#!/usr/bin/env python3
"""
test_pipeline.py — end-to-end check against data with a known answer.

WHY: an analysis that has only ever run on real data cannot be verified. If the
collocate finder returns something plausible about Gaza, there is no way to tell
whether it found a real pattern or a bug that would have returned something
plausible regardless. So we simulate a corpus with a planted effect and a
planted non-effect, and assert both are handled correctly.

  simulate.py  ->  load_db.py  ->  word_freq.py  ->  assertions here

Sections 8-14 (added 2026-09-17) pin the bugs found in the handoff review so
they cannot come back: substring KWIC matching, the asymmetric filler list,
the crawl script's literal backslash-t, a silently missing speaker in the
sampling frame, the hours quota, the QA gate, host resolution with unlabelled
speech, the dashboard status feed and the WhatsApp notifier.
Section 17 (2026-09-20) covers pipeline/segment.py: topic episodes in a meeting.
Section 18 (2026-09-21) covers scripts/role_label.py: dais / public speaker roles.
Section 19 (2026-09-21) covers the meeting pipeline (T1): schema migration,
the Granicus adapter, meeting discovery, captions, classify, episode cuts and
clip registration, the GPU manifest / ingest write-back, roles, meeting QA
rules and the dashboard's episodes panel.
Section 20 (2026-09-21) covers Phase A0: the episode loader with roles, the
term/frame verification counts, name redaction and the exploratory figures.
Section 36 (2026-10-17) covers pipeline/redact.py, the name-redaction pass that
every published episode transcript goes through (DECISIONS #15): each cue rule,
the spaCy PERSON layer, addresses, contacts, the place allowlist, idempotency,
per-episode counts, the audit CSV and the residual scan.

Run:  python3 analysis/test_pipeline.py      (or: make test)
Exit code 0 = all checks pass. Needs only numpy + the standard library.
"""
import csv, hashlib, json, os, random, re, shutil, sqlite3, stat, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
sys.path.insert(0, str(ROOT / "analysis"))

from simulate import NEAR_NODE, NODE, NULL_WORD          # noqa: E402
import word_freq as wf                                    # noqa: E402
import qa_transcripts as qa                               # noqa: E402

fails = []
def check(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail and not ok else ""))
    if not ok:
        fails.append(label)


def section_kwic(db):
    print("\n4. recover the planted effect (through word_freq.kwic_counts, the shipped code)")
    corpus, _ = wf.load(db, host_only=True)
    node_re = wf.node_pattern(NODE)
    for speaker, planted in NEAR_NODE.items():
        near, far, _, hits = wf.kwic_counts(corpus[speaker], node_re, 2)
        n_near, n_far = sum(near.values()), sum(far.values())
        scored = sorted(
            ((w, wf.log_ratio(a, far[w], n_near, n_far),
              wf.log_likelihood(a, far[w], n_near, n_far))
             for w, a in near.items() if a >= 10),
            key=lambda r: -r[1])
        top = [w for w, _, _ in scored[:len(planted)]]
        check(set(top) == set(planted), f"{speaker}: top-{len(planted)} collocates of '{NODE}'",
              f"got {top}, expected {sorted(planted)}")
        check(all(g > 10.83 for _, _, g in scored[:len(planted)]),
              f"{speaker}: planted collocates significant at p<0.001")
        # The other speakers' planted words must NOT appear for this one.
        foreign = {w for s, ws in NEAR_NODE.items() if s != speaker for w in ws}
        check(not (set(top) & foreign), f"{speaker}: no cross-contamination from other speakers")

    print("\n5. the planted NULL must stay non-significant")
    for speaker in NEAR_NODE:
        near, far, _, _ = wf.kwic_counts(corpus[speaker], node_re, 2, keep_fillers=True)
        n_near, n_far = sum(near.values()), sum(far.values())
        g = wf.log_likelihood(near[NULL_WORD], far[NULL_WORD], n_near, n_far)
        lr = wf.log_ratio(near[NULL_WORD], far[NULL_WORD], n_near, n_far)
        check(g < 3.84 and abs(lr) < 0.25,
              f"{speaker}: '{NULL_WORD}' not distinctive", f"G2={g:.1f} LR={lr:+.2f}")


def section_frame(tmp):
    print("\n10. sampling frame: hours quota, missing speaker, determinism, overwrite guard")
    plan = tmp / "frame" / "plan"
    plan.mkdir(parents=True)
    (plan / "channels.tsv").write_text("hasanabi\tu1\nbenshapiro\tu2\ntucker\tu3\n")
    rng = random.Random(11)
    rows = ["speaker\tchannel_url\tvideo_id\tupload_date\tduration_s\tview_count\tlive_status\tcrawled_at\ttitle"]
    spec = {"hasanabi": (15, 900, 5400), "benshapiro": (20, 300, 1500), "tucker": (6, 1800, 7200)}
    for sp, (n, lo, hi) in spec.items():
        for m in range(1, 7):
            for k in range(1 if (sp == "tucker" and m == 3) else n):
                rows.append(f"{sp}\turl\t{sp[:2]}{m:02d}{k:03d}\t2026{m:02d}{rng.randint(1, 28):02d}"
                            f"\t{rng.randint(lo, hi)}\tNA\tNA\tts\tt")
    cat = plan / "catalog.tsv"
    cat.write_text("\n".join(rows) + "\n")
    script = str(ROOT / "plan" / "build_frame.py")
    base = [PY, script, "--catalog", str(cat), "--channels", str(plan / "channels.tsv"),
            "--out-dir", str(plan), "--max-min", "0", "--quota-hours", "3", "--cap-min", "45"]
    r = subprocess.run(base + ["--draw"], capture_output=True, text=True)
    check(r.returncode == 0, "build_frame --draw succeeds on a complete catalog", r.stderr[-300:])
    frame = list(csv.DictReader((plan / "frame.csv").open()))
    cells = {}
    for f in frame:
        cells.setdefault((f["speaker"], f["period"]), []).append(f)
    Q, CAP = 3 * 3600, 45 * 60
    ok = all(Q <= sum(float(f["capped_s"]) for f in fs) < Q + CAP or len(fs) <= 3
             for fs in cells.values())
    check(ok and cells, "every sampled cell holds quota <= capped hours < quota + one capped video")
    check(all(float(f["capped_s"]) <= CAP for f in frame), "no video contributes more than the cap")
    pis = [float(f["pi_hat"]) for f in frame]
    check(pis and all(0 < x <= 1 for x in pis), "every sampled video has an inclusion probability in (0, 1]")
    meta = json.loads((plan / "frame_meta.json").read_text())
    check(meta["periods_dropped"] == ["2026-03"],
          "a month with a thin speaker is dropped for ALL speakers (matched design)",
          f"dropped={meta['periods_dropped']}")
    first = [(f["speaker"], f["period"], f["video_id"]) for f in frame]
    r = subprocess.run(base + ["--draw"], capture_output=True, text=True)
    check(r.returncode != 0 and "exists" in (r.stdout + r.stderr),
          "refuses to overwrite frame.csv without --force")
    lines = cat.read_text().splitlines()
    body = lines[1:]
    random.Random(3).shuffle(body)
    cat.write_text("\n".join([lines[0]] + body) + "\n")
    subprocess.run(base + ["--draw", "--force"], capture_output=True, text=True)
    again = [(f["speaker"], f["period"], f["video_id"]) for f in csv.DictReader((plan / "frame.csv").open())]
    check(sorted(first) == sorted(again), "same seed + reordered catalog -> identical frame")
    cat.write_text("\n".join([lines[0]] + [l for l in body if not l.startswith("tucker")]) + "\n")
    r = subprocess.run(base + ["--draw", "--force"], capture_output=True, text=True)
    check(r.returncode != 0 and "tucker" in (r.stdout + r.stderr),
          "a speaker missing from the catalog aborts the draw (no silent 2-speaker study)")


def section_crawl(tmp):
    print("\n11. crawl script writes real tab-separated columns")
    root = tmp / "crawl"
    (root / "plan").mkdir(parents=True)
    (root / ".venv" / "bin").mkdir(parents=True)
    shutil.copy(ROOT / "plan" / "crawl_metadata.sh", root / "plan" / "crawl_metadata.sh")
    (root / "plan" / "channels.tsv").write_text("# test\nspk\thttps://example.invalid/@x/videos\n")
    shim = root / ".venv" / "bin" / "yt-dlp"
    # Minimal yt-dlp stand-in: renders the --print template it is GIVEN, so a
    # template carrying a literal backslash-t shows up as a broken catalog.
    shim.write_text(f"#!{PY}\n" + r'''import re, sys
a = sys.argv[1:]
t = a[a.index("--print") + 1]
e = {"id": "vid1", "upload_date": "20260105", "duration": "1500", "view_count": "9",
     "live_status": "NA", "title": "A title\twith a tab"}
print(re.sub(r"%\((\w+)\)s", lambda m: e[m.group(1)], t))
''')
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    script = root / "plan" / "crawl_metadata.sh"
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True, cwd=root)
    rows = list(csv.DictReader((root / "plan" / "catalog.tsv").open(), delimiter="\t"))
    ok = (r.returncode == 0 and len(rows) == 1 and rows[0]["video_id"] == "vid1"
          and rows[0]["upload_date"] == "20260105" and rows[0]["title"] == "A title with a tab")
    check(ok, "catalog.tsv has one column per field (the old script wrote literal \\t)",
          f"exit={r.returncode} rows={rows[:1]} err={r.stderr[-200:]}")


def section_qa(tmp):
    print("\n12. transcript QA gate")
    d = tmp / "qa"
    d.mkdir()
    seg = lambda s, t, sp=None: {"start": s, "end": s + 3.5, "text": t, **({"speaker": sp} if sp else {})}
    clean = [seg(i * 4.0, f"sentence number {i} about the budget and the senate vote", "SPEAKER_00")
             for i in range(60)]
    (d / "CLEAN.json").write_text(json.dumps({"video_id": "CLEAN", "segments": clean}))
    # 7 identical segments spanning 27.5 s (a chant, not a loop) vs 20 spanning 79.5 s
    short = [seg(i * 4.0, "thank you so much" if 20 <= i < 27 else f"normal sentence {i} on policy", "SPEAKER_00")
             for i in range(60)]
    (d / "SHORT.json").write_text(json.dumps({"video_id": "SHORT", "segments": short}))
    loop = [seg(i * 4.0, "thank you so much" if 20 <= i < 40 else f"normal sentence {i} on policy", "SPEAKER_00")
            for i in range(60)]
    (d / "LOOP.json").write_text(json.dumps({"video_id": "LOOP", "segments": loop}))
    check(qa.check_video(d / "CLEAN.json")["fail"] == [], "clean diarized transcript passes")
    rs = qa.check_video(d / "SHORT.json")
    check(rs["fail"] == [] and any("7 identical" in m for m in rs["warn"]) and rs["max_dup_span_s"] == 27.5,
          "7 identical consecutive segments over 27.5 s is a WARN, not a FAIL (count is not the unit)", str(rs["warn"]))
    rl = qa.check_video(d / "LOOP.json")
    check(any("decoder loop" in m and "80 s" in m for m in rl["fail"]) and rl["max_dup_span_s"] == 79.5,
          f"identical segments spanning >{qa.DUP_FAIL_S:.0f} s of audio FAIL as a decoder loop", str(rl["fail"]))
    rng = random.Random(2)
    same = all(qa.edit_distance(a, b) == qa.dp_distance(a, b)
               for a, b in ([[rng.choice("abc") for _ in range(rng.randint(0, 30))] for _ in range(2)]
                            for _ in range(200)))
    check(same, "bit-parallel edit distance equals the textbook DP (200 random pairs)")
    ref = "the senator said the bill would pass " * 10
    check(qa.wer(ref, ref + "thanks for watching " * 20) > 0.8,
          "WER divides by the reference: hallucinated text raises it")
    check(qa.norm_words("my [&nbsp;__&nbsp;] word") == ["my", "word"],
          "YouTube's profanity mask is not counted as three words")
    # meeting episode: '<meeting>_e<n>' id with no kind key (b13), two clusters sharing the floor, speaker tags
    ep = [seg(i * 4.0, f"item {i} of the agenda for the board", "SPEAKER_00" if i % 2 else "SPEAKER_01") for i in range(60)]
    ep[10]["text"], ep[11]["text"] = "Male Speaker 1 Director Foley?", "Female Speaker 1"
    (d / "M1_e2.json").write_text(json.dumps({"video_id": "M1_e2", "segments": ep}))
    re_ = qa.check_video(d / "M1_e2.json")
    check(re_["kind"] == "episode" and not any("host share" in m for m in re_["warn"]),
          "'<id>_e<n>' is an episode even without a kind key; no host-share WARN for episodes", str(re_["warn"]))
    check(re_["speaker_tag_segs"] == 2 and any("speaker tag" in m for m in re_["warn"]),
          "subtitle-style 'Male Speaker 1' segments are a WARN", str(re_["warn"]))
    check(qa.check_video(d / "CLEAN.json").get("speaker_tag_segs") == 0, "plain speech has no speaker-tag hits")


def section_host(tmp):
    print("\n13. host resolution ignores unlabelled speech")
    import load_db
    doc = {"segments": [{"start": 0, "end": 50, "text": "a"},                       # unlabelled, longest
                        {"start": 50, "end": 80, "text": "b", "speaker": "SPEAKER_00"},
                        {"start": 80, "end": 90, "text": "c", "speaker": "SPEAKER_01"}]}
    host, conf = load_db.resolve_host(doc)
    check(host == "SPEAKER_00" and conf == round(30 / 90, 3),
          "host = longest LABELLED speaker; share divides by all speech", f"{host} {conf}")
    # planted decoder loop: 5 x "we're not gonna make it" back to back, same speaker
    segs = ([{"start": 0, "end": 4, "text": "the budget vote", "speaker": "SPEAKER_00"}]
            + [{"start": 4 + i, "end": 5 + i, "text": "We're not gonna make it." if i % 2 else "we're not gonna make it.",
                "speaker": "SPEAKER_01"} for i in range(5)]
            + [{"start": 9, "end": 10, "text": "No.", "speaker": "SPEAKER_00"},
               {"start": 10, "end": 11, "text": "No.", "speaker": "SPEAKER_01"},     # other speaker: real exchange
               {"start": 11, "end": 12, "text": "no.", "speaker": "SPEAKER_01"}])    # same speaker again: collapsed
    out, dropped = load_db.collapse_loops(segs)
    check(dropped == 5 and [s["text"] for s in out] == ["the budget vote", "we're not gonna make it.", "No.", "No."]
          and (out[1]["start"], out[1]["end"]) == (4, 9) and out[3]["end"] == 12 and segs[1]["end"] == 5,
          "collapse_loops: same-speaker identical run -> first segment, end extended, 5 dropped; cross-speaker 'No.' kept; input untouched",
          f"dropped={dropped} out={out}")

    # --- host enrolment (scripts/host_enroll.py, 2026-09-20): host_speakers
    # is a SET of labels; every reader honours it and falls back to host_speaker.
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "legacy"))        # Hasan/Tucker one-offs, DECISIONS #16
    import host_enroll, meanness_analysis
    multi = {"video_id": "MULTI", "speaker": "hasanabi", "diarized": True,
             "host_speaker": "SPEAKER_00", "host_confidence": 0.81, "host_method": "embedding",
             "host_speakers": ["SPEAKER_00", "SPEAKER_02"],
             "segments": [{"start": 0, "end": 10, "text": "host a", "speaker": "SPEAKER_00"},
                          {"start": 10, "end": 30, "text": "guest", "speaker": "SPEAKER_01"},
                          {"start": 30, "end": 40, "text": "host b impression", "speaker": "SPEAKER_02"}]}
    single = {"video_id": "SINGLE", "speaker": "tucker", "diarized": True,
              "host_speaker": "SPEAKER_01", "host_confidence": 0.4,
              "segments": [{"start": 0, "end": 10, "text": "guest a", "speaker": "SPEAKER_00"},
                           {"start": 10, "end": 20, "text": "host", "speaker": "SPEAKER_01"},
                           {"start": 20, "end": 30, "text": "guest b", "speaker": "SPEAKER_02"}]}
    check(load_db.host_labels(multi) == {"SPEAKER_00", "SPEAKER_02"} and load_db.host_labels(single) == {"SPEAKER_01"}
          and load_db.host_labels({"segments": []}, "SPEAKER_09") == {"SPEAKER_09"},
          "host_labels: host_speakers set when present; host_speaker fallback; resolve_host answer when neither")
    hs, _ = meanness_analysis.host_segments(multi["segments"], {k: v for k, v in multi.items() if k != "segments"})
    fs, _ = meanness_analysis.host_segments(single["segments"], {k: v for k, v in single.items() if k != "segments"})
    check([s["text"] for s in hs] == ["host a", "host b impression"] and [s["text"] for s in fs] == ["host"],
          "meanness host_segments keeps every host_speakers label; single host_speaker fallback")
    d = tmp / "enroll"
    d.mkdir()
    for doc in (multi, single):
        (d / f"{doc['video_id']}.json").write_text(json.dumps(doc))
    dbp = tmp / "enroll.sqlite"
    subprocess.run([PY, str(ROOT / "analysis" / "load_db.py"), "--src", str(d), "--db", str(dbp)],
                   check=True, capture_output=True)
    con = sqlite3.connect(dbp)
    got = dict(con.execute("SELECT video_id, SUM(is_host) FROM segments GROUP BY 1").fetchall())
    meth = dict(con.execute("SELECT video_id, host_method FROM videos").fetchall())
    con.close()
    check(got == {"MULTI": 2, "SINGLE": 1} and meth == {"MULTI": "embedding", "SINGLE": "talk_time"},
          "load_db is_host: 2 host labels -> 2 host segments; fallback -> 1; host_method recorded", f"{got} {meth}")
    # the decision rule itself (pure function)
    dec = host_enroll.decide
    check(dec([("A", 0.85), ("B", 0.62), ("C", 0.30)]) == ("embedding", ["A", "B"], 0.85, 0.30),
          "decide: every cluster >= 0.60 is host (over-splitting merged); runner-up = best non-host")
    check(dec([("A", 0.85), ("B", 0.78)]) == ("embedding", ["A", "B"], 0.85, None),
          "decide: all clusters host -> runner-up None, still embedding")
    check(dec([("A", 0.70), ("B", 0.58)])[0] == "talk_time" and dec([("A", 0.55), ("B", 0.10)])[0] == "talk_time"
          and dec([("A", 0.75), ("B", 0.59)])[0] == "embedding",
          "decide: gap < 0.15 or best < 0.60 -> talk_time fallback; gap 0.16 -> embedding")
    # QA: an enrolled host with a small share is not a heuristic failure
    q = tmp / "enroll_qa"
    q.mkdir()
    seg = lambda i, sp: {"start": i * 4.0, "end": i * 4.0 + 3.5, "text": f"sentence {i} on the vote", "speaker": sp}
    segs = [seg(i, "SPEAKER_00" if i % 5 == 0 else "SPEAKER_01") for i in range(60)]   # host 20%
    emb = {"video_id": "EMB", "host_speaker": "SPEAKER_00", "host_confidence": 0.8, "host_method": "embedding",
           "host_speakers": ["SPEAKER_00"], "host_similarity": 0.8, "segments": segs}
    tt = {"video_id": "TT", "host_speaker": "SPEAKER_01", "host_confidence": 0.8, "host_method": "talk_time",
          "host_speakers": ["SPEAKER_01"], "host_similarity": 0.5, "host_runner_up": 0.45, "segments": segs}
    (q / "EMB.json").write_text(json.dumps(emb)); (q / "TT.json").write_text(json.dumps(tt))
    re_, rt = qa.check_video(q / "EMB.json"), qa.check_video(q / "TT.json")
    check(re_["host_share"] == 0.2 and re_["host_method"] == "embedding" and not any("host share" in m for m in re_["warn"])
          and rt["host_share"] == 0.8 and rt["host_method"] == "talk_time",
          "QA host share follows host_speakers under embedding and does not WARN 'heuristic unreliable'", f"{re_['warn']}")
    segs2 = [seg(i, "SPEAKER_01" if i % 5 == 0 else "SPEAKER_00") for i in range(60)]
    tt2 = dict(tt, segments=segs2, host_speaker="SPEAKER_00", host_speakers=["SPEAKER_00"])
    (q / "TT2.json").write_text(json.dumps(tt2))
    rt2 = qa.check_video(q / "TT2.json")
    check(rt2["host_share"] == 0.8, "QA talk_time fallback: share of the longest cluster, as before")
    (q / "TT3.json").write_text(json.dumps(dict(tt, segments=[seg(i, "SPEAKER_00" if i % 2 else "SPEAKER_01") for i in range(60)])))
    rt3 = qa.check_video(q / "TT3.json")
    check(any("host share 50%" in m and "did not decide" in m and "0.50" in m for m in rt3["warn"]),
          "QA talk_time fallback under 55% WARNs with the embedding similarity", f"{rt3['warn']}")


def section_ops(tmp):
    print("\n14. dashboard status feed and WhatsApp notifier (no network)")
    sys.path.insert(0, str(ROOT))
    from dashboard import demo_state, status
    db = tmp / "state.sqlite"
    demo_state.build(db, seed=1)
    s = status.collect(db)
    check(s["totals"]["frame_videos"] > 0 and "cells" in s and "batches" in s,
          "status.collect reads a SCHEMA.md-shaped state.sqlite")
    check(any(a["kind"] == "approval" for a in s["alerts"]),
          "a batch awaiting approval raises an alert")
    text = status.as_text(s)
    check(len(text) < 1500 and "approval" in text.lower(), "WhatsApp status text is short and names the pending batch")
    first = status.new_alerts(s, tmp / "seen.json")
    check(first and status.new_alerts(s, tmp / "seen.json") == "",
          "heartbeat alert diff reports an alert once, then stays quiet")
    missing = status.collect(tmp / "nope.sqlite")
    check(missing["state_db"]["exists"] is False, "a missing state DB is reported, not a crash")

    # -- QA snapshot freshness (2026-09-28). status.py reads the CSV the gate
    # left behind; it does not run the gate. A snapshot taken before the last
    # batch finished therefore reads exactly like a clean corpus ("0 FAIL"),
    # which is how 299 ungated transcripts stayed invisible from 2026-09-23 on.
    from datetime import datetime, timedelta, timezone      # noqa: E402
    fin = status._parse_ts(sqlite3.connect(db).execute(
        "SELECT MAX(finished_at) FROM batches WHERE status = 'done'").fetchone()[0])
    qa_csv = tmp / "qa_latest.csv"
    qa_csv.write_text("video_id,status,issues\nV1,OK,\nV2,WARN,coverage 55%\n")
    stamp = lambda t: os.utime(qa_csv, (t.timestamp(), t.timestamp()))

    stamp(fin - timedelta(hours=3))
    st_stale = status.collect(db, qa_csv=qa_csv)
    kinds = {a["kind"]: a for a in st_stale["alerts"]}
    check(kinds.get("qa_stale", {}).get("level") == "serious" and st_stale["qa"]["stale"] is True
          and st_stale["qa"]["ungated"] > 0 and "make qa" in kinds.get("qa_stale", {}).get("text", ""),
          "status: a QA snapshot older than the last finished batch raises a 'serious' qa_stale alert",
          str(st_stale["qa"]) + str(st_stale["alerts"]))
    txt = status.as_text(st_stale)
    check(len(txt) < 1500 and "STALE" in txt and st_stale["qa"]["ran_at"][:10] in txt,
          "status text dates the QA snapshot and marks it stale, still under 1,500 characters", txt)

    stamp(fin + timedelta(hours=1))
    st_fresh = status.collect(db, qa_csv=qa_csv)
    check(st_fresh["qa"]["stale"] is False and not any(a["kind"] == "qa_stale" for a in st_fresh["alerts"])
          and st_fresh["qa"]["ran_at"][:10] in status.as_text(st_fresh),
          "status: a snapshot newer than the last finished batch is not stale, but is still dated in the digest",
          str(st_fresh["qa"]))

    st_none = status.collect(db, qa_csv=tmp / "no_such_qa.csv")
    check(st_none["qa"] is not None and st_none["qa"]["exists"] is False
          and any(a["kind"] == "qa_missing" and a["level"] == "serious" for a in st_none["alerts"])
          and "NO SNAPSHOT" in status.as_text(st_none),
          "status: an absent QA snapshot is a loud 'serious' alert plus a QA line, not silence",
          str(st_none["qa"]) + str(st_none["alerts"]))

    # The trap: --new-alerts runs every 15 min. An elapsed age inside the alert
    # TEXT would change alert_key() on every heartbeat and spam Puck forever.
    stamp(fin - timedelta(hours=3))
    seen = tmp / "seen_qa.json"
    t0 = datetime.now(timezone.utc)      # demo_state dates itself off the wall clock
    beat1 = status.collect(db, qa_csv=qa_csv, now=t0)
    beat2 = status.collect(db, qa_csv=qa_csv, now=t0 + timedelta(minutes=15))
    out1, out2 = status.new_alerts(beat1, seen), status.new_alerts(beat2, seen)
    key = lambda s_: status.alert_key(next(a for a in s_["alerts"] if a["kind"] == "qa_stale"))
    check("QA snapshot is stale" in out1 and out2 == "" and key(beat1) == key(beat2)
          and beat1["qa"]["age_h"] != beat2["qa"]["age_h"],
          "heartbeat: qa_stale fires once and is silent 15 min later, though its reported age moved",
          f"out1={out1!r} out2={out2!r} k1={key(beat1)!r} k2={key(beat2)!r}")

    env = dict(os.environ, NOTIFY_BACKEND="openclaw", OPENCLAW_NOTIFY_TARGET="+10000000000",
               OPENCLAW_NOTIFY_CHANNEL="whatsapp", OPENCLAW_BIN="openclaw")
    r = subprocess.run([PY, str(ROOT / "pipeline" / "notify.py"), "--dry-run", "warn", "test message"],
                       capture_output=True, text=True, env=env)
    check(r.returncode == 0 and "--channel whatsapp" in r.stdout and "+10000000000" in r.stdout,
          "notify.py builds an 'openclaw message send --channel whatsapp' command", r.stdout + r.stderr)


def section_workers(tmp):
    print("\n15. laptop workers: state machine, discovery, downloader, transcode, batch gate, requeue (no network)")
    sys.path.insert(0, str(ROOT / "pipeline"))
    import state, downloader                                # noqa: E402
    import gpu_batch                                        # noqa: E402
    d = tmp / "workers"; d.mkdir()
    db = d / "state.sqlite"
    (d / "key").write_text("not a real key"); (d / "key.pub").write_text("ssh-ed25519 AAAA test")
    env = dict(os.environ, STATE_DB_PATH=str(db), AUDIO_DIR=str(d / "audio"), OPUS_DIR=str(d / "opus"),
               NOTIFY_BACKEND="none", YTPT_OFFLINE="1", TRANSCRIBE_CAP_MIN="0.25",     # cap = 15 s
               BATCH_TARGET_HOURS="50", MONTHLY_SPEND_CAP_USD="40", RUNPOD_API_KEY="x", HF_TOKEN="x",
               GPU_SSH_KEY=str(d / "key"), BATCH_DIR=str(d / "batches"), WHISPER_DIR=str(d / "whisper"))
    run = lambda *args, **kw: subprocess.run([PY, str(ROOT / "pipeline" / args[0]), *args[1:]],
                                             capture_output=True, text=True, env=dict(env, **kw.get("env", {})))
    q = lambda sql, *p: sqlite3.connect(db).execute(sql, p).fetchall()

    # -- state machine
    con = state.connect(db)
    con.execute("INSERT INTO videos (video_id, speaker, channel_url, month, duration_s, crawled_at, in_frame, "
                "status, updated_at) VALUES ('SM1','hasanabi','u','2026-09',30,'t',1,'discovered','t')")
    a = state.claim(con, "SM1", "discovered", "downloading")
    b = state.claim(con, "SM1", "discovered", "downloading")
    check(a and not b, "a conditional claim succeeds once; the second worker gets rowcount 0")
    check(q("SELECT to_status FROM events WHERE video_id='SM1'") == [("downloading",)],
          "every transition writes an events row")
    for i in range(3):
        st = state.fail(con, "SM1", "downloading", f"boom {i}")
        if i < 2:
            state.claim(con, "SM1", "failed", "downloading")
    check(st == "dead" and q("SELECT attempts FROM videos WHERE video_id='SM1'")[0][0] == 3,
          "third failure -> dead, attempts = 3")
    con.execute("DELETE FROM videos WHERE video_id='SM1'"); con.close()

    # -- discovery from a catalog, first-run sample
    cat = d / "catalog.tsv"
    hdr = "speaker\tchannel_url\tvideo_id\tupload_date\tduration_s\tview_count\tlive_status\tcrawled_at\ttitle\n"
    rows = [("hasanabi", "H1", "20260917", "600"), ("hasanabi", "H2", "20260910", "600"), ("hasanabi", "HSHORT", "20260918", "60"),
            ("hasanabi", "HLONG", "20260918", "9000"), ("tucker", "T1", "20260916", "600"), ("tucker", "T2", "20260901", "600")]
    cat.write_text(hdr + "".join(f"{sp}\thttps://x/{sp}\t{vid}\t{dt}\t{dur}\t100\tNA\t2026-09-18T00:00:00Z\ttitle {vid}\n"
                                 for sp, vid, dt, dur in rows))
    r = run("discover.py", "--catalog", str(cat), "--pick-recent", "1")
    picked = sorted(v for (v,) in q("SELECT video_id FROM videos WHERE in_frame=1"))
    check(r.returncode == 0 and picked == ["H1", "T1"],
          "discover: catalog loaded, most recent eligible video per speaker picked (Shorts and >90 min skipped)",
          r.stdout + r.stderr + str(picked))
    r = run("discover.py", "--catalog", str(cat))
    check("0 new" in r.stdout and len(q("SELECT 1 FROM videos")) == 6, "discover is idempotent (ON CONFLICT DO NOTHING)")
    (d / "audio").mkdir(); (d / "opus").mkdir()

    # -- downloader against a fake yt-dlp that writes a real 10-minute tone
    have_ffmpeg = shutil.which("ffmpeg") is not None
    fake = d / "yt-dlp"
    fake.write_text(
        "#!/bin/sh\n"
        "# stand-in for yt-dlp: fail with $FAKE_YTDLP_STDERR, else write <id>.m4a (600 s tone) + <id>.info.json\n"
        'if [ -n "$FAKE_YTDLP_STDERR" ]; then echo "$FAKE_YTDLP_STDERR" >&2; exit 1; fi\n'
        'out=""; url=""\n'
        'while [ $# -gt 0 ]; do [ "$1" = "-o" ] && out="$2"; url="$1"; shift; done\n'
        'id=${url#*v=}; dir=$(dirname "$out")\n'
        'ffmpeg -nostdin -v error -f lavfi -i "sine=frequency=440:duration=600" -c:a aac -y "$dir/$id.m4a"\n'
        'printf \'{"id":"%s","upload_date":"20260917","duration":600,"view_count":5,"like_count":1,"title":"fake"}\' "$id" '
        '> "$dir/$id.info.json"\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    if have_ffmpeg:
        r = run("downloader.py", "--max", "1", env={"YTDLP_BIN": str(fake)})
        row = q("SELECT status, audio_path, date_exact, month FROM videos WHERE video_id='H1'")[0]
        check(r.returncode == 0 and row[0] == "downloaded" and Path(row[1] or "").exists() and row[2] == 1,
              "downloader: fake yt-dlp -> file verified by ffprobe -> 'downloaded', exact date recorded",
              r.stdout + r.stderr + str(row))
    else:
        check(True, "downloader run skipped (no ffmpeg on this machine)")
    r = run("downloader.py", "--max", "1", env={"YTDLP_BIN": str(fake),
                                                 "FAKE_YTDLP_STDERR": "ERROR: [youtube] T1: Sign in to confirm you're not a bot"})
    row = q("SELECT status, attempts, last_error FROM videos WHERE video_id='T1'")[0]
    check(r.returncode == downloader.EXIT_BOT_CHECK and row[0] == "discovered" and row[1] == 0 and "not a bot" in (row[2] or ""),
          "downloader: bot check -> exit 3, row back to 'discovered', no attempt charged, error kept verbatim",
          f"rc={r.returncode} row={row} {r.stdout[-300:]}")
    r = run("downloader.py", "--max", "1", env={"YTDLP_BIN": str(fake),
                                                 "FAKE_YTDLP_STDERR": "ERROR: unable to download video data: HTTP Error 403: Forbidden"})
    row = q("SELECT status, attempts FROM videos WHERE video_id='T1'")[0]
    check(r.returncode == 0 and row == ("failed", 1), "downloader: an ordinary error -> 'failed', attempts = 1", str(row))
    check(downloader.classify_error("HTTP Error 429: Too Many Requests", 0) == "bot"
          and downloader.classify_error("HTTP Error 403: Forbidden", 0) == "error"
          and downloader.classify_error("HTTP Error 403: Forbidden", 2) == "bot",
          "429 stops at once; 403 stops only as a storm (third in a row)")
    import random as _r
    gaps = [downloader.random_gap(45, 180, _r.Random(i)) for i in range(2000)]
    check(min(gaps) >= 22.5 and max(gaps) < 220 and 100 < sum(gaps) / len(gaps) < 125 and len(set(round(g) for g in gaps)) > 100,
          "download pause is random: uniform 45-180 s with jitter, never constant", f"min={min(gaps):.0f} max={max(gaps):.0f}")
    check(downloader.MAX_CONCURRENCY == 2 and "--cookies" not in " ".join(downloader.ytdlp_command("X", d)),
          "yt-dlp command: audio only, no cookies, concurrency hard-capped at 2")

    # -- transcode with the cap
    if have_ffmpeg:
        r = run("transcode.py")
        row = q("SELECT status, opus_path, transcribed_s FROM videos WHERE video_id='H1'")[0]
        check(r.returncode == 0 and row[0] == "transcoded" and row[2] == 15 and Path(row[1] or "").exists(),
              "transcode: 600 s file -> opus cut to the 15 s cap, transcribed_s = 15, m4a kept",
              r.stdout + r.stderr + str(row))
        check(Path(q("SELECT audio_path FROM videos WHERE video_id='H1'")[0][0]).exists(), "the m4a is still there")
        r = run("transcode.py", "--check")
        check(r.returncode == 0 and "0 problem" in r.stdout, "audio-check re-verifies the opus", r.stdout)
    else:
        check(True, "transcode skipped (no ffmpeg)")

    # -- GPU batch approval gate
    r = run("gpu_batch.py", "propose", "--quiet")
    check(r.returncode == 0 and "below the 50 h threshold" in r.stdout and not q("SELECT 1 FROM batches"),
          "propose refuses a batch below BATCH_TARGET_HOURS", r.stdout + r.stderr)
    if have_ffmpeg:
        r = run("gpu_batch.py", "propose", "--quiet", "--min-hours", "0")
        b = q("SELECT status, n_videos, est_usd FROM batches WHERE batch_id='b1'")
        check(r.returncode == 0 and b and b[0][0] == "awaiting_approval" and b[0][1] == 1 and 0 < b[0][2] < 1
              and q("SELECT status FROM videos WHERE video_id='H1'")[0][0] == "queued",
              "propose: batch b1 awaiting_approval with an estimate; its video is 'queued'", r.stdout + r.stderr + str(b))
        r = run("gpu_batch.py", "launch", "b1", "--dry-run")
        check(r.returncode == gpu_batch.EXIT_REFUSED and "not 'approved'" in r.stdout,
              "launch REFUSES an unapproved batch", r.stdout)
        run("gpu_batch.py", "approve", "b1")
        r = run("gpu_batch.py", "launch", "b1", "--dry-run", env={"MONTHLY_SPEND_CAP_USD": "0.001"})
        check(r.returncode == gpu_batch.EXIT_REFUSED and "MONTHLY_SPEND_CAP_USD" in r.stdout,
              "launch REFUSES an approved batch whose estimate breaks the monthly cap", r.stdout)
        r = run("gpu_batch.py", "launch", "b1", "--dry-run")
        check(r.returncode == 0 and "all checks pass" in r.stdout and "dry run" in r.stdout
              and "podFindAndDeployOnDemand" in r.stdout and not q("SELECT 1 FROM pods"),
              "launch --dry-run PASSES with the Phase 2 artefacts and prints the plan; no pod row is written", r.stdout[-600:])
        r = run("gpu_batch.py", "launch", "b1", "--dry-run", env={"GPU_SSH_KEY": str(d / "nokey")})
        check(r.returncode == gpu_batch.EXIT_REFUSED and "ssh-keygen" in r.stdout,
              "launch REFUSES without the pod SSH key (preflight catches it for $0)", r.stdout)
        saved_root, gpu_batch.ROOT = gpu_batch.ROOT, d
        try:
            probs = gpu_batch.launch_checks(state.connect(db), "b1")
        finally:
            gpu_batch.ROOT = saved_root
        check(any("gpu-freeze.txt" in p for p in probs) and any("gpu_job.py" in p for p in probs),
              "launch REFUSES when the Phase 0/2 artefacts are missing", str(probs))
        sqlite3.connect(db).execute("INSERT INTO pods VALUES ('p-live','2026-09-18T00:00:00+00:00',NULL,'running')").connection.commit()
        r = run("gpu_batch.py", "launch", "b1", "--dry-run")
        check("already live" in r.stdout, "launch REFUSES while a pod is live", r.stdout)
        sqlite3.connect(db).execute("DELETE FROM pods").connection.commit()
        r = run("gpu_batch.py", "deny", "b1")
        check(r.returncode == 0 and q("SELECT status FROM batches WHERE batch_id='b1'")[0][0] == "denied"
              and q("SELECT status FROM videos WHERE video_id='H1'")[0][0] == "transcoded" and not q("SELECT 1 FROM pods"),
              "deny: batch denied, video back to 'transcoded', no pod exists", r.stdout + r.stderr)
        check(gpu_batch.estimate_usd(50) < 3 and gpu_batch.estimate_usd(0) > 0,
              "estimate: 50 h at 25x is ~$1-2 with contingency; boot overhead is never zero")
        # -- --speaker filter: a parked 'transcoded' row of another speaker is never swept in
        opus_h1 = q("SELECT opus_path FROM videos WHERE video_id='H1'")[0][0]
        sqlite3.connect(db).execute(
            "INSERT INTO videos (video_id, speaker, channel_url, month, duration_s, transcribed_s, opus_path, "
            "crawled_at, in_frame, status, updated_at) VALUES ('PARK1','tucker','u','2026-09',15,15,?,'t',1,'transcoded','t')",
            (opus_h1,)).connection.commit()
        r = run("gpu_batch.py", "propose", "--quiet", "--min-hours", "0", "--speaker", "nobody")
        check(r.returncode == 0 and "for speaker nobody" in r.stdout and not q("SELECT 1 FROM batches WHERE batch_id='b2'"),
              "propose --speaker with no matching rows proposes nothing", r.stdout + r.stderr)
        r = run("gpu_batch.py", "propose", "--quiet", "--min-hours", "0", "--speaker", "hasanabi, nobody")
        b2 = q("SELECT video_id FROM batch_videos WHERE batch_id='b2'")
        check(r.returncode == 0 and b2 == [("H1",)] and "1 hasanabi" in r.stdout
              and q("SELECT status FROM videos WHERE video_id='PARK1'")[0][0] == "transcoded",
              "propose --speaker hasanabi,nobody: only H1 batched; the parked tucker row stays 'transcoded'",
              r.stdout + r.stderr + str(b2))
        check(gpu_batch.parse_speakers(" a, b ,,c ") == ["a", "b", "c"] and gpu_batch.parse_speakers(None) == [],
              "parse_speakers: comma list, whitespace stripped, empty = no filter")
        run("gpu_batch.py", "deny", "b2")
        sqlite3.connect(db).execute("DELETE FROM videos WHERE video_id='PARK1'").connection.commit()

    # -- requeue and the status feed
    r = run("requeue.py", "T1")
    check(r.returncode == 0 and q("SELECT status, attempts FROM videos WHERE video_id='T1'")[0] == ("discovered", 0),
          "requeue: failed row with no file on disk -> 'discovered', attempts reset", r.stdout)
    from dashboard import status                            # noqa: E402
    s = status.collect(db, cfg=dict(status.settings(), cap_min=0.25))
    check(s["totals"]["frame_videos"] == 2 and any(b["status"] == "denied" for b in s["batches"]),
          "dashboard/status.py reads the real workers' DB (in-frame rows, batches)")


def section_gpu(tmp):
    print("\n16. GPU job (Phase 2) offline: freeze pins, bootstrap syntax, contract, fake pod run, QA-gated ingest")
    sys.path.insert(0, str(ROOT / "pipeline"))
    import state, gpu_job                                   # noqa: E402
    import load_db                                          # noqa: E402
    d = tmp / "gpu"; d.mkdir()

    # -- freeze file: pip-installable lines, the packages that fight, honest header
    lines = (ROOT / "plan" / "gpu-freeze.txt").read_text().splitlines()
    pins = {}
    bad = []
    for l in lines:
        s = l.split("#")[0].strip()
        if not s:
            continue
        name, sep, ver = s.partition("==")
        (pins.__setitem__(name.lower(), ver) if sep and ver else bad.append(l))
    need = {"whisperx", "faster-whisper", "ctranslate2", "pyannote.audio", "torch", "torchaudio", "huggingface-hub"}
    check(not bad and need <= set(pins), "gpu-freeze.txt: every line is name==version; whisperx/pyannote/faster-whisper/ctranslate2/torch pinned",
          f"bad={bad} missing={need - set(pins)}")
    check(pins.get("torch", "").startswith("2.8.") and pins.get("whisperx", "").startswith("3.8.")
          and pins.get("pyannote.audio", "").startswith("4."),
          "gpu-freeze.txt: torch 2.8.x, WhisperX 3.8.x, pyannote-audio 4.x (docs/MODELS.md 3)", str(pins))
    head = "\n".join(lines[:12])
    check("UNVERIFIED" in head or "VERIFIED on" in head, "gpu-freeze.txt says whether a pod has ever run it")

    # -- bootstrap: parses, never echoes the token, refuses without a GPU
    boot = ROOT / "pipeline" / "gpu_bootstrap.sh"
    r = subprocess.run(["bash", "-n", str(boot)], capture_output=True, text=True)
    text = boot.read_text()
    check(r.returncode == 0 and "set -x" not in text and "echo $HF_TOKEN" not in text and 'echo "$HF_TOKEN"' not in text,
          "gpu_bootstrap.sh parses (bash -n) and never prints HF_TOKEN", r.stderr)
    r = subprocess.run(["bash", str(boot), "bZ"], capture_output=True, text=True,
                       env=dict(os.environ, YTPT_JOB_DIR=str(d / "job"), PATH="/usr/bin:/bin", HF_TOKEN="x"))
    check(r.returncode == 2 and "nvidia-smi" in r.stdout, "gpu_bootstrap.sh exits 2 on a machine without a GPU (before pip)", r.stdout[-200:])

    # -- contract: same host rule as the loader, instrument pinned
    meta = {"video_id": "V1", "speaker": "hasanabi", "channel_url": "u", "month": "2026-09", "duration_s": 100.0, "transcribed_s": 90.0}
    segs = [{"start": 0, "end": 50, "text": "a"}, {"start": 50, "end": 80, "text": "b", "speaker": "SPEAKER_00"},
            {"start": 80, "end": 90, "text": "c", "speaker": "SPEAKER_01"}]
    doc = gpu_job.make_doc(meta, segs, gpu_seconds=4.2, batch_id="bZ")
    check(gpu_job.validate_doc(doc, "V1") == [] and (doc["host_speaker"], doc["host_confidence"]) == load_db.resolve_host({"segments": segs})
          and all(k in doc for k in ("transcribed_s", "window_start_s", "model", "diarization_model", "pipeline_version")),
          "make_doc: PLAN.md 1 contract keys present; host/confidence identical to load_db.resolve_host")
    check(doc["model"] == "large-v3" and doc["diarization_model"] == "pyannote/speaker-diarization-community-1",
          "the instrument is pinned: large-v3 + pyannote community-1 (docs/MODELS.md)")

    # -- pod side with a fake WhisperX: per-video atomic files, timing, FAILED, re-run skips
    job = d / "job"; (job / "in").mkdir(parents=True, exist_ok=True); (job / "out").mkdir(exist_ok=True)
    (job / "in" / "V1.opus").write_bytes(b"x"); (job / "in" / "V2.opus").write_bytes(b"x")
    manifest = {"batch_id": "bZ", "videos": [dict(meta, opus="V1.opus"), dict(meta, video_id="V2", opus="V2.opus"),
                                             dict(meta, video_id="V3", opus="V3.opus")]}     # V3: file never pushed
    (job / "in" / "manifest.json").write_text(json.dumps(manifest))
    calls = []
    gpu_job.load_stack = lambda *a, **k: {"fake": True}
    gpu_job.transcribe_one = lambda stack, path, dev, *a, **k: (calls.append(path.name) or (segs, {"asr": 1.0}))   # 4th arg: initial_prompt (2026-09-22)
    rc = gpu_job.run_on_pod("bZ", job / "in" / "manifest.json", job / "in", job / "out", device="cpu")
    out1 = json.loads((job / "out" / "V1.json").read_text())
    timing = [json.loads(l) for l in (job / "out" / "timing.jsonl").read_text().splitlines()]
    check(rc == 1 and (job / "out" / "V2.json").exists() and (job / "out" / "FAILED" / "V3.txt").exists()
          and out1["batch_id"] == "bZ" and len(timing) == 2 and timing[0]["gpu_s"] >= 0 and not list((job / "out").glob("*.tmp")),
          "run_on_pod: one contract JSON per video, timing.jsonl, FAILED/<id>.txt for a missing file, exit 1", f"rc={rc}")
    gpu_job.run_on_pod("bZ", job / "in" / "manifest.json", job / "in", job / "out", device="cpu")
    check(calls == ["V1.opus", "V2.opus"], "run_on_pod re-run is safe: finished videos are skipped", str(calls))

    # -- CUDA retry: raise once -> cache emptied, batch_size halved for the retry, restored after; raise twice -> FAILED
    job2 = d / "job2"; (job2 / "in").mkdir(parents=True); (job2 / "out").mkdir()
    for v in ("V4", "V5", "V6"):
        (job2 / "in" / f"{v}.opus").write_bytes(b"x")
    (job2 / "in" / "manifest.json").write_text(json.dumps({"batch_id": "bY", "videos": [
        dict(meta, video_id="V4", opus="V4.opus"), dict(meta, video_id="V5", opus="V5.opus"), dict(meta, video_id="V6", opus="V6.opus")]}))
    emptied = []
    class _Cuda:  # noqa: E306
        empty_cache = staticmethod(lambda: emptied.append(1))
    class _Torch:  # noqa: E306
        cuda = _Cuda
    fake = {"torch": _Torch, "batch_size": 16}
    seen = []       # (video, batch_size at call time)
    def flaky(stack, path, dev, *a, **k):  # noqa: E306  (4th arg: initial_prompt, 2026-09-22)
        seen.append((path.name, stack["batch_size"]))
        n = sum(1 for p, _ in seen if p == path.name)
        if path.name == "V4.opus" and n == 1:
            raise RuntimeError("CUDA failed with error out of memory")
        if path.name == "V5.opus":
            raise RuntimeError("parallel_for failed: cudaErrorInvalidDevice: invalid device ordinal")
        if path.name == "V6.opus":
            raise RuntimeError("not a GPU problem at all")
        return segs, {"asr": 1.0}
    gpu_job.load_stack = lambda *a, **k: fake
    gpu_job.transcribe_one = flaky
    rc = gpu_job.run_on_pod("bY", job2 / "in" / "manifest.json", job2 / "in", job2 / "out", device="cpu")
    v4 = json.loads((job2 / "out" / "V4.json").read_text())
    tl = [json.loads(l) for l in (job2 / "out" / "timing.jsonl").read_text().splitlines()]
    check(seen == [("V4.opus", 16), ("V4.opus", 8), ("V5.opus", 16), ("V5.opus", 8), ("V6.opus", 16)] and fake["batch_size"] == 16
          and len(emptied) == 2,       # one per CUDA retry; the pre-existing outer OOM-only empty does not fire for V5
          "CUDA RuntimeError -> empty_cache + ONE retry at batch_size/2, then batch_size restored; non-CUDA errors are not retried",
          f"seen={seen} bs={fake['batch_size']} emptied={len(emptied)}")
    check("retried after CUDA error" in v4.get("pod_note", "") and tl[0]["note"] == v4["pod_note"]
          and gpu_job.validate_doc(v4, "V4") == [],
          "a retried video records 'retried after CUDA error' in the doc (pod_note) and timing.jsonl, contract still valid", str(v4.get("pod_note")))
    check(rc == 1 and (job2 / "out" / "FAILED" / "V5.txt").exists() and (job2 / "out" / "FAILED" / "V6.txt").exists()
          and not (job2 / "out" / "V5.json").exists(),
          "a second CUDA error (or a non-CUDA error) still lands in FAILED/<id>.txt, exit 1", f"rc={rc}")
    gpu_job.transcribe_one = lambda stack, path, dev, *a, **k: (calls.append(path.name) or (segs, {"asr": 1.0}))   # 4th arg: initial_prompt (2026-09-22)
    gpu_job.load_stack = lambda *a, **k: {"fake": True}

    # -- laptop side: QA-gated ingest moves state and never loads a FAIL
    db = d / "state.sqlite"
    os.environ["WHISPER_DIR"] = str(d / "whisper")
    con = state.connect(db)
    for vid in ("V1", "LOOP"):
        con.execute("INSERT INTO videos (video_id, speaker, channel_url, month, crawled_at, in_frame, status, updated_at) "
                    "VALUES (?, 'hasanabi', 'u', '2026-09', 't', 1, 'transcribing', 't')", (vid,))
    staging = d / "staging"; staging.mkdir(); (staging / "FAILED").mkdir()
    good = gpu_job.make_doc(meta, [{"start": i * 4.0, "end": i * 4.0 + 3.5, "text": f"sentence {i} on the budget", "speaker": "SPEAKER_00"}
                                   for i in range(30)], batch_id="bZ")
    good["pod_note"] = "retried after CUDA error (test)"
    loop = gpu_job.make_doc(dict(meta, video_id="LOOP"),                     # 17 identical lines spanning 67.5 s
                            [{"start": i * 4.0, "end": i * 4.0 + 3.5, "text": "thank you" if 5 <= i < 22 else f"line {i} on policy",
                              "speaker": "SPEAKER_00"} for i in range(30)], batch_id="bZ")
    (staging / "V1.json").write_text(json.dumps(good)); (staging / "LOOP.json").write_text(json.dumps(loop))
    got = gpu_job.ingest_new(con, "bZ", staging, set(), log=lambda *a: None)
    st = {r[0]: (r[1], r[2]) for r in con.execute("SELECT video_id, status, last_error FROM videos")}
    check(got["done"] == 1 and got["qa_fail"] == 1 and st["V1"][0] == "done" and st["LOOP"][0] == "failed"
          and "decoder loop" in (st["LOOP"][1] or "") and (d / "whisper" / "V1.json").exists() and not (d / "whisper" / "LOOP.json").exists(),
          "ingest: QA OK -> output/whisper + 'done'; QA FAIL (decoder loop) -> 'failed' with the reason, file NOT loaded", str(st))
    ev = con.execute("SELECT note FROM events WHERE video_id = 'V1' AND to_status = 'done'").fetchone()[0]
    check("retried after CUDA error" in (ev or ""), "ingest copies the pod's retry note into the state event", str(ev))
    os.environ.pop("WHISPER_DIR", None)


def section_segment(tmp):
    print("\n17. topic segmentation (pipeline/segment.py): planted block, VTT dedupe, whole-word terms, chapter cross-check")
    sys.path.insert(0, str(ROOT / "pipeline"))
    import segment as sg                                    # noqa: E402
    d = tmp / "seg"; d.mkdir()
    (d / "kw.txt").write_text("# comment\ncoaster\nbluff\ndouble track\ntrain\n")
    kw = sg.load_keywords(d / "kw.txt")

    # -- synthetic hour: filler everywhere, a 6-minute block at 30-36 min dense with the terms
    filler = "the council discussed the budget and the parks and the library tonight"
    dense = "the coaster train on the bluffs needs double track said the coaster rider"
    segs = [{"start": t, "end": t + 4.0, "text": dense if 1800 <= t < 2160 else filler} for t in range(0, 3600, 5)]
    # smoothing over 3 windows may mark one neighbour window on each side, then --pad adds 30 s
    eps = sg.find_episodes(segs, kw, window=60, threshold=1.5, min_episode=120, pad=30)
    check(len(eps) == 1 and 1710 <= eps[0]["start_s"] <= 1800 and 2160 <= eps[0]["end_s"] <= 2250
          and eps[0]["top_keywords"][0] == "coaster",
          "planted 6-min block -> exactly one episode, bounds within one window + pad of the block, top keyword right", str(eps))
    check(sg.find_episodes([dict(s, text=filler) for s in segs], kw) == [], "no keywords anywhere -> zero episodes")
    # two short dense runs 60 s apart (< gap) merge; one lone 60-s run is dropped by --min-episode
    segs2 = [{"start": t, "end": t + 4.0, "text": dense if (600 <= t < 720 or 780 <= t < 900) else filler} for t in range(0, 3600, 5)]
    eps2 = sg.find_episodes(segs2, kw, window=60, threshold=1.5, min_episode=120, pad=0, gap=90)
    check(len(eps2) == 1 and 540 <= eps2[0]["start_s"] <= 600 and 900 <= eps2[0]["end_s"] <= 960, "two runs 60 s apart merge into one episode", str(eps2))
    segs3 = [{"start": t, "end": t + 4.0, "text": dense if 600 <= t < 660 else filler} for t in range(0, 3600, 5)]
    check(sg.find_episodes(segs3, kw, window=60, threshold=1.5, min_episode=200, pad=0) == [], "a lone 60-s run is dropped by --min-episode")

    # -- whole-word terms: 'training' is not 'train'; 'bluffs' is 'bluff'; 'double-track' is 'double track'
    c = sg.hits("training on the bluffs and the double-track and the Coaster's schedule", kw)
    check(c == {"bluff": 1, "double track": 1, "coaster": 1}, "whole-word + plural + hyphenated phrase matching", str(dict(c)))

    # -- YouTube rolling VTT: every line appears twice, with <c> tags and entities; keep each once
    (d / "a.vtt").write_text("WEBVTT\nKind: captions\nLanguage: en\n\n"
                             "00:00:01.000 --> 00:00:03.000 align:start position:0%\n \nOkay<00:00:01.500><c> good</c><00:00:02.000><c> evening</c>\n\n"
                             "00:00:03.000 --> 00:00:03.010 align:start position:0%\nOkay good evening\n \n\n"
                             "00:00:03.010 --> 00:00:05.000 align:start position:0%\nOkay good evening\n&gt;&gt; the<00:00:04.000><c> coaster</c>\n\n"
                             "00:00:05.000 --> 00:00:05.010 align:start position:0%\n&gt;&gt; the coaster\n \n\n"
                             "01:00:05.010 --> 01:00:07.000 align:start position:0%\n&gt;&gt; the coaster\nis<00:01:06.000><c> late</c>\n")
    v = sg.read_segments(d / "a.vtt")
    check([s["text"] for s in v] == ["Okay good evening", ">> the coaster", "is late"] and v[0]["start"] == 1.0 and v[2]["start"] == 3605.01,
          "vtt_to_segments: rolling duplicates and <c> tags removed, entities decoded, hour field honoured", str(v))

    # -- structural cues: chapters from info.json, else description timestamps; overlap by time
    (d / "chap.json").write_text(json.dumps({"duration": 3600, "chapters": [
        {"start_time": 0, "end_time": 1500, "title": "Consent calendar"}, {"start_time": 1500, "end_time": 2400, "title": "Item 7: LOSSAN rail"},
        {"start_time": 2400, "end_time": 3600, "title": "Council reports"}]}))
    (d / "desc.json").write_text(json.dumps({"duration": 3600, "chapters": None,
                                             "description": "Agenda\n0:00 Call to order\n25:00 Item 7 - LOSSAN rail\n40:00 Council reports\n"}))
    ch = sg.structural_chapters(d / "chap.json")
    ds = sg.structural_chapters(d / "desc.json")
    check([t for _, _, t in ch] == ["Consent calendar", "Item 7: LOSSAN rail", "Council reports"]
          and ds == [(0, 1500, "Call to order"), (1500, 2400, "Item 7 - LOSSAN rail"), (2400, 3600, "Council reports")],
          "structural_chapters: info.json chapters, or description timestamps with end = next start / duration", str(ds))
    check(sg.overlapping(eps[0], ch) == ["Item 7: LOSSAN rail"] and sg.overlapping({"start_s": 2380, "end_s": 2500}, ch) == ["Item 7: LOSSAN rail", "Council reports"],
          "overlapping(): the episode maps to the right chapter title(s)")

    # -- 2026-09-20 (T0 test 2): family tags, conditional terms, candidate windows, episode merge, label scorer
    (d / "kd.txt").write_text("@family density\nhousing element\n? parking\n")
    (d / "kt.txt").write_text("@family transit\nbart\n? train\n")
    kw2 = sg.load_keywords(d / "kd.txt", d / "kt.txt")
    check(kw2.families() == ["density", "transit"] and kw2.conditional == {"parking", "train"},
          "load_keywords: several files unioned, @family tags and '?' conditional terms parsed", str(kw2.family))
    check(dict(sg.hits("we train them and the parking garage", kw2)) == {}
          and dict(sg.hits("the bart train and the parking garage", kw2)) == {"bart": 1, "train": 1}
          and dict(sg.family_hits("housing element parking bart", kw2)) == {"density": 2, "transit": 1},
          "conditional terms count only next to an unconditional hit of their own family")
    segs4 = [{"start": t, "end": t + 4.0, "text": "the bart train fare" if 600 <= t < 1200 else filler} for t in range(0, 3600, 5)]
    cands = sg.candidate_windows(segs4, kw2, window=60, threshold=0.5, max_run=240)
    check(2 <= len(cands) <= 4 and cands[0]["start_s"] >= 540 and cands[-1]["end_s"] <= 1260
          and all(c["end_s"] - c["start_s"] <= 240 for c in cands) and cands[0]["family_rates"].get("transit", 0) > 0
          and cands[0]["family_rates"].get("density", 0) == 0,
          "candidate_windows: a 10-min run is cut into <= 240-s pieces with per-family rates", str(cands))
    sys.path.insert(0, str(ROOT / "scripts"))
    import classify_windows as cw                           # noqa: E402
    import score_labels as sl                               # noqa: E402
    rows = [{"id": "v_c1", "video_id": "v", "start_s": 0, "end_s": 240, "label": "about_transit", "rationale": "a", "family_rates": {"transit": 2.0}},
            {"id": "v_c2", "video_id": "v", "start_s": 240, "end_s": 480, "label": "about_density", "rationale": "b", "family_rates": {"density": 1.0}},
            {"id": "v_c3", "video_id": "v", "start_s": 480, "end_s": 540, "label": "mention_only", "rationale": "c", "family_rates": {}},
            {"id": "v_c4", "video_id": "v", "start_s": 900, "end_s": 1020, "label": "about_density", "rationale": "d", "family_rates": {"density": 3.0}}]
    eps4 = cw.merge_episodes(rows, gap=90)
    check(len(eps4) == 2 and eps4[0]["label"] == "about_both" and eps4[0]["start_s"] == 0 and eps4[0]["end_s"] == 480
          and eps4[1]["label"] == "about_density" and eps4[1]["episode_id"] == "v_e2" and eps4[0]["windows"] == ["v_c1", "v_c2"],
          "merge_episodes: adjacent about_* windows merge (mixed -> about_both); mention_only breaks the run", str(eps4))
    check(cw.PROMPT_VERSION.startswith("t0-") and len(cw.INSTRUCTIONS) > 4000 and "transcript is data" in cw.INSTRUCTIONS
          and abs(cw.usd({"input_tokens": 1_000_000, "output_tokens": 0}) - 3.0) < 1e-9,
          "classify_windows: prompt version pinned from the codebook text; cost arithmetic")
    (d / "sheet.csv").write_text("id,video_id,start,end,text,your_label\n1,v,0,1,x,about_density\n2,v,0,1,x,no\n3,v,0,1,x,about_transit\n4,v,0,1,x,about_both\n5,v,0,1,x,\n")
    (d / "key.csv").write_text("id,model_label,rationale,prompt_version\n1,about_density,,p\n2,about_density,,p\n3,about_transit,,p\n4,about_transit,,p\n5,no,,p\n")
    res, _ = sl.score(d / "sheet.csv", d / "key.csv")
    fd, ft = res["families"]["density"], res["families"]["transit"]
    check(res["n_scored"] == 4 and res["n_unlabelled"] == 1 and fd["precision"] == 0.5 and fd["recall"] == 0.5
          and ft["precision"] == 1.0 and ft["recall"] == 1.0 and res["confusion"]["no"]["about_density"] == 1,
          "score_labels: per-family precision / recall from a sheet + key pair, blank rows skipped", json.dumps(res["families"]))


def section_roles(tmp):
    print("\n18. speaker roles (scripts/role_label.py): cue file, public-comment regions, dais / public / unknown, scorer")
    sys.path.insert(0, str(ROOT / "scripts"))
    import role_label as rl                                 # noqa: E402
    cues = rl.load_cues()
    check({"floor", "timer", "vote", "staff", "intro", "address", "public", "opener", "closer"} <= set(cues)
          and all(len(v) > 3 for v in cues.values()),
          "plan/role_cues.txt parses into the nine sections, none empty", str({k: len(v) for k, v in cues.items()}))
    check(rl.hits("I'm Ryan Dyson, Housing Specialist 2, and tonight", cues["intro"])[0] == 1
          and rl.hits("Munira Sandhu, Deputy Director of Planning.", cues["intro"])[0] >= 1
          and rl.hits("thank you general manager powers", cues["intro"])[0] == 0
          and rl.hits("President Hernandez and directors, general manager Powers and staff.", cues["intro"])[0] == 0,
          "intro titles count only in self-introduction form, not when someone is addressed")
    check(rl.hits("Thank you, Mayor.", cues["address"])[0] == 1 and rl.hits("Thank you, Mayor Mahan and members of the council", cues["address"])[0] == 0,
          "'thank you, mayor' is a dais address; 'thank you mayor X and members' (a public greeting) is not")

    # -- synthetic 20-min item: staff report, council Q, public comment (3 speakers), motion
    def say(t, spk, text, dur=4.0):
        return {"start": float(t), "end": float(t) + dur, "speaker": spk, "text": text}
    segs = []
    segs += [say(0, "SPEAKER_00", "We are now on item five. Is there a staff report?")]
    segs += [say(5, "SPEAKER_01", "Good evening, Mayor. Jane Doe, Deputy Director of Planning."),
             *[say(10 + 5 * i, "SPEAKER_01", "the project proposes eighty homes and a park on the site") for i in range(20)]]
    segs += [say(112, "SPEAKER_01", "Staff recommends approval. This concludes staff's presentation.")]
    segs += [say(120, "SPEAKER_04", "Thank you, Mayor. I have a question for staff about the parking.", 8)]
    segs += [say(130, "SPEAKER_01", "yes the garage has one hundred spaces", 6)]
    segs += [say(140, "SPEAKER_04", "great thank you that answers it", 8)]
    segs += [say(150, "SPEAKER_00", "Thank you. We'll now take public comment. Please state your name. You have two minutes.", 6)]
    segs += [say(160, "SPEAKER_02", "My name is Pat Lee and I live on Elm Street. I urge you to deny this project."),
             *[say(165 + 5 * i, "SPEAKER_02", "the traffic on elm street is already bad and this makes it worse") for i in range(20)]]
    segs += [say(268, "SPEAKER_00", "Thank you, that's your time. Next speaker.", 3)]
    segs += [say(272 + 5 * i, "SPEAKER_03", "so what worries people around here is the noise and the construction hours") for i in range(18)]
    segs += [say(365, "SPEAKER_00", "Thank you. Seeing no other speakers, I'll close public comment and bring it back to the council.", 6)]
    segs += [say(372, "SPEAKER_04", "I'll move approval of staff's recommendation with the conditions.", 8)]
    segs += [say(381, "SPEAKER_05", "I'll second.", 2)]
    segs += [say(384, "SPEAKER_00", "We have a motion and a second. Please call the roll.", 5)]
    segs += [say(390 + 3 * i, "SPEAKER_06", f"Councilmember {n}?", 1.0) for i, n in enumerate(("A", "B", "C"))]
    segs += [say(391 + 3 * i, "SPEAKER_04", "Aye.", 0.8) for i in range(3)]
    segs += [say(400, "SPEAKER_06", "Motion passes unanimously.", 3)]
    segs += [say(404, "SPEAKER_00", "Thank you. Moving on to item six.", 4)]
    roles, regions = rl.label_episode(segs, cues, 410.0)
    r = {k: v["role"] for k, v in roles.items()}
    check(len(regions) == 1 and 150 <= regions[0][0] <= 151 and 365 <= regions[0][1] <= 372 and regions[0][2] == "SPEAKER_00" and regions[0][3] is True,
          "one public-comment region from the opener to the closer, credited to the chair", str(regions))
    check(r["SPEAKER_00"] == "dais" and roles["SPEAKER_00"]["method"] == "cue_phrase"
          and r["SPEAKER_01"] == "dais" and r["SPEAKER_04"] == "dais" and r["SPEAKER_06"] == "dais",
          "chair, staff presenter, councilmember and clerk are dais", str(r))
    check(r["SPEAKER_02"] == "public" and roles["SPEAKER_02"]["method"] == "cue_phrase"
          and r["SPEAKER_03"] == "public" and roles["SPEAKER_03"]["method"] == "structure",
          "'my name is / I live on' speaker is public by cue; the cue-less speaker inside the region is public by structure", str(r))
    check(roles["SPEAKER_03"]["in_region_share"] >= 0.95 and roles["SPEAKER_04"]["in_region_share"] == 0.0
          and roles["SPEAKER_04"]["turns"] >= 3 and roles["SPEAKER_04"]["confidence"] > roles["SPEAKER_03"]["confidence"],
          "region share and turn counts are computed; several cues beat one structural cue on confidence")
    check(roles["SPEAKER_04"]["turns"] == 3 and len(rl.turns_of(segs, "SPEAKER_04")) >= 4 and len(rl.turns_of(segs, "SPEAKER_06")) == 1,
          "turns: the councilmember has 3 substantive turns (+ the 'Aye.'s); sub-second interjections do not split the clerk's roll call",
          str((rl.turns_of(segs, "SPEAKER_04"), rl.turns_of(segs, "SPEAKER_06"))))
    summ = rl.summarise(roles)
    check(summ["n_public"] == 2 and summ["n_dais"] == 5 and summ["n_unknown"] == 0 and abs(summ["public_share"] - (roles["SPEAKER_02"]["talk_s"] + roles["SPEAKER_03"]["talk_s"]) / summ["talk_s"]) < 1e-3,
          "per-episode summary: counts and public talk-share", str(summ))

    # -- a clip that starts inside public comment: a timer cue opens the region retroactively
    segs2 = [*[say(5 * i, "SPEAKER_01", "and the bike lane on homestead is too narrow for anyone to use") for i in range(16)],
             say(82, "SPEAKER_00", "Thank you, that's your time. Followed by Kevin.", 3),
             *[say(86 + 5 * i, "SPEAKER_02", "the potholes on that stretch have been reported many times") for i in range(16)],
             say(168, "SPEAKER_00", "That was my last speaker card. I'll close public comment.", 4),
             say(174, "SPEAKER_03", "Thank you, Mayor. I'll move the item.", 5)]
    roles2, regions2 = rl.label_episode(segs2, cues, 180.0)
    r2 = {k: v["role"] for k, v in roles2.items()}
    check(regions2 and regions2[0][0] == 0.0 and r2["SPEAKER_01"] == "public" and r2["SPEAKER_02"] == "public" and r2["SPEAKER_00"] == "dais" and r2["SPEAKER_03"] == "dais",
          "clip that begins mid-public-comment: 'that's your time' opens the region from t=0; both speakers public, chair and mover dais", str((regions2, r2)))

    # -- no public-comment stretch at all: prior says dais, but a public cue still wins
    segs3 = [*[say(5 * i, "SPEAKER_00", "the budget adjustment covers the fleet replacement schedule") for i in range(12)],
             *[say(70 + 5 * i, "SPEAKER_01", "my name is Sam and I live in the neighborhood; please reject this") for i in range(4)]]
    roles3, regions3 = rl.label_episode(segs3, cues, 100.0)
    check(regions3 == [] and roles3["SPEAKER_00"]["role"] == "dais" and roles3["SPEAKER_00"]["method"] == "prior"
          and roles3["SPEAKER_01"]["role"] == "public",
          "no region: cue-less cluster gets the dais prior; a 'my name is' cluster is still public", str({k: (v["role"], v["method"]) for k, v in roles3.items()}))
    check(rl.label_episode([say(0, "SPEAKER_00", "hello", 3)], cues, 10.0)[0]["SPEAKER_00"]["role"] in ("dais", "unknown")
          and rl.label_episode([], cues, 10.0) == ({}, []),
          "degenerate clips (one segment, no segments) do not crash")

    # -- scorer: speech-time accuracy strict / lenient, 'other' excluded, public P/R by cluster
    pred = {"c1": {"A": {"role": "dais", "talk_s": 100}, "B": {"role": "public", "talk_s": 50}, "C": {"role": "unknown", "talk_s": 30},
                   "D": {"role": "public", "talk_s": 20}, "E": {"role": "dais", "talk_s": 200}}}
    truth = {"c1": {"A": "dais", "B": "public", "C": "public", "D": "dais", "E": "other"}}
    sc = rl.score(pred, truth)
    check(sc["speech_s"] == 200 and sc["mixed_excluded_s"] == 200 and abs(sc["accuracy_strict"] - 150 / 200) < 1e-3
          and abs(sc["accuracy_lenient"] - 150 / 170) < 1e-3 and sc["public_tp"] == 1 and sc["public_fp"] == 1 and sc["public_fn"] == 1
          and sc["public_precision"] == 0.5 and sc["public_recall"] == 0.5,
          "score(): strict counts unknown as wrong, lenient drops it, 'other' is excluded, cluster P/R for public", json.dumps({k: v for k, v in sc.items() if k not in ("rows", "errors", "excluded")}))


def section_meetings(tmp):
    print("\n19. meeting pipeline (T1, 2026-09-21): migration, Granicus, discovery, captions, classify, cuts, manifest, ingest, QA, status")
    sys.path.insert(0, str(ROOT / "pipeline"))
    import state, granicus, discover, captions, classify, episodes, transcode, gpu_batch, gpu_job, roles  # noqa: E402
    d = tmp / "meet"; d.mkdir()
    have_ffmpeg = shutil.which("ffmpeg") is not None

    # -- migration: a DB in the 2026-09-18 shape (no kind column, no episodes) with a T0 clip row
    old_db = d / "old.sqlite"
    con = sqlite3.connect(old_db)
    con.executescript("""
        CREATE TABLE videos (video_id TEXT PRIMARY KEY, speaker TEXT NOT NULL, channel_url TEXT NOT NULL, month TEXT NOT NULL,
          title TEXT, upload_date TEXT, date_exact INTEGER DEFAULT 0, duration_s REAL, view_count INTEGER, like_count INTEGER,
          raw_meta TEXT, crawled_at TEXT NOT NULL, in_frame INTEGER DEFAULT 0, status TEXT NOT NULL, audio_path TEXT, opus_path TEXT,
          transcribed_s REAL, attempts INTEGER DEFAULT 0, last_error TEXT, updated_at TEXT NOT NULL, sponsor_segments TEXT);
        CREATE TABLE batches (batch_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL, n_videos INTEGER, audio_hours REAL,
          est_usd REAL, approval_requested_at TEXT, approved_at TEXT, approved_via TEXT, pod_id TEXT, started_at TEXT, finished_at TEXT,
          gpu_minutes REAL, actual_usd REAL, note TEXT);
        INSERT INTO videos (video_id, speaker, channel_url, month, crawled_at, in_frame, status, updated_at) VALUES ('SHOW1','hasanabi','u','2026-09','t',1,'done','t');
        INSERT INTO videos (video_id, speaker, channel_url, month, raw_meta, crawled_at, in_frame, status, updated_at)
          VALUES ('YT1_e2','bart','u','2026-09','{"t0": "test 5 episode clip", "episode_id": "YT1_e2", "source_video_id": "YT1"}','t',1,'done','t');
    """)
    con.commit(); con.close()
    r = state.migrate(old_db, backup=True)
    con = sqlite3.connect(old_db); con.row_factory = sqlite3.Row
    kinds = {x["video_id"]: (x["kind"], x["episode_id"]) for x in con.execute("SELECT video_id, kind, episode_id FROM videos")}
    tables = {x[0] for x in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
    check(r["backup"] and Path(r["backup"]).exists() and Path(r["backup"]).parent.name == "backups"
          and "kind" in r["columns_added"] and {"episodes", "sources", "speaker_roles", "classifier_calls"} <= tables,
          "migrate: backup copy written first, then columns and tables added", str(r))
    check(kinds == {"SHOW1": ("show", None), "YT1_e2": ("episode", "YT1_e2")} and r["episode_rows_backfilled"] == 1,
          "migrate: old rows keep kind=show; the T0 clip row becomes kind=episode with its episode_id", str(kinds))
    r2 = state.migrate(old_db, backup=False)
    check(r2["columns_added"] == [] and r2["tables_added"] == [] and r2["episode_rows_backfilled"] == 0, "migrate is idempotent")
    c2 = state.connect(old_db)
    check("kind" in {x[1] for x in c2.execute("PRAGMA table_info(videos)")} and state.missing_columns(c2) == [],
          "connect() on an old DB adds the missing columns (add-only)")
    c2.close()
    check(transcode.cap_seconds("show") == 45 * 60 and transcode.cap_seconds("meeting") == float("inf")
          and transcode.cap_seconds("episode") == float("inf") and "-t" not in transcode.ffmpeg_command("a", "b", float("inf")),
          "transcode cap: 45 min for shows, none for meeting / episode rows")

    # -- Granicus adapter on the saved T0 pages (both listing templates, the player page)
    F = ROOT / "analysis" / "fixtures" / "granicus"
    oak = granicus.parse_listing((F / "oakland_listing.html").read_text(errors="replace"))
    act = granicus.parse_listing((F / "actransit_listing.html").read_text(errors="replace"))
    check(len(oak) == 12 and oak[0]["clip_id"] == "7755" and "City Council" in oak[0]["name"] and oak[0]["duration_s"] == 13320.0
          and oak[0]["agenda"] and oak[0]["date"] == "September 15, 2026",
          "parse_listing: new template (Oakland) -> clip id, name, date, duration, agenda link", str(oak[:1]))
    check(len(act) == 8 and act[0]["clip_id"] == "135" and act[0]["epoch"] == 1295424000 and act[0]["duration_s"] == 540.0
          and all(a["epoch"] >= b["epoch"] for a, b in zip(act, act[1:])),
          "parse_listing: old template (AC Transit) -> epoch from the hidden span, sorted newest first", str(act[:1]))
    pl = granicus.parse_player((F / "oakland_player_7755.html").read_text(errors="replace"), "oakland")
    check(pl["m3u8"] and pl["m3u8"].endswith("playlist.m3u8") and pl["mp4"] and pl["mp4"].endswith(".mp4")
          and pl["captions_url"] == "https://oakland.granicus.com/videos/7755/captions.vtt" and len(pl["index_points"]) == 8
          and pl["index_points"][0] == (481.0, "1 Call To Order"),
          "parse_player: HLS master, progressive MP4, captions track (absolute), 8 agenda index points with times", str(pl)[:300])
    check(granicus.agenda_item_for(pl["index_points"], 500, 900).startswith("5.1 ") and granicus.agenda_item_for([], 0, 1) is None
          and granicus.agenda_item_for(pl["index_points"], 100, 200) is None,
          "agenda_item_for: the index point whose span overlaps the episode most; none before the first point")
    check(granicus.parse_duration("03h 42m") == 13320.0 and granicus.parse_duration("") is None and granicus.video_id_for("oakland", "7755") == "gr-oakland-7755"
          and granicus.parse_source_url("https://oakland.granicus.com/ViewPublisher.php?view_id=2") == ("oakland", "2")
          and granicus.listing_url("x", 1).endswith("view_id=1") and "clip_id=7" in granicus.player_url("x", 1, 7),
          "granicus helpers: duration, ids, URLs")
    cmd = granicus.audio_slice_command("https://h/playlist.m3u8", 600, 300, "/tmp/x.m4a")
    check(cmd.index("-ss") < cmd.index("-i") and "-vn" in cmd and cmd[cmd.index("-c:a") + 1] == "copy" and "-t" in cmd,
          "audio_slice_command: -ss before -i (seeks on the master), audio only, stream copy")
    # segment.py reads the Granicus caption track unchanged
    sys.path.insert(0, str(ROOT / "pipeline"))
    import segment as sg                                    # noqa: E402
    gsegs = sg.read_segments(F / "oakland_captions_7755_head.vtt")
    check(len(gsegs) > 40 and gsegs[0]["start"] == 320.8 and gsegs[0]["text"] == "Good",
          "vtt_to_segments reads the Granicus captions.vtt (no <c> tags, no rolling duplicates)", str(gsegs[:2]))

    # -- discovery: dates in every title shape, regular-meeting selection, kind=meeting rows
    dates = [discover.parse_meeting_date(x) for x in (
        "SEP 15, 2026 |  City Council", "City Council Meeting 09/08/2026 at 6:00PM", "Board of Directors Regular Meeting, September 10, 2026",
        "* Concurrent Meeting of the Oakland Redevelopment Successor Agency and the City Council on 2026-09-15 3:30 PM",
        "City of Santa Rosa City Council 9-15-26", "City Council Meeting - Sept. 15, 2026", "Peregrine Falcon Camera")]
    check(dates == ["2026-09-15", "2026-09-08", "2026-09-10", "2026-09-15", "2026-09-15", "2026-09-15", None],
          "parse_meeting_date: MON D, YYYY / MM/DD/YYYY / Month D, YYYY / YYYY-MM-DD / M-D-YY / Sept. D, YYYY; none -> None", str(dates))
    ch = discover.read_channels()
    meet = {c["speaker"]: c for c in ch if c["kind"] == "meeting"}
    check({"sanjose", "sanrafael", "bart", "oakland"} <= set(meet) and all(c["kind"] == "show" for c in ch if c["speaker"] in ("hasanabi", "tucker"))
          and all(m["regular_re"] for m in meet.values()),
          "channels.tsv: the four pilot bodies are kind=meeting with a regular-title regex; show rows unchanged")
    # literal backslash-t separators, as T0's list2.sh wrote them; real newlines
    listing = ("A1\\tNA\\t5000\\twas_live\\tSEP 15, 2026 |  City Council\n" "A2\\tNA\\t5000\\twas_live\\t15 Septiembre 2026 | Reunion\n"
               "A3\\tNA\\tNA\\tis_live\\tSEP 22, 2026 |  City Council\n" "A4\\tNA\\t50000\\twas_live\\tSEP 1, 2026 |  City Council\n"
               "A5\\tNA\\t5500\\twas_live\\tSEP 1, 2026 |  City Council\n" "A6\\tNA\\t6000\\twas_live\\tAUG 25, 2026 |  City Council\n"
               "A7\\tNA\\t6000\\twas_live\\tJUN 1, 2025 |  City Council\n" "A8\\tNA\\t900\\twas_live\\tSEP 9, 2026 | Planning Commission\n")
    rows = discover.parse_listing_tsv(listing)
    sel = discover.select_regular(rows, meet["sanjose"]["regular_re"], since="2025-09-20", recent=0)
    check([r["id"] for r in sel] == ["A1", "A5", "A6"] and sel[0]["meeting_date"] == "2026-09-15",
          "select_regular: regex keeps council meetings only, live rows and >12 h placeholders dropped, one row per date, window applied, newest first",
          str([(r["id"], r["meeting_date"]) for r in sel]))
    check([r["id"] for r in discover.select_regular(rows, meet["sanjose"]["regular_re"], since="2025-09-20", recent=2)] == ["A1", "A5"],
          "select_regular --recent 2 keeps the newest two")
    check(discover.parse_listing_tsv("X1\tNA\t10\twas_live\tT\n")[0]["id"] == "X1", "parse_listing_tsv reads real tabs too")
    mdb = d / "meet.sqlite"
    con = state.connect(mdb)
    discover.upsert_source(con, "sanjose", meet["sanjose"]["url"], "meeting", meet["sanjose"]["regular_re"], "youtube", "UCx:streams", "council")
    discover.upsert_source(con, "sanjose", meet["sanjose"]["url"], "meeting", "changed", "youtube", "UCx:streams", "council")
    n1 = discover.insert_meeting(con, "sanjose", meet["sanjose"]["url"], "youtube", "A1", "https://www.youtube.com/watch?v=A1",
                                 "SEP 15, 2026 |  City Council", "2026-09-15", 5000.0, {"tab": "streams"})
    n2 = discover.insert_meeting(con, "sanjose", meet["sanjose"]["url"], "youtube", "A1", "u", "t", "2026-09-15", 5000.0, {})
    row = dict(con.execute("SELECT * FROM videos WHERE video_id = 'A1'").fetchone())
    src = dict(con.execute("SELECT * FROM sources WHERE source_id = 'sanjose:youtube:UCx:streams'").fetchone())
    check(n1 == 1 and n2 == 0 and row["kind"] == "meeting" and row["in_frame"] == 0 and row["status"] == "discovered"
          and row["meeting_id"] == "sanjose:2026-09-15" and row["month"] == "2026-09" and row["platform"] == "youtube"
          and src["regular_re"] == "changed",
          "insert_meeting: kind=meeting, in_frame=0 (launchd workers never select it), meeting_id body:date; re-insert is a no-op; source upserted", str(row))
    # the crawl script skips kind=meeting rows
    croot = d / "crawl"; (croot / "plan").mkdir(parents=True); (croot / ".venv" / "bin").mkdir(parents=True)
    shutil.copy(ROOT / "plan" / "crawl_metadata.sh", croot / "plan" / "crawl_metadata.sh")
    (croot / "plan" / "channels.tsv").write_text("mtg\thttps://example.invalid/streams\tmeeting\t^x\n")
    shim = croot / ".venv" / "bin" / "yt-dlp"; shim.write_text("#!/bin/sh\necho SHOULD_NOT_RUN >&2; exit 1\n"); shim.chmod(0o755)
    r = subprocess.run(["bash", str(croot / "plan" / "crawl_metadata.sh")], capture_output=True, text=True, cwd=croot)
    cat = (croot / "plan" / "catalog.tsv")
    check("SHOULD_NOT_RUN" not in r.stderr and (not cat.exists() or len(cat.read_text().splitlines()) <= 1),
          "crawl_metadata.sh skips kind=meeting rows (they belong to discover.py --meetings)", r.stderr[-200:])

    # -- captions: kind detection, the yt-dlp command, a fake fetch, the T0 cache path, Granicus
    (d / "auto.vtt").write_text("WEBVTT\n\n" + "".join(f"00:00:{i:02d}.000 --> 00:00:{i + 1:02d}.000\nword<00:00:{i:02d}.500><c> two</c>\n\n" for i in range(10)))
    (d / "man.vtt").write_text("WEBVTT\n\n" + "".join(f"00:00:{i:02d}.000 --> 00:00:{i + 1:02d}.000\nword two\n\n" for i in range(10)))
    (d / "empty.vtt").write_text("WEBVTT\n\n")
    check(captions.caption_kind(d / "auto.vtt") == "auto" and captions.caption_kind(d / "man.vtt") == "manual"
          and captions.caption_kind(d / "empty.vtt") == "none" and captions.caption_kind(d / "nope.vtt") == "none",
          "caption_kind: inline word timings = auto, plain cues = manual, < 5 cues = none")
    cmd = captions.ytdlp_captions_command("A1", d)
    check("--skip-download" in cmd and cmd[cmd.index("--sub-langs") + 1] == "en,en-orig" and "--write-info-json" in cmd
          and "--sleep-requests" in cmd and "--cookies" not in " ".join(cmd) and not any("en.*" in c for c in cmd),
          "captions command: subs + info only, en,en-orig (never en.*), sleep-requests, no cookies")
    fake = d / "yt-dlp"
    fake.write_text("#!/bin/sh\n" 'if [ -n "$FAKE_YTDLP_STDERR" ]; then echo "$FAKE_YTDLP_STDERR" >&2; exit 1; fi\n'
                    'out=""; url=""\nwhile [ $# -gt 0 ]; do [ "$1" = "-o" ] && out="$2"; url="$1"; shift; done\n'
                    'id=${url#*v=}; dir=$(dirname "$out")\n'
                    'printf "WEBVTT\\n\\n" > "$dir/$id.en.vtt"; i=0; while [ $i -lt 8 ]; do printf "00:00:0%d.000 --> 00:00:0%d.500\\nhousing<00:00:0%d.100><c> element</c>\\n\\n" $i $i $i >> "$dir/$id.en.vtt"; i=$((i+1)); done\n'
                    'printf \'{"id":"%s","upload_date":"20260916","duration":5000,"title":"SEP 15, 2026 | City Council","subtitles":{},"automatic_captions":{"en":[]}}\' "$id" > "$dir/$id.info.json"\n')
    fake.chmod(0o755)
    os.environ["YTDLP_BIN"] = str(fake)
    subs = d / "subs"; subs.mkdir()
    kind, fields, msg, made = captions.captions_youtube("A1", subs)
    check(kind == "ok" and made and fields["captions"] == "auto" and fields["upload_date"] == "20260916" and fields["date_exact"] == 1
          and fields["duration_s"] == 5000.0 and Path(fields["captions_path"]).name == "A1.en.vtt",
          "captions_youtube: fake yt-dlp -> auto track + exact metadata from info.json", f"{kind} {msg} {fields}")
    kind2, _, msg2, made2 = captions.captions_youtube("A1", subs)
    check(kind2 == "ok" and not made2 and "on disk" in msg2, "captions_youtube: a track already on disk is not requested again")
    saved_t0, captions.T0 = captions.T0, d / "t0"
    (d / "t0" / "B1").mkdir(parents=True)
    shutil.copy(d / "man.vtt", d / "t0" / "B1" / "B1.en.vtt")
    (d / "t0" / "B1" / "B1.info.json").write_text(json.dumps({"id": "B1", "upload_date": "20260901", "duration": 100}))
    kind3, f3, msg3, made3 = captions.captions_youtube("B1", subs)
    captions.T0 = saved_t0
    check(kind3 == "ok" and not made3 and "T0 cache" in msg3 and f3["captions"] == "manual" and (subs / "B1.en.vtt").exists()
          and (d / "t0" / "B1" / "B1.en.vtt").exists(),
          "captions_youtube: a meeting T0 already fetched is copied from output/t0, no request; T0 files untouched", msg3)
    kind4, _, msg4, _ = captions.captions_youtube("C1", subs, consecutive_403=0)
    os.environ["FAKE_YTDLP_STDERR"] = "ERROR: [youtube] C1: Sign in to confirm you're not a bot"
    kind5, _, msg5, _ = captions.captions_youtube("C2", subs)
    os.environ.pop("FAKE_YTDLP_STDERR")
    check(kind4 == "ok" and kind5 == "bot" and "not a bot" in msg5, "captions_youtube: a bot check comes back as 'bot' (the worker stops)")
    # Granicus, with the network functions replaced
    saved = (granicus.fetch_player, granicus.fetch_captions)
    granicus.fetch_player = lambda site, view, clip: ({"m3u8": "https://h/p.m3u8", "mp4": None, "captions_url": "https://h/c.vtt",
                                                       "index_points": [(10.0, "1 Call To Order"), (600.0, "5 Housing Element")]}, {"bytes": 1})
    granicus.fetch_captions = lambda url, dest: (shutil.copy(d / "man.vtt", dest) and (100, 10)) or (100, 10)
    try:
        kg, fg, mg, _ = captions.captions_granicus("gr-oak-1", json.dumps({"site": "oak", "view_id": "2", "clip_id": "1"}), subs)
    finally:
        granicus.fetch_player, granicus.fetch_captions = saved
    pj = json.loads((subs / "gr-oak-1.player.json").read_text())
    check(kg == "ok" and fg["captions"] == "granicus" and (subs / "gr-oak-1.en.vtt").exists() and pj["m3u8"] == "https://h/p.m3u8"
          and json.loads(fg["raw_meta"])["index_points"] == 2,
          "captions_granicus: player page saved as <id>.player.json, captions.vtt fetched, raw_meta carries m3u8 + index count", mg)
    # the stage end to end on the DB: A1 discovered -> captioned
    os.environ["SUBS_DIR"] = str(subs)
    discover.insert_meeting(con, "oakland", "u", "granicus", "gr-oak-1", "u", "Concurrent Meeting ... on 2026-09-15", "2026-09-15", 7000.0,
                            {"site": "oak", "view_id": "2", "clip_id": "1"})
    granicus.fetch_player = lambda site, view, clip: ({"m3u8": "https://h/p.m3u8", "mp4": None, "captions_url": "https://h/c.vtt", "index_points": []}, {"bytes": 1})
    granicus.fetch_captions = lambda url, dest: (100, 10)
    try:
        rc = captions.main(["--state", str(mdb), "--min-gap", "10", "--max-gap", "10"])
    finally:
        granicus.fetch_player, granicus.fetch_captions = saved
    st = {r[0]: (r[1], r[2]) for r in con.execute("SELECT video_id, status, captions FROM videos WHERE kind = 'meeting'")}
    check(rc == 0 and st["A1"] == ("captioned", "auto") and st["gr-oak-1"] == ("captioned", "granicus"),
          "captions.py: every discovered meeting -> 'captioned' with its caption kind", str(st))

    # -- classify: stage 1 candidates, a fake client, cache, episodes rows, cap
    (subs / "A1.en.vtt").write_text("WEBVTT\n\n" + "".join(
        f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}.000 --> {t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60 + 4:02d}.000\n"
        + ("the housing element rezoning sites and the density bonus for units" if 600 <= t < 1200 or 1500 <= t < 1620
           else ("bart train fares and ridership on the bart system" if 3000 <= t < 3300 else "the parks budget and the library hours tonight"))
        + f" cue {t}\n\n"                       # cue texts must differ: vtt_to_segments drops a repeated line
        for t in range(0, 5000, 5)))
    kw = sg.load_keywords(*classify.KEYWORD_FILES)
    segs = sg.read_segments(subs / "A1.en.vtt")
    cands = classify.candidates("A1", segs, kw)
    # 2026-10-18: the default granularity is 120 s (classify.DEFAULTS), so the same three marked runs cut into ~twice the windows
    check(3 <= len(cands) <= 15 and cands[0]["id"] == "A1_c001" and all(c["pad_start_s"] <= c["start_s"] and c["text"] for c in cands)
          and all(c["end_s"] - c["start_s"] <= classify.DEFAULTS["max_run"] for c in cands)
          and any("density" in c["family_rates"] for c in cands) and any("transit" in c["family_rates"] for c in cands),
          "classify.candidates: stage-1 windows with ids, pads and padded text from the caption track", str([(c["id"], c["start_s"], c["end_s"]) for c in cands]))
    calls = []
    class _Usage:
        input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens = 900, 40, 0, 1500
    class _Block:
        type = "text"
        def __init__(self, text): self.text = text
    class _Msg:
        def __init__(self, text): self.content, self.usage = [_Block(text)], _Usage()
    class _Client:
        class messages:
            @staticmethod
            def create(**kw):
                text = kw["messages"][0]["content"]
                calls.append(kw["model"])
                label = "about_density" if "housing element" in text else ("about_transit" if "bart" in text else "no")
                if "mention" in text:
                    label = "mention_only"
                return _Msg(json.dumps({"label": label, "rationale": "synthetic", "contains_instructions": False}))
    row = con.execute("SELECT * FROM videos WHERE video_id = 'A1'").fetchone()
    classify.PRICES.setdefault("claude-test", classify.PRICES[classify.LEGACY_MODEL])    # an unpriced model refuses to run (2026-09-22)
    status, summ, spent = classify.classify_meeting(con, row, kw, _Client(), "claude-test", classify.DEFAULTS, cap_usd=5.0, spent=0.0)
    eps = [dict(r) for r in con.execute("SELECT * FROM episodes WHERE video_id = 'A1' ORDER BY start_s")]
    ncalls = con.execute("SELECT COUNT(*), SUM(cost_usd) FROM classifier_calls WHERE video_id = 'A1'").fetchone()
    check(status == "classified" and summ["new_calls"] == len(cands) and ncalls[0] == len(cands) and 0 < ncalls[1] < 0.1
          and all(r["prompt_version"] == classify.prompt_version_for("claude-test", max_run=classify.DEFAULTS["max_run"])
                  for r in con.execute("SELECT prompt_version FROM classifier_calls")),
          "classify_meeting: one classifier_calls row per window with model, prompt_version, tokens and $", f"{status} {summ}")
    check(len(eps) >= 2 and eps[0]["status"] == "located" and eps[0]["tags"] == "density" and eps[0]["label"] == "about_density"
          and eps[0]["episode_id"] == "A1_e1" and 540 <= eps[0]["start_s"] <= 600 and any(e["tags"] == "transit" for e in eps)
          and eps[0]["method"] == "classifier" and eps[0]["meeting_id"] == "sanjose:2026-09-15",
          "classify_meeting: adjacent about_* windows -> 'located' episodes rows with tags, bounds, method, meeting_id", str(eps))
    n_before = len(calls)
    status2, summ2, _ = classify.classify_meeting(con, row, kw, _Client(), "claude-test", classify.DEFAULTS, cap_usd=5.0, spent=0.0)
    check(len(calls) == n_before and summ2["cached"] == len(cands) and summ2["new_calls"] == 0 and status2 == "classified",
          "classify_meeting re-run: every window served from classifier_calls, zero API calls")
    con.execute("DELETE FROM classifier_calls WHERE video_id = 'A1'")
    status3, summ3, _ = classify.classify_meeting(con, row, kw, _Client(), "claude-test", classify.DEFAULTS, cap_usd=0.0001, spent=0.0)
    check(status3 == "stopped" and "cap" in summ3["reason"], "classify_meeting stops BEFORE a call that could pass --cap-usd")
    # T0 jsonl import
    jl = d / "jsonl"; jl.mkdir()
    (jl / "Z.jsonl").write_text(json.dumps({"id": "Z_c001", "video_id": "Z", "start_s": 0, "end_s": 240, "pad_start_s": 0, "pad_end_s": 270, "words": 500,
                                            "hits_per_100w": 1.2, "family_rates": {"transit": 1.2}, "top_keywords": ["bart"], "label": "about_transit",
                                            "rationale": "r", "contains_instructions": False, "model": "m", "prompt_version": classify.PROMPT_VERSION,
                                            "usage": {"input_tokens": 1, "output_tokens": 1, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
                                            "cost_usd": 0.001, "latency_s": 1.0}) + "\n"
                                + json.dumps({"id": "Z_c002", "video_id": "Z", "start_s": 240, "end_s": 480, "label": None, "model": "m",
                                              "prompt_version": "p", "family_rates": {}}) + "\n")
    f, n, ins = classify.import_jsonl(con, jl)
    f2, n2, ins2 = classify.import_jsonl(con, jl)
    check((f, n, ins) == (1, 2, 1) and ins2 == 0 and con.execute("SELECT label FROM classifier_calls WHERE id = 'Z_c001'").fetchone()[0] == "about_transit",
          "import_jsonl: labelled T0 rows land in classifier_calls once; unlabelled rows and re-imports are skipped")
    m = classify.merge_episodes([{"id": "v_c1", "video_id": "v", "start_s": 0, "end_s": 240, "label": "mention_only", "rationale": "", "family_rates": '{"density": 1}'},
                                 {"id": "v_c2", "video_id": "v", "start_s": 240, "end_s": 300, "label": "mention_only", "rationale": "", "family_rates": {}}],
                                gap=90, labels={"mention_only"}, prefix="m")
    check(len(m) == 1 and m[0]["episode_id"] == "v_m1" and m[0]["end_s"] == 300, "merge_episodes: mention_only runs merge under the _m prefix; JSON-string rates accepted")
    # the stage on the DB: A1 captioned -> classified (client injected through anthropic module)
    import types
    fake_mod = types.ModuleType("anthropic"); fake_mod.Anthropic = lambda: _Client()
    sys.modules["anthropic"] = fake_mod
    rc = classify.main(["run", "--state", str(mdb), "--cap-usd", "5"])
    sys.modules.pop("anthropic", None)
    st = {r[0]: r[1] for r in con.execute("SELECT video_id, status FROM videos WHERE kind = 'meeting'")}
    check(rc == 0 and st["A1"] == "classified" and st["gr-oak-1"] == "no_episodes",
          "classify.py run: A1 -> 'classified'; the Granicus meeting whose captions have no keywords -> 'no_episodes'", str(st))

    # -- episodes: bounds, snapping, cuts from a synthetic whole-meeting audio, registration, linking
    edges = episodes.cue_edges([{"start": 100.0, "end": 104.0}, {"start": 104.0, "end": 108.5}])
    check(episodes.snap(105.2, edges) == 104.0 and episodes.snap(120.0, edges) == 120.0 and episodes.snap(107.0, edges) == 108.5,
          "snap: nearest cue edge within 3 s, else unchanged")
    a, b = episodes.clip_bounds(600.0, 1200.0, 5000.0, [569.0, 1231.5])
    check((a, b) == (569.0, 1231.5) and episodes.clip_bounds(10.0, 100.0, 5000.0, []) == (0.0, 130.0)
          and episodes.clip_bounds(4900.0, 4990.0, 5000.0, []) == (4870.0, 5000.0),
          "clip_bounds: +-30 s pad, clamped to [0, duration], snapped to cue edges", str((a, b)))
    if have_ffmpeg:
        audio = d / "audio"; opus = d / "opus"; audio.mkdir(); opus.mkdir()
        os.environ["AUDIO_DIR"], os.environ["OPUS_DIR"] = str(audio), str(opus)
        full = audio / "A1.opus"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=5000", "-c:a", "libopus", "-b:a", "24k", "-y", str(full)],
                       check=True, capture_output=True)
        con.execute("UPDATE videos SET audio_path = ?, duration_s = 5000 WHERE video_id = 'A1'", (str(full),))
        state.transition(con, "A1", "classified", "downloading"); state.transition(con, "A1", "downloading", "downloaded")
        # a pre-existing clip row (the T0 case) for the first episode: it must be linked, not re-cut
        con.execute("INSERT INTO videos (video_id, speaker, channel_url, month, raw_meta, crawled_at, in_frame, status, updated_at, kind) "
                    "VALUES ('A1_e1','sanjose','u','2026-09','{\"window_start_s\": 570.0, \"window_end_s\": 1230.0}','t',1,'done','t','episode')")
        meeting = dict(con.execute("SELECT * FROM videos WHERE video_id = 'A1'").fetchone())
        ok, bad, secs = episodes.cut_youtube(con, meeting, full, audio, opus, lambda *a: None)
        eps = {r["episode_id"]: dict(r) for r in con.execute("SELECT * FROM episodes WHERE video_id = 'A1'")}
        clips = {r["video_id"]: dict(r) for r in con.execute("SELECT * FROM videos WHERE kind = 'episode' AND meeting_id = 'sanjose:2026-09-15'")}
        e2 = next((e for k, e in eps.items() if k != "A1_e1" and e["status"] == "cut"), None)
        check(bad == 0 and eps["A1_e1"]["status"] == "done" and eps["A1_e1"]["start_s"] == 570.0 and clips["A1_e1"]["episode_id"] == "A1_e1",
              "cut_youtube: an existing clip row is linked (episodes row follows its state and bounds), never re-cut", str(eps.get("A1_e1")))
        check(e2 is not None and Path(e2["opus_path"]).exists() and Path(e2["audio_path"]).exists() and clips[e2["episode_id"]]["status"] == "transcoded"
              and clips[e2["episode_id"]]["in_frame"] == 1 and abs(clips[e2["episode_id"]]["transcribed_s"] - (e2["end_s"] - e2["start_s"])) < 1.5
              and transcode.verify(e2["opus_path"], e2["end_s"] - e2["start_s"])[0] and clips[e2["episode_id"]]["kind"] == "episode",
              "cut_youtube: a new episode is copy-cut, transcoded to 16 kHz mono opus, registered as an in-frame 'transcoded' kind=episode row",
              str(e2))
        ok2, bad2, _ = episodes.cut_youtube(con, meeting, full, audio, opus, lambda *a: None)
        check(ok2 == 0 and bad2 == 0 and con.execute("SELECT COUNT(*) FROM episodes WHERE video_id = 'A1' AND status = 'located'").fetchone()[0] == 0,
              "cut_youtube re-run: nothing left 'located', nothing re-done")
        # gpu_batch.propose over the episode rows: episode-hours wording, episodes rows follow, deny returns them
        env = dict(os.environ, STATE_DB_PATH=str(mdb), NOTIFY_BACKEND="none", BATCH_TARGET_HOURS="50", RUNPOD_API_KEY="x", HF_TOKEN="x")
        r = subprocess.run([PY, str(ROOT / "pipeline" / "gpu_batch.py"), "propose", "--quiet", "--min-hours", "0", "--speaker", "sanjose"],
                           capture_output=True, text=True, env=env)
        bid = con.execute("SELECT batch_id FROM batches ORDER BY created_at DESC LIMIT 1").fetchone()[0]
        check(r.returncode == 0 and "episode clips" in r.stdout and "episode-h" in r.stdout
              and con.execute("SELECT status FROM episodes WHERE episode_id = ?", (e2["episode_id"],)).fetchone()[0] == "queued",
              "propose on episode clips: message in episode clips / episode-h; episodes rows -> 'queued'", r.stdout + r.stderr)
        man, paths, probs = gpu_job.build_manifest(con, bid)
        v = next(x for x in man["videos"] if x["video_id"] == e2["episode_id"])
        check(probs == [] and v["window_start_s"] == e2["start_s"] and v["episode_id"] == e2["episode_id"] and v["meeting_id"] == "sanjose:2026-09-15"
              and v["body"] == "sanjose" and v["tags"] == e2["tags"] and v["kind"] == "episode",
              "build_manifest: window_start_s = the episode's start in the meeting (was hardcoded 0); episode_id / meeting_id / body / tags added", str(v))
        doc = gpu_job.make_doc(v, [{"start": 0, "end": 30, "text": "Good evening, Mayor. Jane Doe, Deputy Director of Planning.", "speaker": "SPEAKER_01"},
                                   {"start": 31, "end": 60, "text": "We'll now take public comment. You have two minutes.", "speaker": "SPEAKER_00"},
                                   {"start": 61, "end": 120, "text": "My name is Pat Lee and I live on Elm Street. I urge you to deny this.", "speaker": "SPEAKER_02"},
                                   {"start": 121, "end": 150, "text": "Thank you. I'll close public comment. Please call the roll.", "speaker": "SPEAKER_00"}],
                              gpu_seconds=2.0, batch_id=bid)
        check(gpu_job.validate_doc(doc, e2["episode_id"]) == [] and doc["episode_id"] == e2["episode_id"] and doc["window_start_s"] == e2["start_s"]
              and doc["episode_start_s"] == e2["start_s"] and doc["body"] == "sanjose" and doc["tags"] == e2["tags"] and doc["kind"] == "episode",
              "make_doc: contract intact; episode keys added at the top level only")
        # ingest: episodes row -> done with transcript_path, speaker_roles rows, roles in the file
        for vid_ in man["videos"]:
            state.transition(con, vid_["video_id"], "queued", "transcribing")
        staging = d / "staging"; staging.mkdir()
        (staging / f"{e2['episode_id']}.json").write_text(json.dumps(doc))
        os.environ["WHISPER_DIR"] = str(d / "whisper")
        got = gpu_job.ingest_new(con, bid, staging, set(), log=lambda *a: None)
        os.environ.pop("WHISPER_DIR", None)
        ep_row = dict(con.execute("SELECT * FROM episodes WHERE episode_id = ?", (e2["episode_id"],)).fetchone())
        sr = {r["speaker"]: r["role"] for r in con.execute("SELECT speaker, role FROM speaker_roles WHERE episode_id = ?", (e2["episode_id"],))}
        loaded = json.loads((d / "whisper" / f"{e2['episode_id']}.json").read_text())
        check(got["done"] == 1 and ep_row["status"] == "done" and ep_row["transcript_path"] and Path(ep_row["transcript_path"]).exists()
              and sr.get("SPEAKER_02") == "public" and sr.get("SPEAKER_00") == "dais" and loaded["roles"]["SPEAKER_02"]["role"] == "public"
              and loaded["roles_cues_version"].startswith("cues-") and "roles_summary" in loaded,
              "ingest_new on an episode: episodes row 'done' + transcript_path, speaker_roles rows written, roles keys added to the file", f"{got} {ep_row['status']} {sr}")
        r = subprocess.run([PY, str(ROOT / "pipeline" / "gpu_batch.py"), "list"], capture_output=True, text=True, env=env)
        check(r.returncode == 0 and bid in r.stdout, "gpu_batch list still works with kind=episode rows")
        for k in ("AUDIO_DIR", "OPUS_DIR"):
            os.environ.pop(k, None)
    else:
        check(True, "episode cuts / manifest / ingest skipped (no ffmpeg)")
    os.environ.pop("SUBS_DIR", None); os.environ.pop("YTDLP_BIN", None)

    # -- QA rules for meeting episodes
    q = d / "qa"; q.mkdir()
    seg = lambda i, t, sp="SPEAKER_00": {"start": i * 4.0, "end": i * 4.0 + 2.2, "text": t, "speaker": sp}   # 55% coverage
    segs = [seg(i, f"sentence {i} about the housing element") for i in range(2, 60)]
    ep_doc = {"video_id": "E1", "kind": "episode", "episode_id": "E1", "transcribed_s": 240.0, "segments": segs}
    show_doc = {"video_id": "S1", "transcribed_s": 240.0, "segments": segs}
    (q / "E1.json").write_text(json.dumps(ep_doc)); (q / "S1.json").write_text(json.dumps(show_doc))
    re_, rs = qa.check_video(q / "E1.json"), qa.check_video(q / "S1.json")
    check(re_["kind"] == "episode" and not any("coverage" in m for m in re_["warn"]) and any("coverage" in m for m in rs["warn"]),
          "QA: 55% coverage is a WARN for a show but not for a meeting episode (WARN 50%, FAIL 30% unchanged)", f"{re_['warn']} {rs['warn']}")
    check(re_["edge_speech_start_s"] == 8.0 and abs(re_["edge_speech_end_s"] - 1.8) < 0.01 and any("edge" in m for m in re_["info"]) and re_["fail"] == []
          and not any("edge" in m for m in re_["warn"]),
          "QA: speech within 5 s of a clip edge is recorded and reported as info, not WARN", str(re_.get("info")))
    stretched = dict(ep_doc, video_id="E2", segments=segs + [{"start": 240, "end": 300, "text": "thank you very much", "speaker": "SPEAKER_00"}], transcribed_s=300.0)
    (q / "E2.json").write_text(json.dumps(stretched))
    r2 = qa.check_video(q / "E2.json")
    check(r2["stretched_segs"] == 1 and any("near-silence" in m for m in r2["warn"]),
          "QA: one segment of 4 words over 60 s is WARNed as text over dead air", str(r2["warn"]))
    low = dict(ep_doc, video_id="E3", segments=[seg(i, "x") for i in range(0, 60, 4)])                          # ~14% coverage
    (q / "E3.json").write_text(json.dumps(low))
    check(any("coverage" in m for m in qa.check_video(q / "E3.json")["fail"]), "QA: coverage under 30% still FAILs an episode")

    # -- dashboard: episodes panel, silence alert, approval wording
    from dashboard import status as dstatus                 # noqa: E402
    con.execute("UPDATE videos SET meeting_date = '2026-01-10' WHERE video_id = 'gr-oak-1'")
    discover.upsert_source(con, "silentcity", "u", "meeting", "^x", "youtube", "UCy:streams", "council")
    s = dstatus.collect(mdb, cfg=dict(dstatus.settings(), cap_min=45, silent_days=60), now=datetime_now())
    bodies = {b["body"]: b for b in s["meetings"]["bodies"]}
    txt = dstatus.as_text(s)
    check(bodies["sanjose"]["meetings"] == 1 and bodies["sanjose"]["located"] >= 2 and s["meetings"]["totals"]["bodies"] == 3
          and "Meetings:" in txt and "sanjose:" in txt,
          "status.collect: per-body meetings / episodes located / cut / transcribed; digest carries the episodes lines", txt)
    kinds_ = {a["kind"]: a["text"] for a in s["alerts"]}
    check("body_silent" in kinds_ and ("oakland" in kinds_["body_silent"] or "silentcity" in kinds_["body_silent"]),
          "status: a body with no meeting in 60 days (or none ever) raises body_silent", str(s["alerts"]))
    if have_ffmpeg:
        appr = [a for a in s["alerts"] if a["kind"] == "approval"]
        check(appr and "episode clips" in appr[0]["text"] and "episode-h" in appr[0]["text"],
              "status: an approval alert for an episode batch states episode-hours", str(appr))
    con.close()


def section_a0(tmp):
    print("\n20. Phase A0: episode loader, term verification, redaction, figures")
    import load_episodes, a0_terms, a0_plots, load_db
    d = tmp / "a0"; (d / "whisper").mkdir(parents=True)
    state = d / "state.sqlite"
    con = sqlite3.connect(state)
    con.executescript("""
      CREATE TABLE videos (video_id TEXT PRIMARY KEY, kind TEXT, speaker TEXT, meeting_date TEXT, month TEXT,
                           duration_s REAL, status TEXT, platform TEXT);
      CREATE TABLE episodes (episode_id TEXT PRIMARY KEY, meeting_id TEXT, video_id TEXT, start_s REAL, end_s REAL,
                             core_start_s REAL, core_end_s REAL, tags TEXT, label TEXT, status TEXT,
                             method TEXT, agenda_item TEXT, transcript_path TEXT);
      CREATE TABLE speaker_roles (episode_id TEXT, speaker TEXT, role TEXT, confidence REAL, method TEXT,
                                  talk_s REAL, cues_version TEXT, PRIMARY KEY (episode_id, speaker));""")
    con.executemany("INSERT INTO videos VALUES (?,?,?,?,?,?,?,?)", [
        ("M1", "meeting", "bart", "2026-08-13", "2026-08", 7200, "cut", "youtube"),
        ("M2", "meeting", "sanjose", "2026-09-01", "2026-09", 3600, "cut", "youtube"),
        ("M3", "meeting", "sanjose", "2026-09-15", "2026-09", 3600, "failed", "youtube")])
    pub = ("My name is Ann Lee and I live in the neighborhood. The traffic on our street is terrible and parking "
           "is impossible. Please deny this project, Councilmember Smith. Thank you for hearing me.")
    dais = ("Thank you, next speaker. Staff recommends approval of the housing element update; the density bonus "
            "and affordable housing units are consistent with state law and the RHNA allocation. Ridership on BART "
            "is up and the fare gates are working. Training for operators continues.")
    def doc(vid, body, segs, roles=None):
        out = {"video_id": vid, "speaker": body, "month": "2026-09", "duration_s": 300, "transcribed_s": 300,
               "diarized": True, "host_speaker": "SPEAKER_00", "host_confidence": 0.5, "model": "t",
               "segments": [{"start": 10.0 * i, "end": 10.0 * i + 8, "speaker": sp, "text": tx} for i, (sp, tx) in enumerate(segs)]}
        if roles:
            out["roles"] = roles
        return out
    v = lambda t, i: f"{t} Point {i}."          # distinct lines, so collapse_loops (identical text) leaves them alone
    files = {
        "M1_e1": doc("M1_e1", "bart", [("SPEAKER_00", v(dais, 1)), ("SPEAKER_01", v(pub, 1)), ("SPEAKER_00", v(dais, 2)), ("SPEAKER_00", v(dais, 3))]),
        "M2_e1": doc("M2_e1", "sanjose", [("SPEAKER_00", v(dais, 1)), ("SPEAKER_01", v(pub, 1)), ("SPEAKER_01", v(pub, 2))]),
        "M2_e2": doc("M2_e2", "sanjose", [("SPEAKER_00", v(dais, 1)), ("SPEAKER_00", v(dais, 2)), ("SPEAKER_00", v(dais, 3))]),
    }
    for eid, dd in files.items():
        (d / "whisper" / f"{eid}.json").write_text(json.dumps(dd))
    # M2_e2 is located by the agenda index, everything else by the classifier:
    # the fixture spans BOTH instrument facets (analysis/facets.py), so the
    # guard and the per-route exposure split are exercised on real rows.
    con.executemany("INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", [
        ("M1_e1", "bart:2026-08-13", "M1", 100, 400, 120, 380, "transit", "about_transit", "done", "classifier", None, str(d / "whisper" / "M1_e1.json")),
        ("M2_e1", "sanjose:2026-09-01", "M2", 0, 300, 0, 300, "density", "about_density", "done", "classifier", None, str(d / "whisper" / "M2_e1.json")),
        ("M2_e2", "sanjose:2026-09-01", "M2", 500, 800, 500, 800, "density", "about_density", "done", "agenda_index", None, str(d / "whisper" / "M2_e2.json")),
        ("M2_e3", "sanjose:2026-09-01", "M2", 900, 960, 900, 960, "density", "about_density", "located", "classifier", None, None),
        ("M2_m1", "sanjose:2026-09-01", "M2", 1000, 1060, 1000, 1060, "density", "mention_only", "mention", "classifier", None, None)])
    con.executemany("INSERT INTO speaker_roles VALUES (?,?,?,?,?,?,?)", [
        ("M1_e1", "SPEAKER_00", "dais", 0.9, "cue_phrase", 24.0, "cues-x"),
        ("M1_e1", "SPEAKER_01", "public", 0.8, "cue_phrase", 8.0, "cues-x")])
    con.commit(); con.close()

    db = d / "corpus.sqlite"
    # the side corpus first (make load), then the episodes on top (make load-episodes)
    subprocess.run([PY, str(ROOT / "analysis" / "load_db.py"), "--db", str(db)], check=True, capture_output=True)
    c, n_ep, n_seg, n_words, sources, n_meet = load_episodes.load(state, db, load_db.collapse_loops)
    check(n_ep == 3 and n_seg == 10 and n_meet == 3, "load_episodes: 3 done episodes, 10 segments, 3 meetings", f"{n_ep} {n_seg} {n_meet}")
    check(sources == {"db": 1, "computed": 2}, "load_episodes: roles from speaker_roles when present, computed otherwise", str(sources))
    roles = dict(c.execute("SELECT speaker, role FROM episode_speakers WHERE episode_id = 'M1_e1'").fetchall())
    check(roles == {"SPEAKER_00": "dais", "SPEAKER_01": "public"}, "load_episodes: DB roles joined by (episode, speaker label)", str(roles))
    r2 = dict(c.execute("SELECT speaker, role FROM episode_speakers WHERE episode_id = 'M2_e1'").fetchall())
    check(r2.get("SPEAKER_01") == "public", "load_episodes: computed roles find the 'my name is' public speaker", str(r2))
    talk, words = c.execute("SELECT talk_s, words FROM episode_speakers WHERE episode_id = 'M1_e1' AND speaker = 'SPEAKER_00'").fetchone()
    exp_words = sum(len(load_episodes.WORD_RE.findall(v(dais, i).lower())) for i in (1, 2, 3))
    check(abs(talk - 24.0) < 0.01 and words == exp_words, "load_episodes: talk seconds and words per (episode, speaker)", f"{talk} {words} vs {exp_words}")
    m = c.execute("SELECT n_about, n_mention, n_done, density_core_s FROM meetings WHERE video_id = 'M2'").fetchone()
    check(m == (3, 1, 2, 660.0), "load_episodes: meeting exposure counts located / mention / done and core seconds", str(m))
    meth = dict(c.execute("SELECT episode_id, method FROM episodes").fetchall())
    check(meth == {"M1_e1": "classifier", "M2_e1": "classifier", "M2_e2": "agenda_index"},
          "load_episodes: episodes.method carries the locate route through from state.sqlite", str(meth))
    ex = {r[0]: r[1:] for r in c.execute(
        "SELECT method, n_about, n_mention, n_done, density_core_s FROM meeting_exposure WHERE video_id = 'M2'")}
    check(ex == {"classifier": (2, 1, 1, 360.0), "agenda_index": (1, 0, 1, 300.0)},
          "load_episodes: meeting_exposure splits the numerator per route and sums back to meetings", str(ex))
    check(c.execute("SELECT COUNT(*) FROM videos").fetchone()[0] > 0, "load_episodes: the side-corpus tables in the same file are untouched")
    c.close()

    segs = a0_terms.load_segments(db)
    cnt = a0_terms.count_terms(["train", "housing element", "fare gate", "neighborhood", "parking minimum"], segs)
    check(cnt["train"]["total"] == 0, "a0_terms: 'train' does not count 'Training' (whole word)")
    check(cnt["housing element"]["total"] == 7 and cnt["fare gate"]["total"] == 7, "a0_terms: multi-word phrases and optional 's count",
          f"{cnt['housing element']['total']} {cnt['fare gate']['total']}")
    check(cnt["neighborhood"]["by_role"]["public"] == 3 and cnt["neighborhood"]["by_role"]["dais"] == 0
          and cnt["neighborhood"]["episodes"] == {"M1_e1", "M2_e1"}, "a0_terms: counts split by role and episode")
    check(cnt["parking minimum"]["total"] == 0, "a0_terms: a term that never occurs is reported as zero, not dropped")
    frames = a0_terms.load_frames(ROOT / "plan" / "frames.yaml")
    check(len(frames) == 12 and all(frames.values()) and "traffic" in frames["traffic_parking"],
          "frames.yaml: 12 frames, every frame has terms, parsed without a YAML library", str(list(frames)))
    red = a0_terms.redact_names("My name is Ann Lee and Councilmember Smith knows San Jose and the Housing Element.")
    check("Ann Lee" not in red and "A. L." in red and "Councilmember S." in red and "San Jose" in red and "Housing Element" in red,
          "a0_terms.redact_names: people become initials, places and titles stay", red)
    names = a0_terms.name_tokens(segs)
    check({"ann", "lee", "smith"} <= names and "housing" not in names and "neighborhood" not in names,
          "a0_terms.name_tokens: name words flagged, content words not", str(sorted(names)))
    lines = a0_terms.kwic("traffic", segs, n=5)
    check(len(lines) == 3 and all("**traffic**" in l[3] for l in lines) and all("Ann Lee" not in l[3] for l in lines),
          "a0_terms.kwic: one line per hit, node marked, names redacted", str(lines))

    out = d / "out"
    import io, contextlib
    import facets

    # -- the facet guard (analysis/facets.py) ---------------------------------
    # This fixture has roles from speaker_roles for one episode and computed
    # roles for two, so it spans the role_source facet exactly as the real pilot
    # does. Every entry point must refuse to draw over both instruments.
    fcon = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    check(sorted(facets.present(fcon)["role_source"]) == ["computed", "db"]
          and sorted(facets.present(fcon)["method"]) == ["agenda_index", "classifier"],
          "facets.present: the fixture spans two role instruments and two locate routes", str(facets.present(fcon)))
    fcon.close()
    for label, fn in (("a0_terms.run",
                       lambda: a0_terms.run(db, out, [ROOT / "plan" / "keywords_density.txt"], ROOT / "plan" / "frames.yaml")),
                      ("a0_plots.main",
                       lambda: a0_plots.main(["--db", str(db), "--out-dir", str(out / "refused"), "--frames", str(ROOT / "plan" / "frames.yaml")]))):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fn()
            refused = False
        except SystemExit as e:
            refused = "refusing to draw a pooled figure" in str(e)
        check(refused, f"facets: {label} REFUSES a corpus spanning two role instruments (no accidental pooled figure)")

    # -- one facet slice draws normally, and is stamped with its level --------
    sl = d / "slice_db.sqlite"
    n_ep, n_words = facets.slice_db(db, sl, "role_source", "computed")
    scon = sqlite3.connect(f"file:{sl}?mode=ro", uri=True)
    check(n_ep == 2 and facets.stamped(scon) == ("role_source", "computed"),
          "facets.slice_db: keeps only that level's episodes and stamps the corpus", f"{n_ep} eps, {facets.stamped(scon)}")
    check(scon.execute("SELECT COUNT(*) FROM episode_segments WHERE episode_id NOT IN "
                       "(SELECT episode_id FROM episodes)").fetchone()[0] == 0,
          "facets.slice_db: segments and speakers are cut with their episodes, never orphaned")
    scon.close()
    with contextlib.redirect_stdout(io.StringIO()):
        rc_slice = a0_plots.main(["--db", str(sl), "--out-dir", str(out / "sliced"), "--frames", str(ROOT / "plan" / "frames.yaml")])
    check(rc_slice == 0 and not list((out / "sliced").glob("POOLED_*")) and (out / "sliced" / "a0_rates.png").exists(),
          "a0_plots: a stamped facet slice draws without --pool and is not stamped POOLED")

    # -- the pooled set is still available, explicitly, and says so -----------
    with contextlib.redirect_stdout(io.StringIO()):
        krows, frows, _ = a0_terms.run(db, out, [ROOT / "plan" / "keywords_density.txt", ROOT / "plan" / "keywords_transit.txt"],
                                       ROOT / "plan" / "frames.yaml", pool=True)
    b = a0_terms.bucket(krows)
    check((out / "a0_term_counts.csv").exists() and (out / "a0_frame_counts.csv").exists() and (out / "a0_kwic.md").exists()
          and len(b["zero"]) > 200 and any(r["term"] == "housing element" and r["total"] == 7 for r in krows),
          "a0_terms.run: writes the three files; never-occurring terms are listed", f"zero={len(b['zero'])}")
    check("POOLED" in (out / "a0_kwic.md").read_text(), "a0_terms.run --pool: the KWIC file says on its face that it is pooled")
    with contextlib.redirect_stdout(io.StringIO()):
        rc = a0_plots.main(["--db", str(db), "--out-dir", str(out / "a0"), "--frames", str(ROOT / "plan" / "frames.yaml"), "--pool"])
    pngs = sorted(p.name for p in (out / "a0").glob("*.png"))
    want = {"POOLED_a0_rates.png", "POOLED_a0_top_terms_role.png", "POOLED_a0_top_terms_public_body.png",
            "POOLED_a0_keyness_public_vs_dais.png", "POOLED_a0_frames_role_body.png", "POOLED_a0_frames_by_date.png",
            "POOLED_a0_speakers.png", "POOLED_a0_density_vs_transit.png"}
    check(rc == 0 and want <= set(pngs), "a0_plots --pool: every figure renders, every filename stamped POOLED", str(pngs))
    csvs = {p.name for p in (out / "a0").glob("*.csv")}
    check({"POOLED_a0_speakers.csv", "POOLED_a0_keyness_public_vs_dais.csv", "POOLED_a0_frames_role_body.csv"} <= csvs,
          "a0_plots: CSV twin per figure", str(csvs))
    with open(out / "a0" / "POOLED_a0_keyness_public_vs_dais.csv") as f:
        kn = {r["term"]: r for r in csv.DictReader(f)}
    check(all(kn[w]["name_like"] == "1" for w in ("ann", "lee") if w in kn), "a0_plots: name-like tokens flagged in the keyness table")


def section_validation(tmp):
    print("\n21. transcription validation (analysis/validate_transcription.py): number rule, lag, per-minute / per-role attribution, silence, diarization flags")
    import validate_transcription as vt
    import numpy as np
    # normalisation: digits and words meet in the middle; "2026" is a year; sb79 splits
    check(vt.norm_tokens("SB 79 passed in 2026, 3,500 units at 15% for $2.5 million, the 21st time")
          == "sb seventy nine passed in twenty twenty six three thousand five hundred units at fifteen percent for two point five million dollars the twenty first time".split(),
          "norm_tokens: digits, years, commas, %, $, decimals and ordinals become words", str(vt.norm_tokens("SB 79 passed in 2026, 3,500 units at 15% for $2.5 million, the 21st time")))
    check(vt.norm_tokens("Twenty twenty-six") == vt.norm_tokens("2026") and vt.norm_tokens("[ __ ] >> hello [Music]") == ["hello"],
          "norm_tokens: '2026' == 'twenty twenty-six'; mask, speaker mark and non-speech tags dropped")
    # rolling auto captions parse to unique text; a Granicus track keeps every cue
    auto = (tmp / "v_auto.vtt")
    auto.write_text("WEBVTT\nKind: captions\n\n00:00:00.000 --> 00:00:03.000 align:start position:0%\n \nthe<00:00:01.000><c> board</c><00:00:02.000><c> meets</c>\n\n"
                    "00:00:03.000 --> 00:00:03.010 align:start position:0%\nthe board meets\n \n\n"
                    "00:00:03.010 --> 00:00:06.000 align:start position:0%\nthe board meets\ntoday<00:00:04.000><c> at</c><00:00:05.000><c> noon</c>\n")
    cues, kind = vt.parse_cues(auto)
    check(kind == "auto" and [c[2] for c in cues] == ["the board meets", "today at noon"], "parse_cues: rolling auto captions -> each phrase once", str(cues))
    gran = tmp / "v_gran.vtt"
    gran.write_text("WEBVTT\n\n00:00:10.000 --> 00:00:12.000\nGood\n\n00:00:12.000 --> 00:00:11.900\nafternoon everyone\n")
    cues, kind = vt.parse_cues(gran)
    check(kind == "manual" and len(cues) == 2 and cues[1][1] >= cues[1][0], "parse_cues: Granicus cues taken whole, end clamped to start", str(cues))
    # alignment: lag, per-minute and per-role attribution, direction of the rate
    ref_items = [(0.0, 30.0, "the board meets today at noon to discuss the fare gates", None),
                 (60.0, 90.0, "my name is Ann Lee and I live in Oakland please deny the project", None)]
    hyp_items = [(5.0, 35.0, "the board meets today at noon to discuss the fair gates", "dais"),
                 (65.0, 95.0, "my name is Ann Lee and I live in Oakland please deny the project thank you next speaker", "public")]
    ref_w = vt.timed_words(ref_items)
    hyp_w = vt.timed_words(hyp_items)
    ops = vt.align([w for _, w, _ in ref_w], [w for _, w, _ in hyp_w])
    lag, n = vt.lag_estimate(ref_w, hyp_w, ops)
    check(abs(lag - 5.0) < 1.0 and n >= 20, "lag_estimate: transcript 5 s behind the captions -> +5 s", f"lag={lag:.2f} n={n}")
    ref_w = [(t, w, "dais" if t < 60 else "public") for t, w, _ in ref_w]
    mins = vt.per_minute(ref_w, hyp_w, ops, 2)
    check(mins[0]["err"] == 1 and mins[0]["ref"] == 11 and mins[1]["err"] == 4 and mins[1]["ref"] == 14,
          "per_minute: one substitution charged to minute 0, four insertions to minute 1", str(mins))
    check(mins[0]["wer"] is None and mins[1]["wer"] is None, "per_minute: minutes under 20 caption words are not rated")
    roles = vt.per_role(ref_w, hyp_w, ops)
    check(roles["dais"] == [1, 11] and roles["public"] == [4, 14], "per_role: errors follow the role of the word", str(roles))
    ref, hyp = "a b c d e f g h i j".split(), "a b c d e f g h i j x y z".split()
    check(vt.edit_distance(ref, hyp) / len(ref) > vt.edit_distance(hyp, ref) / len(hyp), "rate divides by the caption side, so inserted text raises it")
    runs = vt.divergence_runs(ref_w, hyp_w, ops)
    check(runs and runs[0][0] == 4 and "[name]" in vt.redact("my name is Ann Lee and"), "divergence_runs: largest run first; names redacted in anything printed", str(runs[:1]))
    # low-energy audio: 60 s of speech-level noise with a 30 s hole
    rng = np.random.default_rng(1)
    rms = np.full(120, -20.0) + rng.normal(0, 1, 120)
    rms[40:100] = -60.0                                     # frames of 0.5 s: 20-50 s quiet
    stretches, thr = vt.low_energy_stretches(rms)
    check(len(stretches) == 1 and abs(stretches[0][0] - 20) <= 1 and abs(stretches[0][1] - 50) <= 1, "low_energy_stretches: the 30 s hole, nothing else", str(stretches))
    segs = [{"start": 22.0, "end": 45.0, "text": "thank you for watching please subscribe"}, {"start": 0.0, "end": 19.0, "text": "real speech here"},
            {"start": 48.0, "end": 58.0, "text": "back to speech now"}]
    hits = vt.text_over_silence(segs, stretches)
    check(len(hits) == 1 and hits[0][0] == 22.0, "text_over_silence: only the segment inside the quiet stretch", str(hits))
    loop = [{"start": i * 2.0, "end": i * 2.0 + 2, "text": "we're not gonna make it"} for i in range(5)]
    check(vt.repeated_ngrams(loop) and vt.repeated_ngrams(loop)[0][1] == 5 and not vt.repeated_ngrams(segs), "repeated_ngrams: 5 repeats in 10 s caught, plain speech clean")
    # diarization flags
    merged = [{"start": i * 20.0, "end": i * 20.0 + 15, "speaker": "SPEAKER_01", "text": f"my name is P{i} and I live in town, deny it"} for i in range(5)]
    merged.insert(0, {"start": -1, "end": 0, "speaker": "SPEAKER_00", "text": "we'll now take public comment, you have two minutes"})
    d = vt.diarization_check(merged, {"SPEAKER_00": {"role": "dais"}, "SPEAKER_01": {"role": "public"}})
    check(d["intro_events"] == 5 and d["n_public_clusters"] == 1 and any(f.startswith("merge?") for f in d["flags"]), "diarization_check: 5 intros in one public cluster -> merge flag", str(d))
    split = [{"start": i * 5.0, "end": i * 5.0 + 4, "speaker": f"SPEAKER_{i:02d}", "text": "and the next item is the budget"} for i in range(6)]
    d = vt.diarization_check(split, {})
    check(d["sliver_clusters"] == 6 and any(f.startswith("split?") for f in d["flags"]), "diarization_check: six 4-second clusters -> split flag", str(d))
    d = vt.diarization_check(hyp_items and [{"start": 0, "end": 30, "speaker": "SPEAKER_00", "text": "the board meets"}], {})
    check(not d["flags"], "diarization_check: one long cluster, no flags")


def section_review(tmp):
    print("\n22. 2026-09-21 review regressions (docs/REVIEW-2026-09-21.md): money path exits, reclaim scope, launchd loop, stuck rows")
    sys.path.insert(0, str(ROOT / "pipeline"))
    import types, signal, time
    import state, gpu_batch, gpu_job, requeue, episodes, granicus, classify, segment as sg  # noqa: E402
    from dashboard import status as dstatus                 # noqa: E402
    d = tmp / "review"; d.mkdir()
    db = d / "state.sqlite"
    con = state.connect(db)
    now = state.now_iso()
    sent = []
    saved_notify, gpu_batch.notify = gpu_batch.notify, lambda level, msg, **k: sent.append((level, msg)) or True

    # -- R1 teardown: ANY exception from the terminate call marks ORPHAN and pages Puck (was a narrow tuple)
    con.execute("INSERT INTO pods VALUES ('p-json', ?, NULL, 'running')", (now,))
    saved_gql = gpu_batch.runpod_graphql
    gpu_batch.runpod_graphql = lambda *a, **k: json.loads("<html>cloudflare</html>")      # ValueError, not in the old tuple
    try:
        rc = gpu_batch.teardown(con, "p-json")
    finally:
        gpu_batch.runpod_graphql = saved_gql
    check(rc == 1 and con.execute("SELECT status FROM pods WHERE pod_id = 'p-json'").fetchone()[0] == "ORPHAN"
          and sent and sent[-1][0] == "error" and "p-json" in sent[-1][1],
          "teardown: a non-URLError failure (Cloudflare HTML -> ValueError) still -> pods.status ORPHAN + WhatsApp error", f"rc={rc} sent={sent}")
    # -- R2 teardown is idempotent: an already-terminated pod is not billed twice nor flipped to ORPHAN
    con.execute("INSERT INTO pods VALUES ('p-done', ?, ?, 'terminated')", (now, now))
    calls = []

    def not_found(*a, **k):
        calls.append(a)
        raise RuntimeError("RunPod: pod not found")
    gpu_batch.runpod_graphql = not_found
    try:
        rc = gpu_batch.teardown(con, "p-done")
    finally:
        gpu_batch.runpod_graphql = saved_gql
    check(rc == 0 and not calls and con.execute("SELECT status FROM pods WHERE pod_id = 'p-done'").fetchone()[0] == "terminated"
          and con.execute("SELECT COUNT(*) FROM spend WHERE pod_id = 'p-done'").fetchone()[0] == 0,
          "teardown on a pod already 'terminated' here: no API call, no second spend row, not flipped to ORPHAN")

    # -- R3 launch: Ctrl-C and SIGTERM take the failure path (batch failed, videos released, pod torn down)
    def seed_batch(bid, vid):
        con.execute("INSERT INTO videos (video_id, speaker, channel_url, month, crawled_at, in_frame, status, updated_at, kind, episode_id, opus_path) "
                    "VALUES (?, 'oakland', 'u', '2026-09', ?, 1, 'queued', ?, 'episode', ?, ?)", (vid, now, now, vid, str(d / "x.opus")))
        con.execute("INSERT INTO episodes (episode_id, meeting_id, video_id, start_s, end_s, tags, method, status, updated_at) "
                    "VALUES (?, 'oakland:2026-09-01', 'M1', 0, 60, 'density', 'classifier', 'queued', ?)", (vid, now))
        con.execute("INSERT INTO batches (batch_id, created_at, status, n_videos, audio_hours, est_usd) VALUES (?, ?, 'approved', 1, 0.02, 0.05)", (bid, now))
        con.execute("INSERT INTO batch_videos VALUES (?, ?)", (bid, vid))
    torn = []
    saved_checks, saved_teardown = gpu_batch.launch_checks, gpu_batch.teardown
    saved_job = sys.modules.get("gpu_job")
    gpu_batch.launch_checks = lambda con_, bid: []
    gpu_batch.teardown = lambda con_, pod_id, **k: torn.append(pod_id) or 0
    stub = types.ModuleType("gpu_job")

    def create_pod(con_, bid):
        con_.execute("INSERT INTO pods VALUES (?, ?, NULL, 'running')", (f"pod-{bid}", state.now_iso()))
        return f"pod-{bid}"
    stub.create_pod = create_pod
    stub.describe_plan = lambda con_, bid: "plan"
    prev_term = signal.getsignal(signal.SIGTERM)
    try:
        seed_batch("bK", "M1_e1")
        def ctrl_c(con_, bid, pod_id):
            raise KeyboardInterrupt
        stub.run = ctrl_c
        sys.modules["gpu_job"] = stub
        try:
            gpu_batch.launch(con, "bK")
            outcome = "returned"
        except KeyboardInterrupt:
            outcome = "KeyboardInterrupt"
        row = con.execute("SELECT status FROM batches WHERE batch_id = 'bK'").fetchone()[0]
        v = con.execute("SELECT status FROM videos WHERE video_id = 'M1_e1'").fetchone()[0]
        e = con.execute("SELECT status FROM episodes WHERE episode_id = 'M1_e1'").fetchone()[0]
        check(outcome == "KeyboardInterrupt" and row == "failed" and v == "transcoded" and e == "cut" and torn == ["pod-bK"],
              "launch: Ctrl-C during the job -> batch 'failed', video back to 'transcoded', episodes row 'cut', pod torn down (rows were stuck before)",
              f"{outcome} batch={row} video={v} episode={e} torn={torn}")
        seed_batch("bL", "M1_e2")
        def sigterm(con_, bid, pod_id):
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(2)
            return {"gpu_minutes": 1, "actual_usd": 0.01, "note": "should not get here"}
        stub.run = sigterm
        try:
            gpu_batch.launch(con, "bL")
            outcome = "returned"
        except gpu_batch.Interrupted as ex:
            outcome = f"Interrupted: {ex}"
        row = con.execute("SELECT status, note FROM batches WHERE batch_id = 'bL'").fetchone()
        v = con.execute("SELECT status FROM videos WHERE video_id = 'M1_e2'").fetchone()[0]
        check(outcome.startswith("Interrupted") and row[0] == "failed" and "signal" in (row[1] or "") and v == "transcoded" and torn == ["pod-bK", "pod-bL"]
              and signal.getsignal(signal.SIGTERM) == prev_term,
              "launch: SIGTERM (launchctl / exec host) -> same failure path, pod torn down, previous handler restored", f"{outcome} {row} {v} {torn}")
    finally:
        gpu_batch.launch_checks, gpu_batch.teardown = saved_checks, saved_teardown
        if saved_job is not None:
            sys.modules["gpu_job"] = saved_job
        else:
            sys.modules.pop("gpu_job", None)
    gpu_batch.notify = saved_notify

    # -- R4 reclaim_stuck is scoped to the worker's own rows
    con.execute("INSERT INTO videos (video_id, speaker, channel_url, month, crawled_at, in_frame, status, updated_at, kind) VALUES "
                "('SHOWDL', 'hasanabi', 'u', '2026-09', ?, 1, 'downloading', ?, 'show')", (now, now))
    con.execute("INSERT INTO videos (video_id, speaker, channel_url, month, crawled_at, in_frame, status, updated_at, kind) VALUES "
                "('MTGDL', 'oakland', 'u', '2026-09', ?, 0, 'downloading', ?, 'meeting')", (now, now))
    n1 = state.reclaim_stuck(con, "downloading", "classified", kind="meeting")          # episodes.py at start
    n2 = state.reclaim_stuck(con, "downloading", "discovered", in_frame=1)             # downloader at start
    st_ = {r[0]: r[1] for r in con.execute("SELECT video_id, status FROM videos WHERE video_id IN ('SHOWDL', 'MTGDL')")}
    check(n1 == 1 and n2 == 1 and st_ == {"SHOWDL": "discovered", "MTGDL": "classified"},
          "reclaim_stuck: episodes.py's sweep leaves the downloader's show row alone, and vice versa (a show sent to 'classified' was stuck for ever)", str(st_))

    # -- R5 downloader under launchd (--loop): a bot check exits 0 so KeepAlive does not retry YouTube every 10 min
    con.execute("INSERT INTO videos (video_id, speaker, channel_url, month, crawled_at, in_frame, status, updated_at) VALUES "
                "('BOT1', 'hasanabi', 'u', '2026-09', ?, 1, 'discovered', ?)", (now, now))
    fake = d / "yt-dlp"
    fake.write_text("#!/bin/sh\necho \"ERROR: [youtube] BOT1: Sign in to confirm you're not a bot\" >&2\nexit 1\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ, STATE_DB_PATH=str(db), YTDLP_BIN=str(fake), NOTIFY_BACKEND="none", AUDIO_DIR=str(d / "audio"))
    r_loop = subprocess.run([PY, str(ROOT / "pipeline" / "downloader.py"), "--loop", "--max", "1"], capture_output=True, text=True, env=env)
    r_once = subprocess.run([PY, str(ROOT / "pipeline" / "downloader.py"), "--max", "1"], capture_output=True, text=True, env=env)
    check(r_loop.returncode == 0 and "launchd" in r_loop.stdout and r_once.returncode == 3
          and con.execute("SELECT status, attempts FROM videos WHERE video_id = 'BOT1'").fetchone()[:] == ("discovered", 0),
          "downloader: bot check exits 0 under --loop (launchd stays down), 3 by hand; row back to 'discovered', not charged",
          f"loop rc={r_loop.returncode} once rc={r_once.returncode} {r_loop.stdout[-300:]}")

    # -- R6 classify: a meeting with captions='none' goes to 'no_episodes' instead of crashing the run with KeyError
    con.execute("INSERT INTO videos (video_id, speaker, channel_url, month, crawled_at, in_frame, status, updated_at, kind, platform, captions, meeting_id, meeting_date) VALUES "
                "('NOCAP', 'bart', 'u', '2026-09', ?, 0, 'captioned', ?, 'meeting', 'youtube', 'none', 'bart:2026-09-11', '2026-09-11')", (now, now))
    fake_mod = types.ModuleType("anthropic"); fake_mod.Anthropic = lambda: object()
    saved_anthropic = sys.modules.get("anthropic"); sys.modules["anthropic"] = fake_mod
    try:
        rc = classify.main(["run", "--state", str(db), "--cap-usd", "1", "--bodies", "bart"])
    finally:
        if saved_anthropic is not None:
            sys.modules["anthropic"] = saved_anthropic
        else:
            sys.modules.pop("anthropic", None)
    row = con.execute("SELECT status, last_error FROM videos WHERE video_id = 'NOCAP'").fetchone()
    check(rc == 0 and row[0] == "no_episodes", "classify run: captions='none' meeting -> 'no_episodes' (was KeyError 'cached' -> stuck 'captioned')", str(tuple(row)))
    # -- R7 classify cache: a positional id whose bounds moved is NOT reused
    con.execute("INSERT INTO classifier_calls (id, video_id, start_s, end_s, label, model, prompt_version, at) VALUES ('NOCAP_c001', 'NOCAP', 0, 240, 'about_transit', 'm', 'p', ?)", (now,))
    same = classify.cached_calls(con, "NOCAP", "m", "p", [{"id": "NOCAP_c001", "start_s": 0.0, "end_s": 240.0}])
    moved = classify.cached_calls(con, "NOCAP", "m", "p", [{"id": "NOCAP_c001", "start_s": 60.0, "end_s": 300.0}])
    check(set(same) == {"NOCAP_c001"} and moved == {}, "cached_calls: same id + same bounds is cached; same id over a different span is re-classified")

    # -- R8 Granicus: a 403/429 from the stream host stops the cut loop like a YouTube bot check
    con.execute("INSERT INTO videos (video_id, speaker, channel_url, month, crawled_at, in_frame, status, updated_at, kind, platform, meeting_id, meeting_date, duration_s, raw_meta, attempts) VALUES "
                "('gr-x-1', 'gr', 'u', '2026-09', ?, 0, 'classified', ?, 'meeting', 'granicus', 'gr:2026-09-01', '2026-09-01', 5000, '{\"m3u8\": \"https://h/p.m3u8\"}', 0)", (now, now))
    for i in (1, 2):
        con.execute("INSERT INTO episodes (episode_id, meeting_id, video_id, start_s, end_s, core_start_s, core_end_s, tags, method, status, updated_at) "
                    "VALUES (?, 'gr:2026-09-01', 'gr-x-1', ?, ?, ?, ?, 'transit', 'classifier', 'located', ?)", (f"gr-x-1_e{i}", i * 600, i * 600 + 300, i * 600, i * 600 + 300, now))
    slices = []
    saved_slice = granicus.audio_slice
    granicus.audio_slice = lambda m3u8, a, dur, dest: slices.append(a) or (False, ["[https @ 0x1] HTTP error 403 Forbidden", "Server returned 403 Forbidden (access denied)"], 0.2)
    os.environ["AUDIO_DIR"], os.environ["OPUS_DIR"] = str(d / "gaudio"), str(d / "gopus")
    try:
        rc = episodes.main(["--state", str(db), "--bodies", "gr"])
    finally:
        granicus.audio_slice = saved_slice
        os.environ.pop("AUDIO_DIR", None); os.environ.pop("OPUS_DIR", None)
    eps_ = [r[0] for r in con.execute("SELECT status FROM episodes WHERE video_id = 'gr-x-1' ORDER BY episode_id")]
    m = con.execute("SELECT status, attempts FROM videos WHERE video_id = 'gr-x-1'").fetchone()
    check(rc == 3 and len(slices) == 1 and eps_ == ["located", "located"] and tuple(m) == ("classified", 0),
          "episodes.py: Granicus 403 -> ONE request, stop (exit 3), episodes stay 'located', meeting back to 'classified', not charged",
          f"rc={rc} slices={slices} eps={eps_} meeting={tuple(m)}")

    # -- R9 lost clips and unreadable results settle the episodes row too
    seed_batch("bM", "M1_e3")
    state.transition(con, "M1_e3", "queued", "transcribing"); state.set_episode(con, "M1_e3", "transcribing")
    lost = gpu_job.settle_lost(con, "bM", "pod reclaimed")
    e = con.execute("SELECT status, last_error FROM episodes WHERE episode_id = 'M1_e3'").fetchone()
    check(lost == ["M1_e3"] and tuple(e) == ("failed", "pod reclaimed"), "settle_lost: a clip the pod never returned -> video AND episodes row 'failed'", str(tuple(e)))
    seed_batch("bN", "M1_e4")
    state.transition(con, "M1_e4", "queued", "transcribing"); state.set_episode(con, "M1_e4", "transcribing")
    staging = d / "staging"; staging.mkdir(); (staging / "M1_e4.json").write_text("{not json")
    os.environ["WHISPER_DIR"] = str(d / "whisper")
    try:
        got = gpu_job.ingest_new(con, "bN", staging, set(), log=lambda *a: None)
    finally:
        os.environ.pop("WHISPER_DIR", None)
    check(got["qa_fail"] == 1 and con.execute("SELECT status FROM episodes WHERE episode_id = 'M1_e4'").fetchone()[0] == "failed",
          "ingest_new: an unreadable result JSON fails the episodes row as well as the video")
    seed_batch("bO", "M1_e5")
    state.transition(con, "M1_e5", "queued", "transcribing")
    good = gpu_job.make_doc({"video_id": "M1_e5", "speaker": "oakland", "channel_url": "u", "month": "2026-09", "duration_s": 60.0, "transcribed_s": 60.0},
                            [{"start": 0, "end": 30, "text": "hello there", "speaker": "SPEAKER_00"}], gpu_seconds=1.0, batch_id="bO")
    (staging / "M1_e5.json").write_text(json.dumps(good))
    saved_qa = gpu_job._qa
    crashy = types.SimpleNamespace(check_video=lambda f: (_ for _ in ()).throw(ZeroDivisionError("qa bug")))
    gpu_job._qa = lambda: crashy
    os.environ["WHISPER_DIR"] = str(d / "whisper")
    try:
        got = gpu_job.ingest_new(con, "bO", staging, {"M1_e4"}, log=lambda *a: None)
        outcome = "returned"
    except Exception as ex:                                 # noqa: BLE001
        outcome = repr(ex)
    finally:
        gpu_job._qa = saved_qa
        os.environ.pop("WHISPER_DIR", None)
    v5 = con.execute("SELECT status, last_error FROM videos WHERE video_id = 'M1_e5'").fetchone()
    check(outcome == "returned" and got["qa_fail"] == 1 and v5[0] == "failed" and "QA gate crashed" in (v5[1] or "") and not (d / "whisper" / "M1_e5.json").exists(),
          "ingest_new: a QA-gate exception on ONE file fails that video only (it used to fail the batch and tear the pod down)", f"{outcome} {tuple(v5)}")
    # -- R10 requeue of a failed clip puts its episodes row back to 'cut'
    (d / "x.opus").write_text("not really opus")
    out = requeue.requeue(con, "M1_e3")
    check("failed -> transcoded" in out and con.execute("SELECT status FROM episodes WHERE episode_id = 'M1_e3'").fetchone()[0] == "cut",
          "requeue: a failed episode clip -> 'transcoded' and its episodes row -> 'cut' (so propose can pick it up)", out)

    # -- R11 status: a pod whose batch has been 'running' longer than any job can run is an orphan
    from datetime import datetime, timedelta, timezone
    t_now = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
    con.execute("UPDATE batches SET status = 'running', started_at = ?, pod_id = 'p-old' WHERE batch_id = 'bK'", ((t_now - timedelta(hours=20)).isoformat(timespec="seconds"),))
    con.execute("UPDATE batches SET status = 'failed' WHERE batch_id != 'bK'")
    con.execute("DELETE FROM pods"); con.execute("INSERT INTO pods VALUES ('p-old', ?, NULL, 'running')", ((t_now - timedelta(hours=20)).isoformat(timespec="seconds"),))
    s_old = dstatus.collect(db, cfg=dict(dstatus.settings(), cap_min=45), now=t_now)
    con.execute("UPDATE batches SET started_at = ? WHERE batch_id = 'bK'", ((t_now - timedelta(hours=1)).isoformat(timespec="seconds"),))
    s_new = dstatus.collect(db, cfg=dict(dstatus.settings(), cap_min=45), now=t_now)
    k_old = [a for a in s_old["alerts"] if a["kind"] == "orphan_pod"]; k_new = [a for a in s_new["alerts"] if a["kind"] == "orphan_pod"]
    check(len(k_old) == 1 and "bK" in k_old[0]["text"] and k_new == [],
          "status: pod + batch 'running' for 20 h -> CRITICAL orphan_pod naming the batch; 1 h -> legitimate", f"{k_old} {k_new}")

    # -- R12 VTT cues without an hours field parse (WebVTT allows MM:SS.mmm)
    (d / "mmss.vtt").write_text("WEBVTT\n\n05:20.800 --> 05:22.800\nGood evening\n\n1:05:20.800 --> 1:05:22.800\nlater\n")
    segs = sg.read_segments(d / "mmss.vtt")
    check([round(s["start"], 1) for s in segs] == [320.8, 3920.8], "segment.vtt_to_segments: MM:SS.mmm cues are read (were silently zero segments -> 'no_episodes')", str(segs))

    # -- R13 word-level split: a run opened by an untimed word is clamped to the next run's start
    parts = gpu_job.split_by_word_speaker([{"start": 0, "end": 10, "text": "100 x", "speaker": "A",
                                            "words": [{"word": "100", "speaker": "A"}, {"word": "x", "start": 4, "end": 5, "speaker": "B"}]}])
    check([(p["speaker"], p["start"], p["end"]) for p in parts] == [("A", 0, 4), ("B", 4, 5)],
          "split_by_word_speaker: no overlapping runs when a run starts on an untimed word", str(parts))
    con.close()


def section_t2(tmp):
    print("\n23. T2 part 1 (2026-09-22): reconcile, spend caps, classifier model/batches/key, ingest tweaks, vocab, QA episode captions, listings")
    sys.path.insert(0, str(ROOT / "pipeline"))
    import types
    import state, gpu_batch, gpu_job, classify, spend, vocab, discover, granicus  # noqa: E402
    from dashboard import status as dstatus                 # noqa: E402
    d = tmp / "t2"; d.mkdir()
    db = d / "state.sqlite"
    con = state.connect(db)
    now = state.now_iso()
    sent = []
    saved_notify, gpu_batch.notify = gpu_batch.notify, lambda level, msg, **k: sent.append((level, msg)) or True

    def seed_batch(bid, vid, bstatus="launching", vstatus="queued", pod_id=None, pods_row=None):
        con.execute("INSERT INTO videos (video_id, speaker, channel_url, month, crawled_at, in_frame, status, updated_at, kind, episode_id, opus_path) "
                    "VALUES (?, 'oakland', 'u', '2026-09', ?, 1, ?, ?, 'episode', ?, ?)", (vid, now, vstatus, now, vid, str(d / "x.opus")))
        con.execute("INSERT INTO episodes (episode_id, meeting_id, video_id, start_s, end_s, tags, method, status, updated_at) "
                    "VALUES (?, 'oakland:2026-09-01', 'M1', 0, 60, 'density', 'classifier', ?, ?)", (vid, vstatus, now))
        con.execute("INSERT INTO batches (batch_id, created_at, status, n_videos, audio_hours, est_usd, pod_id) VALUES (?, ?, ?, 1, 0.02, 0.05, ?)", (bid, now, bstatus, pod_id))
        con.execute("INSERT INTO batch_videos VALUES (?, ?)", (bid, vid))
        if pods_row:
            con.execute("INSERT INTO pods VALUES (?, ?, NULL, ?)", (pods_row[0], now, pods_row[1]))

    # -- reconcile against a fake RunPod: gone pod -> settled; live pod -> report only; --terminate; never a create
    queries = []
    def fake_gql(pods_by_name, pod_by_id=None):
        def f(query, variables=None, timeout=30):
            queries.append(query)
            assert "podFindAndDeployOnDemand" not in query, "reconcile must never create a pod"
            if "podTerminate" in query:
                return {"podTerminate": None}
            if "pod(input" in query:
                return {"pod": pod_by_id}
            return {"myself": {"pods": pods_by_name}}
        return f
    saved_gql = gpu_batch.runpod_graphql
    try:
        # (a) killed between set_batch('launching') and create returning: provisional row 'creating', nothing on RunPod
        seed_batch("bR1", "R1_e1", "launching", "queued", pods_row=("pending-bR1-1", "creating"))
        gpu_batch.runpod_graphql = fake_gql([])
        rc_dry = gpu_batch.reconcile(con, "bR1", dry_run=True)
        st_dry = (con.execute("SELECT status FROM batches WHERE batch_id = 'bR1'").fetchone()[0],
                  con.execute("SELECT status FROM pods WHERE pod_id = 'pending-bR1-1'").fetchone()[0])
        rc = gpu_batch.reconcile(con, "bR1")
        b = con.execute("SELECT status, note FROM batches WHERE batch_id = 'bR1'").fetchone()
        p_ = con.execute("SELECT status, terminated_at FROM pods WHERE pod_id = 'pending-bR1-1'").fetchone()
        v = con.execute("SELECT status FROM videos WHERE video_id = 'R1_e1'").fetchone()[0]
        e = con.execute("SELECT status FROM episodes WHERE episode_id = 'R1_e1'").fetchone()[0]
        check(rc_dry == 0 and st_dry == ("launching", "creating"), "reconcile --dry-run: looks RunPod up, changes nothing", str(st_dry))
        check(rc == 0 and b[0] == "failed" and "reconciled" in (b[1] or "") and p_[0] == "terminated" and p_[1] and v == "transcoded" and e == "cut"
              and gpu_batch.active_pods(con) == [] and gpu_batch.launch_checks.__name__ == "launch_checks",
              "reconcile: pod gone -> pods row terminated, batch failed (note), video 'transcoded', episodes row 'cut' (was hand SQL: review M7)", f"{tuple(b)} {tuple(p_)} {v} {e}")
        # (b) a live pod: report only, nothing changes; --terminate tears it down (spend row) and settles
        seed_batch("bR2", "R2_e1", "running", "transcribing", pod_id="podLIVE", pods_row=("podLIVE", "running"))
        live = {"id": "podLIVE", "name": "ytpt-bR2", "desiredStatus": "RUNNING", "runtime": {"uptimeInSeconds": 600}}
        gpu_batch.runpod_graphql = fake_gql([live], live)
        rc = gpu_batch.reconcile(con, "bR2")
        untouched = (con.execute("SELECT status FROM batches WHERE batch_id = 'bR2'").fetchone()[0],
                     con.execute("SELECT status FROM pods WHERE pod_id = 'podLIVE'").fetchone()[0],
                     con.execute("SELECT status FROM videos WHERE video_id = 'R2_e1'").fetchone()[0])
        n_term_before = sum("podTerminate" in q for q in queries)
        check(rc == 0 and untouched == ("running", "running", "transcribing") and n_term_before == 0,
              "reconcile: RunPod says the pod is RUNNING -> report only, no terminate call, DB untouched", str(untouched))
        rc = gpu_batch.reconcile(con, "bR2", terminate=True)
        after = (con.execute("SELECT status FROM batches WHERE batch_id = 'bR2'").fetchone()[0],
                 con.execute("SELECT status FROM pods WHERE pod_id = 'podLIVE'").fetchone()[0],
                 con.execute("SELECT status FROM videos WHERE video_id = 'R2_e1'").fetchone()[0],
                 con.execute("SELECT COUNT(*) FROM spend WHERE pod_id = 'podLIVE'").fetchone()[0])
        check(rc == 0 and after == ("failed", "terminated", "transcoded", 1) and sum("podTerminate" in q for q in queries) == 1,
              "reconcile --terminate: teardown (one terminate call, spend recorded), then batch failed and videos released", str(after))
        # (c) not reconcilable / nothing to do
        con.execute("INSERT INTO batches (batch_id, created_at, status, n_videos) VALUES ('bR3', ?, 'done', 1)", (now,))
        gpu_batch.runpod_graphql = fake_gql([])
        check(gpu_batch.reconcile(con, "bR3") == gpu_batch.EXIT_REFUSED and gpu_batch.reconcile(con, "bR1") == 0,
              "reconcile: a 'done' batch is refused (exit 2); an already-settled failed batch is a no-op")
    finally:
        gpu_batch.runpod_graphql = saved_gql
    # the auth header, not the query string (review L1)
    import urllib.request as _ur
    captured = {}
    class _Resp:
        def __init__(self): self.body = b'{"data": {"ok": 1}}'
        def read(self): return self.body
        def __enter__(self): return self
        def __exit__(self, *a): return False
    saved_open = _ur.urlopen
    _ur.urlopen = lambda req, timeout=30: captured.update(url=req.full_url, auth=req.get_header("Authorization")) or _Resp()
    os.environ["RUNPOD_API_KEY"] = "sekrit"
    try:
        gpu_batch.runpod_graphql("query { x }")
    finally:
        _ur.urlopen = saved_open
        os.environ.pop("RUNPOD_API_KEY", None)
    check("api_key" not in captured["url"] and captured["auth"] == "Bearer sekrit", "runpod_graphql: key in the Authorization header, not the URL", str(captured))

    # -- spend ledger and the two hard stops
    con.execute("INSERT INTO spend (pod_id, started_at, usd) VALUES ('podA', ?, 1.0)", (now,))
    con.execute("INSERT INTO batches (batch_id, created_at, status, pod_id, actual_usd) VALUES ('bS1', ?, 'done', 'podA', 1.2)", (now,))
    con.execute("INSERT INTO batches (batch_id, created_at, status, pod_id, actual_usd) VALUES ('bS2', ?, 'done', 'podB', 0.5)", (now,))
    con.execute("INSERT INTO spend (pod_id, started_at, usd) VALUES (NULL, ?, 0.3)", (now,))
    rp = spend.runpod_total(con)
    check(abs(rp - (1.2 + 0.5 + 0.3 + sum(float(r[0]) for r in con.execute("SELECT usd FROM spend WHERE pod_id = 'podLIVE'")))) < 1e-6,
          "runpod_total: per pod max(spend, batch actual_usd) + hand rows, no double count", str(rp))
    os.environ["RUNPOD_CREDIT_USD"], os.environ["RUNPOD_CREDIT_MARGIN_USD"] = "3.0", "0.5"
    try:
        why = spend.runpod_refusal(con, 0.60)
        ok_ = spend.runpod_refusal(con, 0.10)
        s_ = dstatus.collect(db, cfg=dict(dstatus.settings(), cap_min=45))
        txt = dstatus.as_text(s_)
    finally:
        os.environ.pop("RUNPOD_CREDIT_USD", None); os.environ.pop("RUNPOD_CREDIT_MARGIN_USD", None)
    check(why and "stop line $2.50" in why and ok_ is None and "RunPod $" in txt and "of $3.00 credit" in txt and "Claude classifier $" in txt,
          "spend: refusal once recorded + estimate reaches credit - margin; status digest shows both ledgers", f"{why} | {txt.splitlines()[6:8]}")
    # propose / launch_checks refuse past the line
    con.execute("INSERT INTO videos (video_id, speaker, channel_url, month, crawled_at, in_frame, status, updated_at, kind, duration_s, transcribed_s, opus_path) "
                "VALUES ('SP1', 'oakland', 'u', '2026-09', ?, 1, 'transcoded', ?, 'episode', 600, 600, ?)", (now, now, str(d / "x.opus")))
    (d / "x.opus").write_text("x")
    os.environ["RUNPOD_CREDIT_USD"] = "2.5"
    try:
        bid = gpu_batch.propose(con, min_hours=0, quiet=True, speakers=["oakland"])
        con.execute("INSERT INTO batches (batch_id, created_at, status, n_videos, audio_hours, est_usd) VALUES ('bS3', ?, 'approved', 1, 0.1, 0.05)", (now,))
        con.execute("INSERT INTO batch_videos VALUES ('bS3', 'SP1')")
        probs = gpu_batch.launch_checks(con, "bS3")
    finally:
        os.environ.pop("RUNPOD_CREDIT_USD", None)
    check(bid is None and con.execute("SELECT status FROM videos WHERE video_id = 'SP1'").fetchone()[0] == "transcoded"
          and any("stop line" in p for p in probs),
          "propose refuses (row stays transcoded) and launch_checks lists the refusal once the credit stop line is reached", f"{bid} {probs}")

    # -- classifier: per-model instrument id + prices, the (id, prompt_version) key, cap
    check(classify.prompt_version_for("claude-sonnet-4-6") == classify.CODEBOOK_VERSION == "t0-a4a8e55760"
          and classify.prompt_version_for("claude-haiku-4-5-20251001") == "t0-a4a8e55760/claude-haiku-4-5-20251001",
          "prompt_version_for: the T0 instrument keeps the bare hash (316 rows stay cached); every other model gets hash/model")
    u = {"input_tokens": 1000, "output_tokens": 100, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1500}
    check(abs(classify.usd(u, "claude-haiku-4-5-20251001") - (1000 * 1.0 + 100 * 5.0 + 1500 * 0.10) / 1e6) < 1e-9
          and abs(classify.usd(u, "claude-sonnet-4-6", batch=True) - 0.5 * (1000 * 3.0 + 100 * 15.0 + 1500 * 0.30) / 1e6) < 1e-9
          and classify.worst_case_usd("x" * 3200, "claude-haiku-4-5-20251001") > classify.worst_case_usd("x" * 3200, "claude-haiku-4-5-20251001", batch=True),
          "usd: per-model prices, batch = half; Haiku's codebook priced as plain input (cache minimum 4,096 tokens)")
    try:
        classify.price_for("claude-nope")
        priced = False
    except SystemExit:
        priced = True
    check(priced, "price_for: an unpriced model refuses to run")
    row1 = classify.call_row({"id": "V_c001", "video_id": "V", "start_s": 0, "end_s": 240, "label": "no", "model": "claude-sonnet-4-6",
                              "prompt_version": "t0-a4a8e55760", "cost_usd": 0.004})
    row2 = dict(row1, prompt_version="t0-a4a8e55760/claude-haiku-4-5-20251001", model="claude-haiku-4-5-20251001", label="mention_only", cost_usd=0.001)
    row3 = dict(row2, label=None, cost_usd=0.0)
    classify.insert_call(con, row1, replace=True); classify.insert_call(con, row2, replace=True); classify.insert_call(con, row3, replace=True)
    rows_ = {r["prompt_version"]: (r["label"], r["cost_usd"]) for r in con.execute("SELECT prompt_version, label, cost_usd FROM classifier_calls WHERE id = 'V_c001'")}
    check(rows_ == {"t0-a4a8e55760": ("no", 0.004), "t0-a4a8e55760/claude-haiku-4-5-20251001": ("mention_only", 0.001)},
          "insert_call: (id, prompt_version) key -- a second instrument adds a row, never overwrites the first's label or cost; a NULL-label row never replaces a labelled one (review L2)", str(rows_))
    # the migration re-keys an old-shape table without losing a row
    old = d / "old_cc.sqlite"
    c0 = sqlite3.connect(old)
    c0.executescript("""CREATE TABLE videos (video_id TEXT PRIMARY KEY, speaker TEXT NOT NULL, channel_url TEXT NOT NULL, month TEXT NOT NULL,
        raw_meta TEXT, crawled_at TEXT NOT NULL, in_frame INTEGER DEFAULT 0, status TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE classifier_calls (id TEXT PRIMARY KEY, video_id TEXT NOT NULL, start_s REAL NOT NULL, end_s REAL NOT NULL, pad_start_s REAL, pad_end_s REAL,
        words INTEGER, hits_per_100w REAL, family_rates TEXT, top_keywords TEXT, label TEXT, rationale TEXT, contains_instructions INTEGER,
        model TEXT NOT NULL, prompt_version TEXT NOT NULL, input_tokens INTEGER, cache_read_input_tokens INTEGER, cache_creation_input_tokens INTEGER,
        output_tokens INTEGER, cost_usd REAL, latency_s REAL, at TEXT NOT NULL);
        INSERT INTO classifier_calls (id, video_id, start_s, end_s, label, model, prompt_version, cost_usd, at) VALUES ('A_c001', 'A', 0, 1, 'no', 'm', 'p', 0.5, 't');""")
    c0.commit(); c0.close()
    r = state.migrate(old, backup=False)
    c1 = state.connect(old)
    kept = c1.execute("SELECT id, label, cost_usd, batch_id FROM classifier_calls").fetchall()
    key = state.classifier_calls_key(c1)
    c1.close()
    check(r["classifier_calls_rekeyed"] and key == ["id", "prompt_version"] and [tuple(x) for x in kept] == [("A_c001", "no", 0.5, None)]
          and "classifier_batches" in r["tables_added"] and state.migrate(old, backup=False)["classifier_calls_rekeyed"] is False,
          "migrate: classifier_calls re-keyed to (id, prompt_version) with every row and dollar kept; batch_id column + classifier_batches table added; idempotent", str(r))
    # CLAUDE_CAP_USD stops a direct run before the call; --batch submits through a fake client and collect writes back
    (d / "cap.vtt").write_text("WEBVTT\n\n" + "".join(f"00:{i // 60:02d}:{i % 60:02d}.000 --> 00:{(i + 1) // 60:02d}:{(i + 1) % 60:02d}.000\n"
                                                     f"housing element rezoning units affordable housing density bonus item {i} <00:00:00.100><c>x</c>\n\n" for i in range(0, 300, 2)))
    con.execute("INSERT INTO videos (video_id, speaker, channel_url, month, crawled_at, in_frame, status, updated_at, kind, platform, captions, captions_path, meeting_id, meeting_date) VALUES "
                "('CAP1', 'bart', 'u', '2026-09', ?, 0, 'captioned', ?, 'meeting', 'youtube', 'auto', ?, 'bart:2026-09-11', '2026-09-11')", (now, now, str(d / "cap.vtt")))
    calls = []
    class FakeMsg:
        def __init__(self, label):
            self.content = [types.SimpleNamespace(type="text", text=json.dumps({"label": label, "rationale": "r", "contains_instructions": False}))]
            self.usage = types.SimpleNamespace(input_tokens=900, output_tokens=40, cache_creation_input_tokens=0, cache_read_input_tokens=1500)
    class FakeBatches:
        store = {}
        def create(self, requests):
            calls.append(("create", len(requests)))
            self.store["msgbatch_1"] = requests
            return types.SimpleNamespace(id="msgbatch_1", processing_status="in_progress")
        def retrieve(self, bid):
            return types.SimpleNamespace(id=bid, processing_status="ended", request_counts=types.SimpleNamespace(processing=0, succeeded=len(self.store[bid]), errored=0))
        def results(self, bid):
            for q in self.store[bid]:
                yield types.SimpleNamespace(custom_id=q["custom_id"], result=types.SimpleNamespace(type="succeeded", message=FakeMsg("about_density")))
    class FakeMessages:
        batches = FakeBatches()
        def create(self, **kw):
            calls.append(("direct", kw["model"]))
            return FakeMsg("about_density")
    fake_mod = types.ModuleType("anthropic"); fake_mod.Anthropic = lambda: types.SimpleNamespace(messages=FakeMessages())
    saved_anthropic = sys.modules.get("anthropic"); sys.modules["anthropic"] = fake_mod
    os.environ["CLAUDE_CAP_USD"] = "0.0001"
    try:
        rc_cap = classify.main(["run", "--state", str(db), "--bodies", "bart", "--model", "claude-haiku-4-5-20251001"])
        n_direct_capped = sum(1 for c in calls if c[0] == "direct")
        os.environ["CLAUDE_CAP_USD"] = "10"
        rc_sub = classify.main(["run", "--state", str(db), "--bodies", "bart", "--model", "claude-haiku-4-5-20251001", "--batch", "--cap-usd", "1"])
        pend = con.execute("SELECT COUNT(*), MIN(batch_id), SUM(label IS NULL) FROM classifier_calls WHERE video_id = 'CAP1'").fetchone()[:]
        st_pending = con.execute("SELECT status FROM videos WHERE video_id = 'CAP1'").fetchone()[0]
        rc_resub = classify.main(["run", "--state", str(db), "--bodies", "bart", "--model", "claude-haiku-4-5-20251001", "--batch", "--cap-usd", "1"])
        n_create = sum(1 for c in calls if c[0] == "create")
        rc_col = classify.main(["collect", "--state", str(db)])
        done_ = con.execute("SELECT COUNT(*), SUM(label = 'about_density'), ROUND(SUM(cost_usd), 6), MIN(prompt_version), MIN(model) FROM classifier_calls WHERE video_id = 'CAP1'").fetchone()[:]
        st_done = con.execute("SELECT status FROM videos WHERE video_id = 'CAP1'").fetchone()[0]
        eps_ = con.execute("SELECT COUNT(*) FROM episodes WHERE video_id = 'CAP1' AND status = 'located'").fetchone()[0]
        cb = con.execute("SELECT status, n_succeeded, cost_usd FROM classifier_batches WHERE batch_id = 'msgbatch_1'").fetchone()[:]
    finally:
        os.environ.pop("CLAUDE_CAP_USD", None)
        if saved_anthropic is not None:
            sys.modules["anthropic"] = saved_anthropic
        else:
            sys.modules.pop("anthropic", None)
    check(rc_cap == 2 and n_direct_capped == 0 and st_pending in ("captioned",),
          "classify run: CLAUDE_CAP_USD reached -> refused before any call, meeting stays 'captioned'", f"rc={rc_cap} direct={n_direct_capped}")
    check(rc_sub == 0 and pend[0] >= 1 and pend[1] == "msgbatch_1" and pend[2] == pend[0] and st_pending == "captioned" and rc_resub == 0 and n_create == 1,
          "classify run --batch: rows written with label NULL + batch id, one create call; a re-run submits nothing twice; meeting stays 'captioned'", f"{pend} {st_pending} creates={n_create}")
    per_call = 0.5 * (900 * 1.0 + 40 * 5.0 + 1500 * 0.10) / 1e6
    # the default 120-s granularity is named in the instrument id ('/mr120', 2026-10-18); the bare id means 240 s
    check(rc_col == 0 and done_[1] == done_[0] and abs(done_[2] - round(per_call * done_[0], 6)) < 1e-6
          and done_[3] == classify.prompt_version_for("claude-haiku-4-5-20251001", max_run=classify.DEFAULTS["max_run"])
          == "t0-a4a8e55760/claude-haiku-4-5-20251001/mr120"
          and done_[4] == "claude-haiku-4-5-20251001" and st_done == "classified" and eps_ >= 1 and cb[0] == "collected" and cb[1] == done_[0],
          "classify collect: labels + usage written at the batch rate under the Haiku instrument id; episodes written; meeting -> 'classified'; batch row 'collected'",
          f"{done_} {st_done} eps={eps_} {cb}")

    # -- ingest tweaks: speaker-tag prefixes stripped; initial_prompt carried; QA reads the meeting's captions for an episode
    segs = [{"start": 0, "end": 3, "text": "Male Speaker 1 director ames aye", "speaker": "S0"},
            {"start": 3, "end": 5, "text": "Female Speaker: thank you", "speaker": "S1"},
            {"start": 5, "end": 6, "text": "Speaker 2", "speaker": "S1"},
            {"start": 6, "end": 9, "text": "the speaker system in the room", "speaker": "S0"}]
    doc = gpu_job.make_doc({"video_id": "T", "speaker": "bart", "channel_url": "u", "month": "2026-09", "duration_s": 9.0, "transcribed_s": 9.0,
                            "initial_prompt": "BART Board"}, segs)
    check([s["text"] for s in doc["segments"]] == ["director ames aye", "thank you", "the speaker system in the room"] and doc["speaker_tags_stripped"] == 3
          and doc["initial_prompt"] == "BART Board" and gpu_job.validate_doc(doc, "T") == [],
          "make_doc: 'Male Speaker 1' / 'Female Speaker:' / bare 'Speaker 2' prefixes stripped (tag-only segment dropped), count + initial_prompt recorded, contract intact",
          str([s["text"] for s in doc["segments"]]))
    check(vocab.clip("a " * 150) .endswith(".") and len(vocab.clip("a " * 150)) <= vocab.MAX_CHARS and vocab.for_body(con, "nobody") is None
          and all(len(v) <= 200 for v in vocab.read_tsv().values()) and set(vocab.read_tsv()) >= {c["speaker"] for c in discover.read_channels() if c["kind"] == "meeting"},
          "vocab: clipped at a word boundary <= 200 chars; unknown body -> None; plan/vocab.tsv has a line <= 200 chars for every kind=meeting body")
    con.execute("INSERT INTO sources (source_id, body, body_kind, platform, platform_id, url, updated_at, vocab) VALUES ('oakland:granicus:oakland:2', 'oakland', 'council', 'granicus', 'oakland:2', 'u', ?, 'Oakland City Council. BART, AC Transit.')", (now,))
    seed_batch("bV", "V_e1", "approved", "queued")
    man, _, _ = gpu_job.build_manifest(con, "bV")
    check(man["videos"][0]["initial_prompt"] == "Oakland City Council. BART, AC Transit." and vocab.for_body(con, "sanjose") == vocab.clip(vocab.read_tsv()["sanjose"]),
          "build_manifest: initial_prompt from sources.vocab (DB) first, plan/vocab.tsv second", str(man["videos"][0].get("initial_prompt")))
    subs = d / "subs"; subs.mkdir()
    (subs / "MTG1.en.vtt").write_text("WEBVTT\n\n" + "".join(f"00:{(600 + i) // 60:02d}:{(600 + i) % 60:02d}.000 --> 00:{(601 + i) // 60:02d}:{(601 + i) % 60:02d}.000\n"
                                                             f"caption words here now<00:00:00.100><c>x</c>\n\n" for i in range(0, 120, 2)))
    ep_doc = gpu_job.make_doc({"video_id": "MTG1_e1", "speaker": "bart", "channel_url": "u", "month": "2026-09", "duration_s": 120.0, "transcribed_s": 120.0,
                               "window_start_s": 600.0, "kind": "episode", "episode_id": "MTG1_e1"},
                              [{"start": i, "end": i + 1.5, "text": "caption words here now", "speaker": "S0"} for i in range(0, 120, 2)])
    (d / "MTG1_e1.json").write_text(json.dumps(ep_doc))
    rq = qa.check_video(d / "MTG1_e1.json", subs)
    check(rq.get("caption_source") == "meeting" and rq.get("caption_file") == "MTG1.en.vtt" and rq.get("wer_vs_captions") is not None
          and rq.get("missing_text_minutes") == 0 and rq.get("extra_text_minutes") == 0,
          "qa_transcripts: an episode '<meeting>_e<n>' is checked against its MEETING's caption track at window_start_s (per-minute dropped / extra)", str({k: rq.get(k) for k in ("caption_source", "caption_file", "wer_vs_captions", "missing_text_minutes")}))

    # -- the b13 bug (2026-10-16, plan/BART-QIL-ASSESSMENT.md): window_start_s = 0 in a clip
    # that starts 600 s into the meeting. The first 120 s of MTG2's captions are
    # UNRELATED speech, so windowing at 0 scores the clip against the wrong words
    # (the old code gave qIlAln3uR7U_e15 771 % this way). The offset must come
    # from episodes.start_s instead, and with no source there must be no number.
    REAL, REAL_LAST = ("caption words here", "the board will"), ("now", "recess")     # alternating so no run of identical segments trips the loop FAIL
    JUNK, JUNK_LAST = ("budget item roll call", "motion carries five"), ("vote", "zero")
    (subs / "MTG2.en.vtt").write_text("WEBVTT\n\n" + "".join(f"00:{i // 60:02d}:{i % 60:02d}.000 --> 00:{(i + 1) // 60:02d}:{(i + 1) % 60:02d}.000\n"
                                                             f"{JUNK[i // 2 % 2]}<00:00:00.100><c> {JUNK_LAST[i // 2 % 2]}</c>\n\n" for i in range(0, 120, 2))
                                      + "".join(f"00:{(600 + i) // 60:02d}:{(600 + i) % 60:02d}.000 --> 00:{(601 + i) // 60:02d}:{(601 + i) % 60:02d}.000\n"
                                                f"{REAL[i // 2 % 2]}<00:00:00.100><c> {REAL_LAST[i // 2 % 2]}</c>\n\n" for i in range(0, 120, 2)))
    for eid, w0 in (("MTG2_e1", 0), ("MTG2_e2", 0), ("MTG2_e3", None)):
        m = {"video_id": eid, "speaker": "bart", "channel_url": "u", "month": "2026-09", "duration_s": 120.0, "transcribed_s": 120.0}
        if w0 is not None:
            m["window_start_s"] = w0                     # b13 shape: 0, no kind / episode_id / episode_start_s keys
        (d / f"{eid}.json").write_text(json.dumps(gpu_job.make_doc(m, [{"start": i, "end": i + 1.5, "text": f"{REAL[i // 2 % 2]} {REAL_LAST[i // 2 % 2]}", "speaker": "S0"} for i in range(0, 120, 2)])))
    con.execute("INSERT INTO episodes (episode_id, meeting_id, video_id, start_s, end_s, tags, method, status, updated_at) "
                "VALUES ('MTG2_e1', 'bart:2026-09-01', 'MTG2', 600.0, 720.0, 'transit', 'classifier', 'done', ?)", (now,))
    con.commit()
    r_db, r_none, r_off = qa.check_video(d / "MTG2_e1.json", subs, db), qa.check_video(d / "MTG2_e1.json", subs, None), qa.check_video(d / "MTG2_e2.json", subs, db)
    check(r_db.get("window_source") == "state_db" and r_db.get("window_start_s_used") == 600.0 and r_db.get("wer_vs_captions") == 0.0
          and r_db.get("ref_words") == 240 and r_db.get("extra_text_minutes") == 0 and not any("skipped" in w for w in r_db["info"]),
          "qa_transcripts: window_start_s = 0 on an episode is 'unknown', not 0: the offset comes from episodes.start_s and the clip scores 0 % against the right captions",
          str({k: r_db.get(k) for k in ("window_source", "window_start_s_used", "wer_vs_captions", "ref_words", "extra_text_minutes", "warn")}))
    check(qa.wer(" ".join(f"{JUNK[k % 2]} {JUNK_LAST[k % 2]}" for k in range(60)), " ".join(f"{REAL[k % 2]} {REAL_LAST[k % 2]}" for k in range(60))) > 0.9,
          "qa_transcripts: the wrong window (0 s) really would have scored >90 % -- the fixture reproduces the bug shape, not a no-op")
    check("wer_vs_captions" not in r_none and r_none.get("window_source") == "none" and any("caption check skipped" in w for w in r_none["info"]) and not r_none["warn"]
          and "wer_vs_captions" not in r_off and r_off.get("window_source") == "none" and any("caption check skipped" in w for w in r_off["info"]),
          "qa_transcripts: no state DB, or no episodes row, -> NO caption metric and an info note saying why, not a WARN (never a wrong number that looks like a finding)",
          f"none={r_none.get('wer_vs_captions')} {r_none['info']} {r_none['warn']} off={r_off.get('wer_vs_captions')} {r_off['info']}")
    con.execute("INSERT INTO episodes (episode_id, meeting_id, video_id, start_s, end_s, tags, method, status, updated_at) "
                "VALUES ('MTG2_e4', 'bart:2026-09-01', 'MTG2', -0.02, 120.0, 'transit', 'classifier', 'done', ?)", (now,)); con.commit()
    check(qa.episode_window_start({"window_start_s": 0}, "MTG2_e4", db) == ("state_db", 0.0) and qa.episode_window_start({"window_start_s": 0}, "MTG2_e9", db) == (None, None),
          "qa_transcripts: an episodes row is authoritative even at start_s <= 0 (a clip padded off the meeting's start, 9FM5406YagE_e1); no row -> unknown")
    r_meta = qa.check_video(d / "MTG2_e3.json", subs, db)
    doc3 = json.loads((d / "MTG2_e3.json").read_text()); doc3["episode_start_s"] = 600.0; (d / "MTG2_e3.json").write_text(json.dumps(doc3))
    r_meta2 = qa.check_video(d / "MTG2_e3.json", subs, None)
    check(r_meta.get("window_source") == "none" and r_meta2.get("window_source") == "episode_start_s" and r_meta2.get("wer_vs_captions") == 0.0
          and rq.get("window_source") == "window_start_s",
          "qa_transcripts: a missing key is 'unknown' too; episode_start_s in the JSON is the second source; a real window_start_s is still used as-is",
          str((r_meta.get("window_source"), r_meta2.get("window_source"), r_meta2.get("wer_vs_captions"), rq.get("window_source"))))
    rs_show = qa.check_video(d / "MTG1_e1.json", subs, db)
    check(rs_show.get("wer_vs_captions") == rq.get("wer_vs_captions") and qa.check_video(d / "MTG2_e2.json", subs, d / "no_such.sqlite").get("window_source") == "none",
          "qa_transcripts: an absent state DB is tolerated (no crash, no metric); passing one changes nothing for an episode that carries its offset")
    out_csv = d / "qa_state.csv"
    rc_qa = subprocess.run([PY, str(ROOT / "analysis" / "qa_transcripts.py"), "--src", str(d), "--glob", "MTG2_e*.json", "--subs", str(subs),
                            "--state-db", str(db), "--csv", str(out_csv)], capture_output=True, text=True, cwd=ROOT)
    rows_qa = {r["video_id"]: r for r in csv.DictReader(open(out_csv))} if out_csv.exists() else {}
    check(rc_qa.returncode == 0 and rows_qa.get("MTG2_e1", {}).get("window_source") == "state_db" and rows_qa.get("MTG2_e1", {}).get("wer_vs_captions") == "0.0"
          and rows_qa.get("MTG2_e2", {}).get("window_source") == "none" and rows_qa.get("MTG2_e2", {}).get("wer_vs_captions") == ""
          and "info: caption check skipped" in rows_qa.get("MTG2_e2", {}).get("issues", "") and rows_qa.get("MTG2_e2", {}).get("status") == "OK",
          "qa_transcripts CLI: --state-db wires the lookup through to the CSV (window_source column; empty WER and an info note, status OK, when unknown)", rc_qa.stdout[-600:] + rc_qa.stderr[-300:])

    # -- discovery: --list-only saves, --from-listings inserts without the network
    ldir = d / "listings"
    chans = [{"speaker": "sanjose", "url": "https://www.youtube.com/channel/UCx/streams", "kind": "meeting", "regular_re": r"City Council$"},
             {"speaker": "oakland", "url": "https://oakland.granicus.com/ViewPublisher.php?view_id=2", "kind": "meeting", "regular_re": r"^Concurrent"}]
    saved_ly, saved_fl = discover.list_youtube, granicus.fetch_listing
    discover.list_youtube = lambda url, n: ([{"id": "Y1", "upload_date": None, "duration_s": 5000.0, "live_status": "was_live", "title": "SEP 15, 2026 | City Council"},
                                             {"id": "Y2", "upload_date": None, "duration_s": 900.0, "live_status": "was_live", "title": "SEP 9, 2026 | Planning Commission"}], "", 0)
    granicus.fetch_listing = lambda site, view: ([{"clip_id": "7755", "name": "Concurrent Meeting of the Oakland Redevelopment Successor Agency and the City Council on 2026-09-15",
                                                            "date": "September 15, 2026", "duration": "03h 42m", "duration_s": 13320.0, "epoch": None, "agenda": True, "minutes": False}], {})
    try:
        res1 = discover.discover_meetings(con, chans, since="2025-09-20", list_only=str(ldir), sleep=0)
        n_before = con.execute("SELECT COUNT(*) FROM videos WHERE kind = 'meeting' AND speaker IN ('sanjose', 'oakland')").fetchone()[0]
        discover.list_youtube = lambda url, n: (_ for _ in ()).throw(AssertionError("network used with --from-listings"))
        granicus.fetch_listing = discover.list_youtube
        res2 = discover.discover_meetings(con, chans, since="2025-09-20", from_listings=str(ldir), sleep=0)
    finally:
        discover.list_youtube, granicus.fetch_listing = saved_ly, saved_fl
    got = {r[0]: r[1] for r in con.execute("SELECT video_id, meeting_date FROM videos WHERE kind = 'meeting' AND speaker IN ('sanjose', 'oakland')")}
    check((ldir / "sanjose.tsv").exists() and (ldir / "oakland.json").exists() and n_before == 0 and res1["sanjose"][0] == 2,
          "discover --list-only: listings saved per body, nothing inserted", str(res1))
    check(res2 == {"sanjose": (2, 1, 1), "oakland": (1, 1, 1)} and got == {"Y1": "2026-09-15", "gr-oakland-7755": "2026-09-15"},
          "discover --from-listings: regular meetings inserted from the saved listings, no network call", f"{res2} {got}")
    check(discover.body_kind_of("napa") == "county" and discover.body_kind_of("samtrans") == "transit_board" and discover.body_kind_of("cotati") == "council",
          "body_kind_of: county boards, transit boards, councils")
    # the other Granicus listing templates found on 2026-09-22 (fixtures are trimmed live pages)
    F = ROOT / "analysis" / "fixtures" / "granicus"
    sam = granicus.parse_listing((F / "samtrans_listing_excerpt.html").read_text(errors="replace"))
    sf = granicus.parse_listing((F / "sanfrancisco_listing_excerpt.html").read_text(errors="replace"))
    ala = granicus.parse_listing((F / "alameda_listing_excerpt.html").read_text(errors="replace"))
    wc = granicus.parse_listing((F / "walnutcreek_listing_excerpt.html").read_text(errors="replace"))
    check(len(sam) == 4 and sam[0]["clip_id"] == "1532" and sam[0]["name"] == "SamTrans Board of Directors" and sam[0]["date"].startswith("Sep 2, 2026")
          and len(ala) == 3 and ala[0]["duration_s"] == 14460.0 and ala[0]["date"] == "July 22, 2026" and ala[0]["name"].startswith("Transportation Commission on 2026-07-22")
          and len(wc) == 3 and wc[0]["name"] == "City Council Meeting" and wc[0]["date"] == "December 15, 2009",
          "parse_listing: 'listingRow' (SamTrans), bare <tr> (Alameda, Walnut Creek) templates -> clip id, name, date, duration", f"{sam[:1]} {ala[:1]} {wc[:1]}")
    check(len(sf) == 3 and sf[0]["clip_id"] == "48026" and sf[0]["name"] == "SPECIAL Building Inspection Commission" and sf[0]["duration_s"] == 5062.0
          and discover.parse_meeting_date(sf[0]["date"]) == "2024-12-11" and granicus.parse_duration("01:24:22") == 5062.0,
          "parse_listing: SF BoS 'listLine' template -- blank name cell, title from the headers slug, unclassed date, HH:MM:SS duration", str(sf[:1]))
    oak = granicus.parse_listing((F / "oakland_listing.html").read_text(errors="replace"))
    check(len(oak) == 12 and oak[0]["clip_id"] == "7755" and oak[0]["duration_s"] == 13320.0, "parse_listing: the Oakland / AC Transit fixtures parse exactly as before")
    parts = discover.select_regular([{"id": "P1", "duration_s": 5000.0, "live_status": "was_live", "title": "Cupertino City Council Meeting - July 7, 2026 (Part 1)"},
                                     {"id": "P2", "duration_s": 4000.0, "live_status": "was_live", "title": "Cupertino City Council Meeting - July 7, 2026 (Part 2)"},
                                     {"id": "P3", "duration_s": 4000.0, "live_status": "was_live", "title": "Cupertino City Council Meeting - July 7, 2026 (Part 2)"},
                                     {"id": "S1", "duration_s": 4000.0, "live_status": "was_live", "title": "City Council - Regular Meeting 9-8-2026"}],
                                    r"City Council", since="2025-09-20")
    check(sorted(r["id"] for r in parts) == ["P1", "P2", "S1"] and all(r["meeting_date"] == "2026-07-07" for r in parts if r["id"].startswith("P")),
          "select_regular: multi-part uploads keep every Part N of a date (duplicate parts still collapse); M-D-YYYY titles parse", str([(r["id"], r["meeting_date"]) for r in parts]))
    gpu_batch.notify = saved_notify
    con.close()


def section_t2b(tmp):
    print("\n24. T2 part 2 (2026-09-22): caption trickle worker + 24 h backoff, Granicus JS caption track / ASX media, "
          "agenda_index locating, 15 h batch threshold")
    sys.path.insert(0, str(ROOT / "pipeline"))
    from datetime import datetime, timedelta, timezone      # noqa: E402
    import state, captions, granicus, agenda_index, gpu_batch  # noqa: E402
    import segment as sg                                    # noqa: E402
    d = tmp / "t2b"; d.mkdir()
    db = d / "state.sqlite"
    con = state.connect(db)
    now = state.now_iso()
    subs = d / "subs"; subs.mkdir()
    os.environ["SUBS_DIR"] = str(subs)

    # -- the 24 h backoff: written on a bot check, respected by a restart, cleared by a human
    check(captions.backoff_left(con) == (0.0, None), "backoff: none to start")
    until = captions.set_backoff(con, "ERROR: Sign in to confirm you're not a bot")
    left, brow = captions.backoff_left(con)
    check(23.0 < left / 3600 <= 24.0 and brow["reason"].startswith("ERROR") and until > now,
          "backoff: a bot check writes ~24 h into `backoffs` with the exact yt-dlp line", f"{left / 3600:.2f} h")
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(timespec="seconds")
    con.execute("UPDATE backoffs SET until = ? WHERE worker = 'captions'", (past,))
    check(captions.backoff_left(con)[0] == 0.0, "backoff: an expired row is not a backoff")
    con.execute("UPDATE backoffs SET until = 'not a timestamp' WHERE worker = 'captions'")
    check(captions.backoff_left(con)[0] == 0.0, "backoff: an unparseable `until` never wedges the worker shut")
    captions.set_backoff(con, "again")
    check(captions.clear_backoff(con) == 1 and captions.backoff_left(con) == (0.0, None),
          "backoff: --clear-backoff removes it")

    # -- a bot check under --loop: backoff written, row still 'discovered', exit 0 (launchd must not retry)
    fake = d / "yt-dlp"
    fake.write_text("#!/bin/sh\necho \"ERROR: [youtube] X1: Sign in to confirm you're not a bot\" >&2\nexit 1\n")
    fake.chmod(0o755)
    os.environ["YTDLP_BIN"] = str(fake)
    saved_notify, captions.notify = captions.notify, lambda level, msg, **k: True
    con.execute("INSERT INTO videos (video_id, speaker, channel_url, month, crawled_at, in_frame, status, updated_at, kind, platform, meeting_date) "
                "VALUES ('X1','sanjose','u','2026-09',?,1,'discovered',?,'meeting','youtube','2026-09-15')", (now, now))
    calls = d / "calls.txt"
    try:
        rc_loop = captions.main(["--state", str(db), "--loop", "--min-gap", "10", "--max-gap", "10", "--max", "1"])
        st = con.execute("SELECT status FROM videos WHERE video_id = 'X1'").fetchone()[0]
        left_after, _ = captions.backoff_left(con)
        check(rc_loop == 0 and st == "discovered" and 23 < left_after / 3600 <= 24,
              "captions --loop: a bot check exits 0 (launchd leaves it DOWN), row stays 'discovered', 24 h backoff written",
              f"rc={rc_loop} status={st} left={left_after / 3600:.1f} h")
        fake.write_text(f"#!/bin/sh\necho called >> {calls}\nexit 1\n")
        rc2 = captions.main(["--state", str(db), "--loop", "--min-gap", "10", "--max-gap", "10", "--max", "1"])
        check(rc2 == 0 and not calls.exists(),
              "captions: a restart inside the backoff window asks YouTube nothing and exits 0")
        captions.clear_backoff(con)
    finally:
        captions.notify = saved_notify
        os.environ.pop("YTDLP_BIN", None)
    # 2026-09-24: 8-12 min was still too fast -- a second 429 came after only 5
    # requests in 41 min -- so the default is one an hour with a hard daily cap.
    # Section 29 covers the limiter and the queue order it feeds.
    check(captions.LOOP_MIN_GAP == 3600.0 and captions.LOOP_MAX_GAP == 4200.0 and captions.LOOP_DAILY_CAP == 18,
          "captions: one meeting an hour (60-70 min gap) with a hard cap of 18 requests per rolling 24 h")

    # -- parse_player: the Flowplayer subtitles object (Santa Clara / Hayward) and the ASX media route
    js_page = ('<html><div class="index-point" time="600">6. Housing Element Rezoning</div>'
               '<script>var cfg = { subtitles: [{ kind: "subtitles",\n lang: "en",\n label: "English",\n'
               ' src: "\\/videos\\/2592\\/captions.vtt" }]};</script></html>')
    p_js = granicus.parse_player(js_page, "santaclara")
    old_page = ('<meta property="og:video" content="https://x/players/modernplayer.swf?VideoUrl=%2F%2Frichmond.granicus.com'
                '%2FASX.php%3Fintro%3D1%26view_id%3D2%26clip_id%3D6042%26r%3D1a8ec775c2254a8c38254e22653f3ad7%26sn%3D'
                'richmond.granicus.com%26stream_type%3Drtmp" />')
    p_old = granicus.parse_player(old_page, "richmond")
    asx = ('<?xml version="1.0"?><ASX version="3.0"><ENTRY><REF HREF="rtmp://archive-video.granicus.com/richmond/'
           'richmond_2bac08f0-46dc-46da-80db-756153d8ceaf.mp4" /></ENTRY></ASX>')
    check(p_js["captions_url"] == "https://santaclara.granicus.com/videos/2592/captions.vtt" and p_js["captions_lang"] == "en"
          and p_js["captions_label"] == "English" and len(p_js["index_points"]) == 1,
          "parse_player: a Flowplayer subtitles object is a caption track too (Santa Clara / Hayward serve no <track>)", str(p_js)[:200])
    check(granicus.archive_mp4_from_asx(asx) == "https://archive-video.granicus.com/richmond/richmond_2bac08f0-46dc-46da-80db-756153d8ceaf.mp4"
          and granicus.archive_mp4_from_asx("<ASX></ASX>") is None
          and p_old["asx_token"] == "1a8ec775c2254a8c38254e22653f3ad7" and p_old["m3u8"] is None,
          "ASX route: the 2015-era player's rtmp:// REF is the https archive MP4 (Richmond / San Leandro carry no m3u8)",
          str(granicus.archive_mp4_from_asx(asx)))
    trk = granicus.parse_player('<track kind="captions" src="/videos/7/captions.vtt">' + js_page, "oakland")
    check(trk["captions_url"] == "https://oakland.granicus.com/videos/7/captions.vtt",
          "parse_player: a real <track> element still wins over the JS config")

    # -- agenda_index: item spans, keyword match on titles, merge, rows with method='agenda_index'
    pts = [(0.0, "1. Call to Order"), (100.0, "2. CONSENT CALENDAR"), (100.0, "A. Minutes"),
           (600.0, "6. Housing Element Rezoning of Sites and Density Bonus"),
           (1200.0, "7. Zoning Ordinance Amendment for accessory dwelling units"),
           (1800.0, "8. Parks and Recreation Report"), (2400.0, "9. Caltrain Grade Separation Funding")]
    w = agenda_index.item_windows(pts, 3000.0)
    check([(a, b) for a, b, _ in w][:4] == [(0.0, 100.0), (100.0, 600.0), (100.0, 600.0), (600.0, 1200.0)]
          and w[-1][1] == 3000.0,
          "item_windows: an item runs to the NEXT item's time; items sharing a time share the span; the last runs to the meeting end",
          str([(a, b) for a, b, _ in w]))
    w_nodur = agenda_index.item_windows(pts, None)
    check(w_nodur[-1][1] == 2400.0 + 600.0 and agenda_index.item_windows([], None) == [],
          "item_windows: with no duration (Sausalito / SamTrans listings) the last item gets the median item length",
          str(w_nodur[-1]))
    kw = sg.load_keywords(*agenda_index.KEYWORD_FILES)
    matched = agenda_index.match_items(w, kw)
    eps = agenda_index.merge_runs(matched)
    check(len(matched) == 3 and len(eps) == 2 and eps[0]["start_s"] == 600.0 and eps[0]["end_s"] == 1800.0
          and eps[0]["tags"] == "density" and eps[0]["n_items"] == 2
          and "Housing Element" in eps[0]["title"] and "Zoning" in eps[0]["title"]
          and eps[1]["tags"] == "transit" and eps[1]["n_items"] == 1
          and all("Parks and Recreation" not in e["title"] for e in eps),
          "match_items + merge_runs: two CONSECUTIVE density items become one episode; the Caltrain item two items later stays its own; "
          "the parks item between them is not an episode",
          f"{len(matched)} matched -> {[(e['start_s'], e['end_s'], e['tags']) for e in eps]}")
    transit_only = agenda_index.match_items(agenda_index.item_windows([(0.0, "9. Caltrain Grade Separation Funding")], 600.0), kw)
    check(len(transit_only) == 1 and transit_only[0]["tags"] == "transit",
          "match_items: a Caltrain item is tagged transit, not density", str(transit_only))

    con.execute("INSERT INTO videos (video_id, speaker, channel_url, month, crawled_at, in_frame, status, updated_at, kind, platform, "
                "meeting_id, meeting_date, duration_s, captions) VALUES ('gr-t-1','santaclara','u','2026-09',?,1,'no_episodes',?,"
                "'meeting','granicus','santaclara:2026-09-15','2026-09-15',3000.0,'none')", (now, now))
    (subs / "gr-t-1.player.json").write_text(json.dumps({"m3u8": "https://h/p.m3u8", "mp4": None, "captions_url": None,
                                                         "index_points": [[t, ti] for t, ti in pts]}))
    mrow = con.execute("SELECT * FROM videos WHERE video_id = 'gr-t-1'").fetchone()
    summ = agenda_index.locate_meeting(con, mrow, kw)
    e = con.execute("SELECT * FROM episodes WHERE video_id = 'gr-t-1'").fetchall()
    check(summ["episodes"] == 2 and len(e) == 2 and [r["episode_id"] for r in e] == ["gr-t-1_a1", "gr-t-1_a2"]
          and all(r["method"] == "agenda_index" and r["status"] == "located" and r["label"] is None
                  and r["classifier_model"] is None for r in e)
          and "Housing Element" in (e[0]["agenda_item"] or ""),
          "agenda_index rows: id '<meeting>_a<n>' (the classifier's are '_e<n>'), method='agenda_index', NO label / model",
          str(dict(e[0])) if e else "no row")
    state.set_episode(con, "gr-t-1_a1", "cut", start_s=1.0, end_s=2.0)
    agenda_index.locate_meeting(con, mrow, kw)
    after = con.execute("SELECT status, start_s FROM episodes WHERE episode_id = 'gr-t-1_a1'").fetchone()
    check(after["status"] == "cut" and after["start_s"] == 1.0,
          "agenda_index: a re-run leaves an episode that is already cut alone", str(tuple(after)))
    con.execute("INSERT INTO episodes (episode_id, meeting_id, video_id, start_s, end_s, tags, method, status, updated_at) "
                "VALUES ('gr-t-1_e1','santaclara:2026-09-15','gr-t-1',10,20,'density','classifier','located',?)", (now,))
    per_method = dict(con.execute("SELECT method, COUNT(*) FROM episodes WHERE video_id = 'gr-t-1' GROUP BY method").fetchall())
    check(per_method == {"agenda_index": 2, "classifier": 1},
          "one meeting can carry both instruments; n is reported per method, never pooled", str(per_method))
    import re as _re
    check(_re.search(r"_([a-z])(\d+)$", "gr-t-1_a12").group(2) == "12" and qa.EPISODE_ID.search("gr-t-1_a3") is not None,
          "an '_a<n>' clip id still parses for register_clip and still counts as an episode in QA")

    # -- the batch threshold is 15 h, not 50 (2026-09-22)
    saved_setting = state.setting
    state.setting = lambda k, dflt: dflt if k == "BATCH_TARGET_HOURS" else saved_setting(k, dflt)
    try:
        check(gpu_batch.cfg()["target_h"] == 15.0, "BATCH_TARGET_HOURS default is 15 h (was 50)", str(gpu_batch.cfg()["target_h"]))
    finally:
        state.setting = saved_setting
    check(0.15 < gpu_batch.estimate_usd(15.0) < 0.80,
          "15 audio-h at the measured 25x is still a sane batch: well under $1 with the 1.5x contingency",
          f"${gpu_batch.estimate_usd(15.0)}")
    con.close()
    os.environ.pop("SUBS_DIR", None)


def section_t2c(tmp):
    print("\n25. T2 part 2 (2026-09-23): the local ASR locate route (method='local_asr'), a second instrument on a "
          "meeting agenda_index already advanced, and the cutter coming back for it")
    sys.path.insert(0, str(ROOT / "pipeline"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import types
    import state, classify, episodes                        # noqa: E402
    import asr_batch                                        # noqa: E402
    d = tmp / "t2c"; d.mkdir()
    db = d / "state.sqlite"
    con = state.connect(db)
    now = state.now_iso()
    vtt = d / "track.vtt"
    vtt.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:05.000\nhousing element rezoning\n")

    def meeting(vid, status, captions="granicus", path=True, body="santaclara", dur=3600.0):
        con.execute("INSERT INTO videos (video_id, speaker, channel_url, month, crawled_at, in_frame, status, updated_at, "
                    "kind, platform, meeting_id, meeting_date, duration_s, captions, captions_path) "
                    "VALUES (?,?,'u','2026-09',?,1,?,?,'meeting','granicus',?,'2026-09-15',?,?,?)",
                    (vid, body, now, status, now, f"{body}:{vid}", dur, captions, str(vtt) if path else None))
        return con.execute("SELECT * FROM videos WHERE video_id = ?", (vid,)).fetchone()

    def calls(vid, *spans):
        return {f"{vid}_c{i:03d}": {"id": f"{vid}_c{i:03d}", "video_id": vid, "start_s": a, "end_s": b,
                                    "label": lab, "rationale": "staff presentation", "family_rates": {"density": 1.0}}
                for i, (a, b, lab) in enumerate(spans, 1)}

    # -- the route, not the model: an ASR track is its own instrument
    check(classify.method_for("asr_local") == "local_asr" and classify.method_for("granicus") == "classifier"
          and classify.method_for("auto") == "classifier" and classify.method_for("manual") == "classifier"
          and classify.method_for(None) == "classifier",
          "classify.method_for: captions='asr_local' is method='local_asr'; every real caption track stays 'classifier'",
          f"asr_local -> {classify.method_for('asr_local')}, granicus -> {classify.method_for('granicus')}")

    m_asr = meeting("gr-sanleandro-1", "captioned", captions="asr_local", body="sanleandro")
    classify.write_episodes(con, "gr-sanleandro-1", m_asr["meeting_id"], calls("gr-sanleandro-1", (10.0, 70.0, "about_density")),
                            90, None, "claude-sonnet-4-6", "t0-x", classify.method_for(m_asr["captions"]))
    m_cap = meeting("gr-oakland-1", "captioned", body="oakland")
    classify.write_episodes(con, "gr-oakland-1", m_cap["meeting_id"], calls("gr-oakland-1", (10.0, 70.0, "about_density")),
                            90, None, "claude-sonnet-4-6", "t0-x", classify.method_for(m_cap["captions"]))
    got = dict(con.execute("SELECT video_id, method FROM episodes").fetchall())
    check(got == {"gr-sanleandro-1": "local_asr", "gr-oakland-1": "classifier"},
          "episodes rows from an ASR track carry method='local_asr'; from a caption track, 'classifier' (SCHEMA.md: report per method)",
          str(got))
    row = con.execute("SELECT * FROM episodes WHERE video_id = 'gr-sanleandro-1'").fetchone()
    check(row["episode_id"] == "gr-sanleandro-1_e1" and row["status"] == "located" and row["label"] == "about_density"
          and row["classifier_model"] == "claude-sonnet-4-6",
          "a local_asr episode is still a classifier verdict: '_e<n>', label and model on the row", str(tuple(row)[:8]))

    # -- --include-classified: agenda_index moved the row on, the caption track was never read
    m_adv = meeting("gr-santaclara-2", "classified")            # agenda_index left it here
    meeting("gr-santaclara-3", "classified", path=False)        # no caption track: never selected
    plain = types.SimpleNamespace(bodies=None, max=0, include_classified=False)
    wide = types.SimpleNamespace(bodies=None, max=0, include_classified=True)
    ids_plain = {r["video_id"] for r in classify.select_meetings(con, plain)}
    ids_wide = {r["video_id"] for r in classify.select_meetings(con, wide)}
    check(ids_plain == {"gr-sanleandro-1", "gr-oakland-1"} and ids_wide == ids_plain | {"gr-santaclara-2"},
          "select_meetings: --include-classified adds a meeting another instrument advanced AND that still has a caption "
          "track; a row without one is never selected", f"plain={sorted(ids_plain)} wide={sorted(ids_wide)}")
    bodied = types.SimpleNamespace(bodies="santaclara", max=0, include_classified=True)
    check({r["video_id"] for r in classify.select_meetings(con, bodied)} == {"gr-santaclara-2"},
          "select_meetings: --bodies still filters when --include-classified widens the status set")

    # -- finish_status: never clobber the row another instrument owns
    moved_cap = classify.finish_status(con, m_cap, "classified", "note")
    st_cap = con.execute("SELECT status FROM videos WHERE video_id = 'gr-oakland-1'").fetchone()[0]
    moved_adv = classify.finish_status(con, m_adv, "classified", "note")
    st_adv = con.execute("SELECT status FROM videos WHERE video_id = 'gr-santaclara-2'").fetchone()[0]
    ev = con.execute("SELECT COUNT(*) FROM events WHERE video_id = 'gr-santaclara-2' AND note LIKE 'classifier:%'").fetchone()[0]
    check(moved_cap and st_cap == "classified" and not moved_adv and st_adv == "classified" and ev == 1,
          "finish_status: a 'captioned' row still moves to 'classified'; a row agenda_index advanced keeps its status and "
          "gets an events row instead", f"captioned->{st_cap} advanced->{st_adv} events={ev}")
    m_cut = meeting("gr-santaclara-4", "cut")
    moved_cut = classify.finish_status(con, m_cut, "classified", "2 new episodes")
    st_cut = con.execute("SELECT status FROM videos WHERE video_id = 'gr-santaclara-4'").fetchone()[0]
    check(moved_cut and st_cut == "downloaded",
          "finish_status: new located episodes on a meeting the cutter FINISHED send it back to 'downloaded', or "
          "episodes.py would never come back for them", f"cut -> {st_cut}")

    # -- episodes.py selects that meeting again, and only while work is left
    def selected():
        return {r["video_id"] for r in con.execute(f"SELECT video_id FROM videos WHERE {episodes.SELECT_WHERE}")}
    con.execute("UPDATE videos SET status = 'cut' WHERE video_id IN ('gr-santaclara-4', 'gr-oakland-1')")
    con.execute("INSERT INTO episodes (episode_id, meeting_id, video_id, start_s, end_s, tags, method, status, updated_at) "
                "VALUES ('gr-santaclara-4_e1','santaclara:gr-santaclara-4','gr-santaclara-4',0,60,'density','classifier','located',?)", (now,))
    con.execute("UPDATE episodes SET status = 'cut' WHERE video_id = 'gr-oakland-1'")
    sel = selected()
    check("gr-santaclara-4" in sel and "gr-oakland-1" not in sel,
          "episodes.py: a 'cut' meeting with a new 'located' episode is selected again; a 'cut' meeting with none is left alone",
          str(sorted(sel)))
    con.execute("UPDATE episodes SET status = 'cut' WHERE video_id = 'gr-santaclara-4'")
    check("gr-santaclara-4" not in selected(),
          "episodes.py: once its new episodes are cut the meeting drops out of the selection again")

    # -- asr_batch: resumable by the row, not by a file
    con.execute("UPDATE videos SET captions = 'asr_local' WHERE video_id = 'gr-sanleandro-1'")
    meeting("gr-richmond-1", "no_episodes", captions="none", body="richmond", dur=7200.0)
    meeting("gr-richmond-2", "no_episodes", captions=None, body="richmond", dur=1800.0)
    todo = [r["video_id"] for r in asr_batch.select(con, ["richmond", "sanleandro"], "short")]
    done_n, done_h = asr_batch.already_done(con, ["richmond", "sanleandro"])
    check(todo == ["gr-richmond-2", "gr-richmond-1"] and (done_n, round(done_h, 2)) == (1, 1.0),
          "asr_batch: a meeting whose row already says captions='asr_local' is skipped (a killed run resumes by restarting); "
          "--order short is shortest first; captions NULL still counts as to-do", f"{todo} done={done_n}")
    check(asr_batch.BLOCKED.search("Server returned 403 Forbidden") is not None
          and asr_batch.BLOCKED.search("HTTP error 429 Too Many Requests") is not None
          and asr_batch.BLOCKED.search("fetched 132 MB wav in 0.3 min") is None,
          "asr_batch: a 403/429 from the archive host stops the run; an ordinary progress line does not")

    # -- mlx-whisper's Metal buffer limit: a long meeting is transcribed in pieces
    # (gr-richmond-6042, 7.53 h, asked for 4.35 GB and died 2026-09-23)
    import local_asr                                        # noqa: E402
    check(local_asr.chunk_offsets(3600.0, 5400.0) == [] and local_asr.chunk_offsets(None, 5400.0) == []
          and local_asr.chunk_offsets(5400.0, 5400.0) == [],
          "local_asr.chunk_offsets: a meeting that fits in one buffer is transcribed whole, as before")
    offs = local_asr.chunk_offsets(7.53 * 3600, 5400.0)
    check(offs == [0.0, 5400.0, 10800.0, 16200.0, 21600.0, 27000.0] and offs[-1] < 7.53 * 3600,
          "local_asr.chunk_offsets: 7.53 h -> six 90-min pieces, the last one a 108 s remainder; no piece starts past the end",
          str(offs))
    segs = local_asr.shift_segments([{"start": 0.0, "end": 4.5, "text": "housing element"},
                                     {"start": 4.5, "end": 9.0, "text": "rezoning"}], 5400.0)
    check([(s["start"], s["end"]) for s in segs] == [(5400.0, 5404.5), (5404.5, 5409.0)]
          and segs[0]["text"] == "housing element",
          "local_asr.shift_segments: a chunk's cues come back onto the MEETING's clock, or every episode after 90 min "
          "would be cut from the wrong place", str([(s["start"], s["end"]) for s in segs]))
    check(local_asr.vtt_time(5404.5) == "01:30:04.500",
          "local_asr.vtt_time: a shifted cue still writes a valid VTT timestamp", local_asr.vtt_time(5404.5))
    con.close()


def section_t2d(tmp):
    print("\n26. T2 part 2 (2026-09-23): the caption-completeness audit, and the ASR queue refusing to overwrite a "
          "caption track that is fine")
    sys.path.insert(0, str(ROOT / "pipeline"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import state                                           # noqa: E402
    import caption_audit as ca                             # noqa: E402
    import asr_batch                                       # noqa: E402
    d = tmp / "t2d"; d.mkdir()

    # -- words per minute: the measurement the whole audit rests on
    check(abs(ca.words_per_minute(18000, 2 * 3600) - 150.0) < 1e-6,
          "caption_audit.words_per_minute: 18,000 words over a 2 h meeting = 150 wpm (normal speech)",
          str(ca.words_per_minute(18000, 2 * 3600)))
    check(abs(ca.words_per_minute(1174, 0.62 * 3600) - 31.6) < 0.5,
          "caption_audit.words_per_minute: gr-hayward-2458's real numbers (1,174 w / 0.62 h) = ~32 wpm, under the floor",
          f"{ca.words_per_minute(1174, 0.62 * 3600):.1f}")
    check(ca.words_per_minute(5000, 0) is None and ca.words_per_minute(5000, None) is None,
          "caption_audit.words_per_minute: an unknown meeting length is None, never a verdict")
    check(ca.coverage(600.0, 7200.0) == 1 / 12 and ca.coverage(7260.0, 7200.0) == 1.0,
          "caption_audit.coverage: capped at 1.0 -- a caption clock may overrun the listing's duration")

    # -- the failure a word count cannot see: a complete track in the wrong language
    en = ("Thank you. The council will now consider the housing element and the rezoning of the site. "
          "We have a staff report on this item and we will take public comment before we vote on it. ") * 3
    es = ("Gracias. La Junta de Supervisores del Condado de Napa se reunió para considerar el informe "
          "que se presente por el personal y las medidas que se adopten en esta sesión. ") * 3
    check(ca.guess_language(en) == "en", "caption_audit.guess_language: an English council track reads 'en'", ca.guess_language(en))
    check(ca.guess_language(es) == "es",
          "caption_audit.guess_language: Napa's Spanish Granicus track reads 'es' -- 141 wpm and still unreadable by "
          "English keywords", ca.guess_language(es))
    check(ca.guess_language("housing element rezoning") == "unknown",
          "caption_audit.guess_language: too few words is 'unknown', not a confident condemnation")

    # -- the flags on one track
    full = ca.audit_track([{"start": 0.0, "end": 7200.0, "text": " ".join(["word"] * 18000)}], 7200.0)
    check(full["words"] == 18000 and abs(full["wpm"] - 150.0) < 1e-6 and full["coverage"] == 1.0 and not ca.flags(full),
          "caption_audit.audit_track: a full 2 h track at 150 wpm carries no flags", str(full))
    short = ca.audit_track([{"start": 0.0, "end": 600.0, "text": " ".join(["word"] * 900)}], 7200.0)
    check("truncated" in ca.flags(short) and "stops_early" in ca.flags(short),
          "caption_audit.flags: 900 words ending at 10 min of a 2 h meeting is BOTH truncated and stops_early -- the "
          "two failures are reported apart because their remedies differ", str(ca.flags(short)))
    spanish = ca.audit_track([{"start": 0.0, "end": 7200.0, "text": es * 210}], 7200.0)   # ~18,000 words = 150 wpm
    check(ca.flags(spanish) == ["non_english"],
          "caption_audit.flags: a fluent, complete Spanish track is flagged ONLY non_english -- wpm alone passes it",
          str(ca.flags(spanish)))
    check(ca.flags(ca.audit_track([], 7200.0)) == ["empty"],
          "caption_audit.flags: the 8-byte WEBVTT some Granicus sites serve is 'empty', not 'truncated'")

    # -- per-body verdicts: 'no track' is never averaged into a wpm
    check(ca.body_verdict([full, full, full], 0).startswith("healthy"),
          "caption_audit.body_verdict: three good tracks = healthy (Santa Clara's shape)", ca.body_verdict([full, full, full], 0))
    check("NO track" in ca.body_verdict([full, full], 26),
          "caption_audit.body_verdict: Hayward's shape -- the 2 real tracks are fine and 26 meetings have NO track at "
          "all, which is a different failure with a different remedy", ca.body_verdict([full, full], 26))
    check("WRONG LANGUAGE" in ca.body_verdict([spanish] * 20, 10),
          "caption_audit.body_verdict: Napa's shape -- the classifier result is void, not thin",
          ca.body_verdict([spanish] * 20, 10))
    check(ca.body_verdict([], 29) == "NO CAPTION TRACK on any meeting",
          "caption_audit.body_verdict: a body with no track anywhere says so instead of printing a median of nothing")
    check("TRUNCATED" in ca.body_verdict([short, short, full], 0),
          "caption_audit.body_verdict: most tracks truncated -> re-locate by ASR", ca.body_verdict([short, short, full], 0))

    # -- the queue guard: ASR must not replace a caption track that is fine
    good = d / "good.vtt"
    good.write_text("WEBVTT\n\n" + "".join(
        f"00:{m:02d}:00.000 --> 00:{m + 1:02d}:00.000\n" + " ".join(f"w{m}x{k}" for k in range(150)) + "\n\n"
        for m in range(0, 59)), encoding="utf-8")
    thin = d / "thin.vtt"
    thin.write_text("WEBVTT\n\n00:00:00.000 --> 00:10:00.000\n" + " ".join(f"t{k}" for k in range(200)) + "\n",
                    encoding="utf-8")
    db = d / "state.sqlite"
    con = state.connect(db)
    now = state.now_iso()

    def meeting(vid, captions, path, dur=3600.0, status="no_episodes"):
        con.execute("INSERT INTO videos (video_id, speaker, channel_url, month, crawled_at, in_frame, status, updated_at, "
                    "kind, platform, meeting_id, meeting_date, duration_s, captions, captions_path) "
                    "VALUES (?,'hayward','u','2026-09',?,1,?,?,'meeting','granicus',?,'2026-09-15',?,?,?)",
                    (vid, now, status, now, f"hayward:{vid}", dur, captions, str(path) if path else None))

    meeting("gr-hayward-good", "granicus", good)              # 8,850 words in 60 min = 148 wpm
    meeting("gr-hayward-thin", "granicus", thin)              # 200 words in 60 min = 3 wpm
    meeting("gr-hayward-none", "none", None)
    meeting("gr-hayward-cut", "none", None, status="cut")     # agenda_index episodes already cut
    meeting("gr-hayward-done", "asr_local", good)             # already ASR'd: always skipped
    con.commit()

    rows = {r["video_id"] for r in asr_batch.select(con, ["hayward"])}
    check("gr-hayward-good" not in rows,
          "asr_batch.select: a meeting whose caption track PASSES the audit is NOT queued -- local_asr --register would "
          "overwrite captions='granicus' with a deliberately low-quality locate track", str(sorted(rows)))
    check("gr-hayward-thin" in rows,
          "asr_batch.select: a truncated caption track IS queued -- that is what the ASR route is for", str(sorted(rows)))
    check("gr-hayward-none" in rows and "gr-hayward-cut" in rows,
          "asr_batch.select: meetings with no track are queued whatever stage has already touched the row", str(sorted(rows)))
    check("gr-hayward-done" not in rows,
          "asr_batch.select: captions='asr_local' is still the resume key -- a finished meeting is never fetched twice")
    check("gr-hayward-good" in {r["video_id"] for r in asr_batch.select(con, ["hayward"], redo_captioned=True)},
          "asr_batch.select --redo-captioned: a human can still force it, and the help text says the track is replaced")
    check(len(asr_batch.select(con, ["hayward"], limit=2)) == 2,
          "asr_batch.select: --max still limits AFTER the caption-track filter, so it means 'this many meetings to do'")

    # -- the audit over a DB, and the CSV
    audit_rows, per_body = ca.audit(con, ["hayward"])
    by_id = {r["video_id"]: r for r in audit_rows}
    check(by_id["gr-hayward-none"]["flags"] == "no_track" and by_id["gr-hayward-none"]["wpm"] is None,
          "caption_audit.audit: a meeting with no track is recorded as no_track with no wpm, not as 0 wpm -- averaging "
          "zeros is exactly how Hayward read as '1,592 words per meeting'", str(by_id["gr-hayward-none"]["flags"]))
    check(per_body["hayward"]["no_track"] == 2 and per_body["hayward"]["n_tracks"] == 3,
          "caption_audit.audit: the per-body row counts tracks and no-track meetings separately",
          str({k: per_body["hayward"][k] for k in ("meetings", "n_tracks", "no_track")}))
    check(140 < per_body["hayward"]["median_wpm"] < 155,
          "caption_audit.audit: the median wpm is over the TRACKS only", str(per_body["hayward"]["median_wpm"]))
    csv_path = ca.write_csv(audit_rows, d / "audit.csv")
    import csv as _csv
    with open(csv_path, encoding="utf-8") as fh:
        got = list(_csv.DictReader(fh))
    check(len(got) == 5 and {r["video_id"] for r in got} == set(by_id),
          "caption_audit.write_csv: one row per meeting, every meeting present", f"{len(got)} rows")
    check(any(r["wpm"] == "" for r in got),
          "caption_audit.write_csv: an unknown wpm is an empty cell, not a zero")
    con.close()


def section_t2e(tmp):
    print("\n27. T2 part 2 (2026-09-23): one slow slice is one clip's failure, and stage 1 reads the caption track's "
          "own language")
    sys.path.insert(0, str(ROOT / "pipeline"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import state                                           # noqa: E402
    import granicus                                        # noqa: E402
    import classify                                        # noqa: E402
    import segment as sg                                   # noqa: E402
    import caption_audit as ca                             # noqa: E402
    d = tmp / "t2e"; d.mkdir()

    # -- the timeout that took the 2026-09-23 cut run down -------------------
    check(granicus.slice_timeout(1020.0) > 1800.0,
          "granicus.slice_timeout: the 17-min slice that killed the cut run (gr-santaclara-2497_e1) gets more than the "
          "flat 1800 s it used to get", f"{granicus.slice_timeout(1020.0):.0f} s")
    check(granicus.slice_timeout(60.0) == granicus.SLICE_TIMEOUT_FLOOR_S,
          "granicus.slice_timeout: a short slice gets the floor -- a 30 s slice measured 175 s of wall clock, nearly "
          "all of it fixed cost (manifest walk + seek), so scaling on length alone would fail it",
          f"{granicus.slice_timeout(60.0):.0f} s")
    check(granicus.slice_timeout(3600.0) == granicus.SLICE_TIMEOUT_MAX_S,
          "granicus.slice_timeout: capped -- past 3 h ffmpeg is hung, not slow, and an agenda_index item is at most "
          "60 min (agenda_index.MAX_ITEM_S)", f"{granicus.slice_timeout(3600.0):.0f} s")
    check(granicus.slice_timeout(600.0) > 2.25 * 600.0,
          "granicus.slice_timeout: every budget clears the worst wall/duration ratio measured over the 53 slices the "
          "crashed run did cut (2.25x)")

    # ffmpeg picks its muxer from the extension, so the scratch file keeps it:
    # writing to '<name>.m4a.part' failed 26 slices with "Invalid argument"
    part = granicus.part_path(d / "gr-x_e1.m4a")
    check(part.name == "gr-x_e1.part.m4a" and part.suffix == ".m4a",
          "granicus.part_path: the container extension stays LAST, or ffmpeg cannot open the output at all", part.name)

    fake = d / "fakeffmpeg.sh"
    fake.write_text("#!/bin/sh\nsleep 30\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    real_cmd = granicus.audio_slice_command
    granicus.audio_slice_command = lambda *a, **k: [str(fake)]
    try:
        ok, detail, wall = granicus.audio_slice("http://example.invalid/x.m3u8", 0.0, 10.0, d / "slow.m4a", timeout=1.0)
    finally:
        granicus.audio_slice_command = real_cmd
    check(ok is False and "timed out" in str(detail),
          "granicus.audio_slice: a slice that runs past its budget RETURNS a failure -- letting TimeoutExpired escape "
          "killed a 5 h run on its 54th clip and stranded the meeting in 'downloading'", str(detail))
    check(not (d / "slow.m4a").exists(),
          "granicus.audio_slice: nothing lands at the real path unless ffmpeg exited 0, so episodes.py's 'the clip is "
          "already on disk' check can never pick up a truncated slice")

    quick = d / "quickffmpeg.sh"
    quick.write_text("#!/bin/sh\n: > \"$1\"\n")
    quick.chmod(quick.stat().st_mode | stat.S_IEXEC)
    granicus.audio_slice_command = lambda m, s, dur, dest: [str(quick), str(dest)]
    try:
        ok2, _d2, _w2 = granicus.audio_slice("http://example.invalid/x.m3u8", 0.0, 10.0, d / "good.m4a")
    finally:
        granicus.audio_slice_command = real_cmd
    check(ok2 and (d / "good.m4a").exists() and not granicus.part_path(d / "good.m4a").exists(),
          "granicus.audio_slice: a slice that exits 0 is renamed onto the real path and leaves no scratch file")

    # -- stage 1 picks its keyword files by the track's language -------------
    es_files = classify.keyword_files_for("es")
    check([p.name for p in es_files] == ["keywords_density_es.txt", "keywords_transit_es.txt"],
          "classify.keyword_files_for('es'): the Spanish pair", str([p.name for p in es_files]))
    check(classify.keyword_files_for("fr") == classify.keyword_files_for("en")
          and classify.keyword_files_for(None) == classify.keyword_files_for("en"),
          "classify.keyword_files_for: an unsupported or undetected language falls back to English -- a detector "
          "failure must not silently invent a new instrument")
    kw_en, kw_es = classify.keywords_for("en"), classify.keywords_for("es")
    check(kw_es.families() == kw_en.families() == ["density", "transit"],
          "the Spanish files carry the SAME family tags: 'tags' on a Napa episode has to mean what it means on a "
          "Fremont one", str(kw_es.families()))
    check("desplazamiento" in kw_es.conditional,
          "keywords_density_es: 'desplazamiento' is CONDITIONAL -- it is displacement in housing Spanish and a commute "
          "in transport Spanish, the one term in the pair that is genuinely two families")
    check("frecuencia" in kw_es.conditional,
          "keywords_transit_es: 'frecuencia' is conditional although English 'frequency' is not -- 'con frecuencia' is "
          "just 'often' and would fire in every meeting")

    es_text = ("Gracias senor presidente. La Junta de Supervisores va a considerar el elemento de vivienda y la "
               "rezonificacion del sitio, con mas unidades de vivienda asequible cerca de la estacion de tren, y "
               "el personal presentara el informe sobre el transporte publico y la ciclovia que se propone para "
               "esta area de la ciudad. ")
    en_text = ("Thank you. The council will now consider the housing element and the rezoning of this site, with more "
               "affordable housing units near the train station, and staff will present the report on public transit "
               "and the bike lane proposed for this part of the city. ")
    es_segs = [{"start": float(i * 60), "end": float(i * 60 + 60), "text": es_text} for i in range(20)]
    en_segs = [{"start": float(i * 60), "end": float(i * 60 + 60), "text": en_text} for i in range(20)]
    check(sg.segments_language(es_segs) == "es" and sg.segments_language(en_segs) == "en",
          "segment.segments_language: the detector caption_audit reports is the one stage 1 now selects with -- the "
          "audit and the classifier can never disagree about what language a track is in")
    check(ca.guess_language is sg.guess_language,
          "scripts/caption_audit.py uses segment.guess_language, not a second copy of it")

    n_es_with_es = len(sg.candidate_windows(es_segs, kw_es, 60, 0.5, 240))
    n_es_with_en = len(sg.candidate_windows(es_segs, kw_en, 60, 0.5, 240))
    check(n_es_with_es > 0 and n_es_with_en == 0,
          "stage 1 on a Spanish track: the Spanish files find candidate windows where the English files find NONE -- "
          "this is the whole of Napa's 104 audio-h and 0 episodes", f"es={n_es_with_es} en={n_es_with_en}")
    check(len(sg.candidate_windows(en_segs, kw_en, 60, 0.5, 240)) > 0,
          "stage 1 on an English track is unchanged")

    # -- the instrument is NAMED: a Spanish reading is not an English one ----
    check(classify.CODEBOOK_VERSION == "t0-a4a8e55760",
          "the codebook is byte-identical: changing it would re-version -- and re-bill -- every window measured so "
          "far, so the Spanish hint lives in the user turn instead", classify.CODEBOOK_VERSION)
    pv_en = classify.prompt_version_for("claude-sonnet-4-6")
    pv_es = classify.prompt_version_for("claude-sonnet-4-6", "es")
    check(pv_en == classify.CODEBOOK_VERSION and pv_es.endswith("/es") and pv_en != pv_es,
          "classify.prompt_version_for: a Spanish reading gets its own prompt_version, so the two recalls can never "
          "be pooled by a SELECT that does not know about them", f"{pv_en} vs {pv_es}")
    req_en = classify.request_params("claude-sonnet-4-6", "TEXT")
    req_es = classify.request_params("claude-sonnet-4-6", "TEXT", "es")
    check(req_en["system"] == req_es["system"],
          "the cached system block (the codebook) is identical for both languages, so the cache still takes")
    check("SPANISH INTERPRETATION" in req_es["messages"][0]["content"]
          and req_en["messages"][0]["content"].startswith("Transcript window"),
          "an English window is byte-identical to every window classified before 2026-09-23; only the Spanish one "
          "carries the note")
    check(classify.worst_case_usd("TEXT", "claude-sonnet-4-6", lang="es")
          > classify.worst_case_usd("TEXT", "claude-sonnet-4-6"),
          "classify.worst_case_usd counts the language note it will actually send")

    # -- selection: a meeting the ENGLISH instrument answered 'no_episodes' --
    db = d / "state.sqlite"
    con = state.connect(db)
    now = state.now_iso()
    es_vtt = d / "napa.vtt"
    es_vtt.write_text("WEBVTT\n\n" + "".join(
        f"00:{m:02d}:00.000 --> 00:{m:02d}:59.000\n{es_text}\n\n" for m in range(0, 20)), encoding="utf-8")

    def meeting(vid, body, status, path, meta="{}"):
        con.execute("INSERT INTO videos (video_id, speaker, channel_url, month, crawled_at, in_frame, status, updated_at, "
                    "kind, platform, meeting_id, meeting_date, duration_s, captions, captions_path, raw_meta) "
                    "VALUES (?,?,'u','2026-09',?,1,?,?,'meeting','granicus',?,'2026-09-15',3600.0,'granicus',?,?)",
                    (vid, body, now, status, now, f"{body}:{vid}", str(path), meta))

    meeting("gr-napa-a", "napa", "no_episodes", es_vtt)
    meeting("gr-napa-b", "napa", "classified", es_vtt)
    meeting("gr-napa-c", "napa", "captioned", es_vtt)
    con.commit()

    class A:
        bodies = "napa"; max = 0; include_classified = False; include_no_episodes = False
    plain = {r["video_id"] for r in classify.select_meetings(con, A())}
    check(plain == {"gr-napa-c"},
          "classify.select_meetings: 'no_episodes' is a real answer everywhere else and is NOT re-read by default",
          str(sorted(plain)))
    A.include_classified = True; A.include_no_episodes = True
    both = {r["video_id"] for r in classify.select_meetings(con, A())}
    check(both == {"gr-napa-a", "gr-napa-b", "gr-napa-c"},
          "classify.select_meetings --include-no-episodes: the 20 Napa meetings the English keywords could not have "
          "matched become readable again", str(sorted(both)))

    row = con.execute("SELECT * FROM videos WHERE video_id = 'gr-napa-a'").fetchone()
    check(classify.caption_language(row, sg.read_segments(es_vtt)) == "es",
          "classify.caption_language: read off the track itself when the site declares nothing -- Napa's Flowplayer "
          "object has no lang attribute, which is exactly the case this exists for")
    con.execute("UPDATE videos SET raw_meta = ? WHERE video_id = 'gr-napa-b'", (json.dumps({"captions_lang": "en-US"}),))
    declared = con.execute("SELECT * FROM videos WHERE video_id = 'gr-napa-b'").fetchone()
    check(classify.caption_language(declared, sg.read_segments(es_vtt)) == "en",
          "classify.caption_language: a site that DOES declare a language in the <track> element wins over the guess")

    check(classify.finish_status(con, row, "classified", "2 episodes") is True
          and con.execute("SELECT status FROM videos WHERE video_id = 'gr-napa-a'").fetchone()["status"] == "classified",
          "classify.finish_status: a re-read that finds episodes moves the row off 'no_episodes', or episodes.py would "
          "never come for them")
    row_b = con.execute("SELECT * FROM videos WHERE video_id = 'gr-napa-b'").fetchone()
    check(classify.finish_status(con, row_b, "no_episodes", "0 episodes") is False
          and con.execute("SELECT status FROM videos WHERE video_id = 'gr-napa-b'").fetchone()["status"] == "classified",
          "classify.finish_status: a re-read that finds nothing never clobbers another instrument's status")
    con.close()

    # -- coverage: the cheap detector for a Granicus-side truncated track ----
    dense_then_stops = ca.audit_track(
        [{"start": float(i * 60), "end": float(i * 60 + 60), "text": " ".join(["word"] * 139)} for i in range(117)],
        18000.0)
    check(abs(dense_then_stops["coverage"] - 0.39) < 0.01,
          "caption_audit.coverage: gr-fremont-453's shape -- 1,720 cues that stop at 39 % of a 5 h meeting",
          f"{dense_then_stops['coverage']:.2f}")
    check("stops_early" in ca.flags(dense_then_stops),
          "caption_audit.flags: the truncated Fremont track is caught by coverage", str(ca.flags(dense_then_stops)))
    almost = ca.audit_track([{"start": 0.0, "end": 6480.0, "text": " ".join(["word"] * 15000)}], 7200.0)
    check("stops_early" in ca.flags(almost) and "truncated" not in ca.flags(almost),
          "caption_audit.flags: a track that stops at 90 % is caught even though it reads 125 wpm and clears the wpm "
          "floor -- the failure a word count alone cannot see", str(ca.flags(almost)))
    check(not ca.flags(ca.audit_track([{"start": 0.0, "end": 7190.0, "text": " ".join(["word"] * 15000)}], 7200.0)),
          "caption_audit.flags: a complete track is not flagged -- coverage is bimodal (median 1.000, p10 0.993 over "
          "the 168 real tracks), so 0.95 has nothing to make a false positive out of")


def section_t2f(tmp):
    print("\n28. T2 part 2 (2026-09-23): what b17 actually lost the Fremont clips to, and what a window costs")
    sys.path.insert(0, str(ROOT / "pipeline"))
    import classify                                        # noqa: E402
    import transcode                                       # noqa: E402
    d = tmp / "t2f"; d.mkdir()

    # -- the encode path: polarity, then level ------------------------------
    # b17's 30 Fremont failures were NOT quiet audio. Fremont's Granicus MP4s
    # are polarity-inverted stereo, and `-ac 1` averages the two channels into
    # 35 dB of cancellation. Measured on b17: 32/32 Fremont clips inverted,
    # 51/51 clips of the other six bodies at 0.0 dB. These fixtures are the two
    # shapes, synthesised so the test needs no corpus audio.
    def sine(path, filt):
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
                        "-i", "sine=frequency=440:duration=3:sample_rate=48000",
                        "-filter_complex", filt, "-map", "[out]", "-c:a", "aac", str(path)],
                       check=True, capture_output=True)

    inverted = d / "inverted.m4a"                   # L = -R: the Fremont shape
    sine(inverted, "[0:a]asplit[a][b];[b]volume=-1[n];[a][n]amerge=inputs=2[out]")
    dual_mono = d / "dualmono.m4a"                  # L = R: every other body
    sine(dual_mono, "[0:a]asplit[a][b];[a][b]amerge=inputs=2[out]")

    af_inv, why_inv = transcode.downmix_filter(inverted)
    af_dm, why_dm = transcode.downmix_filter(dual_mono)
    check(af_inv == "pan=mono|c0=c0",
          "transcode.downmix_filter: polarity-inverted stereo takes ONE channel -- averaging it is the 35 dB "
          "cancellation that made b17's VAD hear nothing", f"{af_inv} ({why_inv})")
    check(af_dm is None,
          "transcode.downmix_filter: ordinary dual-mono is left to `-ac 1` exactly as before -- 51 of b17's 53 passing "
          "clips must encode byte-for-byte unchanged", f"{af_dm} ({why_dm})")

    lvl_one = transcode.volumedetect(inverted, af="pan=mono|c0=c0")
    lvl_sum = transcode.volumedetect(inverted, af="pan=mono|c0=0.5*c0+0.5*c1")
    check(lvl_sum["mean"] - lvl_one["mean"] <= -transcode.POLARITY_DROP_DB,
          "the fixture really does cancel: the summed downmix sits far below either channel alone",
          f"{lvl_sum['mean'] - lvl_one['mean']:.1f} dB")

    # -- the SAME check on the full-ASR route (b43's 24 Fremont meetings) ----
    # scripts/fullasr.py pulled whole meetings with a plain `-ac 1` long after
    # the clip path learned better, and cancelled all 24 Fremont meetings twice
    # over: 49.3 audio-h at QA coverage 0-13 %. It now asks transcode for the
    # same verdict, and REFUSES when it cannot measure the source -- a silent
    # `-ac 1` fallback is what cost the 49.3 h.
    sys.path.insert(0, str(ROOT / "scripts"))
    import fullasr as fa                                   # noqa: E402
    args_inv, why_i, ok_i = fa.downmix_args(inverted)
    args_dm, why_d, ok_d = fa.downmix_args(dual_mono)
    args_no, why_n, ok_n = fa.downmix_args(d / "there-is-no-such-file.m4a")
    check(ok_i and args_inv[:2] == ["-af", "pan=mono|c0=c0"] and "-ac" in args_inv,
          "fullasr.downmix_args: an inverted whole meeting takes ONE channel and still lands mono",
          f"{args_inv} ({why_i})")
    check(ok_d and args_dm == ["-ac", "1"],
          "fullasr.downmix_args: dual-mono whole meetings keep the original `-ac 1` encode",
          f"{args_dm} ({why_d})")
    check(not ok_n and args_no is None,
          "fullasr.downmix_args: an unmeasurable source REFUSES rather than downmixing blind -- the silent "
          "fallback is what cancelled 49.3 h of Fremont twice", f"{args_no} ({why_n})")

    # ffprobe takes a different option set from ffmpeg: this build has no `-ss`,
    # and passing a seek to both made a stereo file read as "not stereo" and
    # cancel anyway. probe_args is what keeps the seek off ffprobe.
    af_seek, why_seek = transcode.downmix_filter(inverted, 2.0, ["-ss", "1"], probe_args=[])
    check(af_seek == "pan=mono|c0=c0",
          "transcode.downmix_filter: a measurement seek does not reach ffprobe, so the channel count survives it",
          f"{af_seek} ({why_seek})")

    # -- loudnorm fires on level alone, and only below the threshold --------
    quiet = d / "quiet.m4a"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=3:sample_rate=48000",
                    "-af", "volume=-40dB", "-ac", "1", "-c:a", "aac", str(quiet)],
                   check=True, capture_output=True)
    loud = d / "loud.m4a"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=3:sample_rate=48000",
                    "-ac", "1", "-c:a", "aac", str(loud)], check=True, capture_output=True)
    p_quiet, p_loud = transcode.encode_plan(quiet), transcode.encode_plan(loud)
    check(p_quiet["normalised"] and "loudnorm" in p_quiet["af"],
          "transcode.encode_plan: audio under NORMALISE_BELOW_DBFS is normalised",
          f"{p_quiet['pre_mean_dbfs']:.1f} dB -> {p_quiet['af']}")
    check(not p_loud["normalised"] and p_loud["af"] is None,
          "transcode.encode_plan: audio above the threshold gets NO filter at all, so a clip that already passed QA "
          "encodes exactly as it did before", f"{p_loud['pre_mean_dbfs']:.1f} dB")
    check(transcode.ffmpeg_command("a", "b", float("inf")) ==
          ["ffmpeg", "-nostdin", "-v", "error", "-i", "a", "-ac", "1", "-ar", "16000",
           "-c:a", "libopus", "-b:a", "16k", "-y", "b"],
          "transcode.ffmpeg_command with no filter is the command it has always been")

    ok, detail, levels = transcode.encode(inverted, d / "fixed.opus", float("inf"))
    check(ok and levels["post_mean_dbfs"] is not None and levels["post_mean_dbfs"] > lvl_sum["mean"] + 20,
          "transcode.encode: the inverted fixture comes out of the encoder with its speech intact, not 35 dB down",
          f"sum would be {lvl_sum['mean']:.1f} dB, encoded {levels['post_mean_dbfs']:.1f} dB")
    check(inverted.exists() and (d / "fixed.opus").exists(),
          "transcode.encode writes a NEW file and never touches the source m4a")

    # -- what a window costs: the four billed batches ------------------------
    # (n_windows, caption words, language, actual $ charged). The old estimator
    # priced words * 1.35 tokens and credited a cache read on every call.
    M = "claude-sonnet-4-6"
    billed = [(192, 80900, "en", 0.6381), (306, 151259, "en", 0.5518),
              (455, 188516, "en", 1.1051), (93, 39992, "es", 0.3901),
              (59, 29692, "es", 0.2162)]          # Napa's second half, 2026-09-23
    for n, words, lang, actual in billed:
        est = classify.estimate_usd(classify.caption_tokens_for(words, lang), n, M, batch=True)
        check(est >= actual * 0.99,
              f"classify.estimate_usd never comes in under the bill ({n} windows, {lang})",
              f"est ${est:.3f} vs ${actual:.3f} charged")
    napa = classify.estimate_usd(classify.caption_tokens_for(39992, "es"), 93, M, batch=True)
    check(abs(napa / 0.3901 - 1) < 0.15,
          "classify.estimate_usd on the Napa Spanish anchor: within 15 % of the $0.39 actually charged (it was "
          "$0.19 estimated, half the bill, which is what sent this item to the plan)", f"${napa:.3f}")

    check(classify.caption_tokens_for(1000, "es") > classify.caption_tokens_for(1000, "en"),
          "classify.caption_tokens_for: Spanish captions cost more per word (2.74 vs 2.04 measured) -- an interpreter's "
          "paraphrase tokenises worse, and pricing Napa at the English rate is a third of the error")
    check(classify.caption_tokens_for(1000, "fr") == classify.caption_tokens_for(1000, "en"),
          "classify.caption_tokens_for: an unknown language falls back to the English rate, never to zero")
    check(abs(classify.codebook_tokens() - 1531) < 40,
          "classify.codebook_tokens: matches the 1,531 tokens measured as cache_read_input_tokens on a batch that hit "
          "the cache every call", f"{classify.codebook_tokens():.0f}")

    small = classify.codebook_rate(M, classify.CACHE_PAYS_OFF_WINDOWS - 1)
    large = classify.codebook_rate(M, classify.CACHE_PAYS_OFF_WINDOWS)
    check(small == classify.PRICES[M]["input"] and large < small,
          "classify.codebook_rate: below a few hundred windows the prompt cache does not pay off (28 % of the codebook "
          "served from cache at 93 windows, 76 % at 455), so the codebook is priced at plain input",
          f"${small:.2f}/Mtok small, ${large:.2f}/Mtok large")
    check(large > classify.PRICES[M]["cache_read"],
          "classify.codebook_rate: even a large batch is NOT priced at a full cache read -- crediting one understates "
          "the 455-window batch by 26 %")

    # the rail that gates --cap-usd and spend.claude_refusal must stay above the bill
    window = " ".join(["palabra"] * 500)
    check(classify.worst_case_usd(window, M, batch=True, lang="es")
          > classify.estimate_usd(classify.caption_tokens_for(500, "es"), 1, M, batch=True),
          "classify.worst_case_usd stays above the expectation for the same window -- it is the number the spend cap "
          "is checked against")


def section_t2g(tmp):
    print("\n29. T2 part 2 (2026-09-24): caption queue ordered by measured yield, one request an hour, hard daily cap")
    sys.path.insert(0, str(ROOT / "pipeline"))
    from datetime import datetime, timedelta, timezone      # noqa: E402
    import state, captions                                  # noqa: E402
    d = tmp / "t2g"; d.mkdir()
    db = d / "state.sqlite"
    con = state.connect(db)
    now_i = state.now_iso()
    subs = d / "subs"; subs.mkdir()
    os.environ["SUBS_DIR"] = str(subs)

    def meeting(vid, body, date, captions_kind=None, status="discovered"):
        con.execute("INSERT INTO videos (video_id, speaker, channel_url, month, crawled_at, in_frame, status, updated_at, "
                    "kind, platform, meeting_date, captions) VALUES (?,?,'u','2026-09',?,1,?,?,'meeting','youtube',?,?)",
                    (vid, body, now_i, status, now_i, date, captions_kind))

    def episode(eid, vid, n=1):
        for i in range(n):
            con.execute("INSERT INTO episodes (episode_id, meeting_id, video_id, start_s, end_s, tags, method, status, updated_at) "
                        "VALUES (?,?,?,0,60,'transit','classifier','located',?)", (f"{eid}_{i}", vid, vid, now_i))

    # Three bodies with the shape the real queue has: a high-yield body, a
    # measured ZERO-yield body (Rohnert Park: 0 episodes over 5 captioned
    # meetings) and a body never captioned at all (Cotati), which has no
    # measurement -- not a measured zero.
    meeting("H0", "high", "2025-10-01", "auto", "captioned"); episode("eH0", "H0", 10)
    meeting("H1", "high", "2026-03-01"); meeting("H2", "high", "2026-01-01")
    meeting("L0", "low", "2025-10-01", "auto", "captioned"); episode("eL0", "L0", 1)
    meeting("L1", "low", "2026-02-01")
    meeting("Z0", "zero", "2025-10-01", "none", "no_episodes")
    meeting("Z1", "zero", "2025-11-01", "none", "no_episodes")
    meeting("Z2", "zero", "2026-02-01")
    meeting("N0", "new", "2026-02-01")                    # never captioned: no measurement

    yields, pooled = captions.body_yield(con, "youtube")
    check(abs(yields["high"]["yield"] - 10.0) < 1e-9 and abs(yields["low"]["yield"] - 1.0) < 1e-9
          and yields["zero"]["yield"] == 0.0 and yields["zero"]["captioned"] == 2,
          "body_yield: episodes located per captioned meeting, measured from the DB (a caption-less meeting still counts "
          "in the denominator -- it cost a request and returned nothing)", str({k: round(v['yield'], 2) for k, v in yields.items()}))
    check(abs(pooled - 11 / 4) < 1e-9 and not yields["new"]["measured"] and abs(yields["new"]["yield"] - pooled) < 1e-9,
          "body_yield: a body with no captioned meeting yet gets the POOLED rate as a prior and measured=False -- "
          "an unmeasured body is not a zero-yield body", f"pooled={pooled:.2f}")

    rows = captions.select_rows(con)
    ordered = captions.order_queue(rows, yields, pooled)
    ids = [r["video_id"] for r, _, _ in ordered]
    check(ids == ["H2", "H1", "N0", "L1", "Z2"],
          "order_queue: highest-yield body first, OLDEST meeting first inside a body; the unmeasured body sits at the "
          "pooled rate and the measured zero-yield body sorts last", str(ids))
    check(sorted(ids) == sorted(r["video_id"] for r in rows) and len(ids) == len(rows),
          "order_queue is an ORDERING: every waiting meeting is still in the queue, including the zero-yield body's "
          "(which bodies belong in the sample is a DECISIONS.md question, not a worker's)")
    line = captions.yield_line("high", yields, 1, 4, pooled)
    check("rank 1/4" in line and "10.00 ep/meeting" in line and "over 1 captioned" in line
          and "POOLED PRIOR" in captions.yield_line("new", yields, 3, 4, pooled),
          "the log line carries rank, yield and the denominator, so the order can be audited after the fact", line)

    # -- the rate limit, enforced from the DB so a restart cannot skip it -------
    now = datetime.now(timezone.utc)
    check(captions.rate_wait(con, 3600, 18)[0] == 0.0, "rate_wait: nothing recorded -> may request now")
    rid = captions.record_request(con, "H1", now=now - timedelta(minutes=10))
    wait, why = captions.rate_wait(con, 3600, 18, now=now)
    check(2900 < wait < 3100 and "gap" in why,
          "rate_wait: a request 10 min ago holds the next one for ~50 min -- the gap comes out of `worker_requests`, "
          "not a counter inside the process, so launchd's KeepAlive cannot restart into a burst", f"{wait:.0f} s")
    con.execute("DELETE FROM worker_requests WHERE id = ?", (rid,))

    # Five requests in 41 minutes is what the 2026-09-23 run actually did before
    # the 429. Under the new limiter the second of those five never goes out.
    for i in range(5):
        captions.record_request(con, f"burst{i}", now=now - timedelta(minutes=41 - 10 * i))
    check(captions.rate_wait(con, 3600, 18, now=now - timedelta(minutes=31))[0] > 0,
          "rate_wait: the 2026-09-23 burst (5 requests / 41 min) is refused at the second request")
    con.execute("DELETE FROM worker_requests")

    # Cap full: the wait is until the OLDEST request in the window ages out.
    for i in range(3):
        captions.record_request(con, f"c{i}", now=now - timedelta(hours=20 - i))
    wait, why = captions.rate_wait(con, 60, 3, now=now)
    check(3.5 * 3600 < wait < 4.5 * 3600 and "daily cap" in why,
          "rate_wait: with the cap full the wait is until the oldest request leaves the 24 h window -- never a guess, "
          "never longer than the window", f"{wait / 3600:.2f} h -- {why}")
    check(captions.rate_wait(con, 60, 0, now=now)[0] == 0.0, "rate_wait: --daily-cap 0 means no cap")
    check(captions.rate_wait(con, 60, 4, now=now)[0] == 0.0, "rate_wait: under the cap and past the gap -> may request now")

    # A one-shot run that is rate-limited makes NO request, changes no row and
    # -- the part that matters -- does not touch the backoff.
    fake = d / "yt-dlp"
    calls = d / "calls.txt"
    fake.write_text(f"#!/bin/sh\necho called >> {calls}\nexit 1\n")
    fake.chmod(0o755)
    os.environ["YTDLP_BIN"] = str(fake)
    try:
        rc = captions.main(["--state", str(db), "--max", "1", "--min-gap", "3600", "--max-gap", "3600", "--daily-cap", "3"])
        st = con.execute("SELECT status FROM videos WHERE video_id = 'H2'").fetchone()[0]
        check(rc == 0 and not calls.exists() and st == "discovered" and captions.backoff_left(con) == (0.0, None),
              "captions: a rate-limited one-shot run asks YouTube nothing, leaves the row 'discovered' and writes NO "
              "backoff -- the 24 h backoff still means a bot check and nothing else", f"rc={rc} status={st}")
        con.execute("DELETE FROM worker_requests")
        rc2 = captions.main(["--state", str(db), "--max", "1", "--min-gap", "3600", "--max-gap", "3600"])
        n_req = con.execute("SELECT COUNT(*) FROM worker_requests WHERE host = 'youtube'").fetchone()[0]
        called = calls.exists()
        check(rc2 == 0 and called and n_req == 1,
              "captions: the request it does make is recorded once in `worker_requests` (the next run's gap depends on it)",
              f"rc={rc2} called={called} rows={n_req}")
        first = con.execute("SELECT video_id FROM worker_requests WHERE host = 'youtube'").fetchone()[0]
        check(first == "H2", "captions: the meeting it asked for is the highest-yield body's oldest -- H2, not the "
                             "newest meeting of whichever body sorts first alphabetically", first)
    finally:
        os.environ.pop("YTDLP_BIN", None)
        os.environ.pop("SUBS_DIR", None)
    con.close()


def section_t2h(tmp):
    print("\n30. T2 part 2 (2026-09-24): the Swagit locate route (method='swagit_transcript') -- Petaluma leaves the "
          "YouTube queue, and the two traps plan/SOURCES-AUDIT.md paid for stay shut")
    sys.path.insert(0, str(ROOT / "pipeline"))
    sys.path.insert(0, str(ROOT))
    import state, classify, captions, swagit, segment as sg  # noqa: E402
    from analysis import facets                              # noqa: E402
    d = tmp / "t2h"; d.mkdir()
    con = state.connect(d / "state.sqlite")
    now = state.now_iso()

    # -- the route id: a Swagit transcript is its own instrument, never pooled
    check(classify.method_for("swagit") == "swagit_transcript"
          and classify.method_for("asr_local") == "local_asr"
          and classify.method_for("granicus") == "classifier" and classify.method_for("auto") == "classifier",
          "classify.method_for: captions='swagit' is method='swagit_transcript'; the other routes are unchanged",
          f"swagit -> {classify.method_for('swagit')}")
    check(swagit.METHOD in facets.LABELS,
          "analysis/facets.py has a label for the new route, so a per-method figure can name it",
          f"LABELS has {sorted(facets.LABELS)}")

    # -- the player page: agenda index (start AND end), per-word transcript, owner
    page = (
        '<html><head><title>Sep 14, 2026 City Council - Petaluma, CA</title></head><body>'
        '<source src="https://archive-stream.granicus.com/OnDemand/_definst_/mp4:swagitVideo/petalumaca/abc.mp4/playlist.m3u8">'
        # the agenda index is printed TWICE by the page (desktop + mobile)
        + 2 * ('<span class="playerControl" data-id="1" data-ts="0" data-end-ts="60" data-title="CALL TO ORDER"></span>'
               '<span class="playerControl" data-id="2" data-ts="60" data-end-ts="600" data-title="HOUSING ELEMENT"></span>')
        + '<div id="transcript-fragments">'
        '<div class="alert">* This transcript was created by voice-to-text technology.</div>'
        + "".join(f'<a href="#" data-ts="{t}" onclick="seek({t}); return false;" aria-label="x">{w} </a>'
                  for t, w in [(1.0, "GOOD"), (1.4, "EVENING"), (2.0, "EVERYONE"),
                               (30.0, "THE"), (30.4, "HOUSING"), (30.9, "ELEMENT"), (31.4, "REZONING")])
        + '</div></body></html>')
    p = swagit.parse_player(page, "petalumaca")
    check(p["owner"] == "petalumaca" and p["m3u8"] and p["m3u8"].endswith("playlist.m3u8")
          and "archive-stream.granicus.com" in p["m3u8"],
          "swagit.parse_player: the customer slug and the HLS master on archive-stream.granicus.com "
          "(the host granicus.audio_slice already cuts)", f"owner={p['owner']} m3u8={p['m3u8']}")
    check(p["index_points"] == [(0.0, 60.0, "CALL TO ORDER"), (60.0, 600.0, "HOUSING ELEMENT")],
          "swagit.parse_player: the agenda index is de-duplicated -- the page prints it twice, and a doubled item "
          "would weight every agenda_item lookup double", str(p["index_points"]))
    check(p["n_words"] == 7 and p["disclaimer"] and "voice-to-text" in p["disclaimer"],
          "swagit.parse_player: one entry per transcript word, and the voice-to-text provenance is recorded",
          f"{p['n_words']} words, disclaimer={bool(p['disclaimer'])}")

    # -- words -> cues -> .vtt -> segments, LOSSLESS (a numbered VTT reads back doubled)
    vtt = swagit.write_vtt(p["cues"], d / "track.en.vtt")
    segs = sg.read_segments(vtt)
    check(len(segs) == len(p["cues"])
          and sum(len(s["text"].split()) for s in segs) == sum(len(c["text"].split()) for c in p["cues"]) == 7,
          "swagit.write_vtt round-trips through segment.read_segments without gaining or losing a word "
          "(no cue-identifier lines: segment.vtt_to_segments would read them as text)",
          f"{len(p['cues'])} cues -> {len(segs)} segments")
    check(all(c["end"] > c["start"] for c in p["cues"]) and p["cues"][0]["start"] == 1.0,
          "every cue has positive length and starts at its first word", str(p["cues"][:1]))

    # -- TRAP 2: a clip that belongs to another customer must not be ingested
    carmel = swagit.parse_player(page.replace("swagitVideo/petalumaca/", "swagitVideo/carmelin/"), "petalumaca")
    check((carmel["owner"] or "").lower() != "petalumaca",
          "SOURCES-AUDIT trap 2: a page whose stream URL names another customer (Carmel, Indiana) fails the owner "
          "check fetch_player raises on -- /views/1 serves Carmel on every Swagit host", f"owner={carmel['owner']}")
    check("new.swagit.com" in swagit.player_url("petalumaca", "1") and "/views/" not in swagit.PLAYER_URL,
          "SOURCES-AUDIT trap 1: the host pattern is <slug>.new.swagit.com, and this module never reads /views/1",
          swagit.player_url("petalumaca", "1"))
    for bad_url in ("https://www.youtube.com/watch?v=x", "https://r1.googlevideo.com/x"):
        try:
            swagit.fetch(bad_url)
            raised = False
        except ValueError:
            raised = True
        except Exception:                                    # noqa: BLE001 -- anything else means it was not refused
            raised = False
        check(raised, f"swagit.fetch refuses a YouTube host outright ({bad_url.split('/')[2]})", bad_url)

    # -- PrimeGov listing: only council meetings that carry a clip id of their own
    rows = swagit.parse_primegov([
        {"title": "CITY COUNCIL/PCDSA REGULAR MEETING", "date": "Sep 14, 2026", "swagitId": "videos/400885",
         "videoUrl": "https://petalumaca.new.swagit.com/videos/400885"},
        {"title": "Planning Commission Meeting", "date": "Sep 15, 2026", "swagitId": "events/43323", "videoUrl": ""},
        {"title": "CITY COUNCIL/PCDSA REGULAR MEETING", "date": "Aug 03, 2026", "swagitId": None, "videoUrl": ""},
    ])
    check([r["clip_id"] for r in rows] == ["400885"] and rows[0]["date"] == "2026-09-14",
          "swagit.parse_primegov: council meetings with a numeric swagitId only -- a commission, and a council "
          "meeting with no clip id, are both skipped", str([(r["date"], r["clip_id"]) for r in rows]))

    # -- the queue: a meeting with a non-YouTube route is not asked of YouTube
    def meeting(vid, platform, status, date, captions_kind=None, path=None):
        con.execute("INSERT INTO videos (video_id, speaker, channel_url, month, crawled_at, in_frame, status, updated_at, "
                    "kind, platform, meeting_id, meeting_date, captions, captions_path) "
                    "VALUES (?,'petaluma','u','2026-09',?,0,?,?,'meeting',?,?,?,?,?)",
                    (vid, now, status, now, platform, f"petaluma:{date}", date, captions_kind, path))

    meeting("yt-routed", "youtube", "discovered", "2025-10-06")
    meeting("sw-petalumaca-357537", "swagit", "captioned", "2025-10-06", "swagit", str(vtt))
    meeting("yt-alone", "youtube", "discovered", "2025-11-24")          # the short-clip special: no route
    meeting("yt-emptytwin", "youtube", "discovered", "2025-12-01")
    meeting("sw-empty", "swagit", "captioned", "2025-12-01", "none", None)   # a route with no track is no route
    q = [r["video_id"] for r in captions.select_rows(con)]
    check(q == ["yt-alone", "yt-emptytwin"],
          "captions.select_rows drops a meeting whose Swagit twin carries a real transcript, and keeps one whose "
          "twin has captions='none' -- the YouTube row is never deleted or re-stated, only unqueued", str(q))
    q_all = [r["video_id"] for r in captions.select_rows(con, skip_routed=False)]
    check(sorted(q_all) == ["yt-alone", "yt-emptytwin", "yt-routed"],
          "captions.select_rows(skip_routed=False) -- `--include-routed` -- asks for it anyway", str(sorted(q_all)))
    check(con.execute("SELECT status FROM videos WHERE video_id = 'yt-routed'").fetchone()["status"] == "discovered",
          "the superseded YouTube row still says 'discovered': nothing was deleted, no status was invented "
          "(AGENTS.md rule 6)")

    # -- the episodes rows: the route on the row, the agenda item from the page
    calls = {"c1": {"id": "c1", "video_id": "sw-petalumaca-357537", "start_s": 120.0, "end_s": 300.0,
                    "label": "about_density", "rationale": "rezoning", "family_rates": {"density": 2.0}}}
    classify.write_episodes(con, "sw-petalumaca-357537", "petaluma:2025-10-06", calls, 90,
                            swagit.granicus_index_points(p["index_points"]), "claude-sonnet-4-6", "t0-x",
                            classify.method_for("swagit"))
    e = con.execute("SELECT * FROM episodes WHERE video_id = 'sw-petalumaca-357537'").fetchone()
    check(e["method"] == "swagit_transcript" and e["episode_id"] == "sw-petalumaca-357537_e1"
          and e["agenda_item"] == "HOUSING ELEMENT",
          "an episode located on a Swagit transcript carries method='swagit_transcript' and the agenda item the "
          "page's own index puts over that span", f"{e['method']} / {e['agenda_item']}")

    # -- a short clip must not retire a long meeting (the closed-session specials)
    check(swagit.MIN_DURATION_SHARE == 0.5,
          "a Swagit clip shorter than half the known meeting duration is not registered: Petaluma's 2025-09-29 and "
          "2025-11-24 specials publish 2-minute closed-session clips against 42-min and 1.78-h YouTube videos",
          str(swagit.MIN_DURATION_SHARE))
    con.close()


def section_t2i(tmp):
    print("\n31. Granicus local-wav cutter (2026-10-16): a meeting with no m3u8 whose whole audio the local ASR run left "
          "in output/t2/asr/ is cut from that wav, offline -- 94 Richmond / San Leandro episodes sat 'located' behind a no-op")
    sys.path.insert(0, str(ROOT / "pipeline"))
    import io, contextlib
    import state, episodes, transcode  # noqa: E402
    d = tmp / "t2i"; d.mkdir()
    db = d / "state.sqlite"
    con = state.connect(db)
    now = state.now_iso()
    asr = d / "asr"; asr.mkdir()
    have_ffmpeg = shutil.which("ffmpeg") is not None
    # the fixture: a 300 s 16 kHz mono pcm wav exactly as scripts/local_asr.py writes it, and the ASR
    # track whose cue edges the bounds snap to (29.0 is 1 s from the padded start 30.0; 151.0 is 1 s
    # from the padded end 150.0; nothing near 220.0; the second episode is clamped to the wav's 300 s)
    wav = asr / "gr-rich-1.wav"
    if have_ffmpeg:
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=300",
                        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-y", str(wav)], check=True, capture_output=True)
    vtt = asr / "gr-rich-1.en.vtt"
    vtt.write_text("WEBVTT\n\n00:00:28.000 --> 00:00:29.000\nhello\n\n00:02:31.000 --> 00:02:32.000\nworld\n", encoding="utf-8")
    for vid, sp, st, meta in (("gr-rich-1", "rich", "downloaded", '{"m3u8": null, "mp4": null}'),      # wav on disk
                              ("gr-rich-2", "rich", "downloaded", '{"m3u8": null, "mp4": null}'),      # nothing on disk
                              ("yt-other-1", "other", "downloading", "{}")):                            # another run, mid-fetch
        con.execute("INSERT INTO videos (video_id, speaker, channel_url, month, title, upload_date, crawled_at, in_frame, status, updated_at, kind, "
                    "platform, source_url, meeting_id, meeting_date, duration_s, raw_meta, attempts, captions, captions_path) VALUES "
                    "(?, ?, 'u', '2026-02', ?, '20260217', ?, 0, ?, ?, 'meeting', ?, 'https://x/', ?, '2026-02-17', 300, ?, 0, 'asr_local', ?)",
                    (vid, sp, f"{sp} council", now, st, now, "youtube" if vid.startswith("yt-") else "granicus", f"{sp}:2026-02-17",
                     meta, str(vtt) if vid == "gr-rich-1" else None))
    for eid, vid, a, b in (("gr-rich-1_e1", "gr-rich-1", 60, 120), ("gr-rich-1_e2", "gr-rich-1", 250, 295), ("gr-rich-2_e1", "gr-rich-2", 60, 120)):
        con.execute("INSERT INTO episodes (episode_id, meeting_id, video_id, start_s, end_s, core_start_s, core_end_s, tags, method, label, status, updated_at) "
                    "VALUES (?, 'rich:2026-02-17', ?, ?, ?, ?, ?, 'transit', 'local_asr', 'about_transit', 'located', ?)", (eid, vid, a, b, a, b, now))
    os.environ.update({"LOCAL_ASR_DIR": str(asr), "AUDIO_DIR": str(d / "audio"), "OPUS_DIR": str(d / "opus")})
    try:
        check(episodes.local_wav_path(dict(con.execute("SELECT * FROM videos WHERE video_id = 'gr-rich-1'").fetchone())) == wav
              and episodes.local_wav_path(dict(con.execute("SELECT * FROM videos WHERE video_id = 'gr-rich-2'").fetchone())) is None
              and episodes.local_wav_path({"video_id": "gr-rich-1", "platform": "granicus", "raw_meta": '{"m3u8": "https://h/p.m3u8"}'}) is None,
              "local_wav_path: only a Granicus meeting with NO m3u8 and a wav in LOCAL_ASR_DIR takes the fallback; an m3u8 keeps the HLS route")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc_plan = episodes.main(["--state", str(db), "--bodies", "rich", "--plan"])
        plan = buf.getvalue()
        l1 = next((l for l in plan.splitlines() if "gr-rich-1 (" in l), "")
        l2 = next((l for l in plan.splitlines() if "gr-rich-2 (" in l), "")
        check(rc_plan == 0 and "audio via local wav (" in l1 and "gr-rich-1.wav)" in l1 and "audio via granicus" in l2
              and con.execute("SELECT COUNT(*) FROM videos WHERE kind = 'episode'").fetchone()[0] == 0,
              "episodes.py --plan: 'audio via local wav (<dir>/gr-rich-1.wav)' for the fallback meeting, 'audio via granicus' for the one with nothing; no cut",
              f"rc={rc_plan} {l1!r} {l2!r}")
        if have_ffmpeg:
            rc = episodes.main(["--state", str(db), "--bodies", "rich"])
            eps = {r["episode_id"]: dict(r) for r in con.execute("SELECT * FROM episodes ORDER BY episode_id")}
            clips = {r["video_id"]: dict(r) for r in con.execute("SELECT * FROM videos WHERE kind = 'episode'")}
            e1, e2 = eps["gr-rich-1_e1"], eps["gr-rich-1_e2"]
            durs = {k: (episodes.ffprobe_duration(e["audio_path"]) if e["audio_path"] else None) for k, e in eps.items()}
            check(rc == 0 and (e1["start_s"], e1["end_s"]) == (29.0, 151.0) and (e2["start_s"], e2["end_s"]) == (220.0, 300.0)
                  and e1["status"] == e2["status"] == "cut" and e1["method"] == e2["method"] == "local_asr"
                  and durs["gr-rich-1_e1"] and abs(durs["gr-rich-1_e1"] - 122.0) < 0.1 and abs(durs["gr-rich-1_e2"] - 80.0) < 0.1
                  and Path(e1["audio_path"]).suffix == ".wav",
                  "cut_local_wav: core +-30 s, snapped to the ASR cue edges (29.0, 151.0), clamped to the wav (220.0, 300.0); "
                  "sample-exact .wav clips; episodes 'cut' with method STILL 'local_asr'",
                  f"rc={rc} e1={(e1['start_s'], e1['end_s'], e1['status'], e1['method'])} e2={(e2['start_s'], e2['end_s'])} durs={durs}")
            c1 = clips.get("gr-rich-1_e1") or {}
            meta1 = json.loads(c1.get("raw_meta") or "{}")
            check(set(clips) == {"gr-rich-1_e1", "gr-rich-1_e2"} and all(c["status"] == "transcoded" and c["in_frame"] == 1 for c in clips.values())
                  and transcode.verify(c1.get("opus_path") or "", 122.0)[0] and meta1.get("cut_source") == "local_wav"
                  and (meta1.get("window_start_s"), meta1.get("window_end_s")) == (29.0, 151.0) and meta1.get("method") == "local_asr"
                  and con.execute("SELECT status FROM videos WHERE video_id = 'gr-rich-1'").fetchone()[0] == "cut",
                  "register: one in-frame 'transcoded' kind=episode row per clip (16 kHz mono opus verified), raw_meta.cut_source = 'local_wav', "
                  "window bounds recorded, the meeting advances 'downloaded' -> 'cut'",
                  f"clips={{k: (v['status'], v['in_frame']) for k, v in clips.items()}} meta={ {k: meta1.get(k) for k in ('cut_source', 'window_start_s', 'method')} }")
            st = {r[0]: r[1] for r in con.execute("SELECT video_id, status FROM videos WHERE kind = 'meeting'")}
            check(eps["gr-rich-2_e1"]["status"] == "located" and st["gr-rich-2"] == "downloaded" and st["yt-other-1"] == "downloading",
                  "no wav and no m3u8 is still the old no-op (episode 'located', meeting 'downloaded'); the OTHER body's live 'downloading' "
                  "row is not reclaimed by a --bodies run", str(st))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc2 = episodes.main(["--state", str(db), "--bodies", "rich"])
            n_ev = con.execute("SELECT COUNT(*) FROM events WHERE video_id LIKE 'gr-rich-1_e%'").fetchone()[0]
            check(rc2 == 0 and "gr-rich-1 (" not in buf.getvalue() and con.execute("SELECT COUNT(*) FROM videos WHERE kind = 'episode'").fetchone()[0] == 2,
                  "re-run: the cut meeting is not selected again, nothing re-cut", f"rc={rc2} events={n_ev}")
    finally:
        for k in ("LOCAL_ASR_DIR", "AUDIO_DIR", "OPUS_DIR"):
            os.environ.pop(k, None)
    con.close()


def section_redact(tmp):
    print("\n36. pipeline/redact.py: the name-redaction pass (DECISIONS #15, SHOWCASE.md 3)")
    sys.path.insert(0, str(ROOT))
    from pipeline import redact as R

    def only(text, rule=None):
        red, reps = R.redact_text(text)
        return red, [(r["rule"], r["original"]) for r in reps if rule is None or r["rule"] == rule]

    # (a) each cue rule, on its own
    red, reps = only("Good evening, my name is Ann Lee and I live in Alviso.")
    check(red == "Good evening, my name is [name] and I live in Alviso." and reps == [("cue_name_is", "Ann Lee")],
          "cue 'my name is X' -> [name], place after it kept", red)
    red, reps = only("Hi, I'm Jose Hernandez with the Sierra Club and I support BART to San Jose.")
    check(red.startswith("Hi, I'm [name] with the Sierra Club") and "BART to San Jose" in red and reps[0][0] == "cue_im",
          "cue \"I'm X with\" -> [name]; BART and San Jose kept", red)
    red, reps = only("This is Priya Natarajan speaking on behalf of Santa Clara County.")
    check(red == "This is [name] speaking on behalf of Santa Clara County." and reps == [("cue_this_is", "Priya Natarajan")],
          "cue 'this is X speaking' -> [name]", red)
    red, reps = only("John O'Brien, longtime resident of Sunnyvale, opposes item 5.")
    check(red.startswith("[name], longtime resident of Sunnyvale") and reps == [("cue_resident", "John O'Brien")],
          "cue 'X, resident of' -> [name] (apostrophe name)", red)
    red, reps = only("Thank you, Councilmember Smith. Mayor Mahan and Vice Mayor Kamei, Director Santa Clara.")
    check(red == "Thank you, Councilmember [name]. Mayor [name] and Vice Mayor [name], Director Santa Clara."
          and [r for r, _ in reps] == ["cue_honorific"] * 3,
          "honorific + Capitalised -> [name]; the title stays; 'Director Santa Clara' is not a person", red)
    red, reps = only("For the record, Maria de la Cruz, District 3.")
    check(red == "For the record, [name], District 3." and reps == [("cue_record", "Maria de la Cruz")],
          "cue 'for the record, X' -> [name], particles kept inside the name", red)
    red, reps = only("I am Dana Park and I live in Richmond.")
    check(red == "I am [name] and I live in Richmond." and reps == [("cue_im", "Dana Park")],
          "self-introduction keeps a surname that is also a stop word (Park); Richmond kept", red)

    # (b) spaCy PERSON layer (skipped loudly when the model is not installed)
    if R.nlp() is None:
        print("  [SKIP] spaCy en_core_web_sm not installed; PERSON layer untested")
    else:
        red, reps = only("Then Sarah Kim and Robert Garcia-Lopez spoke against the Diridon plan.")
        check("Sarah" not in red and "Garcia" not in red and "Diridon" in red
              and all(r == "person" for r, _ in reps) and len(reps) == 2,
              "spaCy PERSON entities -> [name], Diridon kept", red)
        red, reps = only("Service from Balboa Park to the East Bay; Coleman Avenue is closed near Lake Merritt.")
        check(reps == [], "allowlisted places survive the PERSON layer (Balboa Park, Coleman Avenue, Lake Merritt)", red)

    # (c) address, (d) contact; percent-of-the-way is not an address
    red, reps = only("I live at 1234 Main Street, Apt 2 and I'm 50 percent of the way there.")
    check(red == "I live at [address] and I'm 50 percent of the way there." and reps == [("address", "1234 Main Street, Apt 2")],
          "street address -> [address]; '50 percent of the way' is not one", red)
    red, reps = only("Call me at (415) 555-1234 or ann.lee@example.com.")
    check(red == "Call me at [contact] or [contact]." and [r for r, _ in reps] == ["contact", "contact"],
          "phone and e-mail -> [contact]", red)

    # allowlist survives, including the usual over-redaction suspects
    red, reps = only("Thank you, Oakland. Thank you. Moving on to item 6. Thank you, Madam Chair. "
                     "This is Item 5, consent calendar. The Chair recognizes BART, Caltrain and Santa Clara.")
    check(reps == [], "places, agencies, 'the Chair', 'Madam Chair', 'Item 5' never redacted", str(reps))

    # idempotency: redacting redacted text changes nothing
    src = ("My name is Lawrence Campbell, I'm a resident of Campbell at 12 Oak Lane. "
           "Thank you, Councilmember Smith; Sarah Kim is next.")
    once, reps1 = R.redact_text(src)
    twice, reps2 = R.redact_text(once)
    check(once == twice and reps2 == [] and len(reps1) >= 3, "idempotent: second pass finds nothing", once)

    # a transcript: turns, counts by rule, total words, span across a segment boundary
    doc = {"video_id": "sim_e1", "episode_id": "sim_e1", "kind": "episode",
           "roles": {"SPEAKER_00": {"role": "dais"}, "SPEAKER_01": {"role": "public"}},
           "segments": [
               {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "text": "Our next speaker is Tom"},
               {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_00", "text": "Wu, followed by public comment."},
               {"start": 4.0, "end": 8.0, "speaker": "SPEAKER_01", "text": "Good evening, my name is Ann Lee and I live at 1234 Main Street."},
               {"start": 8.0, "end": 9.0, "speaker": "SPEAKER_01", "text": "Thank you, Mayor Mahan."}]}
    new, audit = R.redact_transcript(doc)
    r = new["redaction"]
    check([s["text"] for s in new["segments"]] ==
          ["Our next speaker is [name]", ", followed by public comment.",
           "Good evening, my name is [name] and I live at [address].", "Thank you, Mayor [name]."],
          "transcript: same segments, text redacted, span crossing a segment boundary handled", str(new["segments"]))
    check(r["by_rule"] == {"address": 1, "cue_honorific": 1, "cue_name_is": 1, "cue_speaker": 1} and r["total"] == 4
          and r["n_turns"] == 2 and r["words"] == R.count_words(" ".join(s["text"] for s in doc["segments"])),
          "transcript: counts by rule, total, turns and words", str(r))
    check([a["turn_idx"] for a in audit] == [0, 1, 1, 1] and audit[0]["role"] == "dais" and audit[1]["role"] == "public"
          and audit[1]["before"].endswith("my name is") and audit[1]["after"].startswith("and I live at"),
          "audit rows: turn index, role, 6-word context", str(audit[:2]))
    for k in ("video_id", "episode_id", "kind", "roles"):
        if new.get(k) != doc.get(k):
            check(False, "transcript: every other key is preserved", k)
            break
    else:
        check(True, "transcript: every other key is preserved")
    # DECISIONS #15: no name must survive in the written file
    check(not any(n in json.dumps(new) for n in ("Tom Wu", "Ann Lee", "Mahan")), "no original name survives in the output JSON")

    # CLI end to end on a temp tree: run --episode, audit, review sample, residual scan
    src_dir, out_dir = tmp / "wh", tmp / "red"
    src_dir.mkdir()
    (src_dir / "sim_e1.json").write_text(json.dumps(doc))
    leak = dict(doc, video_id="sim_e2", episode_id="sim_e2",
                segments=[{"start": 0, "end": 1, "speaker": "S", "text": "Hello, my name is Xqzvtl Pmrwq and that is all."}])
    (src_dir / "sim_e2.json").write_text(json.dumps(leak))
    qa = tmp / "qa.csv"
    qa.write_text("status,file,kind\nOK,sim_e1.json,episode\nWARN,sim_e2.json,episode\nFAIL,sim_e3.json,episode\nOK,show.json,show\n")
    p = subprocess.run([PY, str(ROOT / "pipeline" / "redact.py"), "run", "--all", "--src", str(src_dir), "--qa", str(qa),
                        "--out-dir", str(out_dir), "--audit", str(tmp / "audit.csv"),
                        "--sample-for-review", "3", "--out", str(tmp / "review.csv"), "--residual-scan", "--quiet"],
                       capture_output=True, text=True)
    check(p.returncode == 0 and (out_dir / "sim_e1.json").exists() and (out_dir / "sim_e2.json").exists()
          and not (out_dir / "sim_e3.json").exists() and not (out_dir / "show.json").exists(),
          "CLI run --all: only OK/WARN kind=episode transcripts are redacted", p.stdout + p.stderr)
    rows = list(csv.DictReader(open(tmp / "audit.csv")))
    rev = list(csv.DictReader(open(tmp / "review.csv")))
    check(len(rows) == 5 and {r["episode_id"] for r in rows} == {"sim_e1", "sim_e2"} and len(rev) == 3 and "verdict" in rev[0],
          "audit CSV has one row per replacement; review sample has N rows with a verdict column", f"{len(rows)} rows, {len(rev)} sample")
    check("residual scan: 0 probable misses" in p.stdout, "residual scan is clean after redaction", p.stdout)
    # a planted miss: write an unredacted cue straight into the output dir and scan again
    (out_dir / "sim_e9.json").write_text(json.dumps(dict(leak, episode_id="sim_e9")))
    p2 = subprocess.run([PY, str(ROOT / "pipeline" / "redact.py"), "run", "--out-dir", str(out_dir), "--residual-scan"],
                        capture_output=True, text=True)
    check("1 probable misses" in p2.stdout and "sim_e9" in p2.stdout and "[name_is]" in p2.stdout,
          "residual scan reports a cue still followed by a Capitalised token", p2.stdout)

    # (given_name) a bare first name with no cue and no PERSON hit (CALIBRATION.md 2026-10-17 #2)
    red, reps = only("Gaurav leaves and everything falls apart")
    check(red == "[name] leaves and everything falls apart" and reps == [("given_name", "Gaurav")],
          "given_name: a lexicon-listed first name with no cue is still redacted", red)
    red, reps = only("Grant funding was approved by the council.")
    check(red == "Grant funding was approved by the council." and reps == [],
          "given_name: an ambiguous entry (Grant) alone, no cue, no adjacent name, is kept", red)
    red, reps = only("Will Priya be at the meeting?")
    check(red == "[name] be at the meeting?" and reps == [("given_name", "Will Priya")],
          "given_name: an ambiguous entry (Will) next to another capitalised token is redacted as a run", red)
    red, reps = only("Hi, I'm Jose Hernandez with the Sierra Club and I support BART to San Jose.")
    check(red.startswith("Hi, I'm [name] with the Sierra Club") and "BART to San Jose" in red and reps == [("cue_im", "Jose Hernandez")],
          "given_name does not swallow the capitalised contraction \"I'm\" ahead of a cue-caught name", red)
    check(R._load_given_names()[("grant")] == "ambiguous" and R._load_given_names()["gaurav"] == "name",
          "plan/given_names.txt flags Grant ambiguous and Gaurav a bare name", "")

    # (ci) the lowercase routes. Every layer above keys on a leading capital, so
    # local_asr / mlx-whisper and the YouTube auto-caption arm published names in
    # clear: 57 self-introductions over 30 people in the 2026-10-20 export.
    # These checks are that regression.
    check(R.VERSION == "redact-1.3", "redaction version bumped for the publish-tier and honorific layers", R.VERSION)

    # (publish tier) The 2026-10-20 publish scan found names that the cue-anchored
    # scan reports as clean, because the given_name layer runs at FULL tier, where
    # a lone token that is also a Bay Area place keeps the place's protection.
    # docs/ is published, so the two site exporters pass given_tier=CORE.
    def pub(text):
        red, _ = R.redact_text(text, publish=True)
        return red

    check("[name]" in pub("And then I'm welcoming Carlos and [name] from WILDAN to speak."),
          "publish tier redacts a bare given name that is also a place (Carlos)", pub("I'm welcoming Carlos and x."))
    for place in ("closest to the neighbor at 25 San Carlos.",
                  "lessons learned from another project in San Carlos."):
        check("San Carlos" in pub(place), "publish tier keeps the multi-word place San Carlos", pub(place))
    check("Carlos" not in pub("I'm welcoming Carlos to speak.")
          and "[name]" in pub("Hi, this is Maena Young. I'm the president.")
          and "[name]" in pub("Hi, good evening. This is Delma. I lived in District 5.")
          and "ramses" not in pub("okay hi i'm ramses Long time tenant, first time caller."),
          "publish tier catches the four names that survived the 2026-10-20 export", "")

    # (honorific headings) An agenda item names the official who authored it, and
    # scrub_heading() used to run no person layer at all: 57 of them reached
    # docs/data/episodes.json. spans_honorific() is the one cue cheap enough to
    # run on Title Case, and it must not eat the heading's own structure.
    def head(text):
        spans = sorted(R._spans_address(text) + R._spans_contact(text) + R.spans_honorific(text))
        taken, keep = [], []
        for s, e, orig, rep, rule in spans:
            if any(s < te and e > ts for ts, te in taken):
                continue
            taken.append((s, e))
            keep.append((s, e, orig, rep, rule))
        return R.apply_spans(text, keep)

    check(head("Receive the Annual Report (Report from Development Services Director Buizer)")
          == "Receive the Annual Report (Report from Development Services Director [name])",
          "heading honorific: 'Director Buizer' -> [name]", head("Report from Director Buizer)"))
    check(head("2025 State Legislative Update by State Senator Dr. Aisha Wahab")
          == "2025 State Legislative Update by State Senator Dr. [name]",
          "heading honorific: stacked titles still reach the name", head("by State Senator Dr. Aisha Wahab"))
    check(head("Repeal Policy From: Councilmember Houston Recommendation: Adopt A Resolution")
          == "Repeal Policy From: Councilmember [name] Recommendation: Adopt A Resolution",
          "heading honorific: stops at the heading's own label, not one token past it",
          head("From: Councilmember Houston Recommendation: Adopt"))
    check(head("Adopt the 2025-2033 Housing Element and Certify the Environmental Impact Report")
          == "Adopt the 2025-2033 Housing Element and Certify the Environmental Impact Report"
          and "David Baker Architects" in head("Agreement with David Baker Architects for design"),
          "heading honorific: no cue, no redaction -- policy terms and firm names survive", "")
    check(head("From: Councilmember Ramachandran And Council President Jenkins Recommendation: Adopt")
          == "From: Councilmember [name] And Council President [name] Recommendation: Adopt",
          "heading honorific: two officials in a row, the second cue still reached",
          head("From: Councilmember Ramachandran And Council President Jenkins Recommendation: Adopt"))

    # (ci_honorific) cue_honorific keys on CAP, so on the lowercase routes it read
    # nothing and a published quote carried "good evening mayor gonzalez". A
    # surname is not in plan/given_names.txt, so the cue is the only anchor.
    check(pub("good evening mayor gonzalez council members and welcome everyone")
          == "good evening mayor [name] council members and welcome everyone",
          "ci_honorific: lowercase 'mayor gonzalez' -> [name], stopping before 'council'",
          pub("good evening mayor gonzalez council members and welcome everyone"))
    check(pub("thank you mr chairman and members of the council")
          == "thank you mr chairman and members of the council"
          and pub("the director of public works will present the item")
          == "the director of public works will present the item",
          "ci_honorific: a stacked title or a role phrase is not a name", "")
    check(not [o for _r, o in only("Thank you, Madam Chair. The Chair recognizes BART and Santa Clara.")[1]],
          "ci_honorific stays off cased text, where cue_honorific is competent", "")

    # (span width) spaCy tags only the surname of a two-token name often enough
    # that the wider run has to win, or the published text keeps the first name.
    check(pub("Our next speaker is Dana Ortiz, followed by John Smith.")
          == "Our next speaker is [name], followed by [name].",
          "a wider name run supersedes the narrower span spaCy took",
          pub("Our next speaker is Dana Ortiz, followed by John Smith."))
    red, reps = only("Gaurav leaves and everything falls apart")
    check(red == "[name] leaves and everything falls apart",
          "widening is limited to spaCy's spans: ci_given_pair must not eat 'leaves'", red)
    for lower, upper in [("good evening, my name is dana ortiz and i live in fremont.",
                          "Good evening, my name is Dana Ortiz and I live in Fremont."),
                         ("hi, i'm jose hernandez with the sierra club.",
                          "Hi, I'm Jose Hernandez with the Sierra Club."),
                         ("this is priya natarajan speaking from palo alto.",
                          "This is Priya Natarajan speaking from Palo Alto."),
                         ("for the record, maria de la cruz, district 3.",
                          "For the record, Maria de la Cruz, District 3.")]:
        lo, lo_reps = only(lower)
        up, up_reps = only(upper)
        check("[name]" in lo and "[name]" in up and len(lo_reps) == 1 and len(up_reps) == 1
              and lo.lower() == up.lower(),
              f"lowercase and capitalised self-introduction redact alike: {lower[:34]!r}",
              f"lower={lo!r} upper={up!r}")
    # the same cue words in ordinary speech: a name-like run must stop at stop
    # words, function words, verbs and numbers, or lowercase text has no guard
    for text in ["i'm here to talk about the housing element.",
                 "i'm sorry, i am done speaking.",
                 "this is a continuation of item 5.",
                 "my name is going to be on the list.",
                 "it's important to note the parking requirement.",
                 "call me at the office tomorrow about the 2011 ordinance.",
                 "for the record, the vote was five to zero."]:
        red, reps = only(text)
        check(red == text and reps == [], f"not a self-introduction, nothing redacted: {text[:38]!r}", str(reps))
    # a gazetteer entry that is also an ordinary English word never seeds the
    # case-insensitive pair layer; a strong cue still overrides that
    for text in ["will the council please vote on this item.", "grant funding was approved.",
                 "the may meeting ran long.", "art and recreation funding is in the budget."]:
        red, reps = only(text)
        check(red == text and reps == [], f"gazetteer stopword stays without a cue: {text[:34]!r}", str(reps))
    red, reps = only("my name is will chen and i am a renter.")
    check(red == "my name is [name] and i am a renter." and reps == [("ci_name_is", "will chen")],
          "a strong cue overrides the gazetteer stoplist (lowercase 'will chen')", red)
    # After a WEAK cue the anchor has to be the lexicon or a self-introduction word
    # after a name-shaped run. A trailing comma is not an anchor: it was, and the
    # 2026-10-20 export read "it's [name], not surprising" over "it's raining" on
    # the landing view. Every line here is ordinary speech that a comma, an "and"
    # or a bare context word used to turn into a person.
    for text in ["it's raining, not surprising.", "it's quite cold, honestly.",
                 "it's pouring rain here in richmond.", "i'm frustrated, and so is my neighbour.",
                 "it's confusing, and the parking ratio is wrong.",
                 "we are in san jose tonight.", "this is san jose's housing element."]:
        red, reps = only(text)
        check(red == text and reps == [], f"weak cue needs a real anchor, not a comma: {text[:38]!r}", str(reps))
    # ...and the two shapes that must still be caught: an ambiguous lexicon seed
    # (flagged ambiguous because it is also a place) inside a 2-token run, and a
    # run the lexicon does not know at all followed by a self-introduction word.
    red, reps = only("i'm jose hernandez with the sierra club.")
    check(red == "i'm [name] with the sierra club." and reps == [("ci_intro", "jose hernandez")],
          "weak cue: an ambiguous seed still redacts as part of a 2-token run", red)
    red, reps = only("i'm aaron blackwood speaking on behalf of the chamber.")
    check(red == "i'm [name] speaking on behalf of the chamber." and reps == [("ci_intro", "aaron blackwood")],
          "weak cue: a name the lexicon does not know, anchored by 'speaking'", red)
    red, reps = only("dana ortiz spoke against the rezoning in san leandro.")
    check(red == "[name] spoke against the rezoning in san leandro." and reps == [("ci_given_pair", "dana ortiz")],
          "ci_given_pair: a lowercase given name plus a surname, no cue; the place stays", red)
    red, reps = only("the reed street project and the hunter storm company are listed.")
    check(reps == [], "ci_given_pair keeps the street / company context vetoes", red)
    # idempotency and the allowlist, on lowercase
    src = "my name is ann lee, i'm a resident of campbell at 12 oak lane. my name is dana ortiz."
    once, reps1 = R.redact_text(src)
    twice, reps2 = R.redact_text(once)
    check(once == twice and reps2 == [] and len(reps1) >= 3 and "ann" not in once and "ortiz" not in once,
          "lowercase: idempotent, and no name survives", once)
    red, _ = only("i'm from oakland and i take bart to san jose every day.")
    check(red == "i'm from oakland and i take bart to san jose every day.",
          "lowercase places and agencies survive the case-insensitive layers", red)

    # the residual scan must be able to SEE a lowercase miss. docs/REPORT.md's
    # old scan looked for a cue + a CAPITALISED token, so on the only route that
    # leaks it returned 0 by construction: a check that
    # cannot fail is what let 57 spans through.
    check(len(R.ci_cue_leaks("hello, my name is dana ortiz and that is all.")) == 1
          and R.ci_cue_leaks("hello, my name is [name] and that is all.") == []
          and R.ci_cue_leaks("i'm here to speak on item 5.") == [],
          "ci_cue_leaks sees a lowercase miss, ignores a placeholder and ordinary speech",
          str(R.ci_cue_leaks("hello, my name is dana ortiz and that is all.")))
    lower_leak = dict(doc, video_id="sim_e8", episode_id="sim_e8",
                      segments=[{"start": 0, "end": 1, "speaker": "S",
                                 "text": "hello, my name is xqzvtl pmrwq and that is all."}])
    (out_dir / "sim_e8.json").write_text(json.dumps(lower_leak))
    p3 = subprocess.run([PY, str(ROOT / "pipeline" / "redact.py"), "run", "--out-dir", str(out_dir),
                         "--residual-scan"], capture_output=True, text=True)
    check("2 probable misses" in p3.stdout and "sim_e8" in p3.stdout and "[ci_name_is]" in p3.stdout,
          "residual scan reports a LOWERCASE cue miss, and still counts the capitalised one once", p3.stdout)

    # scripts/redaction_scan.py: the same check over published files, counts and
    # file:line only. It must never echo the leak it found.
    leaky = tmp / "pub.json"
    leaky.write_text(json.dumps({"q": ["hello, my name is dana ortiz.", "i live at 1234 main street.",
                                       "reach me at dana@example.com", "my name is [name], a renter."]}))
    p4 = subprocess.run([PY, str(ROOT / "scripts" / "redaction_scan.py"), str(leaky), "--list", "9"],
                        capture_output=True, text=True)
    check(p4.returncode == 1 and "cue leaks (must be 0): 1" in p4.stdout
          and "ci_name_is   1" in p4.stdout and "address      1" in p4.stdout and "contact      1" in p4.stdout,
          "redaction_scan.py counts cue, address and contact leaks and exits 1", p4.stdout)
    check(not any(w in p4.stdout for w in ("dana", "ortiz", "Main", "main street", "example.com")),
          "redaction_scan.py never prints the matched text", p4.stdout)
    clean = tmp / "clean.json"
    clean.write_text(json.dumps({"q": ["hello, my name is [name].", "i'm here to speak on item 5."]}))
    p5 = subprocess.run([PY, str(ROOT / "scripts" / "redaction_scan.py"), str(clean)],
                        capture_output=True, text=True)
    check(p5.returncode == 0 and "cue leaks (must be 0): 0" in p5.stdout,
          "redaction_scan.py exits 0 on redacted text", p5.stdout)


def section_models(tmp):
    print("\n32. analysis/models: the two showcase Bayesian models (plan/SHOWCASE.md 1)")
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(ROOT))
    from analysis.models import load as L, topic_time as TT, role_gap as RG
    rng = np.random.default_rng(20261017)
    FIT = dict(draws=200, tune=200, chains=2, progress=False)

    # -- 1. topic_time facet guard: a cell table spanning routes is refused ---
    cells_two_routes = pd.DataFrame({
        "body": ["a", "a"], "month": ["2026-01", "2026-01"], "family": ["density", "density"],
        "route": ["classifier", "agenda_index"], "exposure_h": [5.0, 5.0], "y": [10, 10], "minutes": [10.0, 10.0],
        "n_meetings": [1, 1], "n_episodes": [2, 2], "pooled": [False, False]})
    try:
        L.pool_routes(cells_two_routes, pool=False)
        check(False, "load.pool_routes refuses an unpooled multi-route table")
    except L.FacetError:
        check(True, "load.pool_routes refuses an unpooled multi-route table")
    pooled = L.pool_routes(cells_two_routes, pool=True)
    check(bool(pooled["pooled"].all()) and len(pooled) == 1,
          "load.pool_routes(pool=True) collapses the two routes into one stamped-POOLED row", str(pooled.to_dict("records")))

    # -- 2. role_gap facet guard: turns spanning two role instruments refused -
    turns_two_instruments = pd.DataFrame({
        "role": ["dais", "public"] * 40, "family": ["density"] * 80, "body": ["a"] * 80,
        "episode_id": [f"e{i % 6}" for i in range(80)], "words": [150] * 80,
        "stance_median": rng.integers(-2, 3, 80), "role_source": ["db"] * 40 + ["computed"] * 40})
    try:
        RG.fit(turns_two_instruments, outcome="stance", **FIT)
        check(False, "role_gap.fit refuses turns spanning two role_source levels")
    except L.FacetError:
        check(True, "role_gap.fit refuses turns spanning two role_source levels")

    # -- 3. topic_time NegBin recovers a planted body-rate difference ---------
    months = [f"2026-{m:02d}" for m in range(1, 5)]
    rows = []
    for body, true_rate in (("hi", 12.0), ("lo", 2.0)):
        for m in months:
            e = 5.0
            mu = true_rate * e
            y = rng.poisson(mu)
            rows.append({"body": body, "month": m, "family": "density", "route": "classifier",
                        "exposure_h": e, "y": int(y), "minutes": float(y), "pooled": False})
    cells = pd.DataFrame(rows)
    fit_tt = TT.fit(cells, body_effects=True, month_walk=False, **FIT)
    bt = TT.body_table(fit_tt)
    hi_rate = bt.loc[bt["body"] == "hi", "post_median"].iloc[0]
    lo_rate = bt.loc[bt["body"] == "lo", "post_median"].iloc[0]
    check(hi_rate > lo_rate, "topic_time.fit recovers the planted body-rate ordering (hi > lo)",
          f"hi={hi_rate:.2f} lo={lo_rate:.2f}")

    # -- 4. role_gap ordinal path runs end to end on a synthetic turn_stance --
    db = tmp / "models_turn_stance.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE turn_stance (episode_id TEXT, cluster TEXT, turn_idx INT, stance_median INT, "
                "hostility_median INT, role TEXT, body TEXT, family TEXT, words INT)")
    n = 160
    roles = rng.choice(["dais", "public"], n)
    # planted effect: public leans one level higher (less opposed) than dais
    base = rng.integers(-2, 1, n)
    stance = np.clip(base + np.where(roles == "public", 1, 0), -2, 2)
    for i in range(n):
        con.execute("INSERT INTO turn_stance VALUES (?,?,?,?,?,?,?,?,?)",
                    (f"ep{i % 10}", f"spk{i % 5}", i, int(stance[i]), int(rng.integers(0, 5)),
                     roles[i], "a", "density", int(rng.integers(100, 300))))
    con.commit()
    turn_stance_df = L.read_turn_stance(con)
    con.close()
    check(turn_stance_df is not None and len(turn_stance_df) == n and (turn_stance_df["instrument"] == "stance pass").all(),
          "load.read_turn_stance reads the real table and stamps instrument='stance pass'", f"n={0 if turn_stance_df is None else len(turn_stance_df)}")
    fit_rg = RG.fit(turn_stance_df, outcome="stance", target_accept=0.95, **FIT)
    gap = RG.gap_table(fit_rg)
    gap_all = gap.loc[gap["family"] == "all (turn-weighted)", "gap_latent_median"].iloc[0]
    # a 200-draw/2-chain smoke fit on one body and one family is only asked to
    # run end to end and get the sign right, not to converge cleanly
    check(gap_all > 0,
          "role_gap.fit on a synthetic turn_stance table runs and recovers the planted public > dais gap direction",
          f"gap={gap_all:.2f} divergences={fit_rg['diagnostics']['divergences']}")

    # -- 5. cap_by_body keeps every body at or near the cap, not over it ------
    eps = [f"big_e{i}" for i in range(20)] + [f"small_e{i}" for i in range(3)]
    turns_cap = pd.DataFrame({"episode_id": rng.choice(eps, 500),
                              "body": ["big"] * 500})
    turns_cap.loc[turns_cap["episode_id"].str.startswith("small"), "body"] = "small"
    ep_hours = {e: 1.0 for e in eps if e.startswith("big")} | {e: 0.5 for e in eps if e.startswith("small")}
    capped, kept_h = L.cap_by_body(turns_cap, ep_hours, cap_h=5.0)
    check(kept_h["big"] <= 5.0 and abs(kept_h["small"] - 1.5) < 0.01,
          "cap_by_body stops drawing episodes once a body's hours would exceed the cap (big: 20 h available, capped "
          "at 5.0); a body with less than the cap keeps everything (small: 1.5 h available, all kept)",
          f"kept_h={kept_h}")

    # -- 6. bin_standin labels the stand-in and drops short turns -------------
    standin_raw = pd.DataFrame({"words": [10, 500, 150, 99], "compound": [0.1, -0.5, 0.9, 0.0],
                                "hostility_hits": [0, 2, 0, 1]})
    binned = L.bin_standin(standin_raw, min_words=L.MIN_WORDS)
    check(len(binned) == 2 and (binned["instrument"] == L.STANDIN).all() and set(binned["stance_median"]).issubset(set(L.LEVELS)),
          "bin_standin drops turns under MIN_WORDS and stamps every kept row 'STAND-IN: VADER, not stance'",
          f"kept={len(binned)}/{len(standin_raw)}")


def datetime_now():
    from datetime import datetime, timezone
    return datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)

def section_stance(tmp):
    print("\n32. Stance pass (2026-10-17, plan/SHOWCASE.md 4a): the sampler, the codebook as instrument, the schema, "
          "the row-before-the-call ledger and both caps, the k-vote aggregate, the blind coding sheet and the agreement statistics")
    sys.path.insert(0, str(ROOT / "pipeline"))
    import io, contextlib, types
    import state, spend, stance  # noqa: E402

    # --- the sampler: a synthetic turn set with uneven cells
    rng = random.Random(1)
    turns = []
    for body, n_dais, n_pub in (("alpha", 400, 30), ("beta", 60, 2), ("gamma", 9, 0)):
        for role, n in (("dais", n_dais), ("public", n_pub), ("unknown", 5)):
            for i in range(n):
                t = {"episode_id": f"{body}_e1", "cluster": f"SPEAKER_{i % 7:02d}", "turn_idx": len(turns) + 1, "start_s": 0, "end_s": 1,
                     "words": rng.randint(5, 80), "role": role, "body": body, "family": "density", "agenda_item": "",
                     "prev_text": "p", "text": f"turn{len(turns)} " * 6, "next_text": "n"}
                t["turn_id"] = stance.turn_id(t)
                turns.append(t)
    drawn, alloc = stance.draw(turns, 100, ("body", "role"), floor=4, seed=stance.SEED)
    cells = {k: sum(1 for d in drawn if d["cell"] == "|".join(k)) for k in alloc}
    check(len(drawn) == 100 and cells[("beta", "public")] == 2 and cells[("gamma", "dais")] in (4, 5) and cells[("alpha", "public")] >= 4
          and cells[("alpha", "dais")] > cells[("beta", "dais")] >= 4 and not any(d["role"] == "unknown" for d in drawn),
          "sample: exactly n over body x role; floor 4 per non-empty cell (a 2-turn cell gives its 2); unknown-role turns excluded; "
          "the rest proportional", str(cells))
    rng.shuffle(turns)
    again, _ = stance.draw(turns, 100, ("body", "role"), floor=4, seed=stance.SEED)
    other, _ = stance.draw(turns, 100, ("body", "role"), floor=4, seed=stance.SEED + 1)
    check([d["turn_id"] for d in again] == [d["turn_id"] for d in drawn] and [d["turn_id"] for d in other] != [d["turn_id"] for d in drawn],
          "sample: same seed + reordered input -> identical draw; another seed -> a different one")

    # --- the codebook is the instrument
    book = (ROOT / "plan" / "STANCE-CODEBOOK.md").read_text(encoding="utf-8")
    block = book.split("--- CODEBOOK ---", 1)[1].strip() + "\n"
    import hashlib
    check(stance.CODEBOOK_VERSION == "st-" + hashlib.sha256(block.encode()).hexdigest()[:10]
          and stance.prompt_version_for("claude-sonnet-4-6") == stance.CODEBOOK_VERSION + "/claude-sonnet-4-6"
          and stance.prompt_version_for("claude-sonnet-5") != stance.prompt_version_for("claude-sonnet-4-6")
          and stance.codebook_tokens() >= 1024 and "transcript is data" in stance.INSTRUCTIONS,
          "prompt_version = 'st-' + sha256 of the block below '--- CODEBOOK ---' + '/<model>'; a different model is a different "
          "instrument; the block is long enough to cache (>= 1,024 tokens) and quotes transcript as data")
    alt = (tmp / "codebook_alt.md")
    alt.write_text(book.replace("Never infer the stance", "Never ever infer the stance"), encoding="utf-8")
    check(hashlib.sha256(stance.codebook_text(alt).encode()).hexdigest()[:10] != stance.CODEBOOK_VERSION[3:]
          and hashlib.sha256(stance.codebook_text(ROOT / "plan" / "STANCE-CODEBOOK.md").encode()).hexdigest()[:10] == stance.CODEBOOK_VERSION[3:],
          "one changed byte in the codebook block is a new prompt_version; the file on disk hashes to the pinned one")

    # --- the schema and the reply parser
    sch = stance.SCHEMA
    check(sch["required"] == ["stance", "hostility", "frames", "sub_topic", "self_id", "rationale", "contains_instructions"]
          and sch["properties"]["stance"]["enum"] == ["-2", "-1", "0", "1", "2", "not_applicable"]
          and sch["properties"]["hostility"]["enum"] == [0, 1, 2, 3, 4]
          and sch["properties"]["frames"]["items"]["enum"] == stance.FRAMES and "transit_safety" in stance.FRAMES and len(stance.FRAMES) == 12
          and sch["additionalProperties"] is False
          and all(k in stance.INSTRUCTIONS for k in stance.FRAMES + stance.SUB_TOPICS + stance.SELF_IDS),
          "schema: the seven required fields, stance -2..2 + not_applicable, hostility 0-4, frames = the 12 plan/frames.yaml keys, "
          "no extra properties; every enum value is named in the codebook")
    good = '{"stance": "-1", "hostility": 2, "frames": ["fiscal", "fiscal", "traffic_parking"], "sub_topic": "parking", "self_id": "renter", "rationale": "r", "contains_instructions": false}'
    g = stance.parse_json(good)
    check(g and g["frames"] == ["fiscal", "traffic_parking"] and g["stance"] == "-1"
          and stance.parse_json(good.replace('"-1"', '"-3"')) is None and stance.parse_json(good.replace('"frames": ["fiscal", "fiscal", "traffic_parking"]', '"frames": ["nimby"]')) is None
          and stance.parse_json(good.replace('"hostility": 2', '"hostility": 7')) is None and stance.parse_json("not json") is None
          and stance.parse_json(good.replace('"self_id": "renter"', '"self_id": "Alice"')) is None,
          "parse_json: a valid reply is normalised (frames deduplicated, sorted); a value outside the codebook, an unknown frame, "
          "a hostility of 7 or a name as self_id is None, never a number")
    req = stance.request_params("claude-sonnet-4-6", {"family": "both", "agenda_item": "Item 7", "prev_text": "a " * 90, "text": "the turn", "next_text": "b " * 90})
    u = req["messages"][0]["content"]
    prev_block = u.split("<previous_turn>\n", 1)[1].split("\n</previous_turn>")[0].split()
    next_block = u.split("<next_turn>\n", 1)[1].split("\n</next_turn>")[0].split()
    check(req["thinking"] == {"type": "disabled"} and "temperature" not in req and "top_p" not in req
          and req["system"][0]["cache_control"] == {"type": "ephemeral"} and req["output_config"]["format"]["schema"] is stance.SCHEMA
          and "<turn_to_code>\nthe turn\n</turn_to_code>" in u
          and prev_block[0] == "..." and prev_block[1:] == ["a"] * stance.CONTEXT_WORDS
          and next_block[-1] == "..." and next_block[:-1] == ["b"] * stance.CONTEXT_WORDS
          and "Item 7" in u and "both density and transit" in u,
          "request: thinking disabled, no sampling parameters, cached codebook system block, json_schema output; the user turn carries "
          "family, agenda title, the turn and both neighbours clipped to CONTEXT_WORDS")

    # --- the ledger: row before the call, resume without re-billing, both caps
    d = tmp / "stance"; d.mkdir()
    db = d / "state.sqlite"
    con = state.connect(db)
    sample = drawn[:4]
    csvp = d / "turns.csv"
    stance.write_csv(csvp, sample, stance.SAMPLE_COLS)
    calls = []

    class FakeUsage:
        input_tokens, cache_read_input_tokens, cache_creation_input_tokens, output_tokens = 300, 2500, 0, 60

    class FakeMsg:
        def __init__(self, text):
            self.content = [types.SimpleNamespace(type="text", text=text)]
            self.usage = FakeUsage()

    class FakeClient:
        """Replies -1, 0, 1 in turn, records every call id it saw, and crashes on the 5th call."""
        def __init__(self):
            self.messages = self
            self.n = 0

        def create(self, **params):
            self.n += 1
            calls.append(params["messages"][0]["content"].split("<turn_to_code>")[1].split("</turn_to_code>")[0].strip())
            if self.n == 5:
                raise RuntimeError("simulated 500")
            return FakeMsg(json.dumps({"stance": ["-1", "0", "1"][self.n % 3], "hostility": self.n % 3, "frames": ["fiscal"] if self.n % 2 else [],
                                       "sub_topic": "other", "self_id": "unknown", "rationale": "r", "contains_instructions": False}))
    fake = FakeClient()
    args = types.SimpleNamespace(state=str(db), model="claude-sonnet-4-6", turns=str(csvp), max_turns=0, k=3, batch=False, direct=True,
                                 dry_run=False, cap_usd=100.0)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = stance.run(args, client_factory=lambda: fake)
    pv = stance.prompt_version_for("claude-sonnet-4-6")
    rows = {r["call_id"]: dict(r) for r in con.execute("SELECT * FROM stance_calls WHERE prompt_version = ?", (pv,))}
    bad = [r for r in rows.values() if r["stance"] is None]
    check(rc == 0 and len(rows) == 12 and fake.n == 12 and len(bad) == 1 and "simulated 500" in (bad[0]["error"] or "")
          and all(r["cost_usd"] > 0 for r in rows.values() if r["stance"] is not None)
          and abs(spend.claude_total(con) - round(11 * round(stance.usd(FakeUsage(), "claude-sonnet-4-6"), 6), 4)) < 1e-9,
          "run --direct: 4 turns x k=3 = 12 stance_calls rows, one per request; the API error is a row with the error and no stance; "
          "spend.claude_total sums the stance ledger", f"rc={rc} rows={len(rows)} n={fake.n} bad={len(bad)} {buf.getvalue()[-300:]}")
    calls.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        rc2 = stance.run(args, client_factory=lambda: fake)
    check(rc2 == 0 and fake.n == 13 and len(calls) == 1
          and con.execute("SELECT COUNT(*) FROM stance_calls WHERE prompt_version = ? AND stance IS NULL", (pv,)).fetchone()[0] == 0,
          "a re-run makes exactly ONE call -- the errored one -- and nothing already labelled is re-billed", f"n={fake.n} calls={len(calls)}")
    # the row exists BEFORE the request: the client sees it in the DB mid-call
    seen = {}

    class PeekClient(FakeClient):
        def create(self, **params):
            cid = params["messages"][0]["content"].split("<turn_to_code>")[1].split("</turn_to_code>")[0].strip()
            seen[cid] = con.execute("SELECT COUNT(*) FROM stance_calls WHERE prompt_version = ? AND error = 'in flight'", (pv,)).fetchone()[0]
            return super().create(**params)
    with contextlib.redirect_stdout(io.StringIO()):
        stance.run(types.SimpleNamespace(**{**vars(args), "k": 4}), client_factory=PeekClient)
    check(len(seen) == 4 and all(v == 1 for v in seen.values()),
          "the stance_calls row is written BEFORE the request goes out (seen 'in flight' from inside the fake client)", str(seen))
    ts = {(r["turn_idx"]): dict(r) for r in con.execute("SELECT * FROM turn_stance WHERE prompt_version = ?", (pv,))}
    check(len(ts) == 4 and all(t["k"] == 4 and t["n_ok"] == 4 for t in ts.values()) and all(t["stance_votes"].count(";") == 3 for t in ts.values()),
          "turn_stance: one row per turn per prompt_version, re-aggregated over all k=4 votes after the top-up", str({k: v["stance_votes"] for k, v in ts.items()}))
    # the caps
    with contextlib.redirect_stdout(io.StringIO()):
        rc3 = stance.run(types.SimpleNamespace(**{**vars(args), "k": 5, "cap_usd": 0.001}), client_factory=lambda: fake)
    n5 = con.execute("SELECT COUNT(*) FROM stance_calls WHERE prompt_version = ? AND k = 5", (pv,)).fetchone()[0]
    os.environ["CLAUDE_CAP_USD"] = "0.01"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc4 = stance.run(types.SimpleNamespace(**{**vars(args), "k": 5, "cap_usd": 100.0}), client_factory=lambda: fake)
        rc5 = stance.run(types.SimpleNamespace(**{**vars(args), "k": 5, "cap_usd": 100.0, "batch": True, "direct": False}), client_factory=lambda: fake)
    del os.environ["CLAUDE_CAP_USD"]
    n5b = con.execute("SELECT COUNT(*) FROM stance_calls WHERE prompt_version = ? AND k = 5", (pv,)).fetchone()[0]
    check(rc3 == 2 and n5 == 0 and rc4 == 2 and rc5 == 2 and n5b == 0 and "CLAUDE_CAP_USD" in buf.getvalue()
          and con.execute("SELECT COUNT(*) FROM stance_batches").fetchone()[0] == 0,
          "cap refusal: --cap-usd stops before the first call that could pass it (no row written); CLAUDE_CAP_USD refuses both the "
          "direct run and the batch submission before anything is sent", f"rc3={rc3} n5={n5} rc4={rc4} rc5={rc5} n5b={n5b}")
    # batch: rows first with a provisional id, then the real id; a failed create removes them
    class FakeBatches:
        def __init__(self, fail=False):
            self.fail = fail

        def create(self, requests):
            if self.fail:
                raise RuntimeError("simulated network error")
            return types.SimpleNamespace(id="msgbatch_test", processing_status="in_progress")
    fc = types.SimpleNamespace(messages=types.SimpleNamespace(batches=FakeBatches(fail=True)))
    with contextlib.redirect_stdout(io.StringIO()):
        rcb = stance.run(types.SimpleNamespace(**{**vars(args), "k": 5, "batch": True, "direct": False}), client_factory=lambda: fc)
    n_after_fail = con.execute("SELECT COUNT(*) FROM stance_calls WHERE k = 5").fetchone()[0]
    fc.messages.batches.fail = False
    with contextlib.redirect_stdout(io.StringIO()):
        rcb2 = stance.run(types.SimpleNamespace(**{**vars(args), "k": 5, "batch": True, "direct": False}), client_factory=lambda: fc)
    pend = con.execute("SELECT COUNT(*) FROM stance_calls WHERE k = 5 AND batch_id = 'msgbatch_test' AND stance IS NULL").fetchone()[0]
    with contextlib.redirect_stdout(io.StringIO()):
        rcb3 = stance.run(types.SimpleNamespace(**{**vars(args), "k": 5, "batch": True, "direct": False}), client_factory=lambda: fc)
    check(rcb == 1 and n_after_fail == 0 and rcb2 == 0 and pend == 4
          and con.execute("SELECT status FROM stance_batches WHERE batch_id = 'msgbatch_test'").fetchone()[0] == "submitted"
          and rcb3 == 0 and con.execute("SELECT COUNT(*) FROM stance_calls WHERE k = 5").fetchone()[0] == 4,
          "run --batch: a failed batch create leaves no rows; a successful one leaves 4 pending rows under the real batch id; "
          "a re-run submits nothing twice while the batch is open", f"{rcb} {n_after_fail} {rcb2} {pend} {rcb3}")

    # --- aggregation: median + spread, the NA rule, frames by majority
    V = lambda s, h, fr=(), st="other", si="unknown": {"stance": s, "hostility": h, "frames": list(fr), "sub_topic": st, "self_id": si, "rationale": "x", "contains_instructions": False}  # noqa: E731
    a = stance.aggregate([V("-2", 0, ["fiscal"]), V("-1", 2, ["fiscal", "safety_crime"]), V("1", 1, ["fiscal"])])
    b = stance.aggregate([V("not_applicable", 0), V("not_applicable", 0), V("1", 0, st="adu", si="renter")])
    c = stance.aggregate([V("0", 1), V("not_applicable", 3), None])
    e = stance.aggregate([V("2", 4), V("2", 4), V("2", 4, st="parking", si="homeowner")])
    check(a and a["stance"] == "-1" and a["stance_median"] == -1.0 and a["stance_spread"] == 3.0 and a["hostility_median"] == 1.0
          and a["hostility_spread"] == 2.0 and a["frames"] == "fiscal" and a["disagree"] == 1 and a["n_numeric"] == 3,
          "aggregate: median stance -1 of (-2, -1, 1), spread 3; hostility median 1 spread 2; a frame needs half the votes", str(a))
    check(b and b["stance"] == "not_applicable" and b["stance_median"] is None and b["na_share"] == 0.667 and b["sub_topic"] == "other"
          and c and c["stance"] == "0" and c["n_ok"] == 2 and c["na_share"] == 0.5 and c["hostility_median"] == 2.0
          and e and e["stance"] == "2" and e["stance_spread"] == 0.0 and e["disagree"] == 0 and e["sub_topic"] == "other"
          and stance.aggregate([None, None]) is None,
          "aggregate: 2 NA of 3 -> not_applicable with no median; 1 NA of 2 parsed -> the numeric vote (half rule); "
          "unanimous -> spread 0; mode breaks ties by first vote; nothing parsed -> None", f"{b}\n{c}\n{e}")

    # --- the coding sheet carries no model column; the agreement statistics
    sheet = stance.coding_sheet(drawn, 150, seed=stance.SEED)
    sheet2 = stance.coding_sheet(list(reversed(drawn)), 150, seed=stance.SEED)
    sp = d / "sheet.csv"
    stance.write_csv(sp, sheet, stance.SHEET_COLS)
    hdr = next(csv.reader(open(sp, encoding="utf-8")))
    check(len(sheet) == 100 and [r["turn_id"] for r in sheet] == [r["turn_id"] for r in sheet2]
          and all(r[c] == "" for r in sheet for c in ("stance", "hostility", "frames", "sub_topic", "self_id", "notes"))
          and "stance_median" not in hdr and "stance_votes" not in hdr and "rationale" not in hdr and "cost_usd" not in hdr
          and hdr == stance.SHEET_COLS and "text" in hdr and "prev_text" in hdr and "next_text" in hdr,
          "coding sheet: the model's columns (median, votes, rationale, cost) are absent, the coder's columns are blank, "
          "text + both neighbours present, order fixed by the seed; n capped at the turns available", str(hdr))
    A = [{"turn_id": f"t{i}", "stance": s, "hostility": h} for i, (s, h) in enumerate([("-2", 0), ("-1", 1), ("0", 0), ("1", 2), ("2", 3), ("not_applicable", 0), ("1", 0), ("-1", 1)])]
    B = [{"turn_id": f"t{i}", "stance": s, "hostility": h} for i, (s, h) in enumerate([("-2", 0), ("-1", 1), ("0", 0), ("1", 2), ("2", 3), ("not_applicable", 0), ("1", 0), ("-1", 1)])]
    same = stance.agreement(A, B)
    Bx = [dict(r) for r in B]
    Bx[0]["stance"] = "2"; Bx[6]["stance"] = "not_applicable"; Bx[4]["hostility"] = 0
    diff = stance.agreement(A, Bx)
    check(abs(same["stance_kappa_w"] - 1.0) < 1e-9 and same["stance_exact"] == 1.0 and abs(same["hostility_kappa_w"] - 1.0) < 1e-9
          and same["na_agreement"] == 1.0 and abs(same["stance_alpha"] - 1.0) < 1e-9 and same["n_numeric_pairs"] == 7
          and diff["stance_kappa_w"] < 0.6 and diff["n_numeric_pairs"] == 6 and diff["na_agreement"] == 7 / 8
          and diff["hostility_kappa_w"] < 1.0 and len(diff["disagreements"]) == 2,
          "agreement: identical sheets -> weighted kappa 1, exact 1, alpha 1; flipping a -2 to +2 and a number to NA drops kappa below "
          "the 0.6 gate, NA agreement to 7/8, and lists the two disagreements", f"{same}\n{diff}")
    con.close()


def section_methods_page(tmp):
    """The showcase site's Methods page must quote the instruments byte-exactly.

    docs/methods.html renders docs/data/methods.json, which
    scripts/site_export/export_methods.py builds by reading the instruments out
    of this repo. The risk the page introduces is drift: someone re-codes a
    codebook, the hash in classify.py moves, and a stale methods.json keeps
    quoting the old text under the new version string. So every verbatim block
    carries a sha256 of its own text, and these checks recompute those hashes
    from the live sources. A FAIL here means the page is stale -- rebuild it,
    do not edit the JSON.

    This used to test ~/ytpt-site, the standalone site tree the page lived in
    until it moved under docs/ (the Pages root: export_site_data.py writes
    there, and Pages serves main /docs). That tree stopped being rebuilt on
    2026-10-19, so its copies of the exporters still emit `ci95: [None, None]`
    for the renter term and crash on the 2026-10-20 refit -- a failure in a
    tree nobody publishes. Test what ships.
    """
    print("\n33. showcase Methods page: every verbatim block still matches its source")
    site = ROOT / "docs"
    page, exporter, doc_path = site / "methods.html", ROOT / "scripts" / "site_export" / "export_methods.py", site / "data" / "methods.json"

    check(page.exists(), "methods.html exists", str(page))
    check(exporter.exists(), "export/export_methods.py exists", str(exporter))
    if not (page.exists() and exporter.exists()):
        return

    # -- 1. the page builds: the exporter runs clean and writes a parseable doc --
    r = subprocess.run([PY, str(exporter)], capture_output=True, text=True, cwd=str(ROOT))
    check(r.returncode == 0, "export_methods.py runs clean", (r.stderr or r.stdout)[-400:])
    if r.returncode != 0:
        return
    check(doc_path.exists(), "data/methods.json written", str(doc_path))
    try:
        M = json.loads(doc_path.read_text())
    except Exception as e:
        check(False, "data/methods.json parses as JSON", repr(e))
        return
    check(True, "data/methods.json parses as JSON")

    sha = lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest()

    # -- 2. the locate codebooks, against pipeline/classify.py's own constants --
    sys.path.insert(0, str(ROOT / "pipeline"))
    import classify                                    # noqa: PLC0415
    by_name = {c["name"]: c for c in M["locate"]["codebooks"]}
    check(set(by_name) == set(classify.CODEBOOKS),
          "the page carries every codebook classify.py defines, and no others",
          f"page={sorted(by_name)} code={sorted(classify.CODEBOOKS)}")
    for name, text in classify.CODEBOOKS.items():
        c = by_name.get(name)
        if not c:
            continue
        check(c["text"] == text, f"codebook {name}: exported text is byte-identical to classify.CODEBOOKS[{name!r}]",
              f"{len(c['text'])} vs {len(text)} chars")
        check(c["sha256"] == sha(text), f"codebook {name}: exported sha256 matches the source text")
        check(c["version"] == classify.CODEBOOK_VERSIONS[name],
              f"codebook {name}: version string matches classify.CODEBOOK_VERSIONS",
              f"{c['version']} vs {classify.CODEBOOK_VERSIONS[name]}")
    check(M["locate"]["v21_delta"]["text"] == classify.V21_TRANSIT_RULE,
          "the quoted v2.1-minus-v2 sentence is classify.V21_TRANSIT_RULE verbatim")
    check(M["locate"]["model"] == classify.LEGACY_MODEL and M["locate"]["screen"]["max_run_s"] == classify.DEFAULTS["max_run"]
          and M["locate"]["screen"]["pv_max_run_s"] == classify.PV_MAX_RUN,
          "locate model id and window constants come from classify.py",
          f"{M['locate']['model']} {M['locate']['screen']}")

    # -- 3. the stance codebook, against plan/STANCE-CODEBOOK.md ---------------
    import stance                                      # noqa: PLC0415
    cb = stance.codebook_text()
    check(M["stance"]["codebook"]["text"] == cb,
          "stance codebook: exported text is byte-identical to stance.codebook_text()",
          f"{len(M['stance']['codebook']['text'])} vs {len(cb)} chars")
    check(M["stance"]["codebook"]["sha256"] == sha(cb), "stance codebook: exported sha256 matches the source text")
    check(M["stance"]["codebook_version"] == stance.CODEBOOK_VERSION,
          "stance codebook version matches stance.CODEBOOK_VERSION",
          f"{M['stance']['codebook_version']} vs {stance.CODEBOOK_VERSION}")

    # -- 4. the NIMBY/YIMBY prompt, against scripts/nimby_pilot.py ------------
    sys.path.insert(0, str(ROOT / "scripts"))
    import nimby_pilot as nb                           # noqa: PLC0415
    check(M["speakers"]["prompt"]["text"] == nb.SYSTEM,
          "NIMBY prompt: exported text is byte-identical to nimby_pilot.SYSTEM",
          f"{len(M['speakers']['prompt']['text'])} vs {len(nb.SYSTEM)} chars")
    check(M["speakers"]["prompt"]["sha256"] == sha(nb.SYSTEM), "NIMBY prompt: exported sha256 matches the source text")
    check(M["speakers"]["prompt_version"] == nb.PROMPT_VERSION,
          "NIMBY prompt_version matches nimby_pilot.PROMPT_VERSION",
          f"{M['speakers']['prompt_version']} vs {nb.PROMPT_VERSION}")

    # -- 5. the README chart: not published any more, still the version source -
    # The owner removed "The pipeline in one paragraph" and the production
    # flowchart inside it on 2026-10-21 ("remove the pipeline in one paragraph
    # section on the methdos", "and the flow associated"). The chart is still
    # read out of README.md, because section_versions() takes the pinned
    # versions of the non-LLM steps off it rather than retyping them; it is just
    # no longer exported as a section of its own. README.md itself is untouched.
    check("pipeline" not in M, "methods.json no longer carries a pipeline entry", str(sorted(M))[:200])
    fences = re.findall(r"```mermaid\n(.*?)```", (ROOT / "README.md").read_text(), re.S)
    chart = fences[0].rstrip("\n") if fences else ""
    check(bool(fences), "README.md still has the production mermaid fence the versions table reads")
    vr = {v["instrument"]: v for v in M["versions"]}
    for name in ("Keyword screen", "GPU transcription", "Redaction"):
        row = vr.get(name)
        check(bool(row) and row["version"] in chart,
              f"versions row {name!r} carries a version string taken off README.md's chart",
              str(row))

    # -- 6. the versions table is generated, and every hash in it resolves -----
    vrows = {v["instrument"]: v for v in M["versions"]}
    check(len(vrows) == len(M["versions"]) and len(vrows) >= 8,
          "versions table has a row per instrument, no duplicates", f"{len(M['versions'])} rows")
    hashed = [v for v in M["versions"] if v.get("sha256")]
    check(len(hashed) == 5, "the five LLM instruments carry a text hash in the versions table", f"{len(hashed)}")
    all_shas = {c["sha256"] for c in M["locate"]["codebooks"]} | {M["stance"]["codebook"]["sha256"],
                                                                  M["speakers"]["prompt"]["sha256"]}
    check(all(v["sha256"] in all_shas for v in hashed),
          "every hash in the versions table is one of the verbatim blocks on the page")
    for v in M["versions"]:
        check(bool(v["version"]) and bool(v["status"]), f"versions row {v['instrument']!r} has a version and a status", str(v))

    # -- 7. the page's own wiring: the sections and the data file it fetches ---
    html = page.read_text()
    check('d3.json("data/methods.json")' in html, "methods.html fetches data/methods.json")
    for anchor in ("locate", "stance", "speakers", "models", "limitations", "versions"):
        check(f'id="s-{anchor}"' in html and f'id="{anchor}"' in html,
              f"methods.html has the {anchor} section and its anchor")
    # Removed 2026-10-21 with the section itself: no anchor, no render target,
    # and no mermaid import left anywhere on the page. Section 1's "What we did,
    # step by step" flow is a different chart, built from data/flow.json by D3,
    # and must still be here.
    check('id="pipeline"' not in html and 'id="s-pipeline"' not in html,
          "the pipeline section and its anchor are gone from methods.html")
    check("mermaid" not in html, "no mermaid import or chart source is left on methods.html")
    check('id="flow"' in html and 'id="flow-svg"' in html and 'data/flow.json' in html,
          "methods.html still carries the step-by-step flow built from data/flow.json")
    numbered = re.findall(r'<h2 id="([a-z]+)">(\d+)\. ', html)
    check([a for a, _ in numbered] == ["flow", "locate", "stance", "speakers", "models",
                                       "limitations", "versions"]
          and [n for _, n in numbered] == [str(i) for i in range(1, 8)],
          "the Methods sections are numbered 1 to 7 in order, with no gap", str(numbered))
    check(">EXPLORATORY<" in html or "<b>EXPLORATORY</b>" in html, "methods.html carries the EXPLORATORY banner")
    check('href="methods.html"' in (site / "index.html").read_text(), "the dashboard links to the Methods page")

    # -- 8. the exploratory stamp and the two caveats the review requires ------
    # The stamp is now the standard's own sentence (style/DASHBOARD-LANGUAGE.md
    # rule 12) and opens "Exploratory analysis."; it used to open "EXPLORATORY -".
    check(M["exploratory"].lower().startswith("exploratory analysis."),
          "methods.json carries the exploratory stamp", M["exploratory"][:80])
    check("circularity" in M["locate"]["recall"] and len(M["locate"]["recall"]["circularity"]) > 400,
          "the recall section carries the circularity note from VALIDATION 6")
    check(all(r["family"] in ("any", "transit", "density") for r in M["locate"]["recall"]["rows"])
          and all("v1_ref" in r and "adj_ref" in r for r in M["locate"]["recall"]["rows"]),
          "every recall row is one arm scored on both references (like-for-like)")
    check("not been passed" in M["stance"]["kappa_gate"].lower(), "the stance section states the kappa gate is unpassed",
          M["stance"]["kappa_gate"][:120])
    names = {c["name"] for c in M["speakers"]["caveats"]}
    check(any("Sign flip" in n for n in names) and any("one turn" in n for n in names),
          "the speaker section carries the sign-flip and one-scored-turn caveats", str(names))


def main():
    tmp = Path(tempfile.mkdtemp(prefix="pipetest-"))
    db = tmp / "corpus.sqlite"

    print("1. simulate")
    subprocess.run([PY, str(ROOT/"analysis"/"simulate.py"), "--out", "output_sim"],
                   check=True, capture_output=True)
    print("2. load")
    r = subprocess.run([PY, str(ROOT/"analysis"/"load_db.py"), "--db", str(db)],
                       check=True, capture_output=True, text=True)
    print("   " + r.stdout.strip().splitlines()[0])

    print("\n3. statistics unit checks")
    # A word equally frequent in both corpora must have ~0 effect and ~0 G^2.
    check(abs(wf.log_ratio(100, 100, 10_000, 10_000)) < 0.01, "log_ratio: equal rates -> 0")
    check(wf.log_likelihood(100, 100, 10_000, 10_000) < 0.01, "log_likelihood: equal rates -> 0")
    # Doubling the rate must give exactly +1 doubling (within smoothing slack).
    check(abs(wf.log_ratio(200, 100, 10_000, 10_000) - 1.0) < 0.02, "log_ratio: 2x rate -> +1.0")
    # G^2 must GROW with corpus size for the same proportional gap -- this is the
    # property that makes it useless as a ranking key, and the reason we rank on
    # effect size instead. If this ever stops holding, the docs are wrong.
    small = wf.log_likelihood(20, 10, 1_000, 1_000)
    big   = wf.log_likelihood(200, 100, 10_000, 10_000)
    check(big > small * 5, "log_likelihood scales with corpus size (why we rank on effect size)")
    # logDice must not be dominated by rarity the way PMI is.
    check(wf.log_dice(50, 100, 60) > wf.log_dice(2, 100, 2), "log_dice: frequent collocate outranks rare one")

    section_kwic(db)

    print("\n6. host filtering actually removes guest speech")
    all_c, _ = wf.load(db, host_only=False)
    host_c, dropped = wf.load(db, host_only=True)
    check(dropped > 0, "some non-host segments were dropped", f"dropped={dropped}")
    tk_all  = sum(len(v) for v in all_c["tucker"].values())
    tk_host = sum(len(v) for v in host_c["tucker"].values())
    check(tk_host < tk_all * 0.8,
          "tucker (interview-heavy) loses a large share to host filtering",
          f"{tk_host}/{tk_all}")

    print("\n7. cluster bootstrap CI is wider than a naive token-level interval")
    # Concentrate a word in a few videos: the cluster bootstrap must notice the
    # between-video variance that a token-level interval would miss entirely.
    counts = [100, 0, 0, 0, 100, 0, 0, 0, 100, 0]
    totals = [1000] * 10
    lo, hi = wf.cluster_bootstrap_ci(counts, totals)
    rate = sum(counts) / sum(totals) * 1e6
    naive_half = 1.96 * (sum(counts) ** 0.5) / sum(totals) * 1e6
    check((hi - lo) / 2 > naive_half * 2,
          "clustered CI much wider than naive",
          f"cluster +/-{(hi-lo)/2:.0f} vs naive +/-{naive_half:.0f} on rate {rate:.0f}/M")

    print("\n8. KWIC node is a whole word, not a substring")
    ice = wf.node_pattern("ice")
    check(not ice.search("the police and justice"), "'ice' does not match inside police/justice")
    check(bool(ice.search("abolish ICE, now")) and bool(ice.search("ICE's budget")), "'ice' matches ICE and ICE's")
    check(bool(wf.node_pattern("october 7").search("since October  7 happened")), "phrase nodes match across whitespace")

    print("\n9. political direction words survive filtering")
    toks = wf.tokens("the radical left and the far right")
    check("left" in toks and "right" in toks, "'left' and 'right' both kept", f"tokens={toks}")

    for fn in (section_frame, section_crawl, section_qa, section_host, section_ops, section_workers, section_gpu, section_segment, section_roles,
               section_meetings, section_a0, section_validation, section_review, section_t2, section_t2b, section_t2c,
                section_t2d, section_t2e, section_t2f, section_t2g, section_t2h, section_t2i, section_stance, section_redact, section_models, section_methods_page):
        try:
            fn(tmp)
        except Exception as e:                      # a crashed section is a failure, not a traceback
            check(False, f"{fn.__name__} ran without crashing", repr(e))

    shutil.rmtree(tmp, ignore_errors=True)
    print()
    if fails:
        print(f"FAILED {len(fails)} check(s):")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
