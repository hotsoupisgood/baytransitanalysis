#!/usr/bin/env python3
"""
pipeline/roles.py -- speaker-role labelling on ingest (PROPOSAL-transit.md
3g, T0 test 6; wired 2026-09-21). Thin pipeline face of scripts/role_label.py,
which stays the tuning tool (`--score` against hand labels) with
plan/role_cues.txt as the tuning surface.

For every kind='episode' transcript gpu_job.ingest_new accepts, label each
diarized cluster dais / public / unknown (text + timing cues only, no model,
no network), write one `speaker_roles` row per cluster, and add `roles`,
`roles_summary` and `roles_cues_version` to the transcript JSON so the file
stays self-describing (keys added at the top level only, PLAN.md 1).

    python3 pipeline/roles.py <transcript.json> [...]      # print the labels, write nothing
"""
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import role_label as rl  # noqa: E402

CUES_PATH = rl.DEFAULT_CUES


def cues_version(path=CUES_PATH):
    return "cues-" + hashlib.sha1(Path(path).read_bytes()).hexdigest()[:10]


def label_transcript(doc, cues=None, cues_path=CUES_PATH):
    """(roles, regions, summary) for one contract transcript."""
    cues = cues or rl.load_cues(cues_path)
    speakers, regions = rl.label_episode(doc.get("segments") or [], cues, doc.get("transcribed_s") or doc.get("duration_s"))
    roles = {k: {"role": v["role"], "method": v["method"], "confidence": v["confidence"], "talk_s": v["talk_s"]}
             for k, v in speakers.items()}
    return roles, regions, rl.summarise(speakers)


def annotate(doc, cues=None, cues_path=CUES_PATH):
    """Adds roles / roles_summary / roles_cues_version to doc (in place) and returns the roles."""
    roles, regions, summ = label_transcript(doc, cues, cues_path)
    doc["roles"] = roles
    doc["roles_summary"] = dict(summ, n_public_regions=len(regions))
    doc["roles_cues_version"] = cues_version(cues_path)
    return roles


def write_roles(con, episode_id, roles, version):
    """speaker_roles rows for one episode (replace: labels are revisable)."""
    con.execute("BEGIN IMMEDIATE")
    try:
        con.execute("DELETE FROM speaker_roles WHERE episode_id = ?", (episode_id,))
        con.executemany("INSERT INTO speaker_roles (episode_id, speaker, role, confidence, method, talk_s, cues_version) "
                        "VALUES (?,?,?,?,?,?,?)",
                        [(episode_id, spk, r["role"], r["confidence"], r["method"], r["talk_s"], version) for spk, r in roles.items()])
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return len(roles)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cues = rl.load_cues()
    for f in argv:
        doc = json.loads(Path(f).read_text(encoding="utf-8"))
        roles, regions, summ = label_transcript(doc, cues)
        print(f"{Path(f).stem}: {len(roles)} clusters, {len(regions)} public-comment region(s); "
              f"dais {summ['dais_share']:.0%} public {summ['public_share']:.0%} unknown {summ['unknown_share']:.0%}")
        for k, v in roles.items():
            print(f"   {k:<11} {v['role']:<8} {v['method']:<11} conf {v['confidence']:.2f} {v['talk_s']:6.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
