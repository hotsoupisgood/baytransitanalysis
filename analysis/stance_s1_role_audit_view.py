#!/usr/bin/env python3
"""
analysis/stance_s1_role_audit_view.py -- the NIMBY/YIMBY per-cluster view and
the public-vs-dais contrast recomputed under a switchable role column, so the
cue role (scripts/role_label.py) and the audited role (scripts/role_audit.py,
state.sqlite `role_audit`) can be read side by side.

Same frame, same unit and the same bootstrap as analysis/stance_s1_nimby_view.py
(density/both episodes, episode x cluster, T = 0.5, episodes resampled with
replacement inside the body, never pooled across locate routes). The only
change is which column supplies `role`:

  --role-col cue    turn_stance.role            (reproduces the original view)
  --role-col audit  role_audit.cluster_role_audit

With --role-col audit the outputs get a `_roleaudit` suffix:
  speaker_clusters_roleaudit.csv, public_dais_shares_by_body_route_roleaudit.csv,
  public_shares_by_route_roleaudit.csv, summary_roleaudit.json,
  fig_public_yimby_nimby_<route>_roleaudit.png,
  fig_public_yimby_nimby_<route>_stars_roleaudit.png,
  fig_public_vs_dais_<route>_roleaudit.png

EXPLORATORY: the stance kappa gate (SHOWCASE.md 4a) is not passed; every
figure says so. No DB row outside the `role_audit` table is written.

RUN
  .venv/bin/python analysis/stance_s1_role_audit_view.py --role-col audit
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "pipeline"))
import stance, state  # noqa: E402

OUT = ROOT / "output" / "stance_s1" / "nimby_view"
STAMP = "EXPLORATORY, stance κ gate not passed"
LABELS = ("YIMBY", "neutral", "NIMBY")
COL = {"YIMBY": "#2a7f62", "NIMBY": "#b5543a", "neutral": "#898781"}
MIN_CL = 10


def load(pv, frame_csv, role_col):
    con = state.connect(None, create=False)
    ts = pd.read_sql_query(
        "SELECT episode_id, cluster, turn_idx, role, body, family, stance, stance_median, hostility_median, "
        "sub_topic, words FROM turn_stance WHERE prompt_version = ?", con, params=(pv,))
    if role_col == "audit":
        ra = pd.read_sql_query(
            "SELECT episode_id, cluster, turn_idx, cluster_role_audit FROM role_audit WHERE prompt_version = ?",
            con, params=(pv,))
        ts = ts.merge(ra, on=["episode_id", "cluster", "turn_idx"], how="left", validate="one_to_one")
        miss = int(ts.cluster_role_audit.isna().sum())
        if miss:
            print(f"warning: {miss} turns with no role_audit row; cue role kept for those", flush=True)
        ts["role"] = ts.cluster_role_audit.fillna(ts.role)
    con.close()
    fr = pd.read_csv(frame_csv, usecols=["episode_id", "cluster", "turn_idx", "method", "roles_source"])
    return ts.merge(fr, on=["episode_id", "cluster", "turn_idx"], how="inner", validate="one_to_one")


def clusters(df, thresh):
    d = df[df["family"].isin(["density", "both"]) & df["role"].isin(["dais", "public"])].copy()
    g = d.groupby(["episode_id", "cluster", "role", "body", "family", "method", "roles_source"], as_index=False).agg(
        n_turns=("stance", "size"), n_numeric=("stance_median", "count"), mean_stance=("stance_median", "mean"),
        words=("words", "sum"), mean_hostility=("hostility_median", "mean"),
        sub_topics=("sub_topic", lambda s: ";".join(f"{k}:{v}" for k, v in Counter(s).most_common(3))))
    g["label"] = np.where(g["n_numeric"] == 0, "no stance",
                          np.where(g["mean_stance"] >= thresh, "YIMBY",
                                   np.where(g["mean_stance"] <= -thresh, "NIMBY", "neutral")))
    return g


def shares(cl, route, role, min_clusters, reps, rng):
    rows = []
    c = cl[(cl["method"] == route) & (cl["role"] == role) & (cl["label"] != "no stance")]
    for body, g in c.groupby("body"):
        if len(g) < min_clusters:
            continue
        eps = sorted(g["episode_id"].unique())
        M = pd.crosstab(g["episode_id"], g["label"]).reindex(index=eps, columns=list(LABELS), fill_value=0).to_numpy(float)
        tot = M.sum(axis=0); obs = tot / tot.sum()
        boots = np.empty((reps, len(LABELS)))
        for i in range(reps):
            w = rng.multinomial(len(eps), np.full(len(eps), 1.0 / len(eps)))
            t = w @ M; boots[i] = t / t.sum() if t.sum() else np.nan
        lo, hi = np.nanpercentile(boots, [5, 95], axis=0)
        lo95, hi95 = np.nanpercentile(boots, [2.5, 97.5], axis=0)
        net = boots[:, 0] - boots[:, 2]
        st = Counter()
        for s in g["sub_topics"]:
            for kv in s.split(";"):
                k, v = kv.split(":"); st[k] += int(v)
        row = {"route": route, "role": role, "body": body, "n_clusters": int(len(g)), "n_episodes": len(eps),
               "n_no_stance_clusters": int(((cl["method"] == route) & (cl["role"] == role) & (cl["body"] == body) & (cl["label"] == "no stance")).sum()),
               "mean_turns_per_cluster": round(float(g["n_turns"].mean()), 2)}
        for j, l in enumerate(LABELS):
            row.update({f"{l}_share": obs[j], f"{l}_lo90": lo[j], f"{l}_hi90": hi[j], f"{l}_lo95": lo95[j], f"{l}_hi95": hi95[j]})
        row.update({"net_yimby_minus_nimby": obs[0] - obs[2],
                    "net_lo90": float(np.nanpercentile(net, 5)), "net_hi90": float(np.nanpercentile(net, 95)),
                    "star": bool(np.nanpercentile(net, 5) > 0 or np.nanpercentile(net, 95) < 0),
                    "top_sub_topics": ";".join(f"{k}:{v}" for k, v in st.most_common(3))})
        rows.append(row)
    return rows


def _plt():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def figure_shares(res, route, thresh, path, stars):
    plt = _plt()
    res = sorted(res, key=lambda r: r["net_yimby_minus_nimby"])
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(res) + 2.2))
    y = np.arange(len(res))
    for l, off in (("YIMBY", 0.21), ("NIMBY", -0.21)):
        v = np.array([r[f"{l}_share"] for r in res]); lo = np.array([r[f"{l}_lo90"] for r in res]); hi = np.array([r[f"{l}_hi90"] for r in res])
        ax.barh(y + off, v, height=0.4, color=COL[l], label=l, alpha=0.92)
        ax.errorbar(v, y + off, xerr=[np.clip(v - lo, 0, None), np.clip(hi - v, 0, None)],
                    fmt="none", ecolor="#333", elinewidth=0.8, capsize=2)
    lab = [f"{'* ' if (stars and r['star']) else ''}{r['body']} (n={r['n_clusters']} clusters, {r['n_episodes']} ep)" for r in res]
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=8); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("share of public speaker clusters with a stance (90% episode-bootstrap)", fontsize=8)
    sub = "* = 90% interval on YIMBY−NIMBY excludes zero. " if stars else ""
    # Owner, 2026-10-20 22:02: no internal role-audit code in a published title. The
    # role column is still stated on the panel and in summary.json; the EXPLORATORY
    # stamp stays on the figure.
    ax.set_title(f"thtrans: public YIMBY vs NIMBY by body, route={route}\n{sub}{STAMP}", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def figure_public_vs_dais(rows, route, path, tag):
    plt = _plt()
    bodies = sorted({r["body"] for r in rows if r["role"] == "public"} & {r["body"] for r in rows if r["role"] == "dais"},
                    key=lambda b: -next(r["n_clusters"] for r in rows if r["role"] == "public" and r["body"] == b))
    if not bodies:
        return False
    idx = {(r["role"], r["body"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(bodies) + 2.0))
    y = np.arange(len(bodies))
    for role, off, c, m in (("public", 0.14, "#2a7f62", "o"), ("dais", -0.14, "#444444", "s")):
        v = np.array([idx[(role, b)]["net_yimby_minus_nimby"] for b in bodies])
        lo = np.array([idx[(role, b)]["net_lo90"] for b in bodies]); hi = np.array([idx[(role, b)]["net_hi90"] for b in bodies])
        ax.errorbar(v, y + off, xerr=[v - lo, hi - v], fmt=m, color=c, ecolor=c, capsize=2,
                    label=f"{role} (net YIMBY−NIMBY, 90%)")
    ax.axvline(0, ls="--", color="#333", lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels(
        [f"{b} (pub n={idx[('public', b)]['n_clusters']}, dais n={idx[('dais', b)]['n_clusters']})" for b in bodies], fontsize=8)
    ax.set_xlim(-1, 1); ax.invert_yaxis(); ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("net share: YIMBY − NIMBY among speaker clusters with a stance", fontsize=8)
    ax.set_title(f"thtrans: public vs dais by body, route={route} [{tag}]. {STAMP}", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return True


def main(argv=None):
    global MIN_CL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role-col", choices=("cue", "audit"), default="audit")
    ap.add_argument("--frame", default=str(ROOT / "output" / "stance_s1" / "s1_frame.csv"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--suffix", default=None, help="default: '' for cue, '_roleaudit' for audit")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-clusters", type=int, default=10)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261018)
    a = ap.parse_args(argv)
    MIN_CL = a.min_clusters
    sfx = a.suffix if a.suffix is not None else ("_roleaudit" if a.role_col == "audit" else "")
    tag = "role_audit ra-1" if a.role_col == "audit" else "cue role"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pv = stance.prompt_version_for(state.setting("SCORING_MODEL", stance.DEFAULT_MODEL))
    df = load(pv, a.frame, a.role_col)
    cl = clusters(df, a.threshold)
    cl.to_csv(out / f"speaker_clusters{sfx}.csv", index=False)
    rng = np.random.default_rng(a.seed)
    summ = {"prompt_version": pv, "role_col": a.role_col, "threshold": a.threshold,
            "min_clusters_per_body": a.min_clusters, "bootstrap_reps": a.reps, "seed": a.seed, "stamp": STAMP,
            "n_clusters": int(len(cl)),
            "clusters_by_role_label": {f"{r}/{l}": int(n) for (r, l), n in cl.groupby(["role", "label"]).size().items()},
            "routes": {}, "figures": []}
    print(f"[{tag}] {len(cl)} clusters; " + ", ".join(f"{k} {v}" for k, v in summ["clusters_by_role_label"].items()), flush=True)
    all_rows = []
    for route in sorted(cl["method"].unique(), key=lambda r: -(cl["method"] == r).sum()):
        rows_pub = shares(cl, route, "public", a.min_clusters, a.reps, rng)
        rows_dais = shares(cl, route, "dais", a.min_clusters, a.reps, rng)
        all_rows += rows_pub + rows_dais
        summ["routes"][route] = {"bodies_public": len(rows_pub), "bodies_dais": len(rows_dais),
                                 "stars_public": [r["body"] for r in rows_pub if r["star"]]}
        if rows_pub:
            for stars, name in ((False, f"fig_public_yimby_nimby_{route}{sfx}.png"),
                                (True, f"fig_public_yimby_nimby_{route}_stars{sfx}.png")):
                figure_shares(rows_pub, route, a.threshold, out / name, stars)
                summ["figures"].append(name)
            for r in sorted(rows_pub, key=lambda r: r["net_yimby_minus_nimby"]):
                print(f"   {r['body']:<12} n={r['n_clusters']:<4} YIMBY {r['YIMBY_share']:.2f}  NIMBY {r['NIMBY_share']:.2f}  "
                      f"net {r['net_yimby_minus_nimby']:+.2f} ({r['net_lo90']:+.2f},{r['net_hi90']:+.2f})"
                      f"{'  *' if r['star'] else ''}", flush=True)
        if figure_public_vs_dais(rows_pub + rows_dais, route, out / f"fig_public_vs_dais_{route}{sfx}.png", tag):
            summ["figures"].append(f"fig_public_vs_dais_{route}{sfx}.png")
    pd.DataFrame(all_rows).to_csv(out / f"public_dais_shares_by_body_route{sfx}.csv", index=False)
    wide = []
    for route, g in cl[(cl["role"] == "public") & (cl["label"] != "no stance")].groupby("method"):
        c = Counter(g["label"]); n = len(g)
        wide.append({"route": route, "n_public_clusters": n, **{f"{l}_share": c[l] / n for l in LABELS}})
    g = cl[(cl["role"] == "public") & (cl["label"] != "no stance")]
    c = Counter(g["label"]); n = len(g)
    allr = {"route": "ALL", "n_public_clusters": n, **{f"{l}_share": c[l] / n for l in LABELS}}
    wide.append(allr)
    pd.DataFrame(wide).to_csv(out / f"public_shares_by_route{sfx}.csv", index=False)
    summ["public_shares_by_route"] = wide
    print(f"frame-wide public split ({n} clusters): "
          f"YIMBY {allr['YIMBY_share']*100:.1f} / neutral {allr['neutral_share']*100:.1f} / NIMBY {allr['NIMBY_share']*100:.1f}", flush=True)
    (out / f"summary{sfx}.json").write_text(json.dumps(summ, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)) + "\n")
    print(f"wrote {out} (suffix '{sfx}')", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
