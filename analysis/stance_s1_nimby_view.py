#!/usr/bin/env python3
"""
analysis/stance_s1_nimby_view.py -- the YIMBY / NIMBY derived view of the S1
stance pass (plan/STANCE-S1-RESULTS.md 7b, plan/NIMBY-PILOT.md 5). EXPLORATORY;
the stance kappa gate (SHOWCASE.md 4a) is not passed and every output says so.

**The reported role is `role_audit` (rule ra-1, scripts/role_audit.py,
state.sqlite `role_audit`) as of 2026-10-18** -- plan/ROLE-AUDIT-2026-10-18.md
7. The diarization-cue role (scripts/role_label.py) mislabels 20-39 % of the
adjudicable turns in the public bucket as public when the speaker's own words
say `elected` or `staff`; SELF_ID is the second instrument and it reads what
the speaker says about themselves. The cue run is kept as a sensitivity:

  (default)         role_audit.cluster_role_audit  -> unsuffixed outputs
  --role-col cue    turn_stance.role               -> `_cue` outputs

Unit: the anonymous speaker cluster inside one episode (episode x diarized
cluster, DECISIONS #15 -- never linked across meetings). For every cluster
with at least one turn carrying a numeric stance in the S1 pass, the mean of
its turns' stance_median is its density stance; YIMBY if mean >= +T, NIMBY if
<= -T, else neutral (T = 0.5, half a codebook step, the NIMBY-PILOT lexicon
threshold's role). Clusters whose turns are all not_applicable are
"no stance" and sit outside the share denominator. Frame: density-family and
both-family episodes of the S1 frame (stance toward a housing item); transit
episodes are out, since YIMBY/NIMBY is a housing construct. Shares of public
clusters per body with an episode-cluster bootstrap (episodes resampled with
replacement inside the body), one table and one figure per locate route,
never pooled across routes; dais shares in the table for contrast.

Caveat carried from NIMBY-PILOT 3.3: S1 stance is toward the agenda item; on
an item that is itself a restriction (a moratorium, rent control, a height
limit) "supportive" is the NIMBY side, so the mapping flips sign there. The
sub_topic the pass recorded is listed per body so the reader can see where
that bites (affordability_displacement carries the rent-control fights).

Caveat carried from ROLE-AUDIT 8: SELF_ID is model output, not a gold label;
44 % of turns are `self_id = unknown` and keep the cue role, so the audited
public bucket is still contaminated at an unknown rate; and the audit is a
*re-partition* (it moves clusters both ways, public 414 -> 457 with a stance),
not "contamination removed".

The computation lives in analysis/stance_s1_role_audit_view.py, which this
module calls so the two role columns can never drift apart. Figures per route:
fig_public_yimby_nimby_<route>{,_stars}{,_cue}.png,
fig_public_vs_dais_<route>{,_cue}.png.

    .venv/bin/python analysis/stance_s1_nimby_view.py                 # role_audit (reported)
    .venv/bin/python analysis/stance_s1_nimby_view.py --role-col cue  # cue sensitivity -> _cue
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis import stance_s1_role_audit_view as V  # noqa: E402

OUT = V.OUT
#: the reported role column, and the suffix each role column writes under.
#: audit is unsuffixed because it is what STANCE-S1-RESULTS 7b now quotes.
SUFFIX = {"audit": "", "cue": "_cue"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    role_col = "audit"
    if "--role-col" in argv:
        i = argv.index("--role-col")
        role_col = argv[i + 1] if i + 1 < len(argv) else "audit"
    if role_col not in SUFFIX:
        print(f"--role-col must be one of {sorted(SUFFIX)}", file=sys.stderr)
        return 2
    if "--suffix" not in argv:
        argv += ["--suffix", SUFFIX[role_col]]
    if "--role-col" not in argv:
        argv += ["--role-col", role_col]
    print(f"[stance_s1_nimby_view] role column '{role_col}' "
          f"(reported: audit, ROLE-AUDIT-2026-10-18 7); suffix '{SUFFIX[role_col]}'", flush=True)
    return V.main(argv)


if __name__ == "__main__":
    sys.exit(main())
