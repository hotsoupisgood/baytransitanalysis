"""analysis/models -- the two Bayesian models of the showcase (plan/SHOWCASE.md 1).

  load.py        data construction: body x month x family x route cells with
                 meeting-hour exposure; the turn table for the role gap (real
                 `turn_stance` when it exists, the VADER STAND-IN until then)
  topic_time.py  estimand 1: negative-binomial rate of episode-minutes per
                 meeting-hour, body random intercepts, month random walk per
                 family, locate route as a stratum -- and per-route fits
  role_gap.py    estimand 2: cumulative-logit stance / hostility ~ role * family
                 + (1 | body) + (1 | episode)
  plots.py       the figures, in the A0 house style
  run.py         `make models`: everything into output/analysis/models/

Model statements and priors: analysis/models/README.md.
"""
