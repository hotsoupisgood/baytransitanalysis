#!/usr/bin/env Rscript
# turn_nested.R -- the run-6 nested specification, one row set per call
# (2026-10-20).
#
# analysis/r/turn_anyside.R and analysis/r/turn_profile.R each fit one outcome
# with the nested specification. Step 6 of the freeze also needs the same
# specification for hostility and for each outcome with BART dropped, so this
# script takes the row set as an argument and changes nothing else: the fixed
# effects are role + z_renter + role x renter + route, the random intercepts
# are body, meeting and episode, route is relevelled to 'classifier', the
# profile CI is taken from the three-level fit unless it is singular in body
# (then from the identical two-level reparameterisation), and both
# log-likelihoods are written out so the reader can check they agree.
#
#   Rscript analysis/r/turn_nested.R <stem>
#
# reads  output/renter_share_6/data/turns_<stem>.csv
# writes output/renter_share_6/<stem>_{fixed,profile_ci,varcomp}.csv
#        output/renter_share_6/<stem>_profile.json

suppressPackageStartupMessages({ library(lme4); library(jsonlite) })
ROOT <- path.expand("~/ytpt")
OUT <- file.path(ROOT, "output", "renter_share_6")
say <- function(...) { cat(..., "\n"); flush.console() }

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: turn_nested.R <stem>")
STEM <- args[1]

tu <- read.csv(file.path(OUT, "data", paste0("turns_", STEM, ".csv")))
for (k in c("body", "meeting_id", "episode_id", "route")) tu[[k]] <- factor(tu[[k]])
tu$role_x_renter <- tu$role_code * tu$z_renter
if ("classifier" %in% levels(tu$route)) tu$route <- relevel(tu$route, ref = "classifier")

say(sprintf("%s: rows %d ; y %d ; episodes %d ; meetings %d ; bodies %d ; routes %d",
            STEM, nrow(tu), sum(tu$y), nlevels(tu$episode_id),
            nlevels(tu$meeting_id), nlevels(tu$body), nlevels(tu$route)))

ctl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
m3 <- glmer(y ~ role_code + z_renter + role_x_renter + route +
              (1 | body) + (1 | meeting_id) + (1 | episode_id),
            data = tu, family = binomial, control = ctl)
m2 <- glmer(y ~ role_code + z_renter + role_x_renter + route +
              (1 | meeting_id) + (1 | episode_id),
            data = tu, family = binomial, control = ctl)
say(sprintf("3-level logLik %.4f (singular=%s) ; 2-level logLik %.4f (singular=%s)",
            logLik(m3), isSingular(m3), logLik(m2), isSingular(m2)))
say("fixed-effect max |difference| between the two:",
    signif(max(abs(fixef(m3) - fixef(m2))), 3))

vc3 <- as.data.frame(VarCorr(m3))[, c("grp", "vcov", "sdcor")]
print(vc3, digits = 5)

base <- if (isSingular(m3)) m2 else m3
for (i in 1:5) {
  nxt <- update(base, start = list(theta = getME(base, "theta"),
                                   fixef = fixef(base)),
                control = glmerControl(optimizer = "bobyqa",
                                       optCtrl = list(maxfun = 5e5,
                                                      rhoend = 1e-10)))
  gain <- as.numeric(logLik(nxt)) - as.numeric(logLik(base))
  base <- nxt
  say(sprintf("  restart %d: logLik %.6f (gain %.2e)", i, logLik(base), gain))
  if (gain < 1e-8) break
}

nth <- length(getME(base, "theta"))
idx <- c(seq_len(nth), nth + which(names(fixef(base)) %in%
                                   c("role_code", "z_renter", "role_x_renter")))
pp <- NULL
DEVTOL <- NA
for (dt in c(1e-6, 1e-4, 1e-2, 5e-2)) {
  pp <- tryCatch(profile(base, which = idx, signames = FALSE, devtol = dt),
                 error = function(e) { say("  devtol", dt, "failed:",
                                           conditionMessage(e)); NULL })
  if (!is.null(pp)) { say("  profiled at devtol", dt); DEVTOL <- dt; break }
}
if (is.null(pp)) stop("profiling failed at every tolerance")
ci <- as.data.frame(confint(pp, level = 0.95))
ci <- data.frame(parameter = rownames(ci), prof_lo = ci[, 1], prof_hi = ci[, 2],
                 row.names = NULL)
print(ci, digits = 5)

cf <- summary(base)$coefficients
wald <- data.frame(parameter = rownames(cf), est = cf[, 1], se = cf[, 2],
                   p = cf[, 4], row.names = NULL)
print(wald, digits = 5)
vc <- as.data.frame(VarCorr(base))[, c("grp", "vcov", "sdcor")]

write.csv(ci, file.path(OUT, paste0(STEM, "_profile_ci.csv")), row.names = FALSE)
write.csv(wald, file.path(OUT, paste0(STEM, "_fixed.csv")), row.names = FALSE)
write.csv(vc, file.path(OUT, paste0(STEM, "_varcomp.csv")), row.names = FALSE)
write(toJSON(list(stem = STEM, n = nrow(tu), n_y = sum(tu$y),
                  n_episodes = nlevels(tu$episode_id),
                  n_meetings = nlevels(tu$meeting_id),
                  n_bodies = nlevels(tu$body),
                  routes = levels(tu$route),
                  logLik_3level = as.numeric(logLik(m3)),
                  logLik_2level = as.numeric(logLik(m2)),
                  singular_3level = isSingular(m3),
                  singular_2level = isSingular(m2),
                  profiled_levels = if (isSingular(m3)) 2 else 3,
                  profile_devtol = DEVTOL,
                  fixef_max_diff = max(abs(fixef(m3) - fixef(m2))),
                  varcomp_3level = vc3, varcomp = vc,
                  profile = ci, wald = wald),
             auto_unbox = TRUE, digits = 8),
      file.path(OUT, paste0(STEM, "_profile.json")))
say("->", OUT)
