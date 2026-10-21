#!/usr/bin/env Rscript
# turn_anyside.R -- the first step of the role gap in the same nested model as
# the second.
#
# analysis/r/turn_profile.R fits "which side, given a side" (NIMBY vs YIMBY
# turn). This fits "took a side at all" on every stance-bearing turn, with the
# identical specification: role + z_renter + role x renter + route, random
# intercepts for body, meeting and episode. Reporting both steps from one
# estimator is the point -- the dashboard quotes nothing else.
#
# As in turn_profile.R, the three-level fit is expected to be singular in
# `body` (meeting is nested inside body and absorbs it), so the profile CI is
# taken from the identical non-singular two-level reparameterisation and both
# log-likelihoods are written out so the reader can check they agree.

suppressPackageStartupMessages({ library(lme4); library(jsonlite) })
ROOT <- path.expand("~/ytpt")
OUT <- file.path(ROOT, "output", "renter_share_6")
say <- function(...) { cat(..., "\n"); flush.console() }

tu <- read.csv(file.path(OUT, "data", "turns_anyside.csv"))
for (k in c("body", "meeting_id", "episode_id", "route")) tu[[k]] <- factor(tu[[k]])
tu$role_x_renter <- tu$role_code * tu$z_renter
tu$route <- relevel(tu$route, ref = "classifier")
say(sprintf("rows %d ; took a side %d ; episodes %d ; meetings %d ; bodies %d",
            nrow(tu), sum(tu$y), nlevels(tu$episode_id),
            nlevels(tu$meeting_id), nlevels(tu$body)))

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

# The profile is taken off whichever fit is not on a variance boundary; if the
# three-level fit is clean, profile that one and report body with it.
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

write.csv(ci, file.path(OUT, "anyside_profile_ci.csv"), row.names = FALSE)
write.csv(wald, file.path(OUT, "anyside_fixed.csv"), row.names = FALSE)
write.csv(vc, file.path(OUT, "anyside_varcomp.csv"), row.names = FALSE)
write(toJSON(list(n = nrow(tu), n_took_a_side = sum(tu$y),
                  n_episodes = nlevels(tu$episode_id),
                  n_meetings = nlevels(tu$meeting_id),
                  n_bodies = nlevels(tu$body),
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
      file.path(OUT, "anyside_profile.json"))
say("->", OUT)
