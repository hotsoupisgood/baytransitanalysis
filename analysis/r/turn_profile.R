#!/usr/bin/env Rscript
# turn_profile.R -- profile-likelihood CI for the role effect in the nested
# turn-level model.
#
# The three-level fit (1|body) + (1|meeting_id) + (1|episode_id) is singular:
# meeting is nested in body and absorbs it, so var(body) is exactly 0 and
# lme4's profiler cannot step off the boundary. The identical non-singular
# reparameterisation drops the zero term -- same log-likelihood, same fixed
# effects -- and profiles cleanly. Both are reported.

suppressPackageStartupMessages({ library(lme4); library(jsonlite) })
ROOT <- path.expand("~/ytpt")
OUT <- file.path(ROOT, "output", "renter_share_6")
say <- function(...) { cat(..., "\n"); flush.console() }

tu <- read.csv(file.path(OUT, "data", "turns.csv"))
for (k in c("body", "meeting_id", "episode_id", "route")) tu[[k]] <- factor(tu[[k]])
tu$role_x_renter <- tu$role_code * tu$z_renter
tu$route <- relevel(tu$route, ref = "classifier")

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

# bobyqa stops a little short of the true optimum, and the profiler then
# refuses to run ("new, lower deviance"). Restart from the fitted values until
# the deviance stops improving, then escalate devtol if it is still short.
for (i in 1:5) {
  m2b <- update(m2, start = list(theta = getME(m2, "theta"),
                                 fixef = fixef(m2)),
                control = glmerControl(optimizer = "bobyqa",
                                       optCtrl = list(maxfun = 5e5,
                                                      rhoend = 1e-10)))
  gain <- as.numeric(logLik(m2b)) - as.numeric(logLik(m2))
  m2 <- m2b
  say(sprintf("  restart %d: logLik %.6f (gain %.2e)", i, logLik(m2), gain))
  if (gain < 1e-8) break
}

nth <- length(getME(m2, "theta"))
idx <- c(seq_len(nth), nth + which(names(fixef(m2)) %in%
                                   c("role_code", "z_renter", "role_x_renter")))
pp <- NULL
for (dt in c(1e-6, 1e-4, 1e-2, 5e-2)) {
  pp <- tryCatch(profile(m2, which = idx, signames = FALSE, devtol = dt),
                 error = function(e) { say("  devtol", dt, "failed:",
                                           conditionMessage(e)); NULL })
  if (!is.null(pp)) { say("  profiled at devtol", dt); DEVTOL <- dt; break }
}
if (is.null(pp)) stop("profiling failed at every tolerance")
ci <- as.data.frame(confint(pp, level = 0.95))
ci <- data.frame(parameter = rownames(ci), prof_lo = ci[, 1], prof_hi = ci[, 2],
                 row.names = NULL)
print(ci, digits = 5)

cf <- summary(m2)$coefficients
wald <- data.frame(parameter = rownames(cf), est = cf[, 1], se = cf[, 2],
                   p = cf[, 4], row.names = NULL)
vc <- as.data.frame(VarCorr(m2))[, c("grp", "vcov", "sdcor")]
print(vc, digits = 5)

write.csv(ci, file.path(OUT, "turn_profile_ci.csv"), row.names = FALSE)
write.csv(wald, file.path(OUT, "turn_fixed_2level.csv"), row.names = FALSE)
write(toJSON(list(logLik_3level = as.numeric(logLik(m3)),
                  logLik_2level = as.numeric(logLik(m2)),
                  singular_3level = isSingular(m3),
                  singular_2level = isSingular(m2), profile_devtol = DEVTOL,
                  fixef_max_diff = max(abs(fixef(m3) - fixef(m2))),
                  varcomp = vc, profile = ci, wald = wald),
             auto_unbox = TRUE, digits = 8),
      file.path(OUT, "turn_profile.json"))
say("->", OUT)
