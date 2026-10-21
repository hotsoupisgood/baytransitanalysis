#!/usr/bin/env Rscript
# fit_glmm.R -- the R counterpart to analysis/renter_share5_freq.py.
#
# lme4::glmer (nAGQ=25) and glmmTMB beta-binomial on the EXACT primary and
# secondary row sets exported by analysis/r/export_for_r.py, plus the nested
# turn-level model (finding 4 remedy), DHARMa diagnostics, performance::icc,
# leave-one-body-out and grouped 5-fold by body.
#
# Read-only on the data; writes output/renter_share_6/. No network, no spend.

suppressPackageStartupMessages({
  library(lme4); library(glmmTMB); library(DHARMa)
  library(performance); library(emmeans); library(jsonlite)
})

set.seed(20261019)
ROOT <- normalizePath(file.path(dirname(dirname(dirname(
  sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1])))), "."))
if (!dir.exists(file.path(ROOT, "output"))) ROOT <- path.expand("~/ytpt")
DATA <- file.path(ROOT, "output", "renter_share_6", "data")
OUT  <- file.path(ROOT, "output", "renter_share_6")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

say <- function(...) { cat(..., "\n"); flush.console() }
res <- list()

COV <- c("log_income", "log_density", "built_2010_plus", "nhwhite_share",
         "transit_commute", "vacancy_rate", "log_population")
ROUTES <- c("route_agenda_index", "route_fullasr", "route_local_asr",
            "route_swagit_transcript")
FIXED <- paste(c("role_num", "renter_share", "role_x_renter", COV, ROUTES),
               collapse = " + ")
WANT <- c("role_num", "renter_share", "role_x_renter", "transit_commute",
          "log_population", "log_income", "log_density")

num1 <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  if (length(x) != 1) NA_real_ else x
}

tidy_fixed <- function(est, se, lo, hi, nms) {
  data.frame(parameter = nms, est = est, se = se, lo = lo, hi = hi,
             z = est / se, p = 2 * pnorm(-abs(est / se)), row.names = NULL)
}

## ------------------------------------------------------- 1. lme4::glmer
glmer_block <- function(d, tag) {
  f <- as.formula(paste("cbind(y, N - y) ~", FIXED, "+ (1 | body)"))
  m <- glmer(f, data = d, family = binomial(link = "logit"), nAGQ = 25,
             control = glmerControl(optimizer = "bobyqa",
                                    optCtrl = list(maxfun = 2e5)))
  sm <- summary(m)
  cf <- sm$coefficients
  nms <- rownames(cf)
  wald <- tidy_fixed(cf[, 1], cf[, 2], cf[, 1] - 1.96 * cf[, 2],
                     cf[, 1] + 1.96 * cf[, 2], nms)

  sig <- sqrt(unlist(VarCorr(m))[["body"]])
  singular <- isSingular(m, tol = 1e-4)

  # Profile-likelihood CIs. `which` indexes the FULL parameter vector --
  # theta (the random-effect SDs) first, then the fixed effects -- so a mixed
  # character `which` silently profiles theta only. Index numerically, and
  # relax devtol: at a singular fit the profiler finds deviance a shade below
  # the reported optimum and aborts at the 1e-9 default.
  ntheta <- length(getME(m, "theta"))
  idx <- c(seq_len(ntheta), ntheta + which(nms %in% WANT))
  prof <- tryCatch({
    pp <- profile(m, which = idx, signames = FALSE, devtol = 1e-6)
    as.data.frame(confint(pp, level = 0.95))
  }, error = function(e) { say("  profile failed:", conditionMessage(e)); NULL })

  prof_ci <- NULL
  if (!is.null(prof)) {
    rn <- rownames(prof)
    prof_ci <- data.frame(parameter = rn, prof_lo = prof[, 1],
                          prof_hi = prof[, 2], row.names = NULL)
  }

  # LRTs: role + interaction (df2), role alone (df1), interaction (df1)
  nointer <- update(m, as.formula(paste(". ~ . -", "role_x_renter")))
  norole  <- update(nointer, . ~ . - role_num)
  noboth  <- update(m, as.formula(". ~ . - role_num - role_x_renter"))
  a_both  <- anova(noboth, m)
  a_role  <- anova(norole, nointer)
  a_inter <- anova(nointer, m)
  lrt <- data.frame(
    test = c("role + role x renter dropped (df2)",
             "role dropped, no interaction either side (df1)",
             "role x renter dropped (df1)"),
    chisq = c(a_both$Chisq[2], a_role$Chisq[2], a_inter$Chisq[2]),
    df = c(a_both$Df[2], a_role$Df[2], a_inter$Df[2]),
    p = c(a_both$`Pr(>Chisq)`[2], a_role$`Pr(>Chisq)`[2],
          a_inter$`Pr(>Chisq)`[2]), row.names = NULL)

  list(model = m, wald = wald, prof = prof_ci, sigma_body = sig,
       singular = singular, lrt = lrt, loglik = as.numeric(logLik(m)),
       aic = AIC(m), conv = is.null(m@optinfo$conv$lme4$messages),
       conv_msg = paste(m@optinfo$conv$lme4$messages, collapse = "; "))
}

## --------------------------------------------- 2. glmmTMB beta-binomial
tmb_block <- function(d, tag) {
  d$ny <- d$N - d$y
  f0 <- as.formula(paste("cbind(y, ny) ~", FIXED))
  f1 <- as.formula(paste("cbind(y, ny) ~", FIXED, "+ (1 | body)"))

  out <- list()
  for (nm in c("bb_fixed", "bb_ri", "binom_ri")) {
    fam <- if (nm == "binom_ri") binomial() else betabinomial()
    frm <- if (nm == "bb_fixed") f0 else f1
    m <- tryCatch(glmmTMB(frm, data = d, family = fam),
                  error = function(e) { say("  ", nm, "failed:",
                                            conditionMessage(e)); NULL })
    if (is.null(m)) next
    cf <- summary(m)$coefficients$cond
    tab <- tidy_fixed(cf[, 1], cf[, 2], cf[, 1] - 1.96 * cf[, 2],
                      cf[, 1] + 1.96 * cf[, 2], rownames(cf))
    sig <- NA_real_
    if (nm != "bb_fixed") {
      vc <- VarCorr(m)$cond
      if (length(vc)) sig <- sqrt(as.numeric(vc$body))
    }
    phi <- if (nm == "binom_ri") NA_real_ else as.numeric(sigma(m))
    out[[nm]] <- list(model = m, table = tab, sigma_body = sig, phi = phi,
                      rho = if (is.na(phi)) NA_real_ else 1 / (1 + phi),
                      loglik = as.numeric(logLik(m)), aic = AIC(m))
  }
  out
}

## ------------------------------------------------- 4. DHARMa + icc
diag_block <- function(m, label, nsim = 1000) {
  s <- tryCatch(simulateResiduals(m, n = nsim, refit = FALSE, seed = 20261019),
                error = function(e) NULL)
  if (is.null(s)) return(data.frame(fit = label, ok = FALSE, disp_stat = NA_real_, disp_p = NA_real_, ks_stat = NA_real_, ks_p = NA_real_, outlier_p = NA_real_, icc_adj = NA_real_, icc_cond = NA_real_))
  dt <- tryCatch(testDispersion(s, plot = FALSE), error = function(e) NULL)
  ks <- tryCatch(testUniformity(s, plot = FALSE), error = function(e) NULL)
  ol <- tryCatch(testOutliers(s, plot = FALSE, type = "bootstrap"),
                 error = function(e) NULL)
  # icc() returns a bare NA (not a list) when a variance component is zero
  ic <- tryCatch(suppressWarnings(performance::icc(m)), error = function(e) NULL)
  if (!is.list(ic)) ic <- NULL
  data.frame(fit = label, ok = TRUE,
             disp_stat = num1(dt$statistic), disp_p = num1(dt$p.value),
             ks_stat = num1(ks$statistic), ks_p = num1(ks$p.value),
             outlier_p = num1(ol$p.value),
             icc_adj = num1(ic$ICC_adjusted),
             icc_cond = num1(ic$ICC_unadjusted), row.names = NULL)
}
DIAG <- NULL   # every diag_block result, accumulated for one flat CSV
keep_diag <- function(d) { DIAG <<- rbind(DIAG, d); d }

## ----------------------------------------------- run the row-level fits
for (tag in c("primary", "secondary")) {
  d <- read.csv(file.path(DATA, sprintf("rows_%s.csv", tag)))
  d$body <- factor(d$body)
  say(sprintf("[%s] %d rows, %d bodies, %d trials, %d NIMBY", tag, nrow(d),
              nlevels(d$body), sum(d$N), sum(d$y)))

  g <- glmer_block(d, tag)
  say(sprintf("[%s] glmer role %+.4f  sigma_body %.4f  singular=%s  ll %.3f",
              tag, g$wald$est[g$wald$parameter == "role_num"], g$sigma_body,
              g$singular, g$loglik))
  t <- tmb_block(d, tag)
  for (nm in names(t))
    say(sprintf("[%s] glmmTMB %-9s role %+.4f  phi %s  sigma_body %s  AIC %.2f",
                tag, nm, t[[nm]]$table$est[t[[nm]]$table$parameter == "role_num"],
                format(round(t[[nm]]$phi, 3)), format(round(t[[nm]]$sigma_body, 4)),
                t[[nm]]$aic))

  dg <- list(glmer = keep_diag(diag_block(g$model, sprintf("%s glmer", tag))))
  for (nm in names(t))
    dg[[nm]] <- keep_diag(diag_block(t[[nm]]$model, sprintf("%s %s", tag, nm)))

  res[[tag]] <- list(
    n = list(rows = nrow(d), bodies = nlevels(d$body), trials = sum(d$N),
             nimby = sum(d$y)),
    glmer = list(wald = g$wald, profile = g$prof, sigma_body = g$sigma_body,
                 singular = g$singular, lrt = g$lrt, loglik = g$loglik,
                 aic = g$aic, converged = g$conv, conv_msg = g$conv_msg),
    glmmTMB = lapply(t, function(x) x[c("table", "sigma_body", "phi", "rho",
                                        "loglik", "aic")]),
    diagnostics = dg)
  write.csv(g$wald, file.path(OUT, sprintf("glmer_%s_wald.csv", tag)),
            row.names = FALSE)
  if (!is.null(g$prof))
    write.csv(g$prof, file.path(OUT, sprintf("glmer_%s_profile.csv", tag)),
              row.names = FALSE)
  assign(paste0("fit_", tag), g$model)
  assign(paste0("d_", tag), d)
}

## -------------------------------------- 3. nested turn-level model
say("turn level")
tu <- read.csv(file.path(DATA, "turns.csv"))
for (k in c("body", "meeting_id", "episode_id", "route"))
  tu[[k]] <- factor(tu[[k]])
tu$role_x_renter <- tu$role_code * tu$z_renter
tu$route <- relevel(tu$route, ref = "classifier")
say(sprintf("turns %d, NIMBY %d, episodes %d, meetings %d, bodies %d",
            nrow(tu), sum(tu$y), nlevels(tu$episode_id),
            nlevels(tu$meeting_id), nlevels(tu$body)))

turn_specs <- list(
  "body"                   = "(1 | body)",
  "body/meeting"           = "(1 | body) + (1 | meeting_id)",
  "body/meeting/episode"   = "(1 | body) + (1 | meeting_id) + (1 | episode_id)")
turn_out <- list()
for (nm in names(turn_specs)) {
  f <- as.formula(paste("y ~ role_code + z_renter + role_x_renter + route +",
                        turn_specs[[nm]]))
  m <- tryCatch(glmer(f, data = tu, family = binomial, nAGQ = 1,
                      control = glmerControl(optimizer = "bobyqa",
                                             optCtrl = list(maxfun = 2e5))),
                error = function(e) { say("  ", nm, "failed:",
                                          conditionMessage(e)); NULL })
  if (is.null(m)) next
  cf <- summary(m)$coefficients
  vc <- as.data.frame(VarCorr(m))[, c("grp", "vcov", "sdcor")]
  # profile the variance components and the role effect only -- profiling all
  # eleven fixed effects on 1,892 turns with three random intercepts is slow
  # and the route dummies are nuisance
  nth <- length(getME(m, "theta"))
  pr <- tryCatch({
    pp <- profile(m, which = c(seq_len(nth),
                               nth + which(names(fixef(m)) == "role_code")),
                  signames = FALSE, devtol = 1e-6)
    as.data.frame(confint(pp, level = 0.95))
  }, error = function(e) { say("  turn profile failed:",
                               conditionMessage(e)); NULL })
  if (!is.null(pr))
    pr <- data.frame(parameter = rownames(pr), prof_lo = pr[, 1],
                     prof_hi = pr[, 2], row.names = NULL)
  dg <- keep_diag(diag_block(m, sprintf("turn %s", nm)))
  turn_out[[nm]] <- list(
    fixed = tidy_fixed(cf[, 1], cf[, 2], cf[, 1] - 1.96 * cf[, 2],
                       cf[, 1] + 1.96 * cf[, 2], rownames(cf)),
    varcomp = vc, profile = pr, loglik = as.numeric(logLik(m)), aic = AIC(m),
    singular = isSingular(m, tol = 1e-4), diagnostics = dg)
  say(sprintf("  %-22s role %+.4f se %.4f  AIC %.2f  sd: %s", nm,
              cf["role_code", 1], cf["role_code", 2], AIC(m),
              paste(sprintf("%s=%.4f", vc$grp, vc$sdcor), collapse = " ")))
  if (nm == "body/meeting/episode") {
    write.csv(vc, file.path(OUT, "turn_varcomp.csv"), row.names = FALSE)
    turn_full <- m
  }
}
# iid glm for reference, same fixed effects
m_iid <- glm(y ~ role_code + z_renter + role_x_renter + route, data = tu,
             family = binomial)
ci <- suppressMessages(confint(m_iid))
cf <- summary(m_iid)$coefficients
turn_out[["iid glm (reference)"]] <- list(
  fixed = tidy_fixed(cf[, 1], cf[, 2], ci[, 1], ci[, 2], rownames(cf)),
  varcomp = NULL, profile = NULL, loglik = as.numeric(logLik(m_iid)),
  aic = AIC(m_iid), singular = FALSE)
res$turn_level <- list(
  n = list(turns = nrow(tu), nimby = sum(tu$y), episodes = nlevels(tu$episode_id),
           meetings = nlevels(tu$meeting_id), bodies = nlevels(tu$body)),
  fits = turn_out)

## ----------------------------------- 5. LOBO + grouped 5-fold (primary)
say("leave-one-body-out (glmer, primary)")
d <- d_primary
bodies <- levels(d$body)
lobo <- data.frame()
for (b in bodies) {
  sub <- droplevels(d[d$body != b, ])
  # a route dummy can go constant once a body leaves; drop those columns
  rr <- ROUTES[sapply(ROUTES, function(k) length(unique(sub[[k]])) > 1)]
  fx <- paste(c("role_num", "renter_share", "role_x_renter", COV, rr),
              collapse = " + ")
  m <- tryCatch(glmer(as.formula(paste("cbind(y, N - y) ~", fx, "+ (1 | body)")),
                      data = sub, family = binomial, nAGQ = 25,
                      control = glmerControl(optimizer = "bobyqa",
                                             optCtrl = list(maxfun = 2e5))),
                error = function(e) NULL)
  if (is.null(m)) { say("  drop", b, "failed"); next }
  cf <- summary(m)$coefficients
  lobo <- rbind(lobo, data.frame(
    dropped = b, rows = nrow(sub),
    role = cf["role_num", 1], role_se = cf["role_num", 2],
    renter = cf["renter_share", 1], renter_se = cf["renter_share", 2],
    inter = cf["role_x_renter", 1],
    sigma_body = sqrt(unlist(VarCorr(m))[["body"]]),
    routes_dropped = paste(setdiff(ROUTES, rr), collapse = ",")))
}
write.csv(lobo, file.path(OUT, "lobo_glmer_primary.csv"), row.names = FALSE)
res$lobo <- lobo
say(sprintf("  role range %+.3f .. %+.3f ; renter %+.3f .. %+.3f",
            min(lobo$role), max(lobo$role), min(lobo$renter), max(lobo$renter)))

say("grouped 5-fold by body, binomial log-loss per trial")
set.seed(20261019)
fold_of <- setNames(sample(rep_len(1:5, length(bodies))), bodies)
sets <- list(
  "intercept only"              = "1",
  "route only"                  = paste(ROUTES, collapse = " + "),
  "+ role"                      = paste(c("role_num", ROUTES), collapse = " + "),
  "+ role + renter + role:renter" = paste(c("role_num", "renter_share",
                                            "role_x_renter", ROUTES),
                                          collapse = " + "),
  "full (8 covariates)"         = FIXED)
kf <- data.frame()
for (nm in names(sets)) {
  tot_ll <- 0; tot_n <- 0
  for (k in 1:5) {
    tr <- d[fold_of[as.character(d$body)] != k, ]
    te <- d[fold_of[as.character(d$body)] == k, ]
    m <- tryCatch(glm(as.formula(paste("cbind(y, N - y) ~", sets[[nm]])),
                      data = tr, family = binomial), error = function(e) NULL)
    if (is.null(m)) next
    p <- tryCatch(predict(m, newdata = te, type = "response"),
                  error = function(e) rep(mean(tr$y / tr$N), nrow(te)))
    p[is.na(p)] <- sum(tr$y) / sum(tr$N)
    p <- pmin(pmax(p, 1e-9), 1 - 1e-9)
    tot_ll <- tot_ll + sum(te$y * log(p) + (te$N - te$y) * log(1 - p))
    tot_n <- tot_n + sum(te$N)
  }
  kf <- rbind(kf, data.frame(model = nm, logloss = -tot_ll / tot_n,
                             held_out_logdens = tot_ll, trials = tot_n))
  say(sprintf("  %-32s log-loss %.4f  (ll %.1f)", nm, -tot_ll / tot_n, tot_ll))
}
write.csv(kf, file.path(OUT, "kfold_by_body.csv"), row.names = FALSE)
res$kfold <- kf

write.csv(DIAG, file.path(OUT, "dharma_diagnostics.csv"), row.names = FALSE)
res$diagnostics <- DIAG
say("DHARMa / icc table:")
print(DIAG)

## ----------------------------------------------------- session info
si <- sessionInfo()
res$session <- list(
  r_version = si$R.version$version.string, platform = si$platform,
  running = si$running, blas = si$BLAS,
  packages = sapply(c("lme4", "glmmTMB", "DHARMa", "performance", "emmeans",
                      "Matrix", "TMB", "sandwich", "clubSandwich"),
                    function(p) tryCatch(as.character(packageVersion(p)),
                                         error = function(e) NA_character_)))
writeLines(capture.output(print(si)), file.path(OUT, "sessionInfo.txt"))
write(toJSON(res, auto_unbox = TRUE, digits = 8, na = "null", null = "null"),
      file.path(OUT, "r_results.json"))
say("-> ", OUT)
