# install_pkgs.R -- CRAN packages for the run-6 R fits.
# Homebrew's R is a source build with no "well-defined binary type", so
# install.packages must use type="source" (gfortran arrives with brew's gcc).
options(repos = c(CRAN = "https://cloud.r-project.org"),
        Ncpus = max(1, parallel::detectCores() - 1))
pkgs <- c("jsonlite", "lme4", "glmmTMB", "performance", "DHARMa", "emmeans",
          "sandwich", "clubSandwich")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) {
    cat("=== installing", p, "\n"); flush.console()
    try(install.packages(p, type = "source"))
  }
}
cat("=== versions\n")
for (p in pkgs) {
  v <- tryCatch(as.character(packageVersion(p)), error = function(e) "MISSING")
  cat(sprintf("%-14s %s\n", p, v))
}
