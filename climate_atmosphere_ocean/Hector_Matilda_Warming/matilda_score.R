# Weights of a Hector ensemble with Matilda's score_runs(criterion_gmst_obs(), score_bayesian),
# the scoring used in the case study of Brown et al. (2024), Sec. 3, p. 16.
# Usage: Rscript matilda_score.R <gmst.csv> <out.csv>
#   <gmst.csv> header gmst_1950..gmst_2023, one row per run (raw Hector gmst, degC)
#   <out.csv>  columns rmse (Matilda's RMSE_calc with sigma = sd(observed)), weight
# Called by generate_data.py.
suppressPackageStartupMessages({
  library(hector)
  library(matilda)
})
args <- commandArgs(trailingOnly = TRUE)
g <- as.matrix(read.csv(args[1]))
years <- as.integer(sub("gmst_", "", colnames(g)))
n <- nrow(g)
x <- data.frame(
  scenario = "ssp245",
  year = rep(years, times = n),
  variable = GMST(),
  value = as.vector(t(g)),
  units = "degC",
  run_number = rep(seq_len(n), each = length(years))
)
crit <- criterion_gmst_obs()
w <- score_runs(x, crit, score_bayesian)
obs <- crit$obs_values[match(years, crit$years)]
rmse <- apply(g, 1, function(v) RMSE_calc(v, obs, sigma = sd(obs)))
write.csv(format(data.frame(rmse = rmse, weight = w$weights), digits = 17), args[2],
          row.names = FALSE, quote = FALSE)
