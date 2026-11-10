# Runs Hector (through Matilda's set_params) for a table of parameter sets.
# Usage: Rscript hector_worker.R <scenario> <params.csv> <out.csv> [hist]
#   <scenario>   one of ssp119 ssp126 ssp245 ssp370 ssp585 (Hector's bundled INI files)
#   <params.csv> header BETA,Q10_RH,NPP_FLUX0,AERO_SCALE,DIFFUSIVITY,ECS; one row per run
#   <out.csv>    columns warming (degC, mean gmst 2081-2100 minus mean gmst 1850-1900),
#                co2_2100 (ppmv), gmst_1850_1900 (degC, mean raw Hector gmst 1850-1900), and with
#                "hist" also gmst_1950..gmst_2023 (degC, raw Hector gmst)
# A run that fails writes NaN. Called by hector_driver.py.
suppressPackageStartupMessages({
  library(hector)
  library(matilda)
})
args <- commandArgs(trailingOnly = TRUE)
scen <- args[1]
pars <- read.csv(args[2])
hist <- length(args) >= 4 && args[4] == "hist"
ini <- system.file(file.path("input", paste0("hector_", scen, ".ini")), package = "hector")
core <- newcore(ini, name = scen, suppresslogging = TRUE)
hyears <- 1950:2023
out <- matrix(NaN, nrow(pars), 3 + if (hist) length(hyears) else 0)
for (i in seq_len(nrow(pars))) {
  p <- unlist(pars[i, ])
  res <- tryCatch({
    suppressMessages(set_params(core, p))
    reset(core, date = 0)
    run(core, 2100)
    g <- fetchvars(core, 1850:2100, GMST())
    co2 <- fetchvars(core, 2100, CONCENTRATIONS_CO2())$value
    ref <- mean(g$value[g$year <= 1900])
    w <- mean(g$value[g$year >= 2081]) - ref
    v <- c(w, co2, ref)
    if (hist) v <- c(v, g$value[match(hyears, g$year)])
    v
  }, error = function(e) NULL)
  if (!is.null(res)) out[i, ] <- res
}
cn <- c("warming", "co2_2100", "gmst_1850_1900")
if (hist) cn <- c(cn, paste0("gmst_", hyears))
colnames(out) <- cn
write.csv(format(as.data.frame(out), digits = 17), args[3], row.names = FALSE, quote = FALSE)
