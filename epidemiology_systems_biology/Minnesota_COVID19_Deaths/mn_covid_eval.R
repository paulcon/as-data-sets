# mn_covid_eval.R -- evaluate the Minnesota COVID-19 model of Wycoff, Binois and Wild (2021)
# at a batch of input points (part of the as-data-sets data set Minnesota_COVID19_Deaths; MIT).
#
# Usage: Rscript mn_covid_eval.R CODE_DIR INPUT_CSV OUTPUT_CSV [MAX_NEW]
#   CODE_DIR    the unpacked "code" folder of code.zip from the article's supplement
#               (figshare, doi:10.6084/m9.figshare.13611284); it holds helper/covid_allinone.R
#               and data/*.rda. That code is not part of this repository.
#   INPUT_CSV   one point per line, 8 comma-separated values: the arguments x[1..8] of
#               covid_sim(x) (fractions; no header).
#   OUTPUT_CSV  appended to: one line "row,deaths" per point, written as soon as it is done,
#               so an interrupted batch can be resumed. Rows already in OUTPUT_CSV are skipped.
#   MAX_NEW     optional: stop after this many new rows (for runs in chunks).
#
# The model itself is not modified: this script only calls covid_sim() from covid_allinone.R.
args <- commandArgs(trailingOnly = TRUE)
if (!(length(args) %in% c(3, 4))) stop("usage: Rscript mn_covid_eval.R CODE_DIR INPUT_CSV OUTPUT_CSV [MAX_NEW]")
max_new <- if (length(args) == 4) as.integer(args[4]) else .Machine$integer.max
code_dir <- normalizePath(args[1])
inp <- normalizePath(args[2])
outp <- args[3]
X <- as.matrix(read.csv(inp, header = FALSE))
if (ncol(X) != 8) stop("INPUT_CSV must have 8 columns")
done <- integer(0)
if (file.exists(outp) && file.info(outp)$size > 0) {
  done <- read.csv(outp, header = FALSE)[, 1]
}
owd <- setwd(code_dir)            # covid_allinone.R loads ./data/*.rda
suppressPackageStartupMessages(source("helper/covid_allinone.R"))
setwd(owd)
con <- file(outp, open = "a")
n_new <- 0L
for (i in seq_len(nrow(X))) {
  if (i %in% done) next
  if (n_new >= max_new) break
  n_new <- n_new + 1L
  y <- covid_sim(as.numeric(X[i, ]))
  writeLines(sprintf("%d,%.17g", i, y), con)
  flush(con)
}
close(con)
