#!/usr/bin/env Rscript
# Run one chunk of design rows through virsim and write one CSV per chunk.
#
# Usage:
#   Rscript R/run_chunk.R <design_S.csv> <first_row> <last_row> <out_dir> [timeout_s]
#
# Output (in <out_dir>):
#   S_rows_<first>-<last>.csv      QoIs, status and run time per row (final)
#   S_rows_<first>-<last>_ts.csv   daily IC prevalence per row, days 0..550 (final)
# While running, rows are appended to *.partial files, one line per finished
# row; a restarted chunk skips rows already there with status 0. When all rows
# are done the partial files are sorted by row and renamed to the final names.
# A chunk whose final file exists is skipped (idempotent). Set
# VIRSIM_RETRY_ERRORS=1 to rerun rows of a finished chunk whose status is not 0.
#
# status: 0 = ok, 1 = R error (message in <out_dir>/errors.log),
#         2 = time limit reached (timeout_s, default 3600 s of elapsed time).

suppressPackageStartupMessages({
  library(data.table)
  library(virsim)
})
data.table::setDTthreads(1L)
Sys.setenv(OMP_NUM_THREADS = "1")

get_script_dir <- function() {
  a <- commandArgs(FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f) == 1) dirname(normalizePath(f)) else getwd()
}
source(file.path(get_script_dir(), "strategies.R"))

args <- commandArgs(TRUE)
if (length(args) < 4) {
  stop("usage: Rscript run_chunk.R <design.csv> <first_row> <last_row> <out_dir> [timeout_s]")
}
design_file <- args[1]
first <- as.integer(args[2]); last <- as.integer(args[3])
out_dir <- args[4]
timeout_s <- if (length(args) >= 5) as.numeric(args[5]) else 3600
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# --- read the design (two '#' header lines) ---------------------------------
hdr <- readLines(design_file, n = 2)
strategy <- sub(".*strategy=([A-Z]+).*", "\\1", hdr[1])
if (!strategy %in% names(STRATEGY_INPUTS)) stop("no strategy= in ", design_file)
cols <- trimws(sub("\\s*\\[.*\\]$", "", strsplit(sub("^#\\s*", "", hdr[2]), ",")[[1]]))
D <- read.csv(design_file, header = FALSE, comment.char = "#", col.names = cols)
inputs <- c(STRATEGY_INPUTS[[strategy]], COMMON_INPUTS)
stopifnot(all(inputs %in% cols))
rows <- D[D$row >= first & D$row <= last, , drop = FALSE]
if (nrow(rows) == 0) stop("no design rows in ", first, "..", last)

tag <- sprintf("%s_rows_%04d-%04d", strategy, first, last)
f_final <- file.path(out_dir, paste0(tag, ".csv"))
f_ts <- file.path(out_dir, paste0(tag, "_ts.csv"))
f_part <- paste0(f_final, ".partial")
f_tspart <- paste0(f_ts, ".partial")
f_lock <- file.path(out_dir, paste0(tag, ".lock"))
f_err <- file.path(out_dir, "errors.log")

out_cols <- c("row", "status", "runtime_s", "IC_prev_avg_max", "IC_excess_days",
              "seed", inputs, APPLIED_COLS[[strategy]])
out_units <- c("-", "-", "s", "per million", "patient-days per million", "-",
               rep("as design", length(inputs)), rep("d", length(APPLIED_COLS[[strategy]])))
N_DAYS <- RUNTIME + 1L

read_body <- function(f) {
  if (!file.exists(f)) return(NULL)
  L <- readLines(f)
  L <- L[!grepl("^#", L) & nzchar(L)]
  if (length(L) == 0) return(NULL)
  L
}

# --- idempotence -------------------------------------------------------------
retry <- identical(Sys.getenv("VIRSIM_RETRY_ERRORS"), "1")
if (file.exists(f_final)) {
  body <- read_body(f_final)
  st <- as.integer(sapply(strsplit(body, ","), `[`, 2))
  if (!retry || all(st == 0)) {
    message(tag, ": already finished, skipping")
    quit(status = 0)
  }
  # move finished ok rows back to the partial files and rerun the others
  ok_rows <- as.integer(sapply(strsplit(body, ","), `[`, 1))[st == 0]
  writeLines(body[st == 0], f_part)
  tsb <- read_body(f_ts)
  ts_rows <- as.integer(sapply(strsplit(tsb, ","), `[`, 1))
  writeLines(tsb[ts_rows %in% ok_rows], f_tspart)
  invisible(file.remove(f_final, f_ts))
}

# --- lock (one process per chunk) ------------------------------------------
if (!dir.create(f_lock, showWarnings = FALSE)) {
  pidf <- file.path(f_lock, "pid")
  pid <- if (file.exists(pidf)) suppressWarnings(as.integer(readLines(pidf, 1))) else NA
  alive <- !is.na(pid) && dir.exists(file.path("/proc", pid))
  if (alive) {
    message(tag, ": locked by running process ", pid, ", skipping")
    quit(status = 0)
  }
  message(tag, ": removing stale lock")
}
writeLines(as.character(Sys.getpid()), file.path(f_lock, "pid"))

# --- rows already done ---------------------------------------------------------
done <- integer(0)
pb <- read_body(f_part)
if (!is.null(pb)) {
  sp <- strsplit(pb, ",")
  okline <- sapply(sp, length) == length(out_cols) & sapply(sp, `[`, 2) == "0"
  pb <- pb[okline]                                # drop failed or cut-off lines
  done <- as.integer(sapply(strsplit(pb, ","), `[`, 1))
  tsb <- read_body(f_tspart)
  if (!is.null(tsb)) {
    tsp <- strsplit(tsb, ",")
    tsb <- tsb[sapply(tsp, length) == N_DAYS + 1 &
                 as.integer(sapply(tsp, `[`, 1)) %in% done]
    done <- intersect(done, as.integer(sapply(strsplit(tsb, ","), `[`, 1)))
  } else {
    tsb <- character(0); done <- integer(0)
  }
  pb <- pb[as.integer(sapply(strsplit(pb, ","), `[`, 1)) %in% done]
  tsb <- tsb[!duplicated(as.integer(sapply(strsplit(tsb, ","), `[`, 1)))]
  writeLines(pb, f_part)
  writeLines(tsb, f_tspart)
}

fmt <- function(v) {
  # NA is written as nan so that numpy.loadtxt reads every file
  ifelse(is.na(v), "nan", if (is.integer(v)) as.character(v) else sprintf("%.17g", v))
}

# --- run ---------------------------------------------------------------------------
for (k in seq_len(nrow(rows))) {
  r <- rows[k, ]
  if (r$row %in% done) next
  x <- as.list(r[inputs])
  seed <- as.integer(r$seed)
  t0 <- proc.time()[["elapsed"]]
  status <- 0L; q <- c(NA_real_, NA_real_); ts <- rep(NA_real_, N_DAYS)
  applied <- setNames(rep(list(NA_integer_), length(APPLIED_COLS[[strategy]])),
                      APPLIED_COLS[[strategy]])
  res <- tryCatch({
    setTimeLimit(elapsed = timeout_s, transient = TRUE)
    run_one(strategy, x, seed)
  }, error = function(e) e)
  setTimeLimit(elapsed = Inf)
  if (inherits(res, "error")) {
    msg <- conditionMessage(res)
    status <- if (grepl("elapsed time limit", msg)) 2L else 1L
    cat(sprintf("%s %s row %d seed %d status %d: %s\n", format(Sys.time()), strategy,
                r$row, seed, status, gsub("\n", " ", msg)), file = f_err, append = TRUE)
  } else {
    q <- res$qoi
    if (length(res$ts) != N_DAYS || any(res$time != 0:RUNTIME)) {
      status <- 1L
      cat(sprintf("%s %s row %d: unexpected output length %d\n", format(Sys.time()),
                  strategy, r$row, length(res$ts)), file = f_err, append = TRUE)
    } else {
      ts <- res$ts
    }
    for (nm in names(res$applied)) applied[[nm]] <- as.integer(res$applied[[nm]])
  }
  el <- proc.time()[["elapsed"]] - t0
  line <- paste(c(as.character(r$row), as.character(status), sprintf("%.2f", el),
                  fmt(q[1]), fmt(q[2]), as.character(seed),
                  sapply(x, function(v) sprintf("%.17g", v)),
                  sapply(applied, fmt)), collapse = ",")
  # time series first, then the QoI line: a QoI line implies its series exists
  if (status == 0L) {
    cat(paste(c(as.character(r$row), sprintf("%.10g", ts)), collapse = ","), "\n",
        sep = "", file = f_tspart, append = TRUE)
  }
  cat(line, "\n", sep = "", file = f_part, append = TRUE)
  message(sprintf("%s row %d status %d %.1f s", tag, r$row, status, el))
}

# --- finalise: sort, add headers, rename atomically ----------------------------
pb <- read_body(f_part)
prow <- as.integer(sapply(strsplit(pb, ","), `[`, 1))
# keep the last line per row (a rerun error row follows nothing; ok rows are unique)
keep <- !duplicated(prow, fromLast = TRUE)
pb <- pb[keep][order(prow[keep])]
if (!setequal(as.integer(sapply(strsplit(pb, ","), `[`, 1)), rows$row)) {
  unlink(f_lock, recursive = TRUE)
  stop(tag, ": rows missing after run; rerun this chunk")
}
h1 <- sprintf(paste0("# virsim v1.0.5 runs, strategy=%s, design %s rows %d-%d; ",
                     "status 0 ok, 1 error, 2 time limit; R %s, %s"),
              strategy, basename(design_file), first, last,
              paste(R.version$major, R.version$minor, sep = "."), format(Sys.time(), "%Y-%m-%d"))
h2 <- paste0("# ", paste(sprintf("%s [%s]", out_cols, out_units), collapse = ", "))
tmp <- paste0(f_final, ".tmp")
writeLines(c(h1, h2, pb), tmp)
tsb <- read_body(f_tspart)
if (is.null(tsb)) tsb <- character(0)
trow <- as.integer(sapply(strsplit(tsb, ","), `[`, 1))
tsb <- tsb[!duplicated(trow)][order(trow[!duplicated(trow)])]
tmpts <- paste0(f_ts, ".tmp")
writeLines(c(sprintf("# daily IC prevalence (per million) for strategy=%s rows %d-%d (ok rows only)",
                     strategy, first, last),
             paste0("# row [-], ", paste(sprintf("day_%d [per million]", 0:RUNTIME), collapse = ", ")),
             tsb), tmpts)
invisible(file.rename(tmpts, f_ts))
invisible(file.rename(tmp, f_final))
invisible(file.remove(f_part, f_tspart))
unlink(f_lock, recursive = TRUE)
message(tag, ": finished")
