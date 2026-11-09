#!/usr/bin/env bash
# Install R, the R packages virsim needs, and virsim v1.0.5 (built from its
# source tarball) on Ubuntu 24.04 LTS (x86-64):
#
#     bash install_ubuntu2404.sh /path/to/virsim-v1.0.5.tar.gz [INSTALL_DIR]
#
# All R packages come from the Ubuntu 24.04 archive at pinned versions (the
# same versions used for the timing runs), so no CRAN mirror is needed. virsim
# is installed into INSTALL_DIR/Rlib and the log is INSTALL_DIR/install.log.
# INSTALL_DIR (default: the current directory) must exist and must be outside
# this data set's directory. Each stage stops the script with a message if it
# fails. Needs sudo for apt; needs no AWS or other credentials.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DEST="${2:-$PWD}"
[ -d "$DEST" ] || { echo "INSTALL_DIR $DEST is not a directory: create it first" >&2; exit 1; }
KIT="$(cd "$DEST" && pwd -P)"
case "$KIT/" in
  "$HERE/"*) echo "INSTALL_DIR must be outside this data set's directory (not the directory or a folder in it): $KIT" >&2
             exit 1 ;;
esac
TARBALL="${1:-$KIT/virsim-v1.0.5.tar.gz}"
SHA256="e6c629b537adf3ae34fdeb3e9af00fba30aa9b470fdcf7bbaec315fb375edaa6"
LOG="$KIT/install.log"
: > "$LOG"

stop() { echo "STOP at stage $1: $2 (details in $LOG)" >&2; exit 1; }
stage() { echo "== stage $1: $2" | tee -a "$LOG"; }

# ---------------------------------------------------------------------------
stage 1 "check the operating system"
. /etc/os-release
[ "${ID:-}" = "ubuntu" ] && [ "${VERSION_ID:-}" = "24.04" ] \
  || stop 1 "this script is for Ubuntu 24.04; found ${PRETTY_NAME:-unknown}"
[ "$(uname -m)" = "x86_64" ] || stop 1 "expected x86_64, found $(uname -m) (use a c7a/c7i, not a Graviton c7g)"

# ---------------------------------------------------------------------------
stage 2 "check the virsim tarball (sha256)"
[ -f "$TARBALL" ] || stop 2 "tarball not found: $TARBALL"
TARBALL="$(realpath "$TARBALL")"
echo "$SHA256  $TARBALL" | sha256sum -c - >> "$LOG" 2>&1 \
  || stop 2 "sha256 of $TARBALL is not $SHA256"

# ---------------------------------------------------------------------------
stage 3 "install R 4.3.3 and pinned R packages from the Ubuntu archive"
PKGS=(
  r-base-core=4.3.3-2build2
  r-base-dev=4.3.3-2build2
  r-cran-data.table=1.14.10+dfsg-1
  r-cran-foreach=1.5.2-1
  r-cran-iterators=1.0.14-1
  r-cran-doparallel=1.0.17-1
  r-cran-matrixstats=1.2.0-1
  r-cran-rcpp=1.0.12-1
  r-cran-rcpparmadillo=0.12.8.1.0-1build1
  r-cran-usethis=2.2.2-1
  python3-numpy=1:1.26.4+ds-6ubuntu1
  build-essential gfortran liblapack-dev libblas-dev parallel time tmux
)
sudo apt-get update -q >> "$LOG" 2>&1 || stop 3 "apt-get update failed"
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends \
  "${PKGS[@]}" >> "$LOG" 2>&1 \
  || stop 3 "apt-get install failed (a pinned version may have left the archive; see log)"
R --version | head -1 | grep -q "4.3.3" || stop 3 "R is not 4.3.3"

# ---------------------------------------------------------------------------
stage 4 "build virsim 1.0.5 from the tarball into $KIT/Rlib"
# virsim's R/params.R calls usethis::use_data() at build time, hence r-cran-usethis.
mkdir -p "$KIT/Rlib"
BUILD_DIR="$(mktemp -d)"
( cd "$BUILD_DIR" && R CMD INSTALL --no-docs --no-html -l "$KIT/Rlib" "$TARBALL" ) >> "$LOG" 2>&1 \
  || stop 4 "R CMD INSTALL failed"
rm -rf "$BUILD_DIR"

# ---------------------------------------------------------------------------
stage 5 "check package versions and a short virsim run"
R_LIBS="$KIT/Rlib" Rscript - >> "$LOG" 2>&1 <<'EOF' || stop 5 "version check or short run failed"
want <- c(virsim = "1.0.5", data.table = "1.14.10", foreach = "1.5.2",
          doParallel = "1.0.17", matrixStats = "1.2.0", Rcpp = "1.0.12",
          RcppArmadillo = "0.12.8.1.0")
for (p in names(want)) {
  v <- as.character(packageVersion(p))
  cat(p, v, "\n")
  if (v != want[[p]]) stop(p, " is ", v, ", expected ", want[[p]])
}
suppressPackageStartupMessages(library(virsim))
set.seed(1)
s <- virsim(runtime = 30L, n_agent = 1e4, n_cluster = 10, n_supercluster = 2,
            verbose = FALSE, temp_storage = "RAM", output_quick = TRUE)
stopifnot(nrow(s$monitor) > 0)
cat("short run ok\n")
EOF
grep -E "^(virsim|data.table|Rcpp) " "$LOG" || true
echo "install finished: R_LIBS=$KIT/Rlib (run_all.sh sets this)"
