# Settings of the four exit strategies (FC, CT, IL, PO) for virsim v1.0.5.
#
# Written for this run kit from Gugole et al. (2021), PLOS Comput Biol 17(9):
# e1009355 (Methods, pp. 3-10; Table 1, p. 4), with the fixed settings that the
# paper's study scripts pass to virsim (github.com/FGugole/UQ_covid19, commit
# dc6715f, files EasyVVUQ/*_UQ_bio.r). Only parameter values are taken from those
# scripts; this file is new code. virsim itself is an outside package and is
# loaded, never copied.
#
# Each build_<S>() returns the list of arguments for virsim::virsim() for one
# design row `x` (a named list of physical input values), plus `applied`, the
# derived values that the strategy actually used (integer days, etc.).

IC_CAPACITY <- 109      # IC beds per million (p. 6)
AVG_WINDOW <- 30L       # moving-average window in days (p. 6)
RUNTIME <- 550L         # days simulated (p. 5)

base_params <- function(x) {
  # Fixed settings common to all strategies (population of 1 million, p. 4).
  avg_contact_rate <- x$R0 / x$avg_duration_infectiousness   # p. 9
  shp <- x$shape_exposed_time
  list(
    runtime = RUNTIME,
    n_agent = 1e6,
    n_cluster = 1e3,
    n_supercluster = 20,
    cluster_size_sd = 0.95,
    supercluster_size_sd = 0,
    efoi = 0,
    infection_init = 50,
    seed_supercluster = c(rep(9, 2), rep(2, 6), rep(1, 6), rep(0, 6)),
    inc_cum_cond = 9500,
    contact_rate = avg_contact_rate,
    contact_shape = 3.4,
    contact_assort = 0.26,
    population_mixing = 0.05,
    supercluster_mixing = 0.05,
    # Weibull latent period with mean 4.6 days and shape from the design
    exposed_time = list(name = "weibull", shape = shp,
                        scale = exp(log(4.6) - lgamma(1 + 1 / shp))),
    # exponential (Weibull shape 1) infectious period with the design mean
    infected_time = list(name = "weibull", shape = 1,
                         scale = exp(log(x$avg_duration_infectiousness) -
                                       lgamma(2))),
    # Table 1 samples x = intervention_effect_var^-1; the model takes 1/x (p. 9)
    intervention_effect_var = 1 / x$intervention_effect_var_inv,
    temp_storage = "RAM",
    aggregate = "population",
    output_quick = TRUE,
    output_steps = 1L,
    verbose = FALSE
  )
}

# Lockdown preceding every strategy (pp. 4-5): days 0, 10, 17, 70, 100.
PRE_T <- cumsum(c(0, 10, 7, 53, 30))
PRE_EFFECT <- c(1, 0.3, 0.15, 0.25)

zeros <- function(n) rep(0, n)

build_FC <- function(x) {
  p <- base_params(x)
  t <- PRE_T
  p$intervention_t <- t
  p$intervention_effect <- c(PRE_EFFECT, x$intervention_effect)
  p$intervention_uptake <- rep(x$uptake, length(t))
  p$trace_prob_E <- zeros(length(t))
  p$trace_rate_I <- zeros(length(t))
  p$trace_contact_reduction <- zeros(length(t))
  p$efoi <- 50 / 365 / p$n_agent
  list(args = p, applied = list())
}

build_CT <- function(x) {
  p <- base_params(x)
  t <- PRE_T
  p$intervention_t <- t
  p$intervention_effect <- c(PRE_EFFECT, 1)
  p$intervention_uptake <- rep(1, length(t))
  p$trace_prob_E <- c(zeros(4), x$trace_prob_E)
  p$trace_rate_I <- c(zeros(4), x$trace_rate_I)
  p$trace_contact_reduction <- c(zeros(4), x$trace_contact_reduction)
  p$efoi <- 50 / 365 / p$n_agent
  list(args = p, applied = list())
}

build_IL <- function(x) {
  p <- base_params(x)
  # the study scripts truncate both lengths to whole days (as.integer)
  lock <- as.integer(x$lock_length)
  lift <- as.integer(x$lift_length)
  t <- cumsum(c(0, 10, 7, 53, 30, 25, rep(c(lock, lift), 10)))
  p$intervention_t <- t
  p$intervention_effect <- c(PRE_EFFECT, rep(c(1, x$lock_effect), 11))
  p$intervention_uptake <- rep(x$uptake, length(t))
  p$trace_prob_E <- zeros(length(t))
  p$trace_rate_I <- zeros(length(t))
  p$trace_contact_reduction <- zeros(length(t))
  list(args = p, applied = list(lock_days = lock, lift_days = lift))
}

build_PO <- function(x) {
  p <- base_params(x)
  interval <- as.integer(x$intervention_lift_interval)   # truncated, as in the scripts
  sched <- virsim::gen_phased_lift(
    intervention_start = 15,
    init_lockdown_dur = 60,
    lockdown_effect = 0.2,
    n_phases = 20,
    interventions_lift_interval = interval,
    extra_duration_first_interval = 0,
    start_reduce_interval = 20,
    repeat_reduction = 1,
    reduce_interval = 0,
    pl_intervention_effect_hi = x$pl_intervention_effect_hi,
    pl_intervention_effect_lo = 1,
    uptake = x$uptake,
    sc_isolation_effect = 0.5,
    lockdown_isol = FALSE)
  t <- sched$intervention_t
  p$intervention_t <- t
  p$intervention_uptake <- sched$intervention_uptake
  p$intervention_effect <- sched$intervention_effect
  p$sc_isolation_effect <- sched$sc_isolation_effect
  p$trace_prob_E <- zeros(length(t))
  p$trace_rate_I <- zeros(length(t))
  p$trace_contact_reduction <- zeros(length(t))
  list(args = p, applied = list(lift_interval_days = interval))
}

STRATEGY_INPUTS <- list(
  FC = c("intervention_effect", "uptake"),
  CT = c("trace_rate_I", "trace_prob_E", "trace_contact_reduction"),
  IL = c("uptake", "lock_length", "lift_length", "lock_effect"),
  PO = c("uptake", "intervention_lift_interval", "pl_intervention_effect_hi"))
COMMON_INPUTS <- c("avg_duration_infectiousness", "R0",
                   "intervention_effect_var_inv", "shape_exposed_time")
APPLIED_COLS <- list(FC = character(0), CT = character(0),
                     IL = c("lock_days", "lift_days"),
                     PO = c("lift_interval_days"))

# Centred moving average of even order k: weights (1/2, 1, ..., 1, 1/2)/k over
# k+1 days, i.e. forecast::ma(x, order = k, centre = TRUE), which the study
# scripts use; days without a full window are dropped.
centred_ma <- function(v, k = AVG_WINDOW) {
  w <- if (k %% 2 == 0) c(0.5, rep(1, k - 1), 0.5) / k else rep(1, k) / k
  as.numeric(stats::filter(v, w, sides = 2))
}

# One simulation: returns list(qoi = c(IC_prev_avg_max, IC_excess_days), ts = IC_prev)
run_one <- function(strategy, x, seed) {
  b <- switch(strategy, FC = build_FC(x), CT = build_CT(x),
              IL = build_IL(x), PO = build_PO(x),
              stop("unknown strategy ", strategy))
  set.seed(seed)
  sim <- do.call(virsim::virsim, b$args)
  d <- virsim::aggregate_output(sim$monitor)
  d[, c("IC_inc", "IC_prev") := virsim::gen_derived_outcome(inc = inc, time = time)]
  ic <- d$IC_prev
  ma <- centred_ma(ic)
  qoi1 <- max(ma, na.rm = TRUE)                    # QoI 1, p. 6
  qoi2 <- sum(pmax(0, ic - IC_CAPACITY))           # QoI 2, p. 6 (sum over t = 0..T)
  list(qoi = c(qoi1, qoi2), ts = ic, time = d$time, applied = b$applied)
}
