# ============================================================
# POSTERIOR PREDICTIVE CHECKS AND PARAMETER RECOVERY
# Hierarchical expected-loss Rescorla-Wagner models:
# M3, M4, and M7
#
# Run this script from the same project directory used to fit
# the hierarchical models.
#
# Important:
# The original Stan-generated y_rep conditions stickiness on the
# OBSERVED previous choice. The PPC code below instead simulates
# choices recursively, so the previous SIMULATED choice determines
# stickiness on the next trial.
# ============================================================


# ------------------------------------------------------------
# 0. Packages and paths
# ------------------------------------------------------------

library(cmdstanr)
library(posterior)
library(tidyverse)

set.seed(20260722)

trial_file <- "data/processed/learning_trials_clean.rds"
stan_file  <- "models/stan/hierarchical_expected_loss_rw.stan"
fit_dir    <- "data/processed/hierarchical_rw_fits"

ppc_dir <- "results/hierarchical_rw_models/posterior_predictive_checks"
recovery_dir <- "results/hierarchical_rw_models/parameter_recovery"
recovery_fit_dir <- file.path(recovery_dir, "fits")
recovery_true_dir <- file.path(recovery_dir, "generating_parameters")
recovery_result_dir <- file.path(recovery_dir, "replicate_results")

dir.create(ppc_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(recovery_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(recovery_fit_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(recovery_true_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(recovery_result_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(trial_file)) {
  stop("Cannot find trial file: ", trial_file)
}

if (!file.exists(stan_file)) {
  stop("Cannot find Stan file: ", stan_file)
}

# ------------------------------------------------------------
# 1. Model specifications and completed fits
# ------------------------------------------------------------

model_specs <- list(

  M3_block_alpha_stickiness = list(
    K_alpha = 2L,
    K_rho = 1L,
    K_side = 0L
  ),

  M4_block_alpha_full_response = list(
    K_alpha = 2L,
    K_rho = 1L,
    K_side = 1L
  ),

  M7_single_alpha_full_response = list(
    K_alpha = 1L,
    K_rho = 1L,
    K_side = 1L
  )
)

fit_paths <- setNames(
  file.path(
    fit_dir,
    paste0(names(model_specs), "_hierarchical_fit.rds")
  ),
  names(model_specs)
)

missing_fits <- fit_paths[!file.exists(fit_paths)]

if (length(missing_fits) > 0L) {
  stop(
    "Cannot find these fitted model objects:\n",
    paste(missing_fits, collapse = "\n")
  )
}

hierarchical_fits <- map(
  fit_paths,
  readRDS
)

# ------------------------------------------------------------
# 2. Reconstruct the Stan data
# ------------------------------------------------------------

trials <- readRDS(trial_file) |>
  arrange(participant_id, trial) |>
  mutate(
    participant_id = as.character(participant_id),
    condition = as.character(condition),
    is_volatile = as.integer(condition != "Stable")
  )

required_columns <- c(
  "participant_id",
  "trial",
  "condition",
  "choice_blue",
  "losing_blue",
  "blue_magnitude_scaled",
  "green_magnitude_scaled",
  "blue_left"
)

missing_columns <- setdiff(required_columns, names(trials))

if (length(missing_columns) > 0L) {
  stop(
    "Missing trial-data columns: ",
    paste(missing_columns, collapse = ", ")
  )
}

participant_lookup <- trials |>
  distinct(participant_id) |>
  arrange(participant_id) |>
  mutate(subject_index = row_number())

stan_trials <- trials |>
  inner_join(participant_lookup, by = "participant_id") |>
  arrange(subject_index, trial)

trial_counts <- stan_trials |>
  count(subject_index, name = "n_trials")

if (n_distinct(trial_counts$n_trials) != 1L) {
  stop("Participants do not all have the same number of trials.")
}

N_subjects <- nrow(participant_lookup)
N_trials <- unique(trial_counts$n_trials)

make_integer_matrix <- function(x) {
  matrix(
    as.integer(x),
    nrow = N_subjects,
    ncol = N_trials,
    byrow = TRUE
  )
}

make_numeric_matrix <- function(x) {
  matrix(
    as.numeric(x),
    nrow = N_subjects,
    ncol = N_trials,
    byrow = TRUE
  )
}

common_stan_data <- list(

  N = N_subjects,
  T = N_trials,

  choice_blue = make_integer_matrix(
    stan_trials$choice_blue
  ),

  losing_blue = make_integer_matrix(
    stan_trials$losing_blue
  ),

  is_volatile = make_integer_matrix(
    stan_trials$is_volatile
  ),

  blue_left = make_integer_matrix(
    stan_trials$blue_left
  ),

  blue_magnitude = make_numeric_matrix(
    stan_trials$blue_magnitude_scaled
  ),

  green_magnitude = make_numeric_matrix(
    stan_trials$green_magnitude_scaled
  )
)

stopifnot(
  all(common_stan_data$choice_blue %in% c(0L, 1L)),
  all(common_stan_data$losing_blue %in% c(0L, 1L)),
  all(common_stan_data$is_volatile %in% c(0L, 1L)),
  all(common_stan_data$blue_left %in% c(0L, 1L))
)

# ------------------------------------------------------------
# 3. Task probabilities used only for behavioural summaries
# ------------------------------------------------------------

phase_probability_blue_loses <- c(
  Stable = 0.25,
  Volatile1 = 0.80,
  Volatile2 = 0.20,
  Volatile3 = 0.80
)
unknown_conditions <- setdiff(
  unique(stan_trials$condition),
  names(phase_probability_blue_loses)
)

if (length(unknown_conditions) > 0L) {
  stop(
    "The following condition labels do not match the expected labels ",
    "Stable, Volatile1, Volatile2, Volatile3:\n",
    paste(unknown_conditions, collapse = ", "),
    "\nEdit phase_probability_blue_loses near the top of the script."
  )
}


# ------------------------------------------------------------
# 4. Helpers for posterior draws and recursive simulation
# ------------------------------------------------------------

posterior_variables_for_simulation <- function(specification) {

  variables <- c(
    "alpha",
    "beta"
  )

  if (specification$K_rho == 1L) {
    variables <- c(variables, "rho")
  }

  if (specification$K_side == 1L) {
    variables <- c(variables, "side_bias")
  }

  variables
}

recovery_variables <- function(specification) {

  variables <- c(
    "mu_alpha_raw",
    "sigma_alpha_raw",
    "mu_beta_raw",
    "sigma_beta_raw",
    "alpha",
    "beta"
  )

  if (specification$K_rho == 1L) {
    variables <- c(
      variables,
      "mu_rho",
      "sigma_rho",
      "rho"
    )
  }

  if (specification$K_side == 1L) {
    variables <- c(
      variables,
      "mu_side",
      "sigma_side",
      "side_bias"
    )
  }

  if (specification$K_alpha == 2L) {
    variables <- c(
      variables,
      "delta_alpha",
      "delta_log_alpha"
    )
  }

  unique(variables)
}


extract_parameters_from_draw <- function(
  draw_row,
  specification,
  N
) {

  alpha <- matrix(
    NA_real_,
    nrow = N,
    ncol = specification$K_alpha
  )

  for (n in seq_len(N)) {
    for (k in seq_len(specification$K_alpha)) {

      parameter_name <- paste0(
        "alpha[",
        n,
        ",",
        k,
        "]"
      )

      alpha[n, k] <- as.numeric(
        draw_row[parameter_name]
      )
    }
  }

  beta <- vapply(
    seq_len(N),
    function(n) {
      as.numeric(
        draw_row[
          paste0("beta[", n, "]")
        ]
      )
    },
    numeric(1)
  )

  rho <- rep(0, N)

  if (specification$K_rho == 1L) {
    rho <- vapply(
      seq_len(N),
      function(n) {
        as.numeric(
          draw_row[
            paste0("rho[", n, "]")
          ]
        )
      },
      numeric(1)
    )
  }

  side_bias <- rep(0, N)

  if (specification$K_side == 1L) {
    side_bias <- vapply(
      seq_len(N),
      function(n) {
        as.numeric(
          draw_row[
            paste0("side_bias[", n, "]")
          ]
        )
      },
      numeric(1)
    )
  }

  list(
    alpha = alpha,
    beta = beta,
    rho = rho,
    side_bias = side_bias
  )
}


simulate_choice_matrix <- function(
  parameters,
  specification,
  stan_data,
  seed = NULL
) {

  if (!is.null(seed)) {
    set.seed(seed)
  }

  simulated_choice <- matrix(
    0L,
    nrow = stan_data$N,
    ncol = stan_data$T
  )

  for (n in seq_len(stan_data$N)) {

    belief_blue_losing <- 0.5
    previous_simulated_choice <- 0

    for (t in seq_len(stan_data$T)) {

      expected_loss_blue <-
        belief_blue_losing *
        stan_data$blue_magnitude[n, t]

      expected_loss_green <-
        (1 - belief_blue_losing) *
        stan_data$green_magnitude[n, t]

      decision_value <-
        parameters$beta[n] *
        (
          expected_loss_green -
          expected_loss_blue
        )

      if (specification$K_rho == 1L) {
        decision_value <-
          decision_value +
          parameters$rho[n] *
          previous_simulated_choice
      }

      if (specification$K_side == 1L) {
        decision_value <-
          decision_value +
          parameters$side_bias[n] *
          (
            2 * stan_data$blue_left[n, t] - 1
          )
      }

      probability_choose_blue <- plogis(
        decision_value
      )

      simulated_choice[n, t] <- rbinom(
        n = 1,
        size = 1,
        prob = probability_choose_blue
      )

      current_alpha <- if (
        specification$K_alpha == 1L
      ) {

        parameters$alpha[n, 1]

      } else if (
        stan_data$is_volatile[n, t] == 0L
      ) {

        parameters$alpha[n, 1]

      } else {

        parameters$alpha[n, 2]
      }

      prediction_error <-
        stan_data$losing_blue[n, t] -
        belief_blue_losing

      belief_blue_losing <-
        belief_blue_losing +
        current_alpha *
        prediction_error

      # Crucially, use the SIMULATED previous choice
      previous_simulated_choice <-
        2 * simulated_choice[n, t] - 1
    }
  }

  simulated_choice
}
# ------------------------------------------------------------
# 5. Helpers for behavioural posterior predictive summaries
# ------------------------------------------------------------

safe_mean <- function(x) {

  if (length(x) == 0L || all(is.na(x))) {
    return(NA_real_)
  }

  mean(x, na.rm = TRUE)
}

make_behaviour_table <- function(choice_matrix) {

  choice_vector <- as.integer(
    as.vector(
      t(choice_matrix)
    )
  )

  stan_trials |>
    mutate(
      choice_blue_ppc = choice_vector,

      true_p_blue_losing =
        unname(
          phase_probability_blue_loses[condition]
        ),

      true_expected_loss_blue =
        true_p_blue_losing *
        blue_magnitude_scaled,

      true_expected_loss_green =
        (1 - true_p_blue_losing) *
        green_magnitude_scaled,

      optimal_blue = case_when(
        true_expected_loss_blue <
          true_expected_loss_green ~ 1L,

        true_expected_loss_blue >
          true_expected_loss_green ~ 0L,

        TRUE ~ NA_integer_
      ),

      optimal_choice = as.integer(
        choice_blue_ppc == optimal_blue
      ),

      chosen_option_lost = as.integer(
        choice_blue_ppc == losing_blue
      )
    ) |>
    group_by(subject_index) |>
    arrange(trial, .by_group = TRUE) |>
    mutate(
      switched = as.integer(
        choice_blue_ppc != lag(choice_blue_ppc)
      ),

      previous_chosen_option_lost =
        lag(chosen_option_lost)
    ) |>
    ungroup() |>
    group_by(subject_index, condition) |>
    arrange(trial, .by_group = TRUE) |>
    mutate(
      trial_within_phase = row_number()
    ) |>
    ungroup()
}

summarise_overall_behaviour <- function(behaviour_data) {

  behaviour_data |>
    summarise(
      optimal_choice = safe_mean(optimal_choice),
      choose_blue = safe_mean(choice_blue_ppc),
      stay_rate = safe_mean(1 - switched),

      switch_after_loss = safe_mean(
        switched[
          previous_chosen_option_lost == 1L
        ]
      ),

      switch_after_no_loss = safe_mean(
        switched[
          previous_chosen_option_lost == 0L
        ]
      )
    ) |>
    pivot_longer(
      cols = everything(),
      names_to = "metric",
      values_to = "value"
    )
}

summarise_phase_behaviour <- function(behaviour_data) {

  behaviour_data |>
    group_by(condition) |>
    summarise(
      optimal_choice = safe_mean(optimal_choice),
      choose_blue = safe_mean(choice_blue_ppc),
      stay_rate = safe_mean(1 - switched),

      switch_after_loss = safe_mean(
        switched[
          previous_chosen_option_lost == 1L
        ]
      ),

      switch_after_no_loss = safe_mean(
        switched[
          previous_chosen_option_lost == 0L
        ]
      ),

      .groups = "drop"
    ) |>
    pivot_longer(
      cols = -condition,
      names_to = "metric",
      values_to = "value"
    )
}

summarise_reversal_curve <- function(behaviour_data) {

  behaviour_data |>
    filter(condition != "Stable") |>
    group_by(
      condition,
      trial_within_phase
    ) |>
    summarise(
      optimal_choice = safe_mean(optimal_choice),
      choose_blue = safe_mean(choice_blue_ppc),
      .groups = "drop"
    ) |>
    pivot_longer(
      cols = c(
        optimal_choice,
        choose_blue
      ),
      names_to = "metric",
      values_to = "value"
    )
}


# ============================================================
# PART A: POSTERIOR PREDICTIVE CHECKS
# ============================================================


# ------------------------------------------------------------
# 6. Run recursively generated posterior predictive simulations
# ------------------------------------------------------------

# 300 draws are adequate for initial PPC intervals.
# Increase to 500 or 1000 for the final analysis.
N_PPC_DRAWS <- 300L
PPC_SEED <- 20260722L

observed_behaviour <- make_behaviour_table(
  common_stan_data$choice_blue
)

observed_overall <- summarise_overall_behaviour(
  observed_behaviour
)

observed_phase <- summarise_phase_behaviour(
  observed_behaviour
)

observed_reversal <- summarise_reversal_curve(
  observed_behaviour
)

ppc_overall_list <- list()
ppc_phase_list <- list()
ppc_reversal_list <- list()

for (model_name in names(model_specs)) {

  message(
    "\nRunning posterior predictive simulations for ",
    model_name
  )

  specification <- model_specs[[model_name]]
  fit <- hierarchical_fits[[model_name]]

  simulation_variables <-
    posterior_variables_for_simulation(
      specification
    )

  posterior_matrix <- as.matrix(
    fit$draws(
      variables = simulation_variables,
      format = "draws_matrix"
    )
  )

  n_draws_this_model <- min(
    N_PPC_DRAWS,
    nrow(posterior_matrix)
  )

  set.seed(
    PPC_SEED +
      match(model_name, names(model_specs)) * 1000L
  )

  selected_draws <- sample(
    seq_len(nrow(posterior_matrix)),
    size = n_draws_this_model,
    replace = FALSE
  )

  model_overall <- vector(
    "list",
    n_draws_this_model
  )

  model_phase <- vector(
    "list",
    n_draws_this_model
  )

  model_reversal <- vector(
    "list",
    n_draws_this_model
  )

  for (j in seq_len(n_draws_this_model)) {

    if (
      j == 1L ||
      j %% 25L == 0L ||
      j == n_draws_this_model
    ) {
      message(
        model_name,
        ": PPC draw ",
        j,
        " of ",
        n_draws_this_model
      )
    }

    draw_row <- posterior_matrix[
      selected_draws[j],
      ,
      drop = TRUE
    ]

    generating_parameters <-
      extract_parameters_from_draw(
        draw_row = draw_row,
        specification = specification,
        N = N_subjects
      )

    simulated_choice <-
      simulate_choice_matrix(
        parameters = generating_parameters,
        specification = specification,
        stan_data = common_stan_data,
        seed =
          PPC_SEED +
          match(model_name, names(model_specs)) * 100000L +
          j
      )

    simulated_behaviour <-
      make_behaviour_table(
        simulated_choice
      )

    model_overall[[j]] <-
      summarise_overall_behaviour(
        simulated_behaviour
      ) |>
      mutate(
        model = model_name,
        ppc_draw = j,
        .before = 1
      )

    model_phase[[j]] <-
      summarise_phase_behaviour(
        simulated_behaviour
      ) |>
      mutate(
        model = model_name,
        ppc_draw = j,
        .before = 1
      )

    model_reversal[[j]] <-
      summarise_reversal_curve(
        simulated_behaviour
      ) |>
      mutate(
        model = model_name,
        ppc_draw = j,
        .before = 1
      )
  }

  ppc_overall_list[[model_name]] <-
    bind_rows(model_overall)

  ppc_phase_list[[model_name]] <-
    bind_rows(model_phase)

  ppc_reversal_list[[model_name]] <-
    bind_rows(model_reversal)
}

ppc_overall_draws <- bind_rows(
  ppc_overall_list
)

ppc_phase_draws <- bind_rows(
  ppc_phase_list
)

ppc_reversal_draws <- bind_rows(
  ppc_reversal_list
)

write_csv(
  ppc_overall_draws,
  file.path(
    ppc_dir,
    "ppc_overall_draws.csv"
  )
)

write_csv(
  ppc_phase_draws,
  file.path(
    ppc_dir,
    "ppc_phase_draws.csv"
  )
)

write_csv(
  ppc_reversal_draws,
  file.path(
    ppc_dir,
    "ppc_reversal_curve_draws.csv"
  )
)

# ------------------------------------------------------------
# 7. Summarise PPC distributions and Bayesian p-values
# ------------------------------------------------------------

ppc_overall_summary <-
  ppc_overall_draws |>
  left_join(
    observed_overall |>
      rename(observed = value),
    by = "metric"
  ) |>
  group_by(model, metric) |>
  summarise(
    observed = first(observed),
    posterior_predictive_mean = mean(value),
    posterior_predictive_median = median(value),
    lower_95 = quantile(value, 0.025),
    upper_95 = quantile(value, 0.975),

    p_upper = mean(
      value >= observed
    ),

    p_two_sided = min(
      1,
      2 * min(
        mean(value >= observed),
        mean(value <= observed)
      )
    ),

    .groups = "drop"
  )

ppc_phase_summary <-
  ppc_phase_draws |>
  left_join(
    observed_phase |>
      rename(observed = value),
    by = c(
      "condition",
      "metric"
    )
  ) |>
  group_by(
    model,
    condition,
    metric
  ) |>
  summarise(
    observed = first(observed),
    posterior_predictive_mean = mean(value),
    posterior_predictive_median = median(value),
    lower_95 = quantile(value, 0.025),
    upper_95 = quantile(value, 0.975),

    p_upper = mean(
      value >= observed
    ),

    p_two_sided = min(
      1,
      2 * min(
        mean(value >= observed),
        mean(value <= observed)
      )
    ),

    .groups = "drop"
  )

ppc_reversal_summary <-
  ppc_reversal_draws |>
  left_join(
    observed_reversal |>
      rename(observed = value),
    by = c(
      "condition",
      "trial_within_phase",
      "metric"
    )
  ) |>
  group_by(
    model,
    condition,
    trial_within_phase,
    metric
  ) |>
  summarise(
    observed = first(observed),
    posterior_predictive_mean = mean(value),
    posterior_predictive_median = median(value),
    lower_95 = quantile(value, 0.025),
    upper_95 = quantile(value, 0.975),
    .groups = "drop"
  )

write_csv(
  ppc_overall_summary,
  file.path(
    ppc_dir,
    "ppc_overall_summary.csv"
  )
)

write_csv(
  ppc_phase_summary,
  file.path(
    ppc_dir,
    "ppc_phase_summary.csv"
  )
)

write_csv(
  ppc_reversal_summary,
  file.path(
    ppc_dir,
    "ppc_reversal_curve_summary.csv"
  )
)

print(
  ppc_overall_summary,
  n = Inf
)

print(
  ppc_phase_summary |>
    filter(
      metric %in% c(
        "optimal_choice",
        "stay_rate",
        "switch_after_loss"
      )
    ),
  n = Inf
)

# ------------------------------------------------------------
# 8. PPC plots
# ------------------------------------------------------------

metric_labels <- c(
  optimal_choice = "Optimal choice",
  choose_blue = "Choose blue",
  stay_rate = "Choice repetition",
  switch_after_loss = "Switch after loss",
  switch_after_no_loss = "Switch after no loss"
)

ppc_overall_plot <-
  ppc_overall_summary |>
  mutate(
    metric = factor(
      metric,
      levels = names(metric_labels),
      labels = unname(metric_labels)
    )
  ) |>
  ggplot(
    aes(
      x = metric,
      y = posterior_predictive_median
    )
  ) +
  geom_linerange(
    aes(
      ymin = lower_95,
      ymax = upper_95
    )
  ) +
  geom_point() +
  geom_point(
    aes(y = observed),
    shape = 4,
    size = 3,
    stroke = 1
  ) +
  facet_wrap(
    ~ model,
    ncol = 1
  ) +
  coord_cartesian(
    ylim = c(0, 1)
  ) +
  labs(
    x = NULL,
    y = "Proportion",
    title = "Overall posterior predictive checks",
    subtitle = "Intervals: posterior predictive 95%; crosses: observed values"
  ) +
  theme_bw() +
  theme(
    axis.text.x = element_text(
      angle = 35,
      hjust = 1
    )
  )

ggsave(
  filename = file.path(
    ppc_dir,
    "ppc_overall_metrics.png"
  ),
  plot = ppc_overall_plot,
  width = 9,
  height = 11,
  dpi = 300
)


ppc_phase_optimal_plot <-
  ppc_phase_summary |>
  filter(
    metric == "optimal_choice"
  ) |>
  mutate(
    condition = factor(
      condition,
      levels = names(
        phase_probability_blue_loses
      )
    )
  ) |>
  ggplot(
    aes(
      x = condition,
      y = posterior_predictive_median
    )
  ) +
  geom_linerange(
    aes(
      ymin = lower_95,
      ymax = upper_95
    )
  ) +
  geom_point() +
  geom_point(
    aes(y = observed),
    shape = 4,
    size = 3,
    stroke = 1
  ) +
  facet_wrap(
    ~ model,
    ncol = 1
  ) +
  coord_cartesian(
    ylim = c(0, 1)
  ) +
  labs(
    x = "Phase",
    y = "Proportion of optimal choices",
    title = "Phase-specific posterior predictive checks",
    subtitle = "Intervals: posterior predictive 95%; crosses: observed values"
  ) +
  theme_bw()

ggsave(
  filename = file.path(
    ppc_dir,
    "ppc_phase_optimal_choice.png"
  ),
  plot = ppc_phase_optimal_plot,
  width = 9,
  height = 11,
  dpi = 300
)


ppc_reversal_plot <-
  ppc_reversal_summary |>
  filter(
    metric == "optimal_choice"
  ) |>
  mutate(
    condition = factor(
      condition,
      levels = c(
        "Volatile1",
        "Volatile2",
        "Volatile3"
      )
    )
  ) |>
  ggplot(
    aes(
      x = trial_within_phase,
      y = posterior_predictive_median
    )
  ) +
  geom_ribbon(
    aes(
      ymin = lower_95,
      ymax = upper_95
    ),
    alpha = 0.20
  ) +
  geom_line() +
  geom_point(
    aes(y = observed),
    shape = 4,
    size = 1.6,
    stroke = 0.7
  ) +
  facet_grid(
    model ~ condition
  ) +
  coord_cartesian(
    ylim = c(0, 1)
  ) +
  labs(
    x = "Trial within volatile phase",
    y = "Proportion of optimal choices",
    title = "Posterior predictive checks of post-reversal adaptation",
    subtitle = "Bands: posterior predictive 95%; crosses: observed values"
  ) +
  theme_bw()

ggsave(
  filename = file.path(
    ppc_dir,
    "ppc_reversal_adaptation.png"
  ),
  plot = ppc_reversal_plot,
  width = 13,
  height = 9,
  dpi = 300
)

# ============================================================
# PART B: PARAMETER RECOVERY
# ============================================================


# ------------------------------------------------------------
# 9. Compile the existing hierarchical Stan model
# ------------------------------------------------------------

hierarchical_rw_model <- cmdstan_model(
  stan_file,
  force_recompile = FALSE
)

# ------------------------------------------------------------
# 10. Parameter-recovery settings
# ------------------------------------------------------------

# Begin with 5 replications as a computational pilot.
# After confirming that the script and fits work correctly,
# increase this to at least 20 for the final recovery analysis.
N_RECOVERY_REPS <- 5L

RECOVERY_SEED <- 20260723L

RECOVERY_CHAINS <- 4L

RECOVERY_PARALLEL_CHAINS <- min(
  RECOVERY_CHAINS,
  max(
    1L,
    parallel::detectCores(
      logical = FALSE
    ) - 1L
  )
)

RECOVERY_ITER_WARMUP <- 1000L
RECOVERY_ITER_SAMPLING <- 1000L
RECOVERY_ADAPT_DELTA <- 0.97
RECOVERY_MAX_TREEDEPTH <- 13L

# ------------------------------------------------------------
# 11. Recovery helper functions
# ------------------------------------------------------------

parameter_classification <- function(
  parameter,
  model
) {

  case_when(

    str_detect(
      parameter,
      "^alpha\\[[0-9]+,1\\]$"
    ) &
      model ==
      "M7_single_alpha_full_response" ~
      "alpha_general",

    str_detect(
      parameter,
      "^alpha\\[[0-9]+,1\\]$"
    ) ~
      "alpha_stable",

    str_detect(
      parameter,
      "^alpha\\[[0-9]+,2\\]$"
    ) ~
      "alpha_volatile",

    str_detect(
      parameter,
      "^delta_alpha\\["
    ) ~
      "delta_alpha",

    str_detect(
      parameter,
      "^delta_log_alpha\\["
    ) ~
      "delta_log_alpha",

    str_detect(
      parameter,
      "^beta\\["
    ) ~
      "beta",

    str_detect(
      parameter,
      "^rho\\["
    ) ~
      "rho",

    str_detect(
      parameter,
      "^side_bias\\["
    ) ~
      "side_bias",

    parameter == "mu_beta_raw" ~
      "group_mu_beta_raw",

    parameter == "sigma_beta_raw" ~
      "group_sigma_beta_raw",

    str_detect(
      parameter,
      "^mu_alpha_raw\\[1\\]$"
    ) &
      model ==
      "M7_single_alpha_full_response" ~
      "group_mu_alpha_general_raw",

    str_detect(
      parameter,
      "^mu_alpha_raw\\[1\\]$"
    ) ~
      "group_mu_alpha_stable_raw",

    str_detect(
      parameter,
      "^mu_alpha_raw\\[2\\]$"
    ) ~
      "group_mu_alpha_volatile_raw",

    str_detect(
      parameter,
      "^sigma_alpha_raw\\[1\\]$"
    ) &
      model ==
      "M7_single_alpha_full_response" ~
      "group_sigma_alpha_general_raw",

    str_detect(
      parameter,
      "^sigma_alpha_raw\\[1\\]$"
    ) ~
      "group_sigma_alpha_stable_raw",

    str_detect(
      parameter,
      "^sigma_alpha_raw\\[2\\]$"
    ) ~
      "group_sigma_alpha_volatile_raw",

    str_detect(
      parameter,
      "^mu_rho"
    ) ~
      "group_mu_rho",

    str_detect(
      parameter,
      "^sigma_rho"
    ) ~
      "group_sigma_rho",

    str_detect(
      parameter,
      "^mu_side"
    ) ~
      "group_mu_side",

    str_detect(
      parameter,
      "^sigma_side"
    ) ~
      "group_sigma_side",

    TRUE ~
      "other"
  )
}

safe_correlation <- function(
  x,
  y,
  method = "pearson"
) {

  complete <- complete.cases(x, y)

  if (
    sum(complete) < 3L ||
    sd(x[complete]) == 0 ||
    sd(y[complete]) == 0
  ) {
    return(NA_real_)
  }

  cor(
    x[complete],
    y[complete],
    method = method
  )
}


extract_recovery_result <- function(
  recovery_fit,
  true_parameter_vector,
  model_name,
  replication
) {

  variables_this_model <-
    recovery_variables(
      model_specs[[model_name]]
    )

  recovery_summary <-
    recovery_fit$summary(
      variables =
        variables_this_model
    )

  lower_column <- intersect(
    c(
      "q5",
      "q2.5"
    ),
    names(recovery_summary)
  )[1]

  upper_column <- intersect(
    c(
      "q95",
      "q97.5"
    ),
    names(recovery_summary)
  )[1]

  if (
    is.na(lower_column) ||
    is.na(upper_column)
  ) {
    stop(
      "Could not identify posterior interval columns ",
      "in CmdStanR summary output."
    )
  }

  true_table <- tibble(
    parameter =
      names(true_parameter_vector),

    true_value =
      as.numeric(true_parameter_vector)
  )

  recovery_summary |>
    transmute(
      parameter = variable,
      recovered_mean = mean,
      recovered_median = median,
      lower_interval =
        .data[[lower_column]],
      upper_interval =
        .data[[upper_column]],
      rhat = rhat,
      ess_bulk = ess_bulk,
      ess_tail = ess_tail
    ) |>
    inner_join(
      true_table,
      by = "parameter"
    ) |>
    mutate(
      model = model_name,
      replication = replication,

      bias =
        recovered_mean -
        true_value,

      squared_error =
        (
          recovered_mean -
          true_value
        )^2,

      covered =
        true_value >= lower_interval &
        true_value <= upper_interval,

      parameter_class =
        parameter_classification(
          parameter,
          model_name
        ),

      .before = 1
    )
}


extract_recovery_diagnostics <- function(
  recovery_fit,
  model_name,
  replication
) {

  parameter_summary <-
    recovery_fit$summary()

  sampler_diagnostics <-
    as.matrix(
      recovery_fit$sampler_diagnostics(
        format = "draws_matrix"
      )
    )

  tibble(
    model = model_name,
    replication = replication,

    maximum_rhat = max(
      parameter_summary$rhat,
      na.rm = TRUE
    ),

    minimum_bulk_ess = min(
      parameter_summary$ess_bulk,
      na.rm = TRUE
    ),

    minimum_tail_ess = min(
      parameter_summary$ess_tail,
      na.rm = TRUE
    ),

    divergences = sum(
      sampler_diagnostics[, "divergent__"]
    ),

    maximum_treedepth_hits = sum(
      sampler_diagnostics[, "treedepth__"] >=
        RECOVERY_MAX_TREEDEPTH
    )
  )
}

# ------------------------------------------------------------
# 12. Simulate and refit each model
# ------------------------------------------------------------

all_recovery_results <- list()
all_recovery_diagnostics <- list()

for (model_name in names(model_specs)) {

  specification <- model_specs[[model_name]]
  empirical_fit <- hierarchical_fits[[model_name]]

  variables_this_model <-
    recovery_variables(
      specification
    )

  empirical_posterior_matrix <- as.matrix(
    empirical_fit$draws(
      variables = variables_this_model,
      format = "draws_matrix"
    )
  )

  for (
    replication in seq_len(
      N_RECOVERY_REPS
    )
  ) {

    message(
      "\nParameter recovery: ",
      model_name,
      ", replication ",
      replication,
      " of ",
      N_RECOVERY_REPS
    )

    file_stub <- paste0(
      model_name,
      "_recovery_",
      sprintf("%02d", replication)
    )

    true_parameter_path <- file.path(
      recovery_true_dir,
      paste0(
        file_stub,
        "_true_parameters.rds"
      )
    )

    recovery_fit_path <- file.path(
      recovery_fit_dir,
      paste0(
        file_stub,
        "_fit.rds"
      )
    )

    recovery_result_path <- file.path(
      recovery_result_dir,
      paste0(
        file_stub,
        "_parameter_results.csv"
      )
    )

    recovery_diagnostic_path <- file.path(
      recovery_result_dir,
      paste0(
        file_stub,
        "_diagnostics.csv"
      )
    )

    # --------------------------------------------------------
    # Select and save one complete posterior draw as the
    # generating parameter set. This yields realistic parameter
    # values in the empirical region.
    # --------------------------------------------------------

    if (file.exists(true_parameter_path)) {

      true_parameter_vector <-
        readRDS(
          true_parameter_path
        )

    } else {

      set.seed(
        RECOVERY_SEED +
          match(
            model_name,
            names(model_specs)
          ) * 10000L +
          replication
      )

      selected_draw <- sample(
        seq_len(
          nrow(
            empirical_posterior_matrix
          )
        ),
        size = 1L
      )

      true_parameter_vector <-
        empirical_posterior_matrix[
          selected_draw,
          ,
          drop = TRUE
        ]

      saveRDS(
        true_parameter_vector,
        true_parameter_path
      )
    }

    generating_parameters <-
      extract_parameters_from_draw(
        draw_row =
          true_parameter_vector,

        specification =
          specification,

        N =
          N_subjects
      )

    simulated_choice <-
      simulate_choice_matrix(
        parameters =
          generating_parameters,

        specification =
          specification,

        stan_data =
          common_stan_data,

        seed =
          RECOVERY_SEED +
          match(
            model_name,
            names(model_specs)
          ) * 100000L +
          replication
      )

    recovery_common_data <-
      common_stan_data

    recovery_common_data$choice_blue <-
      simulated_choice

    recovery_stan_data <- c(
      recovery_common_data,
      specification
    )

    # --------------------------------------------------------
    # Load a completed recovery fit or run a new one
    # --------------------------------------------------------

    if (file.exists(recovery_fit_path)) {

      message(
        "Loading completed recovery fit: ",
        recovery_fit_path
      )

      recovery_fit <-
        readRDS(
          recovery_fit_path
        )

    } else {

      recovery_fit <-
        hierarchical_rw_model$sample(

          data =
            recovery_stan_data,

          seed =
            RECOVERY_SEED +
            match(
              model_name,
              names(model_specs)
            ) * 1000000L +
            replication,

          chains =
            RECOVERY_CHAINS,

          parallel_chains =
            RECOVERY_PARALLEL_CHAINS,

          iter_warmup =
            RECOVERY_ITER_WARMUP,

          iter_sampling =
            RECOVERY_ITER_SAMPLING,

          adapt_delta =
            RECOVERY_ADAPT_DELTA,

          max_treedepth =
            RECOVERY_MAX_TREEDEPTH,

          refresh =
            100
        )

      recovery_fit$save_object(
        file =
          recovery_fit_path
      )
    }

    # --------------------------------------------------------
    # Extract recovery statistics and diagnostics
    # --------------------------------------------------------

    recovery_result <-
      extract_recovery_result(
        recovery_fit =
          recovery_fit,

        true_parameter_vector =
          true_parameter_vector,

        model_name =
          model_name,

        replication =
          replication
      )

    recovery_diagnostic <-
      extract_recovery_diagnostics(
        recovery_fit =
          recovery_fit,

        model_name =
          model_name,

        replication =
          replication
      )

    write_csv(
      recovery_result,
      recovery_result_path
    )

    write_csv(
      recovery_diagnostic,
      recovery_diagnostic_path
    )

    result_key <- paste0(
      model_name,
      "_",
      replication
    )

    all_recovery_results[[result_key]] <-
      recovery_result

    all_recovery_diagnostics[[result_key]] <-
      recovery_diagnostic
  }
}

# ------------------------------------------------------------
# 13. Combine recovery results
# ------------------------------------------------------------

recovery_results <- bind_rows(
  all_recovery_results
)

recovery_diagnostics <- bind_rows(
  all_recovery_diagnostics
)

write_csv(
  recovery_results,
  file.path(
    recovery_dir,
    "all_parameter_recovery_results.csv"
  )
)

write_csv(
  recovery_diagnostics,
  file.path(
    recovery_dir,
    "all_parameter_recovery_diagnostics.csv"
  )
)

print(
  recovery_diagnostics,
  n = Inf
)

# ------------------------------------------------------------
# 14. Summarise individual-level parameter recovery
# ------------------------------------------------------------

individual_parameter_classes <- c(
  "alpha_general",
  "alpha_stable",
  "alpha_volatile",
  "delta_alpha",
  "delta_log_alpha",
  "beta",
  "rho",
  "side_bias"
)

individual_recovery_by_replication <-
  recovery_results |>
  filter(
    parameter_class %in%
      individual_parameter_classes
  ) |>
  group_by(
    model,
    replication,
    parameter_class
  ) |>
  summarise(
    n_parameters = n(),

    pearson_r = safe_correlation(
      true_value,
      recovered_mean,
      method = "pearson"
    ),

    spearman_rho = safe_correlation(
      true_value,
      recovered_mean,
      method = "spearman"
    ),

    mean_bias = mean(
      bias,
      na.rm = TRUE
    ),

    rmse = sqrt(
      mean(
        squared_error,
        na.rm = TRUE
      )
    ),

    interval_coverage = mean(
      covered,
      na.rm = TRUE
    ),

    .groups = "drop"
  )

individual_recovery_summary <-
  individual_recovery_by_replication |>
  group_by(
    model,
    parameter_class
  ) |>
  summarise(
    n_replications = n(),

    mean_pearson_r = mean(
      pearson_r,
      na.rm = TRUE
    ),

    median_pearson_r = median(
      pearson_r,
      na.rm = TRUE
    ),

    mean_spearman_rho = mean(
      spearman_rho,
      na.rm = TRUE
    ),

    mean_bias = mean(
      mean_bias,
      na.rm = TRUE
    ),

    mean_rmse = mean(
      rmse,
      na.rm = TRUE
    ),

    mean_interval_coverage = mean(
      interval_coverage,
      na.rm = TRUE
    ),

    .groups = "drop"
  )

write_csv(
  individual_recovery_by_replication,
  file.path(
    recovery_dir,
    "individual_recovery_by_replication.csv"
  )
)

write_csv(
  individual_recovery_summary,
  file.path(
    recovery_dir,
    "individual_recovery_summary.csv"
  )
)

print(
  individual_recovery_summary,
  n = Inf
)

# ------------------------------------------------------------
# 15. Summarise group-level hyperparameter recovery
# ------------------------------------------------------------

group_recovery_summary <-
  recovery_results |>
  filter(
    str_detect(
      parameter_class,
      "^group_"
    )
  ) |>
  group_by(
    model,
    parameter_class
  ) |>
  summarise(
    n_replications = n(),

    mean_true_value = mean(
      true_value,
      na.rm = TRUE
    ),

    mean_recovered_value = mean(
      recovered_mean,
      na.rm = TRUE
    ),

    mean_bias = mean(
      bias,
      na.rm = TRUE
    ),

    rmse = sqrt(
      mean(
        squared_error,
        na.rm = TRUE
      )
    ),

    interval_coverage = mean(
      covered,
      na.rm = TRUE
    ),

    .groups = "drop"
  )

write_csv(
  group_recovery_summary,
  file.path(
    recovery_dir,
    "group_hyperparameter_recovery_summary.csv"
  )
)

print(
  group_recovery_summary,
  n = Inf
)

# ------------------------------------------------------------
# 16. Parameter-recovery plots
# ------------------------------------------------------------

individual_recovery_plot <-
  recovery_results |>
  filter(
    parameter_class %in%
      individual_parameter_classes
  ) |>
  ggplot(
    aes(
      x = true_value,
      y = recovered_mean
    )
  ) +
  geom_point(
    alpha = 0.35
  ) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = 2
  ) +
  facet_grid(
    model ~ parameter_class,
    scales = "free"
  ) +
  labs(
    x = "Generating parameter",
    y = "Recovered posterior mean",
    title = "Individual-level parameter recovery"
  ) +
  theme_bw()

ggsave(
  filename = file.path(
    recovery_dir,
    "individual_parameter_recovery.png"
  ),
  plot = individual_recovery_plot,
  width = 17,
  height = 11,
  dpi = 300
)


recovery_correlation_plot <-
  individual_recovery_summary |>
  ggplot(
    aes(
      x = parameter_class,
      y = mean_pearson_r
    )
  ) +
  geom_col() +
  facet_wrap(
    ~ model,
    ncol = 1
  ) +
  coord_cartesian(
    ylim = c(-1, 1)
  ) +
  labs(
    x = NULL,
    y = "Mean Pearson recovery correlation",
    title = "Parameter-recovery correlations"
  ) +
  theme_bw() +
  theme(
    axis.text.x = element_text(
      angle = 35,
      hjust = 1
    )
  )

ggsave(
  filename = file.path(
    recovery_dir,
    "parameter_recovery_correlations.png"
  ),
  plot = recovery_correlation_plot,
  width = 10,
  height = 11,
  dpi = 300
)

# ------------------------------------------------------------
# 17. Final messages
# ------------------------------------------------------------

message(
  "\nPosterior predictive checks completed.\n",
  "PPC results saved in: ",
  ppc_dir
)

message(
  "\nParameter-recovery pilot completed.\n",
  "Recovery results saved in: ",
  recovery_dir,
  "\nAfter checking diagnostics, increase N_RECOVERY_REPS ",
  "from 5 to at least 20 for the final analysis."
)
