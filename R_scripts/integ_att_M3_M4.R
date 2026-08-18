# ============================================================
# Integrated hierarchical attachment models for M3 and M4
#
# Fits:
# 1) M3 + avoidance and anxiety jointly
# 2) M3 + total insecurity
# 3) M4 + avoidance and anxiety jointly
# 4) M4 + total insecurity
#
# Predictors are z-standardised and enter both stable and volatile
# learning-rate distributions. The relative effect is:
# b_volatile - b_stable on the logit(alpha) scale.
# ============================================================

library(cmdstanr)
library(posterior)
library(tidyverse)
library(readxl)

trial_file <- "data/processed/learning_trials_clean.rds"
questionnaire_file <- "65_questionnaires.xlsx"
questionnaire_sheet <- "65_sept25"

stan_dir <- "models/stan"
fit_dir <- "data/processed/integrated_attachment_fits"
result_dir <- "results/hierarchical_rw_models/integrated_attachment_models"

dir.create(stan_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fit_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(result_dir, recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------
# 1. Trial data and participant order
# ------------------------------------------------------------

trials <- readRDS(trial_file) |>
  arrange(participant_id, trial) |>
  mutate(
    participant_id = as.character(participant_id),
    is_volatile = as.integer(condition != "Stable")
  )

required_trial_columns <- c(
  "participant_id", "trial", "condition", "choice_blue",
  "losing_blue", "blue_magnitude_scaled",
  "green_magnitude_scaled", "blue_left"
)

missing_trial_columns <- setdiff(required_trial_columns, names(trials))
if (length(missing_trial_columns) > 0L) {
  stop("Missing trial columns: ",
       paste(missing_trial_columns, collapse = ", "))
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

N <- nrow(participant_lookup)
T <- unique(trial_counts$n_trials)

make_int_matrix <- function(x) {
  matrix(as.integer(x), nrow = N, ncol = T, byrow = TRUE)
}

make_num_matrix <- function(x) {
  matrix(as.numeric(x), nrow = N, ncol = T, byrow = TRUE)
}

common_stan_data <- list(
  N = N,
  T = T,
  choice_blue = make_int_matrix(stan_trials$choice_blue),
  losing_blue = make_int_matrix(stan_trials$losing_blue),
  is_volatile = make_int_matrix(stan_trials$is_volatile),
  blue_left = make_int_matrix(stan_trials$blue_left),
  blue_magnitude = make_num_matrix(stan_trials$blue_magnitude_scaled),
  green_magnitude = make_num_matrix(stan_trials$green_magnitude_scaled)
)

# ------------------------------------------------------------
# 2. Questionnaire data
# ------------------------------------------------------------

normalise_id <- function(x) {
  x |>
    as.character() |>
    stringr::str_trim() |>
    stringr::str_replace("\\.0$", "")
}

to_numeric <- function(x) {
  suppressWarnings(as.numeric(as.character(x)))
}

questionnaire_raw <- read_excel(
  questionnaire_file,
  sheet = questionnaire_sheet
)

required_questionnaire_columns <- c(
  "subIDs", "avoid_avg", "anxiety_avg", "aaq_total_avg"
)

missing_questionnaire_columns <- setdiff(
  required_questionnaire_columns,
  names(questionnaire_raw)
)

if (length(missing_questionnaire_columns) > 0L) {
  stop("Missing questionnaire columns: ",
       paste(missing_questionnaire_columns, collapse = ", "))
}

questionnaires <- questionnaire_raw |>
  transmute(
    participant_id = normalise_id(subIDs),
    avoidance = to_numeric(avoid_avg),
    anxiety = to_numeric(anxiety_avg),
    insecurity = to_numeric(aaq_total_avg)
  ) |>
  filter(!is.na(participant_id), participant_id != "")

duplicate_ids <- questionnaires |>
  count(participant_id) |>
  filter(n > 1L)

if (nrow(duplicate_ids) > 0L) {
  print(duplicate_ids, n = Inf)
  stop("Duplicate participant IDs in questionnaire data.")
}

participant_data <- participant_lookup |>
  mutate(participant_id = normalise_id(participant_id)) |>
  left_join(questionnaires, by = "participant_id") |>
  arrange(subject_index)

missing_scores <- participant_data |>
  filter(if_any(c(avoidance, anxiety, insecurity), is.na))

if (nrow(missing_scores) > 0L) {
  print(missing_scores, n = Inf)
  stop("Some fitted participants have missing attachment scores.")
}

standardisation <- tibble(
  predictor = c("avoidance", "anxiety", "insecurity"),
  mean = c(
    mean(participant_data$avoidance),
    mean(participant_data$anxiety),
    mean(participant_data$insecurity)
  ),
  sd = c(
    sd(participant_data$avoidance),
    sd(participant_data$anxiety),
    sd(participant_data$insecurity)
  )
)

participant_data <- participant_data |>
  mutate(
    avoidance_z = as.numeric(scale(avoidance)),
    anxiety_z = as.numeric(scale(anxiety)),
    insecurity_z = as.numeric(scale(insecurity))
  )

write_csv(
  participant_data,
  file.path(result_dir, "integrated_attachment_participant_data.csv")
)

write_csv(
  standardisation,
  file.path(result_dir, "attachment_predictor_standardisation.csv")
)

# ------------------------------------------------------------
# 3. Stan model
# ------------------------------------------------------------

stan_code <- '
data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1> P;
  int<lower=0, upper=1> K_side;

  matrix[N, P] X;

  array[N, T] int<lower=0, upper=1> choice_blue;
  array[N, T] int<lower=0, upper=1> losing_blue;
  array[N, T] int<lower=0, upper=1> is_volatile;
  array[N, T] int<lower=0, upper=1> blue_left;

  matrix[N, T] blue_magnitude;
  matrix[N, T] green_magnitude;
}

parameters {
  vector[2] mu_alpha_raw;
  vector<lower=0>[2] sigma_alpha_raw;
  matrix[N, 2] z_alpha;

  // Row 1: predictor effects on stable alpha
  // Row 2: predictor effects on volatile alpha
  matrix[2, P] b_alpha;

  real mu_beta_raw;
  real<lower=0> sigma_beta_raw;
  vector[N] z_beta;

  real mu_rho;
  real<lower=0> sigma_rho;
  vector[N] z_rho;

  vector[K_side] mu_side;
  vector<lower=0>[K_side] sigma_side;
  matrix[N, K_side] z_side;
}

transformed parameters {
  matrix[N, 2] alpha;
  vector[N] beta;
  vector[N] rho;
  vector[N] side_bias;

  for (n in 1:N) {
    for (k in 1:2) {
      alpha[n, k] =
        inv_logit(
          mu_alpha_raw[k]
          + X[n] * b_alpha[k]\'
          + sigma_alpha_raw[k] * z_alpha[n, k]
        );
    }
  }

  beta = exp(mu_beta_raw + sigma_beta_raw * z_beta);
  rho = mu_rho + sigma_rho * z_rho;
  side_bias = rep_vector(0, N);

  if (K_side == 1) {
    for (n in 1:N) {
      side_bias[n] =
        mu_side[1] + sigma_side[1] * z_side[n, 1];
    }
  }
}

model {
  mu_alpha_raw ~ normal(logit(0.30), 1.5);
  sigma_alpha_raw ~ normal(0, 0.8);
  to_vector(z_alpha) ~ std_normal();

  // Predictors are standardised. This is a regularising prior.
  to_vector(b_alpha) ~ normal(0, 0.5);

  mu_beta_raw ~ normal(log(3), 1.25);
  sigma_beta_raw ~ normal(0, 0.8);
  z_beta ~ std_normal();

  mu_rho ~ normal(0, 2);
  sigma_rho ~ normal(0, 1);
  z_rho ~ std_normal();

  if (K_side == 1) {
    mu_side ~ normal(0, 2);
    sigma_side ~ normal(0, 1);
    to_vector(z_side) ~ std_normal();
  }

  for (n in 1:N) {
    real belief_blue_losing = 0.5;
    int previous_choice = 0;

    for (t in 1:T) {
      real expected_loss_blue;
      real expected_loss_green;
      real decision_value;
      real current_alpha;
      real prediction_error;

      expected_loss_blue =
        belief_blue_losing * blue_magnitude[n, t];

      expected_loss_green =
        (1 - belief_blue_losing) * green_magnitude[n, t];

      decision_value =
        beta[n] * (expected_loss_green - expected_loss_blue)
        + rho[n] * previous_choice;

      if (K_side == 1) {
        decision_value +=
          side_bias[n] * (2 * blue_left[n, t] - 1);
      }

      choice_blue[n, t] ~ bernoulli_logit(decision_value);

      current_alpha =
        is_volatile[n, t] == 0
        ? alpha[n, 1]
        : alpha[n, 2];

      prediction_error =
        losing_blue[n, t] - belief_blue_losing;

      belief_blue_losing += current_alpha * prediction_error;
      previous_choice = 2 * choice_blue[n, t] - 1;
    }
  }
}

generated quantities {
  vector[P] b_delta_logit_alpha;

  vector[P] effect_alpha_stable_1sd;
  vector[P] effect_alpha_volatile_1sd;
  vector[P] effect_delta_alpha_1sd;
  vector[P] effect_delta_log_alpha_1sd;

  real alpha_stable_at_mean;
  real alpha_volatile_at_mean;
  real delta_alpha_at_mean;
  real delta_log_alpha_at_mean;

  alpha_stable_at_mean = inv_logit(mu_alpha_raw[1]);
  alpha_volatile_at_mean = inv_logit(mu_alpha_raw[2]);

  delta_alpha_at_mean =
    alpha_volatile_at_mean - alpha_stable_at_mean;

  delta_log_alpha_at_mean =
    log(alpha_volatile_at_mean)
    - log(alpha_stable_at_mean);

  for (p in 1:P) {
    real alpha_s_plus;
    real alpha_v_plus;

    b_delta_logit_alpha[p] =
      b_alpha[2, p] - b_alpha[1, p];

    alpha_s_plus =
      inv_logit(mu_alpha_raw[1] + b_alpha[1, p]);

    alpha_v_plus =
      inv_logit(mu_alpha_raw[2] + b_alpha[2, p]);

    effect_alpha_stable_1sd[p] =
      alpha_s_plus - alpha_stable_at_mean;

    effect_alpha_volatile_1sd[p] =
      alpha_v_plus - alpha_volatile_at_mean;

    effect_delta_alpha_1sd[p] =
      (alpha_v_plus - alpha_s_plus)
      - delta_alpha_at_mean;

    effect_delta_log_alpha_1sd[p] =
      (log(alpha_v_plus) - log(alpha_s_plus))
      - delta_log_alpha_at_mean;
  }
}
'

stan_file <- file.path(
  stan_dir,
  "integrated_attachment_M3_M4.stan"
)

writeLines(stan_code, stan_file)

model <- cmdstan_model(
  stan_file,
  force_recompile = FALSE
)

# ------------------------------------------------------------
# 4. Four model specifications
# ------------------------------------------------------------

predictor_sets <- list(
  attachment_dimensions = c("avoidance_z", "anxiety_z"),
  attachment_insecurity = c("insecurity_z")
)

response_models <- list(
  M3_block_alpha_stickiness = list(K_side = 0L),
  M4_block_alpha_full_response = list(K_side = 1L)
)

model_grid <- crossing(
  response_model = names(response_models),
  predictor_set = names(predictor_sets)
) |>
  mutate(
    fit_name = paste(response_model, predictor_set, sep = "__")
  )

print(model_grid, n = Inf)

# ------------------------------------------------------------
# 5. Fit models
# ------------------------------------------------------------

parallel_chains <- min(
  4L,
  max(1L, parallel::detectCores(logical = FALSE) - 1L)
)

fits <- list()

for (i in seq_len(nrow(model_grid))) {

  response_model_name <- model_grid$response_model[i]
  predictor_set_name <- model_grid$predictor_set[i]
  fit_name <- model_grid$fit_name[i]

  predictor_columns <- predictor_sets[[predictor_set_name]]
  predictor_names <- str_remove(predictor_columns, "_z$")

  X <- participant_data |>
    select(all_of(predictor_columns)) |>
    as.matrix()

  current_data <- c(
    common_stan_data,
    list(
      P = ncol(X),
      K_side = response_models[[response_model_name]]$K_side,
      X = X
    )
  )

  fit_path <- file.path(
    fit_dir,
    paste0(fit_name, "_fit.rds")
  )

  write_csv(
    tibble(
      predictor_index = seq_along(predictor_names),
      predictor = predictor_names
    ),
    file.path(
      result_dir,
      paste0(fit_name, "_predictor_lookup.csv")
    )
  )

  if (file.exists(fit_path)) {

    message("Loading completed fit: ", fit_name)
    current_fit <- readRDS(fit_path)

  } else {

    message("Fitting: ", fit_name)

    current_fit <- model$sample(
      data = current_data,
      seed = 20260723 + i,
      chains = 4,
      parallel_chains = parallel_chains,
      iter_warmup = 1500,
      iter_sampling = 1500,
      adapt_delta = 0.97,
      max_treedepth = 13,
      refresh = 100
    )

    current_fit$save_object(fit_path)
  }

  fits[[fit_name]] <- current_fit
}

# ------------------------------------------------------------
# 6. Sampling diagnostics
# ------------------------------------------------------------

diagnostics <- imap_dfr(
  fits,
  function(fit, fit_name) {

    parameter_summary <- fit$summary()

    sampler_matrix <- as.matrix(
      fit$sampler_diagnostics(format = "draws_matrix")
    )

    tibble(
      fit_name = fit_name,
      maximum_rhat = max(parameter_summary$rhat, na.rm = TRUE),
      minimum_bulk_ess =
        min(parameter_summary$ess_bulk, na.rm = TRUE),
      minimum_tail_ess =
        min(parameter_summary$ess_tail, na.rm = TRUE),
      divergences =
        sum(sampler_matrix[, "divergent__"]),
      maximum_treedepth_hits =
        sum(sampler_matrix[, "treedepth__"] >= 13)
    )
  }
) |>
  left_join(model_grid, by = "fit_name") |>
  relocate(response_model, predictor_set, .after = fit_name)

print(diagnostics, n = Inf)

write_csv(
  diagnostics,
  file.path(
    result_dir,
    "integrated_attachment_sampling_diagnostics.csv"
  )
)

# ------------------------------------------------------------
# 7. Posterior effect summaries
# ------------------------------------------------------------

summarise_draws <- function(x) {
  tibble(
    mean = mean(x),
    median = median(x),
    sd = sd(x),
    lower_95 = unname(quantile(x, 0.025)),
    upper_95 = unname(quantile(x, 0.975)),
    probability_positive = mean(x > 0),
    probability_negative = mean(x < 0),
    probability_direction =
      max(mean(x > 0), mean(x < 0))
  )
}

effect_summary <- list()

for (i in seq_len(nrow(model_grid))) {

  response_model_name <- model_grid$response_model[i]
  predictor_set_name <- model_grid$predictor_set[i]
  fit_name <- model_grid$fit_name[i]

  predictor_names <- predictor_sets[[predictor_set_name]] |>
    str_remove("_z$")

  draw_matrix <- as.matrix(
    fits[[fit_name]]$draws(
      variables = c(
        "b_alpha",
        "b_delta_logit_alpha",
        "effect_alpha_stable_1sd",
        "effect_alpha_volatile_1sd",
        "effect_delta_alpha_1sd",
        "effect_delta_log_alpha_1sd"
      ),
      format = "draws_matrix"
    )
  )

  targets <- tibble(
    target = c(
      "stable_logit_alpha",
      "volatile_logit_alpha",
      "relative_difference_logit_alpha",
      "stable_alpha_change_per_1SD",
      "volatile_alpha_change_per_1SD",
      "relative_difference_alpha_change_per_1SD",
      "relative_difference_log_alpha_change_per_1SD"
    ),
    template = c(
      "b_alpha[1,%d]",
      "b_alpha[2,%d]",
      "b_delta_logit_alpha[%d]",
      "effect_alpha_stable_1sd[%d]",
      "effect_alpha_volatile_1sd[%d]",
      "effect_delta_alpha_1sd[%d]",
      "effect_delta_log_alpha_1sd[%d]"
    ),
    scale = c(
      "logit(alpha)",
      "logit(alpha)",
      "difference in logit(alpha)",
      "raw alpha",
      "raw alpha",
      "difference in raw alpha",
      "difference in log(alpha)"
    )
  )

  current_results <- list()

  for (p in seq_along(predictor_names)) {
    for (j in seq_len(nrow(targets))) {

      variable_name <- sprintf(targets$template[j], p)

      current_results[[paste(p, j, sep = "_")]] <-
        summarise_draws(draw_matrix[, variable_name]) |>
        mutate(
          fit_name = fit_name,
          response_model = response_model_name,
          predictor_set = predictor_set_name,
          predictor = predictor_names[p],
          target = targets$target[j],
          scale = targets$scale[j],
          .before = 1
        )
    }
  }

  effect_summary[[fit_name]] <- bind_rows(current_results)
}

effect_summary <- bind_rows(effect_summary) |>
  arrange(response_model, predictor_set, predictor, target)

write_csv(
  effect_summary,
  file.path(
    result_dir,
    "integrated_attachment_effects_summary.csv"
  )
)

primary_effects <- effect_summary |>
  filter(
    target %in% c(
      "stable_logit_alpha",
      "volatile_logit_alpha",
      "relative_difference_logit_alpha"
    )
  )

print(
  primary_effects |>
    select(
      response_model,
      predictor_set,
      predictor,
      target,
      mean,
      lower_95,
      upper_95,
      probability_positive,
      probability_negative,
      probability_direction
    ),
  n = Inf
)

write_csv(
  primary_effects,
  file.path(
    result_dir,
    "integrated_attachment_primary_coefficients.csv"
  )
)

message(
  "\nIntegrated models completed.\n",
  "Primary results: ",
  file.path(
    result_dir,
    "integrated_attachment_primary_coefficients.csv"
  )
)
