# ============================================================
# HIERARCHICAL BAYESIAN RESCORLA-WAGNER MODELS
# Models: M1, M2, M3, M4, M7
# Estimation: hierarchical Bayesian MCMC with cmdstanr
# ============================================================

# ------------------------------------------------------------
# 0. Packages
# ------------------------------------------------------------

library(cmdstanr)
library(tidyverse)
library(posterior)
library(loo)

# ------------------------------------------------------------
# 1. Output directories
# ------------------------------------------------------------

stan_directory <-
  "models/stan"

fit_directory <-
  "data/processed/hierarchical_rw_fits"

result_directory <-
  "results/hierarchical_rw_models"

dir.create(
  stan_directory,
  recursive = TRUE,
  showWarnings = FALSE
)

dir.create(
  fit_directory,
  recursive = TRUE,
  showWarnings = FALSE
)

dir.create(
  result_directory,
  recursive = TRUE,
  showWarnings = FALSE
)


# ------------------------------------------------------------
# 2. Load and validate trial-level data
# ------------------------------------------------------------

trials <- readRDS(
  "data/processed/learning_trials_clean.rds"
) |>
  arrange(
    participant_id,
    trial
  ) |>
  mutate(
    participant_id =
      as.character(participant_id),
    
    is_volatile =
      as.integer(condition != "Stable")
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

missing_columns <- setdiff(
  required_columns,
  names(trials)
)

if (length(missing_columns) > 0) {
  stop(
    "Missing columns: ",
    paste(missing_columns, collapse = ", ")
  )
}


# Check trial counts
trial_counts <- trials |>
  count(
    participant_id,
    name = "n_trials"
  )

print(
  trial_counts |>
    count(n_trials),
  n = Inf
)

if (n_distinct(trial_counts$n_trials) != 1) {
  stop(
    "Participants do not all have the same number of trials."
  )
}

# ------------------------------------------------------------
# 3. Participant index
# ------------------------------------------------------------

participant_lookup <- trials |>
  distinct(participant_id) |>
  arrange(participant_id) |>
  mutate(
    subject_index = row_number()
  )

write_csv(
  participant_lookup,
  file.path(
    result_directory,
    "participant_index_lookup.csv"
  )
)


stan_trials <- trials |>
  inner_join(
    participant_lookup,
    by = "participant_id"
  ) |>
  arrange(
    subject_index,
    trial
  )


N_subjects <-
  nrow(participant_lookup)

N_trials <-
  unique(trial_counts$n_trials)

stopifnot(
  N_subjects == 65,
  N_trials == 120,
  nrow(stan_trials) ==
    N_subjects * N_trials
)

# ------------------------------------------------------------
# 4. Convert trial variables to participant-by-trial matrices
# ------------------------------------------------------------

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
  
  choice_blue =
    make_integer_matrix(
      stan_trials$choice_blue
    ),
  
  losing_blue =
    make_integer_matrix(
      stan_trials$losing_blue
    ),
  
  is_volatile =
    make_integer_matrix(
      stan_trials$is_volatile
    ),
  
  blue_left =
    make_integer_matrix(
      stan_trials$blue_left
    ),
  
  blue_magnitude =
    make_numeric_matrix(
      stan_trials$blue_magnitude_scaled
    ),
  
  green_magnitude =
    make_numeric_matrix(
      stan_trials$green_magnitude_scaled
    )
)


# Basic data checks
stopifnot(
  all(
    common_stan_data$choice_blue %in% c(0L, 1L)
  ),
  
  all(
    common_stan_data$losing_blue %in% c(0L, 1L)
  ),
  
  all(
    common_stan_data$is_volatile %in% c(0L, 1L)
  ),
  
  all(
    common_stan_data$blue_left %in% c(0L, 1L)
  )
)

# ------------------------------------------------------------
# 5. Stan model
# ------------------------------------------------------------

stan_code <- '

data {

  int<lower=1> N;
  int<lower=1> T;

  // One alpha or separate stable/volatile alphas
  int<lower=1, upper=2> K_alpha;

  // Zero or one hierarchical stickiness parameter
  int<lower=0, upper=1> K_rho;

  // Zero or one hierarchical side-bias parameter
  int<lower=0, upper=1> K_side;

  array[N, T] int<lower=0, upper=1> choice_blue;
  array[N, T] int<lower=0, upper=1> losing_blue;
  array[N, T] int<lower=0, upper=1> is_volatile;
  array[N, T] int<lower=0, upper=1> blue_left;

  matrix[N, T] blue_magnitude;
  matrix[N, T] green_magnitude;
}


parameters {

  // ----------------------------------------------------------
  // Learning rates on unconstrained logit scale
  // ----------------------------------------------------------

  vector[K_alpha] mu_alpha_raw;

  vector<lower=0>[K_alpha]
    sigma_alpha_raw;

  matrix[N, K_alpha]
    z_alpha;


  // ----------------------------------------------------------
  // Inverse temperature on log scale
  // ----------------------------------------------------------

  real mu_beta_raw;

  real<lower=0>
    sigma_beta_raw;

  vector[N]
    z_beta;


  // ----------------------------------------------------------
  // Stickiness
  // Empty when K_rho = 0
  // ----------------------------------------------------------

  vector[K_rho]
    mu_rho;

  vector<lower=0>[K_rho]
    sigma_rho;

  matrix[N, K_rho]
    z_rho;


  // ----------------------------------------------------------
  // Side bias
  // Empty when K_side = 0
  // ----------------------------------------------------------

  vector[K_side]
    mu_side;

  vector<lower=0>[K_side]
    sigma_side;

  matrix[N, K_side]
    z_side;
}


transformed parameters {

  matrix[N, K_alpha]
    alpha;

  vector[N]
    beta;

  vector[N]
    rho;

  vector[N]
    side_bias;


  // Hierarchical non-centred learning rates
  for (n in 1:N) {

    for (k in 1:K_alpha) {

      alpha[n, k] =
        inv_logit(
          mu_alpha_raw[k] +
          sigma_alpha_raw[k] *
          z_alpha[n, k]
        );
    }
  }


  // Hierarchical inverse temperature
  beta =
    exp(
      mu_beta_raw +
      sigma_beta_raw *
      z_beta
    );


  // Defaults for models without these parameters
  rho =
    rep_vector(0, N);

  side_bias =
    rep_vector(0, N);


  if (K_rho == 1) {

    for (n in 1:N) {

      rho[n] =
        mu_rho[1] +
        sigma_rho[1] *
        z_rho[n, 1];
    }
  }


  if (K_side == 1) {

    for (n in 1:N) {

      side_bias[n] =
        mu_side[1] +
        sigma_side[1] *
        z_side[n, 1];
    }
  }
}


model {

  // ----------------------------------------------------------
  // Group-level priors
  // ----------------------------------------------------------

  // Learning rate prior centred approximately around alpha = .30
  mu_alpha_raw ~
    normal(
      logit(0.30),
      1.5
    );

  sigma_alpha_raw ~
    normal(
      0,
      0.8
    );

  to_vector(z_alpha) ~
    std_normal();


  // Inverse temperature prior
  mu_beta_raw ~
    normal(
      log(3),
      1.25
    );

  sigma_beta_raw ~
    normal(
      0,
      0.8
    );

  z_beta ~
    std_normal();


  if (K_rho == 1) {

    mu_rho ~
      normal(
        0,
        2
      );

    sigma_rho ~
      normal(
        0,
        1
      );

    to_vector(z_rho) ~
      std_normal();
  }


  if (K_side == 1) {

    mu_side ~
      normal(
        0,
        2
      );

    sigma_side ~
      normal(
        0,
        1
      );

    to_vector(z_side) ~
      std_normal();
  }


  // ----------------------------------------------------------
  // Trial-wise learning and choice likelihood
  // ----------------------------------------------------------

  for (n in 1:N) {

    real belief_blue_losing;
    int previous_choice;

    belief_blue_losing = 0.5;
    previous_choice = 0;


    for (t in 1:T) {

      real expected_loss_blue;
      real expected_loss_green;
      real decision_value;
      real prediction_error;
      real current_alpha;


      expected_loss_blue =
        belief_blue_losing *
        blue_magnitude[n, t];


      expected_loss_green =
        (1 - belief_blue_losing) *
        green_magnitude[n, t];


      // Positive values favour choosing blue
      decision_value =
        beta[n] *
        (
          expected_loss_green -
          expected_loss_blue
        );


      // Choice stickiness
      if (K_rho == 1) {

        decision_value +=
          rho[n] *
          previous_choice;
      }


      // Positive value indicates preference for left option
      if (K_side == 1) {

        decision_value +=
          side_bias[n] *
          (
            2 * blue_left[n, t] - 1
          );
      }


      choice_blue[n, t] ~
        bernoulli_logit(
          decision_value
        );


      // Select current learning rate
      if (K_alpha == 1) {

        current_alpha =
          alpha[n, 1];

      } else {

        if (is_volatile[n, t] == 0) {

          current_alpha =
            alpha[n, 1];

        } else {

          current_alpha =
            alpha[n, 2];
        }
      }


      // Update belief after feedback
      prediction_error =
        losing_blue[n, t] -
        belief_blue_losing;


      belief_blue_losing +=
        current_alpha *
        prediction_error;


      // Blue = +1, green = -1
      previous_choice =
        2 * choice_blue[n, t] - 1;
    }
  }
}


generated quantities {

  vector[N * T]
    log_lik;

  array[N, T]
    int y_rep;


  // These variables simplify later extraction
  matrix[N, 2]
    alpha_block;

  vector[N]
    delta_alpha;

  vector[N]
    delta_log_alpha;


  int observation_index;

  observation_index = 1;


  for (n in 1:N) {

    real belief_blue_losing;
    int previous_choice;


    alpha_block[n, 1] =
      alpha[n, 1];


    if (K_alpha == 2) {

      alpha_block[n, 2] =
        alpha[n, 2];

    } else {

      alpha_block[n, 2] =
        alpha[n, 1];
    }


    delta_alpha[n] =
      alpha_block[n, 2] -
      alpha_block[n, 1];


    delta_log_alpha[n] =
      log(alpha_block[n, 2]) -
      log(alpha_block[n, 1]);


    belief_blue_losing = 0.5;
    previous_choice = 0;


    for (t in 1:T) {

      real expected_loss_blue;
      real expected_loss_green;
      real decision_value;
      real prediction_error;
      real current_alpha;


      expected_loss_blue =
        belief_blue_losing *
        blue_magnitude[n, t];


      expected_loss_green =
        (1 - belief_blue_losing) *
        green_magnitude[n, t];


      decision_value =
        beta[n] *
        (
          expected_loss_green -
          expected_loss_blue
        );


      if (K_rho == 1) {

        decision_value +=
          rho[n] *
          previous_choice;
      }


      if (K_side == 1) {

        decision_value +=
          side_bias[n] *
          (
            2 * blue_left[n, t] - 1
          );
      }


      log_lik[observation_index] =
        bernoulli_logit_lpmf(
          choice_blue[n, t] |
          decision_value
        );


      y_rep[n, t] =
        bernoulli_logit_rng(
          decision_value
        );


      if (K_alpha == 1) {

        current_alpha =
          alpha[n, 1];

      } else {

        if (is_volatile[n, t] == 0) {

          current_alpha =
            alpha[n, 1];

        } else {

          current_alpha =
            alpha[n, 2];
        }
      }


      prediction_error =
        losing_blue[n, t] -
        belief_blue_losing;


      belief_blue_losing +=
        current_alpha *
        prediction_error;


      previous_choice =
        2 * choice_blue[n, t] - 1;


      observation_index += 1;
    }
  }
}
'


stan_file <- file.path(
  stan_directory,
  "hierarchical_expected_loss_rw.stan"
)

writeLines(
  stan_code,
  con = stan_file
)

# ------------------------------------------------------------
# 6. Compile Stan model
# ------------------------------------------------------------

hierarchical_rw_model <- cmdstan_model(
  stan_file,
  force_recompile = FALSE
)


# ------------------------------------------------------------
# 7. Model specifications
# ------------------------------------------------------------

hierarchical_model_specs <- list(
  
  # One alpha + beta
  M1_single_alpha = list(
    K_alpha = 1L,
    K_rho = 0L,
    K_side = 0L
  ),
  
  # Stable alpha + volatile alpha + beta
  M2_block_alpha = list(
    K_alpha = 2L,
    K_rho = 0L,
    K_side = 0L
  ),
  
  # Stable/volatile alpha + beta + stickiness
  M3_block_alpha_stickiness = list(
    K_alpha = 2L,
    K_rho = 1L,
    K_side = 0L
  ),
  
  # Stable/volatile alpha + beta + stickiness + side bias
  M4_block_alpha_full_response = list(
    K_alpha = 2L,
    K_rho = 1L,
    K_side = 1L
  ),
  
  # One alpha + beta + stickiness + side bias
  M7_single_alpha_full_response = list(
    K_alpha = 1L,
    K_rho = 1L,
    K_side = 1L
  )
)


# ------------------------------------------------------------
# 8. Sampling settings
# ------------------------------------------------------------

available_cores <-
  parallel::detectCores(
    logical = FALSE
  )

parallel_chains <-
  min(
    4L,
    available_cores
  )

sampling_settings <- list(
  seed = 20260721,
  chains = 4,
  parallel_chains = parallel_chains,
  iter_warmup = 1000,
  iter_sampling = 1000,
  adapt_delta = 0.95,
  max_treedepth = 12,
  refresh = 100
)


# ------------------------------------------------------------
# 9. Fit the five hierarchical models
# Checkpoint after each completed model
# ------------------------------------------------------------

hierarchical_fits <- list()


for (
  model_name in names(
    hierarchical_model_specs
  )
) {
  
  specification <-
    hierarchical_model_specs[[model_name]]
  
  
  fitted_object_path <- file.path(
    fit_directory,
    paste0(
      model_name,
      "_hierarchical_fit.rds"
    )
  )
  
  
  # Load previously completed fit
  if (file.exists(fitted_object_path)) {
    
    message(
      "Loading completed model: ",
      model_name
    )
    
    hierarchical_fits[[model_name]] <-
      readRDS(
        fitted_object_path
      )
    
    next
  }
  
  
  message(
    "Fitting hierarchical ",
    model_name,
    "..."
  )
  
  
  current_stan_data <- c(
    common_stan_data,
    specification
  )
  
  
  current_fit <-
    hierarchical_rw_model$sample(
      
      data =
        current_stan_data,
      
      seed =
        sampling_settings$seed,
      
      chains =
        sampling_settings$chains,
      
      parallel_chains =
        sampling_settings$parallel_chains,
      
      iter_warmup =
        sampling_settings$iter_warmup,
      
      iter_sampling =
        sampling_settings$iter_sampling,
      
      adapt_delta =
        sampling_settings$adapt_delta,
      
      max_treedepth =
        sampling_settings$max_treedepth,
      
      refresh =
        sampling_settings$refresh
    )
  
  
  # Safely save the completed CmdStanMCMC object
  current_fit$save_object(
    file =
      fitted_object_path
  )
  
  
  hierarchical_fits[[model_name]] <-
    current_fit
  
  
  message(
    "Completed and saved ",
    model_name
  )
}


# Save the list of paths rather than duplicating all fit objects
fit_manifest <- tibble(
  model =
    names(hierarchical_model_specs),
  
  fit_path =
    file.path(
      fit_directory,
      paste0(
        names(hierarchical_model_specs),
        "_hierarchical_fit.rds"
      )
    )
)

write_csv(
  fit_manifest,
  file.path(
    result_directory,
    "hierarchical_fit_manifest.csv"
  )
)


# ------------------------------------------------------------
# 10. Convergence and sampling diagnostics
# ------------------------------------------------------------

diagnostic_summary <- imap_dfr(
  hierarchical_fits,
  function(fit, model_name) {
    
    parameter_summary <-
      fit$summary()
    
    tibble(
      model =
        model_name,
      
      maximum_Rhat =
        max(
          parameter_summary$rhat,
          na.rm = TRUE
        ),
      
      minimum_bulk_ESS =
        min(
          parameter_summary$ess_bulk,
          na.rm = TRUE
        ),
      
      minimum_tail_ESS =
        min(
          parameter_summary$ess_tail,
          na.rm = TRUE
        )
    )
  }
)

print(
  diagnostic_summary,
  n = Inf
)

write_csv(
  diagnostic_summary,
  file.path(
    result_directory,
    "hierarchical_sampling_diagnostics.csv"
  )
)


# Print CmdStan sampler diagnostics separately
for (
  model_name in names(hierarchical_fits)
) {
  
  cat(
    "\n\n==============================\n",
    model_name,
    "\n==============================\n"
  )
  
  print(
    hierarchical_fits[[
      model_name
    ]]$diagnostic_summary()
  )
}


# ------------------------------------------------------------
# 11. posterior parameter summaries
# ------------------------------------------------------------

posterior_parameter_summary <- imap_dfr(
  hierarchical_fits,
  function(fit, model_name) {
    
    specification <-
      hierarchical_model_specs[[model_name]]
    
    # Parameters present in every model
    variables_this_model <- c(
      "mu_alpha_raw",
      "sigma_alpha_raw",
      "mu_beta_raw",
      "sigma_beta_raw",
      "alpha_block",
      "beta",
      "delta_alpha",
      "delta_log_alpha"
    )
    
    # Add stickiness only for M3, M4 and M7
    if (specification$K_rho == 1L) {
      
      variables_this_model <- c(
        variables_this_model,
        "mu_rho",
        "sigma_rho",
        "rho"
      )
    }
    
    # Add side bias only for M4 and M7
    if (specification$K_side == 1L) {
      
      variables_this_model <- c(
        variables_this_model,
        "mu_side",
        "sigma_side",
        "side_bias"
      )
    }
    
    message(
      model_name,
      ": ",
      paste(
        variables_this_model,
        collapse = ", "
      )
    )
    
    fit$summary(
      variables = variables_this_model
    ) |>
      mutate(
        model = model_name,
        .before = 1
      )
  }
)

print(
  posterior_parameter_summary,
  n = 100
)

write_csv(
  posterior_parameter_summary,
  file.path(
    result_directory,
    "hierarchical_posterior_parameter_summary.csv"
  )
)

# ------------------------------------------------------------
# 12. Group-level summaries on interpretable scales
# ------------------------------------------------------------

group_parameter_summary <- imap_dfr(
  hierarchical_fits,
  function(fit, model_name) {
    
    specification <-
      hierarchical_model_specs[[model_name]]
    
    alpha_draws <-
      fit$draws(
        variables =
          "mu_alpha_raw",
        format =
          "draws_df"
      ) |>
      as_tibble()
    
    
    stable_alpha <-
      plogis(
        alpha_draws[[
          "mu_alpha_raw[1]"
        ]]
      )
    
    
    volatile_alpha <- if (
      specification$K_alpha == 2L
    ) {
      
      plogis(
        alpha_draws[[
          "mu_alpha_raw[2]"
        ]]
      )
      
    } else {
      
      stable_alpha
    }
    
    
    beta_draws <-
      fit$draws(
        variables =
          "mu_beta_raw",
        format =
          "draws_df"
      ) |>
      as_tibble()
    
    
    group_beta <-
      exp(
        beta_draws$mu_beta_raw
      )
    
    
    tibble(
      model =
        model_name,
      
      parameter =
        c(
          "group_alpha_stable",
          "group_alpha_volatile",
          "group_delta_alpha",
          "group_beta"
        ),
      
      mean =
        c(
          mean(stable_alpha),
          mean(volatile_alpha),
          mean(
            volatile_alpha -
              stable_alpha
          ),
          mean(group_beta)
        ),
      
      median =
        c(
          median(stable_alpha),
          median(volatile_alpha),
          median(
            volatile_alpha -
              stable_alpha
          ),
          median(group_beta)
        ),
      
      lower_95 =
        c(
          quantile(stable_alpha, .025),
          quantile(volatile_alpha, .025),
          quantile(
            volatile_alpha -
              stable_alpha,
            .025
          ),
          quantile(group_beta, .025)
        ),
      
      upper_95 =
        c(
          quantile(stable_alpha, .975),
          quantile(volatile_alpha, .975),
          quantile(
            volatile_alpha -
              stable_alpha,
            .975
          ),
          quantile(group_beta, .975)
        )
    )
  }
)

print(
  group_parameter_summary,
  n = Inf
)

write_csv(
  group_parameter_summary,
  file.path(
    result_directory,
    "hierarchical_group_parameter_summary.csv"
  )
)


# ------------------------------------------------------------
# 13. PSIS-LOO model comparison
# ------------------------------------------------------------

loo_results <- imap(
  hierarchical_fits,
  function(fit, model_name) {
    
    message(
      "Calculating LOO for ",
      model_name
    )
    
    fit$loo(
      variables =
        "log_lik"
    )
  }
)

saveRDS(
  loo_results,
  file.path(
    result_directory,
    "hierarchical_LOO_results.rds"
  )
)


loo_comparison <-
  loo_compare(
    loo_results
  )

print(
  loo_comparison
)


loo_comparison_table <-
  as.data.frame(
    loo_comparison
  ) |>
  as_tibble() |>
  relocate(model)

write_csv(
  loo_comparison_table,
  file.path(
    result_directory,
    "hierarchical_LOO_model_comparison.csv"
  )
)

# ------------------------------------------------------------
# 14. Check Pareto-k diagnostics
# ------------------------------------------------------------

pareto_summary <- imap_dfr(
  loo_results,
  function(loo_object, model_name) {
    
    pareto_values <-
      loo_object$diagnostics$pareto_k
    
    tibble(
      model =
        model_name,
      
      n_good =
        sum(
          pareto_values <= .5,
          na.rm = TRUE
        ),
      
      n_ok =
        sum(
          pareto_values > .5 &
            pareto_values <= .7,
          na.rm = TRUE
        ),
      
      n_bad =
        sum(
          pareto_values > .7 &
            pareto_values <= 1,
          na.rm = TRUE
        ),
      
      n_very_bad =
        sum(
          pareto_values > 1,
          na.rm = TRUE
        ),
      
      maximum_pareto_k =
        max(
          pareto_values,
          na.rm = TRUE
        )
    )
  }
)

print(
  pareto_summary,
  n = Inf
)

write_csv(
  pareto_summary,
  file.path(
    result_directory,
    "hierarchical_LOO_pareto_diagnostics.csv"
  )
)

