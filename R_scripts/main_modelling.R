# ============================================================
# 02_simple_learning_models.R
# Participant-level maximum-likelihood benchmark models
# ============================================================

library(tidyverse)

trials <- readRDS(
  "data/processed/learning_trials_clean.rds"
) |>
  arrange(participant_id, trial)


# ------------------------------------------------------------
# 1. Model specifications
# ------------------------------------------------------------

model_specs <- list(
  
  M0_magnitude_only = list(
    learning = "none",
    perseveration = FALSE,
    side_bias = FALSE
  ),
  
  M1_single_alpha = list(
    learning = "single",
    perseveration = FALSE,
    side_bias = FALSE
  ),
  
  M2_block_alpha = list(
    learning = "block",
    perseveration = FALSE,
    side_bias = FALSE
  ),
  
  M3_block_alpha_stickiness = list(
    learning = "block",
    perseveration = TRUE,
    side_bias = FALSE
  ),
  
  M4_block_alpha_full_response = list(
    learning = "block",
    perseveration = TRUE,
    side_bias = TRUE
  ),
  
  M5_phase_alpha_full_response = list(
    learning = "phase",
    perseveration = TRUE,
    side_bias = TRUE
  )
)


# ------------------------------------------------------------
# 2. Number and names of parameters
# ------------------------------------------------------------

parameter_names <- function(spec) {
  
  learning_names <- switch(
    spec$learning,
    
    none = character(0),
    
    single = "alpha",
    
    block = c(
      "alpha_stable",
      "alpha_volatile"
    ),
    
    phase = c(
      "alpha_stable",
      "alpha_volatile1",
      "alpha_volatile2",
      "alpha_volatile3"
    )
  )
  
  response_names <- "beta"
  
  if (spec$perseveration) {
    response_names <- c(
      response_names,
      "rho"
    )
  }
  
  if (spec$side_bias) {
    response_names <- c(
      response_names,
      "side_bias"
    )
  }
  
  c(
    learning_names,
    response_names
  )
}


number_parameters <- function(spec) {
  length(parameter_names(spec))
}

# ------------------------------------------------------------
# 3. Transform unconstrained parameters
# ------------------------------------------------------------

decode_parameters <- function(theta, spec) {
  
  index <- 1L
  output <- numeric(0)
  
  if (spec$learning == "single") {
    
    output["alpha"] <-
      plogis(theta[index])
    
    index <- index + 1L
  }
  
  if (spec$learning == "block") {
    
    output["alpha_stable"] <-
      plogis(theta[index])
    
    output["alpha_volatile"] <-
      plogis(theta[index + 1L])
    
    index <- index + 2L
  }
  
  if (spec$learning == "phase") {
    
    output["alpha_stable"] <-
      plogis(theta[index])
    
    output["alpha_volatile1"] <-
      plogis(theta[index + 1L])
    
    output["alpha_volatile2"] <-
      plogis(theta[index + 2L])
    
    output["alpha_volatile3"] <-
      plogis(theta[index + 3L])
    
    index <- index + 4L
  }
  
  # Positive inverse temperature
  output["beta"] <-
    exp(theta[index])
  
  index <- index + 1L
  
  # Unconstrained response parameters
  if (spec$perseveration) {
    
    output["rho"] <-
      theta[index]
    
    index <- index + 1L
  }
  
  if (spec$side_bias) {
    
    output["side_bias"] <-
      theta[index]
  }
  
  output
}


# ------------------------------------------------------------
# 4. Select learning rate on a given trial
# ------------------------------------------------------------

get_alpha <- function(parameters, condition, learning_model) {
  
  if (learning_model == "none") {
    return(0)
  }
  
  if (learning_model == "single") {
    return(parameters["alpha"])
  }
  
  if (learning_model == "block") {
    
    if (condition == "Stable") {
      return(parameters["alpha_stable"])
    }
    
    return(parameters["alpha_volatile"])
  }
  
  if (learning_model == "phase") {
    
    return(
      switch(
        condition,
        
        Stable =
          parameters["alpha_stable"],
        
        Volatile1 =
          parameters["alpha_volatile1"],
        
        Volatile2 =
          parameters["alpha_volatile2"],
        
        Volatile3 =
          parameters["alpha_volatile3"]
      )
    )
  }
  
  stop("Unknown learning model.")
}


# ------------------------------------------------------------
# 5. Run model and return trial-wise values
# ------------------------------------------------------------

run_rw_model <- function(theta, data, spec) {
  
  parameters <-
    decode_parameters(theta, spec)
  
  n_trials <- nrow(data)
  
  # Belief before observing each trial's feedback
  p_blue_losing <- numeric(n_trials)
  
  prediction_error <- numeric(n_trials)
  learning_rate <- numeric(n_trials)
  
  expected_loss_blue <- numeric(n_trials)
  expected_loss_green <- numeric(n_trials)
  
  decision_variable <- numeric(n_trials)
  choice_probability_blue <- numeric(n_trials)
  
  log_likelihood_trial <- numeric(n_trials)
  
  # Initial belief: both colours equally likely to lose
  current_belief <- 0.5
  
  for (t in seq_len(n_trials)) {
    
    p_blue_losing[t] <- current_belief
    
    # Magnitudes were already scaled to the 0-1 range
    expected_loss_blue[t] <-
      current_belief *
      data$blue_magnitude_scaled[t]
    
    expected_loss_green[t] <-
      (1 - current_belief) *
      data$green_magnitude_scaled[t]
    
    # Positive values favour choosing blue
    decision_variable[t] <-
      parameters["beta"] *
      (
        expected_loss_green[t] -
          expected_loss_blue[t]
      )
    
    # Choice perseveration:
    # +1 after blue, -1 after green, 0 on first trial
    if (spec$perseveration) {
      
      previous_choice <- if (t == 1L) {
        0
      } else {
        2 * data$choice_blue[t - 1L] - 1
      }
      
      decision_variable[t] <-
        decision_variable[t] +
        parameters["rho"] *
        previous_choice
    }
    
    # Positive side_bias means a preference for the left option.
    # +1 when blue is on the left, -1 when blue is on the right.
    if (spec$side_bias) {
      
      blue_side <-
        2 * data$blue_left[t] - 1
      
      decision_variable[t] <-
        decision_variable[t] +
        parameters["side_bias"] *
        blue_side
    }
    
    choice_probability_blue[t] <-
      plogis(decision_variable[t])
    
    # Prevent log(0)
    choice_probability_blue[t] <-
      pmin(
        pmax(
          choice_probability_blue[t],
          1e-10
        ),
        1 - 1e-10
      )
    
    log_likelihood_trial[t] <-
      dbinom(
        data$choice_blue[t],
        size = 1,
        prob = choice_probability_blue[t],
        log = TRUE
      )
    
    # Prediction error about the identity of the losing colour
    prediction_error[t] <-
      data$losing_blue[t] -
      current_belief
    
    learning_rate[t] <-
      get_alpha(
        parameters,
        data$condition[t],
        spec$learning
      )
    
    # Update belief after observing feedback
    if (spec$learning != "none") {
      
      current_belief <-
        current_belief +
        learning_rate[t] *
        prediction_error[t]
    }
  }
  
  tibble(
    participant_id =
      data$participant_id,
    
    trial =
      data$trial,
    
    condition =
      data$condition,
    
    choice_blue =
      data$choice_blue,
    
    losing_blue =
      data$losing_blue,
    
    p_blue_losing =
      p_blue_losing,
    
    learning_rate =
      learning_rate,
    
    prediction_error =
      prediction_error,
    
    expected_loss_blue =
      expected_loss_blue,
    
    expected_loss_green =
      expected_loss_green,
    
    decision_variable =
      decision_variable,
    
    choice_probability_blue =
      choice_probability_blue,
    
    log_likelihood_trial =
      log_likelihood_trial
  )
}


# ------------------------------------------------------------
# 6. Negative log likelihood
# ------------------------------------------------------------

negative_log_likelihood <- function(theta, data, spec) {
  
  model_output <-
    run_rw_model(
      theta,
      data,
      spec
    )
  
  negative_log_likelihood <-
    -sum(model_output$log_likelihood_trial)
  
  if (!is.finite(negative_log_likelihood)) {
    return(1e12)
  }
  
  negative_log_likelihood
}


# ------------------------------------------------------------
# 7. Parameter bounds
# ------------------------------------------------------------

parameter_bounds <- function(spec) {
  
  names <-
    parameter_names(spec)
  
  lower <- numeric(length(names))
  upper <- numeric(length(names))
  
  for (j in seq_along(names)) {
    
    parameter <- names[j]
    
    if (str_detect(parameter, "^alpha")) {
      
      # Raw-logit bounds correspond approximately to
      # learning rates between .0003 and .9997
      lower[j] <- -8
      upper[j] <- 8
      
    } else if (parameter == "beta") {
      
      # Parameter is estimated as log(beta)
      lower[j] <- -5
      upper[j] <- 6
      
    } else {
      
      # Perseveration and side bias
      lower[j] <- -10
      upper[j] <- 10
    }
  }
  
  list(
    lower = lower,
    upper = upper
  )
}


# ------------------------------------------------------------
# 8. Fit one participant with multiple starting values
# ------------------------------------------------------------

fit_one_participant <- function(
    data,
    spec,
    model_name,
    n_starts = 10,
    seed = 123
) {
  
  set.seed(
    seed +
      as.integer(as.factor(data$participant_id[1]))
  )
  
  k <-
    number_parameters(spec)
  
  bounds <-
    parameter_bounds(spec)
  
  fits <- vector(
    mode = "list",
    length = n_starts
  )
  
  for (start in seq_len(n_starts)) {
    
    initial_values <-
      runif(
        k,
        min = -0.5,
        max = 0.5
      )
    
    # Give beta a reasonable initial range
    beta_position <-
      which(
        parameter_names(spec) == "beta"
      )
    
    initial_values[beta_position] <-
      log(
        runif(
          1,
          min = 1,
          max = 10
        )
      )
    
    fits[[start]] <-
      optim(
        par = initial_values,
        
        fn = negative_log_likelihood,
        
        data = data,
        spec = spec,
        
        method = "L-BFGS-B",
        
        lower = bounds$lower,
        upper = bounds$upper,
        
        control = list(
          maxit = 5000
        )
      )
  }
  
  objective_values <-
    map_dbl(
      fits,
      "value"
    )
  
  best_fit <-
    fits[[
      which.min(objective_values)
    ]]
  
  parameters <-
    decode_parameters(
      best_fit$par,
      spec
    )
  
  trialwise <-
    run_rw_model(
      best_fit$par,
      data,
      spec
    )
  
  log_likelihood <-
    -best_fit$value
  
  predicted_choice <-
    as.integer(
      trialwise$choice_probability_blue >= 0.5
    )
  
  tibble(
    participant_id =
      data$participant_id[1],
    
    model =
      model_name,
    
    convergence =
      best_fit$convergence,
    
    log_likelihood =
      log_likelihood,
    
    n_parameters =
      k,
    
    AIC =
      -2 * log_likelihood +
      2 * k,
    
    BIC =
      -2 * log_likelihood +
      log(nrow(data)) * k,
    
    prediction_accuracy =
      mean(
        predicted_choice ==
          data$choice_blue
      ),
    
    mean_log_loss =
      -mean(
        trialwise$log_likelihood_trial
      ),
    
    parameters =
      list(parameters),
    
    raw_parameters =
      list(best_fit$par),
    
    trialwise =
      list(trialwise)
  )
}


# ------------------------------------------------------------
# 9. Fit all models
# Fit models with a checkpoint after each model
# ------------------------------------------------------------

dir.create(
  "data/processed/model_checkpoints",
  recursive = TRUE,
  showWarnings = FALSE
)

participant_data <- trials |>
   group_by(participant_id) |>
   group_split()

model_fit_list <- imap(
  model_specs,
  function(spec, model_name) {
    
    checkpoint_path <- file.path(
      "data/processed/model_checkpoints",
      paste0(model_name, "_fits.rds")
    )
    
    # Do not refit a model that has already been completed
    if (file.exists(checkpoint_path)) {
      
      message(
        "Loading completed model: ",
        model_name
      )
      
      return(readRDS(checkpoint_path))
    }
    
    message(
      "Fitting ",
      model_name,
      "..."
    )
    
    model_results <- map_dfr(
      participant_data,
      ~ fit_one_participant(
        data = .x,
        spec = spec,
        model_name = model_name,
        n_starts = 10
      )
    )
    
    # Save immediately after this model finishes
    saveRDS(
      model_results,
      checkpoint_path
    )
    
    message(
      "Saved ",
      model_name
    )
    
    model_results
  }
)

all_fits <- bind_rows(model_fit_list)

saveRDS(
  all_fits,
  "data/processed/simple_model_fits.rds"
)

# convergence check
convergence_summary <- all_fits |>
  count(
    model,
    convergence
  )

print(convergence_summary)

#model comaprison
model_comparison <- all_fits |>
  group_by(model) |>
  summarise(
    n_participants =
      n(),
    
    total_log_likelihood =
      sum(log_likelihood),
    
    summed_AIC =
      sum(AIC),
    
    summed_BIC =
      sum(BIC),
    
    mean_prediction_accuracy =
      mean(prediction_accuracy),
    
    mean_log_loss =
      mean(mean_log_loss),
    
    .groups = "drop"
  ) |>
  arrange(summed_BIC)

print(model_comparison)

# Count participant-level winning models
winner_counts_AIC <- all_fits |>
  group_by(participant_id) |>
  slice_min(
    AIC,
    n = 1,
    with_ties = FALSE
  ) |>
  ungroup() |>
  count(
    model,
    sort = TRUE
  )

winner_counts_BIC <- all_fits |>
  group_by(participant_id) |>
  slice_min(
    BIC,
    n = 1,
    with_ties = FALSE
  ) |>
  ungroup() |>
  count(
    model,
    sort = TRUE
  )

print(winner_counts_AIC)
print(winner_counts_BIC)

#Extract parameter estimates
parameter_estimates <- all_fits |>
  select(
    participant_id,
    model,
    parameters
  ) |>
  unnest_wider(parameters)

print(
  parameter_estimates |>
    group_by(model) |>
    summarise(
      across(
        where(is.numeric),
        list(
          median = ~ median(.x, na.rm = TRUE),
          mean = ~ mean(.x, na.rm = TRUE),
          sd = ~ sd(.x, na.rm = TRUE)
        )
      ),
      .groups = "drop"
    )
)

#Calculate the stable–volatile adaptation parameter
block_parameters <- parameter_estimates |>
  filter(
    model %in% c(
      "M2_block_alpha",
      "M3_block_alpha_stickiness",
      "M4_block_alpha_full_response"
    )
  ) |>
  mutate(
    
    delta_alpha =
      alpha_volatile -
      alpha_stable,
    
    delta_logit_alpha =
      qlogis(
        pmin(
          pmax(alpha_volatile, 1e-6),
          1 - 1e-6
        )
      ) -
      qlogis(
        pmin(
          pmax(alpha_stable, 1e-6),
          1 - 1e-6
        )
      )
  )

block_parameters |>
  group_by(model) |>
  summarise(
    mean_alpha_stable =
      mean(alpha_stable),
    
    mean_alpha_volatile =
      mean(alpha_volatile),
    
    median_delta_alpha =
      median(delta_alpha),
    
    mean_delta_alpha =
      mean(delta_alpha),
    
    .groups = "drop"
  )
