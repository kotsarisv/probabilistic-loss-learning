# ============================================================
# 03_separate_stable_volatile_RW.R
# Separate participant-level RW-softmax fits
# ============================================================

library(tidyverse)

trials <- readRDS(
  "data/processed/learning_trials_clean.rds"
) |>
  arrange(participant_id, trial)

dir.create(
  "data/processed/separate_rw_checkpoints",
  recursive = TRUE,
  showWarnings = FALSE
)

#1. Run the RW–softmax model for one block
run_separate_rw <- function(
    theta,
    data,
    initial_belief = 0.50
) {
  
  # Transform unconstrained optimisation parameters
  alpha <- plogis(theta[1])
  beta  <- exp(theta[2])
  
  n_trials <- nrow(data)
  
  p_blue_losing <- numeric(n_trials)
  prediction_error <- numeric(n_trials)
  
  expected_loss_blue <- numeric(n_trials)
  expected_loss_green <- numeric(n_trials)
  
  decision_variable <- numeric(n_trials)
  probability_choose_blue <- numeric(n_trials)
  trial_log_likelihood <- numeric(n_trials)
  
  current_belief <- initial_belief
  
  for (t in seq_len(n_trials)) {
    
    # Belief before observing this trial's outcome
    p_blue_losing[t] <- current_belief
    
    # Expected losses using displayed magnitudes
    expected_loss_blue[t] <-
      current_belief *
      data$blue_magnitude_scaled[t]
    
    expected_loss_green[t] <-
      (1 - current_belief) *
      data$green_magnitude_scaled[t]
    
    # Positive values favour choosing blue
    decision_variable[t] <-
      beta *
      (
        expected_loss_green[t] -
          expected_loss_blue[t]
      )
    
    probability_choose_blue[t] <-
      plogis(decision_variable[t])
    
    # Avoid log(0)
    probability_choose_blue[t] <-
      pmin(
        pmax(probability_choose_blue[t], 1e-10),
        1 - 1e-10
      )
    
    trial_log_likelihood[t] <-
      dbinom(
        data$choice_blue[t],
        size = 1,
        prob = probability_choose_blue[t],
        log = TRUE
      )
    
    # Environmental outcome:
    # 1 = blue was losing
    # 0 = green was losing
    prediction_error[t] <-
      data$losing_blue[t] -
      current_belief
    
    # Update after feedback
    current_belief <-
      current_belief +
      alpha * prediction_error[t]
  }
  
  list(
    
    parameters = c(
      alpha = alpha,
      beta = beta
    ),
    
    final_belief = current_belief,
    
    trialwise = tibble(
      participant_id = data$participant_id,
      trial = data$trial,
      condition = data$condition,
      
      choice_blue = data$choice_blue,
      losing_blue = data$losing_blue,
      
      p_blue_losing = p_blue_losing,
      prediction_error = prediction_error,
      
      expected_loss_blue = expected_loss_blue,
      expected_loss_green = expected_loss_green,
      
      decision_variable = decision_variable,
      probability_choose_blue = probability_choose_blue,
      trial_log_likelihood = trial_log_likelihood
    )
  )
}

# 2. Negative log-likelihood
separate_rw_nll <- function(
    theta,
    data,
    initial_belief = 0.50
) {
  
  output <- run_separate_rw(
    theta = theta,
    data = data,
    initial_belief = initial_belief
  )
  
  nll <- -sum(
    output$trialwise$trial_log_likelihood
  )
  
  if (!is.finite(nll)) {
    return(1e12)
  }
  
  nll
}

#3. Fit one block using multiple starting values
fit_one_block <- function(
    data,
    block_name,
    initial_belief = 0.50,
    n_starts = 10,
    seed = 123
) {
  
  set.seed(seed)
  
  fits <- vector(
    mode = "list",
    length = n_starts
  )
  
  for (s in seq_len(n_starts)) {
    
    initial_theta <- c(
      # Initial alpha between approximately .15 and .75
      qlogis(runif(1, 0.15, 0.75)),
      
      # Initial beta between 1 and 15
      log(runif(1, 1, 15))
    )
    
    fits[[s]] <- optim(
      par = initial_theta,
      
      fn = separate_rw_nll,
      
      data = data,
      initial_belief = initial_belief,
      
      method = "L-BFGS-B",
      
      lower = c(
        -8,  # alpha on logit scale
        -5   # beta on log scale
      ),
      
      upper = c(
        8,
        6
      ),
      
      control = list(
        maxit = 5000
      )
    )
  }
  
  fit_values <- map_dbl(
    fits,
    "value"
  )
  
  best_fit <- fits[[
    which.min(fit_values)
  ]]
  
  output <- run_separate_rw(
    theta = best_fit$par,
    data = data,
    initial_belief = initial_belief
  )
  
  alpha <- output$parameters["alpha"]
  beta  <- output$parameters["beta"]
  
  log_likelihood <- -best_fit$value
  k <- 2
  n <- nrow(data)
  
  predicted_choice <- as.integer(
    output$trialwise$probability_choose_blue >= 0.50
  )
  
  list(
    
    summary = tibble(
      participant_id =
        as.character(data$participant_id[1]),
      
      block =
        block_name,
      
      alpha =
        as.numeric(alpha),
      
      beta =
        as.numeric(beta),
      
      initial_belief =
        initial_belief,
      
      final_belief =
        output$final_belief,
      
      log_likelihood =
        log_likelihood,
      
      AIC =
        -2 * log_likelihood + 2 * k,
      
      BIC =
        -2 * log_likelihood + log(n) * k,
      
      prediction_accuracy =
        mean(
          predicted_choice ==
            data$choice_blue
        ),
      
      mean_log_loss =
        -mean(
          output$trialwise$trial_log_likelihood
        ),
      
      convergence =
        best_fit$convergence,
      
      alpha_boundary =
        alpha < 0.01 |
        alpha > 0.99,
      
      beta_boundary =
        beta < exp(-4.9) |
        beta > exp(5.9)
    ),
    
    trialwise =
      output$trialwise
  )
}

#4. Fit the stable and volatile blocks for one participant
fit_participant_separate_blocks <- function(
    participant_data,
    n_starts = 10,
    seed = 123
) {
  
  participant_id <-
    as.character(
      participant_data$participant_id[1]
    )
  
  stable_data <- participant_data |>
    filter(condition == "Stable") |>
    arrange(trial)
  
  volatile_data <- participant_data |>
    filter(condition != "Stable") |>
    arrange(trial)
  
  stopifnot(
    nrow(stable_data) == 60,
    nrow(volatile_data) == 60
  )
  
  stable_fit <- fit_one_block(
    data = stable_data,
    block_name = "stable",
    initial_belief = 0.50,
    n_starts = n_starts,
    seed = seed
  )
  
  volatile_fit <- fit_one_block(
    data = volatile_data,
    block_name = "volatile",
    initial_belief = 0.50,
    n_starts = n_starts,
    seed = seed + 10000
  )
  
  block_summary <- bind_rows(
    stable_fit$summary,
    volatile_fit$summary
  )
  
  trialwise <- bind_rows(
    stable_fit$trialwise |>
      mutate(model_block = "stable"),
    
    volatile_fit$trialwise |>
      mutate(model_block = "volatile")
  )
  
  list(
    block_summary = block_summary,
    trialwise = trialwise
  )
}

#5. Fit all participants with checkpoints
checkpoint_path <- file.path(
  "data/processed/separate_rw_checkpoints",
  "separate_rw_participant_fits.rds"
)

trialwise_checkpoint_path <- file.path(
  "data/processed/separate_rw_checkpoints",
  "separate_rw_trialwise_fits.rds"
)


# Load existing participant-level results, when available
if (file.exists(checkpoint_path)) {
  
  block_fits <- readRDS(checkpoint_path)
  
} else {
  
  block_fits <- tibble(
    participant_id = character(),
    block = character(),
    alpha = double(),
    beta = double(),
    initial_belief = double(),
    final_belief = double(),
    log_likelihood = double(),
    AIC = double(),
    BIC = double(),
    prediction_accuracy = double(),
    mean_log_loss = double(),
    convergence = integer(),
    alpha_boundary = logical(),
    beta_boundary = logical()
  )
}


# Load existing trial-wise results, when available
if (file.exists(trialwise_checkpoint_path)) {
  
  trialwise_fits <- readRDS(
    trialwise_checkpoint_path
  )
  
} else {
  
  trialwise_fits <- tibble()
}


# IDs already fitted
completed_ids <- unique(
  as.character(block_fits$participant_id)
)

participant_list <- trials |>
  group_by(participant_id) |>
  group_split()


for (i in seq_along(participant_list)) {
  
  participant_data <- participant_list[[i]]
  
  current_id <- as.character(
    participant_data$participant_id[1]
  )
  
  if (current_id %in% completed_ids) {
    
    message(
      "Skipping completed participant: ",
      current_id
    )
    
    next
  }
  
  message(
    "Fitting participant ",
    i,
    "/",
    length(participant_list),
    ": ",
    current_id
  )
  
  current_fit <- fit_participant_separate_blocks(
    participant_data = participant_data,
    n_starts = 10,
    seed = 123 + i
  )
  
  block_fits <- bind_rows(
    block_fits,
    current_fit$block_summary
  )
  
  trialwise_fits <- bind_rows(
    trialwise_fits,
    current_fit$trialwise
  )
  
  saveRDS(
    block_fits,
    checkpoint_path
  )
  
  saveRDS(
    trialwise_fits,
    trialwise_checkpoint_path
  )
}

# 6. Create one row per participant
participant_parameters <- block_fits |>
  select(
    participant_id,
    block,
    alpha,
    beta,
    log_likelihood,
    AIC,
    BIC,
    prediction_accuracy,
    mean_log_loss,
    convergence,
    alpha_boundary,
    beta_boundary
  ) |>
  pivot_wider(
    names_from = block,
    values_from = c(
      alpha,
      beta,
      log_likelihood,
      AIC,
      BIC,
      prediction_accuracy,
      mean_log_loss,
      convergence,
      alpha_boundary,
      beta_boundary
    )
  ) |>
  mutate(
    
    # Raw learning-rate difference
    delta_alpha =
      alpha_volatile -
      alpha_stable,
    
    # Browning-style relative log learning rate
    delta_log_alpha =
      log(
        pmax(alpha_volatile, 1e-6)
      ) -
      log(
        pmax(alpha_stable, 1e-6)
      ),
    
    # Difference on the unconstrained logit scale
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
      ),
    
    # Decision-temperature differences
    delta_beta =
      beta_volatile -
      beta_stable,
    
    delta_log_beta =
      log(beta_volatile) -
      log(beta_stable),
    
    any_alpha_boundary =
      alpha_boundary_stable |
      alpha_boundary_volatile,
    
    any_beta_boundary =
      beta_boundary_stable |
      beta_boundary_volatile
  )

# 7. Save the parameter dataset
saveRDS(
  participant_parameters,
  "data/processed/separate_rw_participant_parameters.rds"
)

write_csv(
  participant_parameters,
  "data/processed/separate_rw_participant_parameters.csv",
  na = ""
)

# 8. Check convergence and parameter boundaries
participant_parameters |>
  summarise(
    n_participants = n(),
    
    stable_convergence_failures =
      sum(convergence_stable != 0),
    
    volatile_convergence_failures =
      sum(convergence_volatile != 0),
    
    alpha_boundary_cases =
      sum(any_alpha_boundary),
    
    beta_boundary_cases =
      sum(any_beta_boundary)
  ) |>
  print()

participant_parameters |>
  filter(
    convergence_stable != 0 |
      convergence_volatile != 0 |
      any_alpha_boundary |
      any_beta_boundary
  ) |>
  select(
    participant_id,
    alpha_stable,
    alpha_volatile,
    beta_stable,
    beta_volatile,
    delta_alpha,
    convergence_stable,
    convergence_volatile,
    any_alpha_boundary,
    any_beta_boundary
  ) |>
  print(n = Inf)

# 9. Summarise the estimates
participant_parameters |>
  summarise(
    mean_alpha_stable =
      mean(alpha_stable),
    
    sd_alpha_stable =
      sd(alpha_stable),
    
    mean_alpha_volatile =
      mean(alpha_volatile),
    
    sd_alpha_volatile =
      sd(alpha_volatile),
    
    mean_delta_alpha =
      mean(delta_alpha),
    
    sd_delta_alpha =
      sd(delta_alpha),
    
    median_delta_alpha =
      median(delta_alpha),
    
    mean_beta_stable =
      mean(beta_stable),
    
    mean_beta_volatile =
      mean(beta_volatile)
  ) |>
  print()

wilcox.test(
  participant_parameters$alpha_volatile,
  participant_parameters$alpha_stable,
  paired = TRUE,
  exact = FALSE
)

t.test(
  participant_parameters$alpha_volatile,
  participant_parameters$alpha_stable,
  paired = TRUE
)

# 10. Plot individual learning-rate differences
participant_parameters |>
  select(
    participant_id,
    alpha_stable,
    alpha_volatile
  ) |>
  pivot_longer(
    cols = c(
      alpha_stable,
      alpha_volatile
    ),
    names_to = "block",
    values_to = "alpha"
  ) |>
  mutate(
    block = factor(
      block,
      levels = c(
        "alpha_stable",
        "alpha_volatile"
      ),
      labels = c(
        "Stable",
        "Volatile"
      )
    )
  ) |>
  ggplot(
    aes(
      x = block,
      y = alpha,
      group = participant_id
    )
  ) +
  geom_line(alpha = 0.30) +
  geom_point() +
  labs(
    x = NULL,
    y = "Estimated learning rate",
    title = "Stable and volatile learning rates"
  ) +
  theme_classic()

ggplot(
  participant_parameters,
  aes(x = delta_log_alpha)
) +
  geom_histogram(
    bins = 20
  ) +
  geom_vline(
    xintercept = 0,
    linetype = "dashed"
  ) +
  labs(
    x = "log(alpha volatile) - log(alpha stable)",
    y = "Participants",
    title = "Relative adaptation of learning rate"
  ) +
  theme_classic()
