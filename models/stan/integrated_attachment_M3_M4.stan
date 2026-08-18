
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
          + X[n] * b_alpha[k]'
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

