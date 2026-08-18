

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

