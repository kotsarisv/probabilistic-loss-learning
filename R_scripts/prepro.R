# ============================================================
# 00_preprocess_learning_task.R
# Probabilistic loss-learning task
# ============================================================

library(tidyverse)
library(readxl)
library(janitor)

# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------

raw_path <- "65.xlsx"
output_dir <- "data/processed"

dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)


# ------------------------------------------------------------
# 2. Helper functions
# ------------------------------------------------------------

# Convert Excel text/numeric values to numeric without warnings
to_numeric <- function(x) {
  suppressWarnings(readr::parse_double(x))
}

# Convert scientific-notation Excel IDs such as 1.3531305E7
# to character IDs such as "13531305"
to_participant_id <- function(x) {
  
  x_numeric <- suppressWarnings(as.numeric(x))
  
  ifelse(
    is.na(x_numeric),
    NA_character_,
    format(
      x_numeric,
      scientific = FALSE,
      trim = TRUE
    )
  )
}


# ------------------------------------------------------------
# 3. Import raw Gorilla export
# ------------------------------------------------------------

# Reading all columns as text prevents Excel from changing IDs
raw <- readxl::read_xlsx(
  raw_path,
  col_types = "text"
) |>
  janitor::clean_names()

glimpse(raw)


# ------------------------------------------------------------
# 4. Convert relevant columns
# ------------------------------------------------------------

dat <- raw |>
  mutate(
    
    participant_id =
      to_participant_id(participant_private_id),
    
    participant_public_id =
      as.character(participant_public_id),
    
    event_index =
      to_numeric(event_index),
    
    trial =
      to_numeric(trial_number),
    
    screen_number =
      to_numeric(screen_number),
    
    reaction_time_ms =
      to_numeric(reaction_time),
    
    spreadsheet_row =
      to_numeric(spreadsheet_row),
    
    green_magnitude =
      to_numeric(green_v),
    
    blue_magnitude =
      to_numeric(blue_v),
    
    blue_loss_probability =
      to_numeric(blue_p),
    
    green_loss_probability =
      to_numeric(green_p)
  )


# ------------------------------------------------------------
# 5. Extract choice/offer events
# ------------------------------------------------------------

offers <- dat |>
  filter(
    screen_name == "offer",
    !is.na(participant_id),
    between(trial, 1, 120)
  ) |>
  
  # Flag duplicated offer events before selecting one
  group_by(participant_id, trial) |>
  arrange(event_index, .by_group = TRUE) |>
  mutate(
    n_offer_records = n()
  ) |>
  
  # One participant restarted during trial 90.
  # Retain the final offer event from that trial.
  slice_tail(n = 1) |>
  ungroup() |>
  
  transmute(
    participant_id,
    participant_public_id,
    participant_status,
    
    trial,
    offer_event_index = event_index,
    spreadsheet_row,
    condition,
    
    choice_raw = response,
    reaction_time_ms,
    
    green_magnitude,
    blue_magnitude,
    
    blue_loss_probability,
    green_loss_probability,
    
    left_button,
    right_button,
    
    n_offer_records
  )

# ------------------------------------------------------------
# 6. Extract outcome/feedback events
# ------------------------------------------------------------

outcomes <- dat |>
  filter(
    screen_name %in% c("win", "lose"),
    !is.na(participant_id),
    between(trial, 1, 120)
  ) |>
  
  group_by(participant_id, trial) |>
  arrange(event_index, .by_group = TRUE) |>
  mutate(
    n_outcome_records = n()
  ) |>
  slice_tail(n = 1) |>
  ungroup() |>
  
  transmute(
    participant_id,
    trial,
    
    feedback_event_index = event_index,
    feedback_label = screen_name,
    
    # Gorilla's internal labels are counterintuitive:
    # "win"  = participant lost the displayed points
    # "lose" = participant did not lose points
    loss = case_when(
      feedback_label == "win"  ~ 1L,
      feedback_label == "lose" ~ 0L,
      TRUE                     ~ NA_integer_
    ),
    
    n_outcome_records
  )

# ------------------------------------------------------------
# 7. Join choice and outcome information
# ------------------------------------------------------------

trials <- offers |>
  left_join(
    outcomes,
    by = c("participant_id", "trial")
  ) |>
  arrange(participant_id, trial)

# ------------------------------------------------------------
# 8. Construct modelling variables
# ------------------------------------------------------------

trials <- trials |>
  mutate(
    
    # -------------------------
    # Choice coding
    # -------------------------
    
    choice = case_when(
      choice_raw == "blueBox.png"  ~ "blue",
      choice_raw == "greenBox.png" ~ "green",
      TRUE                         ~ NA_character_
    ),
    
    # Binary response for response models:
    # 1 = chose blue
    # 0 = chose green
    choice_blue = case_when(
      choice == "blue"  ~ 1L,
      choice == "green" ~ 0L,
      TRUE              ~ NA_integer_
    ),
    
    reaction_time_s =
      reaction_time_ms / 1000,
    
    # -------------------------
    # Screen-side coding
    # -------------------------
    
    blue_left = case_when(
      left_button == "blueBox.png"  ~ 1L,
      left_button == "greenBox.png" ~ 0L,
      TRUE                          ~ NA_integer_
    ),
    
    # 1 = participant selected the left option
    # 0 = participant selected the right option
    choice_left = case_when(
      choice_blue == 1L ~ blue_left,
      choice_blue == 0L ~ 1L - blue_left,
      TRUE              ~ NA_integer_
    ),
    
    
    # -------------------------
    # Chosen and unchosen values
    # -------------------------
    
    chosen_magnitude = case_when(
      choice_blue == 1L ~ blue_magnitude,
      choice_blue == 0L ~ green_magnitude,
      TRUE              ~ NA_real_
    ),
    
    unchosen_magnitude = case_when(
      choice_blue == 1L ~ green_magnitude,
      choice_blue == 0L ~ blue_magnitude,
      TRUE              ~ NA_real_
    ),
    
    # Scaling may help later model estimation
    blue_magnitude_scaled =
      blue_magnitude / 100,
    
    green_magnitude_scaled =
      green_magnitude / 100,
    
    chosen_magnitude_scaled =
      chosen_magnitude / 100,
    
    # -------------------------
    # Reconstruct losing colour
    # -------------------------
    
    # If blue was chosen:
    #   loss = 1 -> blue was losing
    #   loss = 0 -> green was losing
    #
    # If green was chosen:
    #   loss = 1 -> green was losing
    #   loss = 0 -> blue was losing
    #
    # Therefore, blue is the losing colour whenever
    # choice_blue and loss are equal.
    
    losing_blue =
      as.integer(choice_blue == loss),
    
    losing_colour = case_when(
      losing_blue == 1L ~ "blue",
      losing_blue == 0L ~ "green",
      TRUE              ~ NA_character_
    ),
    
    
    # -------------------------
    # Phase structure
    # -------------------------
    
    block = case_when(
      condition == "Stable" ~ "stable",
      str_detect(condition, "^Volatile") ~ "volatile",
      TRUE ~ NA_character_
    ),
    
    phase = factor(
      condition,
      levels = c(
        "Stable",
        "Volatile1",
        "Volatile2",
        "Volatile3"
      ),
      ordered = TRUE
    ),
    
    phase_number = case_when(
      condition == "Stable"    ~ 0L,
      condition == "Volatile1" ~ 1L,
      condition == "Volatile2" ~ 2L,
      condition == "Volatile3" ~ 3L,
      TRUE                     ~ NA_integer_
    ),
    
    trial_in_phase = case_when(
      trial <= 60  ~ trial,
      trial <= 80  ~ trial - 60,
      trial <= 100 ~ trial - 80,
      trial <= 120 ~ trial - 100,
      TRUE         ~ NA_real_
    ),
    
    volatile_trial = case_when(
      trial > 60 ~ trial - 60,
      TRUE       ~ NA_real_
    ),
    
    # Trials where the underlying contingency changes
    change_point = as.integer(
      trial %in% c(61, 81, 101)
    ),
    
    change_type = case_when(
      trial == 61 ~ "stable_to_volatile",
      trial %in% c(81, 101) ~ "within_volatile_reversal",
      TRUE ~ "none"
    ),
    
    
    # -------------------------
    # Observed trial loss
    # -------------------------
    
    points_lost =
      loss * chosen_magnitude,
    
    trial_payoff =
      -points_lost,
    
    
    # -------------------------
    # Objective expected loss
    # For descriptive/QC analyses only
    # -------------------------
    
    true_expected_loss_blue =
      blue_loss_probability * blue_magnitude,
    
    true_expected_loss_green =
      green_loss_probability * green_magnitude,
    
    # Positive values favour choosing blue
    true_advantage_blue =
      true_expected_loss_green -
      true_expected_loss_blue,
    
    optimal_choice_blue = case_when(
      true_expected_loss_blue <
        true_expected_loss_green ~ 1L,
      
      true_expected_loss_blue >
        true_expected_loss_green ~ 0L,
      
      TRUE ~ NA_integer_
    ),
    
    optimal_choice = case_when(
      is.na(optimal_choice_blue) ~ NA_integer_,
      TRUE ~ as.integer(
        choice_blue == optimal_choice_blue
      )
    )
  ) |>
  
  group_by(participant_id) |>
  arrange(trial, .by_group = TRUE) |>
  mutate(
    cumulative_points_lost =
      cumsum(points_lost),
    
    reconstructed_score =
      5000 - cumulative_points_lost
  ) |>
  ungroup()


# ------------------------------------------------------------
# 9. Order the final columns
# ------------------------------------------------------------

trials <- trials |>
  select(
    participant_id,
    participant_public_id,
    participant_status,
    
    trial,
    block,
    condition,
    phase,
    phase_number,
    trial_in_phase,
    volatile_trial,
    change_point,
    change_type,
    
    choice,
    choice_blue,
    choice_left,
    blue_left,
    
    reaction_time_ms,
    reaction_time_s,
    
    loss,
    losing_colour,
    losing_blue,
    
    blue_magnitude,
    green_magnitude,
    chosen_magnitude,
    unchosen_magnitude,
    
    blue_magnitude_scaled,
    green_magnitude_scaled,
    chosen_magnitude_scaled,
    
    blue_loss_probability,
    green_loss_probability,
    
    true_expected_loss_blue,
    true_expected_loss_green,
    true_advantage_blue,
    optimal_choice_blue,
    optimal_choice,
    
    points_lost,
    cumulative_points_lost,
    reconstructed_score,
    
    spreadsheet_row,
    offer_event_index,
    feedback_event_index,
    feedback_label,
    
    n_offer_records,
    n_outcome_records,
    
    everything()
  )


# ------------------------------------------------------------
# 10. Quality-control summaries
# ------------------------------------------------------------

participant_qc <- trials |>
  group_by(participant_id) |>
  summarise(
    n_trials = n(),
    
    n_missing_choices =
      sum(is.na(choice_blue)),
    
    n_missing_outcomes =
      sum(is.na(loss)),
    
    n_duplicate_offer_trials =
      sum(n_offer_records > 1),
    
    mean_rt_s =
      mean(reaction_time_s, na.rm = TRUE),
    
    proportion_blue =
      mean(choice_blue, na.rm = TRUE),
    
    proportion_loss =
      mean(loss, na.rm = TRUE),
    
    proportion_optimal =
      mean(optimal_choice, na.rm = TRUE),
    
    total_points_lost =
      sum(points_lost, na.rm = TRUE),
    
    final_score_reconstructed =
      last(reconstructed_score),
    
    .groups = "drop"
  )


dataset_qc <- trials |>
  summarise(
    n_participants =
      n_distinct(participant_id),
    
    n_rows =
      n(),
    
    minimum_trials =
      min(table(participant_id)),
    
    maximum_trials =
      max(table(participant_id)),
    
    missing_choices =
      sum(is.na(choice_blue)),
    
    missing_outcomes =
      sum(is.na(loss)),
    
    duplicated_offer_trials =
      sum(n_offer_records > 1),
    
    duplicated_outcome_trials =
      sum(n_outcome_records > 1)
  )

print(dataset_qc)

participant_qc |>
  count(
    n_trials,
    n_duplicate_offer_trials
  ) |>
  print()


# Inspect any duplicated trial records
duplicate_trial_qc <- trials |>
  filter(
    n_offer_records > 1 |
      n_outcome_records > 1
  ) |>
  select(
    participant_id,
    trial,
    condition,
    choice,
    loss,
    reaction_time_ms,
    n_offer_records,
    n_outcome_records
  )

print(duplicate_trial_qc)


# ------------------------------------------------------------
# 11. Automated integrity checks
# ------------------------------------------------------------

# Every participant should have 120 trials
stopifnot(
  all(participant_qc$n_trials == 120)
)

# Choice and outcome should be present on every trial
stopifnot(
  all(!is.na(trials$choice_blue)),
  all(!is.na(trials$loss)),
  all(!is.na(trials$losing_blue))
)

# The displayed magnitudes should sum to 100
stopifnot(
  all(
    abs(
      trials$blue_magnitude +
        trials$green_magnitude -
        100
    ) < 1e-8
  )
)

# The programmed loss probabilities should sum to one
stopifnot(
  all(
    abs(
      trials$blue_loss_probability +
        trials$green_loss_probability -
        1
    ) < 1e-8
  )
)

# Verify trial-condition structure
expected_condition <- case_when(
  trials$trial <= 60  ~ "Stable",
  trials$trial <= 80  ~ "Volatile1",
  trials$trial <= 100 ~ "Volatile2",
  trials$trial <= 120 ~ "Volatile3"
)

stopifnot(
  all(
    trials$condition ==
      expected_condition
  )
)


# ------------------------------------------------------------
# 12. Save processed files
# ------------------------------------------------------------

readr::write_csv(
  trials,
  file.path(
    output_dir,
    "learning_trials_clean.csv"
  ),
  na = ""
)

saveRDS(
  trials,
  file.path(
    output_dir,
    "learning_trials_clean.rds"
  )
)

readr::write_csv(
  participant_qc,
  file.path(
    output_dir,
    "participant_qc.csv"
  ),
  na = ""
)

readr::write_csv(
  duplicate_trial_qc,
  file.path(
    output_dir,
    "duplicate_trial_qc.csv"
  ),
  na = ""
)
