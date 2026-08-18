# ============================================================
# 01_behavioural_validation.R
# ============================================================

library(tidyverse)
library(lme4)
library(emmeans)

# Load the preprocessed trial-level dataset
trials <- readRDS(
  "data/processed/learning_trials_clean.rds"
)


# ------------------------------------------------------------
# 1. Create behavioural validation variables
# ------------------------------------------------------------

behaviour <- trials |>
  group_by(participant_id) |>
  arrange(trial, .by_group = TRUE) |>
  mutate(
    
    # Colour with the objectively lower loss probability
    safer_blue = as.integer(
      blue_loss_probability <
        green_loss_probability
    ),
    
    # Did the participant choose the safer colour?
    chose_safer_colour = as.integer(
      choice_blue == safer_blue
    ),
    
    # Did the participant switch relative to previous trial?
    switched = as.integer(
      choice_blue != lag(choice_blue)
    ),
    
    previous_loss = lag(loss),
    
    # Centred trial within each phase
    trial_in_phase_c =
      trial_in_phase -
      mean(trial_in_phase)
  ) |>
  ungroup()

# basic descriptive validation
phase_by_participant <- behaviour |>
  group_by(
    participant_id,
    phase
  ) |>
  summarise(
    proportion_safer_colour =
      mean(chose_safer_colour, na.rm = TRUE),
    
    proportion_optimal =
      mean(optimal_choice, na.rm = TRUE),
    
    loss_rate =
      mean(loss, na.rm = TRUE),
    
    mean_rt_s =
      mean(reaction_time_s, na.rm = TRUE),
    
    proportion_blue =
      mean(choice_blue, na.rm = TRUE),
    
    .groups = "drop"
  )


phase_summary <- phase_by_participant |>
  group_by(phase) |>
  summarise(
    n = n(),
    
    mean_safer =
      mean(proportion_safer_colour),
    
    sd_safer =
      sd(proportion_safer_colour),
    
    mean_optimal =
      mean(proportion_optimal),
    
    sd_optimal =
      sd(proportion_optimal),
    
    mean_loss_rate =
      mean(loss_rate),
    
    mean_rt_s =
      mean(mean_rt_s),
    
    .groups = "drop"
  )

print(phase_summary)

#Trial-by-trial learning curve
trial_curve <- behaviour |>
  group_by(trial, phase) |>
  summarise(
    proportion_safer =
      mean(chose_safer_colour, na.rm = TRUE),
    
    proportion_optimal =
      mean(optimal_choice, na.rm = TRUE),
    
    se_safer =
      sd(chose_safer_colour, na.rm = TRUE) /
      sqrt(sum(!is.na(chose_safer_colour))),
    
    .groups = "drop"
  )


ggplot(
  trial_curve,
  aes(
    x = trial,
    y = proportion_safer
  )
) +
  geom_hline(
    yintercept = 0.5,
    linetype = "dashed"
  ) +
  geom_vline(
    xintercept = c(60.5, 80.5, 100.5),
    linetype = "dotted"
  ) +
  geom_line() +
  geom_smooth(
    method = "loess",
    formula = y ~ x,
    se = TRUE,
    span = 0.20
  ) +
  coord_cartesian(
    ylim = c(0, 1)
  ) +
  labs(
    x = "Trial",
    y = "Proportion choosing lower-loss-probability colour",
    title = "Probability-learning curve"
  ) +
  theme_classic()

#Adaptation within each volatile phase
volatile_curve <- behaviour |>
  filter(
    phase %in% c(
      "Volatile1",
      "Volatile2",
      "Volatile3"
    )
  ) |>
  group_by(
    phase,
    trial_in_phase
  ) |>
  summarise(
    proportion_safer =
      mean(chose_safer_colour, na.rm = TRUE),
    
    se =
      sd(chose_safer_colour, na.rm = TRUE) /
      sqrt(sum(!is.na(chose_safer_colour))),
    
    .groups = "drop"
  )


ggplot(
  volatile_curve,
  aes(
    x = trial_in_phase,
    y = proportion_safer,
    linetype = phase
  )
) +
  geom_hline(
    yintercept = 0.5,
    linetype = "dashed"
  ) +
  geom_line() +
  geom_point() +
  coord_cartesian(
    ylim = c(0, 1)
  ) +
  labs(
    x = "Trial since contingency change",
    y = "Proportion choosing safer colour",
    linetype = "Phase",
    title = "Adaptation following contingency changes"
  ) +
  theme_classic()

#Simple mixed-effects validation models
#Did performance differ across phases?
model_safer <- glmer(
  chose_safer_colour ~
    phase +
    (1 | participant_id),
  data = behaviour,
  family = binomial
)

summary(model_safer)

emmeans(
  model_safer,
  ~ phase,
  type = "response"
)

#Did participants improve within each phase?
model_adaptation <- glmer(
  chose_safer_colour ~
    phase * trial_in_phase_c +
    (1 | participant_id),
  data = behaviour,
  family = binomial
)

summary(model_adaptation)

emtrends(
  model_adaptation,
  ~ phase,
  var = "trial_in_phase_c"
)

#Feedback sensitivity
switch_summary <- behaviour |>
  filter(
    !is.na(previous_loss),
    !is.na(switched)
  ) |>
  group_by(previous_loss) |>
  summarise(
    switch_probability =
      mean(switched),
    
    n_trials = n(),
    
    .groups = "drop"
  )

print(switch_summary)


model_switch <- glmer(
  switched ~
    previous_loss +
    phase +
    (1 | participant_id),
  data = behaviour |>
    filter(
      !is.na(previous_loss),
      !is.na(switched)
    ),
  family = binomial
)

summary(model_switch)

#Identify participants with potentially problematic behaviour
participant_validation <- behaviour |>
  group_by(participant_id) |>
  summarise(
    proportion_safer =
      mean(chose_safer_colour, na.rm = TRUE),
    
    proportion_optimal =
      mean(optimal_choice, na.rm = TRUE),
    
    proportion_blue =
      mean(choice_blue, na.rm = TRUE),
    
    proportion_left =
      mean(choice_left, na.rm = TRUE),
    
    switch_rate =
      mean(switched, na.rm = TRUE),
    
    mean_rt_s =
      mean(reaction_time_s, na.rm = TRUE),
    
    .groups = "drop"
  ) |>
  mutate(
    extreme_colour_bias =
      proportion_blue < 0.05 |
      proportion_blue > 0.95,
    
    extreme_side_bias =
      proportion_left < 0.05 |
      proportion_left > 0.95
  )


participant_validation |>
  arrange(proportion_optimal) |>
  print(n = 65)

participant_validation |>
  count(
    extreme_colour_bias,
    extreme_side_bias
  )

# refing models
behaviour_refit <- behaviour |>
  mutate(
    phase = factor(
      as.character(phase),
      levels = c(
        "Stable",
        "Volatile1",
        "Volatile2",
        "Volatile3"
      )
    )
  ) |>
  group_by(participant_id, phase) |>
  mutate(
    # Ranges from -0.5 at the beginning to +0.5 at the end
    # of each phase, despite phases having different lengths
    trial_progress =
      (trial_in_phase - min(trial_in_phase)) /
      (max(trial_in_phase) - min(trial_in_phase)) -
      0.5
  ) |>
  ungroup()


model_adaptation_refit <- glmer(
  chose_safer_colour ~
    phase * trial_progress +
    (1 | participant_id),
  data = behaviour_refit,
  family = binomial,
  control = glmerControl(
    optimizer = "bobyqa",
    optCtrl = list(maxfun = 200000)
  )
)

summary(model_adaptation_refit)

emtrends(
  model_adaptation_refit,
  ~ phase,
  var = "trial_progress"
)

model_phase_refit <- glmer(
  chose_safer_colour ~
    phase +
    (1 | participant_id),
  data = behaviour_refit,
  family = binomial,
  control = glmerControl(
    optimizer = "bobyqa",
    optCtrl = list(maxfun = 200000)
  )
)

emmeans(
  model_phase_refit,
  pairwise ~ phase,
  type = "response",
  adjust = "holm"
)
