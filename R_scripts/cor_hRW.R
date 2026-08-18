# ============================================================
# QUESTIONNAIRE–PARAMETER CORRELATIONS
# Hierarchical models: M3, M4 and M7
# ============================================================

# ------------------------------------------------------------
# 0. Packages
# ------------------------------------------------------------

library(cmdstanr)
library(tidyverse)
library(readxl)

# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

trial_file <-
  "data/processed/learning_trials_clean.rds"

fit_directory <-
  "data/processed/hierarchical_rw_fits"

questionnaire_file <-
  "65_questionnaires.xlsx"

questionnaire_sheet <-
  "65_sept25"

result_directory <-
  "results/hierarchical_rw_models/questionnaire_correlations"

dir.create(
  result_directory,
  recursive = TRUE,
  showWarnings = FALSE
)

# Earlier analyses used delta_log_alpha.
# Change to TRUE to also test:
# delta_alpha = alpha_volatile - alpha_stable

INCLUDE_DELTA_ALPHA <- FALSE


# ------------------------------------------------------------
# 2. Load hierarchical fits
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
    fit_directory,
    paste0(
      names(model_specs),
      "_hierarchical_fit.rds"
    )
  ),
  names(model_specs)
)


missing_fits <-
  fit_paths[
    !file.exists(fit_paths)
  ]

if (length(missing_fits) > 0) {
  
  stop(
    "Missing hierarchical fit files:\n",
    paste(
      missing_fits,
      collapse = "\n"
    )
  )
}


hierarchical_fits <-
  map(
    fit_paths,
    readRDS
  )

# ------------------------------------------------------------
# 3. Reconstruct participant order used in Stan
# ------------------------------------------------------------

trials <-
  readRDS(trial_file) |>
  mutate(
    participant_id =
      as.character(participant_id)
  )

participant_lookup <-
  trials |>
  distinct(participant_id) |>
  arrange(participant_id) |>
  mutate(
    subject_index =
      row_number()
  )

print(
  participant_lookup,
  n = Inf
)


# ------------------------------------------------------------
# 4. Parameter-extraction functions
# ------------------------------------------------------------

extract_vector_parameter <- function(
    summary_table,
    stan_name,
    output_name
) {
  
  pattern <- paste0(
    "^",
    stan_name,
    "\\[(\\d+)\\]$"
  )
  
  output <-
    summary_table |>
    filter(
      str_detect(
        variable,
        pattern
      )
    ) |>
    transmute(
      
      subject_index =
        as.integer(
          str_match(
            variable,
            pattern
          )[, 2]
        ),
      
      value = mean
    )
  
  names(output)[2] <-
    output_name
  
  output
}

extract_alpha_parameter <- function(
    summary_table,
    alpha_column,
    output_name
) {
  
  pattern <- paste0(
    "^alpha\\[(\\d+),",
    alpha_column,
    "\\]$"
  )
  
  output <-
    summary_table |>
    filter(
      str_detect(
        variable,
        pattern
      )
    ) |>
    transmute(
      
      subject_index =
        as.integer(
          str_match(
            variable,
            pattern
          )[, 2]
        ),
      
      value = mean
    )
  
  names(output)[2] <-
    output_name
  
  output
}


extract_model_parameters <- function(
    fit,
    model_name,
    specification
) {
  
  variables_this_model <- c(
    "alpha",
    "beta"
  )
  
  
  if (specification$K_alpha == 2L) {
    
    variables_this_model <- c(
      variables_this_model,
      "delta_alpha",
      "delta_log_alpha"
    )
  }
  
  
  if (specification$K_rho == 1L) {
    
    variables_this_model <- c(
      variables_this_model,
      "rho"
    )
  }
  
  
  if (specification$K_side == 1L) {
    
    variables_this_model <- c(
      variables_this_model,
      "side_bias"
    )
  }
  
  
  model_summary <-
    fit$summary(
      variables =
        variables_this_model
    )
  
  
  parameter_tables <- list()
  
  
  # Single-alpha model: M7
  if (specification$K_alpha == 1L) {
    
    parameter_tables[["alpha_general"]] <-
      extract_alpha_parameter(
        summary_table =
          model_summary,
        
        alpha_column =
          1L,
        
        output_name =
          "alpha_general"
      )
    
  } else {
    
    # Block-alpha models: M3 and M4
    parameter_tables[["alpha_stable"]] <-
      extract_alpha_parameter(
        summary_table =
          model_summary,
        
        alpha_column =
          1L,
        
        output_name =
          "alpha_stable"
      )
    
    
    parameter_tables[["alpha_volatile"]] <-
      extract_alpha_parameter(
        summary_table =
          model_summary,
        
        alpha_column =
          2L,
        
        output_name =
          "alpha_volatile"
      )
    
    
    parameter_tables[["delta_log_alpha"]] <-
      extract_vector_parameter(
        summary_table =
          model_summary,
        
        stan_name =
          "delta_log_alpha",
        
        output_name =
          "delta_log_alpha"
      )
    
    
    if (isTRUE(INCLUDE_DELTA_ALPHA)) {
      
      parameter_tables[["delta_alpha"]] <-
        extract_vector_parameter(
          summary_table =
            model_summary,
          
          stan_name =
            "delta_alpha",
          
          output_name =
            "delta_alpha"
        )
    }
  }
  
  
  parameter_tables[["beta"]] <-
    extract_vector_parameter(
      summary_table =
        model_summary,
      
      stan_name =
        "beta",
      
      output_name =
        "beta"
    )
  
  
  if (specification$K_rho == 1L) {
    
    parameter_tables[["rho"]] <-
      extract_vector_parameter(
        summary_table =
          model_summary,
        
        stan_name =
          "rho",
        
        output_name =
          "rho"
      )
  }
  
  
  if (specification$K_side == 1L) {
    
    parameter_tables[["side_bias"]] <-
      extract_vector_parameter(
        summary_table =
          model_summary,
        
        stan_name =
          "side_bias",
        
        output_name =
          "side_bias"
      )
  }
  
  
  reduce(
    parameter_tables,
    full_join,
    by = "subject_index"
  ) |>
    left_join(
      participant_lookup,
      by = "subject_index"
    ) |>
    mutate(
      model =
        model_name,
      .before = 1
    ) |>
    select(
      model,
      subject_index,
      participant_id,
      everything()
    )
}


# ------------------------------------------------------------
# 5. Extract posterior participant parameters
# ------------------------------------------------------------

participant_parameters <-
  imap_dfr(
    hierarchical_fits,
    function(fit, model_name) {
      
      message(
        "Extracting parameters for ",
        model_name
      )
      
      extract_model_parameters(
        fit =
          fit,
        
        model_name =
          model_name,
        
        specification =
          model_specs[[model_name]]
      )
    }
  )


print(
  participant_parameters,
  n = 20
)


write_csv(
  participant_parameters,
  file.path(
    result_directory,
    "hierarchical_M3_M4_M7_participant_parameters.csv"
  )
)


# ------------------------------------------------------------
# 6. Read questionnaires
# ------------------------------------------------------------

questionnaire_raw <-
  read_excel(
    questionnaire_file,
    sheet =
      questionnaire_sheet
  )


required_questionnaire_columns <- c(
  "subIDs",
  "avoid_avg",
  "anxiety_avg",
  "aaq_total_avg",
  "Reappraisal_Mean",
  "Suppression_Mean",
  "MCare",
  "MOverprotection",
  "DCare",
  "DOverprotection"
)


missing_questionnaire_columns <-
  setdiff(
    required_questionnaire_columns,
    names(questionnaire_raw)
  )


if (
  length(
    missing_questionnaire_columns
  ) > 0
) {
  
  stop(
    "Missing questionnaire columns: ",
    paste(
      missing_questionnaire_columns,
      collapse = ", "
    )
  )
}


normalise_id <- function(x) {
  
  x |>
    as.character() |>
    str_trim() |>
    str_replace(
      "\\.0$",
      ""
    )
}


to_numeric <- function(x) {
  
  suppressWarnings(
    as.numeric(
      as.character(x)
    )
  )
}


questionnaires <-
  questionnaire_raw |>
  transmute(
    
    participant_id =
      normalise_id(subIDs),
    
    avoidance =
      to_numeric(avoid_avg),
    
    anxiety =
      to_numeric(anxiety_avg),
    
    insecurity =
      to_numeric(aaq_total_avg),
    
    reappraisal =
      to_numeric(Reappraisal_Mean),
    
    suppression =
      to_numeric(Suppression_Mean),
    
    maternal_care =
      to_numeric(MCare),
    
    maternal_overprotection =
      to_numeric(MOverprotection),
    
    paternal_care =
      to_numeric(DCare),
    
    paternal_overprotection =
      to_numeric(DOverprotection)
  ) |>
  filter(
    !is.na(participant_id),
    participant_id != ""
  )


# Check duplicate questionnaire IDs

duplicate_questionnaire_ids <-
  questionnaires |>
  count(
    participant_id
  ) |>
  filter(
    n > 1
  )


if (
  nrow(
    duplicate_questionnaire_ids
  ) > 0
) {
  
  print(
    duplicate_questionnaire_ids
  )
  
  stop(
    "Duplicate participant IDs found in questionnaire data."
  )
}


# ------------------------------------------------------------
# 7. Merge questionnaires and model parameters
# ------------------------------------------------------------

participant_parameters <-
  participant_parameters |>
  mutate(
    participant_id =
      normalise_id(
        participant_id
      )
  )


analysis_data <-
  participant_parameters |>
  left_join(
    questionnaires,
    by = "participant_id"
  )


matching_summary <-
  analysis_data |>
  group_by(model) |>
  summarise(
    
    n_model_participants =
      n(),
    
    n_matched_questionnaires =
      sum(
        !is.na(avoidance)
      ),
    
    .groups = "drop"
  )


print(
  matching_summary,
  n = Inf
)


write_csv(
  matching_summary,
  file.path(
    result_directory,
    "questionnaire_matching_summary.csv"
  )
)


write_csv(
  analysis_data,
  file.path(
    result_directory,
    "hierarchical_M3_M4_M7_parameters_with_questionnaires.csv"
  )
)


# ------------------------------------------------------------
# 8. Questionnaire families
# ------------------------------------------------------------

trait_dictionary <-
  tribble(
    
    ~predictor,
    ~predictor_label,
    ~trait_family,
    
    "avoidance",
    "Attachment avoidance",
    "Attachment",
    
    "anxiety",
    "Attachment anxiety",
    "Attachment",
    
    "insecurity",
    "Attachment insecurity",
    "Attachment",
    
    "reappraisal",
    "Cognitive reappraisal",
    "Emotion regulation",
    
    "suppression",
    "Expressive suppression",
    "Emotion regulation",
    
    "maternal_care",
    "Maternal care",
    "Parental bonding",
    
    "maternal_overprotection",
    "Maternal overprotection",
    "Parental bonding",
    
    "paternal_care",
    "Paternal care",
    "Parental bonding",
    
    "paternal_overprotection",
    "Paternal overprotection",
    "Parental bonding"
  )


# ------------------------------------------------------------
# 9. Parameters tested for each model
# ------------------------------------------------------------

block_model_outcomes <- c(
  "alpha_stable",
  "alpha_volatile",
  "delta_log_alpha",
  "beta",
  "rho"
)


if (isTRUE(INCLUDE_DELTA_ALPHA)) {
  
  block_model_outcomes <- c(
    "alpha_stable",
    "alpha_volatile",
    "delta_alpha",
    "delta_log_alpha",
    "beta",
    "rho"
  )
}


model_outcomes <- list(
  
  M3_block_alpha_stickiness =
    block_model_outcomes,
  
  M4_block_alpha_full_response =
    c(
      block_model_outcomes,
      "side_bias"
    ),
  
  M7_single_alpha_full_response =
    c(
      "alpha_general",
      "beta",
      "rho",
      "side_bias"
    )
)


# ------------------------------------------------------------
# 10. Spearman-correlation function
# ------------------------------------------------------------

run_spearman <- function(
    data,
    predictor,
    outcome
) {
  
  x <-
    data[[predictor]]
  
  y <-
    data[[outcome]]
  
  
  complete_cases <-
    complete.cases(
      x,
      y
    )
  
  
  x <-
    x[complete_cases]
  
  y <-
    y[complete_cases]
  
  
  n_complete <-
    length(x)
  
  
  if (
    n_complete < 3 ||
    length(unique(x)) < 2 ||
    length(unique(y)) < 2
  ) {
    
    return(
      tibble(
        n = n_complete,
        rho = NA_real_,
        p_value = NA_real_
      )
    )
  }
  
  
  test <-
    suppressWarnings(
      cor.test(
        x,
        y,
        method = "spearman",
        exact = FALSE
      )
    )
  
  
  tibble(
    
    n =
      n_complete,
    
    rho =
      unname(
        test$estimate
      ),
    
    p_value =
      test$p.value
  )
}


# ------------------------------------------------------------
# 11. Run correlations
# ------------------------------------------------------------

correlation_results_raw <-
  imap_dfr(
    model_outcomes,
    function(outcomes, model_name) {
      
      model_data <-
        analysis_data |>
        filter(
          model == model_name
        )
      
      
      correlation_grid <-
        crossing(
          
          predictor =
            trait_dictionary$predictor,
          
          outcome =
            outcomes
        )
      
      
      pmap_dfr(
        correlation_grid,
        function(predictor, outcome) {
          
          run_spearman(
            data =
              model_data,
            
            predictor =
              predictor,
            
            outcome =
              outcome
          ) |>
            mutate(
              
              model =
                model_name,
              
              predictor =
                predictor,
              
              outcome =
                outcome,
              
              .before = 1
            )
        }
      )
    }
  )


# ------------------------------------------------------------
# 12. FDR correction
# ------------------------------------------------------------

correlation_results <-
  correlation_results_raw |>
  left_join(
    trait_dictionary,
    by = "predictor"
  ) |>
  
  # Correction across all correlations within each model
  group_by(model) |>
  mutate(
    p_fdr_model =
      p.adjust(
        p_value,
        method = "BH"
      )
  ) |>
  ungroup() |>
  
  # Correction separately within model and trait family
  group_by(
    model,
    trait_family
  ) |>
  mutate(
    p_fdr_family =
      p.adjust(
        p_value,
        method = "BH"
      )
  ) |>
  ungroup() |>
  
  mutate(
    
    raw_significant =
      !is.na(p_value) &
      p_value < .05,
    
    family_fdr_significant =
      !is.na(p_fdr_family) &
      p_fdr_family < .05,
    
    model_fdr_significant =
      !is.na(p_fdr_model) &
      p_fdr_model < .05
  ) |>
  
  select(
    model,
    predictor,
    predictor_label,
    outcome,
    n,
    rho,
    p_value,
    trait_family,
    p_fdr_model,
    p_fdr_family,
    raw_significant,
    family_fdr_significant,
    model_fdr_significant
  ) |>
  
  arrange(
    model,
    trait_family,
    predictor,
    outcome
  )


# ------------------------------------------------------------
# 13. Save results
# ------------------------------------------------------------

write_csv(
  correlation_results,
  file.path(
    result_directory,
    "hierarchical_M3_M4_M7_trait_parameter_correlations.csv"
  )
)

write_csv(
  correlation_results |>
    filter(
      raw_significant
    ),
  file.path(
    result_directory,
    "hierarchical_M3_M4_M7_raw_significant_correlations.csv"
  )
)

write_csv(
  correlation_results |>
    filter(
      family_fdr_significant |
        model_fdr_significant
    ),
  file.path(
    result_directory,
    "hierarchical_M3_M4_M7_FDR_significant_correlations.csv"
  )
)


# ------------------------------------------------------------
# 14. Print important results
# ------------------------------------------------------------

cat(
  "\nRaw p < .05:\n"
)


print(
  correlation_results |>
    filter(
      raw_significant
    ) |>
    select(
      model,
      trait_family,
      predictor,
      outcome,
      n,
      rho,
      p_value,
      p_fdr_family,
      p_fdr_model
    ),
  n = Inf
)


cat(
  "\nFDR-corrected p < .05:\n"
)


print(
  correlation_results |>
    filter(
      family_fdr_significant |
        model_fdr_significant
    ) |>
    select(
      model,
      trait_family,
      predictor,
      outcome,
      n,
      rho,
      p_value,
      p_fdr_family,
      p_fdr_model
    ),
  n = Inf
)


message(
  "\nCorrelations completed. Results saved in:\n",
  result_directory
)
