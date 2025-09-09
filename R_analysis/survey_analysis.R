library(ordinal)
library(tidyr)
library(dplyr)
library(lme4)
library(lmerTest)
library(bayestestR)
library(emmeans)
library(lsmeans)
library(performance)
library(datawizard)
library(rstudioapi) 
library(reshape2)
library(multcompView)  # Compact letter display for group differences
library(stringr)
library(purrr)
library(glmmTMB)
library(car)


"
Overall script executing all the modelling done for analyzing the survey. 
"




" ***************************************** KPT  ***************************************** "


"
Kill to Pass Test modelling: 

Does the likelihood to kill a creature in the KPT scenarios change 
depending of its specific features? (having I/C/S)?

In this analysis, we test how the presence or absence of each property 
(intentions, sensations, consciousness) affects the likelihood that its killed. 

dependent var: kill
predictors: Consciousness, Intentions, Sensations

Sensitive: to the manipulation; only people who killed at least one but not
all entities are in here. 
"


## Load Data -------------------------------------

data_path <- file.choose()  # "kill_for_test\kill_to_pass_coded_per_entity_sensitive.csv" >> SENSITIVE
save_dir <- dirname(data_path)  # deduce the saving path from this
data <- read.csv(data_path) 

data$response_id <- factor(data$response_id)
data$entity <- factor(data$entity)
data$Consciousness <- factor(data$Consciousness)
data$Intentions <- factor(data$Intentions)
data$Sensations <- factor(data$Sensations)
data$group <- factor(data$group)

## Prepare results saving ------------------------

sink_file_path <- paste0(save_dir, "/kill_for_test_glm_sensitive.txt")
sink(sink_file_path, append = FALSE)

## Analyze Data ----------------------------------

print("=================== Kill per Property =================== ")

# Hypothesis
model_h1 <- glmer(kill ~ Consciousness * Intentions * Sensations + (1| response_id),
                  data = data, family = binomial())


print("-----------------------------model summary-----------------------------")
print(summary(model_h1))

print("-----------------------------random effects-----------------------------")
random_effects <- ranef(model_h1)
print(random_effects)

print("-----------------------------fixed effects-----------------------------")
fixef_summary <- summary(model_h1)$coefficients
print(fixef_summary)


print("-----------------------------odds ratios with 95% CI-----------------------------")
# Calculate Odds Ratios for fixed effects
odds_ratios <- exp(fixef(model_h1))
print(odds_ratios)

# Confidence intervals for Odds Ratios (Wald)
conf_int <- exp(confint(model_h1, parm = "beta_", method = "Wald"))
print(conf_int)

print("In a neat table:")

# Combine into a single data frame for clean output
or_table <- data.frame(
  Term = names(odds_ratios),
  Odds_Ratio = round(odds_ratios, 3),
  CI_lower = round(conf_int[,1], 3),
  CI_upper = round(conf_int[,2], 3)
)

print(or_table)


print("-----------------------------variability explained by the model-----------------------------")
model_r2 <- performance::r2_nakagawa(model_h1)
print(model_r2)

print("-----------------------------post hoc analysis-----------------------------")
emmeans_results <- emmeans(model_h1, ~ Consciousness * Intentions * Sensations)
summary(emmeans_results)
pairwise_results <- contrast(emmeans_results, method = "pairwise")
print(pairwise_results)


## Return output to the console ------------------------
sink() 





" ***************************************** C vs. MS  ***************************************** "


"
Consciousness vs. Moral Status modelling: 

Do attributions of moral status to 24 entities vary depending on their
attributed consciousness? 

In this analysis, we test how attributing moral status (ratings 1-4) were
affected by attributions of consciousness (1-4) to the same entities (items). 

dependent var: moral status ratings
predictors: consciousness ratings
"




## Load Data -------------------------------------

data_path <- file.choose()  # "c_v_ms\c_v_ms_long.csv" 
save_dir <- dirname(data_path)  # deduce the saving path from this
data <- read.csv(data_path) 

data$response_id <- factor(data$response_id)
data$Topic <- factor(data$Topic)
data$Item <- factor(data$Item)


### Preprocess Data -------------------------------------

# cv_data_wider: 4 columns: response_id, Item, Consciousness, Moral Status
data_wide <- data %>% 
  pivot_wider(names_from = Topic, values_from = Rating)

# Compute the response-wise mean of Consciousness per subject (response_id) and per item (entity)
data_wide <- data_wide %>%
  group_by(response_id) %>%
  mutate(Consciousness_ResponseMean = mean(Consciousness, na.rm = TRUE)) %>%
  ungroup() %>%
  group_by(Item) %>%
  mutate(Consciousness_ItemMean = mean(Consciousness, na.rm = TRUE)) %>%
  ungroup()


### Analysis Method : LMM  Function ------------------------

analyze_lmm <- function(formula, data, model_name, save_dir, txt_file_name) {
  
  sink_file_path <- file.path(save_dir, txt_file_name)
  # Start sinking output
  sink(sink_file_path, append = FALSE)
  
  # Fit the model
  print(" ")
  print(paste0("---------------------- LMM: ", model_name, " ----------------------"))
  result <- lmerTest::lmer(formula, data = data,
                           control = lmerControl("bobyqa"))  # to make the model converge
  
  # Print model summary
  result_summary <- summary(result)
  print("                ")
  print("*** Model Summary ***")
  print(result_summary)
  print("                ")
  
  print("                ")
  print("---------------------- Effect Significance ----------------------")
  result_effects <- anova(result, type=2)   
  print(result_effects)
  
  print("                ")
  print("---------------------- Bayes Factors ----------------------")
  result_effects$BF <- p_to_bf(result_effects$"Pr(>F)", n_obs = nobs(result))[["BF"]]
  as.data.frame(result_effects)
  print(result_effects)
  
  # Compute R²
  print("                ")
  print("---------------------- R² (Variance Explained) ----------------------")
  r2_values <- performance::r2(result)
  print(r2_values, tolerance = 1e-12)
  
  # Stop sinking so further output goes back to console
  sink()
  
  # Return all results as a list
  invisible(list(
    model = result,
    summary = result_summary,
    anova = result_effects
  ))
}


### Analyze Data -------------------------------------


## model MS per C ----------------------
# rename to not have spaces
data_wide <- data_wide %>% rename(Moral_Status = `Moral Status`)

result_list <- analyze_lmm(
  formula = Moral_Status ~ 
    Consciousness + Consciousness_ResponseMean + Consciousness_ItemMean +
    (Consciousness + Consciousness_ResponseMean | Item) +
    (Consciousness + Consciousness_ItemMean | response_id),
  data = data_wide,
  model_name = "Moral status ~ Consciousnesss & per subject & per item", 
  save_dir = save_dir, 
  txt_file_name = "lmm_ms_by_c.txt"
  
)
