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




"*******************************************************************************"


"
Another model is to check whether different conceptions of consciousness affect 
killing behaviors in the KPT scenarios. same dataframe is used 
"



print("=================== Kill per Property - Per ICS group =================== ")

# Hypothesis
model_h1 <- glmer(kill ~ group + (1|response_id) + (1|entity),
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
# no pairwise results here as this is just one factor.


print("-----------------------------LRT against null model-----------------------------")
model_null <- glmer(kill ~ 1 + (1|response_id) + (1|entity), data = data, family = binomial())

print("-----------------------------NULL model summary-----------------------------")
print(summary(model_null))

print("-----------------------------Likelihood Ratio Test (LRT)-----------------------------")
lrt_result <- anova(model_null, model_h1, test = "Chisq")
print(lrt_result)

print("LRT statistic")
lrt_statistic <- lrt_result$Chisq[2]
print(lrt_statistic)
print("")
print("LRT degrees of freedom")
lrt_df <- lrt_result$Df[2]
print(lrt_df)
print("")
print("LRT p-value")
lrt_p_value <- lrt_result$`Pr(>Chisq)`[2]
print(lrt_p_value)
print("")

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
  
  # correct p-values with Bonferroni
  result_effects[["p_adjust"]] <- p.adjust(result_effects[["Pr(>F)"]], method = "bonf")
  print(result_effects, digits=3)
  
  print("                ")
  print("---------------------- Bayes Factors ----------------------")
  result_effects$BF <- p_to_bf(result_effects$"Pr(>F)", n_obs = nobs(result))[["BF"]]
  as.data.frame(result_effects)
  print(result_effects)
  
  # Compute R²
  print("                ")
  print("---------------------- R² (Variance Explained) ----------------------")
  r2_values <- performance::r2(result)
  print(r2_values)
  
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



### Compute Off-Diagonal Difference Per Item ----------------------

# Pivot wider: get mean, sd, n per Topic
item_diff <- data %>%
  group_by(Item, Topic) %>%
  summarize(
    mean_rating = mean(Rating, na.rm = TRUE),
    sd_rating = sd(Rating, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  pivot_wider(
    names_from = Topic,
    values_from = c(mean_rating, sd_rating, n)
  )

# Rename columns to safe names
colnames(item_diff) <- c(
  "Item",
  "mean_Consciousness", "mean_MoralStatus",
  "sd_Consciousness", "sd_MoralStatus",
  "n_Consciousness", "n_MoralStatus"
)

# Compute difference and CI
item_diff <- item_diff %>%
  mutate(
    diff = mean_MoralStatus - mean_Consciousness,
    se_diff = sqrt((sd_MoralStatus^2 / n_MoralStatus) + 
                     (sd_Consciousness^2 / n_Consciousness)),
    lower = diff - 1.96 * se_diff,
    upper = diff + 1.96 * se_diff,
    off_diagonal = ifelse(lower > 0 | upper < 0, TRUE, FALSE),
    direction = ifelse(diff > 0, "Moral > Consciousness", "Moral < Consciousness")
  ) %>%
  arrange(desc(abs(diff)))

# Save
write.csv(item_diff, file.path(save_dir, "item_off_diagonal_differences.csv"), row.names = FALSE)

# quick plot
library(ggplot2)
ggplot(item_diff, aes(x = reorder(Item, diff), y = diff, fill = off_diagonal)) +
  geom_col() +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2) +
  coord_flip() +
  labs(
    title = "Off-Diagonal Differences (Moral Status - Consciousness)",
    y = "Difference in Mean Ratings",
    x = "Item"
  ) +
  theme_minimal()




## Estimated Marginal Means: compare Consciousness ratings across Items  ----------------------

# Prepare results saving 
sink_file_path <- paste0(save_dir, "/consciousness_items_emmeans.txt")
sink(sink_file_path, append = FALSE)

# preprocess - the long format
con_data <- data %>% filter(Topic == "Consciousness")
con_data$response_id <- factor(con_data$response_id)
con_data$Item <- factor(con_data$Item)

# fit LMM: Consciousness ~ Item
print("---------------------- Consciousness per Item ----------------------")
model_consciousness <- lmer(Rating ~ Item + (1 | response_id), 
                            data = con_data, control = lmerControl("bobyqa"))
summary(model_consciousness)

# Estimated Marginal Means (EMMs)
print("---------------------- Estimated Marginal Means (EMMs) ----------------------")
# added : , pbkrtest.limit = 10000, lmerTest.limit = 10000 to fix for df's for the large dataset (forcing it to compute them)
emm <- emmeans(model_consciousness, ~ Item, pbkrtest.limit = 10000, lmerTest.limit = 10000) 
print(emm)
emm_table <- summary(emm)

# Pairwise comparisons with Bonferroni correction
print("---------------------- Pairwise comparisons with Bonferroni correction ----------------------")
pairwise_tests <- pairs(emm, adjust = "bonferroni")

# Save RAW results for supplement
write.csv(as.data.frame(pairwise_tests), file.path(save_dir, "consciousness_items_pairwise_comps.csv"), row.names = FALSE)
# create a cleaned version for interpretability
pairwise_df <- as.data.frame(pairwise_tests)
# remove You and newborn as ALL their contrasts are significant
pairwise_clean <- subset(pairwise_df, !grepl("You|Newborn baby", contrast))
# keep only significant comparisons
pairwise_clean <- subset(pairwise_clean, p.value < 0.05)
# save 
write.csv(pairwise_clean, file.path(save_dir, "consciousness_items_pairwise_comps_clean.csv"), row.names = FALSE)


# now save emmeans, with counts of how many pairwise comparisons are significant
item_counts <- table(unlist(str_split(pairwise_df$contrast, " - ")))
emm_table$pairwise_comps <- sapply(as.character(emm_table$Item), function(x) ifelse(x %in% names(item_counts), item_counts[x], 0))
# save
write.csv(as.data.frame(emm_table), file.path(save_dir, "consciousness_items_emmeans.csv"), row.names = FALSE)



# Return output to the console
sink()




## model MS per C and experience with consciousness ----------------------

# ------- load experience data -------
exp_path <- file.choose()  # experience/experience_ratings.csv file
exp_data <- read.csv(exp_path)  
exp_data$response_id <- factor(exp_data$response_id) 
# rename this column as this is self-reported EXPERIENCE with consciousness, avoid confusion
names(exp_path)[names(exp_path) == Consciousness] <- exp_consciousness

cv_with_cexp <- data_wide %>%
  left_join(exp_data %>% select(response_id, exp_consciousness), by = "response_id")

result_list <- analyze_lmm(
  formula = `Moral Status` ~ 
    Consciousness_ResponseMean + exp_consciousness*(Consciousness + Consciousness_ItemMean) +
    (Consciousness*exp_consciousness + Consciousness_ResponseMean | Item) +
    (Consciousness + Consciousness_ItemMean | response_id),
  data = cv_with_cexp,
  model_name = "Moral status ~ Consciousnesss & per subject & per item with Consciouenss Experience", 
  save_dir = save_dir, 
  txt_file_name = "lmm_ms_by_c_withExp.txt"
)



print("=================== MS ratings - Per ICS group =================== ")

"
In this model we will examine whether the conception of moral status as it is
captured in the ICS groups affects the attributed moral status of different 
entities. 

Crucially, we'd have to break it down per item, as:
clmm(rating ~ group * item + (1 | response_id),
                 data = df_long,
                 link = 'logit', 
                 nAGQ = 1, 
                 control = clmm.control(method = 'nlminb')  
                 )
 Does not converge - we have 24 items and 4 groups.                  
"


## Load Data -------------------------------------

data_path <- file.choose()  # "c_v_ms\ms_per_ics.csv" - just consciousness ratings + ICS groups
save_dir <- dirname(data_path)  # deduce the saving path from this
data <- read.csv(data_path) 

## Prepare results saving ------------------------

sink_file_path <- paste0(save_dir, "/ms_per_ics.txt")
sink(sink_file_path, append = FALSE)



### Preprocess Data -------------------------------------

df_long <- melt(data,
                id.vars = c("response_id", "group"),
                variable.name = "item",
                value.name = "rating")

df_long$group <- as.factor(df_long$group)
df_long$item <- as.factor(df_long$item)
df_long$rating <- as.ordered(df_long$rating)


### Analyze Data -------------------------------------

## model C rating per group and item ----------------------
print("=================== MS rating per ICS group and Entity =================== ")

"
We run a CLMM = Cumulative Link Mixed Model
(a special kind of ordinal regression model that also includes random effects), 
because we don't want to treat the ratings as continuous

dependent variable: moral status attributions (ratings, ordinal)
factors: ICS group (group)

And we do this PER ITEM >> note that this means that there is NO VARIATION 
PER PERSON within a given model, because each person rated each item ONCE. 

"

# result aggregation
items <- levels(df_long$item)
all_results <- list()

for (it in items) {
  
  cat(" ---------------------------------- Fitting CLMM for item:", it, "--------------------------------------\n")
  
  df_item <- subset(df_long, item == it)
  
  model <- clm(rating ~ group,  # NO NEED FOR + (1 | response_id), no need for CLMM (just CLM)
                data = df_item,
                link = "logit")
  print("-----------------------------model summary-----------------------------")
  print(summary(model))
  
  # post-hoc pairwise comparisons for groups
  print("-----------------------------post hoc analysis-----------------------------")
  emm <- emmeans(model, ~ group, mode = "latent")
  summary(emm)
  pairwise <- contrast(emm, method = "pairwise", adjust = "none")
  print(pairwise)
  pairwise_df <- as.data.frame(pairwise)
  # add odds ratios and item label
  pairwise_df$odds_ratio <- exp(pairwise_df$estimate)
  pairwise_df$item <- it
  all_results[[it]] <- pairwise_df
}


# combine all per-item contrasts
contrasts_df <- do.call(rbind, all_results)

# Bonferroni correction across all comparisons
contrasts_df$p_adj <- p.adjust(contrasts_df$p.value, method = "bonferroni")

# sort for readability
contrasts_df <- contrasts_df %>%
  arrange(item, contrast)

write.csv(as.data.frame(contrasts_df), file.path(save_dir, "ms_per_ics_group_item_contrasts.csv"), row.names = FALSE)

print(contrasts_df)
  

## Return output to the console ------------------------
sink() 
