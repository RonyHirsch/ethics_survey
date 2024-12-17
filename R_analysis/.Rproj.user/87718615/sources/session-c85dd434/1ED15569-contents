library(ordinal)
library(tidyr)
library(dplyr)
library(lme4)
library(lmerTest)
library(bayestestR)
library(emmeans)
library(lsmeans)
library(psych)
library(ggplot2)
library(poLCA)
library(flexmix) 
library(reshape2)
library(rstudioapi)
library(car)
library(caret) 
library(multcomp)

"
Earth in danger clustering results logistic regression
"



## Load ------------------------

curr_path <- file.choose()  
data <- read.csv(curr_path)  # clusters_with_demographic file

# treat categorical data as such
data$response_id <- factor(data$response_id)
data$Cluster <- factor(data$Cluster)  # making sure that the cluster (a binary tag) is referred to as such


## Model ------------------------
logit_model <- glm(Cluster ~ exp_animals + exp_ethics + exp_ai + exp_consciousness, 
                   data = data, family = binomial)
print(summary(logit_model))


print("                ")
## Model Performance ------------------------
print("------Model Performance------")

# Wald Chi-squared test for overall model fit
print("Wald Chi-squared test")
model_fit <- anova(logit_model, test = "Chisq")
print(model_fit)

print("                ")
print("confusion matrix")
predicted_probs <- predict(logit_model, type = "response")
predicted_classes <- factor(predicted_probs > 0.5, levels = c(FALSE, TRUE), labels = 0:1)
conf_mat <- confusionMatrix(predicted_classes, data$Cluster)
print(conf_mat)
print("                ")


predicted_df <- data.frame(
  Culster = data$Cluster,
  predicted_classes,
  predicted_probs
)

yardstick::roc_curve(predicted_df, truth = Culster, predicted_probs, event_level = "second") %>% autoplot()
yardstick::roc_auc(predicted_df, truth = Culster, predicted_probs, event_level = "second")






## Effect Significance ------------------------
print("                ")
print("------Effect Significance------")
# Anova function from the car package for post-hoc comparisons
result_effects <- car::Anova(logit_model, type = 2)
# correct p-values with Bonferroni
result_effects[["Pr(>Chisq)_corrected"]] <- p.adjust(result_effects[["Pr(>Chisq)"]], method = "bonf")
print(result_effects)

print("                ")
## Bayes Factors ------------------------
print("------Bayes Factors------")
result_effects$BF <- p_to_bf(result_effects$"Pr(>Chisq)", n_obs = nobs(logit_model))[["BF"]]
as.data.frame(result_effects)
print(result_effects)

print("                ")
## Effect Sizes ------------------------
print("------Effect Sizes------")
library(marginaleffects)

avg_slopes(logit_model)


data_for_plot <- bind_rows(
  ai = avg_predictions(logit_model, variables = "exp_ai"),
  animals = avg_predictions(logit_model, variables = "exp_animals"),
  consciousness = avg_predictions(logit_model, variables = "exp_consciousness"),
  ethics = avg_predictions(logit_model, variables = "exp_ethics"), 
  
  .id = "predictor"
) %>% 
  mutate(
    x = coalesce(exp_animals, exp_ethics, exp_consciousness, exp_ai)
  )


ggplot(data_for_plot, aes(x, estimate, fill = predictor, color = predictor)) + 
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), color = NA, alpha = 0.6) + 
  geom_line() + 
  scale_y_continuous(limits = c(0, 1), labels = scales::label_percent())



ggplot(data_for_plot, aes(x, estimate, fill = predictor, color = predictor)) + 
  geom_pointrange(aes(ymin = conf.low, ymax = conf.high)) + 
  geom_line() + 
  scale_y_continuous(limits = c(0, 1), labels = scales::label_percent())

