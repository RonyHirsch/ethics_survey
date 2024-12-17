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
library(performance)

"
Consciousness vs Moral Status Ratings
"




## Load Data ------------------------

# ------- long format -------
long_path <- file.choose()  
cv_data_long <- read.csv(long_path)  # c_v_ms_long file
# treat categorical data as such
cv_data_long$response_id <- factor(cv_data_long$response_id)
cv_data_long$Topic <- factor(cv_data_long$Topic)
cv_data_long$Item <- factor(cv_data_long$Item)


# ------- wide format -------
wide_path <- file.choose()  
cv_data_wide <- read.csv(wide_path)  # c_v_ms file





###### Correlation Between ITEMS' Consciousness & Moral status ------------------------

print("                ")
print("---------------------- Correlation between c and MS ----------------------")
# calculate the correlation between ITEMS' consciousness average consciousness 
# and moral status scores. 

cv_data_for_cor <- cv_data_long %>% 
  group_by(Item, Topic) %>% 
  summarise(
    Rating = mean(Rating)
  ) %>% 
  pivot_wider(names_from = Topic, values_from = Rating)

print("                ")
print("Correlation between Consciousness and Moral Status Item Ratings")
correlation <- cor.test(cv_data_for_cor$Consciousness, cv_data_for_cor$`Moral Status`)
print(correlation)

print("                ")
# calculate the distance between each dot (item) and the regression line
linear_model <- lm(`Moral Status` ~ Consciousness, data=cv_data_for_cor)
cv_data_for_cor$dist_from_regression_line <- residuals(linear_model)
cv_data_for_cor$dist_from_regression_line_abs <- abs(residuals(linear_model))
write.csv(cv_data_for_cor, "c_v_ms_corr_distFromRegLine.csv")
# mean distance between dots and regression line
mean_dist <- mean(cv_data_for_cor$dist_from_regression_line_abs)  
sd_dist <- sd(cv_data_for_cor$dist_from_regression_line_abs)
print(paste("Mean dist from regression line: ", mean_dist, " SD: ", sd_dist))



### plots for sanity check -----------------------------------

# In this plot: each dot = item, regression line in red between them
plot(cv_data_for_cor$Consciousness, cv_data_for_cor$`Moral Status`)
abline(linear_model, col = "red", lwd = 2) 

# In this plot: every subject is one line (fit between all their ratings)
cv_data_long %>% 
  pivot_wider(names_from = Topic, values_from = Rating) %>% 
  filter(
    response_id %in% sample(levels(response_id), 100)
  ) %>% 
  ggplot(aes( Consciousness, `Moral Status`)) + 
  
  geom_smooth(aes(group = response_id), method = "lm", se = FALSE) + 
  geom_smooth(method = "lm", se = FALSE, color = "red", linewidth = 2)  
#+geom_point(data = data_for_cor, size = 3, color = "red")





###### LINEAR MIXED-EFFECTS MODEL ON PARTICIPANTS ------------------------

## Preprocess Data ----------------------

cv_data_wider <- cv_data_long %>% 
  pivot_wider(names_from = Topic, values_from = Rating)

# add a column to indicate expertise ("4" rating in one of the expertise columns)
expert_cols <- c("exp_animals", "exp_ai", "exp_ethics", "exp_consc")
cv_data_wider$is_expert <- ifelse(apply(cv_data_wider[expert_cols], 1, function(row) any(row == 4)), 1, 0)


print("                ")
print("---------------------- LMM between c and MS ----------------------")

result <- lmer(`Moral Status` ~ `Consciousness` * `is_expert` + (`Consciousness`| response_id), data = cv_data_wider)
result_summary <- summary(result)
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
result_effects


print("                ")
print("---------------------- ICC ----------------------")
# Intraclass Correlation Coefficient: 
# Estimates how much of the variance is explained by the variability across people
icc_value <- icc(result)
print(icc_value)
# The difference between adjusted ICC and unadjusted ICC typically refers to 
# whether the ICC calculation takes into account the influence of covariates (fixed effects) in the model
# adjusted ICC = variability explained by the fixed effects is removed before 
# partitioning the remaining variance into between-group and within-group variance
# unadjusted ICC (marginal ICC) = focuses on the total variability in the outcome 
# variable and how much of that is due to differences between individuals. 
# Tells us what proportion of the total variance can be attributed to differences 
# across individuals without factoring in covariates (fixed effects). 
# This gives us a direct understanding of how much of the total variability is 
# due to differences between people.













###### PARTICIPANTS' ratings and their EXPERIENCE ------------------------


### preprocess data -----------------------------------

# take the long data and filter by topic (consciousness, moral status)
data_consciousness <- cv_data_long[cv_data_long$Topic == "Consciousness",]
data_ms <- cv_data_long[cv_data_long$Topic == "Moral Status",]

dataframes <- list(data_consciousness, data_ms) 
dataframes_names <- list("consciousness", "moralstatus") 

column_names <- c("exp_animals", "exp_consc", "exp_ethics", "exp_ai")


### model the data -----------------------------------

# for each topic (consciousness, moral status) check whether experience 
# in any of these fields (column names) affected the ratings. 


print("                ")
print("---------------------- Linear Mixed-Effects Model ----------------------")

options(lmerTest.limit = 500000)  # we have many observations 

for (i in seq_along(dataframes)) {
  df <- dataframes[[i]]
  df_name <- dataframes_names[i]
  print("                ")
  print(paste(" ------- ", df_name, " ------- "))
  
  print("                ")
  print("correlation between experience types")
  corr <- df %>% 
    distinct(response_id, .keep_all = TRUE) %>% 
    dplyr::select(all_of(column_names)) %>% 
    cor()
  print(corr)
  print("                ")
  
  print("                ")
  print("variance between items and people: Model only with Random Intercepts")
  
  empty_mod <- clmm(ordered(Rating) ~ (1 | response_id) + (1 | Item),
                    link = "logit",
                    data = df)
  print("model summary")
  print(summary(empty_mod))
  print("                ")
  
  print("                ")
  print("Model ICC score")
  print(performance::icc(mod, by_group = TRUE))
  print("                ")
  
  
  for (col in column_names){
    print("                ")
    print(paste(" --- ", df_name, " model rating ", "by ", col, " --- "))
    
    mod <- clmm(ordered(Rating) ~ Item * col + (1 | response_id),
                link = "logit",
                data = df)
    print("model summary")
    print(summary(mod))
    print("                ")
    
    print("                ")
    print("effect sizes")
    # This will give the effect of experience column on rating for each item
    emmeans::emtrends(mod, ~ Item, vars = col,
                      mode = "mean.class", infer = TRUE)
    print("                ")
    
    
  }
  
}













