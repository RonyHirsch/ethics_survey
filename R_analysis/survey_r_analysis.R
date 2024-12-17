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

"
Consciousness vs Moral Status Ratings
"




## Load ------------------------

curr_path <- file.choose()  
data <- read.csv(curr_path)  # c_v_ms_long file

# treat categorical data as such
data$response_id <- factor(data$response_id)
data$Topic <- factor(data$Topic)
data$Item <- factor(data$Item)


# filter by topic
data_consciousness <- data[data$Topic == "Consciousness",]
data_ms <- data[data$Topic == "Moral Status",]

dataframes <- list(data_consciousness, data_ms) 
dataframes_names <- list("consciousness", "moralstatus") 

column_names <- c("exp_animals", "exp_consc", "exp_ethics", "exp_ai", "age")


## Perform LMM ------------------------



result <-lmer(Rating ~ Topic * Item + (1 + Topic + Item|response_id), 
              data = data, 
              REML = FALSE)
result_summary <- summary(result)






"
LCA: Latent Class Analysis.
In this analysis, we'll take the consciousness ratings and the moral status ratings 
of the same entities, and see if we can cluster participants based on their ratings
into interesting groups. 
"





## Perform LCA ------------------------
# Use grep to select columns that start with "c_" or "ms_"
columns_to_include <- grep("^c_|^ms_", colnames(data), value = TRUE)
"
The term Latent class regression (LCR) can have two meanings.

In poLCA: 
LCR models refer to latent class models in which the probability of class membership is predicted by one or more covariates. 
However, LCR is also used to refer to regression models in which the manifest variable is partitioned into some 
specified number of latent classes as part of estimating the regression model.
The flexmix function in package flexmix estimates this other type of LCR model:
It is a way to simultaneously fit more than one regression to the data when the latent data partition is unknown. 
Because of these terminology issues, the LCR models this package estimates are sometimes termed 'latent class models with covariates' or 
'concomitant-variable latent class analysis', both of which are accurate descriptions of this model.
"

## poLCA ----------------

# Define the formula for LCA, where each item is treated as a categorical variable
formula_polca <- as.formula(paste("cbind(", paste(columns_to_include, collapse = ", "), ") ~ 1"))

# Perform LCA using poLCA
lca_model <- poLCA(formula_polca, data, nclass = 3, maxiter = 1000, nrep = 5, verbose = TRUE)

summary(lca_model)


## Stats ----------------

log_likelihood <- lca_model$llik   # Log-likelihood
aic_value <- lca_model$aic         # AIC (Akaike Information Criterion)
bic_value <- lca_model$bic         # BIC (Bayesian Information Criterion)
g2_stat <- lca_model$Gsq           # G² (Likelihood Ratio Statistic / Deviance Statistic)
x2_stat <- lca_model$Chisq         # X² (Chi-square statistic)
lca_model$predcell

# add the predicted class membership into the df
predicted_class <-lca_model$predclass 
data$Cluster <- factor(predicted_class)



# Create a data frame for plotting
df_classes <- data.frame(Cluster = factor(predicted_class))
save_path <- selectDirectory()
filename <- "LCA_df_result.csv"
file_path <- file.path(save_path, filename)
write.csv(data, file = file_path, row.names = FALSE)









#########################################################################################

## Load ------------------------
curr_path <- file.choose()  
data <- read.csv(curr_path)
data$response_id <- factor(data$response_id)  # participant code is not a number

## Summary ------------------------
print("overall summary")
print(describe(data))

print("summary per Topic")
print(by(data = data, 
         INDICES = data[,"Topic"], 
         FUN = describe))

## Model ------------------------
result <-lmer(Rating ~ Topic * Item + (Topic|response_id), 
              data = data, 
              REML = FALSE)
result_summary <- summary(result)
print(result_summary)

## Effect Significance ------------------------
print("------Effect Significance------")
result_effects <- anova(result, type = 2)  
print(result_effects)
# correct p-values with Bonferroni
result_effects[["p_adjust"]] <- p.adjust(result_effects[["Pr(>F)"]], method = "bonf")
print(result_effects, digits=3)

## Bayes Factors ------------------------
print("------Bayes Factors------")
result_effects$BF <- p_to_bf(result_effects$"Pr(>F)", n_obs = nobs(result))[["BF"]]
as.data.frame(result_effects)
# round to 3 digits after the decimal point
result_effects[] <- lapply(result_effects, function(x) if(is.numeric(x)) round(x, 3) else x)
result_effects