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
Ratings of agreement with sayings about graded consciousness
"



## Load Data ------------------------

long_path <- file.choose()  
graded_data <- read.csv(long_path)  # graded_experience file
# treat categorical data as such
graded_data$response_id <- factor(graded_data$response_id)

exp_cols <- c("exp_animals", "exp_consc", "exp_ethics", "exp_ai")
agreement_cols <- c("If.two.creatures.systems.are.conscious..they.are.equally.conscious", 
                    "If.two.creatures.systems.are.conscious..they.are.not.necessarily.equally.conscious",
                    "Assuming.two.different.creatures.systems.are.conscious..their.consciousness.is.incomparable")
interest_col <- "Does.it.mean.that.the.interests.of.the.more.conscious.entity.matter.more."


## Process Data ------------------------

# is the participant expert in something
graded_data$is_expert <- ifelse(apply(graded_data[exp_cols], 1, function(row) any(row == 4)), 1, 0)


## Descriptives ------------------------

agreement_summary <- sapply(graded_data[agreement_cols], function(x) c(M = mean(x), SD = sd(x)))
print("Mean (M) and Standard Deviation (SD) of agreement with each saying:")
print(agreement_summary)
