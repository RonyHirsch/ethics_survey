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
library(multcomp)  
library(DHARMa)




## Load Data ------------------------

data_path <- file.choose()  # "kill_for_test\kill_to_pass_coded_per_entity.csv" 
save_dir <- dirname(data_path)  # deduce the saving path from this
data <- read.csv(data_path) 
data$response_id <- factor(data$response_id)
data$entity <- factor(data$entity)
data$Consciousness <- factor(data$Consciousness)
data$Intentions <- factor(data$Intentions)
data$Sensations <- factor(data$Sensations)

## Prepare results saving ------------------------

sink_file_path <- paste0(save_dir, "/kill_for_test.txt")
sink(sink_file_path, append = FALSE)

## Analyze Data ------------------------
"
In this analysis, we test how the presence or absence of each property 
(intentions, sensations, consciousness) affects the likelihood that its killed. 
dependent var: kill
predictors: Consciousness, Intentions, Sensations
"

print("=================== Kill per Property =================== ")

# Hypothesis
model_h1 <- glmer(kill ~ Consciousness * Intentions * Sensations + (1| response_id),
                  data = data, family = binomial())

print("-----------------------------model summary-----------------------------")
print(summary(model_h1))

# 1. Model Testing - Statistical Significance
# We use the summary output for fixed effect significance (p-values) and the random effect variance
# The p-values can be obtained using the z-values in the summary table.

# 2. Interpret the effect of each property (Consciousness, Intentions, Sensations)
# Let's extract and display the fixed effect coefficients and their significance
fixef_summary <- summary(model_h1)$coefficients
print(fixef_summary)

# 3. Variability explained by the model
# We can compute the marginal and conditional R² using the performance package
model_r2 <- performance::r2_nakagawa(model_h1)
print(model_r2)

# 4. Post-hoc analysis for significant predictors
# If you have significant interaction effects, you may want to perform post-hoc comparisons
# Using emmeans to test pairwise comparisons of the significant predictors (if applicable)
# The emmeans and pairwise comparisons show the estimated marginal means (predicted probabilities) 
# for different levels of Consciousness, Intentions, and Sensations.
emmeans_results <- emmeans(model_h1, ~ Consciousness * Intentions * Sensations)
summary(emmeans_results)
pairwise_results <- contrast(emmeans_results, method = "pairwise")
print(pairwise_results)


# 5. Visualization
# Visualizing the effect of the interaction terms (Consciousness * Intentions * Sensations)
# Plotting predicted probabilities based on the model
pred_data <- expand.grid(Consciousness = levels(data$Consciousness), 
                         Intentions = levels(data$Intentions),
                         Sensations = levels(data$Sensations))

# Predict probabilities from the model (ignoring random effects!)
pred_data$predicted_prob <- predict(model_h1, newdata = pred_data, type = "response", re.form = NA)


# 6. Groupings Based on Decisions (Random Effects Analysis)
# Let's check the variance explained by the random effect (response_id)
# some response_ids have very negative or very positive random effects, 
# indicating that some individuals are much more or much less likely to be killed, 
# regardless of the fixed predictors.
random_effects <- ranef(model_h1)
print(random_effects)



# You can also visualize the distribution of random effects to see if there are groupings
random_intercepts <- random_effects$response_id[,1]  # Extract the random intercepts as a vector
names(random_intercepts) <- rownames(random_effects$response_id)  # Optional: add response_id as names

# Convert to a data frame for plotting
random_df <- data.frame(response_id = names(random_intercepts),
                        intercept = as.numeric(random_intercepts))

# Plot histogram of random effects: 
# a histogram showing how much individual participants (identified by response_id) 
# vary in their tendency to "kill", independent of the fixed effects. 
# Groupings or skewness might suggest consistent response patterns among subsets of participants.
ggplot(random_df, aes(x = intercept)) +
  geom_histogram(binwidth = 0.2, fill = "#2B6E8C", color = "black", alpha = 0.8) +
  labs(title = "Distribution of Random Intercepts by Response ID",
       x = "Random Intercept (response tendency)",
       y = "Count") +
  theme_minimal(base_size = 15)





# clustering
set.seed(123)  # For reproducibility
kmeans_result <- kmeans(random_intercepts, centers = 3)  # Assume 3 clusters initially

# Add cluster labels to the data
random_df$cluster <- as.factor(kmeans_result$cluster)

# Visualize the clusters
ggplot(random_df, aes(x = intercept, fill = cluster)) +
  geom_histogram(binwidth = 0.2, color = "black", alpha = 0.7) +
  labs(title = "Distribution of Random Intercepts by Cluster",
       x = "Random Intercept (response tendency)",
       y = "Count") +
  scale_fill_manual(values = c("#2B6E8C", "#AED4E6", "#FCA04A"), 
                    name = "Cluster") +
  theme_minimal(base_size = 15)

# 
cluster_summary <- random_df %>%
  group_by(cluster) %>%
  summarise(mean_intercept = mean(intercept),
            sd_intercept = sd(intercept),
            n = n())

# Print the summary
print(cluster_summary)


# more
data$cluster <- random_df$cluster[match(rownames(data), random_df$response_id)]

# Summarize the distribution of clusters by predictors
table(data$cluster, data$Consciousness)
table(data$cluster, data$Intentions)
table(data$cluster, data$Sensations)

# You can also visualize the distribution of the predictors by cluster
ggplot(data, aes(x = Consciousness, fill = cluster)) +
  geom_bar(position = "dodge") +
  labs(title = "Distribution of Consciousness by Cluster", x = "Consciousness", y = "Count")

ggplot(data, aes(x = Intentions, fill = cluster)) +
  geom_bar(position = "dodge") +
  labs(title = "Distribution of Intentions by Cluster", x = "Intentions", y = "Count")

ggplot(data, aes(x = Sensations, fill = cluster)) +
  geom_bar(position = "dodge") +
  labs(title = "Distribution of Sensations by Cluster", x = "Sensations", y = "Count")


# 
wss <- numeric(15)  # Create an empty vector to store the WSS values

for (k in 1:15) {
  kmeans_result <- kmeans(random_intercepts, centers = k)
  wss[k] <- kmeans_result$tot.withinss
}

# Plot the Elbow curve
ggplot(data.frame(k = 1:15, wss = wss), aes(x = k, y = wss)) +
  geom_line() + 
  geom_point() + 
  labs(title = "Elbow Method for Optimal Number of Clusters", x = "Number of Clusters", y = "Within-Cluster Sum of Squares")







# Fixed effects 
intentions_colors <- c("1" = "#2B6E8C", "0" = "#AED4E6")
sensations_colors <- c("1" = "#95235A", "0" = "#E07BAC")
consciousness_colors <- c("1" = "#CA6302", "0" = "#FCA04A")


plots_info <- list(
  list(x_var = "Consciousness", color_var = "Intentions", color_map = intentions_colors, shape_var = "Sensations", shape_map = c(17, 15), file_name = "plot_consciousness.svg"),
  list(x_var = "Intentions", color_var = "Sensations", color_map = sensations_colors, shape_var = "Consciousness", shape_map = c(17, 15), file_name = "plot_intentions.svg"),
  list(x_var = "Sensations", color_var = "Consciousness", color_map = consciousness_colors, shape_var = "Intentions", shape_map = c(17, 15), file_name = "plot_sensations.svg")
)

# Loop through the combinations and generate plots
for (plot_info in plots_info) {
  # Extract values from the list
  x_var <- plot_info$x_var
  color_var <- plot_info$color_var
  color_map <- plot_info$color_map
  shape_var <- plot_info$shape_var
  shape_map <- plot_info$shape_map
  file_name <- plot_info$file_name
  
  # Create the plot
  p <- ggplot(pred_data, aes_string(x = x_var, y = "predicted_prob", 
                                    color = paste0("as.factor(", color_var, ")"), 
                                    shape = paste0("as.factor(", shape_var, ")"), 
                                    group = paste0("interaction(", color_var, ", ", shape_var, ")"))) +
    geom_point(size = 5) +  # Larger marker size for better visibility
    geom_line() +
    scale_color_manual(values = color_map, 
                       labels = c("Doesn't Have", "Has")) +  # Correct mapping for color variable
    scale_shape_manual(values = shape_map, 
                       labels = c("Has", "Doesn't Have")) +  # Correct mapping for shape variable
    labs(title = paste("Effects on the Probability of Being Killed - ", x_var),
         x = x_var, y = "Predicted Probability of Kill", 
         color = color_var, shape = shape_var) +
    theme_minimal(base_size = 15) +  # Use minimal theme but adjust base size
    theme(
      panel.grid = element_blank(),  # Remove gridlines
      axis.line = element_line(color = "black"),  # Add left and bottom spines
      axis.ticks = element_line(color = "black"),  # Ticks only on left and bottom
      axis.title = element_text(size = 12),  # Font size for axis titles
      axis.text = element_text(size = 10),  # Font size for axis ticks
      legend.title = element_text(size = 12),  # Increase font size of legend titles
      legend.text = element_text(size = 10),  # Increase font size of legend labels
      legend.box.spacing = unit(0.5, "cm")  # Increase space between legend box and plot
    ) +
    scale_y_continuous(limits = c(0, 1)) +  # Set y-axis range between 0 and 1
    scale_x_discrete(labels = c("Doesn't Have", "Has")) +  # Modify axis labels
    guides(
      color = guide_legend(override.aes = list(shape = 20, size = 5)),  # Rectangular legends for color
      shape = guide_legend(override.aes = list(size = 4))  # Increase size of 'Sensations' shapes in legend
    )
  
  # Save the plot to the specified file
  ggsave(paste0(save_dir, "/", file_name), plot = p, width = 8, height = 6, device = "svg")
}




# 7. Summary of how each property influences participants' decisions
# Let's summarize the fixed effects
summary_text <- paste("Fixed effects summary:\n")
for (i in 1:nrow(fixef_summary)) {
  summary_text <- paste(summary_text, 
                        paste(rownames(fixef_summary)[i], ":", 
                              round(fixef_summary[i, "Estimate"], 3), 
                              "(SE =", round(fixef_summary[i, "Std. Error"], 3), 
                              ", z =", round(fixef_summary[i, "z value"], 3), 
                              ", p =", round(fixef_summary[i, "Pr(>|z|)"], 3), ")\n"))
}
cat(summary_text)




## Returns output to the console ------------------------

sink() 
