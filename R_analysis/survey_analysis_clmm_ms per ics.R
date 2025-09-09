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
library(multcompView)  
library(stringr)
library(purrr)
library(glmmTMB)
library(car)



print("=================== MS ratings - Per ICS group =================== ")

"
In this model we will examine whether the conception of moral status as it is
captured in the ICS groups affects the attributed moral status of different 
entities. 
"


## Load Data -------------------------------------

data_path <- file.choose()  # "c_v_ms\ms_per_ics.csv" - just ms  ratings + ICS groups
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
write.csv(df_long, file.path(save_dir, "ms_per_ics_long.csv"), row.names = FALSE)

# Keep a consistent ordering across re-fits
group_levels  <- levels(df_long$group)
item_levels   <- levels(df_long$item)
rating_levels <- levels(df_long$rating)


### Analyze Data -------------------------------------

print("=================== MS rating per ICS group and Entity =================== ")


# 
"
First, we will set the contrast to be 'sum' (and not 'dummy'): Contrasts: SUM coding
With sum coding, each factor’s coefficients represent deviations from the grand 
mean (so “main effects” are averaged over all levels rather than compared to a 
single reference level). 
"

print(" Setting contrasts to SUM (deviation) coding for unordered factors. ")
contrasts(df_long$group) <- contr.sum(length(unique(df_long$group)))
contrasts(df_long$item)  <- contr.sum(length(unique(df_long$item)))

"
Then, fit a CLMM with a logit link, as rating is an ordinal factor
"

print ("fitting CLMM (logit link)")
model <- clmm(
  rating ~ group * item + (1 | response_id),
  data = df_long,
  link = "logit", 
  Hess = TRUE # compute the hessian for standard errors
)
saveRDS(model, file = file.path(save_dir, "clmm_model.rds"))

# load it later
model <- readRDS(file.path(save_dir, "clmm_model.rds"))


print ("model summary")
print(summary(model))

print (" -------- Anova for overall fixed effects ------------")
"
omnibus Wald χ² tests for the fixed effects. 
Type 2: each main effect (group, item) adjusted for other main effects, 
and the interaction effect.
This is to see if there is some effect of group, item, or their interaction, 
complementing the emmeans pairwise comparisons (which ask which *levels* differ).
"

anova_tbl <- car::Anova(model, type = "II")
print(anova_tbl)

anova_df <- anova_tbl |> 
  as.data.frame() |>
  tibble::rownames_to_column("Effect")

write.csv(anova_df,
          file = file.path(save_dir, "anova_typeII_wald.csv"),
          row.names = FALSE)


print (" -------- Estimated marginal means & pairwise tests ------------")

"
Estimated marginal means (EMMs) and pairwise comparisons WITHIN each item
Compute EMMs of group separately for each item 
"
emm_item   <- emmeans(model, ~ group | item, mode = "latent")
pairs_item <- contrast(emm_item, method = "pairwise", adjust = "none")  # no adjustment as we do a tree correction method on all results
# ask emmeans to report these as ratios on the response (odds) scale 
pairs_item_or <- update(pairs_item, tran = make.link("log"), type = "response")


"
summary(pairs_item_or) still prints estimate (log-odds difference) and SE 
because tests live on that scale; 
the response column in summary() and confint() is the OR. 
In the code above I pull ORs from confint() and (optionally) keep the log-odds 
estimate for reference.
"

# get the stats (SE, z, p) from summary(), already in OR scale
pairs_item_sum <- summary(pairs_item_or) |>
  as.data.frame() |>
  tidyr::separate(contrast, into = c("g1","g2"), sep = " - ", remove = FALSE)

# get CIs from confint(), then turn to ORs (and CI) on odds scale
pairs_item_ci <- confint(pairs_item_or) |>
  as.data.frame() |>
  tidyr::separate(contrast, into = c("g1","g2"), sep = " - ", remove = FALSE) |>
  dplyr::rename(OR = response, OR_low = asymp.LCL, OR_high = asymp.UCL) |>
  dplyr::select(item, contrast, g1, g2, OR, OR_low, OR_high)

# merge
pairs_df <- pairs_item_sum |>
  dplyr::left_join(pairs_item_ci, by = c("item","contrast","g1","g2")) |>
  dplyr::mutate(
    significant = p.value < 0.05,
    direction   = ifelse(response > 0, paste0(g1, " > ", g2),
                         paste0(g2, " > ", g1)),
    est_round   = round(response, 3),   # still log-odds difference (for ref)
    SE_round    = round(SE, 3),
    z_round     = round(z.ratio, 3),
    p_round     = signif(p.value, 3),
    OR_round       = round(OR, 3),
    OR_low_round   = round(OR_low, 3),
    OR_high_round  = round(OR_high, 3)
  )


# Only significant per-item
sig_long <- pairs_df |>
  filter(significant) |>
  select(
    item, g1, g2, response, SE, z.ratio, p.value, direction,
    OR, OR_low, OR_high,
    est_round, SE_round, z_round, p_round,
    OR_round, OR_low_round, OR_high_round
  ) |>
  arrange(item, p.value)

# Compact per-item summary
sig_by_item <- sig_long |>
  transmute(
    item,
    summary = paste0(
      g1, " vs ", g2,
      " (", direction,
      ", OR=", OR_round, " [", OR_low_round, ", ", OR_high_round, "]",
      ", est=", est_round,
      ", SE=", SE_round,
      ", z=", z_round,
      ", p=", p_round, ")"
    )
  ) |>
  group_by(item) |>
  summarise(
    n_significant = n(),
    significant_contrasts = paste0(summary, collapse = "; "),
    .groups = "drop"
  ) |>
  arrange(desc(n_significant), item)

"
Estimated marginal means averaged ACROSS items (overall group differences)
Collapse over item to test overall group differences between groups
"
emm_overall   <- emmeans(model, ~ group, mode = "latent")
pairs_overall_obj <- contrast(emm_overall, "pairwise", adjust = "none")  # no adjustment as we do a tree correction method on all results
# Report as proportional odds ratios
pairs_overall_or <- update(pairs_overall_obj, tran = make.link("log"), type = "response")

# get SE, z, p again
pairs_overall_sum <- summary(pairs_overall_or) |>
  as.data.frame() |>
  tidyr::separate(contrast, into = c("g1","g2"), sep = " - ", remove = FALSE)

# CI from emmeans
pairs_overall_ci <- confint(pairs_overall_or) |>
  as.data.frame() |>
  tidyr::separate(contrast, into = c("g1","g2"), sep = " - ", remove = FALSE) |>
  dplyr::rename(OR = response, OR_low = asymp.LCL, OR_high = asymp.UCL) |>
  dplyr::select(contrast, g1, g2, OR, OR_low, OR_high)

# merge
"
Note that this (will be saved as overall_group_pairs) has no degrees of freedom, 
as params in ordinal regression don't have dfs, it's a z-test. 
"
pairs_overall <- pairs_overall_sum |>
  dplyr::left_join(pairs_overall_ci, by = c("contrast","g1","g2")) |>
  dplyr::mutate(
    significant   = p.value < 0.05,
    direction     = ifelse(response > 0, paste0(g1," > ",g2),
                           paste0(g2," > ",g1)),
    est_round     = round(response, 3),
    SE_round      = round(SE, 3),
    z_round       = round(z.ratio, 3),
    p_round       = signif(p.value, 3),
    OR_round      = round(OR, 3),
    OR_low_round  = round(OR_low, 3),
    OR_high_round = round(OR_high, 3)
  )


write.csv(pairs_df,    file.path(save_dir, "pairs_per_item_all.csv"))
write.csv(sig_long,    file.path(save_dir, "pairs_per_item_significant.csv"))
write.csv(sig_by_item, file.path(save_dir, "per_item_significant_summary.csv"))
write.csv(pairs_overall, file.path(save_dir, "overall_group_pairs.csv"))



## Return output to the console ------------------------
sink() 