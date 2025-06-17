import os
import pandas as pd
import numpy as np
import re
from functools import reduce
from itertools import combinations
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.stats import iqr, friedmanchisquare, norm, zscore, pearsonr
import statsmodels.api as sm
import plotter
import process_survey
import survey_mapping
import helper_funcs

CAT_COLOR_DICT = {"Person": "#AF7A6D",  # FAE8EB
                  "Dog": "#CCC7B9",  # 6EA4BF
                  "My pet": "#E2D4BA",
                  "Dictator (person)": "#AF7A6D",  # 4C191B
                  "Person (unresponsive wakefulness syndrome)": "#AF7A6D",
                  "Fruit fly (a conscious one, for sure)": "#CACFD6",
                  "AI (that tells you that it's conscious)": "#074F57",
                  ###
                  survey_mapping.ANS_YES: "#3C5968",  # "#355070"
                  survey_mapping.ANS_NO: "#B53B03",  # "#B26972" #590004, #461D02
                  }

CAT_LABEL_DICT = {"Person": "Person",
                  "Dog": "Dog",
                  "My pet": "My pet",
                  "Dictator (person)": "Dictator",
                  "Person (unresponsive wakefulness syndrome)": "Person (UWS)",
                  "Fruit fly (a conscious one, for sure)": "Conscious fruit-fly",
                  "AI (that tells you that it's conscious)": "Conscious AI",
                  ###
                  "Yes": "Yes",
                  "No": "No"
                  }


def analyze_expertise_effects(df_ratings, rating_cols, expertise_df,
                              expertise_question_col, expertise_label,
                              result_path, item_prefix, expertise_level=4,
                              p_thresh=0.05, max_plot_items=None):
    """
    Generalized function to run MWU tests and plot proportion bars based on expertise level for a given rating type.
    """
    # Merge data
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=[process_survey.COL_ID]), [df_ratings, expertise_df])

    # Binarize expertise: 1 = expert (4+; based on expertise_level), 0 = non-expert (<=3)
    binary_col = f"binary_{expertise_question_col}"
    df_merged[binary_col] = df_merged[expertise_question_col].apply(lambda x: 1 if x >= expertise_level else 0)

    # if rating_cols is ONE COLUMN - single column case
    if isinstance(rating_cols, str):
        item_col = rating_cols
        relevant_df = df_merged[[process_survey.COL_ID, item_col, expertise_question_col, binary_col]].dropna()

        experts = relevant_df[relevant_df[binary_col] == 1]
        nonexperts = relevant_df[relevant_df[binary_col] == 0]

        result = helper_funcs.mann_whitney_utest(
            list_group1=experts[item_col].tolist(),
            list_group2=nonexperts[item_col].tolist()
        )
        result.to_csv(os.path.join(result_path, f"{item_prefix}_{expertise_label}_expertise_MWU.csv"), index=False)

        summary_df = relevant_df.groupby(binary_col)[item_col].agg(["min", "mean", "std", "max"]).reset_index()
        summary_df.to_csv(os.path.join(result_path, f"{item_prefix}_{expertise_label}_expertise_stats.csv"),
                          index=False)

        count_df = relevant_df.groupby([expertise_question_col, item_col]).size().reset_index(name='count')
        count_df["total"] = count_df.groupby(expertise_question_col)["count"].transform('sum')
        count_df["proportion"] = 100 * count_df["count"] / count_df["total"]

        pivot_df = count_df.pivot(index=expertise_question_col, columns=item_col, values="proportion").fillna(0).reset_index()
        pivot_df.to_csv(os.path.join(result_path, f"{item_prefix}_per_{expertise_label}_exp_props.csv"), index=False)

        plotter.plot_expertise_proportion_bars(
            df=pivot_df,
            cols=[1, 2, 3, 4],
            cols_colors={1: "#DB5461", 2: "#fb9a99", 3: "#70a0a4", 4: "#26818B"},
            x_axis_exp_col_name=expertise_question_col,
            x_label=f"Reported experience with {expertise_label.capitalize()}",
            y_ticks=[0, 25, 50, 75, 100],
            save_name=f"{item_prefix}_expertise_{expertise_label}",
            save_path=result_path,
            plt_title=item_col
        )

    else:
        # Multiple columns case
        relevant_df = df_merged[[process_survey.COL_ID] + rating_cols + [expertise_question_col, binary_col]]

        # Run MWU tests
        mw_results, corrected_p_col = helper_funcs.run_group_mann_whitney(
            df=relevant_df,
            comparison_cols=rating_cols,
            group_col=binary_col,
            group_col_name=expertise_label,
            group1_val=0, group1_name="non-experts",
            group2_val=1, group2_name="experts"
        )
        mw_results.to_csv(os.path.join(result_path, f"{item_prefix}_{expertise_label}_expertise_items.csv"), index=False)

        # identify significant items
        significant_items = mw_results[mw_results[corrected_p_col] < p_thresh]["Item"].tolist()

        if max_plot_items:
            significant_items = significant_items[:max_plot_items]

        # Plot ONLY THE SIGNIFICANT items
        for col in significant_items:
            count_df = df_merged.groupby([expertise_question_col, col]).size().reset_index(name='count')
            count_df["total"] = count_df.groupby(expertise_question_col)["count"].transform('sum')
            count_df["proportion"] = 100 * count_df["count"] / count_df["total"]

            pivot_df = count_df.pivot(index=expertise_question_col, columns=col, values="proportion").fillna(0).reset_index()
            pivot_df.to_csv(os.path.join(result_path, f"{item_prefix}_{col}_per_{expertise_label}_exp_props.csv"), index=False)

            plotter.plot_expertise_proportion_bars(
                df=pivot_df,
                cols=[1, 2, 3, 4],
                cols_colors={1: "#DB5461", 2: "#fb9a99", 3: "#70a0a4", 4: "#26818B"},
                x_axis_exp_col_name=expertise_question_col,
                x_label=f"Reported experience with {expertise_label.capitalize()}",
                y_ticks=[0, 25, 50, 75, 100],
                save_name=f"{col}_expertise_{item_prefix}",
                save_path=result_path,
                plt_title=col
            )
    return


def other_creatures(analysis_dict, save_path, sort_together=True, df_earth_cluster=None, load=True):
    # save path
    result_path = os.path.join(save_path, "c_v_ms")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # load relevant data
    df_ms = analysis_dict["other_creatures_ms"].copy()
    df_c = analysis_dict["other_creatures_cons"].copy()
    df = pd.merge(df_c, df_ms, on=[process_survey.COL_ID])
    df.to_csv(os.path.join(result_path, "c_v_ms.csv"), index=False)
    item_columns = [c for c in df_c.columns if c != process_survey.COL_ID]

    # load experience dfs for later
    experience_animal = analysis_dict["animal_exp"].copy()
    experience_ai = analysis_dict["ai_exp"].copy()
    experience_ethics = analysis_dict["ethics_exp"].copy()
    experience_cons = analysis_dict["consciousness_exp"].copy()
    demographics = analysis_dict["demographics"].copy()

    """
    *** Consciousness *** ratings based on experience with different things
    """
    c_rating_cols = [c for c in df_c if c.startswith("c_")]
    ms_rating_cols = [c for c in df_ms if c.startswith("ms_")]

    """
    AI experts: do they rate LLM consciousness different than non-experts?
    """
    c_ai_col = "c_A large language model"

    analyze_expertise_effects(
        df_ratings=df_c,
        rating_cols=c_ai_col,
        expertise_df=experience_ai,
        expertise_question_col=survey_mapping.Q_AI_EXP,
        expertise_label="AI",
        result_path=result_path,
        item_prefix="cRatingLLM"
    )

    """
    AI experts: do they rate self-driving car consciousness different than non-experts?
    """
    c_self_driving_car_col = "c_A self-driving car"

    analyze_expertise_effects(
        df_ratings=df_c,
        rating_cols=c_self_driving_car_col,
        expertise_df=experience_ai,
        expertise_question_col=survey_mapping.Q_AI_EXP,
        expertise_label="selfDrivingCar",
        result_path=result_path,
        item_prefix="cRatingSelfDrivingCar"
    )


    """
    Consciousness experts: for each item, did consciousness experts rate its *CONSCIOUSNESS* differently than non-experts?
    """

    analyze_expertise_effects(
        df_ratings=df_c,
        rating_cols=c_rating_cols,
        expertise_df=experience_cons,
        expertise_question_col=survey_mapping.Q_CONSC_EXP,
        expertise_label="consciousness",
        result_path=result_path,
        item_prefix="cRating"
    )


    """
    Do the same with animal experts vs non-experts, and animal items *** CONSCIOUSNESS *** rating
    """

    animal_rating_cols = [
        "An ant", "An orangutan", "A cow", "A turtle", "A dog",
        "A cat", "A lobster", "A sea urchin", "An octopus", "A salmon",
        "A bat", "A bee", "A mosquito", "A fruit-fly", "A rat", "A pigeon"
    ]

    c_animal_rating_cols = [f"c_{col}" for col in animal_rating_cols]

    analyze_expertise_effects(
        df_ratings=df_c,
        rating_cols=c_animal_rating_cols,
        expertise_df=experience_animal,
        expertise_question_col=survey_mapping.Q_ANIMAL_EXP,
        expertise_label="animal",
        result_path=result_path,
        item_prefix="cRating"
    )



    """
    *** *** ***  Do the same for MORAL STATUS   *** *** ***
    """

    """
    ETHICS experts -> moral status ratings of all items
    """
    analyze_expertise_effects(
        df_ratings=df_ms,
        rating_cols=ms_rating_cols,
        expertise_df=experience_ethics,
        expertise_question_col=survey_mapping.Q_ETHICS_EXP,
        expertise_label="ethics",
        result_path=result_path,
        item_prefix="msRating"
    )


    """
    AI experts and LLM 
    """
    ms_ai_col = "ms_A large language model"

    analyze_expertise_effects(
        df_ratings=df_ms,
        rating_cols=ms_ai_col,
        expertise_df=experience_ai,
        expertise_question_col=survey_mapping.Q_AI_EXP,
        expertise_label="AI",
        result_path=result_path,
        item_prefix="msRatingLLM"
    )

    """
    AI experts and self-driving cars
    """
    ms_self_driving_car_col = "ms_A self-driving car"

    analyze_expertise_effects(
        df_ratings=df_ms,
        rating_cols=ms_self_driving_car_col,
        expertise_df=experience_ai,
        expertise_question_col=survey_mapping.Q_AI_EXP,
        expertise_label="AI",
        result_path=result_path,
        item_prefix="msRatingSelfDrivingCar"
    )


    """
    Consciousness experts -> moral status ratings of all items
    """
    analyze_expertise_effects(
        df_ratings=df_ms,
        rating_cols=ms_rating_cols,
        expertise_df=experience_cons,
        expertise_question_col=survey_mapping.Q_CONSC_EXP,
        expertise_label="consciousness",
        result_path=result_path,
        item_prefix="msRating"
    )

    """
    Animal experts -> moral status ratings of animal items
    """

    ms_animal_rating_cols = [f"ms_{col}" for col in animal_rating_cols]

    analyze_expertise_effects(
        df_ratings=df_ms,
        rating_cols=ms_animal_rating_cols,
        expertise_df=experience_animal,
        expertise_question_col=survey_mapping.Q_ANIMAL_EXP,
        expertise_label="animal",
        result_path=result_path,
        item_prefix="msRating"
    )


    """
    MORAL STATUS PER EXPERTISE TYPE:
    prepare data for a model (R) where we will model moral status ratings of items by the reported experience levels.
    """
    dfs = [df_ms,
           experience_cons.loc[:, [process_survey.COL_ID, survey_mapping.Q_CONSC_EXP]],
           experience_animal.loc[:, [process_survey.COL_ID, survey_mapping.Q_ANIMAL_EXP]],
           experience_ai.loc[:, [process_survey.COL_ID, survey_mapping.Q_AI_EXP]],
           experience_ethics.loc[:, [process_survey.COL_ID, survey_mapping.Q_ETHICS_EXP]]]
    df_ms_with_exp = reduce(lambda left, right: pd.merge(left, right, on=[process_survey.COL_ID]), dfs)
    df_ms_with_exp = df_ms_with_exp.rename(columns=survey_mapping.Q_EXP_DICT)
    df_ms_with_exp.to_csv(os.path.join(result_path, f"ms_and_experience.csv"), index=False)
    # long, for modeling
    df_ms_with_exp_long = df_ms_with_exp.melt(
        id_vars=["response_id"] + list(survey_mapping.Q_EXP_DICT.values()),
        value_vars=ms_rating_cols,
        var_name="item",
        value_name="ms_rating"
    )
    # clean up the 'ms'
    df_ms_with_exp_long["item"] = df_ms_with_exp_long["item"].str.replace("ms_", "", regex=False)
    df_ms_with_exp_long.to_csv(os.path.join(result_path, f"ms_and_experience_long.csv"), index=False)



    # codes and relevant stuff
    items = list(survey_mapping.other_creatures_general_names.keys())  # all rated items
    topic_name_map = {"c": "Consciousness", "ms": "Moral Status"}
    rating_color_list = ["#DB5461", "#fb9a99", "#70a0a4", "#26818B"]
    colors = [rating_color_list[0], rating_color_list[-1]]
    rating_labels = ["Does Not Have", "Probably Doesn't Have", "Probably Has", "Has"]



    """
    *** Relationship between C ratings and MS ratings *** 
    """
    if df_earth_cluster is not None:
        df_c_with_cluster = pd.merge(df_c, df_earth_cluster[[process_survey.COL_ID, "Cluster"]], on=process_survey.COL_ID)
        df_c_with_cluster.to_csv(os.path.join(result_path, f"c_ratings_with_cluster.csv"), index=False)
        c = 3


    # melt the df to a long format
    long_data = pd.melt(df, id_vars=[process_survey.COL_ID], var_name="Item_Topic", value_name="Rating")
    long_data[["Topic", "Item"]] = long_data["Item_Topic"].str.split('_', expand=True)
    long_data = long_data.drop(columns=["Item_Topic"])
    long_data["Topic"] = long_data["Topic"].map(topic_name_map)
    long_data.to_csv(os.path.join(result_path, "c_v_ms_long.csv"), index=False)

    # tagging the entities
    long_data_coded = long_data.copy()
    long_data_coded["non_human_animal"] = long_data_coded["Item"].map(survey_mapping.other_creatures_isNonHumanAnimal)
    long_data_coded["nature"] = long_data_coded["Item"].map(survey_mapping.other_creatures_isTreeHugger)
    if df_earth_cluster is not None:
        long_data_coded = pd.merge(long_data_coded, df_earth_cluster[[process_survey.COL_ID, "Cluster"]], on=process_survey.COL_ID)
    long_data_coded.to_csv(os.path.join(result_path, "c_v_ms_long_entityCoded.csv"), index=False)
    long_data_coded_c = long_data_coded[long_data_coded["Topic"] == "Consciousness"]
    long_data_coded_c.to_csv(os.path.join(result_path, "c_long_entityCoded.csv"), index=False)


    # long df with experience level
    dataframes = [long_data.copy(), experience_animal.loc[:, [process_survey.COL_ID, survey_mapping.Q_ANIMAL_EXP]],
                  experience_ai.loc[:, [process_survey.COL_ID, survey_mapping.Q_AI_EXP]],
                  experience_ethics.loc[:, [process_survey.COL_ID, survey_mapping.Q_ETHICS_EXP]],
                  experience_cons.loc[:, [process_survey.COL_ID, survey_mapping.Q_CONSC_EXP]],
                  demographics.loc[:, [process_survey.COL_ID, survey_mapping.Q_AGE]]]
    long_with_personal = reduce(lambda left, right: pd.merge(left, right, on=[process_survey.COL_ID]), dataframes)
    long_with_personal.to_csv(os.path.join(result_path, "c_v_ms_long_experience.csv"), index=False)

    # stats: average rating (of C & MS) per item (averaged across all respondents)
    long_data_noid = long_data.drop(process_survey.COL_ID, axis=1, inplace=False)
    long_data_mean_rating = long_data_noid.groupby(["Topic", "Item"]).mean().reset_index(drop=False)
    long_data_mean_rating = long_data_mean_rating.pivot(index="Item", columns="Topic", values="Rating").reset_index(
        drop=False)
    # I'll save long_data_mean_rating below after I add some stuff to it
    long_data_mean_rating_stats = long_data_mean_rating.describe()
    long_data_mean_rating_stats.to_csv(os.path.join(result_path, f"c_v_ms_avg_per_item_stats.csv"),
                                       index=True)  # in 'describe' the index is the desc name

    """
    Plot "other creatures" judgments of Consciousness vs. of Moral Status. >> *** ACROSS ITEMS ***
    """
    # prepare data for analyses
    df_pivot = long_data.pivot_table(index="Item", columns="Topic", values="Rating", aggfunc="mean").reset_index(
        drop=False, inplace=False)  # I don't want to 'fillna(0).' this

    # create a plotting version of df-pivot, deleting a prefix of a/an
    df_pivot_plotting = df_pivot.copy()
    # convert names to short names
    df_pivot_plotting["Item"] = df_pivot_plotting["Item"].replace(survey_mapping.other_creatures_general_names)

    # collapsed across everyone, no individuation, diagonal line
    plotter.plot_scatter_xy(df=df_pivot_plotting, identity_col="Item", annotate_id=True,
                            x_col="Consciousness", x_label="Consciousness", x_min=1, x_max=4, x_ticks=1,
                            y_col="Moral Status", y_label="Moral Status", y_min=1, y_max=4, y_ticks=1,
                            save_path=result_path, save_name=f"correlation_c_ms",
                            palette_bounds=colors,  # use the same colors as the individual-item correlations
                            corr_line=False, diag_line=True, fmt="svg",
                            individual_df=None, id_col=None, color_col_colors=None)

    """
    The scatter plot above (plot_scatter_xy) has a diagonal line. The interesting part is the off-diagonal items; 
    those items are ones where people's certainty about them being conscious doesn't correspond to certainty about 
    them having moral status. Let's explore that. 
    """

    # compute the perpendicular distance (how far an item is from the diagonal) : X-axis=Consciousness; Y-axis=MS
    long_data_mean_rating["dist_from_diagonal"] = helper_funcs.calculate_distances(long_data_mean_rating,
                                                                                   x_col="Consciousness",
                                                                                   y_col="Moral Status",
                                                                                   metric="euclidean")
    long_data_mean_rating.to_csv(os.path.join(result_path, f"c_v_ms_avg_per_item.csv"), index=False)

    """
    Let's see if the distances are significantly different from zero (the diagonal). 
    For that we'll do a one-sample t-test with the null hypothesis being that the mean distance is 0. 
    """
    ttest_result = helper_funcs.one_sample_ttest(list_group1=long_data_mean_rating["dist_from_diagonal"].tolist(),
                                                 test_value=0, ci=0.95)
    ttest_result.to_csv(os.path.join(result_path, f"c_v_ms_avg_per_item_diagonal_ttest.csv"), index=False)

    # identify items with the largest deviation from the diagonal
    n = 6
    top_outliers = long_data_mean_rating.nlargest(n, "dist_from_diagonal")
    top_outliers.to_csv(os.path.join(result_path, f"c_v_ms_dist_from_diagonal_top_{n}.csv"), index=False)

    """
    If df_earth_cluster is not None, take the clustering from the Earth-in-danger scenarios, and see if they apply 
    here as well. 
    """
    if df_earth_cluster is not None:
        df_clusters = df_earth_cluster[[process_survey.COL_ID, "Cluster"]]
        df_cols = df.columns.tolist()[1:]
        df_with_cluster = pd.merge(df, df_clusters, how="inner", on=process_survey.COL_ID).reset_index(drop=True,
                                                                                                       inplace=False)
        """
        Check whether moral status ratings are affected by consciousness ratings and by the earth-in-danger clusters. 
        Let's do a mixed-effects model on to model the dependency of moral status ratings on consciousness ratings and 
        cluster membership.  
        """
        c_cols = df_c.columns[1:].tolist()
        ms_cols = df_ms.columns[1:].tolist()
        cluster_col = ["Cluster"]
        indep_cols = c_cols + cluster_col
        long_df_with_cluster = df_with_cluster.melt(id_vars=[process_survey.COL_ID, "Cluster"], value_vars=ms_cols,
                                                    var_name="feature", value_name="moral_status_rating")
        long_df_with_cluster["consciousness_rating"] = \
            df_with_cluster.melt(id_vars=[process_survey.COL_ID, "Cluster"], value_vars=c_cols)["value"]
        result_df, residuals_df, r2_df, summary_df, descriptive_stats, posthoc_df = helper_funcs.mixed_effects_model(
            long_df=long_df_with_cluster, cols_to_standardize=["consciousness_rating", "moral_status_rating"],
            dep_col="moral_status_rating", ind_col1="consciousness_rating", ind_col2="Cluster",
            id_col=process_survey.COL_ID)
        # save all model outputs to excel
        with pd.ExcelWriter(os.path.join(result_path, f"earth_in_danger_clusters_model_outputs.xlsx"),
                            engine="xlsxwriter") as writer:
            result_df.to_excel(writer, sheet_name="Fixed_Effects", index=False)
            residuals_df.to_excel(writer, sheet_name="Residuals", index=False)
            r2_df.to_excel(writer, sheet_name="R2_Values", index=False)
            summary_df.to_excel(writer, sheet_name="Model_Summary", index=False)
            descriptive_stats.to_excel(writer, sheet_name="Descriptive_Stats", index=False)
            posthoc_df.to_excel(writer, sheet_name="Posthoc_Results", index=False)



    """
    *** VARIABILITY IN * CONSCIOUSNESS * RATINGS ***
    """


    """
    ITEM-LEVEL: do people agree or are they divided (on each item)? SD, IQR, CV
    Standard Deviation (SD): maybe the simplest thing we can do to see dispersion. 
    """
    stds = df_c[item_columns].std()

    """
    ITEM-LEVEL: do people agree or are they divided (on each item)? SD, IQR, CV
    Interquartile Range (IQR)
    The range of the middle 50% of responses (75th minus 25th percentile). 
    It is a robust measure of dispersion less affected by outliers. 
    IQR = 0 (all within one category): perfect consensus
    IQR = 1: at least half of the respondents’ ratings fall within 1 scale point.
    """
    iqrs = df_c[item_columns].apply(iqr)

    """
    ITEM-LEVEL: do people agree or are they divided (on each item)? SD, IQR, CV
    CV (coefficient of variation)
    The ratio of the standard deviation to the mean.
    The CV is unitless and tells us how large the spread is relative to each item's mean score. 
    For example, an SD of 1 might be concerning if the mean is 2 (CV = 50%), but less so if the mean is 5 (CV = 20%). 
    However, CV is not universally adopted in psychology – it assumes a ratio scale with a true zero 
    (which Likert scales lack), and interpreting what is a “high” or “low” CV can be difficult. 
    """

    means = df_c[item_columns].mean()
    cv = stds / means

    stats_df = pd.DataFrame({"entity": item_columns,
                             "Mean": means,
                             "SD": stds,
                             "IQR": iqrs,
                             "CV": cv}).reset_index(drop=True, inplace=False)

    stats_df["entity"] = stats_df["entity"].replace({item: item.replace("c_", "") for item in stats_df["entity"]})

    # statistical significance for the CV
    def compute_cv_per_column(matrix):
        means = np.mean(matrix, axis=0)
        stds = np.std(matrix, axis=0, ddof=1)
        return stds / means

    # just the ratings
    ratings_matrix = df_c[item_columns].astype(int).values

    observed_cvs = compute_cv_per_column(ratings_matrix)
    null_cv_distributions = helper_funcs.premutations_for_array(ratings_matrix, compute_cv_per_column,
                                                                n_permutations=1000)
    p_values_cv = (null_cv_distributions >= observed_cvs).sum(axis=0) / null_cv_distributions.shape[0]
    stats_df["CV_p-value"] = p_values_cv  # These p-values indicate how likely such dispersion would occur by chance
    stats_df.to_csv(os.path.join(result_path, "c_ratings_dispersion_by_item.csv"), index=False)

    stats_df["entity_name"] = stats_df["entity"].replace(survey_mapping.other_creatures_general_names, inplace=False)

    stats_df_long = pd.melt(stats_df, id_vars="entity_name", value_vars=["CV", "SD", "IQR"], var_name="Metric", value_name="Value")

    # for ordering:
    stats_df = stats_df.sort_values(by="CV", ascending=True, inplace=False)
    order = stats_df["entity_name"].tolist()

    plotter.plot_categorical_bars_hued(categories_prop_df=stats_df_long, x_col="entity_name", category_col="Metric",
                                       data_col="Value", save_path=result_path, save_name="c_ratings_variance_metrics",
                                       categories_colors={"CV": "#264653", "SD": "#e9c46a", "IQR": "#e76f51"},
                                       fmt="svg", y_min=0, y_max=2.1, y_skip=0.2, delete_y=False, y_label="Score",
                                       x_rotation=90, inch_w=15, inch_h=12, add_pcnt=False, order=order, x_label="")

    # plot the per-item CV
    stats_cv_plot = stats_df.sort_values(by="CV", ascending=True, inplace=False)
    plotter.plot_categorical_scatter(df=stats_cv_plot, x_col="entity", xtick_labels=survey_mapping.other_creatures_general_names, label_x="Entity",
                                     y_col="CV", label_y="Coefficient of Variation (CV)",
                                     color="#4C5B5C", label_title="", size=100,
                                     save_path=result_path, save_name="c_ratings_cv", fmt="svg")
    # same for IQR
    stats_iqr_plot = stats_df.sort_values(by="IQR", ascending=True, inplace=False)
    plotter.plot_categorical_scatter(df=stats_iqr_plot, x_col="entity", xtick_labels=survey_mapping.other_creatures_general_names, label_x="Entity",
                                     y_col="IQR", label_y="Interquartile Range (IQR)",
                                     color="#4C5B5C", label_title="", size=100,
                                     save_path=result_path, save_name="c_ratings_iqr", fmt="svg")
    # same for SD
    stats_sd_plot = stats_df.sort_values(by="SD", ascending=True, inplace=False)
    plotter.plot_categorical_scatter(df=stats_sd_plot, x_col="entity", xtick_labels=survey_mapping.other_creatures_general_names, label_x="Entity",
                                     y_col="SD", label_y="Standard Deviation (SD)",
                                     color="#4C5B5C", label_title="", size=100,
                                     save_path=result_path, save_name="c_ratings_sd", fmt="svg")
    # all of them
    stats_plot = stats_df.sort_values(by="Mean", ascending=True, inplace=False)
    plotter.plot_categorical_multliscatter(df=stats_plot,
                                           x_col="entity", xtick_labels=entity_dict, label_x="Entity",
                                           y_cols=["SD", "IQR", "CV"], labels=["SD", "IQR", "CV"], label_y="Value",
                                           y_ticks=[0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2],
                                           colors=["#55828B", "#87BBA2", "#C9E4CA"],
                                           label_title="", size=100,
                                           save_path=result_path, save_name="c_ratings_by_item_divergence", fmt="svg")
    # plot just the C ratings with M, SD (whiskers)
    df_long = pd.melt(
        df_c,
        id_vars=[process_survey.COL_ID],
        var_name="entity",
        value_name="c_rating"
    )
    df_long = df_long.replace({"entity": entity_dict}, inplace=False)
    # cv_order = [entity_dict[c] for c in stats_df_plot["entity"].tolist()]  # sorted by CV!
    plotter.plot_categorical_scatter_fullresponse(df=df_long, x_col="entity", y_col="c_rating",
                                                  s_scatter=60, y_ticks_list=[1, 2, 3, 4], scatter_alpha=0.4,
                                                  response_id_col="response_id", color_scatter=rating_color_list,
                                                  save_path=result_path, save_name="c_ratings_scatter",
                                                  show_means=True, color_mean="black", s_mean=160,
                                                  show_se=True, show_sd=False,
                                                  color_se="black", se_capsize=7, label_title="",
                                                  label_x="", label_y="Rating", x_jitter=0.2, y_jitter=0.2,
                                                  lines=False, order=None, order_by="mean", fmt="svg")

    """
    GENERAL-LEVEL: is there consensus between people's rating behaviors? 
    Method 1: Krippendorff’s Alpha - Inter-Annotator Data Reliability Metric

    Measures the agreement among observers or raters. 
    Alpha compares the observed disagreement to the disagreement one 
    would expect by chance. 

    It ranges from 1 (perfect agreement) down to 0 (no better than chance); it can even be negative if there is 
    systematic disagreement (i.e. raters consistently oppose each other)
    α = 1: perfect agreement
    α = 0: Agreement is no better than random (essentially a lot of variability)
    α < 0: Active disagreement (people’s ratings consistently oppose each other)

    We use this repo to calculate it: https://github.com/grrrr/krippendorff-alpha/tree/master
    and send a nominal_metric to have the size of the punishment depend on the distance between the ratings. 
    """

    # we use interval_metric (not nominal or ratio_metric)

    alpha_obs = helper_funcs.krippendorff_alpha(data=ratings_matrix, metric=helper_funcs.interval_metric)

    # avoid re-calculating the permutations as it's heavy
    if not load or not (os.path.isfile(os.path.join(result_path, "Krippendorff_null_alphas.csv"))):
        null_alphas = helper_funcs.premutations_for_array(matrix=ratings_matrix,
                                                          metric_func=helper_funcs.krippendorff_alpha,
                                                          n_permutations=1000,
                                                          print_iter=True, metric=helper_funcs.interval_metric)
        null_alphas_df = pd.DataFrame(null_alphas)
        null_alphas_df.to_csv(os.path.join(result_path, "Krippendorff_null_alphas.csv"), index=False)
    else:
        null_alphas_df = pd.read_csv(os.path.join(result_path, "Krippendorff_null_alphas.csv"))
        null_alphas = null_alphas_df.to_numpy()

    alpha_p_value = (null_alphas >= alpha_obs).sum() / len(null_alphas)

    plotter.plot_null_hist(observed_alpha=alpha_obs, null_alphas=null_alphas_df,
                           parameter_name_xlabel="Krippendorff’s Alpha", save_path=result_path,
                           save_name=f"Krippendorff_hist", fmt="svg", observed_alpha_color="red", bins=50, alpha=0.7)
    """
    GENERAL-LEVEL: is there consensus between people's rating behaviors? 
    Kendall’s Coefficient of Concordance (W)
    Unlike α (which can be computed per item), Kendall’s W is a global measure of agreement among raters across 
    a set of items. It asks: Do participants, as a whole, have a similar ranking of the items? 
    If everyone tends to rate the same items high and the same items low, W will be close to 1. 
    If there is no consensus pattern in how items are rated, W will be near 0. >> OVERALL
     Kendall’s W is a single number for the whole set of items, not per item. 
     It won’t directly tell you which specific item has high or low variability; rather, it tells if there’s a broad consensus trend. 

    How it's calculated: 
    For each participant you take their 24 ratings and convert them to ranks (1 = lowest rating given by that person, 
    24 = highest rating, with tied ratings getting average ranks). Then, for each item you sum up its ranks across all 
    participants. If everyone had similar opinions, an item that is generally top-rated will have a high total rank sum, 
    and an item viewed poorly will have a low rank sum. Kendall’s W is derived from the variance of these rank sums. 

    If W comes out to, say, 0.4 or 0.5, that indicates a moderate agreement in how participants rank the items. 
    If W is very low (close to 0), it means different people have very different preferences or perceptions of which 
    items deserve higher ratings. 
    A very high W (close to 1) is unlikely in practice unless the items have an inherent ordering that 
    everyone recognizes – but anything significantly above 0 is evidence of some concordance.
    """

    chi2, p_value_w = friedmanchisquare(*ratings_matrix.T)
    n_raters = df_c.shape[0]
    n_items = len(item_columns)
    kendall_W = chi2 / (n_raters * (n_items - 1))

    # unify into a single table
    consensus_summary = pd.DataFrame({
        "Metric": ["Krippendorff_Alpha", "Kendall_W"],
        "Value": [alpha_obs, kendall_W],
        "p_value": [alpha_p_value, p_value_w]
    })
    consensus_summary.to_csv(os.path.join(result_path, "c_ratings_dispersion_overall.csv"), index=False)







    """
    Use clustering to see if there are items can be clustered to groups based on their distance from the diagonal:
    Actually, clustering when the number of clusters = 2 aligns PERFECTLY with the top 5 outliers..
    """

    nums_clusters = [2]
    for n in nums_clusters:
        kmeans = helper_funcs.perform_kmeans(df_pivot=long_data_mean_rating[["dist_from_diagonal"]],
                                             save_path=result_path,
                                             save_name=f"items_dist_from_diag_{n}",
                                             clusters=n, normalize=False)

        # repeat the figure from before, but now color the dots by their cluster belongings
        # add the cluster data to long_data_mean_rating
        df_plot = long_data_mean_rating.merge(kmeans[0], on="dist_from_diagonal", how="left")
        cluster_colors = {0: "#3C3744", 1: "#F49D37", 2: "#C2C1C2"}
        plotter.plot_scatter_xy(df=df_plot, identity_col="Item",
                                x_col="Consciousness", x_label="Consciousness", x_min=1, x_max=4, x_ticks=1,
                                y_col="Moral Status", y_label="Moral Status", y_min=1, y_max=4, y_ticks=1,
                                save_path=result_path, save_name=f"correlation_c_ms_clustering_{n}", annotate_id=True,
                                color_col_colors=cluster_colors, color_col="Cluster", corr_line=False, diag_line=True,
                                fmt="svg",
                                individual_df=None, id_col=None)







    """
    ****** FIRST OF ALL: JUST CONSCIOUSNESS (W/O MS) ******
    """



    """
    How many non-experts said LLM is conscious?  # Q FOR LIAD, TEMPORARY, RONY REMOVE
    """
    df_cexp = analysis_dict["consciousness_exp"][[process_survey.COL_ID, survey_mapping.Q_CONSC_EXP]]
    df_c_ai = df_c[[process_survey.COL_ID, "c_A large language model"]]
    merged = pd.merge(df_c_ai, df_cexp, on=process_survey.COL_ID)
    just_inexperienced = merged[merged[survey_mapping.Q_CONSC_EXP] <= 2]
    just_inexperienced.groupby("c_A large language model").count().reset_index()
    c = 3



    """
    Compare to the data from Francken et al., 2022 
    https://doi.org/10.1093/nc/niac011

    """
    francken_survey_path = os.path.join(r"C:\Users\Rony\Documents\projects\ethics\survey_analysis", "Francken et al 2022 ans.csv")
    francken_df = pd.read_csv(francken_survey_path)
    francken_c_rating_cols = [c for c in francken_df.columns.tolist() if c.startswith("entities_") or c.startswith("participant")]
    francken_df = francken_df[francken_c_rating_cols]  # just the relevant columns
    c_map = {"participant": process_survey.COL_ID,
             "entities_baby": "c_A newborn baby (human)",
             "entities_bat": "c_A bat",
             "entities_dog": "c_A dog",
             "entities_fish": "c_A salmon",
             "entities_monkey": "c_An orangutan",
             "entities_octopus": "c_An octopus",
             "entities_tree": "c_A tree",
             "entities_yourself": "c_You"}
    rating_cols = [c for c in c_map.values() if c != process_survey.COL_ID]

    francken_df = francken_df.rename(columns=c_map, inplace=False)
    francken_df = francken_df[[c for c in francken_df.columns if c in c_map.values()]]
    """
    In our data, not only the ratings are between 1-4 (and not 1-5) but the MAPPING ITSELF is the OPPOSITE of Francken!
    In our data: 1 = does not have consciousness, 4 = has consciousness
    So to compare Francken to us, we need to:
    (1) flip the 1-5 so 1 >> 5 etc (interprate 5 as high on both scales)
    (2) equate both sacles (turn both into 1-4)
    """
    # (1) flip
    flip_map = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}
    francken_df = francken_df.applymap(lambda x: flip_map.get(x, x))


    """
    Equate columns and add an experience column to ours and Francken's data
    """
    df_c_matched = df_c[[c for c in df_c.columns if c in c_map.values()]]
    # add an 'experience' column
    df_cexp = analysis_dict["consciousness_exp"][[process_survey.COL_ID, survey_mapping.Q_CONSC_EXP]]
    # binarize our experience: 1, 2 --> 0, 3 and up --> 1 [3 is "1" as the Francken paper included RA and undergrads]
    df_cexp[survey_mapping.Q_CONSC_EXP] = (df_cexp[survey_mapping.Q_CONSC_EXP] <= 2).astype(int)
    df_c_matched_cexp = pd.merge(df_c_matched, df_cexp, on=process_survey.COL_ID)

    # all the Francken participants were ASSC attendees so they are Consciousness experts:
    francken_df[survey_mapping.Q_CONSC_EXP] = 1  # BINARY




    """
    (2) Normalize ratings
    I will use use min-max scaling across everyone, to bring both datasets to a shared scale 
    BUT I DON'T normalize per person - to keep their natural rating behavior (and only adjust the bare minimum to be
    able to compare the two dfs). This way, someone who rates everything high still does, and A 4 still means “high,” 
    regardless of which dataset they came from. 
    """
    # min-max scaling WITHOUT normalizing per person
    scaler = MinMaxScaler(feature_range=(1, 4))
    francken_df[rating_cols] = scaler.fit_transform(francken_df[rating_cols])  # rescale Francken
    df_c_matched_cexp[rating_cols] = scaler.fit_transform(df_c_matched_cexp[rating_cols])  # shouldn't be different!

    # unify to compare (in R)
    francken_df["group"] = "Francken"
    df_c_matched_cexp["group"] = "RHNN"
    unified_c = pd.concat([df_c_matched_cexp, francken_df], ignore_index=True, sort=False)
    unified_c.rename(columns={survey_mapping.Q_CONSC_EXP: "experience"}, inplace=True)
    unified_c.to_csv(os.path.join(result_path, "c_rating_francken_comp.csv"), index=False)


    """
    Plot side-by-side: Mean consciousness rating per compared entity (split by group and experience)
    """

    import seaborn as sns
    import matplotlib.pyplot as plt

    custom_palette = {
        'RHNN | Exp:0': '#F7AF9D',
        'RHNN | Exp:1': '#F06542',
        'Francken | Exp:0': '#73B3BF',
        'Francken | Exp:1': '#468C98',
    }

    plt.figure(figsize=(14, 6))
    sns.set_style("ticks")
    plt.rcParams['font.family'] = "Calibri"

    value_vars = [col for col in unified_c.columns if col.startswith('c_')]
    df_long = unified_c.melt(
        id_vars=['response_id', 'experience', 'group'],
        value_vars=value_vars,
        var_name='entity',
        value_name='rating'
    )

    # Clean entity names (remove 'c_')
    df_long['entity_clean'] = df_long['entity'].str.replace('^c_', '', regex=True)
    # Combine group and experience
    df_long['group_experience'] = df_long['group'] + ' | Exp:' + df_long['experience'].astype(str)


    jitter_strength = 0.2
    df_long['rating_jittered'] = df_long['rating'] + np.random.uniform(-jitter_strength, jitter_strength,
                                                                       size=len(df_long))

    # Point plot for means and SE
    ax = sns.pointplot(
        data=df_long,
        x='entity', y='rating', hue='group_experience',
        dodge=0.6, markers='o', capsize=0.1, errorbar=('ci', 95),
        linestyles='', palette=custom_palette
    )

    # Overlay with jittered individual dots
    sns.stripplot(
        data=df_long,
        x='entity', y='rating_jittered', hue='group_experience',
        dodge=True, jitter=True, alpha=0.25, size=3, palette=custom_palette
    )

    # Remove duplicate legends
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), title='Group | Experience')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.xlabel('Entity', fontsize=14)
    plt.ylabel('Rating', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(ticks=[1, 2, 3, 4],fontsize=14)
    plt.title('Entity Ratings by Group and Experience (With Individual Ratings)')
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, f"francken_comp.svg"), format=f"svg", dpi=1000, bbox_inches='tight',
                pad_inches=0.01)



    rony = 4  ## RONYRONY stopped here FRANCKEN BOURGET



    """
    Cross people's ratings with their experience 
    """

    # experience columns
    animal_experience_df = analysis_dict["animal_exp"].loc[:,
                           [process_survey.COL_ID, survey_mapping.Q_ANIMAL_EXP]].rename(
        columns={survey_mapping.Q_ANIMAL_EXP: "exp_animals"}, inplace=False)
    ai_experience_df = analysis_dict["ai_exp"].loc[:, [process_survey.COL_ID, survey_mapping.Q_AI_EXP]].rename(
        columns={survey_mapping.Q_AI_EXP: "exp_ai"}, inplace=False)
    ethics_experience_df = analysis_dict["ethics_exp"].loc[:,
                           [process_survey.COL_ID, survey_mapping.Q_ETHICS_EXP]].rename(
        columns={survey_mapping.Q_ETHICS_EXP: "exp_ethics"}, inplace=False)
    con_experience_df = analysis_dict["consciousness_exp"].loc[:,
                        [process_survey.COL_ID, survey_mapping.Q_CONSC_EXP]].rename(
        columns={survey_mapping.Q_CONSC_EXP: "exp_consc"}, inplace=False)
    demos_df = analysis_dict["demographics"].loc[:, [process_survey.COL_ID, "How old are you?"]].rename(
        columns={"How old are you?": "age"}, inplace=False)

    experience_types = ["exp_animals", "exp_ai", "exp_ethics", "exp_consc"]
    item_cols = df_ms.columns[1:].tolist() + df_c.columns[1:].tolist()
    df_list = [df.copy(), animal_experience_df, ai_experience_df, ethics_experience_df, con_experience_df]
    df_with_experience = reduce(lambda left, right: pd.merge(left, right, on=[process_survey.COL_ID]), df_list)

    # permanova per each experience column to see if there's any type of experience that affects people's rating patterns
    permanova_list = list()
    posthoc_list = list()
    descriptives_list = list()
    for exp in experience_types:
        permanova_result, posthoc, descriptives = helper_funcs.permanova_on_pairwise_distances(data=df_with_experience,
                                                                                               columns=item_cols,
                                                                                               group_col=exp)
        permanova_result["experience type"] = [exp]
        posthoc["experience type"] = exp
        descriptives["experience type"] = exp
        permanova_list.append(permanova_result)
        posthoc_list.append(posthoc)
        descriptives_list.append(descriptives)
    permanova_df = pd.concat(permanova_list, ignore_index=True)
    permanova_df.to_csv(os.path.join(result_path, f"permanova_ratings_per_experience_types.csv"), index=False)
    posthoc_df = pd.concat(posthoc_list, ignore_index=True)
    posthoc_df.to_csv(os.path.join(result_path, f"permanova_ratings_per_experience_types_posthoc.csv"), index=False)
    descriptives_df = pd.concat(descriptives_list, ignore_index=True)
    descriptives_df.to_csv(os.path.join(result_path, f"permanova_ratings_per_experience_types_descriptives.csv"),
                           index=False)




    """
    For each sub calculate the average 'nature' C rating and the average 'non-nature' C rating. 
    Then calculate diff between them. Hypothesis: DIFF in the animal-loving cluster (blue) will be SMALLER than
    in the anthropocentric cluster (yellow). 
    """
    dataframes = [long_data.copy(), animal_experience_df, ai_experience_df, ethics_experience_df, con_experience_df,
                  demos_df]
    result_df = reduce(lambda left, right: pd.merge(left, right, on=[process_survey.COL_ID]), dataframes)
    result_df["non_human_animal"] = result_df["Item"].map(survey_mapping.other_creatures_isNonHumanAnimal)
    result_df["nature"] = result_df["Item"].map(survey_mapping.other_creatures_isTreeHugger)
    result_df.to_csv(os.path.join(result_path, "c_v_ms_long_entityCoded.csv"), index=False)
    result_df_c = result_df[result_df["Topic"] == "Consciousness"]
    result_df_c.to_csv(os.path.join(result_path, "c_ratings_long.csv"), index=False)

    result_df_c_filtered = result_df_c[[process_survey.COL_ID, "Rating", "nature"]]
    grouped = result_df_c_filtered.groupby([process_survey.COL_ID, "nature"])["Rating"].mean().unstack()

    # Rename the columns for clarity
    grouped.columns = ["avg_rating_nature0", "avg_rating_nature1"]

    # Calculate the difference
    grouped["avg_rating_diff"] = grouped["avg_rating_nature1"] - grouped["avg_rating_nature0"]

    # add cluster data
    grouped_merged = pd.merge(grouped, df_earth_cluster, on=process_survey.COL_ID)

    # absolute difference
    df_subset = grouped_merged[[process_survey.COL_ID, "avg_rating_nature0", "avg_rating_nature1", "avg_rating_diff", "Cluster"]].copy()
    df_subset["abs_avg_rating_diff"] = df_subset["avg_rating_diff"].abs()

    # Group by cluster
    groups = [group["abs_avg_rating_diff"].values for name, group in df_subset.groupby("Cluster")]

    from scipy.stats import f_oneway
    from scipy.stats import shapiro, levene

    # Shapiro-Wilk test for normality within each cluster
    normality_results = df_subset.groupby("Cluster")["avg_rating_diff"].apply(shapiro).apply(pd.Series)
    normality_results.columns = ["W statistic", "value"]

    # Levene's test for homogeneity of variance
    clustered_data = [group["avg_rating_diff"].values for _, group in df_subset.groupby("Cluster")]
    levene_stat, levene_p = levene(*clustered_data)

    # Perform one-way ANOVA
    anova_stat, anova_p = f_oneway(*clustered_data)

    # Output results
    normality_results, (levene_stat, levene_p), (anova_stat, anova_p)

    # turn categorical columns into numeric ones for linear modelling
    for col in ["Topic", "Item"]:
        label_encoder = LabelEncoder()
        result_df[col] = label_encoder.fit_transform(result_df[col])

    result_df.to_csv(os.path.join(result_path, "c_v_ms_long_coded.csv"), index=False)

    """
    Plot the ratings individually for each item 
    """

    sorting_method = None
    sorted_suffix = ""

    for topic_code, topic_name in topic_name_map.items():  # Consciousness, Moral Status
        df_topic = df.loc[:, [col for col in df.columns if col.startswith(f"{topic_code}_")]]
        df_topic.columns = df_topic.columns.str.replace(f'^{topic_code}_', '', regex=True)  # get rid of topix prefix
        df_topic.columns = df_topic.columns.str.replace(r'^.*? ', '', regex=True).str.title()  # get rid of "A ..."
        items_sansA = [s.split(' ', 1)[-1].title() if ' ' in s else s for s in items]  # do the same for items

        # Prepare data for each column in the dataframe, they represent the items
        stats = {}
        for col in items_sansA:
            stats[col] = helper_funcs.compute_stats(df_topic[col], possible_values=[1, 2, 3, 4])

        # Create DataFrame for plotting
        plot_data = {}
        for item, (proportions, mean_rating, std_dev, n) in stats.items():
            plot_data[item] = {
                "Proportion": proportions,
                "Mean": mean_rating,
                "Std Dev": std_dev,
                "N": n
            }

        # Define plot size and number of subplots
        num_plots = len(df_topic.columns)

        if sorting_method is None:
            # Sort the data by the proportion of "4" rating (Python 3.7+ dictionaries maintain the insertion order of keys)
            # sorted_plot_data = sorted(plot_data.items(), key=lambda x: x[1]['Proportion'].get(4, 0), reverse=True)

            # Sort the data by the MEAN rating (Python 3.7+ dictionaries maintain the insertion order of keys)
            sorted_plot_data = sorted(plot_data.items(), key=lambda x: x[1]['Mean'], reverse=True)
            if sort_together:  # if false, then it'll sort each of them by the previous condition independently
                sorting_method = list(dict(sorted_plot_data).keys())  # this is the order now
                sorted_suffix = "_sortTogether"

        else:  # sort the second column by the first one's order
            sorted_plot_data = {key: plot_data[key] for key in sorting_method if key in plot_data}.items()

        # plot
        plotter.plot_stacked_proportion_bars(plot_data=sorted_plot_data, num_plots=num_plots,
                                             legend_labels=rating_labels,
                                             colors=rating_color_list, num_ratings=4, title=f"{topic_name.title()}",
                                             bar_relative=False, bar_range_min=1, bar_range_max=4,
                                             save_path=result_path, fmt="svg",
                                             save_name=f"ratings_{topic_name.lower()}{sorted_suffix}")

    """
    Plot "other creatures" judgments of Consciousness vs. of Moral Status. >> *** PER ITEM ***
    """


    data = long_data.pivot(index=["response_id", "Item"], columns=["Topic"], values="Rating").reset_index(drop=False, inplace=False)

    # order the items
    count_df = data.groupby("Item").apply(
        lambda x: pd.Series({
            "count_4": ((x["Consciousness"] == 4) & (x["Moral Status"] == 4)).sum(),
            "count_3": ((x["Consciousness"] == 3) & (x["Moral Status"] == 3)).sum(),
            "count_2": ((x["Consciousness"] == 2) & (x["Moral Status"] == 2)).sum(),
            "count_1": ((x["Consciousness"] == 1) & (x["Moral Status"] == 1)).sum()
        })).reset_index()

    sorted_items_1234 = count_df.sort_values(by=["count_1", "count_2", "count_3", "count_4"],
                                             ascending=[True, True, True, True])["Item"].tolist()

    plotter.plot_multiple_scatter_xy(data=data, identity_col="response_id", x_col="Consciousness",
                                     y_col="Moral Status", x_label="Consciousness", y_label="Moral Status",
                                     x_min=1, x_max=4.2, x_ticks=1, y_min=1, y_max=4.2, y_ticks=1,
                                     save_path=result_path, save_name="correlation_c_ms_panels_sorted1234",
                                     palette_bounds=colors, annotate_id=False,
                                     fmt="svg", size=50, alpha=0.6, corr_line=True, diag_line=True,
                                     vertical_jitter=0.25, horizontal_jitter=0.25,
                                     panel_per_col="Item", panel_order=sorted_items_1234, rows=4, cols=6,
                                     title_size=20, axis_size=14, hide_axes_names=True,
                                     violins=True, violin_alpha=0.75, violin_color="#BAB9CB")

    correlation_per_item = (
        data.groupby("Item")[["Consciousness", "Moral Status"]]
        .corr()
        .unstack()
        .iloc[:, 1]
    )
    sorted_items_corr = correlation_per_item.sort_values(ascending=True).index.tolist()

    plotter.plot_multiple_scatter_xy(data=data, identity_col="response_id", x_col="Consciousness",
                                     y_col="Moral Status", x_label="Consciousness", y_label="Moral Status",
                                     x_min=1, x_max=4.2, x_ticks=1, y_min=1, y_max=4.2, y_ticks=1,
                                     save_path=result_path, save_name="correlation_c_ms_panels_sortedCorr",
                                     palette_bounds=colors, annotate_id=False,
                                     fmt="svg", size=50, alpha=0.6, corr_line=True, diag_line=True,
                                     vertical_jitter=0.25, horizontal_jitter=0.25,
                                     panel_per_col="Item", panel_order=sorted_items_corr, rows=4, cols=6,
                                     title_size=20, axis_size=14, hide_axes_names=True)

    std_per_item = (
        data.groupby("Item")[["Consciousness", "Moral Status"]]
        .std()
        .mean(axis=1)  # Average variability across both ratings
    )
    sorted_items_std = std_per_item.sort_values(ascending=True).index.tolist()
    plotter.plot_multiple_scatter_xy(data=data, identity_col="response_id", x_col="Consciousness",
                                     y_col="Moral Status", x_label="Consciousness", y_label="Moral Status",
                                     x_min=1, x_max=4.2, x_ticks=1, y_min=1, y_max=4.2, y_ticks=1,
                                     save_path=result_path, save_name="correlation_c_ms_panels_sortedSTD",
                                     palette_bounds=colors, annotate_id=False,
                                     fmt="svg", size=50, alpha=0.6, corr_line=True, diag_line=True,
                                     vertical_jitter=0.25, horizontal_jitter=0.25,
                                     panel_per_col="Item", panel_order=sorted_items_std, rows=4, cols=6,
                                     title_size=20, axis_size=14, hide_axes_names=True)


    """
    This is if we want each of them separately
    """
    #for item in long_data["Item"].unique().tolist():
    #    long_data_item = long_data[long_data["Item"] == item].drop(columns=["Item"], inplace=False)
    #    data_item = long_data_item.pivot(index="response_id", columns="Topic", values="Rating").reset_index(drop=False,
    #                                                                                                        inplace=False)
        # plot the relationship between consciousness and moral status ratings FOR THIS ITEM, ACROSS PEOPLE
    #    plotter.plot_scatter_xy(df=data_item, identity_col="response_id",
    #                            x_col="Consciousness", x_label="Consciousness", x_min=1, x_max=4.2, x_ticks=1,
    #                            y_col="Moral Status", y_label="Moral Status", y_min=1, y_max=4.2, y_ticks=1,
    #                            save_path=result_path, save_name=f"correlation_c_ms_{item}", fmt="svg",
    #                            annotate_id=False, palette_bounds=colors, corr_line=False, diag_line=True,
    #                            vertical_jitter=0.15, horizontal_jitter=0.15, size=250, alpha=0.6,
    #                            individual_df=None, id_col=None)




    return


def earth_in_danger(analysis_dict, save_path, cluster_num=2):
    """
    Answers to the "Earth is in danger who would you save?" section
    """
    # save path
    result_path = os.path.join(save_path, "earth_danger")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # load relevant data
    df_earth = analysis_dict["earth_in_danger"]
    questions = df_earth.columns[df_earth.columns != process_survey.COL_ID].tolist()

    # super-simple dumb pie charts to see how many overall selected each option
    for q in questions:
        df_q = df_earth.loc[:, [process_survey.COL_ID, q]]
        counts = df_q[q].value_counts()
        plotter.plot_pie(categories_names=counts.index.tolist(), categories_counts=counts.tolist(),
                         categories_labels=CAT_LABEL_DICT,
                         categories_colors=CAT_COLOR_DICT, title=f"{q}",
                         save_path=result_path, save_name=f"{'_'.join(counts.index.tolist())}", fmt="png")

    """
    Real stuff: Kmeans clustering, crossing with demographics
    """
    df_earth_coded = df_earth.copy()
    for col in questions:
        """
        The map should convert values into 0/1s for the PCA / Kmeans clustering. 
        PCA is a technique that works with numeric data to capture variance. 
        It projects the data into lower dimensions based on linear combinations of the features, so we can't use 
        categorizations (and map them to numbers as the model might falsely interpret these numeric encodings as 
        having ordinal relationships. 
        
        The same goes for Kmeans clustering, it relies on distance measures like Euclidean distance to determine 
        cluster centers, so if using arbitrary numbers for categories, the model will interpret these numbers as having 
        some sort of distance relationship. 
        
        So we will convert everything into binary. However, in order to keep interpretability, I will choose the
        0's and 1's myself (and not simple map each column into binary arbitrarily. 
        """
        col_map = survey_mapping.EARTH_DANGER_QA_MAP[col]
        df_earth_coded[col] = df_earth_coded[col].map(col_map)

    df_earth_coded.set_index([process_survey.COL_ID], inplace=True)

    """
    Perform k-means clustering: group the choices into (k) clusters based on feature similarity.
    Each cluster is represented by a "centroid" (average position of the data points in the cluster).
    Data points are assigned to the cluster whose centroid they are closest to.
    """
    df_pivot, kmeans, cluster_centroids = helper_funcs.perform_kmeans(df_pivot=df_earth_coded, clusters=cluster_num,
                                                                      save_path=result_path, save_name="items")

    """
    Plot the KMeans cluster centroids: this is important. 
    For each cluster (we have cluster_num clusters total), the centroid is the average data point for this cluster 
    (the mean value of the features for all data points in the cluster). We use the centroids to visualize each cluster's
    choice in each earth-is-in-danger dyad, to interpret the differences between them.  
    """

    # Compute the cluster centroids and SEMs
    #cluster_centroids = df_pivot.groupby("Cluster").mean()  # we get this from helper_funcs.perform_kmeans
    cluster_sems = df_pivot.groupby("Cluster").sem()

    # Plot - collapsed (all clusters together)
    helper_funcs.plot_cluster_centroids(cluster_centroids=cluster_centroids, cluster_sems=cluster_sems,
                                        save_path=result_path, save_name="items", fmt="svg",
                                        label_map=survey_mapping.EARTH_DANGER_QA_MAP, binary=True,
                                        label_names_coding=survey_mapping.EARTH_DANGER_ANS_MAP,
                                        threshold=0, overlaid=True, cluster_colors_overlaid=["#EDAE49", "#102E4A"])

    """
    Plot k-means clusters in PCA space: combine the KMeans cluster assignments with the PCA-transformed data into 
    one dataset, to visualize the clusters in the reduced PCA space using a scatter plot.
    This way, we can interpret the clustering in terms of the main patterns of *variation* in the data. 
    In this context we are not really interested in the PCA other than for plotting reasons, as the KMeans is the
    more informative test. 
    """
    # Perform PCA
    pca_df, loadings, explained_variance = helper_funcs.perform_PCA(df_pivot=df_earth_coded, save_path=result_path,
                                                                    save_name="items", components=cluster_num)
    pca_with_cluster = helper_funcs.plot_kmeans_on_PCA(df_pivot=df_pivot, pca_df=pca_df,
                                                       save_path=result_path, save_name="items")
    pca_with_cluster.reset_index(inplace=True, drop=False)

    """
    Examine the clusters demographically
    """
    # then, we want to examine if there are any demographics that are shared within each cluster.
    df_demog = analysis_dict["demographics"]
    df_animalexp = analysis_dict["animal_exp"]
    df_ethicsexp = analysis_dict["ethics_exp"]
    df_aiexp = analysis_dict["ai_exp"]
    df_cexp = analysis_dict["consciousness_exp"]
    df_list = [df_demog, df_animalexp, df_ethicsexp, df_aiexp, df_cexp]
    unified_df = reduce(lambda x, y: x.merge(y, on=process_survey.COL_ID), df_list)
    unified_df_cluster = pd.merge(unified_df, pca_with_cluster[[process_survey.COL_ID, "Cluster"]],
                                  on=process_survey.COL_ID, how="left")
    unified_df_cluster.rename(columns=survey_mapping.Q_EXP_DICT, inplace=True)
    unified_df_cluster.to_csv(os.path.join(result_path, "clusters_with_demographic.csv"), index=False)

    # Perform an independent-sample t-test to see difference in age between the two clusters
    group1 = unified_df_cluster[unified_df_cluster["Cluster"] == 0]["How old are you?"].tolist()
    group2 = unified_df_cluster[unified_df_cluster["Cluster"] == 1]["How old are you?"].tolist()
    continuous_contingency_df = helper_funcs.independent_samples_ttest(list_group1=group1, list_group2=group2)
    continuous_contingency_df[f"per"] = ["age"]

    # Perform a Mann-Whitney U test to test the association between clusters and ordinal data
    ordinal_contingency_cols = {"Education": "What is your education background?",
                                "Experience with Ethics": "exp_ethics",
                                "Experience with Consciousness": "exp_consciousness",
                                "Experience with Animals": "exp_animals",
                                "Experience with AI": "exp_ai"}
    ordinal_contingency_list = list()
    for col in ordinal_contingency_cols:
        if col == "Education":
            unified_df_cluster.loc[:, ordinal_contingency_cols[col]] = unified_df_cluster[
                ordinal_contingency_cols[col]].map(survey_mapping.EDU_MAP)
        group1 = unified_df_cluster[unified_df_cluster["Cluster"] == 0][ordinal_contingency_cols[col]].tolist()
        group2 = unified_df_cluster[unified_df_cluster["Cluster"] == 1][ordinal_contingency_cols[col]].tolist()
        mu_result = helper_funcs.mann_whitney_utest(list_group1=group1, list_group2=group2)
        mu_result[f"per"] = [col]
        ordinal_contingency_list.append(mu_result)
        transformed_df = unified_df_cluster.pivot(index=process_survey.COL_ID, columns="Cluster",
                                                  values=ordinal_contingency_cols[col]).reset_index()
        transformed_df.columns = [process_survey.COL_ID, "0", "1"]
        plotter.plot_raincloud_separate_samples(df=transformed_df, id_col=process_survey.COL_ID,
                                                data_col_names=["0", "1"],
                                                data_col_colors={"0": "#EDAE49", "1": "#102E4A"},
                                                save_path=result_path, save_name=f"clusters_by_{col}",
                                                x_title="Cluster", x_name_dict={"0": "0", "1": "1"},
                                                title="", y_title="Ratings",
                                                ymin=min(unified_df_cluster[ordinal_contingency_cols[col]].tolist()),
                                                ymax=max(unified_df_cluster[ordinal_contingency_cols[col]].tolist()),
                                                yskip=1, fmt="png")
    # now after the demographics are coded
    unified_df_cluster.to_csv(os.path.join(result_path, "clusters_with_demographic_coded.csv"), index=False)

    ordinal_contingency_df = pd.concat(ordinal_contingency_list)

    # Perform a Chi-squared test to examine association between the cluster and gender
    gender_contingency_table = pd.crosstab(unified_df_cluster["Cluster"],
                                           unified_df_cluster["How do you describe yourself?"])
    chisquare_gender = helper_funcs.chi_squared_test(contingency_table=gender_contingency_table)
    chisquare_gender[f"per"] = ["Gender"]

    # Perform a Chi-squared test to examine association between the cluster and country of residence
    country_contingenty_table = pd.crosstab(unified_df_cluster["Cluster"],
                                            unified_df_cluster["In which country do you currently reside?"])
    chisquare_country, expected_df = helper_funcs.chi_squared_test(contingency_table=country_contingenty_table,
                                                                   include_expected=True)
    chisquare_country[f"per"] = ["Country"]

    # together
    clusters_demographic_tests = pd.concat(
        [continuous_contingency_df, ordinal_contingency_df, chisquare_gender, chisquare_country], ignore_index=True)
    clusters_demographic_tests.to_csv(os.path.join(result_path, "clusters_by_demographic_stats.csv"), index=False)

    """
    Examine the standardized residuals to identify which countries contributed most to the difference.
    Residuals show how much the observed counts deviate from the expected counts for each cluster-country pair. 
    Standardized residuals are a way to measure the difference between observed and expected frequencies in a 
    contingency table while accounting for the variability in the data. If the null hypothesis it true, then the
    standardized residuals should follow an approximately standard normal distribution; meaning, about 95% of the values
    should fall within 2 standard deviations of the mean. 
    Haberman, S. J. (1973). The analysis of residuals in cross-classified tables. Biometrics, 205-220.
    Agresti, A. (2012). Categorical data analysis (Vol. 792). John Wiley & Sons. [p.81]
    """
    standardized_residuals = (country_contingenty_table - expected_df) / np.sqrt(expected_df)
    # values outside this range have a 5% probability under the null hypothesis
    significant_residuals = standardized_residuals[standardized_residuals.abs() > 2].dropna(how="all", axis=0).dropna(
        how="all", axis=1)
    significant_countries = significant_residuals.columns
    significant_contingency_table = country_contingenty_table[significant_countries]
    # distribution of countries within each cluster
    significant_distribution = significant_contingency_table.div(significant_contingency_table.sum(axis=1),
                                                                 axis=0) * 100
    # significant residuals with distributions for interpretation
    significant_interpretation_df = pd.concat(
        [significant_contingency_table, significant_residuals, significant_distribution],
        keys=["Observed", "Residuals", "Percentage"], axis=1)
    print("Examination of residuals: significant countries interpretation df:")
    print(significant_interpretation_df)
    print("")
    significant_interpretation_df.to_csv(os.path.join(result_path, f"clusters_by_country_residuals_significant.csv"),
                                         index=True)
    """
    Interpreting Significant Residuals: 
    Residuals - (significant_residuals)
    Positive residuals: The country is *overrepresented* in the cluster compared to what is expected.
    (e.g., a positive residual for "US" in cluster 0 >> people from the US are more likely to belong to cluster 0)
    Negative residuals: The country is *underrepresented* in the cluster compared to what is expected.
    (e.g., a negative residual for "US" in cluster 1 >> people from the US are less likely to belong to cluster 1)
    Distribution - (significant_distribution)
    Showing how individuals are distributed across clusters
    """

    # proportions for each country within each cluster
    country_proportions = country_contingenty_table.div(country_contingenty_table.sum(axis=1), axis=0)
    country_proportions.to_csv(os.path.join(result_path, f"clusters_by_country_proportions_in_cluster.csv"), index=True)

    # return the df with the coded answers and cluster taggings for further analyses
    df_pivot.reset_index(drop=False, inplace=True)
    df_pivot.to_csv(os.path.join(result_path, f"earth_danger_clusters.csv"), index=False)
    return df_pivot


def calculate_ics_proportions(df_ics, save_path, prefix="", suffix=""):
    questions = [c for c in survey_mapping.ICS_Q_NAME_MAP.keys()]
    ans_map = {survey_mapping.ANS_NO: 0, survey_mapping.ANS_YES: 1}
    # plot a collapsed figure where each creature is a bar, with the proportion of how many would kill it
    stats = dict()
    labels = list()
    for q in questions:
        df_q = df_ics.loc[:, [process_survey.COL_ID, q]]
        q_name = survey_mapping.ICS_Q_NAME_MAP[q]
        labels.append(q_name)
        df_q_map = df_q.replace({q: ans_map})
        stats[q_name] = helper_funcs.compute_stats(df_q_map[q], possible_values=df_q_map[q].unique().tolist())

    # Create DataFrame for plotting
    plot_data = {}
    for item, (proportions, mean_rating, std_dev, n) in stats.items():
        plot_data[item] = {
            "Proportion": proportions,
            "Mean": mean_rating,
            "Std Dev": std_dev,
            "N": n
        }
    rating_labels = [survey_mapping.ANS_NO, survey_mapping.ANS_YES]
    rating_colors = [CAT_COLOR_DICT[survey_mapping.ANS_NO], CAT_COLOR_DICT[survey_mapping.ANS_YES]]
    sorted_plot_data = sorted(plot_data.items(), key=lambda x: x[1]["Mean"], reverse=True)
    plotter.plot_stacked_proportion_bars(plot_data=sorted_plot_data, num_plots=4, legend_labels=rating_labels,
                                         ytick_visible=True, text_width=39,
                                         title=f"Do you think a creature/system can be",
                                         show_mean=False, sem_line=False,
                                         colors=rating_colors, num_ratings=2,
                                         annotate_bar=True, annot_font_color="#e0e1dd",
                                         save_path=save_path, save_name=f"{prefix}ics{suffix}", fmt="svg")
    # save data
    plot_df = pd.DataFrame(plot_data)
    plot_df.to_csv(os.path.join(save_path, f"{prefix}ics{suffix}.csv"), index=True)
    return plot_df


def ics_group_map(row):  # x
    if row[survey_mapping.ICS_Q_NAME_MAP[survey_mapping.ICS_Q_CONS_WO_INT]] == 0 and row[survey_mapping.ICS_Q_NAME_MAP[survey_mapping.ICS_Q_CONS_WO_SENS]] == 0:
        return "Multidimensional"
    if row[survey_mapping.ICS_Q_NAME_MAP[survey_mapping.ICS_Q_CONS_WO_INT]] == 0 and row[survey_mapping.ICS_Q_NAME_MAP[survey_mapping.ICS_Q_CONS_WO_SENS]] == 1:
        return "Cognitive"
    if row[survey_mapping.ICS_Q_NAME_MAP[survey_mapping.ICS_Q_CONS_WO_INT]] == 1 and row[survey_mapping.ICS_Q_NAME_MAP[survey_mapping.ICS_Q_CONS_WO_SENS]] == 0:
        return "Valence"
    if row[survey_mapping.ICS_Q_NAME_MAP[survey_mapping.ICS_Q_CONS_WO_INT]] == 1 and row[survey_mapping.ICS_Q_NAME_MAP[survey_mapping.ICS_Q_CONS_WO_SENS]] == 1:
        return "Other"


def ics(analysis_dict, save_path, df_earth_cluster=None, ms_features_order_df=None, feature_colors=None):
    """
    Answers to the "Do you think a creature/system can have intentions/consciousness/sensations w/o having..?" section
    """
    # save path
    result_path = os.path.join(save_path, "i_c_s")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # load relevant data
    df_ics = analysis_dict["ics"].copy()
    df_ics.to_csv(os.path.join(result_path, "i_c_s.csv"), index=False)

    # to cross things with consciousness expertise:
    exp_cons = analysis_dict["consciousness_exp"].copy().loc[:, [process_survey.COL_ID, survey_mapping.Q_CONSC_EXP]]
    exp_cons.rename(columns={survey_mapping.Q_CONSC_EXP: "exp_consc"})

    # calculate the proportions of "yes"/"no" answers to the questions (and plot)
    calculate_ics_proportions(df_ics=df_ics, save_path=result_path, suffix=f"_all")



    """
    Do the answers change based on expertise with consciousness?
    """
    exp_cons = analysis_dict["consciousness_exp"].copy().loc[:, [process_survey.COL_ID, survey_mapping.Q_CONSC_EXP]]
    exp_cons.rename(columns={survey_mapping.Q_CONSC_EXP: "exp_consc"}, inplace=True)
    exp_cons["binary_exp_consc"] = exp_cons["exp_consc"].apply(lambda x: 1 if x >= 4 else 0)

    questions = [c for c in survey_mapping.ICS_Q_NAME_MAP.keys()]

    chisquare_result_list = list()
    for q in questions:
        df_q = df_ics.loc[:, [process_survey.COL_ID, q]]
        q_name = survey_mapping.ICS_Q_NAME_MAP[q]
        df_q_unified = pd.merge(df_q, exp_cons, on=process_survey.COL_ID)
        df_q_unified.rename(columns={q: q_name}, inplace=True)
        """
        create a contingency table for a chi-squared test between answer to the killing question (q_name), 
        and expertise in consciousness (binary_exp_consc)
        """
        # binary! expert vs. non-expert
        contingency_table = pd.crosstab(df_q_unified[q_name], df_q_unified["binary_exp_consc"])
        contingency_table.to_csv(os.path.join(result_path, f"c_expertise_per_kill_{q_name}_contingency.csv"))
        chisquare_result = helper_funcs.chi_squared_test(contingency_table=contingency_table)
        chisquare_result["question"] = q_name
        chisquare_result_list.append(chisquare_result)

        # for plotting
        count_df = df_q_unified.groupby([q_name, "exp_consc"]).size().reset_index(name="count")
        group_totals = df_q_unified.groupby(q_name).size().reset_index(name="total")
        count_df = count_df.merge(group_totals, on=q_name)
        count_df["Proportion"] = (count_df["count"] / count_df["total"]) * 100

        pivot_df = count_df.pivot(index="exp_consc", columns=q_name, values="Proportion").fillna(0).reset_index(drop=False, inplace=False)
        pivot_df.to_csv(os.path.join(result_path, f"c_expertise_per_kill_{q_name}.csv"), index=False)
        plotter.plot_expertise_proportion_bars(df=pivot_df, cols=["No", "Yes"],
                                               cols_colors={"No": '#B33A00', "Yes": '#3B4E58'},
                                               x_axis_exp_col_name="exp_consc",
                                               x_label="Reported experience with consciousness",
                                               y_ticks=[0, 25, 50, 75, 100],
                                               save_name=f"c_expertise_per_kill_{q_name}",
                                               save_path=result_path, plt_title=q_name)

    chisquare_result_total = pd.concat(chisquare_result_list)
    chisquare_result_total.to_csv(os.path.join(result_path, f"chisqaured_c_expertise_per_kill.csv"), index=False)
    d = 3


    """
    Follow up questions: examples for cases of X w/o Y
    """
    example_q_prefix = "Do you have an example of a case of "
    followup_cols = [c for c in df_ics.columns if c.startswith(example_q_prefix)]
    ics_followup = df_ics.loc[:, [process_survey.COL_ID] + followup_cols]

    for col in followup_cols:
        col_savename = col.removeprefix(example_q_prefix).removesuffix("?").replace("/", "_")
        ics_q = ics_followup[ics_followup[col].notnull()]  # actually wrote something in the 'example' field
        ics_q = ics_q[[process_survey.COL_ID, col]]
        ics_q = ics_q[~ics_q[col].str.strip().str.fullmatch(r"No[.,!?]*",
                                                            flags=re.IGNORECASE)]  # and that something isn't a variation of JUST  a *"no"* (some "no, but..blabla" will appear)
        # unify with consciousness experience
        ics_q_unified = pd.merge(ics_q, exp_cons, on=process_survey.COL_ID)
        ics_q_unified.to_csv(os.path.join(result_path, f"{col_savename}.csv"), index=False)  # save answers for examination
        """
        Topic modelling of free text
        """
        helper_funcs.topic_modelling(df=ics_q, text_column=col, save_path=result_path, save_name=col_savename,
                                     num_topics=None)
    c = 3


    """
    Group by people's answers and see how many people are in each group
    - Valence group: think it's possible to have C w/o intentions, but NOT w/o sensations
    - Cognitive group: think it's possible to have C without sensations, but NOT w/o intentions
    - Muldidim group: think it's impossible to have C without either intentions or sensations
    - Other group: think it's possible to have C without the others (Both)
    """

    group_names_map = {"Multidimensional": "Not possible w/o either sensations or intentions",
                       "Cognitive": "Possible w/o sensations, but not w/o intentions",
                       "Valence": "Possible w/o intentions, but not w/o sensations",
                       "Other": "Possible w/o both sensations and intentions"}

    df_qs = df_ics[[process_survey.COL_ID] + list(survey_mapping.ICS_Q_NAME_MAP.keys())]
    df_qs.rename(columns=survey_mapping.ICS_Q_NAME_MAP, inplace=True)
    df_qs.to_csv(os.path.join(result_path, "i_c_s_ans.csv"), index=False)

    c_wo_stuff_cols = [c for c in df_qs.columns if c.startswith("Consciousness")]
    df_relevant = df_qs[[process_survey.COL_ID] + c_wo_stuff_cols]
    df_relevant[c_wo_stuff_cols] = df_relevant[c_wo_stuff_cols].replace(survey_mapping.ANS_YESNO_MAP)
    result = df_relevant.groupby(c_wo_stuff_cols)[process_survey.COL_ID].nunique().reset_index(name="count")
    result["proportion"] = 100 * result["count"] / (result["count"].sum())
    # this is the df where columns are: Consciousness wo Intentions, Consciousness wo Sensations, count, proportion, Group
    result["Group"] = result.apply(lambda row: ics_group_map(row), axis=1)
    result.to_csv(os.path.join(result_path, "c_wo_stuff_groups.csv"), index=False)
    # plot this
    result["Group Name"] = result["Group"].map(group_names_map)
    color_list = ["#e29578", "#ffddd2", "#dedbd2", "#84a98c"]
    plotter.plot_categorical_bars(categories_prop_df=result, data_col="proportion",
                                  order=["Multidimensional", "Cognitive", "Valence", "Other"],
                                  category_col="Group Name", y_min=0, y_max=105, y_skip=10,
                                  delete_y=False, add_pcnt=True,
                                  categories_colors=color_list, text_wrap_width=16,
                                  save_path=result_path, save_name=f"ics_all_necessary ingredients", fmt="svg")

    """
    Do these groups vary based on CONSCIOUSNESS expertise?
    Binarize experience with consciousnss, then do chi squared on experience * group
    """
    expertise_level = 4
    # tag them per group
    df_relevant_tagged = df_relevant.copy()
    df_relevant_tagged["Group"] = df_relevant_tagged.apply(lambda row: ics_group_map(row), axis=1)

    exp_consciousness = analysis_dict["consciousness_exp"].copy().loc[:, [process_survey.COL_ID, survey_mapping.Q_CONSC_EXP]]
    exp_consciousness[f"binary_{survey_mapping.Q_CONSC_EXP}"] = exp_consciousness[survey_mapping.Q_CONSC_EXP].apply(lambda x: 1 if x >= expertise_level else 0)

    merged = pd.merge(df_relevant_tagged, exp_consciousness, on=process_survey.COL_ID)

    """
    create a contingency table for a chi-squared test 
    """
    contingency_table = pd.crosstab(merged[f"binary_{survey_mapping.Q_CONSC_EXP}"], merged["Group"])
    chisquare_result = helper_funcs.chi_squared_test(contingency_table=contingency_table)
    chisquare_result.to_csv(os.path.join(result_path, "c_wo_stuff_conscExpertise.csv"), index=False)

    count_df = merged.groupby([survey_mapping.Q_CONSC_EXP, "Group"]).size().reset_index(name="count")
    group_totals = merged.groupby(survey_mapping.Q_CONSC_EXP).size().reset_index(name="total")
    count_df = count_df.merge(group_totals, on=survey_mapping.Q_CONSC_EXP)
    count_df["Proportion"] = (count_df["count"] / count_df["total"]) * 100

    pivot_df = count_df.pivot(index=survey_mapping.Q_CONSC_EXP, columns="Group", values="Proportion").fillna(0).reset_index(drop=False, inplace=False)
    pivot_df.to_csv(os.path.join(result_path, "ics_group_per_cons_exp_props.csv"), index=False)
    plotter.plot_expertise_proportion_bars(df=pivot_df, cols=["Multidimensional", "Cognitive", "Valence", "Other"],
                                           cols_colors={"Multidimensional": "#e29578", "Cognitive": "#ffddd2", "Valence": "#dedbd2", "Other": "#84a98c"},
                                           x_axis_exp_col_name=survey_mapping.Q_CONSC_EXP,
                                           x_label="Reported experience with consciousness",
                                           y_ticks=[0, 25, 50, 75, 100],
                                           save_name="ics_group_per_cons_exp_props",
                                           save_path=result_path, plt_title="Group")
    x = 3


    """
    Do these groups belong to different clusters in the EiD dilemma?
    """
    # tag them per group
    df_relevant_tagged = df_relevant.copy()
    df_relevant_tagged["Group"] = df_relevant_tagged.apply(lambda row: ics_group_map(row), axis=1)

    if df_earth_cluster is not None:
        df_earth_relevant = df_earth_cluster[[process_survey.COL_ID, "Cluster"]]
        merged = pd.merge(df_relevant_tagged, df_earth_relevant, on=process_survey.COL_ID)
        """
        create a contingency table for a chi-squared test 
        """
        contingency_table = pd.crosstab(merged["Cluster"], merged["Group"])
        chisquare_result = helper_funcs.chi_squared_test(contingency_table=contingency_table)
        chisquare_result.to_csv(os.path.join(result_path, "c_wo_stuff_clusters.csv"), index=False)

        count_df = merged.groupby(["Cluster", "Group"]).size().reset_index(name="count")
        group_totals = merged.groupby("Cluster").size().reset_index(name="total")
        count_df = count_df.merge(group_totals, on="Cluster")
        count_df["Proportion"] = (count_df["count"] / count_df["total"]) * 100
        plotter.plot_categorical_bars_hued(categories_prop_df=count_df,
                                           x_col="Group", x_label=f"ICS Group",
                                           order=["Multidimensional", "Cognitive", "Valence", "Other"],
                                           category_col="Cluster", data_col="Proportion",
                                           categories_colors={0: "#EDAE49", 1: "#102E4A"},
                                           save_path=result_path, save_name=f"ics_group_per_EiDCluster", fmt="svg",
                                           y_min=0, y_max=101, y_skip=10, delete_y=False,
                                           inch_w=15, inch_h=12, add_pcnt=False)






    """
    Do these groups vary in the most important features for MS?
    """
    if ms_features_order_df is not None:
        ms_important_q = "What do you think is important for moral considerations?"
        ms_most_important_q = "Which do you think is the most important for moral considerations?"

        ms_features = analysis_dict["moral_considerations_features"].copy()
        # if the 'most important' column is nan, it means they only selected 1 feature to begin with
        ms_features[ms_most_important_q] = ms_features[ms_most_important_q].replace('nan', pd.NA)
        ms_features[ms_most_important_q] = ms_features[ms_most_important_q].fillna(ms_features[ms_important_q])

        ms_with_group = pd.merge(ms_features, df_relevant_tagged, on=process_survey.COL_ID).reset_index(drop=True, inplace=False)
        groups = ms_with_group["Group"].unique().tolist()
        for group in groups:
            ms_features_group = ms_with_group[ms_with_group["Group"] == group]
            ms_features_order_df, feature_colors, _ = calculate_moral_consideration_features(
                ms_features_df=ms_features_group,
                result_path=result_path,
                save_prefix=f"ms_features_{group}_",
                feature_order_df=ms_features_order_df,
                feature_color_dict=feature_colors)


        relevant_cols = [process_survey.COL_ID, ms_important_q, ms_most_important_q, "Group"]
        ms_features_relevant = ms_with_group[relevant_cols]
        """
        create a contingency table for a chi-squared test 
        """
        contingency_table = pd.crosstab(ms_features_relevant[ms_most_important_q], ms_features_relevant["Group"])
        chisquare_result = helper_funcs.chi_squared_test(contingency_table=contingency_table)
        chisquare_result.to_csv(os.path.join(result_path, "c_wo_stuff_most_important_feature.csv"), index=False)

        count_df = ms_features_relevant.groupby([ms_most_important_q, "Group"]).size().reset_index(name="count")
        group_totals = ms_features_relevant.groupby(ms_most_important_q).size().reset_index(name="total")
        count_df = count_df.merge(group_totals, on=ms_most_important_q)
        count_df["Proportion"] = (count_df["count"] / count_df["total"]) * 100
        count_df = count_df.sort_values(by=["Group", "Proportion"], ascending=[True, False])



    """
    Do these groups have a different stance on the binary/graded C question?
    """
    graded_c = analysis_dict["consciousness_graded"].copy()
    graded_c_unequal = graded_c[[process_survey.COL_ID, survey_mapping.Q_GRADED_UNEQUAL]]
    merged = pd.merge(graded_c_unequal, df_relevant_tagged[[process_survey.COL_ID, "Group"]], on=process_survey.COL_ID)
    """
    create a contingency table for a chi-squared test 
    """
    contingency_table = pd.crosstab(merged[survey_mapping.Q_GRADED_UNEQUAL], merged["Group"])
    chisquare_result = helper_funcs.chi_squared_test(contingency_table=contingency_table)
    chisquare_result.to_csv(os.path.join(result_path, "c_wo_stuff_c_graded.csv"), index=False)

    count_df = merged.groupby([survey_mapping.Q_GRADED_UNEQUAL, "Group"]).size().reset_index(name="count")
    group_totals = merged.groupby(survey_mapping.Q_GRADED_UNEQUAL).size().reset_index(name="total")
    count_df = count_df.merge(group_totals, on=survey_mapping.Q_GRADED_UNEQUAL)
    count_df["Proportion"] = (count_df["count"] / count_df["total"]) * 100
    plotter.plot_categorical_bars_hued(categories_prop_df=count_df,
                                       x_col="Group", x_label=f"ICS Group",
                                       category_col=survey_mapping.Q_GRADED_UNEQUAL, data_col="Proportion",
                                       categories_colors={1: "#DB5461", 2: "#fb9a99", 3: "#70a0a4", 4: "#26818B"},
                                       save_path=result_path, save_name=f"ics_group_per_gradedC", fmt="svg",
                                       y_min=0, y_max=101, y_skip=10, delete_y=False,
                                       inch_w=15, inch_h=12, add_pcnt=True, order=None)


    """
    Do these groups have different attitudes toward the relationship between consciousness and intelligence?
    """
    con_intel = analysis_dict["con_intellect"].copy()
    intel_q = "Do you think consciousness and intelligence are related?"
    con_intel_relevant = con_intel[[process_survey.COL_ID, intel_q]]
    merged = pd.merge(con_intel_relevant, df_relevant_tagged[[process_survey.COL_ID, "Group"]], on=process_survey.COL_ID)

    """
    Multinomial Logistic Regression
    his is the go-to method when your dependent variable is categorical with more than two unordered levels (Group), 
    and you have independent variables (one binary predictor: intel_q).
    It models the log-odds of being in each of the outcome groups relative to a reference group.
    It can tell you how the binary factor affects the probability of being in one group versus another.
    
    Question: does intelligence answer predict belonging to the COGNITIVE GROUP?
    """
    import statsmodels.api as sm
    multinom_df = merged.copy()
    multinom_df[f"binary_{intel_q}"] = multinom_df[intel_q].map({'Yes': 1, 'No': 0})
    # use "Cognitive" as a reference category
    multinom_df["Group"] = pd.Categorical(multinom_df["Group"], categories=["Cognitive", "Multidimensional", "Valence", "Other"], ordered=False)
    X = sm.add_constant(multinom_df[f"binary_{intel_q}"])  # Add intercept
    y = multinom_df["Group"]
    model = sm.MNLogit(y, X)
    result = model.fit()

    with open(os.path.join(result_path, "ics_groups_intelligence_modelSummary.txt"), "w") as f:
        print("=== Multinomial Logistic Regression Summary ===\n", file=f)
        print(result.summary(), file=f)
        print(f"Ref category: {multinom_df['Group'].cat.categories}")
        print("\n=== Odds Ratios ===\n", file=f)
        print(np.exp(result.params), file=f)

        print("\n=== P-values ===\n", file=f)
        print(result.pvalues, file=f)

    params = result.params
    odds_ratios = np.exp(params)
    p_values = result.pvalues
    summary_table = pd.DataFrame({
        'Estimate': params.loc[f"binary_{intel_q}"],
        'Odds_Ratio': odds_ratios.loc[f"binary_{intel_q}"],
        'P_value': p_values.loc[f"binary_{intel_q}"]
    })
    summary_table.to_csv(os.path.join(result_path, "ics_groups_intelligence_modelSummary.csv"), index=True)

    rony = 3


    # extra step - unify "cognitive" and "other"
    to_unify = ["Cognitive", "Other"]
    merged["Group"] = merged["Group"].replace(to_unify, "Cognitive_Other")

    """
    create a contingency table for a chi-squared test 
    """
    contingency_table = pd.crosstab(merged[intel_q], merged["Group"])
    chisquare_result = helper_funcs.chi_squared_test(contingency_table=contingency_table)
    chisquare_result.to_csv(os.path.join(result_path, "c_wo_stuff_c_intelligence_3_groups.csv"), index=False)

    count_df = merged.groupby([intel_q, "Group"]).size().reset_index(name="count")
    group_totals = merged.groupby(intel_q).size().reset_index(name="total")
    count_df = count_df.merge(group_totals, on=intel_q)
    count_df["Proportion"] = (count_df["count"] / count_df["total"]) * 100
    plotter.plot_categorical_bars_hued(categories_prop_df=count_df,
                                       x_col="Group", x_label=f"ICS Group",
                                       category_col=intel_q, data_col="Proportion",
                                       categories_colors={"No": "#B53B03", "Yes": "#3C5968"},
                                       save_path=result_path, save_name=f"ics_group_per_conIntelligence_3_groups", fmt="svg",
                                       y_min=0, y_max=101, y_skip=10, delete_y=False,
                                       inch_w=15, inch_h=12, add_pcnt=True, order=None)


    """
    Do these groups have different attitudes toward the zombie pill?
    """
    zombie_df = analysis_dict["zombification_pill"].copy()
    merged = pd.merge(zombie_df, df_relevant_tagged[[process_survey.COL_ID, "Group"]], on=process_survey.COL_ID)
    """
    create a contingency table for a chi-squared test 
    """
    zombie_q = "Would you take the pill?"
    contingency_table = pd.crosstab(merged[zombie_q], merged["Group"])
    chisquare_result = helper_funcs.chi_squared_test(contingency_table=contingency_table)
    chisquare_result.to_csv(os.path.join(result_path, "c_wo_stuff_zombie.csv"), index=False)

    count_df = merged.groupby([zombie_q, "Group"]).size().reset_index(name="count")
    group_totals = merged.groupby(zombie_q).size().reset_index(name="total")
    count_df = count_df.merge(group_totals, on=zombie_q)
    count_df["Proportion"] = (count_df["count"] / count_df["total"]) * 100
    plotter.plot_categorical_bars_hued(categories_prop_df=count_df,
                                       x_col="Group", x_label=f"ICS Group",
                                       category_col=zombie_q, data_col="Proportion",
                                       categories_colors={"No": "#B53B03", "Yes": "#3C5968"},
                                       save_path=result_path, save_name=f"ics_group_per_zombiePill", fmt="svg",
                                       y_min=0, y_max=101, y_skip=10, delete_y=False,
                                       inch_w=15, inch_h=12, add_pcnt=True, order=None)






    """
    Do different groups assign significantly different average rating to each entity?
    """
    ms_attribution = analysis_dict["other_creatures_ms"].copy()
    ms_attribution_long = pd.melt(ms_attribution, id_vars=[process_survey.COL_ID], var_name="entity", value_name="ms_rating")
    ms_attribution_with_group = pd.merge(ms_attribution_long, df_relevant_tagged[[process_survey.COL_ID, "Group"]], on=process_survey.COL_ID)
    ms_attribution_with_group.to_csv(os.path.join(result_path, "ms_rating_ics_groups.csv"), index=False)
    c = 8888





    """
    Follow up: do the two Earth-in-danger clusters differ in what they think?
    """
    if df_earth_cluster is not None:  # we have an actual df
        clusters = sorted(df_earth_cluster["Cluster"].unique().tolist())  # list all the possible clusters
        for cluster in clusters:
            subs_cluster = df_earth_cluster[df_earth_cluster["Cluster"] == cluster]
            subs_cluster_list = subs_cluster.loc[:, process_survey.COL_ID].tolist()
            df_ics_cluster = df_ics[df_ics[process_survey.COL_ID].isin(subs_cluster_list)].reset_index(drop=True,
                                                                                                       inplace=False)
            # plot
            calculate_ics_proportions(df_ics=df_ics_cluster, save_path=result_path, suffix=f"_cluster{cluster}")

        """
        create a contingency table for a chi-squared test to check whether the clusters significantly differ in  
        their proportion of people who said "Yes"
        """
        questions = [c for c in df_ics.columns if c.startswith("Do you think a creature/system")]
        result_list = list()
        for q in questions:
            df_ics_relevant = df_ics.loc[:, [process_survey.COL_ID, q]]
            df_clusters = df_earth_cluster[[process_survey.COL_ID, "Cluster"]]
            df_ics_relevant_with_cluster = pd.merge(df_ics_relevant, df_clusters, how="inner",
                                                    on=process_survey.COL_ID).reset_index(drop=True, inplace=False)
            """
            create a contingency table for a chi-squared test to check whether the clusters significantly differ in  
            their proportion of people who said "Yes"
            """
            contingency_table = pd.crosstab(df_ics_relevant_with_cluster["Cluster"],
                                            df_ics_relevant_with_cluster[q])
            chisquare_result = helper_funcs.chi_squared_test(contingency_table=contingency_table)
            chisquare_result[f"Question"] = [q]
            chisquare_result[f"per"] = ["Earth-in-danger cluster"]
            result_list.append(chisquare_result)
        result_df = pd.concat(result_list)
        result_df.to_csv(os.path.join(result_path, f"chisqared_earthInDanger_clusters_per_Q.csv"), index=False)
    return


def calculate_kill_for_test(df, save_path, prefix="", suffix=""):
    # all the options for killing (scenarios)
    questions = [c for c in df.columns if c.startswith("A creature/system that")]
    ans_map = {"No (will not kill to pass the test)": 0, "Yes (will kill to pass the test)": 1}

    # plot a collapsed figure where each creature is a bar, with the proportion of how many would kill it
    stats = dict()
    labels = list()
    for q in questions:
        df_q = df.loc[:, [process_survey.COL_ID, q]]
        q_name = survey_mapping.important_test_kill_tokens[q]
        df_q_map = df_q.replace({q: ans_map})
        stats[q_name] = helper_funcs.compute_stats(df_q_map[q], possible_values=df_q_map[q].unique().tolist())
        labels.append(q_name)

    """
    Plot OVERALL 'Yes' / 'No'
    """

    # Create DataFrame for plotting
    plot_data = {}
    for item, (proportions, mean_rating, std_dev, n) in stats.items():
        plot_data[item] = {
            "Proportion": proportions,
            "Mean": mean_rating,
            "Std Dev": std_dev,
            "N": n
        }
    rating_labels = ["No (will not kill to pass the test)", "Yes (will kill to pass the test)"]
    rating_color_list = ["#B53B03", "#3C5968"]
    sorted_plot_data = sorted(plot_data.items(), key=lambda x: x[1]["Mean"], reverse=True)
    plotter.plot_stacked_proportion_bars(plot_data=sorted_plot_data, num_plots=6, legend_labels=rating_labels,
                                         ytick_visible=True, text_width=39, title=f"", show_mean=False, sem_line=False,
                                         colors=rating_color_list, num_ratings=2, annotate_bar=True,
                                         save_path=save_path, save_name=f"{prefix}kill_to_pass{suffix}", fmt="svg")
    # save data
    plot_df = pd.DataFrame(plot_data)
    plot_df.to_csv(os.path.join(save_path, f"{prefix}kill_to_pass_stats{suffix}.csv"), index=True)

    """
    This might be misleading, as some people were not affected at all by the entity, and either would kill none of the 
    creatures, or actually - would kill all of them regardless of the entity. So we'd like to plot those in a reduced
    opacity. 
    """

    # Identify the people who answered 'Yes' or 'No' to all questions
    df_yes_all = df[questions].apply(lambda row: all(row == "Yes (will kill to pass the test)"), axis=1)
    df_no_all = df[questions].apply(lambda row: all(row == "No (will not kill to pass the test)"), axis=1)
    yes_all_proportion = df_yes_all.sum() / len(df)
    no_all_proportion = df_no_all.sum() / len(df)
    # plot again, discounting the all-yes, and all-no
    plotter.plot_stacked_proportion_bars(plot_data=sorted_plot_data, num_plots=6, legend_labels=rating_labels,
                                         ytick_visible=True, text_width=39, title=f"", show_mean=False, sem_line=False,
                                         colors=rating_color_list, num_ratings=2, save_path=save_path, fmt="svg",
                                         save_name=f"{prefix}kill_to_pass_allYesNoDiscount{suffix}", split=True,
                                         yes_all_proportion=yes_all_proportion, no_all_proportion=no_all_proportion,
                                         annotate_bar=True)
    return


def kill_for_test(analysis_dict, save_path, df_earth_cluster=None):
    """
    Answers to the "Do you think a creature/system can have intentions/consciousness/sensations w/o having..?" section
    """
    # save path
    result_path = os.path.join(save_path, "kill_for_test")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # load relevant data
    df_test_orig = analysis_dict["important_test_kill"].copy()
    df_test_orig.to_csv(os.path.join(result_path, f"kill_to_pass.csv"), index=False)

    # calculate the proportions of yes and no for each scenario and plot it
    calculate_kill_for_test(df=df_test_orig, save_path=result_path)

    """
    Does it matter which two features it is? Or rather the more features the merrier?
    Test if people keep alive creatures/systems with 2 features rather than 1 overall.
    """
    df_test = df_test_orig.rename(columns=survey_mapping.important_test_kill_tokens, inplace=False)  # shorter names
    # columns
    one_feature = [survey_mapping.Q_SENSATIONS, survey_mapping.Q_INTENTIONS, survey_mapping.Q_CONSCIOUSNESS]
    two_features = [survey_mapping.Q_CONSCIOUSNESS_SENSATIONS,
                    survey_mapping.Q_SENSATIONS_INTENTIONS, survey_mapping.Q_VULCAN]
    all_features = one_feature + two_features
    df_test_binary = df_test.replace(survey_mapping.ANS_KILLING_MAP, inplace=False)  # convert columns

    # calculate the average 'yes' responses for each person for 1-feature and 2-feature creatures
    df_test_binary["kill_one_avg"] = df_test_binary[one_feature].mean(axis=1)
    df_test_binary["kill_two_avg"] = df_test_binary[two_features].mean(axis=1)
    df_test_binary.to_csv(os.path.join(result_path, f"kill_to_pass_coded.csv"), index=False)

    # paired t-test
    paired_ttest = helper_funcs.dependent_samples_ttest(list_group1=df_test_binary["kill_one_avg"].tolist(),
                                                        list_group2=df_test_binary["kill_two_avg"].tolist())
    paired_ttest.to_csv(os.path.join(result_path, f"kill_oneVtwofeatures_ttest.csv"), index=False)

    # plot it
    df_test_binary["kill_one_avg"] = 100 * df_test_binary["kill_one_avg"]
    df_test_binary["kill_two_avg"] = 100 * df_test_binary["kill_two_avg"]
    plotter.plot_raincloud(df=df_test_binary, id_col="response_id", data_col_names=["kill_one_avg", "kill_two_avg"],
                           data_col_colors={"kill_one_avg": "#457b9d", "kill_two_avg": "#1d3557"},
                           save_path=result_path, save_name=f"kill_oneVtwofeatures", fmt="svg",
                           x_title="", x_name_dict={"kill_one_avg": "One Feature", "kill_two_avg": "Two Features"},
                           title="", y_title="Amount of killed entities", ymin=0, ymax=100, yskip=33.33,
                           y_ticks=["No entities", "One entity", "Two entities", "All entities"], y_jitter=10,
                           data_col_violin_left=None, violin_alpha=0.65, violin_width=0.5, group_spacing=0.5,
                           marker_spread=0.1, marker_size=100, marker_alpha=0.25, scatter_lines=True,
                           size_inches_x=15, size_inches_y=12)

    """
    Now, let's focus on the people - who would kill them all? Who wouldn't kill any?
    """
    all_features = one_feature + two_features

    # all "Yes"
    all_yes = df_test[df_test[all_features].eq(survey_mapping.ANS_KILL).all(axis=1)]
    all_yes_prop = 100 * all_yes.shape[0] / df_test.shape[0]

    # all "No"
    all_nos = df_test[df_test[all_features].eq(survey_mapping.ANS_NOKILL).all(axis=1)]
    all_nos_prop = 100 * all_nos.shape[0] / df_test.shape[0]

    # the rest - would kill at least one, but not all
    rest_prop = 100 * (df_test.shape[0] - all_yes.shape[0] - all_nos.shape[0]) / df_test.shape[0]

    kill_breakdown = pd.DataFrame({"kill_all_N": [all_yes.shape[0]], "kill_all_prop": [all_yes_prop],
                                   "kill_none_N": [all_nos.shape[0]], "kill_none_prop": [all_nos_prop],
                                   "rest_N": [(df_test.shape[0] - all_yes.shape[0] - all_nos.shape[0])],
                                   "rest_prop": [rest_prop]})
    kill_breakdown = kill_breakdown.transpose()
    kill_breakdown.to_csv(os.path.join(result_path, f"all_yes_no.csv"), index=True)

    cat_props = {"Won't kill any": all_nos_prop, "Kill at least one": rest_prop, "Kill all entities": all_yes_prop}
    cat_colors = ["#033860", "#C2948A", "#723D46"]
    category_props = pd.DataFrame(list(cat_props.items()), columns=["killing", "proportion"])
    plotter.plot_categorical_bars(categories_prop_df=category_props,
                                  category_col="killing", y_min=0, y_max=105, y_skip=10,
                                  data_col="proportion", delete_y=False, add_pcnt=True,
                                  categories_colors=cat_colors,
                                  save_path=result_path, save_name=f"all_yes_no", fmt="svg")

    # flatten the selections
    all_selections = all_nos["You wouldn't eliminate any of the creatures; why?"].str.split(',').explode()
    category_props = all_selections.value_counts(normalize=True)
    category_props = category_props.reset_index(drop=False, inplace=False)
    category_props["proportion"] = 100 * category_props["proportion"]
    color_list = ["#006d77", "#83c5be", "#ffddd2", "#e29578"]
    plotter.plot_categorical_bars(categories_prop_df=category_props,
                                  category_col="You wouldn't eliminate any of the creatures; why?",
                                  y_min=0, y_max=105, y_skip=10, inch_w=20, inch_h=12,
                                  data_col="proportion", delete_y=False, add_pcnt=True,
                                  categories_colors=color_list,
                                  save_path=result_path, save_name=f"all_nos_why", fmt="svg")

    # transform the data for modelling
    relevant_cols = [process_survey.COL_ID] + all_features
    df_test_binary_clean = df_test_binary[relevant_cols]
    transformed_data = []
    for index, row in df_test_binary_clean.iterrows():
        response_id = row[process_survey.COL_ID]
        for entity_index, entity_column in enumerate(all_features, start=1):
            entity_id = f"{entity_column}"
            consciousness = survey_mapping.Q_ENTITY_MAP[entity_column]["Consciousness"]
            intentions = survey_mapping.Q_ENTITY_MAP[entity_column]["Intentions"]
            sensations = survey_mapping.Q_ENTITY_MAP[entity_column]["Sensations"]
            kill = 1 if row[entity_column] == 1 else 0
            transformed_data.append([response_id, entity_id, consciousness, intentions, sensations, kill])


    """
    Is killing related to people's attitude towards the relationship between consciousness, sensations and intentions?
    """
    df_ics = analysis_dict["ics"].copy()
    df_ics_relevant = df_ics.loc[:, [process_survey.COL_ID] + list(survey_mapping.ICS_Q_NAME_MAP.keys())]
    df_ics_relevant[list(survey_mapping.ICS_Q_NAME_MAP.keys())] = df_ics_relevant[list(survey_mapping.ICS_Q_NAME_MAP.keys())].replace(survey_mapping.ANS_YESNO_MAP)
    df_ics_relevant.rename(columns=survey_mapping.ICS_Q_NAME_MAP, inplace=True)
    df_ics_relevant["Group"] = df_ics_relevant.apply(lambda row: ics_group_map(row), axis=1)

    df_test_relevant = df_test_orig.loc[:, [process_survey.COL_ID] + list(survey_mapping.important_test_kill_tokens.keys())]
    df_test_relevant.rename(columns=survey_mapping.important_test_kill_tokens, inplace=True)
    df_merged = pd.merge(df_ics_relevant.loc[:, [process_survey.COL_ID, "Group"]], df_test_relevant, on=process_survey.COL_ID)
    kill_columns = list(survey_mapping.important_test_kill_tokens.values())
    df_long = df_merged.melt(id_vars=[process_survey.COL_ID, "Group"],
                             value_vars=kill_columns,
                             var_name="Creature",
                             value_name="Kill_Response")
    df_long["Kill_binary"] = df_long["Kill_Response"].str.startswith("Yes").astype(int)
    df_long.to_csv(os.path.join(result_path, f"kill_per_creature_icsGroup.csv"), index=False)


    """
    Do the killing features explain the clusters?
    - cluster is assigned per person
    - C/S/I are per entity
    so we will ask whether a person’s pattern of decisions across different attribute combinations 
    (e.g., their responses to entities with or without certain attributes) predicts their Cluster.
    """
    # Create the updated DataFrame
    transformed_df = pd.DataFrame(transformed_data, columns=[process_survey.COL_ID, "entity",
                                                             "Consciousness", "Intentions", "Sensations", "kill"])
    if df_earth_cluster is not None:
        df_clusters = df_earth_cluster[[process_survey.COL_ID, "Cluster"]]
        transformed_df = transformed_df.merge(df_clusters, on=process_survey.COL_ID)
    transformed_df.to_csv(os.path.join(result_path, "kill_to_pass_coded_per_entity.csv"), index=False)
    df_earth_cluster_relevant = df_earth_cluster[[process_survey.COL_ID, "Cluster"]]
    merged = pd.merge(df_test_binary_clean, df_earth_cluster_relevant, on=process_survey.COL_ID)
    merged.to_csv(os.path.join(result_path, "kill_to_pass_cluster.csv"), index=False)


    """
    If df_earth_cluster is not None, take the clustering from the Earth-in-danger scenarios, and see if they apply 
    here as well. 
    """
    if df_earth_cluster is not None:
        questions = [c for c in df_test_orig.columns if c.startswith("A creature/system that")]
        df_clusters = df_earth_cluster[[process_survey.COL_ID, "Cluster"]]
        clusters = df_clusters["Cluster"].unique().tolist()
        df_with_cluster = pd.merge(df_test_orig, df_clusters, how="inner", on=process_survey.COL_ID).reset_index(
            drop=True, inplace=False)
        # descriptives and figure
        for cluster in clusters:
            df_cluster = df_with_cluster[df_with_cluster["Cluster"] == cluster]
            calculate_kill_for_test(df=df_cluster, save_path=result_path, prefix=f"earth_cluster{cluster}_", suffix="")
        # statistical analysis
        stats_list = list()
        for q in questions:
            df_kill_relevant = df_test_orig.loc[:, [process_survey.COL_ID, q]]
            df_kill_relevant_with_cluster = pd.merge(df_kill_relevant, df_clusters, how="inner",
                                                     on=process_survey.COL_ID).reset_index(drop=True, inplace=False)
            """
            create a contingency table for a chi-squared test to check whether the clusters significantly differ in  
            their proportion of people who said "Yes"
            """
            contingency_table = pd.crosstab(df_kill_relevant_with_cluster["Cluster"], df_kill_relevant_with_cluster[q])
            chisquare_result = helper_funcs.chi_squared_test(contingency_table=contingency_table)
            chisquare_result[f"Question"] = [q]
            chisquare_result[f"per"] = ["Earth-in-danger cluster"]
            stats_list.append(chisquare_result)
        result_df = pd.concat(stats_list)
        result_df.to_csv(os.path.join(result_path, f"chisqared_earthInDanger_clusters_per_Q.csv"), index=False)


    """
    Can killings be predicted by demographics?
    Random Forest classifier
    """



    demographics_df = analysis_dict["demographics"].copy()
    age_bins = [18, 25, 35, 45, 55, 65, 75, np.inf]
    age_labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "66-75", "76+"]
    demographics_df["age_group"] = pd.cut(demographics_df[survey_mapping.Q_AGE], bins=age_bins, labels=age_labels, right=True, include_lowest=True)

    categorical_cols = ["How do you describe yourself?",  # gender
                        "In which country do you currently reside?",  # country
                        "In what topic?",  # education topic: the education itself is ORDINAL
                        "education_Other: please specify",
                        "Current primary employment domain",
                        "employment_Other: please specify"]

    order_cols = ["age_group",
                  "What is your education background?_coded"]

    result = helper_funcs.run_random_forest_pipeline(dataframe=c_graded_with_personal_info,
                                                     dep_col=f"binary_{question}",
                                                     categorical_cols=categorical_cols,
                                                     save_path=result_path, save_prefix="demographics",
                                                     order_cols=experience_cols, rare_class_threshold=5, cv_folds=10,
                                                     scoring_method="accuracy")

    killing_with_personal = pd.merge(df_test_orig, demographics_df, on=process_survey.COL_ID)

    for col in survey_mapping.ICS_Q_NAME_MAP:
        c = 3

    result = helper_funcs.run_random_forest_pipeline(dataframe=killing_with_personal,
                                                     dep_col="con_intel_related",
                                                     categorical_cols=categorical_cols,
                                                     save_path=result_path, save_prefix="demographics",
                                                     order_cols=experience_cols, rare_class_threshold=5, cv_folds=10,
                                                     scoring_method="accuracy")

    con_demo_relevant = con_demo.drop(columns=["date", "date_start", "date_end", "duration_sec", "language"],
                                      inplace=False)
    df_cluster_relevant = df_earth_cluster.loc[:, [process_survey.COL_ID, "Cluster"]]
    cluster_with_demographics = pd.merge(con_demo_relevant, df_cluster_relevant, on=process_survey.COL_ID)
    cluster_with_demographics.to_csv(os.path.join(result_path, "cluster_with_demographics.csv"), index=False)



    order_cols = ["What is your education background?_coded"]

    """
    Can killings be predicted by experience?
    Random Forest classifier
    """
    ai_exp = analysis_dict["ai_exp"][[process_survey.COL_ID, survey_mapping.Q_AI_EXP]]
    animal_exp = analysis_dict["animal_exp"][[process_survey.COL_ID, survey_mapping.Q_ANIMAL_EXP]]
    ethics_exp = analysis_dict["ethics_exp"][[process_survey.COL_ID, survey_mapping.Q_ETHICS_EXP]]
    con_exp = analysis_dict["consciousness_exp"][[process_survey.COL_ID, survey_mapping.Q_CONSC_EXP]]

    experience_df = reduce(lambda left, right: pd.merge(left, right, on=[process_survey.COL_ID], how='outer'),
                           [ai_exp, animal_exp, ethics_exp, con_exp])

    con_exp_relevant = merged_experience_df.loc[:, [process_survey.COL_ID,
                                                    survey_mapping.Q_CONSC_EXP,
                                                    survey_mapping.Q_AI_EXP,
                                                    survey_mapping.Q_ANIMAL_EXP,
                                                    survey_mapping.Q_ETHICS_EXP]]
    df_cluster_relevant = df_earth_cluster.loc[:, [process_survey.COL_ID, "Cluster"]]
    cluster_with_exp = pd.merge(con_exp_relevant, df_cluster_relevant, on=process_survey.COL_ID)
    cluster_with_exp.to_csv(os.path.join(result_path, "cluster_with_experience.csv"), index=False)

    cluster_with_exp.rename(columns=relevant_cols, inplace=True)

    categorical_cols = []
    order_cols = [relevant_cols[survey_mapping.Q_CONSC_EXP], relevant_cols[survey_mapping.Q_AI_EXP],
                  relevant_cols[survey_mapping.Q_ANIMAL_EXP], relevant_cols[survey_mapping.Q_ETHICS_EXP]]

    helper_funcs.run_random_forest_pipeline(dataframe=cluster_with_exp, dep_col="Cluster",
                                            categorical_cols=categorical_cols,
                                            order_cols=order_cols, save_path=result_path, save_prefix="",
                                            rare_class_threshold=5, n_permutations=1000,
                                            scoring_method="accuracy",
                                            cv_folds=10)

    rony = 5



    return


def zombie_pill(analysis_dict, save_path, feature_order_df=None, feature_color_map=None):
    """
    Answers to the question about whether they would take a zombification pill.
    """
    # save path
    result_path = os.path.join(save_path, "zombie_pill")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # load relevant data
    df_zombie = analysis_dict["zombification_pill"].copy()
    zombie_q = "Would you take the pill?"

    category_counts = df_zombie["Would you take the pill?"].value_counts()
    plotter.plot_pie(categories_names=category_counts.index.tolist(), categories_counts=category_counts.tolist(),
                     categories_colors=CAT_COLOR_DICT, title=zombie_q,
                     save_path=result_path, save_name="take_the_pill_pie", fmt="svg")


    ans_map = {"No": 0, "Yes": 1}
    rating_labels = [survey_mapping.ANS_NO, survey_mapping.ANS_YES]
    df_q_map = df_zombie.replace({zombie_q: ans_map})
    stats = helper_funcs.compute_stats(df_q_map[zombie_q],
                                       possible_values=df_q_map[zombie_q].unique().tolist())
    # Create DataFrame for plotting
    plot_data = {zombie_q: {
        "Proportion": stats[0],
        "Mean": stats[1],
        "Std Dev": stats[2],
        "N": stats[3]
    }}
    zombie_data = pd.DataFrame(plot_data)
    zombie_data.to_csv(os.path.join(result_path, f"take_the_pill.csv"), index=True)  # index is descriptives' names


    """
    Experience with consciousness
    
    Consciousness experts present themselves as experts vs. 'laypeople'; 
    see whether Consciousness experts (self-rated experience with consciousness 4 and up) differ from the rest
    in the proportions of taking the pill
    """

    consciousness_exp = analysis_dict["consciousness_exp"][[process_survey.COL_ID, survey_mapping.Q_CONSC_EXP]]
    zombie_w_exp = pd.merge(df_zombie, consciousness_exp, on=process_survey.COL_ID)

    # "1" (expert) if survey_mapping.Q_CONSC_EXP > 3, "0" (non expert) if survey_mapping.Q_CONSC_EXP <=3
    expertise = [4, 5]
    zombie_w_exp["is_cons_expert"] = zombie_w_exp[survey_mapping.Q_CONSC_EXP].isin(expertise).astype(int)
    contingency_table = pd.crosstab(zombie_w_exp[zombie_q],
                                    zombie_w_exp["is_cons_expert"])
    contingency_table.to_csv(os.path.join(result_path, f"pill_conscExperts_vs_nonExperts_counts.csv"), index=True)
    result = helper_funcs.chi_squared_test(contingency_table=contingency_table)
    result["question"] = zombie_q
    result.to_csv(os.path.join(result_path, f"pill_conscExperts_vs_nonExperts_chisquared.csv"), index=False)

    """
    Plot proportion of Yes-No in the pill question per consciousness expertise level
    """

    zombie_w_exp[f"numeric_{zombie_q}"] = zombie_w_exp[zombie_q].map({'Yes': 1, 'No': 0})
    count_df = zombie_w_exp.groupby([survey_mapping.Q_CONSC_EXP, f"numeric_{zombie_q}"]).size().reset_index(name='count')
    count_df["total"] = count_df.groupby(survey_mapping.Q_CONSC_EXP)["count"].transform('sum')
    count_df["proportion"] = 100 * count_df["count"] / count_df["total"]
    count_df[zombie_q] = count_df[f"numeric_{zombie_q}"].map({1: 'Yes', 0: 'No'})
    zombie_w_exp.to_csv(os.path.join(result_path, "zombie_per_cons_exp.csv"), index=False)

    # for plotting
    pivot_df = count_df.pivot(index=survey_mapping.Q_CONSC_EXP, columns=zombie_q, values="proportion").fillna(0).reset_index(drop=False, inplace=False)
    pivot_df.to_csv(os.path.join(result_path, "zombie_per_cons_exp_props.csv"), index=False)
    plotter.plot_expertise_proportion_bars(df=pivot_df, cols=["No", "Yes"],
                                           cols_colors={"No": '#B33A00', "Yes": '#3B4E58'} ,
                                           x_axis_exp_col_name=survey_mapping.Q_CONSC_EXP,
                                           x_label="Reported experience with consciousness",
                                           y_ticks=[0, 25, 50, 75, 100],
                                           save_name="zombie_per_cons_exp_props",
                                           save_path=result_path, plt_title=zombie_q, annotate_bar=True)


    c = 3  # stopped here
    #############################################################################


    """
    """

    if feature_order_df is not None:
        """
        Cross between the zombie-pill and the features people value most for moral considerations. 
        Do people who agree to be zombies have something in common in terms of what they value for MS? (as from their 
        reply to the zombie question they do not value consciousness)
        """
        ms_features = analysis_dict["moral_considerations_features"].copy()
        c_graded = analysis_dict["consciousness_graded"].copy()
        ms_prios = analysis_dict["moral_considerations_prios"].copy()
        demographics = analysis_dict["demographics"].copy()
        # merge the dfs
        combined = reduce(lambda left, right: pd.merge(left, right, on=[process_survey.COL_ID]),
                          [df_zombie, c_graded, ms_features, ms_prios, demographics])
        # take only the ones who agreed to take the pill
        # for each type of zombie answer, calculate the ms_features thingy
        for answer in rating_labels:
            combined_zombie_ans = combined[combined["Would you take the pill?"] == answer]
            # calculate ms feature thingy
            ms_features_order_df, feature_colors = calculate_moral_consideration_features(
                ms_features_df=combined_zombie_ans,
                result_path=result_path,
                save_prefix=f"ms_features_zombie{answer}_",
                feature_order_df=feature_order_df,
                feature_color_dict=feature_color_map)
        combined.to_csv(os.path.join(result_path, f"zombie_with_extra_info.csv"), index=False)
    return


def calculate_moral_consideration_features(ms_features_df, result_path, save_prefix="",
                                           feature_order_df=None, feature_list=survey_mapping.ALL_FEATURES,
                                           feature_color_dict=None):
    # for dummy creation
    ms_features_copy = ms_features_df.loc[:, [process_survey.COL_ID, "What do you think is important for moral considerations?"]]

    def create_feature_dummies(row):
        return {feature: 1 if feature in row else 0 for feature in feature_list}

    dummies_df = ms_features_copy["What do you think is important for moral considerations?"].apply(
        create_feature_dummies).apply(pd.Series)

    """
    this is the proportion of selecting each category NORMALIZED by the TOTAL NUMBER of RESPONSES (i.e., the number
    of options all participants marked). A X% in this means that X% of all responses were [this] feature. 
    [this basically treats subjects as making single-selections, splitting each multi-selection subject into the 
    corresponding number of "single" dummy subjects, so that we can compare it to the "most important feature" below]
    """
    # proportion several = how many of all people selected a given feature
    proportions_several = (dummies_df.mean() * 100).to_frame(name="Proportion_all").reset_index(drop=False,
                                                                                                inplace=False)
    proportions_several = proportions_several.sort_values("Proportion_all", ascending=False).reset_index(drop=True,
                                                                                                         inplace=False)
    category_order = proportions_several["index"].tolist()

    # if participants selected only one to begin with, we didn't ask them to select which they think is the most important
    # see it here:
    # filtered_data = ms_features_df[ms_features_df["Which do you think is the most important for moral considerations?"].isna()]
    ms_features_df.loc[:, "Which do you think is the most important for moral considerations?"] = ms_features_df[
        "Which do you think is the most important for moral considerations?"].fillna(
        ms_features_df["What do you think is important for moral considerations?"])
    # now after we have the most important, plot it
    most_important = ms_features_df.loc[:, [process_survey.COL_ID, "Which do you think is the most important for moral considerations?"]]
    """
    what the below means is counting the proportions of selecting a single feature. Note that these are amts, 
    and we do not treat within-subject things here. 
    """
    # proportions_one  = how many of all people selected this feature as the most important one
    proportions_one = (most_important["Which do you think is the most important for moral considerations?"].value_counts(
                normalize=True) * 100).to_frame(name="Proportion_one").reset_index(drop=False, inplace=False)
    proportions_one.rename(columns={"Which do you think is the most important for moral considerations?": "index"},
                           inplace=True)
    proportions_one["index"] = pd.Categorical(proportions_one["index"], categories=category_order,
                                              ordered=True)  # match order
    proportions_one = proportions_one.sort_values("index").reset_index(drop=True, inplace=False)

    """
    diff = out of all the people who selected feature X as *one* of the important features,
    how many didn't select it as *the most* important one = the bigger it is, the more people
    who selected it did not select it as THE most important
    """
    df_diff = pd.DataFrame()
    df_diff["index"] = category_order
    df_diff["Proportion_diff"] = proportions_several["Proportion_all"] - proportions_one["Proportion_one"]

    df_unified = reduce(lambda left, right: pd.merge(left, right, on=["index"], how="outer"),
                        [proportions_several, proportions_one, df_diff])
    df_unified.sort_values(by=["Proportion_all"], ascending=False, inplace=True)  # sort by overall proportions
    df_unified.reset_index(drop=True, inplace=True)
    all_people = ms_features_copy.shape[0]  # total number of people this was calculated on
    df_unified["N"] = all_people
    df_unified.to_csv(os.path.join(result_path, f"{save_prefix}important_features.csv"), index=False)

    # plot
    if feature_color_dict is None:  # else, it already IS a dict with categories and colors
        # some nice default colors
        colors = ["#FFBF00", "#F47F38", "#E83F6F", "#855A8A", "#546798", "#2274A5", "#2A848A", "#32936F", "#99C9B7"]
        feature_color_dict = {df_unified.loc[i, "index"]: colors[i] for i in range(df_unified.shape[0])}

    if feature_order_df is None:
        plotter.plot_categorical_bars_layered(categories_prop_df=df_unified, category_col="index",
                                              full_data_col="Proportion_all", partial_data_col="Proportion_one",
                                              categories_colors=feature_color_dict, save_path=result_path,
                                              save_name=f"{save_prefix}important_features", fmt="svg", y_min=0,
                                              y_max=101, y_skip=10, inch_w=22, inch_h=12, order=None,
                                              annotate_bar=True, annot_font_color="#2C333A")
    else:
        plotter.plot_categorical_bars_layered(categories_prop_df=df_unified, category_col="index",
                                              full_data_col="Proportion_all", partial_data_col="Proportion_one",
                                              categories_colors=feature_color_dict, save_path=result_path,
                                              save_name=f"{save_prefix}important_features", fmt="svg", y_min=0,
                                              y_max=101, y_skip=10, inch_w=22, inch_h=12,
                                              order=feature_order_df["index"].tolist(),
                                              annotate_bar=True, annot_font_color="#2C333A")
    # we're getting back the final mapping between features and colors in case we want to re-use it for comparison
    return df_unified, feature_color_dict, most_important


def moral_consideration_features(analysis_dict, save_path, df_earth_cluster=None):
    """
    Answers to the question about which features they think are important for moral considerations >
    This method is only for selecting the population; the actual plotting etc is done by calculate_moral_consideration_features
    """
    # save path
    result_path = os.path.join(save_path, "moral_consideration_features")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    ms_features = analysis_dict["moral_considerations_features"].copy()
    ms_features.to_csv(os.path.join(result_path, "moral_considerations_features.csv"), index=False)

    ms_features_order_df, feature_colors, most_important_df = calculate_moral_consideration_features(ms_features_df=ms_features,
                                                                                  result_path=result_path,
                                                                                  save_prefix="all_",
                                                                                  feature_order_df=None,
                                                                                  feature_color_dict=None)

    most_important_q = "Which do you think is the most important for moral considerations?"

    """
    Break it down by experience: consciousness experts (4/5) vs non-experts (1/2/3)
    """
    c_expertise = analysis_dict["consciousness_exp"].copy()
    c_expertise[f"c_expert"] = c_expertise[survey_mapping.Q_CONSC_EXP].apply(lambda x: 1 if x >= 4 else 0)

    """
    Do consciousness experts differ from non-experts in selecting the Nagel-option as the most important feature
    for moral status? (something it is like to be: survey_mapping.ANS_PHENOMENOLOGY)
    """
    most_important_df["is_phenomenology"] = most_important_df[most_important_q] .apply(lambda x: 1 if x == survey_mapping.ANS_PHENOMENOLOGY else 0)
    c_exp_most_important = pd.merge(c_expertise.loc[:, [process_survey.COL_ID, survey_mapping.Q_CONSC_EXP, "c_expert"]],
                                    most_important_df, on=process_survey.COL_ID)
    """
    Chi squared test
    """
    # Create a combined contingency table
    contingency_table = pd.crosstab(c_exp_most_important["c_expert"], c_exp_most_important["is_phenomenology"])
    contingency_table.to_csv(os.path.join(result_path, f"c_exp_most_important_pheno_contingency.csv"), index=True)
    result_df = helper_funcs.chi_squared_test(contingency_table)
    result_df.to_csv(os.path.join(result_path, f"c_exp_most_important_pheno_chi_squared.csv"), index=False)



    """
    Consciousness experts vs non experts in choosing the most important feature
    """


    c_experts = c_expertise[c_expertise[f"c_expert"] == 1].loc[:, process_survey.COL_ID].tolist()
    ms_features_c_experts = ms_features[ms_features[process_survey.COL_ID].isin(c_experts)].reset_index(drop=True, inplace=False)
    ms_features_c_nonexperts = ms_features[~(ms_features[process_survey.COL_ID].isin(c_experts))].reset_index(drop=True, inplace=False)
    c_expertise_dict = {"cExperts_": ms_features_c_experts, "cNonExperts_": ms_features_c_nonexperts}
    c_expertise_results = {"cExperts_": None, "cNonExperts_": None}
    for k in c_expertise_dict:
        df = c_expertise_dict[k]
        ms_features_order_df, feature_colors, most_important = calculate_moral_consideration_features(ms_features_df=df,
                                                                                                      result_path=result_path,
                                                                                                      save_prefix=k,
                                                                                                      feature_order_df=ms_features_order_df,
                                                                                                      feature_color_dict=feature_colors)
        most_important.to_csv(os.path.join(result_path, f"{k}most_important.csv"), index=False)
        most_important["group"] = k
        c_expertise_results[k] = most_important

    """
    Compare - do consciousness experts differ from non-experts in choosing the most important feature?
    """
    q = "Which do you think is the most important for moral considerations?"
    choices_non_experts = c_expertise_results["cExperts_"].loc[:, ["group", q]]
    choices_experts = c_expertise_results["cNonExperts_"].loc[:, ["group", q]]
    choices_combined = pd.concat([choices_non_experts, choices_experts], ignore_index=True)

    # Create a combined contingency table
    contingency_table = pd.crosstab(choices_combined["group"], choices_combined[q])
    contingency_table.to_csv(os.path.join(result_path, f"cExpertise_most_important_contingency.csv"), index=True)
    result_df = helper_funcs.chi_squared_test(contingency_table)
    result_df.to_csv(os.path.join(result_path, f"cExpertise_most_important_chi_squared.csv"), index=False)


    """
    Relationship between Earth-in-danger clusters and moral consideration features
    """

    if df_earth_cluster is not None:  # we have an actual df
        ms_features_clusters_merged = reduce(lambda left, right: pd.merge(left, right, on=[process_survey.COL_ID], how='outer'), [most_important_df, df_earth_cluster.loc[:, [process_survey.COL_ID, "Cluster"]]])

        """
        Compare - do consciousness experts differ from non-experts in choosing the most important feature?
        """
        q = "Which do you think is the most important for moral considerations?"

        # Create a combined contingency table
        contingency_table = pd.crosstab(ms_features_clusters_merged["Cluster"], choices_combined[q])
        contingency_table.to_csv(os.path.join(result_path, f"clusters_most_important_contingency.csv"), index=True)
        result_df = helper_funcs.chi_squared_test(contingency_table)
        result_df.to_csv(os.path.join(result_path, f"clusters_most_important_chi_squared.csv"), index=False)

        stop_here = 5


        """
        Relationship with consciousness/sensations/intentions: 
        Group moral consideration features together to resemble i_c_s, and show what it does to the proportions
        """

        ms_features_q = "What do you think is important for moral considerations?"
        ms_important_q = "Which do you think is the most important for moral considerations?"

        features_groups = {  # 'c' group: self awareness, sensory abilities, something it is like to be
            survey_mapping.ANS_SELF: "Consciousness",
            survey_mapping.ANS_SENS: "Consciousness",
            survey_mapping.ANS_PHENOMENOLOGY: "Consciousness",
            # 'i' group: planning/goals, thinking, language
            survey_mapping.ANS_PLAN: "Intentions",
            survey_mapping.ANS_THINK: "Intentions",
            survey_mapping.ANS_LANG: "Intentions",
            # 's' group: valence
            survey_mapping.ANS_SENTIENCE: "Sensations",
            # other
            survey_mapping.ANS_OTHER: "Other"
        }

        def map_strings(cell):  # Replace each substring using the mapping, keeping it unchanged if not in the mapping
            result = []
            matched_substrings = set()

            for substring in survey_mapping.ALL_FEATURES:
                if substring in cell and substring not in matched_substrings:
                    result.append(features_groups[substring])
                    matched_substrings.add(substring)

            # remove duplicates (so we won't have 'intentions,intentions,intentions')
            unique_result = sorted(set(result))
            return ','.join(unique_result)

        # map into groups
        ms_features_grouped = ms_features.copy()
        ms_features_grouped[ms_features_q] = ms_features_grouped[ms_features_q].apply(map_strings)  # multi-select
        ms_features_grouped[ms_important_q] = ms_features_grouped[ms_important_q].replace(
            features_groups)  # single-select
        # if the 'most important' column is nan, it means they only selected 1 feature to begin with
        ms_features_grouped[ms_important_q] = ms_features_grouped[ms_important_q].replace('nan', pd.NA)
        ms_features_grouped[ms_important_q] = ms_features_grouped[ms_important_q].fillna(
            ms_features_grouped[ms_features_q])

        calculate_moral_consideration_features(ms_features_df=ms_features_grouped,
                                               result_path=result_path,
                                               save_prefix="all_GROUPED-i_c_s_",
                                               feature_list=["Consciousness", "Intentions", "Sensations", "Other"],
                                               feature_order_df=None,
                                               feature_color_dict=None)

    # return the order and colors so other questions can use them
    return ms_features_order_df, feature_colors


def expert_check(analysis_dict, save_path, df_earth_cluster=None):
    # save path
    result_path = os.path.join(save_path, "expert_check")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # get expert info
    ai_exp = analysis_dict["ai_exp"]
    ethics_exp = analysis_dict["ethics_exp"]
    consciousness_exp = analysis_dict["consciousness_exp"]
    animal_exp = analysis_dict["animal_exp"]

    exp_types = {"ai": ai_exp, "ethics": ethics_exp, "consc": consciousness_exp, "animal": animal_exp}

    """
    EiD clusters based on expertise
    """

    if df_earth_cluster is not None:
        df_earth_cluster = df_earth_cluster[[process_survey.COL_ID, "Cluster"]]
        for exp in exp_types.keys():
            exp_col = [c for c in exp_types[exp].columns.tolist() if c.startswith("On a scale")][0]
            df_exp_only = exp_types[exp][[process_survey.COL_ID, exp_col]]
            df_merged = pd.merge(df_earth_cluster, df_exp_only, on=process_survey.COL_ID)
            df_merged.to_csv(os.path.join(result_path, f"exp_{exp}_data.csv"), index=False)
            df_merged[exp_col] = pd.to_numeric(df_merged[exp_col])

            cluster0 = df_merged[df_merged["Cluster"] == 0].reset_index(drop=True,inplace=False)[exp_col]
            cluster1 = df_merged[df_merged["Cluster"] == 1].reset_index(drop=True,inplace=False)[exp_col]


            """
            Do a Mann-Whitney U test. Experience is ordinal (1-4), clusters are independent
            """
            result = helper_funcs.mann_whitney_utest(list_group1=cluster0.tolist(), list_group2=cluster1.tolist())
            result.to_csv(os.path.join(result_path, f"exp_{exp}_per_EiDCluster_MWU_test.csv"), index=False)

            summary_df = df_merged.groupby("Cluster")[exp_col].agg(["min", "mean", "std", "max"]).reset_index()
            summary_df.to_csv(os.path.join(result_path, f"exp_{exp}_per_EiDCluster_stats.csv"), index=False)


            # plot bars
            count_df = df_merged.groupby(["Cluster", exp_col]).size().reset_index(name="count")
            group_totals = df_merged.groupby("Cluster").size().reset_index(name="total")
            count_df = count_df.merge(group_totals, on="Cluster")
            count_df["Proportion"] = (count_df["count"] / count_df["total"]) * 100
            plotter.plot_categorical_bars_hued(categories_prop_df=count_df,
                                               x_col=exp_col, x_label=f"{exp.title()} Experience",
                                               category_col="Cluster", data_col="Proportion",
                                               categories_colors={0: "#EDAE49", 1: "#102E4A"},
                                               save_path=result_path, save_name=f"exp_{exp}_per_EiDCluster", fmt="svg",
                                               y_min=0, y_max=101, y_skip=10, delete_y=False,
                                               inch_w=15, inch_h=12, add_pcnt=True, order=None)






    """
    RONYRONY
    Francken et al 2022: experts in consciousness (philosophy/neuroscience?) and their ratings of CONSCIOUSNESS
    """

    francken_df = pd.read_csv(r"C:\Users\Rony\Documents\projects\ethics\survey_analysis\Francken et al 2022 ans.csv")
    entities_cols = [c for c in francken_df.columns.tolist() if "entities" in c]
    francken_df_entities = francken_df.loc[:, entities_cols]
    entity_name_conversion = {"entities_bat": "c_A bat",
                              "entities_tree": "c_A tree",
                              "entities_fish": "c_A salmon",
                              "entities_dog": "c_A dog",
                              "entities_octopus": "c_An octopus",
                              "entities_baby": "c_A newborn baby (human)",
                              "entities_yourself": "c_You"}
    francken_df_entities.rename(columns=entity_name_conversion, inplace=True)
    # take just the overlapping creatures
    francken_df_entities = francken_df_entities.loc[:, list(entity_name_conversion.values())]


    """
    Do expert attributions of consciousness align? 
    """
    c_attributions = analysis_dict["other_creatures_cons"]
    c_attributions_w_c_expertise = pd.merge(consciousness_exp.loc[:, [process_survey.COL_ID, survey_mapping.Q_CONSC_EXP]], c_attributions, on=process_survey.COL_ID)
    # filter out non-experts:
    c_attributions_experts = c_attributions_w_c_expertise[c_attributions_w_c_expertise[survey_mapping.Q_CONSC_EXP] >= 3].reset_index(drop=True, inplace=False)
    c_experts_relevant = c_attributions_experts[list(entity_name_conversion.values())]

    # convert ratings into ranks
    ranks_franken = francken_df_entities.rank(axis=1, method="min", ascending=False)
    ranks_us = c_experts_relevant.rank(axis=1, method="min", ascending=False)

    # normalize ranks to scale from 0 to 1 (for comparison)
    franken_scaled_ranks = (ranks_franken - 1) / (ranks_franken.max().max() - 1)
    us_scaled_ranks = (ranks_us - 1) / (ranks_us.max().max() - 1)

    # permutation test to compare rankings between the two groups
    def permutation_test(group1_ranks, group2_ranks, n_permutations=1000):
        observed_stat = np.abs(np.mean(group1_ranks - group2_ranks))

        # Combine the two group rankings and shuffle to create null distribution
        combined_ranks = np.concatenate([group1_ranks.values.flatten(), group2_ranks.values.flatten()])
        perm_stats = []
        for _ in range(n_permutations):
            np.random.shuffle(combined_ranks)
            perm_group1 = combined_ranks[:group1_ranks.size].reshape(group1_ranks.shape)
            perm_group2 = combined_ranks[group1_ranks.size:].reshape(group2_ranks.shape)
            perm_stat = np.abs(np.mean(perm_group1 - perm_group2))
            perm_stats.append(perm_stat)

        perm_stats = np.array(perm_stats)
        p_value = np.mean(perm_stats >= observed_stat)

        return observed_stat, p_value

    # Run the permutation test
    observed_stat, p_value = permutation_test(franken_scaled_ranks, us_scaled_ranks)
    print(f"Observed Statistic: {observed_stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Step 4: Visualizations

    # Visualization 1: Scatter plot of normalized ranks for each fruit
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.scatter(franken_scaled_ranks.values.flatten(), us_scaled_ranks.values.flatten(), label="Entities", color="blue")
    plt.plot([0, 1], [0, 1], 'k--', label="Perfect Agreement")
    plt.title("Scatter Plot of Normalized Rankings: Group 1 vs Group 2")
    plt.xlabel("Group 1 Normalized Ranks")
    plt.ylabel("Group 2 Normalized Ranks")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Visualization 2: Bump chart (Slopegraph) for each fruit
    # Prepare data for bump chart (sorting fruits by Group 1's ranks)
    rank_df = pd.DataFrame({
        'Fruit': group1_df.columns,
        'Group 1 Rank': group1_ranks.mean(axis=0).values,
        'Group 2 Rank': group2_ranks.mean(axis=0).values
    })

    # Sort by Group 1's rank
    rank_df = rank_df.sort_values(by='Group 1 Rank')

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=rank_df, x='Fruit', y='Group 1 Rank', label="Group 1 Rank", marker='o', color='b')
    sns.lineplot(data=rank_df, x='Fruit', y='Group 2 Rank', label="Group 2 Rank", marker='s', color='r')

    plt.xticks(rotation=45, ha='right')
    plt.title('Bump Chart: Rankings by Group 1 and Group 2')
    plt.xlabel('Fruits')
    plt.ylabel('Rank')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return


def age_check(analysis_dict, save_path, df_earth_cluster=None):
    # save path
    result_path = os.path.join(save_path, "age_check")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # get gender info
    demo_df = analysis_dict["demographics"]  # Q_AGE

    """
    Earth in danger: are the clusters age-based?
    """
    age_bins = [18, 25, 35, 45, 55, 65, 75, np.inf]
    age_labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "66-75", "76+"]

    if df_earth_cluster is not None:
        demo_merged = pd.merge(demo_df, df_earth_cluster, on=process_survey.COL_ID)
        demo_merged_filtered = demo_merged[[process_survey.COL_ID, survey_mapping.Q_AGE, "Cluster"]]
        # Create a categorical age group column
        demo_merged_filtered["age_group"] = pd.cut(demo_merged_filtered[survey_mapping.Q_AGE], bins=age_bins, labels=age_labels, right=True, include_lowest=True)
        # Check the distribution of age_group by actual counts
        value_counts = demo_merged_filtered["age_group"].value_counts()
        value_counts.to_csv(os.path.join(result_path, "Age_groups_valuecounts.csv"))

        """
        Chi square test: is the proportion of people in each age category different between Clusters? 
        A significant chi-square would mean the groups have different age composition.
        This approach is very direct for our question and easy to interpret: 
        you can literally see which age ranges contribute to differences. RONYRONY
        """
        # Create a contingency table of group vs age_group
        contingency_table = pd.crosstab(demo_merged_filtered["Cluster"], demo_merged_filtered["age_group"])
        chisquare_result = helper_funcs.chi_squared_test(contingency_table=contingency_table)
        chisquare_result.to_csv(os.path.join(result_path, "Age_per_EiDCluster_chisq.csv"), index=False)
        contingency_table.to_csv(os.path.join(result_path, "Age_per_EiDCluster.csv"), index=True)

        count_df = demo_merged_filtered.groupby(["Cluster", "age_group"]).size().reset_index(name="count")
        group_totals = demo_merged_filtered.groupby("Cluster").size().reset_index(name="total")
        count_df = count_df.merge(group_totals, on="Cluster")
        count_df["Proportion"] = (count_df["count"] / count_df["total"]) * 100

        # plot bars
        plotter.plot_categorical_bars_hued(categories_prop_df=count_df,
                                           x_col="age_group", x_label="Age Group",
                                           category_col="Cluster", data_col="Proportion",
                                           categories_colors={0: "#EDAE49", 1: "#102E4A"},
                                           save_path=result_path, save_name=f"Age_per_EiDCluster", fmt="svg",
                                           y_min=0, y_max=101, y_skip=10, delete_y=False,
                                           inch_w=15, inch_h=12, add_pcnt=True, order=None)



    return


def gender_check(analysis_dict, save_path, df_earth_cluster=None):
    # save path
    result_path = os.path.join(save_path, "gender_check")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # get gender info
    demo_df = analysis_dict["demographics"]
    demo_df_genders = demo_df["How do you describe yourself?"].unique().tolist()
    print(f"There are {len(demo_df_genders)} declared genders in this sample")
    count_df = demo_df.groupby("How do you describe yourself?").size().reset_index(name="count")
    count_df["pctg"] = 100 * count_df["count"] / sum(count_df["count"])

    """
    On the exploratory sample (N=350), 185 are female (52.86%), 159 are male (45.43%), and 6 are the other types. 
    Therefore, the N=6 are negilable, and for testing the relationship between gender and other things they are
    negligible. This might not be the case for the full sample. 
    """

    # filter to only male female, and add a column "is female"
    demo_df = demo_df[(demo_df["How do you describe yourself?"] == "Female") | (demo_df["How do you describe yourself?"] == "Male")]

    """
    Earth in danger: are the clusters gendered?
    """
    if df_earth_cluster is not None:
        demo_merged = pd.merge(demo_df, df_earth_cluster, on=process_survey.COL_ID)
        """
        create a contingency table for a chi-squared test to check whether the earth in danger clusters 
        significantly differ in gender
        """
        contingency_table = pd.crosstab(demo_merged["How do you describe yourself?"],
                                        demo_merged["Cluster"])
        chisquare_result = helper_funcs.chi_squared_test(contingency_table=contingency_table)
        chisquare_result.to_csv(os.path.join(result_path, "Gender_per_EiDCluster_chisq.csv"), index=False)
        contingency_table.to_csv(os.path.join(result_path, "Gender_per_EiDCluster.csv"),
                                 index=True)

        # plot bars RONYRONY
        count_df = demo_merged.groupby(["Cluster", "How do you describe yourself?"]).size().reset_index(name="count")
        group_totals = demo_merged.groupby("Cluster").size().reset_index(name="total")
        count_df = count_df.merge(group_totals, on="Cluster")
        count_df["Proportion"] = (count_df["count"] / count_df["total"]) * 100
        plotter.plot_categorical_bars_hued(categories_prop_df=count_df,
                                           x_col="How do you describe yourself?", x_label="Gender",
                                           category_col="Cluster", data_col="Proportion",
                                           categories_colors={0: "#EDAE49", 1: "#102E4A"},
                                           save_path=result_path, save_name=f"Gender_per_EiDCluster", fmt="svg",
                                           y_min=0, y_max=101, y_skip=10, delete_y=False,
                                           inch_w=15, inch_h=12, add_pcnt=True, order=None)



        # plot bars
        counts = contingency_table.copy()
        counts["total_n"] = contingency_table.sum(axis=1)
        counts["pcnt_total"] = [100 for i in range (counts.shape[0])]  # 100%
        counts["pcnt_cluster 0"] = 100 * counts[0] / counts["total_n"]
        counts.reset_index(drop=False, inplace=True)

        feature_color_dict = {"Male": "#39342D", "Female": "#22333B"}
        plotter.plot_categorical_bars_layered(categories_prop_df=counts, category_col="How do you describe yourself?",
                                              full_data_col="total_n", partial_data_col=0,
                                              categories_colors=feature_color_dict, save_path=result_path,
                                              save_name=f"Gender_per_EiDCluster", fmt="svg", y_min=0,
                                              y_max=round(counts["total_n"].max() + (counts["total_n"].max()/10), -1) + 1, y_skip=10, inch_w=20, inch_h=12, order=None)

    """
    Kill for test: are the killings gendered?
    """

    # load relevant data
    df_test_coded = pd.read_csv(os.path.join(save_path, "kill_for_test", f"kill_to_pass_coded.csv"))
    df_test_coded_relevant = df_test_coded.iloc[:, :7]
    df_test_coded_relevant["response_pattern"] = df_test_coded_relevant[[x for x in df_test_coded_relevant.columns.tolist() if x != process_survey.COL_ID]].\
        apply(lambda x: ''.join(x.astype(str)), axis=1)

    demo_df_relevant = demo_df.loc[:, [process_survey.COL_ID, "How do you describe yourself?"]]
    df_test_coded_merged = pd.merge(df_test_coded_relevant, demo_df_relevant, on=process_survey.COL_ID)

    response_columns = [survey_mapping.Q_SENSATIONS, survey_mapping.Q_INTENTIONS,
                        survey_mapping.Q_CONSCIOUSNESS, survey_mapping.Q_VULCAN,
                        survey_mapping.Q_CONSCIOUSNESS_SENSATIONS, survey_mapping.Q_SENSATIONS_INTENTIONS]

    # Group by 'How do you describe yourself?' and sum the response columns (how many '1's --> kills)
    df_summarized = df_test_coded_merged.groupby("How do you describe yourself?")[response_columns].sum().reset_index(drop=False, inplace=False)
    df_counts = df_test_coded_merged.groupby("How do you describe yourself?")[process_survey.COL_ID].count().reset_index(drop=False, inplace=False)
    df_merged = pd.merge(df_counts, df_summarized, on="How do you describe yourself?")
    df_merged_proportions = df_merged.copy()
    for col in response_columns:
        df_merged_proportions[col] = 100 * df_merged_proportions[col] / df_merged_proportions[process_survey.COL_ID]

    df_merged_proportions.set_index("How do you describe yourself?", inplace=True)
    plotting_df = df_merged_proportions[response_columns].T.reset_index(drop=False, inplace=False)

    plotting_df_melted = plotting_df.melt(id_vars=["index"], value_vars=["Female", "Male"], var_name="Gender", value_name="Proportion")

    plotter.plot_categorical_bars_hued(categories_prop_df=plotting_df_melted, x_col="index", category_col="",
                                       data_col="Proportion",
                                       categories_colors={"Female": "#22333B", "Male": "#39342D"},
                                       save_path=result_path, save_name=f"Gender_per_killForTest", fmt="svg",
                                       y_min=0, y_max=101, y_skip=10, delete_y=False, inch_w=15, inch_h=12,
                                       add_pcnt=True, order=None)
    plotting_df_melted.to_csv(os.path.join(result_path, f"Gender_per_killForTest"), index=False)

    from statsmodels.regression.mixed_linear_model import MixedLM
    import numpy as np
    df_test_coded_merged["coded_gender"] = df_test_coded_merged["How do you describe yourself?"].map({"Female": 0, "Male": 1})

    # Fit the mixed effects logistic regression model
    # 'category_num' is the fixed effect, and 'response_id' is treated as the random effect
    model = MixedLM.from_formula("coded_gender ~ response_pattern", groups=process_survey.COL_ID, data=df_test_coded_merged)

    # Fit the model
    result = model.fit()

    # Show the model summary
    print(result.summary())
    return


def religious_check(analysis_dict, save_path, df_earth_cluster=None):
    """
    Following Demertzi et al. 2009, let's check if religious experience affects stuff
    """

    # save path
    result_path = os.path.join(save_path, "religious_check")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)


    # get religious info
    ethics_df = analysis_dict["ethics_exp"]
    ethics_exp_df = ethics_df[ethics_df["Please specify your experience"].notnull()]
    print(f"There are N={ethics_exp_df.shape[0]} people with ethics experience >= 3 (out of N={ethics_df.shape[0]})")

    religious_str = "Religious/Spiritual Practice (engagement with ethical teachings)"
    religious_df = ethics_exp_df[
        ethics_exp_df["Please specify your experience"].str.contains(religious_str, regex=False)]
    print(f"There are N={religious_df.shape[0]} people who marked having experience with C due to religion")

    religious_ids = religious_df[process_survey.COL_ID].unique()
    ethics_df["religious"] = ethics_df[process_survey.COL_ID].apply(lambda x: 1 if x in religious_ids else 0)


    """
    MORAL CONSIDERATION PRIOS
    """

    q_mc_human_prio = "Do you think some people should have a higher moral status than others?"
    ms_prios = analysis_dict["moral_considerations_prios"].copy()
    religion_combined = pd.merge(ethics_df, ms_prios, on=process_survey.COL_ID)

    """
    create a contingency table for a chi-squared test to check whether the clusters significantly differ in  
    their proportion of people who said "Yes" to the question: 
    'Do you think some people should have a higher moral status than others?'
    Between people who are BOTH experienced in ethics AND it's due to religious experience, vs. rest (i.e., the rest
    of those who are experienced in ethics but not from religion AND the inexperienced ones)
    """
    contingency_table = pd.crosstab(religion_combined[q_mc_human_prio],
                                    religion_combined["religious"])
    chisquare_result = helper_funcs.chi_squared_test(contingency_table=contingency_table)
    chisquare_result.to_csv(os.path.join(result_path, "ReligiousYesNo_people higher ms.csv"), index=False)
    contingency_table.to_csv(os.path.join(result_path, "ReligiousYesNo_people higher ms_contingency_table.csv"),
                             index=True)


    """
    ZOMBIE PILL
    """

    q_pill = "Would you take the pill?"
    df_pill = analysis_dict["zombification_pill"].copy()
    religion_combined = pd.merge(ethics_df, df_pill, on=process_survey.COL_ID)
    contingency_table = pd.crosstab(religion_combined[q_pill], religion_combined["religious"])
    chisquare_result = helper_funcs.chi_squared_test(contingency_table=contingency_table)
    chisquare_result.to_csv(os.path.join(result_path, "ReligiousYesNo_zombie.csv"), index=False)
    contingency_table.to_csv(os.path.join(result_path, "ReligiousYesNo_zombie.csv"),
                             index=True)

    rel_dict = {"religious": religion_combined[religion_combined["religious"] == 1],
                "rest": religion_combined[religion_combined["religious"] == 0]}
    # add pie of only them
    for rel in rel_dict:
        df = rel_dict[rel]
        category_counts = df["Would you take the pill?"].value_counts()
        plotter.plot_pie(categories_names=category_counts.index.tolist(), categories_counts=category_counts.tolist(),
                         categories_colors=CAT_COLOR_DICT, title="Would you take the pill?",
                         save_path=result_path, save_name=f"take_the_pill_pie_{rel}", fmt="svg")

    return


def moral_considreation_prios(analysis_dict, save_path, df_earth_cluster=None):
    """
    Answers to the question about whether different creatures deserve moral considerations
    """
    # save path
    result_path = os.path.join(save_path, "moral_consideration_prios")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    ms_prios = analysis_dict["moral_considerations_prios"].copy()
    ms_prios.to_csv(os.path.join(result_path, "moral_decisions_prios.csv"), index=False)
    questions = [c for c in ms_prios.columns if c.startswith("Do you think")]
    for q in questions:
        df_q = ms_prios.loc[:, [process_survey.COL_ID, q]]
        category_counts = df_q[q].value_counts().reset_index(inplace=False, drop=False)
        plotter.plot_pie(categories_names=category_counts.loc[:, q].tolist(),
                         categories_counts=category_counts.loc[:, "count"].tolist(),
                         categories_colors=CAT_COLOR_DICT, title=f"{q}",
                         save_path=result_path, save_name=f"{q.replace('?', '').replace('/', '-')}", fmt="svg")
        category_counts.to_csv(os.path.join(result_path, f"{q.replace('?', '').replace('/', '-')}.csv"), index=False)

    """
    Follow up on reasons 
    """
    reasons = [c for c in ms_prios.columns if c not in questions]
    for r in reasons:
        df_r = ms_prios.loc[:, [process_survey.COL_ID, r]]
        df_r = df_r[df_r[r].notnull()]
        df_r.to_csv(os.path.join(result_path, f"{r.replace('?', '').replace('/', '-')}.csv"), index=False)



    """
    Do the answers change based on expertise?
    """

    # animal expertise
    df_prio_animal = ms_prios.loc[:, [process_survey.COL_ID, survey_mapping.PRIOS_Q_ANIMALS]]
    exp_animal = analysis_dict["animal_exp"].copy().loc[:, [process_survey.COL_ID, survey_mapping.Q_ANIMAL_EXP]]
    exp_animal.rename(columns={survey_mapping.Q_ANIMAL_EXP: "exp_animal"}, inplace=True)
    exp_animal["binary_exp_animal"] = exp_animal["exp_animal"].apply(lambda x: 1 if x >= 4 else 0)
    df_unified = pd.merge(df_prio_animal, exp_animal, on=process_survey.COL_ID)

    # chi squared
    contingency_table = pd.crosstab(df_unified[survey_mapping.PRIOS_Q_ANIMALS], df_unified["binary_exp_animal"])
    q_name = survey_mapping.PRIOS_Q_NAME_MAP[survey_mapping.PRIOS_Q_ANIMALS]
    chisquare_result = helper_funcs.chi_squared_test(contingency_table=contingency_table)
    contingency_table.to_csv(os.path.join(result_path, f"animal_expertise_per_{q_name}_contingency.csv"))
    chisquare_result.to_csv(os.path.join(result_path, f"animal_expertise_per_{q_name}_chisquare.csv"))
    proportion_table = contingency_table.div(contingency_table.sum(axis=0), axis=1)
    proportion_table.to_csv(os.path.join(result_path, f"animal_expertise_per_{q_name}_descriptives.csv"))

    # for plotting
    df_unified[f"numeric_{survey_mapping.PRIOS_Q_ANIMALS}"] = df_unified[survey_mapping.PRIOS_Q_ANIMALS].map({'Yes': 1, 'No': 0})
    count_df = df_unified.groupby(["exp_animal", f"numeric_{survey_mapping.PRIOS_Q_ANIMALS}"]).size().reset_index(name='count')
    count_df["total"] = count_df.groupby("exp_animal")["count"].transform('sum')
    count_df["proportion"] = 100 * count_df["count"] / count_df["total"]
    count_df[survey_mapping.PRIOS_Q_ANIMALS] = count_df[f"numeric_{survey_mapping.PRIOS_Q_ANIMALS}"].map({1: 'Yes', 0: 'No'})
    df_unified.to_csv(os.path.join(result_path, f"animal_expertise_per_{q_name}.csv"), index=False)

    pivot_df = count_df.pivot(index="exp_animal", columns=survey_mapping.PRIOS_Q_ANIMALS, values="proportion").fillna(0).reset_index(drop=False, inplace=False)
    pivot_df.to_csv(os.path.join(result_path, f"animal_expertise_per_{q_name}_props.csv"), index=False)
    plotter.plot_expertise_proportion_bars(df=pivot_df, cols=["No", "Yes"],
                                           cols_colors={"No": '#B33A00', "Yes": '#3B4E58'},
                                           x_axis_exp_col_name="exp_animal",
                                           x_label="Reported experience with animals",
                                           y_ticks=[0, 25, 50, 75, 100],
                                           save_name=f"animal_expertise_per_{q_name}_props",
                                           save_path=result_path, plt_title=survey_mapping.PRIOS_Q_ANIMALS, annotate_bar=True)
    rony = 3

    """
    Relations to agreement with the assertion that consciousness is a graded phenomenon 
    survey_mapping.Q_GRADED_UNEQUAL
    """
    mc_prios_people = "Do you think some people should have a higher moral status than others?"
    mc_prios_animals = "Do you think some non-human animals should have a higher moral status than others?"
    mc_prios_dict = {"people": mc_prios_people, "animals": mc_prios_animals}

    c_graded = analysis_dict["consciousness_graded"].copy()
    # binarize agreement with the assertion: disagree (0) --> 1, 2; agree (1) --> 3, 4
    agreed = [3, 4]
    c_graded[f"binary_{survey_mapping.Q_GRADED_UNEQUAL}"] = c_graded[survey_mapping.Q_GRADED_UNEQUAL].isin(
        agreed).astype(int)
    # merge graded consciousness answers with the priorities answers
    merged_df = pd.merge(c_graded, ms_prios, on=[process_survey.COL_ID])

    relevant_df = merged_df.loc[:, [process_survey.COL_ID, survey_mapping.Q_GRADED_UNEQUAL,
                                    f"binary_{survey_mapping.Q_GRADED_UNEQUAL}", mc_prios_people, mc_prios_animals]]
    relevant_df.to_csv(os.path.join(result_path, f"graded_con_mc_prios.csv"), index=True)

    for col_name in mc_prios_dict:
        col = mc_prios_dict[col_name]
        # do a contingency table with the BINARY versions of the answers to both questions
        contingency_table = pd.crosstab(relevant_df[f"binary_{survey_mapping.Q_GRADED_UNEQUAL}"], relevant_df[col])
        contingency_table.to_csv(os.path.join(result_path, f"graded_con_mc_{col_name}_prios_counts.csv"), index=True)
        result = helper_funcs.chi_squared_test(contingency_table=contingency_table)
        result["question"] = f"binary_{survey_mapping.Q_GRADED_UNEQUAL}"
        result.to_csv(os.path.join(result_path, f"graded_con_mc_{col_name}_prios_chisquared.csv"), index=False)

        """
        Plot proportion of Yes-No in the pill question per consciousness expertise level
        """

        count_df = relevant_df.groupby([col, f"binary_{survey_mapping.Q_GRADED_UNEQUAL}"]).size().reset_index(name='count')
        count_df["total"] = count_df.groupby(col)["count"].transform('sum')
        count_df["proportion"] = 100 * count_df["count"] / count_df["total"]
        count_df[survey_mapping.Q_GRADED_UNEQUAL] = count_df[f"binary_{survey_mapping.Q_GRADED_UNEQUAL}"].map({1: "Agree", 0: "Disagree"})

        pivot_df = count_df.pivot(index=col, columns=survey_mapping.Q_GRADED_UNEQUAL,values="proportion").fillna(0).reset_index(drop=False, inplace=False)
        pivot_df.to_csv(os.path.join(result_path, f"graded_con_mc_{col_name}_prios_props.csv"), index=False)
        plotter.plot_expertise_proportion_bars(df=pivot_df, cols=["Agree", "Disagree"],
                                               cols_colors={"Disagree": '#B33A00', "Agree": '#3B4E58'},
                                               x_axis_exp_col_name=col,
                                               x_label=f"{col}",
                                               x_map={"No": 0, "Yes": 1},
                                               y_ticks=[0, 25, 50, 75, 100],
                                               save_name=f"graded_con_mc_{col_name}_prios_props",
                                               save_path=result_path, plt_title=survey_mapping.Q_GRADED_UNEQUAL)


    """
    Relation to moral status features: prepare data for analysis:
    Do you think non-conscious creatures/systems should be taken into account in moral decisions by top feature 
    CHI SQUARED TEST
    """
    ms_features = analysis_dict["moral_considerations_features"].copy()
    important_features_q = "What do you think is important for moral considerations?"
    most_important_q = "Which do you think is the most important for moral considerations?"
    # make sure the 'most important' one is filled, as when they only chose one feature we didn't ask them that
    ms_features[most_important_q].fillna(ms_features[important_features_q], inplace=True)
    combined = pd.merge(ms_prios, ms_features, on=process_survey.COL_ID)
    non_conscious_ms = "Do you think non-conscious creatures/systems should be taken into account in moral decisions?"
    combined.rename(columns={most_important_q: "most_important_feature", non_conscious_ms: "non_conscious_ms"}, inplace=True)
    combined.to_csv(os.path.join(result_path, "moral_prios_and_features.csv"), index=False)
    """
    create a contingency table for a chi-squared test to check whether the clusters significantly differ in  
    their proportion of people who said "Yes"
    """
    contingency_table = pd.crosstab(combined["most_important_feature"],
                                    combined["non_conscious_ms"])
    chisquare_result = helper_funcs.chi_squared_test(contingency_table=contingency_table)
    chisquare_result.to_csv(os.path.join(result_path, "moral_features_non-conscious_ms_chisquared.csv"), index=False)
    contingency_table.to_csv(os.path.join(result_path, "moral_features_non-conscious_ms_contingency_table.csv"),
                             index=True)


    """
    If df_earth_cluster is not None, take the clustering from the Earth-in-danger scenarios, and see if they apply 
    here as well. 
    """
    if df_earth_cluster is not None:
        ncon_moral_prio_q = "Do you think non-conscious creatures/systems should be taken into account in moral decisions?"
        people_moral_prio_q = "Do you think some people should have a higher moral status than others?"

        q_map = [ncon_moral_prio_q, people_moral_prio_q]
        result_list = list()
        for q in q_map:
            ms_prios_relevant = ms_prios.loc[:, [process_survey.COL_ID, q]]
            df_clusters = df_earth_cluster[[process_survey.COL_ID, "Cluster"]]
            ms_prios_relevant_with_cluster = pd.merge(ms_prios_relevant, df_clusters, how="inner",
                                                      on=process_survey.COL_ID).reset_index(drop=True, inplace=False)
            """
            create a contingency table for a chi-squared test to check whether the clusters significantly differ in  
            their proportion of people who said "Yes"
            """
            contingency_table = pd.crosstab(ms_prios_relevant_with_cluster["Cluster"],
                                            ms_prios_relevant_with_cluster[q])
            chisquare_result = helper_funcs.chi_squared_test(contingency_table=contingency_table)
            chisquare_result[f"Question"] = [q]
            chisquare_result[f"per"] = ["Earth-in-danger cluster"]
            result_list.append(chisquare_result)
        result_df = pd.concat(result_list)
        result_df.to_csv(os.path.join(result_path, f"chisqared_earthInDanger_clusters_per_Q.csv"), index=False)

    return


def plot_graded_consciousness_given_df(df, save_path, prefix="", suffix=""):
    """
    Plot the answers to the rating questions (agreement) in a stacked bar plot. This can be used for any sub-set
    of the participants and does nothing but plotting
    """
    rating_color_list = ["#DB5461", "#fb9a99", "#70a0a4", "#26818B"]

    rating_labels = ["1", "2", "3", "4"]  # how much do you agree
    rating_questions = [survey_mapping.Q_GRADED_EQUAL, survey_mapping.Q_GRADED_UNEQUAL, survey_mapping.Q_GRADED_INCOMP]

    df_melted = df.melt(id_vars=process_survey.COL_ID, value_vars=rating_questions,
                        var_name='question', value_name='rating')
    counts = df_melted.groupby(['rating', 'question'])[process_survey.COL_ID].nunique().unstack(fill_value=0).reset_index(drop=False, inplace=False)
    counts.to_csv(os.path.join(save_path, "consciousness_graded_rating_counts.csv"), index=False)

    stats = {}
    for col in rating_questions:
        stats[col] = helper_funcs.compute_stats(df[col], possible_values=[1, 2, 3, 4])
    # Create DataFrame for plotting
    plot_data = {}
    for item, (proportions, mean_rating, std_dev, n) in stats.items():
        plot_data[item] = {
            "Proportion": proportions,
            "Mean": mean_rating,
            "Std Dev": std_dev,
            "N": n
        }
    # Sort the data by the MEAN rating (Python 3.7+ dictionaries maintain the insertion order of keys)
    sorted_plot_data = sorted(plot_data.items(), key=lambda x: x[1]["Mean"], reverse=True)
    plotter.plot_stacked_proportion_bars(plot_data=sorted_plot_data, num_plots=3, legend_labels=rating_labels,
                                         colors=rating_color_list, num_ratings=4, title=f"How Much do you Agree?",
                                         save_path=save_path, annotate_bar=True, annot_font_color="#e0e1dd",
                                         show_mean=False, sem_line=False, fmt="svg",
                                         save_name=f"{prefix}consciousness_graded_ratings{suffix}",
                                         text_width=39)
    # save the figure data
    df_result = pd.DataFrame(sorted_plot_data)
    df_result.to_csv(os.path.join(save_path, f"{prefix}consciousness_graded_ratings{suffix}.csv"))
    return


def graded_consciousness(analysis_dict, save_path, df_earth_cluster=None, remove_contradicting=False):
    """
    Answers to the cluster of questions about whether consciousness is graded
    """
    # save path
    result_path = os.path.join(save_path, "graded_consciousness")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    c_graded = analysis_dict["consciousness_graded"]

    # Plot the answers to the rating questions (agreement) in a stacked bar plot
    plot_graded_consciousness_given_df(df=c_graded, save_path=result_path)

    """
    Consciousness experts present themselves as experts vs. 'laypeople'; 
    see whether Consciousness experts (self-rated experience with consciousness 4 and up) differ from the rest
    in the proportions of agreeing with the assertion that consciousness is graded
    ("if two creatures/systems are conscious, they are not necessarily equally conscious"; 
    survey_mapping.Q_GRADED_UNEQUAL)
    """
    con_exp = analysis_dict["consciousness_exp"][[process_survey.COL_ID, survey_mapping.Q_CONSC_EXP]]
    c_graded_with_con_exp = pd.merge(con_exp, c_graded, on=[process_survey.COL_ID])
    # "1" (expert) if survey_mapping.Q_CONSC_EXP > 3, "0" (non expert) if survey_mapping.Q_CONSC_EXP <=3
    expertise = [4, 5]
    c_graded_with_con_exp["is_cons_expert"] = c_graded_with_con_exp[survey_mapping.Q_CONSC_EXP].isin(expertise).astype(int)
    # also binarize agreement with the assertion: disagree (0) --> 1, 2; agree (1) --> 3, 4
    agreed = [3, 4]
    c_graded_with_con_exp[f"binary_{survey_mapping.Q_GRADED_UNEQUAL}"] = c_graded_with_con_exp[survey_mapping.Q_GRADED_UNEQUAL].isin(agreed).astype(int)
    # do it with the BINARY versions of the answers to both questions
    contingency_table = pd.crosstab(c_graded_with_con_exp[f"binary_{survey_mapping.Q_GRADED_UNEQUAL}"],
                                    c_graded_with_con_exp["is_cons_expert"])
    contingency_table.to_csv(os.path.join(result_path, f"graded_con_conscExperts_vs_nonExperts_counts.csv"), index=True)
    result = helper_funcs.chi_squared_test(contingency_table=contingency_table)
    result["question"] = f"binary_{survey_mapping.Q_GRADED_UNEQUAL}"
    result.to_csv(os.path.join(result_path, f"graded_con_conscExperts_vs_nonExperts_chisquared.csv"), index=False)

    """
    Plot proportion of Yes-No in the pill question per consciousness expertise level
    """

    count_df = c_graded_with_con_exp.groupby([survey_mapping.Q_CONSC_EXP, f"binary_{survey_mapping.Q_GRADED_UNEQUAL}"]).size().reset_index(name='count')
    count_df["total"] = count_df.groupby(survey_mapping.Q_CONSC_EXP)["count"].transform('sum')
    count_df["proportion"] = 100 * count_df["count"] / count_df["total"]
    count_df[survey_mapping.Q_GRADED_UNEQUAL] = count_df[f"binary_{survey_mapping.Q_GRADED_UNEQUAL}"].map({1: "Agree", 0: "Disagree"})
    c_graded_with_con_exp.to_csv(os.path.join(result_path, "c_graded_per_cons_exp.csv"), index=False)

    # for plotting
    pivot_df = count_df.pivot(index=survey_mapping.Q_CONSC_EXP, columns=survey_mapping.Q_GRADED_UNEQUAL, values="proportion").fillna(0).reset_index(drop=False, inplace=False)
    pivot_df.to_csv(os.path.join(result_path, "c_graded_per_cons_exp_props.csv"), index=False)
    plotter.plot_expertise_proportion_bars(df=pivot_df, cols=["Agree", "Disagree"],
                                           cols_colors={"Disagree": '#B33A00', "Agree": '#3B4E58'},
                                           x_axis_exp_col_name=survey_mapping.Q_CONSC_EXP,
                                           x_label="Reported experience with consciousness",
                                           y_ticks=[0, 25, 50, 75, 100],
                                           save_name="c_graded_per_cons_exp_props",
                                           save_path=result_path, plt_title=survey_mapping.Q_GRADED_UNEQUAL,
                                           annotate_bar=True)


    """
    If we have the cluster information here, can agreement with the assertions explain the clusters
    """
    assertions = [survey_mapping.Q_GRADED_EQUAL, survey_mapping.Q_GRADED_UNEQUAL, survey_mapping.Q_GRADED_INCOMP]
    if df_earth_cluster is not None:
        df_earth_cluster_relevant = df_earth_cluster[[process_survey.COL_ID, "Cluster"]]
        c_graded_relevant = c_graded[[process_survey.COL_ID] + assertions]
        merged = pd.merge(c_graded_relevant, df_earth_cluster_relevant, on=process_survey.COL_ID)
        # modelling in R
        column_mapping_dict = {survey_mapping.Q_GRADED_EQUAL: "if_c_then_equal",
                               survey_mapping.Q_GRADED_UNEQUAL: "if_c_then_notNec_equal",
                               survey_mapping.Q_GRADED_INCOMP: "if_c_then_incomp"}
        merged.rename(columns=column_mapping_dict, inplace=True)
        assertions_mapped = [column_mapping_dict[a] for a in assertions]
        merged.to_csv(os.path.join(result_path, "graded_qs_with_EiDCluster.csv"), index=False)

        """
        For coherence with other agreement checks, let's binarize agreement
        """
        assertions = ["if_c_then_equal", "if_c_then_notNec_equal", "if_c_then_incomp"]
        results = list()
        stats = list()
        for assertion in assertions:  # RONY agreement was 1-4, so "disagree"=[1, 2] "agree"=[3, 4]
            merged[f"binary_{assertion}"] = (merged[assertion] >= 3).astype(int)
            """
            Chi squared test to check whether agreement (binary) with assertion is different between clusters
            """
            contingency_table = pd.crosstab(merged[f"binary_{assertion}"],
                                            merged["Cluster"])
            contingency_table.to_csv(os.path.join(result_path, f"{assertion}_perEiDcluster.csv"),index=True)
            result = helper_funcs.chi_squared_test(contingency_table=contingency_table)
            result["question"] = f"binary_{assertion}"
            results.append(result)

            # descriptives
            summary_df = merged.groupby("Cluster")[assertion].agg(["mean", "std", "min", "max", "count"]).reset_index()
            summary_df["question"] = assertion
            stats.append(summary_df)

            # plot the agreement with the assertion for each cluster
            merged_relevant = merged.loc[:, [process_survey.COL_ID, f"binary_{assertion}", "Cluster"]]
            count_df = merged_relevant.groupby(["Cluster", f"binary_{assertion}"]).size().reset_index(name='count')
            count_df["total"] = count_df.groupby("Cluster")["count"].transform('sum')
            count_df["proportion"] = 100 * count_df["count"] / count_df["total"]
            count_df[assertion] = count_df[f"binary_{assertion}"].map({1: "Agree", 0: "Disagree"})
            count_df.to_csv(os.path.join(result_path, f"{assertion}_perEiDcluster_counts.csv"), index=False)

            plot_df = count_df.pivot(index="Cluster", columns=assertion, values="proportion").fillna(0)
            cols = ["Agree", "Disagree"]
            cols_colors = {"Disagree": '#B33A00', "Agree": '#3B4E58'}
            plot_df = plot_df.reset_index()  # add Cluster as a column again for use in the plotting
            plotter.plot_expertise_proportion_bars(
                df=plot_df,
                x_axis_exp_col_name="Cluster",
                x_label="Cluster",
                cols=cols,
                cols_colors=cols_colors,
                y_ticks=[0, 25, 50, 75, 100],
                save_name=f"{assertion}_with_EiDCluster",
                save_path=result_path,
                plt_title=f"{assertion} per Cluster",
                fmt='svg',
                x_map={0: '0', 1: '1'})

        result_df = pd.concat(results).reset_index(drop=True, inplace=False)
        result_df.to_csv(os.path.join(result_path, f"graded_qs_with_EiDCluster_ChiSquared.csv"), index=False)
        stats_df = pd.concat(stats)
        stats_df.to_csv(os.path.join(result_path, f"graded_qs_with_EiDCluster_stats.csv"),index=False)

    return



    """
    TAKE THE c_graded_contradiction SUBS (n=12 in exploratory sample) OUT of any follow up analysis using graded_cosnciousness
    """
    c_graded_contradiction = c_graded[(c_graded[survey_mapping.Q_GRADED_EQUAL] == 4) & (c_graded[survey_mapping.Q_GRADED_UNEQUAL] == 4)]
    c_graded_contradiction.to_csv(os.path.join(result_path, f"contradicting_subs_{remove_contradicting}.csv"), index=False)
    if remove_contradicting:
        c_subs = c_graded_contradiction[process_survey.COL_ID].tolist()
        c_graded = c_graded[~c_graded[process_survey.COL_ID].isin(c_subs)].reset_index(drop=True, inplace=False)

    # now, other things


    """
    Relations to expertise
    """

    animal_experience_df = analysis_dict["animal_exp"].loc[:,
                           [process_survey.COL_ID, survey_mapping.Q_ANIMAL_EXP]].rename(
        columns={survey_mapping.Q_ANIMAL_EXP: "exp_animals"}, inplace=False)
    ai_experience_df = analysis_dict["ai_exp"].loc[:, [process_survey.COL_ID, survey_mapping.Q_AI_EXP]].rename(
        columns={survey_mapping.Q_AI_EXP: "exp_ai"}, inplace=False)
    ethics_experience_df = analysis_dict["ethics_exp"].loc[:,
                           [process_survey.COL_ID, survey_mapping.Q_ETHICS_EXP]].rename(
        columns={survey_mapping.Q_ETHICS_EXP: "exp_ethics"}, inplace=False)
    con_experience_df = analysis_dict["consciousness_exp"].loc[:,
                        [process_survey.COL_ID, survey_mapping.Q_CONSC_EXP]].rename(
        columns={survey_mapping.Q_CONSC_EXP: "exp_consc"}, inplace=False)

    df_experience_list = [animal_experience_df, con_experience_df, ethics_experience_df, ai_experience_df]
    df_experience_merged = reduce(lambda left, right: pd.merge(left, right, on=[process_survey.COL_ID], how="outer"),
                                  df_experience_list)
    df_graded_exp = pd.merge(c_graded, df_experience_merged, on=process_survey.COL_ID)
    df_graded_exp.to_csv(os.path.join(result_path, "graded_experience.csv"))

    """
    Does it mean the interests of the more conscious creature matter more?
    """
    all_count = df_graded_exp.shape[0]  # total N
    df_graded_exp_filtered = df_graded_exp[(df_graded_exp[survey_mapping.Q_GRADED_MATTERMORE] == survey_mapping.ANS_YES)
                                           | (df_graded_exp[
                                                  survey_mapping.Q_GRADED_MATTERMORE] == survey_mapping.ANS_NO)]
    not_equal_count = df_graded_exp_filtered.shape[0]  # only those who think C is not equal saw this question
    yes_count = \
        df_graded_exp_filtered[
            df_graded_exp_filtered[survey_mapping.Q_GRADED_MATTERMORE] == survey_mapping.ANS_YES].shape[
            0]
    yes_prop = 100 * (yes_count / not_equal_count)
    no_count = \
        df_graded_exp_filtered[
            df_graded_exp_filtered[survey_mapping.Q_GRADED_MATTERMORE] == survey_mapping.ANS_NO].shape[0]
    no_prop = 100 * (no_count / not_equal_count)

    df_graded_extra = pd.DataFrame({"N": [all_count], "N_interestQ": [not_equal_count],
                                    "Yes_interestQ": [yes_count], "Yes_interestQ_prop": [yes_prop],
                                    "No_interestQ": [no_count], "No_interestQ_prop": [no_prop]})
    df_graded_extra.to_csv(os.path.join(result_path, "graded_experience_interests.csv"))

    """
    Relations to variability in Consciousness Ratings
    """
    other_creatures_c = analysis_dict["other_creatures_cons"]
    # other_creatures_c["c_ratings variability"] = other_creatures_c.iloc[:, 1:].std(axis=1)
    # std_df = other_creatures_c[[process_survey.COL_ID, "c_ratings variability"]]
    # merged_df = pd.merge(c_graded, std_df, on=process_survey.COL_ID)
    # TODO: STOPPED HERE

    return c_graded_contradiction


def consciousness_intelligence(analysis_dict, save_path):
    """
    Answers to the cluster of questions about the relationship between consciousness and intelligence
    """
    # save path
    result_path = os.path.join(save_path, "consciousness_intelligence")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    con_intellect = analysis_dict["con_intellect"]
    question = "Do you think consciousness and intelligence are related?"
    category_counts = con_intellect[question].value_counts()
    plotter.plot_pie(categories_names=category_counts.index.tolist(), categories_counts=category_counts.tolist(),
                     categories_colors=CAT_COLOR_DICT, title=f"{question}",
                     save_path=result_path, save_name=f"{question.replace('?', '').replace('/', '-')}", fmt="svg")
    category_counts.to_csv(os.path.join(result_path, f"{question.replace('?', '').replace('/', '-')}.csv"))

    """
    Follow Up
    """

    follow_up = "How?"
    con_intellect_how = con_intellect[con_intellect[follow_up].notnull()]
    category_props = con_intellect_how[follow_up].value_counts(normalize=True)
    category_props = category_props.reset_index(drop=False, inplace=False)
    category_props["proportion"] = 100 * category_props["proportion"]
    color_list = ["#1d3557", "#457b9d", "#fca311", "#0d1b2a"]
    plotter.plot_categorical_bars(categories_prop_df=category_props,
                                  category_col=follow_up, y_min=0, y_max=105, y_skip=10,
                                  data_col="proportion", delete_y=False, add_pcnt=True,
                                  categories_colors=color_list,
                                  save_path=result_path, save_name=f"{follow_up[:-1]}", fmt="svg")
    c = 4


    """
    Consciousness experts present themselves as experts vs. 'laypeople'; 
    see whether Consciousness experts (self-rated experience with consciousness 4 and up) differ from the rest
    in the proportions of yes-no to this question. 
    """
    con_exp = analysis_dict["consciousness_exp"][[process_survey.COL_ID, survey_mapping.Q_CONSC_EXP]]
    con_intellect_with_con_exp = reduce(lambda left, right: pd.merge(left, right, on=[process_survey.COL_ID], how='outer'),
                                        [con_exp, con_intellect[[process_survey.COL_ID, question]]])
    # "1" (expert) if survey_mapping.Q_CONSC_EXP > 3, "0" (non expert) if survey_mapping.Q_CONSC_EXP <=3
    expertise = [4, 5]
    con_intellect_with_con_exp["is_cons_expert"] = con_intellect_with_con_exp[survey_mapping.Q_CONSC_EXP].isin(expertise).astype(int)
    contingency_table = pd.crosstab(con_intellect_with_con_exp[question],
                                    con_intellect_with_con_exp["is_cons_expert"])
    contingency_table.to_csv(os.path.join(result_path, f"con_intellect_conscExperts_vs_nonExperts_counts.csv"), index=True)
    result = helper_funcs.chi_squared_test(contingency_table=contingency_table)
    result["question"] = question
    result.to_csv(os.path.join(result_path, f"con_intellect_conscExperts_vs_nonExperts_chisquared.csv"), index=False)

    """
    Plot proportion of Yes-No in the consciousness-intelligence question per consciousness expertise level
    """

    count_df = con_intellect_with_con_exp.groupby([survey_mapping.Q_CONSC_EXP, question]).size().reset_index(name='count')
    count_df["total"] = count_df.groupby(survey_mapping.Q_CONSC_EXP)["count"].transform('sum')
    count_df["proportion"] = 100 * count_df["count"] / count_df["total"]
    count_df[f"binary_{question}"] = count_df[question].map({"Yes": 1, "No": 0})
    con_intellect_with_con_exp.to_csv(os.path.join(result_path, "con_intellect_per_cons_exp.csv"), index=False)

    # for plotting
    pivot_df = count_df.pivot(index=survey_mapping.Q_CONSC_EXP, columns=question,
                              values="proportion").fillna(0).reset_index(drop=False, inplace=False)
    pivot_df.to_csv(os.path.join(result_path, "con_intellect_per_cons_exp_props.csv"), index=False)
    plotter.plot_expertise_proportion_bars(df=pivot_df, cols=["Yes", "No"],
                                           cols_colors={"No": '#B33A00', "Yes": '#3B4E58'},
                                           x_axis_exp_col_name=survey_mapping.Q_CONSC_EXP,
                                           x_label="Reported experience with consciousness",
                                           y_ticks=[0, 25, 50, 75, 100],
                                           save_name="con_intellect_per_cons_exp_props",
                                           save_path=result_path, plt_title=question)

    return




    """
    ***************** DEPRECATED *****************
    """


    """
    Demographics on Yes-No
    """
    demographics_df = analysis_dict["demographics"]
    for ans in con_intellect[question].unique().tolist():  # No, Yes
        con_intellect_ans_people = con_intellect[con_intellect[question] == ans][process_survey.COL_ID].tolist()
        demographics_ans = demographics_df[demographics_df[process_survey.COL_ID].isin(con_intellect_ans_people)].reset_index(drop=True)
        specific_result_path = os.path.join(result_path, f"ans_{ans.lower()}")
        if not os.path.isdir(specific_result_path):
            os.mkdir(specific_result_path)
        demographics_age(demographics_df=demographics_ans, save_path=specific_result_path)
        demographics_gender(demographics_df=demographics_ans, save_path=specific_result_path)
        demographics_education(demographics_df=demographics_ans, save_path=specific_result_path)

    age_bins = [18, 25, 35, 45, 55, 65, 75, np.inf]
    age_labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "66-75", "76+"]
    demographics_df["age_group"] = pd.cut(demographics_df[survey_mapping.Q_AGE], bins=age_bins, labels=age_labels,
                                          right=True, include_lowest=True)

    """
    Experience
    """
    ai_exp = analysis_dict["ai_exp"][[process_survey.COL_ID, survey_mapping.Q_AI_EXP]]
    animal_exp = analysis_dict["animal_exp"][[process_survey.COL_ID, survey_mapping.Q_ANIMAL_EXP]]
    ethics_exp = analysis_dict["ethics_exp"][[process_survey.COL_ID, survey_mapping.Q_ETHICS_EXP]]
    con_exp = analysis_dict["consciousness_exp"][[process_survey.COL_ID, survey_mapping.Q_CONSC_EXP]]
    experience_df = reduce(lambda left, right: pd.merge(left, right, on=[process_survey.COL_ID], how='outer'),
                           [ai_exp, animal_exp, ethics_exp, con_exp])
    con_intellect_with_personal_info = reduce(lambda left, right: pd.merge(left, right, on=[process_survey.COL_ID], how='outer'),
                                              [demographics_df, experience_df, con_intellect[[process_survey.COL_ID, question]]])
    experience_cols = [col for col in con_intellect_with_personal_info.columns if col.startswith("On a scale")]

    """
    Random Forest Classifier: Presonal Details --> Cosnciousness & Intelligence
    """
    categorical_cols = ["age_group", "How do you describe yourself?", "What is your education background?", "Current primary employment domain"]
    con_intellect_with_personal_info["con_intel_related"] = con_intellect_with_personal_info[question].str.strip().map({'Yes': 1, 'No': 0})
    con_intellect_with_personal_info = con_intellect_with_personal_info[[process_survey.COL_ID, "con_intel_related"] + experience_cols + categorical_cols]
    result = helper_funcs.run_random_forest_pipeline(dataframe=con_intellect_with_personal_info,
                                                     dep_col="con_intel_related",
                                                     categorical_cols=categorical_cols,
                                                     save_path=result_path, save_prefix="demographics",
                                                     order_cols=experience_cols, rare_class_threshold=5, cv_folds=10,
                                                     scoring_method="accuracy")




    common_denominator = "What is the common denominator?"
    con_intellect_d = con_intellect[con_intellect[common_denominator].notnull()]
    con_intellect_d.to_csv(os.path.join(result_path, "common_denominator.csv"), index=False)
    answers = con_intellect_d[common_denominator].tolist()

    information_processing = [ex for ex in answers if
                              (re.search("process and interpret information", ex, re.IGNORECASE) or
                               (re.search("processing power", ex, re.IGNORECASE)) or
                               (re.search("computation", ex, re.IGNORECASE)))]
    cognition = [ex for ex in answers if (re.search("cognition", ex, re.IGNORECASE) or
                                          (re.search("cognitive processing", ex, re.IGNORECASE)) or
                                          (re.search("cognitive system", ex, re.IGNORECASE)) or
                                          (re.search("attention", ex, re.IGNORECASE)) or
                                          re.search("abstract representation", ex, re.IGNORECASE) or
                                          (re.search("language", ex, re.IGNORECASE)))]
    thinking = [ex for ex in answers if (re.search("think", ex, re.IGNORECASE) or
                                         re.search("thinking", ex, re.IGNORECASE) or
                                         re.search("thoughts", ex, re.IGNORECASE))]
    complexity = [ex for ex in answers if (re.search("complex", ex, re.IGNORECASE) or
                                           re.search("complexity", ex, re.IGNORECASE) or
                                           re.search("adaptation", ex, re.IGNORECASE) or
                                           re.search("advanced brain structure", ex, re.IGNORECASE))]
    emotions = [ex for ex in answers if (re.search("emotion", ex, re.IGNORECASE) or
                                         (re.search("emotions", ex, re.IGNORECASE)))]
    goals_actions = [ex for ex in answers if (re.search("goals", ex, re.IGNORECASE) or
                                              (re.search("actions", ex, re.IGNORECASE)) or
                                              (re.search("decision making", ex, re.IGNORECASE)) or
                                              (re.search("choosing", ex, re.IGNORECASE)))]
    brains = [ex for ex in answers if (re.search("brain", ex, re.IGNORECASE) or
                                       (re.search("neural organization", ex, re.IGNORECASE)) or
                                       (re.search("neural activity", ex, re.IGNORECASE)))]

    lol = information_processing + cognition + thinking + complexity + emotions + goals_actions + brains
    rest = [ex for ex in answers if ex not in lol]
    result_df = pd.DataFrame({"information_processing": [len(information_processing)], "cognition": [len(cognition)],
                              "thinking": [len(thinking)], "complexity": [len(complexity)],
                              "emotions": [len(emotions)], "goals_actions": [len(goals_actions)],
                              "brains": [len(brains)],
                              "misc": [len(rest)]}).transpose()
    result_df.rename(columns={result_df.columns[0]: "count"}, inplace=True)
    result_df.to_csv((os.path.join(result_path, "common denominator_examples.csv")), index=True)  # index is the type
    # save the misc ones
    rest_df = pd.DataFrame({f"common denominator_miscellaneous examples": rest})
    rest_df.to_csv(os.path.join(result_path, "common denominator_examples_misc.csv"), index=False)

    return


def demographics_age(demographics_df, save_path):
    age = "How old are you?"
    age_stats = demographics_df[age].astype(float).describe()
    age_stats.to_csv(os.path.join(save_path, "age_stats.csv"))  # index=True as it includes the information

    age_counts = demographics_df[age].value_counts()
    age_counts_df = age_counts.reset_index(drop=False, inplace=False)
    age_counts_df_sorted = age_counts_df.sort_values(age, ascending=True).reset_index(drop=True, inplace=False)
    age_counts_df_sorted[age] = age_counts_df_sorted[age].astype(int)
    plotter.plot_histogram(df=age_counts_df_sorted, category_col=age, data_col="count", x_label="Age", y_label="Count",
                           save_path=save_path, save_name=f"age", format="svg", color_palette="rocket_r")

    age_props = demographics_df[age].value_counts(normalize=True)
    age_props_df = age_props.reset_index(drop=False, inplace=False)
    age_props_df["proportion"] = 100 * age_props_df["proportion"]

    merged_df = pd.merge(age_counts_df, age_props_df, on=age)
    merged_df.to_csv(os.path.join(save_path, "age.csv"), index=False)

    # descriptives
    ages = np.repeat(age_counts_df[age], age_counts_df["Count"])
    total_count = len(ages)
    average_age = np.mean(ages)
    std_deviation = np.std(ages, ddof=1)
    std_error = std_deviation / np.sqrt(len(ages))
    median_age = np.median(ages)
    min_age = np.min(ages)
    max_age = np.max(ages)
    summary_df = pd.DataFrame({
        "Statistic": ["N", "Average", "Standard Deviation", "Standard Error", "Median", "Minimum", "Maximum"],
        "Value": [total_count, average_age, std_deviation, std_error, median_age, min_age, max_age]
    })
    summary_df.to_csv(os.path.join(save_path, "age_stats.csv"), index=False)
    return


def demographics_gender(demographics_df, save_path):
    gender_order = ["Female", "Male", "Non-binary", "Genderqueer", "Prefer not to say"]
    gender = "How do you describe yourself?"
    category_counts = demographics_df[gender].value_counts()

    gender_color_dict = {"Female": "#d4a373",
                         "Male": "#4a5759",
                         "Non-binary": "#f7e1d7",
                         "Genderqueer": "#edafb8",
                         "Prefer not to say": "#dedbd2"}
    plotter.plot_pie(categories_names=category_counts.index.tolist(), categories_counts=category_counts.tolist(),
                     categories_colors=gender_color_dict, title=f"{gender}", pie_direction=90, edge_color="none",
                     save_path=save_path, save_name=f"gender", fmt="svg",
                     props_in_legend=True, annot_props=False, annot_groups=False,
                     legend=True, legend_order=gender_order, legend_vertical=True)

    category_props = demographics_df[gender].value_counts(normalize=True)
    category_props_df = category_props.reset_index(drop=False, inplace=False)
    category_props_df["proportion"] = 100 * category_props_df["proportion"]
    category_props_df_ordered = category_props_df.sort_values("proportion", ascending=False)
    # plotter.plot_categorical_bars(categories_prop_df=category_props_df_ordered,
    #                              category_col=gender, y_min=0, y_max=60, y_skip=10,
    #                              data_col="proportion",
    #                              categories_colors=color_list,
    #                              save_path=save_path, save_name=f"gender", fmt="svg")

    category_counts_df = category_counts.reset_index(drop=False, inplace=False)
    merged_df = pd.merge(category_counts_df, category_props_df_ordered, on=gender)
    merged_df.to_csv(os.path.join(save_path, "gender.csv"), index=False)
    return


def demographics_education(demographics_df, save_path):
    education = "What is your education background?"
    education_order = [survey_mapping.EDU_NONE, survey_mapping.EDU_PRIM, survey_mapping.EDU_SECD,
                       survey_mapping.EDU_POSTSEC, survey_mapping.EDU_GRAD]
    education_labels = {edu: edu.replace(" education", "") for edu in education_order[1:]}
    education_labels[survey_mapping.EDU_NONE] = survey_mapping.EDU_NONE
    education_labels = {edu: re.sub(r'\(.*?\)', '', education_labels[edu]) for edu in
                        education_labels.keys()}  # remove parantheses

    education_color_dict = {survey_mapping.EDU_NONE: "#DCEDFF",
                            survey_mapping.EDU_PRIM: "#90C3C8",
                            survey_mapping.EDU_SECD: "#759FBC",
                            survey_mapping.EDU_POSTSEC: "#1F5673",
                            survey_mapping.EDU_GRAD: "#463730"}

    education_color_dict_experience = {survey_mapping.EDU_NONE: "#e63946",
                                       survey_mapping.EDU_PRIM: "#f1faee",
                                       survey_mapping.EDU_SECD: "#a8dadc",
                                       survey_mapping.EDU_POSTSEC: "#457b9d",
                                       survey_mapping.EDU_GRAD: "#344968"}

    education_map = {survey_mapping.EDU_NONE: 1,
                     survey_mapping.EDU_PRIM: 2,
                     survey_mapping.EDU_SECD: 3,
                     survey_mapping.EDU_POSTSEC: 4,
                     survey_mapping.EDU_GRAD: 5}
    education_map_reversed = {v: k for k, v in education_map.items()}

    education_counts = demographics_df[education].value_counts().reset_index(drop=False, inplace=False)
    education_props = demographics_df[education].value_counts(normalize=True)
    education_props_df = education_props.reset_index(drop=False, inplace=False)
    education_props_df["proportion"] = 100 * education_props_df["proportion"]

    education_counts_dict = dict(
        zip(education_counts[education], education_counts["count"]))  # a dictionary for faster lookups
    education_counts_pie = [education_counts_dict.get(edu, 0) for edu in education_order]
    education_colors = [education_color_dict[edu] for edu in education_order]

    plotter.plot_pie(categories_names=education_order,
                     categories_counts=education_counts_pie,
                     categories_colors=education_color_dict,
                     title=f"{education}", legend=True, legend_vertical=True,
                     edge_color="none", pie_direction=180,
                     annot_groups=False, annot_props=True,
                     save_path=save_path, save_name=f"education", fmt="svg")

    # Different plotting; rn ugly
    demographics_df[f"{education}_coded"] = demographics_df[education].map(education_map)
    values_order = [1, 2, 3, 4, 5]
    stats = {education: helper_funcs.compute_stats(demographics_df[f"{education}_coded"], possible_values=values_order)}
    plot_data = {}
    for item, (proportions, mean_rating, std_dev, n) in stats.items():
        plot_data[item] = {
            "Proportion": proportions,
            "Mean": mean_rating,
            "Std Dev": std_dev,
            "N": n,
        }
    sorted_plot_data = {key: plot_data[key] for key in list(dict(plot_data).keys()) if key in plot_data}.items()

    rating_labels = [education_map_reversed[i] for i in values_order]
    rating_color_list = ["#E7E7E7", "#B7CED0", "#87B4B9", "#569BA2", "#26818B"]
    topic_name = "Education Level"
    #plotter.plot_stacked_proportion_bars(plot_data=sorted_plot_data, num_plots=1, legend_labels=rating_labels,
    #                                     ytick_visible=False, title=f"{topic_name.title()}",
    #                                     colors=rating_color_list, num_ratings=5,
    #                                     save_path=save_path, save_name=f"education_like_experience")

    order_dict = {survey_mapping.EDU_NONE: 4,
                  survey_mapping.EDU_PRIM: 3,
                  survey_mapping.EDU_SECD: 2,
                  survey_mapping.EDU_POSTSEC: 1,
                  survey_mapping.EDU_GRAD: 0}

    education_props_df_ordered = education_props_df.sort_values(by=education, key=lambda x: x.map(order_dict))
    education_props_df_ordered[f"{education}_label"] = education_props_df_ordered[education].replace(education_labels,
                                                                                                     inplace=False)
    education_props_df_ordered.reset_index(drop=True, inplace=True)
    merged_df = pd.merge(education_counts, education_props_df_ordered, on=education)
    merged_df.to_csv(os.path.join(save_path, "education.csv"), index=False)

    # education field
    field = "In what topic?"

    # people could have selected multiple values here, so handle it to count the right number of each option
    education_field_df = demographics_df.copy()
    education_field_df[field] = education_field_df[field].str.split(',')
    exploded_df = education_field_df.explode(field)
    category_counts = exploded_df[field].value_counts()

    # list names of topics where enough of the population have replied
    topic_props = category_counts.reset_index(drop=False, inplace=False)
    topic_props["proportion"] = (topic_props["count"] / topic_props["count"].sum()) * 100
    topic_props.to_csv(os.path.join(save_path, "education_topic.csv"), index=False)

    threshold = 1.5
    substantial_df = topic_props.loc[topic_props["proportion"] > threshold]
    substantial_list = substantial_df[field].tolist()

    topic_color_dict = {"Computer science / IT": "#1c3b60",
                        "Engineering": "#355171",
                        "Mathematics": "#4f6683",
                        "Statistics": "#687c95",
                        "Physics": "#496280",
                        "Astronomy": "#607690",
                        "Chemistry": "#7789a0",

                        "Biochemistry": "#2d5c73",
                        "Biomedical engineering": "#37718e",
                        "Biology / biotechnology": "#4b7f99",
                        "Neuroscience": "#5f8da5",
                        "Health sciences / public health": "#739cb0",
                        "Nutrition": "#87aabb",

                        "Dentistry": "#497264",
                        "Nursing": "#5b8e7d",
                        "Medicine": "#7ca597",

                        "Agriculture": "#828e6e",
                        "Environmental science": "#a3b18a",
                        "Earth sciences / geography": "#acb996",

                        "Philosophy": "#c78a5f",
                        "Psychology": "#e09c6b",
                        "Anthropology": "#f9ad77",
                        "Sociology": "#fab585",
                        "Social work": "#fabd92",
                        "Religious studies": "#fbc6a0",
                        "Veterinary medicine": "#fbcead",

                        "Architecture": "#f1597d",
                        "Urban planning / civil engineering": "#f1597d",
                        "Art / Art history / fine arts": "#f26c8c",
                        "Graphic design / UX / UI": "#f47e9a",
                        "Music": "#f591a9",
                        "Literature": "#f7a3b7",
                        "Linguistics": "#f9b5c5",

                        "History": "#b3493f",
                        "Journalism": "#c75146",
                        "International relations / political sciences": "#cd6259",
                        "Law": "#d2746b",
                        "Management": "#d8857e",
                        "Business / economics": "#dd9790",
                        "Finance / accounting": "#e3a8a3",
                        "Marketing": "#e9b9b5",
                        "Communications / media": "#eecbc8",

                        "Other": "#8D99AE"
                        }
    topic_order = list(topic_color_dict.keys())  # ordered dicts are supported in this Python version

    category_counts = [category_counts[topic] if topic in category_counts else 0 for topic in topic_order]

    plotter.plot_pie(categories_names=topic_order,
                     categories_counts=category_counts,
                     categories_colors=topic_color_dict,
                     title=f"{field}",
                     pie_direction=180,
                     annot_groups=True, annot_group_selection=substantial_list,
                     annot_props=False, edge_color="none",
                     save_path=save_path, save_name=f"field", fmt="png")
    return


def demographics(analysis_dict, save_path, df_earth_cluster=None):
    """
    Answers to the cluster of questions about participant demographics
    """
    # save path
    result_path = os.path.join(save_path, "demographics")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    con_demo = analysis_dict["demographics"]

    """
    Age
    """
    demographics_age(demographics_df=con_demo, save_path=result_path)

    """
    Country of Residence
    """

    country = "In which country do you currently reside?"
    category_counts = con_demo[country].value_counts(normalize=False).reset_index(drop=False, inplace=False)
    country_proportions = con_demo[country].value_counts(normalize=True).reset_index(drop=False, inplace=False)
    country_proportions = country_proportions.merge(category_counts, on=country)
    country_proportions["proportion"] = 100 * country_proportions["proportion"]  # turn to % s
    country_proportions.to_csv(os.path.join(result_path, "country.csv"), index=False)

    # country & continent world map proportions
    plotter.plot_world_map_proportion(country_proportions_df=country_proportions, data_column=country,
                                      save_path=result_path, save_name="geo", fmt="svg")

    """ 
    Gender
    """

    demographics_gender(demographics_df=con_demo, save_path=result_path)


    """ 
    Education
    """

    demographics_education(demographics_df=con_demo, save_path=result_path)

    """
    Employment
    """

    employment = "Current primary employment domain"
    employment_counts = con_demo[employment].value_counts()

    employment_colors = {"Tech/software development and engineering/IT": "#302F4D",
                         "Engineering/architecture": "#302F4D",
                         "Science/research": "#0e3747",
                         "Healthcare/medicine": "#124559",
                         "Public services/government": "#1E4F62",
                         "Non-profit/volunteering": "#29596B",
                         "Social services/counseling": "#356375",
                         "Farming/agriculture": "#406D7E",
                         "Transportation/logistics": "#4c7787",
                         "Construction/trades": "#597d8b",
                         "Manufacturing/production": "#6e8791",
                         "Business/finance": "#7c8f97",
                         "Administrative/clerical": "#8b9a9d",
                         "Customer services": "#99a6a4",
                         "Hospitality/tourism": "#a8b3ab",
                         "Retail/sales": "#b8b0ad",
                         "Marketing/advertising": "#d0a8a0",
                         "Media/communications": "#e3a392",
                         "Arts/entertainment": "#eab19b",
                         "Legal/paralegal": "#f0b09c",
                         "Management/consulting": "#ffbf9f",
                         "Homemaker/caregiver": "#ffc7b9",
                         "Student (full time)": "#e19d8b",
                         "Retired": "#d1897a",
                         "Other": "#c77a69",
                         "Unemployed": "white"
                         }
    employment_order = list(employment_colors.keys())
    employment_props = employment_counts.reset_index(drop=False, inplace=False)
    employment_props["proportion"] = (employment_props["count"] / employment_props["count"].sum()) * 100
    employment_props.to_csv(os.path.join(result_path, "emplyment.csv"), index=False)

    threshold = 1.5
    substantial_df = employment_props.loc[employment_props["proportion"] > threshold]
    substantial_list = substantial_df[employment].tolist()

    category_counts = [employment_counts[employment] if employment in employment_counts else 0 for employment in
                       employment_order]

    #plotter.plot_pie(categories_names=employment_order,
    #                 categories_counts=category_counts,
    #                 # [employment_counts[employment] for employment in employment_order]
    #                 categories_colors=employment_colors, title=f"{employment}", edge_color="none",
    #                 pie_direction=180, annot_groups=True, annot_group_selection=substantial_list, annot_props=False,
    #                 save_path=result_path, save_name=f"employment", fmt="png")

    """
    Can demographics explain killings (KPT question)?
    """

    con_demo_relevant = con_demo.drop(columns=["date", "date_start", "date_end", "duration_sec", "language"],
                                      inplace=False)
    age_bins = [18, 25, 35, 45, 55, 65, 75, np.inf]
    age_labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "66-75", "76+"]
    age_dict = {age_labels[i]: int(i) for i in range(len(age_labels))}
    con_demo_relevant["age_group"] = pd.cut(con_demo_relevant[survey_mapping.Q_AGE], bins=age_bins,
                                            labels=age_labels, right=True, include_lowest=True)
    # make it an ORDINAL column
    con_demo_relevant["age_group_coded"] = con_demo_relevant["age_group"].map(age_dict)
    # no ? in name
    con_demo_relevant.rename(columns={"What is your education background?_coded":
                                          "What is your education background_coded"}, inplace=True)

    categorical_cols = ["How do you describe yourself?",  # gender
                        "In which country do you currently reside?",  # country
                        "In what topic?",  # education topic: the education itself is ORDINAL
                        "education_Other: please specify",
                        "Current primary employment domain",
                        "employment_Other: please specify"]
    order_cols = ["age_group_coded",  # age group ORDINAL
                  "What is your education background_coded"]

    df_killing = analysis_dict["important_test_kill"].copy()
    df_killing = df_killing.loc[:, [process_survey.COL_ID] + list(survey_mapping.important_test_kill_tokens.keys())]
    df_killing.rename(columns=survey_mapping.important_test_kill_tokens, inplace=True)
    ans_map = {"No (will not kill to pass the test)": 0, "Yes (will kill to pass the test)": 1}

    # TODO: RONY DELETE THIS, for debugging
    df_killing[survey_mapping.Q_INTENTIONS] = df_killing[survey_mapping.Q_INTENTIONS].replace(ans_map)
    df_col = df_killing.loc[:, [process_survey.COL_ID, survey_mapping.Q_INTENTIONS]]
    con_demo_relevant[f"is_South_Africa"] = con_demo_relevant["In which country do you currently reside?"].apply(lambda x: 1 if x == "South Africa" else 0)
    df_merged = pd.merge(con_demo_relevant.loc[:, [process_survey.COL_ID, "Current primary employment domain"]], df_col, on=process_survey.COL_ID)
    df_merged["is_Student"] = df_merged["Current primary employment domain"].apply(lambda x: 1 if x == "Student (full time)" else 0)
    df_merged.groupby("is_Student")[survey_mapping.Q_INTENTIONS].agg(["count", "mean", "std"]).reset_index()
    c = 3


    for col in list(survey_mapping.important_test_kill_tokens.values()):
        print(f" **************** {col} **************** ")
        df_killing[col] = df_killing[col].replace(ans_map)
        df_col = df_killing.loc[:, [process_survey.COL_ID, col]]
        df_merged = pd.merge(con_demo_relevant, df_col, on=process_survey.COL_ID)
        helper_funcs.run_random_forest_pipeline(dataframe=df_merged, dep_col=col,
                                                categorical_cols=categorical_cols,
                                                order_cols=order_cols, save_path=result_path, save_prefix=f"KPT_{col}",
                                                rare_class_threshold=5, n_permutations=1000,
                                                scoring_method="accuracy",
                                                cv_folds=10)

    """
    Can demographics explain clusters?
    """
    con_demo_relevant = con_demo.drop(columns=["date", "date_start", "date_end", "duration_sec", "language"],
                                      inplace=False)
    if df_earth_cluster is not None:
        age_bins = [18, 25, 35, 45, 55, 65, 75, np.inf]
        age_labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "66-75", "76+"]
        age_dict = {age_labels[i]: i for i in range(len(age_labels))}
        con_demo_relevant["age_group"] = pd.cut(con_demo_relevant[survey_mapping.Q_AGE], bins=age_bins,
                                                labels=age_labels, right=True, include_lowest=True)
        # make it an ORDINAL column
        con_demo_relevant["age_group_coded"] = con_demo_relevant["age_group"].map(age_dict)

        df_cluster_relevant = df_earth_cluster.loc[:, [process_survey.COL_ID, "Cluster"]]
        cluster_with_demographics = pd.merge(con_demo_relevant, df_cluster_relevant, on=process_survey.COL_ID)
        cluster_with_demographics.to_csv(os.path.join(result_path, "cluster_with_demographics.csv"), index=False)

        con_demo_relevant.rename(columns={"What is your education background?_coded":
                                          "What is your education background_coded"})

        categorical_cols = ["How do you describe yourself?",  # gender
                            "In which country do you currently reside?",  # country
                            "In what topic?",  # education topic: the education itself is ORDINAL
                            "education_Other: please specify",
                            "Current primary employment domain",
                            "employment_Other: please specify"]
        order_cols = ["age_group_coded",  # age group ORDINAL
                      "What is your education background_coded"]

        helper_funcs.run_random_forest_pipeline(dataframe=cluster_with_demographics, dep_col="Cluster",
                                                categorical_cols=categorical_cols,
                                                order_cols=order_cols, save_path=result_path, save_prefix="clusterEiD",
                                                rare_class_threshold=5, n_permutations=1000,
                                                scoring_method="accuracy",
                                                cv_folds=10)



    return


def gender_cross(analysis_dict, save_path):
    """
    Crosses between gender and other answers
    """

    gender = "How do you describe yourself?"
    gender_order = ["Male", "Non-binary", "Prefer not to say", "Genderqueer", "Female"]
    gender_color_dict = plotter.diverging_palette(color_order=gender_order, left=225, right=45)

    # save path
    result_path = os.path.join(save_path, "effect_gender")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # gender cross with consciousness / moral status ratings
    gender_df = analysis_dict["demographics"].loc[:, [process_survey.COL_ID, gender]]

    long_data = pd.read_csv(os.path.join(save_path, "c_v_ms", "c_v_ms_long.csv"))
    cross_df = pd.merge(long_data, gender_df, on=process_survey.COL_ID, how='left')
    cross_df.to_csv(os.path.join(result_path, "gender_c_v_ms.csv"), index=False)

    for topic in cross_df["Topic"].unique().tolist():
        df_topic = cross_df[cross_df["Topic"] == topic]
        df_topic_meanPerSub = df_topic.groupby([process_survey.COL_ID], as_index=False)["Rating"].mean()
        df_topic_meanPerSub_crossed = pd.merge(df_topic_meanPerSub, gender_df, on=process_survey.COL_ID, how='left')
        plotter.plot_scatter(df=df_topic_meanPerSub_crossed, data_col="Rating", category_col=gender,
                             category_order=gender_order, category_color_dict=gender_color_dict, title_text=f"{gender}",
                             x_label="", y_label=f"Mean {topic} Rating", vertical_jitter=0,
                             y_min=1, y_max=4, y_skip=1, save_path=result_path, save_name=f"gender_{topic.lower()}")
        df_topic.to_csv(os.path.join(result_path, f"gender_{topic.lower()}.csv"), index=False)

    return


def experience(analysis_dict, save_path, df_earth_cluster=None):
    # save path
    result_path = os.path.join(save_path, "experience")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    ethics = analysis_dict["ethics_exp"]
    animals = analysis_dict["animal_exp"]
    ai = analysis_dict["ai_exp"]
    consciousness = analysis_dict["consciousness_exp"]

    ethics_counts = ethics[survey_mapping.Q_ETHICS_EXP].value_counts().sort_index().reset_index()
    ethics_counts.columns = ["rating", "count"]
    ethics_counts["proportion"] = (ethics_counts["count"] / ethics_counts["count"].sum()) * 100
    ethics_counts["experience"] = "ethics"
    animal_counts = animals[survey_mapping.Q_ANIMAL_EXP].value_counts().sort_index().reset_index()
    animal_counts.columns = ["rating", "count"]
    animal_counts["proportion"] = (animal_counts["count"] / animal_counts["count"].sum()) * 100
    animal_counts["experience"] = "animals"
    ai_counts = ai[survey_mapping.Q_AI_EXP].value_counts().sort_index().reset_index()
    ai_counts.columns = ["rating", "count"]
    ai_counts["proportion"] = (ai_counts["count"] / ai_counts["count"].sum()) * 100
    ai_counts["experience"] = "ai"
    consciousness_counts = consciousness[survey_mapping.Q_CONSC_EXP].value_counts().sort_index().reset_index()
    consciousness_counts.columns = ["rating", "count"]
    consciousness_counts["proportion"] = (consciousness_counts["count"] / consciousness_counts["count"].sum()) * 100
    consciousness_counts["experience"] = "consciousness"

    experience_counts = pd.concat([ethics_counts, animal_counts, ai_counts, consciousness_counts], ignore_index=True)
    experience_counts.to_csv(os.path.join(result_path, "experience_proportions.csv"), index=False)

    """
    Frequency of experience pie charts
    """
    experience_colors = {1: "#e63946",
                         2: "#f1faee",
                         3: "#a8dadc",
                         4: "#457b9d",
                         5: "#344968"}

    merged_experience_df = reduce(lambda left, right: pd.merge(left, right, on=[process_survey.COL_ID], how="outer"),
                                  [ethics, animals, ai, consciousness])
    relevant_cols = {survey_mapping.Q_ETHICS_EXP: "exp_ethics",
                     survey_mapping.Q_ANIMAL_EXP: "exp_animals",
                     survey_mapping.Q_AI_EXP: "exp_ai",
                     survey_mapping.Q_CONSC_EXP: "exp_consciousness"}

    relevant_cols_list = list(relevant_cols.values())

    merged_experience_df.rename(columns=relevant_cols, inplace=True)
    merged_experience_df.to_csv(os.path.join(result_path, f"merged_experience_df.csv"), index=False)

    # create a stats df
    stats_list = []
    for col in list(relevant_cols_list):
        col_data = merged_experience_df[col].dropna()
        value_cnts = col_data.value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
        df = pd.DataFrame({
            "experience": [col],
            "N": [len(col_data)],
            "M": [col_data.mean()],
            "SD": [col_data.std()],
            "SE": [col_data.std() / (len(col_data) ** 0.5)],
            "min": [col_data.min()],
            "max": [col_data.max()],
            "N_1": [value_cnts[1]],
            "N_2": [value_cnts[2]],
            "N_3": [value_cnts[3]],
            "N_4": [value_cnts[4]],
            "N_5": [value_cnts[5]],
        })
        stats_list.append(df)
    stats_df = pd.concat(stats_list, ignore_index=True)
    stats_df.to_csv(os.path.join(result_path, f"merged_experience_df_stats.csv"), index=False)


    """
    Plot
    """
    counts_df = merged_experience_df[relevant_cols_list].apply(lambda col: col.value_counts().sort_index()).astype(int)
    counts_df.columns = [f"{relevant_cols[col]}" for col in relevant_cols.keys()]
    counts_df = counts_df.reset_index(drop=False, inplace=False).rename(columns={"index": "rating"}, inplace=False)
    counts_df["rating"] = counts_df["rating"].astype(int)
    target_row = counts_df[counts_df["rating"] == 5].iloc[0]
    sorted_columns = sorted(list(relevant_cols.values()), key=lambda col: target_row[col], reverse=True)
    plotter.plot_stacked_proportion_bars_in_a_batch(df=counts_df, rating_col="rating", y_tick_rotation=90,
                                                    item_cols=sorted_columns,
                                                    annotate=True, annot_font_size=16,
                                                    annot_font_colors=["#24272E", "#2D3039", "#3F4350",
                                                                       "#3F4350", "#BABEC9"],
                                                    color_map=experience_colors, save_path=result_path,
                                                    save_name=f"experience_counts",
                                                    rating_label="", plot_title="Self-Reported Experience Level",
                                                    fmt="svg")

    """
    Can experience explain KPT behavior?
    """
    df_killing = analysis_dict["important_test_kill"].copy()
    df_killing = df_killing.loc[:, [process_survey.COL_ID] + list(survey_mapping.important_test_kill_tokens.keys())]
    df_killing.rename(columns=survey_mapping.important_test_kill_tokens, inplace=True)
    ans_map = {"No (will not kill to pass the test)": 0, "Yes (will kill to pass the test)": 1}
    kill_cols = list(survey_mapping.important_test_kill_tokens.values())
    df_killing[kill_cols] = df_killing[kill_cols].replace(ans_map)

    con_exp_relevant = merged_experience_df.loc[:, [process_survey.COL_ID,
                                                    survey_mapping.Q_CONSC_EXP,
                                                    survey_mapping.Q_AI_EXP,
                                                    survey_mapping.Q_ANIMAL_EXP,
                                                    survey_mapping.Q_ETHICS_EXP]]

    killing_with_exp = pd.merge(con_exp_relevant, df_killing, on=process_survey.COL_ID)
    killing_with_exp.rename(columns=relevant_cols, inplace=True)
    killing_with_exp.to_csv(os.path.join(result_path, "KPT_with_experience.csv"), index=False)

    categorical_cols = []
    order_cols = [relevant_cols[survey_mapping.Q_CONSC_EXP], relevant_cols[survey_mapping.Q_AI_EXP],
                  relevant_cols[survey_mapping.Q_ANIMAL_EXP], relevant_cols[survey_mapping.Q_ETHICS_EXP]]

    for col in list(survey_mapping.important_test_kill_tokens.values()):
        print(f" **************** {col} **************** ")
        df_col = killing_with_exp.loc[:, [process_survey.COL_ID, col] + order_cols]
        helper_funcs.run_random_forest_pipeline(dataframe=df_col, dep_col=col,
                                                categorical_cols=categorical_cols,
                                                order_cols=order_cols, save_path=result_path, save_prefix=f"KPT_{col}",
                                                rare_class_threshold=5, n_permutations=1000,
                                                scoring_method="accuracy",
                                                cv_folds=10)
    exit()


    """
    If we have cluster information: does experience explain the clusters?
    """

    if df_earth_cluster is not None:
        con_exp_relevant = merged_experience_df.loc[:, [process_survey.COL_ID,
                                                         survey_mapping.Q_CONSC_EXP,
                                                         survey_mapping.Q_AI_EXP,
                                                         survey_mapping.Q_ANIMAL_EXP,
                                                         survey_mapping.Q_ETHICS_EXP]]
        df_cluster_relevant = df_earth_cluster.loc[:, [process_survey.COL_ID, "Cluster"]]
        cluster_with_exp = pd.merge(con_exp_relevant, df_cluster_relevant, on=process_survey.COL_ID)
        cluster_with_exp.to_csv(os.path.join(result_path, "cluster_with_experience.csv"), index=False)

        cluster_with_exp.rename(columns=relevant_cols, inplace=True)

        categorical_cols = []
        order_cols = [relevant_cols[survey_mapping.Q_CONSC_EXP], relevant_cols[survey_mapping.Q_AI_EXP],
                      relevant_cols[survey_mapping.Q_ANIMAL_EXP], relevant_cols[survey_mapping.Q_ETHICS_EXP]]

        helper_funcs.run_random_forest_pipeline(dataframe=cluster_with_exp, dep_col="Cluster",
                                                categorical_cols=categorical_cols,
                                                order_cols=order_cols, save_path=result_path, save_prefix="",
                                                rare_class_threshold=5, n_permutations=1000,
                                                scoring_method="accuracy",
                                                cv_folds=10)
    return()



    """
    Does experience with animals affect answers related to animal C / MS ? 
    """

    animals = animals.apply(helper_funcs.replace_animal_other, axis=1)  # replace "Other" if categories do exist

    # first of all, general animal info
    stats = {survey_mapping.Q_ANIMAL_EXP: helper_funcs.compute_stats(animals[survey_mapping.Q_ANIMAL_EXP], possible_values=[1, 2, 3, 4])}
    plot_data = {}
    for item, (proportions, mean_rating, std_dev, n) in stats.items():
        plot_data[item] = {
            "Proportion": proportions,
            "Mean": mean_rating,
            "Std Dev": std_dev,
            "N": n,
        }
    sorted_plot_data = {key: plot_data[key] for key in list(dict(plot_data).keys()) if key in plot_data}.items()
    rating_labels = ["1 (None)", "2", "3", "4", "5 (Extremely)"]
    rating_color_list = ["#E7E7E7", "#B7CED0", "#87B4B9", "#569BA2", "#26818B"]
    topic_name = "experience with animals"
    plotter.plot_stacked_proportion_bars(plot_data=sorted_plot_data, num_plots=1, legend_labels=rating_labels,
                                         ytick_visible=False, title=f"{topic_name.title()}",
                                         colors=rating_color_list, num_ratings=5,
                                         save_path=result_path, save_name=f"{topic_name.lower()}_ratings")

    # If animal experience >= 3, which animals?
    which_animal = animals["Please specify which animals"].str.split(',').explode()
    animal_counts = which_animal.value_counts()

    animal_colors = {"Primates (including apes)": "#FFD380",
                     "Dogs": "#FF9122",
                     "Cats": "#FF7C43",
                     "Birds": "#FB675D",
                     "Rabbits": "#f95d6a",
                     "Livestock": "#F95D6A",
                     "Bears": "#E0547D",
                     "Rodents": "#D45087",
                     "Bats": "#BA518E",
                     "Reptiles": "#A05195",
                     "Snails": "#665191",
                     "Insects": "#544F8A",
                     "Frogs": "#414D83",
                     "Fish": "#2F4B7C",
                     "Cephalopods": "#18456C",
                     "Other marine life": "#003F5C",
                     "Other": "#00202E"
                     }

    animal_order = list(animal_colors.keys())
    animal_props = animal_counts.reset_index(drop=False, inplace=False)
    animal_props["proportion"] = (animal_props["count"] / animal_props["count"].sum()) * 100
    animal_props.to_csv(os.path.join(result_path, "exp_animal_types.csv"), index=False)

    # annotate only important ones
    threshold = 1.5
    substantial_list = animal_props.loc[animal_props["proportion"] > threshold, "Please specify which animals"].tolist()

    category_counts = [animal_counts[animal] if animal in animal_counts else 0 for animal in animal_order]

    plotter.plot_pie(categories_names=animal_order,
                     categories_counts=category_counts,  # [animal_counts[animal] for animal in animal_order]
                     categories_colors=animal_colors, title=f"Animal Experience (3+)", edge_color="none",
                     pie_direction=180, annot_groups=True, annot_group_selection=substantial_list, annot_props=False,
                     save_path=result_path, save_name="exp_animal_types", fmt="png")

    """
    Consciousness / Moral Status in Other Creatures, by experience
    """

    df_ms = analysis_dict["other_creatures_ms"].copy()
    df_c = analysis_dict["other_creatures_cons"].copy()
    df = pd.merge(df_c, df_ms, on=[process_survey.COL_ID])

    # experience columns
    animal_experience_df = analysis_dict["animal_exp"].loc[:,
                           [process_survey.COL_ID, survey_mapping.Q_ANIMAL_EXP]].rename(
        columns={survey_mapping.Q_ANIMAL_EXP: "exp_animals"}, inplace=False)
    ai_experience_df = analysis_dict["ai_exp"].loc[:, [process_survey.COL_ID, survey_mapping.Q_AI_EXP]].rename(
        columns={survey_mapping.Q_AI_EXP: "exp_ai"}, inplace=False)
    ethics_experience_df = analysis_dict["ethics_exp"].loc[:,
                           [process_survey.COL_ID, survey_mapping.Q_ETHICS_EXP]].rename(
        columns={survey_mapping.Q_ETHICS_EXP: "exp_ethics"}, inplace=False)
    con_experience_df = analysis_dict["consciousness_exp"].loc[:,
                        [process_survey.COL_ID, survey_mapping.Q_CONSC_EXP]].rename(
        columns={survey_mapping.Q_CONSC_EXP: "exp_consc"}, inplace=False)
    demos_df = analysis_dict["demographics"].loc[:, [process_survey.COL_ID, "How old are you?"]].rename(
        columns={"How old are you?": "age"}, inplace=False)

    """
    Cross people's ratings with their experience 
    """
    df_list = [df.copy(), animal_experience_df, ai_experience_df, ethics_experience_df, con_experience_df]
    df_with_experience = reduce(lambda left, right: pd.merge(left, right, on=[process_survey.COL_ID]), df_list)

    """
    Experience with AI: 
    Take people's self-reports about their experience with ai (exp_ai). 
    Then, look only at the LLM item of the other creatures (out of the 24 creatures, what did people say about 
    LLM being conscious, and about LLM having moral status). 
    Break it down by experience with AI. 
    """

    ai_exp_col = "exp_ai"
    c_llm_col = "c_A large language model"
    ms_llm_col = "ms_A large language model"

    df = df_with_experience.loc[:, ["response_id", c_llm_col, ms_llm_col, ai_exp_col]]

    rating_color_list = ["#DB5461", "#fb9a99", "#70a0a4", "#26818B"]
    c_rating_labels = ["Does not have consciousness", "Probably doesn't have consciousness",
                       "Probably has consciousness", "Has consciousness"]
    experience_levels = [1, 2, 3, 4, 5]
    stats = {}
    for experience in experience_levels:
        df_experience = df[df[ai_exp_col] == experience]
        # CONSCIOUSNESS
        stats[experience] = helper_funcs.compute_stats(df_experience[c_llm_col], possible_values=[1, 2, 3, 4])
    plot_data = {}
    for item, (proportions, mean_rating, std_dev, n) in stats.items():
        plot_data[item] = {
            "Proportion": proportions,
            "Mean": mean_rating,
            "Std Dev": std_dev,
            "N": n
        }
    # Sort the data by the MEAN rating (Python 3.7+ dictionaries maintain the insertion order of keys)
    sorted_plot_data = sorted(plot_data.items(), key=lambda x: x[1]["Mean"], reverse=True)
    plotter.plot_stacked_proportion_bars(plot_data=sorted_plot_data, num_plots=len(experience_levels),
                                         legend_labels=c_rating_labels, y_title="Experience with AI",
                                         colors=rating_color_list, num_ratings=4, default_ticks=False,
                                         title=f"Indicate whether A large-language model has consciousness",
                                         save_path=result_path,
                                         save_name=f"LLM_conscious_by_experience_with_AI",
                                         text_width=39, relative=False, fmt="svg")

    ma_rating_labels = ["Does not have moral status", "Probably doesn't have moral status",
                        "Probably has moral status", "Has moral status"]
    stats = {}
    for experience in experience_levels:
        df_experience = df[df[ai_exp_col] == experience]
        # CONSCIOUSNESS
        stats[experience] = helper_funcs.compute_stats(df_experience[ms_llm_col], possible_values=[1, 2, 3, 4])
    plot_data = {}
    for item, (proportions, mean_rating, std_dev, n) in stats.items():
        plot_data[item] = {
            "Proportion": proportions,
            "Mean": mean_rating,
            "Std Dev": std_dev,
            "N": n
        }
    # Sort the data by the MEAN rating (Python 3.7+ dictionaries maintain the insertion order of keys)
    sorted_plot_data = sorted(plot_data.items(), key=lambda x: x[1]["Mean"], reverse=True)
    plotter.plot_stacked_proportion_bars(plot_data=sorted_plot_data, num_plots=len(experience_levels),
                                         legend_labels=ma_rating_labels, y_title="Experience with AI",
                                         colors=rating_color_list, num_ratings=4, default_ticks=False,
                                         title=f"Indicate whether A large-language model has moral status",
                                         save_path=result_path,
                                         save_name=f"LLM_ms_by_experience_with_AI",
                                         text_width=39, relative=False, fmt="svg")

    # correlation between LLM consciousness and LLM moral status
    plotter.plot_scatter_xy(df=df, identity_col="response_id", x_col=c_llm_col, x_label="LLM Consciousness",
                            x_min=1, x_max=4, x_ticks=1,
                            y_col=ms_llm_col, y_label="LLM Moral Status", y_min=1, y_max=4, y_ticks=1,
                            save_path=result_path, save_name=f"LLM_c_by_ms", annotate_id=False, title_text="",
                            fmt="svg", size=400, corr_line=True, diag_line=True)
    return


def analyze_survey(sub_df, analysis_dict, save_path, load=True):
    """
    The method which manages all the processing of specific survey data for analyses.
    :param sub_df: the dataframe of all participants' responses
    :param analysis_dict: dictionary where key=topic, value=a dataframe containing all the columns relevant for this
    topic/section
    :param save_path: where the results will be saved (csvs, plots)
    :param load: for stuff that takes a ton of time to run every time
    """

    if load:  # load the earth-in-danger things
        df_earth_cluster = pd.read_csv(os.path.join(save_path, "earth_danger", f"earth_danger_clusters.csv"))

    else:
        df_earth_cluster = earth_in_danger(analysis_dict, save_path)


    graded_consciousness(analysis_dict, save_path, df_earth_cluster=df_earth_cluster, remove_contradicting=True)
    exit()
    ms_features_order_df, feature_colors = moral_consideration_features(analysis_dict=analysis_dict,
                                                                        save_path=save_path,
                                                                        df_earth_cluster=df_earth_cluster)

    experience(analysis_dict, save_path, df_earth_cluster=df_earth_cluster)

    kill_for_test(analysis_dict=analysis_dict, save_path=save_path, df_earth_cluster=df_earth_cluster)

    ics(analysis_dict=analysis_dict, save_path=save_path, df_earth_cluster=df_earth_cluster,
        ms_features_order_df=None, feature_colors=None)

    zombie_pill(analysis_dict, save_path, feature_order_df=None, feature_color_map=None)

    moral_considreation_prios(analysis_dict=analysis_dict, save_path=save_path, df_earth_cluster=df_earth_cluster)

    consciousness_intelligence(analysis_dict, save_path)
    demographics(analysis_dict, save_path, df_earth_cluster=df_earth_cluster)






    other_creatures(analysis_dict=analysis_dict, save_path=save_path, sort_together=False,
                    df_earth_cluster=df_earth_cluster)



    age_check(analysis_dict=analysis_dict, save_path=save_path, df_earth_cluster=df_earth_cluster)

    expert_check(analysis_dict=analysis_dict, save_path=save_path, df_earth_cluster=df_earth_cluster)
    gender_check(analysis_dict=analysis_dict, save_path=save_path, df_earth_cluster=df_earth_cluster)

    religious_check(analysis_dict=analysis_dict, save_path=save_path, df_earth_cluster=df_earth_cluster)


    gender_cross(analysis_dict, save_path)  # move to after the individuals

    return
