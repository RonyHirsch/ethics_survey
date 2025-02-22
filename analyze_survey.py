import os
import pandas as pd
import numpy as np
import re
from functools import reduce
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
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
                  "Yes": "#355070",
                  "No": "#B26972",  # #590004, #461D02
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


def other_creatures(analysis_dict, save_path, sort_together=True, df_earth_cluster=None):
    # save path
    result_path = os.path.join(save_path, "c_v_ms")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # load relevant data
    df_ms = analysis_dict["other_creatures_ms"].copy()
    df_c = analysis_dict["other_creatures_cons"].copy()
    df = pd.merge(df_c, df_ms, on=[process_survey.COL_ID])

    df.to_csv(os.path.join(result_path, "c_v_ms.csv"), index=False)

    # codes and relevant stuff
    items = survey_mapping.other_creatures_general  # all rated items
    topic_name_map = {"c": "Consciousness", "ms": "Moral Status"}

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
    Relationship between C ratings and MS ratings
    """

    # melt the df to a long format
    long_data = pd.melt(df, id_vars=[process_survey.COL_ID], var_name="Item_Topic", value_name="Rating")
    long_data[["Topic", "Item"]] = long_data["Item_Topic"].str.split('_', expand=True)
    long_data = long_data.drop(columns=["Item_Topic"])
    long_data["Topic"] = long_data["Topic"].map(topic_name_map)

    # stats: average rating (of C & MS) per item (averaged across all respondents)
    long_data_noid = long_data.drop(process_survey.COL_ID, axis=1, inplace=False)
    long_data_mean_rating = long_data_noid.groupby(["Topic", "Item"]).mean().reset_index(drop=False)
    long_data_mean_rating = long_data_mean_rating.pivot(index="Item", columns="Topic", values="Rating").reset_index(
        drop=False)
    # I'll save long_data_mean_rating below after I add some stuff to it
    long_data_mean_rating_stats = long_data_mean_rating.describe()
    long_data_mean_rating_stats.to_csv(os.path.join(result_path, f"c_v_ms_avg_per_item_stats.csv"),
                                       index=True)  # in 'describe' the index is the desc name

    dataframes = [long_data.copy(), animal_experience_df, ai_experience_df, ethics_experience_df, con_experience_df,
                  demos_df]
    result_df = reduce(lambda left, right: pd.merge(left, right, on=[process_survey.COL_ID]), dataframes)
    result_df["non_human_animal"] = result_df["Item"].map(survey_mapping.other_creatures_isNonHumanAnimal)
    result_df.to_csv(os.path.join(result_path, "c_v_ms_long.csv"), index=False)

    # turn categorical columns into numeric ones for linear modelling
    for col in ["Topic", "Item"]:
        label_encoder = LabelEncoder()
        result_df[col] = label_encoder.fit_transform(result_df[col])

    result_df.to_csv(os.path.join(result_path, "c_v_ms_long_coded.csv"), index=False)

    """
    Plot at the ratings individually for each item 
    """
    rating_color_list = ["#DB5461", "#fb9a99", "#70a0a4", "#26818B"]
    rating_labels = ["Does Not Have", "Probably Doesn't Have", "Probably Has", "Has"]
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
    colors = [rating_color_list[0], rating_color_list[-1]]

    data = long_data.pivot(index=["response_id", "Item"], columns=["Topic"], values="Rating").reset_index(drop=False, inplace=False)

    # order the items
    count_df = data.groupby("Item").apply(
        lambda x: pd.Series({
            "count_4": ((x["Consciousness"] == 4) & (x["Moral Status"] == 4)).sum(),
            "count_3": ((x["Consciousness"] == 3) & (x["Moral Status"] == 3)).sum(),
            "count_2": ((x["Consciousness"] == 2) & (x["Moral Status"] == 2)).sum(),
            "count_1": ((x["Consciousness"] == 1) & (x["Moral Status"] == 1)).sum()
        })).reset_index()

    sorted_items = count_df.sort_values(by=["count_4", "count_3", "count_2", "count_1"],
                                        ascending=[True, True, True, True])["Item"].tolist()

    plotter.plot_multiple_scatter_xy(data=data, identity_col="response_id", x_col="Consciousness",
                                     y_col="Moral Status", x_label="Consciousness", y_label="Moral Status",
                                     x_min=1, x_max=4.2, x_ticks=1, y_min=1, y_max=4.2, y_ticks=1,
                                     save_path=result_path, save_name="correlation_c_ms_panels",
                                     palette_bounds=colors, annotate_id=False,
                                     fmt="svg", size=50, alpha=0.6, corr_line=True, diag_line=True,
                                     vertical_jitter=0.25, horizontal_jitter=0.25,
                                     panel_per_col="Item", panel_order=sorted_items, rows=4, cols=6,
                                     title_size=20, axis_size=14,hide_axes_names=True)


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


    """
    Plot "other creatures" judgments of Consciousness vs. of Moral Status. >> *** ACROSS ITEMS ***
    """
    # prepare data for analyses
    df_pivot = long_data.pivot_table(index="Item", columns="Topic", values="Rating", aggfunc="mean").reset_index(
        drop=False, inplace=False)  # I don't want to 'fillna(0).' this

    # create a plotting version of df-pivot, deleting a prefix of a/an
    df_pivot_plotting = df_pivot.copy()
    df_pivot_plotting["Item"] = df_pivot_plotting["Item"].str.replace(r'^(A|An)\s+', '', regex=True).str.capitalize()

    # collapsed across everyone, no individuation, diagonal line
    plotter.plot_scatter_xy(df=df_pivot_plotting, identity_col="Item",
                            x_col="Consciousness", x_label="Consciousness", x_min=1, x_max=4, x_ticks=1,
                            y_col="Moral Status", y_label="Moral Status", y_min=1, y_max=4, y_ticks=1,
                            save_path=result_path, save_name=f"correlation_c_ms", annotate_id=True,
                            palette_bounds=colors,  # use the same colors as the individual-item correlations
                            corr_line=False, diag_line=True, fmt="svg",
                            individual_df=None, id_col=None)

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
                         save_path=result_path, save_name=f"{'_'.join(counts.index.tolist())}", format="png")

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
    df_pivot, kmeans = helper_funcs.perform_kmeans(df_pivot=df_earth_coded, clusters=cluster_num,
                                                   save_path=result_path, save_name="items")

    """
    Plot the KMeans cluster centroids: this is important. 
    For each cluster (we have cluster_num clusters total), the centroid is the average data point for this cluster 
    (the mean value of the features for all data points in the cluster). We use the centroids to visualize each cluster's
    choice in each earth-is-in-danger dyad, to interpret the differences between them.  
    """

    # Compute the cluster centroids and SEMs
    cluster_centroids = df_pivot.groupby("Cluster").mean()
    cluster_sems = df_pivot.groupby("Cluster").sem()

    # Plot per cluster
    helper_funcs.plot_cluster_centroids(cluster_centroids=cluster_centroids, cluster_sems=cluster_sems,
                                        save_path=result_path, save_name="items",
                                        label_map=survey_mapping.EARTH_DANGER_QA_MAP, binary=True,
                                        threshold=0, overlaid=False)

    # Plot - collapsed (all clusters together)
    helper_funcs.plot_cluster_centroids(cluster_centroids=cluster_centroids, cluster_sems=cluster_sems,
                                        save_path=result_path, save_name="items", fmt="svg",
                                        label_map=survey_mapping.EARTH_DANGER_QA_MAP, binary=True,
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
    questions = [c for c in df_ics.columns if c.startswith("Do you think a creature/system")]
    ans_map = {"No": 0, "Yes": 1}
    # plot a collapsed figure where each creature is a bar, with the proportion of how many would kill it
    stats = dict()
    labels = list()
    for q in questions:
        df_q = df_ics.loc[:, [process_survey.COL_ID, q]]
        q_name = q.replace("Do you think a creature/system", "")[:-1]
        q_name = q_name.replace("can be", "")
        q_name = q_name.replace("can have", "")
        q_name = q_name.replace("/", "-")
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
    rating_color_list = ["#B26972", "#355070"]
    sorted_plot_data = sorted(plot_data.items(), key=lambda x: x[1]["Mean"], reverse=True)
    plotter.plot_stacked_proportion_bars(plot_data=sorted_plot_data, num_plots=4, legend_labels=rating_labels,
                                         ytick_visible=True, text_width=39,
                                         title=f"Do you think A creature/system can be",
                                         show_mean=False, sem_line=False,
                                         colors=rating_color_list, num_ratings=2,
                                         save_path=save_path, save_name=f"{prefix}ics{suffix}")
    # save data
    plot_df = pd.DataFrame(plot_data)
    plot_df.to_csv(os.path.join(save_path, f"{prefix}ics{suffix}.csv"), index=True)
    return plot_df


def ics(analysis_dict, save_path, df_earth_cluster=None):
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

    plot_df = calculate_ics_proportions(df_ics=df_ics, save_path=result_path, suffix="_all")

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
        ics_q.to_csv(os.path.join(result_path, f"{col_savename}.csv"), index=False)  # save answers for examination

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
    rating_color_list = ["#B26972", "#355070"]
    sorted_plot_data = sorted(plot_data.items(), key=lambda x: x[1]["Mean"], reverse=True)
    plotter.plot_stacked_proportion_bars(plot_data=sorted_plot_data, num_plots=6, legend_labels=rating_labels,
                                         ytick_visible=True, text_width=39, title=f"", show_mean=False, sem_line=False,
                                         colors=rating_color_list, num_ratings=2,
                                         save_path=save_path, save_name=f"{prefix}kill_to_pass{suffix}")
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
                                         colors=rating_color_list, num_ratings=2, save_path=save_path,
                                         save_name=f"{prefix}kill_to_pass_allYesNoDiscount{suffix}", split=True,
                                         yes_all_proportion=yes_all_proportion, no_all_proportion=no_all_proportion)
    return


def kill_for_test(analysis_dict, save_path, df_earth_cluster):
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
    two_features = [survey_mapping.Q_CONSCIOUSNESS_SENSATIONS, survey_mapping.Q_SENSATIONS_INTENTIONS,
                    survey_mapping.Q_VULCAN]
    df_test_binary = df_test.replace(survey_mapping.ANS_KILLING_MAP, inplace=False)  # convert columns
    # calculate the average 'yes' responses for each person for 1-feature and 2-feature creatures
    df_test_binary["kill_one_avg"] = df_test_binary[one_feature].mean(axis=1)
    df_test_binary["kill_two_avg"] = df_test_binary[two_features].mean(axis=1)
    df_test_binary.to_csv(os.path.join(result_path, f"kill_to_pass_coded.csv"), index=False)
    # paired t-test
    paired_ttest = helper_funcs.dependent_samples_ttest(list_group1=df_test_binary["kill_one_avg"].tolist(),
                                                        list_group2=df_test_binary["kill_two_avg"].tolist())
    paired_ttest.to_csv(os.path.join(result_path, f"kill_oneVtwofeatures_ttest.csv"), index=False)

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

    cat_names = ["Won't kill any", "Kill at least one", "Kill all entities"]
    cat_counts = [all_nos_prop, rest_prop, all_yes_prop]
    cat_colors = {"Won't kill any": "#033860", "Kill at least one": "#C2948A", "Kill all entities": "#723D46"}
    plotter.plot_pie(categories_names=cat_names, categories_counts=cat_counts,
                     categories_colors=cat_colors, title=f"Would kill in any of the scenarios",
                     save_path=result_path, save_name="all_yes_no", format="png")

    # why?
    colors = {survey_mapping.ANS_ALLNOS_IMMORAL: "#E7A391",
              survey_mapping.ANS_ALLNOS_KILL: "#E6898B",
              survey_mapping.ANS_ALLNOS_INTERESTS: "#BA7880",
              f"{survey_mapping.ANS_ALLNOS_IMMORAL},{survey_mapping.ANS_ALLNOS_KILL}": "#93032E",
              f"{survey_mapping.ANS_ALLNOS_IMMORAL},{survey_mapping.ANS_ALLNOS_INTERESTS}": "#FB3772",
              f"{survey_mapping.ANS_ALLNOS_INTERESTS},{survey_mapping.ANS_ALLNOS_KILL}": "#FC739C",
              f"{survey_mapping.ANS_ALLNOS_IMMORAL},{survey_mapping.ANS_ALLNOS_INTERESTS},{survey_mapping.ANS_ALLNOS_KILL}": "#FEC3D5",
              f"{survey_mapping.ANS_ALLNOS_IMMORAL},{survey_mapping.ANS_OTHER}": "#E3E7AF",
              f"{survey_mapping.ANS_ALLNOS_IMMORAL},{survey_mapping.ANS_ALLNOS_INTERESTS},{survey_mapping.ANS_ALLNOS_KILL},{survey_mapping.ANS_OTHER}": "#6C6173",
              f"{survey_mapping.ANS_ALLNOS_IMMORAL},{survey_mapping.ANS_ALLNOS_KILL},{survey_mapping.ANS_OTHER}": "#775144",
              f"{survey_mapping.ANS_ALLNOS_IMMORAL},{survey_mapping.ANS_ALLNOS_INTERESTS},{survey_mapping.ANS_OTHER}": "#775144",
              f"{survey_mapping.ANS_OTHER}": "#05B3B3"}

    # flatten the selections
    all_selections = all_nos["You wouldn't eliminate any of the creatures; why?"].str.split(',').explode()
    category_counts = all_selections.value_counts()
    plotter.plot_pie(categories_names=category_counts.index.tolist(), categories_counts=category_counts.tolist(),
                     categories_colors=colors, title=f"You wouldn't eliminate any of the creatures; why?",
                     save_path=result_path, save_name="all_nos_reason", format="png")

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

    c = 3  # TODO: DELETE THIS

    """
    Follow up: Separate the killing choices based on people who even think it's possible to have one feature without
    the others ('Yes' in the ics cluster of questions), and those who do not ('No') and see if they differ.
    """
    df_ics = analysis_dict["ics"].copy()
    # key = ics question (is it possible); value = kill question (would you kill it)
    ics_v_kill = {"Do you think a creature/system can have intentions/goals without being conscious?":
                      [
                          "A creature/system that only has plans/goals and intentions (can plan to perform certain actions in the future), but is not conscious (not experiencing) and cannot feel positive/negative sensations (pleasure/pain)",
                          "A creature/system that can feel positive/negative sensations (pleasure/pain) and also has plans/goals and intentions (can plan to perform certain actions in the future), but is not conscious (not experiencing)"],
                  "Do you think a creature/system can be conscious without having intentions/goals?":
                      [
                          "A creature/system that does not have plans/goals or intentions, and cannot feel positive/negative sensations (pleasure/pain), but is conscious (for example, sees colors, but does not feel anything negative or positive, and cannot plan)",
                          "A creature/system that is both conscious (has experiences) and can feel positive/negative sensations (pleasure/pain), but does not have plans/goals or intentions (for example, can't plan to avoid something that causes pain)"],
                  "Do you think a creature/system can have positive or negative sensations (pleasure/pain) without being conscious?":
                      [
                          "A creature/system that can only feel positive/negative sensations (pleasure/pain), but is not conscious (not experiencing) and does not have plans/goals or intentions (for example, can't plan to avoid something that causes pain)"],
                  "Do you think a creature/system can be conscious without having positive or negative sensations (pleasure/pain)?":
                      [
                          "A creature/system that is both conscious (has experiences) and has plans/goals and intentions (can plan to perform certain actions in the future), but cannot feel positive/negative sensations (pleasure/pain)"]}

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

    # plot
    ans_map = {"No": 0, "Yes": 1}
    rating_labels = [survey_mapping.ANS_NO, survey_mapping.ANS_YES]
    df_q_map = df_zombie.replace({"Would you take the pill?": ans_map})
    stats = helper_funcs.compute_stats(df_q_map["Would you take the pill?"],
                                       possible_values=df_q_map["Would you take the pill?"].unique().tolist())
    # Create DataFrame for plotting
    plot_data = {"Would you take the pill?": {
        "Proportion": stats[0],
        "Mean": stats[1],
        "Std Dev": stats[2],
        "N": stats[3]
    }}
    rating_color_list = ["#B26972", "#355070"]
    sorted_plot_data = sorted(plot_data.items(), key=lambda x: x[1]["Mean"], reverse=True)
    plotter.plot_stacked_proportion_bars(plot_data=sorted_plot_data, num_plots=1, legend_labels=rating_labels,
                                         ytick_visible=True, text_width=39, title=f"", show_mean=False, sem_line=False,
                                         colors=rating_color_list, num_ratings=2, inches_w=18, inches_h=8,
                                         save_path=result_path, save_name=f"take_the_pill")
    zombie_data = pd.DataFrame(plot_data)
    zombie_data.to_csv(os.path.join(result_path, f"take_the_pill.csv"), index=True)  # index is descriptives' names

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
    ms_features_copy = ms_features_df[
        [process_survey.COL_ID, "What do you think is important for moral considerations?"]]

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
    most_important = ms_features_df[
        [process_survey.COL_ID, "Which do you think is the most important for moral considerations?"]]
    """
    what the below means is counting the proportions of selecting a single feature. Note that these are amts, 
    and we do not treat within-subject things here. 
    """
    # proportions_one  = how many of all people selected this feature as the most important one
    proportions_one = (
            most_important["Which do you think is the most important for moral considerations?"].value_counts(
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
                                              y_max=101,
                                              y_skip=10, inch_w=20, inch_h=12, order=None)
    else:
        plotter.plot_categorical_bars_layered(categories_prop_df=df_unified, category_col="index",
                                              full_data_col="Proportion_all", partial_data_col="Proportion_one",
                                              categories_colors=feature_color_dict, save_path=result_path,
                                              save_name=f"{save_prefix}important_features", fmt="svg", y_min=0,
                                              y_max=101,
                                              y_skip=10, inch_w=20, inch_h=12, order=feature_order_df["index"].tolist())
    # we're getting back the final mapping between features and colors in case we want to re-use it for comparison
    return df_unified, feature_color_dict


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

    ms_features_order_df, feature_colors = calculate_moral_consideration_features(ms_features_df=ms_features,
                                                                                  result_path=result_path,
                                                                                  save_prefix="all_",
                                                                                  feature_order_df=None,
                                                                                  feature_color_dict=None)

    """
    Break it down by population: one of the items is ANS_PHENOMENOLOGY, which == consciousness only if you have 
    experience with consciousness science and philosophy. Was this item more important for these people than the rest, 
    who do not necessarily equate or define consciousness this way?
    """

    # prepare the data
    experience_df_copy = analysis_dict["consciousness_exp"].copy()
    demographics_df_copy = analysis_dict["demographics"].copy()
    demographics_df_copy = demographics_df_copy[
        [process_survey.COL_ID, "What is your education background?", "In what topic?"]]
    df_list = [ms_features, experience_df_copy, demographics_df_copy]

    # merge the three dfs
    ms_features_experience = reduce(lambda left, right: pd.merge(left, right, on=[process_survey.COL_ID]), df_list)

    # self-reported consciousness experts (rated 3 and up)
    cons_experts = ms_features_experience[ms_features_experience[survey_mapping.Q_CONSC_EXP] > 2].reset_index(drop=True,
                                                                                                              inplace=False)

    # experts: only academics (experts & have higher education)
    # "contains" as they could have made SEVERAL choices
    cons_experts_academics = cons_experts[
        (cons_experts["What is your education background?"] == survey_mapping.EDU_POSTSEC)
        | (cons_experts["What is your education background?"] == survey_mapping.EDU_GRAD)].reset_index(drop=True,
                                                                                                       inplace=False)

    # experts who are NOT academics
    cons_experts_nonAcademics = cons_experts[
        ~cons_experts[process_survey.COL_ID].isin(cons_experts_academics[process_survey.COL_ID])]

    """
    Calculate proportions - based on expertise
    """
    # [1] academics who marked that their experience with consciousness is specifically derived from *STUDYING* consciousness
    # Clean up the column (strip whitespace and ensure string type)
    cons_experts_academics["Please specify your experience with this topic"] = cons_experts_academics[
        "Please specify your experience with this topic"].astype(str).str.strip()
    cons_experts_academics_expAcademia = cons_experts_academics[
        cons_experts_academics["Please specify your experience with this topic"].str.contains(
            re.escape(survey_mapping.ANS_C_ACADEMIA), na=False, case=False)]
    cons_experts_academics_expAcademia_props, c = calculate_moral_consideration_features(
        ms_features_df=cons_experts_academics_expAcademia,
        result_path=result_path,
        save_prefix="c-experts_expAcademia_",
        feature_order_df=ms_features_order_df,  # have the order YOKED to the original one
        feature_color_dict=feature_colors)

    # these are [self-porcalimed experts] & [have higher education] & [did NOT say their experience is from studying consciousness]
    cons_experts_academics_rest = cons_experts_academics[
        ~cons_experts_academics[process_survey.COL_ID].isin(cons_experts_academics_expAcademia[process_survey.COL_ID])]

    # [2] experts who are not from academia - either academics whose C experience is from other things, OR non-academics
    cons_experts_expNonAcademia = pd.concat([cons_experts_nonAcademics, cons_experts_academics_rest],
                                            ignore_index=True).drop_duplicates(keep=False)
    cons_experts_expNonAcademia_props, c = calculate_moral_consideration_features(
        ms_features_df=cons_experts_expNonAcademia,
        result_path=result_path,
        save_prefix="c-experts_expNonAcademia_",
        feature_order_df=ms_features_order_df,
        feature_color_dict=feature_colors)
    # [3] people who rated 1/2 (not experts)
    cons_nonExperts = ms_features_experience[ms_features_experience[survey_mapping.Q_CONSC_EXP] < 3].reset_index(
        drop=True, inplace=False)
    cons_nonExperts_props, c = calculate_moral_consideration_features(ms_features_df=cons_nonExperts,
                                                                      result_path=result_path,
                                                                      save_prefix="c-nonExperts_",
                                                                      feature_order_df=ms_features_order_df,
                                                                      feature_color_dict=feature_colors)

    # [4] people who ARE experts (3/4/5), have NO higher education [highschool at best], but their expertise DOES stem
    # from academic background (studied/teach these topics) somehow..
    cons_experts_nonAcademics_expYesAcademia = cons_experts_nonAcademics[
        cons_experts_nonAcademics["Please specify your experience with this topic"].str.contains(
            re.escape(survey_mapping.ANS_C_ACADEMIA), na=False, case=False)]

    # either: academic experts whose expertise is not from academia [2], non-experts [3], non-academic experts whose experience IS from academia [4]
    rest = pd.concat([cons_experts_expNonAcademia, cons_nonExperts, cons_experts_nonAcademics_expYesAcademia],
                     ignore_index=True)
    rest_props, c = calculate_moral_consideration_features(ms_features_df=rest, result_path=result_path,
                                                           save_prefix="c-not[exps_academic_expFromAcademia]_",
                                                           feature_order_df=ms_features_order_df,
                                                           feature_color_dict=feature_colors)
    """
    Two Proportion Z Test
    When we want to see if proportions of categories in two groups significantly differ from each other, we use the
    two-proportion z-test. The null hypothesis would be that the proportion of category selection for each item is the
    same between the two groups. 
    """

    # experts from academia vs. NON experts at all
    expsAcedemic_v_nonExps = helper_funcs.two_proportion_ztest(col_items="index", col_prop="Proportion_all", col_n="N",
                                                               group1="experts-academic",
                                                               df1=cons_experts_academics_expAcademia_props,
                                                               group2="non-experts",
                                                               df2=cons_nonExperts_props)
    expsAcedemic_v_nonExps.to_csv(os.path.join(result_path, f"z_test_expsAcademic_nonExps.csv"), index=False)

    # experts from academia vs. OTHER EXPERTS whose expertise is NOT from academia
    expsAcedemic_v_expNonAcademic = helper_funcs.two_proportion_ztest(col_items="index", col_prop="Proportion_all",
                                                                      col_n="N",
                                                                      group1="experts-academic",
                                                                      df1=cons_experts_academics_expAcademia_props,
                                                                      group2="experts-nonAcademia",
                                                                      df2=cons_experts_expNonAcademia_props)
    expsAcedemic_v_expNonAcademic.to_csv(os.path.join(result_path, f"z_test_expsAcademic_expNonAcademia.csv"),
                                         index=False)

    """
    Relationship between Earth-in-danger clusters and moral consideration features
    """
    if df_earth_cluster is not None:  # we have an actual df
        clusters = sorted(df_earth_cluster["Cluster"].unique().tolist())  # list all the possible clusters
        cluster_props = {clusters[i]: pd.DataFrame() for i in range(len(clusters))}
        for cluster in clusters:
            df_cluster = df_earth_cluster[df_earth_cluster["Cluster"] == cluster]  # subset
            cluster_subs = df_cluster[process_survey.COL_ID].tolist()
            ms_features_cluster = ms_features[ms_features[process_survey.COL_ID].isin(cluster_subs)].reset_index(
                drop=True, inplace=False)
            cluster_order_df, cluster_colors = calculate_moral_consideration_features(
                ms_features_df=ms_features_cluster,
                result_path=result_path,
                save_prefix=f"earthInDanger_cluster{cluster}_",
                feature_order_df=ms_features_order_df,
                feature_color_dict=feature_colors)
            cluster_props[cluster] = cluster_order_df

        # test statistical difference between pairs of clusters
        for cluster_pair in combinations(clusters, 2):  # for each pair of clusters
            cluster_comp = helper_funcs.two_proportion_ztest(col_items="index", col_prop="Proportion_all",
                                                             col_n="N",
                                                             group1=f"cluster-{cluster_pair[0]}",
                                                             df1=cluster_props[cluster_pair[0]],
                                                             group2=f"cluster-{cluster_pair[1]}",
                                                             df2=cluster_props[cluster_pair[1]])
            cluster_comp.to_csv(os.path.join(result_path,
                                             f"z_test_earthInDanger_cluster{cluster_pair[0]}_cluster{cluster_pair[1]}.csv"),
                                index=False)

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
                         save_path=result_path, save_name=f"{q.replace('?', '').replace('/', '-')}", format="png")
        category_counts.to_csv(os.path.join(result_path, f"{q.replace('?', '').replace('/', '-')}.csv"), index=False)

    reasons = [c for c in ms_prios.columns if c not in questions]
    for r in reasons:
        df_r = ms_prios.loc[:, [process_survey.COL_ID, r]]
        df_r = df_r[df_r[r].notnull()]
        df_r.to_csv(os.path.join(result_path, f"{r.replace('?', '').replace('/', '-')}.csv"), index=False)

    """
    Relations to graded consciousness: How many people who think some people should have higher MS than others think
    that consciousness is a graded phenomenon?
    """
    c_graded = analysis_dict["consciousness_graded"].copy()
    q_mc_human_prio = "Do you think some people should have a higher moral status than others?"
    q_mc_human_prio_why = "What characterizes people with higher moral status?"
    ms_prios_relevant = ms_prios.loc[:, [process_survey.COL_ID, q_mc_human_prio, q_mc_human_prio_why]]
    combined = pd.merge(ms_prios_relevant, c_graded, on=process_survey.COL_ID)
    combined.to_csv(os.path.join(result_path, "moral_decisions_prios_graded_c.csv"), index=False)

    """
    Mann-Whitney U-test on the ratings of agreement with C-GRADED questions (ordinal) based whether they 
    agree with 'q_mc_human_prio'
    """
    q_mc_human_prio_yes = combined[combined[q_mc_human_prio] == "Yes"].reset_index(inplace=False, drop=True)
    q_mc_human_prio_no = combined[combined[q_mc_human_prio] == "No"].reset_index(inplace=False, drop=True)

    rating_questions = [survey_mapping.Q_GRADED_EQUAL, survey_mapping.Q_GRADED_UNEQUAL, survey_mapping.Q_GRADED_INCOMP]

    result_list = list()
    for col in rating_questions:
        mwu = helper_funcs.mann_whitney_utest(list_group1=q_mc_human_prio_yes.loc[:, col].tolist(),
                                              list_group2=q_mc_human_prio_no.loc[:, col].tolist())
        # add descriptives: yes (some humans' MS > than others)
        mwu["group 1"] = ["q_mc_human_prio_yes"]
        mwu["group 1 M"] = q_mc_human_prio_yes.loc[:, col].mean()
        mwu["group 1 SD"] = q_mc_human_prio_yes.loc[:, col].std()
        # descriptives: no (some humans' MS is not > than others)
        mwu["group 2"] = ["q_mc_human_prio_no"]
        mwu["group 2 M"] = q_mc_human_prio_no.loc[:, col].mean()
        mwu["group 2 SD"] = q_mc_human_prio_no.loc[:, col].std()
        # dependent var & save
        mwu["dependent_var"] = [f"{col}"]
        # df to list
        result_list.append(mwu)

    result_df = pd.concat(result_list, ignore_index=True)
    result_df.to_csv(os.path.join(result_path, "stats_humans unequal_c graded qs.csv"), index=False)

    """
    Plot the graded consciousness q's separately for each group
    """
    human_prios = {"yes": q_mc_human_prio_yes, "no": q_mc_human_prio_no}
    for human_prio in human_prios.keys():
        df = human_prios[human_prio]
        plot_graded_consciousness_given_df(df=df, save_path=result_path, prefix="", suffix=f"_{human_prio}")

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
                                         save_path=save_path,
                                         save_name=f"{prefix}consciousness_graded_ratings{suffix}",
                                         text_width=39)
    # save the figure data
    df_result = pd.DataFrame(sorted_plot_data)
    df_result.to_csv(os.path.join(save_path, f"{prefix}consciousness_graded_ratings{suffix}.csv"))
    return


def graded_consciousness(analysis_dict, save_path):
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

    # contradicting themselves
    c_graded_contradiction = c_graded[
        (c_graded[survey_mapping.Q_GRADED_EQUAL] >= 3) & (c_graded[survey_mapping.Q_GRADED_UNEQUAL] >= 3)]

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

    return


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
                     save_path=result_path, save_name=f"{question.replace('?', '').replace('/', '-')}", format="png")
    category_counts.to_csv(os.path.join(result_path, f"{question.replace('?', '').replace('/', '-')}.csv"))

    follow_up = "How?"
    con_intellect_how = con_intellect[con_intellect[follow_up].notnull()]
    category_counts = con_intellect_how[follow_up].value_counts()
    category_colors = {survey_mapping.ANS_C_NECESSARY: "#F7F0F5",
                       survey_mapping.ANS_I_NECESSARY: "#DECBB7",
                       survey_mapping.ANS_SAME: "#8F857D",
                       survey_mapping.ANS_THIRD: "#5C5552",
                       }
    plotter.plot_pie(categories_names=category_counts.index.tolist(), categories_counts=category_counts.tolist(),
                     categories_colors=category_colors, title=f"{follow_up}",
                     save_path=result_path, save_name=f"{follow_up.replace('?', '').replace('/', '-')}", format="png")
    category_counts.to_csv(os.path.join(result_path, f"{follow_up.replace('?', '').replace('/', '-')}.csv"))

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


def demographics(analysis_dict, save_path):
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

    age = "How old are you?"
    age_stats = con_demo[age].astype(float).describe()
    age_stats.to_csv(os.path.join(result_path, "age_stats.csv"))  # index=True as it includes the information

    age_counts = con_demo[age].value_counts()
    age_counts_df = age_counts.reset_index(drop=False, inplace=False)
    age_counts_df_sorted = age_counts_df.sort_values(age, ascending=True)
    plotter.plot_histogram(df=age_counts_df_sorted, category_col=age, data_col="count",
                           save_path=result_path, save_name=f"age", format="svg")

    age_props = con_demo[age].value_counts(normalize=True)
    age_props_df = age_props.reset_index(drop=False, inplace=False)
    age_props_df["proportion"] = 100 * age_props_df["proportion"]

    merged_df = pd.merge(age_counts_df, age_props_df, on=age)
    merged_df.to_csv(os.path.join(result_path, "age.csv"), index=False)

    """
    Country of Residence
    """

    country = "In which country do you currently reside?"
    category_counts = con_demo[country].value_counts()
    country_proportions = con_demo[country].value_counts(normalize=True).reset_index(drop=False, inplace=False)
    country_proportions["proportion"] = 100 * country_proportions["proportion"]  # turn to % s

    # country & continent world map proportions
    plotter.plot_world_map_proportion(country_proportions_df=country_proportions, data_column=country,
                                      save_path=result_path)

    """ 
    Gender
    """

    gender_order = ["Female", "Male", "Non-binary", "Genderqueer", "Prefer not to say"]
    color_list = ["#8F8F8F"] * len(gender_order)

    gender = "How do you describe yourself?"
    category_counts = con_demo[gender].value_counts()

    # gender_order = ["Male", "Non-binary", "Prefer not to say", "Genderqueer", "Female"]
    # gender_color_dict = plotter.diverging_palette(color_order=gender_order, left=225, right=45)
    # plotter.plot_pie(categories_names=category_counts.index.tolist(), categories_counts=category_counts.tolist(),
    #                 categories_colors=gender_color_dict, title=f"{gender}", pie_direction=90, edge_color="none",
    #                 save_path=result_path, save_name=f"gender", format="png", annot_props=False, annot_groups=False,
    #                 legend=True, legend_order=gender_order, legend_vertical=False)

    category_props = con_demo[gender].value_counts(normalize=True)
    category_props_df = category_props.reset_index(drop=False, inplace=False)
    category_props_df["proportion"] = 100 * category_props_df["proportion"]
    category_props_df_ordered = category_props_df.sort_values("proportion", ascending=False)
    plotter.plot_categorical_bars(categories_prop_df=category_props_df_ordered,
                                  category_col=gender, y_min=0, y_max=60, y_skip=10,
                                  data_col="proportion",
                                  categories_colors=color_list,
                                  save_path=result_path, save_name=f"gender", format="svg")

    category_counts_df = category_counts.reset_index(drop=False, inplace=False)
    merged_df = pd.merge(category_counts_df, category_props_df_ordered, on=gender)
    merged_df.to_csv(os.path.join(result_path, "gender.csv"), index=False)

    """ 
    Education
    """
    education = "What is your education background?"
    education_order = [survey_mapping.EDU_NONE, survey_mapping.EDU_PRIM, survey_mapping.EDU_SECD,
                       survey_mapping.EDU_POSTSEC, survey_mapping.EDU_GRAD]
    education_labels = {edu: edu.replace(" education", "") for edu in education_order[1:]}
    education_labels[survey_mapping.EDU_NONE] = survey_mapping.EDU_NONE
    education_labels = {edu: re.sub(r'\(.*?\)', '', education_labels[edu]) for edu in
                        education_labels.keys()}  # remove parantheses

    education_color_dict = {survey_mapping.EDU_NONE: "#AAD2BA",
                            survey_mapping.EDU_PRIM: "#7BA084",
                            survey_mapping.EDU_SECD: "#6B8F71",
                            survey_mapping.EDU_POSTSEC: "#58735B",
                            survey_mapping.EDU_GRAD: "#445745"}

    education_counts = con_demo[education].value_counts().reset_index(drop=False, inplace=False)
    education_props = con_demo[education].value_counts(normalize=True)
    education_props_df = education_props.reset_index(drop=False, inplace=False)
    education_props_df["proportion"] = 100 * education_props_df["proportion"]

    order_dict = {survey_mapping.EDU_NONE: 4,
                  survey_mapping.EDU_PRIM: 3,
                  survey_mapping.EDU_SECD: 2,
                  survey_mapping.EDU_POSTSEC: 1,
                  survey_mapping.EDU_GRAD: 0}

    education_props_df_ordered = education_props_df.sort_values(by=education, key=lambda x: x.map(order_dict))
    education_props_df_ordered[f"{education}_label"] = education_props_df_ordered[education].replace(education_labels,
                                                                                                     inplace=False)
    education_props_df_ordered.reset_index(drop=True, inplace=True)
    plotter.plot_categorical_bars(categories_prop_df=education_props_df_ordered,
                                  category_col=f"{education}_label", y_min=0, y_max=50, y_skip=10,
                                  data_col="proportion",
                                  categories_colors=color_list,
                                  save_path=result_path, save_name=f"education", format="svg")

    merged_df = pd.merge(education_counts, education_props_df_ordered, on=education)
    merged_df.to_csv(os.path.join(result_path, "education.csv"), index=False)

    # education field
    field = "In what topic?"

    # people could have selected multiple values here, so handle it to count the right number of each option
    education_field_df = con_demo.copy()
    education_field_df[field] = education_field_df[field].str.split(',')
    exploded_df = education_field_df.explode(field)
    category_counts = exploded_df[field].value_counts()

    # list names of topics where enough of the population have replied
    topic_props = category_counts.reset_index(drop=False, inplace=False)
    topic_props["proportion"] = (topic_props["count"] / topic_props["count"].sum()) * 100
    topic_props.to_csv(os.path.join(result_path, "education_topic.csv"), index=False)

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

    plotter.plot_pie(categories_names=topic_order, categories_counts=category_counts,
                     categories_colors=topic_color_dict, title=f"{field}",
                     pie_direction=180, annot_groups=True, annot_group_selection=substantial_list,
                     annot_props=False, edge_color="none",
                     save_path=result_path, save_name=f"field", format="png")

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

    plotter.plot_pie(categories_names=employment_order,
                     categories_counts=category_counts,
                     # [employment_counts[employment] for employment in employment_order]
                     categories_colors=employment_colors, title=f"{employment}", edge_color="none",
                     pie_direction=180, annot_groups=True, annot_group_selection=substantial_list, annot_props=False,
                     save_path=result_path, save_name=f"employment", format="png")

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


def experience(analysis_dict, save_path):
    # save path
    result_path = os.path.join(save_path, "experience")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    ethics = analysis_dict["ethics_exp"]
    animals = analysis_dict["animal_exp"]
    ai = analysis_dict["ai_exp"]
    consciousness = analysis_dict["consciousness_exp"]

    ethics_q = "On a scale from 1 to 5 where 1 means 'none' and 5 means 'extremely', how would you rate your experience and knowledge in ethics and morality?"
    animal_q = "On a scale from 1 to 5 where 1 means 'none' and 5 means 'extremely', how would you rate your level of interaction or experience with animals?"
    ai_q = "On a scale from 1 to 5 where 1 means 'none' and 5 means 'extremely', how would you rate your experience and knowledge in artificial intelligence (AI) systems?"
    consciousness_q = "On a scale from 1 to 5 where 1 means 'none' and 5 means 'extremely', how would you rate your experience and knowledge in the science of consciousness?"

    ethics_counts = ethics[ethics_q].value_counts().sort_index().reset_index()
    ethics_counts.columns = ["rating", "count"]
    ethics_counts["proportion"] = (ethics_counts["count"] / ethics_counts["count"].sum()) * 100
    ethics_counts["experience"] = "ethics"
    animal_counts = animals[animal_q].value_counts().sort_index().reset_index()
    animal_counts.columns = ["rating", "count"]
    animal_counts["proportion"] = (animal_counts["count"] / animal_counts["count"].sum()) * 100
    animal_counts["experience"] = "animals"
    ai_counts = ai[ai_q].value_counts().sort_index().reset_index()
    ai_counts.columns = ["rating", "count"]
    ai_counts["proportion"] = (ai_counts["count"] / ai_counts["count"].sum()) * 100
    ai_counts["experience"] = "ai"
    consciousness_counts = consciousness[consciousness_q].value_counts().sort_index().reset_index()
    consciousness_counts.columns = ["rating", "count"]
    consciousness_counts["proportion"] = (consciousness_counts["count"] / consciousness_counts["count"].sum()) * 100
    consciousness_counts["experience"] = "consciousness"

    experience_counts = pd.concat([ethics_counts, animal_counts, ai_counts, consciousness_counts], ignore_index=True)
    experience_counts.to_csv(os.path.join(result_path, "experience_proportions.csv"), index=False)

    """
    Does experience with animals affect answers related to animal C / MS ? 
    """

    animals = animals.apply(helper_funcs.replace_animal_other, axis=1)  # replace "Other" if categories do exist

    # first of all, general animal info
    stats = {animal_q: helper_funcs.compute_stats(animals[animal_q], possible_values=[1, 2, 3, 4])}
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
                     save_path=result_path, save_name="exp_animal_types", format="png")

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

    other_creatures(analysis_dict=analysis_dict, save_path=save_path, sort_together=False,
                    df_earth_cluster=None)

    kill_for_test(analysis_dict=analysis_dict, save_path=save_path, df_earth_cluster=df_earth_cluster)

    graded_consciousness(analysis_dict, save_path)

    demographics(analysis_dict, save_path)

    experience(analysis_dict, save_path)

    moral_considreation_prios(analysis_dict=analysis_dict, save_path=save_path, df_earth_cluster=df_earth_cluster)

    ics(analysis_dict=analysis_dict, save_path=save_path, df_earth_cluster=df_earth_cluster)

    ms_features_order_df, feature_colors = moral_consideration_features(analysis_dict=analysis_dict,
                                                                        save_path=save_path,
                                                                        df_earth_cluster=df_earth_cluster)

    zombie_pill(analysis_dict, save_path, feature_order_df=ms_features_order_df, feature_color_map=feature_colors)

    consciousness_intelligence(analysis_dict, save_path)

    gender_cross(analysis_dict, save_path)  # move to after the individuals

    return
