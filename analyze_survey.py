import os
import pandas as pd
import numpy as np
import re
from functools import reduce
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.stats import iqr, friedmanchisquare
import process_survey
import survey_mapping
import helper_funcs
import plotter


AGE_BINS = [18, 25, 35, 45, 55, 65, 75, 120]
AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "56-65", "66-75", "76+"]

COUNT = "count"
PROP = "proportion"


def earth_in_danger_clustering(analysis_dict, save_path, cluster_num=2, cluster_colors=None):
    """
    Perform clustering of responses into [cluster_num] clusters based on responses to the Earth-in-Danger dilemma.

    :param analysis_dict: analysis dict of all response-blocks dfs
    :param save_path: path where all the analysis results are saved
    :param cluster_num: number of clusters
    :param cluster_colors: (optional) colors of clusters
    :return: the results of the kmeans clustering: the df with the tagging of the clusters, the kmeans, and the
    cluster centroids
    """
    # save path
    result_path = os.path.join(save_path, "earth_danger")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # load relevant data
    df_earth = analysis_dict["earth_in_danger"]
    questions = df_earth.columns[df_earth.columns != process_survey.COL_ID].tolist()

    """
    Kmeans clustering
    """
    df_earth_coded = df_earth.copy()
    for col in questions:
        """
        Kmeans clustering relies on distance measures like Euclidean distance to determine cluster centers, 
        so if using arbitrary numbers for categories, the model will interpret these numbers as having some sort of 
        distance relationship. 

        So we will convert everything into binary. However, in order to keep interpretability, I will choose the
        0's and 1's myself (and not simple map each column into binary arbitrarily), to identify the same meaning. 
        For that we use: EARTH_DANGER_QA_MAP
        """
        col_map = survey_mapping.EARTH_DANGER_QA_MAP[col]
        df_earth_coded[col] = df_earth_coded[col].map(col_map)

    df_earth_coded.set_index([process_survey.COL_ID], inplace=True)

    """
    Perform k-means clustering: group the choices into k clusters (cluster_num) based on feature similarity.
    Each cluster is represented by a "centroid" (average position of the data points in the cluster).
    Data points are assigned to the cluster whose centroid they are closest to.
    """
    df_pivot, kmeans, cluster_centroids = helper_funcs.perform_kmeans(df_pivot=df_earth_coded, clusters=cluster_num,
                                                                      save_path=result_path, save_name="items")

    """
    Plot the KMeans cluster centroids. For each cluster, the centroid is the average data point for this cluster 
    (the mean value of the features for all data points in the cluster). We use the centroids to visualize each 
    cluster's choice in each earth-is-in-danger dyad, to interpret the differences between them.  
    """

    # Compute the cluster centroids and SEMs
    # cluster_centroids = df_pivot.groupby("Cluster").mean()  # we get this from helper_funcs.perform_kmeans
    cluster_sems = df_pivot.groupby("Cluster").sem()

    # Plot - collapsed (all clusters together)
    if cluster_colors is None:
        cluster_colors = ["#EDAE49", "#102E4A"]
    helper_funcs.plot_cluster_centroids(cluster_centroids=cluster_centroids, cluster_sems=cluster_sems,
                                        save_path=result_path, save_name="items", fmt="svg",
                                        label_map=survey_mapping.EARTH_DANGER_QA_MAP, binary=True,
                                        label_names_coding=survey_mapping.EARTH_DANGER_ANS_MAP,
                                        threshold=0, overlaid=True, cluster_colors_overlaid=cluster_colors)

    return df_pivot, kmeans, cluster_centroids


def demographics_age(demographics_df, save_path):
    """
    Get descriptive statistics about age
    :param demographics_df: the df block of demographic data
    :param save_path: the path to which to save the results
    """

    """
    General descriptives
    """
    age_col = survey_mapping.Q_AGE
    age_stats = demographics_df[age_col].astype(float).describe()
    age_stats.to_csv(os.path.join(save_path, "age_stats.csv"))  # index=True as it includes the information

    """
    Distribution of age by age group 
    (predefined, this is one of the factors based on which we balanced the classes in process_survey)
    """
    # a new column for age group
    demographics_df["age_group"] = pd.cut(
        demographics_df[age_col],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=True,  # Includes the right edge, e.g. 25 is in "18-25"
        include_lowest=True
    )
    # count the number of people in each age group and convert to df
    age_group_counts = demographics_df["age_group"].value_counts(sort=False)
    age_group_counts_df = age_group_counts.reset_index()
    age_group_counts_df.columns = ["age_group", COUNT]
    age_group_counts_df = age_group_counts_df.sort_values("age_group").reset_index(drop=True)
    # proportion
    total_count = age_group_counts_df[""].sum()
    age_group_counts_df[PROP] = 100 * age_group_counts_df[COUNT] / total_count
    age_group_counts_df.to_csv(os.path.join(save_path, "age_group_props.csv"), index=False)

    # plot histogram
    age_color_order = ["#d69f7e", "#cd9777", "#c38e70", "#b07d62", "#9d6b53" "#8a5a44" "#774936"]
    plotter.plot_histogram(df=age_group_counts_df,
                           category_col="age_group", data_col=COUNT,
                           x_label="Age Group", y_label=COUNT, ytick_interval=10,
                           save_path=save_path, save_name=f"age",
                           format="svg",
                           colors=age_color_order)

    return


def demographics_gender(demographics_df, save_path):
    """
    Get descriptive statistics about gender
    :param demographics_df: the df block of demographic data
    :param save_path: the path to which to save the results
    """

    """
    General descriptives
    """
    gender_col = survey_mapping.Q_GENDER
    gender_order = ["Female", "Male", "Non-binary", "Genderqueer", "Prefer not to say"]

    category_counts = demographics_df[gender_col].value_counts()
    category_props = demographics_df[gender_col].value_counts(normalize=True)
    category_df = pd.DataFrame({
        gender_col: category_counts.index,
        COUNT: category_counts.values,
        PROP: category_props.values * 100  # convert to percentage
    })
    # sort by proportion
    category_df = category_df.sort_values(PROP, ascending=False).reset_index(drop=True)
    category_df.to_csv(os.path.join(save_path, "gender.csv"), index=False)

    """
    Plot
    """
    gender_color_dict = {"Female": "#d4a373",
                         "Male": "#4a5759",
                         "Non-binary": "#f7e1d7",
                         "Genderqueer": "#edafb8",
                         "Prefer not to say": "#dedbd2"}

    plotter.plot_pie(categories_names=category_counts.index.tolist(), categories_counts=category_counts.tolist(),
                     categories_colors=gender_color_dict, title=f"{gender_col}", pie_direction=90, edge_color="none",
                     save_path=save_path, save_name=f"gender", fmt="svg",
                     props_in_legend=True, annot_props=True, annot_groups=False,
                     legend=True, legend_order=gender_order, legend_vertical=True)

    return


def demographics_education(demographics_df, save_path):
    """
    Get descriptive statistics about education level
    :param demographics_df: the df block of demographic data
    :param save_path: the path to which to save the results
    """

    """
    General descriptives
    """

    education_col = survey_mapping.Q_EDU
    education_order = survey_mapping.EDU_ORDER

    category_counts = demographics_df[education_col].value_counts()
    category_props = demographics_df[education_col].value_counts(normalize=True)
    category_df = pd.DataFrame({
        education_col: category_counts.index,
        COUNT: category_counts.values,
        PROP: category_props.values * 100  # convert to percentage
    })
    # sort by education level
    category_df[education_col] = pd.Categorical(category_df[education_col], categories=education_order, ordered=True)
    category_df = category_df.sort_values(by=education_col).reset_index(drop=True)
    # clean labels
    education_labels = {edu: edu.replace(" education", "") for edu in education_order[1:]}
    education_labels[survey_mapping.EDU_NONE] = survey_mapping.EDU_NONE
    education_labels = {
        edu: re.sub(r'\(.*?\)', '', label).strip()
        for edu, label in education_labels.items()
    }
    # Map to new "label" column
    category_df["education_label"] = category_df[education_col].map(education_labels)
    # save
    category_df.to_csv(os.path.join(save_path, "education.csv"), index=False)

    """
    Plot
    """
    education_color_order = ["#a9d6e5", "#89c2d9", "#61a5c2", "#468faf", "#2c7da0"]
    plotter.plot_histogram(df=category_df,
                           category_col="education_label", data_col=COUNT,
                           x_label="Education Level", y_label=COUNT, ytick_interval=10,
                           save_path=save_path, save_name=f"education",
                           format="svg",
                           colors=education_color_order)

    """
    Follow up on higher education - topic
    """
    field_col = survey_mapping.Q_EDU_FIELD  # education field
    # handle multiple selections
    education_field_df = demographics_df.copy()
    education_field_df[field_col] = education_field_df[field_col].dropna().astype(str).str.split(',')
    # explode into one row per selected field
    exploded_df = education_field_df.explode(field_col)
    exploded_df[field_col] = exploded_df[field_col].str.strip()  # clean whitespace
    # count and proportion
    category_counts = exploded_df[field_col].value_counts()
    category_props = exploded_df[field_col].value_counts(normalize=True) * 100
    field_df = pd.DataFrame({
        field_col: category_counts.index,
        COUNT: category_counts.values,
        PROP: category_props.values
    }).reset_index(drop=True)
    field_df = field_df.sort_values(by=PROP, ascending=False).reset_index(drop=True)  # descending proportion
    field_df.to_csv(os.path.join(save_path, "education_topic.csv"), index=False)

    return


def demographics_employment(demographics_df, save_path):
    """
    Get descriptive statistics about current employment domain
    :param demographics_df: the df block of demographic data
    :param save_path: the path to which to save the results
    """

    """
    General descriptives
    """

    employment_col = survey_mapping.Q_EMPLOYMENT
    employment_counts = demographics_df[employment_col].value_counts()
    employment_props = demographics_df[employment_col].value_counts(normalize=True) * 100

    employment_df = pd.DataFrame({
        employment_col: employment_counts.index,
        COUNT: employment_counts.values,
        PROP: employment_props.values
    }).reset_index(drop=True)
    employment_df = employment_df.sort_values(by=PROP, ascending=False).reset_index(drop=True)
    employment_df.to_csv(os.path.join(save_path, "employment.csv"), index=False)

    return


def demographics_country(demographics_df, save_path):
    """
    Get descriptive statistics about current country of residence
    :param demographics_df: the df block of demographic data
    :param save_path: the path to which to save the results
    """

    """
    General descriptives
    """

    country_col = survey_mapping.Q_COUNTRY
    country_counts = demographics_df[country_col].value_counts()
    country_props = demographics_df[country_col].value_counts(normalize=True) * 100

    country_df = pd.DataFrame({
        country_col: country_counts.index,
        COUNT: country_counts.values,
        PROP: country_props.values
    }).reset_index(drop=True)
    country_df = country_df.sort_values(by=PROP, ascending=False).reset_index(drop=True)
    country_df.to_csv(os.path.join(save_path, "country.csv"), index=False)

    """
    Plot country and continent proportions on the world map
    """
    plotter.plot_world_map_proportion(country_proportions_df=country_df, data_column=country_col,
                                      save_path=save_path, save_name="country_map", fmt="svg")
    return


def demographics_descriptives(analysis_dict, save_path):
    """
    Get basic descriptives for all demographic data
    :param analysis_dict:
    :param save_path:
    :return:
    """

    # save path
    result_path = os.path.join(save_path, "demographics")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # save the full demographics df based on which we will extract all descriptives
    con_demo = analysis_dict["demographics"]
    con_demo.to_csv(os.path.join(result_path, "demographics_full.csv"), index=False)

    """
    Age
    """
    demographics_age(demographics_df=con_demo, save_path=result_path)

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
    demographics_employment(demographics_df=con_demo, save_path=result_path)

    """
    Country of Residence
    """
    demographics_country(demographics_df=con_demo, save_path=result_path)

    return


def experience_descriptives(analysis_dict, save_path):
    # save path
    result_path = os.path.join(save_path, "experience")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # experience types
    experience_sources = {
        "ethics": (analysis_dict["ethics_exp"], survey_mapping.Q_ETHICS_EXP),
        "animals": (analysis_dict["animal_exp"], survey_mapping.Q_ANIMAL_EXP),
        "ai": (analysis_dict["ai_exp"], survey_mapping.Q_AI_EXP),
        "consciousness": (analysis_dict["consciousness_exp"], survey_mapping.Q_CONSC_EXP),
    }

    """
    General descriptives
    """

    summary_rows = []
    all_ratings = set()
    # collect all possible rating values
    for df, col in experience_sources.values():
        ratings = df[col].dropna().astype(int)
        all_ratings.update(ratings.unique())

    rating_scale = sorted(all_ratings)
    # build summaries
    for label, (df, col) in experience_sources.items():
        ratings = df[col].dropna().astype(int)
        rating_mean = ratings.mean()
        rating_sd = ratings.std(ddof=1)
        rating_se = rating_sd / np.sqrt(len(ratings))

        counts = ratings.value_counts().reindex(rating_scale, fill_value=0)
        props = (counts / counts.sum()) * 100
        row = {
            "experience": label,
            "M": rating_mean,
            "SD": rating_sd,
            "SE": rating_se,
        }
        for r in rating_scale:
            row[f"{COUNT}_{r}"] = counts[r]
            row[f"{PROP}_{r}"] = props[r]

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    ordered_cols = (
            ["experience", "M", "SD", "SE"] +
            [f"{COUNT}_{r}" for r in rating_scale] +
            [f"{PROP}_{r}" for r in rating_scale]
    )
    summary_df = summary_df[ordered_cols]

    # Save to CSV
    summary_df.to_csv(os.path.join(result_path, "experience_proportions.csv"), index=False)

    return


def ics(analysis_dict=analysis_dict, save_path=save_path):
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

    """
    Step 1: basic demographics
    Get what we need to report the standard things we do
    """
    demographics_descriptives(analysis_dict=analysis_dict, save_path=save_path)

    """
    Step 2: expertise
    We collected self-reported expertise levels with various topics. Get what we need to report about that
    """
    experience_descriptives(analysis_dict=analysis_dict, save_path=save_path)

    """
    Step 3: can consciousness be separated from intentions/valence? 
    Answers to the "Do you think a creature/system can have intentions/consciousness/sensations w/o having..?" section
    """
    ics(analysis_dict=analysis_dict, save_path=save_path)

    """
    Step 4: 
    """


    if load:  # load the earth-in-danger things
        df_earth_clusters = pd.read_csv(os.path.join(save_path, "earth_danger", f"earth_danger_clusters.csv"))

    else:
        df_earth_clusters, kmeans, cluster_centroids = earth_in_danger_clustering(analysis_dict, save_path)
