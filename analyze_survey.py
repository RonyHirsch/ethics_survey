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
EXPERTISE = 4  # this ranking (and up) is what will be counted as an 'expert'

COUNT = "count"
PROP = "proportion"

YES_NO_COLORS = {survey_mapping.ANS_YES: "#3C5968",
                 survey_mapping.ANS_NO: "#B53B03"}

EXP_COLORS = {1: "#e63946",
              2: "#f1faee",
              3: "#a8dadc",
              4: "#457b9d",
              5: "#344968"}

ICS_GROUP_COLOR_LIST = ["#B53B03", "#ee9b00", "#005f73", "#3B4E58"]
ICS_GROUP_ORDER_LIST = ["multidimensional", "cognitive-agential", "experiential", "other"]


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
        right=True,
        include_lowest=True
    )
    # count the number of people in each age group and convert to df
    age_group_counts_df = demographics_df["age_group"].value_counts(sort=False).reset_index(inplace=False)
    age_group_counts_df = age_group_counts_df.sort_values("age_group").reset_index(drop=True)
    # proportion
    total_count = age_group_counts_df[COUNT].sum()
    age_group_counts_df[PROP] = 100 * age_group_counts_df[COUNT] / total_count
    age_group_counts_df.to_csv(os.path.join(save_path, "age_group_props.csv"), index=False)

    # plot histogram
    age_color_order = ["#BC3908", "#BC3908", "#BC3908", "#BC3908", "#BC3908", "#BC3908", "#BC3908"]
    plotter.plot_histogram(df=age_group_counts_df,
                           category_col="age_group", data_col=COUNT,
                           x_label="Age Group", y_label=COUNT, ytick_interval=10,
                           save_path=save_path, save_name=f"age",
                           format="svg",
                           colors=age_color_order)

    return demographics_df.loc[:, [process_survey.COL_ID, "age_group"]]


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

    return demographics_df.loc[:, [process_survey.COL_ID, gender_col]]


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

    demographics_df["education_level"] = demographics_df[education_col].map(survey_mapping.EDU_MAP)
    return demographics_df.loc[:, [process_survey.COL_ID, education_col, "education_level"]]


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

    return demographics_df.loc[:, [process_survey.COL_ID, employment_col]]


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
    return  demographics_df.loc[:, [process_survey.COL_ID, country_col]]


def demographics_descriptives(analysis_dict, save_path):
    """
    Get basic descriptives for all demographic data
    :param analysis_dict: dictionary where key=topic, value=a dataframe containing all the columns relevant for this
    topic/section
    :param save_path: where the results will be saved (csvs, plots)
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
    df_age = demographics_age(demographics_df=con_demo, save_path=result_path)

    """ 
    Gender
    """
    df_gender = demographics_gender(demographics_df=con_demo, save_path=result_path)

    """ 
    Education
    """
    df_edu = demographics_education(demographics_df=con_demo, save_path=result_path)

    """
    Employment
    """
    df_employment = demographics_employment(demographics_df=con_demo, save_path=result_path)

    """
    Country of Residence
    """
    df_country = demographics_country(demographics_df=con_demo, save_path=result_path)

    all_dfs = [df_age, df_gender, df_edu, df_employment, df_country]
    merged_demo_df = reduce(lambda left, right: pd.merge(left, right, on=[process_survey.COL_ID], how="outer"), all_dfs)

    return merged_demo_df


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
    all_experience = list()
    summary_rows = list()
    all_ratings = set()
    # collect all possible rating values
    for df, col in experience_sources.values():
        all_experience.append(df)
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
    summary_df.to_csv(os.path.join(result_path, "experience_proportions.csv"), index=False)

    """
    Unified experience table: 
    add all experience types, and also BINARIZE them according to the following logic: 
    Non-experts: ratings 1-3; Experts: ratings 4-5
    """
    all_experience_df = reduce(lambda left, right: pd.merge(left, right, on=[process_survey.COL_ID], how='outer'),
                               all_experience)
    rating_cols = [c for c in survey_mapping.Q_EXP_NAME_DICT.keys()]
    df_ratings = all_experience_df[[process_survey.COL_ID] + rating_cols].dropna()
    df_ratings = df_ratings.rename(columns=survey_mapping.Q_EXP_NAME_DICT)  # rename for ease of use
    plot_rating_cols = [c for c in survey_mapping.Q_EXP_NAME_DICT.values()]

    # binarize ratings based on expertise:
    df_ratings[[f"{col}_expert" for col in plot_rating_cols]] = df_ratings[plot_rating_cols].applymap(lambda x: 1 if x >= EXPERTISE else 0)

    # save full df
    df_ratings.to_csv(os.path.join(result_path, "experience_ratings.csv"), index=False)

    # descriptives on binarized expertise
    expert_cols = [f"{col}_expert" for col in plot_rating_cols]
    total_counts = df_ratings[expert_cols].shape[0]
    experts = df_ratings[expert_cols].sum()
    non_experts = df_ratings[expert_cols].shape[0] - experts
    experts_pct = (experts / total_counts * 100).round(2)
    non_experts_pct = (non_experts / total_counts * 100).round(2)

    expertise_df = pd.DataFrame({
        "Expertise": [col.replace('_expert', '') for col in expert_cols],
        f"Experts (rating >= {EXPERTISE})": experts.values,
        f"Non experts (rating < {EXPERTISE})": non_experts.values,
        f"Experts (rating >= {EXPERTISE}) {PROP}": experts_pct.values,
        f"Non experts (rating < {EXPERTISE}) {PROP}": non_experts_pct.values
    })
    expertise_df.to_csv(os.path.join(result_path, "experience_binarized_proportions.csv"), index=False)

    """
    River (sankey) plot - TODO: CHECK
    """
    import plotly.graph_objects as go
    font_size = 14

    # relevant columns

    rating_levels = ["1", "2", "3", "4", "5"]
    df_ratings[plot_rating_cols] = df_ratings[plot_rating_cols].astype(int).astype(str)

    # Prepare flow data
    all_flows = []
    for i in range(len(plot_rating_cols) - 1):
        flow = df_ratings.groupby([plot_rating_cols[i], plot_rating_cols[i + 1]]).size().reset_index(name=COUNT)
        flow.columns = ['source', 'target', COUNT]
        flow['source_stage'] = plot_rating_cols[i]
        flow['target_stage'] = plot_rating_cols[i + 1]
        all_flows.append(flow)

    all_flows_df = pd.concat(all_flows, ignore_index=True)

    # Create node labels
    labels = []
    stage_label_map = {}
    for stage in plot_rating_cols:
        for val in rating_levels:  # fixed order 1-5
            label = f"{stage}: {val}"
            labels.append(label)
            stage_label_map[(stage, val)] = label

    unique_labels = labels  # already sorted
    label_to_index = {label: i for i, label in enumerate(unique_labels)}

    # Sankey links
    sources, targets, values = [], [], []
    for _, row in all_flows_df.iterrows():
        src_label = stage_label_map[(row['source_stage'], row['source'])]
        tgt_label = stage_label_map[(row['target_stage'], row['target'])]
        sources.append(label_to_index[src_label])
        targets.append(label_to_index[tgt_label])
        values.append(row[COUNT])

    # Customize here
    color_map = EXP_COLORS
    node_colors = [color_map[int(label.split(": ")[1])] for label in unique_labels]

    # Sankey Diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=unique_labels,
            color=node_colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])
    fig.update_layout(title_text="Experience Ratings Sankey Diagram", font_size=font_size)

    #fig.show()

    x = 4  ## TODO: DEBUG HERE to solve this figure
    #flow_data_export = pd.DataFrame({
    #    "source": [unique_labels[s] for s in sources],
    #    "target": [unique_labels[t] for t in targets],
    #    "value": values
    #})

    #flow_data_export.to_csv(os.path.join("experience_flow.csv"), index=False)
    #try:
    #    fig.write_image(os.path.join("experience_flow.svg"))
    #except ValueError as e:
    #    print(f"SVG not saved — you need to install kaleido: pip install -U kaleido\n{e}")

    """
    Alternative figure - stacked bar plot
    """
    counts_df = df_ratings[plot_rating_cols].apply(lambda col: col.value_counts().sort_index()).astype(int)
    counts_df = counts_df.reset_index(drop=False, inplace=False).rename(columns={"index": "rating"}, inplace=False)
    counts_df["rating"] = counts_df["rating"].astype(int)
    target_row = counts_df[counts_df["rating"] == len(rating_levels)].iloc[0]
    sorted_columns = sorted(list(plot_rating_cols), key=lambda col: target_row[col], reverse=True)
    plotter.plot_stacked_proportion_bars(df=counts_df, rating_col="rating", item_cols=sorted_columns,
                                         colors=EXP_COLORS, num_ratings=len(rating_levels),
                                         annotate_bar=True, annot_font_size=20,
                                         annot_font_colors=["#24272E", "#2D3039", "#3F4350", "#3F4350", "#BABEC9"],
                                         ytick_visible=True, text_width=30, title="Self-Reported Experience Level",
                                         save_path=result_path, save_name="experience_proportions", fmt="svg",
                                         show_mean=False, sem_line=False, default_ticks=False)

    return df_ratings, result_path


def calculate_ics_proportions(df_ics, save_path):
    """
    ICS block - do you think it's possible to have intentions wo consciousness, consciousness wo sensations, etc.
    Calculate the proportions of yes/no in response to the ics block
    :param df_ics: df with the ics block
    :param save_path: where the results will be saved
    :returns df_group_cols: a df where row=sub, with the 2 columns of consciousness wo something and another column
    of tagging people to groups based on their conception of consciousness based on the answers to these 2 q's.
    """

    questions = list(survey_mapping.ICS_Q_NAME_MAP.keys())
    q_name_map = survey_mapping.ICS_Q_NAME_MAP
    ans_map = {survey_mapping.ANS_NO: 0, survey_mapping.ANS_YES: 1}
    rating_labels = [survey_mapping.ANS_NO, survey_mapping.ANS_YES]
    rating_colors = [YES_NO_COLORS[survey_mapping.ANS_NO], YES_NO_COLORS[survey_mapping.ANS_YES]]

    """
    counts of responses per question
    """
    response_counts = df_ics[questions].apply(pd.Series.value_counts).fillna(0).astype(int)
    response_counts.index.name = "response"
    response_counts.to_csv(os.path.join(save_path, "ics_response_counts.csv"))
    # map answers to numeric values
    df_mapped = df_ics[questions].replace(ans_map)

    """
    Compute stats per question
    helper_funcs.computer_stats() returns proportions of answers (0, 1), mean, std, and N
    """
    stats = {q_name_map[q]: helper_funcs.compute_stats(df_mapped[q], possible_values=[0, 1]) for q in questions}

    """
    Plot stacked bar plots of yes-no (possible-impossible) options
    """
    plot_data = {
        qname: {  # based on helper_funcs.compute_stats() outputs
            "Proportion": s[0],
            "Mean": s[1],
            "Std Dev": s[2],
            "N": s[3]
        }
        for qname, s in stats.items()
    }
    sorted_plot_data = sorted(plot_data.items(), key=lambda x: x[1]["Mean"], reverse=True)

    plotter.plot_stacked_proportion_bars(plot_data=sorted_plot_data, legend_labels=rating_labels, ytick_visible=True,
                                         text_width=39, title="Do you think a creature/system can be", show_mean=False,
                                         sem_line=False, colors=rating_colors, num_ratings=2, annotate_bar=True,
                                         annot_font_color="#e0e1dd", save_path=save_path, save_name="ics_props",
                                         fmt="svg")

    # save data
    plot_df = pd.DataFrame(plot_data)
    plot_df.to_csv(os.path.join(save_path, f"ics_props.csv"), index=True)

    """
    CONSCIOUSNESS W/O THINGS - What can we learn about it? 
    (1) plot just the 'consciousness w/o...' bars: 
    """
    q_int = q_name_map[survey_mapping.ICS_Q_CONS_WO_INT]
    q_sens = q_name_map[survey_mapping.ICS_Q_CONS_WO_SENS]
    c_wo_stuff = [q_int, q_sens]

    subset_plot_data = [item for item in sorted_plot_data if item[0] in c_wo_stuff]
    plotter.plot_stacked_proportion_bars(plot_data=subset_plot_data, legend_labels=rating_labels,
                                         ytick_visible=True, text_width=39,
                                         title="Do you think a creature/system can be", show_mean=False, sem_line=False,
                                         colors=rating_colors, num_ratings=2, annotate_bar=True,
                                         annot_font_color="#e0e1dd", save_path=save_path,
                                         save_name="consciousness_wo", fmt="svg")

    """
    CONSCIOUSNESS W/O THINGS - What can we learn about it? 
    (2) group by people's answers: 
        (i) multidimensional = impossible to have c w/o intentions or without sensations (i.e., 'no' to both)
        (ii) cognitive-agential = possible to have c w/o sensations, but not w/o intentions/goals ('no' to ICS_Q_CONS_WO_INT, 'yes' to ICS_Q_CONS_WO_SENS)
        (iii) experiential = possible to have c w/o intentions, but not w/o sensations ('yes' to ICS_Q_CONS_WO_INT, 'no' to ICS_Q_CONS_WO_SENS)
        (iv) other = possible to have consciousness w/o both ('yes' to both)
    """

    df_group_cols = df_ics[
        [process_survey.COL_ID, survey_mapping.ICS_Q_CONS_WO_INT, survey_mapping.ICS_Q_CONS_WO_SENS]].rename(
        columns=q_name_map)

    # the 4 options for combinations of answers of consciousness w/o stuff
    conditions = {
        "multidimensional": (df_group_cols[q_int] == "No") & (df_group_cols[q_sens] == "No"),
        "cognitive-agential": (df_group_cols[q_int] == "No") & (df_group_cols[q_sens] == "Yes"),
        "experiential": (df_group_cols[q_int] == "Yes") & (df_group_cols[q_sens] == "No"),
        "other": (df_group_cols[q_int] == "Yes") & (df_group_cols[q_sens] == "Yes"),
    }

    df_group_cols["group"] = np.select(list(conditions.values()), list(conditions.keys()))
    df_groups = df_group_cols.groupby("group")[process_survey.COL_ID].nunique().reset_index(name=COUNT)
    df_groups[PROP] = 100 * df_groups[COUNT] / df_groups[COUNT].sum()
    df_groups.to_csv(os.path.join(save_path, "consciousness_wo_groups.csv"), index=False)

    """
    Plot the groups
    """
    group_names_map = {
        "multidimensional": "Not possible w/o either sensations or intentions",
        "cognitive-agential": "Possible w/o sensations, but not w/o intentions",
        "experiential": "Possible w/o intentions, but not w/o sensations",
        "other": "Possible w/o both sensations and intentions"
    }

    plotter.plot_categorical_bars(categories_prop_df=df_groups, data_col=PROP,
                                  order=ICS_GROUP_ORDER_LIST, name_map=group_names_map,
                                  category_col="group", y_min=0, y_max=105, y_skip=10,
                                  delete_y=False, add_pcnt=True,
                                  categories_colors=ICS_GROUP_COLOR_LIST, text_wrap_width=16, title_text="Consciousness is...",
                                  save_path=save_path, save_name=f"consciousness_wo_groups", fmt="svg")

    return df_group_cols


def ics_descriptives(analysis_dict, save_path):
    """
    Answers to the block of questions asking participants if they think a creature/system can have
    intentions (goals)/consciousness (experience)/sensations (valences, good/bad) without one another.
    :param analysis_dict: dictionary where key=topic, value=a dataframe containing all the columns relevant for this
    topic/section
    :param save_path: where the results will be saved (csvs, plots)
    :returns df_c_groups = the df where each row = sub and the columns are the 2 consciousness wo something Q's, and
    the group tagging column from  calculate_ics_proportions
    """
    result_path = os.path.join(save_path, "i_c_s")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    df_ics = analysis_dict["ics"].copy()
    df_c_groups = calculate_ics_proportions(df_ics=df_ics, save_path=result_path)
    df_c_groups.to_csv(os.path.join(result_path, "ics_with_c_groups.csv"), index=False)

    return df_c_groups, result_path


def perform_chi_square(df1, col1, df2, col2, id_col, save_path, save_name, save_expected=False,
                       grp1_vals=None, grp2_vals=None, grp2_map=None,
                       grp1_color_dict=None, y_tick_list=None, col2_name=None):
    """
    Performs chi square test of independence, testing whether the proportions in one group (df1) differ significantly
    across different groups of df2.
    grp1_vals: the order of the values of the FIRST groupings, the ones that contains MORE THAN TWO GROUPS.
    grp2_vals: chi square test of independence does not output CIs at all. To get 95% CI, we can calculate the
    difference in proportions between TWO GROUPS, and the 95% CI for that DIFFERENCE. For that, we need that
    at least one of thr groups (col1, col2), will only have TWO LEVELS, and note that we need that group to be
    GROUP 2 (so that in the contingency table these will be the COLUMNS). Then, we will pass to ci_cols the levels (2)
    of that grouping variable (e.g., [0, 1], ['no', 'yes'], etc), for the calculation of the CI.
    """

    contingency_table = pd.crosstab(df1[col1], df2[col2])
    chisquare_result, expected_df = helper_funcs.chi_squared_test(contingency_table=contingency_table,
                                                                   include_expected=True, ci_cols=grp2_vals)
    chisquare_result["question1"] = col1
    chisquare_result["question2"] = col2
    chisquare_result.to_csv(os.path.join(save_path, f"{save_name}_chisquared.csv"), index=False)
    if save_expected:
        expected_df.to_csv(os.path.join(save_path, f"{save_name}_chisquared_expected.csv"), index=False)

    # plot
    merged = pd.merge(df1[[id_col, col1]], df2[[id_col, col2]], on=id_col)
    counts = merged.groupby([col2, col1]).size().unstack(fill_value=0)
    grouped_props = counts.div(counts.sum(axis=1), axis=0) * 100
    plot_ready_df = grouped_props.reset_index()
    if not y_tick_list:
        y_tick_list = [0, 25, 50, 75, 100]
    if not grp2_map:
        grp2_map = grp2_vals
    if not col2_name:
        col2_name = col2
    plotter.plot_expertise_proportion_bars(df=plot_ready_df,
                                           cols=grp1_vals, cols_colors=grp1_color_dict,
                                           x_axis_exp_col_name=col2,
                                           x_label=f"{col2_name}", x_map=grp2_map,
                                           y_ticks=y_tick_list,
                                           save_name=f"{save_name}",
                                           save_path=save_path, plt_title=f"", plot_mean=False,
                                           stats_df=None,
                                           annotate_bar=True, annot_font_color="white")
    return


def perform_kruskal_wallis(df_ordinal, ordinal_col, df_group, group_col, id_col, save_path, save_name):
    """
    Ranks all the values from all groups together (from lowest to highest), compares average ranks across groups,
    and tests whether the distributions are significantly different across groups.
    """
    merged_df = pd.merge(df_ordinal[[id_col, ordinal_col]],
                         df_group[[id_col, group_col]],
                         on=id_col)
    kruskal_df = helper_funcs.kruskal_wallis_test(merged_df=merged_df, ordinal_col=ordinal_col, group_col=group_col)
    kruskal_df.to_csv(os.path.join(save_path, f"{save_name}_kruskal.csv"), index=False)
    return


def consc_intell_descriptives(analysis_dict, save_path):
    result_path = os.path.join(save_path, "consciousness_intelligence")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    """
    Plot the simple yes-no answer to the question
    """
    con_intellect = analysis_dict["con_intellect"]
    question = survey_mapping.Q_INTELLIGENCE
    category_counts = con_intellect[question].value_counts()
    plotter.plot_pie(categories_names=category_counts.index.tolist(), categories_counts=category_counts.tolist(),
                     categories_colors=YES_NO_COLORS, title=f"{question}", label_inside=True, prop_fmt=".0f",
                     save_path=result_path, save_name=f"{question.replace('?', '').replace('/', '-')}", fmt="svg")

    """
    If they thought that consciousness and ingelligence are related, we asked them How. Plot these proportions
    """
    follow_up = "How?"
    yes_counts = con_intellect[con_intellect[question] == survey_mapping.ANS_YES][follow_up].value_counts()
    no_count = (con_intellect[question] == survey_mapping.ANS_NO).sum()
    specific_category_counts = pd.concat([yes_counts, pd.Series({survey_mapping.ANS_NO: no_count})])
    how_label_map = {survey_mapping.ANS_C_NECESSARY: "C necessary for I",
                     survey_mapping.ANS_I_NECESSARY: "I necessary for C",
                     survey_mapping.ANS_SAME: "Same",
                     survey_mapping.ANS_THIRD: "Third feature",
                     survey_mapping.ANS_NO: survey_mapping.ANS_NO}
    how_color_map = {survey_mapping.ANS_C_NECESSARY: "#1a4e73",
                     survey_mapping.ANS_I_NECESSARY: "#346182",
                     survey_mapping.ANS_SAME: "#013a63",
                     survey_mapping.ANS_THIRD: "#4d7592",
                     survey_mapping.ANS_NO: YES_NO_COLORS[survey_mapping.ANS_NO]}
    how_order = [survey_mapping.ANS_NO, survey_mapping.ANS_SAME, survey_mapping.ANS_C_NECESSARY,
                 survey_mapping.ANS_I_NECESSARY, survey_mapping.ANS_THIRD]
    specific_category_counts = specific_category_counts.reindex(how_order)
    plotter.plot_pie(categories_names=specific_category_counts.index.tolist(), pie_direction=0,
                     categories_counts=specific_category_counts.tolist(),
                     categories_colors=how_color_map, categories_labels=how_label_map,
                     title=f"{question}", label_inside=True, prop_fmt=".0f",
                     save_path=result_path, save_name=f"{question.replace('?', '').replace('/', '-')}_how", fmt="svg")
    df_counts = specific_category_counts.to_frame(name=COUNT)
    df_counts[PROP] = 100 * (df_counts[COUNT] / df_counts[COUNT].sum())
    yes_total = df_counts.loc[df_counts.index != survey_mapping.ANS_NO, COUNT].sum()
    df_counts[f"{PROP}_out_of_{survey_mapping.ANS_YES}"] = df_counts.apply(
        lambda row: 100 * (row[COUNT] / yes_total) if row.name != survey_mapping.ANS_NO else None, axis=1)
    df_counts.to_csv(os.path.join(result_path, f"{question.replace('?', '').replace('/', '-')}_how.csv"), index=True)
    return con_intellect, result_path


def consc_intell_RF(df_demographics, df_experience, df_con_intell, save_path):
    # create dataframe to pass to a RF classifier
    experience_cols_binary = [x for x in df_experience.columns if "_expert" in x]
    df_experience_binary = df_experience.loc[:, [process_survey.COL_ID] + experience_cols_binary]
    df_c_i_relevant = df_con_intell.loc[:, [process_survey.COL_ID, survey_mapping.Q_INTELLIGENCE]]
    df_demo_relevant = df_demographics.loc[:, [process_survey.COL_ID, "age_group", "education_level",
                                               survey_mapping.Q_GENDER, survey_mapping.Q_EMPLOYMENT,
                                               survey_mapping.Q_COUNTRY]]
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=[process_survey.COL_ID], how='outer'),
                       [df_experience_binary, df_demo_relevant, df_c_i_relevant])

    """
    Process columns to fit RF classifier
    """
    # turn 'age_group' into an ordinal variable (Categorical will turn groups into ints according do the order we have)
    merged_df["age_group_order"] = pd.Categorical(merged_df["age_group"], categories=AGE_LABELS, ordered=True).codes

    # binarize the predicted column
    merged_df["con_intel_related_binary"] = merged_df[survey_mapping.Q_INTELLIGENCE].str.strip().map(survey_mapping.ANS_YESNO_MAP)
    merged_df.to_csv(os.path.join(save_path, "random_forest_dataframe_original.csv"), index=False)

    # define cols:
    categorical_cols = [
        survey_mapping.Q_GENDER,
        survey_mapping.Q_EMPLOYMENT,
        survey_mapping.Q_COUNTRY
    ]
    order_cols = experience_cols_binary + ["education_level", "age_group_order"]
    dep_col = "con_intel_related_binary"
    df = merged_df.loc[:, [process_survey.COL_ID] + categorical_cols + order_cols + [dep_col]]
    df.to_csv(os.path.join(save_path, "random_forest_dataframe_coded.csv"), index=False)
    helper_funcs.run_random_forest_pipeline(dataframe=df, dep_col=dep_col, categorical_cols=categorical_cols,
                                            order_cols=order_cols, save_path=save_path, save_prefix="",
                                            rare_class_threshold=5, n_permutations=1000, scoring_method="accuracy",
                                            cv_folds=10, split_test_size=0.3, random_state=42, n_repeats=50,
                                            shap_plot=True)

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
    Step 1: Basic demographics
    Get what we need to report the standard things we do
    """
    df_demo = demographics_descriptives(analysis_dict=analysis_dict, save_path=save_path)

    """
    Step 2: Expertise
    We collected self-reported expertise levels with various topics. Get what we need to report about that
    returns: the df with all subjects and just the rating columns of 4 experience types(not the 'other' responses etc)
    """
    df_exp_ratings, exp_path = experience_descriptives(analysis_dict=analysis_dict, save_path=save_path)

    """
    Step 3: Can consciousness be separated from intentions/valence? 
    Answers to the "Do you think a creature/system can have intentions/consciousness/sensations w/o having..?" section
    df_c_groups contains a row per subject and focuses on answers to the Consciousness-wo-... questions, contraining
    the answers (y/n) to both (c wo intentions, c wo valence), and the group tagging based on that
    """
    df_c_groups, ics_path = ics_descriptives(analysis_dict=analysis_dict, save_path=save_path)

    """
    Step 4: Do the groups of people from the ics_descriptives differ based on experience with consciousness?
    Note that for this question we BINARIZED experience with consciousness into non-experts (ratings 1-2-3), and 
    experts (ratings 4-5). 
    
    Because of that, we check for relationships between two categorical parameters: 
    ics group belonging, and expertise (binarized). 
    
    Therefore, we need to do a chi square test of independence (and not kruskal), as we test whether the proportions
    of consciousness experts (1 vs. 0) differ significantly across different ics groups.  
    """
    # Consciousness_expert is a binary column where 0 = people who rated themselves < EXPERTISE and 1 otherwise
    #perform_chi_square(df1=df_c_groups, col1="group",
    #                   df2=df_exp_ratings, col2="Consciousness_expert", col2_name="Consciousness Expertise",
    #                   id_col=process_survey.COL_ID,
    #                   save_path=ics_path, save_name="consciousness_exp", save_expected=False,
    #                   grp1_vals=ICS_GROUP_ORDER_LIST, grp2_vals=[0, 1], grp2_map={0: "Non Expert", 1: "Expert"},
    #                   grp1_color_dict={ICS_GROUP_ORDER_LIST[i]: ICS_GROUP_COLOR_LIST[i] for i in range(len(ICS_GROUP_ORDER_LIST))})

    """
    Step 5: Relationship between consciousness and intelligence
    """
    df_c_i, c_i_path = consc_intell_descriptives(analysis_dict=analysis_dict, save_path=save_path)

    """
    Step 6: Does the perceived relationship between consciousness and intelligence depend on demographic factors
    (e.g., age) or expertise (e.g., with AI or with animals?) 
    """
    consc_intell_RF(df_demographics=df_demo, df_experience=df_exp_ratings, df_con_intell=df_c_i, save_path=c_i_path)

    if load:  # load the earth-in-danger things
        df_earth_clusters = pd.read_csv(os.path.join(save_path, "earth_danger", f"earth_danger_clusters.csv"))

    else:
        df_earth_clusters, kmeans, cluster_centroids = earth_in_danger_clustering(analysis_dict, save_path)
