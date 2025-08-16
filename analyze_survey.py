import os
import pandas as pd
import numpy as np
import re
from functools import reduce
from collections import defaultdict
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

EXP_BINARY_NAME_MAP = {0: "Non Expert", 1: "Expert"}
AGREE_BINARY_NAME_MAP = {0: "Disagree", 1: "Agree"}

ICS_GROUP_COLOR_LIST = ["#B53B03", "#ee9b00", "#005f73", "#3B4E58"]
ICS_GROUP_ORDER_LIST = ["multidimensional", "cognitive-agential", "experiential", "other"]

NO_KILL_COLOR_LIST = [YES_NO_COLORS[survey_mapping.ANS_NO] for i in range(len(survey_mapping.ANS_ALLNOS_LIST))]

EARTH_DANGER_COLOR_MAP = {survey_mapping.ANS_PERSON: "#ffbf69",
                          survey_mapping.ANS_DICTATOR: "#e8dab2",
                          survey_mapping.ANS_UWS: "#8d99ae",

                          survey_mapping.ANS_DOG: "#778da9",
                          survey_mapping.ANS_PET: "#415a77",

                          survey_mapping.ANS_FLY: "#917856",
                          survey_mapping.ANS_AI: "#ffd8be"}


EARTH_DANGER_CLUSTER_COLORS = ["#EDAE49", "#102E4A"]

C_V_MS_COLORS = {survey_mapping.ANS_C_MS_1: "#DB5461",
                 survey_mapping.ANS_C_MS_2: "#fb9a99",
                 survey_mapping.ANS_C_MS_3: "#70a0a4",
                 survey_mapping.ANS_C_MS_4: "#26818B"}

IMPORTANT_FEATURE_COLORS = {survey_mapping.ANS_LANG: "#2A848A",
                            survey_mapping.ANS_SENS: "#855A8A",
                            survey_mapping.ANS_SENTIENCE: "#F47F38",
                            survey_mapping.ANS_PLAN: "#546798",
                            survey_mapping.ANS_SELF: "#FFBF00",
                            survey_mapping.ANS_PHENOMENOLOGY: "#2274A5",
                            survey_mapping.ANS_THINK: "#E83F6F",
                            survey_mapping.ANS_OTHER: "#32936F"}


C_I_HOW_LABEL_MAP = {survey_mapping.ANS_C_NECESSARY: "C necessary for I",
                     survey_mapping.ANS_I_NECESSARY: "I necessary for C",
                     survey_mapping.ANS_SAME: "Same",
                     survey_mapping.ANS_THIRD: "Third feature",
                     survey_mapping.ANS_NO: survey_mapping.ANS_NO}
C_I_HOW_COLOR_MAP = {survey_mapping.ANS_C_NECESSARY: "#1a4e73",
                     survey_mapping.ANS_I_NECESSARY: "#346182",
                     survey_mapping.ANS_SAME: "#013a63",
                     survey_mapping.ANS_THIRD: "#4d7592",
                     survey_mapping.ANS_NO: YES_NO_COLORS[survey_mapping.ANS_NO]}
C_I_HOW_ORDER = [survey_mapping.ANS_NO, survey_mapping.ANS_SAME, survey_mapping.ANS_C_NECESSARY,
                 survey_mapping.ANS_I_NECESSARY, survey_mapping.ANS_THIRD]





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
    age_color_order = ["#590d22", "#800f2f", "#a4133c", "#c9184a", "#ff4d6d", "#ff758f", "#ff8fa3"]
    age_colors = {AGE_LABELS[i]: age_color_order[i] for i in range(len(age_color_order))}
    plotter.plot_categorical_bars(categories_prop_df=age_group_counts_df, category_col="age_group", data_col=PROP,
                                  categories_colors=age_colors,
                                  save_path=save_path, save_name=f"age", fmt="svg",
                                  y_min=0, y_max=101, y_skip=10, delete_y=False, inch_w=22, inch_h=12,
                                  layered=False, full_data_col=f"{PROP}", partial_data_col=None,
                                  add_pcnt=True, pcnt_color="#2C333A", pcnt_size=30, pcnt_position="middle")

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

    # relevant columns
    education_col = survey_mapping.Q_EDU
    education_order = survey_mapping.EDU_ORDER  # level of education (ordinal)
    education_topic = survey_mapping.Q_EDU_FIELD  # education field

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
    education_colors = {education_labels[education_order[i]]: education_color_order[i] for i in range(len(education_color_order))}
    plotter.plot_categorical_bars(categories_prop_df=category_df, category_col="education_label", data_col=PROP,
                                  categories_colors=education_colors,
                                  save_path=save_path, save_name=f"education", fmt="svg",
                                  y_min=0, y_max=101, y_skip=10, delete_y=False, inch_w=22, inch_h=12,
                                  layered=False, full_data_col=f"{PROP}_all", partial_data_col=None,
                                  add_pcnt=True, pcnt_color="#2C333A", pcnt_size=30, pcnt_position="middle")

    """
    Follow up on higher education - topic
    """
    # handle multiple selections
    education_field_df = demographics_df.copy()
    education_field_df[education_topic] = education_field_df[education_topic].dropna().astype(str).str.split(',')
    # explode into one row per selected field
    exploded_df = education_field_df.explode(education_topic)
    exploded_df[education_topic] = exploded_df[education_topic].str.strip()  # clean whitespace
    # count and proportion
    category_counts = exploded_df[education_topic].value_counts()
    category_props = exploded_df[education_topic].value_counts(normalize=True) * 100
    field_df = pd.DataFrame({
        education_topic: category_counts.index,
        COUNT: category_counts.values,
        PROP: category_props.values
    }).reset_index(drop=True)
    field_df = field_df.sort_values(by=PROP, ascending=False).reset_index(drop=True)  # descending proportion
    field_df.to_csv(os.path.join(save_path, "education_topic.csv"), index=False)

    demographics_df["education_level"] = demographics_df[education_col].map(survey_mapping.EDU_MAP)
    return demographics_df.loc[:, [process_survey.COL_ID, education_col, "education_level", education_topic]]


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

    """
    Pets
    """
    df_pet = analysis_dict["animal_exp"].loc[:, [process_survey.COL_ID, survey_mapping.Q_PETS]]

    """
    Merge
    """
    all_dfs = [df_age, df_gender, df_edu, df_employment, df_country, df_pet]
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
    # rating columns
    rating_cols = [c for c in survey_mapping.Q_EXP_NAME_DICT.keys()]

    # sources of reported experience
    source_cols = [survey_mapping.Q_CONSC_EXP_FOLLOW_UP,
                   survey_mapping.Q_ETHICS_EXP_FOLLOW_UP,
                   survey_mapping.Q_ANIMAL_EXP_FOLLOW_UP,  # note that the "source" of animal experience was an answer to the Q 'which animals'
                   survey_mapping.Q_AI_EXP_FOLLOW_UP]

    df_ratings = all_experience_df[[process_survey.COL_ID] + rating_cols + source_cols]
    df_ratings = df_ratings.rename(columns=survey_mapping.Q_EXP_NAME_DICT)  # rename EXPERIENCE for ease of use
    df_ratings = df_ratings.rename(columns=survey_mapping.Q_EXP_FOLLOWUP_DICT)  # rename EXPERIENCE SOURCE

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
                                         annotate_bar=True, annot_font_size=22,
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

    ics_questions = list(survey_mapping.ICS_Q_NAME_MAP.keys())
    df_ics_with_groups = pd.merge(df_ics.loc[:, [process_survey.COL_ID] + ics_questions], df_c_groups.loc[:, [process_survey.COL_ID, "group"]], on=process_survey.COL_ID)

    return df_c_groups, df_ics_with_groups, result_path


def perform_chi_square(df1, col1, df2, col2, id_col, save_path, save_name, save_expected=False,
                       grp1_vals=None, grp2_vals=None, grp2_map=None,
                       grp1_color_dict=None, y_tick_list=None, col2_name=None, contingency_back=False):
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

    # ensure we're on the same playing field - unify based on id_column
    merged = pd.merge(df1[[id_col, col1]], df2[[id_col, col2]], on=id_col)
    merged.to_csv(os.path.join(save_path, f"{save_name}_chisquared_data.csv"), index=False)
    # how many people do we have?
    num_obs = merged.shape[0]
    # create the contingency table
    contingency_table = pd.crosstab(merged[col1], merged[col2])
    chisquare_result, expected_df = helper_funcs.chi_squared_test(contingency_table=contingency_table,
                                                                  n=num_obs,
                                                                  include_expected=True)
    chisquare_result["question1"] = col1
    chisquare_result["question2"] = col2

    chisquare_result["question1_counts"] = str(contingency_table.sum(axis=1).to_dict())
    chisquare_result["question2_counts"] = str(contingency_table.sum(axis=0).to_dict())

    chisquare_result.to_csv(os.path.join(save_path, f"{save_name}_chisquared.csv"), index=False)
    if save_expected:
        expected_df.to_csv(os.path.join(save_path, f"{save_name}_chisquared_expected.csv"), index=False)

    # plot
    counts = merged.groupby([col2, col1]).size().unstack(fill_value=0)
    grouped_props = counts.div(counts.sum(axis=1), axis=0) * 100
    plot_ready_df = grouped_props.reset_index()
    if not y_tick_list:
        y_tick_list = [0, 25, 50, 75, 100]
    if not grp2_map:
        if all(isinstance(item, str) for item in grp2_vals):  # if all items in the list are strings, we need ints for the X axis
            grp2_map = {grp2_vals[i]: i for i in range(len(grp2_vals))}
        else:
            grp2_map = {grp2_vals[i]: grp2_vals[i] for i in range(len(grp2_vals))}
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
    if contingency_back:
        return chisquare_result, contingency_table
    return chisquare_result


def perform_kruskal_wallis(df_ordinal, ordinal_col, df_group, group_col, id_col, save_path, save_name):
    """
    Kruskal–Wallis Test: It's a non-parametric alternative to one-way ANOVA.
    Compares more than 2 independent groups (in group_col), with an ordinal data (ordinal_col)
    (Null hypothesis: all groups come from the same distribution)

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
    """
    Answers to the question about the perceived relationship between consciousness and intelligence.
    :param analysis_dict: dictionary where key=topic, value=a dataframe containing all the columns relevant for this
    topic/section
    :param save_path: where the results will be saved (csvs, plots)
    :return: the df with the data from which the descriptives were extracted, and the path where the results are saved
    """
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
    follow_up = survey_mapping.Q_INTELLIGENCE_HOW
    yes_counts = con_intellect[con_intellect[question] == survey_mapping.ANS_YES][follow_up].value_counts()
    no_count = (con_intellect[question] == survey_mapping.ANS_NO).sum()
    specific_category_counts = pd.concat([yes_counts, pd.Series({survey_mapping.ANS_NO: no_count})])
    specific_category_counts = specific_category_counts.reindex(C_I_HOW_ORDER)
    plotter.plot_pie(categories_names=specific_category_counts.index.tolist(), pie_direction=0,
                     categories_counts=specific_category_counts.tolist(),
                     categories_colors=C_I_HOW_COLOR_MAP, categories_labels=C_I_HOW_LABEL_MAP,
                     title=f"{question}", label_inside=True, prop_fmt=".0f",
                     save_path=result_path, save_name=f"{question.replace('?', '').replace('/', '-')}_how", fmt="svg")
    df_counts = specific_category_counts.to_frame(name=COUNT)
    df_counts[PROP] = 100 * (df_counts[COUNT] / df_counts[COUNT].sum())
    yes_total = df_counts.loc[df_counts.index != survey_mapping.ANS_NO, COUNT].sum()
    df_counts[f"{PROP}_out_of_{survey_mapping.ANS_YES}"] = df_counts.apply(
        lambda row: 100 * (row[COUNT] / yes_total) if row.name != survey_mapping.ANS_NO else None, axis=1)
    df_counts.to_csv(os.path.join(result_path, f"{question.replace('?', '').replace('/', '-')}_how.csv"), index=True)


    """
    Free reports: those who said consciousness and intelligence are related to a common third feature were asked what 
    it was. These are their answers. 
    """
    con_intellect_common_denominator = con_intellect[~con_intellect[survey_mapping.Q_INTELLIGENCE_FU].isna()]
    con_intellect_common_denominator.to_csv(os.path.join(result_path, "common_denominator.csv"), index=False)
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


def zombie_pill_descriptives(analysis_dict, save_path):
    """
    Answers to the question about taking a pheno-ectomy pill.
    :param analysis_dict: dictionary where key=topic, value=a dataframe containing all the columns relevant for this
    topic/section
    :param save_path: where the results will be saved (csvs, plots)
    :return: the df with the data from which the descriptives were extracted, and the path where the results are saved
    """
    # save path
    result_path = os.path.join(save_path, "zombie_pill")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # load relevant data
    df_zombie = analysis_dict["zombification_pill"].copy()
    zombie_q = survey_mapping.Q_ZOMBIE

    # plot pie chart
    category_counts = df_zombie[zombie_q].value_counts()
    category_props = df_zombie[zombie_q].value_counts(normalize=True)
    category_df = pd.DataFrame({
        zombie_q: category_counts.index,
        COUNT: category_counts.values,
        PROP: category_props.values * 100  # convert to percentage
    })
    category_df.to_csv(os.path.join(result_path, f"take_the_pill_{PROP}.csv"), index=False)

    """
    Plot
    """
    plotter.plot_pie(categories_names=category_counts.index.tolist(), categories_counts=category_counts.tolist(),
                     categories_colors=YES_NO_COLORS, title=f"{zombie_q}", pie_direction=180, edge_color="none",
                     save_path=result_path, save_name=f"take_the_pill_{PROP}", fmt="svg",
                     props_in_legend=True, annot_props=True, label_inside=True, annot_groups=True,
                     legend=False, legend_order=None, legend_vertical=True)

    return df_zombie.loc[:, [process_survey.COL_ID, zombie_q]], result_path


def kpt_descriptives(analysis_dict, save_path):
    """
    Answers to the block of questions about whether the participant would kill an entity to pass an important test,
    with six possible entities.
    :param analysis_dict: dictionary where key=topic, value=a dataframe containing all the columns relevant for this
    topic/section
    :param save_path: where the results will be saved (csvs, plots)
    :return: the df with the data from which the descriptives were extracted, and the path where the results are saved
    """
    # save path
    result_path = os.path.join(save_path, "kill_for_test")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    df_test = analysis_dict["important_test_kill"].copy()

    # pre-processing: extract and rename the binary kill/no kill questions
    kill_questions = list(survey_mapping.important_test_kill_tokens.keys())
    kill_questions_values = list(survey_mapping.important_test_kill_tokens.values())

    # apply binary conversion for the kill/no kill questions
    df_test_binary = df_test.copy()
    df_test_binary[kill_questions] = df_test_binary[kill_questions].replace({
        survey_mapping.ANS_KILL: survey_mapping.ANS_YES,
        survey_mapping.ANS_NOKILL: survey_mapping.ANS_NO})
    # rename questions into shorter, comprehensible codes
    df_test_binary = df_test_binary.rename(columns={q: survey_mapping.important_test_kill_tokens[q] for q in kill_questions})
    # binarize
    df_test_binary = df_test_binary.replace(survey_mapping.ANS_YESNO_MAP)

    # filter data: identify people who were not sensitive to the manipulation (kill all, not kill all)
    df_test_all_kill = df_test_binary[df_test_binary[kill_questions_values].eq(1).all(axis=1)]
    df_test_all_no_kill = df_test_binary[df_test_binary[kill_questions_values].eq(0).all(axis=1)]
    df_test_sensitive = df_test_binary[~df_test_binary.index.isin(df_test_all_kill.index.union(df_test_all_no_kill.index))]
    df_test_sensitive = df_test_sensitive.loc[:, [process_survey.COL_ID] + kill_questions_values]

    # calculate creature stats (kill vs. no kill) per scenario
    total_respondents = df_test_binary.shape[0]
    df_binary = df_test_binary.loc[:, kill_questions_values]  # just the question cols

    creature_stats = df_binary.apply(lambda col: pd.Series({
        f"kill_{COUNT}": int(col.sum()),
        f"no_kill_{COUNT}": int(total_respondents - col.sum()),
        f"kill_{PROP}": col.mean() * 100,
        f"no_kill_{PROP}": (1 - col.mean()) * 100
    }))
    creature_stats = creature_stats.reset_index().rename(columns={'index': 'Creature/System'})

    # behavior across entities
    all_kill = df_binary[df_binary.eq(1).all(axis=1)]
    all_no_kill = df_binary[df_binary.eq(0).all(axis=1)]
    rest = df_binary[~df_binary.index.isin(all_kill.index.union(all_no_kill.index))]
    summary_stats = pd.DataFrame({
        "Creature/System": ["All Kill", "All No-Kill", "Rest"],
        f"kill_{COUNT}": [all_kill.shape[0], all_no_kill.shape[0], rest.shape[0]],
        f"kill_{PROP}": [100 * all_kill.shape[0] / total_respondents,
                         100 * all_no_kill.shape[0] / total_respondents,
                         100 * rest.shape[0] / total_respondents]})
    summary_stats[f"no_kill_{COUNT}"] = total_respondents - summary_stats[f"kill_{COUNT}"]
    summary_stats[f"no_kill_{PROP}"] = 100 - summary_stats[f"kill_{PROP}"]

    final_df = pd.concat([creature_stats, summary_stats], ignore_index=True)
    final_df[[f"kill_{COUNT}", f"no_kill_{COUNT}"]] = final_df[[f"kill_{COUNT}", f"no_kill_{COUNT}"]].fillna(0).astype(int)
    final_df[[f"kill_{PROP}", f"no_kill_{PROP}"]] = final_df[[f"kill_{PROP}", f"no_kill_{PROP}"]].fillna(0)
    final_df.to_csv(os.path.join(result_path, "kill_to_pass_descriptives.csv"), index=False)

    """
    Plot
    """
    stats = {qname: helper_funcs.compute_stats(df_binary[qname], possible_values=[0, 1]) for qname in df_binary.columns}
    plot_data = {
        qname: {
            "Proportion": stat[0],
            "Mean": stat[1],
            "Std Dev": stat[2],
            "N": stat[3]
        }
        for qname, stat in stats.items()
    }
    sorted_plot_data = sorted(plot_data.items(), key=lambda x: x[1]["Mean"], reverse=True)
    df_yes_all = df_binary.apply(lambda row: all(row == 1), axis=1)
    yes_all_proportion = df_yes_all.sum() / total_respondents
    df_no_all = df_binary.apply(lambda row: all(row == 0), axis=1)
    no_all_proportion = df_no_all.sum() / total_respondents

    rating_labels = [survey_mapping.ANS_NO, survey_mapping.ANS_YES]
    rating_colors = [YES_NO_COLORS[survey_mapping.ANS_NO], YES_NO_COLORS[survey_mapping.ANS_YES]]
    plotter.plot_stacked_proportion_bars(plot_data=sorted_plot_data, legend_labels=rating_labels, ytick_visible=True,
                                         text_width=39, title="Would you kill to pass the test?", show_mean=False,
                                         sem_line=False, colors=rating_colors, num_ratings=2, annotate_bar=True,
                                         annot_font_color="#e0e1dd", save_path=result_path,
                                         save_name="kill_to_pass_discounted", fmt="svg", split=True,
                                         yes_all_proportion=yes_all_proportion, no_all_proportion=no_all_proportion)
    df_result = pd.DataFrame(sorted_plot_data)
    df_result.to_csv(os.path.join(result_path, f"kill_to_pass_discounted.csv"))

    """
    Follow up question on those who replied all 'No's (wouldn't kill any creature)
    """
    df_test_allnos = df_test[df_test.index.isin(all_no_kill.index)]

    # plot
    for item in survey_mapping.ANS_ALLNOS_LIST:
        df_test_allnos[item] = df_test_allnos[survey_mapping.Q_NO_KILL_WHY].apply(lambda x: int(item in x))

    df_test_allnos.to_csv(os.path.join(result_path, "all_nos_why_data.csv"), index=False)
    reason_cols = survey_mapping.ANS_ALLNOS_LIST

    category_counts = df_test_allnos[reason_cols].sum().reset_index()
    category_counts.columns = ["Answer", COUNT]
    category_counts[PROP] = 100 * category_counts[COUNT] / len(df_test_allnos)
    category_counts.to_csv(os.path.join(result_path, "all_nos_why.csv"), index=False)

    colors = {survey_mapping.ANS_ALLNOS_LIST[i]: NO_KILL_COLOR_LIST[i] for i in range(len(NO_KILL_COLOR_LIST))}
    plotter.plot_categorical_bars(categories_prop_df=category_counts, category_col="Answer", data_col=f"{PROP}",
                                  categories_colors=colors, save_path=result_path, save_name="all_nos_why", fmt="svg",
                                  y_min=0, y_max=100, y_skip=10, delete_y=False, inch_w=12, inch_h=15,
                                  order=survey_mapping.ANS_ALLNOS_LIST, text_wrap_width=15, x_label="Reason",
                                  y_fontsize=30, title_text=f"{survey_mapping.Q_NO_KILL_WHY}", flip=True, alpha=0.6,
                                  add_pcnt=True, pcnt_position="middle", pcnt_color="white", pcnt_size=30,
                                  y_tick_fontsize=25)

    return df_test, df_test_sensitive, result_path


def kpt_per_entity(kpt_df_sensitive, cgroups_df, save_path):
    """

    ** NOTE THAT WE ARE ASSUMING THAT PEOPLE WHO ARE INSENSITIVE TO THE MANIPULATION WERE FILTERED OUT **

    Transform the kill-to-pass-test dataframe so that it could be modeled per entity in the follwoing way:
    model <- glmer(kill ~ Consciousness * Intentions * Sensations + (1| response_id), data = data, family = binomial())
    The idea is to be able to test how the presence or absence of each property  (intentions, sensations, consciousness)
    affects the likelihood that its killed. For that, we need to transform our data.
    :param kpt_df_sensitive: The responses to this block of questions
    :param save_path: The path to save the transformed data to
    """

    # make sure we have only the relevant columns, and transform
    all_feature_cols = [col for col in kpt_df_sensitive.columns if set(kpt_df_sensitive[col].unique()) <= {0, 1}]
    relevant_columns = [process_survey.COL_ID] + all_feature_cols
    kpt_df = kpt_df_sensitive[relevant_columns]

    transformed_data = list()
    for index, row in kpt_df.iterrows():
        response_id = row[process_survey.COL_ID]
        for entity_index, entity_column in enumerate(all_feature_cols, start=1):
            entity_id = f"{entity_column}"
            consciousness = survey_mapping.Q_ENTITY_MAP[entity_column]["Consciousness"]
            intentions = survey_mapping.Q_ENTITY_MAP[entity_column]["Intentions"]
            sensations = survey_mapping.Q_ENTITY_MAP[entity_column]["Sensations"]
            kill = 1 if row[entity_column] == 1 else 0
            transformed_data.append([response_id, entity_id, consciousness, intentions, sensations, kill])
    transformed_df = pd.DataFrame(transformed_data, columns=[process_survey.COL_ID, "entity", "Consciousness",
                                                             "Intentions", "Sensations", "kill"])

    # save: NOTE AGAIN - THESE ARE ONLY THE PEOPLE WHO WERE SENSITIVE TO THE MANIPULATION
    transformed_df = pd.merge(transformed_df, cgroups_df.loc[:, [process_survey.COL_ID, "group"]], on=process_survey.COL_ID)
    transformed_df.to_csv(os.path.join(save_path, f"kill_to_pass_coded_per_entity_sensitive.csv"), index=False)

    return


def kpt_per_ics(kpt_df_sensitive, df_ics_groups, save_path):
    """
    This function connects between the block where we asked people about the possibility of having consciousness /
    sensations / intentions without the others (ICS), and the scenarios where they had a chance to kill a creature
    to pass an important test, with the creatures changing based on these dimensions (KPT).
    Note that the logic relies heavily on survey_mapping.scenario_mapping.
    The idea is the following:
    The ICS asks:
    1. Do you think someone can have C without I?
    2. Do you think someone can have C without S?
    3. Do you think someone can have I without C?
    4. Do you think someone can have S without C?
    So we have two pairs of questions: C-I relationship (1&3), and C-S relationship (2&4).

    The KPT asks: would you kill a creature that
    1. has C, without I or S → relevant: can C exist without I / without S?
    2. has I, without C or S → relevant: can I exist without C?
    3. has S, without C or I → relevant: can S exist without C?
    4. has C & I, without S → relevant: can C exist without S?
    5. has C & S, without I → relevant: can C exist without I?
    6. has I & S, without C → relevant: can I exist without C / can S exist without C?
    So for each scenario, we have a subset of questions (one or two) that are relevant.

    For each killing scenario, we will split people into two groups:
    (1) conceptually possible (answers to the RELEVANT ICS questions imply that this creature CAN exist);
    (2) conceptually impossible group (think that at least one of the RELEVANT ICS questions is NOT POSSIBLE).

    Then, we will perform a chi square, for each KPT scenario, asking whether killing behavior varies between people
    who think the creature is possible and those who don't.

    :param kpt_df_sensitive: df containing the id col and the six killing scenarios we presented,
    with the results (yes/no) already binarized (1/0) *** NOTE THAT WE ASSUME THIS DF CONTAINS ONLY PEOPLE WHO WERE
    SENSITIVE TO THE MANIPULATION ***
    :param df_ics_groups: df containing the id col and the four questions about the possibility of having
    consciousness / sensations / intentions without the others, already binarized (1/0)
    :param save_path: path to save the data to
    """
    # binarize ICS responses
    ics_cols = list(survey_mapping.ICS_Q_NAME_MAP.keys())
    df_ics_groups[ics_cols] = df_ics_groups[ics_cols].replace(survey_mapping.ANS_YESNO_MAP)

    # unify for saving
    unified_df = pd.merge(kpt_df_sensitive, df_ics_groups.loc[:, [process_survey.COL_ID] + ics_cols], on=process_survey.COL_ID)
    unified_df.to_csv(os.path.join(save_path, "kpt_sensitive_with_ics.csv"), index=False)

    # killing dict
    kill_dict = {1: "Kill", 0: "Won't Kill"}

    results = list()
    for kill_col, poss_cols in survey_mapping.scenario_mapping.items():
        # subset relevant columns for KILLING responses (KPT)
        df1 = unified_df[[process_survey.COL_ID, kill_col]].copy()
        # kill, no kill: for interpretation
        df1[kill_col] = df1[kill_col].map(kill_dict)
        # subset relevant columns for possibility responses (ICS)
        df2 = unified_df[[process_survey.COL_ID] + poss_cols].copy()
        # create a new column classifying participants as thinking the ICS scenario is "possible" or "not possible"
        df2["is_scenario_possible"] = df2[poss_cols].apply(lambda row: "Possible" if all(row == 1) else "Impossible", axis=1)

        result, contingency = perform_chi_square(df1=df1, col1=kill_col, grp1_vals=["Won't Kill", "Kill"],
                                                 grp1_color_dict={"Won't Kill": YES_NO_COLORS[survey_mapping.ANS_NO],
                                                                  "Kill": YES_NO_COLORS[survey_mapping.ANS_YES]},
                                                 df2=df2, col2="is_scenario_possible", grp2_vals=["Impossible", "Possible"],
                                                 col2_name=f"{kill_col}", # this is not a mistake, it's so that the X axis will describe the case
                                                 id_col=process_survey.COL_ID, save_path=save_path,
                                                 save_name=f"kpt_sensitive_{kill_col}_by_ics",
                                                 save_expected=False, contingency_back=True)
        contingency.to_csv(os.path.join(save_path, f"kpt_sensitive_{kill_col}_by_ics_contingency.csv"), index=True)
        results.append(result)

    result_df = pd.concat(results)
    result_df.to_csv(os.path.join(save_path, f"kpt_sensitive_by_ics_chisquared_all.csv"), index=False)

    return


def eid_descriptives(analysis_dict, save_path):
    """
    Answers to the block of questions about earth-in-danger (who would the participant save out of 2 options)
    :param analysis_dict: dictionary where key=topic, value=a dataframe containing all the columns relevant for this
    topic/section
    :param save_path: where the results will be saved (csvs, plots)
    :return: the df with the data from which the descriptives were extracted, and the path where the results are saved
    """
    # save path
    result_path = os.path.join(save_path, "earth_danger")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # load the relevant data
    df_earth = analysis_dict["earth_in_danger"]
    questions = list(survey_mapping.EARTH_DANGER_QA_MAP.keys())  # keys are the Q's, values are the A's
    df_earth.to_csv(os.path.join(result_path, f"eid_raw.csv"), index=False)

    # simple descriptives
    counts_df = df_earth[questions].apply(lambda col: col.value_counts())
    counts_df = counts_df.reset_index(drop=False, inplace=False).rename(columns={"index": "choice"}, inplace=False)
    counts_df.to_csv(os.path.join(result_path, "eid_counts.csv"), index=False)

    # replace options labels
    #counts_df.loc[:, "choice"] = counts_df["choice"].replace(survey_mapping.EARTH_DANGER_ANS_MAP)

    plot_data = list()
    for scenario in questions:
        counts = counts_df[["choice", scenario]].dropna(subset=[scenario])
        counts = counts.set_index("choice")[scenario]
        counts = counts.astype(float)
        total = counts.sum()
        proportions = {k: 100 * v / total for k, v in counts.items()}
        # enforce_ordering
        ordering = survey_mapping.EARTH_DANGER_QA_MAP[scenario]
        ordered_proportions = {k: proportions[k] for k, _ in sorted(ordering.items(), key=lambda x: x[1])}

        # dummy mean & std
        mean = np.average(list(ordering.values()), weights=list(counts.values))
        std = np.sqrt(np.average((np.array(list(ordering.values())) - mean) ** 2, weights=list(counts.values)))

        plot_data.append((scenario, {
            "Proportion": ordered_proportions,
            "Mean": mean,  # in this context, 'mean' is the proportion of '1' (the option presented on the right)
            "Std Dev": std,
            "N": int(total)
        }))

    plotter.plot_stacked_proportion_bars(plot_data=plot_data, legend_labels=survey_mapping.EARTH_DANGER_ANS_MAP,
                                         colors=EARTH_DANGER_COLOR_MAP, rating_col=None, item_cols=None,
                                         ytick_visible=True, text_width=35, title="Who Would You Save?",
                                         show_mean=False, sem_line=False, num_ratings=2, annotate_bar=True,
                                         annot_font_color="white", annot_font_size=22, save_path=result_path,
                                         save_name="eid_preferences", fmt="svg",
                                         double_ticks=True, double_ticks_bar_titles=False,
                                         ordering_map_dict=survey_mapping.EARTH_DANGER_QA_MAP)
    plot_df = pd.DataFrame(plot_data)
    plot_df.to_csv(os.path.join(result_path, f"eid_preferences.csv"), index=False)

    return df_earth, result_path


def eid_clustering(eid_df, save_path):

    # relevant question columns
    questions = eid_df.columns[eid_df.columns != process_survey.COL_ID].tolist()

    # code it for the sake of clustering
    df_earth_coded = eid_df.copy()
    for col in questions:
        """
        The map should convert values into 0/1s for the Kmeans clustering. 
        Kmeans clustering relies on distance measures like Euclidean distance to determine cluster centroids, 
        so if using arbitrary numbers for categories, the model will interpret these numbers as having some sort of 
        distance relationship. We don't want that (false ordinal relationship).

        So we will convert everything into binary. However, in order to keep interpretability, I will choose the
        0's and 1's myself (and not simple map each column into binary arbitrarily). 
        """
        col_map = survey_mapping.EARTH_DANGER_QA_MAP[col]
        df_earth_coded[col] = df_earth_coded[col].map(col_map)

    df_earth_coded.set_index([process_survey.COL_ID], inplace=True)

    """
    Perform k-means clustering: group the choices into (k) clusters based on feature similarity.
    Each cluster is represented by a "centroid" (average position of the data points in the cluster).
    Data points are assigned to the cluster whose centroid they are closest to.
    """
    #df_pivot, kmeans, cluster_centroids = helper_funcs.perform_kmeans(df_pivot=df_earth_coded, clusters=cluster_num,
    #                                                                  save_path=save_path, save_name="items")
    optimal_k, (df_pivot, kmeans, cluster_centroids), all_scores = helper_funcs.kmeans_optimal_k(df_pivot=df_earth_coded,
                                                                                                 save_path=save_path,
                                                                                                 save_name="items",
                                                                                                 k_range=range(2, 5))
    df_pivot.to_csv(os.path.join(save_path, f"earth_danger_clusters.csv"), index=True)  # index is participant code


    """
    Plot the KMeans cluster centroids:
    For each cluster (we have cluster_num clusters total), the centroid is the average data point for this cluster 
    (the mean value of the features for all data points in the cluster). 
    We use the centroids to visualize each cluster's choice in each earth-is-in-danger dyad, 
    to interpret the differences between them.  
    """

    # Compute the cluster centroids and SEMs
    # cluster_centroids = df_pivot.groupby("Cluster").mean()  # we get this from helper_funcs.perform_kmeans
    cluster_sems = df_pivot.groupby("Cluster").sem()

    # Plot - collapsed (all clusters together)
    helper_funcs.plot_cluster_centroids(cluster_centroids=cluster_centroids, cluster_sems=cluster_sems,
                                        save_path=save_path, save_name="items", fmt="svg",
                                        label_map=survey_mapping.EARTH_DANGER_QA_MAP, binary=True,
                                        label_names_coding=survey_mapping.EARTH_DANGER_ANS_MAP,
                                        threshold=0, overlaid=True, cluster_colors_overlaid=EARTH_DANGER_CLUSTER_COLORS)

    return df_pivot, kmeans, cluster_centroids


def c_v_ms(analysis_dict, save_path):
    """
    Prepare the data of consciousness ratings vs. moral status ratings for modelling in R.
    :param analysis_dict: dictionary where key=topic, value=a dataframe containing all the columns relevant for this
    topic/section
    :param save_path: where the results will be saved (csvs, plots)
    """

    # save path
    result_path = os.path.join(save_path, "c_v_ms")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # load relevant data
    df_ms = analysis_dict["other_creatures_ms"].copy()
    df_c = analysis_dict["other_creatures_cons"].copy()
    # merge
    df = pd.merge(df_c, df_ms, on=[process_survey.COL_ID])
    df.to_csv(os.path.join(result_path, "c_v_ms_raw.csv"), index=False)

    # melt to long
    long_data = pd.melt(df, id_vars=[process_survey.COL_ID], var_name="Item_Topic", value_name="Rating")
    long_data[["Topic", "Item"]] = long_data["Item_Topic"].str.split('_', expand=True)
    long_data = long_data.drop(columns=["Item_Topic"])
    long_data["Topic"] = long_data["Topic"].map({"c": "Consciousness", "ms": "Moral Status"})
    long_data.to_csv(os.path.join(result_path, "c_v_ms_long.csv"), index=False)

    """
    Plot
    """
    # we do NOT want to 'fillna' this -> and there should be no nas!
    df_pivot = long_data.pivot_table(index="Item", columns="Topic", values="Rating", aggfunc="mean").reset_index(drop=False, inplace=False)
    # rename for plotting; get rif of "A"/"An" etc.
    df_pivot["Item"] = df_pivot["Item"].replace(survey_mapping.other_creatures_general_names)

    plotter.plot_scatter_xy(df=df_pivot, identity_col="Item", annotate_id=True,
                            x_col="Consciousness", x_label="Consciousness",
                            x_min=survey_mapping.ANS_C_MS[survey_mapping.ANS_C_MS_1],
                            x_max=survey_mapping.ANS_C_MS[survey_mapping.ANS_C_MS_4], x_ticks=1,
                            y_col="Moral Status", y_label="Moral Status",
                            y_min=survey_mapping.ANS_C_MS[survey_mapping.ANS_C_MS_1],
                            y_max=survey_mapping.ANS_C_MS[survey_mapping.ANS_C_MS_4], y_ticks=1,
                            save_path=result_path, save_name=f"correlation_c_ms",
                            palette_bounds=[C_V_MS_COLORS[survey_mapping.ANS_C_MS_1], C_V_MS_COLORS[survey_mapping.ANS_C_MS_4]],
                            corr_line=False, diag_line=True, fmt="svg",
                            individual_df=None, id_col=None, color_col_colors=None)

    """
    Calculate off-diagonal differences per item (entity). 
    Compute differences & confidence intervals for each item, mark the off-digaonal items as the ones where the 
    entire CI does not include 0. This is basically a non-overlap with zero check - if it happens, it means that the 
    difference between moral status and consciousness is statistically significant at approximately the 0.05 level 
    (without explicitly computing a p-value).
    """

    # group and summarize
    item_diff = long_data.groupby(["Item", "Topic"]).agg( mean_rating=("Rating", "mean"),sd_rating=("Rating", "std"),
                                                          n=("Rating", "count")).reset_index()

    # pivot to wide format
    item_diff = item_diff.pivot(index="Item", columns="Topic", values=["mean_rating", "sd_rating", "n"])
    item_diff.columns = [f"{stat}_{topic}" for stat, topic in item_diff.columns]
    item_diff = item_diff.reset_index()

    # compute difference and confidence intervals
    item_diff["diff"] = item_diff["mean_rating_Moral Status"] - item_diff["mean_rating_Consciousness"]
    item_diff["se_diff"] = np.sqrt(
        (item_diff["sd_rating_Moral Status"] ** 2 / item_diff["n_Moral Status"]) +
        (item_diff["sd_rating_Consciousness"] ** 2 / item_diff["n_Consciousness"])
    )
    item_diff["lower"] = item_diff["diff"] - 1.96 * item_diff["se_diff"]
    item_diff["upper"] = item_diff["diff"] + 1.96 * item_diff["se_diff"]

    # off-diagonal flag (CI does not cross zero)
    item_diff["off_diagonal"] = (item_diff["lower"] > 0) | (item_diff["upper"] < 0)

    # direction
    item_diff["direction"] = np.where(item_diff["diff"] > 0, "above the diagonal", "below the diagonal")
    item_diff.to_csv(os.path.join(result_path, f"item_off_diagonal_differences.csv"), index=False)

    # for plotting (color)
    item_diff["color"] = np.select(
        [
            (item_diff["off_diagonal"]) & (item_diff["diff"] > 0),  # significant off-diagonal & positive
            (item_diff["off_diagonal"]) & (item_diff["diff"] < 0)  # significant off-diagonal & negative
        ],
        [1, -1],  # values for the conditions
        default=0  # if neither condition matches
    )

    # plot it
    colors_dict = {0: "#e5e5e5", 1: YES_NO_COLORS[survey_mapping.ANS_YES], -1: YES_NO_COLORS[survey_mapping.ANS_NO]}
    plotter.plot_item_differences_with_annotations(df=item_diff, id_col="Item", value_col="diff",
                                                   category_col="color", color_map=colors_dict,
                                                   save_path=result_path, save_name="item_off_diagonal_differences",
                                                   se=True, se_col="se_diff", alpha=1.0, annotate=False,
                                                   x_label="Difference", y_label="",
                                                   x_tick_size=22, y_tick_size=22, annotate_fontsize=9,
                                                   label_font_size=25,
                                                   y_ticks_label_map=survey_mapping.other_creatures_general_names,
                                                   plt_title="Off Diagonal", fmt="svg")

    """
    Entities that people though have no moral status but at the same time attributed them with some
    degree of consciousness.
    """

    entity_list = list(survey_mapping.other_creatures_general_names.keys())

    ms_1_result = list()
    c_1_result = list()
    for idx, row in df.iterrows():
        for entity in entity_list:
            c_col = f"c_{entity}"
            ms_col = f"ms_{entity}"
            if row[ms_col] == 1 and row[c_col] > 2:  # if they gave ms = 1 and consciousness > 2 = either "probably has" or "has"
                ms_1_result.append({
                    process_survey.COL_ID: row[process_survey.COL_ID],
                    "Item": entity,
                    "Consciousness": row[c_col]
                })
            if row[c_col] == 1 and row[ms_col] > 2:  # if they gave c = 1 and moral status > 3 = either "probably has" or "has"
                c_1_result.append({
                    process_survey.COL_ID: row[process_survey.COL_ID],
                    "Item": entity,
                    "Moral Status": row[ms_col]
                })

    ms_1_df = pd.DataFrame(ms_1_result)
    c_1_df = pd.DataFrame(c_1_result)

    # aggregate by entity (count how many people qualified)
    ms_1_entity_stats = ms_1_df.groupby("Item").agg(num_people=(process_survey.COL_ID, f"{COUNT}"),
                                                    avg_c_value=("Consciousness", "mean")).reset_index()
    # sort by count
    ms_1_entity_stats = ms_1_entity_stats.sort_values(by="num_people", ascending=False).reset_index(drop=True)
    ms_1_entity_stats.to_csv(os.path.join(result_path, f"ms_1_with_c_3-4_summary.csv"), index=False)

    # same for c
    c_1_entity_stats = c_1_df.groupby("Item").agg(num_people=(process_survey.COL_ID, f"{COUNT}"),
                                                  avg_ms_value=("Moral Status", "mean")).reset_index()
    c_1_entity_stats = c_1_entity_stats.sort_values(by="num_people", ascending=False).reset_index(drop=True)
    c_1_entity_stats.to_csv(os.path.join(result_path, f"c_1_with_ms_3-4_summary.csv"), index=False)

    """
    Plot a pie for each entity such that it'll have all c=1 / ms=1 as a 100%, and proportions of ms / c ratings 
    """
    colors_dict = {i: C_V_MS_COLORS[survey_mapping.ANS_C_MS_LABELS_REVERSED[i]] for i in range(1, 5)}

    # consciousness = 1 : moral status distribution
    # take top 5:
    c_1_top_5 = c_1_entity_stats["Item"].head(5).tolist()
    save_and_plot_pies(df=df, entities=c_1_top_5, filter_prefix="c", value_prefix="ms", result_path=result_path,
                       title_suffix="Moral Status (Consciousness = 1)", file_prefix="c1_ms", colors=colors_dict)

    # moral status = 1 : consciousness distribution
    ms_1_top_5 = ms_1_entity_stats["Item"].head(5).tolist()
    save_and_plot_pies(df=df, entities=ms_1_top_5, filter_prefix="ms", value_prefix="c", result_path=result_path,
                       title_suffix="Consciousness (Moral Status = 1)", file_prefix="ms1_c", colors=colors_dict)

    return long_data, df, result_path


def save_and_plot_pies(df, entities, filter_prefix, value_prefix, result_path, title_suffix, colors,
                       filter_val=1, file_prefix=""):
    for entity in entities:
        filter_col = f"{filter_prefix}_{entity}"
        value_col = f"{value_prefix}_{entity}"

        # filter for the condition
        relevant = df.loc[:, [process_survey.COL_ID, filter_col, value_col]]
        subset = relevant[relevant[filter_col] == filter_val][[value_col]]
        if subset.empty:
            continue

        # get counts & proportions
        counts = subset[value_col].value_counts().sort_index()
        total_n = counts.sum()
        props = counts / total_n * 100

        category_df = pd.DataFrame({
            "Rating": counts.index,
            f"{COUNT}": counts.values,
            f"{PROP}": props.values
        })

        category_df.to_csv(os.path.join(result_path, f"{file_prefix}_{entity}.csv"),index=False)

        # plot pie
        plotter.plot_pie(categories_names=counts.index.tolist(), categories_counts=counts.tolist(),
                         categories_colors=colors,
                         title=f"{title_suffix}: {entity} – N = {total_n}", save_path=result_path,
                         save_name=f"{file_prefix}_{entity}", fmt="svg", props_in_legend=True, annot_props=True,
                         label_inside=True)
    return


def kpt_per_demographics(kpt_df_sensitive, demographics_df, save_path):
    """
    :param kpt_df_sensitive: ASSUMING THIS DF CONTAINS ONLY PEOPLE SENSITIVE TO THE KPT MANIPULATION
    :param demographics_df: the df block of demographic data
    :param save_path: where the results will be saved (csvs, plots)
    """
    # identify the KPT questions (based on them being binary, kill/nokill)
    all_feature_cols = [col for col in kpt_df_sensitive.columns if set(kpt_df_sensitive[col].unique()) <= {0, 1}]
    # merge
    df_merged = pd.merge(kpt_df_sensitive, demographics_df, on=process_survey.COL_ID)
    # sum how many creatures would a person kill
    df_merged["kill_total"] = df_merged[all_feature_cols].sum(axis=1)
    df_merged.to_csv(os.path.join(save_path, "kpt_sensitive_per_demographics.csv"), index=False)

    # define the relevant demographic categories we are interested in checking
    relevant_demo_cols = {survey_mapping.Q_GENDER: "gender", "age_group": "ageGroup"}
    df_clean = df_merged[["kill_total"] + list(relevant_demo_cols.keys())].dropna()

    for col in relevant_demo_cols.keys():
        col_name = relevant_demo_cols[col]
        grouped_df = [
            group["kill_total"].dropna().values
            for _, group in df_clean.groupby(col)
            if not group["kill_total"].dropna().empty
        ]
        kruskal_df = helper_funcs.kruskal_wallis_test(grouped=grouped_df)
        kruskal_df.to_csv(os.path.join(save_path, f"kpt_sensitive_{col_name}_kruskal.csv"), index=False)
        summary_df = (
            df_clean.groupby(col)["kill_total"]
            .agg([f"{COUNT}", "mean", "std", "min", "max"])
            .assign(SE=lambda d: d["std"] / d[f"{COUNT}"] ** 0.5)
            .reset_index()
        )
        summary_df["CI_95_low"] = summary_df["mean"] - 1.96 * summary_df["SE"]
        summary_df["CI_95_high"] = summary_df["mean"] + 1.96 * summary_df["SE"]
        summary_df.to_csv(os.path.join(save_path, f"kpt_sensitive_{col_name}_summary.csv"), index=False)
    return


def c_v_ms_expertise(c_v_ms_df, df_experience, save_path, significant_p_value=0.05):
    """

    :param c_v_ms_df: dataframe with response_id and all the rating columns (c.., ms...)
    :param df_experience: dataframe containing the experience types (ratings) and the binarized expertise columns (_expert)
    :param save_path: path where to save the data to
    """
    rating_types = {"Consciousness": "c_", "Moral Status": "ms_"}
    expertise_rating_dict = {"AI": ["A large language model", "A self-driving car"],
                             "Animals": ["A cow", "A turtle", "An ant", "A dog",
                                         "A cat", "A lobster", "A sea urchin", "An octopus",
                                         "A salmon", "A bat", "A bee", "A mosquito",
                                         "A fruit-fly", "A rat", "A pigeon", "An orangutan"]}

    for expertise_domain in expertise_rating_dict:
        expert_col = f"{expertise_domain}_expert"  # as was defined in the expert - non expert split
        expertise_df = df_experience.loc[:, [process_survey.COL_ID, expert_col]]

        rated_items = expertise_rating_dict[expertise_domain]
        for rating_type in rating_types:
            rating_prefix = rating_types[rating_type]
            rated_cols = [f"{rating_prefix}{item}" for item in rated_items]
            relevant_df = c_v_ms_df.loc[:, [process_survey.COL_ID] + rated_cols]
            # expertise and relevant ratings
            merged_df = reduce(lambda left, right: pd.merge(left, right, on=[process_survey.COL_ID]), [relevant_df, expertise_df])

            # run Mann-Whitney U test
            mw_results, corrected_p_col = helper_funcs.run_group_mann_whitney(
                df=merged_df,
                comparison_cols=rated_cols,
                group_col=expert_col,
                group_col_name=expertise_domain,
                group1_val=survey_mapping.ANS_YESNO_MAP[survey_mapping.ANS_NO], group1_name="non-experts",
                group2_val=survey_mapping.ANS_YESNO_MAP[survey_mapping.ANS_YES], group2_name="experts"
            )
            mw_results.to_csv(os.path.join(save_path, f"{rating_prefix}{expertise_domain}_expertise_items.csv"), index=False)

            """
            Plot
            """
            # identify significant items
            significant_items = mw_results[mw_results[corrected_p_col] < significant_p_value]["Item"].tolist()
            ratings = sorted(list(survey_mapping.ANS_C_MS_LABELS.values()))  # 1 ,2, 3, 4
            rating_colors = {survey_mapping.ANS_C_MS_LABELS[label]: C_V_MS_COLORS[label] for label in list(survey_mapping.ANS_C_MS_LABELS.keys())}

            for item in significant_items:
                item_name = item.removeprefix(rating_prefix)  # Python 3.9+
                count_df = merged_df.groupby([expert_col, item]).size().reset_index(name=f"{COUNT}")
                count_df["total"] = count_df.groupby(expert_col)[f"{COUNT}"].transform("sum")
                count_df[f"{PROP}"] = 100 * count_df[f"{COUNT}"] / count_df["total"]
                pivot_df = count_df.pivot(index=expert_col, columns=item, values=f"{PROP}").fillna(0).reset_index()
                pivot_df.to_csv(os.path.join(save_path, f"{expertise_domain}_expertise_{item}.csv"), index=False)
                plotter.plot_expertise_proportion_bars(df=pivot_df, cols=ratings, cols_colors=rating_colors,
                                                       x_axis_exp_col_name=expert_col, x_map=EXP_BINARY_NAME_MAP,
                                                       x_label=f"Reported experience with {expertise_domain}",
                                                       y_ticks=[0, 25, 50, 75, 100],
                                                       save_name=f"{expertise_domain}_expertise_{item}",
                                                       save_path=save_path, plt_title=item_name,
                                                       annotate_bar=True, annot_font_color="white")
    return


def ms_per_ics(c_v_ms_df, df_ics_groups, save_path):
    """
    We want to answer the question: does being in group A mean you attribute MORAL STATUS to X differently?
    For that, we need to prepare the data for modelling in R.
    The modelling will be done PER ITEM, as otherwise we'd have a model with 24 entities * 4 groups which is hundreds
    of effects.

    :param c_v_ms_df:
    :param df_ics_groups:
    :param save_path:
    :return:
    """
    ms_df = c_v_ms_df.loc[:, [process_survey.COL_ID] + [c for c in c_v_ms_df.columns if c.startswith("ms_")]]  # just MS ratings
    merged = reduce(lambda left, right: pd.merge(left, right, on=[process_survey.COL_ID]),
                    [ms_df, df_ics_groups.loc[:, [process_survey.COL_ID, "group"]]])
    merged.to_csv(os.path.join(save_path, f"ms_per_ics.csv"), index=False)
    return


def eid_per_demographics(eid_df, demographics_df, experience_df, save_path, eid_cluster_df=None):

    expertise_relevant_item_dict = {"AI": [survey_mapping.Q_UWS_AI, survey_mapping.Q_AI_DOG]}
                                    #"Animals": [survey_mapping.Q_PERSON_DOG, survey_mapping.Q_PERSON_PET,
                                    #            survey_mapping.Q_DICTATOR_DOG, survey_mapping.Q_DICTATOR_PET,
                                    #            survey_mapping.Q_UWS_DOG, survey_mapping.Q_UWS_PET,
                                    #            survey_mapping.Q_UWS_FLY, survey_mapping.Q_AI_DOG]

    demographics_relevant_item_dict = {survey_mapping.Q_PETS: [survey_mapping.Q_PERSON_PET,
                                                               survey_mapping.Q_DICTATOR_PET,
                                                               survey_mapping.Q_UWS_PET]}

    for expertise_domain in expertise_relevant_item_dict:
        expert_col = f"{expertise_domain}_expert"  # as was defined in the expert - non expert split
        expertise_df = experience_df.loc[:, [process_survey.COL_ID, expert_col]]

        relevant_decisions = expertise_relevant_item_dict[expertise_domain]
        for decision in relevant_decisions:
            question_code = [k for k, v in survey_mapping.earth_in_danger.items() if v == decision][0]
            eid_df_relevant = eid_df.loc[:, [process_survey.COL_ID, decision]]
            # run a chi squared test
            perform_chi_square(df1=eid_df_relevant, col1=decision,
                               df2=expertise_df, col2=expert_col, col2_name=f"{expertise_domain} Expertise: {decision}",
                               grp2_vals=[0, 1], grp2_map=EXP_BINARY_NAME_MAP, id_col=process_survey.COL_ID,
                               save_path=save_path, save_name=f"eid_expertise_{expertise_domain}_{question_code}", save_expected=False,
                               grp1_vals=eid_df_relevant[decision].unique(),
                               grp1_color_dict=EARTH_DANGER_COLOR_MAP)

    for demo_domain in demographics_relevant_item_dict:
        demo_df = demographics_df.loc[:, [process_survey.COL_ID, demo_domain]]
        relevant_decisions = demographics_relevant_item_dict[demo_domain]
        for decision in relevant_decisions:
            question_code = [k for k, v in survey_mapping.earth_in_danger.items() if v == decision][0]
            eid_df_relevant = eid_df.loc[:, [process_survey.COL_ID, decision]]
            # run a chi squared test
            perform_chi_square(df1=eid_df_relevant, col1=decision,
                               df2=demo_df, col2=demo_domain, col2_name=f"{demo_domain}",
                               grp2_vals=[survey_mapping.ANS_NO, survey_mapping.ANS_YES],
                               id_col=process_survey.COL_ID,
                               save_path=save_path, save_name=f"eid_demographic_{demo_domain.replace('?', '').replace('/', '-')}_{question_code}", save_expected=False,
                               grp1_vals=eid_df_relevant[decision].unique(),
                               grp1_color_dict=EARTH_DANGER_COLOR_MAP)

    """
    If we have the cluster information, try to see if difference in expertise matters for that
    """
    if eid_cluster_df is not None:
        """
        Then we will run a random forest classifier to see if cluster (0/1) can be predicted from self-reported
        expertise level in any of the domains. >> we take the RAW form of experience here (for binarized, uncomment
        and replace)
        """
        experience_cols_raw = list(survey_mapping.Q_EXP_NAME_DICT.values())
        # experience_cols_binary = [c for c in experience_df.columns if c.endswith(f"_expert")]
        exp_relevant = experience_df.loc[:, [process_survey.COL_ID] + experience_cols_raw]
        cluster_relevant = eid_cluster_df.loc[:, [process_survey.COL_ID, "Cluster"]]
        cluster_with_exp = pd.merge(exp_relevant, cluster_relevant, on=process_survey.COL_ID)

        categorical_cols = []  # no categorical columns - self-reported experience is ordinal
        order_cols = experience_cols_raw

        cluster_with_exp.to_csv(os.path.join(save_path, f"model_dataframe_coded.csv"), index=False)
        helper_funcs.run_random_forest_pipeline(dataframe=cluster_with_exp, dep_col="Cluster",
                                                categorical_cols=categorical_cols,
                                                order_cols=order_cols, save_path=save_path, save_prefix="",
                                                rare_class_threshold=5, n_permutations=1000, scoring_method="accuracy",
                                                cv_folds=10, split_test_size=0.3, random_state=42, n_repeats=50,
                                                shap_plot=True, shap_plot_colors=EARTH_DANGER_CLUSTER_COLORS)


    return


def ms_c_prios_descriptives(analysis_dict, save_path):
    """
    Get general counts for answers in this block of questions, specifically counts for: PRIOS_Q_NAME_MAP
    ::param analysis_dict: dictionary where key=topic, value=a dataframe containing all the columns relevant for this
    topic/section
    :param save_path: where the results will be saved (csvs, plots)
    :return:
    """
    # save path
    result_path = os.path.join(save_path, "moral_consideration_prios")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    ms_prios = analysis_dict["moral_considerations_prios"].copy()
    ms_prios.to_csv(os.path.join(result_path, "moral_decisions_prios.csv"), index=False)

    """
    General descriptives
    """
    for question in list(survey_mapping.PRIOS_Q_NAME_MAP.keys()):  # yes/no questions about moral status priorities
        df_q = ms_prios.loc[:, [process_survey.COL_ID, question]]
        category_counts = df_q[question].value_counts()
        category_props = df_q[question].value_counts(normalize=True)
        category_df = pd.DataFrame({
            question: category_counts.index,
            COUNT: category_counts.values,
            PROP: category_props.values * 100  # convert to percentage
        })
        category_df.to_csv(os.path.join(result_path, f"{question.replace('?', '').replace('/', '-')}.csv"), index=False)
        plotter.plot_pie(categories_names=category_counts.index.tolist(), categories_counts=category_counts.tolist(),
                         categories_colors=YES_NO_COLORS, title=f"{question}", pie_direction=180,
                         edge_color="none", legend=False,
                         save_path=result_path, save_name=f"{question.replace('?', '').replace('/', '-')}", fmt="svg",
                         props_in_legend=True, annot_props=True, label_inside=True, annot_groups=True)


    """
    Follow up on reasons (the free-text examples)
    """
    reasons = [c for c in ms_prios.columns if c not in list(survey_mapping.PRIOS_Q_NAME_MAP.keys())]
    for r in reasons:
        df_r = ms_prios.loc[:, [process_survey.COL_ID, r]]
        df_r = df_r[df_r[r].notnull()]
        df_r.to_csv(os.path.join(result_path, f"{r.replace('?', '').replace('/', '-')}.csv"), index=False)

    return ms_prios, result_path


def ms_features_descriptives(analysis_dict, save_path):
    # save path
    result_path = os.path.join(save_path, "moral_consideration_features")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    ms_features = analysis_dict["moral_considerations_features"].copy()
    ms_features.to_csv(os.path.join(result_path, "moral_considerations_features.csv"), index=False)

    """
    Plot: descriptives on the important and MOST important features for moral considerations
    """
    # create dummies for counts
    ms_features_for_dummy = ms_features.loc[:, [process_survey.COL_ID, survey_mapping.Q_FEATURES_IMPORTANT]]

    def create_feature_dummies(row):
        return {feature: 1 if feature in row else 0 for feature in survey_mapping.ALL_FEATURES}
    dummies_df = ms_features_for_dummy[survey_mapping.Q_FEATURES_IMPORTANT].apply(create_feature_dummies).apply(pd.Series)
    """
    this is the proportion of selecting each category NORMALIZED by the TOTAL NUMBER of RESPONSES (i.e., the number
    of options all participants marked). A X% in this means that X% of all responses were [this] feature. 
    [this basically treats subjects as making single-selections, splitting each multi-selection subject into the 
    corresponding number of "single" dummy subjects, so that we can compare it to the "most important feature" below]
    """
    # proportion several = how many of all people selected a given feature
    proportions_several = (dummies_df.mean() * 100).to_frame(name=f"{PROP}_all").reset_index(drop=False,inplace=False)
    proportions_several = proportions_several.sort_values(f"{PROP}_all", ascending=False).reset_index(drop=True, inplace=False)
    category_order = proportions_several["index"].tolist()

    """
    If participants selected only one to begin with, we didn't ask them to select which they think is the most important
    see it by running this:
    filtered_data = ms_features[ms_features[survey_mapping.Q_FEATURES_MOST_IMPORTANT].isna()]
    """
    ms_features.loc[:, survey_mapping.Q_FEATURES_MOST_IMPORTANT] = ms_features[survey_mapping.Q_FEATURES_MOST_IMPORTANT].fillna(ms_features[survey_mapping.Q_FEATURES_IMPORTANT])
    # now  we have the most important
    most_important = ms_features.loc[:, [process_survey.COL_ID, survey_mapping.Q_FEATURES_MOST_IMPORTANT]]

    """
    what the below means is counting the proportions of selecting a single feature. 
    Note that these are amts, and we do not treat within-subject things here. 
    """
    # proportions_one  = how many of ALL people selected this feature as the MOST important one
    proportions_one = (most_important[survey_mapping.Q_FEATURES_MOST_IMPORTANT].value_counts(normalize=True) * 100
                       ).to_frame(name=f"{PROP}_one").reset_index(drop=False, inplace=False)
    proportions_one.rename(columns={survey_mapping.Q_FEATURES_MOST_IMPORTANT: "index"}, inplace=True)
    proportions_one["index"] = pd.Categorical(proportions_one["index"], categories=category_order, ordered=True)  # match order
    proportions_one = proportions_one.sort_values("index").reset_index(drop=True, inplace=False)

    """
    diff = out of all the people who selected feature X as *one* of the important features,
    how many didn't select it as *the most* important one = the bigger it is, the more people
    who selected it did not select it as THE most important
    """
    df_diff = pd.DataFrame()
    df_diff["index"] = category_order
    df_diff[f"{PROP}_diff"] = proportions_several[f"{PROP}_all"] - proportions_one[f"{PROP}_one"]

    df_unified = reduce(lambda left, right: pd.merge(left, right, on=["index"], how="outer"), [proportions_several, proportions_one, df_diff])
    df_unified.sort_values(by=[f"{PROP}_all"], ascending=False, inplace=True)  # sort by overall proportions
    df_unified.reset_index(drop=True, inplace=True)
    all_people = ms_features_for_dummy.shape[0]  # total number of people this was calculated on
    df_unified["N"] = all_people
    df_unified.to_csv(os.path.join(result_path, f"important_features.csv"), index=False)

    """
    Now, plot
    """
    plotter.plot_categorical_bars(categories_prop_df=df_unified, category_col="index",data_col=f"{PROP}_one",
                                  categories_colors=IMPORTANT_FEATURE_COLORS,
                                  save_path=result_path, save_name=f"important_features", fmt="svg",
                                  y_min=0, y_max=101, y_skip=10, delete_y=False, inch_w=22, inch_h=12,
                                  layered=True, full_data_col=f"{PROP}_all", partial_data_col=f"{PROP}_one",
                                  layered_alpha=0.4, add_pcnt=True, pcnt_color="#2C333A",pcnt_size=30,
                                  pcnt_position="top", layered_partial_pcnt_position="middle")

    return df_unified, most_important, result_path


def ms_phenomenology_experts(df_most_important, df_experience, save_path):
    df_most_important["is_phenomenology"] = df_most_important[survey_mapping.Q_FEATURES_MOST_IMPORTANT].apply(
        lambda x: survey_mapping.ANS_YES if x == survey_mapping.ANS_PHENOMENOLOGY else survey_mapping.ANS_NO)

    perform_chi_square(df1=df_most_important, col1="is_phenomenology",  # is phenomenology the most important feature
                       df2=df_experience, col2="Consciousness_expert", col2_name="Consciousness Expertise",
                       id_col=process_survey.COL_ID,
                       save_path=save_path, save_name="phenomenology_per_consciousness_exp", save_expected=False,
                       grp1_vals=[survey_mapping.ANS_NO, survey_mapping.ANS_YES], grp2_vals=[0, 1], grp2_map=EXP_BINARY_NAME_MAP,
                       grp1_color_dict=YES_NO_COLORS)
    return


def experience_with_demographics_descriptives(df_demographics, df_experience, save_path):
    """
    :param df_demographics: output of  demographics_descriptives()
    :param df_experience: output of  experience_descriptives()
    :param save_path: path where all the analysis results are saved
    """

    """
    Education based on reported expertise: cross self-reported expertise with education
    Note that as we didn't collect for experience with animals the source in terms of education (just which animals), 
    we do not check it for this experience type. 
    """
    education_academic = 4  # see survey_mapping.EDU_MAP: 4 and up is academic-level
    education_highschool = 3  # see survey_mapping.EDU_MAP
    experience_academic = survey_mapping.ANS_E_ACADEMIA_PREFIX

    cols_experience = [c for c in survey_mapping.Q_EXP_NAME_DICT.values() if c != "Animals"]
    cols_source = [c for c in df_experience.columns if "source" in c]

    # take only what's relevant for this crossing
    relevant_dfs = [df_demographics.loc[:, [process_survey.COL_ID, "education_level", survey_mapping.Q_EDU_FIELD]],
                    df_experience.loc[:, [process_survey.COL_ID] + cols_experience + cols_source]]
    # unify
    edu_exp_df = reduce(lambda left, right: pd.merge(left, right, on=[process_survey.COL_ID], how="outer"), relevant_dfs)
    edu_exp_df[cols_experience] = edu_exp_df[cols_experience].astype(int)  # to be able to check threshold

    # for each expertise type, take only experts, see if their claimed experience is acadmic, and if so, see their edu
    discrepancy_summary = list()
    for exp in cols_experience:
        print(f"{exp}")
        df_exp = edu_exp_df[edu_exp_df[exp] >= EXPERTISE]  # take only experts in this topic
        df_exp_num = df_exp.shape[0]
        # experience is declared to come (at least in part) from academia (could be other things as well)
        df_exp_academic = df_exp[df_exp[f"{exp}_source"].str.contains(experience_academic, case=False, na=False)]
        df_exp_academic_num = df_exp_academic.shape[0]
        # experience contains academia but they don't HOLD a degree already
        suspected_discrepancies = df_exp_academic[df_exp_academic["education_level"] < education_academic]
        suspected_discrepancies_num = suspected_discrepancies.shape[0]
        # but it could be that they are in academia rn, just don't hold a diploma yet. doesn't mean they lied
        # let's see how many don't hold a highschool diploma
        discrepancies = df_exp_academic[df_exp_academic["education_level"] < education_highschool]
        discrepancies_num = discrepancies.shape[0]
        # aggregate
        discrepancy_summary.append({
            "expertise_type": exp,
            "num_experts": df_exp_num,
            "num_experts_academia": df_exp_academic_num,
            "num_no_degree": suspected_discrepancies_num,
            "num_clear_discrepancies (no highschool diploma)": discrepancies_num})

    df_discrepancy_summary = pd.DataFrame(discrepancy_summary)
    # save
    df_discrepancy_summary.to_csv(os.path.join(save_path, f"experience_academic_against_education.csv"), index=False)
    return


def c_graded_descriptives(analysis_dict, save_path):
    # save path
    result_path = os.path.join(save_path, "graded_consciousness")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    c_graded = analysis_dict["consciousness_graded"].copy()
    c_graded.to_csv(os.path.join(result_path, "consciousness_graded.csv"), index=False)

    # all questions in this section
    rating_questions = [survey_mapping.Q_GRADED_EQUAL, survey_mapping.Q_GRADED_UNEQUAL, survey_mapping.Q_GRADED_INCOMP]

    df_melted = c_graded.melt(id_vars=process_survey.COL_ID, value_vars=rating_questions,
                              var_name='question', value_name='rating')
    counts = df_melted.groupby(['rating', 'question'])[process_survey.COL_ID].nunique().unstack(
        fill_value=0).reset_index(drop=False, inplace=False)
    counts.to_csv(os.path.join(save_path, "consciousness_graded_rating_counts.csv"), index=False)

    """
    Binarize agreement with each assertion: 1/2 = Disagree (0); 3/4 = Agree (1). 
    """
    agreed = [3, 4]
    for col in rating_questions:
        c_graded[f"binary_{col}"] = c_graded[col].isin(agreed).astype(int)

    return c_graded, result_path


def most_important_per_intelligence(df_most_important, df_con_intell, save_path):
    df_most_important["is_thinking_most_important"] = df_most_important[survey_mapping.Q_FEATURES_MOST_IMPORTANT].\
        apply(lambda x: 1 if x == survey_mapping.ANS_THINK else 0)

    result = perform_chi_square(df1=df_con_intell, col1=survey_mapping.Q_INTELLIGENCE, grp1_color_dict=YES_NO_COLORS,
                                grp1_vals=[survey_mapping.ANS_NO, survey_mapping.ANS_YES],
                                df2=df_most_important, col2="is_thinking_most_important",
                                col2_name="Was thinking selected as most important?",
                                grp2_vals=[0, 1], grp2_map=AGREE_BINARY_NAME_MAP,
                                id_col=process_survey.COL_ID, y_tick_list=None,
                                save_path=save_path, save_name=f"thinking most important_intelligence", save_expected=False)

    # now take only the ones who did think consciousness and intelligence were related:
    df_con_intell_yes = df_con_intell[df_con_intell[survey_mapping.Q_INTELLIGENCE] == survey_mapping.ANS_YES].copy().reset_index(drop=True, inplace=False)
    # and only those who also chose thinking as the most important feature
    df_con_intell_yes_thinking = df_con_intell_yes.merge(df_most_important.loc[df_most_important["is_thinking_most_important"] == 1,
    [process_survey.COL_ID, survey_mapping.Q_FEATURES_MOST_IMPORTANT]], on=process_survey.COL_ID, how="inner")
    df_con_intell_yes_thinking.to_csv(os.path.join(save_path, f"thinking most important yes_con intel related yes.csv"), index=False)
    # and now some stats on
    df_con_intell_yes_thinking_values = df_con_intell_yes_thinking[survey_mapping.Q_INTELLIGENCE_HOW].value_counts()
    plotter.plot_pie(categories_names=df_con_intell_yes_thinking_values.index.tolist(), pie_direction=0,
                     categories_counts=df_con_intell_yes_thinking_values.tolist(),
                     categories_colors=C_I_HOW_COLOR_MAP, categories_labels=C_I_HOW_LABEL_MAP,
                     title=f"{survey_mapping.Q_INTELLIGENCE}", label_inside=True, prop_fmt=".0f",
                     save_path=save_path, save_name=f"thinking most important yes_con intel related yes_how", fmt="svg")
    df_counts = df_con_intell_yes_thinking_values.to_frame(name=COUNT)
    df_counts[PROP] = 100 * (df_counts[COUNT] / df_counts[COUNT].sum())
    df_counts.to_csv(os.path.join(save_path, f"thinking most important yes_con intel related yes_how.csv"))

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

    #sub_df.to_csv(os.path.join(save_path, "sub_df.csv"), index=False)

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

    "Extra: Experience and Demographics - all descriptives that cross these two"
    #experience_with_demographics_descriptives(df_demographics=df_demo, df_experience=df_exp_ratings, save_path=exp_path)

    """
    Step 3: Can consciousness be separated from intentions/valence? 
    Answers to the "Do you think a creature/system can have intentions/consciousness/sensations w/o having..?" section
    df_c_groups contains a row per subject and focuses on answers to the Consciousness-wo-... questions, contraining
    the answers (y/n) to both (c wo intentions, c wo valence), and the group tagging based on that
    """
    df_c_groups, df_ics_with_groups, ics_path = ics_descriptives(analysis_dict=analysis_dict, save_path=save_path)

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
    perform_chi_square(df1=df_c_groups, col1="group",
                       df2=df_exp_ratings, col2="Consciousness_expert", col2_name="Consciousness Expertise",
                       id_col=process_survey.COL_ID,
                       save_path=ics_path, save_name="consciousness_exp", save_expected=False,
                       grp1_vals=ICS_GROUP_ORDER_LIST, grp2_vals=[0, 1], grp2_map=EXP_BINARY_NAME_MAP,
                       grp1_color_dict={ICS_GROUP_ORDER_LIST[i]: ICS_GROUP_COLOR_LIST[i] for i in range(len(ICS_GROUP_ORDER_LIST))})

    """
    Step 5: Relationship between consciousness and intelligence
    """
    df_c_i, c_i_path = consc_intell_descriptives(analysis_dict=analysis_dict, save_path=save_path)

    """
    Step 6: Does the perceived relationship between consciousness and intelligence depend on demographic factors
    (e.g., age) or expertise (e.g., with AI or with animals?) 
    """
    #consc_intell_RF(df_demographics=df_demo, df_experience=df_exp_ratings, df_con_intell=df_c_i, save_path=c_i_path)

    """
    Step 7: Examine the relationship between the conception of consciousness (from ICS groups) and the perceived 
    relationship between consciousness and intelligence: 
    Perform a chi square test
    """
    perform_chi_square(df1=df_c_i, col1=survey_mapping.Q_INTELLIGENCE, grp1_color_dict=YES_NO_COLORS,
                       grp1_vals=[survey_mapping.ANS_NO, survey_mapping.ANS_YES],
                       df2=df_c_groups, col2="group", col2_name="Conception of Consciousness",
                       grp2_vals=ICS_GROUP_ORDER_LIST, grp2_map=None,
                       id_col=process_survey.COL_ID, y_tick_list=None,
                       save_path=c_i_path, save_name=f"ics_intelligence", save_expected=False)


    """
    Step 8: Zombie pill dilemma. A variation of Siewert's pheno-ectomy thought experiment. 
    Descriptives
    """
    df_pill, pill_path = zombie_pill_descriptives(analysis_dict=analysis_dict, save_path=save_path)

    """
    Step 9: Do consciousness experts answer the Zombie pill question differently than non-experts?
    Logic is similar to step #4
    """
    # Consciousness_expert is a binary column where 0 = people who rated themselves < EXPERTISE and 1 otherwise
    perform_chi_square(df1=df_pill, col1=survey_mapping.Q_ZOMBIE,
                       df2=df_exp_ratings, col2="Consciousness_expert", col2_name="Consciousness Expertise",
                       id_col=process_survey.COL_ID,
                       save_path=pill_path, save_name="consciousness_exp", save_expected=False,
                       grp1_vals=[survey_mapping.ANS_NO, survey_mapping.ANS_YES],
                       grp2_vals=[0, 1], grp2_map=EXP_BINARY_NAME_MAP,
                       grp1_color_dict=YES_NO_COLORS)

    """
    Step 10: Do the different ICS groups answer the Zombie pill question differently from each other?
    Logic is similar to steps #4 and #9
    Zombie = binary (yes/no), consciousness group = 4 groups (df_c_groups)
    """
    perform_chi_square(df1=df_pill, col1=survey_mapping.Q_ZOMBIE,
                       df2=df_c_groups, col2="group", col2_name="Conception of Consciousness",
                       id_col=process_survey.COL_ID,
                       save_path=pill_path, save_name="ics_group", save_expected=False,
                       grp1_vals=[survey_mapping.ANS_NO, survey_mapping.ANS_YES],
                       grp2_vals=ICS_GROUP_ORDER_LIST, grp2_map=None,
                       grp1_color_dict=YES_NO_COLORS)

    """
    Step 11: Kill to Pass Test (KPT). A moral dilemma of 6 entities with I/C/S (ics), whether you'd kill them or not. 
    Descriptives
    *** NOTE *** df_kpt_sensitive includes ONLY PEOPLE WHO WERE SENSITIVE TO THE MANIPULATION - i.e., those who would
    kill at least one entity, but not all of them. 
    """
    df_kpt, df_kpt_sensitive, kpt_path = kpt_descriptives(analysis_dict=analysis_dict, save_path=save_path)

    """
    * Prepare data for modelling in R * 
    Step 12: Does the likelihood to kill a creature in the KPT scenarios change depending of its specific features? 
    (having I/C/S)?
    Prepare data for modelling (in R): Kill ~ Consciousness * Intentions * Sensations + (1|participant)
    
    Step 13: Does the likelihood to kill a creature in the KPT scenarios change depending on the conception of 
    consciousness? (ics group: df_c_groups)
    Prepare data for modelling (in R): Kill ~ Group + (1|participant) + (1|entity)
    
    *** AGAIN, ASSUMING ONLY PEOPLE WHO WERE SENSITIVE TO THE MANIPULATION ***
    """
    #kpt_per_entity(kpt_df_sensitive=df_kpt_sensitive, cgroups_df=df_c_groups, save_path=kpt_path)

    """
    Step 14: Is the KPT killing behavior affected by thinking that this creature is even possible? 
    
    *** AGAIN, ASSUMING ONLY PEOPLE WHO WERE SENSITIVE TO THE MANIPULATION ***
    """
    kpt_per_ics(kpt_df_sensitive=df_kpt_sensitive, df_ics_groups=df_ics_with_groups, save_path=kpt_path)


    """
    Step 15: Is the KPT killing behavior affected by demographics?
    
    *** AGAIN, ASSUMING ONLY PEOPLE WHO WERE SENSITIVE TO THE MANIPULATION ***
    """
    #kpt_per_demographics(kpt_df_sensitive=df_kpt_sensitive, demographics_df=df_demo, save_path=kpt_path)


    """
    Step 16: Lifeboat Ethics - Earth in Danger (EiD) Block. 
    In this block of questions, earth was in danger, with participants presented with dyads having to choose 
    who to save. 
    """
    df_eid, eid_path = eid_descriptives(analysis_dict=analysis_dict, save_path=save_path)

    """
    Step 18: EiD clusters
    can we cluster people based on their saving patterns into meaningful groups?
    use df_eid, and perform k-means clustering. 
    The function eid_clustering codes the data, prepares it for k-means, and searches for the OPTIMAL number of clusters
    Once it is found, it saves it and there's no need to run again. 
    """
    load=True
    if load:  # we load as this takes a while
        eid_clusters = pd.read_csv(os.path.join(eid_path, f"earth_danger_clusters.csv"))
    else:
        eid_clusters, kmeans, cluster_centroids = eid_clustering(eid_df=df_eid, save_path=eid_path)

    """
    Step 17: EiD per demographics. Is the EiD behavior affected by demographics?
    Similar logic to Step #15 
    """
    eid_per_demographics(eid_df=df_eid, demographics_df=df_demo, experience_df=df_exp_ratings, save_path=eid_path,
                         eid_cluster_df=None)

    """
    Step 18: EiD per consciousness conception group - ics - Do these groups belong to different CLUSTERS?
    Similar logic to Step #10 and  #7
    """
    perform_chi_square(df1=df_c_groups, col1="group",
                       df2=eid_clusters, col2="Cluster", col2_name="Cluster",
                       id_col=process_survey.COL_ID,
                       save_path=eid_path, save_name="ics_group", save_expected=False,
                       grp1_vals=ICS_GROUP_ORDER_LIST, grp2_vals=[0, 1], grp2_map=None,
                       grp1_color_dict={ICS_GROUP_ORDER_LIST[i]: ICS_GROUP_COLOR_LIST[i] for i in range(len(ICS_GROUP_ORDER_LIST))})

    """
    Step 19: consciousness vs moral status (C v MS) 
    In two separate blocks, we presented people with 24 entities (same entities) and asked them about their moral 
    status, and about their consciousness. 
    """
    df_c_v_ms_long, df_c_v_ms, c_v_ms_path = c_v_ms(analysis_dict=analysis_dict, save_path=save_path)

    """
    Step 20: C v MS expertise:
    Does expertise affect consciousness / moral status ratings?
    """
    #c_v_ms_expertise(c_v_ms_df=df_c_v_ms, df_experience=df_exp_ratings, save_path=c_v_ms_path)

    """
    Step 20: Does the conception of consciousness matter for MORAL STATUS ATTRIBUTIONS?
    This is for R script - function with comment 'MS rating per ICS group and Entity'
    """
    ms_per_ics(c_v_ms_df=df_c_v_ms, df_ics_groups=df_ics_with_groups, save_path=c_v_ms_path)

    """
    Step 21: moral consideration priorities: do you think non conscious creatures/systems should be taken into account 
    in moral decisions? And also for conscious creatures. 
    """
    df_ms_c_prios, prios_path = ms_c_prios_descriptives(analysis_dict=analysis_dict, save_path=save_path)


    """
    >> does expertise matter for the prios questions?: checking for Consciousness and Ethics expertise
    """
    for expertise in [survey_mapping.Q_EXP_NAME_DICT[survey_mapping.Q_CONSC_EXP],  # consciousness
                      survey_mapping.Q_EXP_NAME_DICT[survey_mapping.Q_ETHICS_EXP]]:  # ethics
        perform_chi_square(df1=df_ms_c_prios.loc[:, [process_survey.COL_ID, survey_mapping.PRIOS_Q_NONCONS]],
                           col1=survey_mapping.PRIOS_Q_NONCONS,
                           df2=df_exp_ratings, col2=f"{expertise}_expert", col2_name=f"{expertise} Expertise",
                           id_col=process_survey.COL_ID,
                           grp1_vals=[survey_mapping.ANS_NO, survey_mapping.ANS_YES], grp1_color_dict=YES_NO_COLORS,
                           grp2_vals=[0, 1], grp2_map=EXP_BINARY_NAME_MAP,
                           save_path=prios_path, save_name=f"{survey_mapping.PRIOS_Q_NAME_MAP[survey_mapping.PRIOS_Q_NONCONS]}_{expertise.lower()}_exp",
                           save_expected=False)

    """
    Step 22: moral consideration features: which features are most important for moral considerations? Descriptives
    """
    ms_features_df, most_important_df, ms_features_path = ms_features_descriptives(analysis_dict=analysis_dict,
                                                                                   save_path=save_path)

    """
    Step 23: Is there a difference between consciousness experts and non-experts with respect to selecting phenomenology
    as the most important feature for moral considerations?
    """
    ms_phenomenology_experts(df_most_important=most_important_df, df_experience=df_exp_ratings,
                             save_path=ms_features_path)


    """
    Step 23a: what is the relationship between  "thinking" as the most important feature 
    and thinking consciousness and intelligence are related?
    """
    #most_important_per_intelligence(df_most_important=most_important_df, df_con_intell=df_c_i, save_path=ms_features_path)


    """
    Step 24: Graded consciousness
    """
    df_c_graded, graded_path = c_graded_descriptives(analysis_dict=analysis_dict, save_path=save_path)

    """
    >> do people who agree more that some non-human animals should have a higher moral status than others answer
    differently on the assertion that consciousness is a graded phenomanon? 
    """
    for col in list(survey_mapping.Q_GRADED_NAMES.keys()):
        perform_chi_square(df1=df_ms_c_prios.loc[:, [process_survey.COL_ID, survey_mapping.PRIOS_Q_ANIMALS]],
                           col1=survey_mapping.PRIOS_Q_ANIMALS,
                           df2=df_c_graded, col2=f"binary_{col}",  # the BINARIZED agreement
                           col2_name=col,
                           id_col=process_survey.COL_ID,
                           grp1_vals=[survey_mapping.ANS_NO, survey_mapping.ANS_YES], grp1_color_dict=YES_NO_COLORS,
                           grp2_vals=[0, 1], grp2_map=AGREE_BINARY_NAME_MAP,  # 0 = disagree, 1 = agree
                           save_path=graded_path,
                           save_name=f"{survey_mapping.PRIOS_Q_NAME_MAP[survey_mapping.PRIOS_Q_ANIMALS]}_{survey_mapping.Q_GRADED_NAMES[col]}",
                           save_expected=False)


    exit()

