import os
import pandas as pd
import re
import numpy as np
from functools import reduce
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


def other_creatures(analysis_dict, save_path, sort_together=True):
    # save path
    result_path = os.path.join(save_path, "c_v_ms")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # load relevant data
    df_ms = analysis_dict["other_creatures_ms"]
    df_c = analysis_dict["other_creatures_cons"]
    df = pd.merge(df_c, df_ms, on=[process_survey.COL_ID])

    df.to_csv(os.path.join(result_path, "c_v_ms.csv"), index=False)

    # codes and relevant stuff
    items = survey_mapping.other_creatures_general  # all rated items
    topic_name_map = {"c": "Consciousness", "ms": "Moral Status"}

    # melt to a long format
    long_data = pd.melt(df, id_vars=[process_survey.COL_ID], var_name="Item_Topic",
                        value_name="Rating")
    long_data[["Topic", "Item"]] = long_data["Item_Topic"].str.split('_', expand=True)
    long_data = long_data.drop(columns=["Item_Topic"])
    long_data = long_data[[process_survey.COL_ID, "Topic", "Item", "Rating"]]
    long_data["Topic"] = long_data["Topic"].map(topic_name_map)

    # add some demographic numerical columns
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

    result_df = long_data
    for dataframe in [animal_experience_df, ai_experience_df, ethics_experience_df, con_experience_df, demos_df]:
        result_df = pd.merge(result_df, dataframe, on=[process_survey.COL_ID])

    result_df.to_csv(os.path.join(result_path, "c_v_ms_long.csv"), index=False)

    # plot aggregated data

    result_df["non_human_animal"] = result_df["Item"].map(survey_mapping.other_creatures_isNonHumanAnimal)

    for exp in ["exp_animals", "exp_ai", "exp_ethics", "exp_consc"]:
        for topic in ["Consciousness", "Moral Status"]:
            top_df = result_df[result_df["Topic"] == topic]
            exp_name = exp.split("_")[-1]
            plotter.plot_density(df=top_df, x_col="Rating", x_col_name=f"{topic} Rating",
                                 hue_col=exp, hue_col_name=f"Experience with {exp_name}",
                                 save_name=f"{exp}_{topic.lower()}", save_path=result_path)

    # turn categorical columns into numeric ones for linear modelling
    for col in ["Topic", "Item"]:
        label_encoder_country = LabelEncoder()
        result_df[col] = label_encoder_country.fit_transform(result_df[col])

    result_df.to_csv(os.path.join(result_path, "c_v_ms_long_coded.csv"), index=False)

    """
    Look at the ratings individually for each item 
    """

    rating_color_list = ["#DB5461", "#fb9a99", "#70a0a4", "#26818B"]
    rating_labels = ["Does Not Have", "Probably Doesn't Have", "Probably Has", "Has"]

    sorting_method = None

    for topic_code, topic_name in topic_name_map.items():  # Consciousness, Moral Status
        df_topic = df.loc[:, [col for col in df.columns if col.startswith(f"{topic_code}_")]]
        df_topic.columns = df_topic.columns.str.replace(f'^{topic_code}_', '', regex=True)  # get rid of topix prefix
        df_topic.columns = df_topic.columns.str.replace(r'^.*? ', '', regex=True).str.title()  # get rid of "A ..."
        items_sansA = [s.split(' ', 1)[-1].title() if ' ' in s else s for s in items]  # do the same for items

        # Prepare data for each column in the dataframe, they represent the items
        stats = {}
        for col in items_sansA:
            stats[col] = helper_funcs.compute_stats(df_topic[col])

        # Create DataFrame for plotting
        plot_data = {}
        for item, (proportions, mean_rating, std_dev) in stats.items():
            plot_data[item] = {
                'Proportion': proportions,
                'Mean': mean_rating,
                'Std Dev': std_dev
            }

        # Define plot size and number of subplots
        num_plots = len(df_topic.columns)
        # rating_color_list = ["#e31a1c", "#fb9a99", "#a6cee3", "#1f78b4"]

        if sorting_method is None:
            # Sort the data by the proportion of "4" rating (Python 3.7+ dictionaries maintain the insertion order of keys)
            # sorted_plot_data = sorted(plot_data.items(), key=lambda x: x[1]['Proportion'].get(4, 0), reverse=True)

            # Sort the data by the MEAN rating (Python 3.7+ dictionaries maintain the insertion order of keys)
            sorted_plot_data = sorted(plot_data.items(), key=lambda x: x[1]['Mean'], reverse=True)
            if sort_together:  # if false, then it'll sort each of them by the previous condition independently
                sorting_method = list(dict(sorted_plot_data).keys())  # this is the order now

        else:  # sort the second column by the first one's order
            sorted_plot_data = {key: plot_data[key] for key in sorting_method if key in plot_data}.items()

        # plot
        plotter.plot_stacked_proportion_bars(plot_data=sorted_plot_data, num_plots=num_plots, legend=rating_labels,
                                             colors=rating_color_list, num_ratings=4, title=f"{topic_name.title()}",
                                             save_path=result_path, save_name=f"{topic_name.lower()}_ratings")

    """
    Plot "other creatures" judgments of Consciousness vs. of Moral Status. >> PER ITEM
    """

    # prepare data for analyses
    df_pivot = long_data.pivot_table(index="Item", columns="Topic", values="Rating", aggfunc="mean").fillna(
        0).reset_index(drop=False, inplace=False)

    colors = [rating_color_list[0], rating_color_list[-1]]
    individual_data = long_data.pivot_table(index=["response_id", "Item"], columns="Topic",
                                            values="Rating").reset_index(drop=False, inplace=False)
    plotter.plot_scatter_xy(df=df_pivot, identity_col="Item",
                            x_col="Consciousness", x_label="Consciousness", x_min=1, x_max=4, x_ticks=1,
                            y_col="Moral Status", y_label="Moral Status", y_min=1, y_max=4, y_ticks=1,
                            save_path=result_path, save_name=f"correlation_c_ms", annotate_id=True,
                            palette_bounds=colors, format="png", corr_line=True,
                            individual_df=individual_data, id_col="response_id")

    """
    Cluster people based on their rating tendencies of different entities' consciousness and moral status >> PER PERSON
    """
    df_nosub = df.iloc[:, 1:]  # only rating cols

    label_maps = {**survey_mapping.MS_RATINGS, **survey_mapping.C_RATINGS}

    pca_df = helper_funcs.perform_PCA(df_pivot=df_nosub, save_path=result_path, save_name="people", components=2,
                                      clusters=2,
                                      label_map=label_maps, binary=False, threshold=2.5)
    pca_df[process_survey.COL_ID] = df.iloc[:, 0]

    demog_df = analysis_dict["demographics"]

    unified_df = pd.merge(pca_df, demog_df, on=process_survey.COL_ID)
    unified_df.to_csv(os.path.join(result_path, f"people_PCA_result_with_demographics.csv"), index=False)

    """
    CONNECTION TO GRADED CONSCIOUSNESS
    connect between the consciousness and moral status scores, and people's beliefs about graded consciousness
    """

    # load the relevant data  # TODO: STOPPED HERE
    df_graded = analysis_dict["consciousness_graded"]
    x = 3

    """
    CORRELATION ANALYSIS
    Calculate for each item separately the correlation between its consciousness rating and its moral status rating. 
    """
    helper_funcs.corr_per_item(df=df, items=items, save_path=result_path)

    """
    LCA ANALYSIS
    Latent Class Analysis (LCA) on the ratings of consciousness and moral status. 
    """
    # PCA (plotted with cluster analysis)
    # helper_funcs.perform_PCA(df_pivot=df_pivot, save_path=result_path)

    df.drop(columns=["response_id"], inplace=True)  # no need for participants' IDs at this point
    for col in df.columns.tolist():
        df[col] = df[col].astype(int)
    helper_funcs.lca_analysis(df=df, n_classes=3, save_path=result_path)

    return


def earth_in_danger(analysis_dict, save_path):
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

    df_earth_coded = df_earth.copy()
    for col in questions:
        """
        The map should convert values into 0/1s for the PCA and Kmeans clustering. 
        PCA is a technique that works with numeric data to capture variance. 
        It projects the data into lower dimensions based on linear combinations of the features, so we can't use 
        categorizations (and map them to numbers as the model might falsely interpret these numeric encodings as 
        having ordinal relationships. 
        
        The same goes for Kmeans clustering, it relies on distance measures like Euclidean distance to determine 
        cluster centers, so if using arbitrary numbers for categories, the model will interpret these numbers as having 
        some sort of distance relationship. 
        
        So we will convert everything into binary. However, in order to keep interpretability, I will choose the
        0's and 1's myself (and not simple map each column into binary arbitratily. 
        """
        col_map = survey_mapping.EARTH_DANGER_QA_MAP[col]
        df_earth_coded[col] = df_earth_coded[col].map(col_map)

    df_earth_coded.set_index([process_survey.COL_ID], inplace=True)
    pca_df = helper_funcs.perform_PCA(df_pivot=df_earth_coded, save_path=result_path, save_name="items",
                                      components=2, clusters=2, label_map=survey_mapping.EARTH_DANGER_QA_MAP)
    pca_df.reset_index(drop=False, inplace=True)

    # in the pca_df we also get the results of the KMeans clustering we performed.
    # then, we want to examine if there are any demographics that are shared within each cluster.
    df_zombie = analysis_dict["zombification_pill"]
    df_zombie["Would you take the pill?"] = df_zombie["Would you take the pill?"].map({"Yes": 1, "No": 0})
    df_demog = analysis_dict["demographics"]
    df_animalexp = analysis_dict["animal_exp"]
    df_ethicsexp = analysis_dict["ethics_exp"]
    df_aiexp = analysis_dict["ai_exp"]
    df_cexp = analysis_dict["consciousness_exp"]
    df_list = [df_demog, df_animalexp, df_ethicsexp, df_aiexp, df_cexp, df_zombie]
    unified_df = reduce(lambda x, y: x.merge(y, on=process_survey.COL_ID), df_list)
    unified_df_cluster = pd.merge(unified_df, pca_df[[process_survey.COL_ID, "Cluster"]],
                                  on=process_survey.COL_ID, how="left")
    unified_df_cluster.rename(columns=survey_mapping.Q_EXP_DICT, inplace=True)
    unified_df_cluster.to_csv(os.path.join(result_path, "clusters_with_demographic.csv"), index=False)

    import seaborn as sns

    overlap = unified_df_cluster.groupby(["Cluster", "Would you take the pill?"]).size().unstack(fill_value=0)
    print("Count of overlap:\n", overlap)
    count_A1_B1 = len(unified_df_cluster[
                          (unified_df_cluster["Cluster"] == 1) & (unified_df_cluster["Would you take the pill?"] == 1)])
    count_A1_B0 = len(unified_df_cluster[
                          (unified_df_cluster["Cluster"] == 1) & (unified_df_cluster["Would you take the pill?"] == 0)])
    count_A0_B0 = len(unified_df_cluster[
                          (unified_df_cluster["Cluster"] == 0) & (unified_df_cluster["Would you take the pill?"] == 0)])
    count_A0_B1 = len(unified_df_cluster[
                          (unified_df_cluster["Cluster"] == 0) & (unified_df_cluster["Would you take the pill?"] == 1)])

    only_A = len(unified_df_cluster[
                     (unified_df_cluster["Cluster"] == 1) & (unified_df_cluster["Would you take the pill?"] == 0)])
    only_B = len(unified_df_cluster[
                     (unified_df_cluster["Cluster"] == 0) & (unified_df_cluster["Would you take the pill?"] == 1)])
    both_A_and_B = len(unified_df_cluster[(unified_df_cluster["Cluster"] == 1) & (
                unified_df_cluster["Would you take the pill?"] == 1)])
    only_neither = len(unified_df_cluster[(unified_df_cluster["Cluster"] == 0) & (
                unified_df_cluster["Would you take the pill?"] == 0)])
    print(f"Rows where A=1 and B=0: {only_A}")
    print(f"Rows where A=0 and B=1: {only_B}")
    print(f"Rows where A=1 and B=1: {both_A_and_B}")
    print(f"Rows where A=0 and B=0: {only_neither}")

    total_rows = len(unified_df_cluster)
    proportions = pd.DataFrame({
        'Pill=No': [only_A / total_rows, only_neither / total_rows],
        'Pill=Yes': [both_A_and_B / total_rows, only_B / total_rows]
    }, index=['Cluster=1', 'Cluster=0'])
    sns.heatmap(proportions, annot=True, cmap='Blues', cbar=True, fmt='.2f')

    for q in questions:
        df_q = df_earth.loc[:, [process_survey.COL_ID, q]]
        counts = df_q[q].value_counts()
        plotter.plot_pie(categories_names=counts.index.tolist(), categories_counts=counts.tolist(),
                         categories_labels=CAT_LABEL_DICT,
                         categories_colors=CAT_COLOR_DICT, title=f"{q}",
                         save_path=result_path, save_name=f"{'_'.join(counts.index.tolist())}", format="png")
    return


def ics(analysis_dict, save_path):
    """
    Answers to the "Do you think a creature/system can have intentions/consciousness/sensations w/o having..?" section
    """
    # save path
    result_path = os.path.join(save_path, "i_c_s")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # load relevant data
    df_ics = analysis_dict["ics"]
    questions = [c for c in df_ics.columns if c.startswith("Do you think a creature/system")]
    for q in questions:
        df_q = df_ics.loc[:, [process_survey.COL_ID, q]]
        counts = df_q[q].value_counts()
        q_name = q.replace('Do you think a creature/system', '')[:-1]
        q_name = q_name.replace('can be', '')
        q_name = q_name.replace('can have', '')
        q_name = q_name.replace('/', '-')
        plotter.plot_pie(categories_names=counts.index.tolist(), categories_counts=counts.tolist(),
                         categories_labels=CAT_LABEL_DICT,
                         categories_colors=CAT_COLOR_DICT, title=f"{q}",
                         save_path=result_path, save_name=q_name, format="png")
    return


def kill_for_test(analysis_dict, save_path):
    """
    Answers to the "Do you think a creature/system can have intentions/consciousness/sensations w/o having..?" section
    """
    # save path
    result_path = os.path.join(save_path, "kill_for_test")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # load relevant data
    df_test = analysis_dict["important_test_kill"]
    # all the options for killing (scenarios)
    questions = [c for c in df_test.columns if c.startswith("A creature/system that")]
    for q in questions:
        q_name = survey_mapping.important_test_kill_tokens[q]
        df_q = df_test.loc[:, [process_survey.COL_ID, q]]
        counts = df_q[q].value_counts()
        labels = {l: l.replace(' to pass the test', '') for l in counts.index.tolist()}
        colors = {l: CAT_COLOR_DICT[l.split(' ', 1)[0]] for l in counts.index.tolist()}
        plotter.plot_pie(categories_names=counts.index.tolist(), categories_counts=counts.tolist(),
                         categories_labels=labels,
                         categories_colors=colors, title=f"{q}",
                         save_path=result_path, save_name=q_name, format="png")

    # those who answered all "No's"
    all_nos = df_test[df_test[
        "You wouldn't eliminate any of the creatures; why?"].notnull()]  # all people who answered "No" to ALL options
    all_nos_prop = 100 * all_nos.shape[0] / df_test.shape[0]
    rest_prop = 100 * (df_test.shape[0] - all_nos.shape[0]) / df_test.shape[0]
    cat_names = ["Won't kill any", "Kill at least one"]
    cat_counts = [all_nos_prop, rest_prop]
    cat_colors = {"Won't kill any": "#E06C6E", "Kill at least one": "#5288A3"}
    plotter.plot_pie(categories_names=cat_names, categories_counts=cat_counts,
                     categories_colors=cat_colors, title=f"Would kill in any of the scenarios",
                     save_path=result_path, save_name="all_nos_prop", format="png")

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

    # category_counts = all_nos["You wouldn't eliminate any of the creatures; why?"].value_counts()

    plotter.plot_pie(categories_names=category_counts.index.tolist(), categories_counts=category_counts.tolist(),
                     categories_colors=colors, title=f"You wouldn't eliminate any of the creatures; why?",
                     save_path=result_path, save_name="all_nos_reason", format="png")

    return


def zombie_pill(analysis_dict, save_path):
    """
    Answers to the question about whether they would take a zombification pill.
    """
    # save path
    result_path = os.path.join(save_path, "zombie_pill")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # load relevant data
    df_zombie = analysis_dict["zombification_pill"]
    category_counts = df_zombie["Would you take the pill?"].value_counts()
    plotter.plot_pie(categories_names=category_counts.index.tolist(), categories_counts=category_counts.tolist(),
                     categories_colors=CAT_COLOR_DICT, title=f"Would you take the pill?",
                     save_path=result_path, save_name="take_the_pill", format="png")

    return


def moral_consideration_features(analysis_dict, save_path):
    """
    Answers to the question about which features they think are important for moral considerations
    """
    # save path
    result_path = os.path.join(save_path, "moral_consideration_features")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    colors = {survey_mapping.ANS_LANG: "#002642",
              survey_mapping.ANS_SENS: "#840032",
              survey_mapping.ANS_SENTIENCE: "#E59500",
              survey_mapping.ANS_PLAN: "#E5DADA",
              survey_mapping.ANS_SELF: "#FE6D73",
              survey_mapping.ANS_PHENOMENOLOGY: "#FFCB77",
              survey_mapping.ANS_THINK: "#227C9D",
              survey_mapping.ANS_OTHER: "#02040F"
              }

    ms_features = analysis_dict["moral_considerations_features"]

    ms_features_copy = ms_features[[process_survey.COL_ID, "What do you think is important for moral considerations?"]]

    def create_feature_dummies(row):
        return {feature: 1 if feature in row else 0 for feature in survey_mapping.ALL_FEATURES}

    dummies_df = ms_features_copy["What do you think is important for moral considerations?"].apply(create_feature_dummies).apply(pd.Series)

    """
    this is the proportion of selecting each category NORMALIZED by the TOTAL NUMBER of RESPONSES (i.e., the number
    of options all participants marked). A X% in this means that X% of all responses were [this] feature. 
    [this basically treats subjects as making single-selections, splitting each multi-selection subject into the 
    corresponding number of "single" dummy subjects, so that we can compare it to the "most important feature" below]
    """
    # proportion several = how many of all people selected a given feature
    proportions_several = (dummies_df.mean() * 100).to_frame(name="Proportion_all").reset_index(drop=False, inplace=False)
    proportions_several = proportions_several.sort_values("Proportion_all", ascending=False).reset_index(drop=True,
                                                                                                         inplace=False)
    category_order = proportions_several["index"].tolist()

    # if participants selected only one to begin with, we didn't ask them to select which they think is the most important
    # see it here :
    # filtered_data = ms_features[ms_features["Which do you think is the most important for moral considerations?"].isna()]
    ms_features["Which do you think is the most important for moral considerations?"] = ms_features[
        "Which do you think is the most important for moral considerations?"]. \
        fillna(ms_features["What do you think is important for moral considerations?"])
    # now after we have the most important, plot it
    most_important = ms_features[
        [process_survey.COL_ID, "Which do you think is the most important for moral considerations?"]]
    """
    what the below means, is counting the proportions of selecting a single feature. Note that these are amts, 
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

    # diff = out of all the people who selected feature X as *one* of the important features,
    # how many didn't select it as *the most* important one = the bigger it is, the more people
    # who selected it did not select it as THE most important
    df_diff = pd.DataFrame()
    df_diff["index"] = category_order
    df_diff["Proportion_diff"] = proportions_several["Proportion_all"] - proportions_one["Proportion_one"]

    df_unified = reduce(lambda left, right: pd.merge(left, right, on=["index"],
                                                     how="outer"), [proportions_several, proportions_one, df_diff])

    colors = ["#FFBF00", "#F47F38", "#E83F6F", "#855A8A", "#546798", "#2274A5", "#2A848A", "#32936F", "#99C9B7"]
    plotter.plot_categorical_bars_layered(categories_prop_df=df_unified, category_col="index",
                                          full_data_col="Proportion_all", partial_data_col="Proportion_one",
                                          categories_colors=colors, save_path=result_path,
                                          save_name="important_features", format="png", y_min=0, y_max=101,
                                          y_skip=10, inch_w=20, inch_h=12)
    df_unified.to_csv(os.path.join(result_path, "important_features.csv"), index=False)
    return


def moral_considreation_prios(analysis_dict, save_path):
    """
    Answers to the question about whether different creatures deserve moral considerations
    """
    # save path
    result_path = os.path.join(save_path, "moral_consideration_prios")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    ms_prios = analysis_dict["moral_considerations_prios"]
    questions = [c for c in ms_prios.columns if c.startswith("Do you think")]
    for q in questions:
        df_q = ms_prios.loc[:, [process_survey.COL_ID, q]]
        category_counts = df_q[q].value_counts()
        plotter.plot_pie(categories_names=category_counts.index.tolist(), categories_counts=category_counts.tolist(),
                         categories_colors=CAT_COLOR_DICT, title=f"{q}",
                         save_path=result_path, save_name=f"{q.replace('?', '').replace('/', '-')}", format="png")

    reasons = [c for c in ms_prios.columns if c not in questions]
    for r in reasons:
        df_r = ms_prios.loc[:, [process_survey.COL_ID, r]]
        df_r = df_r[df_r[r].notnull()]
        df_r.to_csv(os.path.join(result_path, f"{r.replace('?', '').replace('/', '-')}.csv"), index=False)
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

    """
    Plot the answers to the rating questions (agreement) in a stacked bar plot
    """
    rating_color_list = ["#DB5461", "#fb9a99", "#70a0a4", "#26818B"]
    rating_labels = ["1", "2", "3", "4"]
    rating_questions = [survey_mapping.Q_GRADED_EQUAL, survey_mapping.Q_GRADED_UNEQUAL, survey_mapping.Q_GRADED_INCOMP]
    stats = {}
    for col in rating_questions:
        stats[col] = helper_funcs.compute_stats(c_graded[col])
    # Create DataFrame for plotting
    plot_data = {}
    for item, (proportions, mean_rating, std_dev) in stats.items():
        plot_data[item] = {
            'Proportion': proportions,
            'Mean': mean_rating,
            'Std Dev': std_dev
        }
    # Sort the data by the MEAN rating (Python 3.7+ dictionaries maintain the insertion order of keys)
    sorted_plot_data = sorted(plot_data.items(), key=lambda x: x[1]['Mean'], reverse=True)
    plotter.plot_stacked_proportion_bars(plot_data=sorted_plot_data, num_plots=3, legend=rating_labels,
                                         colors=rating_color_list, num_ratings=4, title=f"How Much do you Agree?",
                                         save_path=result_path, save_name=f"consciousness_graded_ratings",
                                         text_width=39)

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
    df_graded_exp_filtered[df_graded_exp_filtered[survey_mapping.Q_GRADED_MATTERMORE] == survey_mapping.ANS_YES].shape[
        0]
    yes_prop = 100 * (yes_count / not_equal_count)
    no_count = \
    df_graded_exp_filtered[df_graded_exp_filtered[survey_mapping.Q_GRADED_MATTERMORE] == survey_mapping.ANS_NO].shape[0]
    no_prop = 100 * (no_count / not_equal_count)

    df_graded_extra = pd.DataFrame({"N": [all_count], "N_interestQ": [not_equal_count],
                                    "Yes_interestQ": [yes_count], "Yes_interestQ_prop": [yes_prop],
                                    "No_interestQ": [no_count], "No_interestQ_prop": [no_prop]})
    df_graded_extra.to_csv(os.path.join(result_path, "graded_experience_interests.csv"))

    """
    Relations to variability in Consciousness Ratings
    """
    other_creatures_c = analysis_dict["other_creatures_cons"]
    other_creatures_c["c_ratings_variability"] = other_creatures_c.iloc[:, 1:].std(axis=1)
    std_df = other_creatures_c[[process_survey.COL_ID, "c_ratings_variability"]]
    merged_df = pd.merge(c_graded, std_df, on=process_survey.COL_ID)
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

    common_denominator = "What is the common denominator?"
    con_intellect_d = con_intellect[con_intellect[common_denominator].notnull()]
    con_intellect_d.to_csv(os.path.join(result_path, "common_denominator.csv"), index=False)

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

    category_counts = [employment_counts[employment] if employment in employment_counts else 0 for employment in employment_order]

    plotter.plot_pie(categories_names=employment_order,
                     categories_counts=category_counts,  #[employment_counts[employment] for employment in employment_order]
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
        df_topic_meanPerSub = df_topic.groupby(["response_id"], as_index=False)["Rating"].mean()
        df_topic_meanPerSub_crossed = pd.merge(df_topic_meanPerSub, gender_df, on=process_survey.COL_ID, how='left')
        plotter.plot_scatter(df=df_topic_meanPerSub_crossed, data_col="Rating", category_col=gender,
                             category_order=gender_order, category_color_dict=gender_color_dict, title_text=f"{gender}",
                             x_label="", y_label=f"Mean {topic} Rating", vertical_jitter=0,
                             y_min=1, y_max=4, y_skip=1, save_path=result_path, save_name=f"gender_{topic.lower()}")
        df_topic.to_csv(os.path.join(result_path, f"gender_{topic.lower()}.csv"), index=False)

    return


def experience(analysis_dict, save_path):
    ethics = analysis_dict["ethics_exp"]
    animals = analysis_dict["animal_exp"]
    ai = analysis_dict["ai_exp"]
    consciousness = analysis_dict["consciousness_exp"]

    # save path
    result_path = os.path.join(save_path, "experience")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    """
    Does experience with animals affect answers related to animal C / MS ? 
    """

    animals = animals.apply(helper_funcs.replace_animal_other, axis=1)  # replace "Other" if categories do exist

    # first of all, general animal info
    animal_q = "On a scale from 1 to 5 where 1 means 'none' and 5 means 'extremely', how would you rate your level of interaction or experience with animals?"

    stats = {animal_q: helper_funcs.compute_stats(animals[animal_q])}
    plot_data = {}
    for item, (proportions, mean_rating, std_dev) in stats.items():
        plot_data[item] = {
            'Proportion': proportions,
            'Mean': mean_rating,
            'Std Dev': std_dev
        }
    sorted_plot_data = {key: plot_data[key] for key in list(dict(plot_data).keys()) if key in plot_data}.items()
    rating_labels = ["1 (None)", "2", "3", "4", "5 (Extremely)"]
    rating_color_list = ["#E7E7E7", "#B7CED0", "#87B4B9", "#569BA2", "#26818B"]
    topic_name = "experience with animals"
    plotter.plot_stacked_proportion_bars(plot_data=sorted_plot_data, num_plots=1, legend=rating_labels,
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
                     categories_counts=category_counts,  #[animal_counts[animal] for animal in animal_order]
                     categories_colors=animal_colors, title=f"Animal Experience (3+)", edge_color="none",
                     pie_direction=180, annot_groups=True, annot_group_selection=substantial_list, annot_props=False,
                     save_path=result_path, save_name="exp_animal_types", format="png")

    return


def relationship_across(sub_df, analysis_dict, save_path):
    # save path
    result_path = os.path.join(save_path, "clustering")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    """
    Ordinal
    """

    # get ms columns
    ms_creature_cols = list(survey_mapping.other_creatures_ms.values())
    ms_cols = [c for c in analysis_dict["other_creatures_ms"].columns.tolist() if c in ms_creature_cols]

    # get c columns
    c_creature_cols = list(survey_mapping.other_creatures_cons.values())
    c_cols = [c for c in analysis_dict["other_creatures_cons"].columns.tolist() if c in c_creature_cols]

    # map education columns
    education_q = "What is your education background?"
    sub_df[education_q] = sub_df[education_q].map(survey_mapping.EDU_MAP)

    # ordinal columns
    ordinal_cols = ms_cols + c_cols + [education_q, survey_mapping.Q_AI_EXP,
                                       survey_mapping.Q_ANIMAL_EXP,
                                       survey_mapping.Q_CONSC_EXP,
                                       survey_mapping.Q_ETHICS_EXP]

    """
    Categorical 
    """
    gender = "How do you describe yourself?"
    country = "In which country do you currently reside?"
    most_important = "What do you think is important for moral considerations?"

    cat_cols = [gender, country, most_important]
    for col in cat_cols:
        label_encoder_country = LabelEncoder()
        sub_df[col] = label_encoder_country.fit_transform(sub_df[col])

    # earth in danger
    earth_cols = [c for c in analysis_dict["earth_in_danger"].columns.tolist() if "A" in c]
    for c in earth_cols:
        sub_df[c] = sub_df[c].map(survey_mapping.EARTH_DANGER_MAP)

    cat_cols.extend(earth_cols)

    """
    Binary
    """

    # get kill columns
    kill_cols = [c for c in analysis_dict["important_test_kill"].columns.tolist() if "A creature/system" in c]
    for c in kill_cols:
        sub_df[c] = sub_df[c].map(survey_mapping.ANS_KILLING_MAP)

    moral_prios_cols = [c for c in analysis_dict["moral_considerations_prios"].columns.tolist() if "Do you think" in c]
    moral_prios_cols.remove("Do you think conscious creatures/systems should be taken into account in moral decisions?")
    for c in moral_prios_cols:
        sub_df[c] = sub_df[c].map(survey_mapping.ANS_YESNO_MAP)

    intellect_col = "Do you think consciousness and intelligence are related?"
    zombie_col = "Would you take the pill?"
    other_bin_cols = [intellect_col, zombie_col]
    for c in other_bin_cols:
        sub_df[c] = sub_df[c].map(survey_mapping.ANS_YESNO_MAP)

    bin_cols = kill_cols + moral_prios_cols + other_bin_cols

    helper_funcs.perform_kmodes(df=sub_df,
                                numeric_cols=["How old are you?"],
                                ordinal_cols=ordinal_cols,
                                categorical_cols=cat_cols,
                                binary_cols=bin_cols,
                                save_path=result_path)
    return


def analyze_survey(sub_df, analysis_dict, save_path):
    """
    The method which manages all the processing of specific survey data for analyses.
    :param sub_df: the dataframe of all participants' responses
    :param analysis_dict: dictionary where key=topic, value=a dataframe containing all the columns relevant for this
    topic/section
    :param save_path: where the results will be saved (csvs, plots)
    """
    other_creatures(analysis_dict, save_path)
    graded_consciousness(analysis_dict, save_path)
    relationship_across(sub_df, analysis_dict, save_path)
    gender_cross(analysis_dict, save_path)  # move to after the individuals
    demographics(analysis_dict, save_path)
    experience(analysis_dict, save_path)
    zombie_pill(analysis_dict, save_path)
    earth_in_danger(analysis_dict, save_path)  # MUST COME AFTER "zombie_pill"
    ics(analysis_dict, save_path)
    kill_for_test(analysis_dict, save_path)
    moral_consideration_features(analysis_dict, save_path)
    moral_considreation_prios(analysis_dict, save_path)
    consciousness_intelligence(analysis_dict, save_path)

    return
