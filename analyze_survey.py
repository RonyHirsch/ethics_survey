import os
import pandas as pd
import numpy as np
import plotter
import process_survey
import survey_mapping

CAT_COLOR_DICT = {"Person": "#AF7A6D",  #FAE8EB
                  "Dog": "#CCC7B9",  #6EA4BF
                  "My pet": "#E2D4BA",
                  "Dictator (person)": "#AF7A6D",  #4C191B
                  "Person (unresponsive wakefulness syndrome)": "#AF7A6D",
                  "Fruit fly (a conscious one, for sure)": "#CACFD6",
                  "AI (that tells you that it's conscious)": "#074F57",
                  ###
                  "Yes": "#355070",
                  "No": "#B26972"  # #590004, #461D02
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


def other_creatures(analysis_dict, save_path):
    """
    Compare "other creatures" judgments of Consciousness vs. of Moral Status.
    """
    # save path
    result_path = os.path.join(save_path, "c_v_ms")
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # load relevant data
    df_ms = analysis_dict["other_creatures_ms"]
    df_c = analysis_dict["other_creatures_cons"]
    df = pd.merge(df_c, df_ms, on=[process_survey.COL_ID, process_survey.COL_DUR_SEC])
    df.to_csv(os.path.join(result_path, "c_v_ms.csv"), index=False)

    # melt to a long format
    long_data = pd.melt(df, id_vars=[process_survey.COL_ID, process_survey.COL_DUR_SEC], var_name="Item_Topic",
                        value_name="Rating")
    long_data[["Topic", "Item"]] = long_data["Item_Topic"].str.split('_', expand=True)
    long_data = long_data.drop(columns="Item_Topic")
    long_data = long_data[[process_survey.COL_ID, process_survey.COL_DUR_SEC, "Topic", "Item", "Rating"]]
    long_data["Topic"] = long_data["Topic"].map({"c": "Consciousness", "ms": "Moral Status"})
    long_data.to_csv(os.path.join(result_path, "c_v_ms_long.csv"), index=False)

    # other creatures - Consciousness vs Moral status
    for item in survey_mapping.other_creatures_general:
        df_item = df.loc[:, [process_survey.COL_ID] + [col for col in df.columns if item in col]]
        plotter.plot_raincloud(df=df_item, id_col=process_survey.COL_ID,
                               data_col_names=[f"c_{item}", f"ms_{item}"],
                               data_col_colors={f"c_{item}": "#037171", f"ms_{item}": "#985F6F"},  # #D5A021, #93032E
                               x_title="", x_name_dict={f"c_{item}": "Consciousness", f"ms_{item}": "Moral Status"},
                               title=f"{item}", y_title="", ymin=1, ymax=4.04, yskip=1, y_jitter=0.05,
                               y_ticks=["Does not have", "Probably doesn't have", "Probably has", "Has"],
                               data_col_violin_left={f"c_{item}": True, f"ms_{item}": False}, scatter_lines=True,
                               save_path=result_path, save_name=f"{item}", format="png")
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
    df_earth = df_earth.drop(columns=[process_survey.COL_DUR_SEC])
    questions = df_earth.columns[df_earth.columns != process_survey.COL_ID].tolist()
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
    df_ics = df_ics.drop(columns=[process_survey.COL_DUR_SEC])
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
    df_test = df_test.drop(columns=[process_survey.COL_DUR_SEC])
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

    return


def analyze_survey(sub_df, analysis_dict, save_path):
    """
    :param sub_df:
    :param analysis_dict:
    :param save_path:
    :return:
    """
    other_creatures(analysis_dict, save_path)
    earth_in_danger(analysis_dict, save_path)
    ics(analysis_dict, save_path)
    kill_for_test(analysis_dict, save_path)

    return