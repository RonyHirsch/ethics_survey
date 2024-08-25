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
    all_nos = df_test[df_test["You wouldn't eliminate any of the creatures; why?"].notnull()]  # all people who answered "No" to ALL options
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

    #category_counts = all_nos["You wouldn't eliminate any of the creatures; why?"].value_counts()

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
    df_zombie = df_zombie.drop(columns=[process_survey.COL_DUR_SEC])
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
    ms_features = ms_features.drop(columns=[process_survey.COL_DUR_SEC])
    questions = [c for c in ms_features.columns if c.startswith("What do you think") or c.startswith("Which do you think")]
    for q in questions:
        df_q = ms_features.loc[:, [process_survey.COL_ID, q]]
        if q == "What do you think is important for moral considerations?":
            all_selections = df_q[q].str.split(r',(?! )').explode()
            category_counts = all_selections.value_counts()
        else:
            category_counts = df_q[q].value_counts()
        plotter.plot_pie(categories_names=category_counts.index.tolist(), categories_counts=category_counts.tolist(),
                         categories_colors=colors, title=f"{q}",
                         save_path=result_path, save_name=f"{q.replace('?', '')}", format="png")
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
    ms_prios = ms_prios.drop(columns=[process_survey.COL_DUR_SEC])
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
    c_graded = c_graded.drop(columns=[process_survey.COL_DUR_SEC])
    ratings = [c for c in c_graded.columns if c.startswith("If two") or c.startswith("Assuming")]
    rating_colors = {"If two creatures/systems are conscious, they are equally conscious": "#540D6E",
                     "If two creatures/systems are conscious, they are not necessarily equally conscious": "#EE4266",
                     "Assuming two different creatures/systems are conscious, their consciousness is incomparable": "#FFD23F"}
    rating_names = {"If two creatures/systems are conscious, they are equally conscious": f"If two creatures/systems are conscious\nthey are equally conscious",
                    "If two creatures/systems are conscious, they are not necessarily equally conscious": f"If two creatures/systems are conscious\nthey are not necessarily equally conscious",
                     "Assuming two different creatures/systems are conscious, their consciousness is incomparable": f"Assuming two different creatures/systems are conscious\ntheir consciousness is incomparable"}
    rating_violins = {"If two creatures/systems are conscious, they are equally conscious": True,
                     "If two creatures/systems are conscious, they are not necessarily equally conscious": True,
                     "Assuming two different creatures/systems are conscious, their consciousness is incomparable": True}
    plotter.plot_raincloud(df=c_graded, id_col=process_survey.COL_ID,
                           data_col_names=ratings,
                           data_col_colors=rating_colors,
                           x_title="", x_name_dict=rating_names,
                           title=f"", y_title="How much do you agree?", ymin=1, ymax=4.04, yskip=1, y_jitter=0.05,
                           y_ticks=None, group_spacing=1.0, size_inches_x=20,
                           data_col_violin_left=rating_violins, scatter_lines=True,
                           save_path=result_path, save_name=f"consciousness_graded", format="png")

    question = "Does it mean that the interests of the more conscious entity matter more?"
    category_counts = c_graded[question].value_counts()
    plotter.plot_pie(categories_names=category_counts.index.tolist(), categories_counts=category_counts.tolist(),
                     categories_colors=CAT_COLOR_DICT, title=f"{question}",
                     save_path=result_path, save_name=f"{question.replace('?', '').replace('/', '-')}", format="png")

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
    con_intellect = con_intellect.drop(columns=[process_survey.COL_DUR_SEC])
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

    demographic_colors = {"Male": "#084887",
                          "Female": "#F58A07",
                          "Non-binary": "#A3ADCC",
                          "Prefer not to say": "#E7A391",
                          "Genderqueer": "#805FBF"
                          }

    con_demo = analysis_dict["demographics"]
    con_demo = con_demo.drop(columns=[process_survey.COL_DUR_SEC])

    # gender
    gender = "How do you describe yourself?"
    category_counts = con_demo[gender].value_counts()
    plotter.plot_pie(categories_names=category_counts.index.tolist(), categories_counts=category_counts.tolist(),
                     categories_colors=demographic_colors, title=f"{gender}",
                     save_path=result_path, save_name=f"gender", format="png")

    # country
    country = "In which country do you currently reside?"
    category_counts = con_demo[country].value_counts()
    plotter.plot_pie(categories_names=category_counts.index.tolist(), categories_counts=category_counts.tolist(),
                     categories_colors=None, title=f"{country}",
                     save_path=result_path, save_name=f"country", format="png")

    # education
    education = "What is your education background?"
    category_counts = con_demo[education].value_counts()
    plotter.plot_pie(categories_names=category_counts.index.tolist(), categories_counts=category_counts.tolist(),
                     categories_colors=None, title=f"{education}",
                     save_path=result_path, save_name=f"education", format="png")
    # education field
    field = "In what topic?"
    category_counts = con_demo[field].value_counts()
    plotter.plot_pie(categories_names=category_counts.index.tolist(), categories_counts=category_counts.tolist(),
                     categories_colors=None, title=f"{field}",
                     save_path=result_path, save_name=f"field", format="png")

    # employment
    employment = "Current primary employment domain"
    category_counts = con_demo[employment].value_counts()
    plotter.plot_pie(categories_names=category_counts.index.tolist(), categories_counts=category_counts.tolist(),
                     categories_colors=None, title=f"{employment}",
                     save_path=result_path, save_name=f"employment", format="png")
    return


def analyze_survey(sub_df, analysis_dict, save_path):
    """
    :param sub_df:
    :param analysis_dict:
    :param save_path:
    :return:
    """
    demographics(analysis_dict, save_path)
    other_creatures(analysis_dict, save_path)
    earth_in_danger(analysis_dict, save_path)
    ics(analysis_dict, save_path)
    kill_for_test(analysis_dict, save_path)
    zombie_pill(analysis_dict, save_path)
    moral_consideration_features(analysis_dict, save_path)
    moral_considreation_prios(analysis_dict, save_path)
    graded_consciousness(analysis_dict, save_path)
    consciousness_intelligence(analysis_dict, save_path)

    return