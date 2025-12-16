import os
import pickle
import warnings
import numpy as np
import pandas as pd
import survey_mapping
import analyze_survey

# column names
COL_ID = "response_id"

COL_PROGRESS = "Progress"  # % of progress in the survey
PROGRESS_DONE = 100  # 100% progress = finished the survey

COL_FINISHED = "Finished"  # there is a column indicating if the survey has finished (bool)
COL_DUR_SEC = "duration_sec"  # how long it took the subject (in seconds)
MIN_DUR_SEC = 300  # 300 seconds = 5 minutes

COL_BOT = "Q_RecaptchaScore"  # captcha score
BOT_PASS = 0.8  # in the captcha,  1 = not a robot
AGE_CONSENT = 18



def convert_columns(df):
    for column in df.columns:
        try:
            # Try to convert to numeric
            df.loc[:, column] = pd.to_numeric(df[column])
        except ValueError:
            try:
                # If numeric conversion fails, try to convert to datetime
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    df.loc[:, column] = pd.to_datetime(df[column])
            except ValueError:
                # If both conversions fail, leave the column as is
                pass
    return df


def identify_auto_qs(sub_df, prolific_age_mismatch, save_path):
    """
    Identify blocks of questions where the pattern of responses indicates participants may have just skipped
    this part by clicking the same thing over and over
    """

    """
    the rating questions are the easiest ones to identify a person who just clicked on one side all along
    """
    same_ms_rating = sub_df[sub_df[list(survey_mapping.other_creatures_ms.values())].nunique(axis=1) == 1]
    same_c_rating = sub_df[sub_df[list(survey_mapping.other_creatures_cons.values())].nunique(axis=1) == 1]
    intersection = same_ms_rating.merge(same_c_rating, how="inner")
    print(f"{intersection.shape[0]} participants gave identical ratings in the ms-rating section as well as the c-rating section; check manually")
    intersection.to_csv(os.path.join(save_path, f"identical_ratings.csv"), index=False)  # 0 were excluded for that
    if not(intersection.empty):
        # if we have people who gave the same ms rating AND the same c rating to everything, we suspect they skimmed it
        # did they also lie about their age?
        if prolific_age_mismatch is not None:  # not nan, paid sample
            intersection_age = intersection[intersection[COL_ID].isin(prolific_age_mismatch[COL_ID].unique().tolist())]
            if not(intersection_age.empty):
                print(f"** SUSPECTED cheating partipants: N={intersection_age.shape[0]}; filtered out")
                sub_df = sub_df[~sub_df[COL_ID].isin(intersection_age[COL_ID].tolist())]
    return sub_df


def bot_filter(sub_df, save_path, age_mismatch=None):
    """
    Filter out automatic responses or any response that doesn't feel like a genuine person actually responded to the
    survey.
    :param sub_df: the df with the responses to the survey
    :param save_path: path to save data to
    :param age_mismatch: whether to exclude based on mismatch between Prolific age (paid sample) and demographics
    collected during the survey
    :return: the filtered df
    """

    shape_before = sub_df.shape[0]
    print(f"Before filtering: {shape_before} subjects in sample")
    # all those with a reCaptcha score that isn't perfect
    suspected_bot = sub_df[sub_df[COL_BOT] < BOT_PASS]

    """
    test 1: did it take them less than MIN_DUR_SEC to answer the survey?
    """
    test1 = suspected_bot[suspected_bot[COL_DUR_SEC] < MIN_DUR_SEC]
    if not test1.empty:
        sub_df_filtered = sub_df[~sub_df.isin(test1).all(axis=1)]
        sub_df_filtered.reset_index(inplace=True, drop=True)
    else:
        sub_df_filtered = sub_df
    shape_after = sub_df.shape[0]
    print(f"Bot filtering: {shape_before - shape_after} were excluded")  # in the free sample: 0

    """
    test 2: did these people even finish the entire study?
    """
    sub_df_filtered = sub_df_filtered[sub_df_filtered["Finished"] != "False"]
    sub_df_filtered.reset_index(inplace=True, drop=True)
    shape_finish = sub_df_filtered.shape[0]
    print(f"Partial responses: {shape_after - shape_finish} were excluded")

    """
    test 3: are respondents in the age of consent?
    """
    sub_df_age = sub_df_filtered[sub_df_filtered["How old are you?"] >= AGE_CONSENT]
    sub_df_age.reset_index(inplace=True, drop=True)
    shape_final_age = sub_df_age.shape[0]
    print(f"Underage participants: {shape_finish - shape_final_age} were excluded")

    """
    test 4: did people answer automatically (i.e., consistently replied an answer that's on one side of the screen?)
    """
    sub_df_final = identify_auto_qs(sub_df=sub_df_age, prolific_age_mismatch=age_mismatch, save_path=save_path)
    sub_df_final.reset_index(inplace=True, drop=True)
    shape_final = sub_df_final.shape[0]
    print(f"Skipping participants: {shape_final_age - shape_final} were excluded")

    return sub_df_final


def analysis_prep(sub_df):
    """
    Prepare the sample df for analysis - create a dictionary for each question-block (as defined in survey_mapping) for
    easier analysis.
    :param sub_df: the sample df
    :return:
    """
    print(f"ANALYZED SAMPLE: N = {sub_df.shape[0]}")
    analysis_dict = dict()

    relevant_blocks = ["other_creatures_ms", "other_creatures_cons",
                       "moral_considerations_prios",  "moral_considerations_features",
                       "consciousness_exp", "ai_exp", "animal_exp", "ethics_exp", "demographics"]
    blocks = {b: survey_mapping.question_blocks[b] for b in relevant_blocks}

    for question_cluster in blocks:  # for each question group
        # we add identified column & duration column to be able to cross information
        questions = list(blocks[question_cluster].values())
        if COL_ID in questions:  # avoid redundancy
            questions.remove(COL_ID)
        columns = [COL_ID] + questions
        # Filter to only keep columns that exist in sub_df
        existing_columns = [col for col in columns if col in sub_df.columns]
        relevant_df = sub_df[existing_columns]  # take only the questions relevant for this group
        analysis_dict[question_cluster] = relevant_df

    # check which questions columns were not used and why
    used_columns = set()
    for relevant_df in analysis_dict.values():
        used_columns.update(relevant_df.columns)

    unused_columns = set(sub_df.columns) - used_columns
    print(f"Columns not included in any analysis: {unused_columns}")

    return analysis_dict


def process_values(sub_df):
    """
    CORRECT FOR TECHNICAL GLITCHES:
    This function corrects the actual response data to make sure it's unified and doesn't contain any weird things
    that will prevent it from being processed.
    """

    """
    Ordinal scales started with "1" and ended with "4"/"5". However, in qualtrics, the lowest level could have also 
    been not-marked (i.e., 0 or nan), which is why we need to correct for that in the following columns.
    """
    other_creature_cols = [col for col in sub_df.columns for item in survey_mapping.other_creatures_general_names if item in col]
    exp_cols = [survey_mapping.Q_AI_EXP, survey_mapping.Q_ANIMAL_EXP, survey_mapping.Q_CONSC_EXP,survey_mapping.Q_ETHICS_EXP]
    cols = other_creature_cols + exp_cols
    sub_df[cols] = sub_df[cols].replace(np.nan, 1)
    sub_df[cols] = sub_df[cols].replace(0, 1)


    """
    When asked about employment, some responded "Other" but then in the specifics, they wrote something that 
    DOES appear in the options. Let's convert that
    """
    curr_job_col = "Current primary employment domain"
    job_other = "employment_Other: please specify"

    other_dict = {"Homemaker": "Homemaker/caregiver",
                  "care": "Homemaker/caregiver",
                  "Nanny": "Homemaker/caregiver",
                  "homemaker": "Homemaker/caregiver",

                  "not employed": "Unemployed",
                  "unemployed": "Unemployed",
                  "Unemployed": "Unemployed",
                  "I am Unemployed": "Unemployed",
                  "N/a": "Unemployed",
                  "disabled and unable to work": "Unemployed",
                  "enemployed": "Unemployed",
                  "unemployed, looking for a job": "Unemployed",
                  "Unemployed but searching for a job.": "Unemployed",
                  "Disabled": "Unemployed",
                  "Unemployed, looking": "Unemployed",
                  "looking for work": "Unemployed",
                  "No current employment": "Unemployed",
                  "not currently employed": "Unemployed",
                  "I'm a NEET.": "Unemployed",
                  "Unemployed now": "Unemployed",
                  "Looking for work": "Unemployed",
                  "I am currently unemployed.": "Unemployed",

                  "Admin": "Administrative/clerical",
                  "HUMAN RESOURCES": "Administrative/clerical",
                  "hr": "Administrative/clerical",

                  "non-profit org": "Non-profit/volunteering",

                  "Financial accounting": "Business/finance",

                  "Veterinary Medicine": "Healthcare/medicine",
                  "Dietetics": "Healthcare/medicine",

                  "Mixed IT & CS": "Tech/software development and engineering/IT",

                  "Service industry": "Customer services",
                  "IT support": "Customer services",
                  "part time service at a cafe": "Customer services",

                  "Consultant in big companies and a lot of other tasks.": "Management/consulting",
                  "Consulting": "Management/consulting",

                  "Telecommunications": "Media/communications",
                  "Language Services": "Media/communications",

                  "Trader": "Construction/trades",
                  "self-employed e-commerce operator": "Construction/trades",
                  "Forex and Commodities Trading": "Construction/trades",

                  "Real Estate": "Retail/sales",

                  "graphic designer": "Arts/entertainment",
                  "art and media - cultural producer, creative entrepreneur, content creator, human/youth developer,": "Arts/entertainment",
                  "Design": "Arts/entertainment",

                  "Transportation": "Transportation/logistics",

                  }

    # convert "Others" to existing options
    sub_df[curr_job_col] = sub_df.apply(lambda row: other_dict.get(row[job_other].strip(), row[curr_job_col]) if row[curr_job_col] == "Other" else row[curr_job_col], axis=1)
    return sub_df


def process_survey(sub_df, save_path, age_mismatch=None):
    """
    Both the processing of the free and of the paid sample call this method, which takes a df of the survey results
    and processes it.
    :param sub_df: the df to be processed
    :param save_path: where to save the results
    :param age_mismatch: whether to exclude paid participants where the age doesn't match between their Prolific id
    and their reported age
    :return: the processed df
    """
    # filter out bots and any auto-response we suspect:
    print(f"Overall {sub_df.shape[0]} people participated in the study")
    sub_df = bot_filter(sub_df=sub_df, age_mismatch=age_mismatch, save_path=save_path)
    print(f"After filtering out suspected bots: {sub_df.shape[0]} participants")
    # correct for technical glitches
    sub_df = process_values(sub_df)
    return sub_df


def correct_prolific(df, prolific_data_path, save_path, exclude=False):
    """
    :param df: survey dataframe (processed)
    :param prolific_data_path: demographic data from the prolific recruitment platform (path to file)
    :param save_path: where to save the data to
    :param exclude: whether to exclude participants based on mismatches between the survey demographic data and prolific's;
    if True, subjects with inconsistencies will be dropped (based on some condition), otherwise we'll report about them
    and correct based on the prolific as ground-truth
    :return: the corrected/filtered df
    """
    prolific_df = pd.read_csv(prolific_data_path)

    """
    Age
    """

    prolific_df_relevant = prolific_df.loc[:, ["Participant id", "Age"]]
    prolific_df_relevant.rename(columns={"Participant id": "PROLIFIC_PID"}, inplace=True)

    merged_df = pd.merge(df, prolific_df_relevant, on="PROLIFIC_PID", how="left")
    merged_df["age_verification"] = (merged_df["How old are you?"].replace("CONSENT_REVOKED", np.nan).astype(float) == merged_df["Age"].replace("CONSENT_REVOKED", np.nan).astype(float)).astype(int)

    mismatch = merged_df[merged_df["age_verification"] == 0][[COL_ID, "PROLIFIC_PID", "How old are you?", "Age"]]
    mismatch.to_csv(os.path.join(save_path, "age_liars.csv"), index=False)
    print(f"{mismatch.shape[0]} had a mismatch between reported age and Prolific age; took Prolific age")

    if exclude:  # exclude age liars based on some threshold
        age_thresh = 2
        # gap between age as reported in prolific and in the survey
        merged_df["age_gap"] = np.abs(merged_df["How old are you?"].replace("CONSENT_REVOKED", np.nan).astype(float) - merged_df["Age"].replace("CONSENT_REVOKED", np.nan).astype(float))
        big_gap = merged_df[merged_df["age_gap"] > age_thresh]  # gap is larger than threshold
        print(f"**EXCLUSION** where age gap is >= {age_thresh}, N={big_gap.shape[0]} participants are excluded")
        merged_df = merged_df[~merged_df.index.isin(big_gap.index)]
        merged_df.drop(columns=["age_gap"], inplace=True)

    merged_df.loc[merged_df["age_verification"] == 0, "How old are you?"] = merged_df["Age"]
    merged_df.drop(columns=["Age", "age_verification"], inplace=True)

    # nans won't be converted into ints, which is why we convert to floats
    merged_df["How old are you?"] = merged_df["How old are you?"].replace("CONSENT_REVOKED", np.nan).astype(float)

    return merged_df, mismatch


def processed_paid_sample(subject_data_path, prolific_save_path, prolific_data_path, exclude_age_mismatch=False):

    # extract the data
    subject_data_raw = pd.read_csv(subject_data_path)
    # remove automatically-generated columns with no usable data
    subject_data_raw.drop(columns=survey_mapping.redundant, inplace=True)

    subject_data_raw.rename(columns=survey_mapping.questions_name_mapping, inplace=True)  # give meaningful headers
    sub_df = subject_data_raw.iloc[2:]  # delete the other question-indicator rows
    sub_df = convert_columns(sub_df)  # convert numeric and datetime columns to be as such

    # As the sample is paid, cross some of the demographic data with Prolific to ensure correctness
    subject_processed, age_mismatch = correct_prolific(df=sub_df,
                                                       save_path=prolific_save_path,
                                                       prolific_data_path=prolific_data_path,
                                                       exclude=exclude_age_mismatch)

    # process the df
    subject_processed = process_survey(sub_df=subject_processed, age_mismatch=age_mismatch, save_path=prolific_save_path)

    # As the sample is paid, there are a few more columns to take care of to ensure de-identification
    subject_processed.drop(columns=survey_mapping.prolific_redundant, inplace=True, errors="ignore")
    subject_processed.to_csv(os.path.join(prolific_save_path, "processed_data.csv"), index=False)

    # prepare the questions for analysis
    subject_dict = analysis_prep(subject_processed)
    # save to pickle for loading
    with open(os.path.join(prolific_save_path, "processed_data.pkl"), "wb") as f:
        pickle.dump(subject_dict, f)
    return subject_dict, subject_processed

def manage_processing(prolific_data_path, exclude_age_mismatch=False):
    prolific_sub_dict, prolific_sub_df = processed_paid_sample(
        subject_data_path=os.path.join(prolific_data_path, r"raw\raw_data_labels.csv"),
        prolific_save_path=os.path.join(prolific_data_path, r"processed"),
        prolific_data_path=os.path.join(prolific_data_path, r"raw\prolific_demographic_export_68b9346b76bfa17d65f930c3.csv"),
        exclude_age_mismatch=exclude_age_mismatch)
    return prolific_sub_dict, prolific_sub_df


def manage_analysis(prolific_sub_dict, prolific_sub_df, all_save_path):
    # load the right sample:
    sub_df = prolific_sub_df
    analysis_dict = prolific_sub_dict
    # define the folder to save the results
    save_path = all_save_path

    #sub_df.to_csv(os.path.join(save_path, "sub_df.csv"), index=False)

    df_demo = analyze_survey.demographics_descriptives(analysis_dict=analysis_dict, save_path=save_path)
    df_exp_ratings, exp_path = analyze_survey.experience_descriptives(analysis_dict=analysis_dict, save_path=save_path)
    analyze_survey.experience_with_demographics_descriptives(df_demographics=df_demo,
                                                             df_experience=df_exp_ratings, save_path=exp_path)

    df_c_v_ms_long, df_c_v_ms, c_v_ms_path = analyze_survey.c_v_ms(analysis_dict=analysis_dict, save_path=save_path)

    ms_features_df, most_important_df, ms_features_path = analyze_survey.ms_features_descriptives(
        analysis_dict=analysis_dict,
        save_path=save_path)

    df_ms_c_prios, prios_path = analyze_survey.ms_c_prios_descriptives(analysis_dict=analysis_dict, save_path=save_path)

    """
    ------------------------- STATISTICAL ANALYSES and data prep for R modelling --------------------------------------
    """

    for expertise in [survey_mapping.Q_EXP_NAME_DICT[survey_mapping.Q_CONSC_EXP],  # consciousness
                      survey_mapping.Q_EXP_NAME_DICT[survey_mapping.Q_ETHICS_EXP]]:  # ethics
        analyze_survey.perform_chi_square(df1=df_ms_c_prios.loc[:, ["response_id", survey_mapping.PRIOS_Q_NONCONS]],
                                          col1=survey_mapping.PRIOS_Q_NONCONS,
                                          df2=df_exp_ratings,
                                          col2=f"{expertise}_expert", col2_name=f"{expertise} Expertise",
                                          id_col="response_id",
                                          grp1_vals=[survey_mapping.ANS_NO, survey_mapping.ANS_YES],
                                          grp1_color_dict=analyze_survey.YES_NO_COLORS,
                                          grp2_vals=[0, 1], grp2_map=analyze_survey.EXP_BINARY_NAME_MAP,
                                          save_path=prios_path,
                                          save_name=f"{survey_mapping.PRIOS_Q_NAME_MAP[survey_mapping.PRIOS_Q_NONCONS]}_{expertise.lower()}_exp",
                                          save_expected=False)

    analyze_survey.c_v_ms_expertise(c_v_ms_df=df_c_v_ms, df_experience=df_exp_ratings,
                                    save_path=c_v_ms_path, plot_this=False)

    return


if __name__ == '__main__':
    # pre-process data and split to exploratory and replication samples.
    prolific_sub_dict, prolific_sub_df = manage_processing(
        prolific_data_path=r"\survey_analysis\data\2025_10_25_follow-up",
        exclude_age_mismatch=False)

    manage_analysis(prolific_sub_dict=prolific_sub_dict, prolific_sub_df=prolific_sub_df,
                    all_save_path=r"\survey_analysis\data\2025_10_25_follow-up\processed")

