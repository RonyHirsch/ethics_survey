import os

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
BOT_PASS = 1  # in the captcha,  1 = not a robot

COL_CONSENT = "consent_form"  # the field of informed consent
CONSENT_YES = "I consent, begin the study"  # providing consent


def convert_columns(df):
    for column in df.columns:
        try:
            # Try to convert to numeric
            df[column] = pd.to_numeric(df[column])
        except ValueError:
            try:
                # If numeric conversion fails, try to convert to datetime
                df[column] = pd.to_datetime(df[column])
            except ValueError:
                # If both conversions fail, leave the column as is
                pass
    return df


def bot_filter(sub_df):
    # all those with a reCaptcha score that isn't perfect
    suspected_bot = sub_df[sub_df[COL_BOT] != BOT_PASS]
    # test 1: did it take them less than MIN_DUR_SEC to answer the survey?
    test1 = suspected_bot[suspected_bot[COL_DUR_SEC] < MIN_DUR_SEC]
    if not test1.empty:
        sub_df_filtered = sub_df[~sub_df.isin(test1).all(axis=1)]
        sub_df_filtered.reset_index(inplace=True, drop=True)
    else:
        sub_df_filtered = sub_df

    # test 2: did these people even finish the entire study?
    sub_df_filtered = sub_df_filtered[sub_df_filtered["Finished"] != "False"]
    sub_df_filtered.reset_index(inplace=True, drop=True)
    return sub_df_filtered


def analysis_prep(sub_df):
    print(f"ANALYZED SAMPLE: N = {sub_df.shape[0]}")
    analysis_dict = dict()
    blocks = survey_mapping.question_blocks
    for question_cluster in blocks:  # for each question group
        # we add identified column & duration column to be able to cross information
        questions = list(blocks[question_cluster].values())
        if COL_ID in questions:  # avoid redundancy
            questions.remove(COL_ID)
        columns = [COL_ID] + questions
        relevant_df = sub_df[columns]  # take only the questions relevant for this group
        analysis_dict[question_cluster] = relevant_df
    return analysis_dict


def process_values(sub_df):
    """
    CORRECT FOR TECHNICAL GLITCHES
    """

    """
    When asked to rate how much they think a creature has moral status / consciousness, options were:
    1 = "Does not have",
    2 = "Probably doesn't have",
    3 = "Probably has",
    4 = "Has"
    However, Qualtrics allowed them to also select "0". Therefore, we will code "0" as "1": does not have".
    """
    other_creature_cols = [col for col in sub_df.columns for item in survey_mapping.other_creatures_general if item in col]
    additional_cols = ["If two creatures/systems are conscious, they are equally conscious"]
    cols = other_creature_cols + additional_cols
    sub_df[cols] = sub_df[cols].replace(0, 1)

    """
    When asked about their experience level on a scale from 1 to 5, the options were:
    1 = "No experience, 2, 3, 4, 5
    """
    exp_cols = [survey_mapping.Q_AI_EXP, survey_mapping.Q_ANIMAL_EXP, survey_mapping.Q_CONSC_EXP, survey_mapping.Q_ETHICS_EXP]
    for col in exp_cols:
        sub_df[col] = sub_df[col].replace(np.nan, 1)
        sub_df[col] = sub_df[col].replace(0, 1)


    """
    When asked about employment, some responded "Other" but then in the specifics, they wrote something that 
    DOES appear in the options
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
    #print(sub_df[sub_df[curr_job_col] == "Other"][job_other].tolist())
    return sub_df


def process_survey(sub_df):
    sub_df.rename(columns=survey_mapping.questions_name_mapping, inplace=True)  # give meaningful headers
    sub_df = sub_df.iloc[2:]  # delete the other question-indicator rows
    sub_df = convert_columns(sub_df)  # convert numeric and datetime columns to be as such
    # filter out participants who did not provide consent - they are not counted as participants in the study:
    sub_df = sub_df[sub_df[COL_CONSENT] == CONSENT_YES].reset_index(inplace=False, drop=True)
    # filter out bots:
    print(f"Overall {sub_df.shape[0]} people participated in the study")
    sub_df = bot_filter(sub_df)
    print(f"After filtering out suspected bots: {sub_df.shape[0]} participants")
    sub_df = process_values(sub_df)
    return sub_df


def correct_prolific(df, prolific_data_path, save_path):
    prolific_df = pd.read_csv(prolific_data_path)

    """
    Age
    """

    prolific_df_relevant = prolific_df.loc[:, ["Participant id", "Age"]]
    prolific_df_relevant.rename(columns={"Participant id": "PROLIFIC_PID"}, inplace=True)

    merged_df = pd.merge(df, prolific_df_relevant, on="PROLIFIC_PID", how="left")
    merged_df["age_verification"] = (merged_df["How old are you?"].replace("CONSENT_REVOKED", np.nan).astype(float) == merged_df["Age"].replace("CONSENT_REVOKED", np.nan).astype(float)).astype(int)

    mismatch = merged_df[merged_df["age_verification"] == 0][["PROLIFIC_PID", "How old are you?", "Age"]]
    mismatch.to_csv(os.path.join(save_path, "age_liars.csv"), index=False)
    print(f"{mismatch.shape[0]} had a mismatch between reported age and Prolific age; took Prolific age")

    merged_df.loc[merged_df["age_verification"] == 0, "How old are you?"] = merged_df["Age"]
    merged_df.drop(columns=["Age", "age_verification"], inplace=True)

    # nans won't be converted into ints, which is why we convert to floats
    merged_df["How old are you?"] = merged_df["How old are you?"].replace("CONSENT_REVOKED", np.nan).astype(float)

    return merged_df


def processed_paid_sample(subject_data_path, prolific_save_path, prolific_data_path):

    # extract the data
    subject_data_raw = pd.read_csv(subject_data_path)
    # remove automatically-generated columns with no usable data
    subject_data_raw.drop(columns=survey_mapping.redundant, inplace=True)

    # process the df
    subject_processed = process_survey(subject_data_raw)
    # As the sample is paid, cross some of the demographic data with Prolific to ensure correctness
    subject_processed = correct_prolific(df=subject_processed,
                                         save_path=prolific_save_path,
                                         prolific_data_path=prolific_data_path)
    # As the sample is paid, there are a few more columns to take care of to ensure de-identification
    subject_processed.drop(columns=survey_mapping.prolific_redundant, inplace=True, errors="ignore")
    subject_processed.to_csv(os.path.join(prolific_save_path, "processed_data.csv"), index=False)

    # prepare the questions for analysis
    subject_dict = analysis_prep(subject_processed)
    return subject_dict, subject_processed


def processed_free_sample(subject_data_path, free_save_path):

    # extract the data
    subject_data_raw = pd.read_csv(subject_data_path)
    # remove automatically-generated columns with no usable data
    subject_data_raw.drop(columns=survey_mapping.redundant, inplace=True)

    # process the df
    subject_processed = process_survey(subject_data_raw)
    subject_processed.to_csv(os.path.join(free_save_path, "processed_data.csv"), index=False)

    # prepare the questions for analysis
    subject_dict = analysis_prep(subject_processed)

    return subject_dict, subject_processed


if __name__ == '__main__':

    # paid sample
    prolific_sub_dict, prolific_sub_df = processed_paid_sample(subject_data_path=r"C:\Users\rony\Documents\ethics_survey_data\prolific\raw\prolific_paid_labels.csv",
                                                               prolific_save_path=r"C:\Users\rony\Documents\ethics_survey_data\prolific\processed",
                                                               prolific_data_path=r"C:\Users\rony\Documents\ethics_survey_data\prolific\raw\prolific_export_666701f76c6898bb61cdb6c0.csv")
    analyze_survey.analyze_survey(sub_df=prolific_sub_df,
                                  analysis_dict=prolific_sub_dict,
                                  save_path=r"C:\Users\rony\Documents\ethics_survey_data\prolific\processed")

    # free sample
    free_sub_dict, paid_sub_df = processed_free_sample(subject_data_path=r"C:\Users\rony\Documents\ethics_survey_data\free\raw\ethics_free_labels.csv",
                                                       free_save_path=r"C:\Users\rony\Documents\ethics_survey_data\free\processed")
    analyze_survey.analyze_survey(sub_df=paid_sub_df,
                                  analysis_dict=free_sub_dict,
                                  save_path=r"C:\Users\rony\Documents\ethics_survey_data\free\processed")

    # collapse both samples
    total_df = pd.concat([prolific_sub_df, paid_sub_df], ignore_index=True)
    # unify the dicts as well
    total_dict = dict()
    for key in prolific_sub_dict.keys():
        total_dict[key] = pd.concat([prolific_sub_dict[key], free_sub_dict[key]], ignore_index=True)
    analyze_survey.analyze_survey(sub_df=total_df, analysis_dict=total_dict, save_path=r"C:\Users\rony\Documents\ethics_survey_data\all")