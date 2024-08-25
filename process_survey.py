import os
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
    return sub_df_filtered


def analysis_prep(sub_df):
    print(f"ANALYZED SAMPLE: N = {sub_df.shape[0]}")
    analysis_dict = dict()
    blocks = survey_mapping.question_blocks
    for question_cluster in blocks:  # for each question group
        # we add identified column & duration column to be able to cross information
        columns = [COL_ID, COL_DUR_SEC] + list(blocks[question_cluster].values())
        relevant_df = sub_df[columns]  # take only the questions relevant for this group
        analysis_dict[question_cluster] = relevant_df
    return analysis_dict


def process_values(sub_df):
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


if __name__ == '__main__':
    subject_data_path = r"..\raw_data.csv"
    save_path = r"..\processed"

    # extract the data
    subject_data_raw = pd.read_csv(subject_data_path)
    # remove automatically-generated columns with no usable data
    subject_data_raw.drop(columns=survey_mapping.redundant, inplace=True)

    # process the df
    subject_processed = process_survey(subject_data_raw)
    subject_processed.to_csv(os.path.join(save_path, "processed_data.csv"), index=False)
    # prepare the questions for analysis
    subject_dict = analysis_prep(subject_processed)
    # analyze survey
    analyze_survey.analyze_survey(subject_processed, subject_dict, save_path)
