import os
import pandas as pd
import survey_mapping

# column names
COL_PROGRESS = "Progress"  # % of progress in the survey
PROGRESS_DONE = 100  # 100% progress = finished the survey

COL_FINISHED = "Finished"  # there is a column indicating if the survey has finished (bool)
COL_DUR_SEC = "Duration (in seconds)"  # how long it took the subject (in seconds)

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

def process_survey(sub_df):
    sub_df.rename(columns=survey_mapping.questions_name_mapping, inplace=True)  # give meaningful headers
    sub_df = sub_df.iloc[2:]  # delete the other question-indicator rows
    sub_df = convert_columns(sub_df)  # convert numeric and datetime columns to be as such
    # filter out participants who did not provide consent:
    sub_df = sub_df[sub_df[COL_CONSENT] == CONSENT_YES]
    # filter out bots:
    sub_df = sub_df[sub_df[COL_BOT] == BOT_PASS]
    print(f"Overall {sub_df.shape[0]} people participated in the study")
    return sub_df


def analysis_prep(sub_df):
    analysis_dict = dict()
    blocks = survey_mapping.question_blocks
    for question_cluster in blocks:  # for each question group
        relevant_df = sub_df[list(blocks[question_cluster].values())]  # take only the questions relevant for this group
        analysis_dict[question_cluster] = relevant_df
    return analysis_dict


if __name__ == '__main__':
    subject_path = r"...\projects\ethics\survey_analysis\pilot"
    subject_file = "....csv"
    subject_data_path = os.path.join(subject_path, subject_file)
    # extract the data
    subject_data_raw = pd.read_csv(subject_data_path)
    # remove automatically-generated columns with no usable data
    subject_data_raw.drop(columns=survey_mapping.redundant, inplace=True)
    # process the df
    subject_processed = process_survey(subject_data_raw)
    # prepare the questions for analysis
    analysis_dict = analysis_prep(subject_processed)
