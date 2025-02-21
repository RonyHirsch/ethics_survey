import os
import pickle
import warnings
import numpy as np
import pandas as pd
import survey_mapping
import analyze_survey
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

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

COL_CONSENT = "consent_form"  # the field of informed consent
CONSENT_YES = "I consent, begin the study"  # providing consent


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
            intersection_age = intersection[intersection["response_id"].isin(prolific_age_mismatch["response_id"].unique().tolist())]
            if not(intersection_age.empty):
                print(f"** SUSPECTED cheating partipants: N={intersection_age.shape[0]}; filtered out")
                sub_df = sub_df[~sub_df["response_id"].isin(intersection_age["response_id"].tolist())]
    return sub_df


def bot_filter(sub_df, save_path, age_mismatch=None):
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


def process_survey(sub_df, save_path, age_mismatch=None):
    # filter out participants who did not provide consent - they are not counted as participants in the study:
    sub_df = sub_df[sub_df[COL_CONSENT] == CONSENT_YES].reset_index(inplace=False, drop=True)  # in the free sample all of them consented (0 dropouts)
    # filter out bots:
    print(f"Overall {sub_df.shape[0]} people participated in the study")
    sub_df = bot_filter(sub_df=sub_df, age_mismatch=age_mismatch, save_path=save_path)
    print(f"After filtering out suspected bots: {sub_df.shape[0]} participants")
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


def processed_free_sample(subject_data_path, free_save_path):

    # extract the data
    subject_data_raw = pd.read_csv(subject_data_path)
    # remove automatically-generated columns with no usable data
    subject_data_raw.drop(columns=survey_mapping.redundant, inplace=True)

    subject_data_raw.rename(columns=survey_mapping.questions_name_mapping, inplace=True)  # give meaningful headers
    sub_df = subject_data_raw.iloc[2:]  # delete the other question-indicator rows
    sub_df = convert_columns(sub_df)  # convert numeric and datetime columns to be as such

    # process the df
    subject_processed = process_survey(sub_df=sub_df, save_path=free_save_path)
    subject_processed.to_csv(os.path.join(free_save_path, "processed_data.csv"), index=False)

    # prepare the questions for analysis
    subject_dict = analysis_prep(subject_processed)
    with open(os.path.join(free_save_path, "processed_data.pkl"), "wb") as f:
        pickle.dump(subject_dict, f)

    return subject_dict, subject_processed


def define_exploratory_replication_pops(sub_df, sub_dict, replication_prop):
    categorical_cols = [
        "source",
        "Current primary employment domain",
        "What is your education background?",
        "In what topic?",  # might contain NaNs
        "In which country do you currently reside?",
        "How do you describe yourself?",
        "Do you have a pet?"
    ]

    ordinal_cols = [
        "On a scale from 1 to 5 where 1 means 'none' and 5 means 'extremely', how would you rate your experience and knowledge in ethics and morality?",
        "On a scale from 1 to 5 where 1 means 'none' and 5 means 'extremely', how would you rate your experience and knowledge in the science of consciousness?",
        "On a scale from 1 to 5 where 1 means 'none' and 5 means 'extremely', how would you rate your level of interaction or experience with animals?",
        "On a scale from 1 to 5 where 1 means 'none' and 5 means 'extremely', how would you rate your experience and knowledge in artificial intelligence (AI) systems?"
    ]

    age_col = "How old are you?"

    # Convert ordinal variables to numeric, filling NaNs with the median
    for col in ordinal_cols:
        sub_df[col] = pd.to_numeric(sub_df[col], errors="coerce")
        sub_df[col].fillna(sub_df[col].median(), inplace=True)

    # age is numeric, but what we want to balance is actually age-bins and not actual age-numbers
    age_bins = [18, 25, 35, 45, 55, 65, 75, 120]
    age_labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "66-75", "76+"]
    # now, make a categorical column, and that's what we will balance
    sub_df["age_group"] = pd.cut(sub_df[age_col], bins=age_bins, labels=age_labels, include_lowest=True).astype(str)

    # Convert categorical variables to strings and fill NaNs so that they will be trated as a 'category'
    sub_df[categorical_cols] = sub_df[categorical_cols].fillna("None").astype(str)

    # Create a stratification matrix by encoding categorical features
    encoded_cats = pd.get_dummies(sub_df[categorical_cols + ["age_group"]])


    # Normalize ordinal features for fair weighting: NO NEED, as they are all on the same scale
    normalized_ordinals = sub_df[ordinal_cols]

    # Combine categorical and numeric for stratification
    stratification_matrix = pd.concat([encoded_cats, normalized_ordinals], axis=1)

    # Use iterative stratified split to maintain balance across multiple dimensions
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=replication_prop, random_state=42)
    train_idx, test_idx = next(splitter.split(sub_df, stratification_matrix))

    # Create the exploratory and retest datasets
    exploratory_df = sub_df.iloc[train_idx].reset_index(inplace=False, drop=True)
    retest_df = sub_df.iloc[test_idx].reset_index(inplace=False, drop=True)

    # Now create the corresponding exploratory and retests dicts
    exploratory_dict = {key: df[df["response_id"].isin(exploratory_df["response_id"])].reset_index(inplace=False, drop=True) for key, df in sub_dict.items()}
    retest_dict = {key: df[df["response_id"].isin(retest_df["response_id"])].reset_index(inplace=False, drop=True) for key, df in sub_dict.items()}

    return exploratory_df, exploratory_dict, retest_df, retest_dict


def manage_processing(prolific_data_path, free_data_path, all_save_path, load=False, exclude_age_mismatch=False):
    if load:
        # load paid
        with open(os.path.join(prolific_data_path, r"processed\processed_data.pkl"), "rb") as f:
            prolific_sub_dict = pickle.load(f)
        prolific_sub_df = pd.read_csv(os.path.join(prolific_data_path, r"processed\processed_data.csv"))
        # load free
        with open(os.path.join(free_data_path, r"processed\processed_data.pkl"), "rb") as f:
            free_sub_dict = pickle.load(f)
        free_sub_df = pd.read_csv(os.path.join(free_data_path, r"processed\processed_data.csv"))

    else:  # run everything
        # paid sample
        prolific_sub_dict, prolific_sub_df = processed_paid_sample(subject_data_path=os.path.join(prolific_data_path, r"raw\prolific_paid_labels.csv"),
                                                                   prolific_save_path=os.path.join(prolific_data_path, r"processed"),
                                                                   prolific_data_path=os.path.join(prolific_data_path, r"raw\prolific_export_666701f76c6898bb61cdb6c0.csv"),
                                                                   exclude_age_mismatch=exclude_age_mismatch)

        # free sample
        free_sub_dict, free_sub_df = processed_free_sample(subject_data_path=os.path.join(free_data_path, r"raw\ethics_free_labels.csv"),
                                                           free_save_path=os.path.join(free_data_path, r"processed"))

    # collapse both samples
    prolific_sub_df["source"] = "Prolific"
    free_sub_df["source"] = "Free"
    total_df = pd.concat([prolific_sub_df, free_sub_df], ignore_index=True)
    total_df.to_csv(os.path.join(all_save_path, "processed_data.csv"), index=False)
    # unify the dicts as well
    total_dict = dict()
    for key in prolific_sub_dict.keys():
        total_dict[key] = pd.concat([prolific_sub_dict[key], free_sub_dict[key]], ignore_index=True)


    """
    Split the data into training and test. 
    This needs to be balanced, and we chose to balance it by demotraphics. 
    """

    exploratory_df, exploratory_dict, replication_df, replication_dict = define_exploratory_replication_pops(sub_df=total_df,
                                                                                                             sub_dict=total_dict,
                                                                                                             replication_prop=0.7)

    """
    Save the exploratory and replication dataframes for further analysis.
    """
    exploratory_df.to_csv(os.path.join(all_save_path, "exploratory_df.csv"), index=False)
    with open(os.path.join(all_save_path, "exploratory_dict.pkl"), "wb") as f:
        pickle.dump(exploratory_dict, f)

    replication_df.to_csv(os.path.join(all_save_path, "replication_df.csv"), index=False)
    with open(os.path.join(all_save_path, "replication_dict.pkl"), "wb") as f:
        pickle.dump(replication_dict, f)

    return


def manage_analysis(all_save_path, sample="exploratory"):
    """

    :param all_save_path:
    :param sample: exploratory, replication
    :return:
    """

    # load the right sample:
    sub_df = pd.read_csv(os.path.join(all_save_path, f"{sample}_df.csv"))
    with open(os.path.join(all_save_path, rf"{sample}_dict.pkl"), "rb") as f:
        sub_dict = pickle.load(f)

    # define the folder to save the results
    sample_save_path = os.path.join(all_save_path, f"{sample}")
    if not os.path.exists(sample_save_path):
        os.makedirs(sample_save_path)

    analyze_survey.analyze_survey(sub_df=sub_df, analysis_dict=sub_dict, save_path=sample_save_path)
    return


if __name__ == '__main__':
    # pre-process data and split to exploratory and replication samples.
    #manage_processing(prolific_data_path=r"C:\Users\Rony\Documents\projects\ethics\survey_analysis\data\analysis_data\prolific",
    #                  free_data_path=r"C:\Users\Rony\Documents\projects\ethics\survey_analysis\data\analysis_data\free",
    #                  all_save_path=r"C:\Users\Rony\Documents\projects\ethics\survey_analysis\data\analysis_data\all",
    #                  load=False,
    #                  exclude_age_mismatch=False)

    # analyze the relevant sample
    manage_analysis(all_save_path=r"C:\Users\Rony\Documents\projects\ethics\survey_analysis\data\analysis_data\all",
                    sample="exploratory")

