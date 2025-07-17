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
    CORRECT FOR TECHNICAL GLITCHES:
    This function corrects the actual response data to make sure it's unified and doesn't contain any weird things
    that will prevent it from being processed.
    """

    """
    Ordinal scales started with "1" and ended with "4"/"5". However, in qualtrics, the lowest level could have also 
    been not-marked (i.e., 0 or nan), which is why we need to correct for that in the following columns.
    """
    other_creature_cols = [col for col in sub_df.columns for item in survey_mapping.other_creatures_general if item in col]
    exp_cols = [survey_mapping.Q_AI_EXP, survey_mapping.Q_ANIMAL_EXP, survey_mapping.Q_CONSC_EXP,survey_mapping.Q_ETHICS_EXP]
    additional_cols = [survey_mapping.Q_GRADED_EQUAL, survey_mapping.Q_GRADED_UNEQUAL]
    cols = other_creature_cols + exp_cols + additional_cols
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
    #print(sub_df[sub_df[curr_job_col] == "Other"][job_other].tolist())
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
    # filter out participants who did not provide consent - they are not counted as participants in the study:
    sub_df = sub_df[sub_df[COL_CONSENT] == CONSENT_YES].reset_index(inplace=False, drop=True)  # in the free sample all of them consented (0 dropouts)
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


def calculate_proportions(df, categorical_cols, ordinal_cols):
    """
    In a given df, calculate the proportions of different response types in categorical_cols and ordinal_cols
    :return: proportions
    """
    proportions = {}
    for col in categorical_cols:
        proportions[col] = (df[col].value_counts(normalize=True) * 100).to_dict()
    for col in ordinal_cols:
        proportions[col] = (df[col].value_counts(normalize=True) * 100).to_dict()
    proportions_df = pd.DataFrame(proportions).transpose()
    return proportions_df


def compare_proportions(df1, df1_name, df2, df2_name, df3, df3_name):
    """
    Given three dataframes (full sample, exploratory, replication), calculate how similar the proportions of responses
    are across them. This is used in order to check that the exploratory-replication split of the full sample
    preserves the relationships of different response types in the columns which were defined as relevant

    :return: a df containing the comparison
    """
    diff_df1_df2 = (df1 - df2).abs()
    diff_df1_df3 = (df1 - df3).abs()
    diff_df2_df3 = (df2 - df3).abs()

    comparison_df = pd.concat(
        [diff_df1_df2, diff_df1_df3, diff_df2_df3],
        axis=1,
        keys=[f"{df1_name}_vs_{df2_name}", f"{df1_name}_vs_{df3_name}", f"{df2_name}_vs_{df3_name}"]
    )

    # Compute overall mean difference for ranking
    comparison_df["average_difference"] = comparison_df.mean(axis=1)

    # Rank columns based on differences
    most_different_cols = comparison_df["average_difference"].sort_values(ascending=False)

    # re-order the columns (cosmetics)
    comparison_df = comparison_df.T.reset_index(drop=False).T

    return comparison_df, most_different_cols


def define_exploratory_replication_pops(sub_df, sub_dict, categorical_cols, ordinal_cols, replication_prop):
    """
    The method that actually does the splitting between exploratory and replication data given the full sample, and
    the things that needs to be balanced across samples.
    To do the actual splitting, we will use iterative stratification: https://pypi.org/project/iterative-stratification/
    Specifically, MultilabelStratifiedShuffleSplit: https://github.com/trent-b/iterative-stratification

    :param sub_df: the full sample dataframe with all the answers to the survey
    :param sub_dict: the dict dividing the survey responses into blocks
    :param categorical_cols: the categorical columns we need to balance
    :param ordinal_cols: the ordinal columns we need to balance
    :param replication_prop: the proportion of the replication sample out of the full sample between 0-1 (i.e., 0.5 is
    50% so the split will be half exploratory, half replication)
    :return:the df and analysis dicts of the exploratory and replication samples.
    """

    # Convert ordinal variables to numeric, filling NaNs with the median
    for col in ordinal_cols:
        sub_df[col] = pd.to_numeric(sub_df[col], errors="coerce")
        sub_df[col].fillna(sub_df[col].median(), inplace=True)

    # Convert categorical variables to strings and fill NaNs so that they will be trated as a 'category'
    sub_df[categorical_cols] = sub_df[categorical_cols].fillna("None").astype(str)

    # Create a stratification matrix by encoding categorical features
    encoded_cats = pd.get_dummies(sub_df[categorical_cols])

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
    exploratory_dict = {key: df[df[COL_ID].isin(exploratory_df[COL_ID])].reset_index(inplace=False, drop=True) for key, df in sub_dict.items()}
    retest_dict = {key: df[df[COL_ID].isin(retest_df[COL_ID])].reset_index(inplace=False, drop=True) for key, df in sub_dict.items()}

    return exploratory_df, exploratory_dict, retest_df, retest_dict


def manage_processing(prolific_data_path, free_data_path, all_save_path, load=False, exclude_age_mismatch=False):
    """
    This method manages the entire pre-processing pipeline of the full sample, including splitting it to exploratory
    and replication samples. Notably, if we do not want the split to be re-done, do not run this function.

    - Preprocess the samples (processed_paid_sample, processed_free_sample) and unify them
    - Decide which columns are important for balancing between samples
    - Make sure to convert and process them accordingly (e.g., age group)
    - Split the sample into exploratory and replication, with a ratio of 30%-70%
    (see call for define_exploratory_replication_pops)
    - Test the result and report it so we know we're good

    :param prolific_data_path: the path to the folder in which the prolific data is. This parameter is
    expected to be a folder path, in which there is a folder named "raw", and in the "raw" we have 2 important csv
    files: one with the demographics information from prolific (export), and one with the labels (answers export from
    prolific).
    :param free_data_path: he path to the folder in which the free data is. In there, we only have the csv of the
    responses, as the demographics are collected solely from the qualtrics responses.
    :param all_save_path: the path to a folder where we will keep all the data, after preprocessing it and splitting
    it into exploratory and replication samples.
    :param load: if True, loads the processed data AND RE-DOES THE EXPLOTAROTY-REPLICATION SPLIT.
    *** If we do not want to re-do the split, there is NO NEED TO RUN THIS METHOD at all, see "manage_analysis" function
    instead.
    :param exclude_age_mismatch: a parameter that is transferred to "processed_paid_sample", where we have information
    about age from two sources - the prolific profile informaiton, and the survey. If True, then participants with a
    mismatch in ages are excluded; otherwise, they are kept in the sample.
    :return:
    """
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

    """
    Unify both samples
    """
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
    This needs to be balanced, and we chose to balance it by demographics. 
    """

    # so first of all, we need to define the types of data we care about (want to balance)
    categorical_columns = [
        "source",
        survey_mapping.Q_EMPLOYMENT,  # current primary employment domain
        survey_mapping.Q_EDU,  # education level
        survey_mapping.Q_EDU_FIELD,  # might contain NaNs
        survey_mapping.Q_COUNTRY,  # country of residence
        survey_mapping.Q_GENDER,  # gender
        "Do you have a pet?"
    ]
    ordinal_columns = [  # experience columns
        survey_mapping.Q_ETHICS_EXP,
        survey_mapping.Q_CONSC_EXP,
        survey_mapping.Q_ANIMAL_EXP,
        survey_mapping.Q_AI_EXP
    ]

    age_col = survey_mapping.Q_AGE
    # age is numeric, but what we want to balance is actually age-bins and not actual age-numbers - let's balance based on what we later report
    age_bins = analyze_survey.AGE_BINS
    age_labels = analyze_survey.AGE_LABELS
    # now, make a categorical column, and that's what we will balance
    total_df["age_group"] = pd.cut(total_df[age_col], bins=age_bins, labels=age_labels, include_lowest=True).astype(str)
    categorical_columns.append("age_group")  # now this is a categorical column we want to balance with the rest
    total_df.to_csv(os.path.join(all_save_path, "processed_data.csv"), index=False)  # re-save with this column

    """
    Based on the demographic columns of interest (categorical_columns and ordinal_columns), split the data into
    exploratory and replication samples, balancing the proportions of columns so that they will be similar to the 
    overall sample. 
    """
    exploratory_df, exploratory_dict, replication_df, replication_dict = define_exploratory_replication_pops(sub_df=total_df,
                                                                                                             sub_dict=total_dict,
                                                                                                             categorical_cols=categorical_columns,
                                                                                                             ordinal_cols=ordinal_columns,
                                                                                                             replication_prop=0.7)

    """
    Calculate the proportions in the relevant columns to make sure we're good: 
    do that by calculating the proportions of different responses in all columns of interest in each df (full sample, 
    exploratory, replication), and then comparing them (with compare_proportions)
    """
    total_proportions = calculate_proportions(total_df, categorical_columns, ordinal_columns)
    total_proportions.to_csv(os.path.join(all_save_path, f"sample_balance_total.csv"), index=True)
    exploratory_proportions = calculate_proportions(exploratory_df, categorical_columns, ordinal_columns)
    exploratory_proportions.to_csv(os.path.join(all_save_path, f"sample_balance_exploratory.csv"), index=True)
    replication_proportions = calculate_proportions(replication_df, categorical_columns, ordinal_columns)
    replication_proportions.to_csv(os.path.join(all_save_path, f"sample_balance_replication.csv"), index=True)
    # compare the proportions directly
    comparison_df, most_different_cols = compare_proportions(df1=total_proportions, df1_name="total",
                                        df2=exploratory_proportions, df2_name="exploratory",
                                        df3=replication_proportions, df3_name="replication")
    comparison_df.to_csv(os.path.join(all_save_path, f"sample_balance_comparison.csv"), index=True)
    most_different_cols.to_csv(os.path.join(all_save_path, f"sample_balance_comparison_colDiff.csv"), index=True)


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
    Given the sample that we want to analyze (sample), send it to analysis, with the save path provided
    :param all_save_path: the path where all samples are saved, where we need to find the right one
    :param sample: exploratory, replication (folders under 'all_save_path')
    """

    # load the right sample:
    sub_df = pd.read_csv(os.path.join(all_save_path, f"{sample}_df.csv"))  # the df of all the responses
    with open(os.path.join(all_save_path, rf"{sample}_dict.pkl"), "rb") as f:
        sub_dict = pickle.load(f)  # the analysis dict containing dfs per blocks of questions

    # define the folder to save the results
    sample_save_path = os.path.join(all_save_path, f"{sample}")
    if not os.path.exists(sample_save_path):
        os.makedirs(sample_save_path)

    # call the analysis manager
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

