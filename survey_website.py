"""
Survey Data Privacy-Preserving Aggregation Generator for Minds Matter Website: https://ronyhirsch.github.io/minds-matter/

This script processes survey data and generates pre-aggregated JSON files that can be safely exposed via a static website
without revealing individual respondent identities.

Author: RonyHirsch
"""

import pandas as pd
import json
from pathlib import Path
from collections import Counter
import survey_mapping
import analyze_survey
import process_survey


K_THRESHOLD = 5  # privacy threshold

RATING_LABELS = {1: 'Does Not Have', 2: "Probably Doesn't Have", 3: 'Probably Has', 4: 'Has'}

# entity colors for scatter plots of attributions
ENTITY_COLORS = {
    "You": "#BC6266",
    "A newborn baby (human)": "#EABDC0",
    "A fetus (human; 24 weeks)": "#F6EDEC",
    "An orangutan": "#B47857",
    "A cow": "#887458",
    "A dog": "#BC984E",
    "A cat": "#D1A883",
    "A rat": "#3B3B3B",
    "A bat": "#536972",
    "A pigeon": "#7B869A",
    "A bee": "#FFB703",
    "A fruit-fly": "#D4A574",
    "An ant": "#7D4E37",
    "A mosquito": "#6B7280",
    "A turtle": "#23645C",
    "A salmon": "#eb5e28",
    "A lobster": "#A85D5D",
    "An octopus": "#842d72",
    "A sea urchin": "#8B6B99",
    "A tree": "#3D7D4A",
    "A fungus": "#8B7B68",
    "A large language model": "#134074",
    "A self-driving car": "#006E90",
    "An electron": "#5DADE2",
}

# columns which are immediately deleted and will never be exposed
TECHNICAL_COLS = [
    survey_mapping.demographics['StartDate'],
    survey_mapping.demographics['EndDate'],
    'Status',
    process_survey.COL_PROGRESS,
    process_survey.COL_FINISHED,
    survey_mapping.demographics['RecordedDate'],
    process_survey.COL_ID,
    survey_mapping.demographics['UserLanguage'],
    process_survey.COL_BOT,
    process_survey.COL_CONSENT,
]

# demographics; will be exposed with care
DEMOGRAPHIC_COLS = {
    'gender': survey_mapping.Q_GENDER,
    'age': 'age_group',
    'country': survey_mapping.Q_COUNTRY,
    'employment': survey_mapping.Q_EMPLOYMENT,
    'education': survey_mapping.Q_EDU,  # Education background
}

# experience; will be exposed with care
EXPERIENCE_LEVEL_COLS = {
    'ai_exp': survey_mapping.Q_AI_EXP,
    'animal_exp': survey_mapping.Q_ANIMAL_EXP,
    'consciousness_exp': survey_mapping.Q_CONSC_EXP,
    'ethics_exp': survey_mapping.Q_ETHICS_EXP,
}

# experience source; will be exposed with care
EXPERIENCE_SOURCE_COLS = {
    'ai_source': survey_mapping.Q_AI_EXP_FOLLOW_UP,
    'animal_types': survey_mapping.Q_ANIMAL_EXP_FOLLOW_UP,
    'consciousness_source': survey_mapping.Q_CONSC_EXP_FOLLOW_UP,
    'ethics_source': survey_mapping.Q_ETHICS_EXP_FOLLOW_UP,
}

# binary (Yes/No) questions in experience
EXPERIENCE_SIMPLE_COLS = {
    'pets': survey_mapping.Q_PETS,
}

# questions where participants could have selected one (or more) of several choices
MULTI_SELECT_COLS = {
    'education_topic': survey_mapping.demographics['field'],
    'no_kill_reason': survey_mapping.important_test_kill['all_nos'],
    'moral_considerations_important': survey_mapping.moral_considerations_features['m_c'],
    'moral_considerations_most': survey_mapping.moral_considerations_features['m_c_multi_prio'],
}

# questions to which participants responded with free-text
FREE_TEXT_COLS = {
    'goals_without_consciousness_example': survey_mapping.ics['ics_iWc_example'],
    'consciousness_without_goals_example': survey_mapping.ics['ics_cWi_example'],
    'sensations_without_consciousness_example': survey_mapping.ics['ics_sWc_example'],
    'consciousness_without_sensations_example': survey_mapping.ics['ics_cWs_example'],
    'no_kill_other': survey_mapping.important_test_kill['all_nos_other'],
    'moral_considerations_other': survey_mapping.moral_considerations_features['m_c_other'],
    'moral_considerations_most_important_other': survey_mapping.moral_considerations_features['m_c_multi_prio_other'],
    'higher_moral_status_people': survey_mapping.PRIOS_Q_PEOPLE_WHAT,
    'higher_moral_status_animals': survey_mapping.PRIOS_Q_ANIMALS_WHAT,
    'consciousness_intelligence_common': survey_mapping.con_intellect['con_intellect_yes_fu'],
    'ai_exp_other': survey_mapping.ai_exp['ai_exp_other'],
    'animals_other': survey_mapping.animal_exp['animal_other'],
    'consciousness_exp_other': survey_mapping.consciousness_exp['consci_exp_other'],
    'ethics_exp_other': survey_mapping.ethics_exp['ethics_exp_other'],
    'education_other': survey_mapping.demographics['field_other'],
    'employment_other': survey_mapping.demographics['employment_other'],
}

# Theme columns for free-text responses that have thematic coding: what characterizes people/animals with higher ms
FREE_TEXT_THEME_COLS = {
    'higher_moral_status_people': f"{survey_mapping.PRIOS_Q_PEOPLE_WHAT[:-1]} - themes",
    'higher_moral_status_animals': f"{survey_mapping.PRIOS_Q_ANIMALS_WHAT[:-1]} - themes",
}

# free text questions organized by topic with display names
FREE_TEXT_BY_TOPIC = {
    'ics': {
        'label': 'Intentions, Consciousness, Sensations',
        'questions': [
            ('goals_without_consciousness_example', survey_mapping.ics['ics_iWc_example']),
            ('consciousness_without_goals_example', survey_mapping.ics['ics_cWi_example']),
            ('sensations_without_consciousness_example', survey_mapping.ics['ics_sWc_example']),
            ('consciousness_without_sensations_example', survey_mapping.ics['ics_cWs_example']),
        ]
    },
    'kill_test': {
        'label': 'Kill to Pass a Test',
        'questions': [
            ('no_kill_other', f'{survey_mapping.Q_NO_KILL_WHY} - "Other"'),
        ]
    },
    'moral_features': {
        'label': 'Important Features for Moral Considerations',
        'questions': [
            ('moral_considerations_other', 'Important feature - "Other"'),
            ('moral_considerations_most_important_other', 'Most important feature - "Other"'),
        ]
    },
    'moral_cons': {
        'label': 'Moral Considerations',
        'questions': [
            ('higher_moral_status_people', survey_mapping.PRIOS_Q_PEOPLE_WHAT),
            ('higher_moral_status_animals', survey_mapping.PRIOS_Q_ANIMALS_WHAT),
        ]
    },
    'c_intel': {
        'label': 'Consciousness & Intelligence',
        'questions': [
            ('consciousness_intelligence_common', f'{survey_mapping.ANS_THIRD} - {survey_mapping.Q_INTELLIGENCE_FU}'),
        ]
    },
    'experience': {
        'label': 'Experience',
        'questions': [
            ('ai_exp_other', 'AI Experience Source - "Other"'),
            ('animals_other', 'Animal Types - "Other"'),
            ('consciousness_exp_other', 'Consciousness Experience Source - "Other"'),
            ('ethics_exp_other', 'Ethics Experience Source - "Other"'),
        ]
    },
    'demo': {
        'label': 'Demographics',
        'questions': [
            ('employment_other', 'Employment - "Other"'),
        ]
    },
    'education': {
        'label': 'Education',
        'questions': [
            ('education_other', 'Higher Education Topic - "Other"'),
        ]
    },
}

# the "care" part of exposing with care
SAFE_DEMOGRAPHIC_PAIRS = [
    ('gender', 'age'), ('gender', 'education'), ('age', 'education'),
    ('education', 'employment'), ('gender', 'employment'),
]

# pretty names for the website
DISPLAY_NAMES = {}
for entity, short in survey_mapping.other_creatures_general_names.items():
    DISPLAY_NAMES[f'ms_{entity}'] = (f'MS: {short}', f'Moral Status: {entity}')
    DISPLAY_NAMES[f'c_{entity}'] = (f'C: {short}', f'Consciousness: {entity}')

DISPLAY_NAMES[survey_mapping.Q_GENDER] = ('Gender', 'Gender')
DISPLAY_NAMES[survey_mapping.Q_AGE] = ('Age', 'Age')
DISPLAY_NAMES['age_group'] = ('Age Group', 'Age Group')
DISPLAY_NAMES[survey_mapping.Q_COUNTRY] = ('Country', 'Country of Residence')
DISPLAY_NAMES[survey_mapping.Q_EDU] = ('Education', 'Education Background')
DISPLAY_NAMES[survey_mapping.Q_EMPLOYMENT] = ('Employment', 'Employment Domain')
DISPLAY_NAMES[survey_mapping.Q_AI_EXP] = ('AI Experience', 'Experience with Artificial Intelligence')
DISPLAY_NAMES[survey_mapping.Q_ANIMAL_EXP] = ('Animal Experience', 'Experience with Animals')
DISPLAY_NAMES[survey_mapping.Q_CONSC_EXP] = ('Consciousness Experience', 'Experience with Consciousness Science')
DISPLAY_NAMES[survey_mapping.Q_ETHICS_EXP] = ('Ethics Experience', 'Experience with Ethics')
DISPLAY_NAMES[survey_mapping.Q_AI_EXP_FOLLOW_UP] = ('AI Experience Source', 'How did you get your AI experience?')
DISPLAY_NAMES[survey_mapping.Q_ANIMAL_EXP_FOLLOW_UP] = ('Animal Types', 'What animals have you worked with?')
DISPLAY_NAMES[survey_mapping.Q_CONSC_EXP_FOLLOW_UP] = (
    'Consciousness Experience Source', 'How did you get your consciousness knowledge?')
DISPLAY_NAMES[survey_mapping.Q_ETHICS_EXP_FOLLOW_UP] = (
    'Ethics Experience Source', 'How did you get your ethics knowledge?')
DISPLAY_NAMES[survey_mapping.demographics['field']] = ('Education Topic', 'Education Topic')
DISPLAY_NAMES[survey_mapping.important_test_kill['all_nos']] = ('No Kill Reason', 'Reason for not killing any creature')

# Important Features for Moral Status - here we want the full texts
DISPLAY_NAMES[survey_mapping.moral_considerations_features['m_c']] = (
    'What do you think is important for moral considerations?',
    'What do you think is important for moral considerations?')
DISPLAY_NAMES[survey_mapping.moral_considerations_features['m_c_multi_prio']] = (
    'Which do you think is the most important for moral considerations?',
    'Which do you think is the most important for moral considerations?')

# ICS display names (short for charts, full for option labels)
DISPLAY_NAMES[survey_mapping.ICS_Q_INT_WO_CONS] = ('Intentions w/o Consciousness?', survey_mapping.ICS_Q_INT_WO_CONS)
DISPLAY_NAMES[survey_mapping.ICS_Q_CONS_WO_INT] = ('Consciousness w/o Intentions?', survey_mapping.ICS_Q_CONS_WO_INT)
DISPLAY_NAMES[survey_mapping.ICS_Q_SENS_WO_CONS] = ('Sensations w/o Consciousness?', survey_mapping.ICS_Q_SENS_WO_CONS)
DISPLAY_NAMES[survey_mapping.ICS_Q_CONS_WO_SENS] = ('Consciousness w/o Sensations?', survey_mapping.ICS_Q_CONS_WO_SENS)

# Question option labels (for sidebar checkboxes - different from chart titles)
QUESTION_OPTION_LABELS = {
    # ICS - show the rest of the sentence after "Do you think a creature/system can..."
    survey_mapping.ICS_Q_INT_WO_CONS: 'have intentions/goals without being conscious?',
    survey_mapping.ICS_Q_CONS_WO_INT: 'be conscious without having intentions/goals?',
    survey_mapping.ICS_Q_SENS_WO_CONS: 'have positive or negative sensations (pleasure/pain) without being conscious?',
    survey_mapping.ICS_Q_CONS_WO_SENS: 'be conscious without having positive or negative sensations (pleasure/pain)?',

    # Moral Considerations - full questions
    survey_mapping.PRIOS_Q_NONCONS: survey_mapping.PRIOS_Q_NONCONS,
    survey_mapping.PRIOS_Q_CONS: survey_mapping.PRIOS_Q_CONS,
    survey_mapping.PRIOS_Q_PEOPLE: survey_mapping.PRIOS_Q_PEOPLE,
    survey_mapping.PRIOS_Q_ANIMALS: survey_mapping.PRIOS_Q_ANIMALS,

    # Graded Consciousness - full statements
    survey_mapping.Q_GRADED_EQUAL: survey_mapping.Q_GRADED_EQUAL,
    survey_mapping.Q_GRADED_UNEQUAL: survey_mapping.Q_GRADED_UNEQUAL,
    survey_mapping.Q_GRADED_MATTERMORE: survey_mapping.Q_GRADED_MATTERMORE,
    survey_mapping.Q_GRADED_INCOMP: survey_mapping.Q_GRADED_INCOMP,

    # Consciousness & Intelligence - full questions
    survey_mapping.Q_INTELLIGENCE: survey_mapping.Q_INTELLIGENCE,
    survey_mapping.con_intellect['con_intellect_yes']: 'How?',
}

# Kill to Pass a Test option labels - use short names for sidebar/table
# Create mapping from column name to short name and stripped token
KILL_OPTION_LABELS = {}  # For sidebar: column -> short name
KILL_DISPLAY_NAMES = {}  # For plots: column -> stripped token

# First, create reverse mapping: full_token -> column_name
token_to_col = {v: k for k, v in survey_mapping.important_test_kill.items() if k not in ('all_nos', 'all_nos_other')}

for full_token, short_name in survey_mapping.important_test_kill_tokens.items():
    if full_token in token_to_col:
        col_name = full_token  # The column name IS the full token
        # Strip the prefix for display
        stripped = full_token
        if stripped.startswith("A creature/system that "):
            stripped = stripped[len("A creature/system that "):]
        # Sidebar/table label: use stripped text (without "A creature/system that " prefix)
        KILL_OPTION_LABELS[col_name] = stripped
        # Plot label: use same stripped token
        KILL_DISPLAY_NAMES[col_name] = stripped

# Add Kill sidebar labels to QUESTION_OPTION_LABELS
QUESTION_OPTION_LABELS.update(KILL_OPTION_LABELS)

# Kill to Pass a Test display names - use stripped tokens for plots
for col_name, stripped in KILL_DISPLAY_NAMES.items():
    DISPLAY_NAMES[col_name] = (stripped, col_name)

# Earth in Danger display names (full question text, no prefix)
for key, question in survey_mapping.earth_in_danger.items():
    # full question as short name (will be wrapped in display)
    DISPLAY_NAMES[question] = (question, question)

DISPLAY_NAMES['Cluster'] = ('Earth in Danger Clusters', 'Earth in Danger Response Clusters')
DISPLAY_NAMES['group'] = ('ICS Group', 'ICS Response Group')

# Graded Consciousness display names
DISPLAY_NAMES[survey_mapping.Q_GRADED_EQUAL] = ('Graded: Equal', survey_mapping.Q_GRADED_EQUAL)
DISPLAY_NAMES[survey_mapping.Q_GRADED_UNEQUAL] = ('Graded: Unequal', survey_mapping.Q_GRADED_UNEQUAL)
DISPLAY_NAMES[survey_mapping.Q_GRADED_MATTERMORE] = ('Graded: Matters More', survey_mapping.Q_GRADED_MATTERMORE)
DISPLAY_NAMES[survey_mapping.Q_GRADED_INCOMP] = ('Graded: Incomparable', survey_mapping.Q_GRADED_INCOMP)

# Moral Considerations display names
DISPLAY_NAMES[survey_mapping.PRIOS_Q_NONCONS] = ('Non-conscious beings', survey_mapping.PRIOS_Q_NONCONS)
DISPLAY_NAMES[survey_mapping.PRIOS_Q_CONS] = ('Conscious beings', survey_mapping.PRIOS_Q_CONS)
DISPLAY_NAMES[survey_mapping.PRIOS_Q_PEOPLE] = ('some people should have a higher moral status', survey_mapping.PRIOS_Q_PEOPLE)
DISPLAY_NAMES[survey_mapping.PRIOS_Q_ANIMALS] = ('some non-human animals should have a higher moral status', survey_mapping.PRIOS_Q_ANIMALS)

# Consciousness & Intelligence display names
DISPLAY_NAMES[survey_mapping.Q_INTELLIGENCE] = ('Intelligence ~ Consciousness?', survey_mapping.Q_INTELLIGENCE)
DISPLAY_NAMES[survey_mapping.con_intellect['con_intellect_yes']] = (
    'What do they share?', survey_mapping.con_intellect['con_intellect_yes'])

# Zombie Pill
DISPLAY_NAMES[survey_mapping.Q_ZOMBIE] = ('Zombie Pill', survey_mapping.Q_ZOMBIE)

QUESTION_GROUPS = {
    'Moral Status Attributions': list(survey_mapping.other_creatures_ms.values()),
    'Consciousness Attributions': list(survey_mapping.other_creatures_cons.values()),
    'Earth in Danger': list(survey_mapping.earth_in_danger.values()),
    'Intentions, Consciousness, Sensations': [survey_mapping.ICS_Q_INT_WO_CONS, survey_mapping.ICS_Q_CONS_WO_INT,
                                              survey_mapping.ICS_Q_SENS_WO_CONS,
                                              survey_mapping.ICS_Q_CONS_WO_SENS],
    'Kill to Pass a Test': [v for k, v in survey_mapping.important_test_kill.items() if
                            k not in ('all_nos', 'all_nos_other')],
    'Moral Considerations': [survey_mapping.PRIOS_Q_NONCONS, survey_mapping.PRIOS_Q_CONS, survey_mapping.PRIOS_Q_PEOPLE,
                             survey_mapping.PRIOS_Q_ANIMALS],
    'Graded Consciousness': [survey_mapping.Q_GRADED_EQUAL, survey_mapping.Q_GRADED_UNEQUAL,
                             survey_mapping.Q_GRADED_MATTERMORE, survey_mapping.Q_GRADED_INCOMP],
    'Consciousness & Intelligence': [survey_mapping.Q_INTELLIGENCE, survey_mapping.con_intellect['con_intellect_yes']],
    'Zombie Pill': [survey_mapping.Q_ZOMBIE],
    'Calculated Variables': ['Cluster', 'group'],
}

COLORS_EXPERIENCE_SCALE = {'1': '#7EA1C4', '2': '#6586AA', '3': '#4D6C90', '4': '#345176', '5': '#1B365C'}
COLORS_AGREEMENT_SCALE = {'1': '#DB5461', '2': '#fb9a99', '3': '#70a0a4',
                          '4': '#26818B'}  # Same as rating for graded consciousness
# Graded scale for Equal/Unequal/Incomparable questions - blue gradient
COLORS_GRADED_SCALE = {'1': '#8DA9C4', '2': '#6284A5', '3': '#386085', '4': '#0D3B66'}
COLORS_YES_NO = {survey_mapping.ANS_YES: '#134074', survey_mapping.ANS_NO: '#eb5e28'}
COLORS_RATING_SCALE = {'1': '#DB5461', '2': '#fb9a99', '3': '#70a0a4', '4': '#26818B'}
COLORS_KILL = {survey_mapping.ANS_KILL: '#134074', survey_mapping.ANS_NOKILL: '#eb5e28'}

# Education background colors (blue gradient from light to dark)
COLORS_EDUCATION = {
    survey_mapping.EDU_NONE: '#FFFFFF',  # White
    survey_mapping.EDU_PRIM: '#BFCDD9',  # Light blue-gray
    survey_mapping.EDU_SECD: '#809AB3',  # Medium blue
    survey_mapping.EDU_POSTSEC: '#40688C',  # Dark blue
    survey_mapping.EDU_GRAD: '#003566',  # Deep navy
}

# Consciousness & Intelligence "What do they share" colors
COLORS_C_INTEL_SHARE = {
    survey_mapping.ANS_C_NECESSARY: '#41629a',  # Consciousness necessary
    survey_mapping.ANS_I_NECESSARY: '#819dda',  # Intelligence necessary
    survey_mapping.ANS_SAME: '#04396c',  # They are the same
    survey_mapping.ANS_THIRD: '#bad5ff',  # Third factor
}

# EiD Cluster colors
COLORS_EID_CLUSTER = {
    'Anthropocentric': '#C29059',
    'Non-Anthropocentric': '#696A35',
    'anthropocentric': '#C29059',
    'non-anthropocentric': '#696A35',
}

# ICS Group colors
COLORS_ICS_GROUP = {
    'Cognitive-Agential': '#A84A7F',
    'Experiential': '#F6948E',
    'Multidimensional': '#F6C667',
    'Other': '#444B8E',
    'cognitive-agential': '#A84A7F',
    'experiential': '#F6948E',
    'multidimensional': '#F6C667',
    'other': '#444B8E',
}

# Moral features colors - editable dictionary for Important Features for Moral Considerations
COLORS_MORAL_FEATURES = {
    'Language': '#58a6ff',
    'Sensory abilities (detecting things through the senses)': '#f78166',
    'Feelings of pleasure and suffering': '#7ee787',
    'Planning, goals': '#d2a8ff',
    'Self-awareness': '#ffa657',
    'Something it is like to be that creature/system': '#79c0ff',
    'Thinking': '#ff7b72',
    'Other': '#a5d6ff'
}

# Animal types colors - editable dictionary for Experience Sources > Animal Types
COLORS_ANIMAL_TYPES = {
    'Dogs': '#58a6ff',
    'Cats': '#f78166',
    'Birds': '#7ee787',
    'Fish': '#d2a8ff',
    'Rodents': '#ffa657',
    'Reptiles': '#79c0ff',
    'Livestock': '#ff7b72',
    'Other marine life': '#a5d6ff'
}

# Earth in Danger answer colors - harmonious palette
COLORS_EARTH_DANGER = {
    'Person': '#006E90',  # Deep teal
    'Dog': '#E89B87',  # Light terracotta
    'My pet': '#E07A5F',  # Terracotta
    'Dictator': '#F2CC8F',  # Warm sand
    'UWS': '#114b5f',  # Dark teal
    'Fruit fly': '#E76F51',  # Burnt sienna
    'AI': '#A8DADC',  # Soft cyan
    'Dictator (person)': '#F2CC8F',
    'Person (unresponsive wakefulness syndrome)': '#114b5f',
    'Fruit fly (a conscious one, for sure)': '#81B29A',  # Sage green
    'AI (that tells you that it\'s conscious)': '#A8DADC'
}

# Gender colors - updated palette
COLORS_GENDER = {
    'Male': '#FCBF49',  # Golden yellow
    'Female': '#6a994e',  # Sage green
    'Non-binary': '#bc6c25',  # Burnt orange
    'Genderqueer': '#a53860',  # Deep magenta
    'Other': '#7a8b99',  # Steel blue-gray
    'Prefer not to say': '#b8a88a',  # Warm beige
}

# Age group colors - warm gradient from light to dark
COLORS_AGE = {
    '18-25': '#FDF0D5',
    '26-35': '#E7C8B2',
    '36-45': '#D1A08E',
    '46-55': '#BB786B',
    '56-65': '#A45047',
    '66-75': '#8E2824',
    '76+': '#780000'
}


def suppress_small_cells(data, threshold=K_THRESHOLD):
    """Apply k-anonymity suppression to cell counts below threshold."""
    if isinstance(data, dict):
        return {k: (v if v >= threshold else f'<{threshold}') for k, v in data.items()}
    return data


def get_distribution(df, col):
    """Get distribution for a single variable - no suppression needed for privacy
    since single variable distributions don't reveal individual-level data."""
    counts = df[col].value_counts().to_dict()
    counts = {str(k): int(v) for k, v in counts.items()}
    return counts  # No suppression for single variables


def get_crosstab(df, col1, col2, threshold=K_THRESHOLD):
    """Get crosstab with k-anonymity suppression to protect privacy."""
    ct = pd.crosstab(df[col1], df[col2])
    result = {}
    for idx in ct.index:
        result[str(idx)] = {}
        for col in ct.columns:
            val = int(ct.loc[idx, col])
            result[str(idx)][str(col)] = val if val >= threshold else f'<{threshold}'
    return result


def extract_multiselect_options(series, known_options=None):
    """Extract and count options from multiselect responses.

    Args:
        series: Pandas series with comma-separated values
        known_options: Optional list of known valid options to match against
    """
    all_options = []
    for val in series.dropna():
        if isinstance(val, str):
            if known_options:
                # Match known options (handles options with internal commas like "Planning, goals")
                remaining = val
                for opt in known_options:
                    if opt in remaining:
                        all_options.append(opt)
                        remaining = remaining.replace(opt, '', 1)
            else:
                # Simple comma split for unknown options
                all_options.extend([x.strip() for x in val.split(',')])
    return Counter(all_options)


def count_multiselect_respondents(series):
    """Count unique respondents who answered a multiselect question.
    A respondent counts if they have any non-empty response.
    """
    count = 0
    for val in series.dropna():
        if isinstance(val, str) and val.strip() and val.strip().lower() != 'nan':
            count += 1
    return count


# Known multi-select options for proper parsing
KNOWN_MULTISELECT_OPTIONS = {
    'moral_considerations_important': [
        'Language',
        'Sensory abilities (detecting things through the senses)',
        'Feelings of pleasure and suffering',
        'Planning, goals',  # This one has an internal comma!
        'Self-awareness',
        'Something it is like to be that creature/system',
        'Thinking',
        'Other'
    ],
    'moral_considerations_most': [
        'Language',
        'Sensory abilities (detecting things through the senses)',
        'Feelings of pleasure and suffering',
        'Planning, goals',  # This one has an internal comma!
        'Self-awareness',
        'Something it is like to be that creature/system',
        'Thinking',
        'Other'
    ],
    # AI experience source options
    'ai_source': [
        survey_mapping.ANS_AI_ACADEMIA,
        survey_mapping.ANS_AI_PERSON,
        survey_mapping.ANS_AI_PRACTICAL,
        survey_mapping.ANS_AI_PROF,
        survey_mapping.ANS_AI_RESEARCH,
        survey_mapping.ANS_OTHER,
    ],
    # Consciousness experience source options
    'consciousness_source': [
        survey_mapping.ANS_C_ACADEMIA,
        survey_mapping.ANS_C_PERSON,
        survey_mapping.ANS_C_PROF,
        survey_mapping.ANS_C_RESEARCH,
        survey_mapping.ANS_OTHER,
    ],
    # Ethics source has internal comma in "Professional (work involving ethical decisions, law/medicine/social work)"
    'ethics_source': [
        survey_mapping.ANS_E_ACADEMIA,
        survey_mapping.ANS_E_PERSON,
        survey_mapping.ANS_E_PROF,
        survey_mapping.ANS_E_VOLUN,
        survey_mapping.ANS_E_RELIGION,
        survey_mapping.ANS_OTHER,
    ],
    # Animal types - these are the actual values in the data
    'animal_types': [
        'Birds', 'Cats', 'Dogs', 'Livestock', 'Rodents', 'Fish',
        'Reptiles', 'Other marine life', 'Insects', 'Primates (including apes)',
        'Cephalopods', 'Bats', 'Other'
    ],
    # No kill reasons
    'no_kill_reason': [
        survey_mapping.ANS_ALLNOS_IMMORAL,
        survey_mapping.ANS_ALLNOS_KILL,
        survey_mapping.ANS_ALLNOS_INTERESTS,
        survey_mapping.ANS_OTHER,
    ],
}


def get_free_text_responses(df, col, filter_col=None, filter_value=None, theme_col=None):
    """Get free text responses from a column, optionally filtering by another column containing a value.

    Args:
        df: DataFrame to extract from
        col: Column containing free text responses
        filter_col: Optional column to check for filter condition
        filter_value: If filter_col is specified, only include responses where filter_col contains this value
                     as a complete option (handles comma-separated multiselect values)
        theme_col: Optional column containing semicolon-separated themes for each response
    
    Returns:
        If theme_col is None: list of response strings
        If theme_col is specified: list of dicts {'text': str, 'themes': list[str]}
    """
    if filter_col is not None and filter_value is not None and filter_col in df.columns:
        # Filter to only rows where filter_col contains filter_value as a complete option
        # This handles comma-separated multiselect values like "Employed, Other" or "Other, Unemployed"
        # Case-insensitive comparison
        def has_option(cell_value, option):
            if pd.isna(cell_value):
                return False
            cell_str = str(cell_value)
            # Split by comma and check each option (case-insensitive)
            options = [opt.strip().lower() for opt in cell_str.split(',')]
            return option.lower() in options

        mask = df[filter_col].apply(lambda x: has_option(x, filter_value))
        filtered_df = df[mask]
    else:
        filtered_df = df

    # Get valid responses mask
    responses = filtered_df[col].copy()
    valid_mask = responses.notna() & (responses.astype(str).str.strip() != '') & (responses.astype(str).str.lower() != 'nan')
    
    if theme_col is not None and theme_col in filtered_df.columns:
        # Return responses with themes
        result = []
        valid_indices = responses[valid_mask].index
        for idx in valid_indices:
            text = str(responses.loc[idx]).strip()
            theme_val = filtered_df.loc[idx, theme_col]
            if pd.isna(theme_val) or str(theme_val).strip() == '':
                themes = []
            else:
                # Split by '; ' (semicolon + space)
                themes = [t.strip() for t in str(theme_val).split(';') if t.strip()]
            result.append({'text': text, 'themes': themes})
        return result
    else:
        # Return plain list of strings (original behavior)
        responses = responses[valid_mask].astype(str)
        return responses.tolist()


def get_survey_cols(df):
    """
    Return dict of survey columns excluding technical/demographic/free-text columns
    """
    exclude = set(TECHNICAL_COLS)
    exclude.update(DEMOGRAPHIC_COLS.values())
    exclude.update(EXPERIENCE_LEVEL_COLS.values())
    exclude.update(EXPERIENCE_SOURCE_COLS.values())
    exclude.update(EXPERIENCE_SIMPLE_COLS.values())
    exclude.update(MULTI_SELECT_COLS.values())
    exclude.update(FREE_TEXT_COLS.values())
    survey_cols = {}
    for col in df.columns:
        if col not in exclude:
            # Skip columns with _y or _x suffix (duplicate columns from merges)
            if col.endswith('_y') or col.endswith('_x'):
                continue
            base_key = col.replace(' ', '_').replace('/', '_').replace('?', '').replace("'", '').replace(',', '')[:50]
            # Handle key collisions by adding suffix
            key = base_key
            suffix = 2
            while key in survey_cols:
                key = f"{base_key[:47]}_{suffix}"
                suffix += 1
            survey_cols[key] = col
    return survey_cols


def generate_distributions_for_filter(df, survey_cols):
    """
    Generate all distribution dicts for a filtered subset of respondents
    """
    distributions = {}
    multiselect_respondent_counts = {}

    for key, col in DEMOGRAPHIC_COLS.items():
        if col in df.columns:
            distributions[f'demo_{key}'] = get_distribution(df, col)
    for key, col in EXPERIENCE_LEVEL_COLS.items():
        if col in df.columns:
            distributions[f'exp_{key}'] = get_distribution(df, col)
    for key, col in EXPERIENCE_SOURCE_COLS.items():
        if col in df.columns:
            # Use known options if available (for sources with internal commas like ethics_source)
            known_opts = KNOWN_MULTISELECT_OPTIONS.get(key)
            # No suppression for single variable distributions
            distributions[f'expsrc_{key}'] = dict(extract_multiselect_options(df[col], known_opts))
            multiselect_respondent_counts[f'expsrc_{key}'] = count_multiselect_respondents(df[col])

            # Calculate exclusive counts for experience sources (people who chose ONLY that option)
            if known_opts:
                exclusive_counts = {}
                for opt in known_opts:
                    exclusive_counts[opt] = 0
                # Count respondents who selected ONLY one option
                for val in df[col].dropna():
                    val_str = str(val).strip()
                    if not val_str:
                        continue
                    # Extract all options selected
                    selected = []
                    remaining = val_str
                    for opt in known_opts:
                        if opt in remaining:
                            selected.append(opt)
                            remaining = remaining.replace(opt, '', 1)
                    # If exactly one option selected, it's exclusive
                    if len(selected) == 1:
                        exclusive_counts[selected[0]] += 1
                distributions[f'expsrc_{key}_exclusive'] = exclusive_counts
    # Simple Yes/No questions in the Experience domain (like "Do you have a pet?")
    for key, col in EXPERIENCE_SIMPLE_COLS.items():
        if col in df.columns:
            distributions[f'expq_{key}'] = get_distribution(df, col)
    for key, col in MULTI_SELECT_COLS.items():
        if col in df.columns:
            known_opts = KNOWN_MULTISELECT_OPTIONS.get(key)
            # No suppression for single variable distributions
            distributions[f'multiselect_{key}'] = dict(extract_multiselect_options(df[col], known_opts))
            multiselect_respondent_counts[f'multiselect_{key}'] = count_multiselect_respondents(df[col])

            # For no_kill_reason, also calculate exclusive counts (people who chose ONLY that option)
            if key == 'no_kill_reason':
                exclusive_counts = {}
                if known_opts:
                    for opt in known_opts:
                        exclusive_counts[opt] = 0
                    # Count respondents who selected ONLY one option
                    for val in df[col].dropna():
                        val_str = str(val).strip()
                        if not val_str:
                            continue
                        # Extract all options selected
                        selected = []
                        remaining = val_str
                        for opt in known_opts:
                            if opt in remaining:
                                selected.append(opt)
                                remaining = remaining.replace(opt, '', 1)
                        # If exactly one option selected, it's exclusive
                        if len(selected) == 1:
                            exclusive_counts[selected[0]] += 1
                distributions['multiselect_no_kill_reason_exclusive'] = exclusive_counts
    for key, col in survey_cols.items():
        if col in df.columns:
            distributions[f'survey_{key}'] = get_distribution(df, col)

    # Generate combined "most important" that includes single-feature selections
    # When someone selected only ONE feature in "Important", they weren't asked "Most Important"
    # But their single selection IS effectively their most important
    important_col = survey_mapping.moral_considerations_features['m_c']
    most_col = survey_mapping.moral_considerations_features['m_c_multi_prio']

    if important_col in df.columns:
        # Count single-feature selections (those who selected exactly one feature)
        single_feature_counts = Counter()
        combined_most_counts = Counter()

        for idx, row in df.iterrows():
            important_val = row.get(important_col)
            most_val = row.get(most_col)

            # Check if important_val is a single selection (no comma = one feature)
            if pd.notna(important_val) and isinstance(important_val, str):
                important_str = str(important_val).strip()
                # Count features selected
                known_opts = KNOWN_MULTISELECT_OPTIONS.get('moral_considerations_important', [])
                features_selected = []
                if known_opts:
                    remaining = important_str
                    for opt in known_opts:
                        if opt in remaining:
                            features_selected.append(opt)
                            remaining = remaining.replace(opt, '', 1)
                else:
                    features_selected = [x.strip() for x in important_str.split(',')]

                # If exactly one feature selected, it's their "most important" implicitly
                if len(features_selected) == 1:
                    single_feature_counts[features_selected[0]] += 1
                    combined_most_counts[features_selected[0]] += 1
                elif pd.notna(most_val) and isinstance(most_val, str) and most_val.strip():
                    # Multiple features selected, use their explicit "most important" choice
                    combined_most_counts[most_val.strip()] += 1

        # Store the single-feature counts separately (no suppression for single variables)
        distributions['multiselect_moral_considerations_single'] = dict(single_feature_counts)
        # Store the combined most important (explicit + single-feature implied)
        distributions['multiselect_moral_considerations_combined_most'] = dict(combined_most_counts)
        multiselect_respondent_counts['multiselect_moral_considerations_combined_most'] = sum(
            combined_most_counts.values())

    return distributions, multiselect_respondent_counts


def generate_entity_stats(df, entities):
    """
    Compute summary statistics (mean, std, sem, n) for consciousness and moral status per entity
    """
    entity_stats = {}
    for entity in entities:
        c_col, ms_col = f'c_{entity}', f'ms_{entity}'
        if c_col in df.columns and ms_col in df.columns:
            c_data = pd.to_numeric(df[c_col], errors='coerce').dropna()
            ms_data = pd.to_numeric(df[ms_col], errors='coerce').dropna()
            paired = df[[c_col, ms_col]].dropna()
            entity_stats[entity] = {
                'short_name': survey_mapping.other_creatures_general_names[entity],
                'c_mean': float(c_data.mean()) if len(c_data) > 0 else None,
                'c_std': float(c_data.std()) if len(c_data) > 1 else None,
                'c_sem': float(c_data.sem()) if len(c_data) > 1 else None,
                'c_n': int(len(c_data)),
                'ms_mean': float(ms_data.mean()) if len(ms_data) > 0 else None,
                'ms_std': float(ms_data.std()) if len(ms_data) > 1 else None,
                'ms_sem': float(ms_data.sem()) if len(ms_data) > 1 else None,
                'ms_n': int(len(ms_data)),
                'paired_n': int(len(paired)),
                'scatter_data': paired[[c_col, ms_col]].values.tolist() if len(paired) >= K_THRESHOLD else [],
                # Raw values for JS merging
                'consciousness_values': c_data.tolist(),
                'moral_status_values': ms_data.tolist(),
                'paired_data': paired[[c_col, ms_col]].values.tolist() if len(paired) > 0 else [],
            }
    return entity_stats


def generate_survey_stats(df):
    """
    Compute overall survey metadata: total N, duration stats, date range
    """
    stats = {'total_respondents': int(len(df)), 'duration': {}, 'date_range': {}}
    dur_col = process_survey.COL_DUR_SEC
    if dur_col in df.columns:
        dur = pd.to_numeric(df[dur_col], errors='coerce').dropna()
        stats['duration']['all'] = {'mean': float(dur.mean()), 'std': float(dur.std()), 'sem': float(dur.sem()),
                                    'n': int(len(dur))}
        if 'sample' in df.columns:
            stats['duration']['by_sample'] = {}
            for sample in df['sample'].dropna().unique():
                sub = pd.to_numeric(df[df['sample'] == sample][dur_col], errors='coerce').dropna()
                if len(sub) >= K_THRESHOLD:
                    stats['duration']['by_sample'][sample] = {'mean': float(sub.mean()), 'std': float(sub.std()),
                                                              'sem': float(sub.sem()), 'n': int(len(sub))}
        if 'source' in df.columns:
            stats['duration']['by_source'] = {}
            for source in df['source'].dropna().unique():
                sub = pd.to_numeric(df[df['source'] == source][dur_col], errors='coerce').dropna()
                if len(sub) >= K_THRESHOLD:
                    stats['duration']['by_source'][source] = {'mean': float(sub.mean()), 'std': float(sub.std()),
                                                              'sem': float(sub.sem()), 'n': int(len(sub))}
        if 'sample' in df.columns and 'source' in df.columns:
            stats['duration']['by_sample_source'] = {}
            for sample in df['sample'].dropna().unique():
                for source in df['source'].dropna().unique():
                    sub = pd.to_numeric(df[(df['sample'] == sample) & (df['source'] == source)][dur_col],
                                        errors='coerce').dropna()
                    if len(sub) >= K_THRESHOLD:
                        stats['duration']['by_sample_source'][f'{sample}_{source}'] = {'mean': float(sub.mean()),
                                                                                       'std': float(sub.std()),
                                                                                       'sem': float(sub.sem()),
                                                                                       'n': int(len(sub))}
    start_col = survey_mapping.demographics['StartDate']
    if start_col in df.columns:
        try:
            dates = pd.to_datetime(df[start_col], errors='coerce').dropna()
            if len(dates) > 0:
                stats['date_range']['all'] = {'start': dates.min().strftime('%B %Y'),
                                              'end': dates.max().strftime('%B %Y')}
                if 'sample' in df.columns:
                    stats['date_range']['by_sample'] = {}
                    for sample in df['sample'].dropna().unique():
                        sub_dates = pd.to_datetime(df[df['sample'] == sample][start_col], errors='coerce').dropna()
                        if len(sub_dates) >= K_THRESHOLD:
                            stats['date_range']['by_sample'][sample] = {'start': sub_dates.min().strftime('%B %Y'),
                                                                        'end': sub_dates.max().strftime('%B %Y')}
        except Exception as e:
            print(f"Warning: Could not parse dates: {e}")
    return stats


def generate_metadata(df, survey_cols):
    # Sort sample values in preferred order
    sample_order = {'preregistered': 0, 'pre-registered': 0, 'exploratory': 1, 'follow-up': 2, 'follow up': 2,
                    'replication': 3}
    sample_values_raw = df['sample'].dropna().unique().tolist() if 'sample' in df.columns else []
    sample_values = sorted(sample_values_raw, key=lambda x: sample_order.get(x.lower(), 99))

    source_values = sorted(df['source'].dropna().unique().tolist()) if 'source' in df.columns else []

    # Get cluster values for Earth in Danger split-by-cluster feature
    cluster_values = sorted(df['Cluster'].dropna().unique().tolist()) if 'Cluster' in df.columns else []

    # Build entity list with short names and colors
    entity_list = []
    for entity, short in survey_mapping.other_creatures_general_names.items():
        # Add "(24 weeks)" to Fetus for clarity
        display_short = short
        if 'fetus' in entity.lower():
            display_short = short + ' (24 weeks)'
        entity_list.append({
            'key': entity,
            'short': display_short,
            'color': ENTITY_COLORS.get(entity, '#58a6ff')
        })

    return {
        'total_respondents': int(len(df)),
        'total_questions': '>100',
        'demographics': {k: {'label': v, 'values': df[v].dropna().unique().tolist()} for k, v in
                         DEMOGRAPHIC_COLS.items() if v in df.columns},
        'experience_levels': {k: {'label': v, 'values': sorted(df[v].dropna().unique().tolist())} for k, v in
                              EXPERIENCE_LEVEL_COLS.items() if v in df.columns},
        'experience_sources': {k: {'label': v} for k, v in EXPERIENCE_SOURCE_COLS.items()},
        'experience_simple': {k: {'label': v, 'values': sorted(df[v].dropna().unique().tolist())} for k, v in
                              EXPERIENCE_SIMPLE_COLS.items() if v in df.columns},
        'multi_select': {k: {'label': v} for k, v in MULTI_SELECT_COLS.items()},
        'survey_questions': survey_cols,
        'free_text_questions': FREE_TEXT_COLS,
        'safe_demographic_pairs': SAFE_DEMOGRAPHIC_PAIRS,
        'k_threshold': K_THRESHOLD,
        'display_names': DISPLAY_NAMES,
        'question_groups': QUESTION_GROUPS,
        'rating_labels': RATING_LABELS,
        'entities': list(survey_mapping.other_creatures_general_names.keys()),
        'entity_list': entity_list,
        'entity_colors': ENTITY_COLORS,
        'entity_short_names': survey_mapping.other_creatures_general_names,
        'sample_values': sample_values,
        'source_values': source_values,
        'cluster_values': cluster_values,
        'question_option_labels': QUESTION_OPTION_LABELS,
        # Categories that WERE asked in the follow-up sample
        'followup_categories': [
            'ms_attr',  # Moral Status Attributions
            'c_attr',  # Consciousness Attributions
            'combined_attr',  # Entity Combined Attributions
            'moral_features',  # Important Features for Moral Considerations (only "Important", not "Most Important")
            'moral_cons',  # Moral Considerations (partial - NONCONS, CONS only)
            'experience',  # Experience levels (but NOT sources)
            'demo',  # Demographics
            'education',  # Education (background + field)
        ],
        # Categories NOT asked in follow-up (for "All respondents" note)
        # Note: moral_cons has some questions in follow-up (NONCONS, CONS) and some not (PEOPLE, ANIMALS)
        'not_in_followup': [
            'earth_danger',  # Earth in Danger
            'ics',  # ICS
            'kill_test',  # Kill for Test
            'graded_c',  # Graded Consciousness
            'c_intel',  # Consciousness & Intelligence
            'zombie_pill',  # Zombie Pill
            'g_Calculated Variables',  # Calculated Variables
            'ms_no_kill_reason',  # No Kill Reasons multiselect (related to kill_test)
        ],
        # Specific questions NOT in follow-up (for per-question notes)
        'questions_not_in_followup': [
            'some people should have a higher moral status',  # PRIOS_Q_PEOPLE display name
            'some non-human animals should have a higher moral status',  # PRIOS_Q_ANIMALS display name
        ],
        'colors': {
            'experience_scale': COLORS_EXPERIENCE_SCALE,
            'agreement_scale': COLORS_AGREEMENT_SCALE,
            'graded_scale': COLORS_GRADED_SCALE,
            'yes_no': COLORS_YES_NO,
            'rating_scale': COLORS_RATING_SCALE,
            'kill': COLORS_KILL,
            'earth_danger': COLORS_EARTH_DANGER,
            'moral_features': COLORS_MORAL_FEATURES,
            'animal_types': COLORS_ANIMAL_TYPES,
            'gender': COLORS_GENDER,
            'age': COLORS_AGE,
            'c_intel_share': COLORS_C_INTEL_SHARE,
            'eid_cluster': COLORS_EID_CLUSTER,
            'ics_group': COLORS_ICS_GROUP,
            'education': COLORS_EDUCATION,
            'entity': ENTITY_COLORS,
        },
        'education_order': [
            survey_mapping.EDU_NONE,
            survey_mapping.EDU_PRIM,
            survey_mapping.EDU_SECD,
            survey_mapping.EDU_POSTSEC,
            survey_mapping.EDU_GRAD
        ],
        'free_text_by_topic': FREE_TEXT_BY_TOPIC,
    }


def main(df_path, output_dir):
    """
    Generate all privacy-preserving JSON files for the survey website
    """
    print("=" * 60)
    print("SURVEY WEBSITE DATA GENERATOR - VERSION 3.62")
    print("=" * 60)
    print()

    df = pd.read_csv(df_path)
    print(f"Loaded {len(df)} respondents")

    if 'age_group' not in df.columns and survey_mapping.Q_AGE in df.columns:
        df['age_group'] = pd.cut(df[survey_mapping.Q_AGE], bins=analyze_survey.AGE_BINS,
                                 labels=analyze_survey.AGE_LABELS, include_lowest=True).astype(str)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    survey_cols = get_survey_cols(df)
    print(f"Found {len(survey_cols)} survey columns")

    # Check filters
    sample_vals = df['sample'].dropna().unique().tolist() if 'sample' in df.columns else []
    source_vals = df['source'].dropna().unique().tolist() if 'source' in df.columns else []
    print(f"\n=== Filter values ===")
    print(f"  sample: {sample_vals}")
    print(f"  source: {source_vals}")

    entities = list(survey_mapping.other_creatures_general_names.keys())

    # Metadata
    metadata = generate_metadata(df, survey_cols)
    print(f"\nMetadata sample_values: {metadata['sample_values']}")
    print(f"Metadata source_values: {metadata['source_values']}")
    print(f"Metadata entity_list count: {len(metadata['entity_list'])}")

    # Check Kill to Pass a Test items
    kpt_cols = QUESTION_GROUPS['Kill to Pass a Test']
    print(f"\n=== Kill to Pass a Test ({len(kpt_cols)} items) ===")
    for col in kpt_cols:
        dn = DISPLAY_NAMES.get(col, ('?', col))
        print(f"  {dn[0]}")

    # Check Earth in Danger items
    eid_cols = QUESTION_GROUPS['Earth in Danger']
    print(f"\n=== Earth in Danger ({len(eid_cols)} items) ===")
    for col in eid_cols:
        dn = DISPLAY_NAMES.get(col, ('?', col))
        in_survey = any(cn == col for cn in survey_cols.values())
        print(f"  {dn[0][:40]}... (in_data: {in_survey})")

    # Distributions
    distributions, multiselect_respondent_counts = generate_distributions_for_filter(df, survey_cols)

    # Add multiselect respondent counts to metadata
    metadata['multiselect_respondent_counts'] = multiselect_respondent_counts

    # Add theme info for themed free-text questions
    # All unique themes from PEOPLE_THEME_DEFS and ANIMALS_THEME_DEFS in freetext_explorer.py
    metadata['free_text_themes'] = {
        'higher_moral_status_people': [
            'consciousness/sentience',
            'intelligence/cognition', 
            'capacity to suffer/feel pain',
            'vulnerability',
            'ecological role/importance',
            'benefit or harm to other people (from people)'
        ],
        'higher_moral_status_animals': [
            'consciousness/sentience',
            'intelligence/cognition',
            'capacity to suffer/feel pain',
            'similarity/kinship to humans',
            'domestication/pets/companionship',
            'ecological role/importance',
            'endangerment/rarity',
            'benefit or harm to humans (from animals)',
            'taxa-based heuristics'
        ]
    }

    # Entity stats for scatter plots
    entity_stats_all = generate_entity_stats(df, entities)

    # Save
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f)
    print(f"\n>> metadata.json saved")

    with open(output_dir / 'distributions.json', 'w') as f:
        json.dump(distributions, f)
    print(f">> distributions.json saved")

    # Crosstabs
    crosstabs = {}
    for c1, c2 in metadata['safe_demographic_pairs']:
        col1, col2 = DEMOGRAPHIC_COLS.get(c1), DEMOGRAPHIC_COLS.get(c2)
        if col1 and col2 and col1 in df.columns and col2 in df.columns:
            crosstabs[f'demo_{c1}__demo_{c2}'] = get_crosstab(df, col1, col2)
    with open(output_dir / 'crosstabs.json', 'w') as f:
        json.dump(crosstabs, f)
    print(f">> crosstabs.json saved")

    # Free text - generate for all and each filter combination
    def generate_freetext_for_filter(sub_df):
        """Generate free text responses for a filtered subset.
        Since responses are already shown with group labels when viewing combined,
        there's no privacy benefit to suppressing individual subgroups.

        Special handling for employment_other: only include responses where 'Other'
        was actually selected in the employment question.
        
        For questions with theme columns (FREE_TEXT_THEME_COLS), responses are returned
        as objects with 'text' and 'themes' fields."""
        result = {}
        for key, col in FREE_TEXT_COLS.items():
            if col in sub_df.columns:
                # Check if this question has a theme column
                theme_col = FREE_TEXT_THEME_COLS.get(key)
                
                # Special handling for employment_other - only include if 'Other' was selected
                if key == 'employment_other':
                    employment_col = survey_mapping.Q_EMPLOYMENT
                    if employment_col in sub_df.columns:
                        responses = get_free_text_responses(sub_df, col, filter_col=employment_col,
                                                            filter_value='Other', theme_col=theme_col)
                    else:
                        responses = []
                else:
                    responses = get_free_text_responses(sub_df, col, theme_col=theme_col)
                
                if len(responses) > 0:
                    result[key] = {'response_count': len(responses), 'responses': responses}
        return result

    # Generate free text for all respondents
    freetext = {'all': generate_freetext_for_filter(df)}

    # Generate free text for each sample
    if 'sample' in df.columns:
        for smp in df['sample'].dropna().unique():
            sub = df[df['sample'] == smp]
            if len(sub) > 0:
                fkey = f'sample_{smp}'
                freetext[fkey] = generate_freetext_for_filter(sub)

    # Generate free text for each source
    if 'source' in df.columns:
        for src in df['source'].dropna().unique():
            sub = df[df['source'] == src]
            if len(sub) > 0:
                fkey = f'source_{src}'
                freetext[fkey] = generate_freetext_for_filter(sub)

    # Generate free text for each sample+source combination
    if 'sample' in df.columns and 'source' in df.columns:
        for smp in df['sample'].dropna().unique():
            for src in df['source'].dropna().unique():
                sub = df[(df['sample'] == smp) & (df['source'] == src)]
                if len(sub) > 0:
                    fkey = f'sample_{smp}__source_{src}'
                    freetext[fkey] = generate_freetext_for_filter(sub)

    with open(output_dir / 'freetext.json', 'w') as f:
        json.dump(freetext, f)
    print(f">> freetext.json saved (with filters)")

    # Survey stats
    stats = generate_survey_stats(df)
    with open(output_dir / 'survey_stats.json', 'w') as f:
        json.dump(stats, f)
    print(f">> survey_stats.json saved")

    # Entity stats
    entity_stats = {'all': entity_stats_all}
    with open(output_dir / 'entity_stats.json', 'w') as f:
        json.dump(entity_stats, f)
    print(f">> entity_stats.json saved")

    # Filtered distributions
    filtered = {}
    if 'sample' in df.columns:
        for smp in df['sample'].dropna().unique():
            sub = df[df['sample'] == smp]
            if len(sub) >= K_THRESHOLD:
                fkey = f'sample_{smp}'
                dists, resp_counts = generate_distributions_for_filter(sub, survey_cols)
                filtered[fkey] = {
                    'n': int(len(sub)),
                    'distributions': dists,
                    'multiselect_respondent_counts': resp_counts
                }
                entity_stats[fkey] = generate_entity_stats(sub, entities)
    if 'source' in df.columns:
        for src in df['source'].dropna().unique():
            sub = df[df['source'] == src]
            if len(sub) >= K_THRESHOLD:
                fkey = f'source_{src}'
                dists, resp_counts = generate_distributions_for_filter(sub, survey_cols)
                filtered[fkey] = {
                    'n': int(len(sub)),
                    'distributions': dists,
                    'multiselect_respondent_counts': resp_counts
                }
                entity_stats[fkey] = generate_entity_stats(sub, entities)
    if 'sample' in df.columns and 'source' in df.columns:
        for smp in df['sample'].dropna().unique():
            for src in df['source'].dropna().unique():
                sub = df[(df['sample'] == smp) & (df['source'] == src)]
                if len(sub) >= K_THRESHOLD:
                    fkey = f'sample_{smp}__source_{src}'
                    dists, resp_counts = generate_distributions_for_filter(sub, survey_cols)
                    filtered[fkey] = {
                        'n': int(len(sub)),
                        'distributions': dists,
                        'multiselect_respondent_counts': resp_counts
                    }
                    entity_stats[fkey] = generate_entity_stats(sub, entities)

    # Add cluster-based filtering (for Earth in Danger split-by-cluster feature)
    if 'Cluster' in df.columns:
        cluster_vals = df['Cluster'].dropna().unique().tolist()
        print(f"\n=== Adding cluster-based filters for {cluster_vals} ===")
        cluster_keys_created = []

        # By cluster only
        for cluster in cluster_vals:
            sub = df[df['Cluster'] == cluster]
            fkey = f'cluster_{cluster}'
            if len(sub) >= K_THRESHOLD:
                dists, resp_counts = generate_distributions_for_filter(sub, survey_cols)
                filtered[fkey] = {
                    'n': int(len(sub)),
                    'distributions': dists,
                    'multiselect_respondent_counts': resp_counts
                }
                entity_stats[fkey] = generate_entity_stats(sub, entities)
                cluster_keys_created.append(fkey)

        # By sample + cluster
        if 'sample' in df.columns:
            for smp in df['sample'].dropna().unique():
                for cluster in cluster_vals:
                    sub = df[(df['sample'] == smp) & (df['Cluster'] == cluster)]
                    fkey = f'sample_{smp}__cluster_{cluster}'
                    if len(sub) >= K_THRESHOLD:
                        dists, resp_counts = generate_distributions_for_filter(sub, survey_cols)
                        filtered[fkey] = {
                            'n': int(len(sub)),
                            'distributions': dists,
                            'multiselect_respondent_counts': resp_counts
                        }
                        entity_stats[fkey] = generate_entity_stats(sub, entities)
                        cluster_keys_created.append(fkey)

        # By source + cluster
        if 'source' in df.columns:
            for src in df['source'].dropna().unique():
                for cluster in cluster_vals:
                    sub = df[(df['source'] == src) & (df['Cluster'] == cluster)]
                    fkey = f'source_{src}__cluster_{cluster}'
                    if len(sub) >= K_THRESHOLD:
                        dists, resp_counts = generate_distributions_for_filter(sub, survey_cols)
                        filtered[fkey] = {
                            'n': int(len(sub)),
                            'distributions': dists,
                            'multiselect_respondent_counts': resp_counts
                        }
                        entity_stats[fkey] = generate_entity_stats(sub, entities)
                        cluster_keys_created.append(fkey)

        # By sample + source + cluster
        if 'sample' in df.columns and 'source' in df.columns:
            for smp in df['sample'].dropna().unique():
                for src in df['source'].dropna().unique():
                    for cluster in cluster_vals:
                        sub = df[(df['sample'] == smp) & (df['source'] == src) & (df['Cluster'] == cluster)]
                        fkey = f'sample_{smp}__source_{src}__cluster_{cluster}'
                        if len(sub) >= K_THRESHOLD:
                            dists, resp_counts = generate_distributions_for_filter(sub, survey_cols)
                            filtered[fkey] = {
                                'n': int(len(sub)),
                                'distributions': dists,
                                'multiselect_respondent_counts': resp_counts
                            }
                            entity_stats[fkey] = generate_entity_stats(sub, entities)
                            cluster_keys_created.append(fkey)

        print(f"  Created {len(cluster_keys_created)} cluster-based filter keys")

    with open(output_dir / 'filtered_distributions.json', 'w') as f:
        json.dump(filtered, f)
    print(f">> filtered_distributions.json saved")

    with open(output_dir / 'entity_stats.json', 'w') as f:
        json.dump(entity_stats, f)
    print(f">> entity_stats.json saved (with filters)")

    print(f"\n{'=' * 60}")
    print(f"Done! All files saved to {output_dir}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main(
        df_path=r"...\sub_df.csv",
        output_dir=r"...\minds-matter\data"
    )
