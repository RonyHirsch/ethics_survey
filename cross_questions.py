"""
Cross Questions Data Generator for Minds Matter Website: https://ronyhirsch.github.io/minds-matter/

This module generates pre-aggregated cross-tabulation data that can be safely
exposed via a static website without revealing individual respondent identities.

IMPORTANT: This is a completely separate module from survey_website.py.
It generates its own output file (cross_questions_data.json) and does not
modify any existing data files.

Safety is enforced at generation time - if a combination is forbidden,
it is simply not generated and cannot be displayed on the website.

Author: RonyHirsch
"""

import pandas as pd
import json
import re
from pathlib import Path
from datetime import datetime
import survey_mapping
import analyze_survey


# configurations
K_THRESHOLD = 5  # Minimum cell count for k-anonymity
MAX_COUNTRIES = 10  # Maximum number of countries to include per filter

# age binning (same as survey_website.py)
AGE_BINS = analyze_survey.AGE_BINS
AGE_LABELS = analyze_survey.AGE_LABELS

# multi-select answer options for "Why wouldn't you eliminate any creatures"
ANS_ALLNOS_OPTIONS = [
    survey_mapping.ANS_ALLNOS_KILL,       # "Because I wouldn't kill any creature regardless..."
    survey_mapping.ANS_ALLNOS_IMMORAL,    # "Because I think it would be immoral"
    survey_mapping.ANS_ALLNOS_INTERESTS,  # "Because I think all of them have interests..."
    survey_mapping.ANS_OTHER,             # "Other"
]

# Animal Types options (multiselect question about what types of animals they have/had experience with)
# These are extracted dynamically from the data, but we define common ones for reference
ANIMAL_TYPES_OPTIONS = [
    'Dogs',
    'Cats', 
    'Fish',
    'Birds',
    'Rodents (hamster, guinea pig, mouse/rat, rabbit, etc.)',
    'Reptiles/Amphibians',
    'Farm animals (horse, cow, pig, chicken, etc.)',
    'Other',
]

# Consciousness Experience Source options (multiselect)
# These contain commas so cannot be split on comma delimiter
CONSCIOUSNESS_SOURCE_OPTIONS = [
    'Personal Interest (reading texts about consciousness)',
    'Academic Background (studied/teach these topics in university)',
    'Research (conduct research related to consciousness)',
    'Professional Experience (work related to consciousness)',
    'Other',
]

# Ethics Experience Source options (multiselect)
# These contain commas so cannot be split on comma delimiter
# Values from survey_mapping.py: ANS_E_ACADEMIA, ANS_E_PERSON, ANS_E_PROF, ANS_E_VOLUN, ANS_E_RELIGION, ANS_OTHER
ETHICS_SOURCE_OPTIONS = [
    'Academic Background (studied/researched philosophy/ethics in university)',
    'Personal Interest (reading philosophical texts/participant in ethical debates)',
    'Professional (work involving ethical decisions, law/medicine/social work)',
    'Volunteer Work (in organizations focused on ethical issues/moral causes)',
    'Religious/Spiritual Practice (engagement with ethical teachings)',
    'Other',
]

# AI Experience Source options (multiselect)
AI_SOURCE_OPTIONS = [
    'Personal Interest (reading about AI)',
    'Academic Background (studied/teach AI in university)',
    'Research (conduct research related to AI)',
    'Professional Experience (work related to AI)',
    'Other',
]

# Map source key to its known options
EXPERIENCE_SOURCE_OPTIONS = {
    'consciousness_source': CONSCIOUSNESS_SOURCE_OPTIONS,
    'ethics_source': ETHICS_SOURCE_OPTIONS,
    'ai_source': AI_SOURCE_OPTIONS,
    'animal_types': ANIMAL_TYPES_OPTIONS,
}

# Short labels for the ANS_ALLNOS options (for display)
ANS_ALLNOS_SHORT_LABELS = {
    survey_mapping.ANS_ALLNOS_KILL: "Wouldn't kill any creature",
    survey_mapping.ANS_ALLNOS_IMMORAL: "It would be immoral",
    survey_mapping.ANS_ALLNOS_INTERESTS: "All have interests",
    survey_mapping.ANS_OTHER: "Other",
}


# column definitions

# Demographics - keys match what will be used in the JSON
DEMOGRAPHIC_COLS = {
    'gender': survey_mapping.Q_GENDER,
    'age': 'age_group',  # Derived column
    'country': survey_mapping.Q_COUNTRY,
    'employment': survey_mapping.Q_EMPLOYMENT,
    'education': survey_mapping.Q_EDU,
}

# Experience levels (1-5 scale)
EXPERIENCE_LEVEL_COLS = {
    'ai_exp': survey_mapping.Q_AI_EXP,
    'animal_exp': survey_mapping.Q_ANIMAL_EXP,
    'consciousness_exp': survey_mapping.Q_CONSC_EXP,
    'ethics_exp': survey_mapping.Q_ETHICS_EXP,
}

# Experience sources (multiselect - we'll use the binarized columns)
# These are the source question columns that will be split into binary columns
EXPERIENCE_SOURCE_COLS = {
    'ai_source': survey_mapping.Q_AI_EXP_FOLLOW_UP,
    'animal_types': survey_mapping.Q_ANIMAL_EXP_FOLLOW_UP,
    'consciousness_source': survey_mapping.Q_CONSC_EXP_FOLLOW_UP,
    'ethics_source': survey_mapping.Q_ETHICS_EXP_FOLLOW_UP,
}

# Simple Yes/No experience questions (not multiselect)
EXPERIENCE_SIMPLE_COLS = {
    'pets': survey_mapping.Q_PETS,
}

# Survey question groups - organized by topic
SURVEY_QUESTIONS = {
    # Zombie Pill (1 question)
    'zombie_pill': {
        'label': 'Zombie Pill',
        'columns': {
            'zombie': survey_mapping.Q_ZOMBIE,
        }
    },
    
    # ICS - Intentions, Consciousness, Sensations (4 questions)
    'ics': {
        'label': 'Intentions, Consciousness, Sensations',
        'columns': {
            'ics_int_wo_cons': survey_mapping.ICS_Q_INT_WO_CONS,
            'ics_cons_wo_int': survey_mapping.ICS_Q_CONS_WO_INT,
            'ics_sens_wo_cons': survey_mapping.ICS_Q_SENS_WO_CONS,
            'ics_cons_wo_sens': survey_mapping.ICS_Q_CONS_WO_SENS,
        }
    },
    
    # Graded Consciousness (4 questions)
    'graded': {
        'label': 'Graded Consciousness',
        'columns': {
            'graded_equal': survey_mapping.Q_GRADED_EQUAL,
            'graded_unequal': survey_mapping.Q_GRADED_UNEQUAL,
            'graded_mattermore': survey_mapping.Q_GRADED_MATTERMORE,
            'graded_incomp': survey_mapping.Q_GRADED_INCOMP,
        }
    },
    
    # Moral Considerations (4 questions)
    'moral_cons': {
        'label': 'Moral Considerations',
        'columns': {
            'moral_noncons': survey_mapping.PRIOS_Q_NONCONS,
            'moral_cons': survey_mapping.PRIOS_Q_CONS,
            'moral_people': survey_mapping.PRIOS_Q_PEOPLE,
            'moral_animals': survey_mapping.PRIOS_Q_ANIMALS,
        }
    },
    
    # Consciousness & Intelligence (2 questions)
    'c_intel': {
        'label': 'Consciousness & Intelligence',
        'columns': {
            'c_intel_main': survey_mapping.Q_INTELLIGENCE,
            'c_intel_how': survey_mapping.con_intellect['con_intellect_yes'],
        }
    },
    
    # Calculated Variables
    'calculated': {
        'label': 'Calculated Variables',
        'columns': {
            'cluster': 'Cluster',  # EiD Cluster
            'group': 'group',  # ICS Group
        }
    },
}

# Kill to Pass a Test - build dynamically from survey_mapping
KILL_QUESTIONS = {}
for key, col in survey_mapping.important_test_kill.items():
    if key not in ('all_nos_other',):  # Only exclude free-text 'other' field
        # Create a short key from the column name
        short_key = f'kill_{key}'
        KILL_QUESTIONS[short_key] = col

SURVEY_QUESTIONS['kill'] = {
    'label': 'Kill to Pass a Test',
    'columns': KILL_QUESTIONS
}

# Earth in Danger - build from survey_mapping
EID_QUESTIONS = {}
for key, col in survey_mapping.earth_in_danger.items():
    short_key = f'eid_{key}'
    EID_QUESTIONS[short_key] = col

SURVEY_QUESTIONS['earth_danger'] = {
    'label': 'Earth in Danger',
    'columns': EID_QUESTIONS
}

# Moral Features - Important for Moral Considerations
SURVEY_QUESTIONS['moral_features'] = {
    'label': 'Important Features for Moral Considerations',
    'columns': {
        'moral_important': survey_mapping.moral_considerations_features['m_c'],
        'moral_most_important': survey_mapping.moral_considerations_features['m_c_multi_prio'],
    }
}

# Attributions of Consciousness - all 24 entities
C_ATTR_QUESTIONS = {}
for entity_key, col_name in survey_mapping.other_creatures_cons.items():
    C_ATTR_QUESTIONS[f'c_{entity_key}'] = col_name

SURVEY_QUESTIONS['c_attr'] = {
    'label': 'Attributions of Consciousness',
    'columns': C_ATTR_QUESTIONS
}

# Attributions of Moral Status - all 24 entities
MS_ATTR_QUESTIONS = {}
for entity_key, col_name in survey_mapping.other_creatures_ms.items():
    MS_ATTR_QUESTIONS[f'ms_{entity_key}'] = col_name

SURVEY_QUESTIONS['ms_attr'] = {
    'label': 'Attributions of Moral Status',
    'columns': MS_ATTR_QUESTIONS
}


# combinations we will never generate

FORBIDDEN_COMBINATIONS = {
    # Country × any other personal info dimension
    frozenset([('demo', 'country'), ('demo', 'gender')]),
    frozenset([('demo', 'country'), ('demo', 'age')]),
    frozenset([('demo', 'country'), ('demo', 'education')]),
    frozenset([('demo', 'country'), ('demo', 'employment')]),
    frozenset([('demo', 'country'), ('exp', 'ai_exp')]),
    frozenset([('demo', 'country'), ('exp', 'animal_exp')]),
    frozenset([('demo', 'country'), ('exp', 'consciousness_exp')]),
    frozenset([('demo', 'country'), ('exp', 'ethics_exp')]),
    
    # Age × Employment (too sparse)
    frozenset([('demo', 'age'), ('demo', 'employment')]),
}


def is_forbidden(type1, key1, type2, key2):
    """Check if a combination is forbidden."""
    pair = frozenset([(type1, key1), (type2, key2)])
    return pair in FORBIDDEN_COMBINATIONS

# survey × survey pairs to generate

# Format: (topic1, question1_key, topic2, question2_key)
# Or for crossing all questions in a topic: (topic1, '*', topic2, '*')

SURVEY_SURVEY_PAIRS = []

# A. Calculated Variables × Survey Topics
# Cluster × everything
for topic in ['zombie_pill', 'ics', 'graded', 'moral_cons', 'c_intel']:
    for q_key in SURVEY_QUESTIONS[topic]['columns'].keys():
        SURVEY_SURVEY_PAIRS.append(('calculated', 'cluster', topic, q_key))

# group × everything (except ICS itself - that's where it comes from)
for topic in ['zombie_pill', 'graded', 'moral_cons', 'c_intel']:
    for q_key in SURVEY_QUESTIONS[topic]['columns'].keys():
        SURVEY_SURVEY_PAIRS.append(('calculated', 'group', topic, q_key))

# Cluster × group
SURVEY_SURVEY_PAIRS.append(('calculated', 'cluster', 'calculated', 'group'))

# B/C. ICS × Graded Consciousness (4×4 = 16)
for ics_key in SURVEY_QUESTIONS['ics']['columns'].keys():
    for graded_key in SURVEY_QUESTIONS['graded']['columns'].keys():
        SURVEY_SURVEY_PAIRS.append(('ics', ics_key, 'graded', graded_key))

# C. ICS × C&I (4×2 = 8) - includes Zombie × ICS
for ics_key in SURVEY_QUESTIONS['ics']['columns'].keys():
    for ci_key in SURVEY_QUESTIONS['c_intel']['columns'].keys():
        SURVEY_SURVEY_PAIRS.append(('ics', ics_key, 'c_intel', ci_key))

# Zombie × ICS (1×4 = 4)
for ics_key in SURVEY_QUESTIONS['ics']['columns'].keys():
    SURVEY_SURVEY_PAIRS.append(('zombie_pill', 'zombie', 'ics', ics_key))

# D. Graded Consciousness × Moral Considerations (4×4 = 16)
for graded_key in SURVEY_QUESTIONS['graded']['columns'].keys():
    for moral_key in SURVEY_QUESTIONS['moral_cons']['columns'].keys():
        SURVEY_SURVEY_PAIRS.append(('graded', graded_key, 'moral_cons', moral_key))

# G. Kill × (Zombie, Cluster, group)
for kill_key in SURVEY_QUESTIONS['kill']['columns'].keys():
    SURVEY_SURVEY_PAIRS.append(('kill', kill_key, 'zombie_pill', 'zombie'))
    SURVEY_SURVEY_PAIRS.append(('kill', kill_key, 'calculated', 'cluster'))
    SURVEY_SURVEY_PAIRS.append(('kill', kill_key, 'calculated', 'group'))

# Earth in Danger × (Cluster, Group, Zombie)
for eid_key in SURVEY_QUESTIONS['earth_danger']['columns'].keys():
    SURVEY_SURVEY_PAIRS.append(('earth_danger', eid_key, 'calculated', 'cluster'))
    SURVEY_SURVEY_PAIRS.append(('earth_danger', eid_key, 'calculated', 'group'))
    SURVEY_SURVEY_PAIRS.append(('earth_danger', eid_key, 'zombie_pill', 'zombie'))

# H. ICS × Kill to Pass a Test (4 ICS questions × 6 KPT entities = 24)
for ics_key in SURVEY_QUESTIONS['ics']['columns'].keys():
    for kill_key in SURVEY_QUESTIONS['kill']['columns'].keys():
        SURVEY_SURVEY_PAIRS.append(('ics', ics_key, 'kill', kill_key))

print(f"Total survey×survey pairs to generate: {len(SURVEY_SURVEY_PAIRS)}")


# crossings with experience source

EXPERIENCE_SOURCE_SURVEY_PAIRS = []

# Animal Types × Attributions (consciousness and moral status)
for entity_key in SURVEY_QUESTIONS['c_attr']['columns'].keys():
    EXPERIENCE_SOURCE_SURVEY_PAIRS.append(('animal_types', 'c_attr', entity_key))
    
for entity_key in SURVEY_QUESTIONS['ms_attr']['columns'].keys():
    EXPERIENCE_SOURCE_SURVEY_PAIRS.append(('animal_types', 'ms_attr', entity_key))

# Consciousness Experience Source × Various topics
for q_key in SURVEY_QUESTIONS['zombie_pill']['columns'].keys():
    EXPERIENCE_SOURCE_SURVEY_PAIRS.append(('consciousness_source', 'zombie_pill', q_key))

for q_key in SURVEY_QUESTIONS['moral_features']['columns'].keys():
    EXPERIENCE_SOURCE_SURVEY_PAIRS.append(('consciousness_source', 'moral_features', q_key))

for q_key in SURVEY_QUESTIONS['moral_cons']['columns'].keys():
    EXPERIENCE_SOURCE_SURVEY_PAIRS.append(('consciousness_source', 'moral_cons', q_key))

for q_key in SURVEY_QUESTIONS['ics']['columns'].keys():
    EXPERIENCE_SOURCE_SURVEY_PAIRS.append(('consciousness_source', 'ics', q_key))

# Ethics Experience Source × Moral Considerations
for q_key in SURVEY_QUESTIONS['moral_cons']['columns'].keys():
    EXPERIENCE_SOURCE_SURVEY_PAIRS.append(('ethics_source', 'moral_cons', q_key))

print(f"Total experience source × survey pairs to generate: {len(EXPERIENCE_SOURCE_SURVEY_PAIRS)}")


# binary crossings with experience (i.e., experience question × binary survey question)

EXPERIENCE_SIMPLE_SURVEY_PAIRS = []

# Pets × Attributions (consciousness and moral status for animals)
for entity_key in SURVEY_QUESTIONS['c_attr']['columns'].keys():
    EXPERIENCE_SIMPLE_SURVEY_PAIRS.append(('pets', 'c_attr', entity_key))

for entity_key in SURVEY_QUESTIONS['ms_attr']['columns'].keys():
    EXPERIENCE_SIMPLE_SURVEY_PAIRS.append(('pets', 'ms_attr', entity_key))

# Pets × Kill for Test (animal-related ethical decisions)
for q_key in SURVEY_QUESTIONS['kill']['columns'].keys():
    EXPERIENCE_SIMPLE_SURVEY_PAIRS.append(('pets', 'kill', q_key))

# Pets × Earth in Danger (includes questions about animals)
for q_key in SURVEY_QUESTIONS['earth_danger']['columns'].keys():
    EXPERIENCE_SIMPLE_SURVEY_PAIRS.append(('pets', 'earth_danger', q_key))

# Pets × Moral Considerations
for q_key in SURVEY_QUESTIONS['moral_cons']['columns'].keys():
    EXPERIENCE_SIMPLE_SURVEY_PAIRS.append(('pets', 'moral_cons', q_key))

print(f"Total simple experience × survey pairs to generate: {len(EXPERIENCE_SIMPLE_SURVEY_PAIRS)}")


# attributions of C / MS × experience

EXPERIENCE_ATTRIBUTION_PAIRS = []

# Animal Experience × Attributions (consciousness and moral status)
for entity_key in SURVEY_QUESTIONS['c_attr']['columns'].keys():
    EXPERIENCE_ATTRIBUTION_PAIRS.append(('animal_exp', 'c_attr', entity_key))
    EXPERIENCE_ATTRIBUTION_PAIRS.append(('animal_exp', 'ms_attr', entity_key))

# Consciousness Experience × Attributions
for entity_key in SURVEY_QUESTIONS['c_attr']['columns'].keys():
    EXPERIENCE_ATTRIBUTION_PAIRS.append(('consciousness_exp', 'c_attr', entity_key))
    EXPERIENCE_ATTRIBUTION_PAIRS.append(('consciousness_exp', 'ms_attr', entity_key))

# Ethics Experience × Moral Status Attributions
for entity_key in SURVEY_QUESTIONS['ms_attr']['columns'].keys():
    EXPERIENCE_ATTRIBUTION_PAIRS.append(('ethics_exp', 'ms_attr', entity_key))

# AI Experience × All Attributions (we'll let users filter by entity of interest)
for entity_key in SURVEY_QUESTIONS['c_attr']['columns'].keys():
    EXPERIENCE_ATTRIBUTION_PAIRS.append(('ai_exp', 'c_attr', entity_key))
for entity_key in SURVEY_QUESTIONS['ms_attr']['columns'].keys():
    EXPERIENCE_ATTRIBUTION_PAIRS.append(('ai_exp', 'ms_attr', entity_key))

print(f"Total experience × attribution pairs to generate: {len(EXPERIENCE_ATTRIBUTION_PAIRS)}")


# core funcs

def get_crosstab(df, col1, col2, threshold=K_THRESHOLD, suppress=True):
    """
    Generate a cross-tabulation with optional k-anonymity suppression.
    
    Returns a nested dict: {row_value: {col_value: count_or_suppressed}}
    
    SUPPRESSION LOGIC:
    - For BINARY dimensions (exactly 2 values like Male/Female or Yes/No):
      If ANY cell in a row/column is below threshold, ALL cells in that row/column
      are suppressed. This prevents inference attacks where showing partial data + 
      total N would reveal the suppressed value.
    - For NON-BINARY dimensions: All values are shown (no individual cell suppression)
      because suppressed values can be calculated from totals anyway.
    
    Example: For gender_mf crosstabs where rows are Male/Female (binary):
    If a panel (column) has Male=6, Female=1, the entire panel must be suppressed
    because showing Male=6 with N=7 reveals Female=1.
    
    Also includes:
    - '_row_totals': sum of displayed values (for data table row totals)
    - '_col_totals': sum of displayed values (REMOVED when all cells suppressed)
    - '_true_row_totals': actual full row totals (for panel header N display)
    - '_true_col_totals': actual full column totals
    - '_suppressed_rows': list of row keys fully suppressed (when columns are binary)
    - '_suppressed_cols': list of column keys fully suppressed (when rows are binary)
    - '_is_binary_rows': True if exactly 2 rows (e.g., Male/Female)
    - '_is_binary_cols': True if exactly 2 columns
    - '_all_suppressed': True if ALL cells are suppressed (no showable data)
    """
    if col1 not in df.columns or col2 not in df.columns:
        return None
    
    ct = pd.crosstab(df[col1], df[col2])
    result = {}
    true_row_totals = {}  # Full totals for panel headers
    true_col_totals = {}  # Full column totals
    displayed_row_totals = {}  # Sum of non-suppressed cells only (for data table)
    displayed_col_totals = {}  # Sum of non-suppressed cells only (for data table)
    suppressed_rows = []  # Rows suppressed when columns are binary
    suppressed_cols = []  # Columns suppressed when rows are binary
    
    # Check if either dimension is binary (exactly 2 values)
    is_binary_rows = len(ct.index) == 2
    is_binary_cols = len(ct.columns) == 2
    
    # Track if ALL cells end up suppressed
    total_cells = len(ct.index) * len(ct.columns)
    suppressed_cell_count = 0
    
    # Calculate TRUE column totals first (full data)
    for col in ct.columns:
        true_col_totals[str(col)] = int(ct[col].sum())
        displayed_col_totals[str(col)] = 0  # Will accumulate below
    
    # Calculate TRUE row totals (full data)
    for idx in ct.index:
        true_row_totals[str(idx)] = int(ct.loc[idx].sum())
        displayed_row_totals[str(idx)] = 0  # Will accumulate below
    
    # If ROWS are binary (e.g., Male/Female), check each COLUMN for suppression
    # This handles gender_mf crosstabs where panels are survey answers (columns)
    # and bars within panels are Male/Female (rows)
    if is_binary_rows and suppress:
        for col in ct.columns:
            col_has_below_threshold = False
            for idx in ct.index:
                val = int(ct.loc[idx, col])
                if val < threshold:
                    col_has_below_threshold = True
                    break
            if col_has_below_threshold:
                suppressed_cols.append(str(col))
    
    # Build the result with proper suppression
    for idx in ct.index:
        result[str(idx)] = {}
        cells_below_threshold = 0
        
        # First pass: count cells below threshold
        for col in ct.columns:
            val = int(ct.loc[idx, col])
            if suppress and val < threshold:
                cells_below_threshold += 1
        
        # For binary COLUMNS: if ANY cell in this row is below threshold, suppress ALL cells in row
        if is_binary_cols and cells_below_threshold > 0 and suppress:
            for col in ct.columns:
                result[str(idx)][str(col)] = f'<{threshold}'
                suppressed_cell_count += 1
            suppressed_rows.append(str(idx))
            # displayed_row_totals stays 0 for this row (all suppressed)
        else:
            # For non-binary crossings: show actual values (no individual cell suppression)
            # For binary ROWS: suppress entire columns that have any cell below threshold
            for col in ct.columns:
                val = int(ct.loc[idx, col])
                col_str = str(col)
                
                # If rows are binary and this column is in suppressed_cols, suppress this cell
                if is_binary_rows and col_str in suppressed_cols and suppress:
                    result[str(idx)][col_str] = f'<{threshold}'
                    suppressed_cell_count += 1
                else:
                    # Show actual value - only do individual cell suppression for binary dimensions
                    result[str(idx)][col_str] = val
                    # Add to displayed totals
                    displayed_row_totals[str(idx)] += val
                    displayed_col_totals[col_str] += val
    
    # Check if ALL cells are suppressed
    all_suppressed = suppress and (suppressed_cell_count == total_cells)
    
    # Add totals and suppression info as special keys
    # _row_totals/_col_totals = DISPLAYED values only (for data table)
    # _true_row_totals/_true_col_totals = FULL totals (for panel headers)
    result['_row_totals'] = displayed_row_totals
    result['_true_row_totals'] = true_row_totals
    if not all_suppressed:
        result['_col_totals'] = displayed_col_totals
        result['_true_col_totals'] = true_col_totals
    result['_suppressed_rows'] = suppressed_rows
    result['_suppressed_cols'] = suppressed_cols
    result['_is_binary_rows'] = is_binary_rows
    result['_is_binary_cols'] = is_binary_cols
    result['_all_suppressed'] = all_suppressed
    
    return result


def get_allowed_countries(df, country_col, max_countries=MAX_COUNTRIES, k_threshold=K_THRESHOLD):
    """
    Get list of countries allowed for cross-tabulation in this subset.
    
    Returns up to max_countries where each has count >= k_threshold.
    Countries are sorted by count (descending), and we stop at the first
    country below the threshold.
    """
    if country_col not in df.columns:
        return []
    
    counts = df[country_col].value_counts()
    allowed = []
    
    for country, count in counts.items():
        if len(allowed) >= max_countries:
            break
        if count >= k_threshold:
            allowed.append(str(country))
        else:
            # Stop at first country below threshold (they're sorted by count)
            break
    
    return allowed


def get_survey_col(topic, question_key):
    """Get the actual column name for a survey question."""
    if topic not in SURVEY_QUESTIONS:
        return None
    if question_key not in SURVEY_QUESTIONS[topic]['columns']:
        return None
    return SURVEY_QUESTIONS[topic]['columns'][question_key]


def make_crosstab_key(type1, key1, type2, key2):
    """Create a standardized key for a crosstab."""
    return f'{type1}_{key1}__{type2}_{key2}'


def generate_multiselect_crosstab(df, multiselect_col, crossing_col, options, threshold=K_THRESHOLD, suppress=True):
    """
    Generate crosstab for a multi-select question crossed with another variable.
    
    For each option in the multi-select question, counts:
    - exclusive: respondents who selected ONLY this option
    - inclusive: respondents who selected this option (possibly with others)
    
    If suppress=False, all values are shown (for survey×survey crossings).
    
    Returns structure: {
        'type': 'multiselect',
        'data': {
            crossing_value: {
                option: {'exclusive': count, 'inclusive': count},
                ...
            },
            ...
        },
        '_totals': {crossing_value: total_n, ...}
    }
    """
    if multiselect_col not in df.columns or crossing_col not in df.columns:
        return None
    
    # Get valid rows (both columns have data)
    valid_df = df[[multiselect_col, crossing_col]].dropna()
    if len(valid_df) < threshold:
        return None
    
    result = {'type': 'multiselect', 'data': {}, '_totals': {}}
    
    # Get unique values for the crossing column
    crossing_values = valid_df[crossing_col].unique()
    
    for cv in crossing_values:
        cv_str = str(cv)
        cv_df = valid_df[valid_df[crossing_col] == cv]
        result['_totals'][cv_str] = int(len(cv_df))  # Convert to Python int
        result['data'][cv_str] = {}
        
        for option in options:
            # Inclusive: respondents who selected this option (alone or with others)
            # Check if the option appears in the multi-select value
            # Use re.escape to properly escape ALL regex special characters
            escaped_option = re.escape(option)
            inclusive_mask = cv_df[multiselect_col].astype(str).str.contains(escaped_option, regex=True, na=False)
            inclusive_count = int(inclusive_mask.sum())  # Convert to Python int
            
            # Exclusive: respondents who selected ONLY this option
            exclusive_mask = cv_df[multiselect_col].astype(str) == option
            exclusive_count = int(exclusive_mask.sum())  # Convert to Python int
            
            # Apply k-anonymity only if suppress=True
            if suppress:
                if inclusive_count >= threshold:
                    result['data'][cv_str][option] = {
                        'exclusive': exclusive_count if exclusive_count >= threshold else f'<{threshold}',
                        'inclusive': inclusive_count
                    }
                # If inclusive < threshold, don't include this option for this crossing value
            else:
                # No suppression - include all values
                if inclusive_count > 0:
                    result['data'][cv_str][option] = {
                        'exclusive': exclusive_count,
                        'inclusive': inclusive_count
                    }
    
    return result if result['data'] else None


def generate_multiselect_demo_crosstab(df, multiselect_col, demo_col, options, threshold=K_THRESHOLD):
    """
    Generate crosstab for a multi-select question crossed with demographics (employment/country).
    
    Creates panels for each option, showing demographic breakdown.
    Each panel shows respondents who selected that option (inclusive).
    
    Returns structure: {
        'type': 'multiselect_panels',
        'data': {
            option: {demo_value: count, ...},
            ...
        },
        '_panel_totals': {option: total_n, ...}
    }
    """
    if multiselect_col not in df.columns or demo_col not in df.columns:
        return None
    
    # Get valid rows (both columns have data)
    valid_df = df[[multiselect_col, demo_col]].dropna()
    if len(valid_df) < threshold:
        return None
    
    result = {'type': 'multiselect_panels', 'data': {}, '_panel_totals': {}}
    
    for option in options:
        # Get respondents who selected this option (inclusive)
        # Use re.escape to properly escape ALL regex special characters
        escaped_option = re.escape(option)
        option_mask = valid_df[multiselect_col].astype(str).str.contains(escaped_option, regex=True, na=False)
        option_df = valid_df[option_mask]
        
        if len(option_df) < threshold:
            continue
        
        result['_panel_totals'][option] = int(len(option_df))  # Convert to Python int
        result['data'][option] = {}
        
        # Count demographics for this option
        demo_counts = option_df[demo_col].value_counts()
        for demo_val, count in demo_counts.items():
            count_int = int(count)  # Convert to Python int
            if count_int >= threshold:
                result['data'][option][str(demo_val)] = count_int
            else:
                result['data'][option][str(demo_val)] = f'<{threshold}'
    
    return result if result['data'] else None


# generator-funcs

def generate_crosstabs_for_filter(df, allowed_countries):
    """
    Generate all crosstabs for a single filter (subset of data).
    """
    crosstabs = {}
    
    # Helper to add crosstab if valid
    def add_crosstab(key, col1, col2, suppress=True):
        ct = get_crosstab(df, col1, col2, suppress=suppress)
        if ct is not None and len(ct) > 0:
            crosstabs[key] = ct
    
    # survey × demographics
    for topic_key, topic_info in SURVEY_QUESTIONS.items():
        for q_key, q_col in topic_info['columns'].items():
            if q_col not in df.columns:
                continue
            
            # Special handling for kill_all_nos (multi-select question)
            is_multiselect = (topic_key == 'kill' and q_key == 'kill_all_nos')
            
            for demo_key, demo_col in DEMOGRAPHIC_COLS.items():
                if demo_col not in df.columns:
                    continue
                
                # Special handling for multi-select × employment/country
                if is_multiselect and demo_key in ('employment', 'country'):
                    if demo_key == 'country' and len(allowed_countries) == 0:
                        continue
                    
                    work_df = df
                    if demo_key == 'country':
                        work_df = df[df[demo_col].isin(allowed_countries)]
                    
                    key = make_crosstab_key('survey', f'{topic_key}_{q_key}', 'demo', demo_key)
                    ms_ct = generate_multiselect_demo_crosstab(work_df, q_col, demo_col, ANS_ALLNOS_OPTIONS)
                    if ms_ct is not None:
                        ms_ct['note'] = 'NOTE: Only respondents who answered "No" to ALL killing questions were asked this question. Respondents could select multiple options.'
                        crosstabs[key] = ms_ct
                    continue
                
                # Special handling for multi-select × other demographics (gender, age, education)
                if is_multiselect and demo_key in ('gender', 'age', 'education'):
                    key = make_crosstab_key('survey', f'{topic_key}_{q_key}', 'demo', demo_key)
                    # Age and Education are fixed categories - no suppression needed
                    # Age, Education, and Gender are fixed categories - no suppression needed
                    should_suppress = demo_key not in ('age', 'education', 'gender')
                    ms_ct = generate_multiselect_crosstab(df, q_col, demo_col, ANS_ALLNOS_OPTIONS, suppress=should_suppress)
                    if ms_ct is not None:
                        ms_ct['note'] = 'NOTE: Only respondents who answered "No" to ALL killing questions were asked this question. Respondents could select multiple options.'
                        crosstabs[key] = ms_ct
                    continue
                
                # Special handling for country/employment - compute allowed values PER PANEL
                # This avoids the binary suppression issue where a value with enough
                # respondents in one panel gets suppressed because another panel is too small
                if demo_key in ('country', 'employment'):
                    # For country, check if we have allowed countries
                    if demo_key == 'country' and len(allowed_countries) == 0:
                        continue
                    
                    # Get valid data for both columns
                    valid_df = df[[q_col, demo_col]].dropna()
                    if len(valid_df) < K_THRESHOLD:
                        continue
                    
                    # Calculate TRUE row totals from ALL data (for panel headers)
                    survey_values = valid_df[q_col].unique()
                    true_row_totals = {}
                    for sv in survey_values:
                        true_row_totals[str(sv)] = int((valid_df[q_col] == sv).sum())
                    
                    # Build per-panel data
                    # Each panel (survey answer value) gets its own top values
                    result = {}
                    displayed_row_totals = {}
                    displayed_col_totals = {}
                    all_values_seen = set()
                    
                    for survey_val in survey_values:
                        sv_str = str(survey_val)
                        panel_df = valid_df[valid_df[q_col] == survey_val]
                        
                        # Get top values FOR THIS PANEL
                        # For both country and employment, use get_allowed_countries logic
                        # (it works for any categorical column - gets top N values >= k_threshold)
                        panel_values = get_allowed_countries(panel_df, demo_col, MAX_COUNTRIES, K_THRESHOLD)
                        
                        result[sv_str] = {}
                        displayed_row_totals[sv_str] = 0
                        
                        for val in panel_values:
                            count = int((panel_df[demo_col] == val).sum())
                            result[sv_str][val] = count
                            displayed_row_totals[sv_str] += count
                            all_values_seen.add(val)
                            
                            # Track column totals
                            if val not in displayed_col_totals:
                                displayed_col_totals[val] = 0
                            displayed_col_totals[val] += count
                    
                    # Calculate true column totals (for all values that appear in any panel)
                    true_col_totals = {}
                    for val in all_values_seen:
                        true_col_totals[val] = int((valid_df[demo_col] == val).sum())
                    
                    # Only add if we have data
                    if result and any(panel_data for panel_data in result.values()):
                        key = make_crosstab_key('survey', f'{topic_key}_{q_key}', 'demo', demo_key)
                        crosstabs[key] = {
                            **result,
                            '_row_totals': displayed_row_totals,
                            '_true_row_totals': true_row_totals,
                            '_col_totals': displayed_col_totals,
                            '_true_col_totals': true_col_totals,
                            '_suppressed_rows': [],
                            '_suppressed_cols': [],
                            '_is_binary_rows': len(survey_values) == 2,
                            '_is_binary_cols': False,  # Country/Employment are not binary
                            '_all_suppressed': False,
                        }
                
                # Special handling for gender - also generate Male/Female-only version
                elif demo_key == 'gender':
                    # Generate full gender crosstab (for demo × demo crossings, etc.)
                    # Gender is a fixed category - no suppression needed
                    key = make_crosstab_key('survey', f'{topic_key}_{q_key}', 'demo', demo_key)
                    add_crosstab(key, q_col, demo_col, suppress=False)
                    
                    # Generate Male/Female-only crosstab for survey × gender
                    # First, get respondents with valid data for BOTH columns
                    valid_df = df[[q_col, demo_col]].dropna()
                    
                    # Calculate actual totals from those with valid data
                    mf_valid = valid_df[valid_df[demo_col].isin(['Male', 'Female'])]
                    other_valid = valid_df[~valid_df[demo_col].isin(['Male', 'Female'])]
                    
                    n_mf = len(mf_valid)
                    n_other = len(other_valid)
                    n_total_valid = len(valid_df)  # Total with valid data for both columns
                    
                    if n_mf >= K_THRESHOLD:
                        mf_key = make_crosstab_key('survey', f'{topic_key}_{q_key}', 'demo', 'gender_mf')
                        # Get crosstab with gender as ROWS (first dimension) so Male/Female become panels
                        # Gender is a fixed category - no suppression needed
                        mf_ct = get_crosstab(mf_valid, demo_col, q_col, suppress=False)  # Swapped order!
                        if mf_ct is not None and len(mf_ct) > 0:
                            # The data is now: {Male: {survey_answer: count}, Female: {survey_answer: count}}
                            crosstabs[mf_key] = {
                                'data': mf_ct,
                                'n_total': n_mf,  # N of Male/Female respondents with valid data
                                'n_all': n_total_valid,  # Total N with valid data (before filtering)
                                'n_removed': n_other,  # Other genders removed
                                'note': f'NOTE: Other genders were removed due to small N (N={n_other} removed)' if n_other > 0 else None
                            }
                else:
                    key = make_crosstab_key('survey', f'{topic_key}_{q_key}', 'demo', demo_key)
                    # Age and Education are fixed categories - no suppression needed
                    # (unlike Country/Employment where showing small N could identify individuals)
                    should_suppress = demo_key not in ('age', 'education')
                    add_crosstab(key, q_col, demo_col, suppress=should_suppress)
    
    # survey × experience
    for topic_key, topic_info in SURVEY_QUESTIONS.items():
        for q_key, q_col in topic_info['columns'].items():
            if q_col not in df.columns:
                continue
            
            # Special handling for kill_all_nos (multi-select question)
            is_multiselect = (topic_key == 'kill' and q_key == 'kill_all_nos')
                
            for exp_key, exp_col in EXPERIENCE_LEVEL_COLS.items():
                if exp_col not in df.columns:
                    continue
                
                if is_multiselect:
                    # Multi-select × experience level - experience levels are fixed categories
                    key = make_crosstab_key('survey', f'{topic_key}_{q_key}', 'exp', exp_key)
                    ms_ct = generate_multiselect_crosstab(df, q_col, exp_col, ANS_ALLNOS_OPTIONS, suppress=False)
                    if ms_ct is not None:
                        ms_ct['note'] = 'NOTE: Only respondents who answered "No" to ALL killing questions were asked this question. Respondents could select multiple options.'
                        crosstabs[key] = ms_ct
                else:
                    # Survey × experience level - experience levels are fixed categories
                    key = make_crosstab_key('survey', f'{topic_key}_{q_key}', 'exp', exp_key)
                    add_crosstab(key, q_col, exp_col, suppress=False)
    
    # survey × survey
    for topic1, q1_key, topic2, q2_key in SURVEY_SURVEY_PAIRS:
        col1 = get_survey_col(topic1, q1_key)
        col2 = get_survey_col(topic2, q2_key)
        
        if col1 is None or col2 is None:
            continue
        if col1 not in df.columns or col2 not in df.columns:
            continue
        
        # Check if either question is the multi-select kill_all_nos
        is_q1_multiselect = (topic1 == 'kill' and q1_key == 'kill_all_nos')
        is_q2_multiselect = (topic2 == 'kill' and q2_key == 'kill_all_nos')
        
        if is_q1_multiselect or is_q2_multiselect:
            # Multi-select handling: the other variable becomes panels, multi-select becomes stacked bar data
            if is_q1_multiselect:
                multiselect_col, crossing_col = col1, col2
                key = make_crosstab_key('survey', f'{topic1}_{q1_key}', 'survey', f'{topic2}_{q2_key}')
            else:
                multiselect_col, crossing_col = col2, col1
                key = make_crosstab_key('survey', f'{topic1}_{q1_key}', 'survey', f'{topic2}_{q2_key}')
            
            # For survey×survey crossings, don't suppress values <5 since there's no privacy concern
            ms_ct = generate_multiselect_crosstab(df, multiselect_col, crossing_col, ANS_ALLNOS_OPTIONS, suppress=False)
            if ms_ct is not None:
                ms_ct['note'] = 'NOTE: Only respondents who answered "No" to ALL killing questions were asked this question. Respondents could select multiple options.'
                ms_ct['multiselect_var'] = 'kill_all_nos'  # Mark which variable is multiselect
                crosstabs[key] = ms_ct
            continue
        
        key = make_crosstab_key('survey', f'{topic1}_{q1_key}', 'survey', f'{topic2}_{q2_key}')
        # For survey×survey crossings, don't suppress values <5 since there's no privacy concern
        # (these are predefined answer options, not demographics)
        ct = get_crosstab(df, col1, col2, suppress=False)
        if ct is not None and len(ct) > 0:
            crosstabs[key] = ct
    
    # experience × experience
    exp_keys = list(EXPERIENCE_LEVEL_COLS.keys())
    for i, exp1_key in enumerate(exp_keys):
        for exp2_key in exp_keys[i+1:]:  # Only upper triangle to avoid duplicates
            exp1_col = EXPERIENCE_LEVEL_COLS[exp1_key]
            exp2_col = EXPERIENCE_LEVEL_COLS[exp2_key]
            
            if exp1_col not in df.columns or exp2_col not in df.columns:
                continue
            
            key = make_crosstab_key('exp', exp1_key, 'exp', exp2_key)
            add_crosstab(key, exp1_col, exp2_col)
    
    # experience × demographics
    safe_demo_for_exp = ['gender', 'age', 'education']  # NOT country, NOT employment
    
    for exp_key, exp_col in EXPERIENCE_LEVEL_COLS.items():
        if exp_col not in df.columns:
            continue
            
        for demo_key in safe_demo_for_exp:
            demo_col = DEMOGRAPHIC_COLS[demo_key]
            if demo_col not in df.columns:
                continue
            
            # Check it's not forbidden
            if is_forbidden('exp', exp_key, 'demo', demo_key):
                continue
            
            key = make_crosstab_key('exp', exp_key, 'demo', demo_key)
            # Age, Education, and Gender are fixed categories - no suppression needed
            should_suppress = demo_key not in ('age', 'education', 'gender')
            add_crosstab(key, exp_col, demo_col, suppress=should_suppress)
            
            # For gender, also generate Male/Female-only version with gender as panels
            if demo_key == 'gender':
                # Get respondents with valid data for BOTH columns
                valid_df = df[[exp_col, demo_col]].dropna()
                
                # Calculate actual totals
                mf_valid = valid_df[valid_df[demo_col].isin(['Male', 'Female'])]
                other_valid = valid_df[~valid_df[demo_col].isin(['Male', 'Female'])]
                
                n_mf = len(mf_valid)
                n_other = len(other_valid)
                n_total_valid = len(valid_df)
                
                if n_mf >= K_THRESHOLD:
                    mf_key = make_crosstab_key('exp', exp_key, 'demo', 'gender_mf')
                    # Get crosstab with gender as ROWS (first dimension) so Male/Female become panels
                    # Gender is a fixed category - no suppression needed
                    mf_ct = get_crosstab(mf_valid, demo_col, exp_col, suppress=False)  # Swapped order!
                    if mf_ct is not None and len(mf_ct) > 0:
                        crosstabs[mf_key] = {
                            'data': mf_ct,
                            'n_total': n_mf,
                            'n_all': n_total_valid,
                            'n_removed': n_other,
                            'note': f'NOTE: Other genders were removed due to small N (N={n_other} removed)' if n_other > 0 else None
                        }
    
    # demographics × demographics
    safe_demo_pairs = [
        ('gender', 'age'),
        ('gender', 'education'),
        ('age', 'education'),
        ('education', 'employment'),
        ('gender', 'employment'),
    ]
    
    for demo1_key, demo2_key in safe_demo_pairs:
        demo1_col = DEMOGRAPHIC_COLS[demo1_key]
        demo2_col = DEMOGRAPHIC_COLS[demo2_key]
        
        if demo1_col not in df.columns or demo2_col not in df.columns:
            continue
        
        key = make_crosstab_key('demo', demo1_key, 'demo', demo2_key)
        # Employment/Country need suppression; age/education/gender don't
        variable_categories = ('employment', 'country')
        should_suppress = demo1_key in variable_categories or demo2_key in variable_categories
        add_crosstab(key, demo1_col, demo2_col, suppress=should_suppress)
    
    # experience × survey
    # ALL experience sources use multiselect crosstabs with the generate_multiselect_crosstab function
    # This is critical because some options contain commas (e.g., "Professional (work involving ethical decisions, law/medicine/social work)")
    # and naive comma-splitting would break them
    for src_key, topic_key, q_key in EXPERIENCE_SOURCE_SURVEY_PAIRS:
        src_col = EXPERIENCE_SOURCE_COLS.get(src_key)
        survey_col = get_survey_col(topic_key, q_key)
        
        if src_col is None or survey_col is None:
            continue
        if src_col not in df.columns or survey_col not in df.columns:
            continue
        
        key = make_crosstab_key('survey', f'{topic_key}_{q_key}', 'expsrc', src_key)
        
        try:
            temp_df = df[[survey_col, src_col]].dropna()
            if len(temp_df) < K_THRESHOLD:
                continue
            
            # Use predefined options from EXPERIENCE_SOURCE_OPTIONS
            # This avoids comma-splitting issues for categories that contain commas
            options = EXPERIENCE_SOURCE_OPTIONS.get(src_key, [])
            
            if not options:
                # Fallback: try to extract options from data (only safe for animal_types which has no internal commas)
                if src_key == 'animal_types':
                    all_options = set()
                    for val in temp_df[src_col].astype(str).unique():
                        for opt in val.split(','):
                            opt = opt.strip()
                            if opt and opt != 'nan':
                                all_options.add(opt)
                    options = list(all_options) if all_options else ANIMAL_TYPES_OPTIONS
                else:
                    print(f"  Warning: No predefined options for {src_key}, skipping")
                    continue
            
            # Generate multiselect crosstab with survey answers as panels
            ms_ct = generate_multiselect_crosstab(df, src_col, survey_col, options, suppress=False)
            if ms_ct and ms_ct.get('data'):
                # Add the experience note for all experience source crossings
                ms_ct['note'] = 'NOTE: Experience source only includes respondents who rated 3+ on this experience'
                crosstabs[key] = ms_ct
        except Exception as e:
            print(f"  Warning: Could not process {src_key} × {topic_key}_{q_key}: {e}")
            continue
    
    # simple experience × binary (we use regular crosstabs)
    # EXCEPT for kill_all_nos which is multiselect
    for simple_key, topic_key, q_key in EXPERIENCE_SIMPLE_SURVEY_PAIRS:
        simple_col = EXPERIENCE_SIMPLE_COLS.get(simple_key)
        survey_col = get_survey_col(topic_key, q_key)
        
        if simple_col is None or survey_col is None:
            continue
        if simple_col not in df.columns or survey_col not in df.columns:
            continue
        
        # Use expsrc prefix so it appears under Experience Sources in the UI
        key = make_crosstab_key('survey', f'{topic_key}_{q_key}', 'expsrc', simple_key)
        
        # Special handling for kill_all_nos (multi-select question)
        is_multiselect = (topic_key == 'kill' and q_key == 'kill_all_nos')
        if is_multiselect:
            # Use multiselect crosstab with pets (Yes/No) as the crossing variable
            ms_ct = generate_multiselect_crosstab(df, survey_col, simple_col, ANS_ALLNOS_OPTIONS)
            if ms_ct and ms_ct.get('data'):
                ms_ct['note'] = 'NOTE: Only respondents who answered "No" to ALL killing questions were asked this question. Respondents could select multiple options.'
                crosstabs[key] = ms_ct
        else:
            add_crosstab(key, survey_col, simple_col)
    
    # experience × attributions
    for exp_key, topic_key, q_key in EXPERIENCE_ATTRIBUTION_PAIRS:
        exp_col = EXPERIENCE_LEVEL_COLS.get(exp_key)
        survey_col = get_survey_col(topic_key, q_key)
        
        if exp_col is None or survey_col is None:
            continue
        if exp_col not in df.columns or survey_col not in df.columns:
            continue
        
        key = make_crosstab_key('survey', f'{topic_key}_{q_key}', 'exp', exp_key)
        # Experience levels are fixed categories - no suppression needed
        add_crosstab(key, survey_col, exp_col, suppress=False)
    
    return crosstabs


def generate_all_cross_data(df):
    """
    Generate cross-tabulation data for all filter combinations.
    
    Returns the complete data structure for cross_questions_data.json
    """
    country_col = DEMOGRAPHIC_COLS['country']
    
    # Get sample and source values
    sample_vals = sorted(df['sample'].dropna().unique().tolist()) if 'sample' in df.columns else []
    source_vals = sorted(df['source'].dropna().unique().tolist()) if 'source' in df.columns else []
    
    # Sort samples in preferred order
    sample_order = {'preregistered': 0, 'pre-registered': 0, 'exploratory': 1, 'follow-up': 2, 'replication': 3}
    sample_vals = sorted(sample_vals, key=lambda x: sample_order.get(x.lower(), 99))
    
    result = {
        'metadata': {
            'k_threshold': K_THRESHOLD,
            'max_countries': MAX_COUNTRIES,
            'generated_at': datetime.now().isoformat(),
            'survey_survey_pairs_count': len(SURVEY_SURVEY_PAIRS),
            'forbidden_combinations': [list(map(list, fc)) for fc in FORBIDDEN_COMBINATIONS],
            'sample_values': sample_vals,
            'source_values': source_vals,
            'survey_topics': {k: v['label'] for k, v in SURVEY_QUESTIONS.items()},
            'demographic_keys': list(DEMOGRAPHIC_COLS.keys()),
            'experience_keys': list(EXPERIENCE_LEVEL_COLS.keys()),
            'experience_source_keys': list(EXPERIENCE_SOURCE_COLS.keys()),
            'experience_simple_keys': list(EXPERIENCE_SIMPLE_COLS.keys()),
        },
        'filters': {}
    }
    
    # Helper to process a filter
    def process_filter(filter_key, sub_df):
        if len(sub_df) < K_THRESHOLD:
            print(f"  Skipping {filter_key}: n={len(sub_df)} < {K_THRESHOLD}")
            return None
        
        allowed_countries = get_allowed_countries(sub_df, country_col)
        crosstabs = generate_crosstabs_for_filter(sub_df, allowed_countries)
        
        print(f"  {filter_key}: n={len(sub_df)}, countries={len(allowed_countries)}, crosstabs={len(crosstabs)}")
        
        return {
            'n': int(len(sub_df)),
            'allowed_countries': allowed_countries,
            'crosstabs': crosstabs
        }
    
    # 1. All respondents
    print("\n=== Generating crosstabs ===")
    filter_data = process_filter('all', df)
    if filter_data:
        result['filters']['all'] = filter_data
    
    # 2. By sample only (lowercase keys for consistency with JavaScript)
    for sample in sample_vals:
        sub_df = df[df['sample'] == sample]
        filter_key = f'sample_{sample.lower()}'
        filter_data = process_filter(filter_key, sub_df)
        if filter_data:
            result['filters'][filter_key] = filter_data
    
    # 3. By source only (lowercase keys)
    for source in source_vals:
        sub_df = df[df['source'] == source]
        filter_key = f'source_{source.lower()}'
        filter_data = process_filter(filter_key, sub_df)
        if filter_data:
            result['filters'][filter_key] = filter_data
    
    # 4. By sample × source (lowercase keys)
    for sample in sample_vals:
        for source in source_vals:
            sub_df = df[(df['sample'] == sample) & (df['source'] == source)]
            filter_key = f'sample_{sample.lower()}__source_{source.lower()}'
            filter_data = process_filter(filter_key, sub_df)
            if filter_data:
                result['filters'][filter_key] = filter_data
    
    # 5. Multi-sample combinations (2 samples, no source filter)
    from itertools import combinations
    if len(sample_vals) >= 2:
        for combo in combinations(sample_vals, 2):
            # Sort to ensure consistent key ordering (must match JavaScript sampleOrder)
            sorted_combo = sorted(combo, key=lambda x: sample_order.get(x.lower(), 99))
            sub_df = df[df['sample'].isin(combo)]
            filter_key = f'samples_{"_".join(s.lower() for s in sorted_combo)}'
            filter_data = process_filter(filter_key, sub_df)
            if filter_data:
                result['filters'][filter_key] = filter_data
    
    # 6. Multi-sample (2) combinations with source filter
    if len(sample_vals) >= 2:
        for combo in combinations(sample_vals, 2):
            sorted_combo = sorted(combo, key=lambda x: sample_order.get(x.lower(), 99))
            for source in source_vals:
                sub_df = df[(df['sample'].isin(combo)) & (df['source'] == source)]
                filter_key = f'samples_{"_".join(s.lower() for s in sorted_combo)}__source_{source.lower()}'
                filter_data = process_filter(filter_key, sub_df)
                if filter_data:
                    result['filters'][filter_key] = filter_data
    
    # 7. All three samples combined (no source filter)
    if len(sample_vals) >= 3:
        for combo in combinations(sample_vals, 3):
            sorted_combo = sorted(combo, key=lambda x: sample_order.get(x.lower(), 99))
            sub_df = df[df['sample'].isin(combo)]
            filter_key = f'samples_{"_".join(s.lower() for s in sorted_combo)}'
            filter_data = process_filter(filter_key, sub_df)
            if filter_data:
                result['filters'][filter_key] = filter_data
    
    # 8. All three samples with source filter
    if len(sample_vals) >= 3:
        for combo in combinations(sample_vals, 3):
            sorted_combo = sorted(combo, key=lambda x: sample_order.get(x.lower(), 99))
            for source in source_vals:
                sub_df = df[(df['sample'].isin(combo)) & (df['source'] == source)]
                filter_key = f'samples_{"_".join(s.lower() for s in sorted_combo)}__source_{source.lower()}'
                filter_data = process_filter(filter_key, sub_df)
                if filter_data:
                    result['filters'][filter_key] = filter_data
    
    return result


def main(df_path, output_dir):
    """
    Main entry point for generating cross questions data.
    """
    print("=" * 60)
    print("CROSS QUESTIONS DATA GENERATOR")
    print("=" * 60)
    print()
    
    # Load data
    df = pd.read_csv(df_path)
    print(f"Loaded {len(df)} respondents")
    
    # Ensure age_group column exists
    if 'age_group' not in df.columns and survey_mapping.Q_AGE in df.columns:
        df['age_group'] = pd.cut(
            df[survey_mapping.Q_AGE], 
            bins=AGE_BINS,
            labels=AGE_LABELS, 
            include_lowest=True
        ).astype(str)
        print("Created age_group column")
    
    # Check available columns
    print(f"\n=== Column availability ===")
    for topic_key, topic_info in SURVEY_QUESTIONS.items():
        available = sum(1 for col in topic_info['columns'].values() if col in df.columns)
        total = len(topic_info['columns'])
        print(f"  {topic_info['label']}: {available}/{total} columns available")
    
    # Generate all cross data
    cross_data = generate_all_cross_data(df)
    
    # Save to file
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'cross_questions_data.json'
    
    with open(output_path, 'w') as f:
        json.dump(cross_data, f)
    
    # Summary
    total_crosstabs = sum(
        len(f['crosstabs']) 
        for f in cross_data['filters'].values()
    )
    
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Filters generated: {len(cross_data['filters'])}")
    print(f"Total crosstabs: {total_crosstabs}")
    print(f"Output file: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main(
        df_path=r"...\sub_df.csv",
        output_dir=r"...\minds-matter\data"
    )
