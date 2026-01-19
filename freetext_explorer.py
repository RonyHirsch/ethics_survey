"""
Thematic coding of free-text responses to:
- "What characterizes people with higher moral status?"
- "What characterizes animals with higher moral status?"

The thematic coding is based on the following keyword dictionaries, which are mapped to base-themes as detailed below.

Keywords can be:
- Plain words: matched with optional plural suffix
- STEM:prefix: matches any word starting with the prefix (e.g., "STEM:empath" matches "empathy", "empathetic", etc.)
- REGEX:pattern: raw regex injection

Author: RonyHirsch
"""

import re, os
import pandas as pd
import survey_mapping

DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015"), "-")


def normalize_text(s):
    """
    handle case, normalize dashes, and collapse whitespace
    """
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = s.casefold().translate(DASHES)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Theme definitions

PEOPLE_THEME_DEFS = {
    "consciousness/sentience": "Mentions of consciousness, awareness, self-awareness, valenced experience.",
    "intelligence/cognition": "Mentions of intelligence, cognition, memory, planning, learning, language, having a brain.",
    "capacity to suffer/feel pain": "Mentions of pain, suffering, pleasure, emotions.",
    "vulnerability": "Mentions of vulnerability, threat, or being at risk (e.g., marginalized, endangered, scarce).",
    "ecological role/importance": "Mentions of ecosystem/keystone roles, biodiversity, environmental impact.",
    "benefit or harm to other people (from people)": "Mentions of prosocial/antisocial impact on people (e.g., charitable, helpful, harmful, threatening) when talking about humans.",
}

ANIMALS_THEME_DEFS = {
    "consciousness/sentience": "Mentions of consciousness, awareness, self-awareness, valenced experience.",
    "intelligence/cognition": "Mentions of intelligence, cognition, memory, planning, learning, language, having a brain.",
    "capacity to suffer/feel pain": "Mentions of pain, suffering, pleasure, emotions.",
    "similarity/kinship to humans": "Mentions of human similarity, human-like traits, closeness to humans.",
    "domestication/pets/companionship": "Mentions of pets, domestication, loyalty, relationship/affection with people.",
    "ecological role/importance": "Mentions of ecosystem/keystone roles, biodiversity, environmental impact.",
    "endangerment/rarity": "Mentions of endangerment, extinction risk, rarity, conservation priority.",
    "benefit or harm to humans (from animals)": "Mentions of usefulness/benefit to humans or (non-)danger to humans, when talking about animals.",
    "taxa-based heuristics": "Mentions that assign status by taxa examples or contrasts (e.g., mammals>insects, dog>mosquito).",
}


# keywords

KW_CONSCIOUSNESS_BASE = [
    "conscious", "consciousness", "conciousness", "concious", "selfconciousness",
    "counciousness", "counciouness", "consiousness", "consiusness", "consciously",
    "awareness", "awareneness", "aware",
    "experience", "experiences",
    "self awareness", "self-awareness", "self-awareneness", "self awarness",
    "inner-self", "inner self",
    "sentient", "sentience", "valenced", "mind",
]

KW_INTELLIGENCE = [
    "iq", "intelligence", "intellegance", "intelligent", "STEM:intellect", "inteligence", "inteligent", "intelegence",
    "cognative", "cognitive", "cognition", r"REGEX:(?<!I )\bthink(?:ing|s)?\b", "smarter", "capacity for thought",
    "plan", "planning", "learn", "learning", "learned", "STEM:memor",
    "reason", "reasoning", "problem-solving", "problem solving",
    "STEM:understand", "language", "wise", "wisdom"
]

KW_CAPACITY_BASE = [
    "pleasure", "pleasures", "emotion", "emotional", "receptors",
    "nociception", "nociceptors", "sensation", "sensations", "stress", "distress", "fear",
    "anxiety", "cry", "cries", "scream"
]

KW_SIMILARITY = [
    "human", "humans", "people", "person", "persons", "humanlike", "human-like",
    "similar", "similarity", "similaryty", "closer", "like us", "closely related", "kin", "kinship",
    "mammalian", "primate", "primates", "anthropomorphic"
]

KW_DOMESTICATION = [
    "pet", "pets", "domestic", "domesticated", "tame", "tamed", "service",
    "STEM:loyal", "faithful", "relationship", "affection", "friends", "friend",
    "companion", "companionship", "bond", "bonding", "attachment", "attached",
    "owner", "owners", "family", "household",
]

KW_ECOLOGICAL = [
    "ecology", "ecosystem", "ecosystems", "ecological", "environment", "environmental",
    "biodiversity", "collapse", "STEM:pollinat",
    "bees", "bee", "chain", "food web", "food-web", "trophic", "keystone",
    "habitat", "habitats", "predator", "predators", "species",
]

KW_ENDANGERMENT = [
    "STEM:endanger", "STEM:extinct",
    "priority", "rarity", "rare", "scarce", "scarcity", "vulnerable", "threatened",
    "conservation", "conserve", "protected", "protection",
]

KW_VULNERABILITY_PEOPLE_EXTRAS = [
    "disabilities", "disabled", "sick", "ill", "STEM:discriminat"
]

KW_BENEFIT_HARM_BASE = [
    "beneficial", "STEM:benefit", "STEM:use", "utility", "good for humanity", "edible", "edibility",
    "STEM:danger", "STEM:threat", "STEM:kill", "STEM:aggress", "STEM:abus", "STEM:harm",
    "STEM:crime", "STEM:criminal", "crimimal", "service",
    "what they bring to society", "actions towards others", "add something to the lives of others",
    "STEM:charit", "STEM:altru", "STEM:prosocial", "greater good", "helpful",
    "non aggressive", "non-aggressive", "nonaggressive", "not aggressive", "agressive", "do bad things",
    "prey", "pest", "pests", "service dogs", "provide for us", "venom", "venomous", "bite", "bites", "sting", "stings",
    "disease", "diseases", "vector", "vectors", "docile", "friendly",
]

KW_TAXA = [
    "mammal", "mammals", "insect", "insects", "ape", "apes",
    "dog", "dogs", "cat", "cats", "chimp", "chimps", "chimpanzee", "chimpanzees", "orang", "clam", "clams",
    "dolphin", "dolphins", "elephant", "elephants", "lion", "lions",
    "bee", "bees", "monkey", "monkeys", "mosquito", "mosquitoes", "mosquitos",
    "fly", "flies", "sponge", "sponges", "amoeba", "amoebas", "amoebae",
    "bat", "bats", "ant", "ants", "beetle", "beetles", "fish",
    "octopus", "octopi", "octopuses", "crow", "crows", "bird", "birds",
    "reptile", "reptiles", "rat", "rats", "primate", "primates", "rodent", "rodents",
    "spider", "spiders", "shark", "sharks", "whale", "whales",
    "turtle", "turtles", "frog", "frogs", "cow", "cows", "pig", "pigs",
    "sheep", "goat", "goats", "horse", "horses", "crab", "crabs",
    "lobster", "lobsters", "shrimp", "snail", "snails", "worm", "worms",
    "butterfly", "butterflies", "wasp", "wasps", "cuttlefish", "squid",
    "fox", "wolves", "wolf", "bear", "bears",
    "STEM:evolut",
]


"""
People-specific components
"""

KW_CONSCIOUSNESS_PEOPLE = KW_CONSCIOUSNESS_BASE

KW_CAPACITY_PEOPLE_BASE = KW_CAPACITY_BASE

# Regex to capture "feel X pain" or "pain X feel" patterns for capacity theme in people
_FEEL_WORDS = r"(?:feel|feeling|feelings|experience|experiences|sensation|sensations|suffer|suffering|hurting)"
_PAIN_WORDS = r"(?:pain|suffering|distress|hurt|hurts|fear|anxiety)"
KW_CAPACITY_PEOPLE_REGEX = (
    rf"REGEX:\b{_FEEL_WORDS}\b(?:\W+\w+){{0,4}}?\b{_PAIN_WORDS}\b"
    rf"|\b{_PAIN_WORDS}\b(?:\W+\w+){{0,4}}?\b{_FEEL_WORDS}\b"
)

# extra keywords for people's benefit/harm theme (empathy, moral-aim phrases, cause+harm regex)
KW_BENEFIT_HARM_PEOPLE_EXTRAS = [
    "STEM:empath", "STEM:help", "STEM:contribut", "STEM:good", "STEM:respect",
    "considerate", "kind", "kindness", "kindhearted", "thoughtful", "dependable", "doing what is right",
    "impact", "greater good", "STEM:hurt", "mindful", "solve",
    "least amount of pain", "integrity", "intergrity", "authority", "responsibility", "STEM:murder", "hitler",
    r"REGEX:\bcause(?:s|d|ing)?\b(?:\W+\w+){0,5}?(?:\b(?:pain|suffering|harm|distress|fear)\b)"
]


"""
Animal-specific components
"""

# For animals, "feel/feelings" relates to consciousness/sentience
KW_CONSCIOUSNESS_ANIMALS = KW_CONSCIOUSNESS_BASE + ["feel", "feelings"]

# For animals, "empath" stays in capacity theme
KW_CAPACITY_ANIMALS = KW_CAPACITY_BASE + ["STEM:empath", "hurt", "hurts", "hurting", "pain", "painful", "STEM:suffer"]


"""
FINAL THEME DICTIONARIES for the people question and the animal question
"""

PEOPLE_THEME_KEYWORDS = {
    "consciousness/sentience": KW_CONSCIOUSNESS_PEOPLE,
    "intelligence/cognition": KW_INTELLIGENCE,
    "capacity to suffer/feel pain": KW_CAPACITY_PEOPLE_BASE + [
        KW_CAPACITY_PEOPLE_REGEX,
        r"REGEX:\b(?:capacity|ability|able|can)\s+(?:to\s+)?suffer"
    ],
    "vulnerability": KW_ENDANGERMENT + KW_VULNERABILITY_PEOPLE_EXTRAS,
    "ecological role/importance": KW_ECOLOGICAL,
    "benefit or harm to other people (from people)": (
        KW_BENEFIT_HARM_BASE + KW_BENEFIT_HARM_PEOPLE_EXTRAS + KW_DOMESTICATION + KW_TAXA
    ),
}

ANIMALS_THEME_KEYWORDS = {
    "consciousness/sentience": KW_CONSCIOUSNESS_ANIMALS,
    "intelligence/cognition": KW_INTELLIGENCE,
    "capacity to suffer/feel pain": KW_CAPACITY_ANIMALS,
    "similarity/kinship to humans": KW_SIMILARITY,
    "domestication/pets/companionship": KW_DOMESTICATION,
    "ecological role/importance": KW_ECOLOGICAL,
    "endangerment/rarity": KW_ENDANGERMENT,
    "benefit or harm to humans (from animals)": KW_BENEFIT_HARM_BASE,
    "taxa-based heuristics": KW_TAXA,
}


def _kw_to_pattern(kw):
    """
    Convert a keyword spec (plain, STEM:, or REGEX:) to a regex pattern string
    """
    kw = kw.strip().lower()

    # allow direct regex injection
    if kw.startswith("regex:"):
        return kw[len("regex:"):]

    # stem prefix: match any word starting with this
    if kw.startswith("stem:"):
        stem = kw[len("stem:"):]
        return rf"\b{re.escape(stem)}\w*\b"

    # exact-phrase special cases
    if kw in {"self awareness", "self-awareness"}:
        return r"\bself[ -]?awareness\b"
    if kw in {"self-awareneness"}:  # typo variant
        return r"\bself[ -]?awareneness\b"
    if kw in {"inner-self", "inner self"}:
        return r"\binner[ -]?self\b"
    if kw in {"non aggressive", "non-aggressive", "nonaggressive", "not aggressive"}:
        return r"\b(?:non-?aggressive|not aggressive)\b"
    if kw in {"problem-solving", "problem solving"}:
        return r"\bproblem[ -]?solv\w*\b"
    if kw in {"human-like", "humanlike"}:
        return r"\bhuman-?like\b"
    if kw in {"food web", "food-web"}:
        return r"\bfood[ -]?web\b"
    if kw in {"provide for us"}:
        return r"\bprovide(?:s|d|ing)?\s+(?:for\s+)?(?:us|humans|people|humanity)\b"

    if " " in kw:
        # special flexible phrases
        special_phrases = {
            "do bad things": r"\bdo(?:ing)?\W+bad\W+thing(?:s)?\b",
            "commit crimes": r"\bcommit(?:s|ted|ting)?\W+crime(?:s)?\b",
            "good for humanity": r"\bgood\W+for\W+(?:humanity|humans|people|mankind)\b",
            "actions towards others": r"\bactions?\W+toward(?:s)?\W+others?\b",
            "what they bring to society": r"\b(?:what\W+)?they\W+bring(?:s|ing)?\W+to\W+societ(?:y|ies)\b",
            "add something to the lives of others": r"\badd(?:s|ed|ing)?\W+(?:something\W+)?to\W+the\W+(?:life|lives)\W+of\W+others?\b",
        }
        if kw in special_phrases:
            return special_phrases[kw]

        tokens = kw.split()
        first = tokens[0]
        rest = tokens[1:]

        verb_inflect = {"do", "commit", "add", "bring"}
        if first in verb_inflect:
            first_pat = rf"\b{re.escape(first)}(?:s|ed|ing)?"
        else:
            first_pat = rf"\b{re.escape(first)}"

        def tok_pat(tok: str) -> str:
            t = tok
            if t in {"thing", "things"}:
                return r"thing(?:s)?"
            if t in {"crime", "crimes"}:
                return r"crime(?:s)?"
            if t in {"toward", "towards"}:
                return r"toward(?:s)?"
            if t in {"other", "others"}:
                return r"others?"
            if t in {"life", "lives"}:
                return r"(?:life|lives)"
            if t in {"society", "societies"}:
                return r"societ(?:y|ies)"
            return re.escape(t)

        between = r"[ -]+"
        rest_pat = between.join(tok_pat(t) for t in rest) if rest else ""
        if rest_pat:
            return rf"{first_pat}{between}{rest_pat}\b"
        else:
            return rf"{first_pat}\b"

    # single-word handling: allow plural forms by default, with special cases for irregular plurals
    base = re.escape(kw)

    # irregular plurals
    if kw == "fish":
        return r"\bfish(?:es)?\b"
    if kw == "fly":
        return r"\bfly\b|\bflies\b"
    if kw == "person":
        return r"\bperson(?:s)?\b"
    if kw == "monkey":
        return r"\bmonkey(?:s)?\b"

    # default: allow optional plural suffix
    return rf"\b{base}(?:e?s)?\b"


def compile_theme_patterns(theme_keywords):
    """
    keyword lists into a dict of theme:compiled regex
    """
    compiled = {}
    for t, kws in theme_keywords.items():
        parts = [_kw_to_pattern(kw) for kw in kws]
        parts = list(dict.fromkeys(parts))
        compiled[t] = re.compile("|".join(parts), flags=re.IGNORECASE)
    return compiled


def build_themes_for_column(column_name):
    """
    :return: (subject, theme_defs, theme_keywords) for a given column name
    """
    if column_name == survey_mapping.PRIOS_Q_PEOPLE_WHAT[:-1]:
        return "human", PEOPLE_THEME_DEFS, PEOPLE_THEME_KEYWORDS

    elif column_name == survey_mapping.PRIOS_Q_ANIMALS_WHAT[:-1]:
        return "animal", ANIMALS_THEME_DEFS, ANIMALS_THEME_KEYWORDS

    else:
        raise ValueError(
            f"Column header not recognized: {column_name!r}. "
            f"Expected exactly {survey_mapping.PRIOS_Q_PEOPLE_WHAT[:-1]!r} or {survey_mapping.PRIOS_Q_ANIMALS_WHAT[:-1]!r}."
        )


def get_themes_for_text(text, theme_patterns):
    """
    Given a single text response and compiled theme patterns,
    return a semicolon-separated string of matched themes.
    """
    if pd.isna(text) or str(text).strip() == '':
        return ''
    normalized = normalize_text(str(text))
    matched = []
    for theme, pat in theme_patterns.items():
        if pat.search(normalized):
            matched.append(theme)
    return '; '.join(matched)


def process_original_with_themes(input_path, output_path, summary_output_dir=None):
    """
    Process moral_decisions_prios.csv to find themes in both human and animal questions.
    Adds a "[column] - themes" column next to each free-text column,
    preserving response_id and all other columns.
    Prints tagging statistics for each free-text column.

    If summary_output_dir is provided, saves a summary CSV per question with theme counts and percentages.
    """
    df = pd.read_csv(input_path)

    # columns to process: (column_name_in_file, column_name_for_build_themes)
    columns_to_process = [
        (survey_mapping.PRIOS_Q_PEOPLE_WHAT, survey_mapping.PRIOS_Q_PEOPLE_WHAT[:-1]),
        (survey_mapping.PRIOS_Q_ANIMALS_WHAT, survey_mapping.PRIOS_Q_ANIMALS_WHAT[:-1])
    ]

    print("\n" + "=" * 60)
    print("THEMATIC CODING STATISTICS")
    print("=" * 60)

    for col_in_file, col_for_themes in columns_to_process:
        if col_in_file not in df.columns:
            print(f"\nColumn not found in file: {col_in_file!r}, skipping.")
            continue

        # build themes for this column
        subject, theme_defs, theme_keywords = build_themes_for_column(col_for_themes)
        theme_patterns = compile_theme_patterns(theme_keywords)

        # create themes column
        themes_col_name = f"{col_in_file[:-1]} - themes"
        df[themes_col_name] = df[col_in_file].apply(
            lambda x: get_themes_for_text(x, theme_patterns)
        )

        # reorder: place themes column right after the source column
        cols = list(df.columns)
        source_idx = cols.index(col_in_file)
        cols.remove(themes_col_name)
        cols.insert(source_idx + 1, themes_col_name)
        df = df[cols]

        # compute statistics
        non_empty_mask = df[col_in_file].notna() & (df[col_in_file].str.strip() != '')
        total_responses = non_empty_mask.sum()
        tagged_mask = non_empty_mask & (df[themes_col_name] != '')
        tagged_count = tagged_mask.sum()
        tagged_pct = (tagged_count / total_responses * 100) if total_responses > 0 else 0

        # print statistics
        print(f"\n{col_for_themes}:")
        print(f"  Total free-text responses: {total_responses}")
        print(f"  Tagged with at least one theme: {tagged_count} ({tagged_pct:.1f}%)")
        print(f"  Untagged: {total_responses - tagged_count} ({100 - tagged_pct:.1f}%)")

        # generate summary CSV if output directory provided
        if summary_output_dir is not None:
            theme_counts = []
            for theme in theme_keywords.keys():
                # count responses containing this theme
                count = df.loc[non_empty_mask, themes_col_name].str.contains(re.escape(theme), na=False).sum()
                pct = (count / total_responses * 100) if total_responses > 0 else 0
                theme_counts.append({
                    'theme': theme,
                    'count': count,
                    'percentage': round(pct, 1)
                })

            summary_df = pd.DataFrame(theme_counts)
            summary_filename = f"theme_summary_{subject}.csv"
            summary_path = os.path.join(summary_output_dir, summary_filename)
            summary_df.to_csv(summary_path, index=False)
            print(f"  Summary saved: {summary_path}")

    print("\n" + "=" * 60)

    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    return df


if __name__ == "__main__":
    p = r"...\moral_consideration_prios"

    input_file = os.path.join(p, "moral_decisions_prios.csv")
    output_file = os.path.join(p, "moral_decisions_prios_with_themes.csv")
    process_original_with_themes(input_file, output_file, summary_output_dir=p)