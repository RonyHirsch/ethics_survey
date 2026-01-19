import re, os
import pandas as pd
import matplotlib.pyplot as plt
import survey_mapping

DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015"), "-")


def normalize_text(s):
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = s.casefold().translate(DASHES)
    s = re.sub(r"\s+", " ", s).strip()
    return s


BASE_THEME_DEFS = {
    "consciousness/sentience": "Mentions of consciousness, awareness, self-awareness, valenced experience.",
    "intelligence/cognition": "Mentions of intelligence, cognition, memory, planning, learning, language, having a brain.",
    "capacity to suffer/feel pain": "Mentions of pain, suffering, pleasure, emotions, nervous system/receptors.",
    "similarity/kinship to humans": "Mentions of human similarity, human-like traits, closeness to humans.",
    "domestication/pets/companionship": "Mentions of pets, domestication, loyalty, relationship/affection with people.",
    "ecological role/importance": "Mentions of ecosystem/keystone roles, biodiversity, environmental impact.",
    "endangerment/rarity": "Mentions of endangerment, extinction risk, rarity, conservation priority.",
    # benefit/harm label is set per topic (people/animals) below
    "taxa-based heuristics": "Mentions that assign status by taxa examples or contrasts (e.g., mammals>insects, dog>mosquito).",
}


BASE_THEME_KEYWORDS = {
    "consciousness/sentience": [
        "conscious", "consciousness", "awareness", "self awareness", "self-awareness",
        "sentient", "sentience", "aware", "valenced", "mindful", "feel", "feelings",
        "experience", "experiences"
    ],
    "intelligence/cognition": [
        "intelligence", "intelligent", "cognitive", "think", "thinking", "smarter", "capacity for thought",
        "plan", "planning", "learn", "learning", "learned", "memor",
        "reason", "reasoning", "problem-solving", "problem solving",
        "understand", "understanding", "language", "brain", "neuron", "neurons"
    ],
    "capacity to suffer/feel pain": [
        "pain", "painful", "suffer", "suffering", "suffers", "suffered",
        "pleasure", "pleasures", "emotion", "emotional", "empath",
        "nervous", "receptors", "nociception", "nociceptors",
        "hurt", "hurts", "hurting", "sensation", "sensations",
        "stress", "distress", "fear", "anxiety", "cry", "cries", "scream"
    ],
    "similarity/kinship to humans": [
        "human", "humans", "people", "person", "persons", "humanlike", "human-like",
        "similar", "similarity", "closer", "like us", "closely related", "kin", "kinship",
        "mammalian", "primate", "primates", "anthropomorphic"
    ],
    "domestication/pets/companionship": [
        "pet", "pets", "domestic", "domesticated", "tame", "tamed", "service",
        "loyal", "loyalty", "faithful", "relationship", "affection", "friends", "friend",
        "companion", "companionship", "bond", "bonding", "attachment", "attached",
        "owner", "owners", "family", "household"
    ],
    "ecological role/importance": [
        "ecosystem", "ecosystems", "ecological", "environment", "environmental",
        "biodiversity", "collapse", "pollinator", "pollinators", "pollination", "pollinate",
        "bees", "bee", "chain", "food web", "food-web", "trophic", "keystone",
        "habitat", "habitats", "predator", "predators", "prey", "species"
    ],
    "endangerment/rarity": [
        "endanger", "endangered", "endangerment", "extinct", "extinction",
        "priority", "rarity", "rare", "scarce", "scarcity", "vulnerable", "threatened",
        "conservation", "conserve", "protected", "protection"
    ],
    # routed to topic label below (difference between people and animals)
    "_benefit_or_harm_BASE": [
        "beneficial", "benefit", "useful", "usefulness", "utility", "use", "good for humanity",
        "what they bring to society", "actions towards others", "add something to the lives of others",
        "danger", "dangerous", "threat", "threats", "threatening",
        "aggressive", "aggression", "abuse",
        "non aggressive", "non-aggressive", "nonaggressive", "not aggressive",
        "harm", "harmless", "harmful", "commit crimes", "do bad things",
        "prey", "pest", "pests", "service dogs",
        "venom", "venomous", "bite", "bites", "sting", "stings",
        "disease", "diseases", "vector", "vectors",
        "docile", "friendly", "helpful", "charitable", "altruistic", "prosocial", "greater good"
    ],
    "taxa-based heuristics": [
        "mammal", "mammals", "insect", "insects",
        "dog", "dogs", "cat", "cats", "chimp", "chimps", "chimpanzee", "orang",
        "dolphin", "dolphins", "elephant", "elephants", "lion", "lions",
        "bee", "bees", "mosquito", "mosquitoes", "mosquitos",
        "fly", "flies", "sponge", "sponges", "amoeba", "amoebas", "amoebae",
        "bat", "bats", "ant", "ants", "beetle", "beetles", "fish",
        "octopus", "octopi", "octopuses", "crow", "crows", "bird", "birds",
        "reptile", "reptiles", "primate", "primates", "rodent", "rodents",
        "spider", "spiders", "shark", "sharks", "whale", "whales",
        "turtle", "turtles", "frog", "frogs", "cow", "cows", "pig", "pigs",
        "sheep", "goat", "goats", "horse", "horses", "crab", "crabs",
        "lobster", "lobsters", "shrimp", "snail", "snails", "worm", "worms",
        "butterfly", "butterflies", "wasp", "wasps", "cuttlefish", "squid",
        "fox", "wolves", "wolf", "bear", "bears"
    ],
}


def _kw_to_pattern(kw):
    kw = kw.strip().lower()

    # allow direct regex injection
    if kw.startswith("regex:"):
        return kw[len("regex:"):]

    # exact-phrase special cases we already saw
    if kw in {"self awareness", "self-awareness"}:
        return r"\bself[ -]?awareness\b"
    if kw in {"non aggressive", "non-aggressive", "nonaggressive", "not aggressive"}:
        return r"\b(?:non-?aggressive|not aggressive)\b"
    if kw in {"problem-solving", "problem solving"}:
        return r"\bproblem[ -]?solv\w*\b"
    if kw in {"human-like", "humanlike"}:
        return r"\bhuman-?like\b"
    if kw in {"food web", "food-web"}:
        return r"\bfood[ -]?web\b"

    # curated stems: treat as prefixes
    stem_prefixes = {
        "memor", "empath", "pollinat", "understand",
        "benefit", "use", "threat", "aggress", "harm",
        "help", "charit", "altru", "prosocial", "abus"
    }
    if kw in stem_prefixes:
        return rf"\b{re.escape(kw)}\w*\b"

    # multi-word phrase handling with light flexibility
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

    # single-word handling with pluralization for common nouns
    plural_ok = {
        "brain", "bee", "ant", "dog", "cat", "lion", "elephant", "dolphin", "mosquito",
        "fly", "sponge", "bat", "beetle", "fish", "crow", "bird", "reptile", "primate",
        "rodent", "spider", "shark", "whale", "turtle", "frog", "cow", "pig", "goat",
        "horse", "crab", "lobster", "snail", "worm", "butterfly", "wasp", "cuttlefish", "squid",
        "fox", "wolf", "bear", "owner", "family", "species", "person", "people",
        "crime", "thing", "action", "society", "life", "other", "threat", "disease", "vector",
        "friend"
    }
    base = re.escape(kw)
    if kw in plural_ok:
        if kw == "fish":
            return r"\bfish(?:es)?\b"
        if kw == "fly":
            return r"\bfly\b|\bflies\b"
        if kw == "person":
            return r"\bperson(?:s)?\b"
        return rf"\b{base}(?:es|s)?\b"

    # default exact whole-word/phrase
    return rf"\b{base}\b"


def compile_theme_patterns(theme_keywords):
    compiled = {}
    for t, kws in theme_keywords.items():
        parts = [_kw_to_pattern(kw) for kw in kws]
        parts = list(dict.fromkeys(parts))
        compiled[t] = re.compile("|".join(parts), flags=re.IGNORECASE)
    return compiled


def parse_lines(raw, column_name=None):
    raw_lines = raw.splitlines()
    lines = []
    colnorm = normalize_text(column_name) if column_name else None
    for ln in raw_lines:
        n = normalize_text(ln)
        if not n:
            continue
        if n in {"nan", "none"}:
            continue
        if colnorm and n == colnorm:
            continue
        if not re.search(r"[a-z]", n):
            continue
        lines.append(n)
    return lines


def build_themes_for_column(column_name):
    """
    Hard-coded, topic specific themes, also creates the modifications for humans (people) vs animals
    """
    if column_name == survey_mapping.PRIOS_Q_PEOPLE_WHAT[:-1]:
        subject = "human"

        # start from base, drop similarity
        theme_defs = {k: v for k, v in BASE_THEME_DEFS.items() if k != "similarity/kinship to humans"}
        theme_keywords = dict(BASE_THEME_KEYWORDS)
        theme_keywords.pop("similarity/kinship to humans", None)

        """
        In people, feel/feelings is about feeling things and not about conscious vs. not 
        So we'll remove 'feel' and 'feelings' from consciousness/sentience 
        (they remain counted under capacity via regex)
        """
        cons_list = list(theme_keywords["consciousness/sentience"])
        theme_keywords["consciousness/sentience"] = [kw for kw in cons_list if kw not in {"feel", "feelings"}]

        # rename 'endangerment/rarity' - 'vulnerability' for humans
        vul_keywords = theme_keywords.pop("endangerment/rarity")
        theme_keywords["vulnerability"] = vul_keywords

        # update definitions accordingly
        theme_defs.pop("endangerment/rarity", None)
        theme_defs["vulnerability"] = "Mentions of vulnerability, threat, or being at risk (e.g., marginalized, endangered, scarce)."

        # move benefit/harm label to human wording
        base_bh = theme_keywords.pop("_benefit_or_harm_BASE")
        bh_label = "benefit or harm to other people (from people)"
        theme_defs[bh_label] = ("Mentions of prosocial/antisocial impact on people (e.g., charitable, helpful, harmful, threatening) when talking about humans.")

        # take keywords from 'domestication/pets/companionship' and 'taxa-based heuristics' and merge into BH.
        domo_kws = theme_keywords.pop("domestication/pets/companionship")
        taxa_kws = theme_keywords.pop("taxa-based heuristics")
        # remove those themes from definitions, so they don't show as separate rows
        theme_defs.pop("domestication/pets/companionship", None)
        theme_defs.pop("taxa-based heuristics", None)

        # remove 'empath' from capacity list and add to benefit/harm FOR HUMANS only
        cap_list = list(theme_keywords["capacity to suffer/feel pain"])
        theme_keywords["capacity to suffer/feel pain"] = [kw for kw in cap_list if kw != "empath"]
        FEEL_WORDS = r"(?:feel|feeling|feelings|experience|experiences|sensation|sensations|suffer|suffering|hurting)"
        PAIN_WORDS = r"(?:pain|suffering|distress|hurt|hurts|fear|anxiety)"
        cap_core_regex = ( rf"REGEX:\b{FEEL_WORDS}\b(?:\W+\w+){{0,4}}?\b{PAIN_WORDS}\b"
                           rf"|\b{PAIN_WORDS}\b(?:\W+\w+){{0,4}}?\b{FEEL_WORDS}\b")

        kept = [kw for kw in cap_list if kw not in {"empath"}]
        theme_keywords["capacity to suffer/feel pain"] = kept + [cap_core_regex]

        # add empathy + moral-aim phrases to benefit/harm FOR HUMANS only
        human_bh_extras = ["empathy", "empath", "doing what is right", "greater good", "least amount of pain",
                           "REGEX:\\bcause(?:s|d|ing)?\\b(?:\\W+\\w+){0,5}?(?:\\b(?:pain|suffering|harm|distress|fear)\\b)"]

        # merge everything into benefit/harm
        theme_keywords[bh_label] = base_bh + human_bh_extras + domo_kws + taxa_kws

        return subject, theme_defs, theme_keywords

    elif column_name == survey_mapping.PRIOS_Q_ANIMALS_WHAT[:-1]:
        subject = "animal"
        # keep all themes, rename benefit/harm to humans (from animals)
        theme_defs = dict(BASE_THEME_DEFS)
        theme_keywords = dict(BASE_THEME_KEYWORDS)
        base_bh = theme_keywords.pop("_benefit_or_harm_BASE")
        bh_label = "benefit or harm to humans (from animals)"
        theme_defs[bh_label] = "Mentions of usefulness/benefit to humans or (non-)danger to humans, when talking about animals."
        theme_keywords[bh_label] = base_bh
        return subject, theme_defs, theme_keywords

    else:
        # header mismatch
        raise ValueError(
            f"Column header not recognized: {column_name!r}. "
            f"Expected exactly {survey_mapping.PRIOS_Q_PEOPLE_WHAT[:-1]!r} or {survey_mapping.PRIOS_Q_ANIMALS_WHAT[:-1]!r}."
        )


def analyze(raw, column_name=None):

    subject, theme_defs, theme_keywords = build_themes_for_column(column_name or "")
    theme_patterns = compile_theme_patterns(theme_keywords)
    lines = parse_lines(raw, column_name=column_name)
    N = len(lines)

    line_hits = []  # list of dicts: {"response": <str>, "themes": [..]}

    counts = {t: 0 for t in theme_defs}
    examples = {t: [] for t in theme_defs}

    for ln in lines:
        matched_themes = []
        for t, pat in theme_patterns.items():
            if pat.search(ln):
                matched_themes.append(t)
                counts[t] += 1
                if len(examples[t]) < 3:
                    examples[t].append(ln)
        line_hits.append({"response": ln, "themes": "; ".join(matched_themes)})

    # Build theme-level dataframe
    rows = []
    for t in theme_defs:
        n_resp = counts[t]
        prop = round(n_resp / N, 3) if N else 0.0
        rows.append({
            "theme": t,
            "definition": theme_defs[t],
            "n_theme_responses": n_resp,
            "N": N,
            "proportion": prop,
            "example_1": examples[t][0] if len(examples[t]) > 0 else "",
            "example_2": examples[t][1] if len(examples[t]) > 1 else "",
            "example_3": examples[t][2] if len(examples[t]) > 2 else "",
            "subject": subject,
        })

    df = pd.DataFrame(rows).sort_values("n_theme_responses", ascending=False)
    df_line_hits = pd.DataFrame(line_hits)

    return df, N, subject, df_line_hits


def save_outputs(df, N, subject, save_path, save_name, df_line_hits=None):
    csv_path = os.path.join(save_path, f"{save_name}_themes.csv")
    df.to_csv(csv_path, index=False)

    if df_line_hits is not None:
        perline_path = os.path.join(save_path, f"{save_name}_perline_matches.csv")
        df_line_hits.to_csv(perline_path, index=False)

    # chart
    plt.figure(figsize=(8, 6))
    plt.bar(df["theme"], df["n_theme_responses"])
    plt.xticks(rotation=60, ha="right")
    plt.ylabel(f"Responses mentioning theme (N={N})")
    title_subject = "animals" if subject == "animal" else "people"
    plt.title(f"Themes justifying higher moral status for {title_subject}")
    plt.tight_layout()
    chart_path = os.path.join(save_path, f"{save_name}_theme_bar_chart.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    return csv_path, chart_path


if __name__ == "__main__":
    #p = r"C:\Users\Rony\Documents\projects\ethics\survey_analysis\data\analysis_data\all\exploratory\moral_consideration_prios"
    p = r"C:\Users\Rony\Documents\projects\ethics\survey_analysis\data\analysis_data\all\replication_with_follow_up\moral_consideration_prios"
    file_path = os.path.join(p, "moral_decisions_prios_forSupp.csv")
    df_data = pd.read_csv(file_path)
    for col in df_data.columns:
        if col not in (survey_mapping.PRIOS_Q_PEOPLE_WHAT[:-1], survey_mapping.PRIOS_Q_ANIMALS_WHAT[:-1]):
            print(f"Skipping unrecognized column: {col!r}")
            continue
        col_series = df_data[col]
        lines = [(str(v) if pd.notna(v) else "") for v in col_series.tolist()]
        df_doc = '"""\n' + "\n".join(lines) + '\n"""'
        df, N, subject, df_line_hits = analyze(df_doc, column_name=col)
        csv_path, chart_path = save_outputs(df, N, subject, save_path=p, save_name=col, df_line_hits=df_line_hits)
