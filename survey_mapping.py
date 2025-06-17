other_creatures_general_names = {
    "You": "You",
    "A large language model": "LLM",
    "A self-driving car": "Self-driving car",
    "An electron": "Electron",
    "A fungus": "Fungus",
    "A tree": "Tree",
    "A cow": "Cow",
    "A turtle": "Turtle",
    "A dog": "Dog",
    "A cat": "Cat",
    "A lobster": "Lobster",
    "A sea urchin": "Urchin",
    "An octopus": "Octopus",
    "A salmon": "Salmon",
    "A bat": "Bat",
    "A bee": "Bee",
    "A mosquito": "Mosquito",
    "A fruit-fly": "Fruitfly",
    "A rat": "Rat",
    "A pigeon": "Pigeon",
    "An orangutan": "Orangutan",
    "A fetus (human; 24 weeks)": "Fetus",
    "A newborn baby (human)": "Newborn",
    "An ant": "Ant"
}

other_creatures_isNonHumanAnimal = {"You": 0,
                                    "A large language model": 0,
                                    "A self-driving car": 0,
                                    "An electron": 0,
                                    "A fungus": 0,
                                    "A tree": 0,
                                    "A cow": 1,
                                    "A turtle": 1,
                                    "A dog": 1,
                                    "A cat": 1,
                                    "A lobster": 1,
                                    "A sea urchin": 1,
                                    "An octopus": 1,
                                    "A salmon": 1,
                                    "A bat": 1,
                                    "A bee": 1,
                                    "A mosquito": 1,
                                    "A fruit-fly": 1,
                                    "A rat": 1,
                                    "A pigeon": 1,
                                    "An orangutan": 1,
                                    "A fetus (human; 24 weeks)": 0,
                                    "A newborn baby (human)": 0,
                                    "An ant": 1}


# The ONLY diff between this and other_creatures_isNonHumanAnimal is label 1 instead of 0 for: tree, fungus,
other_creatures_isTreeHugger = {"You": 0,
                                "A large language model": 0,
                                "A self-driving car": 0,
                                "An electron": 0,
                                "A fungus": 1,
                                "A tree": 1,
                                "A cow": 1,
                                "A turtle": 1,
                                "A dog": 1,
                                "A cat": 1,
                                "A lobster": 1,
                                "A sea urchin": 1,
                                "An octopus": 1,
                                "A salmon": 1,
                                "A bat": 1,
                                "A bee": 1,
                                "A mosquito": 1,
                                "A fruit-fly": 1,
                                "A rat": 1,
                                "A pigeon": 1,
                                "An orangutan": 1,
                                "A fetus (human; 24 weeks)": 0,
                                "A newborn baby (human)": 0,
                                "An ant": 1}

# other creatures moral status
other_creatures_ms = {
    "other_creatures_ms_1": "ms_You",
    "other_creatures_ms_2": "ms_A large language model",
    "other_creatures_ms_3": "ms_A self-driving car",
    "other_creatures_ms_4": "ms_An electron",
    "other_creatures_ms_5": "ms_A fungus",
    "other_creatures_ms_6": "ms_A tree",
    "other_creatures_ms_7": "ms_A cow",
    "other_creatures_ms_8": "ms_A turtle",
    "other_creatures_ms_9": "ms_A dog",
    "other_creatures_ms_10": "ms_A cat",
    "other_creatures_ms_11": "ms_A lobster",
    "other_creatures_ms_12": "ms_A sea urchin",
    "other_creatures_ms_13": "ms_An octopus",
    "other_creatures_ms_14": "ms_A salmon",
    "other_creatures_ms_15": "ms_A bat",
    "other_creatures_ms_16": "ms_A bee",
    "other_creatures_ms_17": "ms_A mosquito",
    "other_creatures_ms_18": "ms_A fruit-fly",
    "other_creatures_ms_19": "ms_A rat",
    "other_creatures_ms_20": "ms_A pigeon",
    "other_creatures_ms_21": "ms_An orangutan",
    "other_creatures_ms_22": "ms_A fetus (human; 24 weeks)",
    # there is no other_creatures_ms_23
    "other_creatures_ms_24": "ms_A newborn baby (human)",
    "other_creatures_ms_25": "ms_An ant"
}
# answers are numeric in that section (rating; scale 1-4)

# earth is in danger thought experiment

Q_PERSON_DOG = "A random adult person whom you don't know, or a random dog?"
Q_PERSON_PET = "A random adult person whom you don't know, or your pet?"
Q_DICTATOR_DOG = "A dictator whose policy cost the lives of millions of people, or a random dog?"
Q_DICTATOR_PET = "A dictator whose policy cost the lives of millions of people, or your pet?"
Q_UWS_DOG = "A person with a permanent unresponsive wakefulness syndrome, or a random dog?"
Q_UWS_PET = "A person with a permanent  unresponsive wakefulness syndrome, or your pet?"
Q_UWS_FLY = "A person with a permanent unresponsive wakefulness syndrome, or a conscious fruit-fly?"
Q_UWS_AI = "A person with a permanent  unresponsive wakefulness syndrome, or an artificial intelligence (AI) system that can converse, and tells you that it is conscious?"
Q_AI_DOG = "An artificial intelligence (AI) system that can converse and tells you that it is conscious, or a random dog?"

earth_in_danger = {
    "person_dog": Q_PERSON_DOG,
    "person_pet": Q_PERSON_PET,
    "dictator_dog": Q_DICTATOR_DOG,
    "dictator_pet": Q_DICTATOR_PET,
    "uws_dog": Q_UWS_DOG,
    "uws_pet": Q_UWS_PET,
    "uws_Cfly": Q_UWS_FLY,
    "uws_AI": Q_UWS_AI,
    "AI_dog": Q_AI_DOG
}
# answers
ANS_PERSON = "Person"
ANS_DOG = "Dog"
ANS_PET = "My pet"
ANS_DICTATOR = "Dictator (person)"
ANS_UWS = "Person (unresponsive wakefulness syndrome)"
ANS_FLY = "Fruit fly (a conscious one, for sure)"
ANS_AI = "AI (that tells you that it's conscious)"

# answers mapping
EARTH_DANGER_ANS_MAP = {ANS_PERSON: "Person",
                        ANS_DOG: "Dog",
                        ANS_PET: "My pet",
                        ANS_DICTATOR: "Dictator",
                        ANS_UWS: "UWS",
                        ANS_FLY: "Conscious Fruitfly",
                        ANS_AI: "Conscious AI"}

# CATEGORICAL!! not ordinal
EARTH_DANGER_MAP = {ANS_PERSON: 1,
                    ANS_DOG: 2,
                    ANS_PET: 3,
                    ANS_DICTATOR: 4,
                    ANS_UWS: 5,
                    ANS_FLY: 6,
                    ANS_AI: 7}

EARTH_DANGER_MAP_PERSON_DOG = {ANS_PERSON: 1, ANS_DOG: 0}
EARTH_DANGER_MAP_PERSON_PET = {ANS_PERSON: 1, ANS_PET: 0}
EARTH_DANGER_MAP_DICTATOR_DOG = {ANS_DICTATOR: 1, ANS_DOG: 0}
EARTH_DANGER_MAP_DICTATOR_PET = {ANS_DICTATOR: 1, ANS_PET: 0}
EARTH_DANGER_MAP_UWS_DOG = {ANS_UWS: 1, ANS_DOG: 0}
EARTH_DANGER_MAP_UWS_PET = {ANS_UWS: 1, ANS_PET: 0}
EARTH_DANGER_MAP_UWS_FLY = {ANS_UWS: 1, ANS_FLY: 0}
EARTH_DANGER_MAP_UWS_AI = {ANS_UWS: 1, ANS_AI: 0}
EARTH_DANGER_MAP_AI_DOG = {ANS_DOG: 1, ANS_AI: 0}

EARTH_DANGER_QA_MAP = {Q_PERSON_DOG: EARTH_DANGER_MAP_PERSON_DOG,
                       Q_PERSON_PET: EARTH_DANGER_MAP_PERSON_PET,
                       Q_DICTATOR_DOG: EARTH_DANGER_MAP_DICTATOR_DOG,
                       Q_DICTATOR_PET: EARTH_DANGER_MAP_DICTATOR_PET,
                       Q_UWS_DOG: EARTH_DANGER_MAP_UWS_DOG,
                       Q_UWS_PET: EARTH_DANGER_MAP_UWS_PET,
                       Q_UWS_FLY: EARTH_DANGER_MAP_UWS_FLY,
                       Q_UWS_AI: EARTH_DANGER_MAP_UWS_AI,
                       Q_AI_DOG: EARTH_DANGER_MAP_AI_DOG}

# intentions, consciousness, sentience
ICS_Q_INT_WO_CONS = "Do you think a creature/system can have intentions/goals without being conscious?"
ICS_Q_CONS_WO_INT = "Do you think a creature/system can be conscious without having intentions/goals?"
ICS_Q_SENS_WO_CONS = "Do you think a creature/system can have positive or negative sensations (pleasure/pain) without being conscious?"
ICS_Q_CONS_WO_SENS = "Do you think a creature/system can be conscious without having positive or negative sensations (pleasure/pain)?"

ICS_Q_NAME_MAP = {ICS_Q_INT_WO_CONS: "Intentions wo Consciousness",
                  ICS_Q_CONS_WO_INT: "Consciousness wo Intentions",
                  ICS_Q_SENS_WO_CONS: "Sensations wo Consciousness",
                  ICS_Q_CONS_WO_SENS: "Consciousness wo Sensations"}

ics = {
    "ics_iWc": ICS_Q_INT_WO_CONS,
    "ics_iWc_example": "Do you have an example of a case of goals/intentions without consciousness?",
    "ics_cWi": ICS_Q_CONS_WO_INT,
    "ics_cWi_example": "Do you have an example of a case of consciousness without intentions/goals?",
    "ics_sWc": ICS_Q_SENS_WO_CONS,
    "ics_sWc_example": "Do you have an example of a case of positive/negative sensations without consciousness?",
    "ics_cWs": ICS_Q_CONS_WO_SENS,
    "ics_cWs_example": "Do you have an example of a case of consciousness without sensations of pleasure or pain?",
}
# answers
ANS_YES = "Yes"
ANS_NO = "No"
ANS_YESNO_MAP = {ANS_YES: 1, ANS_NO: 0}

# important test (kill a creature/system for success)
important_test_kill = {
    "creature_sensations": "A creature/system that can only feel positive/negative sensations (pleasure/pain), but is not conscious (not experiencing) and does not have plans/goals or intentions (for example, can't plan to avoid something that causes pain)",
    "creature_intentions": "A creature/system that only has plans/goals and intentions (can plan to perform certain actions in the future), but is not conscious (not experiencing) and cannot feel positive/negative sensations (pleasure/pain)",
    "creature_consciousne": "A creature/system that does not have plans/goals or intentions, and cannot feel positive/negative sensations (pleasure/pain), but is conscious (for example, sees colors, but does not feel anything negative or positive, and cannot plan)",
    "creature_vulcan": "A creature/system that is both conscious (has experiences) and has plans/goals and intentions (can plan to perform certain actions in the future), but cannot feel positive/negative sensations (pleasure/pain)",
    "creature_conSense": "A creature/system that is both conscious (has experiences) and can feel positive/negative sensations (pleasure/pain), but does not have plans/goals or intentions (for example, can't plan to avoid something that causes pain)",
    "creature_sensePlan": "A creature/system that can feel positive/negative sensations (pleasure/pain) and also has plans/goals and intentions (can plan to perform certain actions in the future), but is not conscious (not experiencing)",
    "all_nos": "You wouldn't eliminate any of the creatures; why?",
    "all_nos_other": "noKill_Other: please specify"
}

# important test kill creatures
Q_SENSATIONS = "Sensations sans Consciousness & Intentions"
Q_INTENTIONS = "Intentions sans Consciousness & Sensations"
Q_CONSCIOUSNESS = "Consciousness sans Sensations & Intentions"
Q_VULCAN = "Vulcan (Consciousness & Intentions, sans Sensations)"
Q_CONSCIOUSNESS_SENSATIONS = "Consciousness & Sensations, sans Intentions"
Q_SENSATIONS_INTENTIONS = "Sensations & Intentions, sans Consciousness"

Q_ENTITY_MAP = {Q_SENSATIONS: {"Sensations": 1, "Consciousness": 0, "Intentions": 0},
                Q_INTENTIONS: {"Sensations": 0, "Consciousness": 0, "Intentions": 1},
                Q_CONSCIOUSNESS: {"Sensations": 0, "Consciousness": 1, "Intentions": 0},
                Q_VULCAN: {"Sensations": 0, "Consciousness": 1, "Intentions": 1},
                Q_CONSCIOUSNESS_SENSATIONS: {"Sensations": 1, "Consciousness": 1, "Intentions": 0},
                Q_SENSATIONS_INTENTIONS: {"Sensations": 1, "Consciousness": 0, "Intentions": 1}}


important_test_kill_tokens = {
    "A creature/system that can only feel positive/negative sensations (pleasure/pain), but is not conscious (not experiencing) and does not have plans/goals or intentions (for example, can't plan to avoid something that causes pain)": Q_SENSATIONS,
    "A creature/system that only has plans/goals and intentions (can plan to perform certain actions in the future), but is not conscious (not experiencing) and cannot feel positive/negative sensations (pleasure/pain)": Q_INTENTIONS,
    "A creature/system that does not have plans/goals or intentions, and cannot feel positive/negative sensations (pleasure/pain), but is conscious (for example, sees colors, but does not feel anything negative or positive, and cannot plan)": Q_CONSCIOUSNESS,
    "A creature/system that is both conscious (has experiences) and has plans/goals and intentions (can plan to perform certain actions in the future), but cannot feel positive/negative sensations (pleasure/pain)": Q_VULCAN,
    "A creature/system that is both conscious (has experiences) and can feel positive/negative sensations (pleasure/pain), but does not have plans/goals or intentions (for example, can't plan to avoid something that causes pain)": Q_CONSCIOUSNESS_SENSATIONS,
    "A creature/system that can feel positive/negative sensations (pleasure/pain) and also has plans/goals and intentions (can plan to perform certain actions in the future), but is not conscious (not experiencing)": Q_SENSATIONS_INTENTIONS
}

# answers
ANS_KILL = "Yes (will kill to pass the test)"
ANS_NOKILL = "No (will not kill to pass the test)"
ANS_KILLING_MAP = {ANS_KILL: 1, ANS_NOKILL: 0}
ANS_ALLNOS_IMMORAL = "Because I think it would be immoral"
ANS_ALLNOS_INTERESTS = "Because I think all of them have interests of their own"
ANS_ALLNOS_KILL = "Because I wouldn't kill any creature regardless of their interests or capacities"

# pill for no phenomenology; would you take it
zombification_pill = {
    "take_the_pill": "Would you take the pill?"
}
# answers: ANS_YES, ANS_NO


# important features for moral considerations
moral_considerations_features = {
    "m_c": "What do you think is important for moral considerations?",
    "m_c_other": "Other: please specify",
    "m_c_multi_prio": "Which do you think is the most important for moral considerations?",
    "m_c_multi_prio_other": "moralConsiderations_Other: please specify"
}
# answers
ANS_LANG = "Language"
ANS_SENS = "Sensory abilities (detecting things through the senses)"
ANS_SENTIENCE = "Feelings of pleasure and suffering"
ANS_PLAN = "Planning, goals"
ANS_SELF = "Self-awareness"
ANS_PHENOMENOLOGY = "Something it is like to be that creature/system"
ANS_THINK = "Thinking"
ANS_OTHER = "Other"
ALL_FEATURES = [ANS_LANG, ANS_SENS, ANS_SENTIENCE, ANS_PLAN, ANS_SELF, ANS_PHENOMENOLOGY, ANS_THINK, ANS_OTHER]

PRIOS_Q_NONCONS = "Do you think non-conscious creatures/systems should be taken into account in moral decisions?"
PRIOS_Q_CONS = "Do you think conscious creatures/systems should be taken into account in moral decisions?"
PRIOS_Q_PEOPLE = "Do you think some people should have a higher moral status than others?"
PRIOS_Q_ANIMALS = "Do you think some non-human animals should have a higher moral status than others?"

moral_considerations_prios = {
    "mc_Wc": PRIOS_Q_NONCONS,
    "mc_Wc_'no'": PRIOS_Q_CONS,
    "mc_human_prio": PRIOS_Q_PEOPLE,
    "mc_human_prio_why": "What characterizes people with higher moral status?",
    "mc_nonhuman_prio": PRIOS_Q_ANIMALS,
    "mc_nonhuman_prio_why": "What characterizes animals with higher moral status?"
}

PRIOS_Q_NAME_MAP = {PRIOS_Q_NONCONS: "prios_nonCons",
                    PRIOS_Q_CONS: "prios_cons",
                    PRIOS_Q_PEOPLE: "prios_people",
                    PRIOS_Q_ANIMALS: "prios_animals"}

# answers: ANS_YES, ANS_NO, and free text


# is consciousness graded or not
Q_GRADED_EQUAL = "If two creatures/systems are conscious, they are equally conscious"
Q_GRADED_UNEQUAL = "If two creatures/systems are conscious, they are not necessarily equally conscious"
Q_GRADED_MATTERMORE = "Does it mean that the interests of the more conscious entity matter more?"
Q_GRADED_INCOMP = "Assuming two different creatures/systems are conscious, their consciousness is incomparable"
consciousness_graded = {
    "c_binary_1": Q_GRADED_EQUAL,
    "c_graded_1": Q_GRADED_UNEQUAL,
    "c_graded_followup": Q_GRADED_MATTERMORE,
    "c_noncomp_1": Q_GRADED_INCOMP,
}
# answers: numeric, and ANS_YES/NO on the followup


# other creatures CONSCIOUSNESS
other_creatures_cons = {
    "other_creatures_cons_1": "c_You",
    "other_creatures_cons_2": "c_A large language model",
    "other_creatures_cons_3": "c_A self-driving car",
    "other_creatures_cons_4": "c_An electron",
    "other_creatures_cons_5": "c_A fungus",
    "other_creatures_cons_6": "c_A tree",
    "other_creatures_cons_7": "c_A cow",
    "other_creatures_cons_8": "c_A turtle",
    "other_creatures_cons_9": "c_A dog",
    "other_creatures_cons_10": "c_A cat",
    "other_creatures_cons_11": "c_A lobster",
    "other_creatures_cons_12": "c_A sea urchin",
    "other_creatures_cons_13": "c_An octopus",
    "other_creatures_cons_14": "c_A salmon",
    "other_creatures_cons_15": "c_A bat",
    "other_creatures_cons_16": "c_A bee",
    "other_creatures_cons_17": "c_A mosquito",
    "other_creatures_cons_18": "c_A fruit-fly",
    "other_creatures_cons_19": "c_A rat",
    "other_creatures_cons_20": "c_A pigeon",
    "other_creatures_cons_21": "c_An orangutan",
    "other_creatures_cons_22": "c_A fetus (human; 24 weeks)",
    # there is no other_creatures_cons_23
    "other_creatures_cons_24": "c_A newborn baby (human)",
    "other_creatures_cons_25": "c_An ant"
}
# answers are numeric in that section (rating; scale 1-4)


ANS_C_MS = {"Does not have": 1, "Has": 4}

C_RATINGS = {"c_You": {"c_You": 1, "": 4},
             "c_A large language model": {"c_LLM": 1, "": 4},
             "c_A self-driving car": {"c_self driving car": 1, "": 4},
             "c_An electron": {"c_electron": 1, "": 4},
             "c_A fungus": {"c_fungus": 1, "": 4},
             "c_A tree": {"c_tree": 1, "": 4},
             "c_A cow": {"c_cow": 1, "": 4},
             "c_A turtle": {"c_turtle": 1, "": 4},
             "c_A dog": {"c_dog": 1, "": 4},
             "c_A cat": {"c_cat": 1, "": 4},
             "c_A lobster": {"c_lobster": 1, "": 4},
             "c_A sea urchin": {"c_sea urchin": 1, "": 4},
             "c_An octopus": {"c_octopus": 1, "": 4},
             "c_A salmon": {"c_salmon": 1, "": 4},
             "c_A bat": {"c_bat": 1, "": 4},
             "c_A bee": {"c_bee": 1, "": 4},
             "c_A mosquito": {"c_mosquito": 1, "": 4},
             "c_A fruit-fly": {"c_fruitfly": 1, "": 4},
             "c_A rat": {"c_rat": 1, "": 4},
             "c_A pigeon": {"c_pigeon": 1, "": 4},
             "c_An orangutan": {"c_orangutan": 1, "": 4},
             "c_A fetus (human; 24 weeks)": {"c_fetus": 1, "": 4},
             "c_A newborn baby (human)": {"c_newborn": 1, "": 4},
             "c_An ant": {"c_ant": 1, "": 4}}

MS_RATINGS = {"ms_You": {"ms_You": 1, "": 4},
             "ms_A large language model": {"ms_LLM": 1, "": 4},
             "ms_A self-driving car": {"ms_self driving car": 1, "": 4},
             "ms_An electron": {"ms_electron": 1, "": 4},
             "ms_A fungus": {"ms_fungus": 1, "": 4},
             "ms_A tree": {"ms_tree": 1, "": 4},
             "ms_A cow": {"ms_cow": 1, "": 4},
             "ms_A turtle": {"ms_turtle": 1, "": 4},
             "ms_A dog": {"ms_dog": 1, "": 4},
             "ms_A cat": {"ms_cat": 1, "": 4},
             "ms_A lobster": {"ms_lobster": 1, "": 4},
             "ms_A sea urchin": {"ms_sea urchin": 1, "": 4},
             "ms_An octopus": {"ms_octopus": 1, "": 4},
             "ms_A salmon": {"ms_salmon": 1, "": 4},
             "ms_A bat": {"ms_bat": 1, "": 4},
             "ms_A bee": {"ms_bee": 1, "": 4},
             "ms_A mosquito": {"ms_mosquito": 1, "": 4},
             "ms_A fruit-fly": {"ms_fruitfly": 1, "": 4},
             "ms_A rat": {"ms_rat": 1, "": 4},
             "ms_A pigeon": {"ms_pigeon": 1, "": 4},
             "ms_An orangutan": {"ms_orangutan": 1, "": 4},
             "ms_A fetus (human; 24 weeks)": {"ms_fetus": 1, "": 4},
             "ms_A newborn baby (human)": {"ms_newborn": 1, "": 4},
             "ms_An ant": {"ms_ant": 1, "": 4}}

# consciousness and intellect
con_intellect = {
    "con_intellect": "Do you think consciousness and intelligence are related?",
    "con_intellect_yes": "How?",
    "con_intellect_yes_fu": "What is the common denominator?"
}
# answers:
ANS_C_NECESSARY = "Consciousness is necessary for intelligence"
ANS_I_NECESSARY = "Intelligence is necessary for consciousness"
ANS_SAME = "They are the same thing"
ANS_THIRD = "They are related to a common third feature"

# experience with AI
Q_AI_EXP = "On a scale from 1 to 5 where 1 means 'none' and 5 means 'extremely', how would you rate your experience and knowledge in artificial intelligence (AI) systems?"
ai_exp = {
    "ai_exp_1": Q_AI_EXP,
    "ai_exp_fu": "Please specify your experience with AI",
    "ai_exp_other": "aiExp_Other: please specify"
}
# answers:
ANS_AI_ACADEMIA = "Academic background (studied/teach computer science/ML)"
ANS_AI_RESEARCH = "Research (conduct research related to AI systems)"
ANS_AI_PROF = "Professional (develop or implement AI technologies)"
ANS_AI_PRACTICAL = "use AI systems and technologies"
ANS_AI_PERSON = "Personal Interest (follower of AI advancements and development)"

# experience with animals
Q_ANIMAL_EXP = "On a scale from 1 to 5 where 1 means 'none' and 5 means 'extremely', how would you rate your level of interaction or experience with animals?"
animal_exp = {
    "animals_experience_1": Q_ANIMAL_EXP,
    "animals_which": "Please specify which animals",
    "animal_other": "animalsExp_Other: please specify",
    "pets": "Do you have a pet?"
}
# answers
ANS_PRIMATES = "Primates (including apes)"
ANS_FISH = "Fish"
ANS_BIRDS = "Birds"
ANS_RODENTS = "Rodents"
ANS_INSECTS = "Insects"
ANS_BATS = "Bats"
ANS_CEPHAL = "Cephalopods"
ANS_OTHER_MARINE = "Other marine life"
ANS_CATS = "Cats"
ANS_DOGS = "Dogs"
ANS_REPTILES = "Reptiles"
ANS_LIVESTOCK = "Livestock"
# there is also other, and yes/no


# experience with consciousness
Q_CONSC_EXP = "On a scale from 1 to 5 where 1 means 'none' and 5 means 'extremely', how would you rate your experience and knowledge in the science of consciousness?"
consciousness_exp = {
    "consc_experience_1": Q_CONSC_EXP,
    "consc_experience_fu": "Please specify your experience with this topic",
    "consci_exp_other": "consciousExp_Other: please specify"
}
# answers
ANS_C_ACADEMIA = "Academic Background (studied/teach these topics in university)"
ANS_C_RESEARCH = "Research (conduct research related to consciousness)"
ANS_C_PROF = "Professional Experience (work related to consciousness)"
ANS_C_PERSON = "Personal Interest (reading texts about consciousness)"

# experience with ethics
Q_ETHICS_EXP = "On a scale from 1 to 5 where 1 means 'none' and 5 means 'extremely', how would you rate your experience and knowledge in ethics and morality?"
ethics_exp = {
    "ethics_experience_1": Q_ETHICS_EXP,
    "ethics_experience_fu": "Please specify your experience",
    "ethics_exp_other": "ethicsExp_Other: please specify"
}
# answers
ANS_E_ACADEMIA = "Academic Background (studied/researched philosophy/ethics in university)"
ANS_E_PERSON = "Personal Interest (reading philosophical texts/participant in ethical debates)"
ANS_E_PROF = "Professional (work involving ethical decisions, law/medicine/social work)"
ANS_E_VOLUN = "Volunteer Work (in organizations focused on ethical issues/moral causes)"
ANS_E_RELIGION = "Religious/Spitirual Practice (engagement with ethical teachings)"

Q_EXP_DICT = {Q_CONSC_EXP: "exp_consciousness",
              Q_ETHICS_EXP: "exp_ethics",
              Q_ANIMAL_EXP: "exp_animals",
              Q_AI_EXP: "exp_ai"}

# demographics
demographics = {
    "age": "How old are you?",
    "gender": "How do you describe yourself?",
    "country": "In which country do you currently reside?",
    "education": "What is your education background?",
    "field": "In what topic?",
    "field_other": "education_Other: please specify",
    "employment": "Current primary employment domain",
    "employment_other": "employment_Other: please specify",
    "RecordedDate": "date",
    "StartDate": "date_start",
    "EndDate": "date_end",
    "Duration (in seconds)": "duration_sec",
    "ResponseId": "response_id",
    "UserLanguage": "language"
}
Q_AGE = "How old are you?"

# education_options
EDU_NONE = "No formal education"
EDU_PRIM = "Primary education"
EDU_SECD = "Secondary education (hold a high-school diploma or equivalent)"
EDU_POSTSEC = "Post-secondary education (hold a bachelor's or associate degree)"
EDU_GRAD = "Graduate education (hold a master's degree or doctoral degree)"

EDU_MAP = {EDU_NONE: 1,
           EDU_PRIM: 2,
           EDU_SECD: 3,
           EDU_POSTSEC: 4,
           EDU_GRAD: 5}

# columns that contain data we do not care about and do not collect anyway (automatically generated in Qualtrics)
redundant = ["IPAddress", "RecipientLastName", "RecipientFirstName", "RecipientEmail",
             "ExternalReference", "LocationLatitude", "LocationLongitude", "DistributionChannel"]
# columns that contain data we do not want to collect once all paid subjects are compensated (generated by Prolific)
prolific_redundant = ["prolific_id", "PROLIFIC_PID", "catch"]

# list of all question blocks
question_blocks = {"other_creatures_ms": other_creatures_ms,
                   "earth_in_danger": earth_in_danger, "ics": ics,
                   "important_test_kill": important_test_kill,
                   "zombification_pill": zombification_pill,
                   "moral_considerations_features": moral_considerations_features,
                   "moral_considerations_prios": moral_considerations_prios,
                   "consciousness_graded": consciousness_graded,
                   "other_creatures_cons": other_creatures_cons,
                   "con_intellect": con_intellect,
                   "consciousness_exp": consciousness_exp, "ai_exp": ai_exp, "animal_exp": animal_exp,
                   "ethics_exp": ethics_exp, "demographics": demographics}

questions_name_mapping = {k: v for d in list(question_blocks.values()) for k, v in d.items()}
