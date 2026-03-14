import re
import numpy as np

# --- word dictionaries ---

POSITIVE_WORDS = {"love","happy","great","excited","wonderful","good","amazing"}
NEGATIVE_WORDS = {"sad","hate","angry","bad","terrible","awful"}
ANXIETY_WORDS = {"worried","nervous","anxious","stress","afraid"}
ANGER_WORDS = {"angry","mad","furious","annoyed"}

SOCIAL_WORDS = {"friend","friends","party","talk","meet","people","group"}
FAMILY_WORDS = {"mother","father","mom","dad","sister","brother","family"}

CERTAINTY_WORDS = {"always","never","definitely","certainly"}
HEDGING_WORDS = {"maybe","perhaps","probably","kind","sort"}

INSIGHT_WORDS = {"think","know","realize","understand"}
CAUSAL_WORDS = {"because","therefore","since","thus"}

POLITE_WORDS = {"please","thank","sorry","appreciate"}

FIRST_PERSON = {"i","me","my","mine"}
FIRST_PERSON_PLURAL = {"we","us","our"}
SECOND_PERSON = {"you","your"}
THIRD_PERSON = {"he","she","they","them"}


def tokenize(text):

    text = text.lower()

    words = re.findall(r"\b\w+\b", text)

    # attempt normal sentence split
    sentences = re.split(r'[.!?\n]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # fallback if punctuation is missing
    if len(sentences) <= 1:
        approx_sentences = max(1, len(words) // 20)
        sentences = [" ".join(words[i*20:(i+1)*20]) for i in range(approx_sentences)]

    return words, sentences


def ratio(count, total):

    if total == 0:
        return 0

    return count / total


def count_words(words, word_set):

    return sum(1 for w in words if w in word_set)


def extract_features(text):

    words, sentences = tokenize(text)

    word_count = len(words)

    sentence_count = len(sentences)

    unique_words = len(set(words))

    # --- structural features ---

    lexical_diversity = ratio(unique_words, word_count)

    avg_word_length = np.mean([len(w) for w in words]) if words else 0

    avg_sentence_length = ratio(word_count, sentence_count)

    # --- pronouns ---

    first_person = count_words(words, FIRST_PERSON)

    first_person_plural = count_words(words, FIRST_PERSON_PLURAL)

    second_person = count_words(words, SECOND_PERSON)

    third_person = count_words(words, THIRD_PERSON)

    pronoun_density = ratio(
        first_person + first_person_plural + second_person + third_person,
        word_count
    )

    # --- emotion ---

    positive = count_words(words, POSITIVE_WORDS)

    negative = count_words(words, NEGATIVE_WORDS)

    anxiety = count_words(words, ANXIETY_WORDS)

    anger = count_words(words, ANGER_WORDS)

    emotion_intensity = ratio(
        positive + negative + anxiety + anger,
        word_count
    )

    # --- cognition ---

    certainty = count_words(words, CERTAINTY_WORDS)

    hedging = count_words(words, HEDGING_WORDS)

    insight = count_words(words, INSIGHT_WORDS)

    causal = count_words(words, CAUSAL_WORDS)

    # --- social ---

    social = count_words(words, SOCIAL_WORDS)

    family = count_words(words, FAMILY_WORDS)

    politeness = count_words(words, POLITE_WORDS)

    # --- punctuation ---

    exclamation_freq = text.count("!")

    question_freq = text.count("?")

    # --- topic diversity ---

    topic_diversity = ratio(unique_words, sentence_count)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "lexical_diversity": lexical_diversity,
        "avg_word_length": avg_word_length,
        "avg_sentence_length": avg_sentence_length,

        "first_person": ratio(first_person, word_count),
        "first_person_plural": ratio(first_person_plural, word_count),
        "second_person": ratio(second_person, word_count),
        "third_person": ratio(third_person, word_count),
        "pronoun_density": pronoun_density,

        "positive_emotion": ratio(positive, word_count),
        "negative_emotion": ratio(negative, word_count),
        "anxiety_words": ratio(anxiety, word_count),
        "anger_words": ratio(anger, word_count),
        "emotion_intensity": emotion_intensity,

        "certainty_words": ratio(certainty, word_count),
        "hedging_words": ratio(hedging, word_count),
        "insight_words": ratio(insight, word_count),
        "causal_words": ratio(causal, word_count),

        "social_words": ratio(social, word_count),
        "family_words": ratio(family, word_count),
        "politeness_words": ratio(politeness, word_count),

        "exclamation_freq": exclamation_freq,
        "question_freq": question_freq,

        "topic_diversity": topic_diversity
    }