import joblib
import pandas as pd
from src.features import extract_features

TRAITS = ["O","C","E","A","N"]

FEATURE_COLUMNS = [
    "word_count",
    "sentence_count",
    "lexical_diversity",
    "avg_word_length",
    "avg_sentence_length",

    "first_person",
    "first_person_plural",
    "second_person",
    "third_person",
    "pronoun_density",

    "positive_emotion",
    "negative_emotion",
    "anxiety_words",
    "anger_words",
    "emotion_intensity",

    "certainty_words",
    "hedging_words",
    "insight_words",
    "causal_words",

    "social_words",
    "family_words",
    "politeness_words",

    "exclamation_freq",
    "question_freq",

    "topic_diversity"
]

def load_models():

    models = {}

    for trait in TRAITS:
        models[trait] = joblib.load(f"models/{trait}_model.pkl")

    return models


def predict_personality(text):

    features = extract_features(text)

    X = pd.DataFrame([features])[FEATURE_COLUMNS]

    models = load_models()

    predictions = {}

    for trait in TRAITS:

        model = models[trait]

        prob = model.predict_proba(X)[0][1]

        predictions[trait] = prob

    return predictions


if __name__ == "__main__":

    text = input("Enter text: ")

    result = predict_personality(text)

    print("\nPersonality prediction:\n")

    for trait, score in result.items():

        print(f"{trait}: {score:.2f}")