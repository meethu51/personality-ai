from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

def train_model(X, y, trait_name):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print(classification_report(y_test, preds))

    # create models folder if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # save model
    joblib.dump(model, f"models/{trait_name}_model.pkl")

    return model