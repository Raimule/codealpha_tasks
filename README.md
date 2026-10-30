# codealpha_tasks
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Create results folder
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = RESULTS_DIR / "iris_knn_model.joblib"
CSV_PATH = RESULTS_DIR / "user_predictions.csv"

# Load the built-in Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Create a KNN model with feature scaling
model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5),
)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

print("\n" + "=" * 45)
print("IRIS FLOWER CLASSIFICATION")
print("=" * 45)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names,
        zero_division=0,
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model
try:
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved successfully: {MODEL_PATH}")
except (OSError, PermissionError, ValueError) as error:
    print(f"\nCould not save the model: {error}")

# Define input measurements
feature_names = [
    "Sepal length",
    "Sepal width",
    "Petal length",
    "Petal width",
]

print("\nEnter flower measurements in centimetres.")

while True:
    measurements = []

    # Collect and validate four measurements
    try:
        for feature in feature_names:
            while True:
                value = input(f"Enter {feature}: ").strip()

                if value.lower() in ("q", "quit", "exit"):
                    print("Program ended.")
                    raise SystemExit

                try:
                    number = float(value)

                    if not np.isfinite(number) or number <= 0:
                        print(
                            "Error: Enter a positive, finite number."
                        )
                        continue

                    if number > 100:
                        print(
                            "Error: Enter a realistic measurement "
                            "(100 cm or less)."
                        )
                        continue

                    measurements.append(number)
                    break

                except ValueError:
                    print(
                        "Invalid input. Enter a number such as 5.1, "
                        "or type q to exit."
                    )

        # Convert measurements into the required input format
        new_flower = np.array(measurements).reshape(1, -1)

        # Predict the species
        prediction = int(model.predict(new_flower)[0])
        predicted_species = iris.target_names[prediction]

        print("\n" + "=" * 35)
        print("PREDICTION RESULT")
        print("=" * 35)
        print(f"Predicted Iris Species: {predicted_species}")

        # Display prediction probabilities
        probabilities = model.predict_proba(new_flower)[0]

        print("\nPrediction probabilities:")
        for species, probability in zip(
            iris.target_names, probabilities
        ):
            print(f"{species}: {probability * 100:.2f}%")

        # Save prediction to CSV
        try:
            file_exists = CSV_PATH.exists()
            needs_header = (
                not file_exists or CSV_PATH.stat().st_size == 0
            )

            with CSV_PATH.open(
                "a",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.writer(file)

                if needs_header:
                    writer.writerow(
                        feature_names + ["Predicted Species"]
                    )

                writer.writerow(
                    measurements + [predicted_species]
                )

            print(f"\nPrediction saved to: {CSV_PATH}")

        except (OSError, PermissionError, csv.Error) as error:
            print(f"\nCould not save prediction: {error}")

    except (KeyboardInterrupt, EOFError):
        print("\nInput cancelled. Program ended.")
        break

    # Ask whether to predict another flower
    try:
        again = input(
            "\nPredict another flower? (yes/no): "
        ).strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\nProgram ended.")
        break

    if again not in ("yes", "y"):
        print("\nThank you for using Iris Flower Classification!")
        break
