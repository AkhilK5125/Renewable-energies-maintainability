import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_processed_features(features, numerical_features, scaler, column_order):
    features_num = scaler.transform(features[numerical_features])
    features_num_df = pd.DataFrame(
        features_num, columns=numerical_features, index=features.index
    )
    features_processed = pd.concat(
        [features_num_df, features.drop(columns=numerical_features)], axis=1
    )
    return features_processed[column_order]


def save_confusion_matrix_plot(cm, title, output_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Failure", "Failure"],
        yticklabels=["No Failure", "Failure"],
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    # ================= LOAD DATA =================
    data = pd.read_csv("renewable_energies_maintenance_dataset.csv")
    print("Dataset loaded:", data.shape)

    # Standardize column names
    data.columns = [col.strip().lower().replace(" ", "_") for col in data.columns]
    print("\nColumns BEFORE fix:", data.columns.tolist())

    # ================= FIX COLUMN NAMES =================
    data = data.rename(
        columns={
            "ambient_temperature": "ambient_temperature_c",
            "component_temperature": "component_temperature_c",
            "vibration": "vibration_mm_s",
            "power_output": "power_output_kw",
            "efficiency": "efficiency_percent",
            "fault_count_30d": "fault_count_last_30d",
        }
    )
    print("\nColumns AFTER fix:", data.columns.tolist())

    # ================= FEATURE ENGINEERING =================
    categorical_features = ["system_type", "component"]
    target = "failure"

    data_encoded = pd.get_dummies(data, columns=categorical_features)

    expected_cols = [
        "system_type_Solar",
        "system_type_Wind",
        "component_Blade",
        "component_Gearbox",
        "component_Generator",
        "component_Inverter",
        "component_Panel",
    ]

    for col in expected_cols:
        if col not in data_encoded.columns:
            data_encoded[col] = 0

    # Define X and y
    X = data_encoded.drop(columns=["failure", "remaining_useful_life_days"], errors="ignore")
    y = data_encoded[target]

    numerical_features = [
        "operational_hours",
        "ambient_temperature_c",
        "component_temperature_c",
        "vibration_mm_s",
        "power_output_kw",
        "efficiency_percent",
        "fault_count_last_30d",
        "maintenance_delay_days",
        "weather_severity_index",
        "degradation_rate",
    ]

    missing_numerical_features = [col for col in numerical_features if col not in X.columns]
    if missing_numerical_features:
        raise ValueError(
            "Missing required numerical columns: "
            + ", ".join(missing_numerical_features)
        )

    # ================= TRAIN TEST SPLIT =================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTrain shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # ================= SCALING =================
    scaler = StandardScaler()

    scaler.fit(X_train[numerical_features])

    # Keep column order stable for training, evaluation, and downstream inference.
    X_train_processed = build_processed_features(
        X_train, numerical_features, scaler, X.columns
    )
    X_test_processed = build_processed_features(
        X_test, numerical_features, scaler, X.columns
    )
    X_processed = build_processed_features(X, numerical_features, scaler, X.columns)

    print("\nData processed")

    # ================= MODELS =================
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced", n_jobs=1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300, random_state=42
        ),
        "SVC": SVC(probability=True, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }

    best_model = None
    best_score = 0.0
    best_threshold = 0.5
    best_model_name = ""
    model_results = []

    # ================= TRAINING + THRESHOLD =================
    print("\n" + "=" * 60)
    print("TRAINING MODELS WITH THRESHOLD TUNING")
    print("=" * 60)

    for name, model in models.items():
        model.fit(X_train_processed, y_train)

        y_probs = model.predict_proba(X_test_processed)[:, 1]
        thresholds = np.arange(0.2, 0.8, 0.05)
        model_best_score = 0.0
        model_best_threshold = 0.5

        for thresh in thresholds:
            y_pred = (y_probs >= thresh).astype(int)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            if f1 > model_best_score:
                model_best_score = f1
                model_best_threshold = float(thresh)

        model_results.append(
            {
                "Model": name,
                "Best F1 Score": model_best_score,
                "Best Threshold": model_best_threshold,
            }
        )

        print(
            f"{name:<20} Best F1: {model_best_score:.4f} at threshold {model_best_threshold:.2f}"
        )

        if model_best_score > best_score:
            best_score = model_best_score
            best_model = model
            best_threshold = model_best_threshold
            best_model_name = name

    if best_model is None:
        raise RuntimeError("No model produced a valid score during threshold tuning.")

    comparison_df = (
        pd.DataFrame(model_results)
        .sort_values(by="Best F1 Score", ascending=False)
        .reset_index(drop=True)
    )

    # ================= BEST MODEL =================
    print("\n" + "=" * 60)
    print(f"BEST MODEL: {best_model_name}")
    print(f"BEST THRESHOLD: {best_threshold}")
    print(f"BEST F1 SCORE: {best_score:.4f}")
    print("=" * 60)
    print("\nModel comparison summary:\n")
    print(comparison_df.to_string(index=False))

    # ================= FINAL EVALUATION =================
    y_probs_best = best_model.predict_proba(X_test_processed)[:, 1]
    y_pred_best = (y_probs_best >= best_threshold).astype(int)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred_best))

    # ================= SAVE =================
    joblib.dump(best_model, "best_predictive_maintenance_model.joblib")
    joblib.dump(scaler, "best_scaler.joblib")
    joblib.dump(best_threshold, "best_threshold.joblib")

    print("\nModel, scaler, threshold saved")

    # ================= CONFUSION MATRICES =================
    cm_test = confusion_matrix(y_test, y_pred_best)
    save_confusion_matrix_plot(
        cm_test,
        f"{best_model_name} Confusion Matrix (Test Set)",
        "confusion_matrix.png",
    )
    save_confusion_matrix_plot(
        cm_test,
        f"{best_model_name} Confusion Matrix (Test Set)",
        "confusion_matrix_test.png",
    )

    y_probs_all = best_model.predict_proba(X_processed)[:, 1]
    y_pred_all = (y_probs_all >= best_threshold).astype(int)
    cm_all = confusion_matrix(y, y_pred_all)
    save_confusion_matrix_plot(
        cm_all,
        f"{best_model_name} Confusion Matrix (Full Dataset)",
        "confusion_matrix_full_dataset.png",
    )

    print(
        "Confusion matrices saved as confusion_matrix.png, "
        "confusion_matrix_test.png, and confusion_matrix_full_dataset.png"
    )

    # ================= MODEL COMPARISON GRAPH =================
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=comparison_df,
        x="Model",
        y="Best F1 Score",
        hue="Model",
        dodge=False,
        legend=False,
        palette="viridis",
    )
    ax.set_title("Model Comparison by Best F1 Score")
    ax.set_xlabel("Model")
    ax.set_ylabel("Best F1 Score")
    ax.set_ylim(0, min(1.05, comparison_df["Best F1 Score"].max() + 0.12))

    for index, row in comparison_df.iterrows():
        ax.text(
            index,
            row["Best F1 Score"] + 0.015,
            f"F1={row['Best F1 Score']:.3f}\nThr={row['Best Threshold']:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.close()

    print("Model comparison graph saved as model_comparison.png")


if __name__ == "__main__":
    main()
