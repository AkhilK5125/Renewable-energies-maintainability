from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# ================= LOAD =================
model = joblib.load("best_predictive_maintenance_model.joblib")
scaler = joblib.load("best_scaler.joblib")
threshold = joblib.load("best_threshold.joblib")

NUMERICAL_FEATURES = list(getattr(scaler, "feature_names_in_", [])) or [
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
MODEL_FEATURES = list(getattr(model, "feature_names_in_", []))

SYSTEM_TYPE_OPTIONS = sorted(
    feature.replace("system_type_", "") for feature in MODEL_FEATURES if feature.startswith("system_type_")
)
COMPONENT_OPTIONS = sorted(
    feature.replace("component_", "") for feature in MODEL_FEATURES if feature.startswith("component_")
)

DEFAULT_FORM_VALUES = {
    "operational_hours": "",
    "ambient_temperature_c": "",
    "component_temperature_c": "",
    "vibration_mm_s": "",
    "power_output_kw": "",
    "efficiency_percent": "",
    "fault_count_last_30d": "",
    "maintenance_delay_days": "",
    "weather_severity_index": "",
    "degradation_rate": "",
    "system_type": SYSTEM_TYPE_OPTIONS[0] if SYSTEM_TYPE_OPTIONS else "",
    "component": COMPONENT_OPTIONS[0] if COMPONENT_OPTIONS else "",
}


def build_input_frame(form_values):
    input_row = {feature: 0.0 for feature in MODEL_FEATURES}

    numerical_values = [float(form_values[feature]) for feature in NUMERICAL_FEATURES]
    scaled_values = scaler.transform(
        pd.DataFrame([numerical_values], columns=NUMERICAL_FEATURES)
    )[0]

    for feature_name, value in zip(NUMERICAL_FEATURES, scaled_values):
        input_row[feature_name] = float(value)

    system_feature = f"system_type_{form_values['system_type']}"
    component_feature = f"component_{form_values['component']}"

    if system_feature in input_row:
        input_row[system_feature] = 1.0
    if component_feature in input_row:
        input_row[component_feature] = 1.0

    return pd.DataFrame([[input_row[feature] for feature in MODEL_FEATURES]], columns=MODEL_FEATURES)


def get_form_values(source):
    form_values = DEFAULT_FORM_VALUES.copy()
    for feature in NUMERICAL_FEATURES:
        form_values[feature] = source.get(feature, "")
    form_values["system_type"] = source.get("system_type", form_values["system_type"])
    form_values["component"] = source.get("component", form_values["component"])
    return form_values


def compute_model_risk(probability, decision_threshold):
    if probability >= decision_threshold:
        result = "Failure Likely"
        is_failure = True
    else:
        result = "No Immediate Failure"
        is_failure = False

    if probability >= 0.35:
        risk_level = "high"
    elif probability >= 0.20:
        risk_level = "medium"
    else:
        risk_level = "low"

    reason_text = (
        f"Model-only prediction based on a {probability * 100:.2f}% failure probability "
        f"against a saved threshold of {decision_threshold * 100:.2f}%."
    )

    return risk_level, result, reason_text, is_failure


def render_page(**context):
    page_context = {
        **DEFAULT_FORM_VALUES,
        **context,
        "system_type_options": SYSTEM_TYPE_OPTIONS,
        "component_options": COMPONENT_OPTIONS,
    }
    return render_template("index.html", **page_context)


@app.route("/")
def home():
    return render_page()


@app.route("/predict", methods=["POST"])
def predict():
    form_values = get_form_values(request.form)

    try:
        input_df = build_input_frame(form_values)
        probability = float(model.predict_proba(input_df)[0][1])
        probability_percent = round(probability * 100, 2)

        risk_level, result, reason_text, is_failure = compute_model_risk(
            probability, float(threshold)
        )

        return render_page(
            **form_values,
            prediction_text=f"{result} (Probability: {probability_percent}%)",
            probability_percent=probability_percent,
            threshold_percent=round(float(threshold) * 100, 2),
            risk_level=risk_level,
            reason_text=reason_text,
            is_failure=is_failure,
        )
    except Exception as e:
        return render_page(
            **form_values,
            prediction_text=f"Error: {e}",
            risk_level="high",
            reason_text="Please review the form values and try again.",
            is_failure=False,
        )


if __name__ == "__main__":
    app.run(debug=True)
