
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Healthcare Knowledge Assistant",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# APPLICATION DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = BASE_DIR / "healthcare_model.pkl"

    return joblib.load(model_path)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    dataset_path = BASE_DIR / "healthcare_dataset.csv"

    return pd.read_csv(dataset_path)


# ============================================================
# LOAD FILES
# ============================================================

model = load_model()
data = load_dataset()


# ============================================================
# TITLE
# ============================================================

st.title("🩺 AI Powered Healthcare Knowledge Assistant")

st.write(
    "Ask a healthcare-related question and the AI will "
    "predict the most relevant healthcare condition."
)

st.warning(
    "⚠️ Educational purpose only. This application does not "
    "provide a medical diagnosis. Please consult a qualified "
    "healthcare professional for medical advice."
)


# ============================================================
# USER QUESTION
# ============================================================

st.subheader("🔍 Ask Your Healthcare Question")

question = st.text_area(
    "Enter your question:",
    placeholder=(
        "Example: I have fever, cough and body pain. "
        "What could be the reason?"
    ),
    height=130
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button("🔍 Analyze Question", type="primary"):

    # --------------------------------------------------------
    # CHECK QUESTION
    # --------------------------------------------------------

    if not question.strip():

        st.error(
            "Please enter a healthcare question."
        )

        st.stop()


    # --------------------------------------------------------
    # PREDICT CONDITION
    # --------------------------------------------------------

    predicted_condition = model.predict(
        [question]
    )[0]


    # --------------------------------------------------------
    # GET TF-IDF AND CLASSIFIER
    # --------------------------------------------------------

    tfidf = model.named_steps["tfidf"]

    classifier = model.named_steps["classifier"]


    # --------------------------------------------------------
    # TRANSFORM USER QUESTION
    # --------------------------------------------------------

    transformed_question = tfidf.transform(
        [question]
    )


    # --------------------------------------------------------
    # MODEL DECISION SCORE
    # --------------------------------------------------------

    decision_scores = classifier.decision_function(
        transformed_question
    )


    # --------------------------------------------------------
    # TOP 3 PREDICTIONS
    # --------------------------------------------------------

    if decision_scores.ndim == 2:

        scores = decision_scores[0]

        classes = classifier.classes_

        ranked_indices = scores.argsort()[::-1]

        top_predictions = [
            (
                classes[i],
                float(scores[i])
            )
            for i in ranked_indices[:3]
        ]

    else:

        top_predictions = [
            (
                predicted_condition,
                float(decision_scores[0])
            )
        ]


    # --------------------------------------------------------
    # FIND PREDICTED CONDITION IN DATASET
    # --------------------------------------------------------

    condition_rows = data[
        data["condition"] == predicted_condition
    ].copy()


    best_row = None


    # --------------------------------------------------------
    # FIND MOST SIMILAR QUESTION
    # --------------------------------------------------------

    if not condition_rows.empty:

        condition_questions = (
            condition_rows["question"]
            .fillna("")
            .astype(str)
        )

        question_matrix = tfidf.transform(
            condition_questions
        )

        similarity = cosine_similarity(
            transformed_question,
            question_matrix
        )[0]

        best_index = similarity.argmax()

        best_row = condition_rows.iloc[
            best_index
        ]


    # ========================================================
    # DISPLAY PREDICTION
    # ========================================================

    st.success(
        f"🩺 Predicted Condition: {predicted_condition}"
    )


    # ========================================================
    # INFORMATION COLUMNS
    # ========================================================

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # COLUMN 1
    # --------------------------------------------------------

    with col1:

        st.subheader("📌 Category")

        if best_row is not None:
            st.write(
                best_row["category"]
            )
        else:
            st.write(
                "Information not available."
            )


        st.subheader("🩹 Common Symptoms")

        if best_row is not None:
            st.write(
                best_row["common_symptoms"]
            )
        else:
            st.write(
                "Information not available."
            )


        st.subheader("🔎 Possible Cause / Context")

        if best_row is not None:
            st.write(
                best_row[
                    "possible_cause_or_context"
                ]
            )
        else:
            st.write(
                "Information not available."
            )


    # --------------------------------------------------------
    # COLUMN 2
    # --------------------------------------------------------

    with col2:

        st.subheader(
            "💡 General Supportive Care"
        )

        if best_row is not None:
            st.write(
                best_row[
                    "general_supportive_care"
                ]
            )
        else:
            st.write(
                "Information not available."
            )


        st.subheader(
            "🚨 When to Seek Medical Care"
        )

        if best_row is not None:

            st.error(
                best_row[
                    "when_to_seek_medical_care"
                ]
            )

        else:

            st.write(
                "Information not available."
            )


    # ========================================================
    # EDUCATIONAL ANSWER
    # ========================================================

    st.subheader(
        "📚 Educational Answer"
    )

    if best_row is not None:

        st.write(
            best_row["answer"]
        )

    else:

        st.write(
            "No additional information available."
        )


    # ========================================================
    # TOP PREDICTIONS
    # ========================================================

    st.subheader(
        "📊 Top AI Predictions"
    )

    for condition, score in top_predictions:

        st.write(
            f"**{condition}** — "
            f"Model score: `{score:.3f}`"
        )


    # ========================================================
    # MEDICAL DISCLAIMER
    # ========================================================

    st.info(
        "The model score is a ranking score, not a medical "
        "probability. If symptoms are severe, persistent, "
        "or worsening, seek professional medical care."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Powered Healthcare Knowledge Assistant | "
    "TF-IDF + Linear SVM | Educational Use Only"
)
