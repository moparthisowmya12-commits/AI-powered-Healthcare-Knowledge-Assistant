# 🩺 AI Powered Healthcare Knowledge Assistant

An **AI-powered healthcare knowledge assistant** that accepts healthcare-related questions and predicts the most relevant healthcare condition using **TF-IDF text vectorization and a Linear SVM classifier**.

The application is developed using **Python and Streamlit** and provides educational information such as common symptoms, possible causes, supportive care, and when to seek medical care.

> ⚠️ **Medical Disclaimer:** This project is intended for educational purposes only. It does not provide medical diagnosis or replace professional medical advice.

---

## 🚀 Project Overview

The **AI Powered Healthcare Knowledge Assistant** allows users to enter a healthcare-related question in natural language.

The system:

1. Accepts a user's healthcare question.
2. Converts the question into numerical features using **TF-IDF**.
3. Uses a trained **Linear SVM** model to predict the most relevant healthcare condition.
4. Identifies the most similar question from the healthcare dataset.
5. Displays relevant educational information.
6. Shows the **Top 3 AI predictions** with their model scores.

## The Streamlit application is configured with the title **"AI Powered Healthcare Knowledge Assistant"** and provides a healthcare-question input interface.

## ✨ Features

* 🔍 Natural-language healthcare question input
* 🤖 AI-based healthcare condition prediction
* 🧠 TF-IDF text feature extraction
* 📊 Linear SVM classification
* 🏆 Top 3 prediction results
* 📚 Educational healthcare answers
* 🩹 Common symptoms
* 🔎 Possible causes/context
* 💡 General supportive care information
* 🚨 Information about when to seek medical care
* 📱 Interactive Streamlit interface
* ⚠️ Medical safety disclaimer

## The application displays category, symptoms, possible cause/context, supportive care, and medical-care guidance based on the best matching dataset record.

## 🛠️ Technologies Used

| Technology        | Purpose                           |
| ----------------- | --------------------------------- |
| Python            | Main programming language         |
| Streamlit         | Web application interface         |
| Pandas            | Dataset handling                  |
| Scikit-learn      | Machine learning                  |
| TF-IDF            | Text vectorization                |
| Linear SVM        | Condition classification          |
| Joblib            | Loading the trained model         |
| Cosine Similarity | Finding the most similar question |

The trained model is a scikit-learn Pipeline containing a `TfidfVectorizer` and `LinearSVC`. The model uses unigram/bigram features and a Linear SVM classifier.

---

## 📂 Project Structure

```text
AI-Powered-Healthcare-Knowledge-Assistant/
│
├── app.py
├── healthcare_model.pkl
├── healthcare_dataset.csv
├── tfidf_vectorizer.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

### File Description

**`app.py`**
Main Streamlit application.

**`healthcare_model.pkl`**
Trained machine-learning pipeline containing the TF-IDF vectorizer and Linear SVM classifier.

**`healthcare_dataset.csv`**
Healthcare knowledge dataset containing questions, conditions, symptoms, causes/context, supportive care, and medical-care guidance.

**`tfidf_vectorizer.pkl`**
Saved TF-IDF vectorizer file.

**`scaler.pkl`**
Saved scaler file included with the project.

**`requirements.txt`**
Contains the Python libraries required to run the application.

---

## 📊 Dataset

The healthcare dataset contains **750 records and 11 columns**.

The major fields include:

* `record_id`
* `category`
* `condition`
* `question`
* `answer`
* `common_symptoms`
* `possible_cause_or_context`
* `general_supportive_care`
* `when_to_seek_medical_care`
* `source_type`
* `medical_safety_note`

The dataset is used both for healthcare-condition information and for finding the most similar question after prediction.

---

## 🧠 Machine Learning Approach

### 1. TF-IDF Vectorization

The user's question is converted into numerical text features using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

The trained pipeline uses:

```text
TfidfVectorizer
    ├── ngram_range = (1, 2)
    ├── max_df = 0.98
    ├── strip_accents = unicode
    └── sublinear_tf = True
```

This allows the model to consider both individual words and two-word combinations.

### 2. Linear SVM Classification

The transformed text is passed to a **Linear Support Vector Machine (LinearSVC)** classifier.

```text
User Question
      ↓
TF-IDF Vectorization
      ↓
Linear SVM
      ↓
Predicted Healthcare Condition
```

The application obtains the classifier's decision scores and ranks the predictions to display the top three results.

---

## 🔎 Similarity-Based Information Retrieval

After predicting a condition, the application searches the dataset for records belonging to that condition.

It then:

1. Selects questions associated with the predicted condition.
2. Converts those questions into TF-IDF vectors.
3. Calculates **cosine similarity** between the user's question and dataset questions.
4. Selects the most similar question.
5. Uses its associated information to generate the educational response.

This process is implemented using `cosine_similarity`.

---

## 🔄 System Workflow

```text
             User
               │
               ▼
     Enter Healthcare Question
               │
               ▼
       Input Validation
               │
               ▼
        TF-IDF Vectorizer
               │
               ▼
          Linear SVM
               │
               ▼
    Predicted Healthcare Condition
               │
               ▼
       Find Matching Records
               │
               ▼
      Cosine Similarity Search
               │
               ▼
       Best Matching Question
               │
               ▼
     ┌─────────────────────────┐
     │ Category                │
     │ Common Symptoms         │
     │ Possible Cause/Context  │
     │ Supportive Care         │
     │ When to Seek Care       │
     │ Educational Answer      │
     └─────────────────────────┘
               │
               ▼
          User Interface
```

---

## 💻 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/AI-Powered-Healthcare-Knowledge-Assistant.git
```

### Step 2: Open the Project

```bash
cd AI-Powered-Healthcare-Knowledge-Assistant
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

Create a `requirements.txt` file containing:

```text
streamlit
pandas
scikit-learn
joblib
```

---

## ▶️ Run the Application

Run the following command from the project directory:

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

---

## 🧪 Example Input

You can enter a question such as:

```text
I have fever, cough and body pain. What could be the reason?
```

The application will analyze the question and display the predicted healthcare condition along with relevant educational information.

---

## 📋 Example Output

The application displays:

```text
🩺 Predicted Condition: <condition>

📌 Category
<category>

🩹 Common Symptoms
<symptoms>

🔎 Possible Cause / Context
<cause or context>

💡 General Supportive Care
<supportive care>

🚨 When to Seek Medical Care
<medical-care guidance>

📚 Educational Answer
<educational information>

📊 Top AI Predictions
1. <condition> — Model score: <score>
2. <condition> — Model score: <score>
3. <condition> — Model score: <score>
```

The app also clearly states that model scores are **ranking scores and not medical probabilities**.

---

## 🔐 Medical Safety

This application should **not be used for self-diagnosis or emergency medical decisions**.

The application itself displays:

> Educational purpose only. This application does not provide a medical diagnosis.

Users should consult a qualified healthcare professional for medical advice, particularly when symptoms are severe, persistent, or worsening.

---

## 🔮 Future Enhancements

Possible future improvements include:

* 🌐 Multilingual healthcare question support
* 🎤 Voice-based question input
* 💬 Conversational chatbot interface
* 📈 Model performance dashboard
* 🔍 Improved semantic search using transformer models
* 🧑‍⚕️ Doctor/hospital recommendation system
* 📱 Mobile application
* 🔐 User authentication and secure history
* 🏥 Integration with verified healthcare resources
* 🌍 Support for regional languages such as Telugu and Hindi

---

## ⚠️ Limitations

* The model predicts conditions based on patterns learned from the available training data.
* Prediction scores should not be interpreted as medical probabilities.
* The system does not perform clinical diagnosis.
* Dataset coverage may not include every possible healthcare condition.
* Professional medical advice should always be preferred for real medical concerns.

---

## 👩‍💻 Project Information

**Project:** AI Powered Healthcare Knowledge Assistant

**Domain:** Artificial Intelligence / Machine Learning / Healthcare

**Machine Learning:** TF-IDF + Linear SVM

**Interface:** Streamlit

**Language:** Python

**Purpose:** Educational healthcare knowledge assistance

---

## 📜 License

This project is intended for educational and academic purposes.

---

## ❤️ Acknowledgement

This project demonstrates how **Natural Language Processing (NLP), Machine Learning, and Information Retrieval** can be combined to create an educational healthcare knowledge assistant.

---

### ⚕️ Important Notice

**This application is an educational AI project and is not a substitute for a qualified healthcare professional.**
