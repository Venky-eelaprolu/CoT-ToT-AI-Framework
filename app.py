import os
import pandas as pd
from dotenv import load_dotenv
from google import genai

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    ParameterGrid
)

# =====================================================
# LOAD ENVIRONMENT VARIABLES
# =====================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)

# =====================================================
# TASK 1
# SENTIMENT ANALYSIS USING CoT
# =====================================================

print("=" * 60)
print("TASK 1 : SENTIMENT ANALYSIS USING CHAIN OF THOUGHT")
print("=" * 60)

reviews_df = pd.read_csv("reviews.csv")

reviews_text = "\n\n".join(
    [f"Review {i+1}: {review}"
     for i, review in enumerate(reviews_df["review"])]
)

cot_prompt = f"""
You are a sentiment analysis expert.

Classify each review as:
- Positive
- Neutral
- Negative

Follow these steps:

Step 1:
Identify all positive sentiment phrases.

Step 2:
Identify all negative sentiment phrases.

Step 3:
Check whether mixed sentiment exists.

Step 4:
Determine overall sentiment.

Step 5:
Explain reasoning.

Examples:

Review:
"The product is excellent. Battery lasts long."

Reasoning:
Positive phrases:
- excellent
- lasts long

Negative phrases:
None

Mixed sentiment:
No

Final Label:
Positive

Explanation:
Strong positive opinion dominates.

------------------------------------

Review:
"The phone works fine but delivery was delayed."

Reasoning:
Positive phrases:
- works fine

Negative phrases:
- delivery delayed

Mixed sentiment:
Yes

Final Label:
Neutral

Explanation:
Contains both positive and negative points.

------------------------------------

Review:
"Terrible quality. Stopped working after two days."

Reasoning:
Positive phrases:
None

Negative phrases:
- terrible quality
- stopped working

Mixed sentiment:
No

Final Label:
Negative

Explanation:
Strong dissatisfaction expressed.

------------------------------------

Analyze ALL reviews below.

{reviews_text}

For EACH review provide:

Review Number:
Positive phrases:
Negative phrases:
Mixed sentiment:
Final Label:
Explanation:
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=cot_prompt
)

sentiment_output = response.text

print("\n")
print(sentiment_output)

with open(
    "sentiment_results.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(sentiment_output)

print("\nSentiment Results Saved!")

# =====================================================
# TASK 2
# RANDOM FOREST HYPERPARAMETER TUNING
# =====================================================

print("\n")
print("=" * 60)
print("TASK 2 : TREE OF THOUGHT HYPERPARAMETER ANALYSIS")
print("=" * 60)

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

results = []

for params in ParameterGrid(param_grid):

    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        random_state=42
    )

    cv_score = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    ).mean()

    model.fit(X_train, y_train)

    train_score = model.score(
        X_train,
        y_train
    )

    overfit_gap = train_score - cv_score

    results.append({
        "n_estimators":
            params["n_estimators"],
        "max_depth":
            params["max_depth"],
        "min_samples_split":
            params["min_samples_split"],
        "train_accuracy":
            round(train_score, 4),
        "cv_accuracy":
            round(cv_score, 4),
        "overfit_gap":
            round(overfit_gap, 4)
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    "hyperparameter_results.csv",
    index=False
)

print("\nHyperparameter Results Saved!")

print(results_df)

# =====================================================
# TASK 2 - ToT ANALYSIS
# =====================================================

table_text = results_df.to_string(index=False)

tot_prompt = f"""
You are an expert ML Engineer.

Use Tree-of-Thought reasoning.

Goal:
Select the best Random Forest configuration.

For every configuration:

Node 1:
Analyze validation accuracy.

Node 2:
Analyze training accuracy.

Node 3:
Analyze overfitting using train-validation gap.

Node 4:
Analyze complexity.

Node 5:
Classify into:
- High Bias
- Balanced
- High Variance

Create branches for promising parameter ranges.

Compare branches.

Choose the final best configuration.

Explain why it is superior.

Hyperparameter Results:

{table_text}

Provide Tree-of-Thought reasoning.
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=tot_prompt
)

tot_output = response.text

print("\n")
print("=" * 60)
print("TREE OF THOUGHT ANALYSIS")
print("=" * 60)
print(tot_output)

with open(
    "tot_analysis.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(tot_output)

print("\nToT Analysis Saved!")

print("\n")
print("=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)