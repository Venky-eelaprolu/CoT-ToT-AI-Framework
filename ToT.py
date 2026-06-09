from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ParameterGrid
import pandas as pd

from google import genai
from dotenv import load_dotenv
import os

print("=" * 60)
print("TASK 2 : TREE OF THOUGHT HYPERPARAMETER ANALYSIS")
print("=" * 60)     

# Load Breast Cancer Data Set
data = load_breast_cancer()

X = data.data
y = data.target

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Hyperparameter Tuning
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

import time
import pandas as pd

results = []

for params in ParameterGrid(param_grid):

    start_time = time.time()

    rf = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        random_state=42
    )

    cv_score = cross_val_score(
        rf,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    ).mean()

    rf.fit(X_train, y_train)

    train_score = rf.score(X_train, y_train)

    training_time = time.time() - start_time

    overfit_gap = train_score - cv_score

    results.append({
        "n_estimators": params["n_estimators"],
        "max_depth": params["max_depth"],
        "min_samples_split": params["min_samples_split"],
        "train_accuracy": round(train_score, 4),
        "cv_accuracy": round(cv_score, 4),
        "overfit_gap": round(overfit_gap, 4),
        "training_time": round(training_time, 4)
    })

results_df = pd.DataFrame(results)

print("\nHyperparameter Results:")
print(results_df)

results_df.to_csv(
    "hyperparameter_results.csv",
    index=False
)

print("\nHyperparameter Results Saved!")

# ToT Prompt
table_text = results_df.to_string(index=False)

tot_prompt = f"""
You are an expert Machine Learning Engineer.

Use Tree-of-Thought reasoning.

Goal:
Select the best Random Forest hyperparameter configuration.

Follow these steps:

Node 1:
Analyze validation accuracy.

Node 2:
Analyze training accuracy.

Node 3:
Analyze overfitting using train-validation gap.

Node 4:
Compare model complexity.

Node 5:
Classify each configuration into:
- High Bias
- Balanced
- High Variance

Create reasoning branches for promising parameter ranges.

Compare all branches.

Select the best configuration.

Explain why it is superior.

Hyperparameter Results:

{table_text}
"""

# Load Environment
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)

# Model Building
response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
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
