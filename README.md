# CoT-ToT AI Framework using Gemini

## Overview

This project demonstrates advanced Prompt Engineering techniques using Google's Gemini API by implementing two powerful reasoning frameworks:

* **Chain-of-Thought (CoT)** for explainable sentiment analysis
* **Tree-of-Thought (ToT)** for AI-assisted hyperparameter optimization

The framework combines Generative AI with Machine Learning to produce transparent, step-by-step reasoning and improve decision-making processes.

---

## Module 1: Chain-of-Thought (CoT) Sentiment Analysis

### Objective

Analyze customer reviews and classify them into:

* Positive
* Neutral
* Negative

### Features

* Positive phrase extraction
* Negative phrase extraction
* Mixed sentiment detection
* Dominant sentiment identification
* Final sentiment classification
* Explainable AI reasoning

### Workflow

1. Load customer reviews from a CSV file.
2. Send reviews to Gemini API.
3. Apply Chain-of-Thought prompting.
4. Generate step-by-step reasoning.
5. Assign the final sentiment label.
6. Save results to a text file.

---

## Module 2: Tree-of-Thought (ToT) Hyperparameter Analysis

### Objective

Evaluate multiple Random Forest configurations and identify the best-performing model using structured reasoning.

### Features

* Cross-validation analysis
* Training accuracy evaluation
* Overfitting detection
* Bias-Variance analysis
* Model complexity comparison
* Best hyperparameter selection

### Workflow

1. Load the Breast Cancer dataset.
2. Train multiple Random Forest models.
3. Calculate performance metrics.
4. Send evaluation results to Gemini API.
5. Apply Tree-of-Thought reasoning.
6. Compare reasoning branches.
7. Select the optimal configuration.

---

## Technologies Used

* Python
* Gemini API
* Scikit-Learn
* Pandas
* NumPy
* Prompt Engineering
* Chain-of-Thought (CoT)
* Tree-of-Thought (ToT)
* Machine Learning

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Venky-eelaprolu/CoT-ToT-AI-Framework.git
cd CoT-ToT-AI-Framework
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key
```

---

## Run Chain-of-Thought Sentiment Analysis

```bash
python CoT.py
```

Output:

* sentiment_results.txt

---

## Run Tree-of-Thought Hyperparameter Analysis

```bash
python ToT.py
```

Outputs:

* hyperparameter_results.csv
* tot_analysis.txt

---

## Skills Demonstrated

* Prompt Engineering
* Generative AI
* Explainable AI (XAI)
* Large Language Models (LLMs)
* Chain-of-Thought Reasoning
* Tree-of-Thought Reasoning
* Machine Learning
* Hyperparameter Optimization
* Sentiment Analysis
* Model Evaluation

---

## Author

**Venkata Durga Rao Eelaprolu**

GenAI Intern | AI & Machine Learning Enthusiast
