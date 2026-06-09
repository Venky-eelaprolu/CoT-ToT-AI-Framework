# CoT-ToT AI Framework using Gemini API

## Overview

This project demonstrates advanced Prompt Engineering techniques using Google's Gemini API by implementing two powerful reasoning frameworks:

* **Chain-of-Thought (CoT)** for Explainable Sentiment Analysis
* **Tree-of-Thought (ToT)** for AI-Assisted Hyperparameter Optimization

The framework combines Generative AI and Machine Learning to perform transparent, step-by-step reasoning and improve decision-making workflows.

---

## Features

### Chain-of-Thought (CoT) Sentiment Analysis

Analyzes customer reviews and generates explainable sentiment classifications through structured reasoning.

#### Capabilities

* Positive phrase extraction
* Negative phrase extraction
* Mixed sentiment detection
* Dominant sentiment identification
* Sentiment classification (Positive, Neutral, Negative)
* Explainable AI reasoning

#### Workflow

```text
Customer Reviews
       │
       ▼
 Gemini API
       │
       ▼
 Chain-of-Thought Reasoning
       │
       ▼
 Sentiment Classification
       │
       ▼
 sentiment_results.txt
```

---

### Tree-of-Thought (ToT) Hyperparameter Analysis

Uses Gemini to evaluate multiple Random Forest configurations and select the best-performing model through structured reasoning.

#### Capabilities

* Cross-validation analysis
* Training accuracy evaluation
* Overfitting detection
* Bias-Variance analysis
* Model complexity comparison
* Best hyperparameter selection

#### Workflow

```text
Breast Cancer Dataset
          │
          ▼
 Random Forest Models
          │
          ▼
 Hyperparameter Evaluation
          │
          ▼
 Gemini API
          │
          ▼
 Tree-of-Thought Reasoning
          │
          ▼
 Best Configuration Selection
```

---

## Technologies Used

* Python
* Gemini API
* Scikit-Learn
* Pandas
* NumPy
* Prompt Engineering
* Generative AI
* Chain-of-Thought (CoT)
* Tree-of-Thought (ToT)
* Machine Learning

---

## Project Structure

```text
CoT-ToT-AI-Framework/
│
├── sentiment_analysis_cot.py
├── hyperparameter_analysis_tot.py
├── reviews.csv
├── requirements.txt
├── README.md
└── .gitignore
```

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
python sentiment_analysis_cot.py
```

### Output

```text
sentiment_results.txt
```

---

## Run Tree-of-Thought Hyperparameter Analysis

```bash
python hyperparameter_analysis_tot.py
```

### Outputs

```text
hyperparameter_results.csv
tot_analysis.txt
```

---

## Skills Demonstrated

* Prompt Engineering
* Large Language Models (LLMs)
* Generative AI
* Explainable AI (XAI)
* Chain-of-Thought Reasoning
* Tree-of-Thought Reasoning
* Sentiment Analysis
* Hyperparameter Optimization
* Random Forest Modeling
* Model Evaluation
* Cross Validation
* Python Development

---

## Learning Outcomes

Through this project, I explored:

* Advanced prompting strategies for LLMs
* Explainable AI techniques
* AI-assisted decision-making
* Integration of Gemini API with Machine Learning workflows
* Structured reasoning using CoT and ToT frameworks

---

## Author

**Venkata Durga Rao Eelaprolu**

GenAI Intern | AI & Machine Learning Enthusiast

GitHub: https://github.com/Venky-eelaprolu
