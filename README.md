# Chain-of-Thought (CoT) and Tree-of-Thought (ToT) Reasoning using Gemini

## Overview

This project demonstrates advanced Prompt Engineering techniques using Google's Gemini API.

The project consists of two modules:

### 1. Chain-of-Thought (CoT) Sentiment Analysis

Analyzes customer reviews and performs step-by-step reasoning before assigning a sentiment label.

Features:

* Positive phrase extraction
* Negative phrase extraction
* Mixed sentiment detection
* Final sentiment classification
* Explainable AI reasoning

### 2. Tree-of-Thought (ToT) Hyperparameter Analysis

Uses Gemini to evaluate multiple Random Forest configurations and select the best model through structured reasoning.

Features:

* Cross-validation analysis
* Overfitting detection
* Bias-Variance analysis
* Model complexity comparison
* Best hyperparameter selection

## Technologies Used

* Python
* Gemini API
* Scikit-Learn
* Pandas
* Prompt Engineering
* Chain-of-Thought (CoT)
* Tree-of-Thought (ToT)

## Installation

```bash
pip install -r requirements.txt
```

Create a .env file:

```env
GEMINI_API_KEY=your_api_key
```

## Run CoT Sentiment Analysis

```bash
python CoT.py
```

## Run ToT Hyperparameter Analysis

```bash
python ToT.py
```

## Outputs

* sentiment_results.txt
* hyperparameter_results.csv
* tot_analysis.txt

## Author

Venkata Durga Rao Eelaprolu
