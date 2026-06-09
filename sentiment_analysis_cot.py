import os
from dotenv import load_dotenv
import pandas as pd
from google import genai

# Load Environment Variable

load_dotenv()

Api_Key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=Api_Key)

# Task 1
# Sentiment Analysis Using CoT

reviews_df = pd.read_csv("reviews.csv")

reviews_text = "\n\n".join(
    [f"Review {i+1} : {review}"
     for i, review in enumerate(reviews_df["review"])]
)

cot_prompt = f"""
You are an expert Sentiment Analysis Assistant.

Your task is to classify each customer review into one of the following categories:

* Positive
* Neutral
* Negative

Think step-by-step and explain your reasoning before generating the final sentiment label.

Follow the reasoning process below for every review:

Step 1: Identify all positive sentiment phrases.

Step 2: Identify all negative sentiment phrases.

Step 3: Check whether the review contains mixed or contradictory sentiment.

Step 4: Determine which sentiment (positive or negative) is dominant.

Step 5: Assign the final sentiment label.

Step 6: Explain why the review belongs to that sentiment category.

---

Example 1

Review:
"The product is excellent. Battery lasts long and sound quality is fantastic."

Reasoning:

Step 1 - Positive Phrases:

* excellent
* lasts long
* fantastic

Step 2 - Negative Phrases:

* None

Step 3 - Mixed Sentiment:
No

Step 4 - Dominant Sentiment:
Positive

Step 5 - Final Label:
Positive

Step 6 - Explanation:
The review contains multiple strong positive expressions and no negative feedback. The customer is highly satisfied with the product.

---

Example 2

Review:
"The phone works fine, but delivery was delayed."

Reasoning:

Step 1 - Positive Phrases:

* works fine

Step 2 - Negative Phrases:

* delivery was delayed

Step 3 - Mixed Sentiment:
Yes

Step 4 - Dominant Sentiment:
Neither sentiment strongly dominates.

Step 5 - Final Label:
Neutral

Step 6 - Explanation:
The review contains both positive and negative aspects. The product performs adequately, but the delivery issue reduces overall satisfaction.


Example 3

Review:
"Terrible quality. The screen stopped working after two days."

Reasoning:

Step 1 - Positive Phrases:

* None

Step 2 - Negative Phrases:

* terrible quality
* stopped working

Step 3 - Mixed Sentiment:
No

Step 4 - Dominant Sentiment:
Negative

Step 5 - Final Label:
Negative

Step 6 - Explanation:
The review expresses strong dissatisfaction and product failure. No positive feedback is mentioned.


Analyze ALL reviews below:

{reviews_text}

For EACH review provide the following format:

Review Number:

Step 1 - Positive Phrases:

Step 2 - Negative Phrases:

Step 3 - Mixed Sentiment:

Step 4 - Dominant Sentiment:

Step 5 - Final Label:

Step 6 - Explanation:
"""

# Model
response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents=cot_prompt
)

sentiment_output = response.text

print(sentiment_output)

# Create .txt file 
with open("sentiment_results.txt", "w", encoding="utf-8") as f:
    f.write(sentiment_output)

print("\nSentiment Results Saved!")
