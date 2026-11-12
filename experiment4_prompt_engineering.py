"""EXPERIMENT 4: Prompt Engineering for Text Tasks.

AIM:
    To study how clear prompts and instructions guide a program to
    perform text tasks such as summarization, sentiment analysis and
    classification.

ALGORITHM:
    1. Prepare the input text for a selected task.
    2. Define a suitable instruction or rule for the task.
    3. Process the text using the corresponding function.
    4. Generate the summary, sentiment or category.
    5. Display the result.

Sample Output:
    Summary: AI is useful. It helps people.
    Sentiment: Positive
    Category: Technology

RESULT:
    Thus, text-processing tasks were demonstrated using clear
    instructions and prompt-style task definitions.
"""

import re


def summarize(t: str) -> str:
    return " ".join(re.split(r"(?<=[.!?])\s+", t)[:2])


def sentiment(t: str) -> str:
    pos = ["good", "happy", "great", "love", "excellent"]
    neg = ["bad", "sad", "poor", "hate", "worst"]
    words = t.lower().split()
    p, n = sum(x in words for x in pos), sum(x in words for x in neg)
    return "Positive" if p > n else "Negative" if n > p else "Neutral"


def classify(t: str) -> str:
    t = t.lower()
    if any(x in t for x in ["python", "computer", "ai"]):
        return "Technology"
    if any(x in t for x in ["school", "college", "student"]):
        return "Education"
    if any(x in t for x in ["cricket", "football", "match"]):
        return "Sports"
    return "Other"


def main() -> None:
    print("Summary:", summarize("AI is useful. It helps people."))
    print("Sentiment:", sentiment("I am very happy and good"))
    print("Category:", classify("Python is a programming language"))


if __name__ == "__main__":
    main()
