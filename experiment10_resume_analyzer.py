"""EXPERIMENT 10: AI-Based Resume Analyzer using Python.

AIM:
    To develop a Python-based resume analyzer that extracts skills,
    compares them with job requirements and calculates a matching
    percentage.

ALGORITHM:
    1. Create a database of technical and soft skills.
    2. Read the resume and job description.
    3. Extract email and phone number using regular expressions.
    4. Identify skills in both texts and compare them.
    5. Display matched skills, missing skills and the matching percentage.

Sample Output:
    Email: candidate@gmail.com
    Phone: 9876543210
    Matched skills: ['python', 'sql', 'machine learning']
    Missing skills: ['power bi']
    Resume matching score: 75.0 %

RESULT:
    Thus, the AI-based resume analyzer extracted contact details,
    compared skills and calculated the resume matching percentage
    successfully.
"""

import re


def main() -> None:
    skills = ["python", "java", "sql", "machine learning",
              "data analysis", "power bi", "communication"]
    resume = input("Enter resume: ").lower()
    job = input("Enter job description: ").lower()

    email = re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", resume)
    phone = re.findall(r"\b\d{10}\b", resume)
    rs = [s for s in skills if s in resume]
    js = [s for s in skills if s in job]
    matched = [s for s in js if s in rs]
    missing = [s for s in js if s not in rs]
    score = len(matched) / len(js) * 100 if js else 0

    print("Email:", email[0] if email else "Not found")
    print("Phone:", phone[0] if phone else "Not found")
    print("Matched skills:", matched)
    print("Missing skills:", missing)
    print("Resume matching score:", round(score, 2), "%")


if __name__ == "__main__":
    main()
