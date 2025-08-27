🌟 NutriSight

NutriSight is a web-based health assistant that combines machine learning, image processing, and NLP-powered food analysis to support people with Diabetic Retinopathy (DR).

It helps users detect the stage of DR from their retinal fundus images and recommends diabetes-friendly meals from restaurant menus, empowering healthier dining choices.

🚀 Features

🧠 Diabetic Retinopathy Detection

Preprocessing: CLAHE, grayscale conversion

Feature extraction: GLCM, LBP, entropy, vessel density

Classification: Trained XGBoost model

Predicts 5 DR stages with confidence scores

🍲 Personalized Food Advisor

Analyzes restaurant menus via text input or web-scraping

Extracts nutritional data and evaluates dishes against DR-specific dietary rules

Recommends the most diabetes-friendly options

🔗 Integration of Medical + Nutrition Insights

Bridges diagnosis and lifestyle choices

Provides users with personalized, stage-aware recommendations

🛠️ Tech Stack

Machine Learning: XGBoost

Image Processing: OpenCV, scikit-image

NLP & Data Handling: NLTK / difflib, pandas

Frontend: Streamlit

Backend: Python

📊 DR Detection Workflow

Input: Retinal fundus image

Preprocessing: CLAHE enhancement + grayscale

Feature Extraction:

Gray-Level Co-occurrence Matrix (GLCM)

Local Binary Patterns (LBP)

Image entropy

Vessel density

Classification: XGBoost model → predicts No DR, Mild, Moderate, Severe, or Proliferative

Output: Predicted stage + confidence score

🍽️ Nutrition Recommendation Workflow

Input: Restaurant menu (image/text/web-scraped)

Processing: NLP-based dish name matching with nutrition database

Analysis: Evaluates calories, carbs, sugars, sodium, etc.

Output: Safe vs unsafe dishes with personalized recommendations based on DR stage
