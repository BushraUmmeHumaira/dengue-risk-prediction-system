# Dengue Risk Prediction System

Predicts the probability of a high-risk dengue week for each ward in Chattogram, Gazipur and Narayanganj, using weekly weather, ward conditions and past cases. We use classical statistical ML and a small neural network, with no LLM or generative AI in the prediction pipeline.

## Team
- Bushra Umme Humaira - data cleaning and feature engineering
- Sidratul Muntaha - Machine learning prediction engine
- Athkia Tabassum Chowdhury - Streamlit dashboard

## Architecture
```mermaid
flowchart LR
    A[Dengue dataset CSV] --> B[Preprocessing]
    B --> C[Feature engineering]
    C --> D[Train 5 models]
    D --> E[Evaluate and select best]
    E --> F[Ward risk probabilities]
    F --> G[Streamlit UI]
```

## Dataset
Weekly data for 20 wards in 3 cities, 2016-2025 (10,440 rows): weather, ward conditions and weekly dengue cases.

## Models
Logistic Regression, Random Forest, XGBoost, LightGBM and a neural network (MLP). They are trained on 2016-2023 and tested on 2024-2025, and compared with accuracy, precision, recall, F1 and ROC-AUC. See docs/models.md.

## How to run
Requires Python 3.
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
streamlit run app/app.py
```
