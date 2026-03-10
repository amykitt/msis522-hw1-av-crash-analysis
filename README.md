# msis522-hw1-av-crash-analysis
AV Crash Severity Prediction
MSIS 522 Advanced Machine Learning | Foster School of Business, University of Washington
Project Overview
This project applies machine learning to predict fatal crash severity using the NHTSA Fatality Analysis Reporting System (FARS) 2023 dataset. The goal is to classify whether a fatal crash results in a single fatality, two fatalities, or multiple fatalities — a problem directly relevant to autonomous vehicle safety systems.
Dataset

Source: NHTSA FARS 2023 (Fatality Analysis Reporting System)
Size: ~37,600 fatal crashes, 81 features
Target variable: Crash severity (Single_Fatal, Two_Fatals, Multi_Fatal)
Challenge: Severe class imbalance (~85% Single_Fatal)

Models Trained
ModelNotesLogistic RegressionBaseline linear modelDecision TreeTuned via GridSearchCVRandom ForestEnsemble, tuned via GridSearchCVXGBoostBest model — F1 ~0.905, AUC ~0.885Neural Network (MLP)2 hidden layers, 128 units, 50 epochs
Results
XGBoost achieved the best performance with macro F1 of ~0.905 and AUC-ROC of ~0.885. SHAP analysis identified PERSONS and VE_TOTAL as the strongest predictors of multi-fatality outcomes.
Streamlit App
Live App: https://amykitt-msis522-hw1-av-crash-analysis.streamlit.app/
Features 4 tabs:

Executive Summary
Descriptive Analytics
Model Performance
Explainability & Interactive Prediction

How to Run Locally
bashgit clone https://github.com/amykitt/msis522-hw1-av-crash-analysis
cd msis522-hw1-av-crash-analysis
pip install -r requirements.txt
streamlit run app/app.py
```

## Repository Structure
```
├── data/               # NHTSA FARS 2023 dataset
├── notebooks/          # analysis.ipynb - full workflow
├── models/             # Saved model files
├── outputs/            # Figures and model comparison
├── app/                # Streamlit application
└── requirements.txt