# ML Assignment 2

## Problem Statement
Build and compare multiple classification models on a public dataset, then expose the results through an interactive Streamlit app that can evaluate uploaded test data.

## Dataset Description
Dataset used: Breast Cancer Wisconsin (Diagnostic)

Source: UCI Machine Learning Repository

Type: Binary classification

Samples: 569

Features: 30 numeric features

Target classes:

- `0` = malignant
- `1` = benign

## GitHub Repository Link

Replace this placeholder with the final repository URL before submission.

https://github.com/nitinkale14/ML_Assignment2-main

## Models Used

The assignment text lists five explicit models and one line that says six. I implemented the five models that are actually enumerated in the table and instructions.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9825 | 0.9957 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest (Ensemble) | 0.9561 | 0.9931 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

### Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Best overall on this split. It achieved the highest accuracy, AUC, and MCC, which suggests the dataset is largely linearly separable. |
| Decision Tree | The weakest generalizer here. It is easy to explain, but the single-tree structure loses stability compared with the other models. |
| KNN | Strong second-tier result after scaling. Its performance is close to the ensemble model on this split. |
| Naive Bayes | Competitive and fast, but still below the strongest models because of its independence assumption. |
| Random Forest (Ensemble) | Very strong AUC and F1, but it still trails logistic regression on the main ranking metrics for this split. |

Overall winner for this dataset: Logistic Regression as it achieved the highest accuracy, AUC, and MCC, which suggests the dataset is largely linearly separable.

## Streamlit App

Deploy `app.py` on Streamlit Community Cloud after pushing this repo to GitHub.

The app supports:

- CSV upload for test data
- Model selection dropdown
- Metric display
- Confusion matrix and classification report

Submission placeholders to replace before exporting the final PDF:

- Live Streamlit app link: https://mlassignment2-mnpzvuutlnmmbutdvkpjvd.streamlit.app/
- BITS Virtual Lab screenshot: add the captured screenshot file to the submission PDF

## Files Included

- `app.py`
- `train_models.py`
- `ml_utils.py`
- `requirements.txt`
- `test_data.csv`
- `model/`

## How to Recreate

1. Install the requirements : pip install -r requirements.txt
2. Run `python train_models.py` to generate the saved model files and `test_data.csv`.
3. Start the app with `streamlit run app.py`.

## Deployment Steps
1. Push this folder to GitHub.
2. Open Streamlit Community Cloud. 
3. Create a new app from the repository. 
4. Select app.py as the entry point. 
5. Deploy and copy the live app URL into this README and the submitted PDF.