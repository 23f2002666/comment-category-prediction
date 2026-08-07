# Comment Category Prediction

This repository contains a machine learning pipeline to predict the categories assigned to user-generated comments on an online platform. The project utilizes a combination of textual, numerical, and categorical features to perform multi-class classification.

## Project Overview
In online discussion systems, understanding and categorizing user comments is crucial for moderation, engagement analysis, and content recommendation. This project addresses the multi-class classification task by processing text data alongside metadata (such as user engagement signals and symbolic expressions) to predict the final category assigned to each comment.

## Dataset
The dataset represents single comment records and includes:
* **Textual Data:** Raw user comments.
* **Interaction Metadata:** Upvotes, downvotes, and symbolic emoticon usage.
* **Categorical Features:** Race, religion, and the month the comment was posted.
* **Internal Signals:** System-defined indicators (`if_1`, `if_2`) and disability flags.

*Note: The raw datasets are not included in this repository due to file size limitations but can be obtained from the Kaggle competition page.*

## Pipeline & Feature Engineering
A unified pipeline was constructed using scikit-learn's `ColumnTransformer` to process different data types simultaneously:
1. **Numerical Features:** Normalized using `StandardScaler` (e.g., upvotes, downvotes, emoticons).
2. **Categorical Features:** Encoded using `OneHotEncoder` (e.g., race, religion, posting month) with unknown values handled gracefully.
3. **Textual Features:** Vectorized using `TfidfVectorizer` (limited to 8,000 features, sublinear TF scaling, and English stop words removed).

## Models Evaluated
Multiple architectures were explored and evaluated to address the class imbalance present in the dataset:
* **Logistic Regression:** Regularized linear model utilizing balanced class weights to improve minority class detection.
* **Random Forest Classifier:** A tree-based ensemble approach.
* **LightGBM Classifier:** A gradient boosting framework designed for speed and efficiency with high-dimensional sparse data.
* **Calibrated LightGBM:** Post-processing probability calibration applied via `CalibratedClassifierCV` to optimize decision thresholds.
* **Voting Ensemble:** A soft-voting classifier combining the predictions of Random Forest and LightGBM models.

## Key Results
The models were evaluated using the macro-averaged F1-score to ensure fair performance tracking across imbalanced classes. 

| Model | Overall Accuracy | Macro F1-Score | Class 3 F1-Score |
| :--- | :---: | :---: | :---: |
| Random Forest | 90% | 0.75 | 0.44 |
| Logistic Regression | 91% | 0.81 | 0.63 |
| LightGBM | 91% | 0.81 | 0.62 |
| Calibrated LGBM | 91% | 0.81 | 0.62 |
| Voting Ensemble | 91% | 0.81 | 0.64 |

* **Class 3 (Minority Class) Handling:** While the Random Forest model achieved high precision on the rare Class 3, its recall was limited. The Voting Ensemble and Calibrated LightGBM models offered a more balanced trade-off, significantly improving recall for the minority class while maintaining strong overall accuracy.

