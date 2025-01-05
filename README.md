# Bank-Loan-Approval-Classification-Machine-Learning-Project
This machine learning project predicts whether a bank loan application will be approved or not, based on demographic, financial, and credit history data. The project implements multiple classification models, with a focus on achieving high accuracy using an optimized Stacking Algorithm.
# Project Overview
- Objective: To predict loan approval decisions using historical loan application data.
- Dataset: Contains 61,000+ records after balancing. Key features include applicant demographics, credit history, and financial attributes.
# Workflow
1. Exploratory Data Analysis (EDA):
- Analyzed data distributions and identified key correlations between features.
- Visualized missing values, outliers, and class imbalances to improve data quality.
2. Data Preprocessing:
- Handled missing values and treated outliers to ensure data consistency.
- Balanced the dataset using resampling techniques to address class imbalances.
- Standardized features using ```StandardScaler``` to optimize model performance.
3. Model Building:
- Implemented and compared the following models:
- Random Forest Classifier
- Decision Tree Classifier (with hyperparameter tuning and post-pruning)
- AdaBoost Classifier
- Support Vector Classifier (SVC)
- Logistic Regression
- Gaussian Naive Bayes
- Designed a Stacking Algorithm, combining all the above models as base learners, achieving a 98% overall accuracy.
4. Evaluation:
- Evaluated model performance using accuracy, precision, recall, F1-score, and confusion matrix.
- The optimized Stacking Algorithm demonstrated the best generalization across training and testing datasets.
5. Model Deployment:
- Saved the final model as ```Bank Loan Approval Classification.pkl``` for future use and deployment.
# Results
- Achieved ```98%``` overall accuracy with the optimized Stacking Algorithm.
- Significantly improved prediction accuracy through resampling and outlier treatment.
# Tools and Technologies
- Languages: Python
- Libraries: Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn
- Platform: Jupyter Notebook
# Repository Contents
- ```loan_data.csv```: Dataset used for this project.
- ```Bank Loan Approval Classification.ipynb```: Complete project notebook, including EDA, preprocessing, model training, and evaluation.
- ```Bank Loan Approval Classification.pkl```: Dumped file of the final optimized model.
# How to Run
```git clone https://github.com/shubhu111/bank-loan-approval-classification.git```
