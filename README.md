# Breast Cancer Detection using SVM

This project demonstrates the use of Support Vector Machines (SVM) for detecting breast cancer using the Breast Cancer Wisconsin (Diagnostic) dataset. The goal is to build a model that can classify tumors as either benign or malignant based on various features.

## Dataset

The Breast Cancer Wisconsin dataset contains information about tumor characteristics, including radius, texture, perimeter, area, smoothness, and more. The target variable indicates whether a tumor is benign (0) or malignant (1).

## Methodology

1. **Data Loading and Preprocessing:** 
   - The dataset is loaded from scikit-learn. 
   - The data is split into training and testing sets. 
   - Data is standardized using StandardScaler to improve model performance.

2. **Model Training (Initial):**
   - A basic SVM model with a linear kernel is trained on the training data.

3. **Prediction and Evaluation:**
   - The trained model is used to predict the class labels for the test data.
   - The model's accuracy, confusion matrix, and classification report are generated to evaluate its performance.

4. **Hyperparameter Tuning:**
   - GridSearchCV is employed to find the best hyperparameters for the SVM model, including C, gamma, and kernel type.
   - The model is retrained with the best parameters.

5. **New Data Prediction:**
   - A hypothetical new sample is introduced to demonstrate how the model can be used to classify new tumors.


## Usage

This code can be run in a Jupyter Notebook or Google Colab environment. Ensure necessary libraries (numpy, matplotlib, scikit-learn, seaborn, pandas) are installed.


## Results

The model achieves a high accuracy in classifying tumors, with performance measures such as precision, recall, and F1-score displayed in the classification report.  

## Note

This is a basic example for educational purposes. For real-world applications, more sophisticated model selection, feature engineering, and validation techniques are required.
