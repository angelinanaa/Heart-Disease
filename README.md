# 💖 Heart Disease Prediction using Machine Learning

## 📚 Overview

This project aims to predict the likelihood of heart disease in patients using various machine learning algorithms. We employ different classification techniques and compare their performance to determine the best model for predicting heart disease based on medical attributes.

## 📊 Dataset

We utilize the [Heart Disease Dataset from Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset). The dataset includes the following medical attributes:

- **Age**: The age of the patient in years.
- **Sex**: The gender of the patient (1 = male, 0 = female).
- **Chest Pain Type**: Type of chest pain experienced (4 values):
  - 0: Typical angina
  - 1: Atypical angina
  - 2: Non-anginal pain
  - 3: Asymptomatic
- **Resting Blood Pressure**: Resting blood pressure in mm Hg.
- **Serum Cholesterol**: Serum cholesterol in mg/dl.
- **Fasting Blood Sugar**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false).
- **Resting Electrocardiographic Results**: Results of the resting electrocardiogram (values 0, 1, 2).
- **Maximum Heart Rate Achieved**: The maximum heart rate achieved during exercise.
- **Exercise Induced Angina**: Exercise-induced angina (1 = yes, 0 = no).
- **Oldpeak**: ST depression induced by exercise relative to rest.
- **Slope of the Peak Exercise ST Segment**: The slope of the peak exercise ST segment (0, 1, 2).
- **Number of Major Vessels**: Number of major vessels (0-3) colored by fluoroscopy.
- **Thalassemia (Thal)**:
  - 0: Normal
  - 1: Fixed defect
  - 2: Reversible defect

## ⚙️ Installation

1. Clone the repository and navigate to the project directory.
2. Set up a virtual environment and install the required dependencies.
3. Download the dataset from Kaggle and place it in the project directory.

## 🚀 Usage

1. Preprocess the data.
2. Train the models.
3. Evaluate the models.
4. Predict using the best model.

## 🔍 Cross-Validation

We use cross-validation to ensure the robustness and generalizability of our models. Cross-validation involves splitting the data into multiple subsets and training/testing the model on different combinations of these subsets. This helps to mitigate overfitting and provides a better estimate of the model's performance on unseen data.

## 🎯 Hyperparameter Tuning

We perform hyperparameter tuning to optimize the performance of our machine learning models. This process involves selecting the best set of hyperparameters for each model by using techniques such as grid search or random search. Hyperparameter tuning helps in improving the model's accuracy and overall performance.

## 📈 Models and Performance

| Model                   | Accuracy |
|-------------------------|----------|
| Logistic Regression     | 86.5%    |
| Decision Tree           | 89.6%    |
| Random Forest           | 93.7%    |
| SVM                     | 93.7%    |
| KNN                     | 87.5%    |

## 📊 Evaluation Metrics

We use the following evaluation metrics to assess model performance:
- **Accuracy**: Measures the proportion of correctly predicted outcomes.
- **Precision**: Indicates the proportion of true positive predictions among all positive predictions.
- **Recall (Sensitivity)**: Measures the proportion of actual positives that are correctly identified.
- **F1 Score**: Harmonic mean of precision and recall, providing a balance between the two metrics.

## 📉 Learning Curves

Learning curves visualize the model's performance over training iterations:
- **Training Curve**: Plots the model's performance on the training data over successive training batches or epochs.
- **Validation Curve**: Shows how well the model generalizes to unseen data during training.

## 🤝 Contributing

1. Fork the repository.
2. Create a new branch for your feature.
3. Make your changes and commit them.
4. Push your changes and create a Pull Request.

