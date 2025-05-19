# Bank Customer Churn Prediction 

This project predicts customer churn in a bank using machine learning models, as part of an MLOps learning pipeline. It includes:

- Data preprocessing & balancing
- Multiple model experiments
- MLflow experiment tracking
- Model registration in MLflow
- Visualizations
- Clean and modular Python code



## Models & Evaluation
Three models were trained and evaluated:

🌲 Random Forest Classifier
Accuracy: 75.9%

F1 Score: 74.9

Precision: 76.88


Suitable for production due to its robust generalization.


⚡ XGBoost Classifier
Accuracy: 73.9%

F1 Score: 73.14

Precision: 74.2

Fast and efficient. Good alternative to Random Forest.

Slightly lower performance than RF in this case.


✅ Model Registration (via MLflow)
Production model: Random Forest

Chosen for best balance between accuracy, precision, and recall.

Staging model: XGBoost

Slightly behind in accuracy, but still reliable.

🛠 Improvements
Modularized training and preprocessing code.

Rebalanced the dataset for fair learning.

Saved visualizations as MLflow artifacts.

Added model signature for reproducibility.



