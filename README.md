# Predicting Payment Default with XGBoost and SHAP

Welcome to the Predicting Payment Default project! This initiative focuses on developing a robust machine learning pipeline to predict payment defaults based on historical client data. By leveraging numerical data and advanced interpretability tools like SHAP, the project aims to assist financial institutions in identifying high-risk clients and enabling proactive risk management strategies.

## Project Overview

### 🔹 Feature Engineering & Model Implementation
This project emphasizes the creation of predictive features and addresses class imbalance to improve recall for identifying high-risk clients.

**Techniques Used:**
- Feature engineering (e.g., average payment delay, payment trends)
- Class imbalance handling (e.g., SMOTE, undersampling)
- XGBoost for its robustness and high-performance capabilities

**Objective:**
Develop a production-ready model optimized for recall and capable of accurate risk classification.

### 🔹 Model Explainability with SHAP
SHAP (SHapley Additive exPlanations) values are utilized to provide both global and individual-level explanations for model predictions.

**Key Features:**
- SHAP Summary Plots: Highlight overall feature importance.
- SHAP Dependence Plots: Show feature impacts on predictions.
- SHAP Force Plots: Provide individual-level explanations for decision-making.

**Objective:**
Ensure model predictions are interpretable and actionable for stakeholders.

### 🔹 Model Fairness Testing
The project incorporates fairness testing to ensure unbiased predictions across diverse client groups.

**Techniques Used:**
- Fairness testing libraries
- Evaluation of biases and mitigation strategies

**Objective:**
Build a model that is fair and equitable for all client demographics.

### 🔹 Deployment
The final model is deployed in a production environment, enabling real-time predictions for new clients.

---

## Installation
To get started with this project, follow the instructions below:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Predicting-Payment-Default.git
   ```
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Load the Dataset
Load the cleaned dataset (`Client_Data.csv`) into your script:
```python
import pandas as pd
data = pd.read_csv('cleaned_data.csv')
```

### 2. Run the Models
- **Baseline Model:** Run the XGBoost model with basic features. See `model_baseline.py` for details.
- **Enhanced Model:** Run the model with additional engineered features and optimized parameters. See `model_advanced.py` for details.

### 3. SHAP Analysis
Use `shap_analysis.py` to generate:
- SHAP Summary Plots
- SHAP Dependence Plots
- SHAP Force Plots

---

## Results

- **Baseline Model:** Provides initial predictions based on historical client data.
- **Enhanced Model:** Improves recall and accuracy through advanced feature engineering.
- **SHAP Insights:** Highlights key features driving predictions, ensuring transparency and actionable insights.

---

## Contributing 🤝
We welcome contributions from the community! Feel free to fork the repository, open issues, and submit pull requests.

---

## License 📄
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
Special thanks to:
- **SHAP Library:** For providing robust tools to explain machine learning predictions.
- **imbalanced-learn:** For class imbalance handling techniques.
- **scikit-learn:** For its versatile suite of machine learning utilities.

