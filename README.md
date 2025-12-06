# 💳 Intelligent Credit Risk Assessment System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A comprehensive machine learning system for credit risk assessment with explainable AI, achieving 82.69% AUC-ROC and $789K profit improvement.


## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Achievements](#project-achievements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

##  Overview

This project implements an end-to-end machine learning solution for credit risk assessment in the banking sector. It combines predictive modeling with explainable AI (SHAP) to provide transparent, accurate credit decisions that comply with regulatory requirements.

### Problem Statement

Financial institutions face challenges with:
- High loan default rates causing significant losses
- Traditional credit scoring missing complex patterns
- Regulatory requirements for explainable decisions
- Need for automated, scalable assessment systems

### Solution

Our ML-powered system:
- Predicts loan default probability with 82.69% AUC-ROC
- Provides SHAP-based explanations for every decision
- Optimizes business profitability through threshold tuning
- Delivers interactive dashboard for real-time assessments

---

## ✨ Key Features

### 🤖 **Advanced Machine Learning**
- 6 models compared: Logistic Regression, Decision Tree, Random Forest, XGBoost, SMOTE variants
- Random Forest selected as best performer (AUC-ROC: 0.8269)
- Handles class imbalance with SMOTE
- Optimized hyperparameters through cross-validation

### 🔍 **Explainable AI**
- SHAP (SHapley Additive exPlanations) integration
- Global feature importance analysis
- Individual prediction explanations
- Transparent decision-making process

### 💼 **Business Analytics**
- ROI calculation and cost-benefit analysis
- Threshold optimization for maximum profitability
- Expected Loss calculations
- What-if scenario analysis

### 🎨 **Interactive Dashboard**
- Real-time credit risk predictions
- Visual SHAP explanations
- Model performance metrics
- Business impact calculator

---

## 🏆 Project Achievements

| Metric | Value |
|--------|-------|
| **AUC-ROC Score** | 0.8269 |
| **Accuracy** | 75% |
| **Precision** | 83.3% |
| **Recall** | 70% |
| **Default Detection Rate** | 89% (at optimized threshold) |
| **Profit Improvement** | $789,500 |
| **Models Compared** | 6 |

### Business Impact

- ✅ Transformed -$385K loss into +$404K profit
- ✅ Optimized decision threshold from 0.5 to 0.25
- ✅ Increased bad credit detection from 33% to 89%
- ✅ Reduced false negatives while maintaining profitability

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/credit-risk-assessment.git
cd credit-risk-assessment
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare Data and Models

Ensure the following directories exist with required files:
```
project/
├── data/
│   ├── X_test.csv
│   └── y_test.csv
├── models/
│   └── random_forest_model.pkl
└── app.py
```

---

## 💻 Usage

### Running the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your default browser at `http://localhost:8501`

### Running Jupyter Notebooks

Navigate through the notebooks in sequence:

```bash
jupyter notebook
```

1. `01_data_exploration.ipynb` - EDA and data understanding
2. `02_preprocessing.ipynb` - Data cleaning and feature engineering
3. `03_model_development.ipynb` - Model training and comparison
4. `03b_results_analysis.ipynb` - Business impact analysis
5. `04_interpretability_shap.ipynb` - SHAP explanations

### Making Predictions (Python API)

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('models/random_forest_model.pkl')

# Prepare features
features = pd.DataFrame({...})  # Your feature data

# Predict
probability = model.predict_proba(features)[0, 1]
decision = 'REJECT' if probability >= 0.25 else 'APPROVE'

print(f"Default Probability: {probability:.2%}")
print(f"Decision: {decision}")
```

---

## 📁 Project Structure

```
credit-risk-assessment/
│
├── data/                          # Data files
│   ├── raw/                       # Original dataset
│   ├── processed/                 # Preprocessed data
│   ├── X_test.csv
│   └── y_test.csv
│
├── models/                        # Trained models
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   └── logistic_regression_model.pkl
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_development.ipynb
│   ├── 03b_results_analysis.ipynb
│   └── 04_interpretability_shap.ipynb
│
├── reports/                       # Generated reports
│   ├── figures/                   # Visualizations
│   ├── technical_report.pdf
│   └── presentation.pptx
│
├── src/                           # Source code
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── utils.py
│
├── app.py                         # Streamlit dashboard
├── requirements.txt               # Dependencies
├── README.md                      # This file
└── LICENSE                        # MIT License
```

---

## 🔬 Methodology

### 1. Data Collection & Exploration
- **Dataset**: German Credit Data (1,000 instances, 20 attributes)
- **Target**: Binary classification (good/bad credit)
- **Features**: Mix of numerical and categorical variables

### 2. Data Preprocessing
- Handled missing values
- Encoded categorical variables (One-Hot Encoding)
- Feature scaling (StandardScaler)
- Train-test split (80-20)
- SMOTE for class imbalance

### 3. Model Development

| Model | Description | AUC-ROC |
|-------|-------------|---------|
| Logistic Regression | Baseline linear model | 0.7543 |
| Decision Tree | Non-linear, interpretable | 0.6892 |
| **Random Forest** | **Best performer** | **0.8269** |
| XGBoost | Gradient boosting | 0.8145 |
| Random Forest + SMOTE | Balanced training | 0.8156 |
| XGBoost + SMOTE | Balanced training | 0.8089 |

### 4. Model Evaluation
- **Metrics**: AUC-ROC, Accuracy, Precision, Recall, F1-Score
- **Cross-validation**: 5-fold stratified
- **Confusion matrix analysis**
- **ROC curve visualization**

### 5. Interpretability
- **SHAP (SHapley Additive exPlanations)**
  - Global feature importance
  - Individual prediction explanations
  - Feature interaction analysis
- **Regulatory compliance**: Transparent, explainable decisions

### 6. Business Analysis
- **Cost-benefit analysis**
- **Threshold optimization**: Default 0.5 → Optimal 0.25
- **Expected Loss calculations**
- **ROI assessment**: $789K improvement

---

## 📊 Results

### Model Performance

**Random Forest (Selected Model)**
- AUC-ROC: **0.8269** ✅
- Accuracy: **75.0%**
- Precision: **83.3%**
- Recall: **70.0%**
- F1-Score: **76.1%**

### Top Risk Factors (SHAP Importance)

1. **Checking Account Status** - Most influential predictor
2. **Duration of Credit** - Longer duration = higher risk
3. **Credit History** - Past behavior matters
4. **Purpose of Loan** - Certain purposes riskier
5. **Credit Amount** - Higher amounts = higher risk

### Business Impact

**Strategy Comparison:**

| Strategy | Net Profit | Approval Rate | Default Rate |
|----------|-----------|---------------|--------------|
| Approve All | -$1,050,000 | 100% | 30% |
| Default ML (0.5) | -$385,500 | 65% | 20% |
| **Optimized ML (0.25)** | **+$404,000** | **51%** | **11%** |
| Reject All | $0 | 0% | 0% |

**Key Insight**: Optimized ML threshold achieves **$789,500 improvement** over naive approval strategy!

---

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.9+** - Programming language
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting framework
- **imbalanced-learn** - SMOTE implementation

### Visualization & Interpretability
- **Matplotlib** - Static visualizations
- **Seaborn** - Statistical graphics
- **Plotly** - Interactive charts
- **SHAP** - Model explanations

### Dashboard & Deployment
- **Streamlit** - Interactive web application
- **Joblib** - Model serialization

### Development Tools
- **Jupyter Notebook** - Interactive development
- **Git** - Version control
- **VS Code** - Code editor

---

## 📈 Future Enhancements

### Short-term (Next 3 months)
- [ ] Add more datasets (Home Credit Default Risk)
- [ ] Implement deep learning models (Neural Networks)
- [ ] Add LIME for additional explanations
- [ ] Deploy to cloud (Streamlit Cloud / Heroku)

### Medium-term (6 months)
- [ ] Real-time model monitoring dashboard
- [ ] A/B testing framework
- [ ] Automated model retraining pipeline
- [ ] API for production integration

### Long-term (12 months)
- [ ] Multi-country regulatory compliance
- [ ] Federated learning for privacy
- [ ] AutoML for continuous optimization
- [ ] Mobile application

---

##  Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Write unit tests for new features
- Update documentation
- Ensure all tests pass before submitting PR

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

**Your Name**
- Email: Prakashdev33@gmail.com
- LinkedIn: www.linkedin.com/in/prakash-deo
- GitHub: [(https://github.com/Prakashdev33))

**Project Link**: [((https://github.com/Prakashdev33/credit-risk-ml-project?tab=readme-ov-file#contact))]

---

## 🙏 Acknowledgments

- German Credit Data from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29)
- SHAP library by Scott Lundberg
- Streamlit for the amazing dashboard framework
- scikit-learn for comprehensive ML tools

---

## 📚 References

### Academic Papers
1. Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions" (SHAP paper)
2. Siddiqi, N. (2017). "Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards"
3. Basel Committee on Banking Supervision - "Basel III Framework"

### Technical Documentation
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

<div align="center">

**⭐ If you found this project helpful, please consider giving it a star! ⭐**

</div>
