# 🏦 Loan Default Risk Analysis

## Machine Learning Based Credit Risk Assessment System

A production-quality Streamlit web application that predicts whether a borrower is likely to default on a loan using Machine Learning. Built as a final year internship project.

---

## 📋 Project Overview

This project uses Machine Learning to analyse borrower demographics, financial history, and loan characteristics to predict loan default risk. The interactive dashboard provides:

- **Real-time risk prediction** with probability scores
- **Interactive data exploration** with filtering
- **Model performance comparison** across 5 algorithms
- **Business insights** and strategic recommendations

---

## 🛠️ Tech Stack

| Technology    | Purpose                        |
|---------------|--------------------------------|
| Python        | Core programming language      |
| Streamlit     | Web dashboard framework        |
| Pandas        | Data manipulation              |
| NumPy         | Numerical computing            |
| Scikit-learn  | ML model training & evaluation |
| Plotly        | Interactive visualisations     |
| Matplotlib    | Static plots                   |
| Seaborn       | Statistical visualisations     |
| Joblib        | Model serialisation            |

---

## 📁 Project Structure

```
Loan_Default_Risk_Analysis/
│
├── Data/
│   ├── loan_default.csv           # Raw dataset
│   └── loan_default_clean.csv     # Cleaned dataset
│
├── Model/
│   ├── loan_default_model.pkl     # Trained model
│   ├── scaler.pkl                 # Feature scaler
│   ├── encoders.pkl               # Label encoders
│   ├── model_results.pkl          # Model comparison results
│   └── feature_names.pkl          # Feature column names
│
├── Images/
│
├── app.py                         # Streamlit application
├── setup_project.py               # Data generation & model training
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/Loan_Default_Risk_Analysis.git
   cd Loan_Default_Risk_Analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the dashboard**
   ```bash
   streamlit run app.py
   ```

---

## 📊 Dashboard Pages

| Page               | Description                                        |
|--------------------|----------------------------------------------------|
| 🏠 Home            | KPI cards, dataset overview, default distribution  |
| 🔍 Data Exploration| Interactive EDA with filtering and correlation map  |
| 🤖 Model Performance| Model comparison, feature importance, confusion matrix |
| 🔮 Loan Prediction | Real-time prediction form with risk assessment      |
| 💡 Business Insights| Key findings, charts, and banking recommendations  |
| ℹ️ About           | Project details, ML workflow, technologies          |

---

## 🤖 Models Trained

| Model                | Accuracy | Precision | Recall | F1    | ROC AUC |
|----------------------|----------|-----------|--------|-------|---------|
| Decision Tree        | 80.14    | 19.58     | 22.85  | 21.08 | 55.26   |
| Gradient Boosting    | 88.64    | 63.66     | 5.11   | 9.46  | 75.78   |
| Logistic Regression  | 88.52    | 60.24     | 3.32   | 6.30  | 75.31   |
| Random Forest        | 88.53    | 64.15     | 2.87   | 5.49  | 73.63   |
| **Tuned Random Forest** | **78.07** | **26.00** | **50.00** | **35.00** | **73.66** |

**Best Model:** Tuned Random Forest — optimised for Recall to catch the maximum number of defaulters.

---

## 📈 Key Features

- ✅ Real-time loan default prediction
- ✅ Interactive Plotly visualisations
- ✅ Professional banking dashboard UI
- ✅ Model comparison with radar charts
- ✅ Downloadable prediction results
- ✅ Dark mode compatible
- ✅ Responsive layout
- ✅ Business recommendations

---

## 📜 Licence

This project is developed for academic purposes as a final year internship project.

---

*Built with ❤️ using Streamlit and Scikit-learn*
