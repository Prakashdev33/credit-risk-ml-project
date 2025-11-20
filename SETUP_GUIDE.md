# 🚀 Dashboard Setup Instructions

This guide will help you set up and run the Credit Risk Assessment Dashboard.

---

## 📋 Prerequisites Checklist

Before starting, ensure you have:
- ✅ Python 3.9 or higher installed
- ✅ Your trained model file (`random_forest_model.pkl`)
- ✅ Test data files (`X_test.csv`, `y_test.csv`)
- ✅ All Jupyter notebooks completed

---

## 🔧 Step-by-Step Setup

### Step 1: Organize Your Project Structure

Create the following folder structure:

```
your_project/
│
├── app.py                    # Streamlit dashboard (provided)
├── requirements.txt          # Python dependencies (provided)
├── README.md                # Project documentation (provided)
│
├── data/                    # Create this folder
│   ├── X_test.csv          # Your test features
│   └── y_test.csv          # Your test labels
│
├── models/                  # Create this folder
│   └── random_forest_model.pkl   # Your trained model
│
└── notebooks/               # Your existing notebooks
    ├── 01_data_exploration.ipynb
    ├── 02_preprocessing.ipynb
    ├── 03_model_development.ipynb
    ├── 03b_results_analysis.ipynb
    └── 04_interpretability_shap.ipynb
```

### Step 2: Create Required Folders

Open your terminal/command prompt and run:

```bash
# Navigate to your project folder
cd path/to/your/project

# Create necessary folders
mkdir data
mkdir models
```

### Step 3: Copy Your Files

You need to copy three files from your notebooks:

#### A. Save Your Model

In your `03_model_development.ipynb`, add this code at the end:

```python
import joblib

# Save the best model (Random Forest)
joblib.dump(model_rf, 'models/random_forest_model.pkl')
print("✓ Model saved to models/random_forest_model.pkl")
```

#### B. Save Test Data

In your `02_preprocessing.ipynb` or `03_model_development.ipynb`, add:

```python
# Save test data
X_test.to_csv('data/X_test.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)
print("✓ Test data saved!")
```

### Step 4: Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

**Note**: This may take 5-10 minutes depending on your internet speed.

### Step 5: Verify Installation

Check if everything is installed correctly:

```bash
python -c "import streamlit; import pandas; import shap; print('✓ All packages installed!')"
```

If you see "✓ All packages installed!", you're ready to go!

---

## 🎯 Running the Dashboard

### Option 1: Command Line

```bash
streamlit run app.py
```

### Option 2: Python Script

```bash
python -m streamlit run app.py
```

The dashboard should automatically open in your default browser at:
```
http://localhost:8501
```

If it doesn't open automatically, manually navigate to the URL shown in the terminal.

---

## 🎨 Using the Dashboard

### 5 Main Pages:

1. **🏠 Home**
   - Project overview
   - Key metrics summary
   - Quick navigation guide

2. **🔮 Make Prediction**
   - Enter applicant information
   - Get real-time credit risk assessment
   - See SHAP explanations

3. **📈 Model Performance**
   - View ROC curves
   - Confusion matrix
   - Model comparison
   - Classification reports

4. **🎯 Feature Analysis**
   - Global feature importance
   - SHAP summary plots
   - Feature statistics

5. **💼 Business Impact**
   - Financial calculations
   - Threshold optimization
   - Strategy comparison
   - ROI analysis

---

## 🐛 Troubleshooting

### Problem 1: "Model file not found"

**Solution:**
```bash
# Check if model file exists
ls models/random_forest_model.pkl

# If not, re-run the model saving code in your notebook
```

### Problem 2: "Data files not found"

**Solution:**
```bash
# Check if data files exist
ls data/X_test.csv
ls data/y_test.csv

# If not, re-run the data saving code in your notebook
```

### Problem 3: "Module not found" errors

**Solution:**
```bash
# Reinstall requirements
pip install -r requirements.txt --upgrade
```

### Problem 4: SHAP is slow

**Solution:**
- SHAP calculations can take time for large datasets
- The dashboard caches results automatically
- First load may be slow, subsequent loads will be faster

### Problem 5: Port already in use

**Solution:**
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

---

## 💡 Tips for Best Experience

1. **Use Chrome or Firefox** for best compatibility
2. **Full screen mode** for better visualization (F11)
3. **Allow pop-ups** if using prediction features
4. **Wait for SHAP calculations** - they can take 30-60 seconds initially
5. **Use the sidebar** for easy navigation

---

## 🔄 Updating the Dashboard

If you make changes to `app.py`:

1. The dashboard auto-reloads when you save changes
2. Or manually refresh the page (Ctrl+R / Cmd+R)
3. Or restart the Streamlit server

---

## 📊 Customization Options

### Change Default Threshold

In `app.py`, find and modify:
```python
threshold = 0.25  # Change this value
```

### Adjust Business Parameters

Default values:
```python
avg_loan_amount = 10000
profit_margin = 0.05
loss_rate = 0.80
```

Modify these in the dashboard's Business Impact page using the sliders.

### Add More Features

To add custom pages, follow this pattern in `app.py`:
```python
elif page == "Your New Page":
    st.header("Your New Page")
    # Your code here
```

---

## 🎓 Next Steps

After successfully running the dashboard:

1. ✅ Take screenshots for your report
2. ✅ Test all features thoroughly
3. ✅ Prepare demo scenarios
4. ✅ Document any insights
5. ✅ Consider video recording a demo

---

## 📞 Need Help?

If you encounter issues:

1. Check this guide again carefully
2. Review error messages in the terminal
3. Verify file paths are correct
4. Ensure all dependencies are installed
5. Check Python version (must be 3.9+)

---

## ✅ Verification Checklist

Before presenting, verify:

- [ ] Dashboard launches without errors
- [ ] All 5 pages load correctly
- [ ] Prediction page works
- [ ] Charts and visualizations display
- [ ] SHAP explanations generate
- [ ] Business metrics calculate correctly
- [ ] No warning messages appear

---

**🎉 Congratulations! Your dashboard is ready for demonstration!**

Remember: This dashboard showcases your excellent ML work in an interactive, professional format. Take your time to explore all features before presenting.

Good luck! 🚀
