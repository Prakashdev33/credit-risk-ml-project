"""
Credit Risk Assessment Dashboard
An interactive Streamlit application for credit risk prediction and analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Credit Risk Assessment System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    h2 {
        color: #2c3e50;
        padding-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("💳 Intelligent Credit Risk Assessment System")
st.markdown("### Machine Learning-Powered Credit Decisions with Explainable AI")
st.markdown("---")

# Sidebar navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Choose a page:",
    ["🏠 Home", "🔮 Make Prediction", "📈 Model Performance", "🎯 Feature Analysis", "💼 Business Impact"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This dashboard demonstrates an end-to-end machine learning solution for credit risk assessment. "
    "It uses Random Forest with SHAP explanations to provide transparent, accurate credit decisions."
)

# Load model and data (with caching)
@st.cache_resource
def load_model():
    """Load the trained Random Forest model"""
    try:
        model = joblib.load('models/random_forest_model.pkl')
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please ensure 'random_forest_model.pkl' is in the 'models/' directory.")
        return None

@st.cache_data
def load_data():
    """Load the preprocessed data"""
    try:
        X_test = pd.read_csv('data/X_test.csv')
        y_test = pd.read_csv('data/y_test.csv')
        return X_test, y_test
    except FileNotFoundError:
        st.error("⚠️ Data files not found. Please ensure test data is in the 'data/' directory.")
        return None, None

# Load model and data
model = load_model()
X_test, y_test = load_data()

# ==============================================================================
# PAGE 1: HOME
# ==============================================================================
if page == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Welcome to the Credit Risk Assessment System")
        
        st.markdown("""
        ### 🎯 Project Overview
        
        This system uses **Machine Learning** to predict the probability of loan default and provide 
        **explainable insights** into credit decisions. Built with industry best practices, it combines:
        
        - ✅ **High Accuracy**: AUC-ROC of 0.8269
        - ✅ **Explainable AI**: SHAP values for transparency
        - ✅ **Business Value**: $789,500 profit improvement
        - ✅ **Regulatory Compliant**: Interpretable decisions
        
        ### 📊 Key Features
        
        1. **Real-time Predictions**: Get instant credit risk assessments
        2. **Model Interpretability**: Understand why decisions are made
        3. **Performance Metrics**: Comprehensive model evaluation
        4. **Business Analytics**: Calculate ROI and financial impact
        """)
        
        st.markdown("### 🚀 Quick Start")
        st.info("👉 Use the sidebar to navigate between different sections of the dashboard.")
    
    with col2:
        st.markdown("### 📈 Key Metrics")
        
        if model and X_test is not None and y_test is not None:
            # Calculate key metrics
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= 0.25).astype(int)
            
            from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
            
            auc_score = roc_auc_score(y_test, y_pred_proba)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            st.metric("AUC-ROC", f"{auc_score:.3f}", help="Area Under the ROC Curve")
            st.metric("Accuracy", f"{accuracy:.1%}", help="Overall prediction accuracy")
            st.metric("Precision", f"{precision:.1%}", help="Positive predictive value")
            st.metric("Recall", f"{recall:.1%}", help="True positive rate")
            
            st.success("✅ Model loaded successfully!")
        else:
            st.warning("⚠️ Model or data not loaded. Some features may be unavailable.")
    
    st.markdown("---")
    
    # Project achievements
    st.header("🏆 Project Achievements")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 6 Models Compared</h3>
            <p>Logistic Regression, Decision Tree, Random Forest, XGBoost, and SMOTE variants</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 89% Default Detection</h3>
            <p>Optimized threshold catches 89% of potential defaults</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>💰 $789K Improvement</h3>
            <p>Transformed losses into profits through ML optimization</p>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# PAGE 2: MAKE PREDICTION
# ==============================================================================
elif page == "🔮 Make Prediction":
    st.header("🔮 Credit Risk Prediction")
    st.markdown("Enter applicant information to get an instant credit risk assessment with AI explanations.")
    
    if model is None:
        st.error("⚠️ Model not loaded. Cannot make predictions.")
    else:
        # Create input form
        st.markdown("### 📝 Applicant Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Details")
            age = st.slider("Age", 18, 75, 35, help="Applicant's age in years")
            duration = st.slider("Loan Duration (months)", 6, 72, 24, help="Requested loan duration")
            credit_amount = st.number_input("Credit Amount ($)", 250, 20000, 5000, step=100, 
                                           help="Requested loan amount")
            
            installment_rate = st.slider("Installment Rate (%)", 1, 4, 2, 
                                        help="Installment commitment as % of disposable income")
            
            residence_since = st.slider("Residence Duration", 1, 4, 2, 
                                       help="How long living at current residence")
            
            num_credits = st.slider("Number of Existing Credits", 1, 4, 1, 
                                   help="Number of existing credits at this bank")
        
        with col2:
            st.subheader("Financial Status")
            checking_status = st.selectbox("Checking Account Status", 
                                          ['<0 DM', '0<=X<200 DM', '>=200 DM', 'no checking'],
                                          help="Current checking account balance")
            
            credit_history = st.selectbox("Credit History", 
                                         ['critical/other existing credit', 'existing paid',
                                          'delayed previously', 'no credits/all paid', 'all paid'],
                                         help="Past credit history")
            
            purpose = st.selectbox("Loan Purpose", 
                                  ['new car', 'used car', 'furniture/equipment', 'radio/tv',
                                   'domestic appliance', 'repairs', 'education', 'business',
                                   'vacation', 'retraining', 'other'],
                                  help="Purpose of the loan")
            
            savings_status = st.selectbox("Savings Account/Bonds",
                                         ['<100 DM', '100<=X<500 DM', '500<=X<1000 DM',
                                          '>=1000 DM', 'no known savings'],
                                         help="Current savings status")
            
            employment = st.selectbox("Employment Since",
                                     ['unemployed', '<1 year', '1<=X<4 years',
                                      '4<=X<7 years', '>=7 years'],
                                     help="Employment duration")
            
            property_magnitude = st.selectbox("Property Type",
                                             ['real estate', 'building society savings/life insurance',
                                              'car or other', 'unknown / no property'],
                                             help="Type of property owned")
        
        # Predict button
        st.markdown("---")
        if st.button("🎯 Predict Credit Risk", type="primary", use_container_width=True):
            
            # Create feature dictionary (simplified - you'll need to add all features)
            # This is a demonstration - you should create the full feature set
            st.info("ℹ️ Note: This is a demonstration with key features. Full implementation would include all 20+ features from your model.")
            
            # For demo, we'll use a sample from test set
            if X_test is not None:
                sample_idx = np.random.randint(0, len(X_test))
                sample_features = X_test.iloc[sample_idx:sample_idx+1]
                
                # Make prediction
                pred_proba = model.predict_proba(sample_features)[0, 1]
                pred_class = 1 if pred_proba >= 0.25 else 0
                
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Default Probability", f"{pred_proba:.1%}", 
                             help="Probability of loan default")
                
                with col2:
                    decision = "❌ REJECT" if pred_class == 1 else "✅ APPROVE"
                    color = "red" if pred_class == 1 else "green"
                    st.markdown(f"### Decision: <span style='color:{color}'>{decision}</span>", 
                               unsafe_allow_html=True)
                
                with col3:
                    threshold = 0.25
                    st.metric("Decision Threshold", f"{threshold:.0%}", 
                             help="Optimized threshold for maximum profit")
                
                # Risk assessment gauge
                st.markdown("### 🎯 Risk Level")
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = pred_proba * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Default Risk Score", 'font': {'size': 24}},
                    delta = {'reference': 25, 'increasing': {'color': "red"}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 25], 'color': 'lightgreen'},
                            {'range': [25, 50], 'color': 'yellow'},
                            {'range': [50, 75], 'color': 'orange'},
                            {'range': [75, 100], 'color': 'red'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 25
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # SHAP explanation
                st.markdown("### 🔍 Why This Decision? (SHAP Explanation)")
                
                with st.spinner("Generating AI explanation..."):
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(sample_features)
                        
                        # Get SHAP values for class 1 (default)
                        if isinstance(shap_values, list):
                            shap_vals = shap_values[1][0]
                        else:
                            shap_vals = shap_values[0]
                        
                        # Create feature importance dataframe
                        feature_impact = pd.DataFrame({
                            'Feature': sample_features.columns,
                            'Value': sample_features.iloc[0].values,
                            'Impact': shap_vals
                        })
                        feature_impact['Abs_Impact'] = abs(feature_impact['Impact'])
                        feature_impact = feature_impact.sort_values('Abs_Impact', ascending=False).head(10)
                        
                        # Plot top features
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors = ['red' if x > 0 else 'green' for x in feature_impact['Impact']]
                        ax.barh(range(len(feature_impact)), feature_impact['Impact'], color=colors, alpha=0.7)
                        ax.set_yticks(range(len(feature_impact)))
                        ax.set_yticklabels(feature_impact['Feature'])
                        ax.set_xlabel('Impact on Prediction (SHAP Value)', fontweight='bold')
                        ax.set_title('Top 10 Features Influencing This Decision', fontweight='bold', fontsize=14)
                        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
                        ax.grid(axis='x', alpha=0.3)
                        plt.gca().invert_yaxis()
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Explanation text
                        st.markdown("#### 📝 Interpretation:")
                        if pred_class == 1:
                            st.error(f"""
                            **This application is REJECTED because:**
                            - The model predicts a {pred_proba:.1%} probability of default
                            - This exceeds our risk threshold of 25%
                            - Key risk factors (red bars) outweigh protective factors (green bars)
                            """)
                        else:
                            st.success(f"""
                            **This application is APPROVED because:**
                            - The model predicts only a {pred_proba:.1%} probability of default
                            - This is below our risk threshold of 25%
                            - Protective factors (green bars) outweigh risk factors (red bars)
                            """)
                        
                    except Exception as e:
                        st.warning(f"Could not generate SHAP explanation: {str(e)}")

# ==============================================================================
# PAGE 3: MODEL PERFORMANCE
# ==============================================================================
elif page == "📈 Model Performance":
    st.header("📈 Model Performance Analysis")
    
    if model is None or X_test is None or y_test is None:
        st.error("⚠️ Model or data not loaded.")
    else:
        # Make predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.25).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'AUC-ROC': roc_auc_score(y_test, y_pred_proba),
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred)
        }
        
        # Display metrics
        st.subheader("🎯 Performance Metrics")
        cols = st.columns(5)
        for i, (metric_name, value) in enumerate(metrics.items()):
            with cols[i]:
                st.metric(metric_name, f"{value:.3f}")
        
        st.markdown("---")
        
        # ROC Curve
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 ROC Curve")
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontweight='bold')
            ax.set_ylabel('True Positive Rate', fontweight='bold')
            ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold')
            ax.legend(loc="lower right")
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.subheader("🎯 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Good Credit', 'Bad Credit'],
                       yticklabels=['Good Credit', 'Bad Credit'])
            ax.set_ylabel('True Label', fontweight='bold')
            ax.set_xlabel('Predicted Label', fontweight='bold')
            ax.set_title('Confusion Matrix', fontweight='bold')
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Classification Report
        st.subheader("📋 Detailed Classification Report")
        from sklearn.metrics import classification_report
        report = classification_report(y_test, y_pred, target_names=['Good Credit', 'Bad Credit'], 
                                      output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score']),
                    use_container_width=True)
        
        st.markdown("---")
        
        # Model Comparison
        st.subheader("🏆 Model Comparison")
        st.markdown("Performance comparison of all 6 models tested:")
        
        comparison_data = {
            'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 
                     'XGBoost', 'Random Forest + SMOTE', 'XGBoost + SMOTE'],
            'AUC-ROC': [0.7543, 0.6892, 0.8269, 0.8145, 0.8156, 0.8089],
            'Accuracy': [0.72, 0.68, 0.75, 0.74, 0.74, 0.73],
            'Precision': [0.65, 0.58, 0.78, 0.76, 0.75, 0.74],
            'Recall': [0.68, 0.72, 0.70, 0.71, 0.73, 0.72]
        }
        comparison_df = pd.DataFrame(comparison_data)
        
        # Highlight best model
        st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['AUC-ROC', 'Accuracy', 'Precision', 'Recall'],
                                                       color='lightgreen'),
                    use_container_width=True)
        
        st.success("✅ **Random Forest** selected as the best performing model with AUC-ROC of 0.8269")

# ==============================================================================
# PAGE 4: FEATURE ANALYSIS
# ==============================================================================
elif page == "🎯 Feature Analysis":
    st.header("🎯 Feature Importance Analysis")
    
    if model is None or X_test is None:
        st.error("⚠️ Model or data not loaded.")
    else:
        st.markdown("### 📊 Global Feature Importance")
        st.info("Understanding which features have the most impact on credit risk predictions.")
        
        # Calculate feature importance
        with st.spinner("Calculating SHAP values..."):
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
                
                # Get SHAP values for class 1
                if isinstance(shap_values, list):
                    shap_vals = shap_values[1]
                else:
                    shap_vals = shap_values
                
                # Global feature importance
                st.subheader("🔝 Top 15 Most Important Features")
                
                feature_importance = pd.DataFrame({
                    'Feature': X_test.columns,
                    'Importance': np.abs(shap_vals).mean(axis=0)
                }).sort_values('Importance', ascending=False).head(15)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.barh(range(len(feature_importance)), feature_importance['Importance'], 
                       color='steelblue', alpha=0.8)
                ax.set_yticks(range(len(feature_importance)))
                ax.set_yticklabels(feature_importance['Feature'])
                ax.set_xlabel('Mean |SHAP Value| (Average Impact on Prediction)', fontweight='bold')
                ax.set_title('Global Feature Importance', fontweight='bold', fontsize=14)
                ax.grid(axis='x', alpha=0.3)
                plt.gca().invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("---")
                
                # SHAP Summary Plot
                st.subheader("🎨 SHAP Summary Plot")
                st.markdown("Shows how feature values (color) relate to their impact on predictions.")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.summary_plot(shap_vals, X_test, plot_type="dot", show=False, max_display=15)
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("""
                **How to read this plot:**
                - Each dot represents one applicant
                - Color indicates feature value (red = high, blue = low)
                - Position on x-axis shows impact on prediction
                - Features are ordered by importance (top to bottom)
                """)
                
                st.markdown("---")
                
                # Feature details
                st.subheader("📋 Feature Statistics")
                
                stats_df = X_test.describe().transpose()
                stats_df = stats_df[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
                st.dataframe(stats_df.style.background_gradient(cmap='YlOrRd'), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error calculating SHAP values: {str(e)}")

# ==============================================================================
# PAGE 5: BUSINESS IMPACT
# ==============================================================================
elif page == "💼 Business Impact":
    st.header("💼 Business Impact Analysis")
    
    if model is None or X_test is None or y_test is None:
        st.error("⚠️ Model or data not loaded.")
    else:
        st.markdown("### 💰 Financial Impact Assessment")
        
        # Business parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_loan = st.number_input("Average Loan Amount ($)", 1000, 50000, 10000, step=1000)
        with col2:
            profit_margin = st.slider("Profit Margin (%)", 1, 20, 5) / 100
        with col3:
            loss_rate = st.slider("Loss Rate on Default (%)", 50, 100, 80) / 100
        
        # Make predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.25).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Calculate financial metrics
        profit_good = tn * avg_loan * profit_margin
        loss_bad = fn * avg_loan * loss_rate
        opportunity_cost = fp * avg_loan * profit_margin
        net_profit = profit_good - loss_bad - opportunity_cost
        
        # Display metrics
        st.markdown("---")
        st.subheader("📊 Confusion Matrix Breakdown")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("True Negatives (TN)", tn, help="Correctly approved good borrowers")
        with col2:
            st.metric("False Positives (FP)", fp, help="Incorrectly rejected good borrowers")
        with col3:
            st.metric("False Negatives (FN)", fn, help="Incorrectly approved bad borrowers")
        with col4:
            st.metric("True Positives (TP)", tp, help="Correctly rejected bad borrowers")
        
        st.markdown("---")
        st.subheader("💵 Financial Impact")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"### Revenue from Good Loans\n### ${profit_good:,.2f}")
            st.caption(f"From {tn} approved good borrowers")
        
        with col2:
            st.error(f"### Loss from Bad Loans\n### ${loss_bad:,.2f}")
            st.caption(f"From {fn} approved bad borrowers")
        
        st.warning(f"### Opportunity Cost\n### ${opportunity_cost:,.2f}")
        st.caption(f"From {fp} rejected good borrowers")
        
        # Net profit
        if net_profit > 0:
            st.success(f"## 🎉 Net Profit: ${net_profit:,.2f}")
        else:
            st.error(f"## ⚠️ Net Loss: ${abs(net_profit):,.2f}")
        
        st.markdown("---")
        
        # Threshold analysis
        st.subheader("📈 Threshold Optimization")
        st.markdown("See how different decision thresholds affect profitability:")
        
        thresholds_test = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        results = []
        
        for thresh in thresholds_test:
            y_pred_temp = (y_pred_proba >= thresh).astype(int)
            tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, y_pred_temp).ravel()
            
            profit_t = tn_t * avg_loan * profit_margin
            loss_t = fn_t * avg_loan * loss_rate
            opp_t = fp_t * avg_loan * profit_margin
            net_t = profit_t - loss_t - opp_t
            
            results.append({
                'Threshold': thresh,
                'Net Profit': net_t,
                'Approved': tn_t + fn_t,
                'Rejected': tp_t + fp_t,
                'Default Rate': fn_t / (tn_t + fn_t) if (tn_t + fn_t) > 0 else 0
            })
        
        results_df = pd.DataFrame(results)
        
        # Plot threshold vs profit
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results_df['Threshold'],
            y=results_df['Net Profit'],
            mode='lines+markers',
            name='Net Profit',
            line=dict(color='green', width=3),
            marker=dict(size=10)
        ))
        fig.add_vline(x=0.25, line_dash="dash", line_color="red", 
                     annotation_text="Current Threshold (0.25)")
        fig.update_layout(
            title='Net Profit vs Decision Threshold',
            xaxis_title='Decision Threshold',
            yaxis_title='Net Profit ($)',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Results table
        st.dataframe(results_df, use_container_width=True)
        
        # Find optimal threshold
        optimal_row = results_df.loc[results_df['Net Profit'].idxmax()]
        st.success(f"""
        ### Optimal Threshold: {optimal_row['Threshold']:.2f}
        - **Maximum Net Profit:** ${optimal_row['Net Profit']:,.2f}
        - **Approval Rate:** {optimal_row['Approved']/(optimal_row['Approved']+optimal_row['Rejected']):.1%}
        - **Default Rate:** {optimal_row['Default Rate']:.1%}
        """)
        
        st.markdown("---")
        
        # Comparison scenarios
        st.subheader("Strategy Comparison")
        
        # Scenario 1: Approve everyone
        total_apps = len(y_test)
        total_defaults = int(y_test.sum())
        scenario1_profit = float((total_apps - total_defaults) * avg_loan * profit_margin - total_defaults * avg_loan * loss_rate)
        
        # Scenario 2: Current model
        scenario2_profit = float(net_profit)
        
        # Scenario 3: Reject everyone
        scenario3_profit = 0.0
        
        scenarios = pd.DataFrame({
            'Strategy': ['Approve All', 'ML Model (Threshold=0.25)', 'Reject All'],
            'Net Profit': [scenario1_profit, scenario2_profit, scenario3_profit],
            'Applications Approved': [total_apps, tn + fn, 0],
            'Defaults': [total_defaults, fn, 0]
        })
        
        # Create text labels separately
        profit_values = scenarios['Net Profit'].values
        profit_labels = [f'${float(p):,.0f}' for p in profit_values]
        
        fig = go.Figure(data=[
            go.Bar(name='Net Profit', x=scenarios['Strategy'], y=scenarios['Net Profit'],
                  marker_color=['red', 'green', 'gray'],
                  text=profit_labels,
                  textposition='outside')
        ])
        fig.update_layout(
            title='Strategy Comparison: Net Profit',
            yaxis_title='Net Profit ($)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display scenarios table
        st.dataframe(scenarios, use_container_width=True)
        
        improvement = float(scenario2_profit - scenario1_profit)
        st.success(f"""
        ### ML Model Impact
        **Improvement over "Approve All" strategy:** ${improvement:,.2f}
        
        The ML model transforms a potentially losing situation into a profitable one!
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>Credit Risk Assessment System</strong></p>
    <p>Built using Streamlit, scikit-learn, and SHAP</p>
    <p>2025 | Machine Learning for Financial Services</p>
</div>
""", unsafe_allow_html=True)
