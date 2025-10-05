import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.utils import load_object

# Page configuration
st.set_page_config(
    page_title="NASA Exoplanet Classifier",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .confirmed {
        background-color: #C8E6C9;
        border-left: 5px solid #4CAF50;
    }
    .candidate {
        background-color: #FFF9C4;
        border-left: 5px solid #FFC107;
    }
    .false-positive {
        background-color: #FFCDD2;
        border-left: 5px solid #F44336;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and preprocessor
@st.cache_resource
def load_models():
    """Load the trained model and preprocessor"""
    try:
        model_path = os.path.join('artifacts', 'model.pkl')
        preprocessor_path = os.path.join('artifacts', 'proprocessor.pkl')
        
        model = load_object(model_path)
        preprocessor = load_object(preprocessor_path)
        
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# Define feature columns (matching your training data)
FEATURE_COLUMNS = [
    'koi_pdisposition', 'koi_tce_plnt_num', 'koi_fpflag_nt', 'koi_fpflag_ss',
    'koi_fpflag_co', 'koi_fpflag_ec', 'koi_period', 'koi_period_err1',
    'koi_period_err2', 'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2',
    'koi_time0', 'koi_time0_err1', 'koi_time0_err2', 'koi_eccen', 'koi_incl',
    'koi_sma', 'koi_num_transits', 'koi_duration', 'koi_depth', 'koi_impact',
    'koi_dor', 'koi_ror', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_steff',
    'koi_steff_err1', 'koi_steff_err2', 'koi_slogg', 'koi_slogg_err1',
    'koi_slogg_err2', 'koi_smet', 'koi_smet_err1', 'koi_smet_err2', 'koi_srad',
    'koi_srad_err1', 'koi_srad_err2', 'koi_smass', 'koi_smass_err1',
    'koi_smass_err2', 'koi_srho', 'koi_srho_err1', 'koi_srho_err2',
    'koi_model_snr', 'koi_max_sngle_ev', 'koi_max_mult_ev', 'koi_fittype',
    'koi_count', 'koi_bin_oedp_sig', 'ra', 'dec', 'koi_fwm_stat_sig',
    'koi_fwm_sra', 'koi_fwm_sra_err', 'koi_fwm_sdec', 'koi_fwm_sdec_err',
    'koi_fwm_srao', 'koi_fwm_srao_err', 'koi_fwm_sdeco', 'koi_fwm_sdeco_err',
    'koi_fwm_prao', 'koi_fwm_prao_err', 'koi_fwm_pdeco', 'koi_fwm_pdeco_err',
    'koi_dicco_mra', 'koi_dicco_mra_err', 'koi_dicco_mdec', 'koi_dicco_mdec_err',
    'koi_dicco_msky', 'koi_dicco_msky_err', 'koi_dikco_mra', 'koi_dikco_mra_err',
    'koi_dikco_mdec', 'koi_dikco_mdec_err', 'koi_dikco_msky', 'koi_dikco_msky_err',
    'koi_kepmag', 'koi_gmag', 'koi_rmag', 'koi_imag', 'koi_zmag', 'koi_jmag',
    'koi_hmag', 'koi_kmag'
]

def get_prediction_style(prediction):
    """Return CSS class based on prediction"""
    if prediction == 'CONFIRMED':
        return 'confirmed'
    elif prediction == 'CANDIDATE':
        return 'candidate'
    else:
        return 'false-positive'

def main():
    # Header
    st.markdown('<h1 class="main-header">🪐 NASA Exoplanet Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict whether a Kepler Object of Interest is a Confirmed Exoplanet, Candidate, or False Positive</p>', unsafe_allow_html=True)
    
    # Load models
    model, preprocessor = load_models()
    
    if model is None or preprocessor is None:
        st.error("⚠️ Failed to load models. Please ensure model files exist in the 'artifacts' folder.")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://www.nasa.gov/wp-content/themes/nasa/assets/images/nasa-logo@2x.png", width=200)
        st.title("Input Method")
        input_method = st.radio("Choose input method:", ["Manual Input", "CSV Upload"])
        
        st.markdown("---")
        st.markdown("### 📖 About This Project")
        
        with st.expander("🎯 Project Overview", expanded=False):
            st.markdown("""
            #### **NASA Kepler Exoplanet Classifier**
            
            This application predicts whether a Kepler Object of Interest (KOI) is a **confirmed exoplanet**, 
            a **candidate** requiring further investigation, or a **false positive** using advanced machine learning models.
            
            **Classifications:**
            - ✅ **CONFIRMED**: Verified exoplanet
            - ⚠️ **CANDIDATE**: Potential exoplanet needing validation
            - ❌ **FALSE POSITIVE**: Not an exoplanet (stellar eclipse, noise, etc.)
            """)
        
        with st.expander("❓ Problem Statement", expanded=False):
            st.markdown("""
            The **Kepler Space Telescope** observed over 150,000 stars and detected thousands of potential 
            exoplanet signals. However, manually analyzing this massive dataset is extremely time-consuming 
            and requires significant expertise.
            
            **Our Solution:**  
            This automated classifier helps astronomers quickly identify and prioritize exoplanet candidates, 
            accelerating the discovery process and allowing researchers to focus on the most promising objects 
            for follow-up observations.
            """)
        
        with st.expander("🔬 Methodology", expanded=False):
            st.markdown("""
            **Machine Learning Pipeline:**
            
            1. **Data Preprocessing**
               - Missing value imputation (median for numerical, mode for categorical)
               - Rare category handling for categorical features
               - Feature scaling using StandardScaler
               - One-hot encoding for categorical variables
            
            2. **Model Training**
               - Multiple classifiers evaluated: RandomForest, LogisticRegression, DecisionTree, GaussianNB
               - GridSearchCV for hyperparameter tuning
               - Cross-validation to prevent overfitting
               - Best model selection based on accuracy score
            
            3. **Prediction**
               - Trained model predicts disposition class
               - Confidence probabilities provided for all classes
            """)
        
        with st.expander("📊 Dataset Information", expanded=False):
            st.markdown("""
            **Source:** NASA Exoplanet Archive - Kepler Objects of Interest (KOI)  
            **Link:** [NASA Exoplanet Archive](http://exoplanetarchive.ipac.caltech.edu)
            
            **Dataset Details:**
            - **Total Features:** 80+ stellar and planetary characteristics
            - **Key Features:**
              - Orbital parameters (period, eccentricity, semi-major axis)
              - Transit properties (depth, duration, impact parameter)
              - Stellar properties (temperature, radius, mass, metallicity)
              - False positive flags and centroid measurements
            - **Target Classes:** CONFIRMED, CANDIDATE, FALSE POSITIVE
            - **Training Samples:** Thousands of validated KOIs
            
            **Data Quality:**  
            All data undergoes rigorous preprocessing including outlier detection, 
            feature engineering, and validation before model training.
            """)
        
        with st.expander("📖 How to Use", expanded=False):
            st.markdown("""
            **Manual Input Mode:**
            1. Select "Manual Input" from the input method radio button
            2. Enter stellar and planetary parameters in the form
            3. Click the "🚀 Predict" button
            4. View prediction results with confidence scores
            
            **CSV Upload Mode:**
            1. Select "CSV Upload" from the input method radio button
            2. Download the sample template (optional)
            3. Upload your CSV file with KOI data
            4. Click "🚀 Predict All" for batch predictions
            5. Download results as CSV for further analysis
            
            **Tips:**
            - Hover over input fields for helpful tooltips
            - Higher confidence scores indicate more reliable predictions
            - Use batch mode for analyzing multiple objects efficiently
            """)
        
        with st.expander("👥 Credits & References", expanded=False):
            st.markdown("""
            **Developed By:** Space Data Science Team  
            **Project Type:** NASA Space Apps Challenge / Educational ML Project
            
            **Data Source:**
            - NASA Kepler Mission
            - NASA Exoplanet Archive
            - Kepler Input Catalog (KIC)
            
            **Technologies Used:**
            - **ML Framework:** Scikit-learn
            - **Data Processing:** Pandas, NumPy
            - **Web Interface:** Streamlit
            - **Visualization:** 
            - **Model Persistence:** Pickle
            
            **References:**
            - [Kepler Objects of Interest (KOI)](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)
            - [Exoplanet Archive Documentation](https://exoplanetarchive.ipac.caltech.edu/docs/)
            - [Malik, A., Moster, B. P., & Obermeier, C. (2021). Exoplanet detection using machine learning. Monthly Notices of the Royal Astronomical Society, 513(4), 5505–5516.] (https://doi.org/10.1093/mnras/stab3692)
            
            **Acknowledgments:**  
            This project uses publicly available data from NASA's Kepler mission. 
            Special thanks to the Kepler Science Team and NASA Exoplanet Archive.
            """)
        
        st.markdown("---")
        st.markdown("### 🌟 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Features", "80+")
            st.metric("Classes", "3")
        with col2:
            st.metric("Models", "4+")
            st.metric("Accuracy", "~90%")
    
    # Main content
    if input_method == "Manual Input":
        show_manual_input(model, preprocessor)
    else:
        show_csv_upload(model, preprocessor)

def show_manual_input(model, preprocessor):
    """Show manual input form for single prediction"""
    st.header("Enter KOI Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Orbital Parameters")
        koi_period = st.number_input("Orbital Period (days)", value=10.0, help="Time to complete one orbit")
        koi_time0bk = st.number_input("Transit Epoch (BKJD)", value=131.0)
        koi_eccen = st.number_input("Eccentricity", value=0.0, min_value=0.0, max_value=1.0)
        koi_incl = st.number_input("Inclination (degrees)", value=89.0)
        koi_sma = st.number_input("Semi-Major Axis (AU)", value=0.1)
        koi_duration = st.number_input("Transit Duration (hrs)", value=3.0)
        
    with col2:
        st.subheader("Planet Properties")
        koi_depth = st.number_input("Transit Depth (ppm)", value=1000.0)
        koi_prad = st.number_input("Planetary Radius (Earth radii)", value=1.0)
        koi_teq = st.number_input("Equilibrium Temperature (K)", value=300.0)
        koi_insol = st.number_input("Insolation Flux (Earth flux)", value=1.0)
        koi_impact = st.number_input("Impact Parameter", value=0.5)
        koi_dor = st.number_input("Planet-Star Distance/Star Radius", value=10.0)
        
    with col3:
        st.subheader("Stellar Parameters")
        koi_steff = st.number_input("Stellar Effective Temp (K)", value=5778.0)
        koi_slogg = st.number_input("Stellar Surface Gravity (log10)", value=4.5)
        koi_smet = st.number_input("Stellar Metallicity (dex)", value=0.0)
        koi_srad = st.number_input("Stellar Radius (Solar radii)", value=1.0)
        koi_smass = st.number_input("Stellar Mass (Solar mass)", value=1.0)
        koi_num_transits = st.number_input("Number of Transits", value=10, step=1)
    
    # Additional parameters with default values
    col4, col5 = st.columns(2)
    
    with col4:
        st.subheader("False Positive Flags")
        koi_fpflag_nt = st.selectbox("Not Transit-Like", [0, 1], index=0)
        koi_fpflag_ss = st.selectbox("Stellar Eclipse", [0, 1], index=0)
        koi_fpflag_co = st.selectbox("Centroid Offset", [0, 1], index=0)
        koi_fpflag_ec = st.selectbox("Ephemeris Match", [0, 1], index=0)
        
    with col5:
        st.subheader("Categorical Features")
        koi_pdisposition = st.selectbox("Previous Disposition", ["CANDIDATE", "FALSE POSITIVE"])
        koi_fittype = st.selectbox("Fit Type", ["LS", "DV"])
        koi_count = st.number_input("Planet Count", value=1, step=1)
    
    if st.button("🚀 Predict", type="primary"):
        # Create input dataframe with all required features
        input_data = create_input_dict(
            koi_period, koi_time0bk, koi_eccen, koi_incl, koi_sma, koi_duration,
            koi_depth, koi_prad, koi_teq, koi_insol, koi_impact, koi_dor,
            koi_steff, koi_slogg, koi_smet, koi_srad, koi_smass, koi_num_transits,
            koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, koi_fpflag_ec,
            koi_pdisposition, koi_fittype, koi_count
        )
        
        make_prediction(model, preprocessor, input_data)

def create_input_dict(koi_period, koi_time0bk, koi_eccen, koi_incl, koi_sma, koi_duration,
                     koi_depth, koi_prad, koi_teq, koi_insol, koi_impact, koi_dor,
                     koi_steff, koi_slogg, koi_smet, koi_srad, koi_smass, koi_num_transits,
                     koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, koi_fpflag_ec,
                     koi_pdisposition, koi_fittype, koi_count):
    """Create input dictionary with all features, filling missing ones with defaults"""
    
    # Create base dictionary with provided values
    input_dict = {
        'koi_pdisposition': koi_pdisposition,
        'koi_tce_plnt_num': 1,
        'koi_fpflag_nt': koi_fpflag_nt,
        'koi_fpflag_ss': koi_fpflag_ss,
        'koi_fpflag_co': koi_fpflag_co,
        'koi_fpflag_ec': koi_fpflag_ec,
        'koi_period': koi_period,
        'koi_period_err1': koi_period * 0.001,
        'koi_period_err2': -koi_period * 0.001,
        'koi_time0bk': koi_time0bk,
        'koi_time0bk_err1': 0.01,
        'koi_time0bk_err2': -0.01,
        'koi_time0': koi_time0bk + 2454833.0,
        'koi_time0_err1': 0.01,
        'koi_time0_err2': -0.01,
        'koi_eccen': koi_eccen,
        'koi_incl': koi_incl,
        'koi_sma': koi_sma,
        'koi_num_transits': koi_num_transits,
        'koi_duration': koi_duration,
        'koi_depth': koi_depth,
        'koi_impact': koi_impact,
        'koi_dor': koi_dor,
        'koi_ror': np.sqrt(koi_depth / 1e6),
        'koi_prad': koi_prad,
        'koi_teq': koi_teq,
        'koi_insol': koi_insol,
        'koi_steff': koi_steff,
        'koi_steff_err1': koi_steff * 0.01,
        'koi_steff_err2': -koi_steff * 0.01,
        'koi_slogg': koi_slogg,
        'koi_slogg_err1': 0.1,
        'koi_slogg_err2': -0.1,
        'koi_smet': koi_smet,
        'koi_smet_err1': 0.1,
        'koi_smet_err2': -0.1,
        'koi_srad': koi_srad,
        'koi_srad_err1': koi_srad * 0.05,
        'koi_srad_err2': -koi_srad * 0.05,
        'koi_smass': koi_smass,
        'koi_smass_err1': koi_smass * 0.05,
        'koi_smass_err2': -koi_smass * 0.05,
        'koi_srho': 1.41,
        'koi_srho_err1': 0.1,
        'koi_srho_err2': -0.1,
        'koi_model_snr': 20.0,
        'koi_max_sngle_ev': 15.0,
        'koi_max_mult_ev': 25.0,
        'koi_fittype': koi_fittype,
        'koi_count': koi_count,
        'koi_bin_oedp_sig': 0.5,
        'ra': 290.0,
        'dec': 45.0,
    }
    
    # Add flux-weighted centroid features with default values
    fwm_features = {
        'koi_fwm_stat_sig': 0.0, 'koi_fwm_sra': 19.0, 'koi_fwm_sra_err': 0.01,
        'koi_fwm_sdec': 45.0, 'koi_fwm_sdec_err': 0.01, 'koi_fwm_srao': 0.0,
        'koi_fwm_srao_err': 0.1, 'koi_fwm_sdeco': 0.0, 'koi_fwm_sdeco_err': 0.1,
        'koi_fwm_prao': 0.0, 'koi_fwm_prao_err': 0.1, 'koi_fwm_pdeco': 0.0,
        'koi_fwm_pdeco_err': 0.1
    }
    
    # Add difference image centroid features
    dicco_features = {
        'koi_dicco_mra': 0.0, 'koi_dicco_mra_err': 0.1, 'koi_dicco_mdec': 0.0,
        'koi_dicco_mdec_err': 0.1, 'koi_dicco_msky': 0.0, 'koi_dicco_msky_err': 0.1,
        'koi_dikco_mra': 0.0, 'koi_dikco_mra_err': 0.1, 'koi_dikco_mdec': 0.0,
        'koi_dikco_mdec_err': 0.1, 'koi_dikco_msky': 0.0, 'koi_dikco_msky_err': 0.1
    }
    
    # Add magnitude features
    mag_features = {
        'koi_kepmag': 14.0, 'koi_gmag': 14.5, 'koi_rmag': 14.0, 'koi_imag': 13.8,
        'koi_zmag': 13.6, 'koi_jmag': 13.0, 'koi_hmag': 12.8, 'koi_kmag': 12.7
    }
    
    # Merge all features
    input_dict.update(fwm_features)
    input_dict.update(dicco_features)
    input_dict.update(mag_features)
    
    return pd.DataFrame([input_dict])

def show_csv_upload(model, preprocessor):
    """Show CSV upload interface for batch predictions"""
    st.header("Upload CSV File for Batch Predictions")
    
    st.info("""
    📄 **CSV Requirements:**
    - Must contain all required feature columns
    - Can contain multiple rows for batch prediction
    - Download sample template below to see expected format
    """)
    
    # Download sample template
    if st.button("📥 Download Sample CSV Template"):
        sample_data = pd.DataFrame(columns=FEATURE_COLUMNS)
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="Download Template",
            data=csv,
            file_name="koi_template.csv",
            mime="text/csv"
        )
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! {len(df)} rows found.")
            
            st.subheader("Preview of uploaded data")
            st.dataframe(df.head())
            
            if st.button("🚀 Predict All", type="primary"):
                # Ensure all required columns are present
                missing_cols = set(FEATURE_COLUMNS) - set(df.columns)
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                else:
                    make_batch_predictions(model, preprocessor, df)
                    
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")

def make_prediction(model, preprocessor, input_df):
    """Make single prediction and display results"""
    try:
        # Preprocess
        input_transformed = preprocessor.transform(input_df)
        
        # Predict
        prediction = model.predict(input_transformed)[0]
        probabilities = model.predict_proba(input_transformed)[0]
        
        # Get class labels
        classes = model.classes_
        
        # Display result
        st.markdown("---")
        st.header("🔮 Prediction Results")
        
        pred_style = get_prediction_style(prediction)
        
        st.markdown(f"""
        <div class="prediction-box {pred_style}">
            <h2>Classification: {prediction}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Show probabilities
        st.subheader("Confidence Scores")
        prob_df = pd.DataFrame({
            'Classification': classes,
            'Probability': probabilities
        }).sort_values('Probability', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(prob_df, use_container_width=True)
        
        with col2:
            import plotly.express as px
            fig = px.bar(prob_df, x='Classification', y='Probability',
                        color='Probability', color_continuous_scale='Viridis',
                        title='Prediction Confidence')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        st.subheader("📊 Interpretation")
        max_prob = probabilities.max()
        
        if prediction == 'CONFIRMED' and max_prob > 0.8:
            st.success("✅ High confidence that this is a confirmed exoplanet!")
        elif prediction == 'CANDIDATE':
            st.warning("⚠️ This object shows promise but requires further verification.")
        elif prediction == 'FALSE POSITIVE':
            st.error("❌ This object is likely not an exoplanet.")
        else:
            st.info("ℹ️ Prediction made with moderate confidence. Consider additional analysis.")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

def make_batch_predictions(model, preprocessor, df):
    """Make predictions for batch CSV upload"""
    try:
        # Select only the required features
        input_df = df[FEATURE_COLUMNS]
        
        # Preprocess
        input_transformed = preprocessor.transform(input_df)
        
        # Predict
        predictions = model.predict(input_transformed)
        probabilities = model.predict_proba(input_transformed)
        
        # Create results dataframe
        results_df = df.copy()
        results_df['Predicted_Disposition'] = predictions
        
        # Add probability columns for each class
        for i, class_name in enumerate(model.classes_):
            results_df[f'Probability_{class_name}'] = probabilities[:, i]
        
        # Display results
        st.markdown("---")
        st.header("📊 Batch Prediction Results")
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            confirmed = (predictions == 'CONFIRMED').sum()
            st.metric("✅ Confirmed", confirmed)
        
        with col2:
            candidate = (predictions == 'CANDIDATE').sum()
            st.metric("⚠️ Candidates", candidate)
        
        with col3:
            false_pos = (predictions == 'FALSE POSITIVE').sum()
            st.metric("❌ False Positives", false_pos)
        
        # Show results table
        st.subheader("Detailed Results")
        st.dataframe(results_df, use_container_width=True)
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name="koi_predictions.csv",
            mime="text/csv"
        )
        
        # Visualization
        st.subheader("Distribution of Predictions")
        import plotly.express as px
        
        pred_counts = pd.DataFrame({
            'Classification': ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'],
            'Count': [confirmed, candidate, false_pos]
        })
        
        fig = px.pie(pred_counts, values='Count', names='Classification',
                     title='Prediction Distribution',
                     color='Classification',
                     color_discrete_map={
                         'CONFIRMED': '#4CAF50',
                         'CANDIDATE': '#FFC107',
                         'FALSE POSITIVE': '#F44336'
                     })
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error making batch predictions: {str(e)}")

if __name__ == "__main__":
    main()