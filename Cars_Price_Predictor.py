import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import Data_Cleaning as dc
from babel.numbers import format_currency

# Custom CSS for enhanced styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: #f5f7fb;
    }
    
    .hero {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        padding: 4rem 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    .step-number {
        background: #6366f1;
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #4ade80 0%, #22c55e 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.5rem;
        margin: 2rem 0;
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stSelectbox div[data-baseweb="select"] {
        border-radius: 12px !important;
    }
    
    .stNumberInput input, .stTextInput input {
        border-radius: 12px !important;
    }
    
    .gradient-button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: 12px !important;
        transition: transform 0.2s !important;
    }
    
    .gradient-button:hover {
        transform: scale(1.05);
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization and other backend logic remains the same...

def display_homepage():
    # Hero Section
    st.markdown("""
    <div class="hero">
        <h1 style="font-size: 3rem; margin-bottom: 1rem;">🚗 Smart Car Valuation</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">
        Instant Used Car Price Predictions Powered by AI
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Features Grid
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="step-number">📊</div>
            <h3>Market Insights</h3>
            <p>Real-time pricing analysis based on current market trends</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="step-number">🤖</div>
            <h3>AI-Powered</h3>
            <p>Advanced machine learning algorithms for accurate predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="step-number">💡</div>
            <h3 >Smart Analysis</h3>
            <p>Automated data imputation for missing fields and make predictions</p>
        </div>
        """, unsafe_allow_html=True)

    # How It Works
    st.markdown("""
    <div style="margin: 3rem 0;">
        <h2 style="color: #1e293b; text-align: center; margin-bottom: 2rem;">How It Works</h2>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem;">
            <div style="text-align: center;">
                <div class="step-number">1</div>
                <h4>Input Details</h4>
                <p style="color: #64748b;">Provide basic car information</p>
            </div>
            <div style="text-align: center;">
                <div class="step-number">2</div>
                <h4>AI Analysis</h4>
                <p style="color: #64748b;">Our algorithms process the data</p>
            </div>
            <div style="text-align: center;">
                <div class="step-number">3</div>
                <h4>Get Results</h4>
                <p style="color: #64748b;">Instant price estimation</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # CTA Section
    st.markdown("""
    <div style="text-align: center; margin: 4rem 0;">
        <h2 style="color: #1e293b; margin-bottom: 1.5rem;">Ready to Discover Your Car's Value?</h2>
        <a href="/#used-cars-price-predictor" class="gradient-button" style="text-decoration: none;">
        Get Started Now →
        </a>
    </div>
    """, unsafe_allow_html=True)

o_df = pd.read_csv('final_cars_data_for_ml.csv')
df = o_df.copy()
input = None
Price_str = None

with st.sidebar:
    option = option_menu(
        menu_title="Menu",
        options=["Home", "Car Price Predictor"],
        icons=["house-heart", "car-front-fill"],
        menu_icon="cast",
        default_index=0,
        styles={"container": {"width": "100%"}},         
    )

# Home
if option == 'Home':
    display_homepage()

# Prediction Page Enhancements
elif option == 'Car Price Predictor':
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem;">
        <h1 style="color: #2E86C1;">Car Valuation Calculator</h1>
        <p style="color: #64748b;">Fill in the details below to get an instant price estimate</p>
    </div>
    """, unsafe_allow_html=True)

    # Define UI-to-DF mapping
    ui_to_df_mapping = {
        'City': 'City',
        'Body Type': 'Body_Type',
        'Ownerno': 'Ownerno',
        'Company': 'Company',
        'Model': 'Model',
        'Model Year': 'Modelyear',
        'Variant': 'Variantname',
        'Registration Year': 'Registration_Year',
        'Insurance Validity': 'Insurance_Validity',
        'Fuel Type': 'Fuel_Type',
        'Seats': 'Seats',
        'Kms Driven': 'Kms_Driven',
        'Transmission': 'Transmission',
        'Manufacture Year': 'Year_Of_Manufacture',
        'Engine (CC)': 'Engine',
        'Mileage': 'Mileage',
    }

    

    # Define mandatory fields (UI-friendly names)
    mandatory_fields = [
        'City', 'Body Type', 'Ownerno', 'Company', 'Model', 'Model Year', 'Registration Year',
        'Kms Driven', 'Insurance Validity', 'Variant'
    ]

    
    
    # Create columns for inputs
    user_inputs = {}

    # Create 4 columns layout for the inputs with consistent spacing
    st.markdown("<style>.stForm div.row-widget {margin-bottom: 50px;}</style>", unsafe_allow_html=True)
    cols = st.columns(3, gap="medium")

        # Handle "Clear All Fields"
    
     

    for i, (ui_label, df_column) in enumerate(ui_to_df_mapping.items()):
        with cols[i % 3]:
            # Add a star (*) for mandatory fields
            label = f"{ui_label} *" if ui_label in mandatory_fields else ui_label

            # Dynamic field types
            if ui_label in ['City', 'Insurance Validity']:
                # Convert to strings, drop NaNs, capitalize, and sort
                options = [None] + sorted(
                    pd.Series(o_df[df_column].unique()).dropna().astype(str).str.capitalize().tolist()
                )
                user_inputs[df_column] = st.selectbox(label, options, key=df_column)
            elif ui_label == 'Body Type':
                options = [None] + sorted(
                    pd.Series(df['Body_Type'].unique()).dropna().astype(str).str.capitalize().tolist()
                )
                user_inputs[df_column] = st.selectbox(label, options, key=df_column)
                if user_inputs[df_column]:
                    df = df[df['Body_Type'].astype(str).str.capitalize() == user_inputs[df_column]]
            elif ui_label == 'Ownerno':
                user_inputs[df_column] = st.number_input(label, min_value=1, max_value=5, value=1, step=1, key=df_column)
                # user_inputs[df_column] = st.slider(label, min_value=1, max_value=5, value=1, step=1, key=df_column)
            elif ui_label == 'Company':
                options = [None] + sorted(
                    pd.Series(df['Company'].unique()).dropna().astype(str).str.capitalize().tolist()
                )
                user_inputs[df_column] = st.selectbox(label, options, key=df_column)
                if user_inputs[df_column]:
                    df = df[df['Company'].astype(str).str.capitalize() == user_inputs[df_column]]
            elif ui_label == 'Model':
                options = [None] + sorted(
                    pd.Series(df['Model'].unique()).dropna().astype(str).str.capitalize().tolist()
                )
                user_inputs[df_column] = st.selectbox(label, options, key=df_column)
                if user_inputs[df_column]:
                    df = df[df['Model'].astype(str).str.capitalize() == user_inputs[df_column]]
            elif ui_label == 'Variant':
                options = [None] + sorted(
                    pd.Series(df['Variantname'].unique()).dropna().astype(str).str.capitalize().tolist()
                )
                user_inputs[df_column] = st.selectbox(label, options, key=df_column)
            elif ui_label in ['Registration Year', 'Manufacture Year', 'Model Year']:
                user_inputs[df_column] = st.number_input(label, min_value=2000, max_value=2023, value=2015, step=1, key=df_column
                )
            elif ui_label == 'Kms Driven':
                user_inputs[df_column] = st.number_input(label, min_value=0, max_value=500000, value=10000, step=500, key=df_column
                )
            elif ui_label == 'Mileage':
                user_inputs[df_column] = st.text_input(label, value='', key=df_column)
            else:
                options = [None] + sorted(
                    pd.Series(df[df_column].unique()).dropna().astype(str).str.capitalize().tolist()
                )
                user_inputs[df_column] = st.selectbox(label, options, key=df_column)
    
    with cols[1]:
        st.markdown("""
        <style>
        .gradient-button {
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            color: white !important;
            border: none !important;
            padding: 12px 24px !important;
            border-radius: 12px !important;
            transition: transform 0.2s !important;
            width: 100%;
            font-size: 16px;
            font-weight: bold;
        }
        .gradient-button:hover {
            transform: scale(1.05);
            opacity: 0.9;
        }
        </style>
        """, unsafe_allow_html=True)

        if st.button("🚀 Get Estimate", key="predict_button", help="Click to predict the car price"):
            # Check for missing mandatory fields
            missing_fields = [ui_label for ui_label in mandatory_fields if not user_inputs[ui_to_df_mapping[ui_label]]]
            if missing_fields:
                st.error(f"Please fill the mandatory fields: {', '.join(missing_fields)}")
            else:
                with st.spinner('Estimating....'):
                    # Convert user inputs into a DataFrame            
                    input_data = {df_column: (user_inputs[df_column] if user_inputs[df_column] else None)for df_column in ui_to_df_mapping.values()}
                    input_df = pd.DataFrame.from_records([input_data]) 
                    merged_df = dc.process_and_impute_missing_values(o_df, input_df)
                    entered_data = merged_df.iloc[[-1]].reset_index(drop=True)                
                    entered_data = entered_data.drop(columns='Price')
                    input = entered_data
                    preprocessor = joblib.load('preprocessing_pipeline.pkl')
                    processed_inputs = preprocessor.transform(entered_data)
                    model = joblib.load('trained_model.pkl')
                    prediction = model.predict(processed_inputs)
                    formatted_price = format_currency(np.round(np.exp(prediction[0]), 0), 'INR', locale='en_IN').replace('₹', '₹ ')
                    Price_str = f"Price for the {entered_data['Company'].iloc[0]} {entered_data['Model'].iloc[0]} is: {formatted_price}"
if Price_str != None:
    # st.write(Price_str)
    st.markdown(f"""
        <div class="prediction-result">
            <div style="font-size: 2rem; margin-bottom: 1rem;">🎉 Estimated Value</div>
            <div style="font-size: 2.5rem; font-weight: bold;">{formatted_price}</div>
            <div style="margin-top: 1rem; font-size: 1rem;">For {entered_data['Company'].iloc[0]} {entered_data['Model'].iloc[0]}</div>
        </div>
        """, unsafe_allow_html=True)
