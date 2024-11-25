import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import Data_Cleaning as dc
from babel.numbers import format_currency


keys = ['city', 'body_type', 'owner_no', 'company', 'model', 
        'model_year', 'variant_name', 'registration_year', 
        'insurance_validity', 'fuel_type', 'seats', 
        'kms_driven', 'transmission', 'manufacture_year', 
        'engine', 'mileage']

for key in keys:
    if key not in st.session_state:
        st.session_state[key] = None

def display_homepage():
    # Main header
    st.markdown("<h1 style='text-align: center; color: #2E86C1;'>Used CARS Price Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #5D6D7E;'>Get accurate predictions for used car prices in seconds!</h3>", unsafe_allow_html=True)

    # Create a two-column layout for the introduction and app features
    col1, col2 = st.columns(2)

    # with col1:
    #     st.image("https://via.placeholder.com/400x300", caption="Your car valuation partner", use_container_width=True)

    # with col2:
    st.markdown("### About This Application")
    st.markdown("""
    The **Car Price Predictor** is a powerful tool designed to help you estimate the resale price of a used car based on various attributes. 
    Whether you're buying or selling, this app ensures you have accurate pricing data at your fingertips.

    **Key Features:**
    - Input details like City, Ownership History, Company, Model, Variant, and more.
    - Get an instant price prediction for any used car.
    - Smart imputation for missing fields to improve predictions.
    - Easy-to-use interface with step-by-step guidance.
    """)

    # Add a how-to-use section
    st.markdown("## How to Use")
    st.markdown("""
    1. **Enter Mandatory Fields**: Start by filling in the required fields: `City`, `Body Type`, `Ownership Number`, `Company`, `Model`, `Variant`, `Model Year`, `Insurance Validity`, `Registration Year`, and `Kilometers Driven`.
        - These fields are necessary to ensure the most accurate predictions.
    2. **Click Predict**: After filling in the fields, click the **Predict** button.
    3. **View Results**: The application will automatically impute any missing non-mandatory fields and provide you with a predicted car price.
    4. **Error Handling**: If any mandatory field is left blank, the app will alert you to complete the missing details.
    """)

    # Add a visual block for mandatory fields
    st.markdown("### Mandatory Fields")
    st.markdown("""
    To use the app, ensure the following fields are filled in:
    - **City**: Specify the location where the car is being sold.
    - **Body Type**: Enter the Car's Body Type(e.g., SUV, MUV, etc ).
    - **Ownership Number**: Enter the number of previous owners.
    - **Company**: Indicate the car manufacturer (e.g., Toyota, Ford, Hyundai).
    - **Model**: Enter the car's model (e.g., Corolla, Fiesta, i20).
    - **Variant**: Specify the variant (e.g., Petrol, Diesel, EV).
    - **Model Year**: Year of manufacturing.
    - **Registration Year**: Year of registration.
    - **Insurance Validity**: Current Validity of Insurance.
    - **Kilometers Driven**: Total kilometers driven by the car.
    """)

    # Add a visually appealing CTA
    st.markdown("""
    <div style='text-align: center; margin-top: 30px;'>
        <a href="/PricePredictor" style="background-color: #28B463; color: white; padding: 15px 30px; text-decoration: none; border-radius: 5px; font-size: 18px;">
        🚀 Start Predicting Now
        </a>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("© 2024 Car Price Predictor | Built with ❤️ using Streamlit", unsafe_allow_html=True)

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

# Fetch Data
elif option == 'Car Price Predictor':
    st.markdown("<h1 style='text-align: center;'>OLD CAR Price Predictor</h1>", unsafe_allow_html=True)

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
        .custom-button {margin-top: 75px;}
        </style>""", unsafe_allow_html=True)
        # Handle "Predict Price"
        if st.button("Predict Price"):
            # Check for missing mandatory fields
            missing_fields = [ui_label for ui_label in mandatory_fields if not user_inputs[ui_to_df_mapping[ui_label]]]
            if missing_fields:
                st.error(f"Please fill the mandatory fields: {', '.join(missing_fields)}")
            else:
                with st.spinner('Predicting....'):
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
    st.write(Price_str)

    













    

       
