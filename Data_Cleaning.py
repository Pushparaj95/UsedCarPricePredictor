import datetime
import re
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# df = pd.read_csv('extracted_cars_data.csv')

category_mapping = {
    "Gasoline Fuel Injection": [
        "GDi", "MPFi", "PGM - Fi", "Direct Injection", "EFI", "Electronic Fuel Injection",
        "Direct Injection", "Indirect Injection", "MPFI", "SEFI", "Multi-point injection", "Direct Injectio",
        "PGM-FI (Programmed Fuel Injection)", "Gasoline Direct Injection", "Electronic Fuel Injection(EFI)",
        "MFI", "EFI (Electronic Fuel Injection)", "TFSI", "Multi Point Fuel Injection", "MPFI+LPG",
        "PFI", "Dual VVT-i", "SMPI", "Direct Fuel Injection", "Direct Injection Common Rail", "DPFi", "VVT-iE",
        "EFI(Electronic Fuel Injection)", "PGM-FI (Programmed Fuel Inje", "EFI (Electronic Fuel Injection",
        "PGM-FI (Programmed Fuel Inject", "TGDi", "TSI"
    ],
    "Diesel Fuel Injection": [
        "CRDI", "Common Rail", "Common Rail Direct Injection (dCi)", "TDCi", "Common Rail", "CDI",
        "Direct Injection Common Rail", "Common Rail Diesel", "Common Rail Injection", "IDI", "TDi",
        "Advanced Common Rail", "Common-Rail type", "TDi", "DEDST", "DDIS", "common rail system",
        "Common rail direct injection"
    ],
    "Hybrid and Alternative Fuel Injection": [
        "MPFI+CNG", "CNG", "Intelligent-Gas Port Injection", "Direct Injection Common Rail (Dedicated)",
        "ISG", "Electric", "3 Phase AC Induction Motors"
    ],
    "General Electronic Fuel Injection": [
        "EFI", "EFIC", "Electronic Injection System", "Electronic Fuel Injection"
    ],
    "Specific and Specialized Injection Technologies": [
        "MPI", "EGIS", "Single Point Fuel Injection (SPFI)", "3 Phase AC Induction Motors",
        "Distribution Type Fuel Injection", "DOHC", "Multipoint Injection"
    ],
    "Brand-Specific Fuel Injection": [
        "PGM-FI", "TFSI", "Ti-VCT", "D-4S", "VVT-i", "i-VTEC", "ISG"
    ]
}

steering_type_map = {
    "Electronic": ["electric", "electronic", "electrical"]
}

brake_type_map = {
    "Disc": ["Disc", "Disc & Caliper Type", "Solid Disc", "Multilateral Disc", "Disk", "Disc, 236 mm", "Discs",
             "Disc brakes"],
    "Ventilated Disc": ["Ventilated Disc", "Booster assisted ventilated disc", "Ventilated Discs", "Ventilated Disk",
                        "Vantilated Disc", "Ventlated Disc", "Ventillated Disc", "Caliper Ventilated Disc",
                        "Disc,internally ventilated", "Ventialte Disc"],
    "Drum": ["Drum", "Booster assisted drum", "Leading & Trailing Drum", "Drums 180 mm", "Drums", "Ventilated Drum",
             "228.6 mm dia, drums on rear wheels", "Drum`"],
    "Disc & Drum": ["Disc & Drum", "Drum in disc", "Drum in Discs"],
    "Self Adjusting Drum": ["Self-Adjusting Drum", "Self Adjusting Drum", "self adjusting drums"]
}

tyre_type_map = {
    "Tubeless, Radial": ["Tubeless, Radial", "Tubeless,Radial", "Tubeless Radial Tyres", "Tubeless,Radials",
                         "Radial,Tubeless", "Radial, Tubeless", "Tubless, Radial", "Tubeless Tyres, Radial",
                         "Radial, Tubless", "Radial Tyres", "Tubless,Radial", "Radial Tubeless",
                         "Tubeless Radials Tyre",
                         "Radial", "Tubeless, Radials"],
    "Tubeless Tyres": ["Tubeless Tyres", "Tubeless", "tubeless tyre", ],
    "Runflat Tyres": ["Runflat Tyres", "Run-Flat", "Runflat Tyre", "Tubeless,Runflat", "Runflat", "Tubeless. Runflat",
                      "Runflat,Radial", "Tubeless, Runflat"]
}

drive_type_map = {
    "FWD": [
        "FWD", "Front Wheel Drive"
    ],
    "RWD": [
        "RWD", "Rear Wheel Drive", "RWD(with MTT)", "Rear Wheel Drive with ESP"
    ],
    "AWD": [
        "AWD", "4WD", "4X4", "All Wheel Drive", "AWD INTEGRATED MANAGEMENT",
        "Permanent all-wheel drive quattro", "4 WD", "4x2"
    ],
    "2WD": [
        "2WD", "Two Wheel Drive", "2 WD", "2WD "
    ]
}

grouping_keywords = [
                    "ZDI", "5 Str - 5 Seater", "525d", "530i", "740Li", "A 200", "A Option - A, A Petrol - A",
                     "A180", "AC Uniq - AC", "Adventure", "Alpha", "Ambiente", "AMT", "Asta", "Aura", "AX", "B180",
                     "Base", "C 220",
                     "Climber_IC", "CNG", "Cooper", "CRDe", "Creative", "CRDi", "D3", "D4", "D5", "D75", "Delta", "DI",
                     "Dicor_IG",
                     "E i-", "E200", "E250", "E350", "Elegance", "Emotion", "Era", "Exclusive", "Feel Dual", "G80 ",
                     "GLS", "Green ",
                     "H2 ", "Highline", "HTX", "S i-Dtec_IC", "i-DTEC", "i-VTEC ", "LDi Option", "LXI", "Magna",
                     "MULTIJET", "N8", "Platinum",
                     "Progressive", "Puretech", "Quadrajet", "Revotron 1.2", "RXT", "RXZ", "S i-Vtec_IC", "S10", "S11",
                     "S 11 - S11", "Savvy", "sDrive", "Sharp", "Signature", "SLX", "Smart ", "Sports", "Sportz",
                     "Style", "Technology",
                     "Titanium", "Topline", "Trend", "Trendline", "Urban", "VDI", "VXI", "W11", "W4", "W6", "W7", "W8",
                     "W9", "xDrive",
                     "X-Line", "XM", "XO", "XZA", "XZ", "zdi_IC", "Zeta", "ZXI_IC", "4x2_IC", "4x4_IC"
                     ]

gearbox_type_map = {
    "Automatic Transmissions": [
        "CVT", "IVT", "E-CVT", "Fully Automatic", "Single-speed transmission", "Single speed reduction gear",
        "Single Speed", "9-speed automatic", "10-speed Automatic", "6-speed automatic", "9G-TRONIC automatic",
        "Mercedes Benz 7 Speed Automatic", "6-Speed Automatic Transmission", "Six Speed Automatic Gearbox",
        "9G TRONIC", "8-Speed Automatic Transmission", "8-Speed Automatic Transmission", "6-speed automatic",
        "9-Speed Automatic", "8-Speed Tiptronic", "8-Speed Steptronic", "8 Speed CVT",
        "Six Speed Automatic Transmission", "6 speed automatic", "Six Speed Geartronic, Six Speed Automati", "eCVT",
        "Direct Drive"
    ],
    "Dual Clutch Transmission": [
        "7-Speed DCT", "7G DCT 7-Speed Dual Clutch Transmission", "7-Speed DSG", "7 Speed DCT",
        "7-Speed DCT Steptronic",
        "AMG SPEEDSHIFT DCT 8G", "AMG 7-SPEED DCT", "7-speed Stronic", "7-Speed S-Tronic", "7-speed PDK",
        "7-Speed DCT Steptronic", "7G-DCT", "7 Speed 7G-DCT", "7-speed DSG", "7-speed Stronic",
        "7-Speed S Tronic", "7 Speed DSG"
    ],
    "Manual Transmission": [
        "Five Speed", "Five Speed Manual Transmission", "Five Speed Manual Transmission Gearbox",
        "Six Speed Manual", "Six Speed Manual with Paddle Shifter", "Five Speed Manual",
        "Five Speed Manual (Cable Type Gear Shift)", "Five Speed Forward, 1 Reverse", "Six Speed Gearbox",
        "Six Speed MT", "Six Speed iMT", "Six Speed Manual Transmission", "Six-speed DCT", "Six-speed Geartronic",
        "Six Speed Automatic Gearbox", "Six Speed Gearbox", "Five Speed Manual", "5 Speed", "6 Speed", "8 Speed",
        "9 Speed", "4 Speed", "7 Speed", "7-Speed", "5-Speed", "6-Speed", "8-Speed", "8", "6", "5", "5-Speed`",
        "6Speed", "6 Speed MT", "8Speed", "5 Speed Manual (Cable Type Gear Shift)", "Six Speed  Gearbox",
        "5 Speed Forward, 1 Reverse", "5 speed manual", "9 -speed", "4-Speed"
    ],
    "Semi Automatic Transmission": [
        "6 Speed iMT", "6-Speed iMT", "iMT", "6-speed CVT", "6-Speed AT", "6 Speed with Sequential Shift",
        "6-speed IVT", "8 Speed Sport", "8 Speed Tiptronic", "7-Speed Steptronic", "7-Speed S Tronic",
        "6-Speed Steptronic", "6-Speed Steptronic Sport Automatic Transmission", "6-speed autoSHIFT", "6-Speed AT",
        "6-speed DCT", "AGS", "6 Speed AT", "6 Speed IVT", "7 Speed Steptronic Sport", "8-speed Steptronic Automatic",
        "7 Speed S tronic"
    ],
    "Continuously Variable Transmission": [
        "5 Speed CVT", "8-Speed CVT", "7 Speed CVT", "6-speed CVT"
    ],
    "Other/Hybrid Transmissions": [
        "9G-TRONIC", "SPEEDSHIFT TCT 9G", "9 speed Tronic", "9-speed automatic", "7-speed PDK",
        "7-Speed DCT", "8-Speed DCT", "7-speed Stronic", "9G TRONIC", "9-speed automatic", "9-Speed",
        "8-Speed Steptronic", "8 Speed Steptronic", "AMG 7-Speed DCT", "8G-DCT", "9-speed automatic",
        "7-Speed S Tronic", "7-Speed Steptronic Sport", "9-Speed", "9-Speed AT", "8-Speed Steptronic Sport "
                                                                                 "Automatic Transmission",
        "10-speed automatic", "10 speed", "9-speed automatic", "SPEEDSHIFT TCT 9G", "9 speed Tronic", "5 Speed AT+ "
                                                                                                      "Paddle Shifters",

    ]
}

engine_config_map = {
    "DOHC": ["DOHC", "DOHC with VIS", "DOHC with VGT", "16-valve DOHC layout", "DOHC with TIS"],
    "SOHC": ["SOHC"],
    "VTEC": ["VTEC", "iDSI"],
    "Battery Modules": ["16 Modules 48 Cells", "23 Modules 69 Cells"],
    "Other": ["undefined"]
}

insurance_validity_map = {
    "Active": ['1'],
    "Not Available": ['2', "Not Available"],
    "Third Party": ["Third Party", "Third Party insurance"]
}

turbo_charger_map = {
    "Yes": ["Yes", "Twin", "Turbo"],
    "No": ["No"],
}

def clean_month_year(value):
    if pd.isna(value) or value.strip() == '':
        return None

    # Handle year-only values (e.g., "2022") - Add "01" for January
    if len(value) == 4 and value.isdigit():
        return None, value  # Return a tuple for month and year

    value_str = str(value).strip()
    month_year_pattern = r'([A-Za-z]+)[\s-]*(\d{4})'
    match = re.match(month_year_pattern, value_str)
    if match:
        month_str = match.group(1)
        year_str = match.group(2)
        month = pd.Timestamp(month_str + ' 1, 2000').month  # Get month number from month name
        return f"{month:02d}", year_str  # Return a tuple for month and year

    return None


def convert_price(value):
    if pd.isna(value) or value == '':
        return None

    cleaned_value = re.sub(r'[^\d.]', '', value)

    try:
        if 'Lakh' in value:
            return float(cleaned_value) * 1e5
        elif 'Crore' in value:
            return float(cleaned_value) * 1e7
        else:
            return float(cleaned_value)
    except ValueError:
        return None


def extract_first_two_words(variant):
    # Regex to capture first two words
    match = re.match(r'^(\S+\s+\S+)', variant)
    return match.group(0) if match else variant


def remove_regex_pattern(value, pattern=r'\bBS\w*'):
    return re.sub(pattern, '', value)


def standardize_variants(variant, grouping_keywords):
    if pd.isna(variant):
        return variant

    original_variant = str(variant).strip()

    # 1. Process hyphenated replacements with exact word matching
    for keyword in grouping_keywords:
        if " - " in keyword:
            # Handle multiple replacements in one keyword (e.g., "A Option - A, A Petrol - A")
            replacements = keyword.split(", ")
            for replacement in replacements:
                source, target = replacement.split(" - ", 1)
                source = source.strip()
                # Use word boundaries for exact match
                if re.search(rf'\b{re.escape(source)}\b', original_variant, re.IGNORECASE):
                    return target.strip()

    # 2. Process IC cases with exact word matching
    for keyword in grouping_keywords:
        if "_IC" in keyword:
            base_keyword = keyword.replace("_IC", "").strip()
            if re.search(rf'\b{re.escape(base_keyword)}\b', original_variant, re.IGNORECASE):
                return base_keyword

    # 3. Process remaining keywords with exact word matching
    remaining_keywords = [k for k in grouping_keywords if " - " not in k and "_IC" not in k]
    remaining_keywords.sort(key=len, reverse=True)

    for keyword in remaining_keywords:
        keyword = keyword.strip()
        # Use word boundaries for exact match
        if re.search(rf'\b{re.escape(keyword)}\b', original_variant, re.IGNORECASE):
            return keyword

    return original_variant


def extract_numbers(text, target_dtype):
    match = re.search(r'[\d,]+(?:\.\d+)?', str(text))
    if match:
        number = match.group().replace(',', '')
        if target_dtype == int and number.isdigit():
            return int(number)
        return float(number)
    return None


def map_to_category(term):
    term = term.strip().lower() if isinstance(term, str) else term

    for category, terms in category_mapping.items():
        # Check if the lowercase term matches any term in the category list
        if any(term == t.lower() for t in terms):
            return category


def fill_with_key(value, group_map):
    m_value = value.strip().lower() if isinstance(value, str) else value
    for key, values in group_map.items():
        if m_value in (v.lower() for v in values):
            return key
    return value


def extract_max_weight(weight):
    if not isinstance(weight, str):
        return None
    if weight == 'Kerb Weight':
        return None
    weight = weight.lower()
    weight = weight.replace('kg', '').replace('kgs', '').replace(',', '').strip()
    weight = ''.join(c if c.isdigit() or c == '-' else ' ' for c in weight) 
    parts = weight.split('-')

    weights = [int(part.strip()) for part in parts if part.strip().isdigit()]

    if weights:
        return max(weights)
    else:
        return None

    # 'Kerb_Weight' - need to separate weight 1222- 12211


def extract_max_capacity(capacity):
    if not isinstance(capacity, str):
        return None
    match = re.match(r'^(\d+)(?:\s*-\s*(\d+))?', capacity)

    if match:
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start

        return max(start, end)
    else:
        return None


def convert_columns(df):
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            # If the column is numeric (int or float)
            if df[column].dropna().apply(lambda x: x.is_integer()).all():
                # If all values are integers (e.g., 2019.0), convert to int
                df[column] = df[column].astype('Int64')  # Use 'Int64' for nullable integers
            else:
                # Else, keep as float (or convert to float if needed)
                df[column] = df[column].astype(float)
        elif pd.api.types.is_object_dtype(df[column]):
            # If the column is of object type, check if it contains only digits (potential int)
            if df[column].str.isnumeric().all():
                df[column] = df[column].astype('int')
            # else leave as object or apply other transformations for strings
    return df


def clean_and_grouping_columns_with_extracted_keywords(df):
    df['extracted_grouped_variant'] = df['variantName'].apply(extract_first_two_words)
    df['ownerNo'] = df['ownerNo'].replace(0, 1)
    df['grouped_variant'] = df.apply(lambda row: row['extracted_grouped_variant'] if row['variantName'] == row[
        'extracted_grouped_variant'] else row['extracted_grouped_variant'], axis=1)

    df['grouped_variant'] = df['extracted_grouped_variant'].apply(lambda x: remove_regex_pattern(x))

    # extracted_grouped_variant, grouped_variant
    df['variantName'] = df['grouped_variant'].apply(
        lambda x: standardize_variants(x, grouping_keywords)
    )

    columns_to_drop = [
    'extracted_grouped_variant', 'grouped_variant',
    ]
    df = df.drop(columns=columns_to_drop)

    # Extracting BHP, Min RPM and Max RPM
    # df[['Power', 'power_rpm_min', 'power_rpm_max']] = pd.DataFrame(
    #     df['Max_Power'].apply(extract_power_details).tolist(),
    #     index=df.index)

    # df[['Torque','torque_rpm_min', 'torque_rpm_max']] = pd.DataFrame(
    #     df['Max_Torque'].apply(extract_torque_details).tolist(),
    #     index=df.index)

    # df = calculate_ranges(df)
    # df = fill_rpm_ranges(df)

    df['Fuel_Suppy_System'] = df['Fuel_Suppy_System'].apply(map_to_category)

    df['Steering_Type'] = df['Steering_Type'].apply(fill_with_key, args=(steering_type_map,))
    df['Front_Brake_Type'] = df['Front_Brake_Type'].apply(
        fill_with_key, args=(brake_type_map,))
    df['Rear_Brake_Type'] = df['Rear_Brake_Type'].apply(
        fill_with_key, args=(brake_type_map,))
    df['Tyre_Type'] = df['Tyre_Type'].apply(fill_with_key, args=(tyre_type_map,))
    df['Drive_Type'] = df['Drive_Type'].apply(fill_with_key, args=(drive_type_map,))
    df['Gear_Box'] = df['Gear_Box'].apply(fill_with_key, args=(gearbox_type_map,))
    df['Turbo_Charger'] = df['Turbo_Charger'].apply(fill_with_key, args=(turbo_charger_map,))
    df['Super_Charger'] = df['Super_Charger'].apply(fill_with_key, args=(turbo_charger_map,))
    df['Gross_Weight'] = (
        df['Gross_Weight'].apply(extract_max_weight))
    df['Kerb_Weight'] = (
        df['Kerb_Weight'].apply(extract_max_weight))
    df['Cargo_Volumn'] = df['Cargo_Volumn'].apply(extract_max_capacity)
    df['Value_Configuration'] = (
        df['Value_Configuration'].apply(fill_with_key, args=(engine_config_map,)))


    df['Insurance_Validity'] = df['Insurance_Validity'].apply(fill_with_key, args=(insurance_validity_map,))
    
    return df


def impute_missing_values(df, target_column, feature_columns, model_type="classification"):
    """
    Fills missing values in the target_column based on related feature_columns.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - target_column (str): The column with missing values to fill.
    - feature_columns (list): List of columns to use as predictors for filling missing values.
    - model_type (str): Type of model to use, "classification" for discrete values (e.g., month)
                        or "regression" for continuous values (e.g., years).

    Returns:
    - DataFrame with missing values in target_column filled.
    """
    # Separate data into rows with and without missing target_column values
    known_data = df.dropna(subset=[target_column])
    unknown_data = df[df[target_column].isnull()]

    if known_data.empty or unknown_data.empty:
        print(f"No missing values in {target_column} or no non-missing data for training.")
        return df

    # Set up predictors and target variable
    X = known_data[feature_columns]
    y = known_data[target_column]

    # Train/test split for model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choose model based on type
    if model_type == "classification":
        model = RandomForestClassifier(random_state=42)
    elif model_type == "regression":
        model = RandomForestRegressor(random_state=42)
    else:
        raise ValueError("Invalid model type. Choose 'classification' or 'regression'.")

    # Train the model
    model.fit(X_train, y_train)

    # Optional: Model Evaluation
    if model_type == "classification":
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy for {target_column} imputation: {accuracy:.2f}")
    else:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r_squr = r2_score(y_test,y_pred)
        print(f"R2 for {target_column} imputation: {r_squr:.2f}")

    # Impute missing values in the target column
    df.loc[unknown_data.index, target_column] = model.predict(unknown_data[feature_columns])

    return df


def knn_impute_missing_values(df, target_column, feature_columns, n_neighbors=5):
    """
    Fills missing values in the target_column using KNN imputation based on related feature_columns.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - target_column (str): The column with missing values to fill.
    - feature_columns (list): List of columns to use as predictors for filling missing values.
    - n_neighbors (int): Number of neighbors to consider for imputation.

    Returns:
    - DataFrame with missing values in target_column filled.
    """
    # Select the columns needed for imputation
    impute_columns = feature_columns + [target_column]
    
    # Subset the dataframe to include only the necessary columns
    impute_data = df[impute_columns]
    
    # Initialize KNNImputer
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    
    # Perform KNN imputation
    imputed_data = knn_imputer.fit_transform(impute_data)
    
    # Update the target_column in the original DataFrame
    df[target_column] = imputed_data[:, -1]
    
    return df


def adjust_year_prediction(row):
    if row['modelYear'] < 2000 < row['Registration_Year']:
        return row['modelYear']
    return row['Registration_Year']


def label_encoder(df, column):
    le = LabelEncoder()
    non_null_values = df[column].dropna()
    # Fit and transform only the non-null values
    df.loc[non_null_values.index, column] = le.fit_transform(non_null_values)
    return df


    te = TargetEncoder(smoothing=smoothing, min_samples_leaf=min_samples_leaf)
    df[column] = te.fit_transform(df[column], df[target])
    return df

def ordinal_encoder(df, column):
    oe = OrdinalEncoder()
    df[column] = oe.fit_transform(df[[column]])
    return df


def process_and_impute_missing_values(original_df, new_df):
    """
    Merges the original and new DataFrame, processes missing values, encodes columns, 
    imputes missing values, and restores original data types.

    Parameters:
    - original_df: The original DataFrame.
    - new_df: The new DataFrame to be merged with the original.

    Returns:
    - A processed and merged DataFrame with original data types restored.
    """
    # Step 1: Merge the DataFrames
    merged_df = pd.concat([original_df, new_df], ignore_index=True)
    
    # Step 2: Store the original data types
    original_dtypes = original_df.dtypes.to_dict()

    # Step 3: Identify columns with missing values
    missing_df = pd.DataFrame(merged_df.isna().sum()).reset_index()
    missing_df.columns = ['Column', 'Missing Values']
    missing_df = missing_df[missing_df['Missing Values'] > 0]
    missing_columns = missing_df['Column'].tolist()

    # Step 4: Create encoded and imputed copies
    df_encoded = merged_df.copy()
    
    # Identify columns to encode
    columns_to_encode = [column for column in merged_df.columns if column not in missing_columns and column not in ['Engine', 'Mileage', 'Seats']]

    # Step 5: Encode columns
    for column in columns_to_encode:
        df_encoded = label_encoder(df_encoded, column)

    # Step 6: Add derived columns
    current_year = pd.Timestamp.now().year
    merged_df['Age_Old_In_Year'] = current_year - merged_df['Registration_Year']
    merged_df['Price'] = merged_df['Price'].fillna(0)

    # Step 7: Impute missing values
    ignore_columns = ['Age_Old_In_Year', 'Price']
    for column in missing_columns:
        if column not in ignore_columns:
            if column in ['Fuel_Type', 'Transmission']:
                df_encoded = impute_missing_values( df_encoded, target_column=column, feature_columns=['Body_Type', 'Company', 'Model', 'Variantname'],
                    model_type='classification'
                )
            else:
                df_encoded = impute_missing_values( df_encoded, target_column=column, feature_columns=['Body_Type', 'Company', 'Model', 'Variantname'],
                    model_type='regression'
                )

    # Step 8: Combine imputed columns back to merged_df
    for column in merged_df.columns:
        if column in df_encoded.columns:
            merged_df[column] = merged_df[column].combine_first(df_encoded[column])
    
    # Step 9: Restore original data types
    for column, dtype in original_dtypes.items():
        if column in merged_df.columns:
            merged_df[column] = merged_df[column].astype(dtype)

    return merged_df





