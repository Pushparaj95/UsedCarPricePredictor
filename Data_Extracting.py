import pandas as pd
from typing import Dict, List
import json


def flatten_json(json_obj: Dict, column_index: int, prefix: str = '') -> Dict:

    items = {}
    for key, value in json_obj.items():
        new_key = f"{prefix}{key}" if prefix else key

        if isinstance(value, dict):
            items.update(flatten_json(value, f"{new_key}_"))
        elif isinstance(value, list):
            # Special handling for 'top' list containing key-value pairs
            if column_index == 2:
                # Handle 'top' list containing key-value pairs
                if key == 'top' and value and isinstance(value[0], dict):
                    features_values = []
                    for item in value:
                        if 'value' in item:
                            features_values.append(str(item['value']))
                    items['features_values'] = ', '.join(features_values)
                # Handle 'data' list containing sections with lists of key-value pairs
                elif key == 'data' and value and isinstance(value[0], dict):
                    for section in value:
                        if 'heading' in section and 'list' in section:
                            section_name = section['heading']
                            section_values = []
                            for item in section['list']:
                                if isinstance(item, dict) and 'value' in item:
                                    section_values.append(str(item['value']))
                            items[f"{section_name.lower()}_values"] = ', '.join(section_values)
            else:
                if key == 'top' and value and isinstance(value[0], dict):
                    if all(isinstance(item, dict) and 'key' in item and 'value' in item for item in value):
                        for item in value:
                            items[item['key']] = item['value']
                        continue

                # Special handling for 'data' list containing sections with lists of key-value pairs
                elif key == 'data' and value and isinstance(value[0], dict):
                    for section in value:
                        if 'heading' in section and 'list' in section:
                            section_name = section['heading']
                            for item in section['list']:
                                if isinstance(item, dict) and 'key' in item and 'value' in item:
                                    # Use section name as prefix for clarity
                                    items[f"{section_name}_{item['key']}"] = item['value']
                    continue

            # Handle other lists
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    items.update(flatten_json(item, f"{new_key}_{i}_"))
                else:
                    items[f"{new_key}_{i}"] = str(item)
        else:
            items[new_key] = str(value) if value is not None else ''

    return items


def process_json_string(json_str: str) -> Dict:
    """
    Process a JSON string, handling different formats and cleaning
    """
    try:
        # Try direct JSON parsing first
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        try:
            # Try literal eval for cases where single quotes are used
            import ast
            return ast.literal_eval(json_str)
        except (ValueError, SyntaxError, TypeError):
            return {}


def process_excel_files(file_paths: List[str]) -> pd.DataFrame:
    all_data = []
    for file_path in file_paths:
        print(f"Processing {file_path}...")
        df = pd.read_excel(file_path)
        for _, row in df.iterrows():
            flat_row = {'City': file_path.split('/')[-1].split('_')[-2]}  # Add filename as a column
            for col_idx, cell_value in enumerate(row):
                if isinstance(cell_value, str) and cell_value.startswith('{'):
                    json_data = process_json_string(cell_value)
                    if json_data:
                        flat_row.update(flatten_json(json_data, col_idx))
                else:
                    flat_row[f"column_{col_idx}"] = cell_value
            all_data.append(flat_row)

    result_df = pd.DataFrame(all_data)
    result_df.columns = [col.strip().replace(' ', '_') for col in result_df.columns]

    print(
        f"\nProcessing Summary:\nTotal rows processed: {len(result_df)}\nTotal columns created: {len(result_df.columns)}")
    return result_df


def encode_features(df, columns):
    # Initialize a set to collect all unique features
    all_features = set()

    # Convert each specified column to lists of features, stripping extra spaces
    for column in columns:
        df[column] = df[column].apply(
            lambda x: [item.strip() for item in str(x).split(',')] if isinstance(x, str) else [])
        all_features.update(item for sublist in df[column] for item in sublist)

    # Create a DataFrame to hold the binary columns
    binary_columns = []

    # Create a binary column for each unique feature across all specified columns
    for feature in all_features:
        binary_columns.append(
            df[columns].apply(lambda row: any(feature in cell for cell in row), axis=1).rename(feature))

    # Concatenate the original DataFrame with the new binary columns
    df_encoded = pd.concat([df] + binary_columns, axis=1)

    # Optionally, drop the original columns
    df_encoded.drop(columns=columns, inplace=True)

    return df_encoded


excel_files = [
        'Datasets/chennai_cars.xlsx',
        'Datasets/bangalore_cars.xlsx',
        'Datasets/delhi_cars.xlsx',
        'Datasets/hyderabad_cars.xlsx',
        'Datasets/jaipur_cars.xlsx',
        'Datasets/kolkata_cars.xlsx'
        ]

df = process_excel_files(excel_files)

# Grouped features converting it to columns having either True or False
columns_to_encode = ['features_values', 'comfort_&_convenience_values', 'interior_values', 'exterior_values',
                     'safety_values', 'entertainment_&_communication_values']

drop_columns = ['value', 'heading', 'subHeading', 'desc', 'transmission', 'ft', 'Engine_Displacement', 'Engine_and_Transmission_Displacement',
                'km', 'it', 'owner', 'Ownership', 'transmission', 'priceSaving', 'priceFixedText', 'bottomData', 'commonIcon']

df_encoded = encode_features(df, columns_to_encode)
df_encoded = df_encoded.dropna(axis=1, how='all')
df_encoded = df_encoded.loc[:, ~df_encoded.applymap(lambda x: isinstance(x, str) and 'https:' in x).any()]
df_encoded = df_encoded.drop(columns=[column for column in df.columns if column in drop_columns])

df_encoded.to_csv("Datasets/extracted_cars_data.csv", index=False)