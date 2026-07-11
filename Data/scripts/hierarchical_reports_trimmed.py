# outputs list of all manufacturers, families, variants, models, airlines, and how many images per each
# after trimming out classes with low number of images

import pandas as pd
import yaml
import os

# --- Configuration ---
INPUT_CSV = "../metadata/airliners_metadata.csv"
AIRLINE_YAML = "../class_definitions/airline_mapping.yaml"
AIRCRAFT_YAML = "../class_definitions/aircraft_hierarchy.yaml"

AIRLINE_THRESHOLD = 90
VARIANT_THRESHOLD = 90

# Exclude list (case-insensitive)
EXCLUDE_LABELS = ['unknown', 'untitled']

def load_mappings():
    # 1. Load Airline Mappings
    with open(AIRLINE_YAML, 'r', encoding='utf-8') as f:
        airline_raw = yaml.safe_load(f)
    
    airline_lookup = {}
    for true_airline, aliases in airline_raw.items():
        airline_lookup[true_airline] = true_airline
        for alias in aliases:
            airline_lookup[alias] = true_airline

    # 2. Load Aircraft Mappings (Using robust list/dict logic)
    with open(AIRCRAFT_YAML, 'r', encoding='utf-8') as f:
        aircraft_raw = yaml.safe_load(f)
        
    model_to_variant = {}
    variant_to_family = {}
    family_to_manufacturer = {}

    for mfg, families_list in aircraft_raw.items():
        if not families_list: continue
        for family_item in families_list:
            if isinstance(family_item, str):
                family_to_manufacturer[family_item] = mfg
            elif isinstance(family_item, dict):
                for family, variants_list in family_item.items():
                    family_to_manufacturer[family] = mfg
                    if not variants_list: continue
                    for variant_item in variants_list:
                        if isinstance(variant_item, str):
                            variant = variant_item
                            variant_to_family[variant] = family
                            model_to_variant[variant] = variant
                        elif isinstance(variant_item, dict):
                            for variant, models in variant_item.items():
                                variant_to_family[variant] = family
                                model_to_variant[variant] = variant
                                if models and isinstance(models, list):
                                    for model in models:
                                        model_to_variant[model] = variant
                                        
    return airline_lookup, model_to_variant, variant_to_family, family_to_manufacturer

def generate_trimmed_reports():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Could not find {INPUT_CSV}")
        return

    print("Loading data and applying mappings...")
    df = pd.read_csv(INPUT_CSV)
    airline_lookup, model_to_variant, variant_to_family, family_to_manufacturer = load_mappings()

    # Apply mappings
    df['merged_airline'] = df['airline'].map(airline_lookup).fillna(df['airline'])
    df['variant'] = df['aircraft_model'].map(model_to_variant).fillna(df['aircraft_model'])
    df['family'] = df['variant'].map(variant_to_family).fillna("Unknown")
    df['manufacturer'] = df['family'].map(family_to_manufacturer).fillna("Unknown")

    # --- IDENTIFY VALID CLASSES ---
    # 1. Airlines
    air_counts = df['merged_airline'].value_counts()
    valid_airlines = [air for air, count in air_counts.items() 
                      if count >= AIRLINE_THRESHOLD and str(air).lower() not in EXCLUDE_LABELS]
    
    # 2. Variants
    var_counts = df['variant'].value_counts()
    valid_variants = [var for var, count in var_counts.items() 
                      if count >= VARIANT_THRESHOLD and str(var).lower() not in EXCLUDE_LABELS]

    # --- FILTER DATAFRAMES ---
    df_valid_air = df[df['merged_airline'].isin(valid_airlines)]
    df_valid_var = df[df['variant'].isin(valid_variants)]

    print(f"\nFound {len(valid_airlines)} airlines with >= {AIRLINE_THRESHOLD} images.")
    print(f"Found {len(valid_variants)} variants with >= {VARIANT_THRESHOLD} images.\n")

    # --- EXPORT REPORTS ---
    # Airline Export
    air_export = df_valid_air['merged_airline'].value_counts().reset_index()
    air_export.columns = ['Airline', 'Image Count']
    air_export.sort_values(by='Airline').to_csv('counts_airlines_merged_trimmed.csv', index=False)
    print(f"[*] Saved counts_airlines_merged_trimmed.csv")

    # Variant & Hierarchy Exports (Based on valid variants)
    hierarchy_exports = {
        'counts_variants_trimmed.csv': ('variant', 'Variant'),
        'counts_models_trimmed.csv': ('aircraft_model', 'Model'),
        'counts_families_trimmed.csv': ('family', 'Family'),
        'counts_manufacturers_trimmed.csv': ('manufacturer', 'Manufacturer')
    }

    for filename, (col_name, label) in hierarchy_exports.items():
        counts = df_valid_var[col_name].value_counts().reset_index()
        counts.columns = [label, 'Image Count']
        counts.sort_values(by=label).to_csv(filename, index=False)
        print(f"[*] Saved {filename}")

if __name__ == "__main__":
    generate_trimmed_reports()