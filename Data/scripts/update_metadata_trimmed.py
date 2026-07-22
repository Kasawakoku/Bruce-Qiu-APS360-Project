# updates the metadata csv to use up to date variant and airline classes
# after referencing mapping and hierarchy
# after trimming out classes with low number of images
# outputting metadata with invalid airlines trimmed, with invalid variants trimmed
# as well as the AND, OR and XOR of these sets

import pandas as pd
import yaml
import os
import sys

# --- Configuration ---
# Default configurations
DEFAULT_AIRLINE_THRESHOLD = 90
DEFAULT_VARIANT_THRESHOLD = 90
DEFAULT_INPUT_CSV = "../metadata/airliners_metadata.csv"
DEFAULT_AIRLINE_YAML = "../class_definitions/airline_mapping.yaml"
DEFAULT_AIRCRAFT_YAML = "../class_definitions/aircraft_hierarchy.yaml"
# case insensitive
DEFAULT_EXCLUDE_LABELS = ['unknown', 'untitled']

# Positional terminal arguments:
# python update_metadata_trimmed.py [airline_threshold] [variant_threshold] [input_csv] [airline_yaml] [aircraft_yaml] [exclude_labels]

AIRLINE_THRESHOLD = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_AIRLINE_THRESHOLD
VARIANT_THRESHOLD = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_VARIANT_THRESHOLD
INPUT_CSV = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_INPUT_CSV
AIRLINE_YAML = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_AIRLINE_YAML
AIRCRAFT_YAML = sys.argv[5] if len(sys.argv) > 5 else DEFAULT_AIRCRAFT_YAML

# Parse comma-separated exclude labels from terminal
EXCLUDE_LABELS = [
    label.strip() for label in sys.argv[6].split(",")
] if len(sys.argv) > 6 else DEFAULT_EXCLUDE_LABELS

def load_mappings():
    # Identical mapping logic to Script 1
    with open(AIRLINE_YAML, 'r', encoding='utf-8') as f:
        airline_raw = yaml.safe_load(f)
    
    airline_lookup = {}
    for true_airline, aliases in airline_raw.items():
        airline_lookup[true_airline] = true_airline
        for alias in aliases:
            airline_lookup[alias] = true_airline

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

def generate_trimmed_datasets():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Could not find {INPUT_CSV}")
        return

    print(f"Loading raw dataset from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    airline_lookup, model_to_variant, variant_to_family, family_to_manufacturer = load_mappings()

    # 1. Update Airline
    df['airline'] = df['airline'].map(airline_lookup).fillna(df['airline'])

    # 2. Create hierarchical columns based on 'aircraft_model'
    df['aircraft_variant'] = df['aircraft_model'].map(model_to_variant).fillna(df['aircraft_model'])
    df['aircraft_family'] = df['aircraft_variant'].map(variant_to_family).fillna("Unknown")
    df['aircraft_manufacturer'] = df['aircraft_family'].map(family_to_manufacturer).fillna("Unknown")

    # Reorder columns
    cols = list(df.columns)
    if all(c in cols for c in ['aircraft_variant', 'aircraft_family', 'aircraft_manufacturer', 'aircraft_model']):
        for c in ['aircraft_variant', 'aircraft_family', 'aircraft_manufacturer']: cols.remove(c)
        model_idx = cols.index('aircraft_model')
        cols.insert(model_idx + 1, 'aircraft_variant')
        cols.insert(model_idx + 2, 'aircraft_family')
        cols.insert(model_idx + 3, 'aircraft_manufacturer')
        df = df[cols]

    # --- IDENTIFY VALID CLASSES ---
    air_counts = df['airline'].value_counts()
    valid_airlines = set([air for air, count in air_counts.items() 
                          if count >= AIRLINE_THRESHOLD and str(air).lower() not in EXCLUDE_LABELS])
    
    var_counts = df['aircraft_variant'].value_counts()
    valid_variants = set([var for var, count in var_counts.items() 
                          if count >= VARIANT_THRESHOLD and str(var).lower() not in EXCLUDE_LABELS])

    # --- CREATE FILTERED DATAFRAMES ---
    mask_airline = df['airline'].isin(valid_airlines)
    mask_variant = df['aircraft_variant'].isin(valid_variants)

    datasets = {
        '../metadata/airliners_metadata_trimmed_airline_only.csv': df[mask_airline],
        '../metadata/airliners_metadata_trimmed_variant_only.csv': df[mask_variant],
        '../metadata/airliners_metadata_trimmed_intersection.csv': df[mask_airline & mask_variant], # BOTH are valid
        '../metadata/airliners_metadata_trimmed_union.csv': df[mask_airline | mask_variant],         # EITHER is valid
        '../metadata/airliners_metadata_trimmed_symmetric_diff.csv': df[mask_airline ^ mask_variant] # Only ONE is valid. ie. Union - Intersection
    }

    print("\n" + "="*50)
    print("TRIMMED DATASET REPORT")
    print("="*50)
    print(f"Original dataset total images: {len(df)}")
    
    for filename, subset_df in datasets.items():
        subset_df.to_csv(filename, index=False, encoding='utf-8')
        
        # Calculate remaining valid classes inside this specific subset
        rem_airlines = subset_df[subset_df['airline'].isin(valid_airlines)]['airline'].nunique()
        rem_variants = subset_df[subset_df['aircraft_variant'].isin(valid_variants)]['aircraft_variant'].nunique()
        
        print(f"\nSaved: {filename}")
        print(f"  -> Total Images Remaining: {len(subset_df)}")
        print(f"  -> Valid Airlines Remaining: {rem_airlines}")
        print(f"  -> Valid Variants Remaining: {rem_variants}")

    print("="*50)

if __name__ == "__main__":
    generate_trimmed_datasets()