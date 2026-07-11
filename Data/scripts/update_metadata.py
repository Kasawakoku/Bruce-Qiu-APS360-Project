# updates the metadata csv to use up to date variant and airline classes
# after referencing mapping and hierarchy

import pandas as pd
import yaml
import os

# --- Configuration ---
INPUT_CSV = "../metadata/airliners_metadata.csv"
OUTPUT_CSV = "../metadata/airliners_metadata_updated.csv"
AIRLINE_YAML = "../class_definitions/airline_mapping.yaml"
AIRCRAFT_YAML = "../class_definitions/aircraft_hierarchy.yaml"

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

    # ROBUST LOGIC: Handles missing colons and empty lists gracefully
    for mfg, families_list in aircraft_raw.items():
        if not families_list: continue
        
        for family_item in families_list:
            # If the family is missing a colon, treat it as a plain string
            if isinstance(family_item, str):
                family_to_manufacturer[family_item] = mfg
                
            elif isinstance(family_item, dict):
                for family, variants_list in family_item.items():
                    family_to_manufacturer[family] = mfg
                    if not variants_list: continue
                    
                    for variant_item in variants_list:
                        # If the variant has no colon/children, it parses as a string
                        if isinstance(variant_item, str):
                            variant = variant_item
                            variant_to_family[variant] = family
                            model_to_variant[variant] = variant
                            
                        # If the variant has a colon/children, it parses as a dictionary
                        elif isinstance(variant_item, dict):
                            for variant, models in variant_item.items():
                                variant_to_family[variant] = family
                                model_to_variant[variant] = variant # Map variant to itself
                                
                                if models and isinstance(models, list):
                                    for model in models:
                                        model_to_variant[model] = variant
                        
    return airline_lookup, model_to_variant, variant_to_family, family_to_manufacturer

def update_dataset():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Could not find {INPUT_CSV}")
        return

    print(f"Loading raw dataset from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    print("Applying YAML taxonomies...")
    airline_lookup, model_to_variant, variant_to_family, family_to_manufacturer = load_mappings()

    
    # 1. Update Airline Alias to Merged Airline
    df['airline'] = df['airline'].map(airline_lookup).fillna(df['airline'])

    # 2. Read from 'aircraft_model' and CREATE the new 'aircraft_variant' column
    df['aircraft_variant'] = df['aircraft_model'].map(model_to_variant).fillna(df['aircraft_model'])

    # 3. Add Family and Manufacturer columns based on the newly created variant
    df['aircraft_family'] = df['aircraft_variant'].map(variant_to_family).fillna("Unknown")
    df['aircraft_manufacturer'] = df['aircraft_family'].map(family_to_manufacturer).fillna("Unknown")

    # Reorder columns so the hierarchy sits neatly next to the original model
    cols = list(df.columns)
    if 'aircraft_variant' in cols and 'aircraft_family' in cols and 'aircraft_manufacturer' in cols:
        cols.remove('aircraft_variant')
        cols.remove('aircraft_family')
        cols.remove('aircraft_manufacturer')
        
        # Find where the original model column is
        model_idx = cols.index('aircraft_model')
        
        # Insert the new hierarchical columns immediately after it
        cols.insert(model_idx + 1, 'aircraft_variant')
        cols.insert(model_idx + 2, 'aircraft_family')
        cols.insert(model_idx + 3, 'aircraft_manufacturer')
        
        df = df[cols]

    # Save to the NEW file so the old one is untouched
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"\nSuccess! Cleaned dataset saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    update_dataset()