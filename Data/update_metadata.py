import pandas as pd
import yaml
import os

# --- Configuration ---
INPUT_CSV = "airliners_metadata.csv"
OUTPUT_CSV = "airliners_metadata_updated.csv"
AIRLINE_YAML = "airline_mapping.yaml"
AIRCRAFT_YAML = "aircraft_hierarchy.yaml"

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

    # NEW LOGIC: Handling Lists of Dictionaries instead of pure Dictionaries
    for mfg, families_list in aircraft_raw.items():
        if not families_list: continue
        
        for family_dict in families_list:
            for family, variants_list in family_dict.items():
                family_to_manufacturer[family] = mfg
                if not variants_list: continue
                
                for variant_dict in variants_list:
                    for variant, models in variant_dict.items():
                        variant_to_family[variant] = family
                        model_to_variant[variant] = variant # Map variant to itself
                        
                        if models:
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

    # 2. Update Model to Variant
    df['aircraft_variant'] = df['aircraft_variant'].map(model_to_variant).fillna(df['aircraft_variant'])

    # 3. Add Family and Manufacturer columns for future hierarchical training/evaluation
    df['aircraft_family'] = df['aircraft_variant'].map(variant_to_family).fillna("Unknown")
    df['aircraft_manufacturer'] = df['aircraft_family'].map(family_to_manufacturer).fillna("Unknown")

    # Reorder columns slightly to group the aircraft hierarchy together nicely
    cols = list(df.columns)
    # Move family and manufacturer right after variant if possible
    if 'aircraft_family' in cols and 'aircraft_manufacturer' in cols:
        cols.remove('aircraft_family')
        cols.remove('aircraft_manufacturer')
        variant_idx = cols.index('aircraft_variant')
        cols.insert(variant_idx + 1, 'aircraft_family')
        cols.insert(variant_idx + 2, 'aircraft_manufacturer')
        df = df[cols]

    # Save to the NEW file so the old one is untouched
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"\nSuccess! Cleaned dataset saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    update_dataset()