import pandas as pd
import yaml
import os

# --- Configuration ---
INPUT_CSV = "../metadata/airliners_metadata.csv"
AIRLINE_YAML = "../class_definitions/airline_mapping.yaml"
AIRCRAFT_YAML = "../class_definitions/aircraft_hierarchy.yaml"

def load_mappings():
    # 1. Load and invert Airline Mapping (Alias -> True Airline)
    with open(AIRLINE_YAML, 'r', encoding='utf-8') as f:
        airline_raw = yaml.safe_load(f)
    
    airline_lookup = {}
    for true_airline, aliases in airline_raw.items():
        airline_lookup[true_airline] = true_airline # Map it to itself just in case
        for alias in aliases:
            airline_lookup[alias] = true_airline

    # 2. Load and flatten Aircraft Hierarchy
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
                        
    

def generate_reports():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Could not find {INPUT_CSV}")
        return

    print("Loading data and mappings...")
    df = pd.read_csv(INPUT_CSV)
    airline_lookup, model_to_variant, variant_to_family, family_to_manufacturer = load_mappings()

    # Apply mappings. 
    # .fillna() ensures that if a string ISN'T in your YAML, it just keeps the original scraped string.
    df['merged_airline'] = df['airline'].map(airline_lookup).fillna(df['airline'])
    df['variant'] = df['aircraft_model'].map(model_to_variant).fillna(df['aircraft_model'])
    df['family'] = df['variant'].map(variant_to_family).fillna("Unknown Family")
    df['manufacturer'] = df['family'].map(family_to_manufacturer).fillna("Unknown Manufacturer")

    # Define the groupings we want to count
    reports = {
        'counts_airlines_merged.csv': ('merged_airline', 'Airline'),
        'counts_variants.csv': ('variant', 'Variant'),
        'counts_families.csv': ('family', 'Family'),
        'counts_manufacturers.csv': ('manufacturer', 'Manufacturer')
    }

    # Generate, sort, and save each report
    for filename, (col_name, label) in reports.items():
        counts = df[col_name].dropna().value_counts().reset_index()
        counts.columns = [label, 'Image Count']
        counts = counts.sort_values(by=label, ascending=True)
        counts.to_csv(filename, index=False, encoding='utf-8')
        print(f"[*] Saved {len(counts)} unique {label}s to {filename}")

if __name__ == "__main__":
    generate_reports()