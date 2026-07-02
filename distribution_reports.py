import pandas as pd
import os

# --- Configuration ---
INPUT_CSV = "airliners_metadata.csv"
AIRLINES_OUTPUT_CSV = "airline_counts.csv"
VARIANTS_OUTPUT_CSV = "variant_counts.csv"

def generate_distribution_reports():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Could not find '{INPUT_CSV}'. Make sure it is in the same folder.")
        return

    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    total_images = len(df)
    print(f"Total records found: {total_images}\n")

    # --- 1. Process Airlines ---
    # .dropna() removes rows where the airline was blank/None
    # .value_counts() groups them and counts the occurrences
    # .reset_index() turns it back into a standard DataFrame table
    airline_counts = df['airline'].dropna().value_counts().reset_index()
    airline_counts.columns = ['Airline', 'Image Count']
    
    # Sort alphabetically by the 'Airline' column
    airline_counts = airline_counts.sort_values(by='Airline', ascending=True)
    
    # Export to CSV
    airline_counts.to_csv(AIRLINES_OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"[*] Saved {len(airline_counts)} unique airlines to {AIRLINES_OUTPUT_CSV}")

    # --- 2. Process Aircraft Variants ---
    variant_counts = df['aircraft_variant'].dropna().value_counts().reset_index()
    variant_counts.columns = ['Aircraft Variant', 'Image Count']
    
    # Sort alphabetically by the 'Aircraft Variant' column
    variant_counts = variant_counts.sort_values(by='Aircraft Variant', ascending=True)
    
    # Export to CSV
    variant_counts.to_csv(VARIANTS_OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"[*] Saved {len(variant_counts)} unique aircraft variants to {VARIANTS_OUTPUT_CSV}")
    
    print("\nReports generated successfully!")

if __name__ == "__main__":
    generate_distribution_reports()