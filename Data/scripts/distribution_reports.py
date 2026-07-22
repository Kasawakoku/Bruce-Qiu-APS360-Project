# outputs list of all models and airlines and how many images per each

import pandas as pd
import os
import sys

# --- Configuration ---
# Default configurations
DEFAULT_INPUT_CSV = "../metadata/airliners_metadata.csv"
DEFAULT_AIRLINES_OUTPUT_CSV = "../class_definitions/counts_airlines.csv"
DEFAULT_MODEL_OUTPUT_CSV = "../class_definitions/counts_models.csv"

# Positional terminal arguments:
# python distribution_reports.py [input_csv] [airlines_output_csv] [model_output_csv]

INPUT_CSV = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_CSV
AIRLINES_OUTPUT_CSV = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_AIRLINES_OUTPUT_CSV
MODEL_OUTPUT_CSV = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_MODEL_OUTPUT_CSV

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
    counts_airlines = df['airline'].dropna().value_counts().reset_index()
    counts_airlines.columns = ['Airline', 'Image Count']
    
    # Sort alphabetically by the 'Airline' column
    counts_airlines = counts_airlines.sort_values(by='Airline', ascending=True)
    
    # Export to CSV
    counts_airlines.to_csv(AIRLINES_OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"[*] Saved {len(counts_airlines)} unique airlines to {AIRLINES_OUTPUT_CSV}")

    # --- 2. Process Aircraft MODEL ---
    counts_models = df['aircraft_model'].dropna().value_counts().reset_index()
    counts_models.columns = ['Aircraft Model', 'Image Count']
    
    # Sort alphabetically by the 'Aircraft Model' column
    counts_models = counts_models.sort_values(by='Aircraft Model', ascending=True)
    
    # Export to CSV
    counts_models.to_csv(MODEL_OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"[*] Saved {len(counts_models)} unique aircraft models to {MODEL_OUTPUT_CSV}")

    print("\nReports generated successfully!")

if __name__ == "__main__":
    generate_distribution_reports()