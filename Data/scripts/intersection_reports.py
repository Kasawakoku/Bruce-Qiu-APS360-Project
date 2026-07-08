import pandas as pd
import os

# --- Configuration ---
INTERSECTION_CSV = "../metadata/airliners_metadata_trimmed_intersection.csv"
OUTPUT_CSV = "../metadata/intersection_breakdown.csv"

def generate_intersection_breakdown():
    if not os.path.exists(INTERSECTION_CSV):
        print(f"Error: Could not find '{INTERSECTION_CSV}'. Please run the trimming script first.")
        return

    print(f"Loading data from {INTERSECTION_CSV}...")
    df = pd.read_csv(INTERSECTION_CSV)

    print("Calculating airline-variant combinations...")
    
    # Group by both Airline and Variant, count the occurrences, and reset into a clean table
    breakdown_df = df.groupby(['airline', 'aircraft_variant']).size().reset_index(name='Image Count')

    # Sort alphabetically by Airline first, and then by Variant within that Airline
    breakdown_df = breakdown_df.sort_values(by=['airline', 'aircraft_variant'], ascending=[True, True])

    # Capitalize the column names for a cleaner CSV output
    breakdown_df.rename(columns={
        'airline': 'Airline',
        'aircraft_variant': 'Aircraft Variant'
    }, inplace=True)

    # Save the breakdown
    breakdown_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

    # --- Print Report ---
    print("\n" + "="*50)
    print("INTERSECTION BREAKDOWN REPORT")
    print("="*50)
    print(f"Total Unique Airline-Variant Pairs: {len(breakdown_df)}")
    print(f"Total Images Represented:           {breakdown_df['Image Count'].sum()}")
    print("="*50)
    print(f"Breakdown exported to: {OUTPUT_CSV}\n")
    
    # Print a quick preview of the top 5 results so you can see it working
    print("Preview of first 5 rows:")
    print(breakdown_df.head(5).to_string(index=False))

if __name__ == "__main__":
    generate_intersection_breakdown()