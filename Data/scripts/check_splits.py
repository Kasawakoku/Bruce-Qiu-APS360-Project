# For analyzing train/val/test splitting among classes

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# --- Configuration ---

# Helper function to handle string "None" inputs from terminal
def parse_optional_str(val, default):
    if val is None:
        return default
    if str(val).strip().lower() == "none":
        return None
    return val

# Default configurations
DEFAULT_AIRLINE_FILTER = None
DEFAULT_VARIANT_FILTER = None
DEFAULT_TRAIN_CSV = "../metadata/train/train_metadata.csv"
DEFAULT_VAL_CSV = "../metadata/val/val_metadata.csv"
DEFAULT_TEST_CSV = "../metadata/test/test_metadata.csv"
DEFAULT_VALID_AIRLINES_CSV = "../metadata/counts_airlines_merged_trimmed.csv"
DEFAULT_VALID_VARIANTS_CSV = "../metadata/counts_variants_trimmed.csv"

# Positional terminal arguments:
# python check_splits.py [airline_filter] [variant_filter] [train_csv] [val_csv] [test_csv] [valid_airlines_csv] [valid_variants_csv]

AIRLINE_FILTER = parse_optional_str(sys.argv[1] if len(sys.argv) > 1 else None, DEFAULT_AIRLINE_FILTER)
VARIANT_FILTER = parse_optional_str(sys.argv[2] if len(sys.argv) > 2 else None, DEFAULT_VARIANT_FILTER)
TRAIN_CSV = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_TRAIN_CSV
VAL_CSV = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_VAL_CSV
TEST_CSV = sys.argv[5] if len(sys.argv) > 5 else DEFAULT_TEST_CSV
VALID_AIRLINES_CSV = sys.argv[6] if len(sys.argv) > 6 else DEFAULT_VALID_AIRLINES_CSV
VALID_VARIANTS_CSV = sys.argv[7] if len(sys.argv) > 7 else DEFAULT_VALID_VARIANTS_CSV

def visualize_split(airline_filter=None, variant_filter=None):
    """
    Visualizes the train/val/test split of the dataset.
    Leaves filters as None to see the entire dataset split.
    """
    # 1. Load Data
    dataframes = []
    splits = {'Train': TRAIN_CSV, 'Validation': VAL_CSV, 'Test': TEST_CSV}
    
    for split_name, file_path in splits.items():
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Skipping...")
            continue
            
        df = pd.read_csv(file_path)
        df['Split'] = split_name # Tag the data with its split name
        dataframes.append(df)
        
    if not dataframes:
        print("Error: No CSV files found.")
        return
        
    # Combine into a single master dataframe
    master_df = pd.concat(dataframes, ignore_index=True)
    
    # 2. Apply Filters (if specified)
    title = "Dataset Split Distribution"
    
    if airline_filter:
        master_df = master_df[master_df['airline'] == airline_filter]
        title += f"\nAirline: {airline_filter}"
        
    if variant_filter:
        master_df = master_df[master_df['aircraft_variant'] == variant_filter]
        title += f" | Variant: {variant_filter}"
        
    if len(master_df) == 0:
        print(f"No records found matching those filters.")
        return

    # 3. Calculate Counts & Percentages
    split_counts = master_df['Split'].value_counts()
    
    # Force the standard order (Train, Validation, Test) if they exist
    order = [s for s in ['Train', 'Validation', 'Test'] if s in split_counts.index]
    split_counts = split_counts.reindex(order)
    
    total_images = split_counts.sum()
    
    print("="*40)
    print("SPLIT DISTRIBUTION REPORT")
    print("="*40)
    if airline_filter: print(f"Airline: {airline_filter}")
    if variant_filter: print(f"Variant: {variant_filter}")
    print(f"Total Images: {total_images}\n")
    
    for split_name in order:
        count = split_counts[split_name]
        pct = (count / total_images) * 100
        print(f"{split_name + ':':<15} {count} images ({pct:.1f}%)")
    print("="*40)

    # 4. Plot the Bar Graph
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    
    # Define standard colors so Train is always blue, Val is orange, etc.
    colors = {'Train': '#1f77b4', 'Validation': '#ff7f0e', 'Test': '#2ca02c'}
    
    ax = sns.barplot(
        x=split_counts.index, 
        y=split_counts.values, 
        palette=[colors[s] for s in order]
    )
    
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Number of Images", fontsize=12)
    plt.xlabel("Dataset Split", fontsize=12)
    
    # Add the exact numbers on top of each bar
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}', 
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.show()


def load_valid_sets():
    """Loads the valid airlines and variants into fast-lookup sets."""
    valid_airlines = set()
    valid_variants = set()
    
    if os.path.exists(VALID_AIRLINES_CSV):
        df_air = pd.read_csv(VALID_AIRLINES_CSV)
        valid_airlines = set(df_air['Airline'].astype(str))
        
    if os.path.exists(VALID_VARIANTS_CSV):
        df_var = pd.read_csv(VALID_VARIANTS_CSV)
        valid_variants = set(df_var['Variant'].astype(str))
        
    return valid_airlines, valid_variants

def analyze_skew():
    valid_airlines, valid_variants = load_valid_sets()
    
    # 1. Load and combine the splits
    dataframes = []
    splits = {'Train': TRAIN_CSV, 'Val': VAL_CSV, 'Test': TEST_CSV}
    
    for split_name, file_path in splits.items():
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found.")
            continue
        df = pd.read_csv(file_path)
        df['Split'] = split_name
        dataframes.append(df)
        
    if not dataframes:
        print("Error: No split CSVs found.")
        return
        
    master_df = pd.concat(dataframes, ignore_index=True)
    
    # 2. Map invalid/rare classes to "OTHERS"
    master_df['mapped_airline'] = master_df['airline'].apply(
        lambda x: str(x) if str(x) in valid_airlines else "OTHERS"
    )
    master_df['mapped_variant'] = master_df['aircraft_variant'].apply(
        lambda x: str(x) if str(x) in valid_variants else "OTHERS"
    )
    
    # --- HELPER FUNCTION FOR ANALYSIS ---
    def process_column(col_name, display_name):
        print(f"\n" + "="*50)
        print(f"{display_name.upper()} SKEW ANALYSIS")
        print("="*50)
        
        # Count occurrences in each split
        cross_tab = pd.crosstab(master_df[col_name], master_df['Split'])
        
        # Ensure all columns exist just in case
        for s in ['Train', 'Val', 'Test']:
            if s not in cross_tab.columns:
                cross_tab[s] = 0
                
        # Calculate totals and percentages
        cross_tab['Total'] = cross_tab['Train'] + cross_tab['Val'] + cross_tab['Test']
        cross_tab['Train_%'] = (cross_tab['Train'] / cross_tab['Total']) * 100
        cross_tab['Val_%'] = (cross_tab['Val'] / cross_tab['Total']) * 100
        cross_tab['Test_%'] = (cross_tab['Test'] / cross_tab['Total']) * 100
        
        # Separate the "OTHERS" row so it doesn't compete in the Top 3 rankings
        other_stats = None
        if "OTHERS" in cross_tab.index:
            other_stats = cross_tab.loc["OTHERS"]
            cross_tab = cross_tab.drop("OTHERS")
            
        # Top 3 Skewed to Train
        top_train = cross_tab.sort_values(by='Train_%', ascending=False).head(3)
        print("\nTop 3 skewed towards TRAIN:")
        for idx, row in top_train.iterrows():
            print(f"  {idx:<35} | {row['Train_%']:>5.1f}% Train ({int(row['Train'])}/{int(row['Total'])} images)")

        # Top 3 Skewed to Val
        top_val = cross_tab.sort_values(by='Val_%', ascending=False).head(3)
        print("\nTop 3 skewed towards VAL:")
        for idx, row in top_val.iterrows():
            print(f"  {idx:<35} | {row['Val_%']:>5.1f}% Val   ({int(row['Val'])}/{int(row['Total'])} images)")

        # Top 3 Skewed to Test
        top_test = cross_tab.sort_values(by='Test_%', ascending=False).head(3)
        print("\nTop 3 skewed towards TEST:")
        for idx, row in top_test.iterrows():
            print(f"  {idx:<35} | {row['Test_%']:>5.1f}% Test  ({int(row['Test'])}/{int(row['Total'])} images)")

        # Report "OTHERS" stats
        print("\n--- Statistics for 'OTHERS' (Other/Trimmed) ---")
        if other_stats is not None:
            print(f"  Total Images: {int(other_stats['Total'])}")
            print(f"  Train:        {other_stats['Train_%']:>5.1f}% ({int(other_stats['Train'])})")
            print(f"  Val:          {other_stats['Val_%']:>5.1f}% ({int(other_stats['Val'])})")
            print(f"  Test:         {other_stats['Test_%']:>5.1f}% ({int(other_stats['Test'])})")
        else:
            print("  No 'OTHERS' records found in the dataset.")

    # 3. Run Analysis
    process_column('mapped_airline', 'Airline')
    process_column('mapped_variant', 'Variant')
    print("\n")



if __name__ == "__main__":
    analyze_skew()

    visualize_split(airline_filter=AIRLINE_FILTER, variant_filter=VARIANT_FILTER)
    

    # Used to create graphs in progress report
    # visualize_split(airline_filter="Austrian Airlines")
    # visualize_split(airline_filter="OTHERS")

    