## DEPRECATED

import math
import pandas as pd

def generate_split_csvs(
    intersection_csv_path, 
    union_only_csv_path,
    intersection_summary_csv_path, # Your CSV with the 'Image Count' column
    valid_airlines_path, 
    valid_variants_path,
    out_train_path='train_metadata.csv',
    out_val_path='val_metadata.csv',
    out_test_path='test_metadata.csv',
    val_pct=0.10, 
    test_pct=0.10, 
    min_count=5, 
    random_state=42
):
    """
    Reads raw data, applies parameterized 'RARE_GROUP' and masking logic,
    carves out validation and test sets, and exports the final splits to CSVs.
    """
    print("Loading datasets...")
    df_inter = pd.read_csv(intersection_csv_path)
    df_union_only = pd.read_csv(union_only_csv_path)
    
    # Load your pre-computed counts to optimize the rare group identification
    df_summary = pd.read_csv(intersection_summary_csv_path)
    
    valid_airlines = set(pd.read_csv(valid_airlines_path)['Airline'].dropna().unique())
    valid_variants = set(pd.read_csv(valid_variants_path)['Variant'].dropna().unique())
    
    # 1. IDENTIFY RARE CLASSES USING YOUR SUMMARY CSV
    # Create the matching key format: "Airline_Variant"
    df_summary['split_key'] = df_summary['Airline'].astype(str) + "_" + df_summary['Aircraft Variant'].astype(str)
    
    # Filter the summary to find keys below the minimum threshold
    rare_keys = set(df_summary[df_summary['Image Count'] < min_count]['split_key'])
    
    # 2. APPLY SPLIT KEYS TO MAIN INTERSECTION DATA
    df_inter['raw_key'] = df_inter['airline'].astype(str) + "_" + df_inter['aircraft_variant'].astype(str)
    df_inter['split_key'] = df_inter['raw_key'].apply(
        lambda x: 'RARE_GROUP' if x in rare_keys else x
    )
    
    # 3. CUSTOM GROUPBY SPLIT LOGIC
    def split_group(group):
        n_total = len(group)
        n_val = math.ceil(n_total * val_pct)
        n_test = math.ceil(n_total * test_pct)
        
        # Failsafe: Ensure at least 1 image remains for Training 
        if n_val + n_test >= n_total and n_total >= 3:
            n_val = max(1, math.floor(n_total * val_pct))
            n_test = max(1, math.floor(n_total * test_pct))
            if n_val + n_test >= n_total:
                n_val, n_test = 1, 1
                
        # Failsafe: if total is < 3, they all go to train
        if n_total < 3:
            n_val, n_test = 0, 0
            
        shuffled = group.sample(frac=1, random_state=random_state).copy()
        
        shuffled['split_label'] = 'train'
        split_col_idx = shuffled.columns.get_loc('split_label')
        
        if n_val > 0:
            shuffled.iloc[0 : n_val, split_col_idx] = 'val'
        if n_test > 0:
            shuffled.iloc[n_val : n_val + n_test, split_col_idx] = 'test'
        
        return shuffled

    print("Carving out Validation and Test sets...")
    split_df = df_inter.groupby('split_key', group_keys=False).apply(split_group)
    
    # 4. SEPARATE THE SPLITS
    inter_train_df = split_df[split_df['split_label'] == 'train'].copy()
    final_val_df = split_df[split_df['split_label'] == 'val'].copy()
    final_test_df = split_df[split_df['split_label'] == 'test'].copy()
    
    print("Applying -100 ignore_index masks to union data...")
    # 5. MASK INVALID LABELS IN UNION DATA
    df_union_only.loc[~df_union_only['airline'].isin(valid_airlines), 'airline'] = '-100'
    df_union_only.loc[~df_union_only['aircraft_variant'].isin(valid_variants), 'aircraft_variant'] = '-100'
    
    # 6. MERGE AND CLEAN COLUMNS
    # keep_cols = ['photo_id', 'airline', 'aircraft_variant', 'image_filename']
    keep_cols = [
        'photo_id', 'airline', 'aircraft_model', 'aircraft_variant', 
        'aircraft_family', 'aircraft_manufacturer', 'registration', 
        'msn', 'location', 'date', 'photographer', 'image_filename', 'caption'
    ]
    
    final_train_df = pd.concat([
        inter_train_df[keep_cols], 
        df_union_only[keep_cols]
    ], ignore_index=True)
    
    final_val_df = final_val_df[keep_cols]
    final_test_df = final_test_df[keep_cols]
    
    # 7. EXPORT TO CSV
    print(f"Exporting files...")
    final_train_df.to_csv(out_train_path, index=False)
    final_val_df.to_csv(out_val_path, index=False)
    final_test_df.to_csv(out_test_path, index=False)
    
    print("\n--- Pipeline Complete: Splits Saved to Disk ---")
    print(f"Train set: {len(final_train_df)} rows saved to {out_train_path}")
    print(f"Val set:   {len(final_val_df)} rows saved to {out_val_path}")
    print(f"Test set:  {len(final_test_df)} rows saved to {out_test_path}")


generate_split_csvs(
    '../metadata/airliners_metadata_trimmed_intersection.csv',
    '../metadata/airliners_metadata_trimmed_symmetric_diff.csv',
    '../metadata/intersection_breakdown.csv',
    '../metadata/counts_airlines_merged_trimmed.csv',
    '../metadata/counts_variants_trimmed.csv',
    '../metadata/train/train_metadata.csv',
    '../metadata/val/val_metadata.csv',
    '../metadata/test/test_metadata.csv',
    0.2,
    0.2,
    5,
    42
)
