import math
import pandas as pd

def generate_split_csvs(
    intersection_csv_path, 
    union_only_csv_path,
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
    Reads raw data, assigns temporary 'OTHER' categories to invalid labels to 
    include them in the split stratification, applies the -100 mask for training,
    carves out validation and test sets, and exports the final splits to CSVs.
    """
    print("Loading datasets...")
    df_inter = pd.read_csv(intersection_csv_path)
    df_union_only = pd.read_csv(union_only_csv_path)
    
    valid_airlines = set(pd.read_csv(valid_airlines_path)['Airline'].dropna().unique())
    valid_variants = set(pd.read_csv(valid_variants_path)['Variant'].dropna().unique())
    
    # 1. COMBINE DATASETS FOR UNIFIED SPLITTING
    df_all = pd.concat([df_inter, df_union_only], ignore_index=True)
    
    # 2. CREATE TEMPORARY 'OTHER' LABELS FOR STRATIFICATION
    # This groups all private/invalid airlines into an 'OTHER_AIRLINE' bucket,
    # and invalid variants into an 'OTHER_VARIANT' bucket for splitting purposes.
    df_all['temp_airline'] = df_all['airline'].apply(
        lambda x: x if x in valid_airlines else 'OTHER_AIRLINE'
    )
    df_all['temp_variant'] = df_all['aircraft_variant'].apply(
        lambda x: x if x in valid_variants else 'OTHER_VARIANT'
    )
    
    df_all['split_key'] = df_all['temp_airline'].astype(str) + "_" + df_all['temp_variant'].astype(str)
    
    # 3. IDENTIFY RARE CLASSES DYNAMICALLY
    # We calculate this on df_all so the 'OTHER' groups are counted accurately
    counts = df_all['split_key'].value_counts()
    rare_keys = counts[counts < min_count].index
    
    df_all['split_key'] = df_all['split_key'].apply(
        lambda x: 'RARE_GROUP' if x in rare_keys else x
    )
    
    # 4. CUSTOM GROUPBY SPLIT LOGIC
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
    split_df = df_all.groupby('split_key', group_keys=False).apply(split_group)
    
    # 5. SEPARATE THE SPLITS
    final_train_df = split_df[split_df['split_label'] == 'train'].copy()
    final_val_df = split_df[split_df['split_label'] == 'val'].copy()
    final_test_df = split_df[split_df['split_label'] == 'test'].copy()
    
    print("Applying -100 ignore_index masks to final data...")
    # 6. MASK INVALID LABELS WITH -100 FOR PYTORCH
    for df in [final_train_df, final_val_df, final_test_df]:
        df.loc[~df['airline'].isin(valid_airlines), 'airline'] = '-100'
        df.loc[~df['aircraft_variant'].isin(valid_variants), 'aircraft_variant'] = '-100'
    
    # 7. MERGE AND CLEAN COLUMNS
    keep_cols = [
        'photo_id', 'airline', 'aircraft_model', 'aircraft_variant', 
        'aircraft_family', 'aircraft_manufacturer', 'registration', 
        'msn', 'location', 'date', 'photographer', 'image_filename', 'caption'
    ]
    
    final_train_df = final_train_df[keep_cols]
    final_val_df = final_val_df[keep_cols]
    final_test_df = final_test_df[keep_cols]
    
    # 8. EXPORT TO CSV
    print(f"Exporting files...")
    final_train_df.to_csv(out_train_path, index=False)
    final_val_df.to_csv(out_val_path, index=False)
    final_test_df.to_csv(out_test_path, index=False)
    
    print("\n--- Pipeline Complete: Splits Saved to Disk ---")
    print(f"Total images: {len(final_train_df) + len(final_val_df) + len(final_test_df)}")
    print(f"Train set: {len(final_train_df)} rows saved to {out_train_path}")
    print(f"Val set:   {len(final_val_df)} rows saved to {out_val_path}")
    print(f"Test set:  {len(final_test_df)} rows saved to {out_test_path}")


generate_split_csvs(
    '../metadata/airliners_metadata_trimmed_intersection.csv',
    '../metadata/airliners_metadata_trimmed_symmetric_diff.csv',
    #'../metadata/intersection_breakdown.csv',
    '../metadata/counts_airlines_merged_trimmed.csv',
    '../metadata/counts_variants_trimmed.csv',
    '../metadata/train/train_metadata.csv',
    '../metadata/val/val_metadata.csv',
    '../metadata/test/test_metadata.csv',
    0.1,
    0.1,
    5,
    42
)