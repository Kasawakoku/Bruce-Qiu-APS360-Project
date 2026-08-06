# smart_scraper.py
import sys
import os
import time
import random
import re
import pandas as pd
import yaml
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from curl_cffi import requests

# --- Default Configuration ---
# Target & Cap Settings
DEFAULT_VARIANT_CAP = 80
DEFAULT_AIRLINE_CAP = 60
DEFAULT_OTHERS_CAP = 80 

# File Paths - Metadata & Tracking
OLD_MAIN_METADATA_CSV = "../metadata/airliners_metadata.csv"          # READ-ONLY: For duplicate exclusion
NEW_METADATA_CSV = "../metadata/final_test/final_test_airliners_metadata.csv"      # WRITE: The new dataset being scraped
DEFAULT_VARIANT_COUNTS_CSV = "../metadata/final_test/variant_counts.csv"
DEFAULT_AIRLINE_COUNTS_CSV = "../metadata/final_test/airline_counts.csv"
DEFAULT_NEW_MODELS_CSV = "../metadata/final_test/new_models_discovered.csv"
DEFAULT_NEW_AIRLINES_CSV = "../metadata/final_test/new_airlines_discovered.csv"

# File Paths - Definitions
DEFAULT_AIRLINE_YAML = "../class_definitions/airline_mapping.yaml"
DEFAULT_AIRCRAFT_YAML = "../class_definitions/aircraft_hierarchy.yaml"
DEFAULT_VALID_VARIANTS_CSV = "../metadata/counts_variants_trimmed.csv" # Expects column: 'Variant'
DEFAULT_VALID_AIRLINES_CSV = "../metadata/counts_airlines_merged_trimmed.csv" # Expects column: 'Airline'

# File Paths - Seen History (for discovery filtering)
DEFAULT_SEEN_MODELS_CSV = "../metadata/counts_models.csv"     # Edit as needed. Expects first column to contain the strings
DEFAULT_SEEN_AIRLINES_CSV = "../metadata/counts_airlines.csv" # Edit as needed. Expects first column to contain the strings

DEFAULT_IMAGE_SAVE_DIR = "../final_test_airliners_images"
DEFAULT_START_PAGE = 1000
DEFAULT_BASE_URL = "https://www.airliners.net/search?photoCategory=39&sortBy=dateAccepted&sortOrder=desc&perPage=36&display=detail&page={}"

# Base URLs for specific class scraping
# DEFAULT_BASE_URL = "https://www.airliners.net/search?airline=52603&sortBy=dateAccepted&sortOrder=desc&perPage=36&display=detail&page={}" # Spirit Airlines
# DEFAULT_BASE_URL = "https://www.airliners.net/search?aircraftBasicType=8639&sortBy=dateAccepted&sortOrder=desc&perPage=36&display=detail&page={}" # MD-80
# DEFAULT_BASE_URL = "https://www.airliners.net/search?aircraftBasicType=4773&sortBy=dateAccepted&sortOrder=desc&perPage=36&display=detail&page={}" # Embraer 550 Phenom

HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.airliners.net/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}

# --- System Args Override ---
START_PAGE = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_START_PAGE
VARIANT_CAP = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_VARIANT_CAP
AIRLINE_CAP = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_AIRLINE_CAP
OTHERS_CAP = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_OTHERS_CAP

# --- Core Functions ---

def load_mappings():
    """Loads YAML mapping files and handles the dash-list structure."""
    airline_lookup = {}
    if os.path.exists(DEFAULT_AIRLINE_YAML):
        with open(DEFAULT_AIRLINE_YAML, 'r', encoding='utf-8') as f:
            airline_raw = yaml.safe_load(f) or {}
            for true_airline, aliases in airline_raw.items():
                airline_lookup[true_airline] = true_airline
                if aliases:
                    for alias in aliases:
                        airline_lookup[alias] = true_airline

    model_to_variant = {}
    if os.path.exists(DEFAULT_AIRCRAFT_YAML):
        with open(DEFAULT_AIRCRAFT_YAML, 'r', encoding='utf-8') as f:
            aircraft_raw = yaml.safe_load(f) or {}
            for mfg, families_list in aircraft_raw.items():
                if not families_list: continue
                for family_item in families_list:
                    if isinstance(family_item, dict):
                        for family, variants_list in family_item.items():
                            if not variants_list: continue
                            for variant_item in variants_list:
                                if isinstance(variant_item, str):
                                    model_to_variant[variant_item] = variant_item
                                elif isinstance(variant_item, dict):
                                    for variant, models in variant_item.items():
                                        model_to_variant[variant] = variant
                                        if models and isinstance(models, list):
                                            for model in models:
                                                model_to_variant[model] = variant
    return airline_lookup, model_to_variant

def load_valid_classes():
    """Loads the specific lists of variants and airlines from their respective CSVs."""
    valid_variants, valid_airlines = set(), set()
    
    if os.path.exists(DEFAULT_VALID_VARIANTS_CSV):
        df_v = pd.read_csv(DEFAULT_VALID_VARIANTS_CSV)
        if 'Variant' in df_v.columns:
            valid_variants = set(df_v['Variant'].dropna().astype(str).unique())
            
    if os.path.exists(DEFAULT_VALID_AIRLINES_CSV):
        df_a = pd.read_csv(DEFAULT_VALID_AIRLINES_CSV)
        if 'Airline' in df_a.columns:
            valid_airlines = set(df_a['Airline'].dropna().astype(str).unique())
            
    return valid_variants, valid_airlines

def load_seen_classes():
    """Loads historical scraped lists to avoid re-flagging known items as 'discoveries'."""
    seen_models, seen_airlines = set(), set()
    
    if os.path.exists(DEFAULT_SEEN_MODELS_CSV):
        df_m = pd.read_csv(DEFAULT_SEEN_MODELS_CSV)
        col_m = df_m.columns[0] # Grab whatever the first column is named
        seen_models = set(df_m[col_m].dropna().astype(str).unique())
        
    if os.path.exists(DEFAULT_SEEN_AIRLINES_CSV):
        df_a = pd.read_csv(DEFAULT_SEEN_AIRLINES_CSV)
        col_a = df_a.columns[0] # Grab whatever the first column is named
        seen_airlines = set(df_a[col_a].dropna().astype(str).unique())
        
    return seen_models, seen_airlines

def evaluate_class(raw_string, lookup_dict, valid_set):
    """Maps the raw string, heavily prioritizing direct matches in the valid lists."""
    if not raw_string or pd.isna(raw_string): 
        return "OTHERS"
        
    clean_str = str(raw_string).strip()
    
    # 1. Check if the raw string is directly a valid class (bypasses YAML mapping)
    if clean_str in valid_set:
        return clean_str
        
    # 2. Check if the raw string maps to a valid class via YAML
    mapped_val = lookup_dict.get(clean_str)
    if mapped_val and mapped_val in valid_set:
        return mapped_val
        
    # 3. If it's not valid raw and doesn't map to a valid class, it's OTHERS
    return "OTHERS"

def save_discoveries(new_items_set, filepath, column_name):
    """Appends newly discovered unmapped strings to a CSV."""
    if not new_items_set: return
    new_df = pd.DataFrame(list(new_items_set), columns=[column_name])
    if os.path.exists(filepath):
        existing_df = pd.read_csv(filepath)
        combined = pd.concat([existing_df, new_df]).drop_duplicates()
        combined.to_csv(filepath, index=False, encoding='utf-8')
    else:
        new_df.to_csv(filepath, index=False, encoding='utf-8')
    new_items_set.clear()

def scrape_airlines():
    # --- PHASE 1: Initialization & Auto-Synchronization ---
    print("--- Initialization Phase ---")
    
    # FIX 1: Safely ensure ALL directories exist before we try to save to them
    os.makedirs(DEFAULT_IMAGE_SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(NEW_METADATA_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(DEFAULT_VARIANT_COUNTS_CSV), exist_ok=True)
    
    airline_lookup, model_to_variant = load_mappings()
    valid_variants, valid_airlines = load_valid_classes()
    hist_seen_models, hist_seen_airlines = load_seen_classes()
    
    seen_ids = set()
    variant_counts = {}
    airline_counts = {}
    
    # 1. Load OLD metadata strictly for duplicate exclusion (Read-Only)
    if os.path.exists(OLD_MAIN_METADATA_CSV) and os.path.getsize(OLD_MAIN_METADATA_CSV) > 0:
        print(f"Loading old main metadata for exclusion list...")
        df_old = pd.read_csv(OLD_MAIN_METADATA_CSV)
        if 'photo_id' in df_old.columns:
            seen_ids.update(df_old['photo_id'].astype(str))
            print(f"Added {len(df_old)} IDs from the old dataset to the exclusion list.")

    # 2. Load NEW metadata to resume progress (Exclusion + Balance Counting)
    if os.path.exists(NEW_METADATA_CSV) and os.path.getsize(NEW_METADATA_CSV) > 0:
        print(f"Loading new metadata to re-evaluate current scraping balances...")
        df_new = pd.read_csv(NEW_METADATA_CSV)
        
        if 'photo_id' in df_new.columns:
            seen_ids.update(df_new['photo_id'].astype(str))
            
            for _, row in df_new.iterrows():
                eval_var = evaluate_class(row.get('aircraft_model'), model_to_variant, valid_variants)
                eval_air = evaluate_class(row.get('airline'), airline_lookup, valid_airlines)
                
                variant_counts[eval_var] = variant_counts.get(eval_var, 0) + 1
                airline_counts[eval_air] = airline_counts.get(eval_air, 0) + 1
                
            print(f"Resuming with {len(df_new)} previously scraped balanced photos.")
    else:
        print("No new metadata found. Starting fresh class balances.")

    print("Syncing initial class counts to disk...")
    # Because we added os.makedirs above, these will now never crash
    pd.DataFrame(sorted(variant_counts.items()), columns=['variant', 'count']).to_csv(DEFAULT_VARIANT_COUNTS_CSV, index=False)
    pd.DataFrame(sorted(airline_counts.items()), columns=['airline', 'count']).to_csv(DEFAULT_AIRLINE_COUNTS_CSV, index=False)

    # Tracking discovered raw strings not in mappings or historical sets
    new_models = set()
    new_airlines = set()
    session_metadata = []

    # --- PHASE 2: Launch Stealth Browser ---
    print("\nLaunching stealth browser...")
    options = uc.ChromeOptions()
    options.headless = False 
    options.page_load_strategy = 'eager' 
    driver = uc.Chrome(options=options, version_main=149)
    driver.set_page_load_timeout(30) 

    page_num = START_PAGE

    try:
        while True: # Infinite Loop, driven by caps and balances
            start_page_time = time.time()
            print(f"\n--- Scraping page {page_num} ---")
            
            url = DEFAULT_BASE_URL.format(page_num)
            try:
                driver.get(url)
                time.sleep(random.uniform(5.0, 8.0)) # Wait for WAF & images
            except Exception as e:
                print(f"Browser navigation issue on page {page_num}: {e}")
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            photo_containers = soup.select('div.ps-v2-results-display-detail-col.photo')
            
            if not photo_containers:
                print("No photos found on this page. Anti-bot block or end of results?")
                time.sleep(30) # Heavy backoff before retrying
                continue
            
            page_downloads = 0
            page_skips = 0

            for container in photo_containers:
                # Extract ID
                id_element = container.select_one('div.ps-v2-results-col-title-photo-id')
                if not id_element: continue
                photo_id = id_element.text.strip().replace('#', '')
                if not photo_id.isdigit(): continue
                
                if photo_id in seen_ids:
                    page_skips += 1
                    continue
                
                # Helper function
                def extract_text(selector):
                    el = container.select_one(selector)
                    return el.text.strip() if el else None

                # Extract Raw Data
                raw_airline = extract_text('a[href*="airline="]')
                raw_model = extract_text('a[href*="aircraft="]')
                
                # Evaluate Desirability
                eval_var = evaluate_class(raw_model, model_to_variant, valid_variants)
                eval_air = evaluate_class(raw_airline, airline_lookup, valid_airlines)
                
                # Algorithm Rule 1: Skip if both are OTHERS
                if eval_var == "OTHERS" and eval_air == "OTHERS":
                    page_skips += 1
                    continue
                
                # Algorithm Rule 2: Check Caps
                v_cap = OTHERS_CAP if eval_var == "OTHERS" else VARIANT_CAP
                a_cap = OTHERS_CAP if eval_air == "OTHERS" else AIRLINE_CAP
                
                if variant_counts.get(eval_var, 0) >= v_cap and airline_counts.get(eval_air, 0) >= a_cap:
                    page_skips += 1
                    continue

                # PASSES FILTER - Extract remaining metadata & download
                registration = extract_text('a[href*="registrationActual="]')
                msn = extract_text('a[href*="manufacturerSerialNumber="]')
                location = extract_text('a[href*="location="]')
                date_photo = extract_text('a[href*="datePhotographed="]')
                photographer = extract_text('a.ua-name-content')
                caption = extract_text('div.ps-v2-results-col-caption div.ps-v2-results-col-content')

                # Download Image using sharding trick via curl_cffi
                img_filename = None
                if len(photo_id) >= 3:
                    shard = f"{photo_id[-1]}/{photo_id[-2]}/{photo_id[-3]}"
                    img_url = f"https://imgproc.airliners.net/photos/airliners/{shard}/{photo_id}.jpg"
                    img_filename = f"{photo_id}.jpg"
                    img_filepath = os.path.join(DEFAULT_IMAGE_SAVE_DIR, img_filename)
                    
                    if not os.path.exists(img_filepath):
                        try:
                            img_response = requests.get(img_url, headers=HEADERS, impersonate="chrome110", stream=True)
                            if img_response.status_code == 200:
                                with open(img_filepath, 'wb') as f:
                                    for chunk in img_response.iter_content(1024):
                                        f.write(chunk)
                                page_downloads += 1
                                seen_ids.add(photo_id)
                                variant_counts[eval_var] = variant_counts.get(eval_var, 0) + 1
                                airline_counts[eval_air] = airline_counts.get(eval_air, 0) + 1
                            else:
                                img_filename = None # Failed download
                        except Exception as e:
                            img_filename = None
                
                if img_filename: # Only save metadata if download succeeded
                    session_metadata.append({
                        "photo_id": photo_id,
                        "airline": raw_airline,       # DUMP RAW
                        "aircraft_model": raw_model,  # DUMP RAW
                        "registration": registration,
                        "msn": msn,
                        "location": location,
                        "date": date_photo,
                        "photographer": photographer,
                        "caption": caption,
                        "image_filename": img_filename
                    })
                    
                    # Discovery Logging ONLY for images we actually kept
                    if raw_model:
                        if raw_model not in hist_seen_models and raw_model not in model_to_variant and raw_model not in valid_variants:
                            new_models.add(raw_model)
                    if raw_airline:
                        if raw_airline not in hist_seen_airlines and raw_airline not in airline_lookup and raw_airline not in valid_airlines:
                            new_airlines.add(raw_airline)

            # --- PHASE 3: Checkpointing & Reports ---
            elapsed = time.time() - start_page_time
            print(f"Page {page_num} Report | Downloads: {page_downloads} | Skips: {page_skips} | Time: {elapsed:.2f}s")
            
            if (page_num % 5 == 0) and session_metadata:
                print(">>> Saving Checkpoints...")
                
                # 1. Save ONLY to the New Metadata CSV
                new_md_df = pd.DataFrame(session_metadata)
                if os.path.exists(NEW_METADATA_CSV):
                    existing_md = pd.read_csv(NEW_METADATA_CSV)
                    pd.concat([existing_md, new_md_df]).to_csv(NEW_METADATA_CSV, index=False, encoding='utf-8')
                else:
                    new_md_df.to_csv(NEW_METADATA_CSV, index=False, encoding='utf-8')
                
                session_metadata.clear()
                
                # 2. Count State CSVs
                pd.DataFrame(sorted(variant_counts.items()), columns=['variant', 'count']).to_csv(DEFAULT_VARIANT_COUNTS_CSV, index=False)
                pd.DataFrame(sorted(airline_counts.items()), columns=['airline', 'count']).to_csv(DEFAULT_AIRLINE_COUNTS_CSV, index=False)
                
                # 3. Discoveries
                save_discoveries(new_models, DEFAULT_NEW_MODELS_CSV, "raw_model")
                save_discoveries(new_airlines, DEFAULT_NEW_AIRLINES_CSV, "raw_airline")

            page_num += 1

    except KeyboardInterrupt:
        print("\nScraping manually interrupted. Saving final states...")
    finally:
        driver.quit()
        # Final safety save on exit
        if session_metadata:
            new_md_df = pd.DataFrame(session_metadata)
            if os.path.exists(NEW_METADATA_CSV):
                existing_md = pd.read_csv(NEW_METADATA_CSV)
                pd.concat([existing_md, new_md_df]).to_csv(NEW_METADATA_CSV, index=False, encoding='utf-8')
            else:
                new_md_df.to_csv(NEW_METADATA_CSV, index=False, encoding='utf-8')
                
        pd.DataFrame(sorted(variant_counts.items()), columns=['variant', 'count']).to_csv(DEFAULT_VARIANT_COUNTS_CSV, index=False)
        pd.DataFrame(sorted(airline_counts.items()), columns=['airline', 'count']).to_csv(DEFAULT_AIRLINE_COUNTS_CSV, index=False)
        save_discoveries(new_models, DEFAULT_NEW_MODELS_CSV, "raw_model")
        save_discoveries(new_airlines, DEFAULT_NEW_AIRLINES_CSV, "raw_airline")
        print("Shutdown complete.")

if __name__ == "__main__":
    scrape_airlines()