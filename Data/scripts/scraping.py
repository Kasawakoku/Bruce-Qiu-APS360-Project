# main scraping script

import sys
import os
import time
import random
import re
import pandas as pd
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from curl_cffi import requests, CurlOpt
from curl_cffi.requests.session import Session, RetryStrategy

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# --- Default Configuration ---

# Edit as necessary
DEFAULT_START_PAGE = 1
DEFAULT_END_PAGE = 2

DEFAULT_IMAGE_SAVE_DIR = "../airliners_images"
DEFAULT_CSV_FILENAME = "../metadata/airliners_metadata.csv"

# This base URL sorts the Airliners.net database by most recently updated and filter for non-military aircraft
DEFAULT_BASE_URL = "https://www.airliners.net/search?photoCategory=39&sortBy=dateAccepted&sortOrder=desc&perPage=36&display=detail&page={}"

HEADERS = {
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Referer": "https://www.airliners.net/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}

# Positional terminal arguments:
# python scraping.py [start_page] [end_page] [image_dir] [csv_file] [base_url]

START_PAGE = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_START_PAGE
END_PAGE = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_END_PAGE
IMAGE_SAVE_DIR = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_IMAGE_SAVE_DIR
CSV_FILENAME = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_CSV_FILENAME
BASE_URL = sys.argv[5] if len(sys.argv) > 5 else DEFAULT_BASE_URL

start_time = time.time()

if not os.path.exists(IMAGE_SAVE_DIR):
    os.makedirs(IMAGE_SAVE_DIR)

metadata_records = []
total_records = 0

# Metrics for the final report
pages_processed = 0
images_downloaded = 0
images_skipped_existing = 0
failed_downloads = 0

# --- Initialize Stealth Browser ---
print("Launching stealth browser...")
options = uc.ChromeOptions()
options.headless = False 
options.page_load_strategy = 'eager' 

driver = uc.Chrome(options=options, version_main=149)
driver.set_page_load_timeout(30) 


retry_strategy = RetryStrategy(
    count=1,                    # Try 3 times before giving up entirely
    delay=0,                    # Wait 1 second before the first retry
    backoff="linear",      # Double the wait time each failure (1s, 2s, 4s...)
    # status_forcelist=[503, 502, 429] # Only retry on these server errors
)

# --- Main Scraping Loop ---
for page_num in range(START_PAGE, END_PAGE + 1):

    with Session(retry=retry_strategy) as session:
        session.low_speed_limit = 0
        session.low_speed_time = 0

        start_page_time = time.time()

        print(f"\n--- Scraping page {page_num} ---")
        url = BASE_URL.format(page_num)
        
        try:
            driver.get(url)
        except TimeoutException:
            print("Page load hit the 30s timeout limit, but HTML might still be readable. Continuing...")

        try:
            print("Waiting for WAF challenge to pass and photos to load...")
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.ps-v2-results-display-detail-col.photo"))
            )
        except TimeoutException:
            print(f"FAILED: The anti-bot challenge never passed on page {page_num}.")
            continue 
            
        # Extract HTML directly from the browser
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # Use standard CSS selection to ensure we ONLY grab the photo blocks
        photo_containers = soup.select('div.ps-v2-results-display-detail-col.photo')
        print(f"Found {len(photo_containers)} photos on page {page_num}.")
        
        pages_processed += 1
        
        for container in photo_containers:
            
            # 1. Extract Photo ID safely
            id_element = container.select_one('div.ps-v2-results-col-title-photo-id')
            if not id_element:
                continue
                
            photo_id = id_element.text.strip().replace('#', '')
            
            # Guardrail: If it's not a number, skip it immediately
            if not photo_id.isdigit():
                continue
                
            # Helper function
            def extract_text(selector):
                element = container.select_one(selector)
                return element.text.strip() if element else None

            # 2. Extract Metadata (Now includes Caption)
            airline = extract_text('a[href*="airline="]')
            aircraft = extract_text('a[href*="aircraft="]')
            registration = extract_text('a[href*="registrationActual="]')
            msn = extract_text('a[href*="manufacturerSerialNumber="]')
            location = extract_text('a[href*="location="]')
            date_photo = extract_text('a[href*="datePhotographed="]')
            photographer = extract_text('a.ua-name-content')
            caption = extract_text('div.ps-v2-results-col-caption div.ps-v2-results-col-content')
            
            # 3. Extract Security Hash from Thumbnail
            img_element = container.select_one('img.lazy-load')
            hash_val = ""
            if img_element and 'src' in img_element.attrs:
                thumb_src = img_element['src']
                match = re.search(r'-(v[a-f0-9]+)', thumb_src)
                if match:
                    hash_val = match.group(1)
            
            # 4. Construct Exact Image URL & Download
            if len(photo_id) >= 3 and hash_val:
                shard = f"{photo_id[-1]}/{photo_id[-2]}/{photo_id[-3]}"
                img_url = f"https://imgproc.airliners.net/photos/airliners/{shard}/{photo_id}.jpg?v={hash_val}"
                
                img_filename = f"{photo_id}.jpg"
                img_filepath = os.path.join(IMAGE_SAVE_DIR, img_filename)
                
                

                if not os.path.exists(img_filepath):
                    try:
                        img_response = session.get(img_url, headers=HEADERS, impersonate="chrome110", stream=True, timeout=10)
                        if img_response.status_code == 200:
                            with open(img_filepath, 'wb') as f:
                                for chunk in img_response.iter_content(1024):
                                    f.write(chunk)
                            images_downloaded += 1
                        else:
                            print(f"  -> Failed to download {photo_id} (HTTP {img_response.status_code})")
                            failed_downloads += 1
                    except Exception as e:
                        print(f"  -> Error downloading {photo_id}: {e}")
                        failed_downloads += 1
                else:
                    images_skipped_existing += 1
            else:
                img_filename = None
                
            # 5. Store Data
            metadata_records.append({
                "photo_id": photo_id,
                "airline": airline,
                "aircraft_model": aircraft,
                "registration": registration,
                "msn": msn,
                "location": location,
                "date": date_photo,
                "photographer": photographer,
                "caption": caption,
                "image_filename": img_filename
            })

        # --- Save Metadata Every 10 Pages (or at the very end) ---
        if (page_num % 10 == 0) or (page_num == END_PAGE):

            print(f"Saving metadata to {CSV_FILENAME}...")
            new_df = pd.DataFrame(metadata_records)

            if os.path.exists(CSV_FILENAME):
                
                # Load the old data
                existing_df = pd.read_csv(CSV_FILENAME)
                
                # Ensure photo_id is treated as a string to match perfectly
                existing_df['photo_id'] = existing_df['photo_id'].astype(str)
                new_df['photo_id'] = new_df['photo_id'].astype(str)
                
                # Combine the old data and the newly scraped data
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                
                # Drop duplicates based on the ID. 
                # keep='last' ensures that the newly scraped metadata overwrites the old metadata!
                combined_df.drop_duplicates(subset=['photo_id'], keep='last', inplace=True)
                
                # Save it back
                combined_df.to_csv(CSV_FILENAME, index=False, encoding='utf-8')
                total_records = len(combined_df)
            else:
                # If no CSV exists yet, just save the new one
                print(f"Creating new CSV: {CSV_FILENAME} ...")
                new_df.to_csv(CSV_FILENAME, index=False, encoding='utf-8')
                total_records = len(new_df)

            # Clear metadata records to free up memory for the next batch
            metadata_records.clear()
            
        
        if page_num < END_PAGE:
            sleep_time = random.uniform(2.0, 5.0)
            print(f"Page {page_num} complete. Sleeping for {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
        else:
            print(f"Page {page_num} complete. No more pages to scrape.")

        end_page_time = time.time()
        elapsed_page_time = end_page_time - start_page_time

        hours, remainder = divmod(elapsed_page_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"Time spent on this page: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")

driver.quit()



end_time = time.time()
elapsed_time = end_time - start_time

hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)

# --- Final Report ---
print("\n" + "="*40)
print("SCRAPING REPORT")
print("="*40)
print(f"Pages Traversed:      {pages_processed} (Pages {START_PAGE} to {END_PAGE})")
print(f"Images Downloaded:    {images_downloaded}")
print(f"Images Skipped:       {images_skipped_existing} (Metadata Updated)")
print(f"Failed Downloads:     {failed_downloads}")
print(f"Total Unique Records: {total_records}")
print(f"Total time spent: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
print("="*40)
print(f"Metadata exported to: {CSV_FILENAME}")