import pandas as pd
import json
from rapidfuzz import process, fuzz
from collections import defaultdict
import math 
import os
import glob
import time # For timing the process

# --- Configuration ---
CSV_PATH = "/content/all_journals_sjr_unique.csv"
JSON_DIRECTORY = "/content/drive/MyDrive/Rayane_2500_NotNull/" # Assuming all your JSON files are in the content directory
DONE_FILE_PATH = "/content/processed_files.txt"
# ---------------------

def load_and_build_indices(csv_path):
    """Loads the CSV and builds the required lookup dictionaries."""
    print("⏳ Loading CSV and building indices...")
    df = pd.read_csv(csv_path)

    # ISSN -> (title, sjr) dictionary
    issn_to_info = {}
    # Title -> (Title, SJR) dictionary
    csv_titles_info = {}
    
    for _, row in df.iterrows():
        sjr_value_raw = row.get('SJR')
        title = row['Title'].strip()
        
        journal_info = {
            'Title': title,
            'SJR': sjr_value_raw
        }
        
        # Build ISSN index
        if pd.notna(row['ISSN']):
            for issn in str(row['ISSN']).split(','):
                clean_issn = issn.replace('-', '').strip()
                if clean_issn:
                    issn_to_info[clean_issn] = journal_info

        # Build Exact Title index
        if title:
            csv_titles_info[title.lower()] = journal_info
    
    # We keep a simple list of titles for fuzzy candidates lookup
    csv_titles_lower = {k: v['Title'] for k, v in csv_titles_info.items()} 

    # Fuzzy index: by first letter for speed
    fuzzy_index = defaultdict(list)
    for t in df['Title'].dropna():
        key = t[0].lower() if t else '_'
        fuzzy_index[key].append(t.strip())
        
    print("✅ Indices built successfully.")
    return issn_to_info, csv_titles_info, csv_titles_lower, fuzzy_index

def get_processed_files(done_file_path):
    """Loads the set of filenames that have already been processed."""
    if not os.path.exists(done_file_path):
        return set()
    with open(done_file_path, 'r') as f:
        # Use strip() to remove newline characters
        return set(line.strip() for line in f)

def record_processed_file(done_file_path, filename):
    """Appends the filename to the processed file list."""
    with open(done_file_path, 'a') as f:
        f.write(filename + '\n')

def enrich_json_file(file_path, issn_to_info, csv_titles_info, csv_titles_lower, fuzzy_index):
    """Loads a single JSON file, enriches it, and overwrites the file."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        topics = json.load(f)

    enriched_topics = []
    matches_found = 0
    
    for topic in topics:
        primary_location = topic.get('primary_location', {})
        source = primary_location.get('source', {})
        
        # Clean ISSNs
        topic_issns_raw = source.get('issn', [])
        topic_issns = [topic_issns_raw] if isinstance(topic_issns_raw, str) else (topic_issns_raw if isinstance(topic_issns_raw, list) else [])
        topic_issns_clean = [issn.replace('-', '').strip() for issn in topic_issns if issn]

        original_title = source.get('display_name', '').strip()
        original_title_lower = original_title.lower()

        matched_info = None

        # Matching Logic (Same as before)
        # 1️⃣ ISSN match
        for issn in topic_issns_clean:
            if issn in issn_to_info:
                matched_info = issn_to_info[issn]
                break

        # 2️⃣ Exact title match
        if not matched_info and original_title_lower in csv_titles_info:
            matched_info = csv_titles_info[original_title_lower]

        # 3️⃣ Fuzzy matching
        if not matched_info and original_title:
            key = original_title[0].lower() if original_title else '_'
            candidates_titles = fuzzy_index.get(key, list(csv_titles_lower.values()))
            
            match = process.extractOne(
                original_title,
                candidates_titles,
                scorer=fuzz.token_sort_ratio
            )
            
            # Retrieve the full info based on the matched title
            matched_title = match[0] 
            for lower_title, info in csv_titles_info.items():
                if info['Title'] == matched_title:
                    matched_info = info
                    break

        # Enrichment Logic
        enriched_topic = topic.copy() 
        
        if matched_info:
            sjr_value_raw = matched_info.get('SJR')
            sjr_float = None

            # --- ROBUST STRING-TO-FLOAT CONVERSION ---
            if isinstance(sjr_value_raw, str):
                try:
                    sjr_float = float(sjr_value_raw.replace(',', '.'))
                except ValueError:
                    sjr_float = None 
            elif pd.notna(sjr_value_raw) and isinstance(sjr_value_raw, (int, float)):
                sjr_float = float(sjr_value_raw)

            # ADD SJR attribute (Using the float value if successful)
            enriched_topic['SJR'] = sjr_float if sjr_float is not None else sjr_value_raw 
            
            # Calculate SJR_Normalized
            sjr_normalized = None
            if sjr_float is not None and sjr_float >= 0:
                sjr_normalized = math.log1p(sjr_float) 
            
            # ADD SJR_Normalized attribute
            enriched_topic['SJR_Normalized'] = sjr_normalized
            
            # Count successful enrichment
            if 'SJR' in enriched_topic:
                 matches_found += 1
            
        enriched_topics.append(enriched_topic)

    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_topics, f, ensure_ascii=False, indent=4)
        
    return len(enriched_topics), matches_found

# =========================================================================
# === MAIN EXECUTION BLOCK ===
# =========================================================================

# 1. Load CSV indices once (for speed)
issn_to_info, csv_titles_info, csv_titles_lower, fuzzy_index = load_and_build_indices(CSV_PATH)

# 2. Get list of all JSON files
json_files = glob.glob(os.path.join(JSON_DIRECTORY, "*.json"))
total_files = len(json_files)

# 3. Get list of already processed files
processed_files_set = get_processed_files(DONE_FILE_PATH)

print(f"\n📁 Found {total_files} JSON files in the directory.")
print(f"⏩ Skipping {len(processed_files_set)} previously processed files.")
print("--- Starting Batch Enrichment ---")

start_time = time.time()
files_processed_count = 0
total_articles_enriched = 0

for i, file_path in enumerate(json_files):
    file_name = os.path.basename(file_path)
    
    if file_name in processed_files_set:
        continue # Skip already done file

    try:
        articles_count, enriched_count = enrich_json_file(
            file_path, 
            issn_to_info, 
            csv_titles_info, 
            csv_titles_lower, 
            fuzzy_index
        )
        
        # Record successful processing
        record_processed_file(DONE_FILE_PATH, file_name)
        
        files_processed_count += 1
        total_articles_enriched += enriched_count
        
        print(f"✅ [{i+1}/{total_files}] Processed **{file_name}**. Articles: {articles_count}. Enriched: {enriched_count}.")

    except Exception as e:
        print(f"❌ [{i+1}/{total_files}] ERROR processing **{file_name}**: {e}")
        # Continue to the next file if one fails

end_time = time.time()
duration = end_time - start_time

print("\n--- Batch Processing Summary ---")
print(f"⏱️ Total time taken: {duration:.2f} seconds.")
print(f"📊 Files processed in this session: {files_processed_count}")
print(f"📈 Total articles enriched in this session: {total_articles_enriched}")
print(f"📝 The file `{DONE_FILE_PATH}` now tracks all completed files.")
