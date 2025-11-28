import os
import json
import time
import requests
import pandas as pd
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import random

# ========= CONFIG =========
excel_path = "filtered_topics_clean.xlsx"
topic_column = "Unique Topics"
output_dir = "/content/drive/MyDrive/Rayane_2500_NotNull"
max_articles_per_topic = 2_500 
mailto = "rayane.mazrou@ensia.edu.dz"
per_page = 200
max_workers = 4

# Rate limiting configuration
BASE_DELAY = 0.5
RATE_LIMIT_WAIT = 60
MAX_RETRIES_429 = 5

# 🎯 UPDATED: Added referenced_works to the select fields
FIELDS_SELECT = "id,doi,title,publication_year,publication_date,display_name,authorships,cited_by_count,abstract_inverted_index,primary_location,referenced_works"

# 🎯 UPDATED: Added referenced_works to required fields
REQUIRED_FIELDS_FOR_CHECK = [
    "id",
    "doi",
    "title",
    "publication_year",
    "publication_date",
    "display_name",
    "primary_location",
    "authorships",
    "cited_by_count",
    "abstract_inverted_index",
    "referenced_works"  # NEW: Must exist and not be empty
]

REQUIRED_NESTED_FIELDS = {
    "primary_location": ["source"],
    "source": ["id", "display_name", "issn"]
}
# ==========================

os.makedirs(output_dir, exist_ok=True)

def create_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": f"Mozilla/5.0 (compatible; ResearchBot/1.0; mailto:{mailto})",
        "Accept": "application/json"
    })
    return session

# Load topics
try:
    df = pd.read_excel(excel_path)
    topics = df[topic_column].dropna().unique().tolist()
    print(f"📘 Loaded {len(topics)} topics from Excel.")
except FileNotFoundError:
    print(f"❌ ERROR: Excel file not found at '{excel_path}'")
    exit()
except KeyError:
    print(f"❌ ERROR: Column '{topic_column}' not found in the Excel file.")
    exit()

existing_files = [f for f in os.listdir(output_dir) if f.endswith(".json")]
print(f"✅ Found {len(existing_files)} topics already downloaded. They will be skipped.")

def filter_null_fields(works):
    filtered_works = []
    null_count = 0
    for work in works:
        is_valid = True
        
        # 1. Check top-level required fields
        for field in REQUIRED_FIELDS_FOR_CHECK:
            if field not in work or work[field] is None:
                is_valid = False
                break
            
            # Check for empty lists (must have at least one item)
            if isinstance(work[field], list) and not work[field]:
                is_valid = False
                break
                
            # Check for empty dictionaries
            if isinstance(work[field], dict) and not work[field]:
                is_valid = False
                break

        if not is_valid:
            null_count += 1
            continue
            
        # 2. Check nested required fields (primary_location -> source -> id, display_name, issn)
        pl = work.get("primary_location")
        if pl and pl.get("source"):
            source = pl.get("source")
            for field in REQUIRED_NESTED_FIELDS["source"]:
                if source.get(field) is None:
                    is_valid = False
                    break
                
                # 🎯 NEW: Check that issn is not an empty array
                if field == "issn":
                    issn_value = source.get(field)
                    if isinstance(issn_value, list) and not issn_value:
                        is_valid = False
                        break
        else:
            is_valid = False
        
        if is_valid:
            filtered_works.append(work)
        else:
            null_count += 1
            
    return filtered_works, null_count

def fetch_topic(topic):
    safe_name = topic.replace("/", "_").replace("\\", "_").replace(":", "_").strip()
    output_path = os.path.join(output_dir, f"{safe_name}.json")

    if os.path.exists(output_path):
        print(f"⏩ Skipping: {topic}")
        return topic, 0, "skipped"

    print(f"🔍 Starting: {topic}")
    
    session = create_session()
    encoded = quote(topic)
    cursor = "*"
    all_results = []
    total_fetched_api = 0
    total_filtered_out = 0
    consecutive_429_errors = 0

    while True:
        url = (
            f"https://api.openalex.org/works?"
            f"search={encoded}"
            f"&filter=type:article"
            f"&select={FIELDS_SELECT}"
            f"&per-page={per_page}"
            f"&cursor={cursor}"
            f"&mailto={mailto}"
        )

        try:
            r = session.get(url, timeout=30)
            
            if r.status_code == 429:
                consecutive_429_errors += 1
                if consecutive_429_errors > MAX_RETRIES_429:
                    print(f"❌ Too many rate limits for {topic}. Stopping this topic.")
                    break
                
                wait_time = RATE_LIMIT_WAIT * (1.5 ** (consecutive_429_errors - 1))
                jitter = random.uniform(0, 10)
                total_wait = min(wait_time + jitter, 300)
                
                print(f"⚠️ Rate limit #{consecutive_429_errors}: {topic}. Waiting {total_wait:.0f}s...")
                time.sleep(total_wait)
                continue
            else:
                consecutive_429_errors = 0
                
            if r.status_code in [500, 502, 503, 504]:
                print(f"⚠️ Server error {r.status_code}: {topic}. Waiting 30s...")
                time.sleep(30)
                continue
            elif r.status_code == 403:
                print(f"🚫 Access denied: {topic}")
                break
            elif r.status_code == 400:
                print(f"❌ Bad request for {topic}: {r.text}")
                break

            r.raise_for_status()
            data = r.json()

            works = data.get("results", [])
            if not works:
                break

            valid_works, null_count = filter_null_fields(works)
            total_filtered_out += null_count
            
            all_results.extend(valid_works)
            total_fetched_api += len(works)
            
            current_valid_count = len(all_results)

            if current_valid_count >= max_articles_per_topic:
                all_results = all_results[:max_articles_per_topic]
                print(f"✅ Limit reached: {topic} ({len(all_results)} valid articles)")
                break

            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break

            delay = BASE_DELAY + random.uniform(0, 0.3)
            time.sleep(delay)

        except requests.exceptions.RequestException as e:
            print(f"❌ Error: {topic} - {e}")
            time.sleep(15)
            continue

    session.close()

    final_count = len(all_results)
    if final_count > 0:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved: {topic} ({final_count} valid articles). Filtered out {total_filtered_out} due to missing/null fields.")
        return topic, final_count, "completed"
    else:
        print(f"⚠️ No valid articles found: {topic}. Filtered {total_filtered_out} total.")
        return topic, 0, "no_results"

# ======= PARALLEL EXECUTION =======
print(f"\n🚀 Starting download with {max_workers} workers...")
print(f"📝 Required top-level fields: {', '.join(REQUIRED_FIELDS_FOR_CHECK)}")
print(f"📝 Required nested fields: source(id, display_name, issn[not empty])")
print(f"📝 Base delay: {BASE_DELAY}s per request (with jitter)")
print(f"📝 Rate limit wait: {RATE_LIMIT_WAIT}s (with exponential backoff)")
print(f"Article Limit per Topic: {max_articles_per_topic:,}\n")

completed_count = 0
skipped_count = 0
total_articles = 0
errored_topics = []

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    downloaded_names = {f.rsplit(".", 1)[0] for f in existing_files}
    topics_to_process = [t for t in topics if t.replace("/", "_").replace("\\", "_").replace(":", "_").strip() not in downloaded_names]
    
    future_to_topic = {executor.submit(fetch_topic, t): t for t in topics_to_process}
    
    for future in as_completed(future_to_topic):
        topic, count, status = future.result()
        
        if status == "skipped":
            skipped_count += 1
        elif status == "completed":
            completed_count += 1
            total_articles += count
            print(f"✅ [{completed_count + skipped_count}/{len(topics)}] {topic}: {count:,} articles")
        elif status == "no_results":
            print(f"❌ [{completed_count + skipped_count}/{len(topics)}] {topic}: No valid articles found.")
            errored_topics.append(topic)

print("\n" + "=" * 60)
print("🎉 DOWNLOAD COMPLETE!")
print("=" * 60)
print(f"✅ Newly downloaded: {completed_count} topics")
print(f"⏩ Skipped (already exists): {skipped_count} topics")
print(f"❌ Topics with no valid results: {len(errored_topics)}")
print(f"📊 Total valid articles fetched: {total_articles:,}")
print(f"📁 Total topics processed: {len(topics)}")
if errored_topics:
    print(f"\n⚠️ Topics with errors: {', '.join(errored_topics[:10])}")
    if len(errored_topics) > 10:
        print(f"   ... and {len(errored_topics) - 10} more")
print("=" * 60)
