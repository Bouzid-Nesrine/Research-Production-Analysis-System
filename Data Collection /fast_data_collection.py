import os
import json
import time
import requests
import pandas as pd
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ========= CONFIG =========
excel_path = "filtered_topics_clean.xlsx"
topic_column = "Unique Topics"
output_dir = "openalex_data"
max_articles_per_topic = 10_000
mailto = "rayane.mazrou@ensia.edu.dz"
per_page = 200  # OpenAlex max per page
max_workers = 7  # Increased for faster downloads
# ==========================

os.makedirs(output_dir, exist_ok=True)

# Session with connection pooling and retries
def create_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": f"Mozilla/5.0 (compatible; ResearchBot/1.0; mailto:{mailto})",
        "Accept": "application/json"
    })
    return session

# Load topics
df = pd.read_excel(excel_path)
topics = df[topic_column].dropna().unique().tolist()
print(f"📘 Loaded {len(topics)} topics from Excel.")

existing_files = [f for f in os.listdir(output_dir) if f.endswith(".json")]
print(f"✅ Found {len(existing_files)} topics already downloaded. They will be skipped.")

def fetch_topic(topic):
    safe_name = topic.replace("/", "_").replace("\\", "_").replace(":", "_")
    output_path = os.path.join(output_dir, f"{safe_name}.json")

    if os.path.exists(output_path):
        print(f"⏩ Skipping: {topic}")
        return topic, 0, "skipped"

    print(f"🔍 Starting: {topic}")
    
    session = create_session()
    encoded = quote(topic)
    cursor = "*"
    all_results = []
    total_fetched = 0

    while True:
        # ✅ ONLY ARTICLES FILTER ADDED HERE
        url = (
            f"https://api.openalex.org/works?"
            f"search={encoded}"
            f"&filter=type:article"  # 🎯 Filter for articles only
            f"&per-page={per_page}"
            f"&cursor={cursor}"
            f"&mailto={mailto}"
        )

        try:
            r = session.get(url, timeout=30)

            if r.status_code == 429:
                print(f"⚠️ Rate limit: {topic}. Waiting 60s...")
                time.sleep(60)
                continue
            elif r.status_code in [500, 502, 503, 504]:
                print(f"⚠️ Server error {r.status_code}: {topic}. Waiting 30s...")
                time.sleep(30)
                continue
            elif r.status_code == 403:
                print(f"🚫 Access denied: {topic}")
                break

            r.raise_for_status()
            data = r.json()

            works = data.get("results", [])
            if not works:
                break

            all_results.extend(works)
            total_fetched += len(works)

            if total_fetched >= max_articles_per_topic:
                print(f"✅ Limit reached: {topic} ({total_fetched} articles)")
                break

            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break

            time.sleep(0.1)  # Reduced delay for faster fetching

        except requests.exceptions.RequestException as e:
            print(f"❌ Error: {topic} - {e}")
            time.sleep(10)
            continue

    session.close()

    # Save results
    if all_results:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved: {topic} ({len(all_results)} articles)")
        return topic, len(all_results), "completed"
    else:
        print(f"⚠️ No articles found: {topic}")
        return topic, 0, "no_results"


# ======= PARALLEL EXECUTION =======
print(f"\n🚀 Starting download with {max_workers} workers...\n")

completed_count = 0
skipped_count = 0
total_articles = 0

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_topic = {executor.submit(fetch_topic, t): t for t in topics}
    
    for future in as_completed(future_to_topic):
        topic, count, status = future.result()
        
        if status == "skipped":
            skipped_count += 1
        elif status == "completed":
            completed_count += 1
            total_articles += count
            print(f"✅ [{completed_count + skipped_count}/{len(topics)}] {topic}: {count:,} articles")

print("\n" + "=" * 60)
print("🎉 DOWNLOAD COMPLETE!")
print("=" * 60)
print(f"✅ Newly downloaded: {completed_count} topics")
print(f"⏩ Skipped (already exists): {skipped_count} topics")
print(f"📊 Total articles fetched: {total_articles:,}")
print(f"📁 Total topics processed: {len(topics)}")
print("=" * 60)
