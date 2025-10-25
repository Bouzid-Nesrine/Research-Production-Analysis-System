import os
import json
import time
import requests
import pandas as pd
from urllib.parse import quote

# ========= CONFIG =========
excel_path = "filtered_topics_clean.xlsx"   # Path to your Excel file
topic_column = "Unique Topics"              # Name of column with topics
output_dir = "openalex_data"                # Folder to save JSONs
max_articles_per_topic = 10_000             # Limit of works to fetch per topic
mailto = "rayane.mazrou@ensia.edu.dz"      # Your email (for User-Agent)
per_page = 200                              # OpenAlex max per page
# ==========================

os.makedirs(output_dir, exist_ok=True)

headers = {
    "User-Agent": f"Mozilla/5.0 (compatible; ResearchBot/1.0; mailto:{mailto})",
    "Accept": "application/json"
}

# Load topics
df = pd.read_excel(excel_path)
topics = df[topic_column].dropna().unique().tolist()
print(f"📘 Loaded {len(topics)} topics from Excel.")

# Normalize topic names (for comparison)
def normalize_name(name: str) -> str:
    return name.strip().lower().replace("/", "_").replace("\\", "_").replace(":", "_")

# Get list of already downloaded topics
def already_fetched_topics():
    done = []
    for f in os.listdir(output_dir):
        if f.endswith(".json"):
            done.append(normalize_name(os.path.splitext(f)[0]))
    return set(done)

done_topics = already_fetched_topics()
print(f"✅ Found {len(done_topics)} topics already downloaded. Skipping those.")

# ======= MAIN LOOP =======
for idx, topic in enumerate(topics, 1):
    safe_name = topic.replace("/", "_").replace("\\", "_").replace(":", "_")
    normalized = normalize_name(topic)
    output_path = os.path.join(output_dir, f"{safe_name}.json")

    # Skip topics that are already in folder
    if normalized in done_topics:
        print(f"⏩ Skipping already done: {topic}")
        continue

    print(f"\n🔍 Starting topic {idx}/{len(topics)}: {topic}")
    encoded = quote(topic)
    cursor = "*"
    all_results = []
    total_fetched = 0

    while True:
        url = (
            f"https://api.openalex.org/works?"
            f"search={encoded}&filter=type:article&per-page={per_page}&cursor={cursor}&mailto={mailto}"
        )

        try:
            r = requests.get(url, headers=headers, timeout=30)
            
            if r.status_code == 429:
                print("⚠️ Rate limit hit. Sleeping 60s...")
                time.sleep(60)
                continue
            elif r.status_code in [500, 502, 503, 504]:
                print(f"⚠️ Server error {r.status_code}. Waiting 30s...")
                time.sleep(30)
                continue
            elif r.status_code == 403:
                print(f"🚫 Access denied (403) for topic: {topic}")
                break

            r.raise_for_status()
            data = r.json()

            works = data.get("results", [])
            if not works:
                print("⚠️ No more results.")
                break

            all_results.extend(works)
            total_fetched += len(works)
            print(f"  ➕ Fetched {len(works)} (total {total_fetched})")

            # Stop if reached the limit
            if total_fetched >= max_articles_per_topic:
                print("✅ Reached 10,000 article limit for this topic.")
                break

            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                print("✅ Reached end of available results.")
                break

            # Sleep to respect rate limits
            time.sleep(0.2)

        except requests.exceptions.RequestException as e:
            print(f"❌ Network error for '{topic}': {e}")
            time.sleep(10)
            continue

    # Save fetched results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"💾 Saved {len(all_results)} works to {output_path}")

    # Update the list of done topics (in case of interruption later)
    done_topics.add(normalized)
    time.sleep(1.5)

print("\n🎉 Done! All topics processed or already downloaded.")

