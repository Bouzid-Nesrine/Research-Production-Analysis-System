import pandas as pd
import requests
import time
from urllib.parse import quote

# Load your Excel file
df = pd.read_excel("/content/filtered_topics_clean.xlsx")

topic_column = "Unique Topics"  # adjust to your column name

# Improved headers with proper contact info
headers = {
    "User-Agent": "Mozilla/5.0 (compatible; AcademicResearch/1.0; mailto:rayane.mazrou@ensia.edu.dz)",
    "Accept": "application/json"
}

results = []
batch_size = 50  # Save every 50 topics to avoid losing progress

for idx, topic in enumerate(df[topic_column].dropna().unique(), 1):
    try:
        # Use search parameter instead of filter for better compatibility
        # Encode the topic properly
        encoded_topic = quote(topic)
        url = f"https://api.openalex.org/works?search={encoded_topic}&per_page=1"
        
        response = requests.get(url, headers=headers, timeout=10)
        
        # Check for rate limiting
        if response.status_code == 429:
            print(f"Rate limited. Waiting 60 seconds...")
            time.sleep(60)
            response = requests.get(url, headers=headers, timeout=10)
        
        response.raise_for_status()
        
        data = response.json()
        total_articles = data.get("meta", {}).get("count", 0)
        
        print(f"[{idx}] {topic}: {total_articles} articles found")
        
        results.append({
            "Topic": topic,
            "Total Articles": total_articles
        })
        
        # Longer delay to respect rate limits (max 10 requests per second = 100ms minimum)
        # We use 2 seconds to be safe
        time.sleep(2)
        
        # Save progress periodically
        if idx % batch_size == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_excel(f"topics_progress_{idx}.xlsx", index=False)
            print(f"💾 Progress saved at {idx} topics")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data for '{topic}': {e}")
        results.append({
            "Topic": topic,
            "Total Articles": "Error"
        })
        # Wait longer after an error
        time.sleep(5)
    except Exception as e:
        print(f"❌ Unexpected error for '{topic}': {e}")
        results.append({
            "Topic": topic,
            "Total Articles": "Error"
        })

# Save the final results
output_df = pd.DataFrame(results)
output_df.to_excel("topics_with_article_counts_final.xlsx", index=False)

print(f"\n Done! Processed {len(results)} topics")
print(f"Results saved to 'topics_with_article_counts_final.xlsx'")
print(f"Success rate: {len([r for r in results if r['Total Articles'] != 'Error'])}/{len(results)}")
