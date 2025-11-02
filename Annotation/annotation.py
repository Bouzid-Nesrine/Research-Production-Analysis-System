# =========================================================
# Hierarchical Article Classification (OpenRouter + Taxonomy)
# Stage 1: General domain classification (OpenAlex-style)
# Stage 2: Fine-grained classification within the chosen domain (returns full path)
# =========================================================

from openai import OpenAI
import json
import time
import re

# =========================================================
# CONFIGURATION
# =========================================================
OPENROUTER_API_KEY = "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"  # replace with your key
MODEL = "gpt-4o-mini"  # or "anthropic/claude-3-haiku", "meta-llama/llama-3.3-70b-instruct"

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# =========================================================
# LOAD JSON UTILITIES
# =========================================================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_taxonomy_strings(taxonomy_json):
    """Flatten nested taxonomy into a list of full path strings."""
    result = []

    def recurse(subtree, prefix=""):
        if isinstance(subtree, dict):
            for k, v in subtree.items():
                new_prefix = f"{prefix} > {k}" if prefix else k
                recurse(v, new_prefix)
        elif isinstance(subtree, list):
            for item in subtree:
                recurse(item, prefix)
        else:
            result.append(f"{prefix} > {subtree}" if prefix else subtree)

    recurse(taxonomy_json)
    return result

# =========================================================
# LOAD DATA
# =========================================================
taxonomy_json = load_json("final_deepseek.json")
taxonomy_flat = extract_taxonomy_strings(taxonomy_json)
print(f"✅ Loaded taxonomy: {len(taxonomy_flat)} categories")

articles = load_json("/content/drive/MyDrive/NLP Project/Data Collection /Rayane/AI and HR Technologies.json")
print(f"✅ Loaded {len(articles)} articles")

# =========================================================
# OPENALEX TO TAXONOMY MAP
# =========================================================
OPENALEX_TO_TAXONOMY_MAP = {
    "Computer and Information Sciences": ["Natural Science > Computer and Information Science (1.02)"],
    "Engineering and Technology": ["Engineering and Technology (2.00)"],
    "Health Sciences": ["Health Science > Medicine (3.01)"],
    "Social Sciences": ["Social Science (5.00)"],
    "Humanities": ["Humanities (6.00)"],
    "Business and Management": ["Social Science > Business and Management (5.02)"],
    "Economics": ["Social Science > Economics (5.01)"],
    "Psychology": ["Social Science > Psychology (5.03)"],
    "Environmental Science": ["Natural Science > Environmental Science (1.05)"],
    "Mathematics": ["Natural Science > Mathematics (1.01)"],
}

# =========================================================
# STAGE 1: GENERAL DOMAIN CLASSIFICATION
# =========================================================
def classify_general_domain(title, abstract):
    messages = [
        {"role": "system", "content": """
You are an expert in academic article classification.
Classify each article into one of these broad OpenAlex domains:
- Natural Science
- Engineering and Technology
- Health Sciences
- Social Sciences
- Humanities

Return strictly JSON:
{"domain": "...", "reason": "..."}
"""},

        {"role": "user", "content": f"TITLE: {title}\nABSTRACT: {abstract}"}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=200
        )
        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        return result.get("domain", "Unknown"), result.get("reason", "")
    except Exception as e:
        print(f"⚠️ Error in general classification: {e}")
        return "Unknown", "Error"

# =========================================================
# STAGE 2: FINE TAXONOMY CLASSIFICATION (returns full path)
# =========================================================
def classify_fine_category(title, abstract, domain, taxonomy_flat):
    relevant_prefixes = OPENALEX_TO_TAXONOMY_MAP.get(domain, [])
    if relevant_prefixes:
        filtered_taxonomy = [node for node in taxonomy_flat if any(prefix in node for prefix in relevant_prefixes)]
    else:
        filtered_taxonomy = taxonomy_flat

    taxonomy_segment = "\n".join(filtered_taxonomy[:200])

    messages = [
        {"role": "system", "content": f"""
You are an expert academic classifier.
Classify the article into ONE of the taxonomy paths below (each is a full hierarchy):

{taxonomy_segment}

Rules:
- Choose the FULL taxonomy path (not just the leaf).
- Always return exactly one path string from the list above.
- Prefer deeper (more specific) taxonomy paths when relevant.
- Focus on meaning, not keyword matching.
- Output strictly valid JSON:
{{"path": "...", "reason": "..."}}
"""},

        {"role": "user", "content": f"""
TITLE: {title}
ABSTRACT: {abstract}
Domain: {domain}
"""}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=350
        )
        content = response.choices[0].message.content.strip()

        try:
            fine = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            fine = json.loads(match.group(0)) if match else {"path": "Unclassified", "reason": "Parse error"}

        return fine.get("path", "Unclassified"), fine.get("reason", "")
    except Exception as e:
        print(f"⚠️ Error in fine classification: {e}")
        return "Unclassified", "Error"

# =========================================================
# MAIN LOOP
# =========================================================
results = []
print("\n🚀 Starting Hierarchical Classification...\n")

for i, article in enumerate(articles[:50], 1):  # limit to 50 for testing
    title = article.get("title", "")
    abstract = article.get("abstract", "")

    print(f"→ [{i}/{len(articles)}] {title[:70]}...")

    # Stage 1
    general_domain, reason_general = classify_general_domain(title, abstract)
    print(f"  🌍 General Domain: {general_domain}")

    # Stage 2
    fine_path, reason_fine = classify_fine_category(title, abstract, general_domain, taxonomy_flat)
    print(f"  ✅ Full Path: {fine_path}")
    print(f"  💭 Reason: {reason_fine[:120]}...\n")

    results.append({
        "title": title,
        "general_domain": general_domain,
        "fine_category_path": fine_path,
        "reason_general": reason_general,
        "reason_fine": reason_fine
    })

    time.sleep(1)  # avoid rate limits

# =========================================================
# SAVE OUTPUT
# =========================================================
with open("classified_articles.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("✅ Classification complete! Results saved to classified_articles.json")

