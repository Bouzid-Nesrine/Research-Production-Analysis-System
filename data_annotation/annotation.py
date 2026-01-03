from openai import OpenAI
import json, time, re, difflib

# =========================================================
# CONFIGURATION
# =========================================================
OPENROUTER_API_KEY = "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"
MODEL = "gpt-4o-mini"

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# =========================================================
# JSON UTILITIES
# =========================================================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_leaf_paths(subtree, prefix=""):
    """Return only full taxonomy paths ending at leaf nodes."""
    leaves = []
    if isinstance(subtree, dict):
        for k, v in subtree.items():
            new_prefix = f"{prefix} > {k}" if prefix else k
            leaves.extend(extract_leaf_paths(v, new_prefix))
    elif isinstance(subtree, list):
        for item in subtree:
            leaf_path = f"{prefix} > {item}" if prefix else item
            leaves.append(leaf_path)
    else:
        leaves.append(f"{prefix} > {subtree}" if prefix else subtree)
    return leaves

def closest_match(predicted_path, valid_paths):
    """Return the closest valid taxonomy path if an invalid one was generated."""
    matches = difflib.get_close_matches(predicted_path, valid_paths, n=1, cutoff=0.3)
    return matches[0] if matches else "Unclassified"

# =========================================================
# LOAD DATA
# =========================================================
taxonomy_json = load_json("final_deepseek.json")["taxonomy"]
articles = load_json(r"D:\NLP_DATA\openalex_data\AI and HR Technologies.json")

broad_domains = list(taxonomy_json.keys())
print(f"✅ Loaded taxonomy: {len(broad_domains)} top-level domains")
print(f"✅ Loaded {len(articles)} articles")

# =========================================================
# STAGE 1 — BROAD DOMAIN CLASSIFICATION
# =========================================================
def classify_broad_domain(title, abstract):
    domain_list = "\n".join(f"- {d}" for d in broad_domains)

    messages = [
        {
            "role": "system",
            "content": f"""
You are an expert academic classifier.
Classify the following article into EXACTLY ONE of these broad research domains
(from the custom taxonomy):

{domain_list}

Return strictly valid JSON:
{{"domain": "...", "reason": "..."}}
"""
        },
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
        print(f"⚠️ Error in broad classification: {e}")
        return "Unknown", "Error"

# =========================================================
# STAGE 2 — FINE CLASSIFICATION (STRICT LEAF SELECTION)
# =========================================================
def classify_fine_category(title, abstract, domain, taxonomy_json):
    if domain not in taxonomy_json:
        return "Unclassified", "Domain not found in taxonomy"

    domain_subtree = taxonomy_json[domain]
    leaf_paths = extract_leaf_paths(domain_subtree)

    messages = [
        {
            "role": "system",
            "content": f"""
You are an expert in academic taxonomy classification.

Below is the COMPLETE taxonomy for the domain "{domain}".
Each entry represents a FULL hierarchical path down to a LEAF node.

You MUST select EXACTLY ONE path that exists in the provided list.

Rules:
- You are FORBIDDEN from inventing, modifying, or rewording any label.
- Only pick a path that EXACTLY matches one from the list below.
- If multiple paths seem relevant, choose the most specific one.
- If none fit perfectly, select the closest one conceptually.
- Return strictly valid JSON: {{"path": "...", "reason": "..."}}.

Here are the available leaf paths:
{json.dumps(leaf_paths[:700], ensure_ascii=False, indent=2)}
"""
        },
        {"role": "user", "content": f"TITLE: {title}\nABSTRACT: {abstract}\nDOMAIN: {domain}"}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=400
        )
        content = response.choices[0].message.content.strip()

        # Try to parse JSON response
        try:
            fine = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            fine = json.loads(match.group(0)) if match else {"path": "Unclassified", "reason": "Parse error"}

        predicted = fine.get("path", "").strip()
        reason = fine.get("reason", "")

        # Validate: ensure it's a valid path from taxonomy
        if predicted not in leaf_paths:
            corrected = closest_match(predicted, leaf_paths)
            if corrected != "Unclassified":
                reason += f" | Adjusted to closest valid path: {corrected}"
            predicted = corrected

        return predicted, reason
    except Exception as e:
        print(f"⚠️ Error in fine classification: {e}")
        return "Unclassified", "Error"

# =========================================================
# MAIN LOOP
# =========================================================
results = []
print("\n🚀 Starting Full-Depth Hierarchical Classification (Strict Mode)...\n")

for i, article in enumerate(articles[:50], 1):  # limit to 50 for testing
    title = article.get("title", "")
    abstract = article.get("abstract", "")

    print(f"→ [{i}/{len(articles)}] {title[:70]}...")

    # Stage 1: Broad domain
    domain, reason_domain = classify_broad_domain(title, abstract)
    print(f"  🌍 Broad Domain: {domain}")

    # Stage 2: Fine-grained leaf-level
    fine_path, reason_fine = classify_fine_category(title, abstract, domain, taxonomy_json)
    print(f"  ✅ Full Path (Leaf): {fine_path}")
    print(f"  💭 Reason: {reason_fine[:120]}...\n")

    results.append({
        "title": title,
        "broad_domain": domain,
        "fine_category_path": fine_path,
        "reason_domain": reason_domain,
        "reason_fine": reason_fine
    })

    time.sleep(1)  # avoid rate limits

# =========================================================
# SAVE OUTPUT
# =========================================================
with open("classified_articles_validated.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("✅ Classification complete! Saved to classified_articles_validated.json")
