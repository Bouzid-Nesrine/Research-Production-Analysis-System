import pandas as pd

# --------------------------------------------------
# Load articles table
# --------------------------------------------------
articles = pd.read_csv("articles.csv")

# --------------------------------------------------
# Replace / with >
# --------------------------------------------------
articles["researcher_area_path"] = (
    articles["researcher_area_path"]
    .fillna("")
    .str.replace("/", " > ", regex=False)
)

# --------------------------------------------------
# Save result
# --------------------------------------------------
articles.to_csv("articles_updated.csv", index=False)

print("✅ researcher_area_path updated successfully")
