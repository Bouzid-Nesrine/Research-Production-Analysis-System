import pandas as pd

# --------------------------------------------------
# Load tables
# --------------------------------------------------
countries = pd.read_csv("country_updated.csv")
authorship = pd.read_csv("authorships.csv")

# --------------------------------------------------
# 1. Keep only valid country links
# --------------------------------------------------
authorship = authorship.dropna(
    subset=["country_id", "researcher_id", "article_id"]
)

# --------------------------------------------------
# 2. Compute totals per country
# --------------------------------------------------
country_totals = (
    authorship
    .groupby("country_id")
    .agg(
        total_researchers=("researcher_id", "nunique"),
        total_publications=("article_id", "nunique")
    )
    .reset_index()
)

# --------------------------------------------------
# 3. Merge totals into country table
# --------------------------------------------------
countries = countries.merge(
    country_totals,
    left_on="id",
    right_on="country_id",
    how="left"
)

# --------------------------------------------------
# 4. SAFELY overwrite totals (IMPORTANT)
# --------------------------------------------------

# total_researchers
if "total_researchers_y" in countries.columns:
    countries["total_researchers"] = countries["total_researchers_y"]
elif "total_researchers" not in countries.columns:
    countries["total_researchers"] = 0

# total_publications
if "total_publications_y" in countries.columns:
    countries["total_publications"] = countries["total_publications_y"]
elif "total_publications" not in countries.columns:
    countries["total_publications"] = 0

countries["total_researchers"] = (
    countries["total_researchers"].fillna(0).astype(int)
)
countries["total_publications"] = (
    countries["total_publications"].fillna(0).astype(int)
)

# --------------------------------------------------
# 5. Cleanup technical columns
# --------------------------------------------------
countries.drop(
    columns=[c for c in countries.columns if c.endswith("_x") or c.endswith("_y") or c == "country_id"],
    inplace=True,
    errors="ignore"
)

# --------------------------------------------------
# 6. Save result
# --------------------------------------------------
countries.to_csv("countries_updated.csv", index=False)

print("✅ Country totals computed successfully")
