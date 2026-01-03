import pandas as pd

# --------------------------------------------------
# Load tables
# --------------------------------------------------
countries = pd.read_csv("countries.csv")
authorship = pd.read_csv("authorships.csv")
authors = pd.read_csv("researchers_updated.csv")

# --------------------------------------------------
# 1. Keep only valid links
# --------------------------------------------------
authorship = authorship.dropna(subset=["country_id", "researcher_id"])

# --------------------------------------------------
# 2. Join authorship → authors
# --------------------------------------------------
auth_with_authors = authorship.merge(
    authors,
    left_on="researcher_id",
    right_on="id",
    how="inner"
)

# --------------------------------------------------
# 3. Group by country and compute averages
# --------------------------------------------------
country_stats = (
    auth_with_authors
    .groupby("country_id")
    .agg(
        average_h_index=("h_index", "mean"),
        average_rii=("rii", "mean")
    )
    .reset_index()
)

# --------------------------------------------------
# 4. Merge back into country table
# --------------------------------------------------
countries = countries.merge(
    country_stats,
    left_on="id",
    right_on="country_id",
    how="left"
)

# --------------------------------------------------
# 5. SAFELY overwrite / create average columns
# --------------------------------------------------
# Handle average_h_index
if "average_h_index_y" in countries.columns:
    countries["average_h_index"] = countries["average_h_index_y"]
elif "average_h_index" not in countries.columns:
    countries["average_h_index"] = 0

# Handle average_rii
if "average_rii_y" in countries.columns:
    countries["average_rii"] = countries["average_rii_y"]
elif "average_rii" not in countries.columns:
    countries["average_rii"] = 0

# Fill missing values
countries["average_h_index"] = countries["average_h_index"].fillna(0)
countries["average_rii"] = countries["average_rii"].fillna(0)

# --------------------------------------------------
# 6. Compute ranking based on average_rii (DESC)
# --------------------------------------------------
countries["ranking"] = (
    countries["average_rii"]
    .rank(method="dense", ascending=False)
    .astype(int)
)

# --------------------------------------------------
# 7. Cleanup technical columns
# --------------------------------------------------
countries.drop(
    columns=[
        c for c in countries.columns
        if c.endswith("_x") or c.endswith("_y") or c == "country_id"
    ],
    inplace=True,
    errors="ignore"
)

# --------------------------------------------------
# 8. Save result
# --------------------------------------------------
countries.to_csv("country_updated.csv", index=False)
