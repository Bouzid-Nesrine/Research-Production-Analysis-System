import pandas as pd

# --------------------------------------------------
# Load tables
# --------------------------------------------------
institutions = pd.read_csv("institutions_updated.csv")
authorship = pd.read_csv("authorships.csv")

# --------------------------------------------------
# 1. Keep only valid institution links
# --------------------------------------------------
authorship = authorship.dropna(
    subset=["institution_id", "researcher_id", "article_id"]
)

# --------------------------------------------------
# 2. Compute totals per institution
# --------------------------------------------------
institution_totals = (
    authorship
    .groupby("institution_id")
    .agg(
        total_researchers=("researcher_id", "nunique"),
        total_publications=("article_id", "nunique")
    )
    .reset_index()
)

# --------------------------------------------------
# 3. Merge totals into institution table
# --------------------------------------------------
institutions = institutions.merge(
    institution_totals,
    left_on="id",
    right_on="institution_id",
    how="left"
)

# --------------------------------------------------
# 4. SAFELY overwrite totals
# --------------------------------------------------

# total_researchers
if "total_researchers_y" in institutions.columns:
    institutions["total_researchers"] = institutions["total_researchers_y"]
elif "total_researchers" not in institutions.columns:
    institutions["total_researchers"] = 0

# total_publications
if "total_publications_y" in institutions.columns:
    institutions["total_publications"] = institutions["total_publications_y"]
elif "total_publications" not in institutions.columns:
    institutions["total_publications"] = 0

institutions["total_researchers"] = (
    institutions["total_researchers"].fillna(0).astype(int)
)
institutions["total_publications"] = (
    institutions["total_publications"].fillna(0).astype(int)
)

# --------------------------------------------------
# 5. Cleanup technical columns
# --------------------------------------------------
institutions.drop(
    columns=[c for c in institutions.columns if c.endswith("_x") or c.endswith("_y") or c == "institution_id"],
    inplace=True,
    errors="ignore"
)

# --------------------------------------------------
# 6. Save result
# --------------------------------------------------
institutions.to_csv("institutions.csv", index=False)

print("✅ Institution totals computed successfully")
