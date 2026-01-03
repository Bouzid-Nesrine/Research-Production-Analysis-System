import pandas as pd

# --------------------------------------------------
# Load tables
# --------------------------------------------------
institutions = pd.read_csv("institutions.csv")
authorship = pd.read_csv("authorships.csv")
authors = pd.read_csv("researchers_updated.csv")

# --------------------------------------------------
# 1. Keep only valid links
# --------------------------------------------------
authorship = authorship.dropna(subset=["institution_id", "researcher_id"])

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
# 3. Group by institution and compute averages
# --------------------------------------------------
institution_stats = (
    auth_with_authors
    .groupby("institution_id")
    .agg(
        average_h_index=("h_index", "mean"),
        average_rii=("rii", "mean")
    )
    .reset_index()
)

# --------------------------------------------------
# 4. Merge back into institution table
# --------------------------------------------------
institutions = institutions.merge(
    institution_stats,
    left_on="id",
    right_on="institution_id",
    how="left"
)

# --------------------------------------------------
# 5. Safely overwrite / create average columns
# --------------------------------------------------
if "average_h_index_y" in institutions.columns:
    institutions["average_h_index"] = institutions["average_h_index_y"]
elif "average_h_index" not in institutions.columns:
    institutions["average_h_index"] = 0

if "average_rii_y" in institutions.columns:
    institutions["average_rii"] = institutions["average_rii_y"]
elif "average_rii" not in institutions.columns:
    institutions["average_rii"] = 0

institutions["average_h_index"] = institutions["average_h_index"].fillna(0)
institutions["average_rii"] = institutions["average_rii"].fillna(0)

# --------------------------------------------------
# 6. Compute ranking based on average_rii (DESC)
# --------------------------------------------------
institutions["ranking"] = (
    institutions["average_rii"]
    .rank(method="dense", ascending=False)
    .astype(int)
)

# --------------------------------------------------
# 7. Cleanup technical columns
# --------------------------------------------------
institutions.drop(
    columns=[
        c for c in institutions.columns
        if c.endswith("_x") or c.endswith("_y") or c == "institution_id"
    ],
    inplace=True,
    errors="ignore"
)

# --------------------------------------------------
# 8. Save result
# --------------------------------------------------
institutions.to_csv("institutions_updated.csv", index=False)
