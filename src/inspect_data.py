
import pandas as pd

# Load the original dataset
df = pd.read_csv("data/dengue_dataset.csv")

print("\n===== DATASET OVERVIEW =====")
print("Total rows and columns:", df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== BASIC STATISTICS =====")
print(df.describe(include="all"))

# ===== ADDITIONAL DATA QUALITY CHECKS =====

df["date"] = pd.to_datetime(df["date"], errors="coerce")

print("\n===== DATE RANGE =====")
print(df["date"].min(), "to", df["date"].max())

print("\n===== UNIQUE CITIES =====")
print(df["city"].unique())

print("\n===== UNIQUE WARDS =====")
print(df["ward_id"].nunique())

print("\n===== NUMERICAL VALUE RANGES =====")
numeric_cols = df.select_dtypes(include="number").columns
print(df[numeric_cols].agg(["min", "max", "mean"]))

print("\n===== NEGATIVE VALUES CHECK =====")
print((df[numeric_cols] < 0).sum())

# ===== DATA CONSISTENCY CHECKS =====

print("\n===== TEMPERATURE CONSISTENCY =====")

print("Min temp > Max temp:",
      (df["min_temp_c"] > df["max_temp_c"]).sum())

print("Avg temp outside Min-Max:",
      ((df["avg_temp_c"] < df["min_temp_c"]) |
       (df["avg_temp_c"] > df["max_temp_c"])).sum())

print("\n===== SCORE RANGE CHECK =====")

print("Waterlogging outside 0-1:",
      ((df["water_logging_risk_index"] < 0) |
       (df["water_logging_risk_index"] > 1)).sum())

print("Sanitation outside 0-1:",
      ((df["sanitation_score"] < 0) |
       (df["sanitation_score"] > 1)).sum())

print("\n===== HUMIDITY CHECK =====")

print("Humidity outside 0-100:",
      ((df["relative_humidity_pct"] < 0) |
       (df["relative_humidity_pct"] > 100)).sum())

print("\n===== DUPLICATE DATE-CITY-WARD CHECK =====")

print(df.duplicated(
    subset=["date", "city", "ward_id"]
).sum())

print("\n===== RECORDS PER CITY AND WARD =====")

print(df.groupby(["city", "ward_id"]).size())

# ===== DATE AND WEEKLY CONTINUITY CHECKS =====

print("\n===== DATE VALIDATION =====")

# Check invalid dates
print("Invalid dates:", df["date"].isna().sum())

# Check unique city-ward combinations
print("\nUnique city-ward combinations:")
print(df.groupby(["city", "ward_id"]).ngroups)

# Check records per city-ward
records = df.groupby(["city", "ward_id"]).size()

print("\nMinimum records per ward:", records.min())
print("Maximum records per ward:", records.max())

# Check weekly gaps
df_sorted = df.sort_values(["city", "ward_id", "date"])

date_gaps = (
    df_sorted.groupby(["city", "ward_id"])["date"]
    .diff()
    .dt.days
)

print("\n===== WEEKLY GAP CHECK =====")

print("Gaps not equal to 7 days:",
      ((date_gaps.notna()) & (date_gaps != 7)).sum())

print("Gaps greater than 7 days:",
      (date_gaps > 7).sum())

print("Repeated dates within city-ward:",
      df.duplicated(
          subset=["city", "ward_id", "date"]
      ).sum())
# ===== OUTLIER DETECTION =====

print("\n===== OUTLIER DETECTION =====")

numeric_cols = df.select_dtypes(include="number").columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print(f"\nColumn: {col}")
    print("Lower limit:", round(lower, 2))
    print("Upper limit:", round(upper, 2))
    print("Outlier count:", len(outliers))