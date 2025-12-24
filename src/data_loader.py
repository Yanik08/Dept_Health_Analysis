from pathlib import Path
from typing import Tuple
import pandas as pd
def impute_country_means(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    If a country has at least 1 non-missing value for a variable,
    missing values are filled with its country mean.
    If a country has all values missing for that variable,
    the values remain NaN (and those rows may later drop out in training).
    """
    df_filled = df.copy()

    for col in feature_cols:
        if col not in df_filled.columns:
            continue  # skip if column not present

        # Country-specific means for this variable
        country_means = df_filled.groupby("country_code")[col].transform("mean")

        # Fill NaNs with the country mean
        df_filled[col] = df_filled[col].fillna(country_means)

    return df_filled

#### Raw data loading functions:

def load_raw_weo(path: Path) -> pd.DataFrame:
    """
    Load the raw IMF WEO dataset (CSV or Excel).
    """
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def load_raw_crisis(path: Path) -> pd.DataFrame:
    """
    Load the raw crisis Excel file.
    """
    return pd.read_excel(path)


#### Panel building functions:

def build_weo_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build a country-year macro panel from the raw IMF WEO file.
    """
    # Split SERIES_CODE into country, indicator, frequency
    parts = raw["SERIES_CODE"].str.split(".", n=2, expand=True, regex=False)
    raw["country_code"] = parts[0]
    raw["indicator"] = parts[1]
    raw["freq"] = parts[2]

    # Indicators we keep
    keep = [
        "GGXWDG_NGDP",  # Debt %GDP
        "GGXINT_NGDP",  # Interest payments %GDP
        "GGXONLB_NGDP", # Balance/primary balance %GDP
        "NGDP_RPCH",    # GDP growth
        "PCPIPCH",      # Inflation
        "LUR",          # Unemployment
        "NGDPD",        # Nominal GDP
        "LP",           # Population
        "BCA_NGDPD",    # Current account
    ]

    df = raw[raw["indicator"].isin(keep)].copy()

    # Year columns (1980, 1981, ...)
    year_cols = [c for c in df.columns if c.isdigit()]

    # Long format: one row per country-indicator-year
    long = df.melt(
        id_vars=["country_code", "indicator", "freq"],
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    )

    long["year"] = long["year"].astype(int)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    # Wide panel: one row per (country_code, year)
    panel = (
        long.pivot_table(
            index=["country_code", "year"],
            columns="indicator",
            values="value",
        )
        .reset_index()
    )
    panel.columns.name = None

    return panel

#### Crisis panel building function:

def build_crisis_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build a clean crisis panel from the raw global crisis Excel file.
    """
    crisis = raw.copy()

    crisis = crisis.rename(
        columns={
            "CC3": "country_code",
            "Year": "year",
        }
    )

    # Rename crisis columns by position (based on your Excel layout)
    ext1_col = crisis.columns[3]
    ext2_col = crisis.columns[4]
    dom_col = crisis.columns[5]
    cur_col = crisis.columns[6]
    inf_col = crisis.columns[7]

    crisis = crisis.rename(
        columns={
            ext1_col: "external_default_1",
            ext2_col: "external_default_2",
            dom_col: "domestic_default",
            cur_col: "currency_crisis",
            inf_col: "inflation_crisis",
        }
    )

    crisis = crisis.dropna(subset=["year"])
    crisis["year"] = crisis["year"].astype(int)

    keep_cols = [
        "country_code",
        "year",
        "external_default_1",
        "external_default_2",
        "domestic_default",
        "currency_crisis",
        "inflation_crisis",
    ]
    crisis = crisis[keep_cols].copy()

    # Make all crisis indicators numeric 0/1
    crisis_cols = keep_cols[2:]
    for c in crisis_cols:
        crisis[c] = pd.to_numeric(crisis[c], errors="coerce")
        crisis[c] = crisis[c].fillna(0).astype(int)

    sovereign_cols = ["external_default_1", "external_default_2", "domestic_default"]
    crisis["sovereign_crisis"] = (crisis[sovereign_cols] > 0).any(axis=1).astype(int)

    return crisis


#### Master function to build and save panels:

def build_and_save_panels(base_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    #1# WEO panel
    raw_weo_path = raw_dir / "IMF_WEO_dataset.csv"
    raw_weo = load_raw_weo(raw_weo_path)
    weo_panel = build_weo_panel(raw_weo)
    weo_out = processed_dir / "weo_panel.csv"
    weo_panel.to_csv(weo_out, index=False)
    print(f"Saved WEO panel to: {weo_out}")

    #2# Crisis panel
    raw_crisis_path = raw_dir / "global_crisis_data.xlsx"
    raw_crisis = load_raw_crisis(raw_crisis_path)
    crisis_panel = build_crisis_panel(raw_crisis)

    #3# Merge WEO + crisis
    merged = pd.merge(
        weo_panel,
        crisis_panel,
        on=["country_code", "year"],
        how="inner")
    print(f"Merged panel shape before imputation: {merged.shape}")

    #4# Take care of missing values in macrovariables with country mean imputation:
    macro_vars = [
        "GGXWDG_NGDP",
        "GGXONLB_NGDP",
        "NGDP_RPCH",
        "PCPIPCH",
        "LUR",
        "NGDPD",
        "LP",
        "BCA_NGDPD",
    ]
    print( "\n Imputing missing macroeconomic values with country means...")
    merged = impute_country_means(merged, macro_vars)
    print(" Imputation done.")
    # A lot of missinf values for unemployment rate remain, so as a last resort I fill those with global mean:
    merged[macro_vars] = merged[macro_vars].fillna(merged[macro_vars].mean())
    print("Global-mean fallback imputation completed.")

    #5# Create future crisis indicators (within 3, 5, 10 years)
    merged = merged.sort_values(["country_code", "year"])
    old_h_cols = [c for c in merged.columns if c.startswith("crisis_h")]
    if old_h_cols:
        merged = merged.drop(columns=old_h_cols)

    horizons = [3, 5, 10]
    for h in horizons:
        future = [
            merged.groupby("country_code")["sovereign_crisis"].shift(-k)
            for k in range(1, h + 1)
        ]
        merged[f"crisis_h{h}"] = pd.concat(future, axis=1).max(axis=1)

    # quick check (after the loop!)
    print("Horizon cols now:", [c for c in merged.columns if c.startswith("crisis_h")])
    
    merged_out = processed_dir / "merged_weo_crisis.csv"
    merged.to_csv(merged_out, index=False)
    print(f"Saved merged panel to: {merged_out}")

    return weo_panel, merged

