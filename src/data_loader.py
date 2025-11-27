from pathlib import Path
import pandas as pd

#AI helped whith importing excel file
def load_raw_crisis(path: Path) -> pd.DataFrame:
    """
    Load the raw crisis Excel file.

    Parameters
    ----------
    path : Path
        Path to the Excel file.
    
    Returns
    -------
    pd.DataFrame
        Raw crisis_data.
    """
    return pd.read_excel(path)


def load_raw_weo(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


def build_weo_panel(raw: pd.DataFrame) -> pd.DataFrame:
    # Split country / indicator
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

    # Year columns
    years = [col for col in df.columns if col.isdigit()]

    # Reshape
    long = df.melt(
        id_vars=["country_code", "indicator", "freq"],
        value_vars=years,
        var_name="year",
        value_name="value"
    )

    long["year"] = long["year"].astype(int)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    # Pivot
    panel = long.pivot_table(
        index=["country_code", "year"],
        columns="indicator",
        values="value"
    ).reset_index()

    panel.columns.name = None  # remove the pivot table column grouping name
    return panel