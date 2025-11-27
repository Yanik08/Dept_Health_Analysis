from pathlib import Path
import pandas as pd

def build_crisis_panel(path: Path) -> pd.DataFrame:
    
    crisis = pd.read_excel(path)

    crisis = crisis.rename(columns={
        "CC3": "country_code",
        "Year": "year",
    })

    # For condensation, the columns
    # will be renamed by position: (column 1 = 0 in python)
    ext1_col = crisis.columns[3]
    ext2_col = crisis.columns[4]
    dom_col = crisis.columns[5]
    cur_col = crisis.columns[6]
    inf_col = crisis.columns[7]

    crisis = crisis.rename(columns={
        ext1_col: "external_default_1",
        ext2_col: "external_default_2",
        dom_col: "domestic_default",
        cur_col:  "currency_crisis",
        inf_col:  "inflation_crisis",
    })

    crisis = crisis.dropna(subset=["year"])
    crisis["year"] = crisis["year"].astype(int)

    # now I drop "country" column which will not be important (as we have the country code)
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

    crisis["year"] = crisis["year"].astype(int)

    crisis_cols = keep_cols[2:]
    for c in crisis_cols:
        crisis[c] = pd.to_numeric(crisis[c], errors="coerce")
        crisis[c] = crisis[c].fillna(0).astype(int)

    sovereign_cols = ["external_default_1", "external_default_2", "domestic_default"]
    crisis["sovereign_crisis"] = (crisis[sovereign_cols] > 0).any(axis=1).astype(int)

    return crisis