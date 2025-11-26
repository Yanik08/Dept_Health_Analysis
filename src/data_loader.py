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
    """
    Load the raw WEO Excel file.

    Parameters
    ----------
    path : Path
        Path to the Excel file.
    
    Returns
    -------
    pd.DataFrame
        Raw WEO data.
    """

    return pd.read_excel(path, sheet_name="WEOData")

def main() -> None:
    """Test loader: print basic info about the crisis file."""
    excel_path = Path("data/raw/global_crisis_data.xlsx")

    df = load_raw_crisis(excel_path)

    print("\nNumber of rows:", len(df))
    print("\nFirst 20 column names:")
    print(df.columns.tolist()[:20])
    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()