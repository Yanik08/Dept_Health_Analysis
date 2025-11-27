from pathlib import Path
import pandas as pd  # already in your environment.yml

from src.data_loader import load_raw_weo, build_weo_panel
from src.data_merger import build_crisis_panel


def main() -> None:
    """
    Entry point for the Debt Health Analysis project.

    Steps:
    1. Load raw IMF WEO data and build a country–year macro panel.
    2. Load and clean the crisis database (build crisis panel).
    3. Merge both into a single dataset and save it to data/processed/.
    """

    # Project root = folder where main.py lives
    project_root = Path(__file__).parent

    # Always relative paths (course requirement)
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. WEO: build macro panel ----------
    weo_path = raw_dir / "IMF_WEO_dataset.csv"
    if not weo_path.exists():
        raise FileNotFoundError(
            f"WEO file not found at {weo_path}. "
            "Check that 'IMF_WEO_dataset.csv' is in data/raw/."
        )

    print(f"\n[1] Loading WEO data from: {weo_path}")
    weo_raw = load_raw_weo(weo_path)
    print(f"    Raw WEO shape: {weo_raw.shape}")

    weo_panel = build_weo_panel(weo_raw)
    print(f"    WEO panel (country-year) shape: {weo_panel.shape}")

    weo_out = processed_dir / "weo_panel.csv"
    weo_panel.to_csv(weo_out, index=False)
    print(f"    Saved WEO panel to: {weo_out}")

    # ---------- 2. Crisis: build crisis panel ----------
    crisis_path = raw_dir / "global_crisis_data.xlsx"
    if not crisis_path.exists():
        raise FileNotFoundError(
            f"Crisis file not found at {crisis_path}. "
            "Check that 'global_crisis_data.xlsx' is in data/raw/."
        )

    print(f"\n[2] Loading crisis data from: {crisis_path}")
    crisis_panel = build_crisis_panel(crisis_path)
    print(f"    Crisis panel shape: {crisis_panel.shape}")

    # (Optional) you could also save the crisis panel if you want:
    # crisis_out = processed_dir / "crisis_panel.csv"
    # crisis_panel.to_csv(crisis_out, index=False)

    # ---------- 3. Merge WEO + Crisis ----------
    print("\n[3] Merging WEO panel with crisis panel (inner join on country_code, year)...")
    merged = pd.merge(
        weo_panel,
        crisis_panel,
        on=["country_code", "year"],
        how="inner",
    )
    print(f"    Merged dataset shape: {merged.shape}")

    merged_out = processed_dir / "merged_weo_crisis.csv"
    merged.to_csv(merged_out, index=False)
    print(f"    Saved merged dataset to: {merged_out}")

    print("\n✅ main.py finished successfully.\n")


if __name__ == "__main__":
    main()
