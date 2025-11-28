from pathlib import Path
from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

SEED: int = 42

##### Load merged dataset for modeling
def load_model_dataset(path:Path) -> pd.DataFrame:
    """Load the merged dataset for modeling."""
    return pd.read_csv(path)

# select columns used as input from dataframe by excluding certain columns:
def get_feature_columns(df: pd.DataFrame, target: str) -> List[str]: 
    """Get feature columns by excluding the target and identifier columns."""  
    exclude_cols = [target, "country_code", "year"] # columns to exclude from features
    return [col for col in df.columns if col not in exclude_cols] # returns list of column names excluding target and id columns

##### Train logistic regression model
def train_logistic_regression(df: pd.DataFrame, target: str = "sovereign_crisis") -> Tuple[LogisticRegression, pd.DataFrame, pd.Series]:
    """Train a logistic regression model."""
    # Feature engineering
    feature_cols = get_feature_columns(df, target)     # calls function defined above, returns list of feature column names & excludes target and id columns
    X = df[feature_cols].copy()               # creates features dataframe, copying ensures original df is not modified
    y = df[target].copy()               # target series

    # I had problems with non-numeric data in features, so I convert all to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # First, I need to drop rows where the target is NaN
    mask = y.notna()
    X = X[mask]         # keep only rows in X where y is not NaN
    y = y[mask]         # keep only non missing values in y

    # Now, I drop rows with any NaN in features
    X = X.dropna()   
    y = y.loc[X.index]  # align y with the cleaned X


    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    # 20% of data go into test set, 80% into training set
    # SEED for reproducibility, ensuring consistent splits across runs
    # Stratify to maintain class distribution in both sets

    model = LogisticRegression(max_iter=2000, random_state=SEED)
    model.fit(X_train, y_train)
    # Large dataset, so I increased max_iter to ensure convergence
    # random_state for reproducibility, ensuring consistent results across runs
    return model, X_test, y_test