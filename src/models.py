from pathlib import Path
from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

SEED: int = 42

##### Load merged dataset for modeling
def load_model_dataset(path:Path) -> pd.DataFrame:
    """Load the merged dataset for modeling."""
    return pd.read_csv(path)

# select columns used as input from dataframe by excluding certain columns:
def get_feature_columns(df: pd.DataFrame, target: str) -> List[str]: 
    """Get feature columns by excluding the target and identifier columns."""  
    crisis_helper_cols = ["external_default_1", "external_default_2", "domestic_default", "currency_crisis", "inflation_crisis"]
    exclude_cols = {target, "sovereign_crisis", "country_code", "year", *crisis_helper_cols}
    feature_cols: list[str] = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if col.startswith("crisis_h"):
            continue
        feature_cols.append(col)

    return feature_cols
# these columns are excluded because they are either the target variable or identifiers, or they are helper columns used to construct the target variable.
# The remaining columns are considered features for model training.


##### Train logistic regression model
def train_logistic_regression(df: pd.DataFrame, target: str = "sovereign_crisis") -> Tuple[LogisticRegression, pd.DataFrame, pd.Series]:
    """Train a logistic regression model."""
    # Feature engineering
    feature_cols = get_feature_columns(df, target)     # calls function defined above, returns list of feature column names & excludes target and id columns
    X = df[feature_cols].copy()               # creates features dataframe, copying ensures original df is not modified
    y = df[target].copy()               # target series

    # I had problems with non-numeric data in features, so I convert all to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    X = X.dropna(axis=1, how="all")  # drop columns that are all NaN
    feature_cols = list(X.columns)  # update feature_cols to reflect any dropped columns
    
    # First, I need to drop rows where the target is NaN
    mask = y.notna()
    X = X[mask]         # keep only rows in X where y is not NaN
    y = y[mask]         # keep only non missing values in y

    # Then, I need to handle missing values in features
    # I chose to impute missing values using the mean of each feature (documented in report)
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    # 20% of data go into test set, 80% into training set
    # SEED for reproducibility, ensuring consistent splits across runs
    # Stratify to maintain class distribution in both sets
    imputer = SimpleImputer(strategy="mean")

    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,)

    model = LogisticRegression(max_iter=2000, random_state=SEED)
    model.fit(X_train, y_train)
    # Large dataset, so I increased max_iter to ensure convergence
    # random_state for reproducibility, ensuring consistent results across runs
    return model, X_test, y_test

##### Train random forest model #####
def train_random_forest(df: pd.DataFrame, target: str = "sovereign_crisis",) -> Tuple[RandomForestClassifier, pd.DataFrame, pd.Series]:
    """Train a Random Forest classifier."""

    # same feature engineering / cleaning as in train_logistic_regression
    feature_cols = get_feature_columns(df, target) # get feature columns excluding target and id columns
    X = df[feature_cols].copy() # features dataframe, again, copying to avoid modifying original df
    y = df[target].copy() # target series

    # Convert all features to numeric, coerce errors to NaN
    X = X.apply(pd.to_numeric, errors="coerce") # convert features to numeric data types

    # Drop rows where target is NaN
    mask = y.notna() # mask for non-missing target values
    X = X[mask] # keep only rows in X where y is not NaN
    y = y[mask] # keep only non-missing values in y

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    ) # 20% test size, 80% training size, SEED for reproducibility, stratify to maintain class distribution

    imputer = SimpleImputer(strategy="mean")

    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
        )

    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
        )

    ### The model training
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced")

    model.fit(X_train, y_train) # fit the Random Forest model to training data

    return model, X_test, y_test

##### Train XGBoost model #####
def train_xgboost(df: pd.DataFrame, target: str = "sovereign_crisis",) -> Tuple[XGBClassifier, pd.DataFrame, pd.Series]:
    """Train an XGBoost classifier."""

    # same feature engineering / cleaning as in the other models
    feature_cols = get_feature_columns(df, target) # get feature columns excluding target and id columns
    X = df[feature_cols].copy() # features dataframe, again, copying to avoid modifying original df
    y = df[target].copy() # target series

    # Convert all features to numeric, coerce errors to NaN
    X = X.apply(pd.to_numeric, errors="coerce") # convert features to numeric data types

    # Drop rows where target is NaN
    mask = y.notna() # mask for non-missing target values
    X = X[mask] # keep only rows in X where y is not NaN
    y = y[mask] # keep only non-missing values in y


    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
    ) # 20% test size, 80% training size, SEED for reproducibility, stratify to maintain class distribution
    
    imputer = SimpleImputer(strategy="mean")

    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    
    ### The model training
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        objective="binary:logistic",
        eval_metric='logloss',
        n_jobs=-1)

    model.fit(X_train, y_train) # fit the XGBoost model to training data

    return model, X_test, y_test