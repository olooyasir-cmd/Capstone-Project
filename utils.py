"""
utils.py — Shared preprocessing, model training & recommendation logic
for the Google Play Store ML project.
"""

import re
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors

# ─────────────────────────────────────────────
# 1. DATA LOADING & CLEANING
# ─────────────────────────────────────────────

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # ── Remove duplicates (keep last = most recent entry) ──────────────
    df.drop_duplicates(subset="App", keep="last", inplace=True)

    # ── Drop rows where Rating is missing (target variable) ────────────
    df.dropna(subset=["Rating"], inplace=True)

    # ── Fix bad Rating rows (some rows have category-shifted data) ──────
    df = df[pd.to_numeric(df["Rating"], errors="coerce").between(1, 5)]
    df["Rating"] = df["Rating"].astype(float)

    # ── Clean Size → float (MB) ────────────────────────────────────────
    def parse_size(s):
        s = str(s).strip()
        if s in ("Varies with device", "nan", ""):
            return np.nan
        if s.endswith("M"):
            return float(s[:-1])
        if s.endswith("k"):
            return float(s[:-1]) / 1024
        return np.nan

    df["Size_MB"] = df["Size"].apply(parse_size)
    df["Size_MB"].fillna(df["Size_MB"].median(), inplace=True)

    # ── Clean Installs → int ───────────────────────────────────────────
    df["Installs_Clean"] = (
        df["Installs"]
        .str.replace(r"[+,]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # ── Clean Price → float ────────────────────────────────────────────
    df["Price_Clean"] = (
        df["Price"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

    # ── Clean Reviews → int ───────────────────────────────────────────
    df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce").fillna(0).astype(int)

    # ── Drop rows with missing Content Rating or Type ─────────────────
    df.dropna(subset=["Content Rating", "Type"], inplace=True)

    # ── Clean Genres (use first genre only) ───────────────────────────
    df["Genre_Clean"] = df["Genres"].str.split(";").str[0].str.strip()

    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING & ENCODING
# ─────────────────────────────────────────────

def encode_features(df: pd.DataFrame):
    """
    Returns (X, y, encoders_dict, feature_names).
    Encoders are stored so the Streamlit app can reuse them.
    """
    df = df.copy()

    le_cat   = LabelEncoder()
    le_cr    = LabelEncoder()
    le_type  = LabelEncoder()
    le_genre = LabelEncoder()

    df["Category_Enc"]       = le_cat.fit_transform(df["Category"])
    df["ContentRating_Enc"]  = le_cr.fit_transform(df["Content Rating"])
    df["Type_Enc"]           = le_type.fit_transform(df["Type"])
    df["Genre_Enc"]          = le_genre.fit_transform(df["Genre_Clean"])

    feature_cols = [
        "Category_Enc", "Size_MB", "Installs_Clean",
        "Price_Clean", "ContentRating_Enc", "Type_Enc",
        "Genre_Enc", "Reviews",
    ]

    X = df[feature_cols].values
    y = df["Rating"].values

    encoders = {
        "le_cat":   le_cat,
        "le_cr":    le_cr,
        "le_type":  le_type,
        "le_genre": le_genre,
    }
    return X, y, encoders, feature_cols


# ─────────────────────────────────────────────
# 3. SUPERVISED — RANDOM FOREST REGRESSOR
# ─────────────────────────────────────────────

def train_rf(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rf, mae, rmse, X_test, y_test, y_pred


# ─────────────────────────────────────────────
# 4. UNSUPERVISED — KNN RECOMMENDER
# ─────────────────────────────────────────────

def build_recommender(df: pd.DataFrame, encoders: dict):
    """Build a KNN model over the cleaned dataframe."""
    df = df.copy()
    df["Category_Enc"]      = encoders["le_cat"].transform(df["Category"])
    df["ContentRating_Enc"] = encoders["le_cr"].transform(df["Content Rating"])
    df["Genre_Enc"]         = encoders["le_genre"].transform(df["Genre_Clean"])

    rec_features = ["Category_Enc", "Genre_Enc", "Price_Clean",
                    "ContentRating_Enc", "Size_MB"]

    scaler = StandardScaler()
    X_rec = scaler.fit_transform(df[rec_features].values)

    knn = NearestNeighbors(n_neighbors=6, metric="cosine", algorithm="brute")
    knn.fit(X_rec)

    return knn, scaler, rec_features, df.reset_index(drop=True)


def recommend_apps(app_name: str, df: pd.DataFrame, knn, scaler, rec_features,
                   n=5) -> pd.DataFrame:
    """Return top-n similar apps for a given app name."""
    matches = df[df["App"].str.lower() == app_name.lower()]
    if matches.empty:
        # fuzzy fallback: partial match
        matches = df[df["App"].str.lower().str.contains(app_name.lower(), na=False)]
    if matches.empty:
        return pd.DataFrame({"Error": [f"App '{app_name}' not found in dataset."]})

    idx = matches.index[0]
    query_vec = scaler.transform(df.loc[[idx], rec_features].values)
    distances, indices = knn.kneighbors(query_vec, n_neighbors=n + 1)

    recs = df.iloc[indices[0][1:]].copy()   # exclude the app itself
    recs["Similarity"] = (1 - distances[0][1:]).round(3)
    return recs[["App", "Category", "Genre_Clean", "Rating",
                 "Price_Clean", "Installs_Clean", "Similarity"]].rename(
        columns={"Genre_Clean": "Genre", "Price_Clean": "Price",
                 "Installs_Clean": "Installs"}
    )


# ─────────────────────────────────────────────
# 5. PREDICT RATING FOR NEW APP (Streamlit)
# ─────────────────────────────────────────────

def predict_new_app(rf, encoders, category, size_mb, price,
                    content_rating, app_type, genre, reviews=100):
    """
    Predict rating for a hypothetical new app.
    Returns predicted float rating.
    """
    le_cat   = encoders["le_cat"]
    le_cr    = encoders["le_cr"]
    le_type  = encoders["le_type"]
    le_genre = encoders["le_genre"]

    # safe encoding: if unseen label, pick closest
    def safe_encode(le, val):
        if val in le.classes_:
            return le.transform([val])[0]
        return 0  # default to first class

    cat_enc  = safe_encode(le_cat, category)
    cr_enc   = safe_encode(le_cr, content_rating)
    type_enc = safe_encode(le_type, app_type)
    gen_enc  = safe_encode(le_genre, genre)

    X_new = np.array([[cat_enc, size_mb, 1000, price,
                       cr_enc, type_enc, gen_enc, reviews]])
    return float(rf.predict(X_new)[0])
