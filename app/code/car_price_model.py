"""
car_price_model.py  -  the model part of the A3 web app.

The notebook (A3_Predicting_Car_Price_st125973) trains the from-scratch multinomial
LogisticRegression and exports everything that was *learned from the training set* into
app/model/car_price_a3.json:

    * the weights W (row 0 = intercept)            -> the model
    * owner / transmission / fuel mappings          -> fixed .map() dictionaries
    * seller_type classes_ (LabelEncoder)           -> position in the sorted list
    * brand categories_ (OneHotEncoder)             -> one 0/1 column per training brand
    * scaler mean_ and scale_ (StandardScaler)      -> z = (x - mean) / scale
    * medians / modes of the training set           -> defaults for empty inputs
    * the 4 price-class edges (pd.cut bins)         -> to show the price range of a class

This file repeats the notebook's preprocessing in the same order, so a raw car typed into the
web app becomes exactly the same 41 numbers the model was trained on ("fit once, transform
everywhere"). Only numpy and pandas are needed: no pickle, so the file works with any
Python / library version inside the Docker image.
"""
import json
import os

import numpy as np
import pandas as pd

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "model", "car_price_a3.json"
)

# the raw columns the user gives us (same as X before encoding in the notebook)
RAW_COLUMNS = ["year", "km_driven", "fuel", "seller_type", "transmission", "owner",
               "mileage", "engine", "max_power", "seats", "brand"]
NUMERIC_COLUMNS = ["year", "km_driven", "mileage", "engine", "max_power", "seats"]


class CarPriceModel:
    """Load the exported model once, then call predict(...) as many times as needed."""

    def __init__(self, path=DEFAULT_MODEL_PATH):
        with open(path, encoding="utf-8") as f:
            self.bundle = json.load(f)
        b = self.bundle

        self.W = np.array(b["W"], dtype=float)          # shape (n_inputs, k)
        self.k = int(b["k"])                             # number of price classes (4)
        self.feature_columns = b["feature_columns"]      # 40 encoded columns, training order
        self.n_inputs = len(self.feature_columns) + 1    # + 1 for the intercept column
        self.scale_columns = b["scale_columns"]
        self.scaler_mean = np.array(b["scaler_mean"], dtype=float)
        self.scaler_scale = np.array(b["scaler_scale"], dtype=float)
        self.defaults = b["defaults"]
        self.price_bins = b["price_bins"]

        # the weights must fit the features, otherwise the export is broken
        if self.W.shape != (self.n_inputs, self.k):
            raise ValueError(
                f"W has shape {self.W.shape}, expected ({self.n_inputs}, {self.k})"
            )

    # ------------------------------------------------------------------ preprocessing
    def preprocess(self, raw):
        """raw car(s) -> numpy array of shape (m, n_inputs), intercept column first.

        raw can be one dict, a list of dicts or a DataFrame with the RAW_COLUMNS.
        Empty values (None / NaN / missing key) are filled with the training defaults.
        """
        if isinstance(raw, dict):
            df = pd.DataFrame([raw])
        elif isinstance(raw, list):
            df = pd.DataFrame(raw)
        else:
            df = pd.DataFrame(raw).copy()

        b = self.bundle

        # 1) empty input -> median (numbers) / mode (categories) of the TRAINING set
        for col in RAW_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = df[col].astype(object).where(df[col].notna(), self.defaults[col])
        for col in NUMERIC_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="raise").astype(float)

        # 2) fixed mappings (nothing learned -> same dictionaries as the notebook)
        df["owner"] = df["owner"].map(b["owner_mapping"])
        df["transmission"] = df["transmission"].map(b["transmission_mapping"])
        df["fuel"] = df["fuel"].map(b["fuel_mapping"])

        # 3) LabelEncoder.transform = position of the category in the sorted classes_
        seller_lookup = {name: i for i, name in enumerate(b["seller_type_classes"])}
        df["seller_type"] = df["seller_type"].map(seller_lookup)

        # a category the training set never had becomes NaN after .map() -> refuse it
        bad = [c for c in ["owner", "transmission", "fuel", "seller_type"] if df[c].isna().any()]
        if bad:
            raise ValueError(f"Unknown category in column(s): {bad}")

        # 4) OneHotEncoder with the training brands; unknown brand -> all zeros
        #    (same as handle_unknown='ignore' in the notebook)
        for brand in b["brand_categories"]:
            df["brand_" + brand] = (df["brand"] == brand).astype(int)
        df = df.drop(columns="brand")

        # 5) StandardScaler.transform with the TRAINING mean_ / scale_ (never re-fit)
        df[self.scale_columns] = (df[self.scale_columns].astype(float) - self.scaler_mean) / self.scaler_scale

        # 6) same column order as training, then the intercept column of 1s in front
        X = df[self.feature_columns].to_numpy(dtype=float)
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    # ------------------------------------------------------------------ model
    def predict_proba(self, X):
        """X: preprocessed array (m, n_inputs) -> class probabilities (m, k) via softmax."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_inputs:
            raise ValueError(f"expected an array of shape (m, {self.n_inputs}), got {X.shape}")
        z = X @ self.W
        z = z - z.max(axis=1, keepdims=True)          # stable softmax
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict_classes(self, X):
        """X: preprocessed array -> predicted class (0, 1, 2 or 3) for every row, shape (m,)."""
        return np.argmax(self.predict_proba(X), axis=1)

    def predict(self, raw):
        """raw car(s) -> predicted class for every car, shape (m,)."""
        return self.predict_classes(self.preprocess(raw))

    # ------------------------------------------------------------------ helpers for the UI
    def class_price_range(self, c):
        """Human readable price range of class c, from the pd.cut edges used in the notebook."""
        low, high = self.price_bins[c], self.price_bins[c + 1]
        if c == 0:
            return f"up to ₹ {high:,.0f}"
        if c == self.k - 1:
            return f"above ₹ {low:,.0f}"
        return f"₹ {low:,.0f} – ₹ {high:,.0f}"

    def predict_one(self, car):
        """One raw car (dict) -> dict with the class, its price range and all probabilities."""
        X = self.preprocess(car)
        probs = self.predict_proba(X)[0]
        c = int(np.argmax(probs))
        return {
            "price_class": c,
            "price_range": self.class_price_range(c),
            "probabilities": [float(p) for p in probs],
        }
