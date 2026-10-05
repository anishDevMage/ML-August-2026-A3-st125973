"""
Unit tests for the A3 model (run from the repository root with:  pytest -v app/tests)

The assignment asks for two tests:
    1) the model takes the expected input
    2) the output of the model has the expected shape
Two small extra checks make sure bad input is refused and empty fields are filled.
"""
import os
import sys

import numpy as np
import pytest

# make app/code importable (car_price_model.py lives there)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from car_price_model import CarPriceModel  # noqa: E402

model = CarPriceModel()   # loads app/model/car_price_a3.json

SAMPLE_CAR = {"year": 2017, "km_driven": 40000, "fuel": "Diesel", "seller_type": "Individual",
              "transmission": "Manual", "owner": "First Owner", "mileage": 21.5, "engine": 1248,
              "max_power": 74.0, "seats": 5, "brand": "Maruti"}

MORE_CARS = [
    {"year": 2010, "km_driven": 120000, "fuel": "Petrol", "seller_type": "Individual",
     "transmission": "Manual", "owner": "Second Owner", "mileage": 18.9, "engine": 998,
     "max_power": 67.1, "seats": 5, "brand": "Hyundai"},
    {"year": 2019, "km_driven": 15000, "fuel": "Diesel", "seller_type": "Dealer",
     "transmission": "Automatic", "owner": "First Owner", "mileage": 15.0, "engine": 1995,
     "max_power": 190.0, "seats": 5, "brand": "BMW"},
]


def test_model_takes_expected_input():
    """(1) A raw car becomes one row of 41 numbers (intercept + 40 encoded features) and the model accepts it."""
    X = model.preprocess(SAMPLE_CAR)

    assert X.shape == (1, 41)                  # 1 car, 1 intercept + 40 features
    assert X.shape[1] == model.W.shape[0]      # matches the number of rows of W
    assert np.all(X[:, 0] == 1)                # first column is the intercept
    assert not np.isnan(X).any()               # every input was encoded

    probabilities = model.predict_proba(X)     # the model accepts this input without an error
    assert probabilities is not None


def test_model_output_has_expected_shape():
    """(2) For m cars the model returns m class labels in {0, 1, 2, 3} and an (m, 4) probability table."""
    cars = [SAMPLE_CAR] + MORE_CARS
    X = model.preprocess(cars)

    yhat = model.predict_classes(X)
    probabilities = model.predict_proba(X)

    assert yhat.shape == (3,)
    assert set(yhat.tolist()) <= {0, 1, 2, 3}
    assert probabilities.shape == (3, 4)
    assert np.allclose(probabilities.sum(axis=1), 1.0)   # softmax rows add up to 1


def test_model_rejects_input_with_wrong_number_of_features():
    """Extra: an array with the wrong number of columns is refused instead of giving a wrong answer."""
    with pytest.raises(ValueError):
        model.predict_proba(np.ones((1, 10)))


def test_empty_fields_are_filled_with_training_defaults():
    """Extra: a car with every field empty still gives exactly one valid prediction."""
    empty_car = {key: None for key in SAMPLE_CAR}
    result = model.predict_one(empty_car)

    assert result["price_class"] in {0, 1, 2, 3}
    assert len(result["probabilities"]) == 4
