"""
app.py  -  Dash web app for A3: predicts the PRICE CLASS (0-3) of a used car.

Run it locally:     python app/code/app.py          -> http://localhost:8050
Run it in Docker:   cd app && docker compose up -d --build

The model and everything learned from the training set (encoders, scaler, defaults) come from
app/model/car_price_a3.json, exported by the notebook. car_price_model.py repeats the
notebook's preprocessing, so the web app and the notebook give the same predictions.
"""
import os

from dash import Dash, html, dcc, Input, Output, State

from car_price_model import CarPriceModel, DEFAULT_MODEL_PATH

MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
model = CarPriceModel(MODEL_PATH)
print("Model loaded:", model.bundle.get("best_run"), "| inputs:", model.n_inputs, "| classes:", model.k)

# ------------------------------------------------------------------ dropdown options
# only the categories the model was trained on (an unseen value could not be encoded)
OWNER_OPTIONS = [{"label": o, "value": o} for o in model.bundle["owner_mapping"]]
FUEL_OPTIONS = [{"label": f, "value": f} for f in model.bundle["fuel_mapping"]]
TRANSMISSION_OPTIONS = [{"label": t, "value": t} for t in model.bundle["transmission_mapping"]]
SELLER_OPTIONS = [{"label": s, "value": s} for s in model.bundle["seller_type_classes"]]
# "Other" is not a training brand -> all brand columns are 0 (like handle_unknown='ignore')
BRAND_OPTIONS = [{"label": b, "value": b} for b in model.bundle["brand_categories"]]
BRAND_OPTIONS.append({"label": "Other", "value": "Other"})

DEFAULTS = model.defaults

INPUT_STYLE = {"width": "100%", "padding": "6px", "boxSizing": "border-box"}
BLOCK_STYLE = {"marginBottom": "14px"}


def number_input(input_id, label, example):
    return html.Div(
        [html.Label(label),
         dcc.Input(id=input_id, type="number", placeholder=f"Example: {example}", style=INPUT_STYLE)],
        style=BLOCK_STYLE,
    )


def dropdown(input_id, label, options):
    return html.Div(
        [html.Label(label),
         dcc.Dropdown(id=input_id, options=options, placeholder=f"Select {label.lower()}")],
        style=BLOCK_STYLE,
    )


# ------------------------------------------------------------------ layout
app = Dash(__name__)
server = app.server
app.title = "Used Car Price Class Predictor (A3)"

app.layout = html.Div(
    [
        html.H1("Used Car Price Class Predictor", style={"textAlign": "center", "marginBottom": "6px"}),
        html.P(
            "A3 - multinomial logistic regression written from scratch. "
            "Enter the car details; the model predicts which of 4 price classes the car falls into.",
            style={"textAlign": "center", "color": "#555", "marginBottom": "4px"},
        ),
        html.P(
            "Empty fields are filled with the training-set median (numbers) or most common value (categories).",
            style={"textAlign": "center", "color": "#777", "fontSize": "14px", "marginBottom": "26px"},
        ),
        html.Div(
            [
                html.H2("Vehicle information"),
                number_input("year", "Year", 2017),
                number_input("km_driven", "Kilometers driven", 45000),
                dropdown("owner", "Owner", OWNER_OPTIONS),
                number_input("mileage", "Mileage (kmpl)", 20.5),
                number_input("engine", "Engine (CC)", 1197),
                number_input("max_power", "Max power (bhp)", 82),
                number_input("seats", "Number of seats", 5),

                html.H2("Car specifications"),
                dropdown("fuel", "Fuel", FUEL_OPTIONS),
                dropdown("seller_type", "Seller type", SELLER_OPTIONS),
                dropdown("transmission", "Transmission", TRANSMISSION_OPTIONS),
                dropdown("brand", "Brand", BRAND_OPTIONS),

                html.Button(
                    "Predict price class",
                    id="predict-button",
                    n_clicks=0,
                    style={"width": "100%", "padding": "12px", "fontSize": "18px", "cursor": "pointer"},
                ),
                html.Div(id="prediction-output", style={"marginTop": "28px"}),
            ],
            style={"maxWidth": "600px", "margin": "auto", "padding": "24px"},
        ),
    ]
)


# ------------------------------------------------------------------ prediction
def build_result(car):
    """Run the model on one raw car and build the result block (kept separate from the callback)."""
    empty = [col for col, value in car.items() if value is None]
    result = model.predict_one(car)

    rows = []
    for c, p in enumerate(result["probabilities"]):
        bar = html.Div(style={"width": f"{p * 100:.1f}%", "height": "10px", "background": "#3b6fb6"})
        rows.append(
            html.Tr([
                html.Td(f"Class {c}", style={"paddingRight": "10px", "fontWeight": "bold" if c == result["price_class"] else "normal"}),
                html.Td(model.class_price_range(c), style={"paddingRight": "10px"}),
                html.Td(f"{p * 100:.1f} %", style={"paddingRight": "10px", "textAlign": "right"}),
                html.Td(bar, style={"width": "160px"}),
            ])
        )

    filled_note = ""
    if empty:
        filled_note = "Filled with training defaults: " + ", ".join(f"{col} = {DEFAULTS[col]}" for col in empty)

    return html.Div(
        [
            html.H2("Predicted price class", style={"textAlign": "center"}),
            html.H1(f"Class {result['price_class']}", style={"textAlign": "center", "margin": "6px"}),
            html.H3(result["price_range"], style={"textAlign": "center", "color": "#3b6fb6", "marginTop": "0"}),
            html.Table(rows, style={"margin": "auto", "marginTop": "14px"}),
            html.P(filled_note, style={"color": "#777", "fontSize": "14px", "marginTop": "16px"}),
        ]
    )


@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    State("year", "value"),
    State("km_driven", "value"),
    State("owner", "value"),
    State("mileage", "value"),
    State("engine", "value"),
    State("max_power", "value"),
    State("seats", "value"),
    State("fuel", "value"),
    State("seller_type", "value"),
    State("transmission", "value"),
    State("brand", "value"),
    prevent_initial_call=True,
)
def predict_price_class(n_clicks, year, km_driven, owner, mileage, engine, max_power, seats,
                        fuel, seller_type, transmission, brand):
    car = {
        "year": year, "km_driven": km_driven, "fuel": fuel, "seller_type": seller_type,
        "transmission": transmission, "owner": owner, "mileage": mileage, "engine": engine,
        "max_power": max_power, "seats": seats, "brand": brand,
    }
    try:
        return build_result(car)
    except Exception as e:  # show the problem on the page instead of a silent failure
        return html.Div([html.H3("Prediction error", style={"color": "red"}), html.P(str(e))])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
