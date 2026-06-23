import joblib
import pandas as pd

MODEL_PATH = "C:/Users/Admin/Machine Learning Project/invoice_flagging/models/predict_flag_invoice.pkl"
SCALER_PATH = "C:/Users/Admin/Machine Learning Project/invoice_flagging/models/scaler.pkl"


def load_model():
    return joblib.load(MODEL_PATH)


def load_scaler():
    return joblib.load(SCALER_PATH)


def predict_invoice_flag(input_data):

    model = load_model()
    scaler = load_scaler()

    input_df = pd.DataFrame(input_data)

    input_scaled = scaler.transform(input_df)

    input_df["Predicted_Flag"] = model.predict(input_scaled)

    return input_df


if __name__ == "__main__":

    sample_data = {
        "invoice_quantity": [100, 50],
        "invoice_dollars": [18500, 9000],
        "Freight": [250, 120],
        "total_item_quantity": [120, 60],
        "total_item_dollars": [20000, 10000]
    }

    prediction = predict_invoice_flag(sample_data)

    print(prediction)