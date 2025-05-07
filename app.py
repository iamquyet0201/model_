from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # 🔥 Bật CORS để các website khác có thể gọi API

# Load mô hình và công cụ xử lý
model = joblib.load("model_satlo.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/")
def home():
    return jsonify({"message": "Landslide prediction API is running!"})

# 🔧 Trả về favicon mặc định nếu trình duyệt yêu cầu
@app.route('/favicon.ico')
def favicon():
    return '', 204

# 📥 Dự đoán từ dữ liệu gửi lên
@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])

        # Bắt buộc đúng thứ tự cột
        columns = ["c'", "L", "gamma", "h", "u", "phi'", "beta"]
        df = df[columns]

        # Chuẩn hóa
        X_scaled = scaler.transform(df)

        # Dự đoán
        pred = model.predict(X_scaled)
        label = label_encoder.inverse_transform(pred)[0]

        return jsonify({"prediction": label})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
