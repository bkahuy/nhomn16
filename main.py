from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os
from flask import Flask, render_template

app = FastAPI()
appp = Flask(__name__)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


# Đọc dữ liệu
df = pd.read_csv("dataset.csv")

# Chia dữ liệu thành features và target
X = df.drop(columns=['PRICE'])
y = df['PRICE']

# Xử lý giá trị thiếu
if df.isnull().sum().any():
    X.fillna(X.mean(), inplace=True)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Xây dựng các mô hình base
base_models = [
    ('linear', LinearRegression()),
    ('ridge', Ridge(alpha=10.0)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, activation='relu', random_state=42)),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
]

# Final model
final_model = Ridge(alpha=50.0)

# Stacking Regressor
stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=final_model)

# Huấn luyện mô hình
stacking_regressor.fit(X_train_scaled, y_train)

# Dự đoán trên tập kiểm tra
y_pred = stacking_regressor.predict(X_test_scaled)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# print(f"Mean Squared Error (MSE): {mse}")
# print(f"R² Score: {r2}")

# # Lưu mô hình
# with open('best_model.pkl', 'wb') as file:
#     pickle.dump(stacking_regressor, file)

# Mount thư mục static
app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dự đoán giá nhà</title>
    </head>
    <body>
        <h1>Dự đoán giá nhà</h1>
        <form action="/predict" method="POST">
            <label for="bedrooms">Số phòng ngủ:</label><br>
            <input type="number" name="bedrooms" required><br><br>
            
            <label for="garage">Số chỗ để xe:</label><br>
            <input type="number" name="garage" required><br><br>
            
            <label for="land_area">Diện tích đất (m²):</label><br>
            <input type="number" name="land_area" required><br><br>
            
            <label for="floor_area">Diện tích sàn (m²):</label><br>
            <input type="number" name="floor_area" required><br><br>
            
            <label for="build_year">Năm xây dựng:</label><br>
            <input type="number" name="build_year" required><br><br>
            
            <label for="cbd_dist">Khoảng cách đến trung tâm (m):</label><br>
            <input type="number" name="cbd_dist" required><br><br>
            
            <input type="submit" value="Dự đoán">
        </form>
    </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
def predict(bedrooms: int = Form(...), garage: int = Form(...), land_area: int = Form(...), floor_area: int = Form(...), build_year: int = Form(...), cbd_dist: int = Form(...)):
    # Chuẩn bị dữ liệu đầu vào
    input_data = pd.DataFrame([[bedrooms, garage, land_area, floor_area, build_year, cbd_dist]], columns=X.columns)
    input_data_scaled = scaler.transform(input_data)
    
    # Dự đoán giá nhà using the stacking_regressor
    prediction = stacking_regressor.predict(input_data_scaled)[0] 

    return f"""
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Kết quả dự đoán</title>
    </head>
    <body>
        <h1>Kết quả dự đoán</h1>
        <p>Giá nhà dự đoán: {prediction:.2f}</p> 
        <a href="/">Quay lại</a>
    </body>
    </html>
    """