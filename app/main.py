from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

app = FastAPI()

# Mount thư mục static để phục vụ các file tĩnh (CSS, JS, hình ảnh)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Thiết lập Jinja2 cho template rendering
templates = Jinja2Templates(directory="app/templates")

# Đọc dữ liệu từ file CSV
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
    ('linear', Ridge(alpha=10.0)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, activation='relu', random_state=42)),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
]

# Final model
final_model = Ridge(alpha=50.0)

# Stacking Regressor
stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=final_model)

# Huấn luyện mô hình
stacking_regressor.fit(X_train_scaled, y_train)

# Route trang chủ - hiển thị form nhập dữ liệu
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route xử lý dự đoán
@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, bedrooms: int = Form(...), garage: int = Form(...), land_area: int = Form(...), floor_area: int = Form(...), build_year: int = Form(...), cbd_dist: int = Form(...)):
    # Chuẩn bị dữ liệu đầu vào
    input_data = pd.DataFrame([[bedrooms, garage, land_area, floor_area, build_year, cbd_dist]], columns=X.columns)
    input_data_scaled = scaler.transform(input_data)
    
    # Dự đoán giá nhà
    prediction = stacking_regressor.predict(input_data_scaled)[0]

    # Trả kết quả về trang index
    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})
