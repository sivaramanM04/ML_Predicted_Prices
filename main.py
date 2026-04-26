import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("data/train.csv")

# Drop unnecessary column
df = df.drop("Id", axis=1)

print("Initial Shape:", df.shape)

# =========================
# 2. HANDLE MISSING VALUES
# =========================

# Numerical columns
num_cols = df.select_dtypes(include=['int64','float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Categorical columns
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna("None")

# =========================
# 3. ENCODE CATEGORICAL DATA
# =========================
df = pd.get_dummies(df, drop_first=True)

print("After Encoding Shape:", df.shape)

# =========================
# 4. SPLIT FEATURES & TARGET
# =========================
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# =========================
# 5. TRAIN TEST SPLIT
# =========================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 6. TRAIN MODEL
# =========================
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# 7. PREDICTIONS
# =========================
predictions = model.predict(X_test)

# =========================
# 8. EVALUATION
# =========================
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\n===== MODEL PERFORMANCE =====")
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# =========================
# 9. FEATURE IMPORTANCE
# =========================
importance = model.feature_importances_
features = X.columns

imp_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

print("\n===== TOP 10 IMPORTANT FEATURES =====")
print(imp_df.head(10))

# =========================
# 10. VISUALIZATION
# =========================
plt.figure()
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()

# =========================
# 11. SAVE MODEL
# =========================
joblib.dump(model, "house_price_model.pkl")

print("\nModel saved successfully!")