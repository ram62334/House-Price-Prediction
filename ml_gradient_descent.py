import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# -----------------------------
# LOAD AND CLEAN DATA
# -----------------------------
df = pd.read_csv(r"C:\Users\ramya\OneDrive\Desktop\mlproject\House_Rent_Dataset.csv",
                 encoding="utf-8", low_memory=False)

# Drop missing values and duplicates
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Keep only relevant features
features = ["size", "bhk", "furnishing_status", "floor"]
target = "rent"
df = df[features + [target]]

# Encode categorical variables (furnishing_status and floor)
df = pd.get_dummies(df, columns=["furnishing_status", "floor"], drop_first=True)

# Separate features and target
X = df.drop(columns=[target])
y = df[target].values

# -----------------------------
# TRAIN/EVAL/TEST SPLIT
# -----------------------------
# 60% train, 20% validation, 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# -----------------------------
# FEATURE SCALING (Z-score)
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# POLYNOMIAL REGRESSION + RIDGE
# -----------------------------
degrees = [1, 2, 3]
alphas = [0.01, 0.1, 1.0, 10]  # Regularization strength for Ridge
best_val_mse = float('inf')

results = []

for deg in degrees:
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_val_poly = poly.transform(X_val_scaled)
    
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train_poly, y_train)
        
        y_train_pred = model.predict(X_train_poly)
        y_val_pred = model.predict(X_val_poly)
        
        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)
        
        results.append((deg, alpha, train_mse, val_mse))
        
        print(f"Degree {deg}, Alpha {alpha}: Train MSE = {train_mse:.2f}, Val MSE = {val_mse:.2f}")
        
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_model = model
            best_poly = poly
            best_degree = deg
            best_alpha = alpha

print(f"\nBest Model: Degree = {best_degree}, Alpha = {best_alpha}, Validation MSE = {best_val_mse:.2f}")

# -----------------------------
# TEST SET EVALUATION
# -----------------------------
X_test_poly = best_poly.transform(X_test_scaled)
y_test_pred = best_model.predict(X_test_poly)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f"Test MSE = {test_mse:.2f}")

# -----------------------------
# PLOT TRAIN/VAL MSE FOR DEGREES
# -----------------------------
plt.figure(figsize=(8,5))
for deg in degrees:
    val_errors = [r[3] for r in results if r[0]==deg]
    plt.plot(alphas, val_errors, marker='o', label=f'Degree {deg}')
plt.xscale('log')
plt.xlabel("Alpha (Ridge Regularization)")
plt.ylabel("Validation MSE")
plt.title("Validation MSE vs Regularization for Different Degrees")
plt.legend()
plt.show()

# -----------------------------
# PREDICTIONS ON NEW DATA
# -----------------------------
# Example new houses
new_houses = pd.DataFrame({
    'size': [1200, 1500, 2000, 3000],
    'bhk': [2, 3, 3, 4],
    'furnishing_status_Semi-Furnished': [0, 1, 0, 1],
    'furnishing_status_Unfurnished': [1, 0, 1, 0],
    'floor_1 out of 2': [0, 0, 1, 0],
    'floor_1 out of 3': [0, 1, 0, 0],
    'floor_Ground out of 2': [1, 0, 0, 0],
    'floor_Ground out of 4': [0, 0, 0, 1],
    'floor_1 out of 1': [0, 0, 0, 0]  # Add all dummy columns used in training
})

# Scale new data
new_scaled = scaler.transform(new_houses)
new_poly = best_poly.transform(new_scaled)
y_new_pred = best_model.predict(new_poly)

for size, pred in zip(new_houses['size'], y_new_pred):
    print(f"Predicted rent for house {size} sq-ft = {pred:.2f}")
