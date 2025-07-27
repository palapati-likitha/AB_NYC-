
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
import joblib

# Load dataset
df = pd.read_csv("AB_NYC_2019.csv")

# Basic Cleaning
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df[(df['price'] > 0) & (df['price'] < 1000)]

# Drop rows with missing values in selected columns
features = ['room_type', 'neighbourhood', 'latitude', 'longitude', 'minimum_nights',
            'number_of_reviews', 'reviews_per_month', 'availability_365']
df = df[features + ['price']].dropna()

# Add location clustering (KMeans on lat/long)
kmeans = KMeans(n_clusters=5, random_state=42)
df['location_cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']])

# Drop lat/long to avoid multicollinearity
df = df.drop(['latitude', 'longitude'], axis=1)

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=['room_type', 'neighbourhood'], drop_first=True)

# Define input/output
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model comparison
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Linear Regression": LinearRegression()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\nModel: {name}")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R^2 Score:", r2_score(y_test, y_pred))

# Use GridSearchCV to fine-tune Random Forest
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None]
}
grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("\nBest Tuned Random Forest Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R^2 Score:", r2_score(y_test, y_pred))

# Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 4))
sns.histplot(residuals, kde=True)
plt.title("Residuals Distribution")
plt.xlabel("Residual")
plt.show()

# Feature Importance
plt.figure(figsize=(12, 6))
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

sns.barplot(x=importances[indices], y=feature_names[indices])
plt.title("Feature Importances")
plt.show()

# Save the final model
joblib.dump(best_model, "airbnb_price_model.pkl")
