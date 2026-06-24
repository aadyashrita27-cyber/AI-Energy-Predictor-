
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
data = pd.read_csv("static/energy_data.csv")

# Input and Output
X = data[['Hour']]
y = data['Consumption']

# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X, y)

# Save model
joblib.dump(model, 'model.pkl')

print("Model trained successfully!")