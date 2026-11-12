
import pandas as pd
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error,mean_squared_error,root_mean_squared_error)

# Load dataset
df = pd.read_csv(r"D:\DS_73\Project\Projects\electric_vehicle_sales.csv")

# Display dataset
print("Dataset:")
print(df.head())
print(df.shape)

# Check duplicate values
print("\nDuplicate Records:")
print(df.duplicated().sum())
# Remove duplicate records
df = df.drop_duplicates()

df = df.reset_index(drop=True)

# Check minimum and maximum values
print("\nFeature Value Ranges:")
print(df.describe().loc[['min', 'max']].T)

print("Dataset shape after removing duplicates:")
print(df.shape)
print(df.duplicated().sum())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check dataset information
print("\nDataset Information:")
df.info()

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Select input features
x = df[
    [
        'BatteryCapacity_kWh',
        'Range_km',
        'Price_Lakhs',
        'ChargingTime_hrs',
        'BrandRating',
        'Incentives_kWh',
        'ShowroomLocations'
    ]
]

# Select target variable
y = df['MonthlySales']

# Split dataset into training and testing data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=7
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(x_train, y_train)

# Predict values
y_pred = model.predict(x_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

# Display evaluation metrics
print("\nModel Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)


# Save trained model
with open(r'D:\DS_73\Project\Projects\Electric_Vehicle_Sales\Electric_Vehicle_Sales.pkl','wb') as file:
    pickle.dump(model, file)

print("\nModel saved successfully!")
