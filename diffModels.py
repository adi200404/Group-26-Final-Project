import pandas as pd
import warnings
warnings.filterwarnings('ignore')

excel_file_path = '/annual data .xlsx'
df = pd.read_excel(excel_file_path)
#write the excel to a csv so that it is easier to interpret when reading it into functions
df.to_csv('annual data.csv', index=False)

df = pd.read_csv('annual data.csv', skiprows = 1)

df.columns = ['Annual Total', 'Total Fossil Fuels Production', 'Nuclear Electric Power Production', 'Total Renewable Energy Production', 'Total Primary Energy Production', 'Primary Energy Imports', 'Primary Energy Exports',
              'Primary Energy Net Imports', 'Primary Energy Stock Change and Other','Total Fossil Fuels Consumption', 'Nuclear Electric Power Consumption','Total Renewable Energy Consumption','Total Primary Energy Consumption']

df.head()




import matplotlib.pyplot as plt
import seaborn as sns
#visualize data before modelling to detect trends and identify which features contribute to each other

# List of columns that will be printed (if they are numerical)
cols = ['Total Fossil Fuels Production', 'Nuclear Electric Power Production', 'Total Renewable Energy Production', 'Primary Energy Imports', 'Primary Energy Exports',
        'Primary Energy Net Imports', 'Primary Energy Stock Change and Other','Total Fossil Fuels Consumption', 'Nuclear Electric Power Consumption','Total Renewable Energy Consumption']

# Plot the distribution of numerical columns
sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
# plt.savefig('images/10_03.png', dpi=300)

plt.show()





#create a joint plot comparing fossil fuel consumption vs. total renewable energy consumption
#does renewable energy consumption increases as fossil fuel consumption decreases
sns.jointplot(data=df, x='Total Fossil Fuels Consumption', y='Total Renewable Energy Consumption', kind='scatter', color = 'pink')
plt.suptitle('Total Fossil Fuels Consumption vs. Total Renewable Energy Consumption')
#adjust title position to avoid overlap with the plot
plt.subplots_adjust(top=0.95)
plt.show()

#create a joint plot comparing total fossil fuel consumption vs total primary energy consumption
sns.jointplot(data=df, x='Total Fossil Fuels Consumption', y='Total Primary Energy Consumption', kind='scatter', color = 'orange')
plt.suptitle('Fossil Fuel Consumption vs. Total Energy Consumption')
plt.subplots_adjust(top=0.95)
plt.show()

sns.lmplot(data=df, x='Primary Energy Imports', y='Primary Energy Exports',line_kws={"color": "red"})
plt.title('Relationship between Primary Energy Imports and Exports')
plt.xlabel('Primary Energy Imports')
plt.ylabel('Primary Energy Exports')
plt.show()






import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Initialize the scaler and normalize the two production columns
scaler = StandardScaler()
df_normalized = df.copy()  # Create a copy to store normalized data
df_normalized[['Total Fossil Fuels Production', 'Total Renewable Energy Production']] = scaler.fit_transform(
    df[['Total Fossil Fuels Production', 'Total Renewable Energy Production']]
)

# Set up the figure with two subplots for comparison
plt.figure(figsize=(14, 10))

# Plot the non-normalized data
plt.subplot(2, 1, 1)
sns.lineplot(data=df, x='Annual Total', y='Total Fossil Fuels Production', label='Total Fossil Fuels Production', color='blue')
sns.lineplot(data=df, x='Annual Total', y='Total Renewable Energy Production', label='Total Renewable Energy Production', color='green')
plt.title('Total Fossil Fuels Production vs. Total Renewable Energy Production (Non-Normalized)')
plt.xlabel('Annual Total')
plt.ylabel('Production (Quadrillion Btu)')
plt.legend()

# Plot the normalized data
plt.subplot(2, 1, 2)
sns.lineplot(data=df_normalized, x='Annual Total', y='Total Fossil Fuels Production', label='Total Fossil Fuels Production (Normalized)', color='blue')
sns.lineplot(data=df_normalized, x='Annual Total', y='Total Renewable Energy Production', label='Total Renewable Energy Production (Normalized)', color='green')
plt.title('Total Fossil Fuels Production vs. Total Renewable Energy Production (Normalized)')
plt.xlabel('Annual Total')
plt.ylabel('Normalized Production')
plt.legend()

# Final layout adjustments and display
plt.tight_layout()
plt.show()






import numpy as np

cols = ['Total Fossil Fuels Production', 'Nuclear Electric Power Production', 'Total Renewable Energy Production', 'Primary Energy Imports', 'Primary Energy Exports',
        'Primary Energy Net Imports', 'Primary Energy Stock Change and Other','Total Fossil Fuels Consumption', 'Nuclear Electric Power Consumption','Total Renewable Energy Consumption']

# Compute the correlation matrix using numpy's corrcoef function, transposed to correlate features
cm = np.corrcoef(df[cols].values.T)

# Convert the correlation matrix to a DataFrame for easier manipulation
correlation_df = pd.DataFrame(cm, index=cols, columns=cols)

# Set the font scale for the heatmap labels
sns.set(font_scale=1)

# Set up the matplotlib figure with a larger size to accommodate the heatmap
plt.figure(figsize=(12, 10))

# Add a main title to the plot
plt.suptitle('Correlation Heat Map', fontsize=16)

# Create the heatmap using seaborn, with various configurations:
# - cbar=True: shows the color bar
# - annot=True: displays the correlation values on the heatmap
# - square=True: makes each cell square
# - fmt='.2f': formats the annotation to 2 decimal places
# - annot_kws: adjusts the size of the annotations
# - yticklabels and xticklabels: labels for the y and x axes
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 12},
                 yticklabels=cols,
                 xticklabels=cols)

# Adjust the layout to prevent overlap of labels and elements
plt.tight_layout()

# Show the plot on the screen
# plt.savefig('images/10_04.png', dpi=300)  # Saving the figure (commented out)
plt.show()  # Display the heatmap




from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Load data
df = pd.read_csv('annual data.csv',skiprows = [1])

target = 'Total Fossil Fuels Consumption'

cols = ['Total Fossil Fuels Production', 'Nuclear Electric Power Production', 'Total Renewable Energy Production', 'Primary Energy Imports', 'Primary Energy Exports',
        'Nuclear Electric Power Consumption','Total Renewable Energy Consumption']

X = df[cols].values
y = df[target].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Train the model
Mlr = LinearRegression()
Mlr.fit(X_train_std, y_train)

# Prediction
y_test_pred = Mlr.predict(X_test_std)

# Evaluation
r2 = r2_score(y_test, y_test_pred)
rmse = mean_squared_error(y_test, y_test_pred, squared=False)
print(f"R2 = {r2:.2f}")
print(f"RMSE = {rmse:.2f}")








import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Normalize the production columns
scaler = StandardScaler()
df_normalized = df.copy()
df_normalized[['Total Fossil Fuels Production', 'Total Renewable Energy Production']] = scaler.fit_transform(
    df[['Total Fossil Fuels Production', 'Total Renewable Energy Production']]
)

# Step 2: Define features (independent variable) and targets (dependent variables)
X = df[['Annual Total']].values  # Independent variable (year)
y_fossil = df[['Total Fossil Fuels Production']].values  # Target: Fossil fuels production
y_renewable = df[['Total Renewable Energy Production']].values  # Target: Renewable energy production

# Step 3: Train-Test Split
X_train, X_test, y_train_fossil, y_test_fossil = train_test_split(X, y_fossil, test_size=0.2, random_state=42)
X_train, X_test, y_train_renewable, y_test_renewable = train_test_split(X, y_renewable, test_size=0.2, random_state=42)

# Step 4: Build and Train the Linear Regression Models
model_fossil = LinearRegression()
model_fossil.fit(X_train, y_train_fossil)

model_renewable = LinearRegression()
model_renewable.fit(X_train, y_train_renewable)

# Step 5: Evaluate the Models
# Predictions
y_pred_fossil = model_fossil.predict(X_test)
y_pred_renewable = model_renewable.predict(X_test)

# Metrics
rmse_fossil = np.sqrt(mean_squared_error(y_test_fossil, y_pred_fossil))
r2_fossil = r2_score(y_test_fossil, y_pred_fossil)

rmse_renewable = np.sqrt(mean_squared_error(y_test_renewable, y_pred_renewable))
r2_renewable = r2_score(y_test_renewable, y_pred_renewable)

print(f"Fossil Fuels Model - RMSE: {rmse_fossil:.2f}, R²: {r2_fossil:.2f}")
print(f"Renewable Energy Model - RMSE: {rmse_renewable:.2f}, R²: {r2_renewable:.2f}")

# Step 6: Predict Energy Production for the Next 20 Years
future_years = np.arange(df['Annual Total'].max() + 1, df['Annual Total'].max() + 21).reshape(-1, 1)

future_fossil = model_fossil.predict(future_years)
future_renewable = model_renewable.predict(future_years)

# Step 7: Visualize Historical Data and Predictions
plt.figure(figsize=(14, 8))

# Plot historical data
plt.plot(df['Annual Total'], df['Total Fossil Fuels Production'], label='Fossil Fuels Production (Actual)', color='blue')
plt.plot(df['Annual Total'], df['Total Renewable Energy Production'], label='Renewable Energy Production (Actual)', color='green')

# Plot future predictions
plt.plot(future_years, future_fossil, label='Fossil Fuels Production (Predicted)', linestyle='--', color='blue')
plt.plot(future_years, future_renewable, label='Renewable Energy Production (Predicted)', linestyle='--', color='green')

# Add labels and legend
plt.title('Energy Production: Historical Data and 20-Year Predictions')
plt.xlabel('Year')
plt.ylabel('Production (Quadrillion Btu)')
plt.legend()
plt.grid()
plt.show()







from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np


import pandas as pd
import warnings
warnings.filterwarnings('ignore')

excel_file_path = '/annual data .xlsx'
df = pd.read_excel(excel_file_path)
#write the excel to a csv so that it is easier to interpret when reading it into functions
df.to_csv('annual data.csv', index=False)

df = pd.read_csv('annual data.csv', skiprows = 1)

df.columns = ['Annual Total', 'Total Fossil Fuels Production', 'Nuclear Electric Power Production', 'Total Renewable Energy Production', 'Total Primary Energy Production', 'Primary Energy Imports', 'Primary Energy Exports',
              'Primary Energy Net Imports', 'Primary Energy Stock Change and Other','Total Fossil Fuels Consumption', 'Nuclear Electric Power Consumption','Total Renewable Energy Consumption','Total Primary Energy Consumption']

df.head()

# Step 1: Define Features and Targets
X = df[['Annual Total']].values  # Independent variable (year)
y_fossil = df[['Total Fossil Fuels Production']].values  # Target: Fossil fuels production
y_renewable = df[['Total Renewable Energy Production']].values  # Target: Renewable energy production

# Step 2: Apply Polynomial Features
degree = 5 # You can experiment with different degrees
poly = PolynomialFeatures(degree=degree, include_bias=False)

X_poly = poly.fit_transform(X)  # Transform the features into polynomial features

# Step 3: Train-Test Split
X_train, X_test, y_train_fossil, y_test_fossil = train_test_split(X_poly, y_fossil, test_size=0.2, random_state=42)
X_train, X_test, y_train_renewable, y_test_renewable = train_test_split(X_poly, y_renewable, test_size=0.2, random_state=42)

# Step 4: Build and Train Polynomial Regression Models
model_fossil_poly = LinearRegression()
model_fossil_poly.fit(X_train, y_train_fossil)

model_renewable_poly = LinearRegression()
model_renewable_poly.fit(X_train, y_train_renewable)

# Step 5: Evaluate the Models
# Fossil Fuels
y_pred_fossil_poly = model_fossil_poly.predict(X_test)
rmse_fossil_poly = np.sqrt(mean_squared_error(y_test_fossil, y_pred_fossil_poly))
r2_fossil_poly = r2_score(y_test_fossil, y_pred_fossil_poly)

# Renewable Energy
y_pred_renewable_poly = model_renewable_poly.predict(X_test)
rmse_renewable_poly = np.sqrt(mean_squared_error(y_test_renewable, y_pred_renewable_poly))
r2_renewable_poly = r2_score(y_test_renewable, y_pred_renewable_poly)

# Print the Results
print(f"Polynomial Regression (Degree: {degree}) - Fossil Fuels Model:")
print(f"RMSE: {rmse_fossil_poly:.2f}, R²: {r2_fossil_poly:.2f}\n")

print(f"Polynomial Regression (Degree: {degree}) - Renewable Energy Model:")
print(f"RMSE: {rmse_renewable_poly:.2f}, R²: {r2_renewable_poly:.2f}\n")

# Step 6: Predict Future Values (Next 20 Years)
future_years = np.arange(df['Annual Total'].max() + 1, df['Annual Total'].max() + 21).reshape(-1, 1)
future_years_poly = poly.transform(future_years)  # Apply the same polynomial transformation

future_fossil_poly = model_fossil_poly.predict(future_years_poly)
future_renewable_poly = model_renewable_poly.predict(future_years_poly)

# Step 7: Plot Results
plt.figure(figsize=(14, 8))

# Historical data
plt.plot(df['Annual Total'], df['Total Fossil Fuels Production'], label='Fossil Fuels Production (Actual)', color='blue')
plt.plot(df['Annual Total'], df['Total Renewable Energy Production'], label='Renewable Energy Production (Actual)', color='green')

# Predictions
plt.plot(future_years, future_fossil_poly, label='Fossil Fuels Production (Predicted)', linestyle='--', color='blue')
plt.plot(future_years, future_renewable_poly, label='Renewable Energy Production (Predicted)', linestyle='--', color='green')

# Plot settings
plt.title('Energy Production: Polynomial Regression Predictions')
plt.xlabel('Year')
plt.ylabel('Production (Quadrillion Btu)')
plt.legend()
plt.grid()
plt.show()







from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


import pandas as pd
import warnings
warnings.filterwarnings('ignore')

excel_file_path = '/annual data .xlsx'
df = pd.read_excel(excel_file_path)
#write the excel to a csv so that it is easier to interpret when reading it into functions
df.to_csv('annual data.csv', index=False)

df = pd.read_csv('annual data.csv', skiprows = 1)

df.columns = ['Annual Total', 'Total Fossil Fuels Production', 'Nuclear Electric Power Production', 'Total Renewable Energy Production', 'Total Primary Energy Production', 'Primary Energy Imports', 'Primary Energy Exports',
              'Primary Energy Net Imports', 'Primary Energy Stock Change and Other','Total Fossil Fuels Consumption', 'Nuclear Electric Power Consumption','Total Renewable Energy Consumption','Total Primary Energy Consumption']

df.head()

# Step 1: Define Features and Targets
X = df[['Annual Total']].values  # Independent variable (year)
y_fossil = df[['Total Fossil Fuels Production']].values  # Target: Fossil fuels production
y_renewable = df[['Total Renewable Energy Production']].values  # Target: Renewable energy production

# Step 2: Train-Test Split
X_train, X_test, y_train_fossil, y_test_fossil = train_test_split(X, y_fossil, test_size=0.2, random_state=42)
X_train, X_test, y_train_renewable, y_test_renewable = train_test_split(X, y_renewable, test_size=0.2, random_state=42)

# Step 3: Build and Train Random Forest Models
# Fossil Fuels
model_fossil_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_fossil_rf.fit(X_train, y_train_fossil.ravel())

# Renewable Energy
model_renewable_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_renewable_rf.fit(X_train, y_train_renewable.ravel())

# Step 4: Evaluate the Models
# Fossil Fuels
y_pred_fossil_rf = model_fossil_rf.predict(X_test)
rmse_fossil_rf = np.sqrt(mean_squared_error(y_test_fossil, y_pred_fossil_rf))
r2_fossil_rf = r2_score(y_test_fossil, y_pred_fossil_rf)

# Renewable Energy
y_pred_renewable_rf = model_renewable_rf.predict(X_test)
rmse_renewable_rf = np.sqrt(mean_squared_error(y_test_renewable, y_pred_renewable_rf))
r2_renewable_rf = r2_score(y_test_renewable, y_pred_renewable_rf)

# Print the Results
print(f"Random Forest Regression - Fossil Fuels Model:")
print(f"RMSE: {rmse_fossil_rf:.2f}, R²: {r2_fossil_rf:.2f}\n")

print(f"Random Forest Regression - Renewable Energy Model:")
print(f"RMSE: {rmse_renewable_rf:.2f}, R²: {r2_renewable_rf:.2f}\n")

# Step 5: Predict Future Values (Next 20 Years)
future_years = np.arange(df['Annual Total'].max() + 1, df['Annual Total'].max() + 21).reshape(-1, 1)

future_fossil_rf = model_fossil_rf.predict(future_years)
future_renewable_rf = model_renewable_rf.predict(future_years)

# Step 6: Plot Results
plt.figure(figsize=(14, 8))

# Historical data
plt.plot(df['Annual Total'], df['Total Fossil Fuels Production'], label='Fossil Fuels Production (Actual)', color='blue')
plt.plot(df['Annual Total'], df['Total Renewable Energy Production'], label='Renewable Energy Production (Actual)', color='green')

# Predictions
plt.plot(future_years, future_fossil_rf, label='Fossil Fuels Production (Predicted)', linestyle='--', color='blue')
plt.plot(future_years, future_renewable_rf, label='Renewable Energy Production (Predicted)', linestyle='--', color='green')

# Plot settings
plt.title('Energy Production: Random Forest Predictions')
plt.xlabel('Year')
plt.ylabel('Production (Quadrillion Btu)')
plt.legend()
plt.grid()
plt.show()




from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Define Features and Targets
X = df[['Annual Total']].values  # Independent variable (year)
y_imports = df[['Primary Energy Imports']].values  # Target: Energy Imports
y_exports = df[['Primary Energy Exports']].values  # Target: Energy Exports

# Step 2: Apply Polynomial Features
degree = 5  # Degree of the polynomial
poly = PolynomialFeatures(degree=degree, include_bias=False)

X_poly = poly.fit_transform(X)  # Transform the features into polynomial features

# Step 3: Train-Test Split for Imports
X_train, X_test, y_train_imports, y_test_imports = train_test_split(X_poly, y_imports, test_size=0.2, random_state=42)

# Train-Test Split for Exports
_, _, y_train_exports, y_test_exports = train_test_split(X_poly, y_exports, test_size=0.2, random_state=42)

# Step 4: Build and Train Polynomial Regression Models
# Imports Model
model_imports_poly = LinearRegression()
model_imports_poly.fit(X_train, y_train_imports)

# Exports Model
model_exports_poly = LinearRegression()
model_exports_poly.fit(X_train, y_train_exports)

# Step 5: Evaluate the Models
# Imports
y_pred_imports_poly = model_imports_poly.predict(X_test)
rmse_imports_poly = np.sqrt(mean_squared_error(y_test_imports, y_pred_imports_poly))
r2_imports_poly = r2_score(y_test_imports, y_pred_imports_poly)

# Exports
y_pred_exports_poly = model_exports_poly.predict(X_test)
rmse_exports_poly = np.sqrt(mean_squared_error(y_test_exports, y_pred_exports_poly))
r2_exports_poly = r2_score(y_test_exports, y_pred_exports_poly)

# Print the Results
print(f"Polynomial Regression (Degree: {degree}) - Imports Model:")
print(f"RMSE: {rmse_imports_poly:.2f}, R²: {r2_imports_poly:.2f}\n")

print(f"Polynomial Regression (Degree: {degree}) - Exports Model:")
print(f"RMSE: {rmse_exports_poly:.2f}, R²: {r2_exports_poly:.2f}\n")

# Step 6: Predict Future Values (Next 20 Years)
future_years = np.arange(df['Annual Total'].max() + 1, df['Annual Total'].max() + 21).reshape(-1, 1)
future_years_poly = poly.transform(future_years)  # Apply the same polynomial transformation

future_imports_poly = model_imports_poly.predict(future_years_poly)
future_exports_poly = model_exports_poly.predict(future_years_poly)

# Step 7: Plot Results
plt.figure(figsize=(14, 8))

# Historical data for Imports
plt.plot(df['Annual Total'], df['Primary Energy Imports'], label='Imports (Actual)', color='blue')
# Predictions for Imports
plt.plot(future_years, future_imports_poly, label='Imports (Predicted)', linestyle='--', color='blue')

# Historical data for Exports
plt.plot(df['Annual Total'], df['Primary Energy Exports'], label='Exports (Actual)', color='green')
# Predictions for Exports
plt.plot(future_years, future_exports_poly, label='Exports (Predicted)', linestyle='--', color='green')

# Plot settings
plt.title('Energy Imports and Exports: Polynomial Regression Predictions (Degree 5)')
plt.xlabel('Year')
plt.ylabel('Energy (Quadrillion Btu)')
plt.legend()
plt.grid()
plt.show()





# this is for totla prime energy consumption. i did random forest model below. you guys can pick the better one. I ALSO DID SIMPLE LINEAR REGRESSION AS WELL.



from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define Features and Target
X = df[['Annual Total']].values  # Independent variable (year)
y_total_consumption = df[['Total Primary Energy Consumption']].values  # Target: Total Primary Energy Consumption

# Step 2: Apply Polynomial Features
degree = 5

  # Set the polynomial degree
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(X)  # Transform the features into polynomial features

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y_total_consumption, test_size=0.2, random_state=42)

# Step 4: Build and Train Polynomial Regression Model
model_poly = LinearRegression()
model_poly.fit(X_train, y_train)

# Step 5: Evaluate the Model
y_pred = model_poly.predict(X_test)
rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred))
r2_poly = r2_score(y_test, y_pred)

# Print the Results
print(f"Polynomial Regression (Degree: {degree}) - Total Primary Energy Consumption:")
print(f"RMSE: {rmse_poly:.2f}")
print(f"R²: {r2_poly:.2f}")

# Step 6: Predict Future Values (Next 20 Years)
future_years = np.arange(df['Annual Total'].max() + 1, df['Annual Total'].max() + 21).reshape(-1, 1)
future_years_poly = poly.transform(future_years)  # Apply the same polynomial transformation
future_predictions = model_poly.predict(future_years_poly)

# Step 7: Plot Results
plt.figure(figsize=(14, 8))

# Historical data
plt.plot(df['Annual Total'], df['Total Primary Energy Consumption'], label='Total Primary Energy Consumption (Actual)', color='purple')

# Predictions
plt.plot(future_years, future_predictions, label='Total Primary Energy Consumption (Predicted)', linestyle='--', color='orange')

# Plot settings
plt.title(f'Total Primary Energy Consumption: Polynomial Regression Predictions (Degree {degree})')
plt.xlabel('Year')
plt.ylabel('Energy Consumption (Quadrillion Btu)')
plt.legend()
plt.grid()
plt.show()



from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Define Features and Target
X = df[['Annual Total']].values  # Independent variable (year)
y_total_consumption = df[['Total Primary Energy Consumption']].values.ravel()  # Target: Total Primary Energy Consumption

# Step 2: Add Polynomial Features (Optional for Non-Linearity)
degree = 3  # Polynomial degree
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(X)

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y_total_consumption, test_size=0.2, random_state=42)

# Step 4: Build and Train XGBoost Model
model_xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
model_xgb.fit(X_train, y_train)

# Step 5: Evaluate the Model
y_pred = model_xgb.predict(X_test)

# Calculate RMSE and R²
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Print Results
print(f"XGBoost Regression - Total Primary Energy Consumption:")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# Step 6: Predict Future Values (Next 20 Years)
future_years = np.arange(df['Annual Total'].max() + 1, df['Annual Total'].max() + 21).reshape(-1, 1)
future_years_poly = poly.transform(future_years)  # Transform future years to match polynomial features
future_predictions = model_xgb.predict(future_years_poly)

# Step 7: Add a Linear Trend Component to Future Predictions
trend = np.linspace(0, 5, len(future_predictions))  # Linear upward trend
future_predictions_with_trend = future_predictions + trend

# Step 8: Plot Results
plt.figure(figsize=(14, 8))

# Historical data
plt.plot(df['Annual Total'], df['Total Primary Energy Consumption'], label='Total Primary Energy Consumption (Actual)', color='purple')

# Predictions with trend
plt.plot(future_years, future_predictions_with_trend, label='Total Primary Energy Consumption (Predicted - XGBoost with Trend)', linestyle='--', color='orange')

# Plot settings
plt.title(f'Total Primary Energy Consumption: XGBoost Regression with Trend Adjustment (Degree {degree})')
plt.xlabel('Year')
plt.ylabel('Energy Consumption (Quadrillion Btu)')
plt.legend()
plt.grid()
plt.show()
