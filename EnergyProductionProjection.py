import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

def load_and_prepare_data(excel_file_path):
    """
    Load and prepare the energy production data.
    
    Args:
        excel_file_path (str): Path to the Excel file
    
    Returns:
        pd.DataFrame: Prepared dataframe
    """
    # Read Excel file, skipping the first row
    df = pd.read_excel('EnergyDataSS.xlsx', header=10)
    
    # Inspect the columns
    print("Original Columns:", df.columns.tolist())
    
    # Identify key columns dynamically
    year_column = [col for col in df.columns if 'year' in col.lower() or 'total' in col.lower()][0]
    fossil_columns = [col for col in df.columns if 'fossil' in col.lower() and 'production' in col.lower()]
    renewable_columns = [col for col in df.columns if 'renewable' in col.lower() and 'production' in col.lower()]
    
    # Display identified columns
    print("Year Column:", year_column)
    print("Fossil Columns:", fossil_columns)
    print("Renewable Columns:", renewable_columns)
    
    # Rename columns for consistency
    rename_dict = {
        year_column: 'Annual Total',
        fossil_columns[0] if fossil_columns else None: 'Total Fossil Fuels Production',
        renewable_columns[0] if renewable_columns else None: 'Total Renewable Energy Production'
    }
    
    rename_dict = {k: v for k, v in rename_dict.items() if k is not None and v is not None}
    df = df.rename(columns=rename_dict)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    for col in rename_dict.values():
        df[col] = imputer.fit_transform(df[[col]])
    
    return df[list(rename_dict.values())]

def train_polynomial_regression_models(df, degree=5):
    """
    Train polynomial regression models for fossil fuels and renewable energy.
    
    Args:
        df (pd.DataFrame): Input dataframe
        degree (int): Degree of polynomial features
    
    Returns:
        tuple: Trained models, polynomial transformer, performance metrics
    """
    X = df[['Annual Total']].values
    y_fossil = df[['Total Fossil Fuels Production']].values
    y_renewable = df[['Total Renewable Energy Production']].values
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train_fossil, y_test_fossil = train_test_split(X_poly, y_fossil, test_size=0.2, random_state=42)
    _, _, y_train_renewable, y_test_renewable = train_test_split(X_poly, y_renewable, test_size=0.2, random_state=42)
    
    # Train models
    fossil_model = LinearRegression()
    fossil_model.fit(X_train, y_train_fossil)
    
    renewable_model = LinearRegression()
    renewable_model.fit(X_train, y_train_renewable)
    
    # Evaluate models
    y_pred_fossil = fossil_model.predict(X_test)
    rmse_fossil = np.sqrt(mean_squared_error(y_test_fossil, y_pred_fossil))
    r2_fossil = r2_score(y_test_fossil, y_pred_fossil)
    
    y_pred_renewable = renewable_model.predict(X_test)
    rmse_renewable = np.sqrt(mean_squared_error(y_test_renewable, y_pred_renewable))
    r2_renewable = r2_score(y_test_renewable, y_pred_renewable)
    
    # Display metrics
    print(f"Fossil Fuels Model - RMSE: {rmse_fossil:.2f}, R²: {r2_fossil:.2f}")
    print(f"Renewable Energy Model - RMSE: {rmse_renewable:.2f}, R²: {r2_renewable:.2f}")
    
    return (fossil_model, renewable_model), poly

def predict_future_energy_production(df, models, poly, prediction_years=20):
    """
    Predict future energy production for fossil fuels and renewable energy.
    
    Args:
        df (pd.DataFrame): Input dataframe
        models (tuple): Trained models
        poly (PolynomialFeatures): Polynomial transformer
        prediction_years (int): Years to predict into the future
    
    Returns:
        tuple: Future years, fossil fuels predictions, renewable energy predictions
    """
    future_years = np.arange(df['Annual Total'].max() + 1, df['Annual Total'].max() + prediction_years + 1).reshape(-1, 1)
    future_years_poly = poly.transform(future_years)
    
    fossil_model, renewable_model = models
    future_fossil = fossil_model.predict(future_years_poly)
    future_renewable = renewable_model.predict(future_years_poly)
    
    return future_years, future_fossil, future_renewable

def visualize_energy_production(df, future_years, future_fossil, future_renewable):
    """
    Visualize historical and predicted energy production.
    
    Args:
        df (pd.DataFrame): Historical dataframe
        future_years (np.ndarray): Future years for prediction
        future_fossil (np.ndarray): Predicted fossil fuels production
        future_renewable (np.ndarray): Predicted renewable energy production
    """
    plt.figure(figsize=(16, 9))
    
    # Historical stacked areas
    plt.fill_between(df['Annual Total'], 0, df['Total Fossil Fuels Production'], color='blue', alpha=0.3, label='Fossil Fuels (Historical)')
    plt.fill_between(df['Annual Total'], df['Total Fossil Fuels Production'], df['Total Fossil Fuels Production'] + df['Total Renewable Energy Production'], color='green', alpha=0.3, label='Renewable Energy (Historical)')
    
    # Predicted stacked areas
    plt.fill_between(future_years.flatten(), 0, future_fossil.flatten(), color='blue', alpha=0.5, linestyle='--', label='Fossil Fuels (Predicted)')
    plt.fill_between(future_years.flatten(), future_fossil.flatten(), future_fossil.flatten() + future_renewable.flatten(), color='green', alpha=0.5, linestyle='--', label='Renewable Energy (Predicted)')
    
    plt.plot(df['Annual Total'], df['Total Fossil Fuels Production'] + df['Total Renewable Energy Production'], color='purple', linewidth=2, label='Total Energy (Historical)')
    plt.plot(future_years.flatten(), future_fossil.flatten() + future_renewable.flatten(), color='red', linestyle='--', linewidth=2, label='Total Energy (Predicted)')
    
    plt.title("Energy Production: Historical and Predicted", fontsize=16)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Production (Quadrillion Btu)", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.7)
    plt.tight_layout()
    plt.show()

def predict_adjusted_future_energy_production(df, models, poly, prediction_years=20):
    """
    Predict future energy production for fossil fuels and renewable energy,
    applying 2% yearly increase to renewables and 6% yearly decrease to fossils.

    Args:
        df (pd.DataFrame): Input dataframe
        models (tuple): Trained fossil and renewable energy models
        poly (PolynomialFeatures): Polynomial feature transformer
        prediction_years (int): Number of years to predict into the future

    Returns:
        tuple: Future years, fossil fuels predictions, renewable energy predictions
    """
    # Prepare future years
    future_years = np.arange(df['Annual Total'].max() + 1, df['Annual Total'].max() + prediction_years + 1).reshape(-1, 1)
    future_years_poly = poly.transform(future_years)
    
    # Initial predictions from the models
    model_fossil_poly, model_renewable_poly = models
    future_fossil_poly = model_fossil_poly.predict(future_years_poly).flatten()
    future_renewable_poly = model_renewable_poly.predict(future_years_poly).flatten()
    
    # Apply adjustments year-by-year
    adjusted_fossil_poly = []
    adjusted_renewable_poly = []
    
    for i, year in enumerate(future_years.flatten()):
        if i == 0:  # Start with the first predicted value
            adjusted_fossil_poly.append(future_fossil_poly[i])
            adjusted_renewable_poly.append(future_renewable_poly[i])
        else:
            # Adjust fossil and renewable values based on previous year's data
            new_fossil = adjusted_fossil_poly[-1] * (1 - 0.06)  # For Scenario 2 change the subtracting value to 0.06 and for Scenario 3 change it to 0.02
            new_renewable = adjusted_renewable_poly[-1] * (1 + 0.02)  # For Scenario 2 change the adding value to 0.02 and for Scenario 3 change it to 0.06
            adjusted_fossil_poly.append(new_fossil)
            adjusted_renewable_poly.append(new_renewable)
    
    return future_years, np.array(adjusted_fossil_poly), np.array(adjusted_renewable_poly)

def main():
    # File path (modify as needed)
    excel_file_path = ('EnergyDataSS.xlsx')
    
    # Load and prepare data
    df = load_and_prepare_data(excel_file_path)
    
    # Train models
    models, poly = train_polynomial_regression_models(df)
    
    # Predict future production with adjustments
    future_years, future_fossil_poly, future_renewable_poly = predict_adjusted_future_energy_production(df, models, poly)
    
    # Visualize results
    visualize_energy_production(df, future_years, future_fossil_poly, future_renewable_poly)

if __name__ == "__main__":
    main()

