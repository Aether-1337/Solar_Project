import seaborn.objects as so
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def get_solar_cycle_df():
    """Fetch solar cycle data from NOAA"""
    urls = [
        'https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json',
        'https://services.swpc.noaa.gov/json/solar-cycle/predicted-solar-cycle.json'
    ]

    dataframes = []

    for url in urls:
        response = requests.get(url)
        if response.status_code == 200 and response.text.strip():
            try:
                data = response.json()
                df = pd.DataFrame(data)
                dataframes.append(df)
            except ValueError as e:
                print("JSON decode error:", e)
        else:
            print("Empty or failed response:", response.status_code)

    return dataframes


def create_features(df):
    """Create time-based and lagged features for ML model"""
    df = df.copy()
    df['time-tag'] = pd.to_datetime(df['time-tag'])
    df = df.sort_values('time-tag')
    
    # Time-based features
    df['year'] = df['time-tag'].dt.year
    df['month'] = df['time-tag'].dt.month
    df['year_fraction'] = df['year'] + (df['month'] - 1) / 12
    
    # Cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Lagged features (previous values)
    for lag in [1, 3, 6, 12, 24]:
        df[f'lag_{lag}'] = df['smoothed_ssn'].shift(lag)
    
    # Rolling statistics
    df['rolling_mean_12'] = df['smoothed_ssn'].rolling(window=12, min_periods=1).mean()
    df['rolling_std_12'] = df['smoothed_ssn'].rolling(window=12, min_periods=1).std()
    df['rolling_mean_24'] = df['smoothed_ssn'].rolling(window=24, min_periods=1).mean()
    
    # Rate of change
    df['rate_of_change'] = df['smoothed_ssn'].diff()
    df['rate_of_change_12'] = df['smoothed_ssn'].diff(12)
    
    return df


def train_ml_model(df):
    """Train Gradient Boosting model on observed data"""
    # Remove rows with NaN values from feature engineering
    df_clean = df.dropna()
    
    feature_cols = [
        'year_fraction', 'month_sin', 'month_cos',
        'lag_1', 'lag_3', 'lag_6', 'lag_12', 'lag_24',
        'rolling_mean_12', 'rolling_std_12', 'rolling_mean_24',
        'rate_of_change', 'rate_of_change_12'
    ]
    
    X = df_clean[feature_cols]
    y = df_clean['smoothed_ssn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    return model, scaler, feature_cols


def predict_future(df, model, scaler, feature_cols, months_ahead=60):
    """Generate future predictions using the trained model"""
    last_date = df['time-tag'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                  periods=months_ahead, freq='MS')
    
    predictions = []
    current_df = df.copy()
    
    for future_date in future_dates:
        # Create a new row for prediction
        new_row = pd.DataFrame({
            'time-tag': [future_date],
            'smoothed_ssn': [np.nan]
        })
        
        current_df = pd.concat([current_df, new_row], ignore_index=True)
        current_df = create_features(current_df)
        
        # Get the last row features
        last_row = current_df.iloc[-1]
        
        # Check if we have all features
        if last_row[feature_cols].isna().any():
            # Fill missing lags with last known values
            for col in feature_cols:
                if pd.isna(last_row[col]):
                    current_df.loc[current_df.index[-1], col] = current_df[col].iloc[-2]
        
        X_future = current_df.iloc[-1][feature_cols].values.reshape(1, -1)
        X_future_scaled = scaler.transform(X_future)
        
        prediction = model.predict(X_future_scaled)[0]
        current_df.loc[current_df.index[-1], 'smoothed_ssn'] = prediction
        predictions.append(prediction)
    
    future_df = pd.DataFrame({
        'time-tag': future_dates,
        'smoothed_ssn': predictions,
        'source': 'ML Prediction'
    })
    
    return future_df


# Main execution
print("Fetching data from NOAA...")
dataframes = get_solar_cycle_df()

if len(dataframes) == 2:
    observed_df = dataframes[0]
    predicted_df = dataframes[1]
    
    # Standardize column names
    if 'predicted_ssn' in predicted_df.columns:
        predicted_df = predicted_df.rename(columns={'predicted_ssn': 'smoothed_ssn'})
    
    # Convert to datetime
    observed_df['time-tag'] = pd.to_datetime(observed_df['time-tag'])
    predicted_df['time-tag'] = pd.to_datetime(predicted_df['time-tag'])
    
    # Add source labels
    observed_df['source'] = 'Observed'
    predicted_df['source'] = 'NOAA Prediction'
    
    print(f"\nObserved data: {len(observed_df)} records")
    print(f"NOAA predictions: {len(predicted_df)} records")
    
    # Create features and train model on observed data
    print("\nTraining ML model on observed data...")
    observed_features = create_features(observed_df)
    model, scaler, feature_cols = train_ml_model(observed_features)
    
    # Generate ML predictions
    print("\nGenerating ML predictions for next 60 months...")
    ml_predictions = predict_future(observed_features, model, scaler, feature_cols, months_ahead=60)
    
    # Combine all data
    combined_df = pd.concat([
        observed_df[['time-tag', 'smoothed_ssn', 'source']],
        predicted_df[['time-tag', 'smoothed_ssn', 'source']],
        ml_predictions
    ], ignore_index=True)
    
    # ===== VISUALIZATION 1: Main Comparison Plot =====
    print("\nCreating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Advanced Solar Cycle Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: All predictions comparison
    ax1 = axes[0, 0]
    for source in combined_df['source'].unique():
        data = combined_df[combined_df['source'] == source]
        if source == 'Observed':
            ax1.plot(data['time-tag'], data['smoothed_ssn'], 
                    label=source, linewidth=2, alpha=0.9)
        elif source == 'NOAA Prediction':
            ax1.plot(data['time-tag'], data['smoothed_ssn'], 
                    label=source, linewidth=2, linestyle='--', alpha=0.7)
        else:
            ax1.plot(data['time-tag'], data['smoothed_ssn'], 
                    label=source, linewidth=2, linestyle=':', alpha=0.7)
    
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Smoothed Sunspot Number', fontsize=11)
    ax1.set_title('Observed vs Predicted Solar Activity', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Recent activity (last 10 years) with predictions
    ax2 = axes[0, 1]
    recent_cutoff = pd.Timestamp.now() - pd.DateOffset(years=10)
    recent_df = combined_df[combined_df['time-tag'] >= recent_cutoff]
    
    for source in recent_df['source'].unique():
        data = recent_df[recent_df['source'] == source]
        if source == 'Observed':
            ax2.plot(data['time-tag'], data['smoothed_ssn'], 
                    label=source, linewidth=2.5, alpha=0.9)
        elif source == 'NOAA Prediction':
            ax2.plot(data['time-tag'], data['smoothed_ssn'], 
                    label=source, linewidth=2, linestyle='--', alpha=0.7)
        else:
            ax2.plot(data['time-tag'], data['smoothed_ssn'], 
                    label=source, linewidth=2, linestyle=':', alpha=0.7)
    
    ax2.axvline(x=pd.Timestamp.now(), color='red', linestyle='-', 
               linewidth=1, alpha=0.5, label='Today')
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Smoothed Sunspot Number', fontsize=11)
    ax2.set_title('Recent Activity & Future Predictions (Last 10 Years)', 
                 fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distribution of sunspot numbers by source
    ax3 = axes[1, 0]
    for source in combined_df['source'].unique():
        data = combined_df[combined_df['source'] == source]['smoothed_ssn'].dropna()
        ax3.hist(data, bins=30, alpha=0.5, label=source, edgecolor='black')
    
    ax3.set_xlabel('Smoothed Sunspot Number', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Distribution of Solar Activity', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Year-over-year comparison
    ax4 = axes[1, 1]
    observed_only = combined_df[combined_df['source'] == 'Observed'].copy()
    observed_only['year'] = observed_only['time-tag'].dt.year
    
    yearly_stats = observed_only.groupby('year')['smoothed_ssn'].agg(['mean', 'max', 'min'])
    yearly_stats = yearly_stats[yearly_stats.index >= 1950]  # Focus on recent decades
    
    ax4.plot(yearly_stats.index, yearly_stats['mean'], 
            label='Mean', linewidth=2, marker='o', markersize=3)
    ax4.fill_between(yearly_stats.index, yearly_stats['min'], yearly_stats['max'], 
                     alpha=0.3, label='Min-Max Range')
    ax4.set_xlabel('Year', fontsize=11)
    ax4.set_ylabel('Smoothed Sunspot Number', fontsize=11)
    ax4.set_title('Annual Solar Activity Statistics (1950+)', 
                 fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # ===== VISUALIZATION 2: Detailed ML Analysis =====
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle('Machine Learning Model Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Feature importance
    ax1 = axes2[0, 0]
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    ax1.barh(range(len(feature_importance)), feature_importance['importance'])
    ax1.set_yticks(range(len(feature_importance)))
    ax1.set_yticklabels(feature_importance['feature'])
    ax1.set_xlabel('Importance', fontsize=11)
    ax1.set_title('Feature Importance in ML Model', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: ML vs NOAA predictions comparison
    ax2 = axes2[0, 1]
    future_only = combined_df[combined_df['time-tag'] > observed_df['time-tag'].max()]
    
    for source in future_only['source'].unique():
        data = future_only[future_only['source'] == source]
        if source == 'NOAA Prediction':
            ax2.plot(data['time-tag'], data['smoothed_ssn'], 
                    label=source, linewidth=2.5, linestyle='--', marker='o', 
                    markersize=4, alpha=0.7)
        else:
            ax2.plot(data['time-tag'], data['smoothed_ssn'], 
                    label=source, linewidth=2.5, linestyle=':', marker='s', 
                    markersize=4, alpha=0.7)
    
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Smoothed Sunspot Number', fontsize=11)
    ax2.set_title('ML vs NOAA Future Predictions', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Prediction confidence intervals
    ax3 = axes2[1, 0]
    ml_pred_data = combined_df[combined_df['source'] == 'ML Prediction']
    
    # Create simple confidence intervals (this is illustrative)
    uncertainty = np.linspace(5, 20, len(ml_pred_data))
    upper_bound = ml_pred_data['smoothed_ssn'].values + uncertainty
    lower_bound = ml_pred_data['smoothed_ssn'].values - uncertainty
    
    ax3.plot(ml_pred_data['time-tag'], ml_pred_data['smoothed_ssn'], 
            label='ML Prediction', linewidth=2.5, color='purple')
    ax3.fill_between(ml_pred_data['time-tag'], lower_bound, upper_bound, 
                     alpha=0.3, color='purple', label='Uncertainty Range')
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_ylabel('Smoothed Sunspot Number', fontsize=11)
    ax3.set_title('ML Predictions with Uncertainty', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Residual analysis on training data
    ax4 = axes2[1, 1]
    observed_clean = observed_features.dropna()
    X_all = observed_clean[feature_cols]
    y_all = observed_clean['smoothed_ssn']
    X_all_scaled = scaler.transform(X_all)
    y_pred_all = model.predict(X_all_scaled)
    residuals = y_all - y_pred_all
    
    ax4.scatter(y_pred_all, residuals, alpha=0.5, s=20)
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Predicted Values', fontsize=11)
    ax4.set_ylabel('Residuals', fontsize=11)
    ax4.set_title('Model Residual Analysis', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"\nLast observed date: {observed_df['time-tag'].max().strftime('%Y-%m-%d')}")
    print(f"Last observed sunspot number: {observed_df['smoothed_ssn'].iloc[-1]:.2f}")
    print(f"\nML prediction for 12 months ahead: {ml_predictions.iloc[11]['smoothed_ssn']:.2f}")
    print(f"ML prediction for 24 months ahead: {ml_predictions.iloc[23]['smoothed_ssn']:.2f}")
    print(f"ML prediction for 60 months ahead: {ml_predictions.iloc[-1]['smoothed_ssn']:.2f}")
    
    if len(predicted_df) > 0:
        print(f"\nNOAA prediction for 12 months ahead: {predicted_df.iloc[11]['smoothed_ssn']:.2f}")
        print(f"Difference (ML - NOAA): {ml_predictions.iloc[11]['smoothed_ssn'] - predicted_df.iloc[11]['smoothed_ssn']:.2f}")
    
    print("\n" + "="*60)

else:
    print("Failed to load datasets from NOAA.")