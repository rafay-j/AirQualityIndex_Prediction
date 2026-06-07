import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
def create_lag_features(df, columns, lags=[1, 2, 3, 6, 12, 24]):
    df = df.copy()
    for col in columns:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df
def create_rolling_features(df, columns, windows=[3, 6, 12, 24]):
    df = df.copy()
    for col in columns:
        if col in df.columns:
            for window in windows:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
                df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
                df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
    return df
def create_time_features(df, timestamp_col='timestamp'):
    df = df.copy()
    if timestamp_col not in df.columns:
        return df
    df['datetime'] = pd.to_datetime(df[timestamp_col])
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_month'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['quarter'] = df['datetime'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) |
                        (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    return df
def create_interaction_features(df):
    df = df.copy()
    if 'pm25' in df.columns and 'pm10' in df.columns:
        df['pm25_pm10_ratio'] = df['pm25'] / (df['pm10'] + 1)
        df['pm_total'] = df['pm25'] + df['pm10']
    pollutants = ['pm25', 'pm10', 'co', 'no2', 'o3', 'so2']
    available = [p for p in pollutants if p in df.columns]
    if len(available) > 0:
        df['pollutant_sum'] = df[available].sum(axis=1)
        df['pollutant_mean'] = df[available].mean(axis=1)
        df['pollutant_std'] = df[available].std(axis=1)
    if 'no2' in df.columns and 'o3' in df.columns:
        df['no2_o3_product'] = df['no2'] * df['o3']
    if 'temp' in df.columns and 'humidity' in df.columns:
        df['temp_humidity_product'] = df['temp'] * df['humidity']
        df['heat_index'] = df['temp'] + 0.5 * df['humidity']
    if 'pressure' in df.columns and 'temp' in df.columns:
        df['pressure_temp_ratio'] = df['pressure'] / (df['temp'] + 273.15)
    return df
def create_rate_of_change_features(df, columns):
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[f'{col}_diff_1'] = df[col].diff(1)
            df[f'{col}_diff_2'] = df[col].diff(2)
            df[f'{col}_pct_change'] = df[col].pct_change()
    return df
def create_domain_specific_features(df):
    df = df.copy()
    if 'aqi' in df.columns:
        df['aqi_category'] = pd.cut(df['aqi'],
                                    bins=[0, 50, 100, 150, 200, 300, 500],
                                    labels=[0, 1, 2, 3, 4, 5])
        df['aqi_category'] = df['aqi_category'].astype(float)
    if 'pm25' in df.columns:
        df['visibility_index'] = np.exp(-0.01 * df['pm25'])
    if all(col in df.columns for col in ['pm25', 'pm10', 'no2', 'so2']):
        df['respiratory_risk'] = (
            0.5 * df['pm25'] +
            0.3 * df['pm10'] +
            0.1 * df['no2'] +
            0.1 * df['so2']
        )
    if 'o3' in df.columns:
        df['ozone_warning'] = (df['o3'] > 100).astype(int)
    if 'temp' in df.columns and 'pm25' in df.columns:
        df['winter_smog_risk'] = ((df['temp'] < 10) & (df['pm25'] > 50)).astype(int)
    return df
def engineer_features(df, verbose=True):
    if verbose:
        print("=" * 80)
        print("FEATURE ENGINEERING")
        print("=" * 80)
    original_features = len(df.columns)
    df = df.copy()
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('datetime').reset_index(drop=True)
    pollutant_cols = ['pm25', 'pm10', 'co', 'no2', 'o3', 'so2']
    weather_cols = ['temp', 'humidity', 'pressure', 'wind_speed']
    target_col = 'aqi'
    available_pollutants = [c for c in pollutant_cols if c in df.columns]
    available_weather = [c for c in weather_cols if c in df.columns]
    if verbose:
        print(f"\n Step 1: Creating Lag Features")
    if target_col in df.columns:
        df = create_lag_features(df, [target_col], lags=[1, 2, 3, 6, 12, 24])
    df = create_lag_features(df, available_pollutants, lags=[1, 3, 6, 12])
    if verbose:
        lag_features = [c for c in df.columns if '_lag_' in c]
        print(f"   Created {len(lag_features)} lag features")
    if verbose:
        print(f"\n Step 2: Creating Rolling Statistics")
    if target_col in df.columns:
        df = create_rolling_features(df, [target_col], windows=[3, 6, 12, 24])
    df = create_rolling_features(df, available_pollutants, windows=[3, 12, 24])
    if verbose:
        rolling_features = [c for c in df.columns if 'rolling_' in c]
        print(f"   Created {len(rolling_features)} rolling statistic features")
    if verbose:
        print(f"\n Step 3: Creating Time-Based Features")
    before_time = len(df.columns)
    df = create_time_features(df)
    if verbose:
        time_features = len(df.columns) - before_time
        print(f"   Created {time_features} time-based features")
    if verbose:
        print(f"\n Step 4: Creating Interaction Features")
    before_interaction = len(df.columns)
    df = create_interaction_features(df)
    if verbose:
        interaction_features = len(df.columns) - before_interaction
        print(f"   Created {interaction_features} interaction features")
    if verbose:
        print(f"\n Step 5: Creating Rate of Change Features")
    if target_col in df.columns:
        df = create_rate_of_change_features(df, [target_col])
    df = create_rate_of_change_features(df, available_pollutants)
    if verbose:
        roc_features = [c for c in df.columns if '_diff_' in c or '_pct_change' in c]
        print(f"   Created {len(roc_features)} rate of change features")
    if verbose:
        print(f"\n Step 6: Creating Domain-Specific Features")
    before_domain = len(df.columns)
    df = create_domain_specific_features(df)
    if verbose:
        domain_features = len(df.columns) - before_domain
        print(f"   Created {domain_features} domain-specific features")
    if verbose:
        print(f"\n Step 7: Cleaning and Validation")
    initial_rows = len(df)
    df = df.dropna()
    if verbose:
        print(f"   Removed {initial_rows - len(df)} rows with NaN (lag initialization)")
        print(f"   Final rows: {len(df):,}")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    final_features = len(df.columns)
    if verbose:
        print(f"\n" + "=" * 80)
        print(f"FEATURE ENGINEERING SUMMARY")
        print(f"=" * 80)
        print(f"Original features:    {original_features}")
        print(f"Engineered features:  {final_features}")
        print(f"Features added:       {final_features - original_features}")
        print(f"Feature growth:       {(final_features/original_features - 1)*100:.1f}%")
        print(f"Final dataset size:   {len(df):,} rows")
        print("=" * 80)
    feature_groups = {
        'lag': [c for c in df.columns if '_lag_' in c],
        'rolling': [c for c in df.columns if 'rolling_' in c],
        'time': [c for c in df.columns if any(x in c for x in ['hour', 'day', 'month', 'weekend', 'rush', 'night'])],
        'interaction': [c for c in df.columns if any(x in c for x in ['ratio', 'product', 'total', 'sum', 'mean'])],
        'rate_of_change': [c for c in df.columns if '_diff_' in c or 'pct_change' in c],
        'domain': [c for c in df.columns if any(x in c for x in ['category', 'visibility', 'respiratory', 'ozone_warning', 'smog'])]
    }
    return df, feature_groups
if __name__ == "__main__":
    import hopsworks
    print("\nConnecting to Hopsworks...")
    project = hopsworks.login(
        api_key_value="DOXxlrr308Rq2xqN.QmlA3Cfoy8ljM9h8nOiYYpxHA3EoSPGhp9qPBcONsXHRL7XIpsGjbcc80R3OoCz5",
        project="AQI_Project_10"
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)
    df_raw = fg.read()
    print(f"\nRaw data: {len(df_raw):,} rows, {len(df_raw.columns)} columns")
    df_engineered, feature_groups = engineer_features(df_raw, verbose=True)
    print(f"\n Feature Groups:")
    for group, features in feature_groups.items():
        print(f"   {group:20s}: {len(features):3d} features")
    print(f"\n Feature engineering complete!")
    print(f"   Ready for model training with {len(df_engineered.columns)} features")
