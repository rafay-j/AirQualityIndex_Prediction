import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from feature_store import store_features
import os
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "DOXxlrr308Rq2xqN.QmlA3Cfoy8ljM9h8nOiYYpxHA3EoSPGhp9qPBcONsXHRL7XIpsGjbcc80R3OoCz5")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "AQI_Project_10")
def load_historical_csv(filepath='historical_data.csv'):
    """Load historical CSV data"""
    print(f"Loading historical data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f" Loaded {len(df)} rows")
    return df
def transform_to_hopsworks_schema(df):
    """
    Transform historical CSV to match Hopsworks Feature Group schema
    CSV columns: co, no, no2, o3, so2, pm2_5, pm10, nh3, temp, rhum, wspd, pres, aqi
    Missing columns to generate:
    - timestamp, timestamp_unix, datetime
    - station_name, station_url, latitude, longitude
    - dew_point (calculate from temp & humidity)
    - All *_imputed flags
    - Temporal features (hour, day, month, year, etc.)
    - Engineered features
    """
    print("\nTransforming data to Hopsworks schema...")
    transformed = pd.DataFrame()
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=len(df)-1)
    timestamps = pd.date_range(start=start_time, end=end_time, periods=len(df))
    transformed['timestamp'] = timestamps.strftime('%Y-%m-%d %H:%M:%S')
    transformed['timestamp_unix'] = timestamps.astype(np.int64) // 10**9
    transformed['datetime'] = timestamps
    transformed['station_name'] = 'Islamabad, Pakistan'
    transformed['station_url'] = ''
    transformed['latitude'] = 33.6844
    transformed['longitude'] = 73.0479
    transformed['aqi'] = df['aqi'].fillna(-1).round().astype('int64')
    transformed['pm25'] = df['pm2_5'].fillna(-1).round().astype('int64')
    transformed['pm10'] = df['pm10'].fillna(-1).round().astype('int64')
    transformed['o3'] = df['o3'].fillna(-1).round().astype('int64')
    transformed['no2'] = df['no2'].fillna(-1).round().astype('int64')
    transformed['so2'] = df['so2'].fillna(-1).round().astype('int64')
    transformed['co'] = df['co'].fillna(-1).round().astype('int64')
    transformed['temperature'] = df['temp'].fillna(-1)
    transformed['humidity'] = df['rhum'].fillna(-1)
    transformed['pressure'] = df['pres'].fillna(-1)
    transformed['wind_speed'] = df['wspd'].fillna(-1)
    def calc_dew_point(row):
        if row['temp'] > 0 and row['rhum'] > 0:
            return row['temp'] - ((100 - row['rhum']) / 5)
        return -1
    transformed['dew_point'] = df.apply(calc_dew_point, axis=1)
    imputed_cols = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co',
                    'temperature', 'humidity', 'pressure', 'wind_speed', 'dew_point']
    for col in imputed_cols:
        transformed[f'{col}_imputed'] = 0
    transformed['total_imputed_features'] = 0
    transformed['data_quality'] = 'valid'
    transformed['hour'] = timestamps.hour
    transformed['day_of_week'] = timestamps.dayofweek
    transformed['day'] = timestamps.day
    transformed['month'] = timestamps.month
    transformed['year'] = timestamps.year
    transformed['is_weekend'] = (timestamps.dayofweek >= 5).astype(int)
    time_of_day = pd.cut(timestamps.hour, bins=[0, 6, 12, 18, 24],
                         labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                         include_lowest=True)
    transformed['time_of_day'] = time_of_day.astype(str)
    transformed['time_of_day_numeric'] = pd.cut(timestamps.hour, bins=[0, 6, 12, 18, 24],
                                                  labels=[0, 1, 2, 3],
                                                  include_lowest=True).astype(int)
    def categorize_aqi(aqi):
        if pd.isna(aqi) or aqi == -1:
            return 0
        if aqi <= 50:
            return 1
        if aqi <= 100:
            return 2
        if aqi <= 150:
            return 3
        if aqi <= 200:
            return 4
        if aqi <= 300:
            return 5
        return 6
    transformed['aqi_category_numeric'] = df['aqi'].apply(categorize_aqi)
    transformed['temp_humidity_interaction'] = transformed['temperature'] * transformed['humidity']
    transformed['discomfort_index'] = (transformed['temperature'] -
                                       (0.55 - 0.0055 * transformed['humidity']) *
                                       (transformed['temperature'] - 14.5))
    transformed['wind_pm25_interaction'] = transformed['wind_speed'] * transformed['pm25']
    transformed['pressure_temp_ratio'] = transformed['pressure'] / (transformed['temperature'] + 273.15)
    int32_cols = ['pm25_imputed', 'pm10_imputed', 'o3_imputed', 'no2_imputed',
                  'so2_imputed', 'co_imputed', 'temperature_imputed', 'humidity_imputed',
                  'pressure_imputed', 'wind_speed_imputed', 'dew_point_imputed',
                  'is_weekend', 'time_of_day_numeric', 'hour', 'day_of_week', 'day', 'month', 'year']
    for col in int32_cols:
        if col in transformed.columns:
            transformed[col] = transformed[col].astype('int32')
    int64_cols = ['total_imputed_features', 'aqi_category_numeric', 'timestamp_unix']
    for col in int64_cols:
        if col in transformed.columns:
            transformed[col] = transformed[col].astype('int64')
    print(f" Transformed to {len(transformed)} rows with {len(transformed.columns)} columns")
    return transformed
def upload_in_batches(df, batch_size=1000):
    """Upload data in batches to avoid timeouts"""
    print(f"\nUploading {len(df)} rows in batches of {batch_size}...")
    total_batches = (len(df) + batch_size - 1) // batch_size
    success_count = 0
    failed_batches = []
    for i in range(0, len(df), batch_size):
        batch_num = (i // batch_size) + 1
        batch = df.iloc[i:i+batch_size].copy()
        print(f"\nBatch {batch_num}/{total_batches} ({len(batch)} rows)...")
        try:
            success = store_features(
                df=batch,
                api_key=HOPSWORKS_API_KEY,
                project_name=HOPSWORKS_PROJECT,
                feature_group_name="aqi_features",
                version=1
            )
            if success:
                success_count += len(batch)
                print(f" Batch {batch_num} uploaded successfully")
            else:
                failed_batches.append(batch_num)
                print(f" Batch {batch_num} failed")
                batch.to_csv(f'failed_batch_{batch_num}.csv', index=False)
        except Exception as e:
            print(f" Batch {batch_num} error: {e}")
            failed_batches.append(batch_num)
            batch.to_csv(f'failed_batch_{batch_num}.csv', index=False)
    return success_count, failed_batches
def main():
    print("="*70)
    print("HISTORICAL DATA UPLOAD TO HOPSWORKS")
    print("="*70)
    df = load_historical_csv('historical_data.csv')
    transformed_df = transform_to_hopsworks_schema(df)
    print("\n" + "="*70)
    print("SAMPLE OF TRANSFORMED DATA:")
    print("="*70)
    print(transformed_df.head(3)[['timestamp', 'aqi', 'pm25', 'temperature', 'humidity', 'aqi_category_numeric']])
    print("\n" + "="*70)
    print(f"Ready to upload {len(transformed_df)} rows to Hopsworks")
    print("="*70)
    response = input("\nProceed with upload? (yes/no): ").strip().lower()
    if response == 'yes':
        success_count, failed_batches = upload_in_batches(transformed_df, batch_size=500)
        print("\n" + "="*70)
        print("UPLOAD COMPLETE")
        print("="*70)
        print(f" Successfully uploaded: {success_count}/{len(transformed_df)} rows")
        if failed_batches:
            print(f" Failed batches: {failed_batches}")
            print("  Failed batches saved as failed_batch_*.csv")
        else:
            print(" All batches uploaded successfully!")
        print("\nRun 'python check_data_status.py' to verify the data")
    else:
        print("\nUpload cancelled.")
if __name__ == "__main__":
    main()
