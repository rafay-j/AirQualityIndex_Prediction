import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
import time
class WAQIDataFetcher:
    def __init__(self, api_token: str, station_id: str):
        self.api_token = api_token
        self.station_id = station_id
        self.base_url = "https://api.waqi.info"
    def fetch_current_data(self) -> Optional[Dict]:
        url = f"{self.base_url}/feed/{self.station_id}/"
        params = {"token": self.api_token}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "ok":
                return data.get("data")
            else:
                print(f"API Error: {data.get('data')}")
                return None
        except Exception as e:
            print(f"Request failed: {e}")
            return None
    def parse_station_data(self, raw_data: Dict) -> Dict:
        if not raw_data:
            return {}
        iaqi = raw_data.get('iaqi', {})
        city_info = raw_data.get('city', {})
        geo = city_info.get('geo', [None, None])
        time_info = raw_data.get('time', {})
        from datetime import datetime
        collection_time = datetime.now()
        parsed = {
            'timestamp': collection_time.strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp_unix': int(collection_time.timestamp()),
            'aqi': raw_data.get('aqi'),
            'station_name': city_info.get('name'),
            'station_url': '',
            'latitude': geo[0] if len(geo) > 0 else None,
            'longitude': geo[1] if len(geo) > 1 else None,
            'pm25': iaqi.get('pm25', {}).get('v'),
            'pm10': iaqi.get('pm10', {}).get('v'),
            'o3': iaqi.get('o3', {}).get('v'),
            'no2': iaqi.get('no2', {}).get('v'),
            'so2': iaqi.get('so2', {}).get('v'),
            'co': iaqi.get('co', {}).get('v'),
            'temperature': iaqi.get('t', {}).get('v'),
            'pressure': iaqi.get('p', {}).get('v'),
            'humidity': iaqi.get('h', {}).get('v'),
            'wind_speed': iaqi.get('w', {}).get('v'),
            'dew_point': iaqi.get('dew', {}).get('v'),
        }
        return parsed
class DataProcessor:
    @staticmethod
    def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'aqi' in df.columns:
            df['data_quality'] = 'valid'
            df['aqi'] = df['aqi'].fillna(-1)
            if df['aqi'].isna().any():
                df['data_quality'] = 'invalid'
        else:
            df['data_quality'] = 'invalid'
        pollutants = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
        for pollutant in pollutants:
            if pollutant in df.columns:
                df[f'{pollutant}_imputed'] = df[pollutant].isna().astype(int)
                if df[pollutant].notna().any():
                    df[pollutant] = df[pollutant].fillna(df[pollutant].median())
                else:
                    df[pollutant] = df[pollutant].fillna(0)
            else:
                df[pollutant] = 0
                df[f'{pollutant}_imputed'] = 1
        weather_defaults = {
            'temperature': 25.0,
            'humidity': 60.0,
            'pressure': 1013.25,
            'wind_speed': 2.0,
            'dew_point': 15.0
        }
        for weather_param, default_val in weather_defaults.items():
            if weather_param in df.columns:
                df[f'{weather_param}_imputed'] = df[weather_param].isna().astype(int)
                df[weather_param] = df[weather_param].fillna(default_val)
            else:
                df[weather_param] = default_val
                df[f'{weather_param}_imputed'] = 1
        imputed_cols = [col for col in df.columns if col.endswith('_imputed')]
        df['total_imputed_features'] = df[imputed_cols].sum(axis=1) if imputed_cols else 0
        if 'station_url' not in df.columns:
            df['station_url'] = ''
        return df
    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['day'] = df['datetime'].dt.day
            df['month'] = df['datetime'].dt.month
            df['year'] = df['datetime'].dt.year
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['time_of_day'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'], include_lowest=True).astype(str)
            df['time_of_day_numeric'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], labels=[0, 1, 2, 3], include_lowest=True).astype(int)
        else:
            df['datetime'] = pd.Timestamp.now()
            df['hour'] = 0
            df['day_of_week'] = 0
            df['day'] = 0
            df['month'] = 0
            df['year'] = 0
            df['is_weekend'] = 0
            df['time_of_day'] = 'Night'
            df['time_of_day_numeric'] = 0
        if 'aqi' in df.columns:
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
            df['aqi_category_numeric'] = df['aqi'].apply(categorize_aqi)
        else:
            df['aqi_category_numeric'] = 0
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
            df['discomfort_index'] = (df['temperature'] - (0.55 - 0.0055 * df['humidity']) * (df['temperature'] - 14.5))
        else:
            df['temp_humidity_interaction'] = 0
            df['discomfort_index'] = 0
        if 'wind_speed' in df.columns and 'pm25' in df.columns:
            df['wind_pm25_interaction'] = df['wind_speed'] * df['pm25']
        else:
            df['wind_pm25_interaction'] = 0
        if 'pressure' in df.columns and 'temperature' in df.columns:
            df['pressure_temp_ratio'] = df['pressure'] / (df['temperature'] + 273.15)
        else:
            df['pressure_temp_ratio'] = 0
        int32_columns = [
            'pm25_imputed', 'pm10_imputed', 'o3_imputed', 'no2_imputed',
            'so2_imputed', 'co_imputed', 'temperature_imputed', 'humidity_imputed',
            'pressure_imputed', 'wind_speed_imputed', 'dew_point_imputed',
            'is_weekend', 'time_of_day_numeric',
            'hour', 'day_of_week', 'day', 'month', 'year'
        ]
        int64_columns = [
            'total_imputed_features',
            'aqi_category_numeric'
        ]
        for col in int32_columns:
            if col in df.columns:
                df[col] = df[col].astype('int32')
        for col in int64_columns:
            if col in df.columns:
                df[col] = df[col].astype('int64')
        return df
class HistoricalDataCollector:
    def __init__(self, fetcher: WAQIDataFetcher, processor: DataProcessor):
        self.fetcher = fetcher
        self.processor = processor
    def collect_backfill_data(self, days: int = 90, interval_hours: int = 6) -> Optional[pd.DataFrame]:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        total_hours = int((end_date - start_date).total_seconds() / 3600)
        total_samples = (total_hours + interval_hours - 1) // interval_hours
        print(f"Collecting {days} days of historical data ({total_samples} samples at {interval_hours}h intervals)...")
        all_data = []
        collected = 0
        failed = 0
        for hour_offset in range(0, total_hours, interval_hours):
            try:
                raw_data = self.fetcher.fetch_current_data()
                if raw_data:
                    parsed_data = self.fetcher.parse_station_data(raw_data)
                    df_raw = pd.DataFrame([parsed_data])
                    df_cleaned = self.processor.handle_nulls(df_raw)
                    df_features = self.processor.create_features(df_cleaned)
                    all_data.append(df_features)
                    collected += 1
                else:
                    failed += 1
                time.sleep(12)
            except Exception:
                failed += 1
                if failed % 10 == 0:
                    print(f"Failed requests: {failed}")
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f" Collected {collected} samples, {failed} failed")
            return combined_df
        else:
            print(" No data collected")
            return None
    def collect_single_sample(self) -> Optional[pd.DataFrame]:
        try:
            raw_data = self.fetcher.fetch_current_data()
            if raw_data:
                parsed_data = self.fetcher.parse_station_data(raw_data)
                df_raw = pd.DataFrame([parsed_data])
                df_cleaned = self.processor.handle_nulls(df_raw)
                df_features = self.processor.create_features(df_cleaned)
                return df_features
            return None
        except Exception as e:
            print(f"Error collecting sample: {e}")
            import traceback
            traceback.print_exc()
            return None
def fetch_historical_data(api_token: str, station_id: str, days: int = 90, interval_hours: int = 6) -> Optional[pd.DataFrame]:
    fetcher = WAQIDataFetcher(api_token, station_id)
    processor = DataProcessor()
    collector = HistoricalDataCollector(fetcher, processor)
    return collector.collect_backfill_data(days=days, interval_hours=interval_hours)
def fetch_daily_data(api_token: str, station_id: str) -> Optional[pd.DataFrame]:
    fetcher = WAQIDataFetcher(api_token, station_id)
    processor = DataProcessor()
    collector = HistoricalDataCollector(fetcher, processor)
    return collector.collect_single_sample()
