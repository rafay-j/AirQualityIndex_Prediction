from data_fetcher import fetch_daily_data
from feature_store import store_features
from datetime import datetime
import os
WAQI_API_TOKEN = os.getenv("WAQI_API_TOKEN", "088661c637816f9f1463ca3e44d37da6d739d021")
STATION_ID = os.getenv("STATION_ID", "A401143")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "DOXxlrr308Rq2xqN.QmlA3Cfoy8ljM9h8nOiYYpxHA3EoSPGhp9qPBcONsXHRL7XIpsGjbcc80R3OoCz5")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "AQI_Project_10")
def main():
    print(f"=== AQI Daily Update: {datetime.now().strftime('%Y-%m-%d %H:%M')} ===\n")
    if os.getenv("GITHUB_ACTIONS"):
        print(" Running in GitHub Actions environment")
        print(f"   WAQI_API_TOKEN: {'SET' if os.getenv('WAQI_API_TOKEN') else 'NOT SET (using fallback)'}")
        print(f"   STATION_ID: {'SET' if os.getenv('STATION_ID') else 'NOT SET (using fallback)'}")
        print(f"   HOPSWORKS_API_KEY: {'SET' if os.getenv('HOPSWORKS_API_KEY') else 'NOT SET (using fallback)'}")
        print(f"   HOPSWORKS_PROJECT: {'SET' if os.getenv('HOPSWORKS_PROJECT') else 'NOT SET (using fallback)'}")
    else:
        print(" Running locally")
    print(f"\nUsing Station ID: {STATION_ID}")
    print(f"Using Hopsworks Project: {HOPSWORKS_PROJECT}\n")
    df = fetch_daily_data(WAQI_API_TOKEN, STATION_ID)
    if df is not None:
        print(f" Collected: 1 sample with {len(df.columns)} features")
        if 'timestamp' in df.columns:
            print(f"   Timestamp: {df['timestamp'].iloc[0]}")
        success = store_features(
            df=df,
            api_key=HOPSWORKS_API_KEY,
            project_name=HOPSWORKS_PROJECT,
            feature_group_name="aqi_features",
            version=1
        )
        if success:
            print(" Daily update completed successfully")
        else:
            print(" Daily update failed - data not inserted")
            df.to_csv(f"backup_daily_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", index=False)
            print(f"   Backup saved locally")
            exit(1)
    else:
        print(" No data collected from API")
        exit(1)
if __name__ == "__main__":
    main()
