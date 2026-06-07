from data_fetcher import fetch_historical_data
from feature_store import store_features
import os
WAQI_API_TOKEN = os.getenv("WAQI_API_TOKEN", "088661c637816f9f1463ca3e44d37da6d739d021")
STATION_ID = os.getenv("STATION_ID", "A401143")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "DOXxlrr308Rq2xqN.QmlA3Cfoy8ljM9h8nOiYYpxHA3EoSPGhp9qPBcONsXHRL7XIpsGjbcc80R3OoCz5")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "AQI_Project_10")
BACKFILL_DAYS = 30
def main():
    print(" AQI Historical Data Backfill \n")
    df = fetch_historical_data(WAQI_API_TOKEN, STATION_ID, days=BACKFILL_DAYS, interval_hours=12)
    if df is not None:
        print(f"\nDataset: {len(df)} records, {len(df.columns)} features")
        success = store_features(
            df=df,
            api_key=HOPSWORKS_API_KEY,
            project_name=HOPSWORKS_PROJECT,
            feature_group_name="aqi_features",
            version=1
        )
        if success:
            print("\n Backfill completed successfully")
        else:
            print("\n Backfill failed")
            df.to_csv("backup_aqi_features.csv", index=False)
            print("Data saved locally to: backup_aqi_features.csv")
    else:
        print("\n No data collected")
if __name__ == "__main__":
    main()
