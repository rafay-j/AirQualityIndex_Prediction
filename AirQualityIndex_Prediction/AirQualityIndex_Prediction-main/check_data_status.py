import pandas as pd
from feature_store import read_features
from config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    COLLECTION_INTERVAL_HOURS,
    PREDICTION_HORIZON_DAYS,
    BACKFILL_DAYS
)
def analyze_data():
    print("\n" + "="*60)
    print("AQI Data Status Check")
    print("="*60 + "\n")
    print("Current Configuration:")
    print(f"  • Collection Interval: {COLLECTION_INTERVAL_HOURS} hour(s)")
    print(f"  • Prediction Horizon: {PREDICTION_HORIZON_DAYS} day(s)")
    print(f"  • Backfill Days: {BACKFILL_DAYS} day(s)")
    print()
    print("Fetching data from Hopsworks...")
    try:
        df = read_features(
            api_key=HOPSWORKS_API_KEY,
            project_name=HOPSWORKS_PROJECT,
            feature_group_name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION
        )
    except Exception as e:
        print(f" Error fetching data: {e}")
        return
    if df is None or len(df) == 0:
        print(" No data found in Feature Store!")
        print("\nRecommendation: Run backfill_pipeline.py to populate data")
        return
    print(f" Found {len(df)} total rows\n")
    valid_df = df[df['aqi'].notna() & (df['aqi'] != -1)]
    print(f"Data Quality:")
    print(f"  • Valid AQI records: {len(valid_df)}/{len(df)} ({len(valid_df)/len(df)*100:.1f}%)")
    if 'timestamp' in df.columns or 'datetime' in df.columns:
        time_col = 'timestamp' if 'timestamp' in df.columns else 'datetime'
        df[time_col] = pd.to_datetime(df[time_col])
        print(f"  • Time range: {df[time_col].min()} to {df[time_col].max()}")
        time_span = (df[time_col].max() - df[time_col].min()).total_seconds() / 86400
        print(f"  • Time span: {time_span:.1f} days")
    steps_per_day = max(1, int(24 / COLLECTION_INTERVAL_HOURS))
    max_shift = PREDICTION_HORIZON_DAYS * steps_per_day
    rows_after_shift = max(0, len(valid_df) - max_shift)
    print(f"\nPrediction Configuration Analysis:")
    print(f"  • Steps per day: {steps_per_day}")
    print(f"  • Shift for {PREDICTION_HORIZON_DAYS} day(s) ahead: {max_shift} rows")
    print(f"  • Rows available for training: {rows_after_shift}")
    print("\n" + "-"*60)
    if rows_after_shift >= 50:
        print(" Status: EXCELLENT - Plenty of data for training")
        print(f"  You can train with current settings (horizon={PREDICTION_HORIZON_DAYS} days)")
    elif rows_after_shift >= 20:
        print(" Status: GOOD - Sufficient data for training")
        print(f"  Current horizon ({PREDICTION_HORIZON_DAYS} days) should work")
        print("  Consider collecting more data for better model performance")
    elif rows_after_shift >= 10:
        print(" Status: LIMITED - Minimal data available")
        print(f"  Training possible but may need to reduce horizon")
        print(f"  Consider reducing PREDICTION_HORIZON_DAYS to 1")
    else:
        print(" Status: INSUFFICIENT - Not enough data for current settings")
        print("\nRecommendations:")
        for h in range(PREDICTION_HORIZON_DAYS, 0, -1):
            test_shift = h * steps_per_day
            test_rows = len(valid_df) - test_shift
            if test_rows >= 20:
                print(f"  1. Reduce PREDICTION_HORIZON_DAYS to {h} day(s)")
                print(f"     → This would give you {test_rows} training rows")
                break
        else:
            single_step_rows = len(valid_df) - 1
            if single_step_rows >= 10:
                print(f"  1. Use single-step prediction (next hour only)")
                print(f"     → This would give you {single_step_rows} training rows")
            else:
                print("  1. Collect more data - you need at least 20-30 rows")
        print(f"  2. Increase BACKFILL_DAYS in config.py (currently {BACKFILL_DAYS})")
        print("  3. Run: python backfill_pipeline.py")
        print("  4. Wait for more data to accumulate via daily_pipeline.py")
    print("\n" + "="*60)
    numeric_cols = df.select_dtypes(include=['number']).columns
    print(f"\nAvailable Features: {len(numeric_cols)} numeric columns")
    missing_pct = (df.isnull().sum() / len(df) * 100)
    if missing_pct.any():
        print("\nColumns with missing data:")
        for col, pct in missing_pct[missing_pct > 0].items():
            print(f"  • {col}: {pct:.1f}%")
    print("\n" + "="*60 + "\n")
if __name__ == "__main__":
    analyze_data()
