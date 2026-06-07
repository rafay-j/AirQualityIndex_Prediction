import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import RobustScaler
import hopsworks
def clean_data(df, verbose=True):
    if verbose:
        print("=" * 80)
        print("ADVANCED DATA CLEANSING PIPELINE")
        print("=" * 80)
    original_len = len(df)
    df = df.copy()
    if verbose:
        print(f"\n STEP 1: Duplicate Removal")
        print(f"   Original rows: {len(df):,}")
    duplicates = df.duplicated().sum()
    df = df.drop_duplicates()
    if verbose:
        print(f"   Removed {duplicates:,} duplicate rows")
        print(f"   Remaining: {len(df):,}")
    if verbose:
        print(f"\n STEP 2: AQI Validation")
    before = len(df)
    df = df[
        (df['aqi'].notna()) &
        (df['aqi'] > 0) &
        (df['aqi'] < 500)
    ].copy()
    if verbose:
        print(f"   Removed {before - len(df):,} invalid AQI rows")
        print(f"   Valid AQI range: [{df['aqi'].min():.0f}, {df['aqi'].max():.0f}]")
    if verbose:
        print(f"\n  STEP 3: Pollutant Data Cleaning")
    pollutant_cols = ['pm25', 'pm10', 'co', 'no2', 'o3', 'so2']
    for col in pollutant_cols:
        if col not in df.columns:
            continue
        before = len(df)
        df.loc[df[col] < 0, col] = np.nan
        if col in ['pm25', 'pm10']:
            df = df[df[col].notna() & (df[col] >= 0)].copy()
            if verbose and len(df) < before:
                print(f"   {col.upper()}: Removed {before - len(df):,} missing rows")
        else:
            missing = df[col].isna().sum()
            if missing > 0:
                df[col].fillna(df[col].median(), inplace=True)
                if verbose:
                    print(f"   {col.upper()}: Filled {missing:,} missing values with median")
    if verbose:
        print(f"\n STEP 4: Outlier Removal (IQR + Z-Score Combined)")
    before_outliers = len(df)
    def remove_outliers_iqr(df, column, multiplier=1.5):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    def remove_outliers_zscore(df, column, threshold=3):
        z_scores = np.abs(stats.zscore(df[column]))
        return df[z_scores < threshold]
    df = remove_outliers_iqr(df, 'aqi', multiplier=1.5)
    for col in ['pm25', 'pm10']:
        if col in df.columns:
            df = remove_outliers_iqr(df, col, multiplier=1.5)
            if len(df) > 30:
                df = remove_outliers_zscore(df, col, threshold=3)
    outliers_removed = before_outliers - len(df)
    if verbose:
        print(f"   Removed {outliers_removed:,} outlier rows")
        print(f"   Outlier rate: {outliers_removed/before_outliers*100:.2f}%")
    if verbose:
        print(f"\n STEP 5: Domain-Based Value Capping")
    value_caps = {
        'pm25': (0, 500),
        'pm10': (0, 600),
        'co': (0, 50000),
        'no2': (0, 400),
        'o3': (0, 400),
        'so2': (0, 300),
        'temp': (-50, 60),
        'humidity': (0, 100),
        'pressure': (800, 1200),
        'wind_speed': (0, 150)
    }
    capped_count = 0
    for col, (min_val, max_val) in value_caps.items():
        if col in df.columns:
            before_cap = len(df)
            df = df[(df[col] >= min_val) & (df[col] <= max_val)]
            capped = before_cap - len(df)
            capped_count += capped
            if verbose and capped > 0:
                print(f"   {col.upper():12s}: Removed {capped:4d} rows outside [{min_val}, {max_val}]")
    if verbose:
        print(f"   Total capped: {capped_count:,} rows")
    if verbose:
        print(f"\n STEP 6: Temporal Consistency")
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('datetime')
        now = pd.Timestamp.now()
        future_rows = (df['datetime'] > now).sum()
        df = df[df['datetime'] <= now]
        if verbose:
            print(f"   Removed {future_rows} future timestamps")
            print(f"   Time range: {df['datetime'].min()} to {df['datetime'].max()}")
            print(f"   Span: {(df['datetime'].max() - df['datetime'].min()).days} days")
    if verbose:
        print(f"\n STEP 7: Feature Correlation Validation")
    if 'pm25' in df.columns and 'aqi' in df.columns:
        correlation = df['pm25'].corr(df['aqi'])
        if verbose:
            print(f"   PM2.5 vs AQI correlation: {correlation:.3f}")
        df['pm25_aqi_ratio'] = df['pm25'] / (df['aqi'] + 1)
        ratio_Q1 = df['pm25_aqi_ratio'].quantile(0.05)
        ratio_Q3 = df['pm25_aqi_ratio'].quantile(0.95)
        before_ratio = len(df)
        df = df[(df['pm25_aqi_ratio'] >= ratio_Q1) & (df['pm25_aqi_ratio'] <= ratio_Q3)]
        df = df.drop('pm25_aqi_ratio', axis=1)
        if verbose:
            print(f"   Removed {before_ratio - len(df):,} rows with inconsistent PM2.5/AQI ratio")
    if verbose:
        print(f"\n STEP 8: Low Variance Feature Check")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    low_var_cols = []
    for col in numeric_cols:
        if col not in ['timestamp', 'timestamp_unix']:
            variance = df[col].var()
            if variance < 0.01:
                low_var_cols.append(col)
    if verbose:
        if low_var_cols:
            print(f"   Low variance features: {low_var_cols}")
            print(f"   (These will be excluded from training)")
        else:
            print(f"    All features have sufficient variance")
    if verbose:
        print(f"\n STEP 9: Data Smoothing (Rolling Median)")
    smooth_cols = ['pm25', 'pm10', 'co', 'no2', 'o3', 'so2']
    window = 3
    for col in smooth_cols:
        if col in df.columns:
            if len(df) > window:
                df[f'{col}_raw'] = df[col]
                df[col] = df[col].rolling(window=window, center=True, min_periods=1).median()
    if verbose:
        print(f"   Applied rolling median (window={window}) to pollutant sensors")
    if verbose:
        print(f"\n STEP 10: Final Data Validation")
    critical_cols = ['aqi', 'pm25', 'pm10']
    for col in critical_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                df = df[df[col].notna()]
                if verbose:
                    print(f"   Removed {nan_count} rows with NaN in {col}")
    df = df.reset_index(drop=True)
    if verbose:
        print(f"\n" + "=" * 80)
        print(f"CLEANING SUMMARY")
        print(f"=" * 80)
        print(f"Original rows:     {original_len:,}")
        print(f"Final rows:        {len(df):,}")
        print(f"Rows removed:      {original_len - len(df):,}")
        print(f"Retention rate:    {len(df)/original_len*100:.1f}%")
        print(f"\nFinal AQI Stats:")
        print(f"  Mean:  {df['aqi'].mean():.2f}")
        print(f"  Std:   {df['aqi'].std():.2f}")
        print(f"  Range: [{df['aqi'].min():.0f}, {df['aqi'].max():.0f}]")
        print(f"\nData Quality Score: ", end='')
        quality_score = len(df) / original_len * 100
        if quality_score > 90:
            print(f"EXCELLENT ({quality_score:.1f}%)")
        elif quality_score > 80:
            print(f"GOOD ({quality_score:.1f}%)")
        elif quality_score > 70:
            print(f"FAIR ({quality_score:.1f}%)")
        else:
            print(f"POOR ({quality_score:.1f}%) - Consider reviewing data source")
        print("=" * 80)
    return df
if __name__ == "__main__":
    print("\nConnecting to Hopsworks...")
    project = hopsworks.login(
        api_key_value="DOXxlrr308Rq2xqN.QmlA3Cfoy8ljM9h8nOiYYpxHA3EoSPGhp9qPBcONsXHRL7XIpsGjbcc80R3OoCz5",
        project="AQI_Project_10"
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)
    df_raw = fg.read()
    print(f"\nRaw data loaded: {len(df_raw):,} rows\n")
    df_clean = clean_data(df_raw, verbose=True)
    print(f"\n" + "=" * 80)
    print("BEFORE vs AFTER COMPARISON")
    print("=" * 80)
    print(f"\n{'Metric':<20s} {'Before':<15s} {'After':<15s} {'Change':<15s}")
    print("-" * 65)
    metrics = [
        ('Rows', len(df_raw), len(df_clean)),
        ('AQI Mean', df_raw[df_raw['aqi'] > 0]['aqi'].mean(), df_clean['aqi'].mean()),
        ('AQI Std', df_raw[df_raw['aqi'] > 0]['aqi'].std(), df_clean['aqi'].std()),
        ('PM2.5 Mean', df_raw[df_raw['pm25'] > 0]['pm25'].mean(), df_clean['pm25'].mean()),
        ('PM10 Mean', df_raw[df_raw['pm10'] > 0]['pm10'].mean(), df_clean['pm10'].mean()),
    ]
    for metric, before, after in metrics:
        change = ((after - before) / before * 100) if before != 0 else 0
        print(f"{metric:<20s} {before:<15.2f} {after:<15.2f} {change:+.1f}%")
    print("=" * 80)
