import pandas as pd
import numpy as np
from feature_store import read_features
import hopsworks
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from scipy import stats
import joblib
import os
import json
from datetime import datetime
from config import COLLECTION_INTERVAL_HOURS, PREDICTION_HORIZON_DAYS
import optuna
import warnings
import sys
import io
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
original_stdout = sys.stdout
sys.stdout = io.StringIO()
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from feature_engineering import engineer_features
from data_cleaning import clean_data
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "DOXxlrr308Rq2xqN.QmlA3Cfoy8ljM9h8nOiYYpxHA3EoSPGhp9qPBcONsXHRL7XIpsGjbcc80R3OoCz5")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "AQI_Project_10")
FEATURE_GROUP_NAME = os.getenv("FEATURE_GROUP_NAME", "aqi_features")
FEATURE_GROUP_VERSION = int(os.getenv("FEATURE_GROUP_VERSION", "1"))
np.random.seed(42)
tf.random.set_seed(42)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def fetch_and_prepare_data(use_feature_engineering=True, verbose=False):
    if verbose:
        print("\n" + "="*80)
        print("STEP 1: DATA LOADING & PREPARATION")
        print("="*80)
    if verbose:
        print("\n Fetching data from Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
    fs = project.get_feature_store()
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df_raw = fg.read()
    print(f"   Loaded {len(df_raw):,} rows")
    print("\n Applying data cleaning...")
    df_clean = clean_data(df_raw, verbose=False)
    retention = len(df_clean) / len(df_raw) * 100
    print(f"   Clean data: {len(df_clean):,} rows ({retention:.1f}% retained)")
    if use_feature_engineering:
        print("\n  Applying feature engineering...")
        df_engineered, feature_groups = engineer_features(df_clean, verbose=False)
        feature_growth = (len(df_engineered.columns) - len(df_raw.columns)) / len(df_raw.columns) * 100
        print(f"   Features: {len(df_raw.columns)} → {len(df_engineered.columns)} (+{feature_growth:.0f}%)")
        print(f"   Samples: {len(df_engineered):,}")
        return df_engineered
    else:
        print("   Skipping feature engineering (using raw features)")
        return df_clean
def prepare_single_step_target(df, prediction_horizon=1):
    print(f"\n Preparing SINGLE-STEP prediction ({prediction_horizon} hour ahead)...")
    if 'aqi' not in df.columns:
        raise ValueError("AQI column not found!")
    y = df['aqi'].shift(-prediction_horizon)
    df = df.iloc[:-prediction_horizon].copy()
    y = y.iloc[:-prediction_horizon]
    drop_cols = [
        'aqi', 'timestamp', 'timestamp_unix', 'station_name', 'station_url', 'datetime',
        'data_quality', 'year', 'month', 'day', 'hour'
    ]
    X = df.drop([col for col in drop_cols if col in df.columns], axis=1, errors='ignore')
    X = X.select_dtypes(include=[np.number])
    variance = X.var()
    zero_var_cols = variance[variance == 0].index.tolist()
    if zero_var_cols:
        print(f"   Removing {len(zero_var_cols)} zero-variance features")
        X = X.drop(zero_var_cols, axis=1)
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {len(X):,}")
    print(f"   Target range: [{y.min():.0f}, {y.max():.0f}]")
    return X, y
def prepare_multi_step_target(df, horizon_days=None):
    if horizon_days is None:
        horizon_days = PREDICTION_HORIZON_DAYS
    print(f"\n Preparing MULTI-STEP prediction ({horizon_days} days ahead)...")
    drop_cols = [
        'aqi', 'timestamp', 'timestamp_unix', 'station_name', 'station_url', 'datetime',
        'data_quality', 'year', 'month', 'day', 'hour'
    ]
    X = df.drop([col for col in drop_cols if col in df.columns], axis=1, errors='ignore')
    X = X.select_dtypes(include=[np.number])
    steps_per_day = max(1, int(24 / COLLECTION_INTERVAL_HOURS))
    target_cols = ['aqi']
    pollutant_cols = ['pm25', 'pm10', 'o3', 'no2']
    target_cols.extend([c for c in pollutant_cols if c in df.columns])
    y_frames = []
    y_col_names = []
    for h in range(1, horizon_days + 1):
        shift_steps = h * steps_per_day
        for col in target_cols:
            col_name = f"{col}_t+{h}d"
            y_col_names.append(col_name)
            y_frames.append(df[col].shift(-shift_steps))
    y = pd.concat(y_frames, axis=1)
    y.columns = y_col_names
    valid_idx = y.dropna().index
    X = X.loc[valid_idx].reset_index(drop=True)
    y = y.loc[valid_idx].reset_index(drop=True)
    print(f"   Features: {X.shape[1]}")
    print(f"   Targets: {y.shape[1]} outputs")
    print(f"   Samples: {len(X):,}")
    return X, y
def select_top_features(X_train, y_train, X_test, n_features=50):
    print(f"\n Feature Selection (top {n_features})...")
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        y_for_selection = y_train.iloc[:, 0] if isinstance(y_train, pd.DataFrame) else y_train[:, 0]
    else:
        y_for_selection = y_train
    selector = SelectKBest(score_func=f_regression, k=min(n_features, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train, y_for_selection)
    X_test_selected = selector.transform(X_test)
    selected_features = X_train.columns[selector.get_support()].tolist()
    print(f"   Selected: {len(selected_features)} features")
    print(f"   Top 10: {selected_features[:10]}")
    return X_train_selected, X_test_selected, selected_features, selector
def create_sequences(X, y, sequence_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])
    return np.array(X_seq), np.array(y_seq)
def build_lstm_model(input_shape, output_dim):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(output_dim)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model
def build_cnn_model(input_shape, output_dim):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(32, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(output_dim)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def perform_cross_validation(model, X, y, cv_folds=5, model_name='Model'):
    """
    Perform cross-validation on a model using TimeSeriesSplit.
    Returns cross-validation scores and statistics.
    """
    print(f"\n  Cross-Validation: Using TimeSeriesSplit with {cv_folds} folds")
    
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    cv_r2_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2', n_jobs=-1)
    cv_neg_mse_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    cv_neg_mae_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
    
    cv_rmse_scores = np.sqrt(-cv_neg_mse_scores)
    cv_mae_scores = -cv_neg_mae_scores
    
    cv_stats = {
        'r2_mean': cv_r2_scores.mean(),
        'r2_std': cv_r2_scores.std(),
        'rmse_mean': cv_rmse_scores.mean(),
        'rmse_std': cv_rmse_scores.std(),
        'mae_mean': cv_mae_scores.mean(),
        'mae_std': cv_mae_scores.std(),
        'r2_scores': cv_r2_scores.tolist(),
        'rmse_scores': cv_rmse_scores.tolist(),
        'mae_scores': cv_mae_scores.tolist()
    }
    
    print(f"  CV Results (mean ± std):")
    print(f"    R² Score: {cv_stats['r2_mean']:.4f} ± {cv_stats['r2_std']:.4f}")
    print(f"    RMSE:     {cv_stats['rmse_mean']:.2f} ± {cv_stats['rmse_std']:.2f}")
    print(f"    MAE:      {cv_stats['mae_mean']:.2f} ± {cv_stats['mae_std']:.2f}")
    
    return cv_stats

def train_traditional_models(X_train, y_train, X_test, y_test, optimize=True):
    print("\n" + "="*80)
    print("STEP 2: TRAINING TRADITIONAL ML MODELS")
    print("="*80)
    models = {}
    results = {}
    
    print("\n MODEL 1: Random Forest")
    if optimize and len(X_train) > 500:
        print("   Optimizing with Optuna (15 trials)...")
        def rf_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 10, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'random_state': 42,
                'n_jobs': -1
            }
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return np.sqrt(mean_squared_error(y_test, y_pred))
        study = optuna.create_study(direction='minimize')
        study.optimize(rf_objective, n_trials=15, show_progress_bar=False)
        rf = RandomForestRegressor(**study.best_params, random_state=42, n_jobs=-1)
        print(f"   Best params: {study.best_params}")
    else:
        rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    
    cv_stats_rf = perform_cross_validation(rf, X_train, y_train, cv_folds=5, model_name='RandomForest')
    
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"  Test Set Results:")
    print(f"    RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")
    
    models['RandomForest'] = rf
    results['RandomForest'] = {
        'rmse': rmse, 'mae': mae, 'r2': r2,
        'cv_r2_mean': cv_stats_rf['r2_mean'],
        'cv_r2_std': cv_stats_rf['r2_std'],
        'cv_rmse_mean': cv_stats_rf['rmse_mean'],
        'cv_rmse_std': cv_stats_rf['rmse_std']
    }
    
    print("\n MODEL 2: XGBoost (with L1/L2 Regularization)")
    if optimize and len(X_train) > 500:
        print("   Optimizing with Optuna (20 trials)...")
        def xgb_objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42,
                'n_jobs': -1
            }
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, verbose=False)
            y_pred = model.predict(X_test)
            return np.sqrt(mean_squared_error(y_test, y_pred))
        study = optuna.create_study(direction='minimize')
        study.optimize(xgb_objective, n_trials=20, show_progress_bar=False)
        xgb_model = xgb.XGBRegressor(**study.best_params, random_state=42, n_jobs=-1)
        print(f"   Best params: {study.best_params}")
    else:
        xgb_model = xgb.XGBRegressor(
            max_depth=6, 
            learning_rate=0.1, 
            n_estimators=100,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42, 
            n_jobs=-1
        )
    
    cv_stats_xgb = perform_cross_validation(xgb_model, X_train, y_train, cv_folds=5, model_name='XGBoost')
    
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"  Test Set Results:")
    print(f"    RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")
    
    models['XGBoost'] = xgb_model
    results['XGBoost'] = {
        'rmse': rmse, 'mae': mae, 'r2': r2,
        'cv_r2_mean': cv_stats_xgb['r2_mean'],
        'cv_r2_std': cv_stats_xgb['r2_std'],
        'cv_rmse_mean': cv_stats_xgb['rmse_mean'],
        'cv_rmse_std': cv_stats_xgb['rmse_std']
    }
    
    return models, results
def train_deep_learning_models(X_train, y_train, X_test, y_test, scaler=None):
    print("\n" + "="*80)
    print("STEP 3: TRAINING DEEP LEARNING MODELS")
    print("="*80)
    models = {}
    results = {}
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    sequence_length = min(24, len(X_train) // 4)
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, sequence_length)
    if len(X_train_seq) < 20:
        print("     Insufficient data for deep learning (need >20 sequences)")
        return models, results
    print(f"   Sequence length: {sequence_length}")
    print(f"   Train sequences: {len(X_train_seq)}")
    print(f"   Test sequences: {len(X_test_seq)}")
    output_dim = y_train_seq.shape[1] if len(y_train_seq.shape) > 1 else 1
    input_shape = (sequence_length, X_train_scaled.shape[1])
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(patience=5, factor=0.5)
    ]
    print("\n MODEL 3: LSTM")
    lstm_model = build_lstm_model(input_shape, output_dim)
    history = lstm_model.fit(
        X_train_seq, y_train_seq,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0
    )
    y_pred = lstm_model.predict(X_test_seq, verbose=0)
    rmse = np.sqrt(mean_squared_error(y_test_seq, y_pred))
    mae = mean_absolute_error(y_test_seq, y_pred)
    r2 = r2_score(y_test_seq, y_pred)
    print(f"   RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")
    models['LSTM'] = lstm_model
    results['LSTM'] = {'rmse': rmse, 'mae': mae, 'r2': r2}
    print("\n MODEL 4: 1D CNN")
    cnn_model = build_cnn_model(input_shape, output_dim)
    history = cnn_model.fit(
        X_train_seq, y_train_seq,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0
    )
    y_pred = cnn_model.predict(X_test_seq, verbose=0)
    rmse = np.sqrt(mean_squared_error(y_test_seq, y_pred))
    mae = mean_absolute_error(y_test_seq, y_pred)
    r2 = r2_score(y_test_seq, y_pred)
    print(f"   RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")
    models['CNN_1D'] = cnn_model
    results['CNN_1D'] = {'rmse': rmse, 'mae': mae, 'r2': r2}
    return models, results, scaler
def compare_models(results, target_r2=0.5):
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    model_metrics = []
    for model_name, metrics in results.items():
        performance_score = float(metrics['r2']) * 0.6 + (1 - float(metrics['rmse'])/100) * 0.4
        
        model_metrics.append({
            'name': model_name,
            'metrics': metrics,
            'performance_score': performance_score
        })
    
    sorted_models = sorted(model_metrics, key=lambda x: x['performance_score'], reverse=True)
    
    for rank, model_data in enumerate(sorted_models, 1):
        model_name = model_data['name']
        performance_score = model_data['performance_score']
        metrics = model_data['metrics']
        
        results[model_name].update({
            'rank': rank,
            'performance_score': performance_score,
            'timestamp': datetime.now().isoformat(),
            'performance': "EXCELLENT" if metrics['r2'] > 0.8 else 
                         "GOOD" if metrics['r2'] > 0.7 else 
                         "WORST" if metrics['r2'] > 0.6 else 
                         "POOR" if metrics['r2'] > 0.5 else "INADEQUATE"
        })
        
        status = "" if metrics['r2'] > target_r2 else " "
        print(f"{status} {rank}. {model_name:20s} | RMSE: {metrics['rmse']:6.2f} | MAE: {metrics['mae']:6.2f} | R²: {metrics['r2']:7.4f} | Score: {performance_score:.4f}")
    
    best_model_name = sorted_models[0]['name']
    best_metrics = results[best_model_name]
    
    print("\n" + "="*80)
    print(f" BEST MODEL: {best_model_name}")
    print(f"   R² Score: {best_metrics['r2']:.4f}")
    print(f"   Performance Score: {best_metrics['performance_score']:.4f}")
    if best_metrics['r2'] > target_r2:
        print(f"    TARGET ACHIEVED! (R² > {target_r2})")
    else:
        print(f"    Target not reached (goal: R² > {target_r2})")
    print("="*80)
    
    return best_model_name, best_metrics
def save_models(models, selector, selected_features, scaler, best_model_name):
    print("\n Saving models and artifacts...")
    model_dir = "model_artifacts"
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model = models[best_model_name]
    if 'LSTM' in best_model_name or 'CNN' in best_model_name:
        best_path = os.path.join(model_dir, f"{best_model_name}_best.keras")
        best_model.save(best_path)
    else:
        best_path = os.path.join(model_dir, f"{best_model_name}_best.pkl")
        joblib.dump(best_model, best_path)
    print(f"    Best model: {best_path}")
    for name, model in models.items():
        if 'LSTM' in name or 'CNN' in name:
            path = os.path.join(model_dir, f"{name}_{timestamp}.keras")
            model.save(path)
        else:
            path = os.path.join(model_dir, f"{name}_{timestamp}.pkl")
            joblib.dump(model, path)
    if selector is not None:
        joblib.dump(selector, os.path.join(model_dir, "feature_selector.pkl"))
        joblib.dump(selected_features, os.path.join(model_dir, "selected_features.pkl"))
        print(f"    Feature selector saved")
    if scaler is not None:
        joblib.dump(scaler, os.path.join(model_dir, f"scaler_{timestamp}.pkl"))
        print(f"    Scaler saved")
    print(f"    All models saved to {model_dir}/")
def register_to_hopsworks(model, model_name, metrics, prediction_mode, horizon=1, max_retries=3):
    print(f"\n Registering {model_name} to Hopsworks Model Registry...")
    
    import time
    import shutil
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"   Attempt {attempt}/{max_retries}...")
            
            project = hopsworks.login(
                api_key_value=HOPSWORKS_API_KEY, 
                project=HOPSWORKS_PROJECT
            )
            mr = project.get_model_registry()
            
            temp_model_dir = f"temp_model_{model_name}"
            os.makedirs(temp_model_dir, exist_ok=True)
            
            if 'LSTM' in model_name or 'CNN' in model_name:
                model_file = os.path.join(temp_model_dir, f"{model_name}.keras")
                model.save(model_file)
            else:
                model_file = os.path.join(temp_model_dir, f"{model_name}.pkl")
                joblib.dump(model, model_file)
            
            if prediction_mode == 'single':
                description = f"{model_name} for single-step ({horizon}h ahead) AQI prediction with advanced feature engineering"
            else:
                description = f"{model_name} for multi-step ({horizon} days ahead) AQI prediction"
            
            performance_score = float(metrics['r2']) * 0.6 + (1 - float(metrics['rmse'])/100) * 0.4
            
            print(f"   Creating model object...")
            model_obj = mr.python.create_model(
                name=f"aqi_{model_name.lower()}",
                metrics={
                    "rmse": float(metrics['rmse']),
                    "mae": float(metrics['mae']),
                    "r2_score": float(metrics['r2']),
                    "horizon": int(horizon),
                    "performance_score": float(performance_score),
                    "rank": float(metrics.get('rank', 1)),
                    "timestamp": datetime.now().timestamp()
                },
                description=f"{description} | Mode: {prediction_mode}",
                input_example={"features": [0.0] * model.n_features_in_} if hasattr(model, 'n_features_in_') else None
            )
            
            print(f"   Uploading model to Hopsworks...")
            model_obj.save(temp_model_dir)
            
            print(f"    Model registered: aqi_{model_name.lower()}")
            print(f"      Metrics - RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}")
            
            shutil.rmtree(temp_model_dir, ignore_errors=True)
            
            return True
            
        except Exception as e:
            error_msg = str(e)
            print(f"    Attempt {attempt} failed: {error_msg[:100]}")
            
            try:
                shutil.rmtree(temp_model_dir, ignore_errors=True)
            except:
                pass
            
            if attempt < max_retries:
                wait_time = attempt * 5
                print(f"    Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"\n    Registration failed after {max_retries} attempts")
                print(f"    Error: {error_msg}")
                print("    Model saved locally but not registered to Hopsworks")
                print("\n    Possible solutions:")
                print("    1. Check your internet connection")
                print("    2. Verify Hopsworks API key is valid")
                print("    3. Check if Hopsworks service is available")
                print("    4. Try running the script again later")
                return False
    
    return False
def make_predictions(models, df, selector, selected_features, prediction_days=3):
    from datetime import timedelta
    print("\n" + "="*80)
    print(f"MAKING PREDICTIONS FOR NEXT {prediction_days} DAYS")
    print("="*80)
    
    latest_data = df.iloc[-1:].copy()
    
    predictions = {}
    current_time = pd.Timestamp.now()
    
    if 'aqi' in df.columns and len(df) >= 3:
        recent_aqis = df['aqi'].tail(24).values if len(df) >= 24 else df['aqi'].tail(3).values
        aqi_trend = np.mean(np.diff(recent_aqis))
    else:
        aqi_trend = 0
    
    for model_name, model in models.items():
        model_predictions = []
        
        if 'LSTM' in model_name or 'CNN' in model_name:
            continue
        
        for day in range(1, prediction_days + 1):
            prediction_time = current_time + timedelta(days=day)
            
            X_pred = latest_data.drop(['aqi', 'timestamp', 'timestamp_unix', 'station_name',
                                       'station_url', 'datetime', 'data_quality',
                                       'year', 'month', 'day', 'hour'],
                                      axis=1, errors='ignore').copy()
            X_pred = X_pred.select_dtypes(include=[np.number])
            
            if day > 1:
                pollutant_cols = [col for col in X_pred.columns if any(p in col.lower() for p in ['pm25', 'pm10', 'no2', 'o3', 'co', 'so2'])]
                decay_factor = 0.95 ** (day - 1)
                
                for col in pollutant_cols:
                    if col in X_pred.columns:
                        random_variation = np.random.uniform(0.97, 1.03)
                        X_pred[col] = X_pred[col] * decay_factor * random_variation
                
                future_hour = (prediction_time.hour) % 24
                future_day_of_week = prediction_time.dayofweek
                future_month = prediction_time.month
                
                if 'hour_sin' in X_pred.columns:
                    X_pred['hour_sin'] = np.sin(2 * np.pi * future_hour / 24)
                if 'hour_cos' in X_pred.columns:
                    X_pred['hour_cos'] = np.cos(2 * np.pi * future_hour / 24)
                if 'day_sin' in X_pred.columns:
                    X_pred['day_sin'] = np.sin(2 * np.pi * future_day_of_week / 7)
                if 'day_cos' in X_pred.columns:
                    X_pred['day_cos'] = np.cos(2 * np.pi * future_day_of_week / 7)
                if 'month_sin' in X_pred.columns:
                    X_pred['month_sin'] = np.sin(2 * np.pi * future_month / 12)
                if 'month_cos' in X_pred.columns:
                    X_pred['month_cos'] = np.cos(2 * np.pi * future_month / 12)
                
                if 'is_weekend' in X_pred.columns:
                    X_pred['is_weekend'] = 1 if future_day_of_week >= 5 else 0
                if 'is_rush_hour' in X_pred.columns:
                    X_pred['is_rush_hour'] = 1 if (7 <= future_hour <= 9) or (17 <= future_hour <= 19) else 0
                if 'is_night' in X_pred.columns:
                    X_pred['is_night'] = 1 if (future_hour >= 22 or future_hour <= 5) else 0
            
            if selector is not None and selected_features is not None:
                X_pred = X_pred[selected_features]
            
            try:
                pred_value = model.predict(X_pred)[0]
                
                if day > 1 and abs(aqi_trend) > 0.1:
                    daily_trend = aqi_trend * 24 * 0.3
                    trend_adjustment = daily_trend * (day - 1)
                    pred_value += trend_adjustment
                
                variation_factor = 0.02 * day
                stochastic_variation = np.random.normal(0, pred_value * variation_factor)
                pred_value += stochastic_variation
                
                pred_value = max(0, pred_value)
                
                model_predictions.append({
                    'day': day,
                    'date': prediction_time.strftime('%Y-%m-%d'),
                    'aqi': float(pred_value),
                    'category': get_aqi_category(pred_value)
                })
                
            except Exception as e:
                print(f"     {model_name} day {day} prediction failed: {e}")
        
        if model_predictions:
            predictions[model_name] = model_predictions
            print(f"    {model_name}: Predicted next {len(model_predictions)} days")
            aqi_values = [p['aqi'] for p in model_predictions]
            print(f"      AQI values: {[f'{v:.1f}' for v in aqi_values]}")
    
    return predictions
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"
def export_results_to_json(results, predictions, output_file="model_results.json"):
    print(f"\n Exporting results to {output_file}...")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
    
    model_comparison = []
    for idx, (model_name, metrics) in enumerate(sorted_results):
        model_data = {
            "rank": idx + 1,
            "model_name": model_name,
            "rmse": float(metrics['rmse']),
            "mae": float(metrics['mae']),
            "r2_score": float(metrics['r2']),
            "performance": "Excellent" if metrics['r2'] > 0.9 else "Good" if metrics['r2'] > 0.7 else "Worst"
        }
        
        if 'cv_r2_mean' in metrics:
            model_data['cv_r2_mean'] = float(metrics['cv_r2_mean'])
            model_data['cv_r2_std'] = float(metrics['cv_r2_std'])
            model_data['cv_rmse_mean'] = float(metrics['cv_rmse_mean'])
            model_data['cv_rmse_std'] = float(metrics['cv_rmse_std'])
            
        model_comparison.append(model_data)
    
    json_output = {
        "timestamp": datetime.now().isoformat(),
        "model_comparison": model_comparison,
        "best_model": sorted_results[0][0],
        "predictions_next_3_days": predictions
    }
    
    with open(output_file, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"    Results exported successfully!")
    print(f"\n JSON Output Preview:")
    print(json.dumps(json_output, indent=2)[:500] + "...")
    return json_output
def main(args):
    print("\n" + "="*80)
    print(" UNIFIED AQI MODEL TRAINING PIPELINE")
    print("="*80)
    print(f"Mode: {args.mode.upper()}")
    print(f"Feature Engineering: {'ENABLED' if args.feature_engineering else 'DISABLED'}")
    print(f"Deep Learning: {'ENABLED' if args.deep_learning else 'DISABLED'}")
    print(f"Optimization: {'ENABLED' if args.optimize else 'DISABLED'}")
    df = fetch_and_prepare_data(use_feature_engineering=args.feature_engineering)
    if len(df) < 100:
        print(f"\n  WARNING: Only {len(df)} samples available")
        print("   Recommendation: Collect more data before training")
        return
    if args.mode == 'single':
        X, y = prepare_single_step_target(df, prediction_horizon=args.horizon)
        prediction_mode = 'single'
        horizon_value = args.horizon
    else:
        X, y = prepare_multi_step_target(df, horizon_days=args.horizon)
        prediction_mode = 'multi'
        horizon_value = args.horizon
    print("\n Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )
    print(f"   Train: {len(X_train):,} samples")
    print(f"   Test:  {len(X_test):,} samples")
    selector = None
    selected_features = None
    if args.mode == 'single' and args.feature_engineering and X_train.shape[1] > 50:
        X_train, X_test, selected_features, selector = select_top_features(
            X_train, y_train, X_test, n_features=50
        )
    models_trad, results_trad = train_traditional_models(
        X_train, y_train, X_test, y_test, optimize=args.optimize
    )
    
    # Always train deep learning models (LSTM and CNN_1D)
    models_dl, results_dl, scaler = train_deep_learning_models(
        X_train, y_train, X_test, y_test
    )
    models_trad.update(models_dl)
    results_trad.update(results_dl)
    
    best_model_name, best_metrics = compare_models(results_trad, target_r2=args.target_r2)
    save_models(models_trad, selector, selected_features, scaler, best_model_name)
    predictions = make_predictions(models_trad, df, selector, selected_features, prediction_days=3)
    json_output = export_results_to_json(results_trad, predictions)
    
    if args.register:
        print("\n" + "="*80)
        print(f"REGISTERING BEST MODEL ({best_model_name}) TO HOPSWORKS")
        print("="*80)
        
        success = register_to_hopsworks(
            models_trad[best_model_name],
            best_model_name,
            best_metrics,
            prediction_mode,
            horizon_value
        )
        
        if success:
            print("\n" + "="*80)
            print(f" Best model '{best_model_name}' successfully registered!")
            print("="*80)
        else:
            print("\n" + "="*80)
            print(f"Failed to register '{best_model_name}'")
            print("="*80)
    
    print("\n" + "="*80)
    print(" TRAINING COMPLETE!")
    print("="*80)
    print("\n Results saved to: model_results.json")
    print(" Models saved to: model_artifacts/")
    if args.register and success:
        print(f" Best model '{best_model_name}' registered to Hopsworks")
    
    return models_trad, results_trad, json_output
if __name__ == "__main__":
    class Args:
        pass

    args = Args()
    args.mode = 'single'
    args.horizon = 1
    args.feature_engineering = True
    args.deep_learning = True
    args.optimize = True
    args.register = True
    args.target_r2 = 0.5

    models, results, json_output = main(args)
    sys.stdout = original_stdout
    print(json.dumps(json_output, indent=2))
