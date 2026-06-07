import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
import pickle
import hopsworks
from feature_engineering import engineer_features
from data_cleaning import clean_data

st.set_page_config(
    page_title="AQI Prediction System",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .block-container {
        background: white;
        border-radius: 20px;
        padding: 2rem 3rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin-top: 2rem;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #3b82f6 100%);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        color: white;
    }
    
    h1 {
        color: #1e40af;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #3b82f6;
        font-weight: 600;
        border-bottom: 3px solid #60a5fa;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        color: #2563eb;
        font-weight: 600;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 15px;
        border-left: 5px solid #3b82f6;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #f0f9ff 0%, #e0f2fe 100%);
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Radio buttons in sidebar */
    [data-testid="stSidebar"] .stRadio > label {
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    [data-testid="stSidebar"] .stRadio > div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: white !important;
    }
    
    /* Make sidebar headings white */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

HOPSWORKS_API_KEY = st.secrets.get("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT = st.secrets.get("HOPSWORKS_PROJECT", "AQI_Project_10")

@st.cache_resource
def load_best_model_from_hopsworks():
    try:
        project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT
        )
        
        mr = project.get_model_registry()
        
        models_to_check = [
            ('aqi_linearregression', 4),
            ('aqi_linearregression', 3),
            ('aqi_linearregression', 2),
            ('aqi_linearregression', 1),
            ('aqi_xgboost', 3),  # Latest version
            ('aqi_xgboost', 2),
            ('aqi_xgboost', 1),
        ]
        
        # Find best model based on R² score
        best_model_info = None
        best_r2 = -999
        
        for model_name, version in models_to_check:
            try:
                # Get specific version
                model_obj = mr.get_model(model_name, version=version)
                
                # Try different metric keys
                r2_score = -999
                if hasattr(model_obj, 'training_metrics') and model_obj.training_metrics:
                    # Try different possible keys
                    r2_score = (model_obj.training_metrics.get('r2_score') or 
                               model_obj.training_metrics.get('r2') or 
                               model_obj.training_metrics.get('R2') or -999)
                
                st.sidebar.info(f"Checking: {model_obj.name} (v{model_obj.version}) - R²: {r2_score:.4f}")
                
                # Debug: show all available metrics
                if hasattr(model_obj, 'training_metrics') and model_obj.training_metrics:
                    st.sidebar.write(f"Available metrics: {list(model_obj.training_metrics.keys())}")
                
                if r2_score > best_r2 and r2_score > -999:  # Valid R² score
                    best_r2 = r2_score
                    best_model_info = {
                        'name': model_obj.name,
                        'model_obj': model_obj,
                        'r2': r2_score,
                        'version': model_obj.version,
                        'metrics': model_obj.training_metrics
                    }
            except Exception as e:
                # Version might not exist, skip silently
                continue
        
        if not best_model_info or best_r2 == -999:
            st.error("No valid AQI models found in Hopsworks Model Registry with R² scores")
            return None
        
        # Download and load the model
        model_dir = best_model_info['model_obj'].download()
        
        st.sidebar.success(f"Downloaded to: {model_dir}")
        
        # List all files in model directory
        all_files = os.listdir(model_dir)
        st.sidebar.write(f"Files in model dir: {all_files}")
        
        # Try to find and load the model file
        model = None
        
        # Try different file patterns
        file_patterns = [
            f"{best_model_info['name'].replace('aqi_', '').title()}.pkl",
            f"{best_model_info['name'].replace('aqi_', '')}.pkl",
            "model.pkl",
        ]
        
        for pattern in file_patterns:
            for file in all_files:
                if file.lower() == pattern.lower():
                    model_path = os.path.join(model_dir, file)
                    st.sidebar.info(f"Trying to load: {file}")
                    
                    try:
                        import joblib
                        # Try joblib first (common for sklearn models)
                        model = joblib.load(model_path)
                        st.sidebar.success(f"Loaded with joblib: {file}")
                        break
                    except:
                        try:
                            # Try pickle
                            with open(model_path, 'rb') as f:
                                model = pickle.load(f)
                            st.sidebar.success(f"Loaded with pickle: {file}")
                            break
                        except Exception as e:
                            st.sidebar.error(f"Failed to load {file}: {str(e)}")
                            continue
            
            if model is not None:
                break
        
        # If still no model, try any .pkl file
        if model is None:
            for file in all_files:
                if file.endswith('.pkl'):
                    model_path = os.path.join(model_dir, file)
                    st.sidebar.info(f"Trying any pkl file: {file}")
                    
                    try:
                        import joblib
                        model = joblib.load(model_path)
                        st.sidebar.success(f"Loaded with joblib: {file}")
                        break
                    except:
                        try:
                            with open(model_path, 'rb') as f:
                                model = pickle.load(f)
                            st.sidebar.success(f"Loaded with pickle: {file}")
                            break
                        except Exception as e:
                            st.sidebar.warning(f"Skipping {file}: {str(e)}")
                            continue
        
        if model is None:
            st.error("Could not load model file from downloaded directory")
            return None
        
        return {
            'model': model,
            'name': best_model_info['name'].replace('aqi_', '').upper(),
            'r2': best_model_info['r2'],
            'version': best_model_info['version'],
            'metrics': best_model_info['metrics']
        }
        
    except Exception as e:
        st.error(f"Error loading model from Hopsworks: {str(e)}")
        return None

@st.cache_data(ttl=300)
def load_current_features_from_hopsworks():
    """Load current features from Hopsworks Feature Store and apply feature engineering"""
    try:
        # Connect to Hopsworks
        project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT
        )
        
        # Get feature store
        fs = project.get_feature_store()
        
        # Get feature group
        fg = fs.get_feature_group(name="aqi_features", version=1)
        
        # Read ALL data (we need historical data for lag/rolling features)
        df_raw = fg.read()
        
        if len(df_raw) == 0:
            return None
        
        st.sidebar.info(f" Loaded {len(df_raw)} rows from Feature Store")
        
        # Apply data cleaning
        df_clean = clean_data(df_raw, verbose=False)
        st.sidebar.info(f"🧹 After cleaning: {len(df_clean)} rows")
        
        # Apply feature engineering (SAME AS TRAINING)
        df_engineered, feature_groups = engineer_features(df_clean, verbose=False)
        st.sidebar.success(f" Features: {len(df_raw.columns)} → {len(df_engineered.columns)}")
        
        # Get the most recent record (after feature engineering)
        df_latest = df_engineered.sort_values('timestamp', ascending=False).head(1)
        
        st.sidebar.info(f" Latest timestamp: {df_latest['timestamp'].iloc[0]}")
        
        return df_latest
        
    except Exception as e:
        st.error(f"Error loading features from Hopsworks: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_predictions_from_local():
    """Load predictions from model_results.json generated by train_model.py"""
    try:
        model_results_path = "model_results.json"
        
        if not os.path.exists(model_results_path):
            st.error(f" {model_results_path} not found. Please run train_model.py first.")
            return None
        
        with open(model_results_path, 'r') as f:
            results = json.load(f)
        
        # Get best model name
        best_model = results.get('best_model', 'LinearRegression')
        
        # Get predictions for best model
        all_predictions = results.get('predictions_next_3_days', {})
        
        if best_model not in all_predictions:
            st.warning(f"No predictions found for {best_model}, trying first available model")
            best_model = list(all_predictions.keys())[0] if all_predictions else None
        
        if not best_model:
            st.error("No predictions available in model_results.json")
            return None
        
        predictions = all_predictions[best_model]
        
        st.sidebar.success(f" Loaded predictions from {best_model}")
        st.sidebar.info(f" Generated: {results.get('timestamp', 'Unknown')}")
        
        return {
            'predictions': predictions,
            'best_model': best_model,
            'model_comparison': results.get('model_comparison', []),
            'timestamp': results.get('timestamp', 'Unknown')
        }
        
    except Exception as e:
        st.error(f"Error loading predictions: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def get_aqi_category(aqi):
    """Get AQI category with color"""
    if aqi <= 50:
        return {'level': 'Good', 'color': '#10b981'}
    elif aqi <= 100:
        return {'level': 'Moderate', 'color': '#f59e0b'}
    elif aqi <= 150:
        return {'level': 'Unhealthy for Sensitive Groups', 'color': '#ff7e5f'}
    elif aqi <= 200:
        return {'level': 'Unhealthy', 'color': '#ef4444'}
    elif aqi <= 300:
        return {'level': 'Very Unhealthy', 'color': '#991b1b'}
    else:
        return {'level': 'Hazardous', 'color': '#7f1d1d'}


def show_current_conditions(current_features):
    """Display current air quality conditions"""
    st.markdown("<h2 style='text-align: center;'> Current Air Quality Conditions</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Get current AQI
    if 'aqi' in current_features.columns:
        current_aqi = float(current_features['aqi'].iloc[0])
        category = get_aqi_category(current_aqi)
        
        # Hero AQI Display
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {category['color']} 0%, {category['color']}dd 100%); 
                        padding: 40px; border-radius: 20px; text-align: center; 
                        box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
                <p style="color: white; font-size: 1.2rem; margin: 0 0 10px 0; opacity: 0.9;">Air Quality Index</p>
                <h1 style="color: white; font-size: 5rem; margin: 10px 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">{int(current_aqi)}</h1>
                <h3 style="color: white; margin: 10px 0 0 0; font-size: 1.8rem;">{category['level']}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            pm25 = current_features['pm25'].iloc[0] if 'pm25' in current_features.columns else 0
            pm10 = current_features['pm10'].iloc[0] if 'pm10' in current_features.columns else 0
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f59e0b 0%, #f97316 100%); 
                        padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 15px;
                        box-shadow: 0 5px 15px rgba(245,158,11,0.3);">
                <p style="color: white; margin: 0; font-size: 0.9rem; opacity: 0.9;">PM2.5</p>
                <h2 style="color: white; margin: 5px 0; font-size: 2.5rem;">{pm25:.1f}</h2>
                <p style="color: white; margin: 0; font-size: 0.8rem;">µg/m³</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); 
                        padding: 20px; border-radius: 15px; text-align: center;
                        box-shadow: 0 5px 15px rgba(139,92,246,0.3);">
                <p style="color: white; margin: 0; font-size: 0.9rem; opacity: 0.9;">PM10</p>
                <h2 style="color: white; margin: 5px 0; font-size: 2.5rem;">{pm10:.1f}</h2>
                <p style="color: white; margin: 0; font-size: 0.8rem;">µg/m³</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            temp = current_features['temperature'].iloc[0] if 'temperature' in current_features.columns else 0
            humidity = current_features['humidity'].iloc[0] if 'humidity' in current_features.columns else 0
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%); 
                        padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 15px;
                        box-shadow: 0 5px 15px rgba(6,182,212,0.3);">
                <p style="color: white; margin: 0; font-size: 0.9rem; opacity: 0.9;">Temperature</p>
                <h2 style="color: white; margin: 5px 0; font-size: 2.5rem;">{temp:.1f}°</h2>
                <p style="color: white; margin: 0; font-size: 0.8rem;">Celsius</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                        padding: 20px; border-radius: 15px; text-align: center;
                        box-shadow: 0 5px 15px rgba(16,185,129,0.3);">
                <p style="color: white; margin: 0; font-size: 0.9rem; opacity: 0.9;">Humidity</p>
                <h2 style="color: white; margin: 5px 0; font-size: 2.5rem;">{humidity:.0f}%</h2>
                <p style="color: white; margin: 0; font-size: 0.8rem;">Relative</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Detailed Pollutant Levels
    st.markdown("<h3> Detailed Pollutant Analysis</h3>", unsafe_allow_html=True)
    
    pollutant_cols = st.columns(4)
    pollutants = [
        ('pm25', 'PM2.5', 'µg/m³', '#f59e0b'),
        ('pm10', 'PM10', 'µg/m³', '#8b5cf6'),
        ('o3', 'Ozone (O₃)', 'ppb', '#06b6d4'),
        ('no2', 'NO₂', 'ppb', '#ef4444')
    ]
    
    for idx, (col_name, label, unit, color) in enumerate(pollutants):
        if col_name in current_features.columns:
            value = current_features[col_name].iloc[0]
            with pollutant_cols[idx]:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {color}22 0%, {color}11 100%); 
                            padding: 20px; border-radius: 12px; border-left: 4px solid {color};
                            text-align: center;">
                    <p style="color: {color}; margin: 0; font-weight: 600; font-size: 0.9rem;">{label}</p>
                    <h3 style="color: #1e293b; margin: 10px 0 5px 0; font-size: 2rem;">{value:.1f}</h3>
                    <p style="color: #64748b; margin: 0; font-size: 0.8rem;">{unit}</p>
                </div>
                """, unsafe_allow_html=True)

def show_forecast():
    """Display 3-day forecast"""
    st.markdown("<h2 style='text-align: center;'>🔮 3-Day AQI Forecast</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Load predictions from local model_results.json
    pred_data = load_predictions_from_local()
    
    if pred_data:
        predictions = pred_data['predictions']
        best_model = pred_data['best_model']
        timestamp = pred_data['timestamp']
        
        # Display forecast cards
        cols = st.columns(3)
        
        day_names = ['Tomorrow', 'Day After Tomorrow', 'In 3 Days']
        gradient_colors = [
            'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
            'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)'
        ]
        
        for idx, pred in enumerate(predictions):
            with cols[idx]:
                aqi_value = pred['aqi']
                category = get_aqi_category(aqi_value)
                
                st.markdown(f"""
                <div style="background: {gradient_colors[idx]}; 
                            padding: 25px; border-radius: 20px; 
                            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                            text-align: center; min-height: 280px;">
                    <h4 style="color: white; margin: 0 0 10px 0; font-size: 1.3rem;">{day_names[idx]}</h4>
                    <p style="color: rgba(255,255,255,0.9); margin: 5px 0 20px 0; font-size: 0.95rem;">{pred['date']}</p>
                    <h2 style="color: white; font-size: 4rem; margin: 20px 0; text-shadow: 2px 2px 8px rgba(0,0,0,0.3);">
                        {int(aqi_value)}
                    </h2>
                    <div style="background: {category['color']}; 
                                padding: 12px; border-radius: 12px; margin-top: 20px;
                                box-shadow: 0 4px 10px rgba(0,0,0,0.2);">
                        <strong style="color: white; font-size: 1.1rem;">{category['level']}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Trend chart
        st.markdown("<h3> AQI Trend Forecast</h3>", unsafe_allow_html=True)
        
        dates = [pred['date'] for pred in predictions]
        aqis = [pred['aqi'] for pred in predictions]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=aqis,
            mode='lines+markers',
            line=dict(color='#667eea', width=4),
            marker=dict(size=15, color='#764ba2', line=dict(color='white', width=2)),
            name='Predicted AQI',
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.1)'
        ))
        
        # Add AQI zones with labels
        fig.add_hrect(y0=0, y1=50, fillcolor="#10b981", opacity=0.1, line_width=0, 
                     annotation_text="Good", annotation_position="right")
        fig.add_hrect(y0=50, y1=100, fillcolor="#f59e0b", opacity=0.1, line_width=0,
                     annotation_text="Moderate", annotation_position="right")
        fig.add_hrect(y0=100, y1=150, fillcolor="#ff7e5f", opacity=0.1, line_width=0,
                     annotation_text="Unhealthy (Sensitive)", annotation_position="right")
        fig.add_hrect(y0=150, y1=200, fillcolor="#ef4444", opacity=0.1, line_width=0,
                     annotation_text="Unhealthy", annotation_position="right")
        
        fig.update_layout(
            title={
                'text': "3-Day AQI Prediction Trend",
                'font': {'size': 20, 'color': '#1e40af'}
            },
            xaxis_title="Date",
            yaxis_title="AQI Level",
            template="plotly_white",
            height=450,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction details
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%); 
                    padding: 20px; border-radius: 15px; border-left: 5px solid #0284c7; margin-top: 20px;">
            <h4 style="color: #0c4a6e; margin: 0 0 10px 0;"> Prediction Details</h4>
            <p style="color: #075985; margin: 5px 0;"><strong>Model:</strong> {best_model}</p>
            <p style="color: #075985; margin: 5px 0;"><strong>Source:</strong> model_results.json (Local)</p>
            <p style="color: #075985; margin: 5px 0;"><strong>Generated:</strong> {timestamp}</p>
            <p style="color: #075985; margin: 5px 0;"><strong>Update Frequency:</strong> Daily (via GitHub Actions)</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("❌ No predictions available. Please run train_model.py to generate predictions.")

def show_model_analytics():
    """Display model performance analytics"""
    st.markdown("<h2 style='text-align: center;'>Model Performance Analytics</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Load model comparison from local results
    pred_data = load_predictions_from_local()
    
    if pred_data and 'model_comparison' in pred_data:
        model_comparison = pred_data['model_comparison']
        
        st.markdown("<h3>Top Performing Models</h3>", unsafe_allow_html=True)
        
        top3_cols = st.columns(3)
        medal_colors = ['#ffd700', '#c0c0c0', '#cd7f32']
        rank_labels = ['1st', '2nd', '3rd']
        
        for idx, model in enumerate(model_comparison[:3]):
            with top3_cols[idx]:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {medal_colors[idx]}22 0%, {medal_colors[idx]}11 100%); 
                            padding: 20px; border-radius: 15px; border-top: 4px solid {medal_colors[idx]};
                            text-align: center; min-height: 200px;">
                    <div style="font-size: 2rem; margin-bottom: 10px; color: {medal_colors[idx]}; font-weight: bold;">{rank_labels[idx]}</div>
                    <h4 style="color: #1e293b; margin: 10px 0;">{model['model_name']}</h4>
                    <p style="color: #64748b; margin: 5px 0; font-size: 0.9rem;">{model['performance']}</p>
                    <div style="margin-top: 15px;">
                        <p style="color: #334155; margin: 5px 0;"><strong>R²:</strong> {model['r2_score']:.4f}</p>
                        <p style="color: #334155; margin: 5px 0;"><strong>RMSE:</strong> {model['rmse']:.2f}</p>
                        <p style="color: #334155; margin: 5px 0;"><strong>MAE:</strong> {model['mae']:.2f}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        st.markdown("<h3>Complete Model Comparison</h3>", unsafe_allow_html=True)
        
        for model in model_comparison:
            # Color based on performance
            if model['performance'] == 'Excellent':
                border_color = '#10b981'
                bg_color = '#d1fae522'
            elif model['performance'] == 'Good':
                border_color = '#f59e0b'
                bg_color = '#fef3c722'
            else:
                border_color = '#ef4444'
                bg_color = '#fee2e222'
            
            with st.expander(f"#{model['rank']} - {model['model_name']} ({model['performance']})", expanded=(model['rank'] == 1)):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div style="background: {bg_color}; padding: 15px; border-radius: 10px; border-left: 3px solid {border_color};">
                        <p style="color: #64748b; margin: 0; font-size: 0.85rem;">R² Score</p>
                        <h3 style="color: {border_color}; margin: 5px 0;">{model['r2_score']:.4f}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="background: {bg_color}; padding: 15px; border-radius: 10px; border-left: 3px solid {border_color};">
                        <p style="color: #64748b; margin: 0; font-size: 0.85rem;">RMSE</p>
                        <h3 style="color: {border_color}; margin: 5px 0;">{model['rmse']:.2f}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div style="background: {bg_color}; padding: 15px; border-radius: 10px; border-left: 3px solid {border_color};">
                        <p style="color: #64748b; margin: 0; font-size: 0.85rem;">MAE</p>
                        <h3 style="color: {border_color}; margin: 5px 0;">{model['mae']:.2f}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div style="background: {bg_color}; padding: 15px; border-radius: 10px; border-left: 3px solid {border_color};">
                        <p style="color: #64748b; margin: 0; font-size: 0.85rem;">Rank</p>
                        <h3 style="color: {border_color}; margin: 5px 0;">#{model['rank']}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Cross-validation metrics if available
                if 'cv_r2_mean' in model:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("** Cross-Validation Results:**")
                    cv_col1, cv_col2 = st.columns(2)
                    
                    with cv_col1:
                        st.metric("CV R² Mean", f"{model['cv_r2_mean']:.4f}", f"±{model['cv_r2_std']:.4f}")
                        st.metric("CV RMSE Mean", f"{model['cv_rmse_mean']:.2f}", f"±{model['cv_rmse_std']:.2f}")
                    
                    with cv_col2:
                        st.metric("CV R² Std", f"{model['cv_r2_std']:.4f}")
                        st.metric("CV RMSE Std", f"{model['cv_rmse_std']:.2f}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3>R² Score Comparison</h3>", unsafe_allow_html=True)
            
            model_names = [m['model_name'] for m in model_comparison]
            r2_scores = [m['r2_score'] for m in model_comparison]
            colors = ['#10b981' if m['performance'] == 'Excellent' else '#f59e0b' if m['performance'] == 'Good' else '#ef4444' for m in model_comparison]
            
            fig1 = go.Figure(go.Bar(
                y=model_names,
                x=r2_scores,
                orientation='h',
                marker=dict(color=colors, line=dict(color='white', width=1)),
                text=[f"{score:.4f}" for score in r2_scores],
                textposition='outside'
            ))
            
            fig1.update_layout(
                height=350,
                template="plotly_white",
                xaxis_title="R² Score",
                yaxis_title="",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=10)
            )
            
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.markdown("<h3> RMSE Comparison</h3>", unsafe_allow_html=True)
            
            rmse_scores = [m['rmse'] for m in model_comparison]
            
            fig2 = go.Figure(go.Bar(
                y=model_names,
                x=rmse_scores,
                orientation='h',
                marker=dict(color='#667eea', line=dict(color='white', width=1)),
                text=[f"{score:.2f}" for score in rmse_scores],
                textposition='outside'
            ))
            
            fig2.update_layout(
                height=350,
                template="plotly_white",
                xaxis_title="RMSE (Lower is Better)",
                yaxis_title="",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=10)
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        # Training info
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); 
                    padding: 20px; border-radius: 15px; border-left: 5px solid #3b82f6;">
            <h4 style="color: #1e40af; margin: 0 0 15px 0;"> Training Information</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <p style="color: #1e40af; margin: 5px 0;"><strong>Last Training:</strong> {pred_data['timestamp']}</p>
                <p style="color: #1e40af; margin: 5px 0;"><strong>Best Model:</strong> {pred_data['best_model']}</p>
                <p style="color: #1e40af; margin: 5px 0;"><strong>Total Models:</strong> {len(model_comparison)}</p>
                <p style="color: #1e40af; margin: 5px 0;"><strong>Data Source:</strong> Hopsworks Feature Store</p>
                <p style="color: #1e40af; margin: 5px 0; grid-column: 1 / -1;"><strong>Schedule:</strong> Daily via GitHub Actions</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error(" No model comparison data available. Please run train_model.py first.")


def main():
    # Title with emoji and styling
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <h1 style='font-size: 3rem; margin-bottom: 0;'>
            Air Quality Index Prediction System
        </h1>
        <p style='font-size: 1.2rem; color: #64748b; margin-top: 0.5rem;'>
            Real-time AQI monitoring and 3-day forecasting powered by Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h2 style='color: white; font-size: 1.8rem;'> Navigation</h2>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.radio(
            "",
            [" Current Conditions", " 3-Day Forecast", " Model Analytics"],
            label_visibility="collapsed"
        )
        
        st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.2); margin: 2rem 0;'>", unsafe_allow_html=True)
        
        # System Status
        st.markdown("<h3 style='color: white;'> System Status</h3>", unsafe_allow_html=True)
        
        # Check if model_results.json exists
        if os.path.exists("model_results.json"):
            with open("model_results.json", 'r') as f:
                results = json.load(f)
            
            st.markdown(f"""
            <div style='background: rgba(16, 185, 129, 0.2); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;'>
                <p style='margin: 0; color: #10b981; font-weight: 600;'> System Online</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                <p style='margin: 0.3rem 0; color: white;'><strong>Best Model:</strong> {results.get('best_model', 'N/A')}</p>
                <p style='margin: 0.3rem 0; color: white;'><strong>Last Training:</strong> {results.get('timestamp', 'N/A')[:10]}</p>
                <p style='margin: 0.3rem 0; color: white;'><strong>Models Trained:</strong> {len(results.get('model_comparison', []))}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: rgba(239, 68, 68, 0.2); padding: 1rem; border-radius: 10px;'>
                <p style='margin: 0; color: #ef4444; font-weight: 600;'> No Model Results</p>
                <p style='margin: 0.5rem 0 0 0; color: white; font-size: 0.9rem;'>Run train_model.py first</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.2); margin: 2rem 0;'>", unsafe_allow_html=True)
        
        # Data Status
        st.markdown("<h3 style='color: white;'> Data Status</h3>", unsafe_allow_html=True)
        
        with st.spinner("Loading features..."):
            current_features = load_current_features_from_hopsworks()
        
        if current_features is not None and len(current_features) > 0:
            timestamp = current_features['timestamp'].iloc[0] if 'timestamp' in current_features.columns else 'N/A'
            st.markdown(f"""
            <div style='background: rgba(16, 185, 129, 0.2); padding: 1rem; border-radius: 10px;'>
                <p style='margin: 0; color: #10b981; font-weight: 600;'> Features Loaded</p>
                <p style='margin: 0.5rem 0 0 0; color: white; font-size: 0.9rem;'>Updated: {timestamp}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: rgba(239, 68, 68, 0.2); padding: 1rem; border-radius: 10px;'>
                <p style='margin: 0; color: #ef4444; font-weight: 600;'> No Features</p>
            </div>
            """, unsafe_allow_html=True)
            return
    
    # Main content based on selected page
    if page == " Current Conditions":
        show_current_conditions(current_features)
    elif page == " 3-Day Forecast":
        show_forecast()
    else:
        show_model_analytics()

if __name__ == "__main__":
    main()
