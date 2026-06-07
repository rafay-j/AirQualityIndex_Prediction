# Air Quality Index (AQI) Prediction

A comprehensive machine learning project for predicting air quality index values using advanced data processing, feature engineering, and multiple predictive models.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Data Pipeline](#data-pipeline)
- [Model Comparison](#model-comparison)
- [Results](#results)
- [Deployment](#deployment)
- [Contributing](#contributing)

## 🎯 Overview

This project develops a machine learning system to predict Air Quality Index (AQI) values for the next 3 days. It combines real-time data fetching from the World Air Quality Index (WAQI) API, feature engineering, and multiple machine learning models to provide accurate air quality predictions.

**Best Model:** XGBoost with **R² Score: 0.9496** and **RMSE: 9.36**

## ✨ Features

- **Real-time Data Fetching**: Automated hourly data collection from WAQI API
- **Data Cleaning & Preprocessing**: Comprehensive data validation and cleaning pipeline
- **Feature Engineering**: Advanced feature extraction for improved model performance
- **Multiple ML Models**: XGBoost, Random Forest, LSTM, and CNN-1D
- **Model Comparison**: Automated evaluation and benchmarking
- **Feature Store Integration**: Hopsworks integration for feature management
- **Interactive Dashboard**: Streamlit-based web application for predictions and visualization
- **Automated Pipelines**: Daily and backfill data collection pipelines
- **Model Optimization**: Optuna-based hyperparameter tuning

## 📁 Project Structure

```
AirQualityIndex_Prediction/
├── config.py                          # Configuration and environment variables
├── data_fetcher.py                    # Fetch AQI data from WAQI API
├── data_cleaning.py                   # Data validation and cleaning
├── feature_engineering.py             # Feature extraction and transformation
├── exploratory_data_analysis.py       # EDA and visualization
├── train_model.py                     # Model training and evaluation
├── feature_store.py                   # Hopsworks feature store management
├── daily_pipeline.py                  # Daily data collection pipeline
├── backfill_pipeline.py               # Historical data backfill
├── check_data_status.py               # Data quality monitoring
├── streamlit_app.py                   # Interactive web dashboard
├── upload_historical_data.py          # Historical data upload utility
├── requirements.txt                   # Python dependencies
├── runtime.txt                        # Python version specification
├── model_results.json                 # Latest model evaluation results
├── packages.txt                       # System packages
├── github_workflows/                  # CI/CD workflows
└── streamlit/                         # Streamlit configuration
```

## 🚀 Installation

### Prerequisites
- Python 3.11+
- Git

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/rafay-j/AirQualityIndex_Prediction.git
cd AirQualityIndex_Prediction
```

2. **Navigate to the main directory:**
```bash
cd AirQualityIndex_Prediction/AirQualityIndex_Prediction-main
```

3. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

Set the following environment variables in a `.env` file or system environment:

```env
WAQI_API_TOKEN=your_waqi_api_token
STATION_ID=your_station_id
HOPSWORKS_API_KEY=your_hopsworks_api_key
HOPSWORKS_PROJECT=your_project_name
```

### Default Configuration Values (config.py):

| Variable | Default | Description |
|----------|---------|-------------|
| `STATION_ID` | A401143 | WAQI station identifier |
| `BACKFILL_DAYS` | 90 | Days of historical data to backfill |
| `COLLECTION_INTERVAL_HOURS` | 1 | Hourly data collection frequency |
| `FEATURE_GROUP_NAME` | aqi_features | Hopsworks feature group name |
| `PREDICTION_HORIZON_DAYS` | 3 | Number of days to forecast |
| `MIN_TRAINING_ROWS` | 20 | Minimum samples for model training |

## 📊 Usage

### 1. **Run the Interactive Dashboard:**
```bash
streamlit run streamlit_app.py
```
The dashboard provides:
- Real-time AQI visualization
- Model predictions for the next 3 days
- Historical trend analysis
- Model performance metrics

### 2. **Train Models:**
```bash
python train_model.py
```
Trains all models (XGBoost, Random Forest, LSTM, CNN-1D) and generates `model_results.json`

### 3. **Fetch Fresh Data:**
```bash
python daily_pipeline.py
```
Collects the latest AQI data from WAQI API

### 4. **Backfill Historical Data:**
```bash
python backfill_pipeline.py
```
Downloads 90 days of historical data

### 5. **Check Data Status:**
```bash
python check_data_status.py
```
Validates data quality and pipeline status

### 6. **Upload Historical Data:**
```bash
python upload_historical_data.py
```
Uploads historical data to Hopsworks feature store

## 🔄 Data Pipeline

### Data Flow:
```
WAQI API → data_fetcher.py → data_cleaning.py → feature_engineering.py → feature_store.py → train_model.py
```

### Key Steps:

1. **Data Fetching** (`data_fetcher.py`):
   - Hourly collection from WAQI API
   - Real-time AQI readings and auxiliary metrics
   - Error handling and retry logic

2. **Data Cleaning** (`data_cleaning.py`):
   - Missing value handling
   - Outlier detection and treatment
   - Data validation and quality checks

3. **Feature Engineering** (`feature_engineering.py`):
   - Time-based features (hour, day, month, season)
   - Rolling statistics (moving averages)
   - Lag features for temporal patterns
   - Statistical transformations

4. **Model Training** (`train_model.py`):
   - Train-test split
   - Model training with cross-validation
   - Hyperparameter optimization with Optuna
   - Model evaluation and comparison

## 📈 Model Comparison

### Performance Results (from model_results.json):

| Model | R² Score | RMSE | MAE | Rank |
|-------|----------|------|-----|------|
| **XGBoost** | **0.9496** | **9.36** | **5.57** | 🥇 |
| Random Forest | 0.9479 | 9.51 | 5.69 | 🥈 |
| LSTM | 0.8493 | 16.19 | 10.71 | 🥉 |
| CNN-1D | 0.0860 | 39.88 | 30.93 | 4️⃣ |

### Key Insights:
- **XGBoost** provides the best predictive performance (Excellent rating)
- Gradient boosting models outperform deep learning approaches for this dataset
- Cross-validation R² mean: 0.9336 (±0.0255) indicates robust generalization

## 🔮 Predictions

### Next 3-Day Forecast Example:

```
Day 1 (2025-11-15): AQI 181 - Unhealthy
Day 2 (2025-11-16): AQI 184 - Unhealthy
Day 3 (2025-11-17): AQI 200 - Very Unhealthy
```

### AQI Categories:
- 0-50: Good
- 51-100: Moderate
- 101-150: Unhealthy for Sensitive Groups
- 151-200: Unhealthy
- 201-300: Very Unhealthy
- 301+: Hazardous

## 🌐 Deployment

### Streamlit Cloud Deployment:
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Set environment variables in Streamlit Cloud settings
4. Deploy automatically on push

### System Requirements:
- Python 3.11 (specified in `runtime.txt`)
- System packages listed in `packages.txt`
- All Python dependencies from `requirements.txt`

## 🔄 CI/CD Workflows

Automated workflows for:
- Daily data collection
- Weekly model retraining
- Data quality monitoring
- Automated testing

Located in: `github_workflows/`

## 📦 Dependencies

### Core Libraries:
- **streamlit**: Interactive web dashboard
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning models
- **xgboost**: Gradient boosting
- **tensorflow**: Deep learning models
- **optuna**: Hyperparameter optimization
- **hopsworks**: Feature store integration
- **requests**: API interactions
- **plotly**: Interactive visualizations

See `requirements.txt` for complete list with versions.

## 📝 Data Sources

- **Primary Source**: [World Air Quality Index (WAQI)](https://waqi.info/)
- **Features**: PM2.5, PM10, O3, NO2, SO2, CO, Temperature, Humidity, Pressure

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Rafay**
- GitHub: [@rafay-j](https://github.com/rafay-j)

## 📞 Support & Issues

For issues, questions, or suggestions, please open a GitHub issue in the repository.

## 🔗 References

- [WAQI API Documentation](https://waqi.info/api-doc/)
- [Hopsworks Feature Store](https://www.hopsworks.ai/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Last Updated**: June 2026
**Status**: Active Development ✅
