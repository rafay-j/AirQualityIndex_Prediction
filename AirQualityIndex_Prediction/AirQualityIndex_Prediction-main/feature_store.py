import hopsworks
import pandas as pd
from typing import Optional
class HopsworksFeatureStore:
    def __init__(self, api_key: str, project_name: str):
        self.api_key = api_key
        self.project_name = project_name
        self.project = None
        self.fs = None
    def connect(self) -> bool:
        try:
            self.project = hopsworks.login(
                api_key_value=self.api_key,
                project=self.project_name
            )
            self.fs = self.project.get_feature_store()
            print(" Connected to Hopsworks")
            return True
        except Exception as e:
            print(f" Hopsworks connection failed: {e}")
            return False
    def create_or_get_feature_group(self, name: str, version: int = 1,
                                    description: str = "") -> Optional[object]:
        if self.fs is None:
            print(" Not connected to Hopsworks")
            return None
        try:
            try:
                feature_group = self.fs.get_feature_group(name=name, version=version)
                print(f" Using existing feature group: {name} (version {version})")
                return feature_group
            except Exception:
                feature_group = self.fs.create_feature_group(
                    name=name,
                    version=version,
                    description=description,
                    primary_key=['timestamp'],
                    event_time='datetime',
                    online_enabled=False
                )
                print(f" Created new feature group: {name} (version {version})")
                return feature_group
        except Exception as e:
            print(f" Feature group operation failed: {e}")
            return None
    def insert_data(self, feature_group, df: pd.DataFrame) -> bool:
        try:
            feature_group.insert(df, write_options={"wait_for_job": False})
            print(f" Inserted {len(df)} records into feature store")
            return True
        except Exception as e:
            print(f" Data insertion failed: {e}")
            return False
    def read_data(self, feature_group) -> Optional[pd.DataFrame]:
        try:
            df = feature_group.read()
            return df
        except Exception as e:
            print(f" Read failed: {e}")
            return None
def store_features(df: pd.DataFrame, api_key: str, project_name: str,
                   feature_group_name: str = "aqi_features",
                   version: int = 1) -> bool:
    store = HopsworksFeatureStore(api_key, project_name)
    if not store.connect():
        return False
    feature_group = store.create_or_get_feature_group(
        name=feature_group_name,
        version=version,
        description="AQI features with weather and pollutant data"
    )
    if feature_group is None:
        return False
    return store.insert_data(feature_group, df)
def read_features(api_key: str, project_name: str,
                  feature_group_name: str = "aqi_features",
                  version: int = 1) -> Optional[pd.DataFrame]:
    store = HopsworksFeatureStore(api_key, project_name)
    if not store.connect():
        return None
    try:
        feature_group = store.fs.get_feature_group(name=feature_group_name, version=version)
        return store.read_data(feature_group)
    except Exception as e:
        print(f" Feature group not found: {e}")
        return None
