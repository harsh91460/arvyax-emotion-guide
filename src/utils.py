"""
ArvyaX Utils - Feature engineering & helpers
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from typing import Tuple, Optional

def preprocess_text(text_series: pd.Series) -> np.ndarray:
    """TF-IDF text preprocessing"""
    tfidf = TfidfVectorizer(max_features=50, stop_words='english')
    return tfidf.fit_transform(text_series.fillna('missing').astype(str)).toarray()

def safe_fill_na(df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
    """Safe NaN filling with medians"""
    df_filled = df.copy()
    for col in num_cols:
        median = df[col].median()
        df_filled[col] = df[col].fillna(median)
    return df_filled

def create_categorical_dummies(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                             cat_cols: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Safe categorical encoding"""
    all_cat = pd.concat([train_df[cat_cols].fillna('missing'), 
                        test_df[cat_cols].fillna('missing')])
    dummies = pd.get_dummies(all_cat, drop_first=True)
    
    train_cat = dummies.iloc[:len(train_df)].fillna(0)
    test_cat = dummies.iloc[len(train_df):].fillna(0)
    return train_cat, test_cat

def get_decision_engine(state: str, intensity: float, stress: float, 
                       energy: float, timeofday: str) -> Tuple[str, str]:
    """
    Decision Engine - What + When
    Input: predictions + context → Action + Timing
    """
    if state in ['calm', 'neutral'] and intensity <= 2:
        return 'deep_work', 'now'
    elif state == 'overwhelmed' or stress > 3:
        return 'box_breathing', 'now'
    elif energy < 2:
        if timeofday in ['night', 'evening']:
            return 'rest', 'tonight'
        return 'movement', 'within_15_min'
    elif intensity >= 4:
        return 'journaling', 'later_today'
    else:
        return 'grounding', 'now'

def uncertainty_flag(confidence: float) -> int:
    """Simple uncertainty thresholding"""
    return 1 if confidence < 0.6 else 0

def save_production_assets(state_model, int_model, tfidf, path_prefix: str = 'models/'):
    """Save for mobile deployment"""
    joblib.dump(state_model, f'{path_prefix}state_model.pkl')
    joblib.dump(int_model, f'{path_prefix}intensity_model.pkl')
    joblib.dump(tfidf, f'{path_prefix}tfidf.pkl')
    print("Production assets saved (~350KB)")

def feature_importance_names(num_features: int, cat_columns: list) -> list:
    """Generate feature names for analysis"""
    text_names = [f'text_{i}' for i in range(num_features)]
    num_names = ['duration_min', 'sleep_hours', 'energy_level', 'stress_level']
    return text_names + num_names + cat_columns
