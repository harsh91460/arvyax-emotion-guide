import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib

def load_data(data_path="data"):
    """Load train and test data"""
    train = pd.read_excel(f"{data_path}/train.xlsx")
    test = pd.read_excel(f"{data_path}/test.xlsx")
    return train, test

def create_features(df, tfidf=None, cat_columns=None):
    """Create text + metadata features"""
    # Text TF-IDF
    if tfidf is None:
        tfidf = TfidfVectorizer(max_features=50, stop_words='english')
        text_features = tfidf.fit_transform(df['journaltext'].fillna('')).toarray()
    else:
        text_features = tfidf.transform(df['journaltext'].fillna('')).toarray()
    
    # Numerical features
    num_cols = ['durationmin', 'sleephours', 'energylevel', 'stresslevel']
    num_features = df[num_cols].fillna(df[num_cols].median())
    
    # Categorical features
    cat_cols = ['ambiencetype', 'timeofday', 'previousdaymood', 'faceemotionhint', 'reflectionquality']
    cat_features = pd.get_dummies(df[cat_cols].fillna('missing'), drop_first=True)
    
    # Align test categories with train
    if cat_columns is not None:
        cat_features = cat_features.reindex(columns=cat_columns, fill_value=0)
    
    features = np.hstack([text_features, num_features, cat_features])
    return features, tfidf, cat_features.columns.tolist()

def save_preprocessor(tfidf, cat_columns, path="models/preprocessor.joblib"):
    joblib.dump({'tfidf': tfidf, 'cat_columns': cat_columns}, path)
