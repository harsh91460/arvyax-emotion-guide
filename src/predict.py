import pandas as pd
import numpy as np
import joblib
from .preprocess import load_data, create_features
from .decision import get_decision

def make_predictions():
    """Generate predictions for test set"""
    print("Loading models...")
    state_model = joblib.load('models/state_model.joblib')
    int_model = joblib.load('models/intensity_model.joblib')
    preprocessor = joblib.load('models/preprocessor.joblib')
    
    print("Loading test data...")
    _, test = load_data()
    
    print("Creating test features...")
    X_test, _, _ = create_features(test, preprocessor['tfidf'], preprocessor['cat_columns'])
    
    # Predictions
    test['predicted_state'] = state_model.predict(X_test)
    test['predicted_intensity'] = np.clip(int_model.predict(X_test), 1, 5)
    
    # Confidence
    probs = state_model.predict_proba(X_test)
    test['confidence'] = np.max(probs, axis=1)
    test['uncertain_flag'] = (test['confidence'] < 0.6).astype(int)
    
    # Decisions
    test[['what_to_do', 'when_to_do']] = test.apply(
        lambda row: pd.Series(get_decision(
            row['predicted_state'], row['predicted_intensity'],
            row['stresslevel'], row['energylevel'], row['timeofday']
        )), axis=1
    )
    
    # Save
    predictions = test[['id', 'predicted_state', 'predicted_intensity', 
                       'confidence', 'uncertain_flag', 'what_to_do', 'when_to_do']]
    predictions.to_csv('predictions.csv', index=False)
    print("Predictions saved to predictions.csv")
    return predictions
