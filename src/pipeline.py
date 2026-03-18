import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print(" Loading data...")
train = pd.read_excel('train.xlsx')
test = pd.read_excel('test.xlsx')

print(" Columns:", train.columns.tolist())
print("Train shape:", train.shape)

# YOUR EXACT COLUMNS (snake_case)
num_cols = ['duration_min', 'sleep_hours', 'energy_level', 'stress_level']
cat_cols = ['ambience_type', 'time_of_day', 'previous_day_mood', 'face_emotion_hint', 'reflection_quality']

# Clean data
train_clean = train.dropna(subset=['emotional_state', 'intensity']).copy()
print(f"Clean data: {train_clean.shape}")

def create_features(df_train, df_test=None):
    tfidf = TfidfVectorizer(max_features=50, stop_words='english')
    train_text = tfidf.fit_transform(df_train['journal_text'].fillna('missing').astype(str)).toarray()
    
    # Fill numerical NaN
    for col in num_cols:
        median = df_train[col].median()
        df_train[col] = df_train[col].fillna(median)
        if df_test is not None:
            df_test[col] = df_test[col].fillna(median)
    
    train_num = df_train[num_cols].values
    
    # Categorical
    all_cat = pd.concat([df_train[cat_cols].fillna('missing'), 
                        df_test[cat_cols].fillna('missing') if df_test is not None else pd.DataFrame()],
                       axis=0, ignore_index=True)
    
    cat_dummies = pd.get_dummies(all_cat, drop_first=True)
    train_cat = cat_dummies.iloc[:len(df_train)].fillna(0)
    
    X_train = np.hstack([train_text, train_num, train_cat.values])
    
    if df_test is not None:
        test_text = tfidf.transform(df_test['journal_text'].fillna('missing').astype(str)).toarray()
        test_num = df_test[num_cols].values
        test_cat = cat_dummies.iloc[len(df_train):].fillna(0)
        X_test = np.hstack([test_text, test_num, test_cat.values])
        return X_train, X_test, tfidf, list(cat_dummies.columns)
    
    return X_train, None, tfidf, list(cat_dummies.columns)

print("\n Features...")
X, _, tfidf, cat_list = create_features(train_clean)
print(f"Features: {X.shape}")

# Labels
y_state = train_clean['emotional_state'].astype(str)
y_intensity = train_clean['intensity'].fillna(train_clean['intensity'].median()).values

# Split
X_tr, X_val, y_state_tr, y_state_val, y_int_tr, y_int_val = train_test_split(
    X, y_state, y_intensity, test_size=0.2, random_state=42, stratify=y_state
)

print("\n Training...")
state_model = RandomForestClassifier(n_estimators=100, random_state=42)
state_model.fit(X_tr, y_state_tr)
state_pred = state_model.predict(X_val)
print(f" State Accuracy: {accuracy_score(y_state_val, state_pred):.3f}")

int_model = LinearRegression()
int_model.fit(X_tr, y_int_tr)
int_pred = np.clip(int_model.predict(X_val), 1, 5)
print(f" Intensity MSE: {mean_squared_error(y_int_val, int_pred):.3f}")

# Test predictions
print("\n Test...")
X_train_feat, X_test_feat, _, _ = create_features(train_clean, test)

test['predicted_state'] = state_model.predict(X_test_feat)
test['predicted_intensity'] = np.clip(int_model.predict(X_test_feat), 1, 5)

probs = state_model.predict_proba(X_test_feat)
test['confidence'] = np.max(probs, axis=1)
test['uncertain_flag'] = (test['confidence'] < 0.6).astype(int)

# FIXED DECISION FUNCTION - Returns pd.Series!
def get_decision(row):
    state = row['predicted_state']
    intensity = row['predicted_intensity']
    stress = row.get('stress_level', 3)
    energy = row.get('energy_level', 3)
    timeofday = row.get('time_of_day', 'morning')
    
    if state in ['calm', 'neutral'] and intensity <= 2:
        what, when = 'deep_work', 'now'
    elif state == 'overwhelmed' or stress > 3:
        what, when = 'box_breathing', 'now'
    elif energy < 2:
        if timeofday in ['night', 'evening']:
            what, when = 'rest', 'tonight'
        else:
            what, when = 'movement', 'within_15_min'
    elif intensity >= 4:
        what, when = 'journaling', 'later_today'
    else:
        what, when = 'grounding', 'now'
    
    return pd.Series([what, when])

print(" Decisions...")
test[['what_to_do', 'when_to_do']] = test.apply(get_decision, axis=1)

# SAVE
predictions = test[['id', 'predicted_state', 'predicted_intensity', 
                   'confidence', 'uncertain_flag', 'what_to_do', 'when_to_do']]
predictions.to_csv('predictions.csv', index=False)

print("\n  DONE!")
print("predictions.csv SAVED!")
print(predictions.head(10))
print("\n Stats:")
print(predictions['predicted_state'].value_counts())
print(predictions['what_to_do'].value_counts())
