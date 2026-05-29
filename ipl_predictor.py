# =============================================================================
# 🏏 IPL MATCH WINNER PREDICTION MODEL
# =============================================================================

import pandas as pd         
import numpy as np            
import matplotlib.pyplot as plt  
import seaborn as sns         
import warnings
warnings.filterwarnings('ignore') 

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder         
from sklearn.ensemble import RandomForestClassifier   
from sklearn.tree import DecisionTreeClassifier        
from sklearn.metrics import (
    accuracy_score,         
    confusion_matrix,        
    classification_report 
)

import os
import joblib                 
print("✅ All libraries imported successfully!")
print("=" * 60)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'matches.csv')
df = pd.read_csv(DATA_PATH)
print(f"📊 Dataset loaded: {df.shape[0]} matches, {df.shape[1]} columns")
print(f"📅 Seasons covered: {df['season'].min()} to {df['season'].max()}")
print(f"\n📋 Columns available:\n{list(df.columns)}")
print("=" * 60)
print("\n🔍 FIRST 5 ROWS:")
print(df[['season', 'team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'winner']].head())
print(f"\n❓ MISSING VALUES:")
print(df.isnull().sum()[df.isnull().sum() > 0])  # WHY: Only show columns that actually have missing values.
print("=" * 60)
print("\n🧹 CLEANING DATA...")

before = len(df)
df = df.dropna(subset=['winner'])
after = len(df)
print(f"  ✓ Removed {before - after} rows with no winner (e.g., abandoned matches)")

team_name_map = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Rising Pune Supergiants': 'Rising Pune Supergiant',
    'Kings XI Punjab': 'Punjab Kings',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru'
}

for col in ['team1', 'team2', 'toss_winner', 'winner']:
    df[col] = df[col].replace(team_name_map)
print(f"  ✓ Standardized team names (e.g., Delhi Daredevils → Delhi Capitals)")

df['city'] = df['city'].fillna(df['venue'])
print(f"  ✓ Filled missing city values using venue")
features_to_use = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision']
target = 'winner'

df_model = df[features_to_use + [target]].copy()
print(f"  ✓ Selected {len(features_to_use)} features: {features_to_use}")
print(f"  ✓ Target variable: '{target}'")
print(f"  ✓ Final dataset shape: {df_model.shape}")
print("=" * 60)
print("\n🔢 ENCODING TEXT COLUMNS TO NUMBERS...")

label_encoders = {}  
for col in df_model.columns:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le

print(f"  ✓ Encoded columns: {list(df_model.columns)}")
print(f"\n  Example — Team names encoded as:")
team_labels = label_encoders['team1']
for i, name in enumerate(team_labels.classes_[:5]):
    print(f"    {name} → {i}")
print("=" * 60)

X = df_model[features_to_use]  
y = df_model[target]             

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,    
    random_state=42    
)

print(f"\n✂️  TRAIN/TEST SPLIT:")
print(f"  🏋️  Training set : {X_train.shape[0]} matches (80%)")
print(f"  🧪 Testing set  : {X_test.shape[0]} matches (20%)")
print("=" * 60)

print("\n🤖 TRAINING MODELS...")

dt_model = DecisionTreeClassifier(
    max_depth=10,      
    random_state=42
)
dt_model.fit(X_train, y_train)   
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred) * 100
print(f"  🌳 Decision Tree Accuracy : {dt_accuracy:.2f}%")
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=15,     
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred) * 100
print(f"  🌲 Random Forest Accuracy : {rf_accuracy:.2f}%")
print("=" * 60)

best_model = rf_model if rf_accuracy >= dt_accuracy else dt_model
best_pred = rf_pred if rf_accuracy >= dt_accuracy else dt_pred
best_name = "Random Forest" if rf_accuracy >= dt_accuracy else "Decision Tree"

print(f"\n📈 BEST MODEL: {best_name} ({max(rf_accuracy, dt_accuracy):.2f}% accuracy)")
print(f"\n📋 CLASSIFICATION REPORT (Top teams):")
print(classification_report(y_test, best_pred, 
      target_names=label_encoders['winner'].classes_,
      zero_division=0))

print("\n🔑 FEATURE IMPORTANCE (What matters most for winning?):")
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': features_to_use,
    'Importance': importances
}).sort_values('Importance', ascending=False)

for _, row in feature_importance_df.iterrows():
    bar = "█" * int(row['Importance'] * 50)
    print(f"  {row['Feature']:<15} {bar} {row['Importance']:.4f}")
print("=" * 60)

models_dir = os.path.join(BASE_DIR, 'models')
os.makedirs(models_dir, exist_ok=True)

joblib.dump(rf_model, os.path.join(models_dir, 'random_forest.pkl'))
joblib.dump(dt_model, os.path.join(models_dir, 'decision_tree.pkl'))
joblib.dump(label_encoders, os.path.join(models_dir, 'label_encoders.pkl'))

print(f"\n💾 MODELS SAVED to /models/ folder")
print("=" * 60)

def predict_winner(team1, team2, venue, toss_winner, toss_decision):
    """
    Predict the winner of an IPL match.
    
    WHY THIS FUNCTION EXISTS:
        The model needs numbers, not text. This function handles the
        encoding automatically so the user can just pass team names.
    
    Parameters:
        team1 (str)         : Name of team 1
        team2 (str)         : Name of team 2
        venue (str)         : Stadium name
        toss_winner (str)   : Which team won the toss
        toss_decision (str) : 'bat' or 'field'
    
    Returns:
        str: Predicted winning team name
    """
    encoders = joblib.load(os.path.join(models_dir, 'label_encoders.pkl'))
    model = joblib.load(os.path.join(models_dir, 'random_forest.pkl'))
    
    try:
        input_data = pd.DataFrame([[
            encoders['team1'].transform([team1])[0],
            encoders['team2'].transform([team2])[0],
            encoders['venue'].transform([venue])[0],
            encoders['toss_winner'].transform([toss_winner])[0],
            encoders['toss_decision'].transform([toss_decision])[0]
        ]], columns=features_to_use)

        prediction_encoded = model.predict(input_data)[0]
        prediction = encoders['winner'].inverse_transform([prediction_encoded])[0]
        
        proba = model.predict_proba(input_data)[0]
        confidence = max(proba) * 100
        
        return prediction, confidence
    
    except ValueError as e:
        return f"Error: {e} — Check team/venue names are valid.", 0

print("\n🎯 SAMPLE PREDICTIONS:")
print("-" * 60)

test_matches = [
    ("Mumbai Indians", "Punjab Kings", "Wankhede Stadium", "Mumbai Indians", "bat"),
    ("Chennai Super Kings", "Kolkata Knight Riders", "MA Chidambaram Stadium", "Chennai Super Kings", "field"),
    ("Royal Challengers Bengaluru", "Delhi Capitals", "M Chinnaswamy Stadium", "Delhi Capitals", "bat"),
]

for t1, t2, venue, toss_w, toss_d in test_matches:
    winner, confidence = predict_winner(t1, t2, venue, toss_w, toss_d)
    print(f"  🏏 {t1} vs {t2}")
    print(f"     Toss: {toss_w} chose to {toss_d}")
    print(f"     🏆 Predicted Winner : {winner} (Confidence: {confidence:.1f}%)")
    print()

print("=" * 60)
print("✅ IPL Prediction Model complete! Run dashboard.py for visualizations.")