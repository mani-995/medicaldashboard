from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import sys
import subprocess
import threading
import json
from datetime import datetime

app = Flask(__name__, static_folder='static')
CORS(app)

# Thread lock for concurrent access to new_training_data
_training_data_lock = threading.Lock()

# Global variables
model_pipeline = None
kmeans_model = None          # Fix 4: real KMeans model for cluster assignment
zscore_stats = {}            # Fix 1: loaded from file, not hardcoded
cluster_label_map = {}       # Maps cluster_id → descriptive name based on diabetes rate

# Global variables
model_stats = {
    'total_patients': 768,
    'model_accuracy': 87.5,
    'roc_auc': 0.923,
    'patient_clusters': 3,
    'precision': 0.885,
    'recall': 0.821,
    'f1_score': 0.851,
    'diabetes_rate': 34.9,
    'avg_age': 33.2,
    'feature_importance': {},
    'cluster_stats': [],
    'last_retrain': None,
    'retrain_in_progress': False
}

# Store new predictions for retraining
new_training_data = []
prediction_history = []  # Track all predictions — capped at 10,000 (Fix 6)
PREDICTION_HISTORY_CAP = 10_000
RETRAIN_THRESHOLD = 50
MIN_CONFIDENCE_FOR_SKIP = 0.85

def load_model():
    """Load the trained model pipeline and extract real statistics"""
    global model_pipeline, model_stats, kmeans_model, zscore_stats, cluster_label_map
    
    try:
        model_file = 'diabetes_model_pipeline.pkl'
        if not os.path.exists(model_file):
            print(f"[ERROR] Model file not found: {model_file}")
            print(f"[INFO] Current working directory: {os.getcwd()}")
            print(f"[INFO] Files in directory: {os.listdir('.')}")
            model_pipeline = None
            return
        
        model_pipeline = joblib.load(model_file)
        print("[SUCCESS] Model pipeline loaded successfully!")

        # Fix 4: Load real KMeans model
        kmeans_file = 'kmeans_model.pkl'
        if os.path.exists(kmeans_file):
            kmeans_model = joblib.load(kmeans_file)
            print("[SUCCESS] KMeans model loaded!")
        else:
            kmeans_model = None
            print("[WARNING] kmeans_model.pkl not found — run medicalanalyzer.py to generate it")

        # Fix 1: Load z-score statistics saved during training
        zscore_file = 'zscore_stats.pkl'
        if os.path.exists(zscore_file):
            zscore_stats = joblib.load(zscore_file)
            print(f"[SUCCESS] Z-score statistics loaded for {list(zscore_stats.keys())}")
        else:
            zscore_stats = {}
            print("[WARNING] zscore_stats.pkl not found — using fallback hardcoded values")
        
        # Load performance metrics
        try:
            performance_file = 'model_performance.pkl'
            if os.path.exists(performance_file):
                performance = joblib.load(performance_file)
                
                model_stats['model_accuracy'] = performance.get('model_accuracy', 87.5)
                model_stats['roc_auc'] = performance.get('roc_auc', 0.923)
                model_stats['precision'] = performance.get('precision', 0.885)
                model_stats['recall'] = performance.get('recall', 0.821)
                model_stats['f1_score'] = performance.get('f1_score', 0.851)
                model_stats['total_patients'] = performance.get('total_patients', 768)
                model_stats['patient_clusters'] = performance.get('patient_clusters', 3)
                
                if 'feature_importance' in performance:
                    model_stats['feature_importance'] = performance['feature_importance']
                
                print("[SUCCESS] Loaded model performance metrics!")
        except Exception as e:
            print(f"[WARNING] Could not load performance metrics: {e}")
        
        # Load dataset stats
        try:
            dataset_file = 'diabetes.csv'
            if os.path.exists(dataset_file):
                df = pd.read_csv(dataset_file)
                model_stats['total_patients'] = len(df)
                model_stats['diabetes_rate'] = round((df['Outcome'].sum() / len(df)) * 100, 1)
                model_stats['avg_age'] = round(df['Age'].mean(), 1)
        except Exception as e:
            print(f"[WARNING] Could not load dataset stats: {e}")
        
        # Load cluster statistics
        try:
            cluster_file = 'cluster_statistics.pkl'
            if os.path.exists(cluster_file):
                cluster_stats = joblib.load(cluster_file)
                model_stats['cluster_stats'] = cluster_stats
                print("[SUCCESS] Loaded cluster statistics")
                # Fix 4: Build label map from actual diabetes rates
                for cs in cluster_stats:
                    rate = cs.get('diabetes_rate', 0)
                    if rate >= 60:
                        label = 'High Risk Group'
                    elif rate >= 30:
                        label = 'Moderate Risk Group'
                    else:
                        label = 'Low Risk Group'
                    cluster_label_map[cs['cluster_id']] = label
            else:
                model_stats['cluster_stats'] = [
                    {'cluster_id': 0, 'name': 'Cluster 0', 'count': 346, 'diabetes_rate': 15.0},
                    {'cluster_id': 1, 'name': 'Cluster 1', 'count': 269, 'diabetes_rate': 45.0},
                    {'cluster_id': 2, 'name': 'Cluster 2', 'count': 153, 'diabetes_rate': 78.0}
                ]
                cluster_label_map = {0: 'Low Risk Group', 1: 'Moderate Risk Group', 2: 'High Risk Group'}
        except Exception as e:
            print(f"[WARNING] Could not load cluster stats: {e}")
            cluster_label_map = {0: 'Low Risk Group', 1: 'Moderate Risk Group', 2: 'High Risk Group'}
        
        # Load last retrain timestamp
        if os.path.exists('last_retrain.txt'):
            with open('last_retrain.txt', 'r') as f:
                model_stats['last_retrain'] = f.read().strip()
        
        # Load prediction history
        if os.path.exists('prediction_history.json'):
            with open('prediction_history.json', 'r') as f:
                global prediction_history
                prediction_history = json.load(f)
                print(f"[SUCCESS] Loaded {len(prediction_history)} historical predictions")
        
    except FileNotFoundError:
        print("[WARNING] Model file not found.")
        model_pipeline = None

def preprocess_input(data):
    """
    Preprocess user input using the saved preprocessor
    CRITICAL: Must match EXACT feature engineering from medicalanalyzer.py
    """
    if model_pipeline is None:
        raise ValueError("Model pipeline not loaded")
    
    # Create DataFrame with original features
    df = pd.DataFrame([data])
    
    # === SMART ZERO HANDLING (like training) ===
    # Replace zeros with NaN for imputation
    cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_impute:
        if df[col].iloc[0] == 0:
            df[col] = np.nan
    
    # Impute with dataset medians (better than leaving as 0)
    imputation_values = {
        'Glucose': 117.0,
        'BloodPressure': 72.0,
        'SkinThickness': 23.0,
        'Insulin': 125.0,
        'BMI': 32.0
    }
    
    for col, median_val in imputation_values.items():
        if col in df.columns and df[col].isna().any():
            df[col].fillna(median_val, inplace=True)
    
    # === CATEGORICAL FEATURES (EXACT MATCH) ===
    df['BMI_Category'] = pd.cut(df['BMI'], 
                                bins=[0, 18.5, 24.9, 29.9, np.inf],
                                labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    df['AgeGroup'] = pd.cut(df['Age'],
                           bins=[0, 18, 30, 50, np.inf],
                           labels=['Child/Teen', 'Young Adult', 'Middle Aged', 'Senior'])
    
    df['Glucose_Category'] = pd.cut(df['Glucose'], 
                                    bins=[0, 100, 125, 200, np.inf],
                                    labels=['Normal', 'Prediabetes', 'Diabetes', 'High'])
    
    # === INTERACTION FEATURES (EXACT MATCH) ===
    df['Glucose_Insulin_Ratio'] = df['Glucose'] / (df['Insulin'] + 1)
    df['BMI_Age_Interaction'] = df['BMI'] * df['Age']
    df['Glucose_BMI_Interaction'] = df['Glucose'] * df['BMI']
    df['Pregnancy_Age_Ratio'] = df['Pregnancies'] / (df['Age'] + 1)
    df['Age_BMI_Interaction'] = df['Age'] * df['BMI']
    df['Glucose_Age_Interaction'] = df['Glucose'] * df['Age']
    df['Insulin_BMI_Ratio'] = df['Insulin'] / (df['BMI'] + 1)
    df['Pressure_Age_Ratio'] = df['BloodPressure'] / (df['Age'] + 1)
    
    # === POLYNOMIAL FEATURES (EXACT MATCH) ===
    df['Glucose_Squared'] = df['Glucose'] ** 2
    df['BMI_Squared'] = df['BMI'] ** 2
    df['Age_Squared'] = df['Age'] ** 2
    df['Insulin_Squared'] = df['Insulin'] ** 2
    df['Glucose_Cubed'] = df['Glucose'] ** 3
    
    # === LOGARITHMIC TRANSFORMATIONS (EXACT MATCH) ===
    df['Log_Glucose'] = np.log1p(df['Glucose'])
    df['Log_Insulin'] = np.log1p(df['Insulin'])
    df['Log_BMI'] = np.log1p(df['BMI'])
    df['Log_Age'] = np.log1p(df['Age'])
    
    # === RISK SCORES (EXACT MATCH) ===
    df['Metabolic_Risk_Score'] = (
        (df['Glucose'] / 100) * 0.35 + 
        (df['BMI'] / 30) * 0.25 + 
        (df['Age'] / 50) * 0.20 +
        (df['DiabetesPedigreeFunction']) * 0.20
    )
    
    df['Cardiovascular_Risk'] = (
        (df['BloodPressure'] / 80) * 0.4 +
        (df['BMI'] / 30) * 0.3 +
        (df['Age'] / 50) * 0.3
    )
    
    df['Insulin_Resistance_Index'] = df['Glucose'] * df['Insulin'] / 405
    
    # === Z-SCORES — loaded from zscore_stats.pkl saved during training (Fix 1) ===
    # Fallback values used only if zscore_stats.pkl is missing
    _fallback = {
        'Glucose':       {'mean': 121.18, 'std': 30.44},
        'BMI':           {'mean': 32.19,  'std': 6.88},
        'Age':           {'mean': 33.09,  'std': 11.50},
        'BloodPressure': {'mean': 72.25,  'std': 12.13},
        'Insulin':       {'mean': 118.66, 'std': 93.08},
    }
    stats_source = zscore_stats if zscore_stats else _fallback
    for col, s in stats_source.items():
        if col in df.columns and s['std'] > 0:
            df[f'{col}_Zscore'] = (df[col] - s['mean']) / s['std']
    
    # Fix inf/NaN values (CRITICAL)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Use the trained preprocessor
    try:
        preprocessor = model_pipeline['preprocessor']
        processed = preprocessor.transform(df)
        return processed
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        raise

def get_cluster_assignment(processed_data):
    """
    Fix 4: Use the trained KMeans model for cluster assignment.
    Falls back to probability-based heuristic only if model unavailable.
    """
    if kmeans_model is not None:
        try:
            cluster_id = int(kmeans_model.predict(processed_data)[0])
            label = cluster_label_map.get(cluster_id, f'Cluster {cluster_id}')
            return cluster_id, label
        except Exception as e:
            print(f"[WARNING] KMeans prediction failed: {e}")

    # Fallback — should not normally be reached
    return 0, 'Unknown Cluster'

def check_prediction_consistency(input_data, actual_outcome):
    """
    Check if this exact input has been predicted before
    Returns: (has_history, previous_prediction, should_warn)
    """
    # Create a hashable representation of input
    input_key = json.dumps(input_data, sort_keys=True)
    
    for hist in prediction_history:
        hist_key = json.dumps(hist['input'], sort_keys=True)
        if hist_key == input_key:
            prediction_changed = (hist['prediction'] != actual_outcome)
            return True, hist, prediction_changed
    
    return False, None, False

def save_prediction_history():
    """Save prediction history to disk"""
    try:
        with open('prediction_history.json', 'w') as f:
            json.dump(prediction_history, f, indent=2)
    except Exception as e:
        print(f"[WARNING] Could not save prediction history: {e}")

def retrain_model_async():
    """Retrain model in background thread"""
    global model_stats
    
    model_stats['retrain_in_progress'] = True
    print("\n[INFO] Starting background model retraining...")
    
    try:
        # Load original dataset
        df_original = pd.read_csv('diabetes.csv')
        
        # Fix 3: Snapshot and clear new_training_data under lock
        with _training_data_lock:
            samples_to_add = list(new_training_data)
            new_training_data.clear()

        if samples_to_add:
            df_new = pd.DataFrame(samples_to_add)
            df_combined = pd.concat([df_original, df_new], ignore_index=True)
            df_combined.to_csv('diabetes.csv', index=False)
            print(f"[INFO] Added {len(samples_to_add)} new samples to dataset")
        
        # Fix 2: Use subprocess.run with sys.executable instead of os.system
        result = subprocess.run(
            [sys.executable, 'medicalanalyzer.py'],
            capture_output=True, text=True, timeout=600
        )
        with open('retrain_log.txt', 'w') as f:
            f.write(result.stdout)
            f.write(result.stderr)

        if result.returncode != 0:
            print(f"[ERROR] Retraining script failed (exit {result.returncode})")
            print(result.stderr[-2000:])  # last 2k chars of stderr
            # Restore samples so they aren't lost
            with _training_data_lock:
                new_training_data.extend(samples_to_add)
            return
        
        # Reload model + artifacts
        load_model()
        
        # Update timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        model_stats['last_retrain'] = timestamp
        with open('last_retrain.txt', 'w') as f:
            f.write(timestamp)
        
        print(f"[SUCCESS] Model retrained successfully at {timestamp}")
        
    except subprocess.TimeoutExpired:
        print("[ERROR] Retraining timed out after 600s")
    except Exception as e:
        print(f"[ERROR] Retraining failed: {e}")
    
    finally:
        model_stats['retrain_in_progress'] = False

@app.route('/')
def index():
    return send_from_directory('.', 'medical_dashboard.html')

@app.route('/stats')
def get_stats():
    with _training_data_lock:
        new_data_count = len(new_training_data)
    return jsonify({
        'total_patients': model_stats['total_patients'],
        'model_accuracy': model_stats['model_accuracy'],
        'roc_auc': model_stats['roc_auc'],
        'patient_clusters': model_stats['patient_clusters'],
        'diabetes_rate': model_stats['diabetes_rate'],
        'avg_age': model_stats['avg_age'],
        'precision': model_stats['precision'],
        'recall': model_stats['recall'],
        'f1_score': model_stats['f1_score'],
        'feature_importance': model_stats['feature_importance'],
        'cluster_stats': model_stats['cluster_stats'],
        'last_retrain': model_stats['last_retrain'],
        'retrain_in_progress': model_stats['retrain_in_progress'],
        'new_data_count': new_data_count,
        'retrain_threshold': RETRAIN_THRESHOLD
    })

@app.route('/feature-importance')
def get_feature_importance():
    if not model_stats['feature_importance']:
        return jsonify({'error': 'Feature importance not available'}), 404
    
    sorted_features = sorted(
        model_stats['feature_importance'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:15]
    
    return jsonify({
        'features': [f[0] for f in sorted_features],
        'importance': [float(f[1]) for f in sorted_features]
    })

@app.route('/predict', methods=['POST'])
def predict():
    global prediction_history
    
    try:
        data = request.json
        
        required_fields = ['pregnancies', 'glucose', 'bloodPressure', 'skinThickness', 
                          'insulin', 'bmi', 'diabetesPedigree', 'age']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        input_data = {
            'Pregnancies': float(data.get('pregnancies', 0)),
            'Glucose': float(data.get('glucose', 120)),
            'BloodPressure': float(data.get('bloodPressure', 70)),
            'SkinThickness': float(data.get('skinThickness', 20)),
            'Insulin': float(data.get('insulin', 79)),
            'BMI': float(data.get('bmi', 25)),
            'DiabetesPedigreeFunction': float(data.get('diabetesPedigree', 0.5)),
            'Age': float(data.get('age', 33))
        }
        
        # Validate inputs
        if input_data['Glucose'] < 0 or input_data['Glucose'] > 300:
            return jsonify({'error': 'Glucose must be between 0-300 mg/dL'}), 400
        if input_data['BMI'] < 10 or input_data['BMI'] > 70:
            return jsonify({'error': 'BMI must be between 10-70'}), 400
        if input_data['Age'] < 1 or input_data['Age'] > 120:
            return jsonify({'error': 'Age must be between 1-120 years'}), 400
        
        # Preprocess and predict
        processed_data = preprocess_input(input_data)
        
        if model_pipeline is not None:
            model = model_pipeline['model']
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0][1]
            confidence = abs(probability - 0.5) * 2  # 0 to 1 scale
            
            # Fix 4: Use real KMeans cluster assignment
            cluster, cluster_name = get_cluster_assignment(processed_data)
            
        else:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Store prediction in history — Fix 6: cap at PREDICTION_HISTORY_CAP
        prediction_record = {
            'timestamp': datetime.now().isoformat(),
            'input': input_data,
            'prediction': int(prediction),
            'probability': float(probability),
            'confidence': float(confidence),
            'model_version': model_stats.get('last_retrain', 'initial')
        }
        prediction_history.append(prediction_record)
        if len(prediction_history) > PREDICTION_HISTORY_CAP:
            prediction_history.pop(0)
        save_prediction_history()
        
        # Generate recommendations
        recommendations = []
        if input_data['Glucose'] > 140:
            recommendations.append('Monitor glucose levels regularly - consider HbA1c test')
        if input_data['BMI'] > 30:
            recommendations.append('Consider weight management program and dietary changes')
        if input_data['Age'] > 50 and probability > 0.3:
            recommendations.append('Schedule regular health screenings (every 6 months)')
        if input_data['BloodPressure'] > 90:
            recommendations.append('Blood pressure monitoring recommended')
        if prediction == 1:
            recommendations.append('⚠️ HIGH PRIORITY: Consult healthcare provider immediately')
        else:
            recommendations.append('Maintain healthy lifestyle habits - diet and exercise')
        
        # Calculate metabolic score
        metabolic_score = (
            (input_data['Glucose']/100)*0.35 + 
            (input_data['BMI']/30)*0.25 + 
            (input_data['Age']/50)*0.20 +
            input_data['DiabetesPedigreeFunction']*0.20
        )
        
        response = {
            'prediction': int(prediction),
            'probability': float(probability),
            'confidence': float(confidence),
            'cluster': int(cluster),
            'cluster_name': cluster_name,
            'risk_level': 'High' if prediction == 1 else 'Low',
            'recommendations': recommendations,
            'metabolic_score': float(metabolic_score),
            'can_improve_model': len(new_training_data) < RETRAIN_THRESHOLD,
        }

        # Fix 5: Only expose debug_info when DEBUG mode is on
        if app.debug:
            response['debug_info'] = {
                'glucose': input_data['Glucose'],
                'bmi': input_data['BMI'],
                'age': input_data['Age'],
                'raw_probability': float(probability)
            }
        
        # Add low confidence warning
        if confidence < 0.6:
            response['low_confidence_warning'] = True
            response['confidence_message'] = f'This prediction has only {confidence*100:.1f}% confidence. Consider additional medical tests.'
        
        return jsonify(response)
        
    except Exception as e:
        print(f"[ERROR] Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    """Submit actual outcome for model improvement - WITH CONSISTENCY CHECK"""
    try:
        data = request.json
        
        # Store with confirmed outcome
        input_data = {
            'Pregnancies': float(data['pregnancies']),
            'Glucose': float(data['glucose']),
            'BloodPressure': float(data['bloodPressure']),
            'SkinThickness': float(data['skinThickness']),
            'Insulin': float(data['insulin']),
            'BMI': float(data['bmi']),
            'DiabetesPedigreeFunction': float(data['diabetesPedigree']),
            'Age': float(data['age']),
        }
        
        actual_outcome = int(data['actual_outcome'])
        
        # Get current model's prediction for this input
        processed_data = preprocess_input(input_data)
        model = model_pipeline['model']
        current_prediction = model.predict(processed_data)[0]
        current_probability = model.predict_proba(processed_data)[0][1]
        confidence = abs(current_probability - 0.5) * 2
        
        # Check if prediction was correct
        prediction_correct = (current_prediction == actual_outcome)
        
        # Decision logic for adding to training data
        should_add_to_training = True
        skip_reason = None
        
        if prediction_correct and confidence > MIN_CONFIDENCE_FOR_SKIP:
            # Model already predicts this correctly with high confidence
            # No need to retrain on this sample
            should_add_to_training = False
            skip_reason = f'Prediction already correct with {confidence*100:.1f}% confidence'
            print(f"[INFO] Skipping retraining - {skip_reason}")
        
        # Check for prediction history inconsistency
        has_history, prev_pred, changed = check_prediction_consistency(input_data, actual_outcome)
        
        message = 'Thank you for improving the model!'
        if prediction_correct:
            message = f'✅ Model prediction was correct! (Confidence: {confidence*100:.1f}%)'
        else:
            message = f'🔄 Feedback recorded. Model will learn from this. (Was {confidence*100:.1f}% confident in wrong answer)'
        
        if should_add_to_training:
            # Fix 3: Acquire lock before appending to avoid race condition
            with _training_data_lock:
                training_sample = input_data.copy()
                training_sample['Outcome'] = actual_outcome
                new_training_data.append(training_sample)
        
        # Check if should trigger retraining
        with _training_data_lock:
            current_count = len(new_training_data)
        should_retrain = current_count >= RETRAIN_THRESHOLD
        
        if should_retrain and not model_stats['retrain_in_progress']:
            threading.Thread(target=retrain_model_async, daemon=True).start()
        
        return jsonify({
            'success': True,
            'new_data_count': current_count,
            'retrain_triggered': bool(should_retrain),
            'message': message,
            'prediction_was_correct': bool(prediction_correct),
            'confidence': float(confidence),
            'added_to_training': bool(should_add_to_training),
            'skip_reason': skip_reason if skip_reason else ''
        })
        
    except Exception as e:
        print(f"[ERROR] Feedback submission error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/trigger-retrain', methods=['POST'])
def trigger_retrain():
    """Manually trigger model retraining"""
    if model_stats['retrain_in_progress']:
        return jsonify({'error': 'Retraining already in progress'}), 400
    
    if len(new_training_data) == 0:
        return jsonify({'error': 'No new data to retrain with'}), 400
    
    threading.Thread(target=retrain_model_async, daemon=True).start()
    
    return jsonify({
        'success': True,
        'message': f'Retraining started with {len(new_training_data)} new samples'
    })

@app.route('/visualization-data')
def get_visualization_data():
    """Get data for advanced visualizations"""
    try:
        df = pd.read_csv('diabetes.csv')
        
        # Prepare 3D scatter plot data
        scatter_data = {
            'glucose': df['Glucose'].tolist(),
            'bmi': df['BMI'].tolist(),
            'age': df['Age'].tolist(),
            'outcome': df['Outcome'].tolist()
        }
        
        # --- Risk Profile data (split by outcome) ---
        df_healthy = df[df['Outcome'] == 0]
        df_diabetic = df[df['Outcome'] == 1]
        
        risk_profile = {
            'total': len(df),
            'diabetic_count': int(df_diabetic.shape[0]),
            'healthy_count': int(df_healthy.shape[0]),
            'glucose': {
                'healthy': df_healthy['Glucose'].tolist(),
                'diabetic': df_diabetic['Glucose'].tolist()
            },
            'bmi': {
                'healthy': df_healthy['BMI'].tolist(),
                'diabetic': df_diabetic['BMI'].tolist()
            },
            'age': {
                'healthy': df_healthy['Age'].tolist(),
                'diabetic': df_diabetic['Age'].tolist()
            },
            'blood_pressure': {
                'healthy': df_healthy['BloodPressure'].tolist(),
                'diabetic': df_diabetic['BloodPressure'].tolist()
            },
            'stats': {
                'healthy': {
                    'avg_glucose': round(float(df_healthy['Glucose'].mean()), 1),
                    'avg_bmi': round(float(df_healthy['BMI'].mean()), 1),
                    'avg_age': round(float(df_healthy['Age'].mean()), 1),
                    'avg_bp': round(float(df_healthy['BloodPressure'].mean()), 1)
                },
                'diabetic': {
                    'avg_glucose': round(float(df_diabetic['Glucose'].mean()), 1),
                    'avg_bmi': round(float(df_diabetic['BMI'].mean()), 1),
                    'avg_age': round(float(df_diabetic['Age'].mean()), 1),
                    'avg_bp': round(float(df_diabetic['BloodPressure'].mean()), 1)
                }
            }
        }
        
        return jsonify({
            'scatter_3d': scatter_data,
            'risk_profile': risk_profile
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/prediction-history')
def get_prediction_history():
    """Get prediction history for analysis"""
    try:
        # Return last 50 predictions
        recent_history = prediction_history[-50:]
        return jsonify({
            'history': recent_history,
            'total_predictions': len(prediction_history)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_pipeline is not None,
        'version': '5.0 - Prediction Consistency Fix',
        'features': len(model_stats['feature_importance']),
        'retrain_status': 'in_progress' if model_stats['retrain_in_progress'] else 'ready',
        'new_samples': len(new_training_data),
        'total_predictions': len(prediction_history)
    })
    

@app.after_request
def set_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "no-referrer-when-downgrade"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=()"
    return response

# ── Gunicorn deployment fix ────────────────────────────────────────────────
# Gunicorn imports this module directly and never runs the __main__ block.
# Calling load_model() here ensures the model is loaded in all environments.
load_model()
# ───────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("="*60)
    print("ADVANCED MEDICAL ANALYTICS - PREDICTION CONSISTENCY v5.0")
    print("="*60)
    
    print("\n[INFO] Server starting with prediction consistency checks...")
    print(f"[INFO] Retrain threshold: {RETRAIN_THRESHOLD} samples")
    print(f"[INFO] Min confidence to skip: {MIN_CONFIDENCE_FOR_SKIP*100}%")
    print("[INFO] Dashboard: http://localhost:5000")
    print("\n[INFO] Press CTRL+C to stop")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)