"""
Menopause Symptom Prediction and Management Tool
Complete ML Pipeline with Synthetic Dataset Generation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, f1_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# PART 1: SYNTHETIC DATASET GENERATION
# ============================================================================

class MenopauseDataGenerator:
    """Generate medically accurate synthetic menopause data"""
    
    def __init__(self, n_samples=5000):
        self.n_samples = n_samples
        
    def generate_dataset(self):
        """Generate comprehensive synthetic dataset with biomarkers"""
        data = []
        
        for i in range(self.n_samples):
            # Age distribution (35-60 years, peak around 45-52)
            age = np.random.choice(
                np.arange(35, 61),
                p=self._age_distribution()
            )
            
            # Determine menopause stage based on age
            if age < 42:
                stage = 'pre_menopause'
                months_to_menopause = np.random.randint(60, 180)
            elif age < 47:
                stage = 'early_perimenopause'
                months_to_menopause = np.random.randint(24, 84)
            elif age < 52:
                stage = 'late_perimenopause'
                months_to_menopause = np.random.randint(1, 36)
            else:
                stage = 'post_menopause'
                months_to_menopause = 0
            
            # BMI distribution (18-40, mean ~26)
            bmi = np.clip(np.random.normal(26, 5), 18, 40)
            
            # Hormone levels based on stage
            if stage == 'pre_menopause':
                fsh = np.clip(np.random.normal(8, 3), 3, 15)
                estradiol = np.clip(np.random.normal(80, 30), 40, 200)
                lh = np.clip(np.random.normal(10, 4), 5, 20)
            elif stage == 'early_perimenopause':
                fsh = np.clip(np.random.normal(18, 6), 10, 35)
                estradiol = np.clip(np.random.normal(50, 25), 20, 100)
                lh = np.clip(np.random.normal(15, 5), 8, 30)
            elif stage == 'late_perimenopause':
                fsh = np.clip(np.random.normal(35, 10), 20, 60)
                estradiol = np.clip(np.random.normal(25, 15), 5, 50)
                lh = np.clip(np.random.normal(25, 8), 15, 45)
            else:  # post_menopause
                fsh = np.clip(np.random.normal(60, 20), 30, 120)
                estradiol = np.clip(np.random.normal(15, 8), 5, 30)
                lh = np.clip(np.random.normal(35, 10), 20, 60)
            
            # Lifestyle factors
            sleep_quality = np.clip(np.random.normal(6, 2), 1, 10)
            stress_level = np.clip(np.random.normal(5, 2.5), 1, 10)
            exercise_minutes = np.clip(np.random.normal(150, 80), 0, 420)
            
            # Diet quality score (1-10)
            diet_quality = np.clip(np.random.normal(6, 2), 1, 10)
            
            # Smoking status
            smoking_status = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])  # 0=never, 1=former, 2=current
            
            # Medical history
            has_diabetes = np.random.choice([0, 1], p=[0.9, 0.1])
            has_hypertension = np.random.choice([0, 1], p=[0.8, 0.2])
            has_thyroid_disorder = np.random.choice([0, 1], p=[0.85, 0.15])
            
            # Menstrual history
            if stage in ['pre_menopause', 'early_perimenopause']:
                last_period_months = np.random.randint(0, 4)
                cycle_irregularity = np.random.choice([0, 1], p=[0.7, 0.3])
            elif stage == 'late_perimenopause':
                last_period_months = np.random.randint(2, 12)
                cycle_irregularity = 1
            else:
                last_period_months = np.random.randint(12, 60)
                cycle_irregularity = 1
            
            # Symptom severity (influenced by hormones and lifestyle)
            symptom_base = (fsh / 60) * 0.4 + (1 - estradiol / 100) * 0.4 + (stress_level / 10) * 0.2
            
            hot_flash_freq = np.clip(
                np.random.poisson(symptom_base * 15) * (1 if stage != 'pre_menopause' else 0.2),
                0, 30
            )
            
            night_sweats_freq = np.clip(
                np.random.poisson(symptom_base * 10) * (1 if stage != 'pre_menopause' else 0.1),
                0, 20
            )
            
            mood_swings_severity = np.clip(
                np.random.normal(symptom_base * 8, 2),
                0, 10
            )
            
            anxiety_level = np.clip(
                stress_level * 0.6 + symptom_base * 4,
                0, 10
            )
            
            depression_score = np.clip(
                np.random.normal(symptom_base * 6 + stress_level * 0.3, 1.5),
                0, 10
            )
            
            cognitive_issues = np.clip(
                np.random.normal(symptom_base * 7, 2),
                0, 10
            )
            
            fatigue_level = np.clip(
                (10 - sleep_quality) * 0.5 + symptom_base * 5,
                0, 10
            )
            
            libido_change = np.clip(
                np.random.normal(-symptom_base * 6, 2),
                -10, 0
            )
            
            vaginal_dryness = np.clip(
                np.random.normal(symptom_base * 8, 2) * (1 if stage != 'pre_menopause' else 0.1),
                0, 10
            )
            
            joint_pain = np.clip(
                np.random.normal(symptom_base * 6 + age * 0.1, 2),
                0, 10
            )
            
            # Weight change (kg) - many women experience weight gain
            weight_change = np.clip(
                np.random.normal(symptom_base * 10 + (bmi - 25) * 0.2, 3),
                -5, 15
            )
            
            # Heart palpitations
            heart_palpitations_freq = np.clip(
                np.random.poisson(symptom_base * 8) * (1 if stage != 'pre_menopause' else 0.1),
                0, 15
            )
            
            # Calculate overall symptom severity score
            overall_severity = np.mean([
                hot_flash_freq / 30 * 10,
                night_sweats_freq / 20 * 10,
                mood_swings_severity,
                anxiety_level,
                depression_score,
                cognitive_issues,
                fatigue_level,
                abs(libido_change),
                vaginal_dryness,
                joint_pain / 10 * 10
            ])
            
            data.append({
                # Demographics
                'patient_id': f'PT{i:05d}',
                'age': age,
                'bmi': round(bmi, 1),
                
                # Biomarkers
                'fsh_level': round(fsh, 1),
                'estradiol_level': round(estradiol, 1),
                'lh_level': round(lh, 1),
                'testosterone': round(np.clip(np.random.normal(30, 15), 10, 80), 1),
                'progesterone': round(np.clip(np.random.normal(2, 1.5), 0.1, 8), 1),
                'thyroid_tsh': round(np.clip(np.random.normal(2.5, 1.5), 0.5, 10), 2),
                'vitamin_d': round(np.clip(np.random.normal(35, 15), 10, 80), 1),
                'cholesterol_total': round(np.clip(np.random.normal(200, 40), 120, 300), 0),
                
                # Lifestyle
                'sleep_quality': round(sleep_quality, 1),
                'stress_level': round(stress_level, 1),
                'exercise_minutes_per_week': round(exercise_minutes, 0),
                'diet_quality_score': round(diet_quality, 1),
                'smoking_status': int(smoking_status),
                'alcohol_drinks_per_week': int(np.clip(np.random.poisson(3), 0, 20)),
                
                # Medical History
                'has_diabetes': int(has_diabetes),
                'has_hypertension': int(has_hypertension),
                'has_thyroid_disorder': int(has_thyroid_disorder),
                'family_history_early_menopause': int(np.random.choice([0, 1], p=[0.85, 0.15])),
                
                # Menstrual History
                'last_period_months_ago': int(last_period_months),
                'cycle_irregularity': int(cycle_irregularity),
                'age_at_menarche': int(np.clip(np.random.normal(13, 2), 9, 17)),
                
                # Symptoms
                'hot_flash_frequency_per_day': round(hot_flash_freq, 1),
                'night_sweats_frequency_per_week': round(night_sweats_freq, 1),
                'mood_swings_severity': round(mood_swings_severity, 1),
                'anxiety_level': round(anxiety_level, 1),
                'depression_score': round(depression_score, 1),
                'cognitive_issues_score': round(cognitive_issues, 1),
                'fatigue_level': round(fatigue_level, 1),
                'libido_change': round(libido_change, 1),
                'vaginal_dryness_score': round(vaginal_dryness, 1),
                'joint_pain_score': round(joint_pain, 1),
                'weight_change_kg': round(weight_change, 1),
                'heart_palpitations_per_week': round(heart_palpitations_freq, 1),
                'overall_symptom_severity': round(overall_severity, 2),
                
                # Target Variables
                'months_to_menopause': int(months_to_menopause),
                'menopause_stage': stage,
                'high_symptom_severity': int(overall_severity > 6)
            })
        
        return pd.DataFrame(data)
    
    def _age_distribution(self):
        """Create realistic age distribution for menopause study"""
        ages = np.arange(35, 61)
        # Peak around 47-52
        probs = np.exp(-0.5 * ((ages - 49) / 8) ** 2)
        return probs / probs.sum()

# ============================================================================
# PART 2: MACHINE LEARNING MODELS
# ============================================================================

class MenopausePredictionModel:
    """ML model for predicting menopause onset and symptom severity"""
    
    def __init__(self):
        self.regression_model = None
        self.classification_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_features(self, df):
        """Prepare features for modeling"""
        feature_cols = [
            'age', 'bmi', 'fsh_level', 'estradiol_level', 'lh_level',
            'testosterone', 'progesterone', 'thyroid_tsh', 'vitamin_d',
            'cholesterol_total', 'sleep_quality', 'stress_level',
            'exercise_minutes_per_week', 'diet_quality_score', 'smoking_status',
            'alcohol_drinks_per_week', 'has_diabetes', 'has_hypertension',
            'has_thyroid_disorder', 'family_history_early_menopause',
            'last_period_months_ago', 'cycle_irregularity', 'age_at_menarche',
            'hot_flash_frequency_per_day', 'night_sweats_frequency_per_week',
            'mood_swings_severity', 'anxiety_level', 'depression_score',
            'cognitive_issues_score', 'fatigue_level', 'vaginal_dryness_score',
            'joint_pain_score', 'weight_change_kg', 'heart_palpitations_per_week'
        ]
        
        self.feature_names = feature_cols
        return df[feature_cols]
    
    def train_regression_model(self, X_train, y_train):
        """Train regression model to predict months to menopause"""
        print("Training regression model for months to menopause prediction...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Random Forest Regressor
        self.regression_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.regression_model.fit(X_train_scaled, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.regression_model, X_train_scaled, y_train,
            cv=5, scoring='neg_mean_absolute_error'
        )
        print(f"CV MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")
        
        return self.regression_model
    
    def train_classification_model(self, X_train, y_train):
        """Train classification model for high symptom severity"""
        print("\nTraining classification model for symptom severity...")
        
        X_train_scaled = self.scaler.transform(X_train)
        
        # Gradient Boosting Classifier
        self.classification_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        self.classification_model.fit(X_train_scaled, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.classification_model, X_train_scaled, y_train,
            cv=5, scoring='f1'
        )
        print(f"CV F1-Score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        return self.classification_model
    
    def evaluate_models(self, X_test, y_reg_test, y_clf_test):
        """Evaluate both models on test set"""
        X_test_scaled = self.scaler.transform(X_test)
        
        # Regression evaluation
        y_reg_pred = self.regression_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_reg_test, y_reg_pred)
        
        print("\n" + "="*60)
        print("REGRESSION MODEL PERFORMANCE (Months to Menopause)")
        print("="*60)
        print(f"Mean Absolute Error: {mae:.2f} months")
        print(f"Accuracy (within ±6 months): {np.mean(np.abs(y_reg_pred - y_reg_test) <= 6) * 100:.1f}%")
        
        # Classification evaluation
        y_clf_pred = self.classification_model.predict(X_test_scaled)
        y_clf_proba = self.classification_model.predict_proba(X_test_scaled)[:, 1]
        
        f1 = f1_score(y_clf_test, y_clf_pred)
        roc_auc = roc_auc_score(y_clf_test, y_clf_proba)
        
        print("\n" + "="*60)
        print("CLASSIFICATION MODEL PERFORMANCE (High Symptom Severity)")
        print("="*60)
        print(f"F1-Score: {f1:.3f}")
        print(f"ROC-AUC: {roc_auc:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_clf_test, y_clf_pred))
        
        return {
            'regression_mae': mae,
            'classification_f1': f1,
            'classification_roc_auc': roc_auc
        }
    
    def get_feature_importance(self):
        """Get feature importance from both models"""
        reg_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.regression_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        clf_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.classification_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return reg_importance, clf_importance
    
    def predict(self, X):
        """Make predictions for new data"""
        X_scaled = self.scaler.transform(X)
        months_pred = self.regression_model.predict(X_scaled)
        severity_proba = self.classification_model.predict_proba(X_scaled)[:, 1]
        
        return months_pred, severity_proba
    
    def save_models(self, path='models/'):
        """Save trained models"""
        import os
        os.makedirs(path, exist_ok=True)
        
        joblib.dump(self.regression_model, f'{path}regression_model.pkl')
        joblib.dump(self.classification_model, f'{path}classification_model.pkl')
        joblib.dump(self.scaler, f'{path}scaler.pkl')
        joblib.dump(self.feature_names, f'{path}feature_names.pkl')
        print(f"\nModels saved to {path}")

# ============================================================================
# PART 3: VISUALIZATION AND ANALYSIS
# ============================================================================

def visualize_results(df, model, X_test, y_reg_test, y_clf_test):
    """Create comprehensive visualizations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Menopause Prediction Model Analysis', fontsize=16, fontweight='bold')
    
    # 1. Feature Importance (Regression)
    reg_imp, clf_imp = model.get_feature_importance()
    axes[0, 0].barh(reg_imp['feature'][:10], reg_imp['importance'][:10])
    axes[0, 0].set_xlabel('Importance')
    axes[0, 0].set_title('Top 10 Features - Menopause Timing')
    axes[0, 0].invert_yaxis()
    
    # 2. Feature Importance (Classification)
    axes[0, 1].barh(clf_imp['feature'][:10], clf_imp['importance'][:10], color='coral')
    axes[0, 1].set_xlabel('Importance')
    axes[0, 1].set_title('Top 10 Features - Symptom Severity')
    axes[0, 1].invert_yaxis()
    
    # 3. Age vs FSH by Stage
    stage_colors = {'pre_menopause': 'green', 'early_perimenopause': 'yellow', 
                    'late_perimenopause': 'orange', 'post_menopause': 'red'}
    for stage, color in stage_colors.items():
        stage_data = df[df['menopause_stage'] == stage]
        axes[0, 2].scatter(stage_data['age'], stage_data['fsh_level'], 
                          c=color, label=stage.replace('_', ' ').title(), alpha=0.6)
    axes[0, 2].set_xlabel('Age')
    axes[0, 2].set_ylabel('FSH Level')
    axes[0, 2].set_title('Age vs FSH by Menopause Stage')
    axes[0, 2].legend()
    
    # 4. Symptom Severity Distribution
    axes[1, 0].hist(df['overall_symptom_severity'], bins=30, color='purple', alpha=0.7)
    axes[1, 0].axvline(6, color='red', linestyle='--', label='High Severity Threshold')
    axes[1, 0].set_xlabel('Overall Symptom Severity')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Symptom Severity')
    axes[1, 0].legend()
    
    # 5. Predicted vs Actual (Regression)
    X_test_scaled = model.scaler.transform(X_test)
    y_pred = model.regression_model.predict(X_test_scaled)
    axes[1, 1].scatter(y_reg_test, y_pred, alpha=0.5)
    axes[1, 1].plot([0, max(y_reg_test)], [0, max(y_reg_test)], 'r--', lw=2)
    axes[1, 1].set_xlabel('Actual Months to Menopause')
    axes[1, 1].set_ylabel('Predicted Months')
    axes[1, 1].set_title('Prediction Accuracy - Regression')
    
    # 6. Correlation Heatmap (Top Features)
    top_features = ['age', 'fsh_level', 'estradiol_level', 'hot_flash_frequency_per_day', 
                    'sleep_quality', 'stress_level', 'months_to_menopause']
    corr = df[top_features].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=axes[1, 2], 
                square=True, linewidths=1)
    axes[1, 2].set_title('Feature Correlations')
    
    plt.tight_layout()
    plt.savefig('model_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'model_analysis.png'")

# ============================================================================
# PART 4: MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("="*60)
    print("MENOPAUSE PREDICTION & MANAGEMENT TOOL")
    print("ML Pipeline Execution")
    print("="*60)
    
    # Generate synthetic dataset
    print("\n[1/5] Generating synthetic dataset...")
    generator = MenopauseDataGenerator(n_samples=5000)
    df = generator.generate_dataset()
    df.to_csv('menopause_synthetic_dataset.csv', index=False)
    print(f"Dataset generated: {len(df)} samples")
    print(f"Saved to: menopause_synthetic_dataset.csv")
    
    # Display dataset info
    print("\nDataset Overview:")
    print(df.describe())
    print("\nMenopause Stage Distribution:")
    print(df['menopause_stage'].value_counts())
    
    # Prepare data
    print("\n[2/5] Preparing features and targets...")
    model = MenopausePredictionModel()
    X = model.prepare_features(df)
    y_regression = df['months_to_menopause']
    y_classification = df['high_symptom_severity']
    
    # Train-test split
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_regression, y_classification, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train models
    print("\n[3/5] Training machine learning models...")
    model.train_regression_model(X_train, y_reg_train)
    model.train_classification_model(X_train, y_clf_train)
    
    # Evaluate models
    print("\n[4/5] Evaluating model performance...")
    metrics = model.evaluate_models(X_test, y_reg_test, y_clf_test)
    
    # Feature importance
    print("\n[5/5] Analyzing feature importance...")
    reg_imp, clf_imp = model.get_feature_importance()
    print("\nTop 10 Features for Menopause Timing Prediction:")
    print(reg_imp.head(10).to_string(index=False))
    print("\nTop 10 Features for Symptom Severity Prediction:")
    print(clf_imp.head(10).to_string(index=False))
    
    # Visualizations
    print("\nGenerating visualizations...")
    visualize_results(df, model, X_test, y_reg_test, y_clf_test)
    
    # Save models
    model.save_models()
    
    # Test case validation
    print("\n" + "="*60)
    print("TEST CASE VALIDATION (from hackathon requirements)")
    print("="*60)
    
    # Test Case 1: Symptom Prediction
    test_case_1 = pd.DataFrame([{
        'age': 47, 'bmi': 26, 'fsh_level': 45, 'estradiol_level': 20,
        'lh_level': 30, 'testosterone': 25, 'progesterone': 1.5,
        'thyroid_tsh': 2.0, 'vitamin_d': 30, 'cholesterol_total': 200,
        'sleep_quality': 4, 'stress_level': 8, 'exercise_minutes_per_week': 60,
        'diet_quality_score': 5, 'smoking_status': 0, 'alcohol_drinks_per_week': 2,
        'has_diabetes': 0, 'has_hypertension': 0, 'has_thyroid_disorder': 0,
        'family_history_early_menopause': 0, 'last_period_months_ago': 4,
        'cycle_irregularity': 1, 'age_at_menarche': 13,
        'hot_flash_frequency_per_day': 8, 'night_sweats_frequency_per_week': 5,
        'mood_swings_severity': 7, 'anxiety_level': 7, 'depression_score': 6,
        'cognitive_issues_score': 6, 'fatigue_level': 8, 'vaginal_dryness_score': 6,
        'joint_pain_score': 5, 'weight_change_kg': 5, 'heart_palpitations_per_week': 3
    }])
    
    months_pred, severity_prob = model.predict(test_case_1)
    print(f"\nTest Case 1 - High-risk patient (Age 47, high FSH, low estradiol, high stress):")
    print(f"  Predicted months to menopause: {months_pred[0]:.0f} months")
    print(f"  High symptom severity probability: {severity_prob[0]*100:.1f}%")
    print(f"  Status: {'✓ PASS' if months_pred[0] <= 12 else '✗ FAIL'} (Expected: within 12 months)")
    
    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*60)
    print("\nDeliverables created:")
    print("  ✓ menopause_synthetic_dataset.csv")
    print("  ✓ model_analysis.png")
    print("  ✓ models/regression_model.pkl")
    print("  ✓ models/classification_model.pkl")
    print("  ✓ models/scaler.pkl")
    print("\nModel Performance Summary:")
    print(f"  • Regression MAE: {metrics['regression_mae']:.2f} months")
    print(f"  • Classification F1-Score: {metrics['classification_f1']:.3f}")
    print(f"  • Classification ROC-AUC: {metrics['classification_roc_auc']:.3f}")

if __name__ == "__main__":
    main()