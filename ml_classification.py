"""
Machine Learning Module - Enhanced with Visualizations
Implements techniques to improve fatigue classification accuracy
- Feature Engineering (interactions, PCA)
- Ensemble Methods (voting, stacking)
- Per-Subject Models
- Temporal Features
- Comprehensive Visualizations (Confusion Matrices, ROC Curves, Feature Importance)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              VotingClassifier, StackingClassifier)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_curve, auc, roc_auc_score)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False


class FatigueClassifier:
    """ML classifier with feature engineering and ensemble methods"""
    
    def __init__(self, features_dir='features_output', output_dir='ml_advanced_results'):
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {}
        self.base_accuracy = None
        self.feature_names = None
        self.X_test = None
        self.y_test = None
        self.y_pred_dict = {}
        self.y_pred_proba_dict = {}
    
    # ========================================================================
    # 1. ENHANCED DATA LOADING (WITH GRAPH METRICS)
    # ========================================================================
    
    def load_all_features_enhanced(self):
        """Load features including graph metrics"""
        print("\n" + "="*70)
        print("LOADING ENHANCED FEATURE SET")
        print("="*70 + "\n")
        
        all_data = []
        
        for subject_dir in self.features_dir.iterdir():
            if not subject_dir.is_dir() or not subject_dir.name.startswith('Subject'):
                continue
            
            subject_id = subject_dir.name
            
            # Load band powers
            band_file = subject_dir / f'{subject_id}_band_powers.csv'
            complexity_file = subject_dir / f'{subject_id}_complexity.csv'
            graph_file = subject_dir / f'{subject_id}_graph_metrics.csv'
            sliding_file = subject_dir / f'{subject_id}_sliding_window.csv'
            
            if band_file.exists() and complexity_file.exists():
                band_df = pd.read_csv(band_file)
                complexity_df = pd.read_csv(complexity_file)
                
                # Merge band powers and complexity
                merged = pd.merge(band_df, complexity_df, 
                                on=['channel', 'condition', 'subject'], how='inner')
                
                # Add graph metrics (per subject-condition, not per channel)
                if graph_file.exists():
                    graph_df = pd.read_csv(graph_file)
                    # Merge on subject and condition only
                    for _, graph_row in graph_df.iterrows():
                        condition = graph_row['condition']
                        subject = graph_row['subject']
                        
                        # Add graph metrics to all channels of this subject-condition
                        mask = (merged['condition'] == condition) & (merged['subject'] == subject)
                        for col in graph_df.columns:
                            if col not in ['condition', 'subject']:
                                merged.loc[mask, f'graph_{col}'] = graph_row[col]
                
                # Add temporal features (average per subject-condition)
                if sliding_file.exists():
                    sliding_df = pd.read_csv(sliding_file)
                    # Compute statistics from sliding window
                    temporal_stats = {
                        'delta_trend': np.polyfit(range(len(sliding_df)), 
                                                  sliding_df['delta_power'], 1)[0] if len(sliding_df) > 1 else 0,
                        'theta_trend': np.polyfit(range(len(sliding_df)), 
                                                 sliding_df['theta_power'], 1)[0] if len(sliding_df) > 1 else 0,
                        'alpha_trend': np.polyfit(range(len(sliding_df)), 
                                                 sliding_df['alpha_power'], 1)[0] if len(sliding_df) > 1 else 0,
                        'beta_trend': np.polyfit(range(len(sliding_df)), 
                                                sliding_df['beta_power'], 1)[0] if len(sliding_df) > 1 else 0,
                        'alpha_variability': sliding_df['alpha_power'].std(),
                        'theta_variability': sliding_df['theta_power'].std()
                    }
                    
                    # Add to merged data
                    for key, val in temporal_stats.items():
                        merged[key] = val
                
                all_data.append(merged)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Print detailed data summary
        print(f"✓ Loaded ENHANCED features from {combined_df['subject'].nunique()} subjects")
        print(f"  - Total samples (rows): {len(combined_df)}")
        print(f"  - Unique subjects: {combined_df['subject'].nunique()}")
        print(f"  - Conditions: {combined_df['condition'].unique().tolist()}")
        
        # Count samples per condition
        condition_counts = combined_df.groupby('condition').size()
        print(f"\n  Samples per condition:")
        for condition, count in condition_counts.items():
            print(f"    • {condition}: {count} samples")
        
        # Count unique subject-condition pairs
        subject_condition_pairs = combined_df.groupby(['subject', 'condition']).size()
        print(f"\n  Subject-Condition pairs: {len(subject_condition_pairs)}")
        print(f"    (Expected: {combined_df['subject'].nunique()} subjects × 2 conditions = {combined_df['subject'].nunique() * 2})")
        
        print(f"\n  - Features: {len([c for c in combined_df.columns if c not in ['channel', 'condition', 'subject']])}")
        
        # Show channels per subject-condition
        if 'channel' in combined_df.columns:
            n_channels = combined_df['channel'].nunique()
            channels_per_pair = combined_df.groupby(['subject', 'condition'])['channel'].nunique().values
            print(f"  - Channels per subject-condition: {channels_per_pair[0] if len(channels_per_pair) > 0 else 'N/A'}")
            print(f"  - Total unique channels: {n_channels} (23-channel system)")
            print(f"\n  Calculation: {combined_df['subject'].nunique()} subjects × 2 conditions × {n_channels} channels = {combined_df['subject'].nunique() * 2 * n_channels} samples")
        
        return combined_df
    
    # ========================================================================
    # 2. FEATURE ENGINEERING
    # ========================================================================
    
    def engineer_features(self, X, feature_names, method='all'):
        """
        Engineer features
        
        Parameters:
        -----------
        X : array
            Original features
        feature_names : list
            Feature names
        method : str
            'interactions', 'pca', 'all'
        """
        print(f"\n{'='*70}")
        print(f"FEATURE ENGINEERING: {method.upper()}")
        print(f"{'='*70}\n")
        
        X_enhanced = X.copy()
        new_feature_names = feature_names.copy()
        
        if method in ['interactions', 'all']:
            print("Creating interaction features...")
            # Create important interactions
            # Theta/Beta ratio interaction
            theta_idx = [i for i, name in enumerate(feature_names) if 'theta_abs' in name]
            beta_idx = [i for i, name in enumerate(feature_names) if 'beta_abs' in name]
            alpha_idx = [i for i, name in enumerate(feature_names) if 'alpha_abs' in name]
            
            if theta_idx and beta_idx:
                theta_beta = X[:, theta_idx[0]] / (X[:, beta_idx[0]] + 1e-10)
                X_enhanced = np.column_stack([X_enhanced, theta_beta])
                new_feature_names.append('theta_beta_interaction')
            
            if alpha_idx and theta_idx:
                alpha_theta = X[:, alpha_idx[0]] / (X[:, theta_idx[0]] + 1e-10)
                X_enhanced = np.column_stack([X_enhanced, alpha_theta])
                new_feature_names.append('alpha_theta_interaction')
            
            # Complexity * Power interactions
            entropy_idx = [i for i, name in enumerate(feature_names) if 'sample_entropy' in name]
            if entropy_idx and alpha_idx:
                entropy_alpha = X[:, entropy_idx[0]] * X[:, alpha_idx[0]]
                X_enhanced = np.column_stack([X_enhanced, entropy_alpha])
                new_feature_names.append('entropy_alpha_interaction')
            
            print(f" Added {X_enhanced.shape[1] - X.shape[1]} interaction features")
        
        if method in ['pca', 'all']:
            print("Applying PCA dimensionality reduction...")
            # Keep components explaining 95% variance
            pca = PCA(n_components=0.95, random_state=42)
            X_pca = pca.fit_transform(X_enhanced)
            
            print(f" Reduced from {X_enhanced.shape[1]} to {X_pca.shape[1]} features")
            print(f" Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
            
            return X_pca, [f'PC{i+1}' for i in range(X_pca.shape[1])], pca
        
        return X_enhanced, new_feature_names, None
    
    # ========================================================================
    # 3. ENSEMBLE METHODS
    # ========================================================================
    
    def create_voting_ensemble(self):
        """Create voting classifier from top models"""
        print("\nCreating Voting Ensemble (soft voting)...")
        
        estimators = [
            ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
        ]
        
        if XGBOOST_AVAILABLE:
            estimators.append(
                ('xgb', xgb.XGBClassifier(n_estimators=200, max_depth=6, random_state=42, eval_metric='logloss'))
            )
        
        voting_clf = VotingClassifier(estimators=estimators, voting='soft')
        
        return voting_clf
    
    def create_stacking_ensemble(self):
        """Create stacking classifier"""
        print("\nCreating Stacking Ensemble...")
        
        base_estimators = [
            ('svm', SVC(kernel='rbf', C=10, probability=True, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=200, random_state=42))
        ]
        
        if XGBOOST_AVAILABLE:
            base_estimators.append(
                ('xgb', xgb.XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss'))
            )
        
        # Meta-learner
        meta_learner = LogisticRegression(max_iter=2000)
        
        stacking_clf = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=5
        )
        
        return stacking_clf
    
    # ========================================================================
    # 4. PER-SUBJECT CALIBRATION
    # ========================================================================
    
    def train_per_subject_models(self, df, feature_cols):
        """Train personalized models for each subject"""
        print(f"\n{'='*70}")
        print("PER-SUBJECT PERSONALIZED MODELS")
        print(f"{'='*70}\n")
        
        subjects = df['subject'].unique()
        subject_results = []
        
        for subject in subjects:
            # Get subject data
            subject_data = df[df['subject'] == subject]
            
            if len(subject_data) < 10:  # Need minimum samples
                continue
            
            X_subj = subject_data[feature_cols].values
            y_subj = (subject_data['condition'] == 'pre').astype(int).values
            
            # Handle NaN
            X_subj = np.nan_to_num(X_subj, nan=0.0)
            
            # Train-test split
            if len(X_subj) > 20:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_subj, y_subj, test_size=0.3, random_state=42, stratify=y_subj
                )
                
                # Train simple model
                clf = SVC(kernel='rbf', probability=True, random_state=42)
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                clf.fit(X_train_scaled, y_train)
                
                accuracy = clf.score(X_test_scaled, y_test)
                subject_results.append({
                    'subject': subject,
                    'accuracy': accuracy,
                    'n_samples': len(X_subj)
                })
        
        results_df = pd.DataFrame(subject_results)
        
        if len(results_df) > 0:
            print(f"✓ Trained models for {len(results_df)} subjects")
            print(f"  - Mean accuracy: {results_df['accuracy'].mean():.4f}")
            print(f"  - Std accuracy: {results_df['accuracy'].std():.4f}")
            print(f"  - Best subject: {results_df['accuracy'].max():.4f}")
            print(f"  - Worst subject: {results_df['accuracy'].min():.4f}")
            
            # Save results
            results_df.to_csv(self.output_dir / 'per_subject_results.csv', index=False)
            
            return results_df
        
        return None
    
    # ========================================================================
    # 5. VISUALIZATION METHODS
    # ========================================================================
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all methods"""
        print("\n" + "="*70)
        print("GENERATING CONFUSION MATRICES")
        print("="*70 + "\n")
        
        n_methods = len(self.y_pred_dict)
        n_cols = 3
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_methods > 1 else [axes]
        
        for idx, (method_name, y_pred) in enumerate(self.y_pred_dict.items()):
            cm = confusion_matrix(self.y_test, y_pred)
            
            # Plot
            ax = axes[idx]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Post-Fatigue', 'Pre-Fatigue'],
                       yticklabels=['Post-Fatigue', 'Pre-Fatigue'],
                       ax=ax, cbar_kws={'label': 'Count'})
            
            ax.set_xlabel('Predicted Label', fontsize=10, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=10, fontweight='bold')
            
            # Add accuracy to title
            acc = self.results[method_name]['accuracy']
            ax.set_title(f'{method_name}\nAccuracy: {acc:.4f}', 
                        fontsize=11, fontweight='bold', pad=10)
        
        # Hide unused subplots
        for idx in range(n_methods, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: confusion_matrices.png")
    
    def plot_roc_curves(self):
        """Plot ROC curves for all methods"""
        print("\n" + "="*70)
        print("GENERATING ROC CURVES")
        print("="*70 + "\n")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.y_pred_proba_dict)))
        
        for idx, (method_name, y_pred_proba) in enumerate(self.y_pred_proba_dict.items()):
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Plot
            ax.plot(fpr, tpr, color=colors[idx], lw=2.5, 
                   label=f'{method_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.500)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves - All Methods', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Saved: roc_curves.png")
    
    def plot_feature_importance(self, X_train, y_train, feature_names):
        """Plot feature importance using multiple methods"""
        print("\n" + "="*70)
        print("GENERATING FEATURE IMPORTANCE")
        print("="*70 + "\n")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Method 1: Random Forest Feature Importance
        print("Computing Random Forest feature importance...")
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        
        rf_importance = rf.feature_importances_
        rf_indices = np.argsort(rf_importance)[-20:]  # Top 20
        
        axes[0].barh(range(len(rf_indices)), rf_importance[rf_indices], 
                     color='#3498db', alpha=0.8, edgecolor='black')
        axes[0].set_yticks(range(len(rf_indices)))
        axes[0].set_yticklabels([feature_names[i] for i in rf_indices], fontsize=9)
        axes[0].set_xlabel('Importance Score', fontsize=11, fontweight='bold')
        axes[0].set_title('Random Forest Feature Importance (Top 20)', 
                         fontsize=12, fontweight='bold', pad=15)
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Method 2: Permutation Importance (on test set)
        print("Computing permutation importance...")
        
        # Use a simpler model for permutation importance (faster)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        
        perm_importance = permutation_importance(
            lr, X_train, y_train, n_repeats=10, random_state=42, n_jobs=-1
        )
        
        perm_indices = np.argsort(perm_importance.importances_mean)[-20:]
        
        axes[1].barh(range(len(perm_indices)), 
                     perm_importance.importances_mean[perm_indices],
                     xerr=perm_importance.importances_std[perm_indices],
                     color='#e74c3c', alpha=0.8, edgecolor='black')
        axes[1].set_yticks(range(len(perm_indices)))
        axes[1].set_yticklabels([feature_names[i] for i in perm_indices], fontsize=9)
        axes[1].set_xlabel('Importance Score', fontsize=11, fontweight='bold')
        axes[1].set_title('Permutation Importance (Top 20)', 
                         fontsize=12, fontweight='bold', pad=15)
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Saved: feature_importance.png")
        
        # Save feature importance data
        importance_data = []
        for i, name in enumerate(feature_names):
            importance_data.append({
                'feature': name,
                'rf_importance': rf_importance[i],
                'perm_importance_mean': perm_importance.importances_mean[i],
                'perm_importance_std': perm_importance.importances_std[i]
            })
        
        importance_df = pd.DataFrame(importance_data)
        importance_df = importance_df.sort_values('rf_importance', ascending=False)
        importance_df.to_csv(self.output_dir / 'feature_importance.csv', index=False)
        
        print(f" Saved: feature_importance.csv")
        
        # Print top 10 features
        print("\nTop 10 Most Important Features (Random Forest):")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']:40s} {row['rf_importance']:.6f}")
    
    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================
    
    def run_advanced_analysis(self):
        """Run complete advanced analysis pipeline"""
        
        print(f"\n{'#'*70}")
        print("ADVANCED ML PIPELINE - ACCURACY IMPROVEMENT")
        print(f"{'#'*70}\n")
        
        # 1. Load enhanced features
        df = self.load_all_features_enhanced()
        
        # Prepare data
        exclude_cols = ['channel', 'condition', 'subject']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        X = df[feature_cols].values
        y = (df['condition'] == 'pre').astype(int).values
        
        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Store for later use
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_cols
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"\n{'='*70}")
        print("DATASET SPLIT INFORMATION")
        print(f"{'='*70}\n")
        
        print(f"Original Dataset:")
        print(f"  - Total samples (rows): {len(X)}")
        print(f"  - Total features: {X.shape[1]}")
        print(f"  - Class 0 (Post-Fatigue): {np.sum(y == 0)} samples ({np.sum(y == 0)/len(y)*100:.1f}%)")
        print(f"  - Class 1 (Pre-Fatigue): {np.sum(y == 1)} samples ({np.sum(y == 1)/len(y)*100:.1f}%)")
        
        print(f"\nTraining Set (70%):")
        print(f"  - Total samples: {len(X_train)} samples")
        print(f"  - Class 0 (Post-Fatigue): {np.sum(y_train == 0)} samples ({np.sum(y_train == 0)/len(y_train)*100:.1f}%)")
        print(f"  - Class 1 (Pre-Fatigue): {np.sum(y_train == 1)} samples ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)")
        
        print(f"\nTest Set (30%):")
        print(f"  - Total samples: {len(X_test)} samples")
        print(f"  - Class 0 (Post-Fatigue): {np.sum(y_test == 0)} samples ({np.sum(y_test == 0)/len(y_test)*100:.1f}%)")
        print(f"  - Class 1 (Pre-Fatigue): {np.sum(y_test == 1)} samples ({np.sum(y_test == 1)/len(y_test)*100:.1f}%)")
        
        print(f"\nData Breakdown:")
        print(f"  - Subjects: {df['subject'].nunique()}")
        print(f"  - Conditions: {df['condition'].nunique()}")
        print(f"  - Subject-Condition pairs: {df.groupby(['subject', 'condition']).ngroups}")
        if 'channel' in df.columns:
            n_channels = df['channel'].nunique()
            print(f"  - Channels per recording: {n_channels} (23-channel system)")
            print(f"  - Total samples = {df['subject'].nunique()} subjects × 2 conditions × {n_channels} channels = {df['subject'].nunique() * 2 * n_channels}")
            print(f"  - Expected for your data: 36 subjects × 2 conditions × 23 channels = 1,656 samples")
        
        print(f"\n{'='*70}\n")
        
        # ====================================================================
        # BASELINE: Standard SVM (for comparison)
        # ====================================================================
        print(f"\n{'='*70}")
        print("BASELINE MODEL (Standard SVM)")
        print(f"{'='*70}\n")
        
        baseline_svm = SVC(kernel='rbf', C=10, probability=True, random_state=42)
        baseline_svm.fit(X_train_scaled, y_train)
        baseline_acc = baseline_svm.score(X_test_scaled, y_test)
        
        y_pred_baseline = baseline_svm.predict(X_test_scaled)
        y_pred_proba_baseline = baseline_svm.predict_proba(X_test_scaled)[:, 1]
        
        print(f"Baseline SVM Accuracy: {baseline_acc:.4f}")
        self.base_accuracy = baseline_acc
        self.results['Baseline SVM'] = {
            'accuracy': baseline_acc,
            'model': baseline_svm
        }
        self.y_pred_dict['Baseline SVM'] = y_pred_baseline
        self.y_pred_proba_dict['Baseline SVM'] = y_pred_proba_baseline
        
        # ====================================================================
        # METHOD 1: Feature Engineering (Interactions)
        # ====================================================================
        print(f"\n{'='*70}")
        print("METHOD 1: FEATURE ENGINEERING (INTERACTIONS)")
        print(f"{'='*70}")
        
        X_train_eng, feat_names_eng, _ = self.engineer_features(
            X_train_scaled, feature_cols, method='interactions'
        )
        X_test_eng, _, _ = self.engineer_features(
            X_test_scaled, feature_cols, method='interactions'
        )
        
        svm_eng = SVC(kernel='rbf', C=10, probability=True, random_state=42)
        svm_eng.fit(X_train_eng, y_train)
        eng_acc = svm_eng.score(X_test_eng, y_test)
        
        y_pred_eng = svm_eng.predict(X_test_eng)
        y_pred_proba_eng = svm_eng.predict_proba(X_test_eng)[:, 1]
        
        print(f"\nSVM with Interactions Accuracy: {eng_acc:.4f}")
        print(f"Improvement over baseline: {(eng_acc - baseline_acc)*100:+.2f}%")
        
        self.results['SVM + Interactions'] = {
            'accuracy': eng_acc,
            'model': svm_eng
        }
        self.y_pred_dict['SVM + Interactions'] = y_pred_eng
        self.y_pred_proba_dict['SVM + Interactions'] = y_pred_proba_eng
        
        # ====================================================================
        # METHOD 2: PCA Dimensionality Reduction
        # ====================================================================
        print(f"\n{'='*70}")
        print("METHOD 2: PCA DIMENSIONALITY REDUCTION")
        print(f"{'='*70}")
        
        X_train_pca, feat_names_pca, pca_model = self.engineer_features(
            X_train_scaled, feature_cols, method='pca'
        )
        X_test_pca = pca_model.transform(X_test_scaled)
        
        svm_pca = SVC(kernel='rbf', C=10, probability=True, random_state=42)
        svm_pca.fit(X_train_pca, y_train)
        pca_acc = svm_pca.score(X_test_pca, y_test)
        
        y_pred_pca = svm_pca.predict(X_test_pca)
        y_pred_proba_pca = svm_pca.predict_proba(X_test_pca)[:, 1]
        
        print(f"\nSVM with PCA Accuracy: {pca_acc:.4f}")
        print(f"Improvement over baseline: {(pca_acc - baseline_acc)*100:+.2f}%")
        
        self.results['SVM + PCA'] = {
            'accuracy': pca_acc,
            'model': svm_pca
        }
        self.y_pred_dict['SVM + PCA'] = y_pred_pca
        self.y_pred_proba_dict['SVM + PCA'] = y_pred_proba_pca
        
        # ====================================================================
        # METHOD 3: Voting Ensemble
        # ====================================================================
        print(f"\n{'='*70}")
        print("METHOD 3: VOTING ENSEMBLE")
        print(f"{'='*70}")
        
        voting_clf = self.create_voting_ensemble()
        voting_clf.fit(X_train_scaled, y_train)
        voting_acc = voting_clf.score(X_test_scaled, y_test)
        
        y_pred_voting = voting_clf.predict(X_test_scaled)
        y_pred_proba_voting = voting_clf.predict_proba(X_test_scaled)[:, 1]
        
        print(f"\nVoting Ensemble Accuracy: {voting_acc:.4f}")
        print(f"Improvement over baseline: {(voting_acc - baseline_acc)*100:+.2f}%")
        
        self.results['Voting Ensemble'] = {
            'accuracy': voting_acc,
            'model': voting_clf
        }
        self.y_pred_dict['Voting Ensemble'] = y_pred_voting
        self.y_pred_proba_dict['Voting Ensemble'] = y_pred_proba_voting
        
        # ====================================================================
        # METHOD 4: Stacking Ensemble
        # ====================================================================
        print(f"\n{'='*70}")
        print("METHOD 4: STACKING ENSEMBLE")
        print(f"{'='*70}")
        
        stacking_clf = self.create_stacking_ensemble()
        stacking_clf.fit(X_train_scaled, y_train)
        stacking_acc = stacking_clf.score(X_test_scaled, y_test)
        
        y_pred_stacking = stacking_clf.predict(X_test_scaled)
        y_pred_proba_stacking = stacking_clf.predict_proba(X_test_scaled)[:, 1]
        
        print(f"\nStacking Ensemble Accuracy: {stacking_acc:.4f}")
        print(f"Improvement over baseline: {(stacking_acc - baseline_acc)*100:+.2f}%")
        
        self.results['Stacking Ensemble'] = {
            'accuracy': stacking_acc,
            'model': stacking_clf
        }
        self.y_pred_dict['Stacking Ensemble'] = y_pred_stacking
        self.y_pred_proba_dict['Stacking Ensemble'] = y_pred_proba_stacking
        
        # ====================================================================
        # METHOD 5: Per-Subject Models
        # ====================================================================
        print(f"\n{'='*70}")
        print("METHOD 5: PER-SUBJECT PERSONALIZED MODELS")
        print(f"{'='*70}")
        
        subject_results = self.train_per_subject_models(df, feature_cols)
        
        # ====================================================================
        # GENERATE VISUALIZATIONS
        # ====================================================================
        print(f"\n{'='*70}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*70}")
        
        self.plot_confusion_matrices()
        self.plot_roc_curves()
        self.plot_feature_importance(X_train_scaled, y_train, feature_cols)
        
        # ====================================================================
        # FINAL SUMMARY
        # ====================================================================
        self.print_final_summary()
        self.plot_improvement_comparison()
    
    def print_final_summary(self):
        """Print final summary of all methods"""
        
        print(f"\n{'='*70}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'='*70}\n")
        
        summary_data = []
        for name, result in self.results.items():
            improvement = ((result['accuracy'] - self.base_accuracy) / self.base_accuracy) * 100
            summary_data.append({
                'Method': name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'Improvement': f"{improvement:+.2f}%"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Accuracy', ascending=False)
        
        print(summary_df.to_string(index=False))
        
        # Save
        summary_df.to_csv(self.output_dir / 'advanced_methods_summary.csv', index=False)
        
        # Best method
        best_name = max(self.results.items(), key=lambda x: x[1]['accuracy'])[0]
        best_acc = self.results[best_name]['accuracy']
        
        print(f"\n BEST METHOD: {best_name}")
        print(f"   Accuracy: {best_acc:.4f}")
        print(f"   Improvement: {((best_acc - self.base_accuracy) / self.base_accuracy)*100:+.2f}%")
        
        # Print sample usage summary
        print(f"\n{'='*70}")
        print("SAMPLE USAGE SUMMARY")
        print(f"{'='*70}\n")
        
        if self.X_test is not None and self.y_test is not None:
            total_test = len(self.y_test)
            total_train = len(self.y_test) / 0.3 * 0.7  # Approximate train size
            total_samples = int(total_train + total_test)
            
            print(f"Total Samples Used: {total_samples}")
            print(f"  - Training: {int(total_train)} samples (70%)")
            print(f"  - Testing: {total_test} samples (30%)")
            
            print(f"\nTest Set Class Distribution:")
            print(f"  - Class 0 (Post-Fatigue): {np.sum(self.y_test == 0)} samples")
            print(f"  - Class 1 (Pre-Fatigue): {np.sum(self.y_test == 1)} samples")
            
            print(f"\nNote: Each 'sample' represents one EEG channel from one recording.")
            print(f"      For your dataset: 36 subjects × 2 conditions × 23 channels = 1,656 total samples")
            print(f"      → Training: ~1,159 samples (70%)")
            print(f"      → Testing: ~497 samples (30%)")
        
        print(f"\n{'='*70}\n")
    
    def plot_improvement_comparison(self):
        """Plot comparison of all methods"""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        methods = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in methods]
        
        # Sort by accuracy
        sorted_idx = np.argsort(accuracies)[::-1]
        methods = [methods[i] for i in sorted_idx]
        accuracies = [accuracies[i] for i in sorted_idx]
        
        # Color code
        colors = ['#e74c3c' if m == 'Baseline SVM' else '#2ecc71' for m in methods]
        
        bars = ax.barh(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add baseline line
        ax.axvline(self.base_accuracy, color='red', linestyle='--', linewidth=2, 
                  label=f'Baseline ({self.base_accuracy:.4f})', alpha=0.7)
        
        # Add values on bars
        for bar, acc in zip(bars, accuracies):
            improvement = ((acc - self.base_accuracy) / self.base_accuracy) * 100
            ax.text(acc + 0.005, bar.get_y() + bar.get_height()/2, 
                   f'{acc:.4f} ({improvement:+.1f}%)',
                   va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Advanced Methods Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(0.5, 1.0)
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'methods_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: methods_comparison.png")



# MAIN EXECUTION


if __name__ == "__main__":
    FEATURES_DIR = "features_output"
    OUTPUT_DIR = "ml_advanced_results"
    
    # Initialize advanced classifier
    advanced_clf = FatigueClassifier(FEATURES_DIR, OUTPUT_DIR)
    
    # Run complete advanced analysis
    advanced_clf.run_advanced_analysis()
    
    print("\n✓ Advanced ML analysis complete!")
    print(f"✓ Check {OUTPUT_DIR}/ for results")