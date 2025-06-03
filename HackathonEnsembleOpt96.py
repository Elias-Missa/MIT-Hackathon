"""
🚀 ENSEMBLE HYPERPARAMETER OPTIMIZATION FOR 96%!
=================================================
Taking the proven 95.87% ensemble solution from HackathonNN.py 
and adding extensive hyperparameter optimization to reach 96%

Strategy: Same ensemble (4 model types × 5-fold CV) + Hyperparameter tuning = 96%+
Expected: 95.87% → 96.0%+ with optimal hyperparameters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import category_encoders as ce
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from itertools import product
import random

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 ENSEMBLE HYPERPARAMETER OPTIMIZATION FOR 96%")
print(f"Device: {device}")
print("=" * 60)

def load_and_merge_data():
    """Load and merge travel and survey data - SAME AS HACKATHONNN.PY"""
    print("📊 Loading and merging datasets...")
    
    travel_train = pd.read_csv('Traveldata_train.csv')
    survey_train = pd.read_csv('Surveydata_train.csv')
    travel_test = pd.read_csv('Traveldata_test.csv')
    survey_test = pd.read_csv('Surveydata_test.csv')
    
    train_df = pd.merge(travel_train, survey_train, on='ID', how='inner')
    test_df = pd.merge(travel_test, survey_test, on='ID', how='inner')
    
    print(f"✅ Training data: {train_df.shape}")
    print(f"✅ Test data: {test_df.shape}")
    print(f"✅ Target distribution: {train_df['Overall_Experience'].value_counts().to_dict()}")
    
    return train_df, test_df

def create_advanced_features(df):
    """Create comprehensive engineered features - SAME AS HACKATHONNN.PY"""
    print("🔬 Creating advanced engineered features...")
    
    df = df.copy()
    original_cols = len(df.columns)
    
    # 1. DELAY ANALYTICS
    df['Total_Delay'] = df['Departure_Delay_in_Mins'].fillna(0) + df['Arrival_Delay_in_Mins'].fillna(0)
    df['Delay_Ratio'] = df['Arrival_Delay_in_Mins'].fillna(0) / (df['Departure_Delay_in_Mins'].fillna(0) + 1)
    df['Is_Delayed'] = (df['Total_Delay'] > 0).astype(int)
    df['Severe_Delay'] = (df['Total_Delay'] > df['Total_Delay'].quantile(0.9)).astype(int)
    df['Delay_Squared'] = df['Total_Delay'] ** 2
    df['Delay_Log'] = np.log1p(df['Total_Delay'])
    
    # 2. SERVICE QUALITY SCORES
    service_cols = ['Seat_Comfort', 'Catering', 'Platform_Location', 'Onboard_Wifi_Service', 
                   'Onboard_Entertainment', 'Online_Support', 'Ease_of_Online_Booking', 
                   'Onboard_Service', 'Legroom', 'Baggage_Handling', 'CheckIn_Service', 
                   'Cleanliness', 'Online_Boarding']
    
    rating_map = {'Poor': 1, 'Needs Improvement': 2, 'Acceptable': 3, 'Good': 4, 'Excellent': 5}
    
    score_cols = []
    for col in service_cols:
        if col in df.columns:
            df[f'{col}_Score'] = df[col].map(rating_map).fillna(df[col].map(rating_map).median())
            score_cols.append(f'{col}_Score')
    
    # Service quality aggregations
    df['Service_Quality_Mean'] = df[score_cols].mean(axis=1)
    df['Service_Quality_Std'] = df[score_cols].std(axis=1).fillna(0)
    df['Service_Quality_Min'] = df[score_cols].min(axis=1)
    df['Service_Quality_Max'] = df[score_cols].max(axis=1)
    df['Service_Quality_Range'] = df['Service_Quality_Max'] - df['Service_Quality_Min']
    df['Service_Quality_Median'] = df[score_cols].median(axis=1)
    
    # 3. CUSTOMER PROFILE
    df['Is_Business_Travel'] = (df['Type_Travel'] == 'Business travel').astype(int)
    df['Is_Loyal_Customer'] = (df['Customer_Type'] == 'Loyal Customer').astype(int)
    df['Is_Business_Class'] = (df['Travel_Class'] == 'Business').astype(int)
    df['Is_Female'] = (df['Gender'] == 'Female').astype(int)
    
    # Age processing
    df['Age_filled'] = df['Age'].fillna(df['Age'].median())
    df['Age_Squared'] = df['Age_filled'] ** 2
    df['Is_Senior'] = (df['Age_filled'] >= 60).astype(int)
    df['Is_Young'] = (df['Age_filled'] <= 30).astype(int)
    
    # 4. TRAVEL PATTERNS
    df['Distance_Log'] = np.log1p(df['Travel_Distance'])
    df['Distance_Squared'] = df['Travel_Distance'] ** 2
    df['Is_Long_Distance'] = (df['Travel_Distance'] > df['Travel_Distance'].quantile(0.75)).astype(int)
    df['Distance_Per_Delay'] = df['Travel_Distance'] / (df['Total_Delay'] + 1)
    
    # 5. INTERACTION FEATURES
    df['Business_x_Service'] = df['Is_Business_Travel'] * df['Service_Quality_Mean']
    df['Loyal_x_Service'] = df['Is_Loyal_Customer'] * df['Service_Quality_Mean']
    df['Class_x_Service'] = df['Is_Business_Class'] * df['Service_Quality_Mean']
    df['Age_x_Service'] = df['Age_filled'] * df['Service_Quality_Mean']
    df['Distance_x_Service'] = df['Travel_Distance'] * df['Service_Quality_Mean']
    
    # 6. COUNT FEATURES
    df['Excellent_Services_Count'] = (df[score_cols] == 5).sum(axis=1)
    df['Poor_Services_Count'] = (df[score_cols] == 1).sum(axis=1)
    df['Service_Excellence_Ratio'] = df['Excellent_Services_Count'] / (len(score_cols) + 1)
    
    new_cols = len(df.columns) - original_cols
    print(f"✅ Created {new_cols} new features")
    
    return df

def preprocess_data(train_df, test_df):
    """Comprehensive data preprocessing - SAME AS HACKATHONNN.PY"""
    print("🧹 Preprocessing data...")
    
    # Combine for consistent preprocessing
    combined_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    
    # Handle categorical variables
    categorical_cols = combined_df.select_dtypes(include=['object']).columns.tolist()
    if 'ID' in categorical_cols:
        categorical_cols.remove('ID')
    
    # Fill missing values
    for col in categorical_cols:
        if col != 'Overall_Experience':
            mode_val = combined_df[col].mode()
            if len(mode_val) > 0:
                combined_df[col] = combined_df[col].fillna(mode_val[0])
    
    # Handle numerical columns
    numerical_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numerical_cols:
        if col != 'Overall_Experience' and combined_df[col].isnull().sum() > 0:
            combined_df[col] = combined_df[col].fillna(combined_df[col].median())
    
    # Outlier handling
    delay_cols = ['Departure_Delay_in_Mins', 'Arrival_Delay_in_Mins', 'Total_Delay']
    for col in delay_cols:
        if col in combined_df.columns:
            q99 = combined_df[col].quantile(0.99)
            combined_df[col] = combined_df[col].clip(upper=q99)
    
    # Split back
    train_size = len(train_df)
    cleaned_train_df = combined_df[:train_size].copy()
    cleaned_test_df = combined_df[train_size:].copy()
    
    return cleaned_train_df, cleaned_test_df

def encode_features(train_df, test_df):
    """Encode features for neural networks - SAME AS HACKATHONNN.PY"""
    print("🔢 Encoding features...")
    
    # Separate features and target
    X_train = train_df.drop(['Overall_Experience'], axis=1)
    y_train = train_df['Overall_Experience']
    X_test = test_df.drop(['Overall_Experience'], axis=1, errors='ignore')
    
    # Store IDs
    test_ids = X_test['ID'].copy()
    X_train = X_train.drop(['ID'], axis=1)
    X_test = X_test.drop(['ID'], axis=1)
    
    # Target encoding for categorical variables
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        target_encoder = ce.TargetEncoder(cols=categorical_cols, smoothing=1.0)
        X_train = target_encoder.fit_transform(X_train, y_train)
        X_test = target_encoder.transform(X_test)
    
    print(f"✅ Encoded features: {X_train.shape[1]} total features")
    
    return X_train, X_test, y_train, test_ids

# ============ SAME MODEL ARCHITECTURES AS HACKATHONNN.PY ============

class DeepNeuralNetwork(nn.Module):
    """Deep neural network with advanced regularization - SAME AS HACKATHONNN.PY"""
    
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], dropout=0.3, use_batch_norm=True):
        super(DeepNeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

class ResidualNN(nn.Module):
    """Neural network with residual connections - SAME AS HACKATHONNN.PY"""
    
    def __init__(self, input_dim, hidden_dim=256, num_blocks=3, dropout=0.3):
        super(ResidualNN, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            )
            self.blocks.append(block)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        x = F.relu(self.input_bn(self.input_layer(x)))
        
        # Apply residual blocks
        for block in self.blocks:
            residual = x
            x = block(x)
            x = F.relu(x + residual)  # Residual connection
        
        return self.output_layer(x)

class WideDeepNN(nn.Module):
    """Wide & Deep neural network - SAME AS HACKATHONNN.PY"""
    
    def __init__(self, input_dim, deep_dims=[256, 128, 64], dropout=0.3):
        super(WideDeepNN, self).__init__()
        
        # Wide component (linear)
        self.wide = nn.Linear(input_dim, 1)
        
        # Deep component
        deep_layers = []
        prev_dim = input_dim
        
        for dim in deep_dims:
            deep_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        deep_layers.append(nn.Linear(prev_dim, 1))
        self.deep = nn.Sequential(*deep_layers)
        
        # Combine wide and deep
        self.output = nn.Linear(2, 1)
    
    def forward(self, x):
        wide_out = self.wide(x)
        deep_out = self.deep(x)
        combined = torch.cat([wide_out, deep_out], dim=1)
        return self.output(combined)

# ============ HYPERPARAMETER OPTIMIZATION FUNCTIONS ============

def get_hyperparameter_grid():
    """Define comprehensive hyperparameter search space for powerful PC"""
    return {
        # Learning rates
        'lr': [0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005],
        
        # Weight decay
        'weight_decay': [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4],
        
        # Batch sizes
        'batch_size': [128, 256, 512, 1024],
        
        # Dropout rates
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
        
        # Optimizers
        'optimizer': ['adamw', 'adam'],
        
        # Scheduler parameters
        'scheduler_patience': [5, 8, 10, 12, 15],
        'scheduler_factor': [0.3, 0.5, 0.6, 0.7, 0.8],
        
        # Architecture-specific parameters
        'DeepNN_hidden_dims': [
            [512, 256, 128, 64],      # Original successful
            [768, 384, 192, 96],      # Wider
            [512, 384, 256, 128, 64], # Deeper
            [1024, 512, 256, 128],    # Even wider
            [640, 320, 160, 80],      # Alternative scaling
            [512, 256, 128],          # Simpler
            [1024, 768, 512, 256, 128] # Very deep
        ],
        'ResidualNN_hidden_dim': [256, 320, 384, 448, 512, 640],
        'ResidualNN_num_blocks': [2, 3, 4, 5, 6],
        'WideDeepNN_deep_dims': [
            [256, 128, 64],
            [384, 192, 96],
            [512, 256, 128],
            [320, 160, 80],
            [512, 256, 128, 64],
            [768, 384, 192]
        ]
    }

def train_single_model_with_hyperparams(model_fn, model_params, train_params, 
                                       X_train, y_train, X_val, y_val, epochs=150):
    """Train a single model with specific hyperparameters"""
    
    # Create model
    model = model_fn(**model_params).to(device)
    
    # Setup optimizer
    if train_params['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), 
                               lr=train_params['lr'], 
                               weight_decay=train_params['weight_decay'])
    else:
        optimizer = optim.Adam(model.parameters(), 
                              lr=train_params['lr'], 
                              weight_decay=train_params['weight_decay'])
    
    # Setup scheduler
    scheduler = ReduceLROnPlateau(optimizer, 
                                 patience=train_params['scheduler_patience'], 
                                 factor=train_params['scheduler_factor'])
    
    # Prepare data
    X_train_tensor = torch.FloatTensor(X_train.values).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).to(device)
    X_val_tensor = torch.FloatTensor(X_val.values).to(device)
    y_val_tensor = torch.FloatTensor(y_val.values).to(device)
    
    # Data loader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'], shuffle=True)
    
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_acc = 0
    best_model_state = None
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor).squeeze()
            val_preds = torch.sigmoid(val_outputs) > 0.5
            val_acc = (val_preds == y_val_tensor).float().mean().item()
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_acc

def optimize_model_hyperparameters(model_name, model_fn, X_train, y_train, max_trials=50):
    """Optimize hyperparameters for a specific model type - EXTENSIVE SEARCH"""
    print(f"\n🔍 Optimizing hyperparameters for {model_name} ({max_trials} trials)...")
    
    hyperparams = get_hyperparameter_grid()
    
    # Create validation split
    X_opt_train, X_opt_val, y_opt_train, y_opt_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Scale features
    scaler = StandardScaler()
    X_opt_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_opt_train),
        columns=X_opt_train.columns
    )
    X_opt_val_scaled = pd.DataFrame(
        scaler.transform(X_opt_val),
        columns=X_opt_val.columns
    )
    
    best_score = 0
    best_params = None
    all_results = []
    
    for trial in range(max_trials):
        # Sample hyperparameters
        train_params = {
            'lr': random.choice(hyperparams['lr']),
            'weight_decay': random.choice(hyperparams['weight_decay']),
            'batch_size': random.choice(hyperparams['batch_size']),
            'optimizer': random.choice(hyperparams['optimizer']),
            'scheduler_patience': random.choice(hyperparams['scheduler_patience']),
            'scheduler_factor': random.choice(hyperparams['scheduler_factor'])
        }
        
        # Model-specific parameters
        if model_name == 'DeepNN_Large':
            model_params = {
                'input_dim': X_opt_train_scaled.shape[1],
                'hidden_dims': random.choice(hyperparams['DeepNN_hidden_dims']),
                'dropout': random.choice(hyperparams['dropout'])
            }
        elif model_name == 'DeepNN_Deep':
            model_params = {
                'input_dim': X_opt_train_scaled.shape[1],
                'hidden_dims': [256, 256, 256, 128, 64],  # Fixed deep architecture
                'dropout': random.choice(hyperparams['dropout'])
            }
        elif model_name == 'ResidualNN':
            model_params = {
                'input_dim': X_opt_train_scaled.shape[1],
                'hidden_dim': random.choice(hyperparams['ResidualNN_hidden_dim']),
                'num_blocks': random.choice(hyperparams['ResidualNN_num_blocks']),
                'dropout': random.choice(hyperparams['dropout'])
            }
        elif model_name == 'WideDeepNN':
            model_params = {
                'input_dim': X_opt_train_scaled.shape[1],
                'deep_dims': random.choice(hyperparams['WideDeepNN_deep_dims']),
                'dropout': random.choice(hyperparams['dropout'])
            }
        
        print(f"  Trial {trial+1:2d}/{max_trials}: lr={train_params['lr']:.4f}, wd={train_params['weight_decay']:.0e}, dropout={model_params['dropout']:.1f}")
        
        try:
            model, score = train_single_model_with_hyperparams(
                model_fn, model_params, train_params,
                X_opt_train_scaled, y_opt_train, X_opt_val_scaled, y_opt_val,
                epochs=100  # Reduced for hyperparameter search
            )
            
            all_results.append({
                'trial': trial + 1,
                'score': score,
                'model_params': model_params.copy(),
                'train_params': train_params.copy()
            })
            
            print(f"    → Score: {score:.4f} ({score*100:.2f}%)")
            
            if score > best_score:
                best_score = score
                best_params = {
                    'model_params': model_params,
                    'train_params': train_params
                }
                print(f"    🎯 NEW BEST: {best_score:.4f} ({best_score*100:.2f}%)")
        
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            continue
    
    print(f"  Final best {model_name} score: {best_score:.4f} ({best_score*100:.2f}%)")
    
    # Show top 5 results
    sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)[:5]
    print(f"  Top 5 configurations:")
    for i, result in enumerate(sorted_results):
        print(f"    {i+1}. Score: {result['score']:.4f}, lr: {result['train_params']['lr']:.4f}")
    
    return best_params, best_score, all_results

def train_optimized_ensemble(X_train, y_train, optimized_hyperparams):
    """Train ensemble with optimized hyperparameters using 5-fold CV - SAME AS HACKATHONNN.PY"""
    print("\n🧠 Training optimized ensemble with 5-fold CV...")
    
    input_dim = X_train.shape[1]
    
    # Model definitions with optimized parameters
    def create_model_fn(model_class, params):
        def model_fn():
            return model_class(**params['model_params'])
        return model_fn, params['train_params']
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_models = {}
    all_results = {}
    
    for name in optimized_hyperparams.keys():
        print(f"\n🔄 Training optimized {name}...")
        
        params = optimized_hyperparams[name]
        model_fn, train_params = create_model_fn(
            DeepNeuralNetwork if 'DeepNN' in name else 
            ResidualNN if name == 'ResidualNN' else WideDeepNN,
            params
        )
        
        fold_scores = []
        fold_models = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_fold_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_fold_train),
                columns=X_fold_train.columns
            )
            X_fold_val_scaled = pd.DataFrame(
                scaler.transform(X_fold_val),
                columns=X_fold_val.columns
            )
            
            # Train model with optimized hyperparameters
            model, val_acc = train_single_model_with_hyperparams(
                model_fn, {}, train_params,  # model_params already in model_fn
                X_fold_train_scaled, y_fold_train,
                X_fold_val_scaled, y_fold_val,
                epochs=150  # Full training for final ensemble
            )
            
            fold_scores.append(val_acc)
            fold_models.append((model, scaler))
            
            print(f"  Fold {fold + 1}: {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        mean_accuracy = np.mean(fold_scores)
        std_accuracy = np.std(fold_scores)
        
        all_models[name] = fold_models
        all_results[name] = {
            'fold_scores': fold_scores,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy
        }
        
        print(f"  Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f} ({mean_accuracy*100:.2f}%)")
    
    return all_models, all_results

def generate_optimized_predictions(models, X_test, test_ids):
    """Generate ensemble predictions from optimized models - SAME AS HACKATHONNN.PY"""
    print("\n🎯 Generating optimized ensemble predictions...")
    
    if not models:
        print("⚠️ No models available! Using simple baseline predictions...")
        # Fallback: predict based on majority class
        final_predictions = np.ones(len(test_ids), dtype=int)  # Predict positive (common in satisfaction)
        
        submission = pd.DataFrame({
            'ID': test_ids,
            'Overall_Experience': final_predictions
        })
        
        submission.to_csv('hackathon_ensemble_opt96_submission.csv', index=False)
        print(f"✅ Saved fallback predictions to 'hackathon_ensemble_opt96_submission.csv'")
        print(f"📊 Fallback prediction distribution: Positive={len(final_predictions)}, Negative=0")
        print(f"📊 Positive ratio: 1.0000")
        
        return submission
    
    all_predictions = []
    
    for name, fold_models in models.items():
        model_preds = []
        
        for model, scaler in fold_models:
            # Scale test features
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns
            )
            
            # Make predictions
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test_scaled.values).to(device)
                outputs = model(X_test_tensor).squeeze()
                probs = torch.sigmoid(outputs).cpu().numpy()
                model_preds.append(probs)
        
        # Average across folds
        model_avg_preds = np.mean(model_preds, axis=0)
        all_predictions.append(model_avg_preds)
        print(f"  {name}: Generated predictions (mean prob: {model_avg_preds.mean():.3f})")
    
    # Ensemble prediction (equal weights)
    final_probs = np.mean(all_predictions, axis=0)
    final_predictions = (final_probs > 0.5).astype(int)
    
    # Create submission
    submission = pd.DataFrame({
        'ID': test_ids,
        'Overall_Experience': final_predictions
    })
    
    submission.to_csv('hackathon_ensemble_opt96_submission.csv', index=False)
    print(f"✅ Saved predictions to 'hackathon_ensemble_opt96_submission.csv'")
    
    # Safe prediction distribution printing
    try:
        pred_counts = np.bincount(final_predictions)
        if len(pred_counts) >= 2:
            print(f"📊 Final prediction distribution: Negative={pred_counts[0]}, Positive={pred_counts[1]}")
        else:
            print(f"📊 Final prediction distribution: Only class {pred_counts.argmax()} predicted")
        print(f"📊 Positive ratio: {final_predictions.mean():.4f}")
    except Exception as e:
        print(f"📊 Prediction summary: {len(final_predictions)} predictions generated")
    
    return submission

def plot_optimization_results(optimization_results, ensemble_results):
    """Plot comprehensive optimization and ensemble results"""
    print("\n📊 Creating comprehensive result plots...")
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Hyperparameter optimization scores
    ax1 = plt.subplot(2, 3, 1)
    if optimization_results:
        model_names = list(optimization_results.keys())
        opt_scores = [optimization_results[name][1] for name in model_names]
        
        bars = ax1.bar(model_names, opt_scores, alpha=0.7, color='skyblue')
        ax1.set_title('Hyperparameter Optimization Results', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Best Validation Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, score in zip(bars, opt_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Ensemble cross-validation results
    ax2 = plt.subplot(2, 3, 2)
    if ensemble_results:
        model_names = list(ensemble_results.keys())
        means = [ensemble_results[name]['mean_accuracy'] for name in model_names]
        stds = [ensemble_results[name]['std_accuracy'] for name in model_names]
        
        bars = ax2.barh(model_names, means, xerr=stds, capsize=5, alpha=0.7, color='lightgreen')
        ax2.set_title('5-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Accuracy')
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax2.text(mean + std + 0.001, i, f'{mean:.4f}±{std:.4f}', 
                    va='center', fontsize=10, fontweight='bold')
    
    # 3. Fold-wise performance
    ax3 = plt.subplot(2, 3, 3)
    if ensemble_results:
        for name, result in ensemble_results.items():
            ax3.plot(range(1, 6), result['fold_scores'], 'o-', label=name, linewidth=2, markersize=6)
        ax3.set_title('Fold-wise Performance', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Fold')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(alpha=0.3)
    
    # 4. Hyperopt vs Ensemble comparison
    ax4 = plt.subplot(2, 3, 4)
    if optimization_results and ensemble_results:
        comparison_data = []
        for name in model_names:
            if name in optimization_results and name in ensemble_results:
                comparison_data.append({
                    'Model': name,
                    'Hyperopt_Score': optimization_results[name][1],
                    'Ensemble_Score': ensemble_results[name]['mean_accuracy']
                })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            x_pos = np.arange(len(comp_df))
            width = 0.35
            
            ax4.bar(x_pos - width/2, comp_df['Hyperopt_Score'], width, 
                   label='Hyperopt Validation', alpha=0.7, color='orange')
            ax4.bar(x_pos + width/2, comp_df['Ensemble_Score'], width, 
                   label='5-Fold CV', alpha=0.7, color='green')
            
            ax4.set_title('Hyperopt vs Ensemble Performance', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Accuracy')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(comp_df['Model'], rotation=45)
            ax4.legend()
    
    # 5. Score distribution
    ax5 = plt.subplot(2, 3, 5)
    if optimization_results:
        all_scores = []
        for name, (params, best_score, all_results) in optimization_results.items():
            if all_results:
                scores = [r['score'] for r in all_results]
                all_scores.extend(scores)
                ax5.hist(scores, alpha=0.6, label=name, bins=10)
        
        ax5.set_title('Hyperparameter Search Score Distributions', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Validation Score')
        ax5.set_ylabel('Frequency')
        ax5.legend()
    
    # 6. Performance summary table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')
    
    if ensemble_results:
        # Create summary table
        table_data = []
        for name, result in ensemble_results.items():
            table_data.append([
                name,
                f"{result['mean_accuracy']:.4f}",
                f"{result['std_accuracy']:.4f}",
                f"{max(result['fold_scores']):.4f}",
                f"{min(result['fold_scores']):.4f}"
            ])
        
        table = ax6.table(cellText=table_data,
                         colLabels=['Model', 'Mean', 'Std', 'Max', 'Min'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax6.set_title('Performance Summary Table', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('ensemble_hyperopt96_comprehensive_results.png', dpi=300, bbox_inches='tight')
    print("📊 Saved comprehensive plots to 'ensemble_hyperopt96_comprehensive_results.png'")

# Main execution
if __name__ == "__main__":
    print("🎯 ENSEMBLE HYPERPARAMETER OPTIMIZATION FOR 96%")
    print("Taking the proven 95.87% ensemble and optimizing it to 96%+")
    print("=" * 60)
    
    # Load and prepare data (SAME AS HACKATHONNN.PY)
    train_df, test_df = load_and_merge_data()
    
    # Feature engineering (SAME AS HACKATHONNN.PY)
    train_df = create_advanced_features(train_df)
    test_df = create_advanced_features(test_df)
    
    # Preprocessing (SAME AS HACKATHONNN.PY)
    train_clean, test_clean = preprocess_data(train_df, test_df)
    
    # Feature encoding (SAME AS HACKATHONNN.PY)
    X_train, X_test, y_train, test_ids = encode_features(train_clean, test_clean)
    
    # ============ HYPERPARAMETER OPTIMIZATION PHASE ============
    print("\n🔍 PHASE 1: EXTENSIVE HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print("This will take time but will find the optimal hyperparameters!")
    
    # Model definitions for optimization
    model_functions = {
        'DeepNN_Large': DeepNeuralNetwork,
        'DeepNN_Deep': DeepNeuralNetwork,
        'ResidualNN': ResidualNN,
        'WideDeepNN': WideDeepNN
    }
    
    # Optimize each model type extensively
    optimization_results = {}
    for name, model_fn in model_functions.items():
        best_params, best_score, all_results = optimize_model_hyperparameters(
            name, model_fn, X_train, y_train, max_trials=50  # EXTENSIVE SEARCH
        )
        optimization_results[name] = (best_params, best_score, all_results)
    
    # ============ ENSEMBLE TRAINING PHASE ============
    print("\n🧠 PHASE 2: OPTIMIZED ENSEMBLE TRAINING")
    print("=" * 60)
    
    # Extract best hyperparameters
    optimized_hyperparams = {
        name: params for name, (params, score, results) in optimization_results.items() 
        if params is not None
    }
    
    # Train ensemble with optimized hyperparameters
    ensemble_models, ensemble_results = train_optimized_ensemble(
        X_train, y_train, optimized_hyperparams
    )
    
    # ============ PREDICTION GENERATION ============
    print("\n🎯 PHASE 3: FINAL OPTIMIZED PREDICTIONS")
    print("=" * 60)
    
    # Generate predictions
    submission = generate_optimized_predictions(ensemble_models, X_test, test_ids)
    
    # Create comprehensive plots
    plot_optimization_results(optimization_results, ensemble_results)
    
    # ============ FINAL SUMMARY ============
    print(f"\n🏆 ENSEMBLE HYPERPARAMETER OPTIMIZATION COMPLETE!")
    print("=" * 70)
    
    if ensemble_results:
        # Calculate ensemble average
        ensemble_scores = [result['mean_accuracy'] for result in ensemble_results.values()]
        overall_ensemble_score = np.mean(ensemble_scores)
        overall_ensemble_std = np.std(ensemble_scores)
        
        print(f"🎯 Overall Ensemble Score: {overall_ensemble_score:.4f} ± {overall_ensemble_std:.4f}")
        print(f"🎯 Percentage: {overall_ensemble_score*100:.2f}% ± {overall_ensemble_std*100:.2f}%")
        print(f"🎯 Target: 96.00%")
        
        if overall_ensemble_score >= 0.96:
            print("\n🎉 MISSION ACCOMPLISHED! 96%+ ACHIEVED! 🎉")
        else:
            improvement = (overall_ensemble_score - 0.9587) * 100
            print(f"\n📈 Improvement from baseline 95.87%: +{improvement:.2f}%")
            if overall_ensemble_score > 0.9587:
                print("✅ Successfully improved the baseline!")
                remaining = (0.96 - overall_ensemble_score) * 100
                print(f"📊 Only {remaining:.2f}% away from 96% target!")
            
        print("\n📊 Hyperparameter Optimization Results:")
        for name, (params, score, results) in optimization_results.items():
            if params:
                print(f"  {name:15s}: {score:.4f} ({score*100:.2f}%) - {len(results)} trials")
        
        print("\n📊 Final Ensemble Performance:")
        for name, result in ensemble_results.items():
            mean_acc = result['mean_accuracy']
            std_acc = result['std_accuracy']
            print(f"  {name:15s}: {mean_acc:.4f} ± {std_acc:.4f} ({mean_acc*100:.2f}%)")
    
    print(f"\n📁 Files generated:")
    print(f"  • hackathon_ensemble_opt96_submission.csv (OPTIMIZED PREDICTIONS)")
    print(f"  • ensemble_hyperopt96_comprehensive_results.png (DETAILED PLOTS)")
    
    print(f"\n🚀 READY FOR 96%+ SUBMISSION!")
    print("🔥 This optimized ensemble should push beyond the 95.87% baseline!") 