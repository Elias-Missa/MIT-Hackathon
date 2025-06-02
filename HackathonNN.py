"""
🚄 HACKATHON NEURAL NETWORK - Shinkansen Satisfaction Prediction
================================================================
Target: Maximum Accuracy with Deep Learning

Robust neural network pipeline featuring:
- Advanced feature engineering (50+ features)
- Multiple proven neural network architectures
- Comprehensive preprocessing and normalization
- Ensemble neural networks with cross-validation
- Hyperparameter optimization and regularization

Author: Expert Deep Learning Engineer
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

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚄 HACKATHON NEURAL NETWORK PIPELINE")
print(f"Device: {device}")
print("=" * 60)

def load_and_merge_data():
    """Load and merge travel and survey data"""
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
    """Create comprehensive engineered features"""
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
    """Comprehensive data preprocessing"""
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
    """Encode features for neural networks"""
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

class DeepNeuralNetwork(nn.Module):
    """Deep neural network with advanced regularization"""
    
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
    """Neural network with residual connections"""
    
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
    """Wide & Deep neural network"""
    
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

def train_neural_network(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=512, lr=0.001):
    """Train neural network with early stopping"""
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train.values).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).to(device)
    X_val_tensor = torch.FloatTensor(X_val.values).to(device)
    y_val_tensor = torch.FloatTensor(y_val.values).to(device)
    
    # Data loader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=False)
    
    best_val_acc = 0
    best_model_state = None
    patience = 15
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
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: Val Acc = {val_acc:.4f}")
        
        if patience_counter >= patience:
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model, best_val_acc

def train_ensemble_models(X_train, y_train):
    """Train ensemble of neural networks"""
    print("\n🧠 Training ensemble of neural networks...")
    
    input_dim = X_train.shape[1]
    models = {}
    results = {}
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    architectures = {
        'DeepNN_Large': lambda: DeepNeuralNetwork(input_dim, [512, 256, 128, 64]),
        'DeepNN_Deep': lambda: DeepNeuralNetwork(input_dim, [256, 256, 256, 128, 64]),
        'ResidualNN': lambda: ResidualNN(input_dim),
        'WideDeepNN': lambda: WideDeepNN(input_dim)
    }
    
    for name, model_fn in architectures.items():
        print(f"\n🔄 Training {name}...")
        
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
                columns=X_fold_train.columns,
                index=X_fold_train.index
            )
            X_fold_val_scaled = pd.DataFrame(
                scaler.transform(X_fold_val),
                columns=X_fold_val.columns,
                index=X_fold_val.index
            )
            
            # Train model
            model = model_fn()
            trained_model, val_acc = train_neural_network(
                model, X_fold_train_scaled, y_fold_train, 
                X_fold_val_scaled, y_fold_val
            )
            
            fold_scores.append(val_acc)
            fold_models.append((trained_model, scaler))
            
            print(f"  Fold {fold + 1}: {val_acc:.4f}")
        
        mean_accuracy = np.mean(fold_scores)
        std_accuracy = np.std(fold_scores)
        
        models[name] = fold_models
        results[name] = {
            'fold_scores': fold_scores,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy
        }
        
        print(f"  Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    
    return models, results

def generate_predictions(models, X_test, test_ids):
    """Generate ensemble predictions"""
    print("\n🎯 Generating ensemble predictions...")
    
    all_predictions = []
    
    for name, fold_models in models.items():
        model_preds = []
        
        for model, scaler in fold_models:
            # Scale test features
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
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
        print(f"  {name}: Generated predictions")
    
    # Ensemble prediction (equal weights)
    final_probs = np.mean(all_predictions, axis=0)
    final_predictions = (final_probs > 0.5).astype(int)
    
    # Create submission
    submission = pd.DataFrame({
        'ID': test_ids,
        'Overall_Experience': final_predictions
    })
    
    submission.to_csv('hackathon_nn_submission.csv', index=False)
    print(f"✅ Saved predictions to 'hackathon_nn_submission.csv'")
    
    return submission

def plot_results(results):
    """Plot model performance comparison"""
    print("\n📊 Model Performance Analysis:")
    print("=" * 50)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_accuracy'], reverse=True)
    
    for name, result in sorted_results:
        print(f"{name:15s}: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")
    
    # Create performance plot
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    names = [name for name, _ in sorted_results]
    means = [result['mean_accuracy'] for _, result in sorted_results]
    stds = [result['std_accuracy'] for _, result in sorted_results]
    
    bars = plt.barh(names, means, xerr=stds, capsize=5, alpha=0.7)
    plt.xlabel('Accuracy')
    plt.title('Neural Network Model Comparison')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{mean:.4f}', ha='left', va='center', fontsize=10)
    
    plt.subplot(1, 2, 2)
    for name, result in results.items():
        plt.plot(result['fold_scores'], 'o-', label=name, alpha=0.7, linewidth=2)
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Cross-Validation Performance')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('neural_network_performance.png', dpi=300, bbox_inches='tight')
    print("📊 Saved performance plot to 'neural_network_performance.png'")

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    train_df, test_df = load_and_merge_data()
    
    # Feature engineering
    train_df = create_advanced_features(train_df)
    test_df = create_advanced_features(test_df)
    
    # Preprocessing
    train_clean, test_clean = preprocess_data(train_df, test_df)
    
    # Feature encoding
    X_train, X_test, y_train, test_ids = encode_features(train_clean, test_clean)
    
    # Train ensemble models
    models, results = train_ensemble_models(X_train, y_train)
    
    # Generate predictions
    submission = generate_predictions(models, X_test, test_ids)
    
    # Analyze performance
    plot_results(results)
    
    # Print final summary
    best_model = max(results.keys(), key=lambda k: results[k]['mean_accuracy'])
    best_accuracy = results[best_model]['mean_accuracy']
    
    print(f"\n🏆 FINAL SUMMARY:")
    print(f"🏆 Best Model: {best_model}")
    print(f"🏆 Best CV Accuracy: {best_accuracy:.4f} ± {results[best_model]['std_accuracy']:.4f}")
    print(f"🎯 Predictions saved to: hackathon_nn_submission.csv") 