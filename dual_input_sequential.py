import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import logging
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Suppress warnings and configure logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#########################################
# Data Loading Functions
#########################################
def load_engineered_features(db_path='bybit_data.db', table_name="training_set"):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df

def load_sequential_data(db_path='bybit_data.db', table_name="custom_5min"):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    return df

#########################################
# Scaler Fitting Function
#########################################
def fit_scalers(df_eng, df_seq, seq_feature_cols):
    eng_cols = df_eng.columns.drop('date')
    X_eng = df_eng[eng_cols].values.astype(np.float32)
    eng_scaler = StandardScaler()
    eng_scaler.fit(X_eng)
    
    X_seq = df_seq[seq_feature_cols].values.astype(np.float32)
    seq_scaler = StandardScaler()
    seq_scaler.fit(X_seq)
    
    return eng_scaler, seq_scaler

#########################################
# Custom Dataset for Dual Input with Date Filtering
#########################################
class DualInputTradingDataset(Dataset):
    def __init__(self, db_path='bybit_data.db', seq_expected=288, date_range=None, 
                 eng_scaler=None, seq_scaler=None):
        """
        Loads engineered features from training_set and sequential 5‑min data from custom_5min.
        Optionally filters by date_range (a tuple of start_date and end_date), and applies normalization.
        """
        self.db_path = db_path
        self.seq_expected = seq_expected
        
        self.df_eng = load_engineered_features(db_path)
        self.df_seq = load_sequential_data(db_path)
        
        if date_range is not None:
            start_date, end_date = date_range
            self.df_eng = self.df_eng[(self.df_eng['date'] >= start_date) & (self.df_eng['date'] <= end_date)]
        
        self.dates = np.sort(self.df_eng['date'].unique())
        self.seq_feature_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'open_interest']
        
        # Fit scalers if not provided
        if eng_scaler is None or seq_scaler is None:
            self.eng_scaler, self.seq_scaler = fit_scalers(self.df_eng, self.df_seq, self.seq_feature_cols)
        else:
            self.eng_scaler = eng_scaler
            self.seq_scaler = seq_scaler
        
        # Normalize engineered features (all columns except 'date')
        eng_cols = self.df_eng.columns.drop('date')
        eng_features = self.df_eng[eng_cols].values.astype(np.float32)
        self.df_eng.loc[:, eng_cols] = self.eng_scaler.transform(eng_features)
        
        # Normalize sequential features
        self.df_seq.loc[:, self.seq_feature_cols] = self.seq_scaler.transform(
            self.df_seq[self.seq_feature_cols].values.astype(np.float32)
        )
        
        # Debug: Print a sample of normalized sequential data for the first available date
        sample_date = self.dates[0]
        sample_seq = self.df_seq[self.df_seq['date'] == sample_date][self.seq_feature_cols].head()
        logger.info(f"Sample normalized sequential data for {sample_date}:\n{sample_seq}")

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        day = self.dates[idx]
        day_str = str(day)
        
        eng_row = self.df_eng[self.df_eng['date'] == day]
        if eng_row.empty:
            raise ValueError(f"No engineered features for date {day}")
        eng_data = eng_row.drop(columns=['date']).iloc[0]
        target_val = eng_data['target']
        if isinstance(target_val, str):
            target_val = 1.0 if target_val.strip().upper() == "TRUE" else 0.0
        else:
            target_val = 1.0 if float(target_val) >= 0.5 else 0.0
        eng_features = torch.tensor(eng_data.drop(labels=['target']).values, dtype=torch.float32)
        
        seq_data = self.df_seq[self.df_seq['date'] == day].sort_values('datetime')
        seq_values = seq_data[self.seq_feature_cols].values.astype(np.float32)
        if len(seq_values) < self.seq_expected:
            pad_count = self.seq_expected - len(seq_values)
            pad_array = np.repeat(seq_values[-1].reshape(1, -1), pad_count, axis=0)
            seq_values = np.vstack([seq_values, pad_array])
        elif len(seq_values) > self.seq_expected:
            seq_values = seq_values[:self.seq_expected]
        seq_features = torch.tensor(seq_values, dtype=torch.float32)
        
        target = torch.tensor(target_val, dtype=torch.float32)
        return day_str, seq_features, eng_features, target

#########################################
# Dual-Input Model Definition with Dropout
#########################################
class DualInputModel(nn.Module):
    def __init__(self, seq_input_dim, seq_hidden_dim, seq_num_layers, eng_input_dim, fc_hidden_dim, dropout_prob=0.2):
        super(DualInputModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=seq_hidden_dim,
            num_layers=seq_num_layers,
            batch_first=True
        )
        # Engineered feature branch with dropout
        self.fc_eng = nn.Sequential(
            nn.Linear(eng_input_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(fc_hidden_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        # Combined branch with dropout
        self.fc_combined = nn.Sequential(
            nn.Linear(seq_hidden_dim + fc_hidden_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(fc_hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, seq_input, eng_input):
        lstm_out, (hn, _) = self.lstm(seq_input)
        seq_rep = hn[-1]  # Last layer's hidden state
        eng_rep = self.fc_eng(eng_input)
        combined = torch.cat((seq_rep, eng_rep), dim=1)
        # Diagnostic logging: print pre-activation statistics for the first batch only
        pre_activation = self.fc_combined[0](combined)
        logger.info(f"Pre-activation mean: {pre_activation.mean().item():.4f}, std: {pre_activation.std().item():.4f}")
        output = self.fc_combined(combined)
        return output

#########################################
# Training Routine with Epoch Summary
#########################################
def train_model(model, dataloader, criterion, optimizer, num_epochs=50, device='cpu'):
    model.to(device)
    epoch_summary = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for _, seq_input, eng_input, target in dataloader:
            seq_input = seq_input.to(device)
            eng_input = eng_input.to(device)
            target = target.to(device).view(-1, 1)
            optimizer.zero_grad()
            outputs = model(seq_input, eng_input)
            loss = criterion(outputs, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item() * seq_input.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {epoch_loss:.4f}")
            epoch_summary.append((epoch+1, epoch_loss))
    if epoch_summary:
        df_summary = pd.DataFrame(epoch_summary, columns=["Epoch", "Average_Loss"])
        logger.info("Epoch Summary (every 10 epochs):\n" + df_summary.to_string(index=False))
    return model

#########################################
# Main Routine: Training and Testing Split
#########################################
def main():
    db_path = 'bybit_data.db'
    
    # Define training and testing date ranges.
    training_date_range = (pd.to_datetime('2023-02-01').date(), pd.to_datetime('2024-02-01').date())
    testing_date_range  = (pd.to_datetime('2024-02-01').date(), pd.to_datetime('2025-02-01').date())
    
    # Create datasets.
    training_dataset = DualInputTradingDataset(db_path=db_path, seq_expected=288, date_range=training_date_range)
    testing_dataset  = DualInputTradingDataset(db_path=db_path, seq_expected=288, date_range=testing_date_range)
    
    # Create DataLoaders.
    train_loader = DataLoader(training_dataset, batch_size=4, shuffle=True)
    test_loader  = DataLoader(testing_dataset, batch_size=4, shuffle=False)
    
    # Determine input dimensions.
    seq_input_dim = 7  # (open, high, low, close, volume, turnover, open_interest)
    eng_input_dim = training_dataset.df_eng.drop(columns=['date', 'target']).shape[1]
    seq_hidden_dim = 64
    seq_num_layers = 2
    fc_hidden_dim = 32
    
    model = DualInputModel(seq_input_dim, seq_hidden_dim, seq_num_layers, eng_input_dim, fc_hidden_dim, dropout_prob=0.2)
    logger.info(model)
    
    # Compute class weights from training data.
    targets = training_dataset.df_eng['target'].apply(lambda x: 1 if str(x).strip().upper() == "TRUE" or float(x) >= 0.5 else 0)
    count_0 = (targets == 0).sum()
    count_1 = (targets == 1).sum()
    logger.info(f"Training set class distribution: 0's: {count_0}, 1's: {count_1}")
    # Use a simple weight for the positive class:
    weight_for_one = count_0 / count_1 if count_1 != 0 else 1.0
    class_weight = torch.tensor([weight_for_one], dtype=torch.float32)
    
    # Use BCE loss with class weight.
    criterion = nn.BCELoss(weight=class_weight)
    # Increase learning rate to help escape trivial solutions.
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    logger.info("Starting training on 2023-02-01 to 2024-02-01 data...")
    trained_model = train_model(model, train_loader, criterion, optimizer, num_epochs=50, device='cpu')
    
    # Evaluate on the testing dataset.
    trained_model.eval()
    predictions = []
    dates = []
    actuals = []
    with torch.no_grad():
        for batch_date, seq_input, eng_input, target in test_loader:
            outputs = trained_model(seq_input.to('cpu'), eng_input.to('cpu'))
            predictions.extend(outputs.squeeze().cpu().numpy())
            dates.extend(batch_date)
            actuals.extend(target.cpu().numpy())
    
    df_results = pd.DataFrame({
        'date': dates,
        'prediction': predictions,
        'predicted_flag': (np.array(predictions) >= 0.5).astype(int),
        'actual_SND': actuals
    })
    
    conn = sqlite3.connect(db_path)
    df_results.to_sql('predictions_testing', conn, if_exists='replace', index=False)
    conn.close()
    logger.info("Testing predictions successfully saved to database table 'predictions_testing'.")

if __name__ == "__main__":
    main()
