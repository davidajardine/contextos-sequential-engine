import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
def load_engineered_features(db_path='bybit_data.db'):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM training_set", conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df

def load_sequential_data(db_path='bybit_data.db'):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM custom_5min", conn)
    conn.close()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    return df

#########################################
# Scaler Fitting Function
#########################################
def fit_scalers(df_eng, df_seq, seq_feature_cols):
    # For engineered features: drop non-numeric columns ('date' and 'target')
    eng_cols = df_eng.columns.drop('date')
    X_eng = df_eng[eng_cols].values.astype(np.float32)
    eng_scaler = StandardScaler()
    eng_scaler.fit(X_eng)
    
    # For sequential features: use specified columns
    X_seq = df_seq[seq_feature_cols].values.astype(np.float32)
    seq_scaler = StandardScaler()
    seq_scaler.fit(X_seq)
    
    return eng_scaler, seq_scaler

#########################################
# Custom Dataset for Dual Input
#########################################
class DualInputTradingDataset(Dataset):
    def __init__(self, db_path='bybit_data.db', seq_expected=288, eng_scaler=None, seq_scaler=None):
        """
        Loads:
          - Engineered features from training_set (one row per trading day).
          - Sequential 5-min data from custom_5min.
        Applies normalization using provided or fitted scalers.
        """
        self.db_path = db_path
        self.seq_expected = seq_expected
        
        # Load data
        self.df_eng = load_engineered_features(db_path)
        self.df_seq = load_sequential_data(db_path)
        self.dates = np.sort(self.df_eng['date'].unique())
        
        # Sequential data columns
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
        
        # Normalize sequential features for the entire df_seq
        self.df_seq.loc[:, self.seq_feature_cols] = self.seq_scaler.transform(
            self.df_seq[self.seq_feature_cols].values.astype(np.float32)
        )

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        day = self.dates[idx]
        # Convert day to string for proper collation
        day_str = str(day)
        
        # Retrieve engineered features for the day
        eng_row = self.df_eng[self.df_eng['date'] == day]
        if eng_row.empty:
            raise ValueError(f"No engineered features for date {day}")
        eng_data = eng_row.drop(columns=['date']).iloc[0]
        # Convert the 'target' value to numeric 0/1
        target_val = eng_data['target']
        if isinstance(target_val, str):
            target_val = 1.0 if target_val.strip().upper() == "TRUE" else 0.0
        else:
            target_val = 1.0 if float(target_val) >= 0.5 else 0.0
        # Drop the target from engineered features for input
        eng_features = torch.tensor(eng_data.drop(labels=['target']).values, dtype=torch.float32)
        
        # Retrieve sequential 5-min data for this day, sorted by datetime
        seq_data = self.df_seq[self.df_seq['date'] == day].sort_values('datetime')
        seq_values = seq_data[self.seq_feature_cols].values.astype(np.float32)
        # Pad or truncate to a fixed length (seq_expected)
        if len(seq_values) < self.seq_expected:
            pad_count = self.seq_expected - len(seq_values)
            pad_array = np.repeat(seq_values[-1].reshape(1, -1), pad_count, axis=0)
            seq_values = np.vstack([seq_values, pad_array])
        elif len(seq_values) > self.seq_expected:
            seq_values = seq_values[:self.seq_expected]
        seq_features = torch.tensor(seq_values, dtype=torch.float32)
        
        target = torch.tensor(target_val, dtype=torch.float32)
        
        # Return the date (as string), sequential features, engineered features, and target
        return day_str, seq_features, eng_features, target

#########################################
# Dual-Input Model Definition
#########################################
class DualInputModel(nn.Module):
    def __init__(self, seq_input_dim, seq_hidden_dim, seq_num_layers, eng_input_dim, fc_hidden_dim):
        super(DualInputModel, self).__init__()
        # LSTM branch for sequential 5-min data
        self.lstm = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=seq_hidden_dim,
            num_layers=seq_num_layers,
            batch_first=True
        )
        # Feed-forward branch for engineered features
        self.fc_eng = nn.Sequential(
            nn.Linear(eng_input_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, fc_hidden_dim),
            nn.ReLU()
        )
        # Combined fully connected layers for final prediction
        self.fc_combined = nn.Sequential(
            nn.Linear(seq_hidden_dim + fc_hidden_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, seq_input, eng_input):
        # Process sequential input with LSTM
        lstm_out, (hn, _) = self.lstm(seq_input)
        seq_features = hn[-1]  # Use last hidden state from final LSTM layer
        
        # Process engineered features with feed-forward branch
        eng_features = self.fc_eng(eng_input)
        
        # Concatenate both features and pass through combined layers
        combined = torch.cat((seq_features, eng_features), dim=1)
        output = self.fc_combined(combined)
        return output

#########################################
# Training Routine with Epoch Summary & Gradient Monitoring
#########################################
def train_dual_model(model, dataloader, criterion, optimizer, num_epochs=50, device='cpu'):
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
            
            # Log gradient norm for first linear layer in fc_eng as an example
            for name, param in model.fc_eng[0].named_parameters():
                if param.grad is not None:
                    logger.info(f"Epoch {epoch+1} {name} grad norm: {param.grad.norm().item():.4f}")
                    break
                    
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item() * seq_input.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {epoch_loss:.4f}")
            epoch_summary.append((epoch+1, epoch_loss))
            
    if epoch_summary:
        df_summary = pd.DataFrame(epoch_summary, columns=["Epoch", "Average_Loss"])
        logger.info("Epoch Summary (every 10 epochs):")
        logger.info("\n" + df_summary.to_string(index=False))
    return model

#########################################
# Main Routine
#########################################
def main():
    db_path = 'bybit_data.db'
    dataset = DualInputTradingDataset(db_path=db_path, seq_expected=288)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Determine dimensions:
    seq_input_dim = 7  # For 5-min features: open, high, low, close, volume, turnover, open_interest
    # For engineered features: drop 'date' and 'target'
    eng_input_dim = dataset.df_eng.drop(columns=['date']).shape[1] - 1  
    seq_hidden_dim = 64
    seq_num_layers = 2
    fc_hidden_dim = 32
    
    model = DualInputModel(seq_input_dim, seq_hidden_dim, seq_num_layers, eng_input_dim, fc_hidden_dim)
    logger.info(model)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    trained_model = train_dual_model(model, dataloader, criterion, optimizer, num_epochs=50, device='cpu')
    
    # Evaluate the model and collect predictions with dates
    trained_model.eval()
    predictions_list = []
    dates_list = []
    actual_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch_date, seq_input, eng_input, target = batch
            outputs = trained_model(seq_input.to('cpu'), eng_input.to('cpu'))
            predictions_list.extend(outputs.squeeze().cpu().numpy())
            dates_list.extend(batch_date)
            actual_list.extend(target.cpu().numpy())
    
    # Build the results DataFrame
    df_results = pd.DataFrame({
        'date': dates_list,
        'prediction': predictions_list,
        'predicted_flag': (np.array(predictions_list) >= 0.5).astype(int),
        'actual_SND': actual_list
    })
    
    conn = sqlite3.connect(db_path)
    df_results.to_sql('predictions', conn, if_exists='replace', index=False)
    conn.close()
    logger.info("Predictions successfully saved to database table 'predictions'.")

if __name__ == '__main__':
    main()
