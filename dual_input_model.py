import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import logging
import warnings

# Suppress warnings and set display options
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#########################################
# Reaggregation Functions
#########################################
def resample_by_trading_day(df, rule, include_funding=False):
    """
    Group 5-min data by trading_day and resample within each trading day.
    """
    resampled_list = []
    for trading_day, group in df.groupby('trading_day'):
        trading_day_start = pd.to_datetime(group['trading_day_start'].iloc[0])
        trading_day_end = trading_day_start + timedelta(hours=24)
        group = group[(group['datetime'] >= trading_day_start) & (group['datetime'] < trading_day_end)]
        if group.empty:
            continue
        group = group.set_index('datetime')
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'turnover': 'sum',
            'open_interest': 'last'
        }
        if include_funding:
            agg_dict['funding_rate'] = 'mean'
        resampled = group.resample(rule).agg(agg_dict)
        resampled['trading_day_start'] = trading_day_start
        resampled_list.append(resampled.reset_index())
    if resampled_list:
        return pd.concat(resampled_list, ignore_index=True)
    else:
        return pd.DataFrame()

def reaggregate_custom_tables():
    """
    Reaggregate custom_5min data into multiple timeframes and save them to the DB.
    Creates the tables: agg_custom_15min, agg_custom_1h, agg_custom_3h, and agg_custom_daily.
    """
    db_path = 'bybit_data.db'
    conn = sqlite3.connect(db_path)
    df_5min = pd.read_sql_query("SELECT * FROM custom_5min", conn)
    conn.close()
    # Ensure proper types
    df_5min['datetime'] = pd.to_datetime(df_5min['datetime'])
    df_5min['trading_day'] = pd.to_datetime(df_5min['trading_day']).dt.date
    df_5min['trading_day_start'] = pd.to_datetime(df_5min['trading_day_start'])
    
    df_15min = resample_by_trading_day(df_5min, '15T', include_funding=False)
    df_1h    = resample_by_trading_day(df_5min, '1H', include_funding=True)
    df_3h    = resample_by_trading_day(df_5min, '3H', include_funding=False)
    df_daily = resample_by_trading_day(df_5min, 'D', include_funding=False)
    
    conn = sqlite3.connect(db_path)
    df_15min.to_sql('agg_custom_15min', conn, if_exists='replace', index=False)
    df_1h.to_sql('agg_custom_1h', conn, if_exists='replace', index=False)
    df_3h.to_sql('agg_custom_3h', conn, if_exists='replace', index=False)
    df_daily.to_sql('agg_custom_daily', conn, if_exists='replace', index=False)
    conn.close()
    logger.info("Re-aggregated custom tables saved: agg_custom_15min, agg_custom_1h, agg_custom_3h, agg_custom_daily.")

#########################################
# Preprocessing and Indicator Functions
#########################################
def preprocess_data(df):
    """
    Convert the 'datetime' column to datetime objects, sort, and extract a 'date' column.
    """
    if 'datetime' not in df.columns and 'trading_day_start' in df.columns:
        df['datetime'] = pd.to_datetime(df['trading_day_start'])
    else:
        df['datetime'] = pd.to_datetime(df['datetime'])
    df.sort_values(by='datetime', inplace=True)
    df['date'] = df['datetime'].dt.date
    return df

def compute_EMAs(df, ema_periods=[100, 350]):
    """
    Compute exponential moving averages on the 'close' column.
    """
    for period in ema_periods:
        col_name = f'EMA_{period}'
        df[col_name] = df['close'].ewm(span=period, adjust=False).mean()
    return df

def compute_ATR(df, period=7):
    """
    Compute the Average True Range (ATR) using a rolling window.
    """
    df = df.copy()
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df[f'ATR_{period}'] = df['true_range'].rolling(window=period).mean()
    df.drop(columns=['prev_close', 'tr1', 'tr2', 'tr3', 'true_range'], inplace=True)
    return df

def merge_ema_atr_features(df_features):
    """
    Merge EMA and ATR indicators into the feature dataset.
    EMAs are computed from agg_custom_15min, and ATR is computed from agg_custom_daily.
    """
    db_path = 'bybit_data.db'
    # Process 15-minute data for EMA indicators
    conn = sqlite3.connect(db_path)
    df_15min = pd.read_sql_query("SELECT * FROM agg_custom_15min", conn)
    conn.close()
    if 'datetime' in df_15min.columns:
        df_15min['datetime'] = pd.to_datetime(df_15min['datetime'])
    elif 'trading_day_start' in df_15min.columns:
        df_15min['datetime'] = pd.to_datetime(df_15min['trading_day_start'])
    else:
        raise KeyError("No datetime or trading_day_start column in agg_custom_15min")
    df_15min.sort_values('datetime', inplace=True)
    df_15min['date'] = df_15min['datetime'].dt.date
    df_15min = compute_EMAs(df_15min, ema_periods=[100, 350])
    daily_ema = df_15min.groupby('date').agg({'EMA_100': 'last', 'EMA_350': 'last'}).reset_index()
    
    # Process daily data for ATR indicator
    conn = sqlite3.connect(db_path)
    df_daily_custom = pd.read_sql_query("SELECT * FROM agg_custom_daily", conn)
    conn.close()
    if 'datetime' in df_daily_custom.columns:
        df_daily_custom['datetime'] = pd.to_datetime(df_daily_custom['datetime'])
    elif 'trading_day_start' in df_daily_custom.columns:
        df_daily_custom['datetime'] = pd.to_datetime(df_daily_custom['trading_day_start'])
    else:
        raise KeyError("No datetime or trading_day_start column in agg_custom_daily")
    df_daily_custom.sort_values('datetime', inplace=True)
    df_daily_custom['date'] = df_daily_custom['datetime'].dt.date
    df_daily_custom = compute_ATR(df_daily_custom, period=7)
    daily_atr = df_daily_custom.drop_duplicates(subset='date')[['date', 'ATR_7']]
    
    df_features = df_features.merge(daily_ema, on='date', how='left')
    df_features = df_features.merge(daily_atr, on='date', how='left')
    return df_features

#########################################
# Feature Extraction Functions
#########################################
def extract_prev_day_candle_features(df_3h, current_day):
    """
    Extract candle features from the previous trading day using 3-hour data.
    For the defined trading day (03:00 to 03:00 UTC):
      - P1: 18:00 to 21:00 on the previous day
      - P2: 21:00 on the previous day to 03:00 of the current trading day
    """
    prev_day = current_day - timedelta(days=1)
    P1_start = datetime.combine(prev_day, time(18, 0, 0))
    P1_end = datetime.combine(prev_day, time(21, 0, 0))
    P2_start = P1_end
    P2_end = datetime.combine(current_day, time(3, 0, 0))
    
    candles_P1 = df_3h[(df_3h['datetime'] >= P1_start) & (df_3h['datetime'] < P1_end)].copy()
    candles_P2 = df_3h[(df_3h['datetime'] >= P2_start) & (df_3h['datetime'] < P2_end)].copy()
    
    if candles_P1.empty or candles_P2.empty:
        return pd.Series({
            'P1_range': np.nan,
            'P2_range': np.nan,
            'P1_body': np.nan,
            'P2_body': np.nan,
            'P2_P1_close_diff': np.nan,
            'P1_wick_ratio': np.nan,
            'P2_wick_ratio': np.nan,
            'P2_open': np.nan
        })
    
    def select_candle(candles, start, end):
        midpoint = start + (end - start) / 2
        candles['diff'] = abs(candles['datetime'] - midpoint)
        return candles.sort_values('diff').iloc[0]
    
    P1 = select_candle(candles_P1, P1_start, P1_end)
    P2 = select_candle(candles_P2, P2_start, P2_end)
    
    P1_range = P1['high'] - P1['low']
    P2_range = P2['high'] - P2['low']
    P1_body = abs(P1['close'] - P1['open'])
    P2_body = abs(P2['close'] - P2['open'])
    P1_wick_ratio = ((P1_range - P1_body) / P1_range) if P1_range != 0 else 0
    P2_wick_ratio = ((P2_range - P2_body) / P2_range) if P2_range != 0 else 0
    close_diff = P2['close'] - P1['close']
    P2_open = P2['open']
    
    return pd.Series({
        'P1_range': P1_range,
        'P2_range': P2_range,
        'P1_body': P1_body,
        'P2_body': P2_body,
        'P2_P1_close_diff': close_diff,
        'P1_wick_ratio': P1_wick_ratio,
        'P2_wick_ratio': P2_wick_ratio,
        'P2_open': P2_open
    })

def extract_A1_features(df_3h, current_day, P2_features):
    """
    Extract features from the A1 candle (from 00:00 to 03:00 of the current trading day)
    using 3-hour data. Also calculate the A1 upper and lower wicks and volume.
    """
    A1_start = datetime.combine(current_day, time(0, 0, 0))
    A1_end = datetime.combine(current_day, time(3, 0, 0))
    candles_A1 = df_3h[(df_3h['datetime'] >= A1_start) & (df_3h['datetime'] < A1_end)].copy()
    
    if candles_A1.empty:
        return pd.Series({
            'A1_range': np.nan,
            'A1_body': np.nan,
            'A1_close': np.nan,
            'A1_deviation_from_P2_mid_pct': np.nan,
            'A1_upper_wick': np.nan,
            'A1_lower_wick': np.nan,
            'A1_volume': np.nan
        })
    
    def select_candle(candles, start, end):
        midpoint = start + (end - start) / 2
        candles['diff'] = abs(candles['datetime'] - midpoint)
        return candles.sort_values('diff').iloc[0]
    
    A1 = select_candle(candles_A1, A1_start, A1_end)
    A1_range = A1['high'] - A1['low']
    A1_body = abs(A1['close'] - A1['open'])
    A1_close = A1['close']
    # Calculate upper and lower wicks:
    A1_upper_wick = A1['high'] - max(A1['open'], A1['close'])
    A1_lower_wick = min(A1['open'], A1['close']) - A1['low']
    A1_volume = A1.get('volume', np.nan)
    
    P2_range = P2_features.get('P2_range', np.nan)
    P2_open = P2_features.get('P2_open', np.nan)
    if np.isnan(P2_range) or np.isnan(P2_open) or P2_range == 0:
        deviation = np.nan
    else:
        P2_mid = P2_open + 0.5 * P2_range
        deviation = (A1_close - P2_mid) / (0.25 * P2_range)
    
    return pd.Series({
        'A1_range': A1_range,
        'A1_body': A1_body,
        'A1_close': A1_close,
        'A1_deviation_from_P2_mid_pct': deviation,
        'A1_upper_wick': A1_upper_wick,
        'A1_lower_wick': A1_lower_wick,
        'A1_volume': A1_volume
    })

#########################################
# Build Training Set from Aggregated Data
#########################################
def build_feature_dataset(df_daily, df_3h, db_path='bybit_data.db'):
    """
    Build the feature dataset by merging daily data, 3h candle features, computed indicators,
    and the SND flag from btcusdt_daily. Then shift the target (SND) upward so that the features
    from day D are used to predict the SND flag for day D+1. Also filter out rows where the target day
    (i.e., the next day's date) falls on Saturday or Sunday.
    """
    # Preprocess daily and 3h data
    df_daily = preprocess_data(df_daily)
    df_daily = df_daily.drop_duplicates(subset='date', keep='last')
    df_3h = preprocess_data(df_3h)
    
    # Merge SND data from btcusdt_daily
    conn = sqlite3.connect(db_path)
    df_btc = pd.read_sql_query("SELECT * FROM btcusdt_daily", conn)
    conn.close()
    df_btc['datetime'] = pd.to_datetime(df_btc['datetime'])
    df_btc.sort_values('datetime', inplace=True)
    df_btc['date'] = df_btc['datetime'].dt.date
    df_btc = df_btc[['date', 'SND']]
    df_daily = df_daily.merge(df_btc, on='date', how='left')
    
    # Shift the SND flag upward so that features from day D predict the SND flag for day D+1
    df_daily['target'] = df_daily['SND'].shift(-1)
    # Create a temporary column for the target day (next day's date)
    df_daily['target_date'] = df_daily['date'].shift(-1)
    # Filter out rows where the target day is Saturday (5) or Sunday (6)
    df_daily = df_daily[df_daily['target_date'].apply(lambda d: d is not None and d.weekday() not in [5,6])]
    # Drop the temporary target_date column and any rows with NaN in the target column
    df_daily = df_daily.drop(columns=['target_date'])
    df_daily = df_daily.dropna(subset=['target'])
    
    # Compute 5-day EMA on daily closes, shifted by one day
    df_daily['ema_5'] = df_daily['close'].ewm(span=5, adjust=False).mean().shift(1)
    
    features = []
    # Iterate over daily data starting from the second row (ensuring previous day exists)
    for idx, row in df_daily.iloc[1:].iterrows():
        current_day = row['date']
        prev_day = current_day - timedelta(days=1)
        
        # Extract candle features from 3h data for the previous trading day
        prev_candle_feats = extract_prev_day_candle_features(df_3h, current_day)
        
        # Get previous day's daily data for additional features
        prev_daily_row = df_daily[df_daily['date'] == prev_day]
        if prev_daily_row.empty:
            logger.warning(f"Missing previous day's daily data for {current_day}")
            continue
        prev_daily = prev_daily_row.iloc[0]
        prev_daily_range = prev_daily['high'] - prev_daily['low']
        prev_daily_pct_change = (prev_daily['close'] - prev_daily['open']) / prev_daily['open'] if prev_daily['open'] else np.nan
        ema_5_val = prev_daily_row['ema_5'].iloc[0] if 'ema_5' in prev_daily_row.columns else np.nan
        
        diff_open_close = row['close'] - row['open']
        daily_pct_change = (row['close'] / row['open']) - 1
        body_to_range_ratio = abs(row['close'] - row['open']) / (row['high'] - row['low']) if (row['high'] - row['low']) != 0 else np.nan
        prev_day_gap = row['open'] - prev_daily.get('close', np.nan)
        
        # Extract A1 candle features (from 00:00 to 03:00 of the current trading day)
        A1_feats = extract_A1_features(df_3h, current_day, prev_candle_feats)
        
        # Use the shifted target from the row (next day's SND flag)
        target = 1 if str(row.get('target', '')).strip().upper() == "TRUE" else 0
        
        feature_row = {
            'date': current_day,
            'P1_range': prev_candle_feats.get('P1_range', np.nan),
            'P2_range': prev_candle_feats.get('P2_range', np.nan),
            'P1_body': prev_candle_feats.get('P1_body', np.nan),
            'P2_body': prev_candle_feats.get('P2_body', np.nan),
            'P2_P1_close_diff': prev_candle_feats.get('P2_P1_close_diff', np.nan),
            'P1_wick_ratio': prev_candle_feats.get('P1_wick_ratio', np.nan),
            'P2_wick_ratio': prev_candle_feats.get('P2_wick_ratio', np.nan),
            'daily_open': row['open'],
            'daily_close': row['close'],
            'prev_daily_range': prev_daily_range,
            'prev_daily_pct_change': prev_daily_pct_change,
            'prev_atr': prev_daily_range,  # ATR proxy; actual ATR will be merged later.
            'ema_5': ema_5_val,
            'diff_open_close': diff_open_close,
            'daily_pct_change': daily_pct_change,
            'body_to_range_ratio': body_to_range_ratio,
            'prev_day_gap': prev_day_gap,
            'A1_range': A1_feats.get('A1_range', np.nan),
            'A1_body': A1_feats.get('A1_body', np.nan),
            'A1_close': A1_feats.get('A1_close', np.nan),
            'A1_deviation_from_P2_mid_pct': A1_feats.get('A1_deviation_from_P2_mid_pct', np.nan),
            'A1_upper_wick': A1_feats.get('A1_upper_wick', np.nan),
            'A1_lower_wick': A1_feats.get('A1_lower_wick', np.nan),
            'A1_volume': A1_feats.get('A1_volume', np.nan),
            'target': target
        }
        features.append(feature_row)
    df_features = pd.DataFrame(features).dropna()
    df_features.sort_values(by='date', inplace=True)
    logger.info(f"Feature dataset built with {len(df_features)} records before filtering.")
    return df_features

#########################################
# Additional Feature Merging Functions
#########################################
def merge_5min_features(df_features):
    """
    Merge summary statistics from raw 5-min data into the feature set.
    """
    db_path = 'bybit_data.db'
    conn = sqlite3.connect(db_path)
    df_5min = pd.read_sql_query("SELECT * FROM custom_5min", conn)
    conn.close()
    df_5min['datetime'] = pd.to_datetime(df_5min['datetime'])
    df_5min['date'] = df_5min['datetime'].dt.date
    daily_5min = df_5min.groupby('date').agg({
        'open': 'first',
        'close': 'last',
        'high': 'max',
        'low': 'min',
        'volume': 'sum'
    }).reset_index()
    daily_5min.rename(columns={
         'open': 'first_5min_open',
         'close': 'last_5min_close',
         'high': 'max_5min_high',
         'low': 'min_5min_low',
         'volume': 'total_5min_volume'
    }, inplace=True)
    df_features = df_features.merge(daily_5min, on='date', how='left')
    return df_features

def merge_1h_features(df_features):
    """
    Merge summary statistics from 1-hour aggregated data into the feature set.
    """
    db_path = 'bybit_data.db'
    conn = sqlite3.connect(db_path)
    df_1h = pd.read_sql_query("SELECT * FROM agg_custom_1h", conn)
    conn.close()
    if 'datetime' in df_1h.columns:
        df_1h['datetime'] = pd.to_datetime(df_1h['datetime'])
    else:
        df_1h['datetime'] = pd.to_datetime(df_1h['trading_day_start'])
    df_1h['date'] = df_1h['datetime'].dt.date
    daily_1h = df_1h.groupby('date').agg({
        'open': 'first',
        'close': 'last',
        'high': 'max',
        'low': 'min',
        'volume': 'sum'
    }).reset_index()
    daily_1h.rename(columns={
         'open': 'first_1h_open',
         'close': 'last_1h_close',
         'high': 'max_1h_high',
         'low': 'min_1h_low',
         'volume': 'total_1h_volume'
    }, inplace=True)
    df_features = df_features.merge(daily_1h, on='date', how='left')
    return df_features

#########################################
# Main Routine: Build and Save Training Set
#########################################
def main():
    logger.info("Re-aggregating custom tables from custom_5min...")
    reaggregate_custom_tables()
    
    db_path = 'bybit_data.db'
    conn = sqlite3.connect(db_path)
    df_daily = pd.read_sql_query("SELECT * FROM agg_custom_daily", conn)
    df_3h = pd.read_sql_query("SELECT * FROM agg_custom_3h", conn)
    conn.close()
    
    # Build the feature dataset from aggregated daily and 3h data (merging SND from btcusdt_daily)
    df_features = build_feature_dataset(df_daily, df_3h, db_path=db_path)
    logger.info(f"Feature dataset built with {len(df_features)} records before filtering.")
    
    # Filter dataset to desired period: 1 Feb 2024 to 1 Feb 2025
    start_date = pd.to_datetime('2024-02-01').date()
    end_date = pd.to_datetime('2025-02-01').date()
    df_features = df_features[(df_features['date'] >= start_date) & (df_features['date'] <= end_date)]
    logger.info(f"Feature dataset size after filtering: {len(df_features)} records.")
    
    # Merge additional indicators (EMA_100, EMA_350, ATR_7)
    df_features = merge_ema_atr_features(df_features)
    
    # Merge 5min and 1h candle features into the feature set
    df_features = merge_5min_features(df_features)
    df_features = merge_1h_features(df_features)
    
    # Write the final training set into the database as a new table
    conn = sqlite3.connect(db_path)
    df_features.to_sql('training_set', conn, if_exists='replace', index=False)
    conn.close()
    logger.info("Training set successfully saved to database table 'training_set'.")

if __name__ == "__main__":
    main()
