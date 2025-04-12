#!/usr/bin/env python3
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#########################################
# Data Loading Functions
#########################################

def load_5min_data(db_path='bybit_data.db'):
    """Loads 5-min candle data from the database."""
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM btcusdt_5min", conn)
        conn.close()
        df['datetime'] = pd.to_datetime(df['datetime'])
        logger.info("5-min data loaded with {} rows".format(len(df)))
        return df
    except Exception as e:
        logger.error("Error loading 5-min data: {}".format(e))
        raise

def load_aggregated_data(db_path='bybit_data.db'):
    """Loads aggregated_set from the database to get previous day P2 metrics."""
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT trading_day, high_P2, low_P2 FROM aggregated_set", conn)
        conn.close()
        df['trading_day'] = pd.to_datetime(df['trading_day']).dt.date
        logger.info("Aggregated data loaded with {} rows".format(len(df)))
        return df
    except Exception as e:
        logger.error("Error loading aggregated data: {}".format(e))
        raise

#########################################
# Helper Functions
#########################################

def filter_A1_session(df):
    """Returns only the 5-min candles for the A1 session (00:00 to 02:59 UTC)."""
    return df[df['datetime'].dt.hour < 3]

def get_calendar_date(dt):
    """Returns the calendar date of a candle (using its datetime)."""
    return dt.date()

#########################################
# Sequential Feature Extraction
#########################################

def simulate_day_5min(group):
    """
    For a given day’s A1 session 5-min data (group),
    compute:
      - simulated_DOP: open of first candle
      - dop_cross_count: count of crosses (excluding first candle)
      - first_dop_cross_time: timestamp of first cross event
      - max_deviation_percent: maximum percent deviation of any candle's close from simulated_DOP
      - total_volume: sum of volume in the A1 session
      - time_to_breach: first time the close moves outside the previous day’s P2 50% zone (if available)
      - time_to_reversal: if a breach occurs, the time when the price returns inside the zone
    """
    group = group.sort_values("datetime")
    simulated_DOP = group.iloc[0]['open']
    
    # Count crosses (exclude first candle)
    cross_count = 0
    first_cross_time = None
    max_dev = 0
    total_vol = group['volume'].sum()
    
    for idx, row in group.iloc[1:].iterrows():
        op = row['open']
        cl = row['close']
        hi = row['high']
        lo = row['low']
        current_time = row['datetime']
        dev = abs(cl - simulated_DOP) / simulated_DOP * 100
        if dev > max_dev:
            max_dev = dev
        # Cross defined as: open and close on opposite sides OR candle range spans simulated_DOP.
        if (op - simulated_DOP) * (cl - simulated_DOP) < 0 or (hi >= simulated_DOP and lo <= simulated_DOP):
            cross_count += 1
            if first_cross_time is None:
                first_cross_time = current_time

    features = {
        "simulated_DOP": simulated_DOP,
        "dop_cross_count": cross_count,
        "first_dop_cross_time": first_cross_time,
        "max_deviation_percent": max_dev,
        "total_volume": total_vol
    }
    return features

def compute_P2_zone(p2_high, p2_low):
    """
    Computes the P2 50% zone boundaries:
      P2_mid = (p2_high + p2_low) / 2
      P2_range = p2_high - p2_low
      Lower bound = P2_mid - 0.25 * P2_range
      Upper bound = P2_mid + 0.25 * P2_range
    Returns (lower_bound, upper_bound, p2_mid).
    """
    p2_mid = (p2_high + p2_low) / 2
    p2_range = p2_high - p2_low
    lower_bound = p2_mid - 0.25 * p2_range
    upper_bound = p2_mid + 0.25 * p2_range
    return lower_bound, upper_bound, p2_mid

def analyze_P2_interaction(simulated_DOP, p2_zone):
    """
    Checks whether the simulated_DOP falls within the P2 50% zone.
    p2_zone: (lower_bound, upper_bound, p2_mid)
    Returns a flag (1 if within zone, 0 otherwise) and the deviation (as a percentage of p2_mid).
    """
    lower_bound, upper_bound, p2_mid = p2_zone
    in_zone = 1 if lower_bound <= simulated_DOP <= upper_bound else 0
    deviation_from_mid = abs(simulated_DOP - p2_mid) / p2_mid * 100
    return in_zone, deviation_from_mid

def simulate_all_days_5min(df_5min, agg_df, start_date, end_date):
    """
    Simulates sequential features for each day in the specified calendar date range (A1 session, 5-min data).
    
    Also integrates previous day's P2 data:
      For each day d, the relevant P2 data is from day (d - 1).
    
    Returns a DataFrame with:
      - date (calendar date)
      - simulated_DOP, dop_cross_count, first_dop_cross_time, max_deviation_percent, total_volume
      - P2_lower_bound, P2_upper_bound, P2_mid, a1_in_zone (flag), and deviation_from_P2_mid (%)
      - time_to_breach and time_to_reversal (if a breach of P2 zone occurs)
    """
    # Filter 5-min data to A1 session and assign calendar date.
    df_A1 = filter_A1_session(df_5min)
    df_A1['date'] = df_A1['datetime'].dt.date
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    df_A1 = df_A1[(df_A1['date'] >= start) & (df_A1['date'] <= end)]
    
    # Create a dictionary for P2 data from aggregated_set, keyed by trading_day.
    # For a given day d, we want the P2 of the previous day (d - 1).
    p2_dict = {}
    for _, row in agg_df.iterrows():
        day = row['trading_day']
        if pd.notna(row['high_P2']) and pd.notna(row['low_P2']):
            p2_dict[day] = (row['high_P2'], row['low_P2'])
    
    results = []
    for day, group in df_A1.groupby("date"):
        group = group.sort_values("datetime")
        if group.empty:
            continue
        features = simulate_day_5min(group)
        features['date'] = day
        
        # Retrieve P2 for the previous calendar day.
        prev_day = day - timedelta(days=1)
        if prev_day in p2_dict:
            p2_high, p2_low = p2_dict[prev_day]
            p2_zone = compute_P2_zone(p2_high, p2_low)
            features['P2_lower_bound'] = p2_zone[0]
            features['P2_upper_bound'] = p2_zone[1]
            features['P2_mid'] = p2_zone[2]
            in_zone, dev_from_p2_mid = analyze_P2_interaction(features['simulated_DOP'], p2_zone)
            features['a1_in_P2_zone'] = in_zone
            features['deviation_from_P2_mid_percent'] = dev_from_p2_mid
        else:
            features['P2_lower_bound'] = np.nan
            features['P2_upper_bound'] = np.nan
            features['P2_mid'] = np.nan
            features['a1_in_P2_zone'] = np.nan
            features['deviation_from_P2_mid_percent'] = np.nan

        # Time-to-breach and time-to-reversal within A1 relative to P2 zone:
        # Define breach as the first candle whose close is outside the P2 50% zone.
        time_to_breach = None
        time_to_reversal = None
        breach_time = None
        reversal_time = None
        
        if pd.notna(features['P2_lower_bound']):
            lower_bound, upper_bound, _ = p2_zone
            breached = False
            for idx, row in group.iloc[1:].iterrows():
                cl = row['close']
                current_time = row['datetime']
                # Check if the close is outside the P2 zone.
                if not (lower_bound <= cl <= upper_bound):
                    if not breached:
                        breach_time = current_time
                        breached = True
                elif breached and reversal_time is None:
                    reversal_time = current_time
                    break
            if breach_time is not None:
                time_to_breach = (breach_time - group.iloc[0]['datetime']).total_seconds() / 60.0  # minutes
            if breach_time is not None and reversal_time is not None:
                time_to_reversal = (reversal_time - breach_time).total_seconds() / 60.0  # minutes
            
        features['time_to_breach_minutes'] = time_to_breach
        features['time_to_reversal_minutes'] = time_to_reversal
        
        results.append(features)
    
    return pd.DataFrame(results)

#########################################
# Main Execution
#########################################

if __name__ == "__main__":
    # Specify the date range for simulation, e.g., November 1 to November 10, 2024.
    START_DATE = '2024-11-01'
    END_DATE = '2024-11-10'
    
    # Load 5-min data.
    df_5min = load_5min_data()
    
    # Load aggregated_set data to retrieve P2 features.
    # We assume aggregated_set has at least: trading_day, high_P2, low_P2.
    try:
        conn = sqlite3.connect('bybit_data.db')
        agg_df = pd.read_sql_query("SELECT trading_day, high_P2, low_P2 FROM aggregated_set", conn)
        conn.close()
        agg_df['trading_day'] = pd.to_datetime(agg_df['trading_day']).dt.date
        logger.info("Aggregated (P2) data loaded with {} rows".format(len(agg_df)))
    except Exception as e:
        logger.error("Error loading aggregated data: {}".format(e))
        agg_df = pd.DataFrame()  # empty
    
    # Simulate sequential features for each day in the specified date range.
    df_seq = simulate_all_days_5min(df_5min, agg_df, START_DATE, END_DATE)
    
    print("Sequential Simulation Results (Expanded) for {} to {}:".format(START_DATE, END_DATE))
    print(df_seq)
