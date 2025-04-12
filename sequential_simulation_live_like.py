#!/usr/bin/env python3
"""
Sequential Simulation Live-Like Backtest for A1 Session

This script processes 5-min candle data from Bybit (A1 session: 00:00-02:59 UTC) to compute:

1. Basic DOP Metrics:
   - simulated_DOP: Open of the first 5-min A1 candle (used as the Daily Open Price for A1).
   - dop_cross_count: Count of all DOP crosses (excluding the first candle). A cross is defined as either:
         (a) the candle's open and close lie on opposite sides of the simulated_DOP, OR
         (b) the candle's range spans the simulated_DOP.
   - first_dop_cross_time: Timestamp of the first cross event.
   - max_deviation_percent: Maximum percentage deviation (|close - simulated_DOP|/simulated_DOP*100).
   - total_volume, avg_volume, avg_open_interest, volume_change_percent: Liquidity measures in A1.
   - avg_funding_rate: Average funding rate during A1.

2. Interaction with Previous Day's Key Levels:
   - Loads aggregated data to retrieve previous day’s P2 (and P1) data.
   - Computes the P2 zone: using P2_mid = (high_P2 + low_P2)/2 and zone bounds = P2_mid ± 0.25*(high_P2-low_P2).
   - Determines if simulated_DOP breaches this zone (records breach_direction: "above", "below", or "inside") and deviation from P2_mid.
   - Computes time-to-breach (minutes from first A1 candle until the first candle with close outside P2 zone) and time-to-reversal (if price returns inside the zone).

3. Directional Bias:
   - Retrieves previous day’s P1 and P2 directional bias (bullish if close > open, bearish otherwise).

The script processes the data sequentially over a specified calendar date range (using the calendar date from the 5-min data) so that the backtest mimics live conditions.

Note: This script is intended for backtesting purposes. In live trading, the same logic would be updated with each incoming 5-min candle.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging for output to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


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
    """
    Loads aggregated_set data for previous day metrics.
    We require at least: trading_day, high_P2, low_P2, open_P1, close_P1, open_P2, close_P2.
    """
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT trading_day, high_P2, low_P2, open_P1, close_P1, open_P2, close_P2 FROM aggregated_set", conn)
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
    """Filters to 5-min candles in the A1 session (00:00 to 02:59 UTC)."""
    return df[df['datetime'].dt.hour < 3]

def get_calendar_date(dt):
    """Extracts calendar date from datetime."""
    return dt.date()

def compute_P2_zone(p2_high, p2_low):
    """
    Computes the P2 50% zone:
      P2_mid = (p2_high + p2_low) / 2
      Zone lower bound = P2_mid - 0.25*(p2_high - p2_low)
      Zone upper bound = P2_mid + 0.25*(p2_high - p2_low)
    Returns (lower_bound, upper_bound, P2_mid).
    """
    P2_mid = (p2_high + p2_low) / 2
    p2_range = p2_high - p2_low
    lower_bound = P2_mid - 0.25 * p2_range
    upper_bound = P2_mid + 0.25 * p2_range
    return lower_bound, upper_bound, P2_mid

def analyze_P2_interaction(simulated_DOP, p2_zone):
    """
    Determines whether simulated_DOP lies outside or inside the P2 zone.
    Returns:
      - breach_direction: "above" if simulated_DOP > upper_bound, "below" if simulated_DOP < lower_bound, "inside" otherwise.
      - deviation_from_P2_mid_percent: |simulated_DOP - P2_mid| / P2_mid * 100.
    """
    lower_bound, upper_bound, P2_mid = p2_zone
    if simulated_DOP > upper_bound:
        breach_direction = "above"
    elif simulated_DOP < lower_bound:
        breach_direction = "below"
    else:
        breach_direction = "inside"
    deviation = abs(simulated_DOP - P2_mid) / P2_mid * 100
    return breach_direction, deviation

def get_direction(open_price, close_price):
    """Returns 'bullish' if close > open, else 'bearish'."""
    return "bullish" if close_price > open_price else "bearish"

#########################################
# Sequential Feature Extraction
#########################################

def simulate_day_5min(group):
    """
    For a given day's 5-min A1 data (group), compute sequential features.
    - simulated_DOP: Open of the first 5-min candle.
    - dop_cross_count: Count of crosses (excluding first candle), where a cross is:
         (open - simulated_DOP)*(close - simulated_DOP) < 0 OR candle range spans simulated_DOP.
    - first_dop_cross_time: Timestamp of first cross event.
    - max_deviation_percent: Maximum percentage deviation of close from simulated_DOP.
    - total_volume, avg_volume, avg_open_interest: Liquidity metrics over A1.
    - avg_funding_rate: Average funding rate over A1.
    """
    group = group.sort_values("datetime")
    simulated_DOP = group.iloc[0]['open']
    
    dop_cross_count = 0
    first_dop_cross_time = None
    max_deviation_percent = 0
    
    for _, row in group.iloc[1:].iterrows():
        op = row['open']
        cl = row['close']
        hi = row['high']
        lo = row['low']
        current_time = row['datetime']
        # Cross conditions (as defined)
        transition_cross = (op - simulated_DOP) * (cl - simulated_DOP) < 0
        range_cross = (hi >= simulated_DOP) and (lo <= simulated_DOP)
        if transition_cross or range_cross:
            dop_cross_count += 1
            if first_dop_cross_time is None:
                first_dop_cross_time = current_time
        deviation_percent = abs(cl - simulated_DOP) / simulated_DOP * 100
        if deviation_percent > max_deviation_percent:
            max_deviation_percent = deviation_percent
    
    total_volume = group['volume'].sum()
    avg_volume = group['volume'].mean()
    avg_open_interest = group['open_interest'].mean() if 'open_interest' in group.columns else np.nan
    avg_funding_rate = group['funding_rate'].mean() if 'funding_rate' in group.columns else np.nan

    return {
        "simulated_DOP": simulated_DOP,
        "dop_cross_count": dop_cross_count,
        "first_dop_cross_time": first_dop_cross_time,
        "max_deviation_percent": max_deviation_percent,
        "total_volume": total_volume,
        "avg_volume": avg_volume,
        "avg_open_interest": avg_open_interest,
        "avg_funding_rate": avg_funding_rate
    }

def simulate_all_days_5min(df_5min, agg_df, start_date, end_date):
    """
    Simulates sequential features for each calendar day in the specified date range (A1 session, 5-min data).
    Uses calendar date from the 5-min data.
    
    For each day, it computes sequential A1 features and integrates previous day's aggregated data (for P2/P1):
      - P2 zone from previous day’s high_P2 and low_P2.
      - Breach analysis: whether simulated_DOP breaches the P2 zone and by how much.
      - Timing metrics: time-to-breach (from first candle to first candle with close outside P2 zone) and
        time-to-reversal (if price returns inside the zone).
      - Directional bias: from previous day’s P1 and P2.
      - Also passes along liquidity metrics and funding rate from A1.
    
    Returns a DataFrame with these features.
    """
    # Filter for A1 session and assign calendar date.
    df_A1 = filter_A1_session(df_5min)
    df_A1['date'] = df_A1['datetime'].dt.date
    
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    df_A1 = df_A1[(df_A1['date'] >= start) & (df_A1['date'] <= end)]
    
    # Build a dictionary from aggregated data keyed by trading_day.
    agg_dict = {}
    for _, row in agg_df.iterrows():
        agg_dict[row['trading_day']] = row
    
    results = []
    for day, group in df_A1.groupby("date"):
        group = group.sort_values("datetime")
        if group.empty:
            continue
        features = simulate_day_5min(group)
        features['date'] = day
        
        # Retrieve previous day's aggregated data.
        prev_day = day - timedelta(days=1)
        if prev_day in agg_dict:
            prev_data = agg_dict[prev_day]
            # Compute P2 zone from previous day's P2 high/low.
            if pd.notna(prev_data['high_P2']) and pd.notna(prev_data['low_P2']):
                p2_zone = compute_P2_zone(prev_data['high_P2'], prev_data['low_P2'])
                features['P2_lower_bound'] = p2_zone[0]
                features['P2_upper_bound'] = p2_zone[1]
                features['P2_mid'] = p2_zone[2]
                breach_direction, dev_from_mid = analyze_P2_interaction(features['simulated_DOP'], p2_zone)
                features['a1_breach_direction'] = breach_direction
                features['deviation_from_P2_mid_percent'] = dev_from_mid
            else:
                features['P2_lower_bound'] = np.nan
                features['P2_upper_bound'] = np.nan
                features['P2_mid'] = np.nan
                features['a1_breach_direction'] = np.nan
                features['deviation_from_P2_mid_percent'] = np.nan
            
            # Compute time-to-breach and time-to-reversal relative to P2 zone.
            if pd.notna(features.get('P2_lower_bound')):
                lower_bound, upper_bound, _ = p2_zone
                breach_time = None
                reversal_time = None
                for idx, row in group.iloc[1:].iterrows():
                    cl = row['close']
                    current_time = row['datetime']
                    # If the close is outside the P2 zone, mark breach.
                    if cl < lower_bound or cl > upper_bound:
                        if breach_time is None:
                            breach_time = current_time
                    # If a breach has occurred and price returns within the zone, mark reversal.
                    elif breach_time is not None and reversal_time is None:
                        reversal_time = current_time
                        break
                if breach_time is not None:
                    features['time_to_breach_minutes'] = (breach_time - group.iloc[0]['datetime']).total_seconds() / 60.0
                else:
                    features['time_to_breach_minutes'] = np.nan
                if breach_time is not None and reversal_time is not None:
                    features['time_to_reversal_minutes'] = (reversal_time - breach_time).total_seconds() / 60.0
                else:
                    features['time_to_reversal_minutes'] = np.nan
            else:
                features['time_to_breach_minutes'] = np.nan
                features['time_to_reversal_minutes'] = np.nan
            
            # Directional bias from previous day's P1 and P2.
            features['P1_direction'] = get_direction(prev_data['open_P1'], prev_data['close_P1']) if pd.notna(prev_data['open_P1']) and pd.notna(prev_data['close_P1']) else np.nan
            features['P2_direction'] = get_direction(prev_data['open_P2'], prev_data['close_P2']) if pd.notna(prev_data['open_P2']) and pd.notna(prev_data['close_P2']) else np.nan
        else:
            features['P2_lower_bound'] = np.nan
            features['P2_upper_bound'] = np.nan
            features['P2_mid'] = np.nan
            features['a1_breach_direction'] = np.nan
            features['deviation_from_P2_mid_percent'] = np.nan
            features['time_to_breach_minutes'] = np.nan
            features['time_to_reversal_minutes'] = np.nan
            features['P1_direction'] = np.nan
            features['P2_direction'] = np.nan
        
        results.append(features)
    
    return pd.DataFrame(results)

#########################################
# Main Execution
#########################################

if __name__ == "__main__":
    # Define the simulation date range (e.g., November 1 to November 10, 2024).
    START_DATE = '2024-11-01'
    END_DATE = '2024-11-10'
    
    # Load 5-min A1 data.
    df_5min = load_5min_data()
    
    # Load aggregated data (for previous day P2/P1 metrics).
    agg_df = load_aggregated_data()
    
    # Run the sequential simulation.
    df_seq = simulate_all_days_5min(df_5min, agg_df, START_DATE, END_DATE)
    
    print("Sequential Simulation Results (Expanded) for {} to {}:".format(START_DATE, END_DATE))
    print(df_seq)
