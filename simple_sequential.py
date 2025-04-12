#!/usr/bin/env python3
"""
Aggregator-Style Backtest with Refined Logic:
- P2 50% zone breakout/reentry check
- Cross counts in final 3 hours (00:00–02:59)
- Volume factor
- Additional aggregator signals (p2 bias, p2 wick, daily wick/body, return to open)
- Weighted scoring approach
- Logs results to 'backtest_predictions' table
- Displays final scoreboard

Test Range: aggregator_day from 2025-01-01 to 2025-01-10
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Set up Pandas & Logging
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== WEIGHTS & THRESHOLDS ====================
WEIGHTS = {
    'p2_bias': 50,               # near top/bottom => less S&D, else more S&D
    'p2_wick_strength': 20,      # big wicks => more S&D
    'liquidity_sweep': 20,       # if aggregator A1 or P2 surpass each other => uncertain
    'dop_crosses': {0: -15, 3: 5, 5: 10, 8: 25, 10: 30},
    'wick_size': 10,             # aggregator A1 wick
    'volume_spike': 40,          # aggregator P2 vs P1 or additional volume factor
    'return_to_open': 50,        # if aggregator day rebalanced
    'daily_wick_pct': 20,        # daily candle has large wick => more S&D
    'daily_body_median': 10,     # smaller body => indecision => more S&D
    'p2_50_breakout': 80,        # if A1 breaks out of P2 50% zone => less S&D, else more
    'p2_50_reentry': 50          # reentry => big S&D risk
}

# Additional numeric thresholds
WICK_SIZE_THRESHOLD = 1.3       # e.g. 1.5%
VOLUME_SPIKE_THRESHOLD = 1.2    # 150%
DAILY_WICK_THRESH_HIGH = 65     # >65% is "large daily wick"
DAILY_WICK_THRESH_LOW  = 50     # <50% is "clean daily wick"
P2_ZONE_CONSECUTIVE = 3         # how many consecutive candles needed for breakout

def load_data(db_path='bybit_data.db'):
    """
    Loads aggregator data, daily SND, and 5-minute raw data.
    """
    conn = sqlite3.connect(db_path)
    df_agg = pd.read_sql("SELECT * FROM aggregated_set", conn)
    df_agg['trading_day'] = pd.to_datetime(df_agg['trading_day']).dt.date

    df_daily = pd.read_sql("SELECT datetime as trading_day, SND FROM btcusdt_daily", conn)
    df_daily['trading_day'] = pd.to_datetime(df_daily['trading_day']).dt.date

    df_5min = pd.read_sql("""
        SELECT datetime, open, high, low, close, volume
        FROM btcusdt_5min
    """, conn)
    conn.close()

    df_5min['datetime'] = pd.to_datetime(df_5min['datetime'])
    return df_agg, df_daily, df_5min

def get_daily_body_median(df_agg):
    """
    For daily body size weighting, using aggregator columns daily_open/daily_close if present.
    """
    return abs(df_agg['daily_close'] - df_agg['daily_open']).median()

def count_a1_crosses(df_5min, aggregator_day):
    """
    aggregator_day => e.g. 2025-01-01
    We'll define midnight start => aggregator_day + 1 at 00:00 to 02:59
    use the candle at exactly 00:00 as DOP (open).
    Then count how many subsequent candles have [low..high] containing DOP
    (skipping the first candle).
    """
    next_day = pd.Timestamp(aggregator_day) + pd.Timedelta(days=1)
    midnight_start = next_day
    midnight_end   = next_day + pd.Timedelta(hours=3)

    zero_mask = (df_5min['datetime'] == midnight_start)
    zero_candle = df_5min[zero_mask]
    if zero_candle.empty:
        return 0, np.nan
    dop = zero_candle.iloc[0]['open']

    session_mask = (df_5min['datetime'] >= midnight_start) & (df_5min['datetime'] < midnight_end)
    session_data = df_5min[session_mask].copy().sort_values('datetime')
    if session_data.empty:
        return 0, dop

    cross_count = 0
    for i in range(len(session_data)):
        if i == 0:
            continue
        c = session_data.iloc[i]
        if c['low'] <= dop <= c['high']:
            cross_count += 1

    return cross_count, dop

def evaluate_p2_50_zone(a1_5min_data, p2_low, p2_high):
    """
    Returns (break_occurred, reentry_occurred)

    - We define P2 zone boundaries => 25%-75% around midpoint:
      lower_50 = p2_low + 0.25*(range), upper_50 = p2_high - 0.25*(range)
    - We track consecutive candles outside the zone to confirm breakout,
      if breakout occurs, track reentry if a subsequent candle closes back in.

    We only do this if a1_5min_data is non-empty. We'll skip the first candle from logic if you prefer.
    """
    p2_range = p2_high - p2_low
    if p2_range <= 0:
        return False, False

    lower_50 = p2_low + 0.25 * p2_range
    upper_50 = p2_high - 0.25 * p2_range

    outside_count = 0
    break_occurred = False
    reentry_occurred = False
    breakout_direction = None

    a1_5min_data = a1_5min_data.copy()
    a1_5min_data.sort_values('datetime', inplace=True)

    for i, c in a1_5min_data.iloc[1:].iterrows():  # skip first candle if desired
        close_price = c['close']
        if break_occurred:
            # If we already had a breakout, check reentry
            if breakout_direction == 'up' and close_price < upper_50:
                reentry_occurred = True
                break
            elif breakout_direction == 'down' and close_price > lower_50:
                reentry_occurred = True
                break
        else:
            # no breakout yet
            if close_price > upper_50:
                outside_count += 1
                if outside_count >= P2_ZONE_CONSECUTIVE:
                    break_occurred = True
                    breakout_direction = 'up'
            elif close_price < lower_50:
                outside_count += 1
                if outside_count >= P2_ZONE_CONSECUTIVE:
                    break_occurred = True
                    breakout_direction = 'down'
            else:
                # candle is inside => reset outside_count
                outside_count = 0

    return break_occurred, reentry_occurred

def score_aggregator_day(row, cross_count, dop, daily_body_median, a1_5min_data):
    """
    Combine all signals into a single numeric score, referencing the WEIGHTS dict.
    - row includes aggregator signals (P2 open/close/high/low, A1 open/close, daily_..)
    - cross_count from midnight chunk
    - daily_body_median for daily body weighting
    - a1_5min_data for advanced P2 50% zone checks
    """
    score = 0

    # 1) P2 bias
    p2_range = (row['high_P2'] - row['low_P2'])
    if p2_range > 0:
        near_top = (row['high_P2'] - row['close_P2']) < 0.15 * p2_range
        near_bottom = (row['close_P2'] - row['low_P2']) < 0.15 * p2_range
        if near_top or near_bottom:
            # strong directional => less S&D
            score -= WEIGHTS['p2_bias']
        else:
            # center => more S&D
            score += WEIGHTS['p2_bias']

    # 2) P2 wick strength
    if p2_range > 0:
        upper_wick_pct = (row['high_P2'] - max(row['open_P2'], row['close_P2'])) / p2_range * 100
        lower_wick_pct = (min(row['open_P2'], row['close_P2']) - row['low_P2']) / p2_range * 100
        if upper_wick_pct > 60 or lower_wick_pct > 60:
            # big wicks => more S&D
            score += WEIGHTS['p2_wick_strength']
        else:
            score -= WEIGHTS['p2_wick_strength']

    # 3) Liquidity sweep (A1 vs P2)
    # e.g. if aggregator A1 range extends beyond P2 range => uncertain
    sweep = (row['high_A1'] > row['high_P2']) or (row['low_A1'] < row['low_P2'])
    if sweep:
        score += WEIGHTS['liquidity_sweep']
    else:
        score -= WEIGHTS['liquidity_sweep']

    # 4) Cross count (midnight chunk)
    cross_points = next((pts for crs, pts in sorted(WEIGHTS['dop_crosses'].items(), reverse=True) if cross_count >= crs), -10)
    score += cross_points

    # 5) aggregator A1 wick size
    if row['open_A1'] > 0:
        a1_wick_pct = abs(row['high_A1'] - row['low_A1']) / row['open_A1'] * 100
        if a1_wick_pct > WICK_SIZE_THRESHOLD:
            score += WEIGHTS['wick_size']
        else:
            score -= WEIGHTS['wick_size']

    # 6) volume spike (P2 vs P1)
    if row['volume_P2'] > row['volume_P1'] * VOLUME_SPIKE_THRESHOLD:
        score += WEIGHTS['volume_spike']
    else:
        score -= WEIGHTS['volume_spike']

    # 7) daily wick
    daily_range = row['daily_high'] - row['daily_low']
    if daily_range > 0:
        daily_wick_pct = (daily_range - abs(row['daily_close'] - row['daily_open'])) / daily_range * 100
        if daily_wick_pct > DAILY_WICK_THRESH_HIGH:
            score += WEIGHTS['daily_wick_pct']
        elif daily_wick_pct < DAILY_WICK_THRESH_LOW:
            score -= WEIGHTS['daily_wick_pct']

    # 8) daily body < median => S&D
    aggregator_daily_body = abs(row['daily_close'] - row['daily_open'])
    if aggregator_daily_body < daily_body_median:
        score += WEIGHTS['daily_body_median']
    else:
        score -= WEIGHTS['daily_body_median']

    # 9) return_to_open
    rebalanced = abs(row['close_A1'] - row['daily_open']) < 0.001 * row['daily_open']
    if rebalanced:
        score -= WEIGHTS['return_to_open']
    else:
        score += WEIGHTS['return_to_open']

    # 10) P2 50% zone breakout / reentry
    # Evaluate on actual A1 5-min data if available
    # We'll gather aggregator_day+1 00:00->02:59 again, then do the breakout logic
    # We have it from a1_5min_data
    if a1_5min_data is not None and not a1_5min_data.empty:
        break_occurred, reentry_occurred = evaluate_p2_50_zone(a1_5min_data, row['low_P2'], row['high_P2'])
        if break_occurred and not reentry_occurred:
            # real breakout => less S&D
            score -= WEIGHTS['p2_50_breakout']
        elif break_occurred and reentry_occurred:
            # fake breakout => big S&D
            score += WEIGHTS['p2_50_reentry']
        else:
            # never broke out => stuck inside => more S&D
            score += WEIGHTS['p2_50_breakout']

    # final classification
    pred_result = 1 if score >= 0 else 0
    return pred_result, score

def get_a1_5min_data(df_5min, aggregator_day):
    """
    Returns the 5-min data from aggregator_day+1 00:00 -> 02:59 for use in P2 50% zone logic
    """
    next_day = pd.Timestamp(aggregator_day) + pd.Timedelta(days=1)
    midnight_start = next_day
    midnight_end   = next_day + pd.Timedelta(hours=3)
    mask = (df_5min['datetime'] >= midnight_start) & (df_5min['datetime'] < midnight_end)
    a1_data = df_5min[mask].copy().sort_values('datetime')
    return a1_data

def run_backtest(df_agg, df_daily, df_5min):
    results = []
    daily_body_median = get_daily_body_median(df_agg)

    for _, row in df_agg.iterrows():
        aggregator_day = row['trading_day']

        # 1) get cross_count & dop from the final chunk
        cross_count, dop = count_a1_crosses(df_5min, aggregator_day)

        # 2) also get that chunk's data for P2 50% zone logic
        a1_5min_data = get_a1_5min_data(df_5min, aggregator_day)

        # 3) unify aggregator signals + new signals
        pred_result, pred_score = score_aggregator_day(
            row,
            cross_count,
            dop,
            daily_body_median,
            a1_5min_data
        )

        # aggregator_day+1 => the "actual" day for SND
        next_cal_day = pd.Timestamp(aggregator_day) + pd.Timedelta(days=1)
        daily_snd_row = df_daily[df_daily['trading_day'] == next_cal_day.date()]
        if not daily_snd_row.empty:
            actual_snd = 1 if daily_snd_row['SND'].iloc[0] == 'TRUE' else 0
        else:
            actual_snd = np.nan

        results.append({
            'aggregator_day': aggregator_day,
            'A1_dop': dop,
            'A1_cross_count': cross_count,
            'prediction_score': pred_score,
            'prediction_result': pred_result,
            'actual_snd': actual_snd
        })

    results_df = pd.DataFrame(results)

    conn = sqlite3.connect('bybit_data.db')
    results_df.to_sql('backtest_predictions', conn, if_exists='replace', index=False)
    conn.close()

    # Summarize ignoring NaNs
    valid_df = results_df.dropna(subset=['actual_snd'])
    if not valid_df.empty:
        accuracy = (valid_df['prediction_result'] == valid_df['actual_snd']).mean() * 100
        fp = len(valid_df[(valid_df['prediction_result'] == 1) & (valid_df['actual_snd'] == 0)])
        fn = len(valid_df[(valid_df['prediction_result'] == 0) & (valid_df['actual_snd'] == 1)])
    else:
        accuracy, fp, fn = 0.0, 0, 0

    logger.info("=== Final Scorecard ===")
    logger.info(f"Total Aggregator Days: {len(results_df)}")
    logger.info(f"Valid Days (with actual_snd): {len(valid_df)}")
    logger.info(f"Accuracy: {accuracy:.2f}%")
    logger.info(f"False Positives: {fp}")
    logger.info(f"False Negatives: {fn}")
    logger.info("=======================")

if __name__ == "__main__":
    df_agg, df_daily, df_5min = load_data()

    # Limit aggregator_day to 2025-01-01..2025-01-10 for continuity
    df_agg = df_agg[
        (df_agg['trading_day'] >= datetime(2025,1,1).date()) &
        (df_agg['trading_day'] <= datetime(2025,1,30).date())
    ]

    run_backtest(df_agg, df_daily, df_5min)
