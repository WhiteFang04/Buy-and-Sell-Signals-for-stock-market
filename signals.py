import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Stock price buy & sell signal")
st.markdown("Data file should have [Date,Close,High,Low,Open] columns in Data")

uploaded_file = st.file_uploader("Upload your stock file (with Date, Open, High, Low, Close columns)",type=["xlsx","csv"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date',inplace=True)
    df = df.sort_values(by='Date', ascending=True)

    df.dropna(inplace =True)

    st.subheader("Raw data")
    st.dataframe(df.tail())
    def detect_candlestick_patterns(df):
        patterns = [None] * len(df)

        for i in range(2, len(df)):
            o, h, l, c = df['Open'].iloc[i], df['High'].iloc[i], df['Low'].iloc[i], df['Close'].iloc[i]
            prev_o, prev_c = df['Open'].iloc[i - 1], df['Close'].iloc[i - 1]
            prev2_c = df['Close'].iloc[i - 2]

            body = abs(c - o)
            range_candle = h - l
            upper_wick = h - max(c, o)
            lower_wick = min(c, o) - l

            # ✅ Bullish Patterns
            if body < 0.3 * range_candle and lower_wick > 2 * body:
                patterns[i] = 'Hammer'
            elif body < 0.3 * range_candle and upper_wick > 2 * body:
                patterns[i] = 'Shooting Star'
            elif prev_c < prev_o and c > o and c > prev_o and o < prev_c:
                patterns[i] = 'Bullish Engulfing'
            elif prev_c > prev_o and c < o and o > prev_c and c < prev_o:
                patterns[i] = 'Bearish Engulfing'
            elif prev_c < prev_o and c > o and o < l and c > (prev_c + prev_o) / 2:
                patterns[i] = 'Piercing Line'
            elif prev2_c > prev_o and abs(prev_c - prev_o) < 0.1 * (prev_c + prev_o) and c < prev_c:
                patterns[i] = 'Evening Star'
            elif prev2_c < prev_o and abs(prev_c - prev_o) < 0.1 * (prev_c + prev_o) and c > prev_c:
                patterns[i] = 'Morning Star'
            elif abs(c - o) <= 0.005 * c:
                if c > prev_c:
                    patterns[i] = 'Doji (at support)'
                else:
                    patterns[i] = 'Gravestone Doji'

        df['Candlestick_Pattern'] = patterns
        return df

    def detect_chart_patterns(df, window=5):
        highs = df['High'].values
        lows = df['Low'].values
        patterns = [None] * len(df)

        local_max = argrelextrema(highs, np.greater_equal, order=window)[0]
        local_min = argrelextrema(lows, np.less_equal, order=window)[0]

        for i in range(window, len(df)):
            # Double Bottom (W)
            if i - 2*window >= 0 and (
                lows[i-window] == min(lows[i-2*window:i]) and
                lows[i] == min(lows[i-window:i+1]) and
                abs(lows[i] - lows[i-window]) / lows[i] < 0.02
            ):
                patterns[i] = 'Double Bottom'

            # Double Top (M)
            elif i - 2*window >= 0 and (
                highs[i-window] == max(highs[i-2*window:i]) and
                highs[i] == max(highs[i-window:i+1]) and
                abs(highs[i] - highs[i-window]) / highs[i] < 0.02
            ):
                patterns[i] = 'Double Top'

            # Ascending Triangle (Flat top + higher lows)
            elif i >= window and (
                highs[i-window:i+1].max() - highs[i] < 0.5 and
                lows[i-window:i+1].min() < lows[i]
            ):
                patterns[i] = 'Ascending Triangle'

            # Descending Triangle (Flat bottom + lower highs)
            elif i >= window and (
                lows[i-window:i+1].min() - lows[i] < 0.5 and
                highs[i-window:i+1].max() > highs[i]
            ):
                patterns[i] = 'Descending Triangle'

            # Cup & Handle (U followed by dip)
            elif i - window >= 0 and (
                lows[i-window] > lows[i-window//2] < lows[i] and
                highs[i] < highs[i-window]
            ):
                patterns[i] = 'Cup and Handle'

             # Falling Wedge (Lower highs + lower lows narrowing)
            elif i >= window and (
                highs[i] < highs[i - window] and
                lows[i] > lows[i - window]
            ):
                patterns[i] = 'Falling Wedge'

            # Rising Wedge
            elif i >= window and (
                highs[i] > highs[i - window] and
                lows[i] < lows[i - window]
            ):
                patterns[i] = 'Rising Wedge'

            # Bullish Flag (after strong up move)
            elif i >= window and (
                df['Close'].iloc[i-window] * 1.05 < df['Close'].iloc[i] and
                (highs[i] - lows[i]) < 0.02 * df['Close'].iloc[i]
            ):
                patterns[i] = 'Bullish Flag'

            # Bearish Flag (after strong down move)
            elif i >= window and (
                df['Close'].iloc[i-window] * 0.95 > df['Close'].iloc[i] and
                (highs[i] - lows[i]) < 0.02 * df['Close'].iloc[i]
            ):
                patterns[i] = 'Bearish Flag'

        df['Chart_Pattern'] = patterns
        return df
    
    def generate_signals(df):
        signals = []

        for i in range(len(df)):
            candle = df['Candlestick_Pattern'].iloc[i]
            chart = df['Chart_Pattern'].iloc[i]

            # ✅ Bullish Patterns → BUY Signal
            bullish_candles = ['Hammer', 'Bullish Engulfing', 'Piercing Line', 'Morning Star', 'Doji (at support)', 'Inverted Hammer']
            bullish_charts = ['Double Bottom', 'Inverse Head & Shoulders', 'Ascending Triangle', 'Falling Wedge', 'Cup and Handle', 'Bullish Flag']

            # ❌ Bearish Patterns → SELL Signal
            bearish_candles = ['Shooting Star', 'Bearish Engulfing', 'Evening Star', 'Dark Cloud Cover', 'Gravestone Doji']
            bearish_charts = ['Double Top', 'Head & Shoulders', 'Descending Triangle', 'Rising Wedge', 'Bearish Flag']

            if candle in bullish_candles or chart in bullish_charts:
                signals.append('BUY')
            elif candle in bearish_candles or chart in bearish_charts:
                signals.append('SELL')
            else:
                signals.append(None)

        df['Signal'] = signals
        return df
    df = detect_candlestick_patterns(df)
    df = detect_chart_patterns(df, window=5)
    df = generate_signals(df)

    st.subheader("signals")
    st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Candlestick_Pattern', 'Chart_Pattern', 'Signal']].dropna())

    st.subheader("Buy and sell signal")
    def find_support_resistance_levels(df, window=10, threshold=0.01):
        highs = df['High'].values
        lows = df['Low'].values
        support_levels = []
        resistance_levels = []

        # Detect local minima/maxima
        local_max = argrelextrema(highs, np.greater, order=window)[0]
        local_min = argrelextrema(lows, np.less, order=window)[0]

        # Filter significant levels (remove similar/close levels)
        def is_far_enough(levels, new_level):
            return all(abs(new_level - lv) / lv > threshold for lv in levels)

        for i in local_min:
            level = lows[i]
            if is_far_enough(support_levels, level):
                support_levels.append(level)

        for i in local_max:
            level = highs[i]
            if is_far_enough(resistance_levels, level):
                resistance_levels.append(level)

        return support_levels, resistance_levels


    def plot_candlestick_with_patterns_and_slider(df, prices):
        fig = go.Figure()

    # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='green',
            decreasing_line_color='red'
        ))

        # Buy signals
        buy_signals = df[df['Signal'] == 'BUY']
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['Low'] * 0.995,
            mode='markers+text',
            marker=dict(color='green', size=10, symbol='triangle-up'),
            text=buy_signals['Candlestick_Pattern'],
            name='Buy Signal',
            textposition='bottom center'
        ))

        # Sell signals
        sell_signals = df[df['Signal'] == 'SELL']
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['High'] * 1.005,
            mode='markers+text',
            marker=dict(color='red', size=10, symbol='triangle-down'),
            text=sell_signals['Candlestick_Pattern'],
            name='Sell Signal',
            textposition='top center'
        ))

        # Chart pattern annotations + connecting lines
        for idx, row in df.iterrows():
            if pd.notna(row['Chart_Pattern']):
                fig.add_annotation(
                    x=idx,
                    y=row['High'] * 1.02,
                    text=row['Chart_Pattern'],
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='yellow',
                    font=dict(size=10, color="yellow")
                )

        # Connect detected chart patterns with lines (optional visual aid)
        pattern_indices = df[df['Chart_Pattern'].notna()].index
        for pattern_name in df['Chart_Pattern'].dropna().unique():
            pattern_df = df[df['Chart_Pattern'] == pattern_name]
            if len(pattern_df) >= 2:
                fig.add_trace(go.Scatter(
                    x=pattern_df.index,
                    y=pattern_df['Close'],
                    mode='lines',
                    name=pattern_name,
                    line=dict(width=2, dash='dot')
                ))

        # Support and resistance lines
        supports, resistances = find_support_resistance_levels(df)

        for lvl in supports:
            fig.add_hline(y=lvl, line=dict(color='green', width=1, dash='dot'),
                          annotation_text=f"Support @ {lvl:.2f}", annotation_position="bottom left")

        for lvl in resistances:
            fig.add_hline(y=lvl, line=dict(color='red', width=1, dash='dot'),
                          annotation_text=f"Resistance @ {lvl:.2f}", annotation_position="top left")

        # Layout
        fig.update_layout(
            title='Candlestick Chart with Buy/Sell Signals & Chart Patterns',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark',
            hovermode='x unified',
            height=700,
            width=1400,
            xaxis=dict(
                rangeslider=dict(visible=False),
                type="date"
            ),
            showlegend=True
        )

        fig.show()

    plot_candlestick_with_patterns_and_slider(df,prices=df['Close'])
    