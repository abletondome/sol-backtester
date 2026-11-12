import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from datetime import datetime

st.set_page_config(page_title="SOL/USDT AI Strategy Developer & Backtester", layout="wide")

st.title("🚀 SOL/USDT AI Strategy Developer & Backtester")
st.markdown("**OKX historical data** alapú backtesting és stratégia fejlesztő alkalmazás. Töltsd fel a CSV-t, állíts be stratégiát, tesztelj demo számlával.")

# Sidebar for settings
st.sidebar.header("⚙️ Beállítások")

uploaded_file = st.sidebar.file_uploader("Töltsd fel a historical OHLCV CSV-t (OKX SOL/USDT)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        # Try to detect columns
        df.columns = df.columns.str.lower().str.strip()
        
        # Common column mappings
        col_map = {}
        for col in df.columns:
            if 'time' in col or 'date' in col or 'timestamp' in col:
                col_map['datetime'] = col
            elif col in ['open', 'o']:
                col_map['open'] = col
            elif col in ['high', 'h']:
                col_map['high'] = col
            elif col in ['low', 'l']:
                col_map['low'] = col
            elif col in ['close', 'c']:
                col_map['close'] = col
            elif 'vol' in col:
                col_map['volume'] = col
        
        if 'datetime' in col_map:
            df['datetime'] = pd.to_datetime(df[col_map['datetime']])
            df = df.sort_values('datetime').reset_index(drop=True)
        else:
            st.error("Nem található dátum/idő oszlop. Győződj meg róla, hogy van timestamp vagy date oszlop.")
            st.stop()
        
        required = ['open', 'high', 'low', 'close']
        missing = [r for r in required if r not in col_map]
        if missing:
            st.error(f"Hiányzó oszlopok: {missing}. Szükséges: open, high, low, close")
            st.stop()
        
        # Rename for consistency
        df = df.rename(columns={
            col_map['open']: 'open',
            col_map['high']: 'high',
            col_map['low']: 'low',
            col_map['close']: 'close'
        })
        if 'volume' in col_map:
            df = df.rename(columns={col_map['volume']: 'volume'})
        
        st.sidebar.success(f"✅ {len(df)} gyertya betöltve")
        
        # Data preview
        with st.expander("📊 Adat előnézet"):
            st.dataframe(df.head(20))
            st.write(f"Időszak: {df['datetime'].min()} - {df['datetime'].max()}")
        
        # Strategy selection
        st.sidebar.header("📈 Stratégia")
        strategy_type = st.sidebar.selectbox(
            "Stratégia típusa",
            ["SMA Crossover", "RSI", "MACD", "Bollinger Bands", "EMA + RSI kombinált", "Swing: 200 EMA + SuperTrend 10-5", "Fast vs Slow MA (24/65 EMA)"]
        )
        
        initial_capital = st.sidebar.number_input("Kezdő tőke (USDT)", value=10000.0, min_value=100.0)
        fee_pct = st.sidebar.slider("Kereskedési díj (%)", 0.0, 0.5, 0.1, 0.01) / 100
        position_size_pct = st.sidebar.slider("Pozíció méret (% tőke)", 10, 100, 95) / 100
        
        # Strategy parameters
        params = {}
        if strategy_type == "SMA Crossover":
            params['fast'] = st.sidebar.slider("Fast SMA", 5, 50, 10)
            params['slow'] = st.sidebar.slider("Slow SMA", 20, 200, 50)
        elif strategy_type == "RSI":
            params['rsi_period'] = st.sidebar.slider("RSI periódus", 7, 21, 14)
            params['oversold'] = st.sidebar.slider("Oversold", 20, 40, 30)
            params['overbought'] = st.sidebar.slider("Overbought", 60, 80, 70)
        elif strategy_type == "MACD":
            params['fast'] = st.sidebar.slider("MACD Fast", 8, 16, 12)
            params['slow'] = st.sidebar.slider("MACD Slow", 20, 30, 26)
            params['signal'] = st.sidebar.slider("Signal", 5, 12, 9)
        elif strategy_type == "Bollinger Bands":
            params['period'] = st.sidebar.slider("BB periódus", 10, 50, 20)
            params['std'] = st.sidebar.slider("Szórás", 1.5, 3.0, 2.0, 0.1)
        elif strategy_type == "EMA + RSI kombinált":
            params['ema_fast'] = st.sidebar.slider("EMA Fast", 5, 30, 12)
            params['ema_slow'] = st.sidebar.slider("EMA Slow", 20, 100, 26)
            params['rsi_period'] = st.sidebar.slider("RSI", 7, 21, 14)
            params['rsi_buy'] = st.sidebar.slider("RSI Buy under", 20, 50, 40)
        elif strategy_type == "Swing: 200 EMA + SuperTrend 10-5":
            params['ema_period'] = st.sidebar.slider("EMA periódus", 100, 300, 200)
            params['st_period'] = st.sidebar.slider("SuperTrend ATR periódus", 7, 20, 10)
            params['st_multiplier'] = st.sidebar.slider("SuperTrend szorzó", 2.0, 7.0, 5.0, 0.5)
            st.sidebar.markdown("---")
            st.sidebar.subheader("Risk Management")
            params['tp_pct'] = st.sidebar.slider("Take Profit (%)", 5.0, 30.0, 12.0, 0.5)
            params['sl_pct'] = st.sidebar.slider("Stop Loss (%)", 1.0, 10.0, 3.0, 0.1)
        elif strategy_type == "Fast vs Slow MA (24/65 EMA)":
            params['fast'] = st.sidebar.slider("Fast MA", 5, 50, 24)
            params['slow'] = st.sidebar.slider("Slow MA", 20, 150, 65)
            params['ma_type'] = st.sidebar.selectbox("MA típus", ["EMA", "SMA"])
        
        # Optimization section
        st.sidebar.markdown("---")
        optimize = st.sidebar.checkbox("🔍 Paraméter optimalizálás")
        
        if optimize and strategy_type in ["Swing: 200 EMA + SuperTrend 10-5", "Fast vs Slow MA (24/65 EMA)"]:
            st.sidebar.subheader("Optimalizálási tartomány")
            if strategy_type == "Swing: 200 EMA + SuperTrend 10-5":
                ema_range = st.sidebar.slider("EMA tartomány", 50, 400, (100, 300))
                st_mult_range = st.sidebar.slider("SuperTrend szorzó tartomány", 1.5, 8.0, (2.0, 7.0), 0.5)
                tp_range = st.sidebar.slider("TP tartomány (%)", 5.0, 30.0, (8.0, 20.0), 1.0)
                sl_range = st.sidebar.slider("SL tartomány (%)", 0.5, 10.0, (1.5, 6.0), 0.5)
            else:  # Fast vs Slow
                fast_range = st.sidebar.slider("Fast MA tartomány", 10, 50, (15, 35))
                slow_range = st.sidebar.slider("Slow MA tartomány", 40, 120, (50, 90))
        
        # Generate signals
        def generate_signals(df, strategy, params):
            signals = pd.Series(0, index=df.index)
            close = df['close']
            
            if strategy == "SMA Crossover":
                sma_fast = close.rolling(params['fast']).mean()
                sma_slow = close.rolling(params['slow']).mean()
                signals[(sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))] = 1  # Buy
                signals[(sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))] = -1  # Sell
            
            elif strategy == "RSI":
                rsi = ta.momentum.RSIIndicator(close, window=params['rsi_period']).rsi()
                signals[(rsi < params['oversold']) & (rsi.shift(1) >= params['oversold'])] = 1
                signals[(rsi > params['overbought']) & (rsi.shift(1) <= params['overbought'])] = -1
            
            elif strategy == "MACD":
                macd = ta.trend.MACD(close, window_slow=params['slow'], window_fast=params['fast'], window_sign=params['signal'])
                macd_line = macd.macd()
                signal_line = macd.macd_signal()
                signals[(macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))] = 1
                signals[(macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))] = -1
            
            elif strategy == "Bollinger Bands":
                bb = ta.volatility.BollingerBands(close, window=params['period'], window_dev=params['std'])
                lower = bb.bollinger_lband()
                upper = bb.bollinger_hband()
                signals[(close < lower) & (close.shift(1) >= lower.shift(1))] = 1
                signals[(close > upper) & (close.shift(1) <= upper.shift(1))] = -1
            
            elif strategy == "EMA + RSI kombinált":
                ema_fast = ta.trend.EMAIndicator(close, window=params['ema_fast']).ema_indicator()
                ema_slow = ta.trend.EMAIndicator(close, window=params['ema_slow']).ema_indicator()
                rsi = ta.momentum.RSIIndicator(close, window=params['rsi_period']).rsi()
                signals[(ema_fast > ema_slow) & (rsi < params['rsi_buy'])] = 1
                signals[(ema_fast < ema_slow)] = -1
            
            elif strategy == "Swing: 200 EMA + SuperTrend 10-5":
                # 200 EMA
                ema200 = ta.trend.EMAIndicator(close, window=params['ema_period']).ema_indicator()
                
                # SuperTrend calculation
                high = df['high']
                low = df['low']
                atr = ta.volatility.AverageTrueRange(high, low, close, window=params['st_period']).average_true_range()
                
                hl2 = (high + low) / 2
                basic_upper = hl2 + params['st_multiplier'] * atr
                basic_lower = hl2 - params['st_multiplier'] * atr
                
                final_upper = basic_upper.copy()
                final_lower = basic_lower.copy()
                supertrend = pd.Series(index=df.index, dtype=float)
                direction = pd.Series(1, index=df.index)  # 1 = up, -1 = down
                
                for i in range(1, len(df)):
                    # Final Upper Band
                    if basic_upper.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
                        final_upper.iloc[i] = basic_upper.iloc[i]
                    else:
                        final_upper.iloc[i] = final_upper.iloc[i-1]
                    
                    # Final Lower Band
                    if basic_lower.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
                        final_lower.iloc[i] = basic_lower.iloc[i]
                    else:
                        final_lower.iloc[i] = final_lower.iloc[i-1]
                    
                    # SuperTrend direction
                    if close.iloc[i] <= final_upper.iloc[i]:
                        direction.iloc[i] = -1
                        supertrend.iloc[i] = final_upper.iloc[i]
                    else:
                        direction.iloc[i] = 1
                        supertrend.iloc[i] = final_lower.iloc[i]
                
                # Signals: Buy when price > 200 EMA and SuperTrend is up
                # Sell when SuperTrend flips down or price < 200 EMA
                st_up = direction == 1
                above_ema = close > ema200
                
                signals[(st_up) & (above_ema) & (~st_up.shift(1).fillna(False))] = 1  # SuperTrend flip to up + above EMA
                signals[(~st_up) & (st_up.shift(1).fillna(True))] = -1  # SuperTrend flip to down
                signals[(close < ema200) & (close.shift(1) >= ema200.shift(1))] = -1  # Cross below EMA
            
            elif strategy == "Fast vs Slow MA (24/65 EMA)":
                if params.get('ma_type', 'EMA') == "EMA":
                    fast_ma = ta.trend.EMAIndicator(close, window=params['fast']).ema_indicator()
                    slow_ma = ta.trend.EMAIndicator(close, window=params['slow']).ema_indicator()
                else:
                    fast_ma = close.rolling(params['fast']).mean()
                    slow_ma = close.rolling(params['slow']).mean()
                
                signals[(fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))] = 1  # Buy crossover
                signals[(fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))] = -1  # Sell crossunder
            
            return signals
        
        def run_backtest(df, signals, initial_capital, fee_pct, position_size_pct, params=None, strategy_type=None):
            """Reusable backtest function"""
            capital = initial_capital
            position = 0.0
            equity = []
            trades = []
            entry_price = 0.0
            entry_time = None
            tp_price = 0.0
            sl_price = 0.0
            
            use_tp_sl = strategy_type == "Swing: 200 EMA + SuperTrend 10-5" and params is not None
            
            for i in range(len(df)):
                price = df['close'].iloc[i]
                high = df['high'].iloc[i]
                low = df['low'].iloc[i]
                signal = signals.iloc[i]
                dt = df['datetime'].iloc[i]
                
                if position > 0 and use_tp_sl:
                    hit_tp = high >= tp_price
                    hit_sl = low <= sl_price
                    if hit_tp or hit_sl or signal == -1 or i == len(df)-1:
                        exit_price = tp_price if hit_tp else (sl_price if hit_sl else price)
                        sell_value = position * exit_price * (1 - fee_pct)
                        capital += sell_value
                        pnl_pct = (exit_price / entry_price - 1) * 100
                        position = 0
                elif position > 0 and (signal == -1 or i == len(df)-1):
                    sell_value = position * price * (1 - fee_pct)
                    capital += sell_value
                    position = 0
                
                if signal == 1 and position == 0:
                    invest = capital * position_size_pct
                    position = invest / price * (1 - fee_pct)
                    capital -= invest
                    entry_price = price
                    entry_time = dt
                    if use_tp_sl:
                        tp_price = entry_price * (1 + params['tp_pct'] / 100)
                        sl_price = entry_price * (1 - params['sl_pct'] / 100)
                
                current_equity = capital + (position * price if position > 0 else 0)
                equity.append(current_equity)
            
            equity = pd.Series(equity)
            total_return = (equity.iloc[-1] / initial_capital - 1) * 100 if len(equity) > 0 else 0
            returns = equity.pct_change().dropna()
            sharpe = (returns.mean() / returns.std() * np.sqrt(365*24*4)) if returns.std() > 0 else 0
            max_dd = ((equity.cummax() - equity) / equity.cummax()).max() * 100 if len(equity) > 0 else 0
            
            return total_return, sharpe, max_dd
        
        if optimize and strategy_type in ["Swing: 200 EMA + SuperTrend 10-5", "Fast vs Slow MA (24/65 EMA)"]:
            if st.sidebar.button("🚀 Optimalizálás indítása"):
                with st.spinner("Optimalizálás folyamatban... (ez eltarthat egy ideig)"):
                    results = []
                    
                    if strategy_type == "Swing: 200 EMA + SuperTrend 10-5":
                        ema_values = range(ema_range[0], ema_range[1]+1, 25)
                        mult_values = np.arange(st_mult_range[0], st_mult_range[1]+0.1, 1.0)
                        tp_values = np.arange(tp_range[0], tp_range[1]+0.1, 2.0)
                        sl_values = np.arange(sl_range[0], sl_range[1]+0.1, 1.0)
                        
                        for ema in ema_values:
                            for mult in mult_values:
                                for tp in tp_values:
                                    for sl in sl_values:
                                        opt_params = {
                                            'ema_period': int(ema),
                                            'st_period': 10,
                                            'st_multiplier': float(mult),
                                            'tp_pct': float(tp),
                                            'sl_pct': float(sl)
                                        }
                                        signals = generate_signals(df, strategy_type, opt_params)
                                        ret, sharpe, dd = run_backtest(df, signals, initial_capital, fee_pct, position_size_pct, opt_params, strategy_type)
                                        results.append({
                                            'EMA': int(ema),
                                            'ST_Mult': round(float(mult), 1),
                                            'TP%': round(float(tp), 1),
                                            'SL%': round(float(sl), 1),
                                            'Return%': round(ret, 2),
                                            'Sharpe': round(sharpe, 2),
                                            'MaxDD%': round(dd, 2)
                                        })
                    
                    else:  # Fast vs Slow
                        fast_values = range(fast_range[0], fast_range[1]+1, 5)
                        slow_values = range(slow_range[0], slow_range[1]+1, 10)
                        
                        for fast in fast_values:
                            for slow in slow_values:
                                if fast >= slow:
                                    continue
                                opt_params = {'fast': fast, 'slow': slow, 'ma_type': 'EMA'}
                                signals = generate_signals(df, strategy_type, opt_params)
                                ret, sharpe, dd = run_backtest(df, signals, initial_capital, fee_pct, position_size_pct, None, strategy_type)
                                results.append({
                                    'Fast': fast,
                                    'Slow': slow,
                                    'Return%': round(ret, 2),
                                    'Sharpe': round(sharpe, 2),
                                    'MaxDD%': round(dd, 2)
                                })
                    
                    if results:
                        results_df = pd.DataFrame(results)
                        results_df = results_df.sort_values(['Return%', 'Sharpe'], ascending=[False, False])
                        
                        # Filter for profitable ones
                        good_results = results_df[results_df['Return%'] >= 12.0]
                        
                        st.subheader("🏆 Optimalizálási eredmények")
                        
                        if len(good_results) > 0:
                            st.success(f"✅ Találtam **{len(good_results)}** olyan kombinációt, ami **≥ 12%** hozamot produkált!")
                            st.dataframe(good_results.head(15), use_container_width=True)
                            best = good_results.iloc[0]
                            st.balloons()
                            st.success(f"**Legjobb (≥12%):** Return: **{best['Return%']}%** | Sharpe: {best['Sharpe']} | MaxDD: {best['MaxDD%']}%")
                        else:
                            st.warning("⚠️ A megadott tartományban nem találtam ≥12% hozamú kombinációt.")
                            st.dataframe(results_df.head(15), use_container_width=True)
                            best = results_df.iloc[0]
                            st.info(f"Eddigi legjobb: Return: {best['Return%']}% | Sharpe: {best['Sharpe']}")
                        
                        # Download all results
                        csv = results_df.to_csv(index=False)
                        st.download_button("📥 Összes eredmény letöltése (CSV)", csv, "optimization_results.csv", "text/csv")
        
        if st.sidebar.button("▶️ Backtest indítása", type="primary"):
            with st.spinner("Backtest fut..."):
                signals = generate_signals(df, strategy_type, params)
                
                # Backtest engine (with TP/SL support)
                capital = initial_capital
                position = 0.0  # SOL amount
                equity = []
                trades = []
                entry_price = 0.0
                entry_time = None
                tp_price = 0.0
                sl_price = 0.0
                
                use_tp_sl = strategy_type == "Swing: 200 EMA + SuperTrend 10-5"
                
                for i in range(len(df)):
                    price = df['close'].iloc[i]
                    high = df['high'].iloc[i]
                    low = df['low'].iloc[i]
                    signal = signals.iloc[i]
                    dt = df['datetime'].iloc[i]
                    
                    # Check TP / SL if in position
                    if position > 0 and use_tp_sl:
                        hit_tp = high >= tp_price
                        hit_sl = low <= sl_price
                        
                        if hit_tp or hit_sl or signal == -1 or i == len(df)-1:
                            exit_price = tp_price if hit_tp else (sl_price if hit_sl else price)
                            sell_value = position * exit_price * (1 - fee_pct)
                            capital += sell_value
                            pnl = sell_value - (position * entry_price * (1 + fee_pct))
                            pnl_pct = (exit_price / entry_price - 1) * 100
                            reason = "TP" if hit_tp else ("SL" if hit_sl else "Signal")
                            trades.append({
                                'entry_time': entry_time,
                                'exit_time': dt,
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'pnl': pnl,
                                'pnl_pct': pnl_pct,
                                'size': position,
                                'exit_reason': reason
                            })
                            position = 0
                    
                    # Normal signal exit (for other strategies)
                    elif position > 0 and (signal == -1 or i == len(df)-1):
                        sell_value = position * price * (1 - fee_pct)
                        capital += sell_value
                        pnl = sell_value - (position * entry_price * (1 + fee_pct))
                        pnl_pct = (price / entry_price - 1) * 100
                        trades.append({
                            'entry_time': entry_time,
                            'exit_time': dt,
                            'entry_price': entry_price,
                            'exit_price': price,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'size': position,
                            'exit_reason': 'Signal'
                        })
                        position = 0
                    
                    # Open position on buy signal
                    if signal == 1 and position == 0:
                        invest = capital * position_size_pct
                        position = invest / price * (1 - fee_pct)
                        capital -= invest
                        entry_price = price
                        entry_time = dt
                        if use_tp_sl:
                            tp_price = entry_price * (1 + params['tp_pct'] / 100)
                            sl_price = entry_price * (1 - params['sl_pct'] / 100)
                    
                    # Current equity
                    current_equity = capital + (position * price if position > 0 else 0)
                    equity.append(current_equity)
                
                equity = pd.Series(equity, index=df['datetime'])
                
                # Metrics
                total_return = (equity.iloc[-1] / initial_capital - 1) * 100
                returns = equity.pct_change().dropna()
                sharpe = (returns.mean() / returns.std() * np.sqrt(365*24*4)) if returns.std() > 0 else 0  # approx for 15min candles
                max_dd = ((equity.cummax() - equity) / equity.cummax()).max() * 100
                win_trades = [t for t in trades if t['pnl'] > 0]
                win_rate = len(win_trades) / len(trades) * 100 if trades else 0
                
                # Display results
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Teljes hozam", f"{total_return:.2f}%")
                col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
                col3.metric("Max Drawdown", f"{max_dd:.2f}%")
                col4.metric("Win Rate", f"{win_rate:.1f}% ({len(trades)} trade)")
                
                # Equity chart
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.05, row_heights=[0.7, 0.3])
                
                fig.add_trace(go.Scatter(x=equity.index, y=equity, name="Equity", line=dict(color='green')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df['datetime'], y=df['close'], name="SOL Price", line=dict(color='blue')), row=2, col=1)
                
                # Mark trades
                for trade in trades:
                    fig.add_vline(x=trade['entry_time'], line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
                    fig.add_vline(x=trade['exit_time'], line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
                
                fig.update_layout(height=700, title="Equity Curve & Price", showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Trades table
                if trades:
                    st.subheader("📋 Trades")
                    trades_df = pd.DataFrame(trades)
                    format_dict = {
                        'entry_price': '{:.4f}',
                        'exit_price': '{:.4f}',
                        'pnl': '{:.2f}',
                        'pnl_pct': '{:.2f}%'
                    }
                    st.dataframe(trades_df.style.format(format_dict))
                    
                    # Extra stats for TP/SL
                    if 'exit_reason' in trades_df.columns:
                        tp_count = (trades_df['exit_reason'] == 'TP').sum()
                        sl_count = (trades_df['exit_reason'] == 'SL').sum()
                        st.info(f"TP találatok: {tp_count} | SL találatok: {sl_count}")
                
                # Drawdown
                dd = (equity.cummax() - equity) / equity.cummax() * 100
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=dd.index, y=dd, fill='tozeroy', name='Drawdown', line=dict(color='red')))
                fig_dd.update_layout(title="Drawdown", height=300)
                st.plotly_chart(fig_dd, use_container_width=True)
                
                st.success("✅ Backtest kész! Finomítsd a paramétereket a jobb eredményekért.")
        
        # Price chart always
        st.subheader("💰 Ár grafikon")
        fig_price = go.Figure()
        fig_price.add_trace(go.Candlestick(
            x=df['datetime'],
            open=df['open'], high=df['high'],
            low=df['low'], close=df['close'],
            name='SOL/USDT'
        ))
        fig_price.update_layout(height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_price, use_container_width=True)
        
    except Exception as e:
        st.error(f"Hiba az adat feldolgozása során: {str(e)}")
        st.info("CSV formátum példa: datetime,open,high,low,close,volume")

else:
    st.info("👈 Töltsd fel a historical adatokat a bal oldalon a kezdéshez!")
    st.markdown("""
    ### Használat:
    1. Töltsd fel az OKX SOL/USDT historical CSV-t (OHLCV)
    2. Válassz stratégiát és állítsd be a paramétereket
    3. Indítsd a backtestet demo számlával
    4. Elemezd az eredményeket és finomítsd a stratégiát
    
    **Támogatott stratégiák:** SMA Crossover, RSI, MACD, Bollinger Bands, EMA+RSI
    
    *AI tipp:* Próbálj ki különböző paramétereket, és figyeld a Sharpe ratio és drawdown értékeket!
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("**AI Strategy Developer** v1.0")
st.sidebar.markdown("Készült Grok-kal a stratégia fejlesztéshez")
