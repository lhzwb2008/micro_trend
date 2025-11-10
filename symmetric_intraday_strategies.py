"""
对称日内趋势策略合集
可以同时做多做空的经典策略

策略包括：
1. Dual Thrust - 双向突破策略
2. R-Breaker - 反转突破策略
3. 菲阿里四价 - 四价突破策略
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ================================================================================
# 配置
# ================================================================================

CONFIG = {
    'data_path': 'btc_15m.csv',
    'initial_capital': 100000,
    'start_date': '2020-01-01',
    'end_date': '2025-11-09',
    
    # 策略选择（1=Dual Thrust, 2=R-Breaker, 3=菲阿里四价）
    'strategy': 1,
    
    # Dual Thrust 参数
    'dual_thrust': {
        'lookback_days': 1,      # 回看天数
        'k1': 0.5,               # 上轨系数
        'k2': 0.5,               # 下轨系数
        'stop_loss_pct': 0.03,   # 止损百分比
        'use_time_exit': True,   # 是否使用时间退出
        'exit_hour': 23,         # 退出小时（UTC时间）
        'exit_minute': 45,       # 退出分钟
    },
    
    # R-Breaker 参数
    'r_breaker': {
        'lookback_days': 1,      # 回看天数
        'stop_loss_pct': 0.03,   # 止损百分比
        'use_time_exit': True,
        'exit_hour': 23,
        'exit_minute': 45,
    },
    
    # 菲阿里四价参数
    'phy_four_price': {
        'lookback_days': 1,      # 回看天数
        'stop_loss_pct': 0.03,   # 止损百分比
        'use_time_exit': True,
        'exit_hour': 23,
        'exit_minute': 45,
    },
    
    'print_trades': False,
}


# ================================================================================
# 数据加载
# ================================================================================

def load_data(config):
    """加载数据"""
    df = pd.read_csv(config['data_path'])
    
    if 'DateTime' not in df.columns:
        if 'timestamp' in df.columns:
            df['DateTime'] = pd.to_datetime(df['timestamp'], unit='ms')
        else:
            raise ValueError("数据文件需要包含DateTime或timestamp列")
    else:
        df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    df = df.sort_values('DateTime').reset_index(drop=True)
    
    # 过滤日期范围
    if config.get('start_date'):
        df = df[df['DateTime'] >= config['start_date']]
    if config.get('end_date'):
        df = df[df['DateTime'] <= config['end_date']]
    
    # 添加日期列
    df['Date'] = df['DateTime'].dt.date
    
    print(f"✅ 数据加载完成")
    print(f"   数据范围: {df['DateTime'].min()} 至 {df['DateTime'].max()}")
    print(f"   数据点数: {len(df)}")
    print(f"   价格范围: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}\n")
    
    return df


# ================================================================================
# 策略1: Dual Thrust
# ================================================================================

def calculate_dual_thrust_levels(df, config):
    """
    计算Dual Thrust策略的上下轨
    
    原理：
    1. 计算前N日的HH（最高价的最高）、LL（最低价的最低）、HC（收盘价的最高）、LC（收盘价的最低）
    2. Range = max(HH - LC, HC - LL)
    3. 上轨 = Open + k1 * Range
    4. 下轨 = Open - k2 * Range
    5. 价格突破上轨做多，跌破下轨做空
    """
    params = config['dual_thrust']
    lookback = params['lookback_days']
    k1 = params['k1']
    k2 = params['k2']
    
    df = df.copy()
    
    # 按日期分组，计算每日的开盘价
    daily_open = df.groupby('Date')['Open'].first()
    df['DayOpen'] = df['Date'].map(daily_open)
    
    # 按日期计算前一日的最高价的最高、最低价的最低、收盘价的最高和最低
    daily_hh = df.groupby('Date')['High'].max()
    daily_ll = df.groupby('Date')['Low'].min()
    daily_hc = df.groupby('Date')['Close'].max()
    daily_lc = df.groupby('Date')['Close'].min()
    
    # 将这些值映射回每行，并向前移动lookback天
    df['HH_prev'] = df['Date'].map(daily_hh).shift(lookback*96)
    df['LL_prev'] = df['Date'].map(daily_ll).shift(lookback*96)
    df['HC_prev'] = df['Date'].map(daily_hc).shift(lookback*96)
    df['LC_prev'] = df['Date'].map(daily_lc).shift(lookback*96)
    
    # 填充NaN（使用向前填充）
    df['HH_prev'].fillna(method='ffill', inplace=True)
    df['LL_prev'].fillna(method='ffill', inplace=True)
    df['HC_prev'].fillna(method='ffill', inplace=True)
    df['LC_prev'].fillna(method='ffill', inplace=True)
    
    # 计算Range
    df['Range'] = np.maximum(df['HH_prev'] - df['LC_prev'], df['HC_prev'] - df['LL_prev'])
    
    # 计算上下轨
    df['BuyLine'] = df['DayOpen'] + k1 * df['Range']
    df['SellLine'] = df['DayOpen'] - k2 * df['Range']
    
    return df


def run_dual_thrust_backtest(df, config):
    """执行Dual Thrust策略回测"""
    params = config['dual_thrust']
    initial_capital = config['initial_capital']
    stop_loss_pct = params['stop_loss_pct']
    use_time_exit = params['use_time_exit']
    print_trades = config['print_trades']
    
    capital = initial_capital
    position = None
    trades = []
    equity_curve = []
    
    for idx, row in df.iterrows():
        current_price = row['Close']
        current_time = row['DateTime']
        buy_line = row['BuyLine']
        sell_line = row['SellLine']
        
        # 记录权益曲线
        current_equity = capital
        if position:
            if position['type'] == 'long':
                unrealized_pnl = (current_price - position['entry_price']) * position['shares']
            else:
                unrealized_pnl = (position['entry_price'] - current_price) * position['shares']
            current_equity = capital + unrealized_pnl
        equity_curve.append({'DateTime': current_time, 'Equity': current_equity})
        
        # 时间退出检查
        should_time_exit = False
        if use_time_exit and position:
            if current_time.hour == params['exit_hour'] and current_time.minute >= params['exit_minute']:
                should_time_exit = True
        
        # 持仓管理
        if position:
            entry_price = position['entry_price']
            position_type = position['type']
            
            # 计算盈亏
            if position_type == 'long':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price
            
            # 检查退出条件
            should_close = False
            close_reason = None
            
            # 止损
            if pnl_pct <= -stop_loss_pct:
                should_close = True
                close_reason = '止损'
            
            # 时间退出
            elif should_time_exit:
                should_close = True
                close_reason = '时间退出'
            
            # 反向信号
            elif position_type == 'long' and current_price < sell_line:
                should_close = True
                close_reason = '反向信号'
            elif position_type == 'short' and current_price > buy_line:
                should_close = True
                close_reason = '反向信号'
            
            # 执行平仓
            if should_close:
                shares = position['shares']
                if position_type == 'long':
                    pnl = (current_price - entry_price) * shares
                else:
                    pnl = (entry_price - current_price) * shares
                capital += pnl
                
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'direction': position_type,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'shares': shares,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct * 100,
                    'reason': close_reason
                })
                
                if print_trades:
                    print(f"[{current_time}] 平仓{position_type} | {close_reason} | "
                          f"入:{entry_price:.1f} 出:{current_price:.1f} | "
                          f"盈亏:${pnl:+,.2f} ({pnl_pct*100:+.2f}%)")
                
                position = None
        
        # 开仓信号（无持仓时）
        if not position and pd.notna(buy_line) and pd.notna(sell_line):
            # 做多信号
            if current_price > buy_line:
                shares = int(capital / current_price)
                if shares > 0:
                    position = {
                        'type': 'long',
                        'entry_price': current_price,
                        'entry_time': current_time,
                        'shares': shares
                    }
                    if print_trades:
                        print(f"[{current_time}] 开多仓 | 价格:{current_price:.1f} 突破:{buy_line:.1f}")
            
            # 做空信号
            elif current_price < sell_line:
                shares = int(capital / current_price)
                if shares > 0:
                    position = {
                        'type': 'short',
                        'entry_price': current_price,
                        'entry_time': current_time,
                        'shares': shares
                    }
                    if print_trades:
                        print(f"[{current_time}] 开空仓 | 价格:{current_price:.1f} 突破:{sell_line:.1f}")
    
    # 强制平仓最后持仓
    if position:
        current_price = df.iloc[-1]['Close']
        entry_price = position['entry_price']
        shares = position['shares']
        position_type = position['type']
        
        if position_type == 'long':
            pnl = (current_price - entry_price) * shares
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl = (entry_price - current_price) * shares
            pnl_pct = (entry_price - current_price) / entry_price
        capital += pnl
        
        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': df.iloc[-1]['DateTime'],
            'direction': position_type,
            'entry_price': entry_price,
            'exit_price': current_price,
            'shares': shares,
            'pnl': pnl,
            'pnl_pct': pnl_pct * 100,
            'reason': 'Force Close'
        })
    
    return trades, capital, equity_curve


# ================================================================================
# 策略2: R-Breaker
# ================================================================================

def calculate_r_breaker_levels(df, config):
    """
    计算R-Breaker策略的关键价位
    
    原理：
    基于前一日的High, Low, Close计算6个价位：
    - Pivot = (H + L + C) / 3
    - 突破买入价 (Bbreak) = 2*Pivot - Low
    - 观察卖出价 (Ssetup) = Pivot + (High - Low)
    - 反转买入价 (Benter) = 2*Pivot - High
    - 反转卖出价 (Senter) = 2*Pivot + Low
    - 观察买入价 (Bsetup) = Pivot - (High - Low)
    - 突破卖出价 (Sbreak) = 2*Pivot + High
    
    交易逻辑：
    1. 趋势突破：价格突破Bbreak做多，跌破Sbreak做空
    2. 反转交易：价格触及Ssetup后回落至Senter做空，触及Bsetup后反弹至Benter做多
    """
    params = config['r_breaker']
    
    df = df.copy()
    
    # 按日期分组，计算前一日的H, L, C
    df['Date'] = df['DateTime'].dt.date
    daily_high = df.groupby('Date')['High'].max()
    daily_low = df.groupby('Date')['Low'].min()
    daily_close = df.groupby('Date')['Close'].last()
    
    # 映射到每行，并向前移动一天
    df['PrevHigh'] = df['Date'].map(daily_high).shift(96)
    df['PrevLow'] = df['Date'].map(daily_low).shift(96)
    df['PrevClose'] = df['Date'].map(daily_close).shift(96)
    
    # 填充NaN
    df['PrevHigh'].fillna(method='ffill', inplace=True)
    df['PrevLow'].fillna(method='ffill', inplace=True)
    df['PrevClose'].fillna(method='ffill', inplace=True)
    
    # 计算Pivot
    df['Pivot'] = (df['PrevHigh'] + df['PrevLow'] + df['PrevClose']) / 3
    
    # 计算6个关键价位
    df['Bbreak'] = 2 * df['Pivot'] - df['PrevLow']  # 突破买入
    df['Ssetup'] = df['Pivot'] + (df['PrevHigh'] - df['PrevLow'])  # 观察卖出
    df['Benter'] = 2 * df['Pivot'] - df['PrevHigh']  # 反转买入
    df['Senter'] = 2 * df['Pivot'] - df['PrevLow']  # 反转卖出（与Bbreak相同）
    df['Bsetup'] = df['Pivot'] - (df['PrevHigh'] - df['PrevLow'])  # 观察买入
    df['Sbreak'] = df['PrevHigh'] + 2 * (df['Pivot'] - df['PrevLow'])  # 突破卖出
    
    return df


def run_r_breaker_backtest(df, config):
    """执行R-Breaker策略回测"""
    params = config['r_breaker']
    initial_capital = config['initial_capital']
    stop_loss_pct = params['stop_loss_pct']
    use_time_exit = params['use_time_exit']
    print_trades = config['print_trades']
    
    capital = initial_capital
    position = None
    trades = []
    equity_curve = []
    touched_high = False  # 是否触及过Ssetup
    touched_low = False   # 是否触及过Bsetup
    
    for idx, row in df.iterrows():
        current_price = row['Close']
        current_high = row['High']
        current_low = row['Low']
        current_time = row['DateTime']
        
        # 记录权益曲线
        current_equity = capital
        if position:
            if position['type'] == 'long':
                unrealized_pnl = (current_price - position['entry_price']) * position['shares']
            else:
                unrealized_pnl = (position['entry_price'] - current_price) * position['shares']
            current_equity = capital + unrealized_pnl
        equity_curve.append({'DateTime': current_time, 'Equity': current_equity})
        
        # 每日重置触及标志
        if idx > 0 and df.iloc[idx]['Date'] != df.iloc[idx-1]['Date']:
            touched_high = False
            touched_low = False
        
        # 更新触及标志
        if pd.notna(row['Ssetup']) and current_high >= row['Ssetup']:
            touched_high = True
        if pd.notna(row['Bsetup']) and current_low <= row['Bsetup']:
            touched_low = True
        
        # 时间退出检查
        should_time_exit = False
        if use_time_exit and position:
            if current_time.hour == params['exit_hour'] and current_time.minute >= params['exit_minute']:
                should_time_exit = True
        
        # 持仓管理
        if position:
            entry_price = position['entry_price']
            position_type = position['type']
            
            # 计算盈亏
            if position_type == 'long':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price
            
            # 检查退出条件
            should_close = False
            close_reason = None
            
            # 止损
            if pnl_pct <= -stop_loss_pct:
                should_close = True
                close_reason = '止损'
            
            # 时间退出
            elif should_time_exit:
                should_close = True
                close_reason = '时间退出'
            
            # 反向突破信号
            elif position_type == 'long' and current_price < row['Sbreak']:
                should_close = True
                close_reason = '反向突破'
            elif position_type == 'short' and current_price > row['Bbreak']:
                should_close = True
                close_reason = '反向突破'
            
            # 执行平仓
            if should_close:
                shares = position['shares']
                if position_type == 'long':
                    pnl = (current_price - entry_price) * shares
                else:
                    pnl = (entry_price - current_price) * shares
                capital += pnl
                
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'direction': position_type,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'shares': shares,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct * 100,
                    'reason': close_reason
                })
                
                if print_trades:
                    print(f"[{current_time}] 平仓{position_type} | {close_reason} | "
                          f"入:{entry_price:.1f} 出:{current_price:.1f} | "
                          f"盈亏:${pnl:+,.2f} ({pnl_pct*100:+.2f}%)")
                
                position = None
        
        # 开仓信号（无持仓时）
        if not position and pd.notna(row['Bbreak']):
            # 趋势突破做多
            if current_price > row['Bbreak']:
                shares = int(capital / current_price)
                if shares > 0:
                    position = {
                        'type': 'long',
                        'entry_price': current_price,
                        'entry_time': current_time,
                        'shares': shares
                    }
                    if print_trades:
                        print(f"[{current_time}] 趋势突破多 | 价格:{current_price:.1f} > {row['Bbreak']:.1f}")
            
            # 趋势突破做空
            elif current_price < row['Sbreak']:
                shares = int(capital / current_price)
                if shares > 0:
                    position = {
                        'type': 'short',
                        'entry_price': current_price,
                        'entry_time': current_time,
                        'shares': shares
                    }
                    if print_trades:
                        print(f"[{current_time}] 趋势突破空 | 价格:{current_price:.1f} < {row['Sbreak']:.1f}")
            
            # 反转做多（触及Bsetup后反弹至Benter）
            elif touched_low and current_price > row['Benter']:
                shares = int(capital / current_price)
                if shares > 0:
                    position = {
                        'type': 'long',
                        'entry_price': current_price,
                        'entry_time': current_time,
                        'shares': shares
                    }
                    if print_trades:
                        print(f"[{current_time}] 反转做多 | 价格:{current_price:.1f} > {row['Benter']:.1f}")
            
            # 反转做空（触及Ssetup后回落至Senter）
            elif touched_high and current_price < row['Senter']:
                shares = int(capital / current_price)
                if shares > 0:
                    position = {
                        'type': 'short',
                        'entry_price': current_price,
                        'entry_time': current_time,
                        'shares': shares
                    }
                    if print_trades:
                        print(f"[{current_time}] 反转做空 | 价格:{current_price:.1f} < {row['Senter']:.1f}")
    
    # 强制平仓最后持仓
    if position:
        current_price = df.iloc[-1]['Close']
        entry_price = position['entry_price']
        shares = position['shares']
        position_type = position['type']
        
        if position_type == 'long':
            pnl = (current_price - entry_price) * shares
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl = (entry_price - current_price) * shares
            pnl_pct = (entry_price - current_price) / entry_price
        capital += pnl
        
        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': df.iloc[-1]['DateTime'],
            'direction': position_type,
            'entry_price': entry_price,
            'exit_price': current_price,
            'shares': shares,
            'pnl': pnl,
            'pnl_pct': pnl_pct * 100,
            'reason': 'Force Close'
        })
    
    return trades, capital, equity_curve


# ================================================================================
# 策略3: 菲阿里四价
# ================================================================================

def calculate_phy_four_price_levels(df, config):
    """
    计算菲阿里四价策略的关键价位
    
    原理：
    基于前一日的Open, High, Low, Close计算4个价位：
    - 突破买入价 = High
    - 突破卖出价 = Low
    - 反转买入价 = 2*Low - High
    - 反转卖出价 = 2*High - Low
    
    简化版交易逻辑：
    1. 价格突破前日高点做多
    2. 价格跌破前日低点做空
    """
    params = config['phy_four_price']
    
    df = df.copy()
    
    # 按日期分组，计算前一日的O, H, L, C
    df['Date'] = df['DateTime'].dt.date
    daily_open = df.groupby('Date')['Open'].first()
    daily_high = df.groupby('Date')['High'].max()
    daily_low = df.groupby('Date')['Low'].min()
    daily_close = df.groupby('Date')['Close'].last()
    
    # 映射到每行，并向前移动一天
    df['PrevOpen'] = df['Date'].map(daily_open).shift(96)
    df['PrevHigh'] = df['Date'].map(daily_high).shift(96)
    df['PrevLow'] = df['Date'].map(daily_low).shift(96)
    df['PrevClose'] = df['Date'].map(daily_close).shift(96)
    
    # 填充NaN
    df['PrevOpen'].fillna(method='ffill', inplace=True)
    df['PrevHigh'].fillna(method='ffill', inplace=True)
    df['PrevLow'].fillna(method='ffill', inplace=True)
    df['PrevClose'].fillna(method='ffill', inplace=True)
    
    # 计算四价
    df['BuyBreak'] = df['PrevHigh']  # 突破买入
    df['SellBreak'] = df['PrevLow']  # 突破卖出
    df['BuyReverse'] = 2 * df['PrevLow'] - df['PrevHigh']  # 反转买入
    df['SellReverse'] = 2 * df['PrevHigh'] - df['PrevLow']  # 反转卖出
    
    return df


def run_phy_four_price_backtest(df, config):
    """执行菲阿里四价策略回测"""
    params = config['phy_four_price']
    initial_capital = config['initial_capital']
    stop_loss_pct = params['stop_loss_pct']
    use_time_exit = params['use_time_exit']
    print_trades = config['print_trades']
    
    capital = initial_capital
    position = None
    trades = []
    equity_curve = []
    
    for idx, row in df.iterrows():
        current_price = row['Close']
        current_time = row['DateTime']
        
        # 记录权益曲线
        current_equity = capital
        if position:
            if position['type'] == 'long':
                unrealized_pnl = (current_price - position['entry_price']) * position['shares']
            else:
                unrealized_pnl = (position['entry_price'] - current_price) * position['shares']
            current_equity = capital + unrealized_pnl
        equity_curve.append({'DateTime': current_time, 'Equity': current_equity})
        
        # 时间退出检查
        should_time_exit = False
        if use_time_exit and position:
            if current_time.hour == params['exit_hour'] and current_time.minute >= params['exit_minute']:
                should_time_exit = True
        
        # 持仓管理
        if position:
            entry_price = position['entry_price']
            position_type = position['type']
            
            # 计算盈亏
            if position_type == 'long':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price
            
            # 检查退出条件
            should_close = False
            close_reason = None
            
            # 止损
            if pnl_pct <= -stop_loss_pct:
                should_close = True
                close_reason = '止损'
            
            # 时间退出
            elif should_time_exit:
                should_close = True
                close_reason = '时间退出'
            
            # 反向信号
            elif position_type == 'long' and current_price < row['SellBreak']:
                should_close = True
                close_reason = '反向信号'
            elif position_type == 'short' and current_price > row['BuyBreak']:
                should_close = True
                close_reason = '反向信号'
            
            # 执行平仓
            if should_close:
                shares = position['shares']
                if position_type == 'long':
                    pnl = (current_price - entry_price) * shares
                else:
                    pnl = (entry_price - current_price) * shares
                capital += pnl
                
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'direction': position_type,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'shares': shares,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct * 100,
                    'reason': close_reason
                })
                
                if print_trades:
                    print(f"[{current_time}] 平仓{position_type} | {close_reason} | "
                          f"入:{entry_price:.1f} 出:{current_price:.1f} | "
                          f"盈亏:${pnl:+,.2f} ({pnl_pct*100:+.2f}%)")
                
                position = None
        
        # 开仓信号（无持仓时）
        if not position and pd.notna(row['BuyBreak']):
            # 突破做多
            if current_price > row['BuyBreak']:
                shares = int(capital / current_price)
                if shares > 0:
                    position = {
                        'type': 'long',
                        'entry_price': current_price,
                        'entry_time': current_time,
                        'shares': shares
                    }
                    if print_trades:
                        print(f"[{current_time}] 突破做多 | 价格:{current_price:.1f} > {row['BuyBreak']:.1f}")
            
            # 突破做空
            elif current_price < row['SellBreak']:
                shares = int(capital / current_price)
                if shares > 0:
                    position = {
                        'type': 'short',
                        'entry_price': current_price,
                        'entry_time': current_time,
                        'shares': shares
                    }
                    if print_trades:
                        print(f"[{current_time}] 突破做空 | 价格:{current_price:.1f} < {row['SellBreak']:.1f}")
    
    # 强制平仓最后持仓
    if position:
        current_price = df.iloc[-1]['Close']
        entry_price = position['entry_price']
        shares = position['shares']
        position_type = position['type']
        
        if position_type == 'long':
            pnl = (current_price - entry_price) * shares
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl = (entry_price - current_price) * shares
            pnl_pct = (entry_price - current_price) / entry_price
        capital += pnl
        
        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': df.iloc[-1]['DateTime'],
            'direction': position_type,
            'entry_price': entry_price,
            'exit_price': current_price,
            'shares': shares,
            'pnl': pnl,
            'pnl_pct': pnl_pct * 100,
            'reason': 'Force Close'
        })
    
    return trades, capital, equity_curve


# ================================================================================
# 结果分析
# ================================================================================

def analyze_results(trades, final_capital, initial_capital, df, equity_curve):
    """分析回测结果"""
    if not trades:
        print("\n⚠️  没有产生任何交易！\n")
        return None
    
    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)
    
    # 基础指标
    total_return = (final_capital - initial_capital) / initial_capital * 100
    num_trades = len(trades_df)
    
    # 分离多空交易
    long_trades = trades_df[trades_df['direction'] == 'long']
    short_trades = trades_df[trades_df['direction'] == 'short']
    
    # 盈利/亏损交易
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] < 0]
    
    win_rate = len(winning_trades) / num_trades * 100 if num_trades > 0 else 0
    
    # 计算年化收益和波动率
    start_date = df['DateTime'].min()
    end_date = df['DateTime'].max()
    days = (end_date - start_date).days
    years = days / 365.25
    
    annualized_return = ((final_capital / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    # 计算波动率
    equity_df['Returns'] = equity_df['Equity'].pct_change()
    daily_volatility = equity_df['Returns'].std()
    annualized_volatility = daily_volatility * np.sqrt(96 * 365) * 100  # 15分钟数据，一天96个bar
    
    # 夏普比率
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
    
    # 最大回撤
    equity_df['Peak'] = equity_df['Equity'].cummax()
    equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['Peak']) / equity_df['Peak'] * 100
    max_drawdown = equity_df['Drawdown'].min()
    
    # Calmar比率
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Buy & Hold对比
    bh_return = (df.iloc[-1]['Close'] / df.iloc[0]['Close'] - 1) * 100
    bh_annualized = ((df.iloc[-1]['Close'] / df.iloc[0]['Close']) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    # 计算Buy & Hold的夏普比率
    df_bh = df.copy()
    df_bh['BH_Value'] = initial_capital * (df_bh['Close'] / df_bh.iloc[0]['Close'])
    df_bh['BH_Returns'] = df_bh['BH_Value'].pct_change()
    bh_volatility = df_bh['BH_Returns'].std() * np.sqrt(96 * 365) * 100
    bh_sharpe = bh_annualized / bh_volatility if bh_volatility > 0 else 0
    
    # 打印结果
    print(f"\n{'='*80}")
    print(f"📊 回测结果")
    print(f"{'='*80}\n")
    
    print(f"💰 资金表现:")
    print(f"   初始资金: ${initial_capital:,.0f}")
    print(f"   最终资金: ${final_capital:,.0f}")
    print(f"   总回报率: {total_return:+.2f}%")
    print(f"   年化收益率: {annualized_return:+.2f}%\n")
    
    print(f"📊 风险指标:")
    print(f"   年化波动率: {annualized_volatility:.2f}%")
    print(f"   最大回撤: {max_drawdown:.2f}%")
    print(f"   夏普比率: {sharpe_ratio:.3f}")
    print(f"   Calmar比率: {calmar_ratio:.3f}\n")
    
    print(f"📈 交易统计:")
    print(f"   总交易次数: {num_trades}笔")
    print(f"   多头交易: {len(long_trades)}笔 | 空头交易: {len(short_trades)}笔")
    print(f"   胜率: {win_rate:.1f}%")
    print(f"   盈利交易: {len(winning_trades)}笔 (平均: ${winning_trades['pnl'].mean():,.2f})")
    print(f"   亏损交易: {len(losing_trades)}笔 (平均: ${losing_trades['pnl'].mean():,.2f})")
    if len(losing_trades) > 0:
        profit_loss_ratio = abs(winning_trades['pnl'].mean() / losing_trades['pnl'].mean())
        print(f"   盈亏比: {profit_loss_ratio:.2f}\n")
    
    print(f"🎯 vs Buy & Hold:")
    print(f"   B&H总回报: {bh_return:+.2f}%")
    print(f"   B&H年化: {bh_annualized:+.2f}%")
    print(f"   B&H夏普: {bh_sharpe:.3f}")
    print(f"   策略超额回报: {total_return - bh_return:+.2f}%")
    print(f"   夏普比率优势: {sharpe_ratio - bh_sharpe:+.3f}\n")
    
    print(f"{'='*80}\n")
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'num_trades': num_trades,
        'num_long': len(long_trades),
        'num_short': len(short_trades),
    }


# ================================================================================
# 主函数
# ================================================================================

def main():
    strategy_names = {
        1: 'Dual Thrust (双向突破)',
        2: 'R-Breaker (反转突破)',
        3: '菲阿里四价 (四价突破)'
    }
    
    print(f"\n{'='*80}")
    print(f"📈 对称日内趋势策略回测系统")
    print(f"{'='*80}\n")
    print(f"策略: {strategy_names[CONFIG['strategy']]}\n")
    
    # 加载数据
    df = load_data(CONFIG)
    
    # 执行策略
    if CONFIG['strategy'] == 1:
        print(f"{'='*80}")
        print(f"📊 计算 Dual Thrust 上下轨...")
        print(f"{'='*80}\n")
        df = calculate_dual_thrust_levels(df, CONFIG)
        print(f"✅ 参数: k1={CONFIG['dual_thrust']['k1']}, k2={CONFIG['dual_thrust']['k2']}, "
              f"回看={CONFIG['dual_thrust']['lookback_days']}天\n")
        
        print(f"{'='*80}")
        print(f"🚀 开始回测...")
        print(f"{'='*80}\n")
        trades, final_capital, equity_curve = run_dual_thrust_backtest(df, CONFIG)
    
    elif CONFIG['strategy'] == 2:
        print(f"{'='*80}")
        print(f"📊 计算 R-Breaker 关键价位...")
        print(f"{'='*80}\n")
        df = calculate_r_breaker_levels(df, CONFIG)
        print(f"✅ 回看={CONFIG['r_breaker']['lookback_days']}天\n")
        
        print(f"{'='*80}")
        print(f"🚀 开始回测...")
        print(f"{'='*80}\n")
        trades, final_capital, equity_curve = run_r_breaker_backtest(df, CONFIG)
    
    elif CONFIG['strategy'] == 3:
        print(f"{'='*80}")
        print(f"📊 计算菲阿里四价价位...")
        print(f"{'='*80}\n")
        df = calculate_phy_four_price_levels(df, CONFIG)
        print(f"✅ 回看={CONFIG['phy_four_price']['lookback_days']}天\n")
        
        print(f"{'='*80}")
        print(f"🚀 开始回测...")
        print(f"{'='*80}\n")
        trades, final_capital, equity_curve = run_phy_four_price_backtest(df, CONFIG)
    
    # 分析结果
    if trades:
        analyze_results(trades, final_capital, CONFIG['initial_capital'], df, equity_curve)


if __name__ == '__main__':
    main()

