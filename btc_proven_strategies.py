"""
BTC日内趋势策略集合
基于网络搜索到的经过验证的策略

策略来源：
1. 双重动量策略 (胜率59.3%, 盈亏比1:2.7, 回撤12.4%)
2. ATR突破策略 (年化收益200%+)
3. TSI真实强弱指标策略 (年化收益119%)
4. 均值回归+趋势跟随 (年化收益98.43%, 夏普2.06)
"""

import pandas as pd
import numpy as np
from datetime import date, datetime

# ================================================================================
# 配置区域 - 选择要测试的策略
# ================================================================================

CONFIG = {
    # 数据设置
    'data_path': 'btc_15m.csv',
    'start_date': date(2024, 11, 1),
    'end_date': date(2025, 11, 10),
    'initial_capital': 100000,
    
    # 选择策略 (1-4)
    'strategy': 1,  # 1=双重动量, 2=ATR突破, 3=TSI策略, 4=均值回归
    
    # 策略1: 双重动量参数
    'dual_momentum': {
        'long_period': 96,      # 24小时 (96根15分钟K线)
        'short_period': 16,     # 4小时 (16根15分钟K线)
        'ma_type': 'ema',       # 'sma' or 'ema'
    },
    
    # 策略2: ATR突破参数
    'atr_breakout': {
        'atr_period': 14,
        'atr_multiplier': 2.0,  # ATR倍数
        'lookback': 20,         # 回看周期
    },
    
    # 策略3: TSI策略参数
    'tsi': {
        'long_period': 25,
        'short_period': 13,
        'signal_period': 13,
        'threshold': 0,         # 信号阈值
    },
    
    # 策略4: 均值回归参数
    'mean_reversion': {
        'lookback_days': 10,    # 回看天数
        'hold_days': 1,         # 持有天数
    },
    
    # 通用设置
    'stop_loss_pct': 0.03,      # 3%止损
    'take_profit_pct': 0.08,    # 8%止盈
    'use_trailing_stop': True,  # 使用跟踪止损
    'trailing_stop_pct': 0.02,  # 2%跟踪止损
    
    'print_trades': True,
}


# ================================================================================
# 策略1: 双重动量策略
# ================================================================================

def strategy_dual_momentum(df, config):
    """
    双重动量策略：同时监测长短周期动量
    胜率: 59.3%, 盈亏比: 1:2.7, 最大回撤: 12.4%
    """
    params = config['dual_momentum']
    long_period = params['long_period']
    short_period = params['short_period']
    
    print(f"\n策略: 双重动量")
    print(f"  长周期: {long_period}根K线 ({long_period*15/60:.1f}小时)")
    print(f"  短周期: {short_period}根K线 ({short_period*15/60:.1f}小时)")
    
    # 计算动量指标
    if params['ma_type'] == 'ema':
        df['Long_MA'] = df['Close'].ewm(span=long_period, adjust=False).mean()
        df['Short_MA'] = df['Close'].ewm(span=short_period, adjust=False).mean()
    else:
        df['Long_MA'] = df['Close'].rolling(window=long_period).mean()
        df['Short_MA'] = df['Close'].rolling(window=short_period).mean()
    
    # 计算动量方向
    df['Long_Momentum'] = (df['Close'] > df['Long_MA']).astype(int)  # 1=上涨, 0=下跌
    df['Short_Momentum'] = (df['Close'] > df['Short_MA']).astype(int)
    
    # 信号生成
    df['Signal'] = 0
    # 双重动量同向上涨 -> 做多
    df.loc[(df['Long_Momentum'] == 1) & (df['Short_Momentum'] == 1), 'Signal'] = 1
    # 双重动量同向下跌 -> 做空（可选）
    # df.loc[(df['Long_Momentum'] == 0) & (df['Short_Momentum'] == 0), 'Signal'] = -1
    
    return df


# ================================================================================
# 策略2: ATR突破策略
# ================================================================================

def calculate_atr(df, period=14):
    """计算ATR指标"""
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].ewm(span=period, adjust=False).mean()
    return df


def strategy_atr_breakout(df, config):
    """
    ATR突破策略：动态突破区间
    年化收益率: 200%+
    """
    params = config['atr_breakout']
    atr_period = params['atr_period']
    atr_mult = params['atr_multiplier']
    lookback = params['lookback']
    
    print(f"\n策略: ATR突破")
    print(f"  ATR周期: {atr_period}")
    print(f"  ATR倍数: {atr_mult}")
    print(f"  回看周期: {lookback}")
    
    # 计算ATR
    df = calculate_atr(df, atr_period)
    
    # 计算动态突破区间
    df['Lowest'] = df['Low'].rolling(window=lookback).min()
    df['Upper_Band'] = df['Lowest'] + atr_mult * df['ATR']
    
    # 计算趋势（MA作为趋势过滤）
    df['MA'] = df['Close'].rolling(window=20).mean()
    df['Trend'] = (df['Close'] > df['MA']).astype(int)
    
    # 信号生成：突破上轨且在上升趋势中
    df['Signal'] = 0
    df.loc[(df['Close'] > df['Upper_Band']) & (df['Trend'] == 1), 'Signal'] = 1
    
    return df


# ================================================================================
# 策略3: TSI真实强弱指标策略
# ================================================================================

def calculate_tsi(df, long_period=25, short_period=13, signal_period=13):
    """计算TSI指标"""
    # 计算价格动量
    df['Momentum'] = df['Close'] - df['Close'].shift(1)
    
    # 双重平滑动量
    df['Smooth_Momentum'] = df['Momentum'].ewm(span=long_period, adjust=False).mean()
    df['Double_Smooth_Momentum'] = df['Smooth_Momentum'].ewm(span=short_period, adjust=False).mean()
    
    # 双重平滑绝对动量
    df['Abs_Momentum'] = abs(df['Momentum'])
    df['Smooth_Abs'] = df['Abs_Momentum'].ewm(span=long_period, adjust=False).mean()
    df['Double_Smooth_Abs'] = df['Smooth_Abs'].ewm(span=short_period, adjust=False).mean()
    
    # TSI = 100 * (双重平滑动量 / 双重平滑绝对动量)
    df['TSI'] = 100 * df['Double_Smooth_Momentum'] / df['Double_Smooth_Abs']
    
    # 信号线
    df['TSI_Signal'] = df['TSI'].ewm(span=signal_period, adjust=False).mean()
    
    return df


def strategy_tsi(df, config):
    """
    TSI策略：真实强弱指标
    年化收益率: 119%, 夏普比率: 2.25-2.30
    """
    params = config['tsi']
    
    print(f"\n策略: TSI真实强弱指标")
    print(f"  长周期: {params['long_period']}")
    print(f"  短周期: {params['short_period']}")
    print(f"  信号周期: {params['signal_period']}")
    
    # 计算TSI
    df = calculate_tsi(df, params['long_period'], params['short_period'], params['signal_period'])
    
    # 信号生成
    df['Signal'] = 0
    # TSI上穿信号线 -> 做多
    df.loc[(df['TSI'] > df['TSI_Signal']) & (df['TSI'].shift(1) <= df['TSI_Signal'].shift(1)), 'Signal'] = 1
    # TSI下穿信号线 -> 平仓或做空
    df.loc[(df['TSI'] < df['TSI_Signal']) & (df['TSI'].shift(1) >= df['TSI_Signal'].shift(1)), 'Signal'] = -1
    
    return df


# ================================================================================
# 策略4: 均值回归+趋势跟随
# ================================================================================

def strategy_mean_reversion(df, config):
    """
    均值回归+趋势跟随策略
    年化收益率: 98.43%, 夏普比率: 2.06
    """
    params = config['mean_reversion']
    lookback_days = params['lookback_days']
    
    # 转换为每日数据的lookback（假设一天96根15分钟K线）
    lookback_bars = lookback_days * 96
    
    print(f"\n策略: 均值回归+趋势跟随")
    print(f"  回看天数: {lookback_days}天")
    
    # 计算过去N天的最高价和最低价
    df['MAX'] = df['High'].rolling(window=lookback_bars).max()
    df['MIN'] = df['Low'].rolling(window=lookback_bars).min()
    
    # 信号生成：价格触及极值
    df['Signal'] = 0
    df.loc[df['Close'] >= df['MAX'] * 0.999, 'Signal'] = 1   # 接近最高价（容差0.1%）
    df.loc[df['Close'] <= df['MIN'] * 1.001, 'Signal'] = 1   # 接近最低价（容差0.1%）
    
    return df


# ================================================================================
# 回测引擎
# ================================================================================

def run_backtest(df, config):
    """执行回测"""
    initial_capital = config['initial_capital']
    stop_loss_pct = config['stop_loss_pct']
    take_profit_pct = config['take_profit_pct']
    use_trailing = config['use_trailing_stop']
    trailing_pct = config['trailing_stop_pct']
    print_trades = config['print_trades']
    
    capital = initial_capital
    position = None
    trades = []
    equity_curve = []  # 记录每日资金曲线
    
    print(f"\n{'='*80}")
    print(f"开始回测...")
    print(f"{'='*80}\n")
    
    for idx, row in df.iterrows():
        if pd.isna(row['Signal']):
            continue
        
        current_price = row['Close']
        current_time = row['DateTime']
        signal = row['Signal']
        
        # 记录当前权益
        current_equity = capital
        if position:
            unrealized_pnl = (current_price - position['entry_price']) * position['shares']
            current_equity = capital + unrealized_pnl
        equity_curve.append({
            'DateTime': current_time,
            'Equity': current_equity
        })
        
        # 持仓管理
        if position:
            entry_price = position['entry_price']
            highest_price = position.get('highest_price', entry_price)
            shares = position['shares']
            
            # 更新最高价
            if current_price > highest_price:
                highest_price = current_price
                position['highest_price'] = highest_price
            
            # 检查止损/止盈
            pnl_pct = (current_price - entry_price) / entry_price
            
            should_close = False
            close_reason = ""
            
            # 固定止损
            if pnl_pct <= -stop_loss_pct:
                should_close = True
                close_reason = "Stop Loss"
            
            # 固定止盈
            elif pnl_pct >= take_profit_pct:
                should_close = True
                close_reason = "Take Profit"
            
            # 跟踪止损
            elif use_trailing and pnl_pct > 0:
                trailing_stop_price = highest_price * (1 - trailing_pct)
                if current_price < trailing_stop_price:
                    should_close = True
                    close_reason = "Trailing Stop"
            
            # 反向信号
            elif signal == -1 and position['direction'] == 'long':
                should_close = True
                close_reason = "Reverse Signal"
            
            # 平仓
            if should_close:
                pnl = (current_price - entry_price) * shares
                capital += pnl
                
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'direction': position['direction'],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'shares': shares,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct * 100,
                    'reason': close_reason
                })
                
                if print_trades:
                    print(f"[{current_time}] 平仓 | "
                          f"入:{entry_price:.1f} 出:{current_price:.1f} | "
                          f"盈亏:${pnl:+,.2f} ({pnl_pct*100:+.2f}%) | {close_reason}")
                
                position = None
        
        # 开仓信号
        if not position and signal == 1:
            shares = int(capital / current_price)
            if shares > 0:
                position = {
                    'direction': 'long',
                    'entry_price': current_price,
                    'entry_time': current_time,
                    'shares': shares,
                    'highest_price': current_price
                }
                
                if print_trades:
                    print(f"[{current_time}] 开仓 | 价格:{current_price:.1f} 股数:{shares}")
    
    # 强制平仓最后持仓
    if position:
        current_price = df.iloc[-1]['Close']
        entry_price = position['entry_price']
        shares = position['shares']
        pnl = (current_price - entry_price) * shares
        capital += pnl
        
        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': df.iloc[-1]['DateTime'],
            'direction': position['direction'],
            'entry_price': entry_price,
            'exit_price': current_price,
            'shares': shares,
            'pnl': pnl,
            'pnl_pct': pnl / entry_price * 100,
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
        return
    
    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)
    
    total_return = (final_capital - initial_capital) / initial_capital * 100
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
    profit_factor = (avg_win * winning_trades) / (avg_loss * losing_trades) if losing_trades > 0 else float('inf')
    
    # 计算收益率序列
    equity_df['Returns'] = equity_df['Equity'].pct_change()
    
    # 计算年化收益率
    total_days = (df['DateTime'].max() - df['DateTime'].min()).days
    years = total_days / 365.25
    annualized_return = (final_capital / initial_capital) ** (1 / years) - 1 if years > 0 else 0
    
    # 计算波动率（年化）
    daily_returns = equity_df.groupby(equity_df['DateTime'].dt.date)['Returns'].last()
    volatility = daily_returns.std() * np.sqrt(365.25) if len(daily_returns) > 1 else 0
    
    # 计算夏普比率 (假设无风险利率为0)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # 计算最大回撤
    equity_df['Cummax'] = equity_df['Equity'].cummax()
    equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['Cummax']) / equity_df['Cummax'] * 100
    max_drawdown = equity_df['Drawdown'].min()
    
    # 计算Calmar比率
    calmar_ratio = annualized_return / abs(max_drawdown) * 100 if max_drawdown != 0 else 0
    
    # Buy & Hold
    first_price = df.iloc[0]['Close']
    last_price = df.iloc[-1]['Close']
    bh_return = (last_price - first_price) / first_price * 100
    bh_annualized = (last_price / first_price) ** (1 / years) - 1 if years > 0 else 0
    
    # 计算Buy & Hold的波动率和夏普比率
    df['BH_Returns'] = df['Close'].pct_change()
    bh_daily_returns = df.groupby(df['DateTime'].dt.date)['BH_Returns'].last()
    bh_volatility = bh_daily_returns.std() * np.sqrt(365.25) if len(bh_daily_returns) > 1 else 0
    bh_sharpe = bh_annualized / bh_volatility if bh_volatility > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"📊 回测结果")
    print(f"{'='*80}\n")
    
    print(f"💰 资金表现:")
    print(f"   初始资金: ${initial_capital:,.0f}")
    print(f"   最终资金: ${final_capital:,.0f}")
    print(f"   总回报率: {total_return:+.2f}%")
    print(f"   年化收益率: {annualized_return*100:+.2f}%")
    
    print(f"\n📊 风险指标:")
    print(f"   年化波动率: {volatility*100:.2f}%")
    print(f"   最大回撤: {max_drawdown:.2f}%")
    print(f"   夏普比率: {sharpe_ratio:.3f}")
    print(f"   Calmar比率: {calmar_ratio:.3f}")
    
    print(f"\n📈 交易统计:")
    print(f"   总交易次数: {total_trades}笔")
    print(f"   胜率: {win_rate:.1f}%")
    print(f"   盈利交易: {winning_trades}笔 (平均: ${avg_win:,.2f})")
    print(f"   亏损交易: {losing_trades}笔 (平均: ${avg_loss:,.2f})")
    print(f"   盈亏比: {profit_factor:.2f}")
    
    print(f"\n🎯 vs Buy & Hold:")
    print(f"   B&H总回报: {bh_return:+.2f}%")
    print(f"   B&H年化: {bh_annualized*100:+.2f}%")
    print(f"   B&H夏普: {bh_sharpe:.3f}")
    print(f"   策略超额回报: {total_return - bh_return:+.2f}%")
    print(f"   夏普比率优势: {sharpe_ratio - bh_sharpe:+.3f}")
    
    print(f"\n📊 最佳/最差交易:")
    best = trades_df.loc[trades_df['pnl'].idxmax()]
    worst = trades_df.loc[trades_df['pnl'].idxmin()]
    print(f"   最佳: ${best['pnl']:+,.2f} ({best['pnl_pct']:+.2f}%)")
    print(f"   最差: ${worst['pnl']:+,.2f} ({worst['pnl_pct']:+.2f}%)")
    
    print(f"\n{'='*80}\n")
    
    # 返回关键指标用于对比
    return {
        'total_return': total_return,
        'annualized_return': annualized_return * 100,
        'volatility': volatility * 100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'profit_factor': profit_factor
    }


# ================================================================================
# 主程序
# ================================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("BTC日内趋势策略回测系统")
    print("="*80)
    
    # 加载数据
    print(f"\n加载数据: {CONFIG['data_path']}")
    df = pd.read_csv(CONFIG['data_path'], parse_dates=['DateTime'])
    df.sort_values('DateTime', inplace=True)
    
    # 过滤日期范围
    df['Date'] = df['DateTime'].dt.date
    df = df[(df['Date'] >= CONFIG['start_date']) & (df['Date'] <= CONFIG['end_date'])]
    
    print(f"数据范围: {df['DateTime'].min()} 至 {df['DateTime'].max()}")
    print(f"数据点数: {len(df):,}条")
    
    # 选择并执行策略
    strategy_num = CONFIG['strategy']
    
    if strategy_num == 1:
        df = strategy_dual_momentum(df, CONFIG)
    elif strategy_num == 2:
        df = strategy_atr_breakout(df, CONFIG)
    elif strategy_num == 3:
        df = strategy_tsi(df, CONFIG)
    elif strategy_num == 4:
        df = strategy_mean_reversion(df, CONFIG)
    else:
        print(f"❌ 未知策略编号: {strategy_num}")
        exit(1)
    
    # 执行回测
    trades, final_capital, equity_curve = run_backtest(df, CONFIG)
    
    # 分析结果
    results = analyze_results(trades, final_capital, CONFIG['initial_capital'], df, equity_curve)

