"""
BTC动量策略灵活回测系统
支持自定义配置的简化回测框架
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta, time

# ================================================================================
# 📝 配置区域 - 在这里修改所有参数
# ================================================================================

CONFIG = {
    # === 数据设置 ===
    'data_path': 'btc_15m.csv',              # 基础K线数据文件（目前使用15分钟，可换成btc_1m.csv）
    'boundary_kline': '1d',                  # 边界计算用的K线周期：'1m', '15m', '1h', '4h', '1d'
    'boundary_data_path': 'btc_1d.csv',      # 边界K线数据文件（如果与data_path不同）
    
    # === 回测时间 ===
    'start_date': date(2024, 11, 1),
    'end_date': date(2025, 11, 10),
    
    # === 交易参数 ===
    'initial_capital': 100000,               # 初始资金
    'check_interval_minutes': 60,            # 检查间隔（分钟），如15, 60, 240, 720等
    'K1': 0.8,                               # 上边界系数（调小，更容易触发）
    'K2': 0.8,                               # 下边界系数
    
    # === Sigma计算方式 ===
    'sigma_method': 'rolling_window',        # 'time_based'(按时间点) 或 'rolling_window'(滚动窗口)
    'lookback_days': 1,                      # time_based方法: 回溯天数
    'rolling_window_hours': 12,               # rolling_window方法: 滚动窗口小时数（最佳: 12小时）
    
    # === 交易方向 ===
    'enable_long': True,                     # 是否允许做多
    'enable_short': False,                   # 是否允许做空（牛市禁用！）
    
    # === 交易日分割设置 ===
    'use_custom_day_split': True,            # 是否使用自定义交易日分割
    'day_split_hour': 0,                     # 交易日分割时刻（小时，0-23），例如0表示凌晨0点
    'force_close_at_split': False,           # 取消强制平仓，让趋势跑起来！
    
    # === 其他设置 ===
    'print_trades': True,                    # 是否打印每笔交易
    'print_daily_summary': True,             # 是否打印每日汇总
}

# ================================================================================
# 核心回测函数
# ================================================================================

def load_and_prepare_data(config):
    """加载并准备数据"""
    print(f"\n{'='*80}")
    print(f"📊 BTC动量策略回测 - 灵活配置版")
    print(f"{'='*80}\n")
    
    # 1. 加载1分钟K线数据
    print(f"📂 加载数据: {config['data_path']}")
    df = pd.read_csv(config['data_path'], parse_dates=['DateTime'])
    df.sort_values('DateTime', inplace=True)
    df['Date'] = df['DateTime'].dt.date
    df['Time'] = df['DateTime'].dt.strftime('%H:%M')
    df['Hour'] = df['DateTime'].dt.hour
    df['Minute'] = df['DateTime'].dt.minute
    
    print(f"   ✓ 数据点数: {len(df):,}条")
    print(f"   ✓ 时间范围: {df['DateTime'].min()} 至 {df['DateTime'].max()}\n")
    
    # 2. 如果使用自定义交易日分割，重新计算Date
    if config['use_custom_day_split']:
        split_hour = config['day_split_hour']
        print(f"🕐 使用自定义交易日分割: 每日{split_hour}点作为交易日分界")
        
        # 调整Date：如果时间早于split_hour，归入前一天
        df['TradingDate'] = df.apply(
            lambda row: row['Date'] if row['Hour'] >= split_hour 
            else (pd.Timestamp(row['Date']) - pd.Timedelta(days=1)).date(),
            axis=1
        )
        print(f"   ✓ 交易日已重新计算\n")
    else:
        df['TradingDate'] = df['Date']
        print(f"📅 使用自然日作为交易日\n")
    
    # 3. 加载边界计算用的K线数据
    boundary_kline = config['boundary_kline']
    print(f"📈 边界计算周期: {boundary_kline}")
    
    if boundary_kline == '1d':
        # 使用日K线
        boundary_path = config.get('boundary_data_path', 'btc_1d.csv')
        print(f"   加载日K线: {boundary_path}")
        daily_df = pd.read_csv(boundary_path, parse_dates=['DateTime'])
        daily_df['TradingDate'] = daily_df['DateTime'].dt.date
        
        # 提取开盘和收盘价
        boundary_df = daily_df[['TradingDate', 'Open', 'Close']].rename(
            columns={'Open': 'BoundaryOpen', 'Close': 'BoundaryClose'}
        )
    else:
        # 从1分钟数据重采样得到指定周期的K线
        print(f"   从1分钟数据重采样到 {boundary_kline}")
        
        # 设置重采样频率
        freq_map = {'1m': '1T', '15m': '15T', '1h': '1H', '4h': '4H'}
        freq = freq_map.get(boundary_kline, '1H')
        
        # 按TradingDate分组重采样
        resampled_data = []
        for trading_date, group in df.groupby('TradingDate'):
            group_resampled = group.set_index('DateTime').resample(freq).agg({
                'Open': 'first',
                'Close': 'last'
            }).dropna()
            
            if len(group_resampled) > 0:
                resampled_data.append({
                    'TradingDate': trading_date,
                    'BoundaryOpen': group_resampled['Open'].iloc[0],
                    'BoundaryClose': group_resampled['Close'].iloc[-1]
                })
        
        boundary_df = pd.DataFrame(resampled_data)
    
    # 4. 合并边界数据到主DataFrame
    df = df.merge(boundary_df, on='TradingDate', how='left')
    df['BoundaryOpen'] = df['BoundaryOpen'].ffill()
    df['BoundaryClose'] = df['BoundaryClose'].ffill()
    
    print(f"   ✓ 边界数据已合并\n")
    
    # 5. 筛选日期范围
    start_date = config['start_date']
    end_date = config['end_date']
    df = df[(df['TradingDate'] >= start_date) & (df['TradingDate'] <= end_date)]
    
    print(f"📅 回测期间: {start_date} 至 {end_date}")
    print(f"   ✓ 筛选后数据: {len(df):,}条\n")
    
    return df


def calculate_boundaries(df, config):
    """计算交易边界"""
    print(f"📐 计算交易边界...")
    
    K1 = config['K1']
    K2 = config['K2']
    sigma_method = config.get('sigma_method', 'time_based')
    
    # 1. 计算每日的参考价格
    df['prev_close'] = df.groupby('TradingDate')['BoundaryClose'].transform('first').shift(1)
    df['day_open'] = df.groupby('TradingDate')['BoundaryOpen'].transform('first')
    
    # 参考价格
    df['upper_ref'] = df[['day_open', 'prev_close']].max(axis=1)
    df['lower_ref'] = df[['day_open', 'prev_close']].min(axis=1)
    
    # 2. 计算每分钟相对开盘的回报率
    df['ret'] = df['Close'] / df['day_open'] - 1
    
    # 3. 计算噪声区间（sigma）- 支持两种方法
    if sigma_method == 'time_based':
        # 方法A: 基于历史同时刻的波动（原方法，适合美股）
        lookback_days = config['lookback_days']
        print(f"   Sigma方法: 时间点历史波动 (lookback={lookback_days}天)")
        
        pivot = df.pivot(index='TradingDate', columns='Time', values='ret').abs()
        sigma = pivot.rolling(window=lookback_days, min_periods=lookback_days).mean().shift(1)
        sigma = sigma.stack().reset_index(name='sigma')
        
        # 合并sigma
        df = df.merge(sigma, on=['TradingDate', 'Time'], how='left')
        
    elif sigma_method == 'rolling_window':
        # 方法B: 滚动时间窗口波动率（适合BTC 24小时交易）
        rolling_hours = config.get('rolling_window_hours', 4)
        window_size = int(rolling_hours * 60 / 15)  # 转换为15分钟K线数量
        
        print(f"   Sigma方法: 滚动窗口波动率 (窗口={rolling_hours}小时, {window_size}根K线)")
        
        # 计算滚动窗口的波动率
        # 使用expanding window避免未来数据泄露
        df['abs_ret'] = df['ret'].abs()
        df['sigma'] = df['abs_ret'].rolling(window=window_size, min_periods=window_size).mean().shift(1)
        
        # 填充初始NaN值（使用全局平均）
        global_sigma = df['abs_ret'].mean()
        df['sigma'] = df['sigma'].fillna(global_sigma)
        
    else:
        raise ValueError(f"未知的sigma_method: {sigma_method}")
    
    # 4. 计算上下边界
    df['upper_bound'] = df['upper_ref'] * (1 + K1 * df['sigma'])
    df['lower_bound'] = df['lower_ref'] * (1 - K2 * df['sigma'])
    
    # 去除无效数据
    df = df.dropna(subset=['upper_bound', 'lower_bound'])
    
    print(f"   ✓ K1={K1}, K2={K2}")
    print(f"   ✓ 有效数据: {len(df):,}条")
    print(f"   ✓ Sigma统计: 均值={df['sigma'].mean():.6f}, 中位数={df['sigma'].median():.6f}\n")
    
    return df


def run_backtest(df, config):
    """执行回测"""
    print(f"{'='*80}")
    print(f"🚀 开始回测交易...")
    print(f"{'='*80}\n")
    
    # 配置参数
    initial_capital = config['initial_capital']
    check_interval = config['check_interval_minutes']
    enable_long = config['enable_long']
    enable_short = config['enable_short']
    force_close_at_split = config['force_close_at_split']
    split_hour = config['day_split_hour']
    print_trades = config.get('print_trades', True)
    print_daily = config.get('print_daily_summary', True)
    
    print(f"💰 初始资金: ${initial_capital:,.0f}")
    print(f"⏱️  检查间隔: {check_interval}分钟")
    print(f"📊 交易方向: ", end="")
    if enable_long and enable_short:
        print("做多 + 做空")
    elif enable_long:
        print("仅做多")
    elif enable_short:
        print("仅做空")
    else:
        print("❌ 错误：至少需要启用一个交易方向！")
        return None
    
    if config['use_custom_day_split'] and force_close_at_split:
        print(f"🕐 强制平仓时间: 每日{split_hour}:00")
    
    print(f"\n{'-'*80}\n")
    
    # 筛选检查时间点
    check_times = df[df['Minute'] % check_interval == 0].copy()
    
    # 回测变量
    capital = initial_capital
    position = None
    trades = []
    daily_trades = {}
    
    # 遍历每个检查点
    for idx, row in check_times.iterrows():
        current_time = row['DateTime']
        current_date = row['TradingDate']
        current_price = row['Close']
        upper_bound = row['upper_bound']
        lower_bound = row['lower_bound']
        current_hour = row['Hour']
        
        # === 检查是否需要强制平仓（到达交易日分割点） ===
        if position and force_close_at_split and current_hour == split_hour and row['Minute'] == 0:
            shares = position['shares']
            entry_price = position['entry_price']
            direction = position['direction']
            
            if direction == 'long':
                pnl = (current_price - entry_price) * shares
            else:  # short
                pnl = (entry_price - current_price) * shares
            
            capital += pnl
            
            trade = {
                'entry_time': position['entry_time'],
                'exit_time': current_time,
                'date': current_date,
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': current_price,
                'shares': shares,
                'pnl': pnl,
                'reason': 'Day Split Force Close'
            }
            trades.append(trade)
            
            if print_trades:
                print(f"[{current_time}] 平仓 {direction.upper():5} | "
                      f"入:{entry_price:8.1f} 出:{current_price:8.1f} | "
                      f"盈亏:${pnl:+10,.2f} | 原因:交易日分割")
            
            # 记录每日交易
            if current_date not in daily_trades:
                daily_trades[current_date] = []
            daily_trades[current_date].append(trade)
            
            position = None
        
        # === 持仓管理 ===
        if position:
            should_close = False
            close_reason = ""
            
            # 获取当天的最后一个检查点（在交易日结束前）
            next_split_time = None
            if config['use_custom_day_split']:
                # 找到下一个split时刻
                if current_hour < split_hour:
                    next_split_time = current_time.replace(hour=split_hour, minute=0, second=0)
                else:
                    next_split_time = (current_time + timedelta(days=1)).replace(hour=split_hour, minute=0, second=0)
            
            # 检查是否是当天最后一个检查点（在分割点之前）
            same_day_checks = check_times[check_times['TradingDate'] == current_date]
            if next_split_time:
                same_day_checks = same_day_checks[same_day_checks['DateTime'] < next_split_time]
            
            is_last_check = (same_day_checks.index[-1] == idx) if len(same_day_checks) > 0 else False
            
            if is_last_check and not force_close_at_split:
                should_close = True
                close_reason = "Market Close"
            elif position['direction'] == 'long':
                # 多头止损：跌破下边界
                if current_price < lower_bound:
                    should_close = True
                    close_reason = "Stop Loss"
            else:  # short
                # 空头止损：突破上边界
                if current_price > upper_bound:
                    should_close = True
                    close_reason = "Stop Loss"
            
            # 平仓
            if should_close:
                shares = position['shares']
                entry_price = position['entry_price']
                direction = position['direction']
                
                if direction == 'long':
                    pnl = (current_price - entry_price) * shares
                else:  # short
                    pnl = (entry_price - current_price) * shares
                
                capital += pnl
                
                trade = {
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'date': current_date,
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'shares': shares,
                    'pnl': pnl,
                    'reason': close_reason
                }
                trades.append(trade)
                
                if print_trades:
                    print(f"[{current_time}] 平仓 {direction.upper():5} | "
                          f"入:{entry_price:8.1f} 出:{current_price:8.1f} | "
                          f"盈亏:${pnl:+10,.2f} | 原因:{close_reason}")
                
                # 记录每日交易
                if current_date not in daily_trades:
                    daily_trades[current_date] = []
                daily_trades[current_date].append(trade)
                
                position = None
        
        # === 开仓信号 ===
        if not position:
            direction = None
            
            # 突破上边界
            if current_price > upper_bound:
                if enable_long:
                    direction = 'long'
            # 突破下边界
            elif current_price < lower_bound:
                if enable_short:
                    direction = 'short'
            
            # 开仓
            if direction:
                shares = int(capital / current_price)
                if shares > 0:
                    position = {
                        'direction': direction,
                        'entry_price': current_price,
                        'entry_time': current_time,
                        'shares': shares
                    }
                    
                    if print_trades:
                        print(f"[{current_time}] 开仓 {direction.upper():5} | "
                              f"价格:{current_price:8.1f} 股数:{shares} | "
                              f"上界:{upper_bound:8.1f} 下界:{lower_bound:8.1f}")
    
    # === 打印每日汇总 ===
    if print_daily and len(daily_trades) > 0:
        print(f"\n{'-'*80}")
        print(f"📅 每日交易汇总")
        print(f"{'-'*80}\n")
        
        for trading_date in sorted(daily_trades.keys()):
            day_trades = daily_trades[trading_date]
            day_pnl = sum([t['pnl'] for t in day_trades])
            print(f"{trading_date} | 交易:{len(day_trades):2}笔 | 盈亏:${day_pnl:+10,.2f}")
    
    return {
        'trades': trades,
        'final_capital': capital,
        'daily_trades': daily_trades
    }


def analyze_results(result, config):
    """分析回测结果"""
    if not result or not result['trades']:
        print("\n⚠️  没有产生任何交易！\n")
        return
    
    trades = pd.DataFrame(result['trades'])
    initial_capital = config['initial_capital']
    final_capital = result['final_capital']
    
    # 基础指标
    total_return = (final_capital - initial_capital) / initial_capital * 100
    total_trades = len(trades)
    winning_trades = len(trades[trades['pnl'] > 0])
    losing_trades = len(trades[trades['pnl'] < 0])
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    # 分多空统计
    long_trades = trades[trades['direction'] == 'long']
    short_trades = trades[trades['direction'] == 'short']
    
    # 计算最大回撤
    daily_trades = result['daily_trades']
    daily_capital = [initial_capital]
    running_capital = initial_capital
    
    for d in sorted(daily_trades.keys()):
        for t in daily_trades[d]:
            running_capital += t['pnl']
            daily_capital.append(running_capital)
    
    peak = daily_capital[0]
    max_drawdown = 0
    for cap in daily_capital:
        if cap > peak:
            peak = cap
        dd = (peak - cap) / peak
        if dd > max_drawdown:
            max_drawdown = dd
    
    # 输出结果
    print(f"\n{'='*80}")
    print(f"📊 回测结果分析")
    print(f"{'='*80}\n")
    
    print(f"💰 资金表现:")
    print(f"   初始资金: ${initial_capital:,.0f}")
    print(f"   最终资金: ${final_capital:,.0f}")
    print(f"   总回报率: {total_return:+.2f}%")
    print(f"   最大回撤: {max_drawdown*100:.2f}%")
    
    print(f"\n📈 交易统计:")
    print(f"   总交易次数: {total_trades}笔")
    print(f"   做多交易: {len(long_trades)}笔 (盈亏:${long_trades['pnl'].sum():+,.2f})")
    print(f"   做空交易: {len(short_trades)}笔 (盈亏:${short_trades['pnl'].sum():+,.2f})")
    print(f"   胜率: {win_rate:.1f}%")
    print(f"   盈利交易: {winning_trades}笔 (平均:${trades[trades['pnl']>0]['pnl'].mean() if winning_trades>0 else 0:,.2f})")
    print(f"   亏损交易: {losing_trades}笔 (平均:${trades[trades['pnl']<0]['pnl'].mean() if losing_trades>0 else 0:,.2f})")
    
    if winning_trades > 0 and losing_trades > 0:
        avg_win = trades[trades['pnl'] > 0]['pnl'].mean()
        avg_loss = abs(trades[trades['pnl'] < 0]['pnl'].mean())
        profit_factor = (avg_win * winning_trades) / (avg_loss * losing_trades)
        print(f"   盈亏比: {profit_factor:.2f}")
    
    print(f"\n📊 最佳/最差交易:")
    best = trades.loc[trades['pnl'].idxmax()]
    worst = trades.loc[trades['pnl'].idxmin()]
    print(f"   最佳: {best['date']} {best['direction']} ${best['pnl']:+,.2f}")
    print(f"   最差: {worst['date']} {worst['direction']} ${worst['pnl']:+,.2f}")
    
    # 时间统计
    trading_days = len(daily_trades)
    total_days = (config['end_date'] - config['start_date']).days + 1
    
    print(f"\n📅 时间统计:")
    print(f"   回测天数: {total_days}天")
    print(f"   交易天数: {trading_days}天")
    print(f"   平均每交易日: {total_trades/trading_days:.1f}笔")
    
    print(f"\n{'='*80}\n")


# ================================================================================
# 主程序
# ================================================================================

if __name__ == "__main__":
    # 显示当前配置
    print("\n" + "="*80)
    print("⚙️  当前配置")
    print("="*80)
    print(f"数据文件: {CONFIG['data_path']}")
    print(f"边界周期: {CONFIG['boundary_kline']}")
    print(f"回测时间: {CONFIG['start_date']} 至 {CONFIG['end_date']}")
    print(f"检查间隔: {CONFIG['check_interval_minutes']}分钟")
    print(f"边界参数: K1={CONFIG['K1']}, K2={CONFIG['K2']}")
    print(f"Sigma方法: {CONFIG.get('sigma_method', 'time_based')}", end="")
    if CONFIG.get('sigma_method') == 'rolling_window':
        print(f" (窗口={CONFIG.get('rolling_window_hours', 4)}小时)")
    else:
        print(f" (lookback={CONFIG.get('lookback_days', 1)}天)")
    print(f"交易方向: 做多={CONFIG['enable_long']}, 做空={CONFIG['enable_short']}")
    print(f"自定义交易日: {CONFIG['use_custom_day_split']} (分割时刻:{CONFIG['day_split_hour']}点)")
    print("="*80)
    
    # 执行回测
    df = load_and_prepare_data(CONFIG)
    df = calculate_boundaries(df, CONFIG)
    result = run_backtest(df, CONFIG)
    analyze_results(result, CONFIG)

