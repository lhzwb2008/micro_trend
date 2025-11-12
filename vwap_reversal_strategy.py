#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VWAP反向策略回测系统

策略原理：
当价格过度偏离VWAP（成交量加权平均价）时进行反向交易，预期价格会回归到均值。
- 价格显著高于VWAP → 做空（预期回落）
- 价格显著低于VWAP → 做多（预期反弹）

策略特点：
1. 均值回归：利用价格偏离后的回归特性
2. 对称交易：可同时做多做空
3. 短线交易：持仓时间较短，适合日内
4. 风险控制：固定止损、止盈和时间退出
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==================== 策略配置 ====================

CONFIG = {
    # 数据文件路径
    'data_file': 'btc_15m.csv',
    
    # VWAP计算周期（单位：K线数量，15分钟K线）
    'vwap_period': 96,  # 96个15分钟 = 24小时
    
    # 偏离阈值（标准差倍数）
    'entry_std_multiplier': 2.0,  # 进场阈值：价格偏离VWAP达到2倍标准差
    
    # 或者使用固定百分比偏离（如果设置，将覆盖标准差方法）
    'use_fixed_deviation': False,  # 是否使用固定百分比偏离
    'entry_deviation_pct': 0.03,   # 固定偏离百分比（3%）
    
    # 止盈止损
    'take_profit_pct': 0.015,      # 止盈：1.5%
    'stop_loss_pct': 0.02,         # 止损：2%
    
    # 回归目标
    'exit_on_vwap_touch': True,    # 价格回归到VWAP时平仓
    'vwap_touch_threshold': 0.002, # VWAP触碰阈值（0.2%以内视为回归）
    
    # 时间控制
    'max_hold_periods': 8,         # 最大持仓周期（8个15分钟 = 2小时）
    'force_close_hour': 23,        # 强制平仓小时（UTC时间）
    'force_close_minute': 45,      # 强制平仓分钟
    
    # 交易控制
    'enable_long': True,           # 是否允许做多
    'enable_short': True,          # 是否允许做空
    
    # 信号确认
    'require_volume_confirm': True,  # 是否需要成交量确认
    'volume_ma_period': 20,         # 成交量均线周期
    'volume_threshold': 1.0,        # 成交量阈值（当前成交量/均线）
    
    # 回测设置
    'initial_capital': 10000,      # 初始资金
    'commission': 0.001,           # 手续费率（0.1%）
    'slippage': 0.0005,            # 滑点（0.05%）
    
    # 输出控制
    'print_trades': False,         # 是否打印每笔交易
    'print_daily_summary': False,  # 是否打印每日统计
}

# ==================== 预设配置 ====================

PRESETS = {
    'balanced': {  # 平衡型：8%偏离，交易次数适中（推荐）⭐
        'use_fixed_deviation': True,
        'entry_deviation_pct': 0.08,  # 8%偏离才入场
        'take_profit_pct': 0.03,
        'stop_loss_pct': 0.015,
        'max_hold_periods': 12,
        'exit_on_vwap_touch': True,
        'vwap_touch_threshold': 0.003,
        'require_volume_confirm': True,
    },
    'high_return': {  # 高收益型：10%偏离，交易次数少但盈利因子高
        'use_fixed_deviation': True,
        'entry_deviation_pct': 0.10,  # 10%偏离才入场
        'take_profit_pct': 0.035,
        'stop_loss_pct': 0.02,
        'max_hold_periods': 16,
        'exit_on_vwap_touch': True,
        'vwap_touch_threshold': 0.004,
        'require_volume_confirm': True,
    },
    'moderate': {  # 适中型：7%偏离
        'use_fixed_deviation': True,
        'entry_deviation_pct': 0.07,
        'take_profit_pct': 0.025,
        'stop_loss_pct': 0.015,
        'max_hold_periods': 12,
        'exit_on_vwap_touch': True,
        'vwap_touch_threshold': 0.003,
        'require_volume_confirm': True,
    },
    'frequent': {  # 高频型：6%偏离，更多交易机会
        'use_fixed_deviation': True,
        'entry_deviation_pct': 0.06,
        'take_profit_pct': 0.02,
        'stop_loss_pct': 0.015,
        'max_hold_periods': 10,
        'exit_on_vwap_touch': True,
        'vwap_touch_threshold': 0.003,
        'require_volume_confirm': True,
    },
}

# 选择预设（None表示使用CONFIG中的默认值）
ACTIVE_PRESET = 'balanced'  # 可选: 'balanced'⭐, 'high_return', 'moderate', 'frequent'

# ==================== 数据加载 ====================

def load_data(file_path):
    """加载BTC历史数据"""
    print(f"正在加载数据: {file_path}")
    df = pd.read_csv(file_path)
    
    # 统一列名为小写
    df.columns = df.columns.str.lower()
    
    # 处理时间列（可能是timestamp或datetime）
    if 'datetime' in df.columns:
        df.rename(columns={'datetime': 'timestamp'}, inplace=True)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"数据范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    print(f"数据点数: {len(df)}")
    print(f"字段: {df.columns.tolist()}\n")
    
    return df

# ==================== VWAP计算 ====================

def calculate_vwap(df, period):
    """
    计算VWAP（成交量加权平均价）及其标准差
    
    VWAP = Σ(价格 × 成交量) / Σ(成交量)
    """
    df = df.copy()
    
    # 典型价格（Typical Price）
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    
    # 成交量加权价格
    df['tp_volume'] = df['typical_price'] * df['volume']
    
    # 计算滚动VWAP
    df['vwap'] = df['tp_volume'].rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    
    # 计算价格偏离VWAP的标准差
    df['price_deviation'] = df['close'] - df['vwap']
    df['deviation_std'] = df['price_deviation'].rolling(window=period).std()
    
    # 计算价格相对VWAP的偏离百分比
    df['deviation_pct'] = (df['close'] - df['vwap']) / df['vwap']
    
    return df

def calculate_volume_ma(df, period):
    """计算成交量移动平均"""
    df = df.copy()
    df['volume_ma'] = df['volume'].rolling(window=period).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    return df

# ==================== 信号生成 ====================

def generate_signals(df, config):
    """
    生成VWAP反向交易信号
    
    信号逻辑：
    1. 价格过度偏离VWAP时产生信号
    2. 偏离方向决定交易方向（做反向）
    3. 可选：需要成交量确认
    """
    df = df.copy()
    
    # 初始化信号列
    df['signal'] = 0  # 0=无信号, 1=做多, -1=做空
    
    # 确定偏离阈值
    if config['use_fixed_deviation']:
        # 使用固定百分比偏离
        upper_threshold = config['entry_deviation_pct']
        lower_threshold = -config['entry_deviation_pct']
        
        # 生成信号
        df.loc[df['deviation_pct'] > upper_threshold, 'signal'] = -1  # 价格过高，做空
        df.loc[df['deviation_pct'] < lower_threshold, 'signal'] = 1   # 价格过低，做多
    else:
        # 使用标准差倍数
        multiplier = config['entry_std_multiplier']
        
        # 计算上下阈值
        df['upper_band'] = df['vwap'] + multiplier * df['deviation_std']
        df['lower_band'] = df['vwap'] - multiplier * df['deviation_std']
        
        # 生成信号
        df.loc[df['close'] > df['upper_band'], 'signal'] = -1  # 价格过高，做空
        df.loc[df['close'] < df['lower_band'], 'signal'] = 1   # 价格过低，做多
    
    # 成交量确认
    if config['require_volume_confirm']:
        # 只在成交量高于平均时产生信号
        df.loc[df['volume_ratio'] < config['volume_threshold'], 'signal'] = 0
    
    # 根据配置禁用某个方向
    if not config['enable_long']:
        df.loc[df['signal'] == 1, 'signal'] = 0
    if not config['enable_short']:
        df.loc[df['signal'] == -1, 'signal'] = 0
    
    return df

# ==================== 回测引擎 ====================

def run_backtest(df, config):
    """运行VWAP反向策略回测"""
    
    # 初始化
    capital = config['initial_capital']
    position = 0  # 0=空仓, 1=多头, -1=空头
    entry_price = 0
    entry_time = None
    entry_idx = 0
    
    trades = []
    equity_curve = []
    
    commission_rate = config['commission']
    slippage_rate = config['slippage']
    
    # 遍历数据
    for idx, row in df.iterrows():
        current_time = row['timestamp']
        current_price = row['close']
        current_signal = row['signal']
        current_vwap = row['vwap']
        
        # 记录权益曲线
        if position == 0:
            equity = capital
        elif position == 1:
            equity = capital * (1 + (current_price / entry_price - 1))
        else:  # position == -1
            equity = capital * (1 - (current_price / entry_price - 1))
        
        equity_curve.append({
            'timestamp': current_time,
            'equity': equity,
            'position': position
        })
        
        # ===== 持仓管理 =====
        if position != 0:
            # 计算持仓盈亏
            if position == 1:
                pnl_pct = (current_price / entry_price - 1)
            else:
                pnl_pct = -(current_price / entry_price - 1)
            
            hold_periods = idx - entry_idx
            
            exit_reason = None
            
            # 1. 止盈检查
            if pnl_pct >= config['take_profit_pct']:
                exit_reason = 'take_profit'
            
            # 2. 止损检查
            elif pnl_pct <= -config['stop_loss_pct']:
                exit_reason = 'stop_loss'
            
            # 3. VWAP回归检查
            elif config['exit_on_vwap_touch']:
                vwap_distance_pct = abs(current_price - current_vwap) / current_vwap
                if vwap_distance_pct <= config['vwap_touch_threshold']:
                    exit_reason = 'vwap_touch'
            
            # 4. 反向信号
            if exit_reason is None:
                if (position == 1 and current_signal == -1) or (position == -1 and current_signal == 1):
                    exit_reason = 'reverse_signal'
            
            # 5. 最大持仓时间
            if exit_reason is None and hold_periods >= config['max_hold_periods']:
                exit_reason = 'max_hold'
            
            # 6. 强制平仓时间
            if exit_reason is None:
                if current_time.hour == config['force_close_hour'] and current_time.minute >= config['force_close_minute']:
                    exit_reason = 'force_close'
            
            # 平仓
            if exit_reason:
                exit_price = current_price * (1 - slippage_rate if position == 1 else 1 + slippage_rate)
                
                if position == 1:
                    pnl = capital * (exit_price / entry_price - 1)
                else:
                    pnl = capital * (1 - exit_price / entry_price)
                
                # 扣除手续费
                entry_commission = capital * commission_rate
                exit_commission = (capital + pnl) * commission_rate
                pnl = pnl - entry_commission - exit_commission
                
                capital += pnl
                
                trade_record = {
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'direction': 'LONG' if position == 1 else 'SHORT',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'entry_vwap': df.loc[entry_idx, 'vwap'],
                    'exit_vwap': current_vwap,
                    'hold_periods': hold_periods,
                    'pnl': pnl,
                    'pnl_pct': pnl / (capital - pnl) * 100,
                    'exit_reason': exit_reason,
                    'capital_after': capital
                }
                
                trades.append(trade_record)
                
                if config['print_trades']:
                    print(f"[{exit_reason.upper()}] {trade_record['direction']} | "
                          f"Entry: {entry_price:.2f} @ {entry_time.strftime('%Y-%m-%d %H:%M')} | "
                          f"Exit: {exit_price:.2f} @ {current_time.strftime('%Y-%m-%d %H:%M')} | "
                          f"PnL: {pnl:+.2f} ({trade_record['pnl_pct']:+.2f}%) | "
                          f"Hold: {hold_periods}期 | Capital: {capital:.2f}")
                
                position = 0
        
        # ===== 开仓信号 =====
        if position == 0 and current_signal != 0:
            # 检查数据有效性
            if pd.isna(current_vwap):
                continue
            
            position = current_signal
            entry_price = current_price * (1 + slippage_rate if position == 1 else 1 - slippage_rate)
            entry_time = current_time
            entry_idx = idx
    
    # 回测结束时强制平仓
    if position != 0:
        exit_price = df.iloc[-1]['close']
        if position == 1:
            pnl = capital * (exit_price / entry_price - 1)
        else:
            pnl = capital * (1 - exit_price / entry_price)
        
        entry_commission = capital * commission_rate
        exit_commission = (capital + pnl) * commission_rate
        pnl = pnl - entry_commission - exit_commission
        capital += pnl
        
        trades.append({
            'entry_time': entry_time,
            'exit_time': df.iloc[-1]['timestamp'],
            'direction': 'LONG' if position == 1 else 'SHORT',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_vwap': df.loc[entry_idx, 'vwap'],
            'exit_vwap': df.iloc[-1]['vwap'],
            'hold_periods': len(df) - 1 - entry_idx,
            'pnl': pnl,
            'pnl_pct': pnl / (capital - pnl) * 100,
            'exit_reason': 'backtest_end',
            'capital_after': capital
        })
    
    return trades, equity_curve, capital

# ==================== 结果分析 ====================

def analyze_results(trades, equity_curve, final_capital, initial_capital, config):
    """分析回测结果"""
    
    if len(trades) == 0:
        print("\n" + "="*60)
        print("没有产生任何交易信号")
        print("="*60)
        return
    
    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)
    
    # 基础统计
    total_return = (final_capital - initial_capital) / initial_capital
    num_trades = len(trades_df)
    
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] <= 0]
    
    win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
    
    # 计算盈亏比
    profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else float('inf')
    
    # 最大回撤
    equity_df['cummax'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
    max_drawdown = equity_df['drawdown'].min()
    
    # 计算年化收益率
    start_date = equity_df['timestamp'].min()
    end_date = equity_df['timestamp'].max()
    days = (end_date - start_date).days
    years = days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # 夏普比率（假设无风险利率为0）
    equity_df['returns'] = equity_df['equity'].pct_change()
    sharpe_ratio = equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(365.25 * 96) if equity_df['returns'].std() != 0 else 0
    
    # Calmar比率
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # 方向统计
    long_trades = trades_df[trades_df['direction'] == 'LONG']
    short_trades = trades_df[trades_df['direction'] == 'SHORT']
    
    # 平仓原因统计
    exit_reasons = trades_df['exit_reason'].value_counts()
    
    # 持仓时间统计
    avg_hold = trades_df['hold_periods'].mean()
    
    # ===== 打印结果 =====
    print("\n" + "="*60)
    print("VWAP反向策略回测结果")
    print("="*60)
    
    print("\n【策略配置】")
    print(f"VWAP周期: {config['vwap_period']}个K线")
    if config['use_fixed_deviation']:
        print(f"进场偏离: ±{config['entry_deviation_pct']*100:.2f}% (固定百分比)")
    else:
        print(f"进场偏离: ±{config['entry_std_multiplier']:.1f}倍标准差")
    print(f"止盈/止损: {config['take_profit_pct']*100:.2f}% / {config['stop_loss_pct']*100:.2f}%")
    print(f"最大持仓: {config['max_hold_periods']}个周期")
    print(f"VWAP回归: {'开启' if config['exit_on_vwap_touch'] else '关闭'}")
    print(f"成交量确认: {'开启' if config['require_volume_confirm'] else '关闭'}")
    
    print("\n【整体表现】")
    print(f"初始资金: ${initial_capital:,.2f}")
    print(f"最终资金: ${final_capital:,.2f}")
    print(f"总收益: ${final_capital - initial_capital:+,.2f}")
    print(f"总收益率: {total_return*100:+.2f}%")
    print(f"年化收益率: {annual_return*100:+.2f}%")
    print(f"夏普比率: {sharpe_ratio:.3f}")
    print(f"最大回撤: {max_drawdown*100:.2f}%")
    print(f"Calmar比率: {calmar_ratio:.3f}")
    
    print("\n【交易统计】")
    print(f"总交易次数: {num_trades}")
    print(f"  └─ 多头: {len(long_trades)} ({len(long_trades)/num_trades*100:.1f}%)")
    print(f"  └─ 空头: {len(short_trades)} ({len(short_trades)/num_trades*100:.1f}%)")
    print(f"胜率: {win_rate*100:.2f}%")
    print(f"  └─ 盈利交易: {len(winning_trades)}")
    print(f"  └─ 亏损交易: {len(losing_trades)}")
    print(f"盈亏比: {avg_win/avg_loss:.2f}" if avg_loss > 0 else "盈亏比: N/A")
    print(f"盈利因子: {profit_factor:.2f}")
    print(f"平均盈利: ${avg_win:,.2f}")
    print(f"平均亏损: ${avg_loss:,.2f}")
    print(f"平均持仓: {avg_hold:.1f}个周期 ({avg_hold*15:.0f}分钟)")
    
    print("\n【方向表现】")
    if len(long_trades) > 0:
        long_win_rate = len(long_trades[long_trades['pnl'] > 0]) / len(long_trades)
        long_pnl = long_trades['pnl'].sum()
        print(f"多头: {len(long_trades)}笔 | 胜率{long_win_rate*100:.1f}% | 盈亏${long_pnl:+,.2f}")
    
    if len(short_trades) > 0:
        short_win_rate = len(short_trades[short_trades['pnl'] > 0]) / len(short_trades)
        short_pnl = short_trades['pnl'].sum()
        print(f"空头: {len(short_trades)}笔 | 胜率{short_win_rate*100:.1f}% | 盈亏${short_pnl:+,.2f}")
    
    print("\n【退出原因】")
    for reason, count in exit_reasons.items():
        print(f"{reason}: {count} ({count/num_trades*100:.1f}%)")
    
    # 最佳和最差交易
    best_trade = trades_df.loc[trades_df['pnl'].idxmax()]
    worst_trade = trades_df.loc[trades_df['pnl'].idxmin()]
    
    print("\n【极值交易】")
    print(f"最佳交易: {best_trade['direction']} | {best_trade['entry_time'].strftime('%Y-%m-%d %H:%M')} | "
          f"PnL: ${best_trade['pnl']:+,.2f} ({best_trade['pnl_pct']:+.2f}%)")
    print(f"最差交易: {worst_trade['direction']} | {worst_trade['entry_time'].strftime('%Y-%m-%d %H:%M')} | "
          f"PnL: ${worst_trade['pnl']:+,.2f} ({worst_trade['pnl_pct']:+.2f}%)")
    
    print("\n" + "="*60)
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_hold_periods': avg_hold
    }

# ==================== 主函数 ====================

def main():
    """主函数"""
    
    # 应用预设配置
    config = CONFIG.copy()
    if ACTIVE_PRESET and ACTIVE_PRESET in PRESETS:
        print(f"\n使用预设配置: {ACTIVE_PRESET}")
        config.update(PRESETS[ACTIVE_PRESET])
    
    # 加载数据
    df = load_data(config['data_file'])
    
    # 计算VWAP
    print("计算VWAP...")
    df = calculate_vwap(df, config['vwap_period'])
    
    # 计算成交量指标
    if config['require_volume_confirm']:
        print("计算成交量指标...")
        df = calculate_volume_ma(df, config['volume_ma_period'])
    
    # 生成信号
    print("生成交易信号...")
    df = generate_signals(df, config)
    
    signal_count = (df['signal'] != 0).sum()
    long_signals = (df['signal'] == 1).sum()
    short_signals = (df['signal'] == -1).sum()
    print(f"信号统计: 总计{signal_count}个 (多头{long_signals}个, 空头{short_signals}个)")
    
    # 运行回测
    print("\n开始回测...")
    trades, equity_curve, final_capital = run_backtest(df, config)
    
    # 分析结果
    results = analyze_results(trades, equity_curve, final_capital, config['initial_capital'], config)
    
    return df, trades, equity_curve, results

if __name__ == '__main__':
    df, trades, equity_curve, results = main()

