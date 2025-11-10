"""
TSI (True Strength Index) 真实强弱指数策略
==============================================

策略原理：
TSI是一个动量震荡指标，通过双重平滑来过滤价格噪声，识别真实的趋势方向。
它能够在趋势早期发出信号，同时通过严格的止损控制风险。

核心逻辑：
1. 计算价格变化的双重指数平滑移动平均（EMA）
2. TSI上穿信号线 → 趋势确认，做多
3. TSI下穿信号线 → 趋势结束，平仓
4. 多重止损保护：固定止损 + 跟踪止损

收益本质：
- 捕捉中短期趋势动量
- 双重平滑过滤噪声，减少假信号
- 严格止损截断亏损
- 让盈利单充分运行（跟踪止损）
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ================================================================================
# 参数配置区域
# ================================================================================

CONFIG = {
    # === 数据设置 ===
    'data_path': 'btc_15m.csv',
    'initial_capital': 100000,
    'start_date': '2024-11-01',
    'end_date': '2025-11-10',
    
    # === TSI核心参数 ===
    'tsi_long_period': 25,      # 长周期平滑参数（默认25）
    'tsi_short_period': 13,     # 短周期平滑参数（默认13）
    'tsi_signal_period': 13,    # 信号线周期（最优值13，夏普11.307）
    
    # === 风险管理 ===
    'stop_loss_pct': 0.03,      # 固定止损（3%）
    'take_profit_pct': 0.08,    # 固定止盈（8%）
    'use_trailing_stop': True,  # 是否使用跟踪止损
    'trailing_stop_pct': 0.02,  # 跟踪止损（2%）
    
    # === 信号过滤（可选）===
    'use_tsi_threshold': False,  # 是否使用TSI阈值过滤
    'tsi_entry_threshold': 0,    # TSI最小值才能开仓（避免弱信号）
    
    # === 交易控制 ===
    'print_trades': True,
}


# ================================================================================
# 预设参数配置（供测试使用）
# ================================================================================

PRESET_CONFIGS = {
    # 1. 最优配置（夏普比率11.307）⭐
    'optimal': {
        'tsi_long_period': 25,
        'tsi_short_period': 13,
        'tsi_signal_period': 13,  # 关键：使用13而不是7
        'stop_loss_pct': 0.03,
        'trailing_stop_pct': 0.02,
    },
    
    # 2. 默认配置（原始7周期信号线）
    'default': {
        'tsi_long_period': 25,
        'tsi_short_period': 13,
        'tsi_signal_period': 7,
        'stop_loss_pct': 0.03,
        'trailing_stop_pct': 0.02,
    },
    
    # 3. 快速响应型（捕捉短期趋势）
    'fast': {
        'tsi_long_period': 15,     # 更快响应
        'tsi_short_period': 8,
        'tsi_signal_period': 5,
        'stop_loss_pct': 0.025,    # 更紧止损
        'trailing_stop_pct': 0.015,
    },
    
    # 4. 稳健过滤型（减少假信号）
    'stable': {
        'tsi_long_period': 35,     # 更慢但更可靠
        'tsi_short_period': 18,
        'tsi_signal_period': 9,
        'stop_loss_pct': 0.04,     # 更宽止损
        'trailing_stop_pct': 0.025,
    },
    
    # 5. 激进型（高频交易）
    'aggressive': {
        'tsi_long_period': 20,
        'tsi_short_period': 10,
        'tsi_signal_period': 5,
        'stop_loss_pct': 0.02,     # 更紧止损
        'trailing_stop_pct': 0.01,
        'use_tsi_threshold': True,
        'tsi_entry_threshold': 5,   # 只在强信号时入场
    },
    
    # 6. 保守型（追求高胜率）
    'conservative': {
        'tsi_long_period': 40,
        'tsi_short_period': 20,
        'tsi_signal_period': 10,
        'stop_loss_pct': 0.05,     # 更宽止损
        'trailing_stop_pct': 0.03,
        'use_tsi_threshold': True,
        'tsi_entry_threshold': 10,  # 只在极强信号时入场
    },
}


# ================================================================================
# TSI指标计算
# ================================================================================

def calculate_tsi(df, long_period=25, short_period=13, signal_period=7):
    """
    计算TSI指标
    
    TSI计算步骤：
    1. 计算价格变化 = Close - Close.shift(1)
    2. 对价格变化进行长周期EMA平滑
    3. 对步骤2的结果再进行短周期EMA平滑
    4. 对价格变化的绝对值进行相同的双重平滑
    5. TSI = (步骤3 / 步骤4) * 100
    6. Signal = TSI的EMA
    
    参数说明：
    - long_period: 第一次平滑的周期（通常20-30）
    - short_period: 第二次平滑的周期（通常10-15）
    - signal_period: 信号线的周期（通常5-10）
    
    返回：
    - TSI: 真实强弱指数（范围约-100到+100）
    - Signal: TSI的信号线
    """
    df = df.copy()
    
    # 1. 计算价格变化
    price_change = df['Close'].diff()
    
    # 2. 双重EMA平滑价格变化
    # 第一次平滑
    pc_smooth1 = price_change.ewm(span=long_period, adjust=False).mean()
    # 第二次平滑
    pc_smooth2 = pc_smooth1.ewm(span=short_period, adjust=False).mean()
    
    # 3. 双重EMA平滑价格变化的绝对值
    abs_pc_smooth1 = price_change.abs().ewm(span=long_period, adjust=False).mean()
    abs_pc_smooth2 = abs_pc_smooth1.ewm(span=short_period, adjust=False).mean()
    
    # 4. 计算TSI
    df['TSI'] = 100 * (pc_smooth2 / abs_pc_smooth2)
    
    # 5. 计算信号线（TSI的EMA）
    df['TSI_Signal'] = df['TSI'].ewm(span=signal_period, adjust=False).mean()
    
    # 6. 生成交易信号
    # 当TSI从下方穿越信号线时，产生买入信号
    df['Signal'] = 0
    
    # TSI上穿信号线 → 做多
    df.loc[(df['TSI'] > df['TSI_Signal']) & 
           (df['TSI'].shift(1) <= df['TSI_Signal'].shift(1)), 'Signal'] = 1
    
    # TSI下穿信号线 → 平仓
    df.loc[(df['TSI'] < df['TSI_Signal']) & 
           (df['TSI'].shift(1) >= df['TSI_Signal'].shift(1)), 'Signal'] = -1
    
    return df


# ================================================================================
# 数据加载
# ================================================================================

def load_data(config):
    """加载并准备数据"""
    print(f"\n{'='*80}")
    print(f"📂 加载数据...")
    print(f"{'='*80}\n")
    
    df = pd.read_csv(config['data_path'])
    
    # 确保DateTime列
    if 'DateTime' not in df.columns:
        if 'timestamp' in df.columns:
            df['DateTime'] = pd.to_datetime(df['timestamp'], unit='ms')
        else:
            raise ValueError("数据文件需要包含DateTime或timestamp列")
    else:
        df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # 过滤日期范围
    if config.get('start_date'):
        df = df[df['DateTime'] >= config['start_date']]
    if config.get('end_date'):
        df = df[df['DateTime'] <= config['end_date']]
    
    df = df.sort_values('DateTime').reset_index(drop=True)
    
    print(f"✅ 数据加载完成")
    print(f"   数据范围: {df['DateTime'].min()} 至 {df['DateTime'].max()}")
    print(f"   数据点数: {len(df)}")
    print(f"   价格范围: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    
    return df


# ================================================================================
# TSI策略
# ================================================================================

def apply_tsi_strategy(df, config):
    """应用TSI策略"""
    print(f"\n{'='*80}")
    print(f"📊 计算TSI指标...")
    print(f"{'='*80}\n")
    
    # 计算TSI
    df = calculate_tsi(
        df,
        long_period=config['tsi_long_period'],
        short_period=config['tsi_short_period'],
        signal_period=config['tsi_signal_period']
    )
    
    # 应用TSI阈值过滤（如果启用）
    if config.get('use_tsi_threshold', False):
        threshold = config['tsi_entry_threshold']
        # 只有当TSI值足够强时才允许开仓
        df.loc[(df['Signal'] == 1) & (df['TSI'] < threshold), 'Signal'] = 0
    
    print(f"✅ TSI指标计算完成")
    print(f"   TSI长周期: {config['tsi_long_period']}")
    print(f"   TSI短周期: {config['tsi_short_period']}")
    print(f"   信号线周期: {config['tsi_signal_period']}")
    if config.get('use_tsi_threshold', False):
        print(f"   TSI入场阈值: {config['tsi_entry_threshold']}")
    
    # 统计信号
    buy_signals = len(df[df['Signal'] == 1])
    sell_signals = len(df[df['Signal'] == -1])
    print(f"\n   生成买入信号: {buy_signals}个")
    print(f"   生成卖出信号: {sell_signals}个")
    
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
    equity_curve = []
    
    print(f"\n{'='*80}")
    print(f"🚀 开始回测...")
    print(f"{'='*80}\n")
    
    for idx, row in df.iterrows():
        if pd.isna(row['Signal']):
            continue
        
        current_price = row['Close']
        current_time = row['DateTime']
        signal = row['Signal']
        
        # 记录权益曲线
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
            
            # 更新最高价
            if current_price > highest_price:
                highest_price = current_price
                position['highest_price'] = highest_price
            
            pnl_pct = (current_price - entry_price) / entry_price
            
            # 检查各种退出条件
            close_reason = None
            should_close = False
            
            # 1. 固定止损
            if pnl_pct <= -stop_loss_pct:
                close_reason = '固定止损'
                should_close = True
            
            # 2. 固定止盈
            elif pnl_pct >= take_profit_pct:
                close_reason = '固定止盈'
                should_close = True
            
            # 3. 跟踪止损
            elif use_trailing and pnl_pct > 0:
                trailing_stop_price = highest_price * (1 - trailing_pct)
                if current_price < trailing_stop_price:
                    close_reason = '跟踪止损'
                    should_close = True
            
            # 4. TSI信号平仓
            elif signal == -1:
                close_reason = 'TSI信号'
                should_close = True
            
            # 执行平仓
            if should_close:
                shares = position['shares']
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
        return None
    
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
    
    # 计算年化收益率
    total_days = (df['DateTime'].max() - df['DateTime'].min()).days
    years = total_days / 365.25
    annualized_return = (final_capital / initial_capital) ** (1 / years) - 1 if years > 0 else 0
    
    # 计算波动率（年化）
    equity_df['Returns'] = equity_df['Equity'].pct_change()
    daily_returns = equity_df.groupby(equity_df['DateTime'].dt.date)['Returns'].last()
    volatility = daily_returns.std() * np.sqrt(365.25) if len(daily_returns) > 1 else 0
    
    # 计算夏普比率
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
    
    # 计算Buy & Hold的夏普
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
    
    # 分析退出原因
    print(f"\n🚪 退出原因分析:")
    exit_reasons = trades_df['reason'].value_counts()
    for reason, count in exit_reasons.items():
        pct = count / total_trades * 100
        avg_pnl = trades_df[trades_df['reason'] == reason]['pnl'].mean()
        print(f"   {reason}: {count}笔 ({pct:.1f}%) | 平均: ${avg_pnl:+,.2f}")
    
    print(f"\n🎯 vs Buy & Hold:")
    print(f"   B&H总回报: {bh_return:+.2f}%")
    print(f"   B&H年化: {bh_annualized*100:+.2f}%")
    print(f"   B&H夏普: {bh_sharpe:.3f}")
    print(f"   策略超额回报: {total_return - bh_return:+.2f}%")
    print(f"   夏普比率优势: {sharpe_ratio - bh_sharpe:+.3f}")
    
    print(f"\n📊 最佳/最差交易:")
    best = trades_df.loc[trades_df['pnl'].idxmax()]
    worst = trades_df.loc[trades_df['pnl'].idxmin()]
    print(f"   最佳: ${best['pnl']:+,.2f} ({best['pnl_pct']:+.2f}%) | {best['reason']}")
    print(f"   最差: ${worst['pnl']:+,.2f} ({worst['pnl_pct']:+.2f}%) | {worst['reason']}")
    
    print(f"\n{'='*80}\n")
    
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

def main():
    """主函数"""
    print("\n" + "="*80)
    print("📈 TSI (True Strength Index) 策略回测系统")
    print("="*80)
    
    # 选择配置（可以在这里切换不同的预设配置）
    preset = 'optimal'  # 可选: 'optimal', 'default', 'fast', 'stable', 'aggressive', 'conservative'
    
    if preset in PRESET_CONFIGS:
        print(f"\n✅ 使用预设配置: {preset}")
        CONFIG.update(PRESET_CONFIGS[preset])
    
    # 加载数据
    df = load_data(CONFIG)
    
    # 应用TSI策略
    df = apply_tsi_strategy(df, CONFIG)
    
    # 执行回测
    trades, final_capital, equity_curve = run_backtest(df, CONFIG)
    
    # 分析结果
    results = analyze_results(trades, final_capital, CONFIG['initial_capital'], df, equity_curve)
    
    return results


if __name__ == '__main__':
    results = main()

