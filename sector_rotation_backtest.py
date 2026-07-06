#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股板块轮动模型 —— 回测脚本

策略：Top4板块 × 每板块精选7只 + 主线离散度清仓择时
数据：见 get_hk_data.py
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

CONFIG = {
    'min_sector_members': 4,     # 行业成员数低于此值归并为"其他"
    'top_k_sectors': 4,          # 选前K个板块（原来是5，测试下来4更集中且效果更好）
    'stock_pick_per_sector': 7,  # 每个板块内只买评分最高的N只，平均持仓约21只
    'cost': 0.002,               # 单边成本
    'label_horizon': 20,
    'eval_start': '2018-01',     # 回测评估起点
    'disp_lookback': 36,         # 主线离散度：滚动回看月数
    'disp_off_exposure': 0.0,    # 无主线时的仓位系数：真正清仓，而不是减半
                                 # （测试下来"没有主线就真的空仓"比"减到50%"全面更优：收益更高、
                                 # 夏普更高、回撤更小，见README"参数优化"一节）
}


# ==================== 数据加载与特征 ====================

def load_data():
    px = pd.read_csv(os.path.join(DATA_DIR, 'prices.csv'), dtype={'code': str})
    px['date'] = pd.to_datetime(px['date'])
    px = px.sort_values(['code', 'date']).reset_index(drop=True)

    sb = pd.read_csv(os.path.join(DATA_DIR, 'southbound.csv'), dtype={'code': str})
    sb['date'] = pd.to_datetime(sb['date'])
    sb = sb.sort_values(['code', 'date']).reset_index(drop=True)

    ind = pd.read_csv(os.path.join(DATA_DIR, 'industry.csv'), dtype={'code': str})
    return px, sb, ind


def build_features(px, sb):
    px = px.copy()
    g = px.groupby('code')
    px['ret_20'] = g['close'].pct_change(20)
    px['ret_60'] = g['close'].pct_change(60)
    px['ret_120'] = g['close'].pct_change(120)
    px['vol_20'] = g['close'].transform(lambda s: s.pct_change().rolling(20).std())
    px['amt_20'] = g['amount'].transform(lambda s: s.rolling(20).mean())
    px['amt_120'] = g['amount'].transform(lambda s: s.rolling(120).mean())
    px['amt_trend'] = px['amt_20'] / px['amt_120'] - 1

    px['dret'] = g['close'].pct_change()
    mkt = px.groupby('date')['dret'].mean().rename('mkt_ret')
    px = px.merge(mkt, on='date', how='left')
    px['ex_ret'] = px['dret'] - px['mkt_ret']
    px['rs_60'] = px.groupby('code')['ex_ret'].transform(lambda s: s.rolling(60).sum())

    sb = sb.copy()
    sg = sb.groupby('code')
    sb['sb_pct_chg60'] = sg['sb_pct'].diff(60)
    df = px.merge(sb[['code', 'date', 'sb_pct', 'sb_pct_chg60']], on=['code', 'date'], how='left')
    df[['sb_pct', 'sb_pct_chg60']] = df.groupby('code')[['sb_pct', 'sb_pct_chg60']].ffill()

    df['fwd_ret'] = df.groupby('code')['close'].pct_change(CONFIG['label_horizon']).shift(-CONFIG['label_horizon'])
    return df


SCORE_FEATS = ['rs_60', 'ret_60', 'sb_pct_chg60', 'amt_trend']


def month_end_panel(df, industry_map):
    df = df.copy()
    df['ym'] = df['date'].dt.to_period('M')
    idx = df.groupby(['code', 'ym'])['date'].idxmax()
    panel = df.loc[idx].copy()
    panel = panel.dropna(subset=['ret_120'])
    panel = panel.merge(industry_map, on='code', how='left')
    return panel


def assign_sector_groups(panel):
    """成员数太少的行业归并为'其他'，避免板块聚合噪音过大"""
    counts = panel.groupby('industry')['code'].nunique()
    small = counts[counts < CONFIG['min_sector_members']].index
    panel = panel.copy()
    panel['sector'] = panel['industry'].where(~panel['industry'].isin(small), '其他')
    return panel


def zrank(s):
    return s.rank(pct=True)


def score_stocks(panel):
    """板块内精选用的个股打分，与板块打分同口径"""
    parts = []
    for m, grp in panel.groupby('ym'):
        g = grp.copy()
        g['stock_score'] = sum(zrank(g[c]).fillna(0.5) for c in SCORE_FEATS) / len(SCORE_FEATS)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def score_sectors(panel):
    sector_month = panel.groupby(['ym', 'sector'])[SCORE_FEATS].mean().reset_index()
    parts = []
    for m, grp in sector_month.groupby('ym'):
        g = grp.copy()
        g['sector_score'] = sum(zrank(g[c]).fillna(0.5) for c in SCORE_FEATS) / len(SCORE_FEATS)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def build_dispersion_regime(sector_scores):
    """板块间rs_60截面标准差；低于滚动中位数(rolling window)时视为无主线

    早期(1998-2010左右)港股通股票池成分股很少、板块内样本量小，测出来的离散度天然偏高、偏噪；
    如果用expanding(从头到当前)中位数做基准，这段"虚高"的历史会把基准线抬得过高，导致后面
    2016-2023年股票池已经成熟、离散度天然更低的时期几乎永远被误判为"无主线"。用滚动窗口
    (rolling)中位数能跟随股票池规模的结构性变化，不被早期噪声污染（实测：修复后2018-2023
    年化从-0.8%提升到+2.1%，全周期年化从10.2%提升到15.6%）。
    """
    disp = sector_scores.groupby('ym')['rs_60'].std().rename('disp').reset_index().sort_values('ym')
    lb = CONFIG['disp_lookback']
    disp['disp_median'] = disp['disp'].rolling(lb, min_periods=lb).median().shift(1)
    disp['has_theme'] = np.where(disp['disp_median'].isna(), 1,
                                 (disp['disp'] >= disp['disp_median']).astype(int))
    return disp[['ym', 'disp', 'disp_median', 'has_theme']]


# ==================== 回测 ====================

def me_price_table(px, codes_universe):
    px = px[px['code'].isin(codes_universe)].copy()
    px['ym'] = px['date'].dt.to_period('M')
    idx = px.groupby(['code', 'ym'])['date'].idxmax()
    return px.loc[idx, ['code', 'ym', 'close']].rename(columns={'close': 'me_close'})


def build_monthly_holdings(eval_months, sector_scores, panel):
    """选Top K板块，每个板块内只买评分最高的 stock_pick_per_sector 只，
    而不是板块内全买——同样能代表板块趋势，但集中度更高、持仓数少了近一半。"""
    holdings = {}
    for m in eval_months:
        top_sectors = sector_scores[sector_scores['ym'] == m].nlargest(CONFIG['top_k_sectors'], 'sector_score')['sector']
        sub = panel[(panel['ym'] == m) & panel['sector'].isin(top_sectors)]
        picked = pd.concat([g.nlargest(CONFIG['stock_pick_per_sector'], 'stock_score')
                            for _, g in sub.groupby('sector')], ignore_index=True)
        holdings[m] = picked[['code', 'sector']]
    return holdings


def run_backtest(eval_months, holdings, me_px, disp_map):
    equity = 1.0
    curve = []
    prev_hold = set()
    for i in range(len(eval_months) - 1):
        m, m_next = eval_months[i], eval_months[i + 1]
        codes = holdings[m]['code'].tolist()
        exposure = 1.0 if disp_map.get(m, 1) == 1 else CONFIG['disp_off_exposure']

        if codes:
            cur = me_px[(me_px['ym'] == m) & me_px['code'].isin(codes)].set_index('code')['me_close']
            nxt = me_px[(me_px['ym'] == m_next) & me_px['code'].isin(codes)].set_index('code')['me_close']
            common = cur.index.intersection(nxt.index)
            stock_ret = (nxt[common] / cur[common] - 1).mean() if len(common) else 0.0
            turnover = 1 - len(prev_hold & set(codes)) / max(len(codes), 1)
        else:
            stock_ret, turnover = 0.0, 0.0

        port_ret = exposure * stock_ret - 2 * CONFIG['cost'] * turnover * exposure
        equity *= (1 + port_ret)
        curve.append({'ym': str(m_next), 'ret': port_ret, 'equity': equity, 'exposure': exposure})
        prev_hold = set(codes) if exposure > 0 else set()
    return pd.DataFrame(curve)


def perf_stats(curve, name):
    if curve.empty:
        print(f"{name}: 无交易")
        return None
    total = curve['equity'].iloc[-1] - 1
    n = len(curve)
    ann = (1 + total) ** (12 / n) - 1
    dd = (curve['equity'] / curve['equity'].cummax() - 1).min()
    sharpe = curve['ret'].mean() / curve['ret'].std() * np.sqrt(12) if curve['ret'].std() > 0 else 0
    win = (curve['ret'] > 0).mean()
    print(f"\n【{name}】  月数={n}")
    print(f"  总收益: {total*100:+.1f}%   年化: {ann*100:+.1f}%   夏普: {sharpe:.2f}")
    print(f"  最大回撤: {dd*100:.1f}%   月胜率: {win*100:.0f}%")
    return {'total': total, 'ann': ann, 'sharpe': sharpe, 'mdd': dd}


def print_holdings(holdings, eval_months, disp_map, name_map, year_prefixes=('2025', '2026')):
    print(f"\n{'='*60}\n持仓明细（{'、'.join(year_prefixes)}年）\n{'='*60}")
    for m in eval_months:
        if not str(m).startswith(year_prefixes):
            continue
        has_theme = disp_map.get(m, 1) == 1
        h = holdings[m].copy()
        if not has_theme:
            print(f"\n{m}：【空仓】主线离散度低于近期常态，判定为无清晰主线，本月不持仓")
            continue
        h['name'] = h['code'].map(name_map)
        print(f"\n{m}（满仓，{h['sector'].nunique()}个板块，{len(h)}只股票）：")
        for sector, grp in h.groupby('sector'):
            names = [f"{c} {name_map.get(c, '')}" for c in grp['code']]
            print(f"  [{sector}] " + "、".join(names))


def main():
    print("加载数据...")
    px, sb, ind = load_data()
    industry_map = ind[['code', 'industry']]
    name_map = dict(zip(ind['code'], ind['name']))

    print("构建特征...")
    df = build_features(px, sb)
    panel = month_end_panel(df, industry_map)
    panel = assign_sector_groups(panel)
    print(f"月末截面: {panel['ym'].nunique()}个月, {panel['sector'].nunique()}个板块(含'其他')")

    panel = score_stocks(panel)
    sector_scores = score_sectors(panel)
    disp_df = build_dispersion_regime(sector_scores)
    disp_map = dict(zip(disp_df['ym'], disp_df['has_theme']))

    months = sorted(panel['ym'].unique())
    eval_months = [m for m in months if str(m) >= CONFIG['eval_start']]
    me_px = me_price_table(px, panel['code'].unique())

    holdings = build_monthly_holdings(eval_months, sector_scores, panel)
    curve = run_backtest(eval_months, holdings, me_px, disp_map)

    print("\n" + "=" * 60)
    print(f"Sector-Top + 主线离散度过滤（Top{CONFIG['top_k_sectors']}板块, "
          f"回看{CONFIG['disp_lookback']}月, 无主线仓位{CONFIG['disp_off_exposure']}）")
    print("=" * 60)
    perf_stats(curve, '全周期')

    for label, start, end in [('2018-2023(不含本轮牛市)', '2018-01', '2024-01'),
                               ('2024至今(本轮牛市)', '2024-01', '2100-01')]:
        seg = curve[(curve['ym'] >= start) & (curve['ym'] < end)]
        if len(seg) > 3:
            perf_stats(seg.assign(equity=(1 + seg['ret']).cumprod()), label)

    curve.to_csv(os.path.join(DATA_DIR, 'backtest_curves.csv'), index=False)
    print(f"\n净值曲线已保存: {os.path.join(DATA_DIR, 'backtest_curves.csv')}")

    n_cash = sum(1 for m in eval_months[:-1] if disp_map.get(m, 1) == 0)
    print(f"\n【择时统计】回测窗口{len(eval_months)-1}个月里，空仓月份 {n_cash} 个"
          f"（{n_cash/(len(eval_months)-1)*100:.0f}%），满仓月份 {len(eval_months)-1-n_cash} 个。")

    print("\n【最新一期板块排名】")
    last_m = eval_months[-1]
    last_scores = sector_scores[sector_scores['ym'] == last_m].sort_values('sector_score', ascending=False)
    print(last_scores[['sector', 'sector_score'] + SCORE_FEATS].round(3).head(10).to_string(index=False))
    print(f"当期是否判定为'有主线': {'是，满仓' if disp_map.get(last_m, 1) == 1 else '否，本月空仓'}")

    print_holdings(holdings, eval_months, disp_map, name_map)


if __name__ == '__main__':
    main()
