#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股板块轮动模型 - 数据下载

数据来源：akshare（新浪 + 东方财富接口）
输出目录：data/
"""

import os
import time
import socket
import pandas as pd
import akshare as ak

socket.setdefaulttimeout(20)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')

CONFIG = {
    'universe_size': 200,
    'price_end_str': '2026-07-06',
    'sleep': 0.4,
    'max_retry': 4,
}


def retry_call(fn, *args, **kwargs):
    last_err = None
    for i in range(CONFIG['max_retry']):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            wait = 2 * (i + 1)
            print(f"  重试 {i+1}/{CONFIG['max_retry']} ({e})，等待{wait}s")
            time.sleep(wait)
    raise last_err


def get_universe():
    path = os.path.join(DATA_DIR, 'universe.csv')
    if os.path.exists(path):
        df = pd.read_csv(path, dtype={'code': str})
        print(f"股票池已存在: {len(df)}只")
        return df
    comp = retry_call(ak.stock_hk_ggt_components_em)
    comp = comp.rename(columns={'代码': 'code', '名称': 'name', '成交额': 'amount'})
    comp = comp.sort_values('amount', ascending=False).head(CONFIG['universe_size'])
    df = comp[['code', 'name']].reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"股票池: {len(df)}只")
    return df


def download_prices(universe):
    """新浪全历史日线，覆盖到上市首日"""
    out_path = os.path.join(DATA_DIR, 'prices.csv')
    frames = []
    for i, row in universe.iterrows():
        code = row['code']
        cache = os.path.join(CACHE_DIR, f'px_{code}.csv')
        if os.path.exists(cache):
            df = pd.read_csv(cache)
        else:
            try:
                df = retry_call(ak.stock_hk_daily, symbol=code, adjust='qfq')
            except Exception as e:
                print(f"[{i+1}/{len(universe)}] {code} 行情失败: {e}")
                continue
            df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']].copy()
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df = df[df['date'] <= CONFIG['price_end_str']]
            df.to_csv(cache, index=False)
            time.sleep(CONFIG['sleep'])
        df['code'] = code
        frames.append(df)
        if (i + 1) % 20 == 0:
            print(f"行情进度 {i+1}/{len(universe)}")
    all_px = pd.concat(frames, ignore_index=True)
    all_px.to_csv(out_path, index=False)
    span = f"{all_px['date'].min()} ~ {all_px['date'].max()}"
    print(f"行情完成: {len(all_px)}行, {span} -> {out_path}")


def download_southbound(universe):
    out_path = os.path.join(DATA_DIR, 'southbound.csv')
    frames = []
    for i, row in universe.iterrows():
        code = row['code']
        cache = os.path.join(CACHE_DIR, f'sb_{code}.csv')
        if os.path.exists(cache):
            df = pd.read_csv(cache)
        else:
            try:
                df = retry_call(ak.stock_hsgt_individual_em, symbol=code)
            except Exception as e:
                print(f"[{i+1}/{len(universe)}] {code} 南向持仓失败: {e}")
                continue
            df = df.rename(columns={
                '持股日期': 'date', '持股数量': 'sb_shares', '持股市值': 'sb_value',
                '持股数量占A股百分比': 'sb_pct'})
            df = df[['date', 'sb_shares', 'sb_value', 'sb_pct']]
            df.to_csv(cache, index=False)
            time.sleep(CONFIG['sleep'])
        df['code'] = code
        frames.append(df)
        if (i + 1) % 20 == 0:
            print(f"南向进度 {i+1}/{len(universe)}")
    all_sb = pd.concat(frames, ignore_index=True)
    all_sb.to_csv(out_path, index=False)
    print(f"南向持仓完成: {len(all_sb)}行 -> {out_path}")


def download_industry(universe):
    out_path = os.path.join(DATA_DIR, 'industry.csv')
    rows = []
    for i, row in universe.iterrows():
        code = row['code']
        cache = os.path.join(CACHE_DIR, f'ind_{code}.csv')
        if os.path.exists(cache):
            ind = pd.read_csv(cache)['industry'].iloc[0]
        else:
            try:
                df = retry_call(ak.stock_hk_company_profile_em, symbol=code)
                ind = df['所属行业'].iloc[0] if len(df) else None
            except Exception as e:
                print(f"[{i+1}/{len(universe)}] {code} 行业获取失败: {e}")
                ind = None
            pd.DataFrame({'industry': [ind]}).to_csv(cache, index=False)
            time.sleep(CONFIG['sleep'])
        rows.append({'code': code, 'name': row['name'], 'industry': ind})
        if (i + 1) % 20 == 0:
            print(f"行业进度 {i+1}/{len(universe)}")
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"行业分类完成: {len(out)}只, {out['industry'].nunique()}个行业 -> {out_path}")


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    universe = get_universe()
    download_prices(universe)
    download_southbound(universe)
    download_industry(universe)


if __name__ == '__main__':
    main()
