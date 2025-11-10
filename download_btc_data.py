#!/usr/bin/env python3
"""
BTC历史数据下载和处理脚本
从Binance Vision下载BTCUSDT 1分钟数据并转换为指定格式
支持自定义时间范围配置
"""

import os
import sys
import requests
import zipfile
import pandas as pd# pyright: ignore
from datetime import datetime, date, timedelta
import calendar
from typing import List, Tuple, Optional, Dict, Any
import time
import argparse

# ==================== 配置区域 ====================
# 您可以在这里修改下载的时间范围
CONFIG = {
    # 数据下载时间范围配置
    'START_DATE': '2020-01-01',  # 开始日期，格式：YYYY-MM-DD（近5年）
    'END_DATE': '2025-11-10',    # 结束日期，格式：YYYY-MM-DD（到最新）
    
    # 输出文件配置
    'OUTPUT_FILENAME': 'btc_15m.csv',  # 输出文件名（直接在当前目录）
    
    # 下载配置
    'SYMBOL': 'BTCUSDT',              # 交易对
    'INTERVAL': '15m',                # 时间间隔（15分钟）
    'DELAY_BETWEEN_DOWNLOADS': 0.5,   # 下载间隔（秒）
}
# ================================================

class BTCDataDownloader:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化下载器
        
        Args:
            config: 配置字典，如果为None则使用全局CONFIG
        """
        if config is None:
            config = CONFIG
            
        self.base_url = "https://data.binance.vision/data/spot"
        self.symbol = config['SYMBOL']
        self.interval = config['INTERVAL']
        self.output_filename = config['OUTPUT_FILENAME']
        self.delay = config['DELAY_BETWEEN_DOWNLOADS']
        self.temp_dir = "temp"
        
        # 创建临时目录
        os.makedirs(self.temp_dir, exist_ok=True)
        
        print(f"交易对: {self.symbol}")
        print(f"时间间隔: {self.interval}")
        print(f"输出文件: {self.output_filename}")
        print(f"临时目录: {self.temp_dir}")
    
    def get_date_ranges(self, start_date: date, end_date: date) -> List[Tuple[str, str]]:
        """
        获取需要下载的日期范围
        优先下载月度数据，不足一个月的下载日度数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            List of (period_type, date_string) tuples
        """
        ranges = []
        current_date = start_date
        
        while current_date <= end_date:
            # 检查是否可以下载整个月的数据
            last_day_of_month = calendar.monthrange(current_date.year, current_date.month)[1]
            month_end = date(current_date.year, current_date.month, last_day_of_month)
            
            # 如果当前月的第一天开始，且整个月都在范围内
            if (current_date.day == 1 and month_end <= end_date):
                # 下载月度数据
                month_str = current_date.strftime("%Y-%m")
                ranges.append(("monthly", month_str))
                # 跳到下个月
                if current_date.month == 12:
                    current_date = date(current_date.year + 1, 1, 1)
                else:
                    current_date = date(current_date.year, current_date.month + 1, 1)
            else:
                # 下载日度数据
                day_str = current_date.strftime("%Y-%m-%d")
                ranges.append(("daily", day_str))
                current_date += timedelta(days=1)
        
        return ranges
    
    def download_file(self, period_type: str, date_str: str) -> Optional[str]:
        """
        下载单个文件
        
        Args:
            period_type: "daily" 或 "monthly"
            date_str: 日期字符串 (YYYY-MM-DD 或 YYYY-MM)
            
        Returns:
            下载的文件路径，如果失败返回None
        """
        if period_type == "daily":
            url = f"{self.base_url}/daily/klines/{self.symbol}/{self.interval}/{self.symbol}-{self.interval}-{date_str}.zip"
            filename = f"{self.symbol}-{self.interval}-{date_str}.zip"
        else:  # monthly
            url = f"{self.base_url}/monthly/klines/{self.symbol}/{self.interval}/{self.symbol}-{self.interval}-{date_str}.zip"
            filename = f"{self.symbol}-{self.interval}-{date_str}.zip"
        
        filepath = os.path.join(self.temp_dir, filename)
        
        print(f"正在下载: {filename}")
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"下载完成: {filename}")
            return filepath
            
        except requests.exceptions.RequestException as e:
            print(f"下载失败: {filename} - {str(e)}")
            return None
    
    def extract_and_process_file(self, zip_filepath: str) -> pd.DataFrame:
        """
        解压并处理CSV文件
        
        Args:
            zip_filepath: ZIP文件路径
            
        Returns:
            处理后的DataFrame
        """
        try:
            with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                # 获取ZIP文件中的CSV文件名
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                if not csv_files:
                    print(f"警告: {zip_filepath} 中没有找到CSV文件")
                    return pd.DataFrame()
                
                csv_filename = csv_files[0]
                
                # 解压CSV文件到临时目录
                zip_ref.extract(csv_filename, self.temp_dir)
                csv_filepath = os.path.join(self.temp_dir, csv_filename)
                
                # 读取CSV文件
                # Binance数据格式: Open time,Open,High,Low,Close,Volume,Close time,Quote asset volume,Number of trades,Taker buy base asset volume,Taker buy quote asset volume,Ignore
                df = pd.read_csv(csv_filepath, header=None)
                df.columns = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 
                             'Close_time', 'Quote_asset_volume', 'Number_of_trades', 
                             'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore']
                
                # 删除临时CSV文件
                os.remove(csv_filepath)
                
                return df
                
        except Exception as e:
            print(f"处理文件失败: {zip_filepath} - {str(e)}")
            return pd.DataFrame()
    
    def convert_to_target_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        转换为目标格式
        
        Args:
            df: 原始DataFrame
            
        Returns:
            转换后的DataFrame
        """
        if df.empty:
            return df
        
        # 转换时间戳为datetime格式
        # 注意：从2025年1月1日开始，Binance使用微秒时间戳
        try:
            # 先尝试毫秒格式
            df['DateTime'] = pd.to_datetime(df['Open_time'], unit='ms')
        except:
            try:
                # 如果失败，尝试微秒格式
                df['DateTime'] = pd.to_datetime(df['Open_time'], unit='us')
            except:
                # 如果还是失败，尝试纳秒格式或直接转换
                df['DateTime'] = pd.to_datetime(df['Open_time'], unit='ns')
        
        # 选择需要的列并重命名
        result_df = df[['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # 格式化DateTime为字符串 (YYYY-MM-DD HH:MM:SS)
        result_df['DateTime'] = result_df['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return result_df
    
    def download_and_merge_data(self, start_date: date, end_date: Optional[date] = None) -> Optional[str]:
        """
        下载并合并数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期，默认为今天
            
        Returns:
            合并后的文件路径
        """
        if end_date is None:
            # 使用2024年12月31日作为默认结束日期，因为这是已知可用的最新完整数据
            end_date = date(2024, 12, 31)
        
        print(f"开始下载数据: {start_date} 到 {end_date}")
        
        # 获取日期范围
        date_ranges = self.get_date_ranges(start_date, end_date)
        print(f"需要下载 {len(date_ranges)} 个文件")
        
        all_dataframes = []
        successful_downloads = 0
        
        for period_type, date_str in date_ranges:
            # 下载文件
            zip_filepath = self.download_file(period_type, date_str)
            
            if zip_filepath:
                # 处理文件
                df = self.extract_and_process_file(zip_filepath)
                
                if not df.empty:
                    # 转换格式
                    formatted_df = self.convert_to_target_format(df)
                    all_dataframes.append(formatted_df)
                    successful_downloads += 1
                
                # 删除ZIP文件
                os.remove(zip_filepath)
            
            # 添加延迟避免请求过于频繁
            time.sleep(self.delay)
        
        print(f"成功下载并处理了 {successful_downloads} 个文件")
        
        if not all_dataframes:
            print("没有成功下载任何数据")
            return None
        
        # 合并所有数据
        print("正在合并数据...")
        merged_df = pd.concat(all_dataframes, ignore_index=True)
        
        # 按时间排序
        merged_df['DateTime'] = pd.to_datetime(merged_df['DateTime'])
        merged_df = merged_df.sort_values('DateTime')
        
        # 去重
        merged_df = merged_df.drop_duplicates(subset=['DateTime'])
        
        # 保存合并后的文件到当前目录
        output_filepath = self.output_filename
        
        merged_df.to_csv(output_filepath, index=False)
        
        print("数据合并完成!")
        print(f"总共 {len(merged_df)} 条记录")
        print(f"时间范围: {merged_df['DateTime'].min()} 到 {merged_df['DateTime'].max()}")
        print(f"输出文件: {output_filepath}")
        
        return output_filepath
    
    def cleanup_temp_files(self):
        """清理临时文件"""
        try:
            if os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
                print("临时文件清理完成")
        except Exception as e:
            print(f"清理临时文件时出错: {str(e)}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='下载BTC历史数据')
    parser.add_argument('--start-date', type=str, default=None, 
                       help=f'开始日期 (YYYY-MM-DD), 默认: {CONFIG["START_DATE"]}')
    parser.add_argument('--end-date', type=str, default=None,
                       help=f'结束日期 (YYYY-MM-DD), 默认: {CONFIG["END_DATE"]}')
    parser.add_argument('--output-filename', type=str, default=None,
                       help=f'输出文件名, 默认: {CONFIG["OUTPUT_FILENAME"]}')
    
    args = parser.parse_args()
    
    try:
        # 创建配置副本，允许命令行参数覆盖
        config = CONFIG.copy()
        
        # 使用命令行参数或配置文件中的默认值
        start_date_str = args.start_date or config['START_DATE']
        end_date_str = args.end_date or config['END_DATE']
        
        if args.output_filename:
            config['OUTPUT_FILENAME'] = args.output_filename
        
        # 解析日期
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        
        if end_date_str:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        else:
            end_date = None
        
        print("=" * 60)
        print("BTC历史数据下载器")
        print("=" * 60)
        print(f"开始日期: {start_date_str}")
        print(f"结束日期: {end_date_str or '自动确定'}")
        print(f"输出文件: {config['OUTPUT_FILENAME']}")
        print("=" * 60)
        
        # 创建下载器
        downloader = BTCDataDownloader(config)
        
        # 下载并合并数据
        output_file = downloader.download_and_merge_data(start_date, end_date)
        
        if output_file:
            print(f"\n✅ 数据下载完成！")
            print(f"输出文件: {output_file}")
        else:
            print(f"\n❌ 数据下载失败！")
            sys.exit(1)
            
    except ValueError as e:
        print(f"日期格式错误: {str(e)}")
        print("请使用 YYYY-MM-DD 格式")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n用户取消下载")
        sys.exit(1)
    except Exception as e:
        print(f"发生错误: {str(e)}")
        sys.exit(1)
    finally:
        # 清理临时文件
        if 'downloader' in locals():
            downloader.cleanup_temp_files()

if __name__ == "__main__":
    main() 