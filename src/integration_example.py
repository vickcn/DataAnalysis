#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
離散數據聚合與相關性分析整合範例
展示如何將聚合功能整合到現有的數據分析流程中
"""

import asyncio
import os
import sys
import pandas as pd
import numpy as np

# 添加父目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.discrete_aggregator import aggregate_discrete_data
from package.data_analysis import mic_matrix, plotCorrelation
import DataAnalysis as DA

async def analyze_with_aggregation(file_path: str, 
                                 discrete_columns: list = None,
                                 max_categories: int = 10,
                                 use_fast_correlation: bool = False,
                                 n_jobs: int = None):
    """
    帶有離散數據聚合的完整分析流程
    
    Parameters:
    -----------
    file_path : str
        數據文件路徑
    discrete_columns : list, optional
        需要聚合的離散欄位列表
    max_categories : int
        聚合後的最大類別數
    use_fast_correlation : bool
        是否使用快速相關性計算
    n_jobs : int, optional
        並行計算的進程數
    """
    
    print("=" * 60)
    print("離散數據聚合 + 相關性分析整合流程")
    print("=" * 60)
    
    # 1. 載入數據
    print("1. 載入數據...")
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("不支援的文件格式")
        
        print(f"   數據形狀: {df.shape}")
        print(f"   欄位: {list(df.columns)}")
        
    except Exception as e:
        print(f"   載入數據失敗: {e}")
        return None
    
    # 2. 分析離散欄位
    print("\n2. 分析離散欄位...")
    discrete_info = {}
    
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].nunique() > max_categories:
            unique_count = df[col].nunique()
            discrete_info[col] = {
                'unique_count': unique_count,
                'needs_aggregation': unique_count > max_categories,
                'sample_values': df[col].dropna().head(5).tolist()
            }
            print(f"   {col}: {unique_count} 個唯一值 {'(需要聚合)' if unique_count > max_categories else ''}")
    
    # 3. 執行離散數據聚合
    if discrete_columns is None:
        discrete_columns = [col for col, info in discrete_info.items() if info['needs_aggregation']]
    
    if discrete_columns:
        print(f"\n3. 聚合離散數據 ({len(discrete_columns)} 個欄位)...")
        
        df_aggregated, aggregation_results = await aggregate_discrete_data(
            df,
            columns=discrete_columns,
            max_categories=max_categories
        )
        
        print("   聚合結果:")
        for result in aggregation_results:
            if result['aggregated']:
                print(f"     {result['column_name']}: {result['original_categories']} -> {result['final_categories']} 類別")
        
    else:
        print("\n3. 無需聚合離散數據")
        df_aggregated = df.copy()
        aggregation_results = []
    
    # 4. 計算相關性矩陣
    print(f"\n4. 計算相關性矩陣...")
    print(f"   使用並行計算: {n_jobs if n_jobs else '自動'}")
    print(f"   快速相關性: {use_fast_correlation}")
    
    try:
        # 只選擇數值型和聚合後的類別型欄位
        analysis_columns = []
        for col in df_aggregated.columns:
            if (df_aggregated[col].dtype in ['int64', 'float64'] or 
                df_aggregated[col].nunique() <= max_categories):
                analysis_columns.append(col)
        
        df_for_analysis = df_aggregated[analysis_columns]
        print(f"   分析欄位: {analysis_columns}")
        
        # 計算 MIC 矩陣
        corr_matrix = mic_matrix(
            df_for_analysis,
            n_jobs=n_jobs,
            use_fast_correlation=use_fast_correlation
        )
        
        print(f"   相關性矩陣形狀: {corr_matrix.shape}")
        
    except Exception as e:
        print(f"   相關性計算失敗: {e}")
        return None
    
    # 5. 生成相關性圖表
    print(f"\n5. 生成相關性圖表...")
    
    try:
        ret = {}
        success = DA.PlotCorrelation(
            df_for_analysis,
            method='mic',
            exp_fd='tmp/aggregated_analysis',
            ret=ret,
            n_jobs=n_jobs,
            use_fast_correlation=use_fast_correlation
        )
        
        if success:
            print("   相關性圖表生成成功")
            if 'pd_corr' in ret:
                print(f"   相關性矩陣已保存")
        else:
            print("   相關性圖表生成失敗")
            
    except Exception as e:
        print(f"   圖表生成失敗: {e}")
    
    # 6. 輸出結果摘要
    print(f"\n6. 結果摘要:")
    print(f"   原始數據: {df.shape[0]} 行 × {df.shape[1]} 欄")
    print(f"   分析數據: {df_for_analysis.shape[0]} 行 × {df_for_analysis.shape[1]} 欄")
    print(f"   聚合欄位: {len([r for r in aggregation_results if r['aggregated']])} 個")
    
    if aggregation_results:
        print(f"\n   聚合詳情:")
        for result in aggregation_results:
            if result['aggregated']:
                print(f"     - {result['column_name']}: {result['data_type']} 類型")
                print(f"       {result['original_categories']} -> {result['final_categories']} 類別")
    
    return {
        'original_data': df,
        'aggregated_data': df_aggregated,
        'analysis_data': df_for_analysis,
        'correlation_matrix': corr_matrix,
        'aggregation_results': aggregation_results
    }

async def create_sample_data():
    """創建示例數據用於測試"""
    print("創建示例數據...")
    
    np.random.seed(42)
    n_samples = 200
    
    # 生成複雜的離散數據
    countries = ['中國', '美國', '日本', '德國', '法國', '英國', '澳洲', '加拿大', 
                '韓國', '印度', '巴西', '俄羅斯', '義大利', '西班牙', '荷蘭']
    
    cities = ['北京', '上海', '紐約', '洛杉磯', '東京', '大阪', '柏林', '慕尼黑',
             '巴黎', '里昂', '倫敦', '曼徹斯特', '雪梨', '墨爾本', '多倫多']
    
    transportation = ['汽車', '公車', '捷運', '自行車', '步行', '機車', '計程車', 
                     '火車', '高鐵', '飛機', '船', '電動車', '滑板車', 'Uber', 'Grab']
    
    products = ['iPhone 14', 'Samsung Galaxy', 'MacBook Pro', 'Dell XPS', 'iPad Pro',
               'Surface Pro', 'AirPods', 'Sony WH-1000XM4', 'Canon EOS', 'Nikon D850']
    
    # 生成時間數據（更複雜）
    base_time = pd.Timestamp('2023-01-01')
    times = []
    for i in range(n_samples):
        random_days = np.random.randint(0, 365)
        random_hours = np.random.randint(0, 24)
        random_minutes = np.random.randint(0, 60)
        time_str = (base_time + pd.Timedelta(days=random_days, hours=random_hours, minutes=random_minutes)).strftime('%Y-%m-%d %H:%M:%S')
        times.append(time_str)
    
    df = pd.DataFrame({
        '時間戳記': times,
        '國家': np.random.choice(countries, n_samples),
        '城市': np.random.choice(cities, n_samples),
        '交通方式': np.random.choice(transportation, n_samples),
        '產品名稱': np.random.choice(products, n_samples),
        '年齡': np.random.randint(18, 80, n_samples),
        '收入': np.random.normal(50000, 20000, n_samples),
        '評分': np.random.uniform(1, 5, n_samples),
        '購買數量': np.random.poisson(3, n_samples),
        '滿意度': np.random.choice(['非常滿意', '滿意', '普通', '不滿意', '非常不滿意'], n_samples)
    })
    
    # 保存示例數據
    sample_file = 'tmp/sample_data_for_aggregation.csv'
    os.makedirs('tmp', exist_ok=True)
    df.to_csv(sample_file, index=False, encoding='utf-8')
    
    print(f"示例數據已保存到: {sample_file}")
    print(f"數據形狀: {df.shape}")
    
    return sample_file

async def main():
    """主函數"""
    print("離散數據聚合整合範例")
    
    # 創建示例數據
    sample_file = await create_sample_data()
    
    # 執行完整分析流程
    results = await analyze_with_aggregation(
        file_path=sample_file,
        discrete_columns=['時間戳記', '國家', '城市', '交通方式', '產品名稱', '滿意度'],
        max_categories=8,
        use_fast_correlation=False,  # 使用 MIC 以處理離散數據
        n_jobs=4  # 使用 4 個進程並行計算
    )
    
    if results:
        print("\n" + "=" * 60)
        print("分析完成！")
        print("=" * 60)
        
        # 顯示一些統計信息
        corr_matrix = results['correlation_matrix']
        print(f"\n最高相關性配對:")
        
        # 找出最高的非對角線相關性
        mask = np.triu(np.ones_like(corr_matrix.values), k=1).astype(bool)
        correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if not np.isnan(corr_matrix.iloc[i, j]):
                    correlations.append({
                        'col1': corr_matrix.columns[i],
                        'col2': corr_matrix.columns[j], 
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        # 排序並顯示前5個
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        for i, corr in enumerate(correlations[:5]):
            print(f"  {i+1}. {corr['col1']} ↔ {corr['col2']}: {corr['correlation']:.3f}")

if __name__ == "__main__":
    asyncio.run(main())
