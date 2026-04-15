#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
離散型數據聚合功能模組
使用 LLM 智能聚合離散數據，減少類別數量以提升分析效率
"""

import os
import sys
import json
import re
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

# 添加父目錄到路徑以導入 package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from package import LOGger

def load_llm_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    從 config.json 載入 LLM 配置
    
    Parameters:
    -----------
    config_path : str, optional
        配置檔案路徑，None 表示使用預設路徑
        
    Returns:
    --------
    Dict[str, Any]
        LLM 配置字典
    """
    if config_path is None:
        # 預設配置檔案路徑
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, 'config.json')
    
    default_config = {
        'textprocessor_url': 'http://10.1.3.127:6017',
        'provider': 'remote',
        'model': 'remote8b',
        'max_categories': 20,
        'aggregation_threshold': 20,  # 唯一值數量超過此值才需要聚合
        'min_frequency': 1
    }
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                llm_config = config.get('llm', {})
                
                # 合併預設值和配置值
                result = default_config.copy()
                result.update(llm_config)
                
                return result
    except Exception as e:
        # 使用 print 因為此時 LOGger 可能尚未完全初始化
        print(f'警告: 載入 LLM 配置失敗: {e}，使用預設配置')
    
    return default_config

class DiscreteAggregator:
    """離散數據聚合器"""
    
    def __init__(self, 
                 textprocessor_url: Optional[str] = None,
                 llm_provider: Optional[str] = None,
                 llm_model: Optional[str] = None,
                 max_categories: Optional[int] = None,
                 aggregation_threshold: Optional[int] = None,
                 min_frequency: Optional[int] = None,
                 config_path: Optional[str] = None,
                 exp_fd: Optional[str] = None):
        """
        初始化聚合器
        
        Parameters:
        -----------
        textprocessor_url : str, optional
            TextProcessor 服務的 URL，None 表示從 config.json 讀取
        llm_provider : str, optional
            LLM 提供者 (openai, remote 等)，None 表示從 config.json 讀取
        llm_model : str, optional
            LLM 模型別名，None 表示從 config.json 讀取
        max_categories : int, optional
            聚合後的最大類別數，None 表示從 config.json 讀取
        aggregation_threshold : int, optional
            聚合閾值，唯一值數量超過此值才需要聚合，None 表示從 config.json 讀取
        min_frequency : int, optional
            保留類別的最小頻率，None 表示從 config.json 讀取
        config_path : str, optional
            配置檔案路徑，None 表示使用預設路徑
        """
        # 從 config.json 載入配置
        llm_config = load_llm_config(config_path)
        
        # 驗證並設置 textprocessor_url
        if textprocessor_url is not None and isinstance(textprocessor_url, str) and textprocessor_url.startswith(('http://', 'https://')):
            self.textprocessor_url = textprocessor_url
        else:
            # 如果參數無效，使用 config.json 中的值
            if textprocessor_url is not None and textprocessor_url != llm_config['textprocessor_url']:
                print(f'警告: 無效的 textprocessor_url 參數 "{textprocessor_url}"，使用 config.json 中的值: {llm_config["textprocessor_url"]}')
            self.textprocessor_url = llm_config['textprocessor_url']
        
        # 驗證 URL 格式
        if not isinstance(self.textprocessor_url, str) or not self.textprocessor_url.startswith(('http://', 'https://')):
            raise ValueError(f'無效的 textprocessor_url: {self.textprocessor_url}，必須是有效的 HTTP/HTTPS URL')
        
        # 使用參數值覆蓋配置值（如果提供且有效）
        self.llm_provider = llm_provider if llm_provider is not None else llm_config['provider']
        self.llm_model = llm_model if llm_model is not None else llm_config['model']
        self.max_categories = max_categories if max_categories is not None else llm_config['max_categories']
        self.aggregation_threshold = aggregation_threshold if aggregation_threshold is not None else llm_config['aggregation_threshold']
        self.min_frequency = min_frequency if min_frequency is not None else llm_config['min_frequency']
        
        # 日誌設置
        self.logger = LOGger
        self.log_file = os.path.join('log', 'discrete_aggregator_%t.txt')
        self.m_addlog = LOGger.addloger(logfile=self.log_file)
        
        # 記錄實際使用的配置（確認 URL 正確）
        self.m_addlog(f'DiscreteAggregator 初始化完成: textprocessor_url={self.textprocessor_url}, provider={self.llm_provider}, model={self.llm_model}', 
                     colora=LOGger.OKCYAN)
        
        # JSON 日誌目錄（優先使用 exp_fd，否則使用預設的 jsonlog 目錄）
        if exp_fd:
            # 確保 exp_fd 目錄存在
            os.makedirs(exp_fd, exist_ok=True)
            self.jsonlog_dir = exp_fd
        else:
            self.jsonlog_dir = os.path.join('jsonlog', datetime.now().strftime('%Y%m%d'))
            os.makedirs(self.jsonlog_dir, exist_ok=True)
    
    def _is_numeric_data(self, series: pd.Series) -> bool:
        """檢查是否為數值型數據（應保持連續性，不進行聚合）"""
        try:
            # 首先檢查數據類型
            if series.dtype in ['int64', 'float64', 'int32', 'float32', 'Int64', 'Float64']:
                return True
            
            # 嘗試轉換為數值（檢查是否可以全部轉換）
            sample = series.head(100)
            if len(sample) == 0:
                return False
            
            # 檢查樣本中能轉換為數值的比例
            numeric_count = 0
            for val in sample:
                try:
                    float(val)
                    numeric_count += 1
                except (ValueError, TypeError):
                    pass
            
            # 如果超過 90% 的樣本都是數值，視為數值型
            return numeric_count / len(sample) >= 0.9
        except Exception:
            return False
        
    def analyze_column(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """
        分析單一欄位的離散數據特性
        
        Parameters:
        -----------
        series : pd.Series
            要分析的數據序列
        column_name : str
            欄位名稱
            
        Returns:
        --------
        Dict[str, Any]
            分析結果
        """
        # 移除空值
        clean_series = series.dropna()
        
        # 統計唯一值
        value_counts = clean_series.value_counts()
        unique_count = len(value_counts)
        
        # 檢查是否為數值型數據
        is_numeric = self._is_numeric_data(clean_series)
        
        # 分析數據類型
        data_type = self._detect_data_type(clean_series)
        
        # 如果是數值型數據，不需要聚合（保持連續性）
        needs_aggregation = unique_count > self.aggregation_threshold and not is_numeric
        
        analysis = {
            'column_name': column_name,
            'total_count': len(series),
            'non_null_count': len(clean_series),
            'unique_count': unique_count,
            'data_type': data_type,
            'is_numeric': is_numeric,
            'needs_aggregation': needs_aggregation,
            'value_counts': value_counts.to_dict(),
            'top_values': value_counts.head(20).to_dict()
        }
        
        # 記錄分析結果
        if is_numeric:
            self.m_addlog(f'分析欄位 {column_name}: 唯一值數量={unique_count}, 數據類型={data_type}, 數值型（跳過聚合）', 
                         colora=LOGger.OKCYAN)
        else:
            self.m_addlog(f'分析欄位 {column_name}: 唯一值數量={unique_count}, 數據類型={data_type}', 
                         colora=LOGger.OKBLUE)
        
        return analysis
    
    def _detect_data_type(self, series: pd.Series) -> str:
        """檢測數據類型"""
        sample_values = series.head(100).astype(str).tolist()
        
        # 檢查是否為時間格式
        if self._is_datetime_like(sample_values):
            return 'datetime'
        
        # 檢查是否為地理位置
        if self._is_geographic_like(sample_values):
            return 'geographic'
        
        # 檢查是否為交通方式
        if self._is_transportation_like(sample_values):
            return 'transportation'
        
        # 檢查是否為類別編碼
        if self._is_category_code_like(sample_values):
            return 'category_code'
        
        return 'general'
    
    def _is_datetime_like(self, values: List[str]) -> bool:
        """檢查是否為時間格式"""
        datetime_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
            r'\d{2}:\d{2}:\d{2}',  # HH:MM:SS
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'  # YYYY-MM-DD HH:MM:SS
        ]
        
        for pattern in datetime_patterns:
            matches = sum(1 for v in values[:10] if re.search(pattern, str(v)))
            if matches >= len(values[:10]) * 0.7:  # 70% 匹配
                return True
        return False
    
    def _is_geographic_like(self, values: List[str]) -> bool:
        """檢查是否為地理位置"""
        geo_keywords = ['國', '省', '市', '縣', '區', '州', '洲', 'country', 'city', 'state']
        matches = sum(1 for v in values[:10] 
                     for keyword in geo_keywords 
                     if keyword.lower() in str(v).lower())
        return matches >= len(values[:10]) * 0.3
    
    def _is_transportation_like(self, values: List[str]) -> bool:
        """檢查是否為交通方式"""
        transport_keywords = ['車', '公車', '捷運', '火車', '飛機', '船', 'bus', 'train', 'car', 'plane']
        matches = sum(1 for v in values[:10] 
                     for keyword in transport_keywords 
                     if keyword.lower() in str(v).lower())
        return matches >= len(values[:10]) * 0.3
    
    def _is_category_code_like(self, values: List[str]) -> bool:
        """檢查是否為類別編碼"""
        # 檢查是否大多數值都是短字串或數字編碼
        short_values = sum(1 for v in values[:10] if len(str(v)) <= 5)
        return short_values >= len(values[:10]) * 0.7
    
    async def aggregate_column(self, series: pd.Series, column_name: str, 
                             analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        聚合單一欄位的離散數據
        
        Parameters:
        -----------
        series : pd.Series
            要聚合的數據序列
        column_name : str
            欄位名稱
        analysis : Dict[str, Any], optional
            預先分析的結果
            
        Returns:
        --------
        Dict[str, Any]
            聚合結果，包含映射關係
        """
        if analysis is None:
            analysis = self.analyze_column(series, column_name)
        
        if not analysis['needs_aggregation']:
            self.m_addlog(f'欄位 {column_name} 不需要聚合', colora=LOGger.WARNING)
            return {
                'column_name': column_name,
                'aggregated': False,
                'mapping': {},
                'original_categories': analysis['unique_count'],
                'final_categories': analysis['unique_count']
            }
        
        # 根據數據類型選擇聚合策略
        data_type = analysis['data_type']
        value_counts = analysis['value_counts']
        
        self.m_addlog(f'開始聚合欄位 {column_name}, 類型: {data_type}', colora=LOGger.WARNING)
        
        if data_type == 'datetime':
            mapping = await self._aggregate_datetime(value_counts, column_name)
        elif data_type == 'geographic':
            mapping = await self._aggregate_geographic(value_counts, column_name)
        elif data_type == 'transportation':
            mapping = await self._aggregate_transportation(value_counts, column_name)
        else:
            mapping = await self._aggregate_general(value_counts, column_name, data_type)
        
        # 記錄聚合結果
        result = {
            'column_name': column_name,
            'data_type': data_type,
            'aggregated': True,
            'mapping': mapping,
            'original_categories': len(value_counts),
            'final_categories': len(set(mapping.values())),
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存到 JSON 日誌
        self._save_aggregation_log(result)
        
        self.m_addlog(f'聚合完成: {column_name} ({len(value_counts)} -> {len(set(mapping.values()))} 類別)', 
                     colora=LOGger.OKCYAN)
        
        return result
    
    async def _aggregate_datetime(self, value_counts: Dict[str, int], column_name: str) -> Dict[str, str]:
        """聚合時間數據"""
        prompt = f"""
        請將以下時間數據聚合成不超過{self.max_categories}個時間段。

        欄位名稱: {column_name}
        數據範例: {list(value_counts.keys())[:20]}

        聚合原則:
        1. 將相似的時間聚合成有意義的時間段
        2. 考慮時間的自然分組（如：上午、下午、晚上；工作日、週末；季節等）
        3. 最終類別數不超過{self.max_categories}個
        4. 使用中文描述時間段

        請以JSON格式回傳映射關係，格式如下:
        {{"原始值1": "聚合後類別1", "原始值2": "聚合後類別2", ...}}
        """
        
        return await self._call_llm_for_mapping(prompt, value_counts)
    
    async def _aggregate_geographic(self, value_counts: Dict[str, int], column_name: str) -> Dict[str, str]:
        """聚合地理數據"""
        prompt = f"""
        請將以下地理位置數據聚合成不超過{self.max_categories}個地理區域。

        欄位名稱: {column_name}
        數據範例: {list(value_counts.keys())[:20]}

        聚合原則:
        1. 按照地理區域聚合（如：亞洲、歐洲、北美洲等）
        2. 或按照行政區劃聚合（如：華北、華南、華東等）
        3. 最終類別數不超過{self.max_categories}個
        4. 使用中文描述地理區域

        請以JSON格式回傳映射關係，格式如下:
        {{"原始值1": "聚合後類別1", "原始值2": "聚合後類別2", ...}}
        """
        
        return await self._call_llm_for_mapping(prompt, value_counts)
    
    async def _aggregate_transportation(self, value_counts: Dict[str, int], column_name: str) -> Dict[str, str]:
        """聚合交通方式數據"""
        prompt = f"""
        請將以下交通方式數據聚合成不超過6個主要交通類別。

        欄位名稱: {column_name}
        數據範例: {list(value_counts.keys())[:20]}

        聚合原則:
        1. 按照交通工具類型聚合（如：陸運、海運、空運）
        2. 或按照交通方式聚合（如：公共交通、私人交通、步行等）
        3. 最終類別數不超過6個
        4. 使用中文描述交通類別

        請以JSON格式回傳映射關係，格式如下:
        {{"原始值1": "聚合後類別1", "原始值2": "聚合後類別2", ...}}
        """
        
        return await self._call_llm_for_mapping(prompt, value_counts)
    
    async def _aggregate_general(self, value_counts: Dict[str, int], column_name: str, data_type: str) -> Dict[str, str]:
        """聚合一般類別數據"""
        prompt = f"""
        請將以下類別數據聚合成不超過{self.max_categories}個主要類別。

        欄位名稱: {column_name}
        數據類型: {data_type}
        數據範例: {list(value_counts.keys())[:20]}
        頻率資訊: {dict(list(value_counts.items())[:10])}

        聚合原則:
        1. 根據語義相似性聚合相關類別
        2. 保留高頻率的重要類別
        3. 將低頻率的類別合併到相似的高頻類別中
        4. 最終類別數不超過{self.max_categories}個
        5. 使用簡潔明確的中文描述

        請以JSON格式回傳映射關係，格式如下:
        {{"原始值1": "聚合後類別1", "原始值2": "聚合後類別2", ...}}
        """
        
        return await self._call_llm_for_mapping(prompt, value_counts)
    
    async def _call_llm_for_mapping(self, prompt: str, value_counts: Dict[str, int]) -> Dict[str, str]:
        """調用 LLM 生成映射關係"""
        try:
            # 驗證 URL
            if not self.textprocessor_url or not isinstance(self.textprocessor_url, str):
                raise ValueError(f'textprocessor_url 無效: {self.textprocessor_url}')
            
            # 確保 URL 格式正確
            url = self.textprocessor_url.rstrip('/')
            if not url.startswith(('http://', 'https://')):
                raise ValueError(f'textprocessor_url 必須以 http:// 或 https:// 開頭: {url}')
            
            chat_url = f"{url}/chat"
            self.m_addlog(f'調用 LLM: {chat_url}, provider={self.llm_provider}, model={self.llm_model}')
            
            # 調用 TextProcessor 的 chat 服務
            response = requests.post(
                chat_url,
                json={
                    "prompt": prompt,
                    "provider": self.llm_provider,
                    "model": self.llm_model,
                    "system_prompt": "你是專業的數據分析師，擅長數據聚合和分類。請嚴格按照JSON格式回傳結果。",
                    "max_tokens": 2000,
                    "temperature": 0.3
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                llm_output = result.get('output', '')
                
                # 解析 LLM 輸出的 JSON
                try:
                    # 提取 JSON 部分
                    json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
                    if json_match:
                        mapping = json.loads(json_match.group())
                        
                        # 驗證映射關係
                        if isinstance(mapping, dict):
                            self.m_addlog(f'LLM 成功生成映射關係，包含 {len(mapping)} 個映射', 
                                        colora=LOGger.OKCYAN)
                            return mapping
                    
                    raise ValueError("無法解析 LLM 輸出的 JSON")
                    
                except json.JSONDecodeError as e:
                    self.m_addlog(f'JSON 解析失敗: {e}', colora=LOGger.FAIL)
                    self.m_addlog(f'LLM 輸出: {llm_output}')
                    
            else:
                self.m_addlog(f'LLM 調用失敗: HTTP {response.status_code}', colora=LOGger.FAIL)
                
        except Exception as e:
            self.m_addlog(f'LLM 調用異常: {e}', colora=LOGger.FAIL)
        
        # 回退到基於頻率的簡單聚合
        return self._fallback_frequency_aggregation(value_counts)
    
    def _fallback_frequency_aggregation(self, value_counts: Dict[str, int]) -> Dict[str, str]:
        """基於頻率的回退聚合策略"""
        self.m_addlog('使用回退聚合策略', colora=LOGger.WARNING)
        
        # 按頻率排序
        sorted_items = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        
        mapping = {}
        category_counter = 1
        
        for value, count in sorted_items:
            if count >= self.min_frequency and category_counter <= self.max_categories:
                # 高頻值保持原樣或簡化
                if len(str(value)) > 10:
                    mapping[value] = f'類別{category_counter}'
                else:
                    mapping[value] = str(value)
                category_counter += 1
            else:
                # 低頻值歸類為"其他"
                mapping[value] = '其他'
        
        return mapping
    
    def apply_aggregation(self, df: pd.DataFrame, aggregation_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        將聚合結果應用到 DataFrame
        
        Parameters:
        -----------
        df : pd.DataFrame
            原始數據
        aggregation_results : List[Dict[str, Any]]
            聚合結果列表
            
        Returns:
        --------
        pd.DataFrame
            聚合後的數據
        """
        df_aggregated = df.copy()
        
        for result in aggregation_results:
            if result['aggregated']:
                column_name = result['column_name']
                mapping = result['mapping']
                
                # 應用映射
                df_aggregated[column_name] = df_aggregated[column_name].map(mapping).fillna('其他')
                
                self.m_addlog(f'已應用聚合到欄位 {column_name}', colora=LOGger.OKBLUE)
        
        return df_aggregated
    
    def _save_aggregation_log(self, result: Dict[str, Any]):
        """保存聚合日誌到 JSON 文件"""
        timestamp = datetime.now()
        log_filename = f"aggregation_{timestamp.strftime('%H%M%S')}_{result['column_name']}.json"
        log_path = os.path.join(self.jsonlog_dir, log_filename)
        
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            self.m_addlog(f'聚合日誌已保存: {log_path}')
            
        except Exception as e:
            self.m_addlog(f'保存聚合日誌失敗: {e}', colora=LOGger.FAIL)

# 便利函數
async def aggregate_discrete_data(df: pd.DataFrame, 
                                columns: Optional[List[str]] = None,
                                **kwargs) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    聚合 DataFrame 中的離散數據
    
    Parameters:
    -----------
    df : pd.DataFrame
        要處理的數據
    columns : List[str], optional
        要聚合的欄位列表，None 表示自動檢測
    **kwargs
        傳遞給 DiscreteAggregator 的參數
        
    Returns:
    --------
    Tuple[pd.DataFrame, List[Dict[str, Any]]]
        聚合後的數據和聚合結果
    """
    aggregator = DiscreteAggregator(**kwargs)
    
    if columns is None:
        # 自動檢測需要聚合的欄位（排除數值型數據）
        columns = []
        for col in df.columns:
            # 檢查是否為數值型
            is_numeric = aggregator._is_numeric_data(df[col])
            
            # 只聚合非數值型且超過閾值的欄位
            if not is_numeric and df[col].nunique() > aggregator.aggregation_threshold:
                columns.append(col)
            elif is_numeric:
                # 記錄跳過的數值型欄位
                aggregator.m_addlog(f'跳過數值型欄位 {col}（保持連續性）', colora=LOGger.OKCYAN)
    
    aggregation_results = []
    
    for col in columns:
        analysis = aggregator.analyze_column(df[col], col)
        if analysis['needs_aggregation']:
            result = await aggregator.aggregate_column(df[col], col, analysis)
            aggregation_results.append(result)
    
    # 應用聚合
    df_aggregated = aggregator.apply_aggregation(df, aggregation_results)
    
    return df_aggregated, aggregation_results
