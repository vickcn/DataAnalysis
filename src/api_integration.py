#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
離散數據聚合 API 整合
為 DataAnalysis API 添加離散數據聚合端點
"""

import asyncio
import os
import sys
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from fastapi import HTTPException
from fastapi.responses import JSONResponse
# 添加父目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.discrete_aggregator import aggregate_discrete_data
import DataAnalysis as DA

# API 請求模型
class DiscreteAggregationRequest(BaseModel):
    filePath: str
    sheet: Optional[int] = 0
    discrete_columns: Optional[List[str]] = None
    max_categories: Optional[int] = None  # None 表示從 config.json 讀取
    textprocessor_url: Optional[str] = None  # None 表示從 config.json 讀取
    llm_provider: Optional[str] = "remote"  # None 表示從 config.json 讀取
    llm_model: Optional[str] = "remote8b"  # None 表示從 config.json 讀取
    ret: Dict[str, Any] = {}

class DiscreteAggregationWithAnalysisRequest(BaseModel):
    filePath: str
    sheet: Optional[int] = 0
    discrete_columns: Optional[List[str]] = None
    max_categories: Optional[int] = None  # None 表示從 config.json 讀取
    textprocessor_url: Optional[str] = None  # None 表示從 config.json 讀取
    llm_provider: Optional[str] = "remote"  # None 表示從 config.json 讀取
    llm_model: Optional[str] = "remote8b"  # None 表示從 config.json 讀取
    use_fast_correlation: bool = False
    n_jobs: Optional[int] = None
    ret: Dict[str, Any] = {}

async def discrete_aggregation_only(request: DiscreteAggregationRequest):
    """
    僅執行離散數據聚合
    
    Returns:
    --------
    Dict containing:
    - success: bool
    - aggregated_data: DataFrame (as dict)
    - aggregation_results: List[Dict]
    - message: str
    """
    try:
        # 載入數據
        ret = request.ret.copy()
        success = DA.DataParsingFromFile(request.filePath, ret=ret, sht=request.sheet)
        
        if not success or 'matrix' not in ret:
            return {
                "success": False,
                "message": "數據載入失敗",
                "aggregated_data": None,
                "aggregation_results": []
            }
        
        df = ret['matrix']
        
        # 準備聚合參數，過濾無效值
        aggregation_kwargs = {}
        
        # 只傳遞有效的參數
        if request.max_categories is not None:
            aggregation_kwargs['max_categories'] = request.max_categories
        
        if request.textprocessor_url is not None and isinstance(request.textprocessor_url, str) and request.textprocessor_url.startswith(('http://', 'https://')):
            aggregation_kwargs['textprocessor_url'] = request.textprocessor_url
        
        # 如果提供了 llm_provider 且與預設值不同，才使用（預設值從 config.json 讀取）
        if request.llm_provider is not None and request.llm_provider != "remote":
            aggregation_kwargs['llm_provider'] = request.llm_provider
        
        # 如果提供了 llm_model 且與預設值不同，才使用（預設值從 config.json 讀取）
        if request.llm_model is not None and request.llm_model != "remote8b":
            aggregation_kwargs['llm_model'] = request.llm_model
        
        # 執行離散數據聚合（None 值會被 aggregate_discrete_data 從 config.json 讀取）
        df_aggregated, aggregation_results = await aggregate_discrete_data(
            df,
            columns=request.discrete_columns,
            **aggregation_kwargs
        )
        
        # 更新 ret 中的數據
        ret['matrix'] = df_aggregated
        ret['aggregation_results'] = aggregation_results
        
        # 構建響應
        try:
            # 限制返回的數據量（最多返回前1000行，避免響應過大）
            max_rows = 1000
            if len(df_aggregated) > max_rows:
                df_preview = df_aggregated.head(max_rows)
                data_preview = df_preview.to_dict('records')
                data_summary = {
                    "total_rows": len(df_aggregated),
                    "returned_rows": max_rows,
                    "preview_only": True
                }
            else:
                data_preview = df_aggregated.to_dict('records')
                data_summary = {
                    "total_rows": len(df_aggregated),
                    "returned_rows": len(df_aggregated),
                    "preview_only": False
                }
            
            # 清理 ret 中不可序列化的對象
            serializable_ret = {}
            for key, value in ret.items():
                try:
                    # 跳過 DataFrame 對象，我們已經處理了
                    if hasattr(value, 'to_dict'):
                        serializable_ret[key] = f"<DataFrame: {value.shape[0]} rows x {value.shape[1]} cols>"
                    else:
                        # 嘗試序列化測試
                        import json
                        json.dumps(value, default=str)
                        serializable_ret[key] = value
                except (TypeError, ValueError) as e:
                    # 如果無法序列化，轉換為字符串描述
                    serializable_ret[key] = f"<{type(value).__name__} object>"
            
            response = {
                "success": True,
                "message": f"離散數據聚合完成，處理了 {len([r for r in aggregation_results if r['aggregated']])} 個欄位",
                "data_summary": data_summary,
                "aggregated_data": data_preview,
                "aggregation_results": aggregation_results,
                "ret": serializable_ret
            }
            
            # 驗證響應可以序列化
            import json
            json.dumps(response, default=str, ensure_ascii=False)
            
            return response
            
        except Exception as serialization_error:
            # 如果序列化失敗，返回簡化版本的響應
            return {
                "success": True,
                "message": f"離散數據聚合完成，處理了 {len([r for r in aggregation_results if r['aggregated']])} 個欄位，但數據過大無法完整返回",
                "data_summary": {
                    "total_rows": len(df_aggregated),
                    "columns": list(df_aggregated.columns),
                    "serialization_error": str(serialization_error)
                },
                "aggregated_data": None,
                "aggregation_results": aggregation_results,
                "ret": {}
            }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"離散數據聚合失敗: {str(e)}",
            "aggregated_data": None,
            "aggregation_results": []
        }

async def discrete_aggregation_with_analysis(request: DiscreteAggregationWithAnalysisRequest):
    """
    執行離散數據聚合 + 相關性分析
    
    Returns:
    --------
    Dict containing:
    - success: bool
    - aggregated_data: DataFrame (as dict) 
    - aggregation_results: List[Dict]
    - correlation_analysis: bool
    - message: str
    """
    try:
        # 1. 載入數據
        ret = request.ret.copy()
        success = DA.DataParsingFromFile(request.filePath, ret=ret, sht=request.sheet)
        
        if not success or 'matrix' not in ret:
            return {
                "success": False,
                "message": "數據載入失敗",
                "aggregated_data": None,
                "aggregation_results": [],
                "correlation_analysis": False
            }
        
        df = ret['matrix']
        
        # 2. 準備聚合參數，過濾無效值
        aggregation_kwargs = {}
        
        # 只傳遞有效的參數
        if request.max_categories is not None:
            aggregation_kwargs['max_categories'] = request.max_categories
        
        if request.textprocessor_url is not None and isinstance(request.textprocessor_url, str) and request.textprocessor_url.startswith(('http://', 'https://')):
            aggregation_kwargs['textprocessor_url'] = request.textprocessor_url
        
        # 如果提供了 llm_provider 且與預設值不同，才使用（預設值從 config.json 讀取）
        if request.llm_provider is not None and request.llm_provider != "remote":
            aggregation_kwargs['llm_provider'] = request.llm_provider
        
        # 如果提供了 llm_model 且與預設值不同，才使用（預設值從 config.json 讀取）
        if request.llm_model is not None and request.llm_model != "remote8b":
            aggregation_kwargs['llm_model'] = request.llm_model
        
        # 執行離散數據聚合（None 值會被 aggregate_discrete_data 從 config.json 讀取）
        df_aggregated, aggregation_results = await aggregate_discrete_data(
            df,
            columns=request.discrete_columns,
            **aggregation_kwargs
        )
        
        # 3. 準備分析數據（只包含適合分析的欄位）
        analysis_columns = []
        for col in df_aggregated.columns:
            if (df_aggregated[col].dtype in ['int64', 'float64'] or 
                df_aggregated[col].nunique() <= request.max_categories):
                analysis_columns.append(col)
        
        df_for_analysis = df_aggregated[analysis_columns]
        
        # 4. 執行相關性分析
        correlation_success = False
        try:
            # 設定輸出目錄
            exp_fd = DA.autoExportSet(None, source_file=request.filePath)
            exp_fd = os.path.join(exp_fd, 'aggregated_analysis')
            
            # 執行相關性分析
            correlation_success = DA.PlotCorrelation(
                df_for_analysis,
                method='mic',
                exp_fd=exp_fd,
                ret=ret,
                n_jobs=request.n_jobs,
                use_fast_correlation=request.use_fast_correlation
            )
            
        except Exception as e:
            print(f"相關性分析失敗: {e}")
        
        # 更新 ret 中的數據
        ret['matrix'] = df_aggregated
        ret['analysis_matrix'] = df_for_analysis
        ret['aggregation_results'] = aggregation_results
        
        # 構建響應
        try:
            # 限制返回的數據量（最多返回前1000行，避免響應過大）
            max_rows = 1000
            
            # 處理聚合後的數據
            if len(df_aggregated) > max_rows:
                df_agg_preview = df_aggregated.head(max_rows)
                agg_data_preview = df_agg_preview.to_dict('records')
                agg_data_summary = {
                    "total_rows": len(df_aggregated),
                    "returned_rows": max_rows,
                    "preview_only": True
                }
            else:
                agg_data_preview = df_aggregated.to_dict('records')
                agg_data_summary = {
                    "total_rows": len(df_aggregated),
                    "returned_rows": len(df_aggregated),
                    "preview_only": False
                }
            
            # 處理分析數據
            if len(df_for_analysis) > max_rows:
                df_ana_preview = df_for_analysis.head(max_rows)
                ana_data_preview = df_ana_preview.to_dict('records')
                ana_data_summary = {
                    "total_rows": len(df_for_analysis),
                    "returned_rows": max_rows,
                    "preview_only": True
                }
            else:
                ana_data_preview = df_for_analysis.to_dict('records')
                ana_data_summary = {
                    "total_rows": len(df_for_analysis),
                    "returned_rows": len(df_for_analysis),
                    "preview_only": False
                }
            
            # 清理 ret 中不可序列化的對象
            serializable_ret = {}
            for key, value in ret.items():
                try:
                    # 跳過 DataFrame 對象，我們已經處理了
                    if hasattr(value, 'to_dict'):
                        serializable_ret[key] = f"<DataFrame: {value.shape[0]} rows x {value.shape[1]} cols>"
                    else:
                        # 嘗試序列化測試
                        import json
                        json.dumps(value, default=str)
                        serializable_ret[key] = value
                except (TypeError, ValueError) as e:
                    # 如果無法序列化，轉換為字符串描述
                    serializable_ret[key] = f"<{type(value).__name__} object>"
            
            response = {
                "success": True,
                "message": f"離散數據聚合和相關性分析完成。聚合了 {len([r for r in aggregation_results if r['aggregated']])} 個欄位，分析了 {len(analysis_columns)} 個欄位",
                "aggregated_data_summary": agg_data_summary,
                "aggregated_data": agg_data_preview,
                "analysis_data_summary": ana_data_summary,
                "analysis_data": ana_data_preview,
                "aggregation_results": aggregation_results,
                "correlation_analysis": correlation_success,
                "analysis_columns": analysis_columns,
                "ret": serializable_ret
            }
            
            # 驗證響應可以序列化
            import json
            json.dumps(response, default=str, ensure_ascii=False)
            
            return response
            
        except Exception as serialization_error:
            # 如果序列化失敗，返回簡化版本的響應
            return {
                "success": True,
                "message": f"離散數據聚合和相關性分析完成，但數據過大無法完整返回",
                "aggregated_data_summary": {
                    "total_rows": len(df_aggregated),
                    "columns": list(df_aggregated.columns),
                },
                "analysis_data_summary": {
                    "total_rows": len(df_for_analysis),
                    "columns": list(df_for_analysis.columns),
                },
                "aggregated_data": None,
                "analysis_data": None,
                "aggregation_results": aggregation_results,
                "correlation_analysis": correlation_success,
                "analysis_columns": analysis_columns,
                "serialization_error": str(serialization_error),
                "ret": {}
            }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"處理失敗: {str(e)}",
            "aggregated_data": None,
            "aggregation_results": [],
            "correlation_analysis": False
        }

# 添加到 FastAPI 應用的端點定義
def add_discrete_aggregation_endpoints(app):
    """
    將離散數據聚合端點添加到 FastAPI 應用
    
    Parameters:
    -----------
    app : FastAPI
        FastAPI 應用實例
    """
    
    @app.post("/discrete-aggregation")
    async def discrete_data_aggregation(request: DiscreteAggregationRequest):
        """離散數據聚合端點"""
        try:
            result = await discrete_aggregation_only(request)
            # 使用 JSONResponse 明確返回，確保正確序列化
            return JSONResponse(content=result, status_code=200)
        except Exception as e:
            import traceback
            error_detail = {
                "success": False,
                "message": f"離散數據聚合失敗: {str(e)}",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            return JSONResponse(content=error_detail, status_code=500)
    
    @app.post("/discrete-aggregation-with-analysis")
    async def discrete_aggregation_and_analysis(request: DiscreteAggregationWithAnalysisRequest):
        """離散數據聚合 + 相關性分析端點"""
        try:
            result = await discrete_aggregation_with_analysis(request)
            # 使用 JSONResponse 明確返回，確保正確序列化
            return JSONResponse(content=result, status_code=200)
        except Exception as e:
            import traceback
            error_detail = {
                "success": False,
                "message": f"處理失敗: {str(e)}",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            return JSONResponse(content=error_detail, status_code=500)

# 使用範例
async def example_usage():
    """使用範例"""
    
    # 範例 1: 僅聚合
    request1 = DiscreteAggregationRequest(
        filePath="path/to/your/data.csv",
        discrete_columns=["country", "transportation", "category"],
        max_categories=8,
        ret={}
    )
    
    result1 = await discrete_aggregation_only(request1)
    print("聚合結果:", result1["success"])
    
    # 範例 2: 聚合 + 分析
    request2 = DiscreteAggregationWithAnalysisRequest(
        filePath="path/to/your/data.csv",
        discrete_columns=["country", "transportation", "category"],
        max_categories=8,
        use_fast_correlation=False,
        n_jobs=4,
        ret={}
    )
    
    result2 = await discrete_aggregation_with_analysis(request2)
    print("聚合+分析結果:", result2["success"])

if __name__ == "__main__":
    asyncio.run(example_usage())
