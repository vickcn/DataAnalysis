"""
使用 API 調用 MDC 模型的模組
替代原本直接使用 mdc 物件的方式
"""

import requests
import pandas as pd
import numpy as np
import os
import json
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Dict, Any, List

# 導入必要的模組（從 data_analysis.py）
m_print = lambda *args, **kwargs: print(*args)
try:
    from . import visualization3 as vs3
    DFP = vs3.DFP
    plt = vs3.plt
    vs2 = vs3.vs2
    vs = vs2.vs
    LOGger = DFP.LOGger
    from package.data_analysis import (
        determineDataType,
        performStatisticalTests,
        addStatisticalInfoToPlot,
        getProjectOutputPath
    )

    m_print = LOGger.addloger(logfile='')
except ImportError:
    # 如果函數不存在，定義空函數或使用替代方案
    def determineDataType(data):
        """判斷數據類型"""
        if data.dtype in ['int64', 'float64']:
            return 'continuous'
        return 'categorical'
    
    def performStatisticalTests(*args, **kwargs):
        """執行統計檢定（佔位函數）"""
        return False
    
    def addStatisticalInfoToPlot(*args, **kwargs):
        """添加統計資訊到圖表（佔位函數）"""
        pass
    
    def getProjectOutputPath(output_dir, filename):
        """獲取專案輸出路徑"""
        return os.path.join(output_dir, filename)


def load_config(config_path=None):
    """
    從 config.json 讀取配置
    
    參數:
        config_path: config.json 的路徑（預設: 專案根目錄下的 config.json）
    
    返回:
        dict: 配置字典，如果讀取失敗則返回空字典
    """
    if config_path is None:
        # 預設路徑：專案根目錄下的 config.json
        # 從當前文件位置向上找到專案根目錄
        current_file = os.path.abspath(__file__)
        package_dir = os.path.dirname(current_file)
        project_root = os.path.dirname(package_dir)
        config_path = os.path.join(project_root, 'config.json')
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        else:
            return {}
    except Exception as e:
        LOGger.addlog(f"讀取配置文件失敗: {e}", logfile='', colora=LOGger.WARNING)
        return {}

def get_default_api_url():
    """
    從 config.json 讀取預設的 MDC API URL
    
    返回:
        str: API URL，如果讀取失敗則返回 None
    """
    config = load_config()
    if 'mdc_api' in config and 'url' in config['mdc_api']:
        return config['mdc_api']['url']
    return None


def get_mdc_headers_via_api(
        api_url: str, 
        model_name: str = "ACAngle", 
        version: str = "v0-0-2-0",
        timeout: int = 30
    ) -> tuple:
    """
    替代 mdc.xheader 和 mdc.yheader
    通過 API 獲取模型的輸入和輸出欄位
    
    Args:
        api_url: API 服務器 URL
        model_name: 模型名稱
        version: 模型版本
        timeout: 請求超時時間（秒）
        
    Returns:
        (xheader列表, yheader列表)
    """
    try:
        response = requests.get(
            f"{api_url}/api/mdc",
            params={
                "model_name": model_name,
                "version": version
            },
            timeout=timeout
        )
        response.raise_for_status()
        result = response.json()
        
        if result.get("success"):
            mdc_info = result["mdc_info"]
            xheader = mdc_info.get("xheader", [])
            yheader = mdc_info.get("yheader", [])
            return xheader, yheader
        else:
            error_detail = result.get("detail", {})
            error_msg = error_detail.get("message", "未知錯誤")
            LOGger.addlog(f"獲取模型資訊失敗: {error_msg}", logfile='', colora=LOGger.FAIL)
            return [], []
    except requests.exceptions.Timeout:
        LOGger.addlog(f"API 請求超時（{timeout} 秒）", logfile='', colora=LOGger.FAIL)
        return [], []
    except requests.exceptions.ConnectionError:
        LOGger.addlog(f"無法連接到 API 服務器: {api_url}", logfile='', colora=LOGger.FAIL)
        return [], []
    except Exception as e:
        LOGger.addlog(f"獲取模型資訊時發生錯誤: {str(e)}", logfile='', colora=LOGger.FAIL)
        return [], []


def predict_via_api(
        api_url: str,
        model_name: str,
        version: str,
        input_data: pd.DataFrame,
        timeout: int = 30
    ) -> Optional[np.ndarray]:
    """
    替代 mdc.predict(predict_data[mdc.xheader])
    通過 API 執行預測（支援批量數據）
    
    Args:
        api_url: API 服務器 URL
        model_name: 模型名稱
        version: 模型版本
        input_data: 輸入數據 DataFrame（支援單筆或多筆）
        timeout: 請求超時時間（秒）
        
    Returns:
        預測結果 numpy array，失敗則返回 None
    """
    try:
        # 將 DataFrame 轉換為列表格式（按行分組）：[{column: value}, ...]
        if isinstance(input_data, pd.DataFrame):
            # 先複製 DataFrame 避免修改原始數據
            df_copy = input_data.copy()
            
            # 轉換為列表格式：每行一個字典
            data_list = []
            for idx, row in df_copy.iterrows():
                row_dict = {}
                for col in df_copy.columns:
                    val = row[col]
                    # 處理 NaN 值：轉換為 None（JSON 的 null）
                    if pd.isna(val):
                        row_dict[col] = None
                    else:
                        # 確保值是 Python 原生類型
                        if isinstance(val, (np.integer, np.int64, np.int32)):
                            row_dict[col] = int(val)
                        elif isinstance(val, (np.floating, np.float64, np.float32)):
                            row_dict[col] = float(val)
                        elif isinstance(val, (np.bool_, bool)):
                            row_dict[col] = bool(val)
                        else:
                            row_dict[col] = val
                data_list.append(row_dict)
            
            # 如果只有一筆數據，可以選擇使用單筆格式或批量格式
            # 這裡統一使用列表格式（批量格式），API 會自動處理
            api_data = data_list if len(data_list) > 1 else data_list[0] if len(data_list) == 1 else {}
            
        elif isinstance(input_data, dict):
            # 如果已經是字典格式（單筆），轉換為列表格式
            api_data = [input_data]
        elif isinstance(input_data, list):
            # 如果已經是列表格式（批量），直接使用
            api_data = input_data
        else:
            # 不支援的格式
            m_print(f"數據格式不支援: {type(input_data)}", colora=LOGger.FAIL)
            return None
        
        # 發送請求前，先驗證數據是否可以序列化
        import json
        try:
            json.dumps(api_data)
        except (TypeError, ValueError) as e:
            m_print(f"數據無法序列化為 JSON: {str(e)}", colora=LOGger.FAIL)
            return None
        
        response = requests.post(
            f"{api_url}/api/predict",
            json={
                "model_name": model_name,
                "version": version,
                "data": api_data  # 使用列表格式（支援批量）
            },
            timeout=timeout
        )
        
        # 如果返回 422，記錄詳細錯誤信息
        if response.status_code == 422:
            try:
                error_detail = response.json()
                m_print(f"API 預測 422 錯誤詳情: {error_detail}", colora=LOGger.FAIL)
            except:
                m_print(f"API 預測 422 錯誤，無法解析響應: {response.text[:500]}", colora=LOGger.FAIL)
        
        response.raise_for_status()
        result = response.json()
        
        if result.get("success"):
            prediction_result = result.get("result", [])
            if prediction_result:
                # 轉換為 numpy array
                pred_array = np.array(prediction_result)
                
                # 處理結果格式
                # API 返回格式：
                # - 單筆：[[value]] 或 [value]
                # - 批量：[[value1], [value2], ...] 或 [[value1, value2, ...], ...]
                
                # 如果是二維陣列
                if pred_array.ndim == 2:
                    # 如果每行只有一個值，展平為一維
                    if pred_array.shape[1] == 1:
                        pred_array = pred_array.flatten()
                    # 否則保持二維（多輸出情況）
                # 如果是一維陣列，直接使用
                
                return pred_array
        else:
            error_detail = result.get("detail", {})
            error_msg = error_detail.get("message", "未知錯誤")
            m_print(f"API 預測失敗: {error_msg}", colora=LOGger.FAIL)
        return None
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 422:
            try:
                error_detail = e.response.json()
                m_print(f"API 預測 422 錯誤詳情: {error_detail}", colora=LOGger.FAIL)
            except:
                m_print(f"API 預測 422 錯誤，無法解析響應: {e.response.text[:500]}", colora=LOGger.FAIL)
        m_print(f"預測時發生 HTTP 錯誤: {str(e)}", colora=LOGger.FAIL)
        return None
    except requests.exceptions.Timeout:
        m_print(f"API 請求超時（{timeout} 秒）", colora=LOGger.FAIL)
        return None
    except requests.exceptions.ConnectionError:
        m_print(f"無法連接到 API 服務器: {api_url}", colora=LOGger.FAIL)
        return None
    except Exception as e:
        m_print(f"預測時發生錯誤: {str(e)}", colora=LOGger.FAIL)
        return None

def analyzeContinuousToContinuous(
        source_data: pd.DataFrame,
        xheader: str,
        yheader: str,
        fixed_values: Optional[Dict[str, Any]] = None,
        include_model_prediction: bool = True,
        ret: Optional[Dict[str, Any]] = None,
        api_url: Optional[str] = None,
        model_name: str = "ACAngle",
        version: str = "v0-0-2-0",
        output_dir: Optional[str] = None,
        timeout: int = 30,
        **kwags
    ) -> bool:
    """
    連續型x對連續型y的分析（使用 API 調用 MDC 模型）
    
    Parameters
    ----------
    source_data : pd.DataFrame
        原始資料
    xheader : str
        x軸header名稱
    yheader : str
        y軸header名稱
    fixed_values : dict, optional
        指定其他X因子的固定值，格式為 {'變數名': 固定值}
    include_model_prediction : bool
        是否包含模型預測
    ret : dict, optional
        回傳結果字典
    api_url : str, optional
        API 服務器 URL，如果為 None 則不執行模型預測
    model_name : str
        模型名稱
    version : str
        模型版本
    output_dir : str, optional
        輸出目錄
    timeout : int
        API 請求超時時間（秒）
    **kwags
        其他參數
        
    Returns
    -------
    bool
        是否成功分析
    """
    ret = ret if isinstance(ret, dict) else {}
    
    try:
        # 檢查是否需要 API
        if include_model_prediction and api_url is None:
            m_print("警告：未提供 api_url，跳過模型預測", colora=LOGger.WARNING)
            include_model_prediction = False
        
        # 設定中文字型
        vs3.plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
        
        # 取得x和y的範圍
        x_data = source_data[xheader].dropna()
        y_data = source_data[yheader].dropna()
        
        x_min, x_max = x_data.min(), x_data.max()
        margin = (x_max - x_min) * 0.1
        x_range = np.linspace(x_min - margin, x_max + margin, 50)
        
        # 繪製圖表
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pred = None
        
        # 條件性執行模型預測
        if include_model_prediction and api_url:
            # 獲取模型資訊
            xheaders, yheaders = get_mdc_headers_via_api(api_url, model_name, version, timeout)
            
            if not xheaders:
                LOGger.addlog("無法獲取模型資訊，跳過模型預測", logfile='', colora=LOGger.WARNING)
                include_model_prediction = False
            else:
                # 準備預測用的資料
                fixed_values = fixed_values if isinstance(fixed_values, dict) else {}
                predict_data = pd.DataFrame()
                
                for col in xheaders:
                    if col == xheader:
                        predict_data[col] = x_range
                    elif col in fixed_values:
                        # 使用指定的固定值，廣播到所有行
                        fixed_value = fixed_values[col]
                        predict_data[col] = [fixed_value] * len(x_range)
                    else:
                        if col in source_data.columns:
                            col_data = source_data[col].dropna()
                            if determineDataType(col_data) == 'continuous':
                                # 連續型用平均數，廣播到所有行
                                mean_value = col_data.mean()
                                predict_data[col] = [mean_value] * len(x_range)
                            else:
                                # 非連續型用眾數，廣播到所有行
                                mode_value = col_data.mode().iloc[0] if len(col_data.mode()) > 0 else col_data.iloc[0]
                                predict_data[col] = [mode_value] * len(x_range)
                        else:
                            predict_data[col] = [0] * len(x_range)  # 預設值，廣播到所有行
                
                # 執行預測
                p_npData = predict_via_api(api_url, model_name, version, predict_data, timeout)
                
                if p_npData is not None:
                    # 處理預測結果
                    if yheader in yheaders:
                        y_pred_index = yheaders.index(yheader)
                        if len(p_npData.shape) > 1 and p_npData.shape[1] > y_pred_index:
                            y_pred = p_npData[:, y_pred_index]
                        else:
                            y_pred = p_npData
                    else:
                        # 如果找不到對應的 yheader，使用第一個輸出
                        y_pred = p_npData[:, 0] if len(p_npData.shape) > 1 else p_npData
                    
                    # 繪製預測曲線
                    ax.plot(x_range, y_pred, 'b-', linewidth=2, label=f'{yheader} 預測曲線')
                else:
                    LOGger.addlog("預測失敗，跳過預測曲線繪製", logfile='', colora=LOGger.WARNING)
        
        # 繪製真實資料散點圖
        real_x = source_data[xheader].dropna()
        real_y = source_data[yheader].dropna()
        common_index = real_x.index.intersection(real_y.index)
        if len(common_index) > 0:
            ax.scatter(real_x[common_index], real_y[common_index], 
                    c='red', alpha=0.6, s=30, label='真實資料')
        
        ax.set_xlabel(xheader, fontproperties=vs.MJHfontprop())
        ax.set_ylabel(yheader, fontproperties=vs.MJHfontprop())
        ax.set_title(f'{xheader} 對 {yheader} 的單維度變化分析', fontproperties=vs.MJHfontprop())
        ax.legend(prop=vs.MJHfontprop())
        ax.grid(True, alpha=0.3)
        
        # 執行統計檢定並添加到圖表
        if 'statistical_tests' in kwags:
            addStatisticalInfoToPlot(ax, kwags['statistical_tests'], xheader, yheader)
        else:
            x_type = determineDataType(source_data[xheader])
            y_type = determineDataType(source_data[yheader])
            stats_ret = {}
            if performStatisticalTests(source_data, xheader, yheader, x_type, y_type, ret=stats_ret):
                addStatisticalInfoToPlot(ax, stats_ret['statistical_tests'], xheader, yheader)
        
        # 添加固定值 legend（左上角）
        if 'fixed_values_used' in kwags and kwags['fixed_values_used']:
            legend_text = "其他變數固定值:\n"
            for var_name, var_value in kwags['fixed_values_used'].items():
                if var_name != xheader:  # 排除分析變數本身
                    legend_text += f"{var_name}: {var_value}\n"
            
            # 在左上角添加文字框
            props = dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8)
            ax.text(0.02, 0.98, legend_text.strip(), transform=ax.transAxes, 
                    fontsize=9, verticalalignment='top', horizontalalignment='left',
                    bbox=props, fontproperties=vs3.vs.MJHfontprop())
        
        # 儲存圖片到專案資料夾
        output_filename = f'single_dimension_continuous_{xheader}_{yheader}.png'
        output_dir = output_dir if LOGger.isinstance_not_empty(output_dir, str) else 'analysis_output'
        if LOGger.isinstance_not_empty(output_dir, str):
            os.makedirs(output_dir, exist_ok=True)
            output_path = getProjectOutputPath(output_dir, output_filename)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 準備回傳資料
        if include_model_prediction and y_pred is not None:
            ret['model_output'] = {
                'xdata': x_range.tolist(),
                'pdata': y_pred.tolist() if isinstance(y_pred, np.ndarray) else [float(y_pred)]
            }
        else:
            ret['model_output'] = None
        ret['record_data'] = {
            'x_data': real_x[common_index].tolist() if len(common_index) > 0 else [],
            'y_data': real_y[common_index].tolist() if len(common_index) > 0 else []
        }
        ret['image_path'] = output_path if 'output_path' in locals() else None
        ret['analysis_type'] = 'continuous_to_continuous'
        
        return True
        
    except Exception as e:
        LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
        ret['msg'] = f'analyzeContinuousToContinuous error: {str(e)}'
        return False

def analyzeCategoricalToContinuous(
        source_data: pd.DataFrame,
        xheader: str,
        yheader: str,
        fixed_values: Optional[Dict[str, Any]] = None,
        include_model_prediction: bool = True,
        ret: Optional[Dict[str, Any]] = None,
        api_url: Optional[str] = None,
        model_name: str = "ACAngle",
        version: str = "v0-0-2-0",
        output_dir: Optional[str] = None,
        timeout: int = 30,
        **kwags
    ) -> bool:
    """
    非連續型x對連續型y的分析（使用 API 調用 MDC 模型）
    
    Parameters
    ----------
    source_data : pd.DataFrame
        原始資料
    xheader : str
        x軸header名稱
    yheader : str
        y軸header名稱
    fixed_values : dict, optional
        指定其他X因子的固定值
    include_model_prediction : bool
        是否包含模型預測
    ret : dict, optional
        回傳結果字典
    api_url : str, optional
        API 服務器 URL，如果為 None 則不執行模型預測
    model_name : str
        模型名稱
    version : str
        模型版本
    output_dir : str, optional
        輸出目錄
    timeout : int
        API 請求超時時間（秒）
    **kwags
        其他參數
        
    Returns
    -------
    bool
        是否成功分析
    """
    ret = ret if isinstance(ret, dict) else {}
    
    try:
        # 檢查是否需要 API
        if include_model_prediction and api_url is None:
            LOGger.addlog("警告：未提供 api_url，跳過模型預測", logfile='', colora=LOGger.WARNING)
            include_model_prediction = False
        
        # 設定中文字型
        vs3.plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
        
        # 取得x的唯一值
        x_categories = source_data[xheader].dropna().unique()
        n_categories = len(x_categories)
        
        # 使用vs3.cm_rainbar分配顏色
        colors = vs3.vs2.cm_rainbar(n_categories, c_alpha=0.3)
        
        # 取得y的範圍用於繪製分布圖
        y_data = source_data[yheader].dropna()
        y_min, y_max = y_data.min(), y_data.max()
        
        # 繪製圖表
        fig, ax = vs3.plt.subplots(figsize=(12, 8))
        
        # 準備回傳資料
        if include_model_prediction:
            ret['model_output'] = {'xdata': [], 'pdata': []}
        else:
            ret['model_output'] = None
        ret['record_data'] = {}
        
        # 獲取模型資訊（如果需要預測）
        xheaders, yheaders = [], []
        if include_model_prediction and api_url:
            xheaders, yheaders = get_mdc_headers_via_api(api_url, model_name, version, timeout)
            if not xheaders:
                LOGger.addlog("無法獲取模型資訊，跳過模型預測", logfile='', colora=LOGger.WARNING)
                include_model_prediction = False
        
        for i, category in enumerate(x_categories):
            # 取得該類別的y值分布
            category_y = source_data[source_data[xheader] == category][yheader].dropna()
            
            if len(category_y) > 0:
                # 繪製分布圖（直方圖）
                color = colors[i] if i < len(colors) else colors[0]
                ax.hist(category_y, bins=20, alpha=0.3, color=color[:3], 
                    label=f'{category} 分布', orientation='horizontal')
                
                # 條件性執行模型預測
                if include_model_prediction and api_url and xheaders:
                    # 準備預測用的資料
                    fixed_values = fixed_values if isinstance(fixed_values, dict) else {}
                    predict_data = pd.DataFrame()
                    for col in xheaders:
                        if col == xheader:
                            predict_data[col] = [category]
                        elif col in fixed_values:
                            predict_data[col] = [fixed_values[col]]
                        else:
                            if col in source_data.columns:
                                col_data = source_data[col].dropna()
                                if determineDataType(col_data) == 'continuous':
                                    predict_data[col] = [col_data.mean()]
                                else:
                                    mode_value = col_data.mode().iloc[0] if len(col_data.mode()) > 0 else col_data.iloc[0]
                                    predict_data[col] = [mode_value]
                            else:
                                predict_data[col] = [0]
                    
                    # 執行預測
                    p_npData = predict_via_api(api_url, model_name, version, predict_data, timeout)
                    
                    if p_npData is not None:
                        if yheader in yheaders:
                            y_pred_index = yheaders.index(yheader)
                            if len(p_npData.shape) > 1 and p_npData.shape[1] > y_pred_index:
                                y_pred = p_npData[0, y_pred_index]
                            else:
                                y_pred = p_npData[0]
                        else:
                            y_pred = p_npData[0] if len(p_npData.shape) > 1 else p_npData
                        
                        # 繪製預測線
                        ax.axhline(y=y_pred, color=color[:3], linestyle='--', linewidth=2, 
                            label=f'{category} 預測值')
                        
                        # 儲存模型預測資料
                        ret['model_output']['xdata'].append(str(category))
                        ret['model_output']['pdata'].append(float(y_pred))
                
                ret['record_data'][f'category_{i+1}'] = {
                    'value': str(category),
                    'y_data': category_y.tolist()
                }
        
        ax.set_ylabel(yheader, fontproperties=vs.MJHfontprop())
        ax.set_xlabel('頻率', fontproperties=vs.MJHfontprop())
        ax.set_title(f'{xheader} 對 {yheader} 的分類分析', fontproperties=vs.MJHfontprop())
        ax.legend(prop=vs.MJHfontprop())
        ax.grid(True, alpha=0.3)
        
        # 執行統計檢定並添加到圖表
        if 'statistical_tests' in kwags:
            addStatisticalInfoToPlot(ax, kwags['statistical_tests'], xheader, yheader)
        else:
            x_type = determineDataType(source_data[xheader])
            y_type = determineDataType(source_data[yheader])
            stats_ret = {}
            if performStatisticalTests(source_data, xheader, yheader, x_type, y_type, ret=stats_ret):
                addStatisticalInfoToPlot(ax, stats_ret['statistical_tests'], xheader, yheader)
        
        # 添加固定值 legend（左上角）
        if 'fixed_values_used' in kwags and kwags['fixed_values_used']:
            legend_text = "其他變數固定值:\n"
            for var_name, var_value in kwags['fixed_values_used'].items():
                if var_name != xheader:
                    legend_text += f"{var_name}: {var_value}\n"
            
            props = dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8)
            ax.text(0.02, 0.98, legend_text.strip(), transform=ax.transAxes, 
                    fontsize=9, verticalalignment='top', horizontalalignment='left',
                    bbox=props, fontproperties=vs3.vs.MJHfontprop())
        
        # 儲存圖片到專案資料夾
        output_filename = f'single_dimension_categorical_to_continuous_{xheader}_{yheader}.png'
        output_dir = output_dir if LOGger.isinstance_not_empty(output_dir, str) else 'analysis_output'
        if LOGger.isinstance_not_empty(output_dir, str):
            os.makedirs(output_dir, exist_ok=True)
            output_path = getProjectOutputPath(output_dir, output_filename)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        ret['image_path'] = output_path if 'output_path' in locals() else None
        ret['analysis_type'] = 'categorical_to_continuous'
        
        return True
        
    except Exception as e:
        LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
        ret['msg'] = f'analyzeCategoricalToContinuous error: {str(e)}'
        return False

def analyzeCategoricalToCategorical(
        source_data: pd.DataFrame,
        xheader: str,
        yheader: str,
        fixed_values: Optional[Dict[str, Any]] = None,
        ret: Optional[Dict[str, Any]] = None,
        api_url: Optional[str] = None,
        model_name: str = "ACAngle",
        version: str = "v0-0-2-0",
        output_dir: Optional[str] = None,
        timeout: int = 30,
        **kwags
    ) -> bool:
    """
    非連續型x對非連續型y的分析（使用 API 調用 MDC 模型）
    """
    ret = ret if isinstance(ret, dict) else {}
    
    try:
        # 設定中文字型
        vs3.plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
        
        # 建立交叉表分析
        import seaborn as sns
        crosstab = pd.crosstab(source_data[xheader], source_data[yheader], margins=True)
        
        # 繪製熱力圖
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 繪製計數熱力圖
        sns.heatmap(crosstab.iloc[:-1, :-1], annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title(f'{xheader} vs {yheader} 交叉分析 (計數)', fontproperties=vs.MJHfontprop())
        ax1.set_xlabel(yheader, fontproperties=vs.MJHfontprop())
        ax1.set_ylabel(xheader, fontproperties=vs.MJHfontprop())
        
        # 計算比例並繪製比例熱力圖
        crosstab_pct = crosstab.iloc[:-1, :-1].div(crosstab.iloc[:-1, :-1].sum(axis=1), axis=0) * 100
        sns.heatmap(crosstab_pct, annot=True, fmt='.1f', cmap='Oranges', ax=ax2)
        ax2.set_title(f'{xheader} vs {yheader} 交叉分析 (百分比)', fontproperties=vs.MJHfontprop())
        ax2.set_xlabel(yheader, fontproperties=vs.MJHfontprop())
        ax2.set_ylabel(xheader, fontproperties=vs.MJHfontprop())
        
        # 進行預測分析
        x_categories = source_data[xheader].dropna().unique()
        ret['model_output'] = {'xdata': [], 'pdata': []}
        ret['record_data'] = {}
        
        # 獲取模型資訊（如果需要預測）
        xheaders, yheaders = [], []
        if api_url:
            xheaders, yheaders = get_mdc_headers_via_api(api_url, model_name, version, timeout)
        
        for i, x_category in enumerate(x_categories):
            y_pred = None
            # 準備預測用的資料
            if api_url and xheaders:
                fixed_values = fixed_values if isinstance(fixed_values, dict) else {}
                predict_data = pd.DataFrame()
                for col in xheaders:
                    if col == xheader:
                        predict_data[col] = [x_category]
                    elif col in fixed_values:
                        predict_data[col] = [fixed_values[col]]
                    else:
                        if col in source_data.columns:
                            col_data = source_data[col].dropna()
                            if determineDataType(col_data) == 'continuous':
                                predict_data[col] = [col_data.mean()]
                            else:
                                mode_value = col_data.mode().iloc[0] if len(col_data.mode()) > 0 else col_data.iloc[0]
                                predict_data[col] = [mode_value]
                        else:
                            predict_data[col] = [0]
                
                # 執行預測
                p_npData = predict_via_api(api_url, model_name, version, predict_data, timeout)
                if p_npData is not None:
                    if yheader in yheaders:
                        y_pred_index = yheaders.index(yheader)
                        if len(p_npData.shape) > 1 and p_npData.shape[1] > y_pred_index:
                            y_pred = p_npData[0, y_pred_index]
                        else:
                            y_pred = p_npData[0]
                    else:
                        y_pred = p_npData[0] if len(p_npData.shape) > 1 else p_npData
                    
                    ret['model_output']['xdata'].append(str(x_category))
                    ret['model_output']['pdata'].append(str(y_pred))
            
            # 取得實際的y值分布
            actual_y = source_data[source_data[xheader] == x_category][yheader].dropna()
            ret['record_data'][f'category_{i+1}'] = {
                'x_value': str(x_category),
                'y_predicted': str(y_pred) if y_pred is not None else None,
                'y_actual_distribution': actual_y.value_counts().to_dict()
            }
        
        fig.tight_layout()
        
        # 執行統計檢定
        if 'statistical_tests' in kwags:
            addStatisticalInfoToPlot(ax1, kwags['statistical_tests'], xheader, yheader)
        else:
            x_type = determineDataType(source_data[xheader])
            y_type = determineDataType(source_data[yheader])
            stats_ret = {}
            if performStatisticalTests(source_data, xheader, yheader, x_type, y_type, ret=stats_ret):
                addStatisticalInfoToPlot(ax1, stats_ret['statistical_tests'], xheader, yheader)
        
        # 儲存圖片
        output_filename = f'single_dimension_categorical_to_categorical_{xheader}_{yheader}.png'
        output_dir = output_dir if LOGger.isinstance_not_empty(output_dir, str) else 'analysis_output'
        if LOGger.isinstance_not_empty(output_dir, str):
            os.makedirs(output_dir, exist_ok=True)
            output_path = getProjectOutputPath(output_dir, output_filename)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        ret['image_path'] = output_path if 'output_path' in locals() else None
        ret['analysis_type'] = 'categorical_to_categorical'
        
        return True
        
    except Exception as e:
        LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
        ret['msg'] = f'analyzeCategoricalToCategorical error: {str(e)}'
        return False

def twoDimensionAnalysis(
        source_data: pd.DataFrame,
        xheader1: str,
        xheader2: str,
        yheader: str,
        fixed_values: Optional[Dict[str, Any]] = None,
        include_model_prediction: bool = True,
        ret: Optional[Dict[str, Any]] = None,
        api_url: Optional[str] = None,
        model_name: str = "ACAngle",
        version: str = "v0-0-2-0",
        output_dir: Optional[str] = None,
        timeout: int = 30,
        **kwags
    ) -> bool:
    """
    二維度分析（使用 API 調用 MDC 模型）
    自動判斷資料類型並調用對應的分析函數
    """
    ret = ret if isinstance(ret, dict) else {}
    
    try:
        # 判斷資料類型
        x1_type = determineDataType(source_data[xheader1])
        x2_type = determineDataType(source_data[xheader2])
        y_type = determineDataType(source_data[yheader])
        
        LOGger.addlog(f'Two-dimension analysis types: {xheader1}({x1_type}) + {xheader2}({x2_type}) -> {yheader}({y_type})', 
            logfile='', colora=LOGger.OKCYAN)
        
        # 記錄使用的固定值
        fixed_values = fixed_values if isinstance(fixed_values, dict) else {}
        ret['fixed_values_used'] = fixed_values.copy()
        
        # 根據資料類型選擇分析方法
        if x1_type == 'continuous' and x2_type == 'continuous':
            success = analyzeTwoContinuousToY(source_data, xheader1, xheader2, yheader, y_type, 
                fixed_values=fixed_values, include_model_prediction=include_model_prediction, 
                ret=ret, api_url=api_url, model_name=model_name, version=version, 
                output_dir=output_dir, timeout=timeout, **kwags)
        elif (x1_type == 'categorical' and x2_type == 'continuous') or (x1_type == 'continuous' and x2_type == 'categorical'):
            success = analyzeMixedToY(source_data, xheader1, xheader2, yheader, x1_type, x2_type, y_type,
                fixed_values=fixed_values, include_model_prediction=include_model_prediction,
                ret=ret, api_url=api_url, model_name=model_name, version=version,
                output_dir=output_dir, timeout=timeout, **kwags)
        elif x1_type == 'categorical' and x2_type == 'categorical':
            success = analyzeTwoCategoricalToY(source_data, xheader1, xheader2, yheader, y_type,
                fixed_values=fixed_values, include_model_prediction=include_model_prediction,
                ret=ret, api_url=api_url, model_name=model_name, version=version,
                output_dir=output_dir, timeout=timeout, **kwags)
        else:
            ret['msg'] = f'Unsupported data type combination: {x1_type} + {x2_type} -> {y_type}'
            return False
        
        if success:
            ret['xheader1'] = xheader1
            ret['xheader2'] = xheader2
            ret['yheader'] = yheader
            ret['x1_type'] = x1_type
            ret['x2_type'] = x2_type
            ret['y_type'] = y_type
            ret['analysis_type'] = f'{x1_type}_{x2_type}_to_{y_type}'
        
        return success
        
    except Exception as e:
        LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
        ret['msg'] = f'twoDimensionAnalysis error: {str(e)}'
        return False

def map_source_headers_to_model_headers(
        xheader1: str,
        xheader2: str,
        xheaders: List[str]
    ) -> Dict[str, str]:
    """
    將 source_data 的欄位名稱對應到模型的輸入欄位名稱
    
    此函數處理兩種情況：
    1. 實際欄位名稱直接匹配的情況（例如 compare_and_analyze.py：x1='waxAngle', x2='SpecLoft'）
    2. 測試數據中欄位被重命名為 X1, X2 的情況（x1='X1', x2='X2'，但 xheaders=['waxAngle', 'SpecLoft']）
    
    參數:
        xheader1: source_data 的第一個 x 欄位名稱
        xheader2: source_data 的第二個 x 欄位名稱
        xheaders: 模型的輸入欄位名稱列表
    
    返回:
        dict: 包含 'model_x1_header' 和 'model_x2_header' 的字典
    """
    # 先確定哪些 xheaders 已經被 xheader1 或 xheader2 使用
    used_headers = set()
    
    # 確定 model_x1_header
    if xheader1 in xheaders:
        # 如果 xheader1 在 xheaders 中，直接使用（這是 compare_and_analyze.py 的正常情況）
        model_x1_header = xheader1
        used_headers.add(xheader1)
    else:
        # 如果 xheader1 不在 xheaders 中（例如是重命名後的 'X1'），使用第一個未使用的 xheader
        available_headers = [h for h in xheaders if h not in used_headers]
        model_x1_header = available_headers[0] if len(available_headers) > 0 else xheader1
        if model_x1_header in xheaders:
            used_headers.add(model_x1_header)
    
    # 確定 model_x2_header
    if xheader2 in xheaders:
        # 如果 xheader2 在 xheaders 中，直接使用（這是 compare_and_analyze.py 的正常情況）
        model_x2_header = xheader2
        used_headers.add(xheader2)
    else:
        # 如果 xheader2 不在 xheaders 中（例如是重命名後的 'X2'），使用第一個未使用的 xheader
        available_headers = [h for h in xheaders if h not in used_headers]
        model_x2_header = available_headers[0] if len(available_headers) > 0 else xheader2
        if model_x2_header in xheaders:
            used_headers.add(model_x2_header)
    
    return {
        'model_x1_header': model_x1_header,
        'model_x2_header': model_x2_header
    }

def validate_scatter_surface_consistency(
        scatter_x1: np.ndarray,
        scatter_x2: np.ndarray,
        scatter_y: np.ndarray,
        surface_x1_range: np.ndarray,
        surface_x2_range: np.ndarray,
        surface_Z: np.ndarray,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
    """
    驗證 scatter 點與 surface 的一致性
    
    參數:
        scatter_x1: scatter 點的 X1 座標
        scatter_x2: scatter 點的 X2 座標
        scatter_y: scatter 點的 Y 值（真實值）
        surface_x1_range: surface 的 X1 網格範圍
        surface_x2_range: surface 的 X2 網格範圍
        surface_Z: surface 的 Z 值（預測值），形狀為 (len(surface_x2_range), len(surface_x1_range))
        threshold: 差異門檻（預設 None，會自動使用 scatter_y 的 1 倍標準差）。如果指定數值，則使用該值
    
    返回:
        dict: 包含驗證結果的字典
            - exceeded_count: 超出門檻的點數
            - threshold: 使用的門檻值
            - max_diff: 最大差異
            - mean_diff: 平均差異
            - diffs: 所有點的差異列表
    """
    from scipy.interpolate import griddata
    
    # 如果 threshold 為 None，使用 scatter_y 的 1 倍標準差
    if threshold is None:
        threshold = float(np.std(scatter_y))
    
    # 準備 surface 的網格點和對應的 Z 值
    # surface_Z 的形狀是 (len(surface_x2_range), len(surface_x1_range))
    # 需要轉換為 (N, 3) 的格式：[(x1, x2, z), ...]
    X1_grid, X2_grid = np.meshgrid(surface_x1_range, surface_x2_range)
    surface_points = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])
    surface_values = surface_Z.ravel()
    
    # 使用 griddata 對 scatter 點進行插值，獲取對應的 surface Z 值
    scatter_points = np.column_stack([scatter_x1, scatter_x2])
    surface_y_interpolated = griddata(
        surface_points,
        surface_values,
        scatter_points,
        method='linear',
        fill_value=np.nan
    )
    
    # 確保輸入為數值類型
    scatter_y = np.asarray(scatter_y, dtype=np.float64)
    surface_y_interpolated = np.asarray(surface_y_interpolated, dtype=np.float64)
    
    # 計算差異（絕對值）
    diffs = np.abs(scatter_y - surface_y_interpolated)
    
    # 過濾掉 NaN 值（超出插值範圍的點）
    valid_mask = ~np.isnan(diffs)
    valid_diffs = diffs[valid_mask]
    
    # 統計結果
    exceeded_mask = valid_diffs > threshold
    exceeded_count = np.sum(exceeded_mask)
    
    result = {
        'exceeded_count': int(exceeded_count),
        'threshold': threshold,
        'max_diff': float(np.max(valid_diffs)) if len(valid_diffs) > 0 else 0.0,
        'mean_diff': float(np.mean(valid_diffs)) if len(valid_diffs) > 0 else 0.0,
        'diffs': valid_diffs.tolist(),
        'total_points': len(scatter_y),
        'valid_points': int(np.sum(valid_mask))
    }
    
    return result

def analyzeTwoContinuousToY(
        source_data: pd.DataFrame,
        xheader1: str,
        xheader2: str,
        yheader: str,
        y_type: str,
        fixed_values: Optional[Dict[str, Any]] = None,
        include_model_prediction: bool = True,
        ret: Optional[Dict[str, Any]] = None,
        api_url: Optional[str] = None,
        model_name: str = "ACAngle",
        version: str = "v0-0-2-0",
        output_dir: Optional[str] = None,
        timeout: int = 30,
        **kwags
    ) -> bool:
    """
    兩個連續型變數對Y的分析（響應面分析，使用 API 調用 MDC 模型）
    """
    ret = ret if isinstance(ret, dict) else {}
    
    try:
        # 如果未提供 api_url，嘗試從 config.json 讀取
        if api_url is None:
            api_url = get_default_api_url()
        
        # 檢查是否需要 API
        if include_model_prediction and api_url is None:
            LOGger.addlog("警告：未提供 api_url，跳過模型預測", logfile='', colora=LOGger.WARNING)
            include_model_prediction = False
        
        # 設定中文字型
        vs3.plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
        
        # 取得x1和x2的範圍
        x1_data = source_data[xheader1].dropna()
        x2_data = source_data[xheader2].dropna()
        y_data = source_data[yheader].dropna()
        
        x1_min, x1_max = x1_data.min(), x1_data.max()
        x2_min, x2_max = x2_data.min(), x2_data.max()
        
        # 繪製3D響應面圖
        fig = plt.figure(figsize=(15, 10))
        
        # 獲取模型資訊（如果需要預測）
        xheaders, yheaders = [], []
        fixed_headers = [x for x in xheaders if x not in [xheader1, xheader2]]
        if include_model_prediction and api_url:
            xheaders, yheaders = get_mdc_headers_via_api(api_url, model_name, version, timeout)
            if not xheaders:
                LOGger.addlog("無法獲取模型資訊，跳過模型預測", logfile='', colora=LOGger.WARNING)
                include_model_prediction = False
            if not xheader1 in xheaders or not xheader2 in xheaders:
                LOGger.addlog("xheader1 或 xheader2 不在模型資訊中，跳過模型預測", logfile='', colora=LOGger.WARNING)
                include_model_prediction = False
            if fixed_headers:
                fixed_assignments_set = set(fixed_values.keys())
                if not set(fixed_headers).issubset(fixed_assignments_set):
                    print(LOGger.FAIL + f"固定欄位無校：\nfixed_headers: {fixed_headers}\nfixed_values: {fixed_values}")
                include_model_prediction = False
        
        # 條件性執行模型預測
        Z = None
        x1_range = None
        x2_range = None
        if include_model_prediction and api_url and xheaders:
            # 建立網格
            x1_margin = (x1_max - x1_min) * 0.1
            x2_margin = (x2_max - x2_min) * 0.1
            x1_range = np.linspace(x1_min - x1_margin, x1_max + x1_margin, 20)
            x2_range = np.linspace(x2_min - x2_margin, x2_max + x2_margin, 20)
            X1, X2 = np.meshgrid(x1_range, x2_range)

            input_frame = pd.DataFrame(np.column_stack([X1.ravel(), X2.ravel()]), columns=[xheader1, xheader2])
            for x in fixed_headers:
                input_frame[x] = fixed_values[x]
            input_frame = input_frame[xheaders]
            
            # 執行預測
            p_npData = predict_via_api(api_url, model_name, version, input_frame, timeout)
            
            if p_npData is not None:
                if yheader in yheaders:
                    y_pred_index = yheaders.index(yheader)
                    if len(p_npData.shape) > 1 and p_npData.shape[1] > y_pred_index:
                        y_pred = p_npData[:, y_pred_index]
                    else:
                        y_pred = p_npData
                else:
                    y_pred = p_npData[:, 0] if len(p_npData.shape) > 1 else p_npData
                
                # 重塑為網格形狀
                # 注意：循環順序是 for i in x1_range, for j in x2_range
                # 所以 y_pred 的順序是 [x1[0],x2[0]], [x1[0],x2[1]], ..., [x1[0],x2[n]], [x1[1],x2[0]], ...
                # 這需要 reshape 為 (len(x1_range), len(x2_range))，然後轉置以匹配 meshgrid 的順序
                # meshgrid 返回的網格第一個維度是 x2_range，第二個維度是 x1_range
                # Z = y_pred.reshape(len(x1_range), len(x2_range)).T

                Z = y_pred.reshape(len(x1_range), len(x2_range))
                
                # 驗證 scatter 點與 surface 的一致性
                real_data = source_data[[xheader1, xheader2, yheader]].dropna()
                validation_result = None
                if len(real_data) > 0:
                    # 門檻預設為 None，會自動使用 y 資料的 1 倍標準差
                    validation_result = validate_scatter_surface_consistency(
                        real_data[xheader1].values,
                        real_data[xheader2].values,
                        real_data[yheader].values,
                        x1_range,
                        x2_range,
                        Z
                        # threshold 預設為 None，會自動使用 y 資料的 1 倍標準差
                    )
                    # 將驗證結果記錄到 ret 字典中
                    if 'model_output' not in ret:
                        ret['model_output'] = {}
                    ret['model_output']['scatter_surface_validation'] = validation_result
                    
                    if validation_result['exceeded_count'] > 0:
                        LOGger.addlog(
                            f"警告：有 {validation_result['exceeded_count']} 個數據點與響應面的差異超過門檻 {validation_result['threshold']:.2f} (y 資料的 1 倍標準差)",
                            logfile='',
                            colora=LOGger.WARNING
                        )
                        LOGger.addlog(
                            f"最大差異：{validation_result['max_diff']:.4f}，平均差異：{validation_result['mean_diff']:.4f}",
                            logfile='',
                            colora=LOGger.WARNING
                        )
                
                # 3D 響應面
                ax1 = fig.add_subplot(221, projection='3d')
                surf = ax1.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
                
                # 添加真實資料點（real_data 已在驗證函數中定義）
                if len(real_data) > 0:
                    ax1.scatter(real_data[xheader1], real_data[xheader2], real_data[yheader], 
                            c='red', s=20, alpha=0.6, label='真實資料')
                
                ax1.set_xlabel(xheader1, fontproperties=vs.MJHfontprop())
                ax1.set_ylabel(xheader2, fontproperties=vs.MJHfontprop())
                ax1.set_zlabel(yheader, fontproperties=vs.MJHfontprop())
                ax1.set_title(f'{xheader1} + {xheader2} 對 {yheader} 的響應面分析', fontproperties=vs.MJHfontprop())
                fig.colorbar(surf, ax=ax1, shrink=0.5)
                
                # 2D 等高線圖
                ax2 = fig.add_subplot(222)
                # 注意：contour(X, Y, Z) 中 X 是 x 軸，Y 是 y 軸
                # meshgrid(x1_range, x2_range) 返回：
                #   X1[i,j] = x1_range[j] (x 軸，對應 xheader1)
                #   X2[i,j] = x2_range[i] (y 軸，對應 xheader2)
                # Z[i,j] 對應 (x1_range[j], x2_range[i])，即 (X1[i,j], X2[i,j])
                # 所以 contour(X1, X2, Z) 應該是正確的
                contour = ax2.contour(X1, X2, Z, levels=15)
                ax2.clabel(contour, inline=True, fontsize=8)
                contourf = ax2.contourf(X1, X2, Z, levels=15, alpha=0.6, cmap='viridis')
                
                # 添加真實資料點到等高線圖
                if len(real_data) > 0:
                    ax2.scatter(real_data[xheader1], real_data[xheader2], 
                            c=real_data[yheader], s=30, cmap='viridis', 
                            edgecolors='black', alpha=0.8, label='真實資料')
                
                ax2.set_xlabel(xheader1, fontproperties=vs.MJHfontprop())
                ax2.set_ylabel(xheader2, fontproperties=vs.MJHfontprop())
                ax2.set_title(f'{yheader} 等高線圖', fontproperties=vs.MJHfontprop())
                fig.colorbar(contourf, ax=ax2)
                
                # X2 固定時的切片分析
                ax3 = fig.add_subplot(223)
                mid_x2_idx = len(x2_range) // 2
                x2_fixed_value = x2_range[mid_x2_idx]
                ax3.plot(x1_range, Z[mid_x2_idx, :], 'b-', linewidth=2, 
                        label=f'{xheader2}={x2_fixed_value:.2f}時的預測')
                
                # 添加對應的真實資料
                margin = (x2_max - x2_min) * 0.1
                real_slice1 = real_data[(real_data[xheader2] < x2_max + margin) &
                                        (real_data[xheader2] > x2_min - margin)]
                if len(real_slice1) > 0:
                    ax3.scatter(real_slice1[xheader1], real_slice1[yheader], 
                            c='red', s=30, alpha=0.6, label='真實資料')
                
                ax3.set_xlabel(xheader1, fontproperties=vs.MJHfontprop())
                ax3.set_ylabel(yheader, fontproperties=vs.MJHfontprop())
                ax3.set_title(f'固定 {xheader2} 的切片分析', fontproperties=vs.MJHfontprop())
                ax3.legend(prop=vs.MJHfontprop())
                ax3.grid(True, alpha=0.3)
                
                # X1 固定時的切片分析
                ax4 = fig.add_subplot(224)
                mid_x1_idx = len(x1_range) // 2
                x1_fixed_value = x1_range[mid_x1_idx]
                ax4.plot(x2_range, Z[:, mid_x1_idx], 'g-', linewidth=2, 
                        label=f'{xheader1}={x1_fixed_value:.2f}時的預測')
                
                # 添加對應的真實資料
                margin = (x1_max - x1_min) * 0.1
                real_slice2 = real_data[(real_data[xheader1] < x1_max + margin) &
                                        (real_data[xheader1] > x1_min - margin)]
                if len(real_slice2) > 0:
                    ax4.scatter(real_slice2[xheader2], real_slice2[yheader], 
                            c='red', s=30, alpha=0.6, label='真實資料')
                
                ax4.set_xlabel(xheader2, fontproperties=vs.MJHfontprop())
                ax4.set_ylabel(yheader, fontproperties=vs.MJHfontprop())
                ax4.set_title(f'固定 {xheader1} 的切片分析', fontproperties=vs.MJHfontprop())
                ax4.legend(prop=vs.MJHfontprop())
                ax4.grid(True, alpha=0.3)
                
                # 準備回傳資料
                ret['model_output'] = {
                    'x1_range': x1_range.tolist(),
                    'x2_range': x2_range.tolist(),
                    'z_data': Z.tolist()
                }
            else:
                include_model_prediction = False
        
        if not include_model_prediction or Z is None:
            # 不包含模型預測時，只顯示真實資料的散點圖
            real_data = source_data[[xheader1, xheader2, yheader]].dropna()
            
            ax1 = fig.add_subplot(221, projection='3d')
            if len(real_data) > 0:
                ax1.scatter(real_data[xheader1], real_data[xheader2], real_data[yheader], 
                        c='blue', s=30, alpha=0.7, label='真實資料')
            
            ax1.set_xlabel(xheader1, fontproperties=vs.MJHfontprop())
            ax1.set_ylabel(xheader2, fontproperties=vs.MJHfontprop())
            ax1.set_zlabel(yheader, fontproperties=vs.MJHfontprop())
            ax1.set_title(f'{xheader1} + {xheader2} 對 {yheader} 的真實資料分布', fontproperties=vs.MJHfontprop())
            
            # 2D 散點圖
            ax2 = fig.add_subplot(222)
            if len(real_data) > 0:
                scatter = ax2.scatter(real_data[xheader1], real_data[xheader2], 
                                    c=real_data[yheader], cmap='viridis', alpha=0.7)
                fig.colorbar(scatter, ax=ax2)
            
            ax2.set_xlabel(xheader1, fontproperties=vs.MJHfontprop())
            ax2.set_ylabel(xheader2, fontproperties=vs.MJHfontprop())
            ax2.set_title(f'{yheader} 真實資料分布', fontproperties=vs.MJHfontprop())
            
            # X1 vs Y 散點圖
            ax3 = fig.add_subplot(223)
            if len(real_data) > 0:
                ax3.scatter(real_data[xheader1], real_data[yheader], 
                        c='blue', s=30, alpha=0.7, label='真實資料')
            ax3.set_xlabel(xheader1, fontproperties=vs.MJHfontprop())
            ax3.set_ylabel(yheader, fontproperties=vs.MJHfontprop())
            ax3.set_title(f'{xheader1} 對 {yheader}', fontproperties=vs.MJHfontprop())
            ax3.grid(True, alpha=0.3)
            
            # X2 vs Y 散點圖
            ax4 = fig.add_subplot(224)
            if len(real_data) > 0:
                ax4.scatter(real_data[xheader2], real_data[yheader], 
                        c='green', s=30, alpha=0.7, label='真實資料')
            ax4.set_xlabel(xheader2, fontproperties=vs.MJHfontprop())
            ax4.set_ylabel(yheader, fontproperties=vs.MJHfontprop())
            ax4.set_title(f'{xheader2} 對 {yheader}', fontproperties=vs.MJHfontprop())
            ax4.grid(True, alpha=0.3)
            
            ret['model_output'] = None
        
        # 添加固定值 legend
        if 'fixed_values_used' in kwags and kwags['fixed_values_used']:
            legend_text = "其他變數固定值:\n"
            for var_name, var_value in kwags['fixed_values_used'].items():
                if var_name not in [xheader1, xheader2]:
                    legend_text += f"{var_name}: {var_value}\n"
            
            props = dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8)
            fig.text(0.02, 0.98, legend_text.strip(), 
                    fontsize=9, verticalalignment='top', horizontalalignment='left',
                    bbox=props, fontproperties=vs3.vs.MJHfontprop())
        
        # 儲存圖片
        output_filename = f'two_dimension_continuous_{xheader1}_{xheader2}_{yheader}.png'
        output_dir = output_dir if LOGger.isinstance_not_empty(output_dir, str) else 'analysis_output'
        if LOGger.isinstance_not_empty(output_dir, str):
            os.makedirs(output_dir, exist_ok=True)
            output_path = getProjectOutputPath(output_dir, output_filename)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        ret['image_path'] = output_path if 'output_path' in locals() else None
        ret['analysis_type'] = 'two_continuous_to_y'
        ret['record_data'] = {
            'x1_data': x1_data.tolist(),
            'x2_data': x2_data.tolist(),
            'y_data': y_data.tolist()
        }
        
        return True
        
    except Exception as e:
        LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
        ret['msg'] = f'analyzeTwoContinuousToY error: {str(e)}'
        return False

def analyzeMixedToY(
        source_data: pd.DataFrame,
        xheader1: str,
        xheader2: str,
        yheader: str,
        x1_type: str,
        x2_type: str,
        y_type: str,
        fixed_values: Optional[Dict[str, Any]] = None,
        include_model_prediction: bool = True,
        ret: Optional[Dict[str, Any]] = None,
        api_url: Optional[str] = None,
        model_name: str = "ACAngle",
        version: str = "v0-0-2-0",
        output_dir: Optional[str] = None,
        timeout: int = 30,
        **kwags
    ) -> bool:
    """
    混合型變數對Y的分析（一個連續型 + 一個類別型 -> Y，使用 API 調用 MDC 模型）
    注意：此函數目前為簡化實現，完整實現需要參考 data_analysis.py
    """
    ret = ret if isinstance(ret, dict) else {}
    
    try:
        # 如果未提供 api_url，嘗試從 config.json 讀取
        if api_url is None:
            api_url = get_default_api_url()
        
        # 檢查是否需要 API
        if include_model_prediction and api_url is None:
            LOGger.addlog("警告：未提供 api_url，跳過模型預測", logfile='', colora=LOGger.WARNING)
            include_model_prediction = False
        
        # 設定中文字型
        vs3.plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
        
        # 確定哪個是連續型，哪個是類別型
        continuous_header = xheader1 if x1_type == 'continuous' else xheader2
        categorical_header = xheader2 if x1_type == 'continuous' else xheader1
        
        # 取得類別的唯一值
        categories = source_data[categorical_header].dropna().unique()
        
        # 繪製圖表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{xheader1} + {xheader2} 對 {yheader} 的混合分析', fontproperties=vs.MJHfontprop())
        
        # 獲取模型資訊（如果需要預測）
        xheaders, yheaders = [], []
        fixed_headers = []
        if include_model_prediction and api_url:
            xheaders, yheaders = get_mdc_headers_via_api(api_url, model_name, version, timeout)
            if not xheaders:
                LOGger.addlog("無法獲取模型資訊，跳過模型預測", logfile='', colora=LOGger.WARNING)
                include_model_prediction = False
            if not xheader1 in xheaders or not xheader2 in xheaders:
                LOGger.addlog("xheader1 或 xheader2 不在模型資訊中，跳過模型預測", logfile='', colora=LOGger.WARNING)
                include_model_prediction = False
            fixed_headers = [x for x in xheaders if x not in [xheader1, xheader2]]
            if fixed_headers:
                fixed_assignments_set = set(fixed_values.keys()) if fixed_values else set()
                if not set(fixed_headers).issubset(fixed_assignments_set):
                    print(LOGger.FAIL + f"固定欄位無校：\nfixed_headers: {fixed_headers}\nfixed_values: {fixed_values}")
                    include_model_prediction = False
            if include_model_prediction:
                # 建立模型輸入欄位名稱與 source_data 欄位名稱的對應關係
                # 即使目前沒有實現模型預測的響應面繪製，也先建立對應關係以備將來使用
                header_mapping = map_source_headers_to_model_headers(xheader1, xheader2, xheaders)
                model_x1_header = header_mapping['model_x1_header']
                model_x2_header = header_mapping['model_x2_header']
                
                # 記錄對應關係（用於調試）
                if model_x1_header != xheader1 or model_x2_header != xheader2:
                    LOGger.addlog(
                        f"欄位名稱對應：source_data 的 {xheader1} -> 模型的 {model_x1_header}, {xheader2} -> 模型的 {model_x2_header}",
                        logfile='',
                        colora=LOGger.OKCYAN
                    )
        
        # 為每個類別繪製分析圖
        colors = vs3.vs2.cm_rainbar(len(categories), c_alpha=0.7)
        for idx, category in enumerate(categories):
            category_data = source_data[source_data[categorical_header] == category]
            if len(category_data) > 0:
                color = colors[idx] if idx < len(colors) else colors[0]
                
                # 繪製散點圖
                ax = axes[idx // 2, idx % 2]
                ax.scatter(category_data[continuous_header], category_data[yheader],
                          c=color[:3], alpha=0.6, s=30, label=str(category))
                ax.set_xlabel(continuous_header, fontproperties=vs.MJHfontprop())
                ax.set_ylabel(yheader, fontproperties=vs.MJHfontprop())
                ax.set_title(f'{categorical_header}={category}', fontproperties=vs.MJHfontprop())
                ax.legend(prop=vs.MJHfontprop())
                ax.grid(True, alpha=0.3)
        
        # 儲存圖片
        output_filename = f'two_dimension_mixed_{xheader1}_{xheader2}_{yheader}.png'
        output_dir = output_dir if LOGger.isinstance_not_empty(output_dir, str) else 'analysis_output'
        if LOGger.isinstance_not_empty(output_dir, str):
            os.makedirs(output_dir, exist_ok=True)
            output_path = getProjectOutputPath(output_dir, output_filename)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        ret['image_path'] = output_path if 'output_path' in locals() else None
        ret['analysis_type'] = 'mixed_to_y'
        ret['model_output'] = None if not include_model_prediction else {}
        
        return True
        
    except Exception as e:
        LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
        ret['msg'] = f'analyzeMixedToY error: {str(e)}'
        return False

def analyzeTwoCategoricalToY(
        source_data: pd.DataFrame,
        xheader1: str,
        xheader2: str,
        yheader: str,
        y_type: str,
        fixed_values: Optional[Dict[str, Any]] = None,
        include_model_prediction: bool = True,
        ret: Optional[Dict[str, Any]] = None,
        api_url: Optional[str] = None,
        model_name: str = "ACAngle",
        version: str = "v0-0-2-0",
        output_dir: Optional[str] = None,
        timeout: int = 30,
        **kwags
    ) -> bool:
    """
    兩個類別型變數對Y的分析（使用 API 調用 MDC 模型）
    注意：此函數目前為簡化實現，完整實現需要參考 data_analysis.py
    """
    ret = ret if isinstance(ret, dict) else {}
    
    try:
        # 如果未提供 api_url，嘗試從 config.json 讀取
        if api_url is None:
            api_url = get_default_api_url()
        
        # 檢查是否需要 API
        if include_model_prediction and api_url is None:
            LOGger.addlog("警告：未提供 api_url，跳過模型預測", logfile='', colora=LOGger.WARNING)
            include_model_prediction = False
        
        # 獲取模型資訊（如果需要預測）
        xheaders, yheaders = [], []
        fixed_headers = []
        if include_model_prediction and api_url:
            xheaders, yheaders = get_mdc_headers_via_api(api_url, model_name, version, timeout)
            if not xheaders:
                LOGger.addlog("無法獲取模型資訊，跳過模型預測", logfile='', colora=LOGger.WARNING)
                include_model_prediction = False
            if not xheader1 in xheaders or not xheader2 in xheaders:
                LOGger.addlog("xheader1 或 xheader2 不在模型資訊中，跳過模型預測", logfile='', colora=LOGger.WARNING)
                include_model_prediction = False
            fixed_headers = [x for x in xheaders if x not in [xheader1, xheader2]]
            if fixed_headers:
                fixed_assignments_set = set(fixed_values.keys()) if fixed_values else set()
                if not set(fixed_headers).issubset(fixed_assignments_set):
                    print(LOGger.FAIL + f"固定欄位無校：\nfixed_headers: {fixed_headers}\nfixed_values: {fixed_values}")
                    include_model_prediction = False
            if include_model_prediction:
                # 建立模型輸入欄位名稱與 source_data 欄位名稱的對應關係
                # 即使目前沒有實現模型預測的響應面繪製，也先建立對應關係以備將來使用
                header_mapping = map_source_headers_to_model_headers(xheader1, xheader2, xheaders)
                model_x1_header = header_mapping['model_x1_header']
                model_x2_header = header_mapping['model_x2_header']
                
                # 記錄對應關係（用於調試）
                if model_x1_header != xheader1 or model_x2_header != xheader2:
                    LOGger.addlog(
                        f"欄位名稱對應：source_data 的 {xheader1} -> 模型的 {model_x1_header}, {xheader2} -> 模型的 {model_x2_header}",
                        logfile='',
                        colora=LOGger.OKCYAN
                    )
        
        # 設定中文字型
        vs3.plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
        
        # 建立交叉表分析
        import seaborn as sns
        pivot_table = source_data.pivot_table(values=yheader, index=xheader1, columns=xheader2, aggfunc='mean')
        
        # 繪製熱力圖
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='viridis', ax=ax)
        ax.set_title(f'{xheader1} + {xheader2} 對 {yheader} 的雙類別分析', fontproperties=vs.MJHfontprop())
        ax.set_xlabel(xheader2, fontproperties=vs.MJHfontprop())
        ax.set_ylabel(xheader1, fontproperties=vs.MJHfontprop())
        
        # 儲存圖片
        output_filename = f'two_dimension_categorical_{xheader1}_{xheader2}_{yheader}.png'
        output_dir = output_dir if LOGger.isinstance_not_empty(output_dir, str) else 'analysis_output'
        if LOGger.isinstance_not_empty(output_dir, str):
            os.makedirs(output_dir, exist_ok=True)
            output_path = getProjectOutputPath(output_dir, output_filename)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        ret['image_path'] = output_path if 'output_path' in locals() else None
        ret['analysis_type'] = 'two_categorical_to_y'
        ret['model_output'] = None if not include_model_prediction else {}
        ret['pivot_table'] = pivot_table.to_dict()
        
        return True
        
    except Exception as e:
        LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
        ret['msg'] = f'analyzeTwoCategoricalToY error: {str(e)}'
        return False

