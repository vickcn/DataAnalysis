from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
import os
import sys
import json
import math
import numpy as np
import pandas as pd

# 加入當前目錄到 Python 路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from package import LOGger
from src import analyze_engine as ae
m_print = LOGger.addloger(logfile='')
m_logfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api_server_%t.log')
m_addlog = LOGger.addloger(logfile=m_logfile)
m_DataSetPath = ae.m_DataSetPath
m_ExportSetPath = ae.m_ExportSetPath
# 不穩定度 API 預設輸出根目錄（issues/不穩定度計算功能API化.iss）；TODO: 可改讀 config.json
m_instability_api_output_default = r"\\10.1.3.127\ml_home\DataAnalysis\apiOutput"

# 載入配置檔案
def load_config():
    """載入 config.json 配置檔案"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['load_config'])
        m_addlog(f"載入配置檔案失敗", stamps=['load_config'], colora=LOGger.FAIL)
        return {
            "Host_IP": "127.0.0.1",
            "Host_Port": 8000,
            "source_dir": "."
        }

# 載入配置
config = load_config()

# 導入 DataAnalysis 模組
import DataAnalysis as DA
from package import dataframeprocedure as DFP
from src import heteroscedastic as het_instability

app = FastAPI(
    title="DataAnalysis API",
    description="資料分析 API 服務",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 掛載靜態文件目錄（CSS、JS）
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 模板目錄路徑
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'template')

# "filePath": "\\\\10.1.3.129\\EIMSFileData\\ExportSet\\63\\1\\graph\\corr.xlsx",
# "sheet": 0,
# "method": "mic",
# "exp_fd": "tmp\\micCorr",
# "stamps": [
# ],
# "ret": {}

# 請求模型
class FilePathRequest(BaseModel):
    filePath: str
    sheet: Optional[int] = 0
    output_dir: Optional[str] = None

class AnalysisRequest(FilePathRequest):
    xheader: str
    yheader: str
    fixed_values: Optional[Dict[str, Any]] = None
    include_model_prediction: Optional[bool] = True
    ret: Optional[Dict[str, Any]] = {}

class AnalysisViaApiRequest(AnalysisRequest):
    """API 版本的分析請求模型"""
    api_url: Optional[str] = "http://10.1.3.236:5678"
    model_name: Optional[str] = "ACAngle"
    version: Optional[str] = "v0-0-2-0"
    timeout: Optional[int] = 30

class TwoDimensionAnalysisRequest(FilePathRequest):
    xheader1: str
    xheader2: str
    yheader: str
    fixed_values: Optional[Dict[str, Any]] = None
    include_model_prediction: Optional[bool] = True
    ret: Optional[Dict[str, Any]] = {}

class MixedAnalysisRequest(FilePathRequest):
    xheader1: str
    xheader2: str
    yheader: str
    x1_type: str
    x2_type: str
    y_type: str
    fixed_values: Optional[Dict[str, Any]] = None
    include_model_prediction: Optional[bool] = True
    ret: Optional[Dict[str, Any]] = {}

class TwoDimensionAnalysisViaApiRequest(TwoDimensionAnalysisRequest):
    """API 版本的雙維度分析請求模型"""
    api_url: Optional[str] = "http://10.1.3.236:5678"
    model_name: Optional[str] = "ACAngle"
    version: Optional[str] = "v0-0-2-0"
    timeout: Optional[int] = 30
    y_type: Optional[str] = "continuous"  # 用於 analyzeTwoContinuousToY 和 analyzeTwoCategoricalToY

class MixedAnalysisViaApiRequest(MixedAnalysisRequest):
    """API 版本的混合分析請求模型"""
    api_url: Optional[str] = "http://10.1.3.236:5678"
    model_name: Optional[str] = "ACAngle"
    version: Optional[str] = "v0-0-2-0"
    timeout: Optional[int] = 30

class CorrelationRequest(FilePathRequest):
    method: Optional[str] = 'mic'
    exp_fd: Optional[str] = 'micCorr'
    stamps: Optional[List[str]] = None
    ret: Optional[Dict[str, Any]] = {}
    data_config_file: Optional[str] = None

class DataParsingRequest(FilePathRequest):
    ret: Optional[Dict[str, Any]] = {}
    exp_fd: Optional[str] = 'micCorr'

class DataParsingAndPlotCorrelationRequest(FilePathRequest):
    method: Optional[str] = 'mic'
    exp_fd: Optional[str] = 'micCorr'
    stamps: Optional[List[str]] = None
    ret: Optional[Dict[str, Any]] = {}
    selectHeader: Optional[List[str]] = None
    isAggregate: Optional[bool] = True

class StandardTestsRequest(FilePathRequest):
    column: str
    group_by: Optional[str] = None
    alpha: Optional[float] = 0.05
    output_dir: Optional[str] = None
    use_api: Optional[bool] = False
    plan_id: Optional[str] = None
    seq_no: Optional[str] = None

class PointsRequest(FilePathRequest):
    columns: List[str]                      # 要取出的欄位（2D: [x,y], 3D: [x1,x2,y]）
    fixed_values: Optional[Dict[str, Any]] = None
    sample_n: Optional[int] = 2000
    seed: Optional[int] = 7
    include_extra_columns: Optional[List[str]] = None  # 額外想看詳情的欄位（可選）


class InstabilityBaseRequest(FilePathRequest):
    """不穩定度 API 共用欄位；見 issues/不穩定度計算功能API化.iss"""
    x_col: str
    y_col: str
    layers: Optional[List[int]] = None
    ret: Optional[Dict[str, Any]] = {}
    # layers: None 表三層皆要；否則 1/2/3 — TODO: pydantic 驗證與 heteroscedastic 對齊


class InstabilityComputeRequest(InstabilityBaseRequest):
    """僅 JSON 回傳計算結果，不寫檔、不畫圖（邏輯在 heteroscedastic.compute_instability_payload）"""


class InstabilitySaveRequest(InstabilityBaseRequest):
    """寫檔；output_dir 未給則用 m_instability_api_output_default"""


class InstabilityPlotRequest(InstabilityBaseRequest):
    """繪圖輸出；output_dir 未給則用 m_instability_api_output_default"""


# 安全序列化函數
def safe_serialize(obj, max_depth=3, current_depth=0, _seen=None, **kwags):
    """
    安全地序列化物件，避免循環引用和無窮遞迴
    """
    # 初始化 seen 集合來追蹤已處理的物件
    if _seen is None:
        _seen = set()
        m_print(f'[safe_serialize] 開始序列化，類型: {type(obj).__name__}', colora=LOGger.OKCYAN)
    
    # 檢查循環引用
    obj_id = id(obj)
    if obj_id in _seen:
        m_print(f'[safe_serialize] 檢測到循環引用: {type(obj).__name__}', colora=LOGger.WARNING)
        return f"<circular_reference: {type(obj).__name__}>"
    
    # 防止無窮遞迴
    if current_depth > max_depth:
        m_print(f'[safe_serialize] 達到最大深度限制: {current_depth} > {max_depth}, 類型: {type(obj).__name__}', colora=LOGger.WARNING)
        return f"<max_depth_reached: {type(obj).__name__}>"
    
    # None, bool, str 直接返回
    if obj is None or isinstance(obj, (bool, str)):
        if current_depth == 0:
            m_print(f'[safe_serialize] 簡單類型直接返回: {type(obj).__name__}', colora=LOGger.OKGREEN)
        return obj
    
    # 處理整數類型（包括 numpy 整數類型）
    if isinstance(obj, (int, np.integer)):
        try:
            result = int(obj)
            if current_depth == 0:
                m_print(f'[safe_serialize] 整數類型轉換成功: {type(obj).__name__} -> int', colora=LOGger.OKGREEN)
            return result
        except Exception as e:
            m_print(f'[safe_serialize] 整數轉換失敗: {type(obj).__name__}, 錯誤: {str(e)}', colora=LOGger.FAIL)
            return None
    
    # 處理 float 類型（包括 numpy 浮點類型），檢查是否為 inf、-inf 或 nan
    if isinstance(obj, (float, np.floating)):
        try:
            if math.isnan(obj):
                m_print(f'[safe_serialize] 檢測到 NaN 值 (深度: {current_depth})', colora=LOGger.WARNING)
                return None  # 將 inf、-inf 和 nan 轉換為 None（JSON 中的 null）
            if math.isinf(obj):
                m_print(f'[safe_serialize] 檢測到 Inf 值: {obj} (深度: {current_depth})', colora=LOGger.WARNING)
                return None  # 將 inf、-inf 和 nan 轉換為 None（JSON 中的 null）
            result = float(obj)
            if current_depth == 0:
                m_print(f'[safe_serialize] 浮點數類型轉換成功: {type(obj).__name__} -> float', colora=LOGger.OKGREEN)
            return result
        except (ValueError, TypeError, OverflowError) as e:
            # 如果轉換失敗，返回 None
            m_print(f'[safe_serialize] 浮點數轉換失敗: {type(obj).__name__}, 值: {obj}, 錯誤: {str(e)}', colora=LOGger.FAIL)
            return None
    
    # 將當前物件加入 seen 集合（僅對可變物件）
    if isinstance(obj, (dict, list)):
        _seen.add(obj_id)
    
    try:
        # DataFrame
        if isinstance(obj, pd.DataFrame):
            m_print(f'[safe_serialize] 處理 DataFrame (深度: {current_depth}), 形狀: {obj.shape}', colora=LOGger.OKCYAN)
            try:
                # 安全處理 columns
                columns_list = []
                for col in obj.columns:
                    try:
                        if isinstance(col, (float, np.floating)):
                            if math.isnan(col) or math.isinf(col):
                                columns_list.append("<nan_or_inf_column>")
                            else:
                                columns_list.append(str(float(col)))
                        else:
                            columns_list.append(str(col))
                    except Exception as e:
                        m_print(f'[safe_serialize] DataFrame column 處理失敗: {col}, 錯誤: {str(e)}', colora=LOGger.FAIL)
                        columns_list.append("<column_error>")
                
                # 安全處理 dtypes
                dtypes_dict = {}
                for col, dtype in obj.dtypes.items():
                    try:
                        col_str = str(col) if not (isinstance(col, (float, np.floating)) and (math.isnan(col) or math.isinf(col))) else "<nan_or_inf_column>"
                        dtypes_dict[col_str] = str(dtype)
                    except Exception as e:
                        m_print(f'[safe_serialize] DataFrame dtype 處理失敗: {col}, 錯誤: {str(e)}', colora=LOGger.FAIL)
                        dtypes_dict["<dtype_error>"] = str(dtype)
                
                return {
                    "_type": "DataFrame",
                    "shape": list(obj.shape),
                    "columns": columns_list,
                    "dtypes": dtypes_dict
                }
            except Exception as e:
                m_print(f'[safe_serialize] DataFrame 處理失敗: {str(e)}', colora=LOGger.FAIL)
                raise
        
        # Series
        if isinstance(obj, pd.Series):
            m_print(f'[safe_serialize] 處理 Series (深度: {current_depth}), 形狀: {obj.shape}', colora=LOGger.OKCYAN)
            try:
                return {
                    "_type": "Series",
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "name": str(obj.name) if obj.name is not None else None
                }
            except Exception as e:
                m_print(f'[safe_serialize] Series 處理失敗: {str(e)}', colora=LOGger.FAIL)
                raise
        
        # Numpy array
        if isinstance(obj, np.ndarray):
            m_print(f'[safe_serialize] 處理 ndarray (深度: {current_depth}), 形狀: {obj.shape}, dtype: {obj.dtype}', colora=LOGger.OKCYAN)
            try:
                return {
                    "_type": "ndarray",
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype)
                }
            except Exception as e:
                m_print(f'[safe_serialize] ndarray 處理失敗: {str(e)}', colora=LOGger.FAIL)
                raise
        
        # 函數
        if callable(obj):
            m_print(f'[safe_serialize] 處理函數: {getattr(obj, "__name__", "unknown")}', colora=LOGger.OKCYAN)
            return f"<function: {getattr(obj, '__name__', 'unknown')}>"
        
        # 字典
        if isinstance(obj, dict):
            m_print(f'[safe_serialize] 處理字典 (深度: {current_depth}), 鍵數量: {len(obj)}', colora=LOGger.OKCYAN)
            result = {}
            for idx, (k, v) in enumerate(obj.items()):
                try:
                    # 處理鍵
                    if isinstance(k, (float, np.floating)):
                        if math.isnan(k) or math.isinf(k):
                            m_print(f'[safe_serialize] 字典鍵 {idx} 為 inf/nan，使用替代鍵名', colora=LOGger.WARNING)
                            key_str = "<nan_or_inf_key>"
                        else:
                            key_str = str(float(k))
                    else:
                        key_str = str(k)
                    
                    # 處理值
                    m_print(f'[safe_serialize] 處理字典鍵 "{key_str}" (深度: {current_depth})', colora=LOGger.OKCYAN)
                    serialized_value = safe_serialize(v, max_depth, current_depth + 1, _seen)
                    result[key_str] = serialized_value
                except Exception as e:
                    m_print(f'[safe_serialize] 字典鍵 "{k}" 處理失敗: {str(e)}', colora=LOGger.FAIL)
                    LOGger.exception_process(e, logfile='', stamps=['safe_serialize'])
                    try:
                        result[str(k)] = f"<error: {str(e)[:30]}>"
                    except:
                        result["<key_error>"] = f"<error: {str(e)[:30]}>"
            return result
        
        # 列表或元組
        if isinstance(obj, (list, tuple)):
            m_print(f'[safe_serialize] 處理 {type(obj).__name__} (深度: {current_depth}), 長度: {len(obj)}', colora=LOGger.OKCYAN)
            try:
                if len(obj) > 50:  # 限制列表長度
                    m_print(f'[safe_serialize] 列表過長 ({len(obj)} > 50)，只處理前5個元素', colora=LOGger.WARNING)
                    return {
                        "_type": type(obj).__name__,
                        "length": len(obj),
                        "sample": [safe_serialize(item, max_depth, current_depth + 1, _seen) for item in list(obj)[:5]]
                    }
                return [safe_serialize(item, max_depth, current_depth + 1, _seen) for item in obj]
            except Exception as e:
                m_print(f'[safe_serialize] {type(obj).__name__} 處理失敗: {str(e)}', colora=LOGger.FAIL)
                LOGger.exception_process(e, logfile='', stamps=['safe_serialize'])
                return f"<{type(obj).__name__}: {len(obj)} items, error: {str(e)[:20]}>"
        
        # 其他物件
        m_print(f'[safe_serialize] 未知類型: {type(obj).__name__} (深度: {current_depth})', colora=LOGger.WARNING)
        return f"<{type(obj).__name__}>"
    
    except Exception as e:
        m_print(f'[safe_serialize] 處理 {type(obj).__name__} 時發生例外 (深度: {current_depth}): {str(e)}', colora=LOGger.FAIL)
        LOGger.exception_process(e, logfile='', stamps=['safe_serialize'])
        return f"<{type(obj).__name__}: error: {str(e)[:50]}>"
    finally:
        # 從 seen 集合中移除（允許在其他分支中再次處理）
        if isinstance(obj, (dict, list)) and obj_id in _seen:
            _seen.discard(obj_id)
            if current_depth == 0:
                m_print(f'[safe_serialize] 序列化完成', colora=LOGger.OKGREEN)

# 通用資料載入函數
def load_data(file_path: str, sheet: int = 0):
    """載入資料檔案"""
    try:
        # 如果檔案路徑不是絕對路徑，則從 source_dir 開始尋找
        if not os.path.isabs(file_path):
            # 先檢查是否在當前目錄
            if os.path.exists(file_path):
                full_path = file_path
            else:
                # 從 source_dir 開始尋找
                source_dir = config.get('source_dir', '.')
                full_path = os.path.join(source_dir, file_path)
        else:
            full_path = file_path
        
        if not os.path.exists(full_path):
            # 嘗試在 referenceDirs 中尋找
            reference_dirs = config.get('referenceDirs', [])
            found = False
            for ref_dir in reference_dirs:
                test_path = os.path.join(ref_dir, file_path)
                if os.path.exists(test_path):
                    full_path = test_path
                    found = True
                    break
            
            if not found:
                raise HTTPException(status_code=404, detail=f"檔案不存在: {file_path}")
        
        data = DFP.import_data(full_path, sht=sheet)
        if data is None:
            raise HTTPException(status_code=400, detail="無法載入資料檔案")
        
        return data
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['load_data'])
        raise HTTPException(status_code=500, detail=f"載入資料時發生錯誤: {str(e)}")

# 前端模板路由
@app.get("/index.html", response_class=HTMLResponse)
async def index_html():
    """返回首頁（動態注入配置）"""
    index_path = os.path.join(template_dir, 'index.html')
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # 從 config.json 取得 API 地址
        api_host = config.get("Host_IP", "127.0.0.1")
        api_port = config.get("Host_Port", 8000)
        api_base = f"http://{api_host}:{api_port}"
        
        # 替換硬編碼的 API 地址
        html_content = html_content.replace(
            'http://10.1.3.127:6030',
            api_base
        )
        
        return HTMLResponse(content=html_content)
    raise HTTPException(status_code=404, detail="index.html 不存在")

@app.get("/main.html", response_class=HTMLResponse)
async def main_html():
    """返回主介面頁面（動態注入配置）"""
    main_path = os.path.join(template_dir, 'main.html')
    if os.path.exists(main_path):
        with open(main_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # 從 config.json 取得 API 地址
        api_host = config.get("Host_IP", "127.0.0.1")
        api_port = config.get("Host_Port", 8000)
        api_base = f"http://{api_host}:{api_port}"
        
        # 替換硬編碼的 API 地址
        html_content = html_content.replace(
            'http://10.1.3.127:6030',
            api_base
        )
        
        return HTMLResponse(content=html_content)
    raise HTTPException(status_code=404, detail="main.html 不存在")

@app.get("/heteroscedastic.html", response_class=HTMLResponse)
async def heteroscedastic_html():
    """返回不穩定度頁面（動態注入配置）"""
    page_path = os.path.join(template_dir, 'heteroscedastic.html')
    if os.path.exists(page_path):
        with open(page_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # 從 config.json 取得 API 地址
        api_host = config.get("Host_IP", "127.0.0.1")
        api_port = config.get("Host_Port", 8000)
        api_base = f"http://{api_host}:{api_port}"

        # 替換硬編碼的 API 地址
        html_content = html_content.replace(
            'http://10.1.3.127:6030',
            api_base
        )

        return HTMLResponse(content=html_content)
    raise HTTPException(status_code=404, detail="heteroscedastic.html 不存在")

@app.get("/app.css")
async def app_css():
    """返回 CSS 文件"""
    css_path = os.path.join(static_dir, 'css', 'app.css')
    if os.path.exists(css_path):
        return FileResponse(css_path, media_type="text/css")
    raise HTTPException(status_code=404, detail="app.css 不存在")

@app.get("/app.js")
async def app_js():
    """返回 JavaScript 文件"""
    js_path = os.path.join(static_dir, 'js', 'app.js')
    if os.path.exists(js_path):
        return FileResponse(js_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="app.js 不存在")

# API 端點
@app.get("/")
async def root():
    """根路由：返回 API 資訊或重定向到首頁"""
    # 檢查是否有 index.html，如果有則返回 HTML（動態注入配置），否則返回 JSON
    index_path = os.path.join(template_dir, 'index.html')
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # 從 config.json 取得 API 地址
        api_host = config.get("Host_IP", "127.0.0.1")
        api_port = config.get("Host_Port", 8000)
        api_base = f"http://{api_host}:{api_port}"
        
        # 替換硬編碼的 API 地址
        html_content = html_content.replace(
            'http://10.1.3.127:6030',
            api_base
        )
        
        return HTMLResponse(content=html_content)
    return {
        "message": "DataAnalysis API 服務運行中",
        "config": {
            "host_ip": config.get("Host_IP", "127.0.0.1"),
            "host_port": config.get("Host_Port", 8000),
            "source_dir": config.get("source_dir", "."),
            "reference_dirs": config.get("referenceDirs", [])
        }
    }

@app.get("/config")
async def get_config():
    """取得服務配置資訊"""
    return {
        "host_ip": config.get("Host_IP", "127.0.0.1"),
        "host_port": config.get("Host_Port", 8000),
        "source_dir": config.get("source_dir", "."),
        "reference_dirs": config.get("referenceDirs", []),
        "gpu_memory_limit": config.get("gpuMemoLimit", 10000)
    }

@app.get("/files")
async def list_files(directory: Optional[str] = None):
    """列出 root（source_disk / source_dir）下指定目錄的檔案與子目錄（僅允許瀏覽 root 範圍）"""
    try:
        # 瀏覽根目錄：優先使用 source_disk（config.json: source_disk），否則退回 source_dir
        browse_root = config.get("source_disk") or config.get("source_dir") or "."
        browse_root = str(browse_root).strip() if browse_root is not None else "."
        # Windows drive root: "X:" -> "X:\\"
        if len(browse_root) == 2 and browse_root[1] == ":" and browse_root[0].isalpha():
            browse_root = browse_root + os.sep
        browse_root_abs = os.path.abspath(browse_root)

        source_dir = config.get("source_dir", ".")
        source_dir_abs = os.path.abspath(source_dir)
        # 預設顯示目錄：source_dir 必須位於 browse_root 底下，否則退回 browse_root
        try:
            default_is_under_root = os.path.commonpath([source_dir_abs, browse_root_abs]) == browse_root_abs
        except ValueError:
            default_is_under_root = False
        default_dir_abs = source_dir_abs if default_is_under_root else browse_root_abs

        # directory:
        # - None（未給參數）: 回傳預設目錄（通常是 source_dir）
        # - ""（明確給空字串）: 回傳瀏覽根目錄（browse_root）
        # - 其他: 相對於 browse_root 的子路徑，或絕對路徑（仍需在 browse_root 內）
        if directory is None:
            target_dir_abs = default_dir_abs
        else:
            directory = str(directory).strip()
            if directory == "":
                target_dir_abs = browse_root_abs
            elif os.path.isabs(directory):
                target_dir_abs = os.path.abspath(directory)
            else:
                target_dir_abs = os.path.abspath(os.path.join(browse_root_abs, directory))

        # 僅允許瀏覽 source_dir 範圍（防止跳出根目錄）
        try:
            is_under_source = os.path.commonpath([target_dir_abs, browse_root_abs]) == browse_root_abs
        except ValueError:
            is_under_source = False
        if not is_under_source:
            raise HTTPException(status_code=400, detail=f"僅允許瀏覽 root 底下: {browse_root_abs}")

        if not os.path.exists(target_dir_abs):
            raise HTTPException(status_code=404, detail=f"目錄不存在: {target_dir_abs}")
        if not os.path.isdir(target_dir_abs):
            raise HTTPException(status_code=400, detail=f"不是目錄: {target_dir_abs}")

        directories = []
        files = []
        for item in os.listdir(target_dir_abs):
            item_path = os.path.join(target_dir_abs, item)
            if os.path.isdir(item_path):
                directories.append({
                    "name": item,
                    "path": item_path,
                    "relative_path": os.path.relpath(item_path, browse_root_abs),
                    "modified": os.path.getmtime(item_path)
                })
            elif os.path.isfile(item_path):
                files.append({
                    "name": item,
                    "path": item_path,
                    "relative_path": os.path.relpath(item_path, browse_root_abs),
                    "size": os.path.getsize(item_path),
                    "modified": os.path.getmtime(item_path)
                })

        directories = sorted(directories, key=lambda x: x.get("name", "").lower())
        files = sorted(files, key=lambda x: x.get("name", "").lower())

        relative_directory = os.path.relpath(target_dir_abs, browse_root_abs)
        if relative_directory == ".":
            relative_directory = ""
            parent_relative_directory = None
        else:
            parent_abs = os.path.dirname(target_dir_abs)
            parent_relative_directory = os.path.relpath(parent_abs, browse_root_abs)
            if parent_relative_directory == ".":
                parent_relative_directory = ""

        return {
            # 前端瀏覽用的根目錄（「最上層」）
            "source_dir": browse_root_abs,
            # 原本 config.json 的 source_dir（預設顯示起點），方便 debug
            "source_dir_config": source_dir_abs,
            "directory": target_dir_abs,
            "relative_directory": relative_directory,
            "parent_relative_directory": parent_relative_directory,
            "directories": directories,
            "files": files,
            "count": len(files),
            "dir_count": len(directories),
        }
    except HTTPException:
        raise
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['list_files'])
        raise HTTPException(status_code=500, detail=f"列出檔案時發生錯誤: {str(e)}")

@app.get("/search-file")
async def search_file(filename: str):
    """搜尋檔案在配置的目錄中"""
    try:
        search_results = []
        
        # 搜尋來源目錄
        source_dir = config.get("source_dir", ".")
        if os.path.exists(source_dir):
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    if filename.lower() in file.lower():
                        search_results.append({
                            "name": file,
                            "path": os.path.join(root, file),
                            "directory": root,
                            "found_in": "source_dir"
                        })
        
        # 搜尋參考目錄
        reference_dirs = config.get("referenceDirs", [])
        for ref_dir in reference_dirs:
            if os.path.exists(ref_dir):
                for root, dirs, files in os.walk(ref_dir):
                    for file in files:
                        if filename.lower() in file.lower():
                            search_results.append({
                                "name": file,
                                "path": os.path.join(root, file),
                                "directory": root,
                                "found_in": "reference_dir"
                            })
        
        return {
            "filename": filename,
            "results": search_results,
            "count": len(search_results)
        }
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['search_file'])
        raise HTTPException(status_code=500, detail=f"搜尋檔案時發生錯誤: {str(e)}")

@app.post("/plot-correlation")
async def plot_correlation(request: CorrelationRequest):
    """繪製相關性圖表"""
    try:
        data = load_data(request.filePath, request.sheet)
        
        if request.data_config_file:
            selected_header = None
            m_print(f'[plot_correlation] 載入配置檔案: {request.data_config_file}', colora=LOGger.OKGREEN)
            data_config = LOGger.load_json(request.data_config_file)
            m_print(f'[plot_correlation] 配置檔案內容: {list(data_config.keys()) if isinstance(data_config, dict) else "非字典格式"}', colora=LOGger.OKGREEN)
            if isinstance(data_config, dict):
                # 優先使用 selected_header
                selected_header = data_config.get('selected_header')
                m_print(f'[plot_correlation] selected_header: {selected_header}', colora=LOGger.OKCYAN)
                # 如果沒有 selected_header，從 HEADER 配置中提取
                if not selected_header and 'HEADER' in data_config:
                    header_config = data_config.get('HEADER', {})
                    m_print(f'[plot_correlation] HEADER 配置: {header_config}', colora=LOGger.OKCYAN)
                    if header_config:
                        header_names = []
                        for key, value in header_config.items():
                            if isinstance(value, str) and value:
                                header_names.append(value)
                        if header_names:
                            selected_header = header_names
                            m_print(f'[plot_correlation] 從 HEADER 提取的欄位: {header_names}', colora=LOGger.OKGREEN)
            if isinstance(selected_header, list):
                # 只保留存在於資料中的欄位
                available_headers = [h for h in selected_header if h in data.columns]
                if available_headers:
                    m_print(f'[plot_correlation] 使用選取的欄位: {available_headers}', colora=LOGger.OKGREEN)
                    data = data[available_headers]
                else:
                    m_print(f'[plot_correlation] 警告: 配置的欄位都不存在於資料中: {selected_header}', colora=LOGger.WARNING)
        # 設定預設輸出目錄
        exp_fd = request.exp_fd or os.path.join('tmp', 'micCorr')
        
        result = DA.PlotCorrelation(
            matrix=data,
            method=request.method,
            exp_fd=exp_fd,
            stamps=request.stamps,
            ret=request.ret
        )
        
        return {
            "success": result,
            "message": "相關性圖表繪製完成" if result else "相關性圖表繪製失敗",
            "output_directory": exp_fd
        }
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['plot_correlation'])
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/calculate-correlation")
async def calculate_correlation(request: CorrelationRequest):
    """計算相關性"""
    try:
        data = load_data(request.filePath, request.sheet)
        if request.data_config_file:
            selected_header = None
            m_print(f'[calculate_correlation] 載入配置檔案: {request.data_config_file}', colora=LOGger.OKCYAN)
            data_config = LOGger.load_json(request.data_config_file)
            m_print(f'[calculate_correlation] 配置檔案內容: {list(data_config.keys()) if isinstance(data_config, dict) else "非字典格式"}', colora=LOGger.OKCYAN)
            if isinstance(data_config, dict):
                # 優先使用 selected_header
                selected_header = data_config.get('selected_header')
                m_print(f'[calculate_correlation] selected_header: {selected_header}', colora=LOGger.OKCYAN)
                # 如果沒有 selected_header，從 HEADER 配置中提取
                if not selected_header and 'HEADER' in data_config:
                    header_config = data_config.get('HEADER', {})
                    m_print(f'[calculate_correlation] HEADER 配置: {header_config}', colora=LOGger.OKCYAN)
                    if header_config:
                        header_names = []
                        for key, value in header_config.items():
                            if isinstance(value, str) and value:
                                header_names.append(value)
                        if header_names:
                            selected_header = header_names
                            m_print(f'[calculate_correlation] 從 HEADER 提取的欄位: {header_names}', colora=LOGger.OKGREEN)
            if isinstance(selected_header, list):
                # 只保留存在於資料中的欄位
                available_headers = [h for h in selected_header if h in data.columns]
                if available_headers:
                    m_print(f'[calculate_correlation] 使用選取的欄位: {available_headers}', colora=LOGger.OKGREEN)
                    data = data[available_headers]
                else:
                    m_print(f'[calculate_correlation] 警告: 配置的欄位都不存在於資料中: {selected_header}', colora=LOGger.WARNING)

        # 設定預設輸出目錄
        exp_fd = request.exp_fd or os.path.join('tmp', 'micCorr')
        
        result = DA.CalculateCorrelation(
            matrix=data,
            method=request.method,
            exp_fd=exp_fd,
            stamps=request.stamps,
            ret=request.ret
        )
        
        # 使用安全序列化
        m_print('使用安全序列化', colora=LOGger.WARNING)
        serializable_ret = safe_serialize(request.ret)
        
        response_dict = {
            "success": result,
            "message": "相關性計算完成" if result else "相關性計算失敗",
            "output_directory": str(exp_fd),  # 確保是字符串
            "res": serializable_ret
        }
        
        # 驗證響應字典中的所有值都是可序列化的
        try:
            import json
            json.dumps(response_dict)  # 測試是否可以序列化
            m_print(f'[calculate_correlation] 響應字典序列化測試通過', colora=LOGger.OKGREEN)
        except (ValueError, TypeError) as e:
            m_print(f'[calculate_correlation] 響應字典序列化測試失敗: {str(e)}', colora=LOGger.FAIL)
            # 如果失敗，再次對整個響應進行安全序列化
            response_dict = safe_serialize(response_dict)
            m_print(f'[calculate_correlation] 已對響應進行安全序列化', colora=LOGger.WARNING)
        
        return response_dict
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['calculate_correlation'])
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data-parsing")
async def data_parsing(request: DataParsingRequest):
    """資料解析"""
    try:
        data = load_data(request.filePath, request.sheet)
        
        result = DA.DataParsingFromFile(data, ret=request.ret)
        
        # 使用安全序列化
        serializable_ret = safe_serialize(request.ret)
        
        return {
            "success": result,
            "message": "資料解析完成" if result else "資料解析失敗",
            "res": serializable_ret
        }
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['data_parsing'])
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dppc")
async def data_parsing_and_plot_correlation(request: DataParsingAndPlotCorrelationRequest):
    """資料解析並繪製相關性圖表"""
    try:
        print(f"[API] /dppc 開始處理，filePath: {request.filePath}, stamps: {request.stamps}, exp_fd: {request.exp_fd}")
        
        result = DA.DataParsingAndPlotCorrelation(
            request.filePath, 
            ret=request.ret, 
            selectHeader=request.selectHeader,
            exp_fd=request.exp_fd,
            stamps=request.stamps,
            method=request.method,
            isAggregate=request.isAggregate
        )
        
        m_addlog(f"DataParsingAndPlotCorrelation 執行完成，result: {result}", stamps=['data_parsing_and_plot_correlation'], colora=LOGger.OKCYAN)
        
        if not result:
            m_addlog(f"DataParsingAndPlotCorrelation 返回 False，但繼續處理序列化", stamps=['data_parsing_and_plot_correlation'], colora=LOGger.WARNING)
        
        # 使用安全序列化函數處理 ret
        try:
            serializable_ret = safe_serialize(request.ret)
            m_addlog(f"序列化完成", stamps=['data_parsing_and_plot_correlation'], colora=LOGger.OKCYAN)
        except Exception as serialize_error:
            m_addlog(f"序列化失敗: {str(serialize_error)}", stamps=['data_parsing_and_plot_correlation'], colora=LOGger.FAIL)
            # 如果序列化失敗，至少返回基本的結果
            serializable_ret = {"error": "序列化失敗", "message": str(serialize_error)}
        
        return {
            "success": result,
            "message": "資料解析並繪製相關性圖表完成" if result else "資料解析並繪製相關性圖表失敗",
            "res": serializable_ret
        }
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['data_parsing_and_plot_correlation'])
        m_addlog(f"/dppc 端點錯誤", stamps=['data_parsing_and_plot_correlation'], colora=LOGger.FAIL)
        # 確保錯誤訊息能正確返回
        raise HTTPException(status_code=500, detail=f"處理失敗: {str(e)}")

@app.post("/analyze-continuous-to-continuous")
async def analyze_continuous_to_continuous(request: AnalysisRequest):
    """連續變數對連續變數分析"""
    try:
        data = load_data(request.filePath, request.sheet)
        
        result = DA.analyzeContinuousToContinuous(
            data=data,
            xheader=request.xheader,
            yheader=request.yheader,
            fixed_values=request.fixed_values,
            include_model_prediction=request.include_model_prediction,
            ret=request.ret,
            mdc=None,
            output_dir=request.output_dir
        )
        
        return {
            "success": result,
            "message": "連續變數對連續變數分析完成" if result else "分析失敗",
            "xheader": request.xheader,
            "yheader": request.yheader
        }
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['analyze_continuous_to_continuous'])
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-categorical-to-continuous")
async def analyze_categorical_to_continuous(request: AnalysisRequest):
    """類別變數對連續變數分析"""
    try:
        data = load_data(request.filePath, request.sheet)
        
        result = DA.analyzeCategoricalToContinuous(
            data=data,
            xheader=request.xheader,
            yheader=request.yheader,
            fixed_values=request.fixed_values,
            include_model_prediction=request.include_model_prediction,
            ret=request.ret,
            mdc=None,
            output_dir=request.output_dir
        )
        
        return {
            "success": result,
            "message": "類別變數對連續變數分析完成" if result else "分析失敗",
            "xheader": request.xheader,
            "yheader": request.yheader
        }
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['analyze_continuous_to_categorical'])
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-continuous-to-categorical")
async def analyze_continuous_to_categorical(request: AnalysisRequest):
    """連續變數對類別變數分析"""
    try:
        data = load_data(request.filePath, request.sheet)
        
        result = DA.analyzeContinuousToCategorical(
            data=data,
            xheader=request.xheader,
            yheader=request.yheader,
            fixed_values=request.fixed_values,
            ret=request.ret,
            mdc=None,
            output_dir=request.output_dir
        )
        
        return {
            "success": result,
            "message": "連續變數對類別變數分析完成" if result else "分析失敗",
            "xheader": request.xheader,
            "yheader": request.yheader
        }
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['analyze_categorical_to_categorical'])
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-categorical-to-categorical")
async def analyze_categorical_to_categorical(request: AnalysisRequest):
    """類別變數對類別變數分析"""
    try:
        data = load_data(request.filePath, request.sheet)
        
        result = DA.analyzeCategoricalToCategorical(
            data=data,
            xheader=request.xheader,
            yheader=request.yheader,
            fixed_values=request.fixed_values,
            ret=request.ret,
            mdc=None
        )
        
        return {
            "success": result,
            "message": "類別變數對類別變數分析完成" if result else "分析失敗",
            "xheader": request.xheader,
            "yheader": request.yheader
        }
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['analyze_categorical_to_categorical'])
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-continuous-to-continuous-via-api")
async def analyze_continuous_to_continuous_via_api(request: AnalysisViaApiRequest):
    """連續變數對連續變數分析（API 版本）"""
    try:
        data = load_data(request.filePath, request.sheet)
        
        result = DA.analyzeContinuousToContinuous_via_api(
            data=data,
            xheader=request.xheader,
            yheader=request.yheader,
            fixed_values=request.fixed_values,
            include_model_prediction=request.include_model_prediction,
            ret=request.ret,
            output_dir=request.output_dir,
            api_url=request.api_url,
            model_name=request.model_name,
            version=request.version,
            timeout=request.timeout
        )
        
        # 使用安全序列化處理 ret
        serializable_ret = safe_serialize(request.ret)
        
        return {
            "success": result,
            "message": "連續變數對連續變數分析完成（API 版本）" if result else "分析失敗",
            "xheader": request.xheader,
            "yheader": request.yheader,
            "api_url": request.api_url,
            "model_name": request.model_name,
            "version": request.version,
            "res": serializable_ret
        }
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['analyze_continuous_to_continuous_via_api'])
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-categorical-to-continuous-via-api")
async def analyze_categorical_to_continuous_via_api(request: AnalysisViaApiRequest):
    """類別變數對連續變數分析（API 版本）"""
    try:
        data = load_data(request.filePath, request.sheet)
        
        result = DA.analyzeCategoricalToContinuous_via_api(
            data=data,
            xheader=request.xheader,
            yheader=request.yheader,
            fixed_values=request.fixed_values,
            include_model_prediction=request.include_model_prediction,
            ret=request.ret,
            output_dir=request.output_dir,
            api_url=request.api_url,
            model_name=request.model_name,
            version=request.version,
            timeout=request.timeout
        )
        
        # 使用安全序列化處理 ret
        serializable_ret = safe_serialize(request.ret)
        
        return {
            "success": result,
            "message": "類別變數對連續變數分析完成（API 版本）" if result else "分析失敗",
            "xheader": request.xheader,
            "yheader": request.yheader,
            "api_url": request.api_url,
            "model_name": request.model_name,
            "version": request.version,
            "res": serializable_ret
        }
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['analyze_categorical_to_continuous_via_api'])
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-continuous-to-categorical-via-api")
async def analyze_continuous_to_categorical_via_api(request: AnalysisViaApiRequest):
    """連續變數對類別變數分析（API 版本）"""
    try:
        data = load_data(request.filePath, request.sheet)
        
        result = DA.analyzeContinuousToCategorical_via_api(
            data=data,
            xheader=request.xheader,
            yheader=request.yheader,
            fixed_values=request.fixed_values,
            ret=request.ret,
            output_dir=request.output_dir,
            api_url=request.api_url,
            model_name=request.model_name,
            version=request.version,
            timeout=request.timeout
        )
        
        # 使用安全序列化處理 ret
        serializable_ret = safe_serialize(request.ret)
        
        return {
            "success": result,
            "message": "連續變數對類別變數分析完成（API 版本）" if result else "分析失敗",
            "xheader": request.xheader,
            "yheader": request.yheader,
            "api_url": request.api_url,
            "model_name": request.model_name,
            "version": request.version,
            "res": serializable_ret
        }
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['analyze_continuous_to_categorical_via_api'])
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-categorical-to-categorical-via-api")
async def analyze_categorical_to_categorical_via_api(request: AnalysisViaApiRequest):
    """類別變數對類別變數分析（API 版本）"""
    try:
        data = load_data(request.filePath, request.sheet)
        
        result = DA.analyzeCategoricalToCategorical_via_api(
            data=data,
            xheader=request.xheader,
            yheader=request.yheader,
            fixed_values=request.fixed_values,
            ret=request.ret,
            output_dir=request.output_dir,
            api_url=request.api_url,
            model_name=request.model_name,
            version=request.version,
            timeout=request.timeout
        )
        
        # 使用安全序列化處理 ret
        serializable_ret = safe_serialize(request.ret)
        
        return {
            "success": result,
            "message": "類別變數對類別變數分析完成（API 版本）" if result else "分析失敗",
            "xheader": request.xheader,
            "yheader": request.yheader,
            "api_url": request.api_url,
            "model_name": request.model_name,
            "version": request.version,
            "res": serializable_ret
        }
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['analyze_categorical_to_categorical_via_api'])
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/two-dimension-analysis")
async def two_dimension_analysis(request: TwoDimensionAnalysisRequest):
    """二維度分析"""
    try:
        data = load_data(request.filePath, request.sheet)
        
        result = DA.twoDimensionAnalysis(
            data=data,
            xheader1=request.xheader1,
            xheader2=request.xheader2,
            yheader=request.yheader,
            fixed_values=request.fixed_values,
            include_model_prediction=request.include_model_prediction,
            ret=request.ret
        )
        
        return {
            "success": result,
            "message": "二維度分析完成" if result else "分析失敗",
            "xheader1": request.xheader1,
            "xheader2": request.xheader2,
            "yheader": request.yheader
        }
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['two_dimension_analysis'])
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-two-continuous-to-y")
async def analyze_two_continuous_to_y(request: TwoDimensionAnalysisRequest):
    """兩個連續變數對 Y 分析"""
    try:
        data = load_data(request.filePath, request.sheet)
        
        result = DA.analyzeTwoContinuousToY(
            data=data,
            xheader1=request.xheader1,
            xheader2=request.xheader2,
            yheader=request.yheader,
            y_type="continuous",  # 預設為連續變數
            fixed_values=request.fixed_values,
            include_model_prediction=request.include_model_prediction,
            ret=request.ret
        )
        
        return {
            "success": result,
            "message": "兩個連續變數對 Y 分析完成" if result else "分析失敗",
            "xheader1": request.xheader1,
            "xheader2": request.xheader2,
            "yheader": request.yheader
        }
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['analyze_two_continuous_to_y'])
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-mixed-to-y")
async def analyze_mixed_to_y(request: MixedAnalysisRequest):
    """混合變數對 Y 分析"""
    try:
        data = load_data(request.filePath, request.sheet)
        
        result = DA.analyzeMixedToY(
            data=data,
            xheader1=request.xheader1,
            xheader2=request.xheader2,
            yheader=request.yheader,
            x1_type=request.x1_type,
            x2_type=request.x2_type,
            y_type=request.y_type,
            fixed_values=request.fixed_values,
            include_model_prediction=request.include_model_prediction,
            ret=request.ret
        )
        
        return {
            "success": result,
            "message": "混合變數對 Y 分析完成" if result else "分析失敗",
            "xheader1": request.xheader1,
            "xheader2": request.xheader2,
            "yheader": request.yheader,
            "x1_type": request.x1_type,
            "x2_type": request.x2_type,
            "y_type": request.y_type
        }
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['analyze_mixed_to_y'])
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-two-categorical-to-y")
async def analyze_two_categorical_to_y(request: TwoDimensionAnalysisRequest):
    """兩個類別變數對 Y 分析"""
    try:
        data = load_data(request.filePath, request.sheet)
        
        result = DA.analyzeTwoCategoricalToY(
            data=data,
            xheader1=request.xheader1,
            xheader2=request.xheader2,
            yheader=request.yheader,
            y_type="categorical",  # 預設為類別變數
            fixed_values=request.fixed_values,
            include_model_prediction=request.include_model_prediction,
            ret=request.ret
        )
        
        return {
            "success": result,
            "message": "兩個類別變數對 Y 分析完成" if result else "分析失敗",
            "xheader1": request.xheader1,
            "xheader2": request.xheader2,
            "yheader": request.yheader
        }
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['analyze_two_categorical_to_y'])
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-two-continuous-to-y-via-api")
async def analyze_two_continuous_to_y_via_api(request: TwoDimensionAnalysisViaApiRequest):
    """兩個連續變數對 Y 分析（API 版本）"""
    try:
        data = load_data(request.filePath, request.sheet)
        
        result = DA.analyzeTwoContinuousToY_via_api(
            data=data,
            xheader1=request.xheader1,
            xheader2=request.xheader2,
            yheader=request.yheader,
            y_type=request.y_type or "continuous",
            fixed_values=request.fixed_values,
            include_model_prediction=request.include_model_prediction,
            ret=request.ret,
            output_dir=request.output_dir,
            api_url=request.api_url,
            model_name=request.model_name,
            version=request.version,
            timeout=request.timeout
        )
        
        # 使用安全序列化處理 ret
        serializable_ret = safe_serialize(request.ret)
        
        return {
            "success": result,
            "message": "兩個連續變數對 Y 分析完成（API 版本）" if result else "分析失敗",
            "xheader1": request.xheader1,
            "xheader2": request.xheader2,
            "yheader": request.yheader,
            "y_type": request.y_type or "continuous",
            "api_url": request.api_url,
            "model_name": request.model_name,
            "version": request.version,
            "res": serializable_ret
        }
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['analyze_two_continuous_to_y_via_api'])
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-mixed-to-y-via-api")
async def analyze_mixed_to_y_via_api(request: MixedAnalysisViaApiRequest):
    """混合變數對 Y 分析（API 版本）"""
    try:
        data = load_data(request.filePath, request.sheet)
        
        result = DA.analyzeMixedToY_via_api(
            data=data,
            xheader1=request.xheader1,
            xheader2=request.xheader2,
            yheader=request.yheader,
            x1_type=request.x1_type,
            x2_type=request.x2_type,
            y_type=request.y_type,
            fixed_values=request.fixed_values,
            include_model_prediction=request.include_model_prediction,
            ret=request.ret,
            output_dir=request.output_dir,
            api_url=request.api_url,
            model_name=request.model_name,
            version=request.version,
            timeout=request.timeout
        )
        
        # 使用安全序列化處理 ret
        serializable_ret = safe_serialize(request.ret)
        
        return {
            "success": result,
            "message": "混合變數對 Y 分析完成（API 版本）" if result else "分析失敗",
            "xheader1": request.xheader1,
            "xheader2": request.xheader2,
            "yheader": request.yheader,
            "x1_type": request.x1_type,
            "x2_type": request.x2_type,
            "y_type": request.y_type,
            "api_url": request.api_url,
            "model_name": request.model_name,
            "version": request.version,
            "res": serializable_ret
        }
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['analyze_mixed_to_y_via_api'])
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-two-categorical-to-y-via-api")
async def analyze_two_categorical_to_y_via_api(request: TwoDimensionAnalysisViaApiRequest):
    """兩個類別變數對 Y 分析（API 版本）"""
    try:
        data = load_data(request.filePath, request.sheet)
        
        result = DA.analyzeTwoCategoricalToY_via_api(
            data=data,
            xheader1=request.xheader1,
            xheader2=request.xheader2,
            yheader=request.yheader,
            y_type=request.y_type or "categorical",
            fixed_values=request.fixed_values,
            include_model_prediction=request.include_model_prediction,
            ret=request.ret,
            output_dir=request.output_dir,
            api_url=request.api_url,
            model_name=request.model_name,
            version=request.version,
            timeout=request.timeout
        )
        
        # 使用安全序列化處理 ret
        serializable_ret = safe_serialize(request.ret)
        
        return {
            "success": result,
            "message": "兩個類別變數對 Y 分析完成（API 版本）" if result else "分析失敗",
            "xheader1": request.xheader1,
            "xheader2": request.xheader2,
            "yheader": request.yheader,
            "y_type": request.y_type or "categorical",
            "api_url": request.api_url,
            "model_name": request.model_name,
            "version": request.version,
            "res": serializable_ret
        }
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['analyze_two_categorical_to_y_via_api'])
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/standard-tests")
async def standard_tests(request: StandardTestsRequest):
    """標準檢定分析"""
    try:
        data = load_data(request.filePath, request.sheet)
        
        # 建立 handler 物件
        class Handler:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        handler = Handler(
            file_path=request.filePath,
            column=request.column,
            group_by=request.group_by,
            sheet=request.sheet,
            alpha=request.alpha,
            output_dir=request.output_dir or os.path.join('tmp', 'distribution_results'),
            use_api=request.use_api,
            plan_id=request.plan_id,
            seq_no=request.seq_no
        )
        
        # 執行標準檢定
        DA.standardTests(handler)
        
        return {
            "success": True,
            "message": "標準檢定分析完成",
            "column": request.column,
            "group_by": request.group_by,
            "output_directory": handler.output_dir
        }
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['standard_tests'])
        raise HTTPException(status_code=500, detail=str(e))


def _instability_resolve_output_dir(request: InstabilityBaseRequest) -> str:
    # TODO: 與 config 合併、子目錄命名（job_id 等）
    return request.output_dir or m_instability_api_output_default


@app.post("/instability/compute")
async def instability_compute(request: InstabilityComputeRequest):
    """
    不穩定度計算，主體回 JSON（issues/不穩定度計算功能API化.iss）。
    TODO: 將 ret 與 heteroscedastic 參數（離散閾值、norm、tol、density_shrink 等）對齊。
    """
    try:
        df = load_data(request.filePath, request.sheet)
        payload = het_instability.compute_instability_payload(
            df,
            request.x_col,
            request.y_col,
            layers=request.layers,
            extra=request.ret,
        )
        return {
            "success": True,
            "message": "不穩定度計算（骨架）",
            "filePath": request.filePath,
            "x_col": request.x_col,
            "y_col": request.y_col,
            "layers": request.layers,
            "res": safe_serialize(payload),
        }
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['instability_compute'])
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/instability/save")
async def instability_save(request: InstabilitySaveRequest):
    """不穩定度結果寫檔（與 compute / plot 分離）。"""
    try:
        df = load_data(request.filePath, request.sheet)
        out_dir = _instability_resolve_output_dir(request)
        # TODO: os.makedirs(out_dir, exist_ok=True)
        saved = het_instability.save_instability_to_disk(
            df,
            request.x_col,
            request.y_col,
            output_base_dir=out_dir,
            layers=request.layers,
            extra=request.ret,
        )
        return {
            "success": True,
            "message": "不穩定度存檔（骨架）",
            "output_dir": out_dir,
            "res": safe_serialize(saved),
        }
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['instability_save'])
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/instability/plot")
async def instability_plot(request: InstabilityPlotRequest):
    """不穩定度繪圖（與 compute / save 分離）。"""
    try:
        df = load_data(request.filePath, request.sheet)
        out_dir = _instability_resolve_output_dir(request)
        plotted = het_instability.plot_instability_figures(
            df,
            request.x_col,
            request.y_col,
            output_base_dir=out_dir,
            layers=request.layers,
            extra=request.ret,
        )
        return {
            "success": True,
            "message": "不穩定度繪圖（骨架）",
            "output_dir": out_dir,
            "res": safe_serialize(plotted),
        }
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['instability_plot'])
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data/points")
async def get_points(request: PointsRequest):
    """
    回傳前端繪圖用的資料點（2D/3D），支援：
    - 指定 columns（必要）
    - fixed_values 條件過濾（可選）
    - sample_n 抽樣（避免點太多）
    - include_extra_columns 額外欄位，供 hover/click 顯示更多特徵
    """
    try:
        df = load_data(request.filePath, request.sheet)

        cols = list(dict.fromkeys(request.columns + (request.include_extra_columns or [])))
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"欄位不存在: {missing[:10]}")

        sub = df[cols].copy()

        # fixed_values filter
        if isinstance(request.fixed_values, dict) and request.fixed_values:
            for k, v in request.fixed_values.items():
                if k in sub.columns:
                    sub = sub[sub[k] == v]

        # drop rows with NaN in required columns
        required = request.columns
        sub = sub.dropna(subset=required)

        # sampling
        n = int(request.sample_n or 2000)
        if n > 0 and len(sub) > n:
            rng = np.random.default_rng(int(request.seed or 7))
            idx = rng.choice(len(sub), size=n, replace=False)
            sub = sub.iloc[idx]

        # to jsonable
        # - 將 numpy/pandas 型別轉成 python 基本型別，避免 JSON 序列化問題
        rows = []
        for i, row in sub.iterrows():
            d = {}
            for c in cols:
                val = row[c]
                if pd.isna(val):
                    d[c] = None
                elif isinstance(val, (np.integer,)):
                    d[c] = int(val)
                elif isinstance(val, (np.floating,)):
                    d[c] = float(val)
                else:
                    d[c] = str(val) if isinstance(val, (pd.Timestamp,)) else val
            d["_index"] = int(i) if isinstance(i, (int, np.integer)) else str(i)
            rows.append(d)

        return {
            "success": True,
            "count": len(rows),
            "required_columns": request.columns,
            "columns": cols,
            "rows": rows
        }
    except HTTPException:
        raise
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['get_points'])
        raise HTTPException(status_code=500, detail=str(e))


class EvaluationFitRequest(BaseModel):
    plan_id: int = Field(..., gt=0, description="計劃序號，必須為正整數")
    seq_no: int = Field(1, gt=0, description="序列號，必須為正整數，預設為 1")
    stamps: Optional[List[str]] = Field(None, description="額外的標記列表")
    output_dir: Optional[str] = Field(None, description="輸出目錄，可選")
    provider: Optional[str] = Field(None, description="LLM provider，預設為None，後面系統自動帶")
    model: Optional[str] = Field("remote8b", description="模型名稱，預設為 'remote8b'")
    user_prompt_template: Optional[str] = Field(None, description="用戶提示模板，可選")
    system_prompt: Optional[str] = Field(None, description="系統提示模板，可選")
    prompt_temperature: Optional[float] = Field(None, description="提示溫度，可選")
    personality: Optional[str] = Field(None, description="LLM 回復個性描述，可選。例如：'你是一位專業的數據分析師，要用精準專業的詞彙進行分析。'")
    enable_llm_analysis: Optional[bool] = Field(True, description="是否進行 LLM 分析，預設為 True")
    fH_prompt_alias_level: Optional[int] = Field(3, description="個性的使用等級，預設為2；初步有1~3三種選項，越高越初階")

@app.post("/evaluationPlanTask")
async def evaluationPlanTask(request: EvaluationFitRequest):
    """
    評估模型適配性
    """
    try:
        result = {}
        # 處理 stamps：如果為 None，則使用空列表
        base_stamp = f'Proj_{request.plan_id}-{request.seq_no}'
        stamps = [base_stamp] + (request.stamps if request.stamps else [])
        if not LOGger.isinstance_not_empty(request.output_dir, str):
            output_dir = os.path.join(m_ExportSetPath, str(request.plan_id), str(request.seq_no))
        if not ae.evaluationFit(request.plan_id, request.seq_no, ret=result, stamps=stamps, output_dir=output_dir):
            error_message = result.get('message', '未知錯誤')
            raise HTTPException(status_code=500, detail=f"評估模型適配性失敗: {error_message}")
        
        return {
            "success": True,
            "message": "評估模型適配性完成",
            "result": result
        }
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['evaluationPlanTask'])
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluationPlanTaskLLM")
async def evaluationFitAnalysis(request: EvaluationFitRequest):
    """
    LLM評估模型適配性分析
    """
    try:
        result = {}
        # 處理 stamps：如果為 None，則使用空列表
        base_stamp = f'Proj_{request.plan_id}-{request.seq_no}'
        stamps = [base_stamp] + (request.stamps if request.stamps else [])
        if not LOGger.isinstance_not_empty(request.output_dir, str):
            output_dir = os.path.join(m_ExportSetPath, str(request.plan_id), str(request.seq_no), "Analysis")
        if not ae.evaluationFitAnalysis(request.plan_id, request.seq_no, ret=result, stamps=stamps, output_dir=output_dir, provider=request.provider, model=request.model, user_prompt_template=request.user_prompt_template, system_prompt=request.system_prompt, prompt_temperature=request.prompt_temperature, personality=request.personality, enable_llm_analysis=request.enable_llm_analysis, fH_prompt_alias_level=request.fH_prompt_alias_level, md_output_dir=output_dir):
            error_message = result.get('message', '未知錯誤')
            raise HTTPException(status_code=500, detail=f"LLM評估模型適配性分析失敗: {error_message}")
        
        # 提取 token 和費用信息
        token_summary = result.get('token_summary')
        billing_summary = result.get('billing_summary')
        
        return {
            "success": True,
            "message": "LLM評估模型適配性分析完成",
            "result": result,
            "token_summary": token_summary,
            "billing_summary": billing_summary
        }
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['evaluationPlanTaskLLM'])
        raise HTTPException(status_code=500, detail=str(e))


# 檔案預覽和轉換相關端點
class FilePreviewRequest(BaseModel):
    filePath: str
    sheet: Optional[int] = 0  # xlsx 檔案需要指定工作表
    preview_rows: Optional[int] = 5  # 預覽行數，預設5行

class FileToPklRequest(BaseModel):
    filePath: str
    sheet: Optional[int] = 0  # xlsx 檔案需要指定工作表
    protocol: Optional[int] = 4  # pickle protocol 版本
    output_dir: Optional[str] = "apiOutput"  # 輸出目錄，預設 apiOutput
    output_path: Optional[str] = None  # 輸出檔案路徑（相對於 output_dir），None 表示自動生成

class PklProtocolConvertRequest(BaseModel):
    filePath: str
    protocol: Optional[int] = 4  # 目標 pickle protocol 版本
    output_dir: Optional[str] = "apiOutput"  # 輸出目錄，預設 apiOutput
    output_path: Optional[str] = None  # 輸出檔案路徑（相對於 output_dir），None 表示覆蓋原檔案名

@app.post("/file/preview")
async def preview_file(request: FilePreviewRequest):
    """
    偵測檔案欄位並預覽前N項資料
    
    - 支援 .xlsx, .xls, .csv, .pkl 格式
    - 如果是 xlsx 需要有 sheet 參數
    - 值使用 DFP.parse(x, digit=4) 轉成字串並截取前20字元
    """
    try:
        import pandas as pd
        import pickle
        
        # 解析檔案路徑
        if not os.path.isabs(request.filePath):
            source_dir = config.get('source_dir', '.')
            full_path = os.path.join(source_dir, request.filePath)
            if not os.path.exists(full_path):
                reference_dirs = config.get('referenceDirs', [])
                for ref_dir in reference_dirs:
                    test_path = os.path.join(ref_dir, request.filePath)
                    if os.path.exists(test_path):
                        full_path = test_path
                        break
                else:
                    raise HTTPException(status_code=404, detail=f"檔案不存在: {request.filePath}")
        else:
            full_path = request.filePath
        
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail=f"檔案不存在: {full_path}")
        
        # 檢查檔案類型
        file_ext = os.path.splitext(full_path)[1].lower()
        if file_ext not in ['.xlsx', '.xls', '.csv', '.pkl']:
            raise HTTPException(status_code=400, detail=f"不支援的檔案格式: {file_ext}，僅支援 .xlsx, .xls, .csv, .pkl")
        
        # 載入資料
        if file_ext in ['.xlsx', '.xls', '.pkl']:
            data = DFP.import_data(full_path, sht=request.sheet)
        else:  # .csv
            data = DFP.import_data(full_path)
        
        if data is None or data.empty:
            raise HTTPException(status_code=400, detail="無法載入資料或資料為空")
        
        print(data.head())
        # 預覽資料處理函數
        def format_value(x):
            """格式化值：使用 DFP.parse 轉成字串，digit=4，並截取前20字元"""
            try:
                parsed = DFP.parse(x, digit=4)
                result = str(parsed)[:20]
                return result
            except:
                return str(x)[:20] if x is not None else ''

        # 欄名處理函數：避免 NaN/Inf 欄名導致 JSON 編碼失敗
        def format_column_name(col):
            try:
                if isinstance(col, (float, np.floating)):
                    if math.isnan(col):
                        return "<nan_column>"
                    if math.isinf(col):
                        return "<inf_column>"
                    return str(float(col))
                return str(col)
            except Exception:
                return "<column_error>"
        
        # 預覽前 N 行
        preview_count = min(request.preview_rows, len(data))
        preview_data = data.head(preview_count)
        
        # 格式化預覽資料
        preview_formatted = {}
        for col in preview_data.columns:
            col_name = format_column_name(col)
            preview_formatted[col_name] = [
                format_value(val) for val in preview_data[col].head(preview_count).tolist()
            ]
        
        # 欄位資訊
        columns_info = []
        for col in data.columns:
            col_info = {
                'name': format_column_name(col),
                'dtype': str(data[col].dtype),
                'non_null_count': int(data[col].notna().sum()),
                'null_count': int(data[col].isna().sum()),
                'unique_count': int(data[col].nunique()),
                'sample_values': [format_value(val) for val in data[col].dropna().head(3).tolist()]
            }
            columns_info.append(col_info)

        response_data = {
            "success": True,
            "file_path": full_path,
            "file_type": file_ext,
            "shape": {
                "rows": int(data.shape[0]),
                "columns": int(data.shape[1])
            },
            "columns": columns_info,
            "preview": {
                "rows_shown": preview_count,
                "data": preview_formatted
            }
        }
        return safe_serialize(response_data, max_depth=8)
        
    except HTTPException:
        raise
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['preview_file'])
        raise HTTPException(status_code=500, detail=f"預覽檔案失敗: {str(e)}")

@app.post("/file/convert-to-pkl")
async def convert_file_to_pkl(request: FileToPklRequest):
    """
    將 xlsx 或 csv 檔案轉換為 pkl 檔案
    
    - 支援 .xlsx, .xls, .csv 格式
    - xlsx 需要指定 sheet 參數
    - protocol 預設為 4
    """
    try:
        import pickle
        import pandas as pd
        
        # 解析檔案路徑
        if not os.path.isabs(request.filePath):
            source_dir = config.get('source_dir', '.')
            full_path = os.path.join(source_dir, request.filePath)
            if not os.path.exists(full_path):
                reference_dirs = config.get('referenceDirs', [])
                for ref_dir in reference_dirs:
                    test_path = os.path.join(ref_dir, request.filePath)
                    if os.path.exists(test_path):
                        full_path = test_path
                        break
                else:
                    raise HTTPException(status_code=404, detail=f"檔案不存在: {request.filePath}")
        else:
            full_path = request.filePath
        
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail=f"檔案不存在: {full_path}")
        
        # 檢查檔案類型
        file_ext = os.path.splitext(full_path)[1].lower()
        if file_ext not in ['.xlsx', '.xls', '.csv']:
            raise HTTPException(status_code=400, detail=f"不支援的檔案格式: {file_ext}，僅支援 .xlsx, .xls, .csv")
        
        # 載入資料
        if file_ext in ['.xlsx', '.xls']:
            data = DFP.import_data(full_path, sht=request.sheet)
        else:
            data = DFP.import_data(full_path)
        
        if data is None or data.empty:
            raise HTTPException(status_code=400, detail="無法載入資料或資料為空")
        
        # 決定輸出路徑
        output_dir = request.output_dir if request.output_dir else "apiOutput"
        
        if request.output_path:
            # 如果指定了 output_path
            if os.path.isabs(request.output_path):
                # 絕對路徑，直接使用
                output_path = request.output_path
            else:
                # 相對路徑，相對於 output_dir
                output_path = os.path.join(output_dir, request.output_path)
        else:
            # 自動生成：將原檔名改為 .pkl，放在 output_dir 下
            base_name = os.path.splitext(os.path.basename(full_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}.pkl")
        
        # 確保輸出目錄存在
        output_dir_path = os.path.dirname(output_path)
        if output_dir_path and not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path, exist_ok=True)
        
        # 儲存為 pkl 檔案
        with open(output_path, 'wb') as f:
            pickle.dump(data, f, protocol=request.protocol)
        
        return {
            "success": True,
            "message": f"檔案已成功轉換為 pkl 格式",
            "source_file": full_path,
            "output_file": output_path,
            "protocol": request.protocol,
            "shape": {
                "rows": int(data.shape[0]),
                "columns": int(data.shape[1])
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['convert_file_to_pkl'])
        raise HTTPException(status_code=500, detail=f"轉換檔案失敗: {str(e)}")

@app.post("/file/convert-pkl-protocol")
async def convert_pkl_protocol(request: PklProtocolConvertRequest):
    """
    轉換 pkl 檔案的 protocol 版本
    
    - 僅支援 .pkl 檔案
    - protocol 預設為 4
    - 其他格式檔案會被拒絕
    """
    try:
        import pickle
        
        # 解析檔案路徑
        if not os.path.isabs(request.filePath):
            source_dir = config.get('source_dir', '.')
            full_path = os.path.join(source_dir, request.filePath)
            if not os.path.exists(full_path):
                reference_dirs = config.get('referenceDirs', [])
                for ref_dir in reference_dirs:
                    test_path = os.path.join(ref_dir, request.filePath)
                    if os.path.exists(test_path):
                        full_path = test_path
                        break
                else:
                    raise HTTPException(status_code=404, detail=f"檔案不存在: {request.filePath}")
        else:
            full_path = request.filePath
        
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail=f"檔案不存在: {full_path}")
        
        # 檢查檔案類型（只接受 .pkl）
        file_ext = os.path.splitext(full_path)[1].lower()
        if file_ext != '.pkl':
            raise HTTPException(status_code=400, detail=f"僅支援 .pkl 檔案，收到: {file_ext}")
        
        # 讀取 pkl 檔案
        try:
            with open(full_path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            LOGger.exception_process(e, logfile='', stamps=['convert_pkl_protocol'])
            raise HTTPException(status_code=400, detail=f"無法讀取 pkl 檔案: {str(e)}")
        
        # 決定輸出路徑
        output_dir = request.output_dir if request.output_dir else "apiOutput"
        
        if request.output_path:
            # 如果指定了 output_path
            if os.path.isabs(request.output_path):
                # 絕對路徑，直接使用
                output_path = request.output_path
            else:
                # 相對路徑，相對於 output_dir
                output_path = os.path.join(output_dir, request.output_path)
        else:
            # 自動生成：保持原檔名，放在 output_dir 下
            base_name = os.path.basename(full_path)
            output_path = os.path.join(output_dir, base_name)
        
        # 確保輸出目錄存在
        output_dir_path = os.path.dirname(output_path)
        if output_dir_path and not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path, exist_ok=True)
        
        # 以新 protocol 儲存
        with open(output_path, 'wb') as f:
            pickle.dump(data, f, protocol=request.protocol)
        
        # 獲取資料形狀（如果是 DataFrame）
        shape_info = None
        if hasattr(data, 'shape'):
            shape_info = {
                "rows": int(data.shape[0]),
                "columns": int(data.shape[1]) if len(data.shape) > 1 else int(data.shape[0])
            }
        
        return {
            "success": True,
            "message": f"pkl 檔案 protocol 已轉換為 {request.protocol}",
            "source_file": full_path,
            "output_file": output_path,
            "protocol": request.protocol,
            "shape": shape_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=['convert_pkl_protocol'])
        raise HTTPException(status_code=500, detail=f"轉換 pkl protocol 失敗: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # 添加離散數據聚合端點
    try:
        from src.api_integration import add_discrete_aggregation_endpoints
        add_discrete_aggregation_endpoints(app)
        print("✓ 離散數據聚合端點已添加")
    except ImportError as e:
        print(f"⚠ 離散數據聚合功能載入失敗: {e}")
    
    # 從配置檔案取得主機和埠號設定
    host = config.get("Host_IP", "127.0.0.1")
    port = config.get("Host_Port", 8000)
    
    print(f"正在啟動 DataAnalysis API 服務...")
    print(f"服務地址: http://{host}:{port}")
    print(f"API 文檔: http://{host}:{port}/docs")
    print(f"配置資訊: http://{host}:{port}/config")
    print(f"來源目錄: {config.get('source_dir', '.')}")
    print(f"參考目錄: {config.get('referenceDirs', [])}")
    
    uvicorn.run(app, host=host, port=port)
