import os
import json
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from mcp.server.fastmcp import FastMCP


API_BASE = os.getenv("DA_API_BASE", "http://10.3.1.127:6030").rstrip("/")
API_TIMEOUT = float(os.getenv("DA_API_TIMEOUT", "600"))

mcp = FastMCP("dataanalysis-api")

_LOG_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "WARNING": 30, "ERROR": 40}
_LOG_ENABLED = os.getenv("DA_MCP_LOG", "1").strip().lower() not in ("0", "false", "no", "off")
_LOG_LEVEL = os.getenv("DA_MCP_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO")).strip().upper()
_LOG_LEVEL_NUM = _LOG_LEVELS.get(_LOG_LEVEL, 20)


def _log(level: str, msg: str) -> None:
    # MCP stdio transport uses stdout for protocol messages.
    # Always write debug/info logs to stderr to avoid breaking the client.
    if not _LOG_ENABLED:
        return
    level_u = (level or "INFO").strip().upper()
    if _LOG_LEVELS.get(level_u, 20) < _LOG_LEVEL_NUM:
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {level_u} DA-MCP: {msg}", file=sys.stderr, flush=True)


class ApiError(RuntimeError):
    pass


def _url(path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    return f"{API_BASE}{path}"


def _request(method: str, path: str, payload: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = _url(path)
    started = time.monotonic()
    _log(
        "DEBUG",
        f"HTTP {method.upper()} {url} payload_keys={list((payload or {}).keys())} params_keys={list((params or {}).keys())}",
    )
    try:
        resp = requests.request(
            method=method.upper(),
            url=url,
            json=payload,
            params=params,
            timeout=API_TIMEOUT,
        )
    except requests.RequestException as e:
        elapsed = time.monotonic() - started
        _log("ERROR", f"HTTP {method.upper()} {url} request failed after {elapsed:.2f}s: {e}")
        raise ApiError(f"無法連線到 DataAnalysis API: {url}\n{e}") from e

    elapsed = time.monotonic() - started
    _log("DEBUG", f"HTTP {method.upper()} {url} -> {resp.status_code} in {elapsed:.2f}s")

    content_type = resp.headers.get("content-type", "")
    if not resp.ok:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        detail_preview = str(detail)
        if len(detail_preview) > 500:
            detail_preview = detail_preview[:500] + "..."
        _log("ERROR", f"HTTP {method.upper()} {url} -> {resp.status_code} {resp.reason}; detail={detail_preview}")
        raise ApiError(f"API 呼叫失敗: {resp.status_code} {resp.reason}\nURL: {url}\n回應: {detail}")

    if "application/json" in content_type:
        return resp.json()
    return {
        "success": True,
        "status_code": resp.status_code,
        "content_type": content_type,
        "text": resp.text,
    }


def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return _request("POST", path, payload=payload)


def _get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return _request("GET", path, params=params)


@mcp.tool()
def health() -> Dict[str, Any]:
    """檢查 DataAnalysis API 是否可連線，並回傳根路由資訊。"""
    return _get("/")


@mcp.tool()
def get_config() -> Dict[str, Any]:
    """取得 api_server.py 的服務配置資訊。"""
    return _get("/config")


@mcp.tool()
def list_files(directory: Optional[str] = None) -> Dict[str, Any]:
    """列出可瀏覽的目錄、子目錄與檔案。directory 可為相對路徑、絕對路徑，或空字串表示 root。"""
    params: Dict[str, Any] = {}
    if directory is not None:
        params["directory"] = directory
    return _get("/files", params=params)


@mcp.tool()
def search_file(filename: str) -> Dict[str, Any]:
    """依檔名關鍵字搜尋 source_dir / referenceDirs 中的檔案。"""
    return _get("/search-file", params={"filename": filename})


@mcp.tool()
def preview_file(file_path: str, sheet: int = 0, preview_rows: int = 5) -> Dict[str, Any]:
    """預覽 xlsx/xls/csv/pkl 的欄位、型別、前幾列樣本。"""
    return _post(
        "/file/preview",
        {
            "filePath": file_path,
            "sheet": sheet,
            "preview_rows": preview_rows,
        },
    )


@mcp.tool()
def convert_file_to_pkl(
    file_path: str,
    sheet: int = 0,
    protocol: int = 4,
    output_dir: str = "apiOutput",
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """把 xlsx/xls/csv 轉成 pkl。"""
    payload: Dict[str, Any] = {
        "filePath": file_path,
        "sheet": sheet,
        "protocol": protocol,
        "output_dir": output_dir,
    }
    if output_path:
        payload["output_path"] = output_path
    return _post("/file/convert-to-pkl", payload)


@mcp.tool()
def convert_pkl_protocol(
    file_path: str,
    protocol: int = 4,
    output_dir: str = "apiOutput",
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """轉換既有 pkl 的 protocol 版本。"""
    payload: Dict[str, Any] = {
        "filePath": file_path,
        "protocol": protocol,
        "output_dir": output_dir,
    }
    if output_path:
        payload["output_path"] = output_path
    return _post("/file/convert-pkl-protocol", payload)


@mcp.tool()
def calculate_correlation(
    file_path: str,
    sheet: int = 0,
    method: str = "mic",
    exp_fd: str = "micCorr",
    stamps: Optional[List[str]] = None,
    data_config_file: Optional[str] = None,
) -> Dict[str, Any]:
    """呼叫 /calculate-correlation 計算相關性，例如 MIC。"""
    payload: Dict[str, Any] = {
        "filePath": file_path,
        "sheet": sheet,
        "method": method,
        "exp_fd": exp_fd,
        "ret": {},
    }
    if stamps is not None:
        payload["stamps"] = stamps
    if data_config_file:
        payload["data_config_file"] = data_config_file
    return _post("/calculate-correlation", payload)


@mcp.tool()
def plot_correlation(
    file_path: str,
    sheet: int = 0,
    method: str = "mic",
    exp_fd: str = "micCorr",
    stamps: Optional[List[str]] = None,
    data_config_file: Optional[str] = None,
    async_job: bool = True,
    timeout_sec: Optional[float] = None,
    n_jobs: Optional[int] = None,
    is_aggregate: bool = False,
    spill_to_disk: bool = False,
    spill_backend: str = "pkl",
    spill_keep_files: bool = False,
) -> Dict[str, Any]:
    """呼叫 /plot-correlation 產生相關性圖。預設非阻塞：回傳 job_id；async_job=False 則同步等完成。"""
    payload: Dict[str, Any] = {
        "filePath": file_path,
        "sheet": sheet,
        "method": method,
        "exp_fd": exp_fd,
        "ret": {},
        "async_job": async_job,
        "spill_to_disk": spill_to_disk,
        "spill_backend": spill_backend,
        "spill_keep_files": spill_keep_files,
    }
    if stamps is not None:
        payload["stamps"] = stamps
    if data_config_file:
        payload["data_config_file"] = data_config_file
    if timeout_sec is not None:
        payload["timeout_sec"] = timeout_sec
    if n_jobs is not None:
        payload["n_jobs"] = n_jobs
    payload["isAggregate"] = bool(is_aggregate)
    return _post("/plot-correlation", payload)


@mcp.tool()
def plot_correlation_status(job_id: str) -> Dict[str, Any]:
    """查詢 /plot-correlation 背景任務狀態與 output_files。"""
    return _get(f"/plot-correlation/status/{job_id}")


@mcp.tool()
def plot_correlation_cancel(job_id: str) -> Dict[str, Any]:
    """中止 /plot-correlation 背景任務（最佳努力）。"""
    return _post(f"/plot-correlation/jobs/{job_id}/cancel", {})


@mcp.tool()
def data_parsing(file_path: str, sheet: int = 0, exp_fd: str = "micCorr") -> Dict[str, Any]:
    """僅做資料解析，回傳解析後的 metadata。"""
    return _post(
        "/data-parsing",
        {
            "filePath": file_path,
            "sheet": sheet,
            "exp_fd": exp_fd,
            "ret": {},
        },
    )


@mcp.tool()
def dppc(
    file_path: str,
    sheet: int = 0,
    method: str = "mic",
    exp_fd: str = "micCorr",
    select_header: Optional[List[str]] = None,
    is_aggregate: bool = True,
    stamps: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """DataParsingAndPlotCorrelation：可選欄位、可做離散聚合，再繪製相關性圖。"""
    payload: Dict[str, Any] = {
        "filePath": file_path,
        "sheet": sheet,
        "method": method,
        "exp_fd": exp_fd,
        "ret": {},
        "isAggregate": is_aggregate,
    }
    if select_header is not None:
        payload["selectHeader"] = select_header
    if stamps is not None:
        payload["stamps"] = stamps
    return _post("/dppc", payload)


@mcp.tool()
def get_points(
    file_path: str,
    columns: List[str],
    sheet: int = 0,
    fixed_values: Optional[Dict[str, Any]] = None,
    sample_n: int = 2000,
    seed: int = 7,
    include_extra_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """取得前端 2D/3D 視覺化用資料點。"""
    payload: Dict[str, Any] = {
        "filePath": file_path,
        "sheet": sheet,
        "columns": columns,
        "sample_n": sample_n,
        "seed": seed,
    }
    if fixed_values is not None:
        payload["fixed_values"] = fixed_values
    if include_extra_columns is not None:
        payload["include_extra_columns"] = include_extra_columns
    return _post("/data/points", payload)


@mcp.tool()
def analyze_continuous_to_continuous(
    file_path: str,
    xheader: str,
    yheader: str,
    sheet: int = 0,
    fixed_values: Optional[Dict[str, Any]] = None,
    include_model_prediction: bool = True,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """連續 X 對連續 Y 分析。"""
    payload: Dict[str, Any] = {
        "filePath": file_path,
        "sheet": sheet,
        "xheader": xheader,
        "yheader": yheader,
        "fixed_values": fixed_values,
        "include_model_prediction": include_model_prediction,
        "ret": {},
        "output_dir": output_dir,
    }
    return _post("/analyze-continuous-to-continuous", payload)


@mcp.tool()
def analyze_categorical_to_continuous(
    file_path: str,
    xheader: str,
    yheader: str,
    sheet: int = 0,
    fixed_values: Optional[Dict[str, Any]] = None,
    include_model_prediction: bool = True,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """類別 X 對連續 Y 分析。"""
    payload: Dict[str, Any] = {
        "filePath": file_path,
        "sheet": sheet,
        "xheader": xheader,
        "yheader": yheader,
        "fixed_values": fixed_values,
        "include_model_prediction": include_model_prediction,
        "ret": {},
        "output_dir": output_dir,
    }
    return _post("/analyze-categorical-to-continuous", payload)


@mcp.tool()
def analyze_continuous_to_categorical(
    file_path: str,
    xheader: str,
    yheader: str,
    sheet: int = 0,
    fixed_values: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """連續 X 對類別 Y 分析。"""
    payload: Dict[str, Any] = {
        "filePath": file_path,
        "sheet": sheet,
        "xheader": xheader,
        "yheader": yheader,
        "fixed_values": fixed_values,
        "ret": {},
        "output_dir": output_dir,
    }
    return _post("/analyze-continuous-to-categorical", payload)


@mcp.tool()
def analyze_categorical_to_categorical(
    file_path: str,
    xheader: str,
    yheader: str,
    sheet: int = 0,
    fixed_values: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """類別 X 對類別 Y 分析。"""
    payload: Dict[str, Any] = {
        "filePath": file_path,
        "sheet": sheet,
        "xheader": xheader,
        "yheader": yheader,
        "fixed_values": fixed_values,
        "ret": {},
    }
    return _post("/analyze-categorical-to-categorical", payload)


@mcp.tool()
def analyze_two_continuous_to_y(
    file_path: str,
    xheader1: str,
    xheader2: str,
    yheader: str,
    sheet: int = 0,
    fixed_values: Optional[Dict[str, Any]] = None,
    include_model_prediction: bool = True,
) -> Dict[str, Any]:
    """兩個連續 X 對 Y 分析。"""
    return _post(
        "/analyze-two-continuous-to-y",
        {
            "filePath": file_path,
            "sheet": sheet,
            "xheader1": xheader1,
            "xheader2": xheader2,
            "yheader": yheader,
            "fixed_values": fixed_values,
            "include_model_prediction": include_model_prediction,
            "ret": {},
        },
    )


@mcp.tool()
def analyze_mixed_to_y(
    file_path: str,
    xheader1: str,
    xheader2: str,
    yheader: str,
    x1_type: str,
    x2_type: str,
    y_type: str,
    sheet: int = 0,
    fixed_values: Optional[Dict[str, Any]] = None,
    include_model_prediction: bool = True,
) -> Dict[str, Any]:
    """混合型 X1/X2 對 Y 分析。"""
    return _post(
        "/analyze-mixed-to-y",
        {
            "filePath": file_path,
            "sheet": sheet,
            "xheader1": xheader1,
            "xheader2": xheader2,
            "yheader": yheader,
            "x1_type": x1_type,
            "x2_type": x2_type,
            "y_type": y_type,
            "fixed_values": fixed_values,
            "include_model_prediction": include_model_prediction,
            "ret": {},
        },
    )


@mcp.tool()
def analyze_two_categorical_to_y(
    file_path: str,
    xheader1: str,
    xheader2: str,
    yheader: str,
    sheet: int = 0,
    fixed_values: Optional[Dict[str, Any]] = None,
    include_model_prediction: bool = True,
) -> Dict[str, Any]:
    """兩個類別 X 對 Y 分析。"""
    return _post(
        "/analyze-two-categorical-to-y",
        {
            "filePath": file_path,
            "sheet": sheet,
            "xheader1": xheader1,
            "xheader2": xheader2,
            "yheader": yheader,
            "fixed_values": fixed_values,
            "include_model_prediction": include_model_prediction,
            "ret": {},
        },
    )


@mcp.tool()
def standard_tests(
    file_path: str,
    column: str,
    sheet: int = 0,
    group_by: Optional[str] = None,
    alpha: float = 0.05,
    output_dir: Optional[str] = None,
    use_api: bool = False,
    plan_id: Optional[str] = None,
    seq_no: Optional[str] = None,
    tol: Optional[float] = None,
    spec_mode: str = "tost",
    confidence: float = 0.95,
    p_adjust: str = "holm",
) -> Dict[str, Any]:
    """執行標準檢定分析。tol 有值時一併傳規格參數至 /standard-tests（issues/檢定容入規格判斷.iss）。"""
    payload: Dict[str, Any] = {
        "filePath": file_path,
        "sheet": sheet,
        "column": column,
        "alpha": alpha,
        "use_api": use_api,
    }
    if group_by is not None:
        payload["group_by"] = group_by
    if output_dir is not None:
        payload["output_dir"] = output_dir
    if plan_id is not None:
        payload["plan_id"] = plan_id
    if seq_no is not None:
        payload["seq_no"] = seq_no
    if tol is not None:
        payload["tol"] = tol
        payload["spec_mode"] = spec_mode
        payload["confidence"] = confidence
        payload["p_adjust"] = p_adjust
    return _post("/standard-tests", payload)


@mcp.tool()
def instability_compute(
    file_path: str,
    x_col: str,
    y_col: str,
    sheet: int = 0,
    layers: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """計算不穩定度，回傳 JSON。"""
    payload: Dict[str, Any] = {
        "filePath": file_path,
        "sheet": sheet,
        "x_col": x_col,
        "y_col": y_col,
        "ret": {},
    }
    if layers is not None:
        payload["layers"] = layers
    return _post("/instability/compute", payload)


@mcp.tool()
def instability_save(
    file_path: str,
    x_col: str,
    y_col: str,
    sheet: int = 0,
    output_dir: Optional[str] = None,
    layers: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """計算並寫出不穩定度結果。"""
    payload: Dict[str, Any] = {
        "filePath": file_path,
        "sheet": sheet,
        "x_col": x_col,
        "y_col": y_col,
        "ret": {},
        "output_dir": output_dir,
    }
    if layers is not None:
        payload["layers"] = layers
    return _post("/instability/save", payload)


@mcp.tool()
def instability_plot(
    file_path: str,
    x_col: str,
    y_col: str,
    sheet: int = 0,
    output_dir: Optional[str] = None,
    layers: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """計算並輸出不穩定度圖表。"""
    payload: Dict[str, Any] = {
        "filePath": file_path,
        "sheet": sheet,
        "x_col": x_col,
        "y_col": y_col,
        "ret": {},
        "output_dir": output_dir,
    }
    if layers is not None:
        payload["layers"] = layers
    return _post("/instability/plot", payload)


@mcp.tool()
def evaluation_plan_task(
    plan_id: int,
    seq_no: int = 1,
    stamps: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """執行 evaluationPlanTask。"""
    payload: Dict[str, Any] = {
        "plan_id": plan_id,
        "seq_no": seq_no,
    }
    if stamps is not None:
        payload["stamps"] = stamps
    if output_dir is not None:
        payload["output_dir"] = output_dir
    return _post("/evaluationPlanTask", payload)


@mcp.tool()
def evaluation_plan_task_llm(
    plan_id: int,
    seq_no: int = 1,
    stamps: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    provider: Optional[str] = None,
    model: str = "remote8b",
    user_prompt_template: Optional[str] = None,
    system_prompt: Optional[str] = None,
    prompt_temperature: Optional[float] = None,
    personality: Optional[str] = None,
    enable_llm_analysis: bool = True,
    fH_prompt_alias_level: int = 3,
) -> Dict[str, Any]:
    """執行 evaluationPlanTaskLLM。"""
    payload: Dict[str, Any] = {
        "plan_id": plan_id,
        "seq_no": seq_no,
        "model": model,
        "enable_llm_analysis": enable_llm_analysis,
        "fH_prompt_alias_level": fH_prompt_alias_level,
    }
    if stamps is not None:
        payload["stamps"] = stamps
    if output_dir is not None:
        payload["output_dir"] = output_dir
    if provider is not None:
        payload["provider"] = provider
    if user_prompt_template is not None:
        payload["user_prompt_template"] = user_prompt_template
    if system_prompt is not None:
        payload["system_prompt"] = system_prompt
    if prompt_temperature is not None:
        payload["prompt_temperature"] = prompt_temperature
    if personality is not None:
        payload["personality"] = personality
    return _post("/evaluationPlanTaskLLM", payload)


@mcp.tool()
def call_api_endpoint(
    path: str,
    method: str = "POST",
    payload_json: Optional[str] = None,
    query_json: Optional[str] = None,
) -> Dict[str, Any]:
    """萬用工具：直接呼叫任意 DataAnalysis API 端點。payload_json / query_json 請傳 JSON 字串。"""
    payload = json.loads(payload_json) if payload_json else None
    params = json.loads(query_json) if query_json else None
    return _request(method, path, payload=payload, params=params)


if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    _log(
        "INFO",
        f"Starting MCP server (transport={transport}) pid={os.getpid()} API_BASE={API_BASE} timeout={API_TIMEOUT}s",
    )
    try:
        # mcp.run()
        mcp.run(transport="stdio")
    finally:
        _log("INFO", "MCP server stopped")
