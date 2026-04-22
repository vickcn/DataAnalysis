"""
Zone / header 解析（issues/動態互動對齊AIAPI.iss）。
對齊 AIAPI：xheader/x_cols、yheader/y_col 與 xheader_zones/yheader_zones[*].core。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple


def _normalize_path(path: Optional[str]) -> str:
    if not isinstance(path, str) or not path.strip():
        return ""
    return os.path.normpath(os.path.abspath(os.path.expanduser(path.strip())))


def _read_json_dict(path: str, *, label: str) -> Dict[str, Any]:
    normalized = _normalize_path(path)
    if not normalized:
        raise ValueError(f"{label} 路徑不可為空")
    if not os.path.isfile(normalized):
        raise FileNotFoundError(f"{label} 不存在：{normalized}")

    last_error: Optional[Exception] = None
    text: Optional[str] = None
    for encoding in ("utf-8-sig", "utf-8", "cp950", "big5"):
        try:
            with open(normalized, "r", encoding=encoding) as f:
                text = f.read()
            break
        except UnicodeDecodeError as e:
            last_error = e
    if text is None:
        raise ValueError(f"{label} 編碼無法讀取：{normalized} ({last_error})")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{label} JSON 解析失敗：{normalized} ({e})") from e

    if not isinstance(payload, dict):
        raise ValueError(f"{label} 根節點必須是 JSON 物件：{normalized}")
    return payload


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        v = str(item).strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _to_str_list(value: Any, *, split_csv: bool) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if split_csv:
            return [token.strip() for token in raw.split(",") if token.strip()]
        return [raw]
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            out.extend(_to_str_list(item, split_csv=split_csv))
        return out
    return []


def _extract_x_direct(cfg: Dict[str, Any]) -> List[str]:
    x_cols: List[str] = []
    for key in ("xheader", "x_cols"):
        x_cols.extend(_to_str_list(cfg.get(key), split_csv=True))
    return _dedupe_keep_order(x_cols)


def _extract_y_direct(cfg: Dict[str, Any]) -> Optional[str]:
    for key in ("yheader", "y_col", "y_cols"):
        values = _to_str_list(cfg.get(key), split_csv=True)
        if values:
            return values[0]
    return None


def _extract_input_file(merged: Dict[str, Any], input_file_hint: Optional[str]) -> Optional[str]:
    for key in ("source_data_file", "source_data_filepath", "input_file", "filePath"):
        value = merged.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(input_file_hint, str) and input_file_hint.strip():
        return input_file_hint.strip()
    return None


def flatten_zone_cores(zones: Any) -> List[str]:
    """
    將 xheader_zones / yheader_zones 中各 zone 的 core 展平。
    支援 dict 或 list 型式的 zones。
    """
    if isinstance(zones, dict):
        zone_items = zones.values()
    elif isinstance(zones, (list, tuple)):
        zone_items = zones
    else:
        return []

    cores: List[str] = []
    for zone in zone_items:
        if isinstance(zone, dict):
            core = zone.get("core")
            cores.extend(_to_str_list(core, split_csv=False))
    return _dedupe_keep_order(cores)


def validate_config_dict(cfg: Dict[str, Any]) -> Tuple[bool, str]:
    """
    驗證設定是否可解析出 x/y 欄位；無 zones 時允許走 xheader/x_cols、yheader/y_col。
    """
    if not isinstance(cfg, dict):
        return False, "設定內容必須是 JSON 物件。"

    x_direct = _extract_x_direct(cfg)
    y_direct = _extract_y_direct(cfg)
    x_zone_cores = flatten_zone_cores(cfg.get("xheader_zones"))
    y_zone_cores = flatten_zone_cores(cfg.get("yheader_zones"))

    if "xheader_zones" in cfg and cfg.get("xheader_zones") not in (None, {}) and not x_zone_cores and not x_direct:
        return False, "xheader_zones 存在但找不到有效 core，且 xheader/x_cols 也為空。"
    if "yheader_zones" in cfg and cfg.get("yheader_zones") not in (None, {}) and not y_zone_cores and not y_direct:
        return False, "yheader_zones 存在但找不到有效 core，且 yheader/y_col 也為空。"

    x_cols = x_direct if x_direct else x_zone_cores
    y_col = y_direct or (y_zone_cores[0] if y_zone_cores else None)

    if not x_cols:
        return False, "無法解析 x 欄位：請提供 xheader/x_cols，或 xheader_zones[*].core。"
    if not y_col:
        return False, "無法解析 y 欄位：請提供 yheader/y_col，或 yheader_zones[*].core。"
    return True, ""


def extract_xy_from_config(
        merged: Dict[str, Any],
        *,
        config_path: Optional[str] = None,
        input_file_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    由合併後設定解析 input_file、x_cols、y_col（xheader/x_cols 優先，zones 為 fallback）。
    """
    if not isinstance(merged, dict):
        merged = {}

    x_direct = _extract_x_direct(merged)
    y_direct = _extract_y_direct(merged)
    x_zone_cores = flatten_zone_cores(merged.get("xheader_zones"))
    y_zone_cores = flatten_zone_cores(merged.get("yheader_zones"))

    x_cols = x_direct if x_direct else x_zone_cores
    y_col = y_direct or (y_zone_cores[0] if y_zone_cores else None)

    if y_col:
        x_cols = [x for x in x_cols if x != y_col]

    source_paths = merged.get("_source_paths") if isinstance(merged.get("_source_paths"), dict) else {}
    resolved_config_path = config_path or source_paths.get("config") or source_paths.get("model")

    return {
        "input_file": _extract_input_file(merged, input_file_hint),
        "x_cols": _dedupe_keep_order(x_cols),
        "y_col": y_col,
        "xheader_zones": merged.get("xheader_zones"),
        "yheader_zones": merged.get("yheader_zones"),
        "config_path": resolved_config_path,
        "implemented": True,
    }


def _guess_default_data_path(config_file_path: str) -> Optional[str]:
    if not config_file_path:
        return None
    candidate = os.path.join(os.path.dirname(config_file_path), "defaultData.json")
    return candidate if os.path.isfile(candidate) else None


def _guess_model_config_path(config_file_path: str, data_cfg: Dict[str, Any]) -> Optional[str]:
    if not config_file_path:
        return None
    base_dir = os.path.dirname(config_file_path)
    stamp = data_cfg.get("ID")
    if stamp is None:
        stamps = data_cfg.get("stamps")
        if isinstance(stamps, (list, tuple)) and stamps:
            stamp = stamps[0]
    candidates: List[str] = []
    if stamp is not None and str(stamp).strip():
        candidates.append(os.path.join(base_dir, "warehouse", str(stamp).strip(), "version", "config.json"))
    candidates.append(os.path.join(base_dir, "warehouse", "version", "config.json"))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def merge_config_sources(
        *,
        config_path: Optional[str] = None,
        default_data_path: Optional[str] = None,
        model_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    依序合併三種設定來源：config.json -> defaultData.json -> model config。
    未提供路徑時會嘗試從 config.json 推估 defaultData / model config。
    """
    merged: Dict[str, Any] = {}

    resolved_config_path = _normalize_path(config_path)
    if resolved_config_path:
        config_cfg = _read_json_dict(resolved_config_path, label="config_path")
    else:
        config_cfg = {}

    resolved_default_path = _normalize_path(default_data_path)
    if not resolved_default_path:
        guessed = _guess_default_data_path(resolved_config_path)
        resolved_default_path = _normalize_path(guessed)
    if resolved_default_path:
        default_cfg = _read_json_dict(resolved_default_path, label="default_data_path")
    else:
        default_cfg = {}

    resolved_model_path = _normalize_path(model_config_path)
    if not resolved_model_path:
        guessed = _guess_model_config_path(resolved_config_path, config_cfg)
        resolved_model_path = _normalize_path(guessed)
    if resolved_model_path:
        model_cfg = _read_json_dict(resolved_model_path, label="model_config_path")
    else:
        model_cfg = {}

    for payload in (config_cfg, default_cfg, model_cfg):
        merged.update(payload)

    merged["_source_paths"] = {
        "config": resolved_config_path or None,
        "default_data": resolved_default_path or None,
        "model": resolved_model_path or None,
    }
    return merged
