# -*- coding: utf-8 -*-
"""
欄位分檔暫存（plot-correlation spill-to-disk）：
介面 + Pkl 實作；manifest 與欄位 pkl 位於同根目錄下。
"""
from __future__ import annotations

import abc
import json
import os
import pickle
import re
import shutil
from typing import Any, Dict, List, Optional

import pandas as pd

_SPILL_VERSION = 1


def _safe_col_path(name: Any, index: int) -> str:
    s = re.sub(r"[^a-zA-Z0-9_\-_.]+", "_", str(name))[:120]
    if not s:
        s = f"col_{index}"
    return f"{index:04d}__{s}"


class SpillStore(abc.ABC):
    """先抽象介面，之後可換 Arrow 等實作。"""

    @property
    @abc.abstractmethod
    def root(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def columns(self) -> List:
        """原始欄位名稱（與 DataFrame 一致）。"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def n_rows(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def read_column(self, col: Any) -> pd.Series:
        raise NotImplementedError

    @abc.abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        """還原供繪圖用（此路徑會佔用較大記憶體，僅在 MIC 完成後使用）。"""
        raise NotImplementedError

    @abc.abstractmethod
    def cleanup(self) -> None:
        """刪除 spill 目錄。"""
        raise NotImplementedError


class PklSpillStore(SpillStore):
    """
    每欄一檔：columns/{idx}__<safe>.pkl
    根目錄下 spill_manifest.json
    """

    def __init__(self, root: str, meta: Dict[str, Any], column_file_map: Dict[str, str]) -> None:
        self._root = os.path.abspath(str(root).strip())
        self._meta = dict(meta)
        self._col_files = dict(column_file_map)

    @property
    def root(self) -> str:
        return self._root

    @property
    def columns(self) -> List:
        return list(self._meta.get("columns", []))

    @property
    def n_rows(self) -> int:
        try:
            return int(self._meta.get("rows", 0))
        except (TypeError, ValueError):
            return 0

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        root: str,
    ) -> "PklSpillStore":
        if not isinstance(df, pd.DataFrame) or df.size == 0:
            raise ValueError("Spill 需要非空 DataFrame")
        os.makedirs(root, exist_ok=True)
        col_dir = os.path.join(root, "columns")
        os.makedirs(col_dir, exist_ok=True)

        cols: List = list(df.columns)
        meta: Dict[str, Any] = {
            "version": _SPILL_VERSION,
            "backend": "pkl",
            "columns": cols,
            "dtypes": {str(c): str(df[c].dtype) for c in cols},
            "rows": int(len(df)),
        }
        fmap: Dict[str, str] = {}
        for i, c in enumerate(cols):
            safe = _safe_col_path(c, i)
            fname = f"{safe}.pkl"
            fpath = os.path.join(col_dir, fname)
            fmap[str(c)] = fname
            with open(fpath, "wb") as fp:
                pickle.dump(df[c], fp, protocol=4)
        meta["files"] = fmap
        mpath = os.path.join(root, "spill_manifest.json")
        with open(mpath, "w", encoding="utf-8") as fp:
            json.dump(meta, fp, ensure_ascii=False, indent=2)
        return cls(root, meta, fmap)

    @classmethod
    def open_existing(cls, root: str) -> "PklSpillStore":
        mpath = os.path.join(str(root), "spill_manifest.json")
        with open(mpath, "r", encoding="utf-8") as fp:
            meta = json.load(fp)
        fmap: Dict[str, str] = dict(meta.get("files") or {})
        return cls(str(root), meta, fmap)

    def _file_name_for(self, col: Any) -> str:
        for c in self.columns:
            if c == col or str(c) == str(col):
                k = str(c)
                if k in self._col_files:
                    return str(self._col_files[k])
        if str(col) in self._col_files:
            return str(self._col_files[str(col)])
        if col in self._col_files:
            return str(self._col_files[col])
        raise KeyError(f"欄位不在 manifest: {col!r}")

    def read_column(self, col: Any) -> pd.Series:
        name = self._file_name_for(col)
        p = os.path.join(self._root, "columns", name)
        with open(p, "rb") as fp:
            obj: Any = pickle.load(fp)
        if isinstance(obj, pd.Series):
            return obj
        return pd.Series(obj)

    def to_dataframe(self) -> pd.DataFrame:
        parts: Dict[Any, pd.Series] = {}
        for c in self.columns:
            parts[c] = self.read_column(c)
        return pd.DataFrame(parts)

    def cleanup(self) -> None:
        if self._root and os.path.isdir(self._root):
            try:
                shutil.rmtree(self._root, ignore_errors=True)
            except Exception:
                pass
