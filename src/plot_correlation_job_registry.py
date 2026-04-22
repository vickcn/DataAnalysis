# -*- coding: utf-8 -*-
"""
/plot-correlation 背景任務 in-memory 登錄（單程序內有效；重啟後 job 消失）。

環境變數：
  DATAANALYSIS_JOB_TIMEOUT_SEC — 預設 job 逾時秒數；0 或空表示不限制（僅在請求未帶 timeout_sec 時套用）。
"""
from __future__ import annotations

import glob
import json
import os
import signal
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def default_job_timeout_sec() -> float:
    raw = os.environ.get("DATAANALYSIS_JOB_TIMEOUT_SEC", "0")
    try:
        v = float(str(raw).strip())
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, v)


@dataclass
class PlotCorrelationJob:
    job_id: str
    status: str  # queued | running | success | failed | cancelled
    requested_exp_fd: str
    resolved_exp_fd: str
    effective_n_jobs: int = 1
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    error: Optional[str] = None
    output_files: List[str] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    timeout_sec: float = 0.0
    method: str = "mic"
    file_path: str = ""
    sheet: Any = 0
    worker_pids: List[int] = field(default_factory=list)
    cleanup_cancelled_futures: int = 0
    cleanup_killed_workers: int = 0
    cleanup_thread_stop_requested: bool = False
    cleanup_memory_refs_cleared: bool = False
    cleanup_errors: List[str] = field(default_factory=list)
    thread_ref: Any = field(default=None, repr=False, compare=False)
    timer_ref: Any = field(default=None, repr=False, compare=False)


_lock = threading.RLock()
_jobs: Dict[str, PlotCorrelationJob] = {}


def _plot_job_to_public_dict(job: PlotCorrelationJob) -> Dict[str, Any]:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "requested_output_directory": job.requested_exp_fd,
        "resolved_output_directory": job.resolved_exp_fd,
        "effective_n_jobs": int(job.effective_n_jobs or 1),
        "error": job.error,
        "output_files": list(job.output_files or []),
        "started_at": job.started_at,
        "ended_at": job.ended_at,
        "timeout_sec": job.timeout_sec,
        "method": job.method,
        "file_path": job.file_path,
    }


def persist_job_json(job_id: str) -> None:
    """寫入 resolved_exp_fd/job.json（重啟後可讀檔得知最後狀態）。"""
    with _lock:
        job = _jobs.get(job_id)
        if not job or not job.resolved_exp_fd:
            return
        pdir = str(job.resolved_exp_fd)
        fpath = os.path.join(pdir, "job.json")
        payload = {
            "job_id": job.job_id,
            "status": job.status,
            "requested_output_directory": job.requested_exp_fd,
            "resolved_output_directory": job.resolved_exp_fd,
            "effective_n_jobs": int(job.effective_n_jobs or 1),
            "error": job.error,
            "output_files": list(job.output_files or []),
            "started_at": job.started_at,
            "ended_at": job.ended_at,
            "timeout_sec": job.timeout_sec,
            "method": job.method,
            "file_path": job.file_path,
            "updated_at": time.time(),
        }
    try:
        os.makedirs(pdir, exist_ok=True)
        with open(fpath, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
    except Exception:
        pass


def list_all_job_summaries() -> List[Dict[str, Any]]:
    with _lock:
        return [_plot_job_to_public_dict(j) for j in _jobs.values()]


def _kill_pid(pid: int) -> bool:
    """Best-effort kill by pid (Windows + POSIX)."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False

    if os.name == "nt":
        try:
            import ctypes
            PROCESS_TERMINATE = 0x0001
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
            if not handle:
                return False
            try:
                return bool(ctypes.windll.kernel32.TerminateProcess(handle, 1))
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            return False

    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except Exception:
        return False


def create_job(
    requested_exp_fd: str,
    output_base_resolved: str,
    *,
    effective_n_jobs: int = 1,
    method: str = "mic",
    file_path: str = "",
    sheet: Any = 0,
    timeout_sec: float = 0.0,
) -> PlotCorrelationJob:
    job_id = str(uuid.uuid4())
    base = os.path.abspath(str(output_base_resolved).strip())
    resolved = os.path.join(base, job_id)
    job = PlotCorrelationJob(
        job_id=job_id,
        status="queued",
        requested_exp_fd=requested_exp_fd,
        resolved_exp_fd=resolved,
        effective_n_jobs=int(effective_n_jobs or 1),
        timeout_sec=float(timeout_sec or 0.0),
        method=method,
        file_path=file_path,
        sheet=sheet,
    )
    with _lock:
        _jobs[job_id] = job
    try:
        os.makedirs(resolved, exist_ok=True)
    except Exception:
        pass
    persist_job_json(job_id)
    return job


def get_job(job_id: str) -> Optional[PlotCorrelationJob]:
    with _lock:
        return _jobs.get(job_id)


def update_job(job_id: str, **kwargs: Any) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        for k, v in kwargs.items():
            if hasattr(job, k):
                setattr(job, k, v)


def set_job_thread(job_id: str, thread_obj: Any) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job:
            job.thread_ref = thread_obj


def set_job_timer(job_id: str, timer_obj: Any) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job:
            job.timer_ref = timer_obj


def clear_job_timer(job_id: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job:
            job.timer_ref = None


def set_worker_pids(job_id: str, pids: List[int]) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        seen = set()
        merged: List[int] = []
        for pid in list(job.worker_pids) + list(pids or []):
            try:
                ipid = int(pid)
            except (TypeError, ValueError):
                continue
            if ipid <= 0 or ipid in seen:
                continue
            seen.add(ipid)
            merged.append(ipid)
        job.worker_pids = merged


def note_cancelled_futures(job_id: str, n: int) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        try:
            nn = int(n)
        except (TypeError, ValueError):
            nn = 0
        if nn > 0:
            job.cleanup_cancelled_futures += nn


def note_killed_workers(job_id: str, n: int) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        try:
            nn = int(n)
        except (TypeError, ValueError):
            nn = 0
        if nn > 0:
            job.cleanup_killed_workers += nn


def note_memory_cleared(job_id: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job:
            job.cleanup_memory_refs_cleared = True


def note_cleanup_error(job_id: str, message: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job and message:
            job.cleanup_errors.append(str(message))


def _cancel_timer(timer_obj: Any) -> None:
    if timer_obj is None:
        return
    try:
        timer_obj.cancel()
    except Exception:
        pass


def _kill_registered_workers(pids: List[int]) -> Tuple[int, List[str]]:
    killed = 0
    errors: List[str] = []
    for pid in sorted(set(pids)):
        ok = _kill_pid(pid)
        if ok:
            killed += 1
        else:
            errors.append(f"failed to kill worker pid={pid}")
    return killed, errors


def mark_running(job_id: str) -> None:
    now = time.time()
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        if job.status != "queued":
            return
        if job.cancel_event.is_set():
            job.status = "cancelled"
            job.ended_at = now
            job.error = job.error or "cancelled"
        else:
            job.status = "running"
            job.started_at = now
    persist_job_json(job_id)


def request_cancel(job_id: str) -> bool:
    """回傳是否找到 job 並已送出取消信號。"""
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return False
        job.cancel_event.set()
        job.cleanup_thread_stop_requested = True
        timer_obj = job.timer_ref
        pids = list(job.worker_pids)

    _cancel_timer(timer_obj)
    killed, errs = _kill_registered_workers(pids)

    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return False
        if job.status in ("queued", "running"):
            job.status = "cancelled"
            job.ended_at = time.time()
            job.error = job.error or "cancelled by client"
        if killed > 0:
            job.cleanup_killed_workers += killed
        if errs:
            job.cleanup_errors.extend(errs)
    persist_job_json(job_id)
    return True


def try_mark_success(job_id: str, output_files: List[str]) -> bool:
    """若 job 仍為 running 且未取消，標為 success 並寫入 output_files。"""
    now = time.time()
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return False
        if job.status in ("success", "failed", "cancelled"):
            return False
        if job.cancel_event.is_set():
            job.status = "cancelled"
            job.ended_at = now
            job.error = job.error or "cancelled"
        else:
            job.status = "success"
            job.output_files = list(output_files)
            job.ended_at = now
            job.error = None
    persist_job_json(job_id)
    with _lock:
        job2 = _jobs.get(job_id)
    return bool(job2 and job2.status == "success")


def try_mark_failed(job_id: str, error: str) -> None:
    """終止為 failed（不覆寫已 success / 已明確 cancelled 的狀態）。"""
    now = time.time()
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        if job.status in ("success", "failed", "cancelled"):
            return
        job.status = "failed"
        job.error = str(error)
        job.ended_at = now
    persist_job_json(job_id)


def try_mark_timeout(job_id: str) -> None:
    """逾時：設 cancel 並標 failed（與 try_mark_failed 分開以便 worker 辨識）。"""
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        if job.status in ("success", "failed", "cancelled"):
            return
        job.cancel_event.set()
        job.cleanup_thread_stop_requested = True
        timer_obj = job.timer_ref
        pids = list(job.worker_pids)

    _cancel_timer(timer_obj)
    killed, errs = _kill_registered_workers(pids)

    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        if job.status in ("success", "failed", "cancelled"):
            return
        job.status = "failed"
        job.error = "job_timeout"
        job.ended_at = time.time()
        if killed > 0:
            job.cleanup_killed_workers += killed
        if errs:
            job.cleanup_errors.extend(errs)
    persist_job_json(job_id)


def get_cleanup_snapshot(job_id: str) -> Dict[str, Any]:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return {}
        return {
            "thread_stop_requested": bool(job.cleanup_thread_stop_requested),
            "cancelled_futures": int(job.cleanup_cancelled_futures),
            "killed_workers": int(job.cleanup_killed_workers),
            "memory_refs_cleared": bool(job.cleanup_memory_refs_cleared),
            "cleanup_errors": list(job.cleanup_errors),
            "worker_pids": list(job.worker_pids),
        }


def scan_correlation_output_files(resolved_dir: str) -> List[str]:
    """掃描 MIC 相關輸出：corr.pkl、corr.xlsx、corrplot 圖檔。"""
    out: List[str] = []
    if not resolved_dir or not os.path.isdir(resolved_dir):
        return out
    base = os.path.abspath(resolved_dir)
    for name in ("corr.pkl", "corr.xlsx"):
        p = os.path.join(base, name)
        if os.path.isfile(p):
            out.append(p)
    for pattern in ("corrplot*.png", "*corrplot*.png"):
        out.extend(sorted(glob.glob(os.path.join(base, pattern))))
    # 去重並保持順序
    seen = set()
    uniq: List[str] = []
    for p in out:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            uniq.append(ap)
    return uniq
