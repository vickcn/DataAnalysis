import sys
import os
import requests
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import re
import pandas as pd
import numpy as np
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from package import dataframeprocedure as DFP
from package import LOGger


m_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
m_config = LOGger.load_json(m_config_path)
m_llm_url = m_config.get('data_llm', {}).get('textprocessor_url', 'http://10.1.3.127:6017')
# m_provider = m_config.get('data_llm', {}).get('provider', 'openai')
m_model = m_config.get('data_llm', {}).get('model', 'gpt4o_chat')
m_max_tokens = m_config.get('data_llm', {}).get('max_tokens', 12048)
m_temperature = m_config.get('data_llm', {}).get('temperature', 0.1)
m_top_p = m_config.get('data_llm', {}).get('top_p', 0.95)
m_show_billing = m_config.get('data_llm', {}).get('show_billing', True)

m_prompt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'prompt')
m_prompt_path = os.path.join(m_prompt_dir, 'evaluationFit.json')
m_evaluationFit_promptInfo = LOGger.load_json(m_prompt_path)
m_formatHelper_prompt_path = os.path.join(m_prompt_dir, 'formatHelper.json')
m_formatHelper_promptInfo = LOGger.load_json(m_formatHelper_prompt_path)
m_formatHelper_prompt_paths = {
    1:  r'C:\ML_HOME\DataAnalysis\prompt\formatHelper_level1.json',
    2:  r'C:\ML_HOME\DataAnalysis\prompt\formatHelper_level2.json',
    3:  r'C:\ML_HOME\DataAnalysis\prompt\formatHelper_level3.json'
}
m_fH_prompt_alias_level = m_config.get('data_llm',{}).get('fH_prompt_alias_level',3)


m_DataSetPath = m_config.get('dataExpfd', '')
m_ExportSetPath = m_config.get('mainExpfd', '')

@dataclass
class EvaluationFit:
    general_info: Dict[str, Any] = field(default_factory=dict)
    regression_info: Dict[str, Any] = field(default_factory=dict)
    classification_info: Dict[str, Any] = field(default_factory=dict)
    particular_info: Dict[str, Any] = field(default_factory=dict)
    stamps: List[Any] = field(default_factory=list)
    
    def __post_init__(self):
        self.infos = {
            'general_info': self.general_info,
            'particular_info': self.particular_info,
        }

    def to_markdown(self) -> str:
        markdown = ""
        markdown += f"# General Info\n"
        markdown += f"## {self.infos['general_info']}\n"
        # markdown += f"# Regression Info\n"
        # markdown += f"## {self.infos['regression_info']}\n"
        # markdown += f"# Classification Info\n"
        # markdown += f"## {self.infos['classification_info']}\n"
        markdown += f"# Particular Info\n"
        markdown += f"## {self.infos['particular_info']}\n"
        for k,v in self.infos.items():
            if k in ['general_info', 'regression_info', 'classification_info', 'particular_info']:
                continue
            markdown += f"# {k}\n"
            if isinstance(v, dict):
                for k2,v2 in v.items():
                    markdown += f"## {k2}\n"
                    markdown += f"### {v2}\n"
            elif isinstance(v, list):
                # 處理列表類型，特別是包含字典的列表（如 record）
                if len(v) > 0 and isinstance(v[0], dict):
                    # 如果列表包含字典，用 JSON 格式化以便 LLM 能正確讀取
                    markdown += f"```json\n{json.dumps(v, ensure_ascii=False, indent=4)}\n```\n\n"
                else:
                    # 如果是普通列表，使用 DFP.parse
                    markdown += f"## {DFP.parse(v,digit=4)}\n"
            else:
                markdown += f"## {DFP.parse(v,digit=4)}\n"
            
        markdown += f"# Memo\n"
        markdown += f"如果有特徵對沒有被提及他的MIC重要度，那就是0"
        return markdown

    def save_to_file(self, file_path: Optional[str] = None, exp_fd: str = "tmp_ana"):
        if not file_path or not isinstance(file_path, str):
            output_fn = LOGger.stamp_process('', ["ETMD", *self.stamps], '','','','_',for_file=True)
            file_path = os.path.join(exp_fd, output_fn + '.md')
        # 確保目錄存在
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(self.to_markdown())
        return file_path

    def load_from_file(self, file_path: str):
        # with open(file_path, 'r', encoding='utf-8') as file:
        #     self.to_markdown = file.read()
        raise NotImplementedError("Not implemented")

    def add_info(self, key: str, value: Any):
        """
        直接添加新的項目到 self.infos
        
        Args:
            key: str, 要添加的鍵名
            value: Any, 要添加的值（可以是字典或其他類型）
        """
        if isinstance(value, dict):
            if key not in self.infos:
                self.infos[key] = {}
            self.infos[key].update(value)
        else:
            self.infos[key] = value
    
    def load_from_dict(self, data: Dict[str, Any], key: Optional[str] = None):
        """
        從字典載入數據到對應的 info 屬性，並動態擴增 self.infos
        
        Args:
            data: Dict[str, Any], 要載入的數據字典
            key: Optional[str], 如果提供，則將數據載入到 self.infos[key]；否則從 data 中讀取預定義的鍵
        """
        if key:
            # 如果提供了 key，直接將數據載入到該鍵
            if key in ['general_info', 'regression_info', 'classification_info', 'particular_info']:
                # 如果是預定義的鍵，更新對應的屬性
                setattr(self, key, data)
                self.infos[key] = getattr(self, key)
            else:
                # 如果是新的鍵，動態擴增到 infos
                if key not in self.infos:
                    self.infos[key] = {}
                if isinstance(data, dict):
                    self.infos[key].update(data)
                else:
                    self.infos[key] = data
        else:
            # 如果沒有提供 key，從 data 中讀取預定義的鍵
            if 'general_info' in data:
                self.general_info = data.get('general_info', {})
                self.infos['general_info'] = self.general_info
            if 'regression_info' in data:
                self.regression_info = data.get('regression_info', {})
                self.infos['regression_info'] = self.regression_info
            if 'classification_info' in data:
                self.classification_info = data.get('classification_info', {})
                self.infos['classification_info'] = self.classification_info
            if 'particular_info' in data:
                self.particular_info = data.get('particular_info', {})
                self.infos['particular_info'] = self.particular_info

def grep_from_PlanTask_log(log_path: str, ret: Optional[Dict[str, Any]] = None, front_sym_inLine:  Optional[str] = None, back_sym_inLine: Optional[str] = None, **kwargs) -> bool:
    """
    Args:
        log_path: str, the path to the log file
        ret: Optional[Dict[str, Any]], the return value
            - (str)
            - loss curve data
        **kwargs: Any, the keyword arguments
    Returns:
        bool, True if successful, False otherwise
    """
    lines = []
    LOGger.read_txt(log_path, lines, front_sym_inLine=front_sym_inLine, back_sym_inLine=back_sym_inLine)
    content = '\n'.join(lines)
    print(LOGger.OKBLUE + f"log {log_path} content: {content}")
    if isinstance(ret, dict): ret['log'] = {'log': f'\n```\n{content}\n```', 'path': log_path}
    return True

def grep_from_PlanTask_log_by_PlanSeq(plan_id: int, seq_no: int = 1, ret: Optional[Dict[str, Any]] = None, front_sym_inLine:  Optional[str] = None, back_sym_inLine: Optional[str] = None, **kwargs) -> bool:
    """
    Args:
        plan_id: int, the plan sequence number
        seq_no: int, the sequence number
        ret: Optional[Dict[str, Any]], the return value
            - (str)
            - loss curve data
        **kwargs: Any, the keyword arguments
    Returns:
        bool, True if successful, False otherwise
    """
    log_path = m_ExportSetPath + f'\\{plan_id}\\{seq_no}\\log.txt'
    retTemp = {}
    if(not grep_from_PlanTask_log(log_path, ret=retTemp, front_sym_inLine=front_sym_inLine, back_sym_inLine=back_sym_inLine, **kwargs)):
        return False
    if isinstance(ret, dict): ret.update(retTemp)
    return True

def grep_train_valid_loss_from_log_content(log_content: str) -> Tuple[List[float], Optional[List[float]]]:
    """
    從(str)標準log規律中找出訓練跟驗證集的loss數據
    
    Args:
        log_content: str, 日誌內容字符串
        
    Returns:
        Tuple[List[float], Optional[List[float]]], the training and validation loss data
            - (List[float]): 訓練損失列表
            - (Optional[List[float]]): 驗證損失列表（如果沒有驗證集，返回None）
    
    支持的日誌格式：
    1. [XXX]訓練回數:X, 最後訓練損失:X.XX, 最後試驗損失:X.XX
    2. [logLoss][XXX]核心訓練進行中.... Epoch X: loss: X.XXXX, val_loss: X.XXXX
    """
    train_losses = []
    valid_losses = []
    
    if not log_content or not isinstance(log_content, str):
        return train_losses, None if not valid_losses else valid_losses
    
    lines = log_content.split('\n')
    
    # 模式1: 提取包含 "最後訓練損失" 和 "最後試驗損失" 的行
    # 格式: [XXX]訓練回數:X, 最後訓練損失:X.XX, 最後試驗損失:X.XX
    pattern1 = r'最後訓練損失[:：](\d+\.?\d*).*?最後試驗損失[:：](\d+\.?\d*)'
    
    # 模式2: 提取 epoch 級別的 loss 和 val_loss
    # 格式: Epoch X: loss: X.XXXX, val_loss: X.XXXX
    pattern2 = r'Epoch\s+\d+:\s+loss:\s+(\d+\.?\d*),\s+val_loss:\s+(\d+\.?\d*)'
    
    for line in lines:
        # 匹配模式1: 最後訓練損失/最後試驗損失（同一行）
        match1 = re.search(pattern1, line)
        if match1:
            train_val = match1.group(1)
            valid_val = match1.group(2)
            if train_val:
                try:
                    train_losses.append(float(train_val))
                except ValueError:
                    pass
            if valid_val:
                try:
                    valid_losses.append(float(valid_val))
                except ValueError:
                    pass
            continue
        
        # 匹配模式2: Epoch 級別的 loss 和 val_loss
        match2 = re.search(pattern2, line)
        if match2:
            train_val = match2.group(1)
            valid_val = match2.group(2)
            if train_val:
                try:
                    train_losses.append(float(train_val))
                except ValueError:
                    pass
            if valid_val:
                try:
                    valid_losses.append(float(valid_val))
                except ValueError:
                    pass
    
    # 如果沒有找到驗證損失，返回 None
    if not valid_losses:
        return train_losses, None
    
    return train_losses, valid_losses

def grep_train_valid_loss_from_log(log_path: str, ret: Optional[Dict[str, Any]] = None, front_sym_inLine:  Optional[str] = None, back_sym_inLine: Optional[str] = None, **kwargs) -> bool:
    """
    Args:
        log_path: str, the path to the log file
        ret: Optional[Dict[str, Any]], the return value
            - (dict)
            ```
                {{
                    'train_losses': [float, ...],
                    'valid_losses': [float, ...] or None
                }}
            ```
        front_sym_inLine: Optional[str], the front symbol in line
        back_sym_inLine: Optional[str], the back symbol in line
        **kwargs: Any, the keyword arguments
    Returns:
        bool, True if successful, False otherwise
    """
    if not os.path.exists(log_path):
        print(LOGger.FAIL + f"Log file not found: {log_path}")
        return False
    
    lines = []
    LOGger.read_txt(log_path, lines, front_sym_inLine=front_sym_inLine, back_sym_inLine=back_sym_inLine)
    content = '\n'.join(lines)
    train_losses, valid_losses = grep_train_valid_loss_from_log_content(content)
    if isinstance(ret, dict): 
        ret.update({'train_losses': train_losses, 'valid_losses': valid_losses})
    return True

def grep_train_valid_loss_from_log_by_PlanSeq(plan_id: int, seq_no: int = 1, ret: Optional[Dict[str, Any]] = None, front_sym_inLine: Optional[str] = None, back_sym_inLine: Optional[str] = None, **kwargs) -> bool:
    """
    Args:
        plan_id: int, the plan sequence number
        seq_no: int, the sequence number
        ret: Optional[Dict[str, Any]], the return value
            - (dict)
            ```
                {{
                    'train_losses': [float, ...],
                    'valid_losses': [float, ...] or None
                }}
            ```
        front_sym_inLine: Optional[str], the front symbol in line
        back_sym_inLine: Optional[str], the back symbol in line
        **kwargs: Any, the keyword arguments
    Returns:
        bool, True if successful, False otherwise
    """
    log_path = m_ExportSetPath + f'\\{plan_id}\\{seq_no}\\log.txt'
    return grep_train_valid_loss_from_log(log_path, ret=ret, front_sym_inLine=front_sym_inLine, back_sym_inLine=back_sym_inLine, **kwargs)

def grep_from_record(record_path: str, sheet: int = 0, filters: Optional[Dict[str, Any]] = None, ret: Optional[Dict[str, Any]] = None, **kwargs) -> bool:
    """
    Args:
        record_path: str, the path to the record file
        sheet: int, the sheet number
        filters: Optional[Dict[str, Any]], the filters to apply to the data
        ret: Optional[Dict[str, Any]], the return value
            - (dict)
            ```
                {{
                    timestampindex: {{
                        ... 指標 ...
                    }},
                    ...
                }}
            ```
            - standard evaluation data
        **kwargs: Any, the keyword arguments
    Returns:
        bool, True if successful, False otherwise
    """
    print(LOGger.WARNING + f"grep_from_record start")
    df = DFP.import_data(record_path, sht=sheet, xlsx_formula_header=['hyperlink'])
    if not isinstance(df, DFP.pd.DataFrame):
        print(LOGger.FAIL + f"{record_path} is not a DataFrame")
        return False
    if not 'hyperlink' in df.columns:
        print(LOGger.FAIL + f"{record_path} has no hyperlink column")
        return False
    # print(LOGger.OKBLUE + f"df['hyperlink'].astype(str): {df['hyperlink'].astype(str)}")

    for k, flt in filters.items():
        try:
            df = flt(df)
        except Exception as e:
            print(LOGger.FAIL + f"Filter {k} failed: {e}")
            return False
    if df is None:
        print(LOGger.FAIL + f"No data found")
        return False
    if getattr(df, 'empty', False):
        print(LOGger.FAIL + f"No data found")
        return False
    
    print(LOGger.OKBLUE + (df.index if hasattr(df, 'index') else str(df)[:500]))
    if isinstance(ret, dict): ret['record'] = df.to_dict(orient='records')
    return True


def _parse_hyperlink(x):
    try:
        ret = eval(LOGger.mystr(x).brackets("(",")"))[0]
    except:
        ret = ""
    return ret

def grep_from_record_by_PlanSeq(plan_id: int, seq_no: int = 1, filters: Optional[Dict[str, Any]] = None, ret: Optional[Dict[str, Any]] = None, **kwargs) -> bool:
    """
    Args:
        plan_id: int, the plan sequence number
        seq_no: int, the sequence number
        filters: Optional[Dict[str, Any]], the filters to apply to the data
        ret: Optional[Dict[str, Any]], the return value
            - (dict)
            ```
                {{
                    timestampindex: {{
                        ... 指標 ...
                    }},
                    ...
                }}
            ```
            - standard evaluation data
        **kwargs: Any, the keyword arguments
    Returns:
        bool, True if successful, False otherwise
    """
    record_path = m_ExportSetPath + f'\\{plan_id}\\records_Plan{plan_id}.xlsx'
    if not os.path.exists(record_path):
        print(LOGger.FAIL + f"Record file not found: {record_path}")
        return False
    filters = filters or {}
    if isinstance(seq_no, int): filters['SeqNo'] = lambda df: df[df['hyperlink'].map(_parse_hyperlink) == str(seq_no)]
    retTemp = {}
    if(not grep_from_record(record_path, filters=filters, ret=retTemp, **kwargs)):
        return False
    if isinstance(ret, dict): ret.update(retTemp)
    return True


def grep_from_factory_importance_info(info_path, ret: Optional[Dict[str, Any]], **kwargs) -> bool: 
    print(LOGger.WARNING + f"grep_from_factory_importance_info start")
    df = DFP.import_data(info_path)

    infos = {}
    for i,row in enumerate(df.index):
        for j,col in enumerate(df.columns):
            if i<j:
                score = df.loc[row, col]
                infos[f"({row},{col})"] = score
    info_stg = '\n'.join([f"{k}:{DFP.parse(v)}" for k,v in infos.items()])
    if isinstance(ret, dict):
        ret['mic_info_stg'] = info_stg
        ret['mic_infos'] = infos

    return True


def grep_from_factory_importance_info_by_PlanSeq(plan_id, seq_no, ret: Optional[Dict[str, Any]], **kwargs) -> bool:
    corr_path = m_ExportSetPath + f'\\{plan_id}\\{seq_no}\\graph\\corr.xlsx'
    if not os.path.exists(corr_path):
        print(LOGger.FAIL + f"corr file not found: {corr_path}")
        return False
    retTemp = {}
    if(not grep_from_factory_importance_info(corr_path, ret=retTemp, **kwargs)):
        return False
    if isinstance(ret, dict): ret.update(retTemp)
    return True

def grep_execution_info(exec_file_path: str, ret: Optional[Dict[str, Any]] = None, selected_sheets: List[Any] = [], **kwargs) -> bool:
    """
    Args:
        exec_file_path: str, the path to the execution file
        ret: Optional[Dict[str, Any]], the return value
            - (dict)
            ```
                {{
                    ...  ...
                }}
            ```
        **kwargs: Any, the keyword arguments
    Returns:
        bool, True if successful, False otherwise
    """
    if not LOGger.isinstance_not_empty(exec_file_path, str):
        print(LOGger.FAIL + f"Execution file path is not a valid string: \n{exec_file_path}")
        return False
    if not exec_file_path.endswith('.xlsx'):
        print(LOGger.FAIL + f"Execution file is not an Excel file: {exec_file_path}")
        return False
    if not os.path.exists(exec_file_path):
        print(LOGger.FAIL + f"Execution file not found: {exec_file_path}")
        return False
    datas = {}
    for sheet in selected_sheets:
        if not LOGger.isinstance_not_empty(sheet, (int, str)):
            print(LOGger.FAIL + f"Sheet is not a valid input: \n{sheet}")
            return False
        df = DFP.import_data(exec_file_path, sht=sheet)
        datas[sheet] = df.to_dict(orient='records')
    if isinstance(ret, dict): ret['datas'] = datas
    return True

def grep_json_info(config_path: str, ret: Optional[Dict[str, Any]] = None, stamps: Optional[List[Any]] = None, **kwargs) -> bool:
    """
    Args:
        config_path: str, the path to the config file
        ret: Optional[Dict[str, Any]], the return value
            - (dict)
            ```
                {{
                    ...  ...
                }}
            ```
        **kwargs: Any, the keyword arguments
    Returns:
        bool, True if successful, False otherwise
    """
    if not LOGger.isinstance_not_empty(config_path, str):
        print(LOGger.FAIL + f"Config file path is not a valid string: \n{config_path}")
        return False
    if not config_path.endswith('.json'):
        print(LOGger.FAIL + f"Config file is not a JSON file: {config_path}")
        return False
    if not os.path.exists(config_path):
        print(LOGger.FAIL + f"Config file not found: {config_path}")
        return False
    infos = LOGger.load_json(config_path, ret=ret, **kwargs)
    stamps = stamps or []
    stamp = LOGger.stamp_process('', stamps, '','','','_',for_file=True)
    if isinstance(ret, dict): ret[stamp] = infos
    return True

def grep_exeuction_info_by_PlanSeq(plan_id: int, seq_no: int = 1, ret: Optional[Dict[str, Any]] = None, **kwargs) -> bool:
    """
    Args:
        plan_id: int, the plan sequence number
        seq_no: int, the sequence number
        ret: Optional[Dict[str, Any]], the return value
            - (dict)
            ```
                {{
                    ...  ...
                }}
            ```
        **kwargs: Any, the keyword arguments
    Returns:
        bool, True if successful, False otherwise
    """
    if not LOGger.isinstance_not_empty(plan_id, int):
        print(LOGger.FAIL + f"Plan sequence number is not a valid integer: \n{plan_id}")
        return False
    if not LOGger.isinstance_not_empty(seq_no, int):
        print(LOGger.FAIL + f"Sequence number is not a valid integer: \n{seq_no}")
        return False
    exec_file_path = m_ExportSetPath + f'\\{plan_id}\\{seq_no}\\excel\\Plan{plan_id}_{plan_id}_detailed_evaluation.xlsx'
    if not os.path.exists(exec_file_path):
        exec_file_path = None
        files, _ = DFP.explore_folder(m_ExportSetPath + f'\\{plan_id}\\{seq_no}\\excel')
        for file in files:
            if file.endswith('.xlsx'):
                exec_file_path = file
                print(LOGger.OKGREEN + f"Execution file found: {exec_file_path}")
                break
        if not exec_file_path:
            print(LOGger.FAIL + f"Execution file not found: {exec_file_path}")
            return False
    if not grep_execution_info(exec_file_path, ret=ret, **kwargs):
        return False
    return True

def grep_json_info_by_PlanSeq(plan_id: int, seq_no: int = 1, ret: Optional[Dict[str, Any]] = None, **kwargs) -> bool:
    """
    Args:
        plan_id: int, the plan sequence number
        seq_no: int, the sequence number
        ret: Optional[Dict[str, Any]], the return value
            - (dict)
            ```
                {{
                    ...  ...
                }}
            ```
        **kwargs: Any, the keyword arguments
    Returns:
        bool, True if successful, False otherwise
    """
    if not LOGger.isinstance_not_empty(plan_id, int):
        print(LOGger.FAIL + f"Plan sequence number is not a valid integer: \n{plan_id}")
        return False
    if not LOGger.isinstance_not_empty(seq_no, int):
        print(LOGger.FAIL + f"Sequence number is not a valid integer: \n{seq_no}")
        return False
    config_path = m_ExportSetPath + f'\\{plan_id}\\{seq_no}\\config.json'
    if not os.path.exists(config_path):
        print(LOGger.FAIL + f"Config file not found: {config_path}")
        return False
    if not grep_json_info(config_path, ret=ret, stamps=['config'], **kwargs):
        return False
    return True

def summary_project_info(plan_id: int, seq_no: int = 1, ret: Optional[Dict[str, Any]] = None, stamps: Optional[List[Any]] = None, **kwargs) -> bool:
    """
    Args:
        plan_id: int, the plan sequence number
        seq_no: int, the sequence number
        ret: Optional[Dict[str, Any]], the return value
            - (dict)
            ```
                {{
                    ...  ...
                }}
            ```
        **kwargs: Any, the keyword arguments
    Returns:
        bool, True if successful, False otherwise
    """
    stamps = stamps or []
    if not LOGger.isinstance_not_empty(plan_id, int):
        print(LOGger.FAIL + f"Plan sequence number is not a valid integer: \n{plan_id}")
        return False
    if not LOGger.isinstance_not_empty(seq_no, int):
        print(LOGger.FAIL + f"Sequence number is not a valid integer: \n{seq_no}")
        return False
    project_path = m_ExportSetPath + f'\\{plan_id}\\{seq_no}'
    if not os.path.exists(project_path):
        print(LOGger.FAIL + f"Project path not found: {project_path}")
        return False
    stamp = LOGger.stamp_process('', stamps, '','','','_',for_file=True)
    info = ret.get(stamp, {}) or {}
    grep_methods = [
        grep_json_info_by_PlanSeq,
        grep_exeuction_info_by_PlanSeq,
        grep_from_record_by_PlanSeq,
        grep_from_factory_importance_info_by_PlanSeq,
        # lambda plan_id,seq_no=1,ret=None,front_sym_inLine='    ',back_sym_inLine=None,**kwargs: grep_from_PlanTask_log_by_PlanSeq(plan_id=plan_id,seq_no=seq_no,ret=ret, front_sym_inLine=front_sym_inLine, back_sym_inLine=back_sym_inLine, **kwargs),
        grep_train_valid_loss_from_log_by_PlanSeq
    ]
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for i, method in enumerate(grep_methods):
            futures.append(executor.submit(method, plan_id, seq_no, ret=info))
        for future in as_completed(futures):
            future.result()
    end_time = time.time()
    print(LOGger.OKBLUE + f"Time taken: {end_time - start_time} seconds")
    return True


def evaluationFit(plan_id: int, seq_no: int = 1, ret: Optional[Dict[str, Any]] = None, stamps: Optional[List[Any]] = None, output_dir: Optional[str] = None, **kwargs) -> bool:
    """
    Args:
        plan_id: int, the plan sequence number
        seq_no: int, the sequence number
        ret: Optional[Dict[str, Any]], the return value
        output_dir: Optional[str], the output directory
        **kwargs: Any, the keyword arguments
    Returns:
        bool, True if successful, False otherwise
    """
    etmd = EvaluationFit(stamps=stamps)
    stamp = LOGger.stamp_process('', stamps, '','','','_')
    info = {stamp: etmd.infos}
    if not summary_project_info(plan_id, seq_no, ret=info, stamps=stamps, **kwargs):
        print(LOGger.FAIL + f"Grep project info failed")
        return False
    print(LOGger.OKBLUE + f"info: {','.join(list(info.keys()))}")
    
    # 從 info 中提取實際的數據結構
    # info 的結構可能是 {'info': {'datas': {}, 'config': {...}}} 或直接是 {'datas': {}, 'config': {...}}
    actual_info = info.get(stamp, {}) 

    # 移除只用於中間計算的文字版 MIC，避免輸出重複（留數值版 mic_infos 即可）
    actual_info.pop('mic_info_stg', None)
    
    # 提取各種信息
    # config 鍵包含 JSON 配置信息（之前是 infos）
    infos_data = actual_info.get('config', actual_info.get('infos', {}))
    mic_info = actual_info.get('mic_infos', [])
    exec_info = actual_info.get('datas', {})
    record_info = actual_info.get('record', [])
    log_info = actual_info.get('log', '')
    train_losses = actual_info.get('train_losses', [])
    valid_losses = actual_info.get('valid_losses', None)
    
    
    
    # 處理 infos：可能是列表或字典
    if isinstance(infos_data, list):
        print(LOGger.OKCYAN + f"infos_data: {LOGger.stamp_process('',infos_data)}")
        # 如果是列表，將每個項目作為一個 info 項
        for idx, item in enumerate(infos_data):
            if isinstance(item, dict):
                # 使用項目的 'type' 或其他唯一標識作為鍵名
                key = item.get('type', f'info_{idx}')
                etmd.add_info(key, item)
                print(LOGger.OKBLUE + f"Added info item: {key} = \n{item}")
    elif isinstance(infos_data, dict):
        print(LOGger.OKCYAN + f"infos_data: {LOGger.stamp_process('',infos_data)}")
        # # 如果是字典，直接遍歷
        # for k, v in infos_data.items():
        #     print(LOGger.OKBLUE + f"json_info: {k} = \n{v}")
        #     etmd.load_from_dict(v, k)
        # 將整包 config 收成單一區塊，避免後續輸出重複列出各欄位
        etmd.add_info('config', infos_data)
    
    if isinstance(mic_info, dict):
        print(LOGger.OKCYAN + f"mic_infos: {LOGger.stamp_process('',mic_info)}")
        for k, v in mic_info.items():
            if v < 1e-4:
                continue
            print(LOGger.OKBLUE + f"mic_pair: {k} = \n{v}")
            etmd.add_info(f"mic_pair{k}", v)
    
    # 處理 exec_info (datas)
    if isinstance(exec_info, dict):
        print(LOGger.OKCYAN + f"exec_info: {LOGger.stamp_process('',exec_info)}")
        for k, v in exec_info.items():
            print(LOGger.OKBLUE + f"exec_info: {k} = \n{v}")
            etmd.add_info(f'exec_{k}', v)
    
    # 處理 record 數據
    if record_info:
        print(LOGger.OKCYAN + f"record_info: {len(record_info)} records")
        etmd.add_info('record', record_info)
        print(LOGger.OKBLUE + f"Added record data: {len(record_info)} items")
    
    # 處理 loss 數據
    if train_losses:
        loss_data = {'train_losses': train_losses}
        if valid_losses is not None:
            loss_data['valid_losses'] = valid_losses
        etmd.add_info('loss_curve', loss_data)
        print(LOGger.OKBLUE + f"loss_curve: train_losses={len(train_losses)} items, valid_losses={'None' if valid_losses is None else len(valid_losses)} items")
    
    # 處理 log_info
    if log_info:
        etmd.add_info('log', log_info)
        print(LOGger.OKBLUE + f"log_info: \n{log_info}")
    else:
        print(LOGger.OKBLUE + f"log_info: (empty)")
    # 按照格式寫成給llm看、來評估資料分布與模型能力的md
    md = etmd.to_markdown()
    if isinstance(ret, dict): ret['md'] = md
    # etmd.save_to_file() # debug
    print(LOGger.WARNING + f"[evaluationFit] md_output_dir: {output_dir}")
    if LOGger.isinstance_not_empty(output_dir, str):
        if not etmd.save_to_file(exp_fd=output_dir):
            print(LOGger.FAIL + f"Save etmd failed")
            return False
        # if not save_result(etmd, output_dir, stamps):
        #     print(LOGger.FAIL + f"Save etmd failed")
        #     return False
    return True


def _format_token_info(result: Dict[str, Any]) -> None:
    """
    格式化並顯示 Token 使用情況
    
    Args:
        result: API 響應結果字典，包含 usage、token_summary 等信息
    """
    print(f"\nToken 使用情況:")
    
    # 優先使用單次請求的 usage 信息（如果存在）
    usage_info = result.get('usage')
    found_token = False
    
    if usage_info and isinstance(usage_info, dict):
        prompt_tokens = usage_info.get('prompt_tokens', 0)
        completion_tokens = usage_info.get('completion_tokens', 0)
        total_tokens = usage_info.get('total_tokens', 0)
        
        if prompt_tokens > 0 or completion_tokens > 0 or total_tokens > 0:
            print(f"  輸入 tokens: {prompt_tokens}")
            print(f"  輸出 tokens: {completion_tokens}")
            print(f"  總 tokens: {total_tokens}")
            found_token = True
    
    # 如果沒有單次請求的 usage，則從 token_summary 中查找本次請求的 provider/model 的 token
    if not found_token and 'token_summary' in result:
        token_info = result['token_summary']
        current_provider = result.get('provider', '')
        current_model = result.get('model_alias', '')
        
        if isinstance(token_info, dict):
            # 優先從 by_provider_model 中查找本次請求的 provider/model 的 token
            by_provider_model = token_info.get('by_provider_model', [])
            for item in by_provider_model:
                if (item.get('provider') == current_provider and 
                    item.get('model_alias') == current_model):
                    prompt_tokens = item.get('prompt_tokens', 0)
                    completion_tokens = item.get('completion_tokens', 0)
                    total_tokens = item.get('total_tokens', 0)
                    count = item.get('count', 0)
                    
                    if prompt_tokens > 0 or completion_tokens > 0 or total_tokens > 0:
                        if count > 1:
                            # 如果是累計統計（count > 1），顯示累計和平均
                            avg_prompt = prompt_tokens // count
                            avg_completion = completion_tokens // count
                            avg_total = total_tokens // count
                            print(f"  輸入 tokens: {prompt_tokens} (累計，平均: {avg_prompt}/次)")
                            print(f"  輸出 tokens: {completion_tokens} (累計，平均: {avg_completion}/次)")
                            print(f"  總 tokens: {total_tokens} (累計，平均: {avg_total}/次)")
                            print(f"  呼叫次數: {count}")
                        else:
                            print(f"  輸入 tokens: {prompt_tokens}")
                            print(f"  輸出 tokens: {completion_tokens}")
                            print(f"  總 tokens: {total_tokens}")
                        found_token = True
                        break
            
            # 如果找不到本次請求的 provider/model，且 total 有值，顯示 total（累計統計）
            if not found_token and 'total' in token_info and isinstance(token_info['total'], dict):
                total_info = token_info['total']
                prompt_tokens = total_info.get('prompt_tokens', 0)
                completion_tokens = total_info.get('completion_tokens', 0)
                total_tokens = total_info.get('total_tokens', 0)
                calls = total_info.get('calls', 0)
                
                if prompt_tokens > 0 or completion_tokens > 0 or total_tokens > 0:
                    print(f"  [累計統計] 輸入 tokens: {prompt_tokens}")
                    print(f"  [累計統計] 輸出 tokens: {completion_tokens}")
                    print(f"  [累計統計] 總 tokens: {total_tokens}")
                    if calls > 0:
                        print(f"  [累計統計] 呼叫次數: {calls}")
                    found_token = True
    
    if not found_token:
        print(f"  輸入 tokens: N/A (此 provider 不支援 token 回報或尚未記錄)")
        print(f"  輸出 tokens: N/A (此 provider 不支援 token 回報或尚未記錄)")
        print(f"  總 tokens: N/A (此 provider 不支援 token 回報或尚未記錄)")


def _format_billing_info(result: Dict[str, Any]) -> None:
    """
    格式化並顯示費用估計
    
    Args:
        result: API 響應結果字典，包含 current_cost_usd、billing_summary 等信息
    """
    print(f"\n費用估計:")
    
    # 優先使用單次請求的費用（current_cost_usd）
    current_cost = result.get('current_cost_usd')
    if current_cost is not None:
        if current_cost > 0:
            print(f"  本次請求費用: ${current_cost:.8f}")
        else:
            print(f"  本次請求費用: $0.0 (此 provider 不計費)")
    else:
        # 如果沒有單次請求的費用，從 billing_summary 中查找本次請求的 provider/model 的費用
        if 'billing_summary' in result:
            billing_info = result['billing_summary']
            current_provider = result.get('provider', '')
            current_model = result.get('model_alias', '')
            
            if isinstance(billing_info, dict):
                # 優先從 by_provider_model 中查找本次請求的 provider/model 的費用
                by_provider_model = billing_info.get('by_provider_model', [])
                found_cost = False
                
                for item in by_provider_model:
                    if (item.get('provider') == current_provider and 
                        item.get('model_alias') == current_model):
                        cost_usd = item.get('cost_usd', 0.0)
                        count = item.get('count', 0)
                        
                        if count > 1:
                            # 如果是累計統計，顯示平均費用
                            avg_cost = cost_usd / count
                            print(f"  [累計統計] 總費用: ${cost_usd:.8f} ({count} 次請求，平均: ${avg_cost:.8f}/次)")
                        else:
                            print(f"  本次請求費用: ${cost_usd:.8f}")
                        found_cost = True
                        break
                
                # 如果找不到本次請求的 provider/model，顯示總費用（累計統計）
                if not found_cost:
                    total_cost = billing_info.get('total_cost_usd', 0.0)
                    total_calls = billing_info.get('total_calls', 0)
                    if total_cost > 0 or total_calls > 0:
                        print(f"  [累計統計] 總費用: ${total_cost:.8f} ({total_calls} 次請求)")
                    else:
                        print(f"  費用: $0.0 (此 provider 不計費或尚未記錄)")
            else:
                print(f"  {billing_info}")
        else:
            print(f"  未提供（可能此 provider/model 不支援費用追蹤）")


def get_model_provider(model_name, **kwargs):
    if model_name.find('gpt')>-1:
        return "openai"
    elif model_name.find('remote')>-1:
        return "remote"
    else:
        return "sentence_transformer"



def analyze_content(content: str, provider: Optional[str] = None, model: Optional[str] = None, user_prompt_template: Optional[str] = None, system_prompt: Optional[str] = None, prompt_temperature: Optional[float] = None, personality: Optional[str] = None, fH_prompt_alias_level: Optional[int] = m_fH_prompt_alias_level) -> Dict[str, Any]:
    """
    Args:
        content: str, the content to analyze
        provider: Optional[str], LLM provider，預設為 m_provider
        model: Optional[str], 模型名稱，預設為 m_model
        user_prompt_template: Optional[str], 用戶提示模板，預設為 m_evaluationFit_promptInfo.get('user_prompt', '')
            - 如果 user_prompt_template 中不包含 {{content}}，則會自動添加
        system_prompt: Optional[str], 系統提示模板，預設為 m_evaluationFit_promptInfo.get('system_prompt', '')
        prompt_temperature: Optional[float], 提示溫度，預設為 m_temperature
        personality: Optional[str], LLM 回復個性描述，預設為 m_formatHelper_promptInfo.get('personality', '')
            - formatHelper 配置固定使用系統預設的 m_formatHelper_promptInfo（從 formatHelper.json 載入）
        prompt_alias_level: 提示講解的等級，初步開發1~3，越高越初階。若為空，回歸 m_formatHelper_promptInfo
    Returns:
        Dict[str, Any], 包含以下鍵的字典：
            - 'analysis_result': str, LLM 分析結果
            - 'token_summary': dict, Token 使用情況（如果可用）
            - 'billing_summary': dict, 費用估計（如果可用）
            - 'post_id': str, LLM API 回應 ID（如果可用）
            - 'timestamp': str, 時間戳（如果可用）
    """
    formatHelper_prompt_path = m_formatHelper_prompt_paths.get(fH_prompt_alias_level, m_formatHelper_prompt_path)
    print(LOGger.OKGREEN + f"使用{formatHelper_prompt_path}個性進行分析...")
    formatHelper_promptInfo = LOGger.load_json(formatHelper_prompt_path)
    
    user_prompt_template = user_prompt_template if user_prompt_template else m_evaluationFit_promptInfo.get('user_prompt', '')
    if not f'{{content}}' in user_prompt_template:
        user_prompt_template = f'{user_prompt_template}\n\n{{content}}'

    system_prompt = system_prompt if system_prompt else m_evaluationFit_promptInfo.get('system_prompt', '')
    
    # 加入個性描述
    if personality:
        system_prompt = f'{personality}\n\n{system_prompt}'
    elif formatHelper_promptInfo and formatHelper_promptInfo.get('personality'):
        system_prompt = f'{formatHelper_promptInfo.get("personality")}\n\n{system_prompt}'
    
    # 加入格式說明
    explain_main_info = "你就是本系統的開發者，「你們系統」就是「本系統」，本系統的數據分析格式與一般標準用法有以下特殊之處，自己清楚就好，需要時再跟讀者報告：1. batch_size 參數的特殊用法：\n   - 如果 batch_size 是整數（如 100、256），則表示實際的批次大小\n   - 如果 batch_size 是 0 到 1 之間的小數（如 0.1、0.2），則表示訓練資料的比例\n   - 例如：batch_size=0.1 且訓練資料有 1000 筆，則實際批次大小會轉換為 100（即 10% 的資料）\n   - 轉換公式：batch_size = int(batch_size * 訓練資料筆數)\n\n2. 本系統的訓練會透過設定 auto_train_max_count 來決定要跑多少次初始化初始動能的梯度遞減訓練，每回裡面才是 epoch 梯度遞減的回數。所以在每一次 auto_train_count 中，早停會發揮作用，不依定會跑滿 epoch，最後 loss_list 會蒐集每一次實跑的 epochs\n\n3. stratify 參數的特殊用法：\n   - 本系統的 stratify 不是簡單的單一欄位名稱，而是通過 produce_stratify 函數動態生成\n   - stratifyLabels 可以是多個欄位的列表，系統會將這些欄位的值組合起來（使用 ' | ' 連接）\n   - 例如：如果 stratifyLabels=['欄位A', '欄位B']，則 stratify 的值會是 '值A | 值B' 的組合形式\n   - 系統會使用 adjust_stratify 函數調整 stratify 的分類數量，以確保訓練/測試集分割的平衡\n   - 這與一般 scikit-learn 的 stratify 參數（通常是單一欄位名稱或陣列）用法不同\n\n請在分析模型配置時，特別注意這些參數的實際含義，避免誤解為標準用法。"
    system_prompt = f'{system_prompt}\n\n{explain_main_info}'
    # 從 prompt 配置中讀取溫度，如果沒有則使用預設值 0.1
    prompt_temperature = prompt_temperature if prompt_temperature else m_evaluationFit_promptInfo.get('temperature', m_temperature)

    user_prompt = user_prompt_template.format(content=content)
    
    # 使用傳入的參數，如果為 None 則使用預設值
    llm_model = model if model is not None else m_model
    llm_provider = provider if provider is not None else get_model_provider(model)
    
    # 根據模型類型選擇正確的參數名稱 # 20260202 改由TextProcessor 分配參數名稱
    # max_token_key = 'max_completion_tokens' if ('gpt52' in llm_model.lower() or 'gpt5.2' in llm_model.lower() or 'gpt-5.2' in llm_model.lower()) else 'max_tokens' # 20260202 改由TextProcessor 分配參數名稱
    # 構建請求數據
    request_data = {
        "prompt": user_prompt,
        "provider": llm_provider,
        "model": llm_model,
        "system_prompt": system_prompt,
        "temperature": prompt_temperature,
        "top_p": m_top_p,
        "max_tokens": m_max_tokens # 20260202 改由TextProcessor 分配參數名稱
    }
    
    # 發送請求
    print(f"\n正在發送請求到: {m_llm_url}")
    print(f"Provider: {llm_provider}, Model: {llm_model}")
    
    try:
        response = requests.post(
            f"{m_llm_url}/chat",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=300  # 5分鐘超時
        )
        
        # 檢查響應狀態
        response.raise_for_status()
        
        # 解析 JSON 響應
        result = response.json()
        
        print("\n" + "="*80)
        print("分析完成！")
        print("="*80)
        print(f"\n回應 ID: {result.get('post_id', 'N/A')}")
        print(f"時間戳: {result.get('timestamp', 'N/A')}")
        
        # 顯示 Token 使用情況
        _format_token_info(result)
        
        # 顯示費用估計
        _format_billing_info(result)
        
        # 提取分析結果（不包含原 prompt）
        analysis_output = result.get('output', '')
        
        # 確保返回的是字符串類型（LLM 分析回復）
        if not isinstance(analysis_output, str):
            analysis_output = str(analysis_output) if analysis_output is not None else ''
        
        print("\n" + "="*80)
        print("分析報告:")
        print("="*80)
        print(analysis_output)
        print("="*80)
        
        # 返回包含分析結果、當次請求的 token 和費用信息的字典
        return {
            'analysis_result': analysis_output,
            'usage': result.get('usage'),  # 當次請求的 token 使用情況
            'current_cost_usd': result.get('current_cost_usd'),  # 當次請求的費用
            'post_id': result.get('post_id'),
            'timestamp': result.get('timestamp')
        }
        
    except requests.exceptions.RequestException as e:
        print(f"\n請求失敗: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"響應狀態碼: {e.response.status_code}")
            print(f"響應內容: {e.response.text}")
        raise
    except json.JSONDecodeError as e:
        print(f"\nJSON 解析失敗: {e}")
        print(f"響應內容: {response.text}")
        raise
    
    
def evaluationFitAnalysis(plan_id: int, seq_no: int = 1, ret: Optional[Dict[str, Any]] = None, stamps: Optional[List[Any]] = None, output_dir: Optional[str] = None, provider: Optional[str] = None, model: Optional[str] = None, user_prompt_template: Optional[str] = None, system_prompt: Optional[str] = None, prompt_temperature: Optional[float] = None, personality: Optional[str] = None, enable_llm_analysis: bool = True, fH_prompt_alias_level: Optional[int]=m_fH_prompt_alias_level, md_output_dir: Optional[str] = None, **kwargs) -> bool:
    """
    Args:
        plan_id: int, the plan sequence number
        seq_no: int, the sequence number
        ret: Optional[Dict[str, Any]], the return value
        stamps: Optional[List[Any]], the stamps list
        output_dir: Optional[str], the output directory
        provider: Optional[str], LLM provider，預設為 m_provider
        model: Optional[str], 模型名稱，預設為 m_model
        user_prompt_template: Optional[str], 用戶提示模板，預設為 m_evaluationFit_promptInfo.get('user_prompt', '')
            - 如果 user_prompt_template 中不包含 {{content}}，則會自動添加
        system_prompt: Optional[str], 系統提示模板，預設為 m_evaluationFit_promptInfo.get('system_prompt', '')
        prompt_temperature: Optional[float], 提示溫度，預設為 m_temperature
        personality: Optional[str], LLM 回復個性描述，預設為 m_formatHelper_promptInfo.get('personality', '')
            - formatHelper 配置固定使用系統預設的 m_formatHelper_promptInfo（從 formatHelper.json 載入）
        enable_llm_analysis: bool, 是否進行 LLM 分析，預設為 False
            - 如果為 False，只生成 Markdown 報告（md），不進行 LLM 分析
            - 如果為 True，生成 Markdown 報告後繼續進行 LLM 分析
        md_output_dir: Optional[str], the markdown output directory
        **kwargs: Any, the keyword arguments
    Returns:
        bool, True if successful, False otherwise
    """
    if not LOGger.isinstance_not_empty(plan_id, int):
        print(LOGger.FAIL + f"Plan sequence number is not a valid integer: \n{plan_id}")
        return False
    if not LOGger.isinstance_not_empty(seq_no, int):
        print(LOGger.FAIL + f"Sequence number is not a valid integer: \n{seq_no}")
        return False
    if not evaluationFit(plan_id, seq_no, ret=ret, stamps=stamps, output_dir=md_output_dir, **kwargs):
        print(LOGger.FAIL + f"Evaluation fit failed")
        return False
    md = ret['md']
    # print(LOGger.OKBLUE + f"md:\n{md}")
    
    # 如果 enable_llm_analysis 為 False，跳過 LLM 分析
    if not enable_llm_analysis:
        if LOGger.isinstance_not_empty(output_dir, str):
            if not save_result(ret, output_dir, stamps=["basic"]):
                print(LOGger.FAIL + f"Save result failed")
                return False
        return True
    
    # 進行 LLM 分析
    # 使用傳入的參數，如果為 None 則使用預設值
    llm_model = model if model is not None else m_model
    llm_provider = provider if provider is not None else get_model_provider(llm_model)
    analysis_response = analyze_content(md, provider=llm_provider, model=llm_model, user_prompt_template=user_prompt_template, system_prompt=system_prompt, prompt_temperature=prompt_temperature, personality=personality, fH_prompt_alias_level=fH_prompt_alias_level)
    
    # analyze_content 現在返回字典，包含 analysis_result、當次請求的 usage 和 current_cost_usd
    if isinstance(analysis_response, dict):
        if isinstance(ret, dict):
            ret['analysis_result'] = analysis_response.get('analysis_result', '')
            ret['usage'] = analysis_response.get('usage')  # 當次請求的 token 使用情況
            ret['current_cost_usd'] = analysis_response.get('current_cost_usd')  # 當次請求的費用
            ret['llm_post_id'] = analysis_response.get('post_id')
            ret['llm_timestamp'] = analysis_response.get('timestamp')
    else:
        # 向後兼容：如果返回的是字符串（舊版本）
        if isinstance(ret, dict):
            ret['analysis_result'] = analysis_response
    if LOGger.isinstance_not_empty(output_dir, str):
        if not save_result(ret, output_dir, stamps=[str(llm_model), str(fH_prompt_alias_level)]):
            print(LOGger.FAIL + f"Save result failed")
            return False
    return True

def save_result(result: Dict[str, Any], output_dir: str, stamps: List[Any]) -> bool:
    """
    Args:
        result: Dict[str, Any], the result to save
        output_dir: str, the output directory
        stamps: List[Any], the stamps to save
    Returns:
        bool, True if successful, False otherwise
    """
    stamps = stamps or ["evaluationFitAnalysis"]
    # stamp = LOGger.stamp_process('', stamps, '','','','_',for_file=True)
    if not LOGger.isinstance_not_empty(result, dict):
        print(LOGger.FAIL + f"Result is not a valid dictionary: \n{result}")
        return False
    print(LOGger.WARNING + f"Saving result to: {output_dir} ...")
    if LOGger.isinstance_not_empty(output_dir, str):
        os.makedirs(output_dir, exist_ok=True)
    
    result_fn = LOGger.stamp_process('', stamps, '','','','_',for_file=True)
    result_file = os.path.join(output_dir, result_fn + '.json')
    # if not LOGger.save_json(result, file=result_file):
    #     print(LOGger.FAIL + f"Save result json failed")
    #     return False
    # print(LOGger.OKCYAN + f"Result json saved to: {result_file}")
    
    if 'analysis_result' in result:
        result_analysis_fn = LOGger.stamp_process('', stamps, '','','','_',for_file=True)
        result_analysis_file = os.path.join(output_dir, result_analysis_fn + '.md')
        with open(result_analysis_file, 'w', encoding='utf-8') as f:
            f.write(result['analysis_result'])
        print(LOGger.OKCYAN + f"Result analysis saved to: {result_analysis_file}")
    if 'infos' in result:
        result_info_fn = LOGger.stamp_process('', ['info', *stamps],'','','_',for_file=True)
        result_info_file = os.path.join(output_dir, result_info_fn + '.json')
        if not LOGger.save_json(result['infos'], file=result_info_file):
            print(LOGger.FAIL + f"Save result info json failed")
            return False
        print(LOGger.OKCYAN + f"Result info saved to: {result_info_file}")
    return True


#########################################################################################
#########################################################################################
# 分析MIC、資料前台的功能
#########################################################################################
#########################################################################################
from package import EIMSDataProc as EDP


def grep_from_EIMSDataProc(filePath: str, selected_header: Optional[List[str]] = None, ret: Optional[Dict[str, Any]] = None, stamps: Optional[List[Any]] = None, dataClassAsgin: Optional[Dict[str, Any]] = None, sht: Optional[int] = 0, **kwargs) -> bool:
    """
    Args:
        filePath: str, the file path
        selected_header: Optional[List[str]], the selected header
        ret: Optional[Dict[str, Any]], the return value
        stamps: Optional[List[Any]], the stamps list
        dataClassAsgin: Optional[Dict[str, Any]], the data class assign
        sht: Optional[int], the sheet number
        **kwargs: Any, the keyword arguments
    Returns:
        bool, True if successful, False otherwise
    """
    # 讀取原始資料
    raw_data = DFP.import_data(filePath, sht=sht)
    if isinstance(selected_header, list):
        raw_data = raw_data[selected_header]
    # 添加 dataClassAsgin 的 log 記錄
    print(LOGger.OKBLUE + f'接收到的 dataClassAsgin: {dataClassAsgin}', stamps=['grep_from_EIMSDataProc'])
    print(LOGger.OKBLUE + f'dataClassAsgin 類型: {type(dataClassAsgin)}', stamps=['grep_from_EIMSDataProc'])

    result = EDP.EIMSDataSetCore(raw_data, dataClassAsgin=dataClassAsgin, **kwargs)
    OutTab_intype_num, DataSetD, stats_info = result

    # <-- return DataSetD: DataFrame 所有欄位的敘述統計
    # <-- return stats_info: dict 統計量資訊

    DataSetD_info = DataSetD.to_dict(orient='records')
    # 隨機抽樣最多300筆資料
    if len(OutTab_intype_num) > 300:
        OutTab_intype_num_sampled = OutTab_intype_num.sample(n=300, random_state=None)
    else:
        OutTab_intype_num_sampled = OutTab_intype_num
    OutTab_intype_num_info = OutTab_intype_num_sampled.to_dict(orient='records')

    if isinstance(ret, dict):
        ret['DataSetD_info'] = DataSetD_info
        ret['datas'] = OutTab_intype_num_info
        ret['stats_info'] = stats_info
    return True

    
def summary_datas(filePath: str, selected_header: Optional[List[str]] = None, ret: Optional[Dict[str, Any]] = None, stamps: Optional[List[Any]] = None, dataClassAsgin: Optional[Dict[str, Any]] = None, sht: Optional[int] = 0, **kwargs) -> bool:
    """
    Args:
        filePath: str, the file path
        selected_header: Optional[List[str]], the selected header
        ret: Optional[Dict[str, Any]], the return value
        stamps: Optional[List[Any]], the stamps list
        dataClassAsgin: Optional[Dict[str, Any]], the data class assign
        sht: Optional[int], the sheet number
        **kwargs: Any, the keyword arguments
    Returns:
        bool, True if successful, False otherwise
    """
    etmd = EvaluationFit()
    if not grep_from_EIMSDataProc(filePath, selected_header=selected_header, ret=ret, stamps=stamps, dataClassAsgin=dataClassAsgin, sht=sht, **kwargs):
        return False
    DataSetD_info = ret['DataSetD_info']
    datas = ret['datas']
    stats_info = ret['stats_info']
    if isinstance(ret, dict):
        ret['DataSetD_info'] = DataSetD_info
        ret['datas'] = datas
        ret['stats_info'] = stats_info
    return True
