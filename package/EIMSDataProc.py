# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 13:38:41 2023
@author: Ian.
EIMSDataProc
"""
import sys
import pandas as pd
import statistics as st
import json
from package import data_analysis as ds
import matplotlib
vs3 = ds.vs3
vs = vs3.vs
import datetime as dtC
from dateutil import parser as dtp
DFP = vs3.DFP
np = DFP.np
dcp = DFP.dcp
LOGger = DFP.LOGger
os=LOGger.os

# 移除熱力圖API客戶端依賴，避免循環引用

#%%
if(False):
    __file__ = 'EIMSDataProc.py'
json_file = '%s_buffer.json'%os.path.basename(__file__.replace('.py',''))
curfn = os.path.basename(__file__)
m_theme = curfn.replace('.py','')
m_config = LOGger.load_json('%s_buffer.json'%m_theme)
m_DataExpfd = m_config.get('dataExpfd', os.path.join('D:\\','JobProject','EIMSsystem','EIMSFileData','DataSet'))
m_heatMapExpfd = m_config.get('heatMapExpfd', os.path.join('D:\\','JobProject','EIMSsystem','EIMSFileData','Heatmap'))
#%%
m_logfile = os.path.join(os.path.dirname(__file__), 'log','log_%t.txt')
LOGger.CreateContainer(m_logfile)
m_addlog = LOGger.addloger(logfile=m_logfile)
m_print = LOGger.addloger(logfile='')
m_debug = LOGger.myDebuger(stamps=[*os.path.basename(__file__).split('.')[:-1]])
m_rename_header = {'count':'dCount', 'mean':'dMean', 'std':'dStd', 'min':'dMin', 'max':'dMax','50%':'dMedian'}
m_plotCorrMatrixMonitor = None
m_plotCorrMatrixBuffer = {'standBy': []}

# 初始化 heatmap buffer replacement
# 移除熱力圖API客戶端依賴，避免循環引用

#%%
def chineseModuleActivate():
    vs3.plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
    
def DataTypeing(s):
    """
    資料型態 DataTypeing
    --> s:DataFrame
    <-- return Date:日期/Int:整數/Float:浮點數/Text:文字/Img:圖片/Path:路徑/Unknown:未知
    """
    s_nonnan = s[np.logical_not(s.isna())]
    is_full_series_same_type = len(set(map(type, s_nonnan)))==1
    if(not is_full_series_same_type):
        m_addlog('series type is not the same...', colora=LOGger.WARNING, stamps=[s.name])
        m_print(set(map(type, s_nonnan)), colora=LOGger.WARNING, stamps=[s.name])
        return 'Unknown'
    s_nonnan_first = s_nonnan.iloc[0]
    if isinstance(s_nonnan_first, (pd.Timestamp, dtC.datetime, dtC.date, dtC.time)):
        return 'Date'
    elif(DFP.astype(s_nonnan_first, d_type=dtp.parse)!=None):
        return 'Date' 
    elif(DFP.astype(s_nonnan_first)!=None):
        if(DFP.astype(s_nonnan_first)==np.nan):
            m_addlog('s_nonnan_first == np.nan', colora=LOGger.WARNING, stamps=[s.name])
            return 'Unknown'
        elif(int(s_nonnan_first//1)==float(s_nonnan_first)):
            return 'Int'
        else:
            return 'Float'
    elif(isinstance(s_nonnan_first, str)):
        if(not (s_nonnan.map(lambda x:x.find('.')==-1)).any()):
            if(s_nonnan_first[-4:]=='.jpg' or s_nonnan_first[-4:]=='.jpg' or s_nonnan_first[-4:]=='.bmp'):
                return 'Img'
            return 'Path'
        return 'Text'
    else:
        return 'Unknown'

def showDataTableShape(**args):
    m_print(**args, colora=LOGger.WARNING)

def EIMSDataSetCore(outputData, dataClassAsgin=None, showTableShape=False, **kwags):
    """
    敘述統計 EIMSDataSetD
    --> outputData:int 資料集
    <-- return OutTab_intype_num: DataFrame 所有欄位的敘述統計
    <-- return DataSetD: DataFrame 所有欄位的敘述統計
    <-- return stats_info: dict 統計量資訊
    """
    LOGger.addDebug(f'EIMSDataSetCore~')
    OutTab = pd.DataFrame(outputData)
    OutTab_columns = dcp(list(OutTab.columns))
    OutTab_intype = OutTab.applymap(DFP.astype_datetime_float_or_remain) #能轉成時間的先轉、再轉成數字、再不然就不動
    if(showTableShape): m_print(
        'OutTab.shape', str(OutTab.shape), 
        colora=LOGger.WARNING, stamps=['phase1'])
    DataSetD = OutTab_intype.describe()
    if(showTableShape): m_print(
        'OutTab.shape', str(OutTab.shape), 
        'DataSetD.shape', str(DataSetD.shape), 
        colora=LOGger.WARNING, stamps=['phase2'])
    # 保留分位數資訊供採樣使用（在 drop 之前提取）
    quantile_info = None
    if '25%' in DataSetD.index and '75%' in DataSetD.index:
        try:
            quantile_info = DataSetD.loc[['min', '25%', '50%', '75%', 'max']].copy()
        except Exception:
            quantile_info = None
    DataSetD = DataSetD.drop(['25%','75%'])
    #全距
    DataSetD = DataSetD.append(pd.DataFrame(
        [[np.max(OutTab_intype[hd]) - np.min(OutTab_intype[hd]) for hd in DataSetD]], columns=DataSetD.columns, index=['dRange']), sort=False)
    if(showTableShape): m_print(
        'DataSetD.shape', str(DataSetD.shape), 
        colora=LOGger.WARNING, stamps=['phase3'])
    #加入其他欄位
    for hd in list(set(OutTab.columns) - set(DataSetD)):
        #DataSetD[hd] = [np.sum(1 - OutTab[hd].isna()), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        DataSetD[hd] = [np.sum(1 - OutTab[hd].isna()), 0, 0, 0, 0, 0, 0]
    if(showTableShape): m_print(
        'DataSetD.shape', str(DataSetD.shape), 
        colora=LOGger.WARNING, stamps=['phase4'])
    #眾數
    dMode_infrm = OutTab.apply(lambda x:LOGger.mode_statistics(x, return_count=True), axis=0)
    dMode_infrm.index = ['dMode','dModeCount']
    DataSetD = DataSetD.append(dMode_infrm, sort=False)
    if(showTableShape): m_print(
        'DataSetD.shape', str(DataSetD.shape), 
        colora=LOGger.WARNING, stamps=['phase5'])
    # DataSetD = DataSetD.append(pd.DataFrame(dMode.values.reshape(1,-1), columns=dMode.index, index=['dMode']), sort=False)
    LOGger.addDebug(f'EIMSDataSetCore| a......')
    #值的種類數
    dAmount = OutTab.apply(lambda x:len(set(x)), axis=0)
    DataSetD = DataSetD.append(pd.DataFrame(dAmount.values.reshape(1,-1), columns=dAmount.index, index=['dAmount']), sort=False)
    if(showTableShape): m_print(
        'DataSetD.shape', str(DataSetD.shape), 
        colora=LOGger.WARNING, stamps=['phase6'])
    #資料型態
    DataType = OutTab_intype.apply(DataTypeing, axis=0)
    DataSetD = DataSetD.append(pd.DataFrame(DataType.values.reshape(1,-1), columns=dAmount.index, index=['DataType']), sort=False)
    if(showTableShape): m_print(
        'DataSetD.shape', str(DataSetD.shape), 
        colora=LOGger.WARNING, stamps=['phase7'])
    #資料類別 R:連續/C:分類
    DataClass = np.full(OutTab.shape[1], 'R') # 一開始全部預設連續
    DataClass = np.where((DataType!='Int') & (DataType!='Float') & (DataType!='Date'), 'C', DataClass)
    DataClass = np.where((DataType=='Int') & (dAmount<10), 'C', DataClass)
    if(isinstance(dataClassAsgin, dict)):
        mask = OutTab.columns.map(lambda x:dataClassAsgin[x] if(x in dataClassAsgin) else 'U')
        DataClass = np.where(mask!='U', mask, DataClass)
        m_print(str(DataClass), colora=LOGger.WARNING)
    DataSetD = DataSetD.append(pd.DataFrame(DataClass.reshape(1,-1), columns=dAmount.index, index=['DataClass']), sort=False)
    if(showTableShape): m_print(
        'DataSetD.shape', str(DataSetD.shape), 
        colora=LOGger.WARNING, stamps=['phase8'])
    #缺失值
    dMissCount = np.sum(OutTab.isna())
    DataSetD = DataSetD.append(pd.DataFrame(dMissCount.values.reshape(1,-1), columns=dMissCount.index, index=['dMissCount']), sort=False)
    if(showTableShape): m_print(
        'DataSetD.shape', str(DataSetD.shape), 
        colora=LOGger.WARNING, stamps=['phase9'])
    #還原原columns順序
    DataSetD = DataSetD[OutTab_columns]
    LOGger.addDebug(f'EIMSDataSetCore| b......')
    if(showTableShape): m_print(
        'DataSetD.shape', str(DataSetD.shape), 
        colora=LOGger.WARNING, stamps=['phase10'])
    #轉置
    DataSetD = DataSetD.T
    DataSetD["count"] = DataSetD["count"].astype(int)
    DataSetD["dModeCount"] = DataSetD["dModeCount"].astype(int)
    #特徵值    
    DataSetD["FieldName"] = OutTab_columns
    DataSetD["xHeader"] = OutTab_columns
    if(showTableShape): m_print(
        'DataSetD.shape', str(DataSetD.shape), 
        colora=LOGger.WARNING, stamps=['phase12'])
    #序號
    DataSetD["SeqNo"] = [i + 1 for i in range(len(DataSetD))]
    if(showTableShape): m_print(
        'DataSetD.shape', str(DataSetD.shape), 
        colora=LOGger.WARNING, stamps=['phase13'])
    #啟用
    DataSetD["IsEnable"] = "true"
    LOGger.addDebug(f'EIMSDataSetCore| c......')
    if(showTableShape): m_print(
        'DataSetD.shape', str(DataSetD.shape), 
        colora=LOGger.WARNING, stamps=['phase14'])
    #重新命名
    DataSetD = DataSetD.rename(columns=m_rename_header)
    #轉str
    DataSetD = DataSetD.astype(str)
    # 處理 DataSetD 中的 nan 值，將其替換為空白字串
    DataSetD = DataSetD.replace('nan', '')
    OutTab_intype_num = OutTab_intype[OutTab_intype.columns[DataClass=='R']].copy()
    for col in OutTab_intype_num.columns:
        if OutTab_intype_num[col].dtype.kind in ['M', 'O']:  # datetime 或 object 類型
            if pd.api.types.is_datetime64_any_dtype(OutTab_intype_num[col]):
                OutTab_intype_num[col] = pd.to_numeric(OutTab_intype_num[col], errors='coerce')
    OutTab_intype_str = OutTab_intype[OutTab_intype.columns[DataClass!='R']].copy()
    OutTab_intype_str = OutTab_intype_str.fillna('')
    OutTab_intype_str = OutTab_intype_str.astype(str)
    if(showTableShape): m_print(
        'OutTab_intype.shape', str(OutTab_intype.shape), 
        'OutTab_intype_num.shape', str(OutTab_intype_num.shape), 
        'OutTab_intype_str.shape', str(OutTab_intype_str.shape),
        'DataClass', str(DataClass),
        'DataSetD.shape', str(DataSetD.shape), 
        colora=LOGger.WARNING, stamps=['phase17'])
    OutTab_new = pd.concat([OutTab_intype_num, OutTab_intype_str], axis=1)
    # 準備統計量資訊供採樣使用
    stats_info = {
        'quantiles': quantile_info,  # 分位數資訊
        'data_class': dict(zip(OutTab_columns, DataClass)),  # 資料類別 (R/C)
        'data_type': dict(zip(OutTab_columns, DataType)),  # 資料型態
        'describe': DataSetD.T  # 完整的統計資訊（轉置後，每欄一行）
    }
    # return OutTab_intype_num, DataSetD
    return OutTab_new, DataSetD, stats_info

def EIMSDataSetD(outputData, CorrMatrixFilePath=None, plan_id=None, seq_no=None, ret=None, stamps=None, max_samples=None, **kwags):
    result = EIMSDataSetCore(outputData, **kwags)
    OutTab_intype_num, DataSetD, stats_info = result
    
    # 計算 MIC 並儲存到全域變數中，供 reportImportancies 使用
    try:
        if not OutTab_intype_num.empty and plan_id is not None and seq_no is not None:
            m_addlog(f'開始計算MIC，數據形狀: {OutTab_intype_num.shape}', colora=LOGger.WARNING)
            
            # 使用ds.saveCorrelation計算MIC並儲存pkl, xlsx
            stamps = [f'PlanID_{plan_id}', f'SeqNo_{seq_no}'] if plan_id and seq_no else ['direct_calc']
            
        success = ds.saveCorrelation(
            OutTab_intype_num, 
            method='mic', 
            exp_fd=CorrMatrixFilePath if CorrMatrixFilePath else m_heatMapExpfd, 
            stamps=stamps, 
            ret=ret,
            max_samples=max_samples,
            stats_info=stats_info,  # 傳遞統計量資訊供採樣使用
            **kwags
        )
        if(not success):    
            ret['pd_corr'] = None

    except Exception as e:
        m_addlog(f'MIC計算過程異常: {e}', colora=LOGger.FAIL)
        LOGger.exception_process(e, logfile=m_logfile, stamps=['MIC_calculation'])
    
    return DataSetD.to_dict(orient ='records') #轉成list形式

def EIMSDataSetH(outputData, stamps=None, **kwags):
    # if(not LOGger.isinstance_not_empty(CorrMatrixFileName, str)):
    #     m_addlog('EIMSDataSetH: CorrMatrixFileName is not a valid string:\n', f"`{CorrMatrixFileName}`", colora=LOGger.FAIL)
    #     return None
    # CorrMatrixFileBasename = CorrMatrixFileName if(CorrMatrixFileName[-4:] in ['.png','.jpg']) else f"{CorrMatrixFileName}.png"
    # CorrMatrixFilePath = os.path.join(m_heatMapExpfd, CorrMatrixFileBasename)
    _, DataSetD, _ = EIMSDataSetCore(outputData, stamps=stamps, **kwags)
    return DataSetD.to_dict(orient ='records')

def saveCorrelationJob(stamps=None, ret=None, **kwags):
    try:
        itemStamps = []
        stamps = stamps if(isinstance(stamps, list)) else []
        global m_plotCorrMatrixBuffer
        processed_count = 0
        
        # 持續處理 buffer 中的任務
        while True:
            # 找到符合條件的任務索引（目錄路徑，不包含檔案名）
            target_index = -1
            for i, item in enumerate(m_plotCorrMatrixBuffer['standBy']):
                if len(item) >= 2 and item[1].find('.') == -1:
                    target_index = i
                    break
            
            # 如果沒有符合條件的任務，結束處理
            if target_index == -1:
                m_addlog('saveCorrelationJob: No more tasks to process', colora=LOGger.WARNING, stamps=stamps)
                break
                
            # 從全域 buffer 中移除任務
            OutTab_intype_num, FilePath, itemStamps = m_plotCorrMatrixBuffer['standBy'].pop(target_index)
            processed_count += 1
            
            m_addlog('saveCorrelationJob start. Counts:%d, Processing #%d'%(len(m_plotCorrMatrixBuffer['standBy']), processed_count), colora=LOGger.WARNING, stamps=stamps)
            m_addlog(OutTab_intype_num.shape, FilePath, itemStamps, colora=LOGger.WARNING, stamps=stamps)
            itemStamps = itemStamps if(isinstance(itemStamps, list)) else []
            
            if not ds.saveCorrelation(OutTab_intype_num, method='mic', exp_fd=FilePath, stamps=[*stamps, *itemStamps], ret=ret, **kwags):
                m_addlog(f'saveCorrelation failed for {FilePath}', colora=LOGger.FAIL, stamps=stamps)
                
    except Exception as e:
        LOGger.exception_process(e, logfile=m_logfile, stamps=[*stamps, *itemStamps])
        return False
    finally:
        remaining_tasks = len([x for x in m_plotCorrMatrixBuffer['standBy'] if x[1].find('.')==-1])
        m_addlog('saveCorrelationJob over. Processed:%d, Remain:%d'%(processed_count, remaining_tasks), colora=LOGger.WARNING, stamps=[*stamps,*itemStamps])
    return True

def plotCorrMatrixJob(stamps=None, ret=None, **kwags):
    try:
        itemStamps = []
        stamps = stamps if(isinstance(stamps, list)) else []
        global m_plotCorrMatrixBuffer
        processed_count = 0
        
        # 持續處理 buffer 中的任務，直到清空
        while True:
            standBy = m_plotCorrMatrixBuffer['standBy']
            if not standBy:  # buffer 空了就結束
                m_addlog('plotCorrMatrixJob: No more tasks to process', colora=LOGger.WARNING, stamps=stamps)
                break
                
            m_addlog('plotCorrMatrixJob start. Counts:%d, Processing #%d'%(len(standBy), processed_count + 1), colora=LOGger.WARNING, stamps=stamps)
            OutTab_intype_num, FilePath, itemStamps = standBy.pop(0)
            processed_count += 1
            itemStamps = itemStamps if(isinstance(itemStamps, list)) else []
            
            # 檢查檔案是否已存在（應該檢查 heatmap.png 檔案）
            heatmap_file = os.path.join(FilePath, 'heatmap.png') if os.path.isdir(FilePath) else FilePath
            if os.path.isfile(heatmap_file):
                m_addlog(f"File {heatmap_file} already exists, skipping...", colora=LOGger.WARNING, stamps=stamps)
                continue  # 跳過這個任務，處理下一個
                
            if not plotCorrMatrix(OutTab_intype_num, FilePath, stamps=[*stamps, *itemStamps], ret=ret, **kwags):
                m_addlog(f'plotCorrMatrix failed for {FilePath}', colora=LOGger.FAIL, stamps=stamps)
                # 繼續處理其他任務，不要因為一個失敗就停止
                
    except Exception as e:
        LOGger.exception_process(e, logfile=m_logfile, stamps=[*stamps, *itemStamps])
        return False
    finally:
        remaining_tasks = len(m_plotCorrMatrixBuffer['standBy'])
        m_addlog('plotCorrMatrixJob over. Processed:%d, Remain:%d'%(processed_count, remaining_tasks), colora=LOGger.WARNING, stamps=[*stamps,*itemStamps])
    return True
    
def plotCorrMatrix(OutTab_intype_num, FilePath, stamps=None, ret=None, **kwags):
    try:
        stamps = stamps if(isinstance(stamps, list)) else []
        # 確保 ret 是字典
        if not isinstance(ret, dict):
            ret = {}
        
        # 設定 matplotlib 使用非互動式後端，避免 GUI 執行緒問題
        import matplotlib
        matplotlib.use('Agg')  # 使用非互動式後端
        
        OutTab_intype_num.corr() #測試所謂數據型資料能否生成熱力圖
        # FilePath 是目錄路徑，需要加上檔名
        hmfile = os.path.join(FilePath, 'heatmap.png')
        LOGger.CreateFile(hmfile, lambda f:ds.plotRegressionHeatmap(
            OutTab_intype_num, stamps=stamps, file=f, ret=ret, **kwags))
    except Exception as e:
        LOGger.exception_process(e, logfile=m_logfile, stamps=stamps)
        return False
    return True

def plotCorrMatrixMonitorThreading():
    try:
        global m_plotCorrMatrixMonitor
        m_addlog('m_plotCorrMatrixMonitor is activating.....', colora=LOGger.WARNING)
        if(m_plotCorrMatrixMonitor is not None):
            m_addlog('m_plotCorrMatrixMonitor is activated!!!!', colora=LOGger.WARNING)
            return True
        m_plotCorrMatrixMonitor = LOGger.myThreadAgent(target_core=plotCorrMatrixJob, time_waiting=60, immediate_start=True)
        m_addlog('m_plotCorrMatrixMonitor', m_plotCorrMatrixMonitor)
    except Exception as e:
        LOGger.exception_process(e, logfile=m_logfile, stamps=['plotCorrMatrixMonitoring initial'])
        return False
    return True

if(False):
    with open("..\\EIMSFileData\\DataSet\\1\\20231218180124.json", "r") as f:
        outputData= json.load(f)
        OutTab = pd.DataFrame(outputData)
        OutTab.describe()
#%%
def EIMSDataProcInitalTest():
    dataJsonFile = os.path.join(m_DataExpfd, '3','202506170930.json')
    if(not os.path.isfile(dataJsonFile)):
        return {}, None
    FileName = os.path.basename(dataJsonFile)
    outputData = LOGger.load_json(dataJsonFile)
    OutTab = pd.DataFrame(outputData)
    print(str(OutTab)[:200])
    return outputData, FileName

def scenario():
    try:
        if(not plotCorrMatrixMonitorThreading()):
            return False
        outputData, FilePath = EIMSDataProcInitalTest()
        EIMSDataSetD(outputData,FilePath)
        while(True):
            LOGger.time.sleep(1)
    finally:
        if(m_plotCorrMatrixMonitor):     m_plotCorrMatrixMonitor.stop()
    return True

#%%
if(__name__=='__main__'):
    scenario()