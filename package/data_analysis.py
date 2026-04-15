#
# For data conditioning
#
import pickle
from keras.models import model_from_json
from matplotlib import axis
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder as LBC
from sklearn.preprocessing import OneHotEncoder as OHC
from sklearn.metrics import cohen_kappa_score as cohkpa
from sklearn.metrics import mean_squared_log_error
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
from numpy.random import seed
from numpy.random import choice
from minepy import MINE
import sys
import joblib
# import matplotlib.patches as mpatches
# 
# Other essential libraries
#
from copy import deepcopy as dcp
import numpy as np
from numpy import argmax
from numpy import argmin
import pandas as pd
from sklearn.preprocessing import StandardScaler as Stdscr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics as skm
from datetime import datetime as dt
from mpl_toolkits.mplot3d import Axes3D
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

# 設定 matplotlib 使用非互動式後端，避免 GUI 執行緒問題
import matplotlib
matplotlib.use('Agg')

from package import visualization3 as vs3
DFP = vs3.DFP
plt = vs3.plt
vs2 = vs3.vs2
vs = vs2.vs
import seaborn as sns
LOGger = DFP.LOGger
import time
#%%
if(False):
    __file__ = 'data_analysis.py'
json_file = '%s_buffer.json'%os.path.basename(__file__.replace('.py',''))
m_addlog = LOGger.addloger(logfile=os.path.join('log', 'log_%t.txt'))
m_theme = LOGger.stamp_process('',[os.path.basename(os.path.dirname(__file__)), 'dataAnalysis'],'','','','_',for_file=True)
m_debug = LOGger.myDebuger(stamps=[*os.path.basename(__file__).split('.')[:-1]]) # 'debug.pkl'
m_print = LOGger.addloger(logfile='')
#%%   
def serializeZone(zone, **kwags):
    """
    Standard

    Parameters
    ----------
    zone : TYPE
        DESCRIPTION.
    **kwags : TYPE
        DESCRIPTION.

    Returns
    -------
    zone_serialized : TYPE
        DESCRIPTION.

    """
    zoneClassName = LOGger.type_string(zone)
    zone_serialized = {}
    zone_serialized['core'] = list(zone)
    zone_serialized['zoneClass'] = zoneClassName
    for k in getattr(zone, 'mainAttrs', []):
        zone_serialized[k] = LOGger.transform_class2dict(getattr(zone,k))
        zone_serialized[k].pop('export', None) if(isinstance(zone_serialized[k], dict)) else None
    if(getattr(zone,'mv',None) is not None):
        mv = zone.mv
        for k,v in mv.mainAttrs.items():
            if(k in ['all_categories']):
                zone_serialized[k] = getattr(zone, '%s_file'%k, zone_serialized[k])
                continue
            zone_serialized[k] = getattr(zone, k, v)
    return zone_serialized

def deserialPathLike(zone, attrName, attr, exactCriterion=None, stamps=None, source_dir='.', **kwags):
    stamps = stamps if(isinstance(stamps, list)) else []
    exactCriterion = exactCriterion if(callable(exactCriterion)) else (lambda x:not DFP.isiterable(x) and x is not None)
    attrPath = dcp(attr)
    attr = None
    # LOGger.addDebug('attrPath', attrPath)
    # LOGger.addDebug('attr', attr)
    if(exactCriterion(attrPath)):
        if(LOGger.isinstance_not_empty(attrPath, str)):
            file = os.path.join(source_dir, attrPath)
            if(os.path.exists(file)):
                jsonDict = LOGger.load_json(file)
                # TODO: 要更客製化
                attr = dcp(jsonDict.get('classes_', jsonDict) if(isinstance(jsonDict, dict)) else jsonDict)
                setattr(zone, '%s_file'%attrName, attrPath)
                return attr
        LOGger.addlog('`%s`'%attrPath, "invalid", logfile='', colora=LOGger.FAIL, stamps=['deserialPathLike']+stamps+[attrName])
        m_debug.updateDump({'attrName':attrName, 'attr':attr, 'zone':zone, 'stamps':stamps, 'source_dir':source_dir})
    return attr

def deserializeZone(zone, dic, config=None, config_dir='.', **kwags):
    """
    Standard

    Parameters
    ----------
    zone : TYPE
        DESCRIPTION.
    dic : TYPE
        DESCRIPTION.
    **kwags : TYPE
        DESCRIPTION.

    Returns
    -------
    bool
        DESCRIPTION.

    """
    dic = LOGger.mydict(dic if(isinstance(dic, dict)) else {})
    config = config if(isinstance(config, dict)) else {}
    # config 設定 zone.mainAttrs
    for k in zone.mainAttrs:
        if(not k in dic and not k in config and not hasattr(zone, k)):
            LOGger.addlog('zone of type %s missing %s'%(type(zone), k), logfile='', stamps=['deserializeZone']+zone.stamps, colora=LOGger.FAIL)
            return False
        if(k in dic):    attr = dic[k]
        elif(k in config):    attr = config[k]
        else:   attr = getattr(zone, k)
        attr = LOGger.transform_dict2class(attr) if(isinstance(attr, dict) or isinstance(attr, str)) else attr
        setattr(zone, k, attr)
        LOGger.addlog('set %s'%attr, logfile='', colora=LOGger.OKCYAN, stamps=['deserializeZone']+zone.stamps+[k])
    # config 設定 mv.mainAttrs
    for k,v in getattr(getattr(zone, 'mv', None), 'mainAttrs', {}).items():
        attr = dcp(dic.get(k, config.get(k,v)))
        # iterable的時後不需要source_dir，pathLike的時候source_dir用config_dir
        if(k in ['all_categories']):
            all_categories = getattr(zone, k, None)
            attr = LOGger.mylist(tuple(all_categories)) if(DFP.isiterable(all_categories)) else deserialPathLike(zone, k, attr, source_dir=config_dir)
        zone.mv.mainAttrs[k] = attr
        if(k in zone.mv.drawAttrs): zone.mv.drawAttrs[k] = attr
        setattr(zone, k, attr) # mv.mainAttrs 傳給 zone.mainAttrs
        LOGger.addlog('set %s'%attr, logfile='', colora=LOGger.OKBLUE, stamps=['deserializeZone mv']+zone.stamps+[k])
    return True

class HEADER_ZONE(LOGger.mylist):
    def __init__(self, core, data_prop='continuous', preprocessing='', preprocessor_file=None, preprocessor_dir='.', 
                 cell_size=None, feaVarAxis=None, reshapeThruMethod='reshapeThruFlatten', activation=None, pad_value=0, exp_fd='.',
                 all_categories=None, stamps=None, hidden_layer_sizes=None, hidden_layer_nns=None, 
                 score_threshold=None, default_value=np.nan, **kwags):
        super().__init__(core)
        self.stamps = stamps if(isinstance(stamps, list)) else []
        self.data_prop = LOGger.mystr(data_prop)
        self.preprocessor = None
        self.referenceData = None
        self.preprocessedData = None
        self.preprocessor_file = preprocessor_file
        self.preprocessor_dir = preprocessor_dir
        self.cell_size = cell_size
        self.feaVarAxis = feaVarAxis
        self.pad_value = pad_value
        self.default_value = default_value
        self.all_categories = None
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_layer_nns = hidden_layer_nns
        self.mainAttrs = ['data_prop','preprocessing','preprocessor_file','all_categories','cell_size','feaVarAxis',
                          'activation','hidden_layer_sizes','pad_value','default_value','hidden_layer_nns']
        self.exp_fd = exp_fd
        self.fn = LOGger.stamp_process('',stamps,'','','','_',for_file=1)
        self.score_threshold = score_threshold
        if(isinstance(self.preprocessor_file, str)):
            self.preprocessor = joblib.load(os.path.join(os.path.join(self.preprocessor_dir, self.preprocessor_file)))
        
    def serializeZone(self):
        return serializeZone(self)
    
    def deserializeZone(self, dic=None, config=None, **kwags):
        return deserializeZone(self, dic, config=config, **kwags)
    
def activateZone(dic, zoneClassName=None, preprocessor_dir='.', stamps=None, exp_fd='.', config=None, **kwags):
    """
    

    Parameters
    ----------
    dic : TYPE
        DESCRIPTION.
    zoneClassName : TYPE, optional
        DESCRIPTION. The default is None.
    default_zoneClassName : TYPE, optional
        DESCRIPTION. The default is 'HEADER_ZONE'.
    preprocessor_dir : TYPE, optional
        DESCRIPTION. The default is '.'.
    **kwags : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    data_prop = LOGger.transform_dict2class(dic['data_prop'])
    preprocessing = LOGger.transform_dict2class(dic.get('preprocessing'))
    headerZone = HEADER_ZONE(dic['core'], data_prop=data_prop, preprocessing=preprocessing, exp_fd=exp_fd, 
                             cell_size=dic.get('cell_size',None), 
                             preprocessor_file=dic.get('preprocessor_file',None),
                             preprocessor_dir=preprocessor_dir, pad_value=dic.get('pad_value',0), 
                             activation=dic.get('activation',''), #LOGger.transform_dict2class在內部
                             lossmethod=dic.get('lossmethod',''), #LOGger.transform_dict2class在內部
                             all_categories=dic.get('all_categories',None),
                             stamps=stamps)
    print('headerZone.editor', getattr(headerZone,'editor','?'))
    if(not headerZone.deserializeZone(dic, config=config, config_dir=preprocessor_dir)):
        return None
    return headerZone

def saveMINE(handler, **kwags):
    mine_info = {
            'alpha': kwags['mineAlpha'],  # 從handler中獲取參數
            'c': kwags['mineC'],  
            'results': {
                'mic': handler.estimator.mic(),
                'mas': handler.estimator.mas(),
                'mev': handler.estimator.mev(),
                'mcn': handler.estimator.mcn()
            }
        }
    LOGger.save_json(mine_info, os.path.join(handler.exp_fd, 'mine_info.json'))

def loadMINE(handler, ret=None,**kwags):
    mine_info = LOGger.load_json(handler.estimatorFile)
    mine = MINE(alpha=mine_info['alpha'], c=mine_info['c'])
    if(isinstance(ret, dict)):  ret['mine_info'] = mine_info
    return mine

def factorRelativityAnalysisMIC(inspectData, otherData, mine=MINE(alpha=0.6, c=15), ret=None, stamps=None, addlog=m_addlog, handler=None, **kwags):
    if(isinstance(inspectData, (list,tuple))):
        if(not factorRelativityAnalysisMIC(np.array(inspectData), otherData, mine=mine, ret=ret, stamps=stamps, 
                                           addlog=addlog, handler=handler, **kwags)):
            return False
        return True
    if(not hasattr(inspectData,'shape')):
        return False
    if(len(inspectData.shape)==0 or len(inspectData.shape)>2):
        return False
    stamps = stamps if(isinstance(stamps, list)) else []
    inspectData = getattr(inspectData, 'values', inspectData)
    if(len(inspectData.shape)==2):
        if(inspectData.shape[1]==1):
            if(not factorRelativityAnalysisMIC(inspectData.reshape(-1), otherData, mine=mine, ret=ret, stamps=stamps, 
                                               addlog=addlog, handler=handler, **kwags)):
                return False
            return True
    if(len(inspectData.shape)==2):
        # 很多個inspectDataDimension
        mic_dfs = {}
        retTemp = {}
        for i in range(inspectData.shape[1]):
            retTemp.clear()
            arr = dcp(inspectData[:,i])
            if(not factorRelativityAnalysisMIC(arr, otherData, mine=mine, stamps=stamps, handler=handler, addlog=addlog, ret=retTemp, **kwags)):
                if(addlog): addlog('failed!!!', colora=LOGger.FAIL, stamps=[*stamps, i])
                return False
            outputStamp = dcp(LOGger.stamp_process('',[*stamps,i],'','','','_'))
            mic_dfs[outputStamp] = dcp(retTemp)
        if(isinstance(ret, dict)):  ret['eva_dfs'] = dcp(mic_dfs)
        if(handler is not None):   
            setattr(handler, 'eva_dfs', dcp(mic_dfs))
        return True
    elif(len(inspectData.shape)>2 or len(inspectData.shape)==0):
        if(addlog): addlog('failed with inspectData.shape:%s!!!'%str(inspectData.shape), colora=LOGger.FAIL, stamps=stamps)
        return False
    mic_results = {}
    
    for column in otherData.columns:
        mine.compute_score(otherData[column], inspectData)
        mic_results[column] = mine.mic()
    
    # 將結果轉為 DataFrame 並排序
    mic_df = pd.DataFrame(list(mic_results.items()), columns=["Variable", "MIC"])
    mic_df = mic_df.sort_values(by="MIC", ascending=False)
    
    if(isinstance(ret,dict)):   ret['eva_df'] = mic_df
    if(handler is not None):    handler.eva_df = mic_df
    if(addlog): addlog(str(mic_df), stamps=stamps, colora=LOGger.WARNING)
    return True

def factorRelativityAnalysis(dfX, dfy, handler=None, corMethod=None):
    pass
    
def evaluationScenario(handler, **kwags):
    eva_df = handler.eva_df
    eva_dfs = getattr(handler,'eva_dfs',None)
    if(LOGger.isinstance_not_empty(eva_dfs, dict)):
        xlsxFile = os.path.join(handler.exp_fd, 'eva_dfs.xlsx')
        LOGger.CreateContainer(xlsxFile)
        wrt = pd.ExcelWriter(xlsxFile)
        for k,v in eva_dfs.items():
            if(isinstance(v.get('eva_df'), pd.core.frame.DataFrame)): v['eva_df'].to_excel(wrt, sheet_name=k)
            LOGger.addDebug(k, v)
        wrt.save()
    if(kwags.get('selectThreshold',None) is not None):
        selectThreshold = float(kwags['selectThreshold'])
        selected = list(tuple(eva_df['Variable'][eva_df['MIC']>selectThreshold].values))
        handler.addlog('selected:', ', '.join(list(map((lambda x:'"%s"'%x), selected))), colora=LOGger.WARNING)
        LOGger.save_json(selected, os.path.join(handler.exp_fd, 'selected.json'))
    if(getattr(handler,'estimator',None) is not None):
        saveMINE(handler, **kwags)
    return True

class BaseSampler:
    """採樣類別基礎類別，定義統一的 sample 方法接口"""
    
    def __init__(self, random_state=42, stats_info=None, **kwargs):
        self.random_state = random_state
        self.stats_info = stats_info  # 預先計算的統計量資訊
        self.kwargs = kwargs
    
    def sample(self, df, max_samples):
        """
        採樣方法，子類別必須實作
        
        參數:
            df: pandas DataFrame
            max_samples: int, 最大樣本數
        
        返回:
            pandas DataFrame: 採樣後的資料
        """
        raise NotImplementedError("子類別必須實作 sample 方法")


class NoneSampler(BaseSampler):
    """不採樣"""
    
    def sample(self, df, max_samples):
        return df


class RandomSampler(BaseSampler):
    """簡單隨機採樣"""
    
    def sample(self, df, max_samples):
        return df.sample(n=max_samples, random_state=self.random_state).reset_index(drop=True)


class QuantileSampler(BaseSampler):
    """分位數採樣：確保各分位數都有代表，保留極值"""
    
    def __init__(self, random_state=42, n_quantiles=5, stats_info=None, **kwargs):
        super().__init__(random_state, stats_info, **kwargs)
        self.n_quantiles = n_quantiles
    
    def sample(self, df, max_samples):
        sampled_indices = set()
        
        # 對每個數值欄位進行分位數採樣
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return df.sample(n=max_samples, random_state=self.random_state).reset_index(drop=True)
        
        for col in numeric_cols:
            # 優先使用預先計算的分位數資訊
            if self.stats_info and self.stats_info.get('quantiles') is not None:
                quantile_info = self.stats_info['quantiles']
                if col in quantile_info.columns:
                    # 使用預先計算的分位數值
                    quantile_values = quantile_info[col].values
                    # 如果需要的分位數多於預先計算的，補充計算
                    if len(quantile_values) < self.n_quantiles + 2:
                        quantiles = np.linspace(0, 1, self.n_quantiles + 2)
                        quantile_values = df[col].quantile(quantiles).values
                else:
                    # 該欄位沒有預先計算的分位數，重新計算
                    quantiles = np.linspace(0, 1, self.n_quantiles + 2)
                    quantile_values = df[col].quantile(quantiles).values
            else:
                # 沒有預先計算的分位數資訊，重新計算
                quantiles = np.linspace(0, 1, self.n_quantiles + 2)
                quantile_values = df[col].quantile(quantiles).values
            
            # 為每個分位數找到最近的資料點
            for q_val in quantile_values:
                idx = (df[col] - q_val).abs().idxmin()
                sampled_indices.add(idx)
        
        # 如果採樣點不足，隨機補充
        remaining = max_samples - len(sampled_indices)
        if remaining > 0:
            remaining_indices = set(df.index) - sampled_indices
            if len(remaining_indices) > 0:
                additional = np.random.choice(list(remaining_indices), 
                                             min(remaining, len(remaining_indices)), 
                                             replace=False)
                sampled_indices.update(additional)
        
        return df.loc[list(sampled_indices)[:max_samples]].reset_index(drop=True)


class HybridSampler(BaseSampler):
    """混合採樣：結合分位數採樣和隨機採樣"""
    
    def __init__(self, random_state=42, quantile_ratio=0.3, **kwargs):
        super().__init__(random_state, **kwargs)
        self.quantile_ratio = quantile_ratio
    
    def sample(self, df, max_samples):
        n_quantile = int(max_samples * self.quantile_ratio)
        n_random = max_samples - n_quantile
        
        # 分位數採樣（先獲取索引，不 reset_index）
        sampled_indices = set()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return df.sample(n=max_samples, random_state=self.random_state).reset_index(drop=True)
        
        for col in numeric_cols:
            quantiles = np.linspace(0, 1, 6)  # 5 quantiles + min + max
            quantile_values = df[col].quantile(quantiles)
            for q_val in quantile_values:
                idx = (df[col] - q_val).abs().idxmin()
                sampled_indices.add(idx)
        
        # 如果採樣點不足，隨機補充
        remaining_quantile = n_quantile - len(sampled_indices)
        if remaining_quantile > 0:
            remaining_indices = set(df.index) - sampled_indices
            if len(remaining_indices) > 0:
                additional = np.random.choice(list(remaining_indices), 
                                             min(remaining_quantile, len(remaining_indices)), 
                                             replace=False)
                sampled_indices.update(additional)
        
        quantile_indices = list(sampled_indices)[:n_quantile]
        quantile_samples = df.loc[quantile_indices]
        
        # 隨機採樣剩餘部分
        remaining_df = df[~df.index.isin(quantile_indices)]
        if len(remaining_df) > 0:
            random_samples = remaining_df.sample(n=min(n_random, len(remaining_df)), 
                                                random_state=self.random_state)
            result = pd.concat([quantile_samples, random_samples], ignore_index=True)
        else:
            result = quantile_samples.reset_index(drop=True)
        
        return result


class StratifiedSampler(BaseSampler):
    """分層採樣：根據類別變數分層，保持各類別比例"""
    
    def __init__(self, random_state=42, stratify_col=None, stats_info=None, **kwargs):
        super().__init__(random_state, stats_info, **kwargs)
        self.stratify_col = stratify_col
    
    def sample(self, df, max_samples):
        stratify_col = self.stratify_col
        if stratify_col is None:
            # 優先使用預先計算的 DataClass 資訊
            if self.stats_info and self.stats_info.get('data_class'):
                data_class = self.stats_info['data_class']
                # 選擇第一個分類變數（DataClass='C'）
                cat_cols = [col for col, dc in data_class.items() if dc == 'C']
                if len(cat_cols) > 0:
                    stratify_col = cat_cols[0]
                else:
                    # 如果沒有分類變數，使用第一個類別變數
                    cat_cols = df.select_dtypes(include=['object', 'category']).columns
                    if len(cat_cols) > 0:
                        stratify_col = cat_cols[0]
                    else:
                        return df.sample(n=max_samples, random_state=self.random_state).reset_index(drop=True)
            else:
                # 自動選擇第一個類別變數
                cat_cols = df.select_dtypes(include=['object', 'category']).columns
                if len(cat_cols) > 0:
                    stratify_col = cat_cols[0]
                else:
                    return df.sample(n=max_samples, random_state=self.random_state).reset_index(drop=True)
        
        if stratify_col not in df.columns:
            LOGger.addlog(f'分層欄位 {stratify_col} 不存在，使用隨機採樣', colora=LOGger.WARNING)
            return df.sample(n=max_samples, random_state=self.random_state).reset_index(drop=True)
        
        # 計算各層的採樣數量
        value_counts = df[stratify_col].value_counts()
        proportions = value_counts / len(df)
        
        sampled_dfs = []
        sampled_indices = set()
        for value, count in value_counts.items():
            layer_df = df[df[stratify_col] == value]
            n_sample = max(1, int(proportions[value] * max_samples))
            n_sample = min(n_sample, len(layer_df))
            sampled_layer = layer_df.sample(n=n_sample, random_state=self.random_state)
            sampled_dfs.append(sampled_layer)
            sampled_indices.update(sampled_layer.index.tolist())
        
        result = pd.concat(sampled_dfs, ignore_index=True)
        
        # 如果總數不足，隨機補充
        if len(result) < max_samples:
            remaining = max_samples - len(result)
            remaining_df = df[~df.index.isin(sampled_indices)]
            if len(remaining_df) > 0:
                additional = remaining_df.sample(n=min(remaining, len(remaining_df)), 
                                               random_state=self.random_state)
                result = pd.concat([result, additional], ignore_index=True)
        
        return result


class DensitySampler(BaseSampler):
    """密度感知採樣：根據資料密度採樣，確保稀疏區域也有代表"""
    
    def sample(self, df, max_samples):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            return df.sample(n=max_samples, random_state=self.random_state).reset_index(drop=True)
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # 使用 KMeans 聚類
            n_clusters = min(max_samples // 10, 50)
            n_clusters = max(n_clusters, 2)
            
            # 標準化數值欄位
            X = df[numeric_cols].fillna(df[numeric_cols].median())
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # KMeans 聚類
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            # 從每個簇中採樣
            samples_per_cluster = max_samples // n_clusters
            sampled_indices = []
            
            for cluster_id in range(n_clusters):
                cluster_mask = clusters == cluster_id
                cluster_df = df[cluster_mask]
                
                if len(cluster_df) > 0:
                    n_sample = min(samples_per_cluster, len(cluster_df))
                    sampled = cluster_df.sample(n=n_sample, random_state=self.random_state)
                    sampled_indices.extend(sampled.index.tolist())
            
            # 如果還有名額，隨機補充
            remaining = max_samples - len(sampled_indices)
            if remaining > 0:
                remaining_indices = set(df.index) - set(sampled_indices)
                if len(remaining_indices) > 0:
                    additional = np.random.choice(list(remaining_indices), 
                                                 min(remaining, len(remaining_indices)), 
                                                 replace=False)
                    sampled_indices.extend(additional)
            
            sampled_indices = list(set(sampled_indices))[:max_samples]
            return df.loc[sampled_indices].reset_index(drop=True)
            
        except Exception as e:
            LOGger.addlog(f'密度採樣失敗: {str(e)}，回退到隨機採樣', colora=LOGger.WARNING)
            return df.sample(n=max_samples, random_state=self.random_state).reset_index(drop=True)


# 採樣類別映射字典
SAMPLER_CLASSES = {
    'none': NoneSampler,
    'random': RandomSampler,
    'quantile': QuantileSampler,
    'hybrid': HybridSampler,
    'stratified': StratifiedSampler,
    'density': DensitySampler,
}


def get_sampler(sampling_method='hybrid', random_state=42, stats_info=None, **kwargs):
    """
    根據採樣方法名稱獲取對應的採樣類別實例
    
    參數:
        sampling_method: str, 採樣方法名稱
        random_state: int, 隨機種子
        stats_info: dict, 預先計算的統計量資訊（包含 quantiles, data_class 等）
        **kwargs: 其他參數（傳遞給對應的採樣類別）
    
    返回:
        BaseSampler: 採樣類別實例
    """
    sampler_class = SAMPLER_CLASSES.get(sampling_method.lower())
    if sampler_class is None:
        LOGger.addlog(f'未知的採樣方法: {sampling_method}，使用隨機採樣', colora=LOGger.WARNING)
        sampler_class = RandomSampler
    
    return sampler_class(random_state=random_state, stats_info=stats_info, **kwargs)

def to_numeric_codes(s, fill_missing=False):
    """
    將任意 Series 轉為數值向量與是否為離散特徵的旗標
    參考 calImp.py 的實現
    
    參數:
        s: pandas Series
        fill_missing: bool, 對於連續數據是否填充缺失值 (預設False，保留NaN)
    
    返回:
        tuple: (數值向量, 是否為離散特徵)
    """
    if np.issubdtype(s.dtype, np.number):
        # 連續數值：保留 NaN 不填充，讓 MIC 計算判斷是否有效
        x = s.astype(float).copy()
        if fill_missing:
            med = np.nanmedian(x) if np.isfinite(np.nanmedian(x)) else 0.0
            x = x.fillna(med).to_numpy()
        else:
            x = x.to_numpy()
        return x, False
    else:
        # 離散類別：factorize，包含 NaN 作為一個類別
        vals = s.astype("object").fillna("__NaN__")
        codes, _ = pd.factorize(vals, sort=True)
        return codes.astype(float), True

def _compute_mic_pair(args):
    """計算一對變量的MIC值（用於並行處理）"""
    i, j, col_i, col_j, x_data, y_data, alpha, c = args
    try:
        if i == j:
            return (i, j, 1.0)
        
        mine = MINE(alpha=alpha, c=c)
        
        # 過濾掉任一序列中的 NaN，只計算有效的配對
        valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
        
        if valid_mask.sum() > 0:  # 至少有有效數據點
            x_valid = x_data[valid_mask]
            y_valid = y_data[valid_mask]
            
            if len(x_valid) > 0 and len(y_valid) > 0:
                mine.compute_score(x_valid, y_valid)
                mic = mine.mic()
                return (i, j, mic)
        
        return (i, j, np.nan)
    except Exception as e:
        return (i, j, np.nan)

def mic_matrix(df, handle_discrete=True, n_jobs=None, use_fast_correlation=False,
               max_samples=None, sampling_method='hybrid', random_state=42, stats_info=None, **kwargs):
    """
    計算DataFrame中所有列之間的MIC值矩陣
    支援離散數據處理和並行計算
    
    參數:
        df: pandas DataFrame
        handle_discrete: bool, 是否處理離散數據 (預設True)
        n_jobs: int, 並行處理的工作進程數。None表示使用所有CPU核心，-1表示使用所有核心-1
        use_fast_correlation: bool, 是否使用快速的Pearson相關性代替MIC（僅適用於純數值數據）
        max_samples: int, 最大樣本數。None 表示不採樣
        sampling_method: str, 採樣方法
            - 'none': 不採樣
            - 'random': 簡單隨機採樣
            - 'quantile': 分位數採樣（保留極值和分位數）
            - 'hybrid': 混合採樣（推薦，結合分位數和隨機採樣）
            - 'stratified': 分層採樣（需要指定 stratify_col）
            - 'density': 密度感知採樣（基於聚類）
        random_state: int, 隨機種子
        **kwargs: 其他參數
            - stratify_col: str, 分層採樣的欄位名稱（用於 stratified 方法）
            - quantile_ratio: float, 混合採樣中分位數採樣的比例（預設 0.3）
            - n_quantiles: int, 分位數採樣的數量（預設 5）
    
    返回:
        mic_df: 包含MIC值的DataFrame
    """
    # 資料採樣
    if max_samples is not None and len(df) > max_samples:
        original_size = len(df)
        LOGger.addDebug(f'資料量 {original_size} 超過 {max_samples}，使用 {sampling_method} 採樣...')
        sampler = get_sampler(sampling_method=sampling_method, random_state=random_state, stats_info=stats_info, **kwargs)
        df = sampler.sample(df, max_samples=max_samples)
        LOGger.addDebug(f'採樣後資料量: {len(df)} (減少 {original_size - len(df)} 筆)')
    
    n = df.shape[1]  # 獲取列數
    columns = getattr(df,'columns', np.arange(n).tolist())
    
    # 快速相關性選項（僅適用於純數值數據）
    if use_fast_correlation and handle_discrete:
        numeric_only = df.select_dtypes(include=[np.number])
        if numeric_only.shape[1] == df.shape[1]:
            LOGger.addDebug('Using fast Pearson correlation instead of MIC')
            corr_matrix = numeric_only.corr().values
            return pd.DataFrame(corr_matrix, index=columns, columns=columns)
    
    # 預處理數據：轉換為數值格式
    processed_data = {}
    discrete_flags = {}
    
    if handle_discrete:
        for col in df.columns:
            # 離散數據時，fill_missing=True 讓 factorize 處理 NaN
            processed_data[col], discrete_flags[col] = to_numeric_codes(df[col], fill_missing=False)
    else:
        # 原有邏輯：直接使用數值，但保留 NaN
        for col in df.columns:
            processed_data[col] = df[col].astype(float).to_numpy()
            discrete_flags[col] = False
    
    # 創建空的MIC矩陣，使用 NaN 初始化
    mic_values = np.full((n, n), np.nan)
    
    # 準備並行計算的參數
    alpha = 0.6
    c = 15
    
    # 決定是否使用並行計算
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)  # 預設使用所有核心-1
    elif n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    # 準備所有需要計算的配對
    pairs_to_compute = []
    for i, col_i in enumerate(columns):
        for j, col_j in enumerate(columns[i:], i):  # 只需計算上三角矩陣
            if i == j:
                mic_values[i, j] = 1.0  # 對角線為1
            else:
                x_data = processed_data[col_i]
                y_data = processed_data[col_j]
                pairs_to_compute.append((i, j, col_i, col_j, x_data, y_data, alpha, c))
    
    # 並行計算MIC
    if len(pairs_to_compute) > 0 and n_jobs > 1:
        LOGger.addDebug(f'Computing MIC matrix in parallel with {n_jobs} processes...')
        try:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                results = list(executor.map(_compute_mic_pair, pairs_to_compute))
            
            # 填充結果到矩陣
            for i, j, mic in results:
                mic_values[i, j] = mic
                mic_values[j, i] = mic  # MIC矩陣是對稱的
        except Exception as e:
            LOGger.addlog(f'Parallel computation failed, falling back to serial: {e}', colora=LOGger.WARNING)
            # 回退到串行計算
            for i, j, col_i, col_j, x_data, y_data, alpha, c in pairs_to_compute:
                result = _compute_mic_pair((i, j, col_i, col_j, x_data, y_data, alpha, c))
                _, _, mic = result
                mic_values[i, j] = mic
                mic_values[j, i] = mic
    else:
        # 串行計算
        LOGger.addDebug('Computing MIC matrix serially...')
        for i, j, col_i, col_j, x_data, y_data, alpha, c in pairs_to_compute:
            result = _compute_mic_pair((i, j, col_i, col_j, x_data, y_data, alpha, c))
            _, _, mic = result
            mic_values[i, j] = mic
            mic_values[j, i] = mic
    
    # 轉換為DataFrame
    mic_df = pd.DataFrame(mic_values, 
                         index=columns, 
                         columns=columns)
    
    return mic_df

def plotDataDistribution(data, fig=None, visualizeMethod='report', file='', handler=None, exp_fd='.', stamps=None, figsize=(10,15), **kwags):
    if(exp_fd is None): exp_fd = LOGger.execute('exp_fd', handler, default='.')
    if(fig is None): fig = LOGger.execute('fig', handler, default=None)
    stamps = stamps if(isinstance(stamps, list)) else []
    kwags['infrms'] = kwags.get('infrms', {})
    kwags['infrm_default'] = kwags.get('infrm_default', {})
    res = getattr(vs3,'visualizeMethod', vs3.report)(data=data, fig=fig, file=file, **kwags)
    file = os.path.join(handler.exp_fd, '%s.jpg'%LOGger.stamp_process('',[*stamps, 'dataDistribution'],'','','','_',for_file=True))
    LOGger.CreateFile(file, lambda f:vs3.end(fig, file=f))
    return res

def get_title_bgcolor(r):
        # 使用相关系数的绝对值作为颜色强度
        intensity = np.abs(r)
        
        if r >= 0:
            # 正相关：从白色渐变到红色
            return (1, 1*(1-intensity), 1*(1-intensity), 0.3)
        else:
            # 负相关：从白色渐变到蓝色
            return (1*(1-intensity), 1*(1-intensity), 1, 0.3)

def saveCorrelation(data, method='mic', exp_fd='.', stamps=None, ret=None, **kwags):
    try:
        stamps = stamps if(isinstance(stamps, list)) else []
        
        # 支持並行計算
        n_jobs = kwags.get('n_jobs', None)
        use_fast_correlation = kwags.get('use_fast_correlation', False)
        max_samples = kwags.get('max_samples', None)  # 預設不採樣
        sampling_method = kwags.get('sampling_method', 'hybrid')
        random_state = kwags.get('random_state', 42)
        stats_info = kwags.get('stats_info', None)  # 預先計算的統計量資訊
        corr = mic_matrix(data, n_jobs=n_jobs, use_fast_correlation=use_fast_correlation,
                         max_samples=max_samples, sampling_method=sampling_method, 
                         random_state=random_state, stats_info=stats_info, **kwags)
        m_print(f'[RET_TRACE] mic_matrix 計算完成，corr 形狀: {corr.shape}', stamps=stamps, colora=LOGger.WARNING)
        
        if(isinstance(ret, dict)):
            ret['corr'] = corr
        
        # 將corr轉換為numpy數組
        pd_corr = pd.DataFrame(corr, index=data.columns, columns=data.columns)
        
        if(isinstance(ret, dict)):
            ret['pd_corr'] = pd_corr
        
        DFP.save(pd_corr, exp_fd=exp_fd, fn='corr',save_types=['pkl','xlsx'])
        m_print(f'[RET_TRACE] saveCorrelation 完成，路徑: {exp_fd}', stamps=stamps, colora=LOGger.OKCYAN)
            
    except Exception as e:
        m_print(f'[RET_TRACE] saveCorrelation 發生錯誤: {str(e)}', stamps=stamps, colora=LOGger.FAIL)
        LOGger.exception_process(e, logfile='', stamps=stamps)
        LOGger.addlog('saveCorrelation failed!!!', logfile='', stamps=stamps, colora=LOGger.FAIL)
        return False
    finally:
        LOGger.addDebug('saveCorrelation over!!!!')
    return True

def is_numeric_column(series):
    """判斷欄位是否為數值型（連續型）"""
    # 首先檢查原始數據類型
    if series.dtype in ['int64', 'float64', 'int32', 'float32']:
        return True
    
    # 檢查是否可以轉換為數值
    try:
        pd.to_numeric(series, errors='raise')
        return True
    except (ValueError, TypeError):
        pass
    
    # 檢查是否為聚合後的數值型數據（字符串格式的數值）
    if series.dtype == 'object':
        # 取樣本檢查是否為數值字符串
        sample = series.dropna().head(10)
        numeric_count = 0
        for val in sample:
            try:
                float(str(val))
                numeric_count += 1
            except (ValueError, TypeError):
                pass
        
        # 如果大部分樣本都是數值字符串，視為數值型
        if len(sample) > 0 and numeric_count / len(sample) >= 0.8:
            return True
    
    return False

def is_categorical_column(series):
    """判斷欄位是否為類別型（離散型）"""
    return not is_numeric_column(series) or series.nunique() <= 10

def drawCorrmap(df, stamps=None, handler=None, ret=None, mask=None, maskColumnName='mask', 
                    height=3, diag_kind='auto', columns=None, corr=None, suptitle='', **kwags):
    """
        自定義的相關性矩陣繪圖函數，根據資料型態選擇不同的繪圖方式
        
        Parameters
        ----------
        df : pd.DataFrame
            要繪製的資料
        stamps : list, optional
            時間戳記
        handler : object, optional
            處理器物件
        ret : dict, optional
            回傳結果字典
        mask : array-like, optional
            遮罩陣列
        maskColumnName : str, optional
            遮罩欄位名稱
        height : int, optional
            圖表高度
        columns : list, optional
            要繪製的欄位
        corr : array-like, optional
            相關性矩陣
        suptitle : str, optional
            圖表標題
        **kwags : dict
            其他參數
            
        Returns
        -------
        bool
            是否成功繪製
    """
    try:
        # 設定中文字型
        vs3.plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
        
        if(mask is not None):
            if(mask.shape[0]!=df.shape[0]):
                LOGger.addlog('shape inconsistence!!', mask.shape[0], df.shape[0], colora=LOGger.FAIL)
                return False
            df[maskColumnName] = mask
        stamps = stamps if(isinstance(stamps, list)) else []
        if(not isinstance(columns,list)):    columns = LOGger.mylist(df.columns)
        
        # 移除遮罩欄位從繪圖欄位中
        plot_columns = [c for c in columns if c != maskColumnName]
        n_cols = len(plot_columns)
        
        # 檢查是否有可繪製的欄位
        if n_cols == 0:
            LOGger.addlog('No columns to plot after filtering', colora=LOGger.FAIL)
            return False
        
        # 創建子圖
        fig, axes = vs3.plt.subplots(n_cols, n_cols, figsize=(height*n_cols, height*n_cols))
        
        # 確保 axes 是 2D 陣列
        if n_cols == 1:
            axes = np.array([[axes]])
        elif n_cols > 1 and axes.ndim == 1:
            # 這種情況不應該發生，但為了安全起見
            axes = axes.reshape(n_cols, n_cols)
        elif n_cols > 1 and axes.ndim == 2:
            # 正常情況，不需要改變
            pass
        
        # 為每個軸對繪製圖表
        LOGger.addDebug(f'drawCorrmap: plotting {n_cols}x{n_cols} grid for columns: {plot_columns}')
        
        # 性能優化：預先計算欄位索引映射，避免在循環中重複查找
        column_to_index = {col: idx for idx, col in enumerate(df.columns)}
        
        # 性能優化：預先判斷所有欄位的數值型態
        column_is_numeric = {}
        for col in plot_columns:
            if col in df.columns:
                column_is_numeric[col] = is_numeric_column(df[col])
        
        # 資源優化：在繪圖循環中定期讓出 CPU 時間
        yield_interval = kwags.get('yield_plot_interval', 10)  # 每處理 10 個子圖休息一次（提高效率）
        plot_count = 0
        
        for i, col_i in enumerate(plot_columns):
            for j, col_j in enumerate(plot_columns):
                plot_count += 1
                
                # 每處理一定數量的子圖後，讓出 CPU 時間
                if plot_count % yield_interval == 0:
                    import time
                    time.sleep(1)  # 短暫休息，讓其他程序有機會執行
                    # 注意：不使用 plt.pause()，因為在非互動式後端（Agg）下可能會阻塞
                try:
                    ax = axes[i, j]
                except IndexError as e:
                    LOGger.addlog(f'Axis index error: i={i}, j={j}, axes.shape={axes.shape}', colora=LOGger.FAIL)
                    raise e
                
                if i == j:
                    # 對角線：繪製單變量分布
                    data_col = df[col_i].dropna()
                    if column_is_numeric.get(col_i, False):
                        # 連續型：直方圖
                        ax.hist(data_col, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                        ax.set_title(f'{col_i} 分布', fontproperties=vs3.MJHfontprop())
                    else:
                        # 類別型：計數圖
                        value_counts = data_col.value_counts()
                        ax.bar(range(len(value_counts)), value_counts.values, color='lightcoral')
                        ax.set_xticks(range(len(value_counts)))
                        ax.set_xticklabels(value_counts.index, rotation=45)
                        ax.set_title(f'{col_i} 計數', fontproperties=vs3.MJHfontprop())
                else:
                    # 非對角線：根據兩個變數的型態選擇繪圖方式
                    # 性能優化：使用預先計算的數值型態判斷
                    x_is_numeric = column_is_numeric.get(col_j, False)
                    y_is_numeric = column_is_numeric.get(col_i, False)
                    
                    # 取得共同的非空值索引（只對需要的欄位進行操作）
                    if x_is_numeric and y_is_numeric:
                        # 連續 vs 連續：只需要兩個欄位
                        common_idx = df[[col_i, col_j]].dropna().index
                        x_clean = df.loc[common_idx, col_j]
                        y_clean = df.loc[common_idx, col_i]
                    else:
                        # 其他情況也需要兩個欄位
                        common_idx = df[[col_i, col_j]].dropna().index
                        x_clean = df.loc[common_idx, col_j]
                        y_clean = df.loc[common_idx, col_i]
                    
                    if x_is_numeric and y_is_numeric:
                        # 連續 vs 連續：散點圖 + 回歸線
                        ax.scatter(x_clean, y_clean, alpha=0.6, color='blue')
                        # 添加回歸線
                        try:
                            z = np.polyfit(x_clean.astype(float), y_clean.astype(float), 1)
                            p = np.poly1d(z)
                            ax.plot(x_clean, p(x_clean.astype(float)), "r--", alpha=0.8)
                        except:
                            pass
                    elif x_is_numeric and not y_is_numeric:
                        # 連續 vs 類別：箱線圖（垂直）
                        # x軸是連續型，y軸是類別型，但箱線圖需要按類別分組
                        categories = y_clean.unique()
                        box_data = [x_clean[y_clean == cat].astype(float) for cat in categories]
                        # ax.boxplot(box_data, labels=categories, axis='y')
                        ax.boxplot(box_data, labels=categories, vert=False)
                        # ax.tick_params(axis='x', rotation=45)
                        # # 軸標籤：x軸顯示類別，y軸顯示數值
                        # ax.set_xlabel(col_j, fontproperties=vs3.MJHfontprop())  # 類別軸
                        # ax.set_ylabel(col_i, fontproperties=vs3.MJHfontprop())  # 數值軸
                        # continue  # 跳過後面的通用軸標籤設置
                    elif not x_is_numeric and y_is_numeric:
                        # 類別 vs 連續：箱線圖（垂直）
                        # x軸是類別型，y軸是連續型
                        categories = x_clean.unique()
                        box_data = [y_clean[x_clean == cat].astype(float) for cat in categories]
                        ax.boxplot(box_data, labels=categories, vert=True)
                        ax.tick_params(axis='x', rotation=45)
                        # # 軸標籤：x軸顯示類別，y軸顯示數值
                        # ax.set_xlabel(col_j, fontproperties=vs3.MJHfontprop())  # 類別軸
                        # ax.set_ylabel(col_i, fontproperties=vs3.MJHfontprop())  # 數值軸
                        # continue  # 跳過後面的通用軸標籤設置
                    else:
                        # 類別 vs 類別：熱力圖
                        crosstab = pd.crosstab(y_clean, x_clean)
                        # crosstab = pd.crosstab(x_clean, y_clean)
                        im = ax.imshow(crosstab.values, cmap='Blues', aspect='auto')
                        ax.set_xticks(range(len(crosstab.columns)))
                        ax.set_yticks(range(len(crosstab.index)))
                        ax.set_xticklabels(crosstab.columns, rotation=45)
                        ax.set_yticklabels(crosstab.index)
                        
                        # 性能優化：只在類別數量較少時添加數值標註（避免過多文字導致繪圖變慢）
                        max_categories_for_text = kwags.get('max_categories_for_text', 10)
                        if len(crosstab.columns) <= max_categories_for_text and len(crosstab.index) <= max_categories_for_text:
                            for x in range(len(crosstab.columns)):
                                for y in range(len(crosstab.index)):
                                    ax.text(x, y, crosstab.iloc[y, x], ha='center', va='center', rotation=45)
                    ax.set_ylabel(col_i, fontproperties=vs3.MJHfontprop())
                # 設定軸標籤
                ax.set_xlabel(col_j, fontproperties=vs3.MJHfontprop())
                
                # 添加相關性係數（如果有）
                if hasattr(corr,'shape') and corr.shape[0] > i and corr.shape[1] > j:
                    try:
                        # 性能優化：使用預先計算的索引映射
                        if col_i in column_to_index and col_j in column_to_index:
                            i_idx = column_to_index[col_i]
                            j_idx = column_to_index[col_j]
                            if i_idx == j_idx:
                                continue
                            if i_idx < corr.shape[0] and j_idx < corr.shape[1]:
                                r = np.clip(corr[i_idx, j_idx], a_max=1.0, a_min=0.0)
                                if not np.isnan(r):  # 只顯示有效的相關性係數
                                    facecolorDyeMethod = kwags.get('facecolorDyeMethod')
                                    infrm = f'corr: {r:.3f}'
                                    title = ax.set_title(infrm, fontsize=8)
                                    if callable(facecolorDyeMethod):
                                        title.set_bbox(dict(
                                            facecolor=facecolorDyeMethod(r),
                                            edgecolor='none',
                                            alpha=0.7
                                        ))
                    except (IndexError, KeyError, ValueError) as e:
                        LOGger.addDebug(f'Correlation coefficient access failed for {col_i}-{col_j}: {e}')
                        pass
        
        # 設定整體標題
        if suptitle:
            fig.suptitle(suptitle, fontproperties=vs3.MJHfontprop(), fontsize=16)
        
        try:
            fig.tight_layout()
        except Exception as layout_e:
            LOGger.addlog('tight_layout failed:', str(layout_e), colora=LOGger.WARNING)
        
        # 創建一個類似 seaborn pairplot 的回傳物件
        class CustomPairGrid:
            def __init__(self, fig):
                self.fig = fig
        
        snsed = CustomPairGrid(fig)
        
        if isinstance(ret, dict):
            ret['fig'] = fig
            ret['snsed'] = snsed
            LOGger.addDebug('drawCorrmap: snsed set successfully')
        else:
            LOGger.addlog('drawCorrmap: ret is not dict', type(ret), colora=LOGger.FAIL)
        
        return True
        
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=stamps)
        return False

def plotCorrelationMonitorThreading(data, stamps=None, handler=None, exp_fd=None, file=None, numColor=None, 
                                    mask=None, height=5, ret=None, **kwags):
    """
    使用線程在背景執行 plotCorrelation，避免阻塞主程序
    """
    try:
        # 強制禁用背景執行，避免無限遞迴
        thread_kwags = kwags.copy()
        thread_kwags['use_background'] = False  # 在線程中強制同步執行
        thread_kwags['background_async'] = False  # 禁用異步背景執行
        
        thd = LOGger.threading.Thread(
            target=plotCorrelation, 
            args=[data], 
            kwargs={
                'handler': handler, 
                'stamps': stamps, 
                'exp_fd': exp_fd, 
                'file': file, 
                'numColor': numColor,
                'mask': mask, 
                'height': height, 
                'ret': ret,
                **thread_kwags
            }
        )
        thd.daemon = True  # 設為守護線程，主程序結束時自動結束
        thd.start()
        LOGger.addDebug('plotCorrelationMonitorThreading start', stamps=stamps)
        if hasattr(handler, 'thds'):
            handler.thds.append(thd)
        return True
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=stamps)
        return False

def plotCorrelation(data, stamps=None, handler=None, exp_fd=None, file=None, numColor=None, 
                          mask=None, height=5, ret=None, **kwags):
    try:
        # 檢查資料量，如果超過閾值則使用背景執行和資源優化
        data_size_threshold = kwags.get('data_size_threshold', 10)
        use_background = kwags.get('use_background', None)  # None 表示自動判斷
        
        # 自動判斷是否使用背景執行
        if use_background is None:
            use_background = data.shape[0] > data_size_threshold
        
        # 如果使用背景執行，直接返回（不等待完成）
        if use_background and kwags.get('background_async', True):
            LOGger.addlog(f'資料量較大 (n={data.shape[0]})，使用背景執行繪圖', 
                        stamps=stamps, colora=LOGger.OKCYAN)
            return plotCorrelationMonitorThreading(
                data, stamps=stamps, handler=handler, exp_fd=exp_fd, 
                file=file, numColor=numColor, mask=mask, height=height, 
                ret=ret, **kwags
            )
        
        fbasename = 'corrplot.png'
        stamps = stamps if(isinstance(stamps, list)) else []
        if(LOGger.isinstance_not_empty(file, str)): 
            exp_fd = (os.path.dirname(file) or '.')
            fbasename = os.path.basename(file)
        elif(LOGger.isinstance_not_empty(exp_fd, str)): 
            exp_fd = LOGger.execute('exp_fd', handler, default=exp_fd, not_found_alarm=False)
            fbasename = LOGger.stamp_process('',[*stamps, fbasename],'','','','_',for_file=True)
        
        # corr = data.corr() #測試所謂數據型資料能否生成熱力圖
        # 支持並行計算和快速相關性選項
        n_jobs = kwags.get('n_jobs', None)  # None表示自動選擇，-1表示所有核心
        use_fast_correlation = kwags.get('use_fast_correlation', False)
        
        # 資料量大時，降低計算優先級或使用快速模式
        if data.shape[0] > data_size_threshold:
            if not use_fast_correlation:
                # 自動啟用快速相關性計算（如果資料是純數值型）
                use_fast_correlation = kwags.get('auto_fast_correlation', True)
            # 降低繪圖解析度
            if 'dpi' not in kwags:
                kwags['dpi'] = kwags.get('low_res_dpi', 100)  # 預設降低到 100 DPI
            # 設定 matplotlib 為非互動模式，減少資源使用
            matplotlib.pyplot.ioff()  # 關閉互動模式
            LOGger.addlog(f'資料量較大 (n={data.shape[0]})，啟用資源優化模式', 
                        stamps=stamps, colora=LOGger.OKCYAN)
        
        # 休息點 1：計算 MIC 前
        if data.shape[0] > data_size_threshold:
            import time
            time.sleep(0.05)  # 讓出 CPU 時間
        
        corr = mic_matrix(data, n_jobs=n_jobs, use_fast_correlation=use_fast_correlation) #算MIC
        
        # 休息點 2：計算 MIC 後
        if data.shape[0] > data_size_threshold:
            time.sleep(0.05)  # 讓出 CPU 時間
        
        # 將corr轉換為numpy數組
        pd_corr = pd.DataFrame(corr, index=data.columns, columns=data.columns)
        if(isinstance(ret, dict)):  ret['pd_corr'] = pd_corr
        DFP.save(pd_corr, exp_fd=exp_fd, fn='corr',save_types=['pkl','xlsx'])

        LOGger.addDebug('plotCorrelation start!!!! file:', os.path.join(exp_fd, fbasename))
        # 確保 corr 矩陣與 data 的欄位順序一致
        corr_values = corr.values if hasattr(corr, 'values') else corr
        
        # 休息點 3：繪圖前
        if data.shape[0] > data_size_threshold:
            time.sleep(0.05)  # 讓出 CPU 時間
        
        if(not drawCorrmap(data, stamps=stamps, mask=mask, height=height, corr=corr_values, ret=ret,
                                         facecolorDyeMethod=get_title_bgcolor, suptitle='n:%s'%(data.shape[0]), **kwags)):
            return False
        file = os.path.join(exp_fd, fbasename)
        if isinstance(ret, dict) and 'snsed' in ret:
            snsed = ret['snsed']
            # 資料量大時，降低保存時的解析度
            if data.shape[0] > data_size_threshold and 'dpi' in kwags:
                # 調整 figure 的 DPI
                if hasattr(snsed, 'fig'):
                    snsed.fig.set_dpi(kwags['dpi'])
            LOGger.CreateFile(file, lambda f:vs3.end(snsed.fig, file=file, dpi=kwags.get('dpi', None)))
        else:
            LOGger.addlog('ret or snsed not found, cannot save heatmap', colora=LOGger.FAIL, stamps=stamps)
            return False
        LOGger.addDebug('plotCorrelation end!!!! file:', os.path.join(exp_fd, fbasename))
        return True
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=stamps)
        return False
    finally:
        LOGger.addDebug('plotCorrelation over!!!!')

def plotRegressionHeatmap(data, stamps=None, handler=None, exp_fd=None, file=None, numColor=None, 
                          mask=None, height=5, ret=None, **kwags):
    try:
        fbasename = 'corrplot.png'
        stamps = stamps if(isinstance(stamps, list)) else []
        if(LOGger.isinstance_not_empty(file, str)): 
            exp_fd = (os.path.dirname(file) or '.')
            fbasename = os.path.basename(file)
        elif(LOGger.isinstance_not_empty(exp_fd, str)): 
            exp_fd = LOGger.execute('exp_fd', handler, default=exp_fd, not_found_alarm=False)
            fbasename = LOGger.stamp_process('',[*stamps, fbasename],'','','','_',for_file=True)
        LOGger.addDebug('plotRegressionHeatmap start!!!! file:', os.path.join(exp_fd, fbasename))
        # corr = data.corr() #測試所謂數據型資料能否生成熱力圖
        # 支持並行計算
        n_jobs = kwags.get('n_jobs', None)
        use_fast_correlation = kwags.get('use_fast_correlation', False)
        corr = mic_matrix(data, n_jobs=n_jobs, use_fast_correlation=use_fast_correlation) #算MIC
        
        # 將corr轉換為numpy數組
        pd_corr = pd.DataFrame(corr, index=data.columns, columns=data.columns)
        if(isinstance(ret, dict)):  ret['pd_corr'] = pd_corr
        DFP.save(pd_corr, exp_fd=exp_fd, fn='corr',save_types=['pkl','xlsx'])

        LOGger.addDebug('plotRegressionHeatmap start!!!! file:', os.path.join(exp_fd, fbasename))
        if(not vs3.drawRegressionCorrmap(data, stamps=stamps, mask=mask, height=height, corr=corr, ret=ret, facecolorDyeMethod=get_title_bgcolor, suptitle='n:%s'%(data.shape[0]), **kwags)):
            return False
        file = os.path.join(exp_fd, fbasename)
        if isinstance(ret, dict) and 'snsed' in ret:
            snsed = ret['snsed']
            LOGger.CreateFile(file, lambda f:vs3.end(snsed.fig, file=file))
        else:
            LOGger.addlog('ret or snsed not found, cannot save heatmap', colora=LOGger.FAIL, stamps=stamps)
            return False
        LOGger.addDebug('plotRegressionHeatmap end!!!! file:', os.path.join(exp_fd, fbasename))
        return True
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=stamps)
        return False
    finally:
        LOGger.addDebug('plotRegressionHeatmap over!!!!')

def plotRegressionHeatmapMonitorThreading(OutTab_intype_num, handler=None, stamps=None, file='', height=5, ret=None, **kwags):
    try:
        thd = LOGger.threading.Thread(target=plotRegressionHeatmap, args=[OutTab_intype_num], 
                                      kwargs={'handler': handler, 'stamps':stamps, 'file':file, 'height':height, 'ret':ret})
        thd.start()
        LOGger.addDebug('plotRegressionHeatmapMonitorThreading start')
        if(hasattr(handler,'thds')):    handler.thds.append(thd)
    except Exception as e:
        LOGger.excetion_process(e, logfile='', stamps=['plotRegressionHeatmapMonitoring initial'], colora=LOGger.FAIL)
        return False
    return True

def regressionHeatmapScenario(data, selectedHeader=None, handler=None, file='', stamps=None, fig=None, exp_fd=None, height=5, **kwags):
    if(exp_fd is None): exp_fd = LOGger.execute('exp_fd', handler, default='.')
    if(fig is None): fig = LOGger.execute('fig', handler, default=None)
    stamps = stamps if(isinstance(stamps, list)) else []
    if(isinstance(selectedHeader, list)):   data = data[selectedHeader]
    mask = np.full(data.shape[1], False)
    if(data.applymap(lambda x:DFP.astype(x,default=np.nan)).isna().any().any()):
        mask = data.applymap(lambda x:DFP.astype(x,default=np.nan)).isna().any().values
        nanColumns = data.columns[mask]
        m_print('nanColumns', str(nanColumns), colora=LOGger.FAIL)
    data = data[list(tuple(data.columns[np.logical_not(mask)]))].astype(float, errors='ignore')
    if(not plotRegressionHeatmapMonitorThreading(data, handler=handler, file=file, height=height, **kwags)):
        return False
    return True

def cleanNaDataByKeyHeader(data, keyHeader, **kwags):
    dataTemp = dcp(data)
    m_print('data before', dataTemp.shape[0], stamps=['cleanDataByKeyHeader'])
    dataTemp = dataTemp.applymap(lambda x:DFP.astype(x,default=np.nan))
    dataTemp = dataTemp.dropna(subset=keyHeader, axis=0, how='any')
    m_print('data after', dataTemp.shape[0], 'keyHeader', DFP.parse(keyHeader), 
            stamps=['cleanDataByKeyHeader'])
    return dataTemp

def drawHistogram(OutTab_intype_num, stamps=None, handler=None, file='', **kwags):
    vs3.report_normhist(OutTab_intype_num, stamps=stamps, handler=handler, file=file, **kwags)
    return True

def plotHistogramScenario(data, selectedHeader=None, handler=None, mask=None, maskHeader=None, 
                          file='', stamps=None, fig=None, exp_fd=None, **kwags):
    stamps = stamps if(isinstance(stamps, list)) else []
    exp_fd = exp_fd if(LOGger.isinstance_not_empty(exp_fd, str)) else LOGger.execute('exp_fd', handler, default='.', not_found_alarm=False)
    file = file if(LOGger.isinstance_not_empty(file, str)) else LOGger.execute(
        'file', handler, not_found_alarm=False, 
        default=os.path.join(exp_fd, LOGger.stamp_process('',[*stamps, 'Histogram'],'','','','_',for_file=True)))
    # LOGger.addDebug(str(data))
    if(mask is None):
        if(maskHeader is not None):
            mask = data[maskHeader].applymap(lambda x:DFP.parse(x)).values
    dataTemp = (data[selectedHeader] if(isinstance(selectedHeader, list)) else data).copy()
    dataTemp = cleanNaDataByKeyHeader(dataTemp, selectedHeader, **kwags)
    if(not drawHistogram(dataTemp, stamps=stamps, handler=handler, file=file, mask=mask, **kwags)):
        return False
    return True

if(True): #mdc服務
    def checkHeaderCompatibility(mdc, xheader, yheader, ret=None, **kwags):
        """
        檢查header是否適合進行單維度分析
        
        Parameters
        ----------
        mdc : object
            模型核心物件
        xheader : str
            x軸header名稱
        yheader : str  
            y軸header名稱
        ret : dict, optional
            回傳結果字典
            
        Returns
        -------
        bool
            是否適合進行分析
        """
        ret = ret if isinstance(ret, dict) else {}
        
        try:
            # 檢查xheader是否在有cell_size的header_zone中
            for zone_name, zone in mdc.xheader_zones.items():
                if xheader in getattr(zone, 'core', []):
                    cell_size = getattr(zone, 'cell_size', None)
                    if DFP.isiterable(cell_size):
                        ret['msg'] = f'xheader "{xheader}" 位於有cell_size的header_zone中，無法進行單維度分析'
                        ret['compatible'] = False
                        return False
            
            # 檢查yheader是否在有cell_size的header_zone中  
            for zone_name, zone in mdc.yheader_zones.items():
                if yheader in getattr(zone, 'core', []):
                    cell_size = getattr(zone, 'cell_size', None)
                    if DFP.isiterable(cell_size):
                        ret['msg'] = f'yheader "{yheader}" 位於有cell_size的header_zone中，無法進行單維度分析'
                        ret['compatible'] = False
                        return False
            
            ret['compatible'] = True
            return True
            
        except Exception as e:
            LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
            ret['msg'] = f'checkHeaderCompatibility error: {str(e)}'
            ret['compatible'] = False
            return False

    def determineDataType(data_series, **kwags):
        """
        判斷資料是否為連續型
        
        Parameters
        ----------
        data_series : pd.Series
            資料序列
            
        Returns
        -------
        str
            'continuous' 或 'categorical'
        """
        try:
            # 移除NaN值
            clean_data = data_series.dropna()
            
            if len(clean_data) == 0:
                return 'categorical'
            
            # 檢查是否全部都是數值型
            numeric_data = pd.to_numeric(clean_data, errors='coerce')
            numeric_ratio = numeric_data.notna().sum() / len(clean_data)
            
            # 如果超過80%是數值且唯一值數量大於10，視為連續型
            unique_count = len(clean_data.unique())
            
            if numeric_ratio > 0.8 and unique_count > 10:
                return 'continuous'
            else:
                return 'categorical'
                
        except Exception as e:
            LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
            return 'categorical'

    def addStatisticalInfoToPlot(ax, statistical_tests, xheader, yheader, **kwags):
        """
        在圖表上添加統計檢定信息
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            圖表軸對象
        statistical_tests : dict
            統計檢定結果
        xheader : str
            x軸變數名稱
        yheader : str
            y軸變數名稱
        """
        try:
            # 創建統計分析管理器
            stat_manager = StatisticalAnalysisManager()
            
            # 使用管理器添加統計信息到圖表
            return stat_manager.add_statistical_info_to_plot(ax, statistical_tests, xheader, yheader)
            
        except Exception as e:
            LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
            return False



    def analyzeContinuousToContinuous(source_data, xheader, yheader, fixed_values=None, include_model_prediction=True, ret=None, mdc=None, output_dir=None, **kwags):
        """
            連續型x對連續型y的分析
            
            Parameters
            ----------
            mdc : object
                模型核心物件
            source_data : pd.DataFrame
                原始資料
            xheader : str
                x軸header名稱
            yheader : str
                y軸header名稱
            fixed_values : dict, optional
                指定其他X因子的固定值，格式為 {'變數名': 固定值}
            ret : dict, optional
                回傳結果字典
                
            Returns
            -------
            bool
                是否成功分析
        """
        ret = ret if isinstance(ret, dict) else {}
        
        try:
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
            
            # 條件性執行模型預測
            if include_model_prediction and mdc is not None:
                # 準備預測用的資料
                # 其他維度使用固定值、平均數或眾數填充
                fixed_values = fixed_values if isinstance(fixed_values, dict) else {}
                predict_data = pd.DataFrame()
                
                for col in mdc.xheader:
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
                p_npData = mdc.predict(predict_data[mdc.xheader])
                y_pred_index = mdc.yheader.index(yheader) if yheader in mdc.yheader else 0
                y_pred = p_npData[:, y_pred_index] if len(p_npData.shape) > 1 else p_npData
                
                # 繪製預測曲線
                ax.plot(x_range, y_pred, 'b-', linewidth=2, label=f'{yheader} 預測曲線')
            
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
            output_dir = output_dir if LOGger.isinstance_not_empty(output_dir, str) else getattr(mdc, 'report_fd', 'analysis_output')
            if LOGger.isinstance_not_empty(output_dir, str):
                os.makedirs(output_dir, exist_ok=True)
                output_path = getProjectOutputPath(output_dir, output_filename)
                fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # 準備回傳資料
            if include_model_prediction and mdc is not None:
                ret['model_output'] = {
                    'xdata': x_range.tolist(),
                    'pdata': y_pred.tolist()
                }
            else:
                ret['model_output'] = None
            ret['record_data'] = {
                'x_data': real_x[common_index].tolist() if len(common_index) > 0 else [],
                'y_data': real_y[common_index].tolist() if len(common_index) > 0 else []
            }
            ret['image_path'] = output_path
            ret['analysis_type'] = 'continuous_to_continuous'
            
            return True
            
        except Exception as e:
            LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
            ret['msg'] = f'analyzeContinuousToContinuous error: {str(e)}'
            return False

    def analyzeCategoricalToContinuous(source_data, xheader, yheader, fixed_values=None, include_model_prediction=True, ret=None, mdc=None, output_dir=None, **kwags):
        """
        非連續型x對連續型y的分析
        
        Parameters
        ----------
        mdc : object
            模型核心物件
        source_data : pd.DataFrame
            原始資料
        xheader : str
            x軸header名稱
        yheader : str
            y軸header名稱
        ret : dict, optional
            回傳結果字典
            
        Returns
        -------
        bool
            是否成功分析
        """
        ret = ret if isinstance(ret, dict) else {}
        
        try:
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
            if include_model_prediction and mdc is not None:
                ret['model_output'] = {'xdata': [], 'pdata': []}
            else:
                ret['model_output'] = None
            ret['record_data'] = {}
            
            for i, category in enumerate(x_categories):
                # 取得該類別的y值分布
                category_y = source_data[source_data[xheader] == category][yheader].dropna()
                
                if len(category_y) > 0:
                    # 繪製分布圖（直方圖）
                    color = colors[i] if i < len(colors) else colors[0]
                    ax.hist(category_y, bins=20, alpha=0.3, color=color[:3], 
                        label=f'{category} 分布', orientation='horizontal')
                    
                    # 條件性執行模型預測
                    if include_model_prediction and mdc is not None:
                        # 準備預測用的資料
                        fixed_values = fixed_values if isinstance(fixed_values, dict) else {}
                        predict_data = pd.DataFrame()
                        for col in mdc.xheader:
                            if col == xheader:
                                predict_data[col] = [category]
                            elif col in fixed_values:
                                # 使用指定的固定值
                                predict_data[col] = [fixed_values[col]]
                            else:
                                if col in source_data.columns:
                                    col_data = source_data[col].dropna()
                                    if determineDataType(col_data) == 'continuous':
                                        predict_data[col] = col_data.mean()
                                    else:
                                        # print(str(col_data))
                                        mode_value = col_data.mode().iloc[0] if len(col_data.mode()) > 0 else col_data.iloc[0]
                                        predict_data[col] = [mode_value]
                                        # print(str(predict_data[col]))
                                else:
                                    predict_data[col] = 0
                        
                        # 執行預測
                        p_npData = mdc.predict(predict_data[mdc.xheader])
                        y_pred_index = mdc.yheader.index(yheader) if yheader in mdc.yheader else 0
                        y_pred = p_npData[0, y_pred_index] if len(p_npData.shape) > 1 else p_npData[0]
                        
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
                    if var_name != xheader:  # 排除分析變數本身
                        legend_text += f"{var_name}: {var_value}\n"
                
                # 在左上角添加文字框
                props = dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8)
                ax.text(0.02, 0.98, legend_text.strip(), transform=ax.transAxes, 
                        fontsize=9, verticalalignment='top', horizontalalignment='left',
                        bbox=props, fontproperties=vs3.vs.MJHfontprop())
            
            # 儲存圖片到專案資料夾
            output_filename = f'single_dimension_categorical_to_continuous_{xheader}_{yheader}.png'
            output_dir = output_dir if LOGger.isinstance_not_empty(output_dir, str) else getattr(mdc, 'report_fd', 'analysis_output')
            if LOGger.isinstance_not_empty(output_dir, str):
                os.makedirs(output_dir, exist_ok=True)
                output_path = getProjectOutputPath(output_dir, output_filename)
                fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            ret['image_path'] = output_path
            ret['analysis_type'] = 'categorical_to_continuous'
            
            return True
            
        except Exception as e:
            LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
            ret['msg'] = f'analyzeCategoricalToContinuous error: {str(e)}'
            return False

    def analyzeContinuousToCategorical(source_data, xheader, yheader, fixed_values=None, ret=None, mdc=None, output_dir=None, **kwags):
        """
        連續型x對非連續型y的分析（角色互換版本）
        
        Parameters
        ----------
        mdc : object
            模型核心物件
        source_data : pd.DataFrame
            原始資料
        xheader : str
            x軸header名稱
        yheader : str
            y軸header名稱
        ret : dict, optional
            回傳結果字典
            
        Returns
        -------
        bool
            是否成功分析
        """
        ret = ret if isinstance(ret, dict) else {}
        
        try:
            # 設定中文字型
            vs3.plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
            
            # 取得y的唯一值
            y_categories = source_data[yheader].dropna().unique()
            n_categories = len(y_categories)
            
            # 使用vs3.cm_rainbar分配顏色
            colors = vs2.cm_rainbar(n_categories, c_alpha=0.3)
            
            # 取得x的範圍用於繪製分布圖
            x_data = source_data[xheader].dropna()
            x_min, x_max = x_data.min(), x_data.max()
            
            # 繪製圖表
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 準備回傳資料
            ret['model_output'] = {'xdata': [], 'pdata': []}
            ret['record_data'] = {}
            
            for i, category in enumerate(y_categories):
                # 取得該類別的x值分布
                category_x = source_data[source_data[yheader] == category][xheader].dropna()
                
                if len(category_x) > 0:
                    # 繪製分布圖（直方圖）
                    color = colors[i] if i < len(colors) else colors[0]
                    ax.hist(category_x, bins=20, alpha=0.3, color=color[:3], 
                        label=f'{category} 分布')
                    
                    # 準備預測用的資料（使用該類別的x值平均）
                    x_mean = category_x.mean()
                    if mdc is not None:
                        fixed_values = fixed_values if isinstance(fixed_values, dict) else {}
                        predict_data = pd.DataFrame()
                        for col in mdc.xheader:
                            if col == xheader:
                                predict_data[col] = [x_mean]
                            elif col in fixed_values:
                                # 使用指定的固定值
                                predict_data[col] = [fixed_values[col]]
                            else:
                                if col in source_data.columns:
                                    col_data = source_data[col].dropna()
                                    if determineDataType(col_data) == 'continuous':
                                        predict_data[col] = col_data.mean()
                                    else:
                                        mode_value = col_data.mode().iloc[0] if len(col_data.mode()) > 0 else col_data.iloc[0]
                                        predict_data[col] = [mode_value]
                                else:
                                    predict_data[col] = 0
                        
                        # 執行預測
                        p_npData = mdc.predict(predict_data[mdc.xheader])
                        y_pred_index = mdc.yheader.index(yheader) if yheader in mdc.yheader else 0
                        y_pred = p_npData[0, y_pred_index] if len(p_npData.shape) > 1 else p_npData[0]
                        
                        # 繪製預測線
                        ax.axvline(x=x_mean, color=color[:3], linestyle='--', linewidth=2, 
                                label=f'{category} 平均值')
                        
                        # 儲存資料
                        ret['model_output']['xdata'].append(float(x_mean))
                        ret['model_output']['pdata'].append(str(category))
                    ret['record_data'][f'category_{i+1}'] = {
                        'value': str(category),
                        'x_data': category_x.tolist()
                    }
            
            ax.set_xlabel(xheader, fontproperties=vs.MJHfontprop())
            ax.set_ylabel('頻率', fontproperties=vs.MJHfontprop())
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
                    if var_name != xheader:  # 排除分析變數本身
                        legend_text += f"{var_name}: {var_value}\n"
                
                # 在左上角添加文字框
                props = dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8)
                ax.text(0.02, 0.98, legend_text.strip(), transform=ax.transAxes, 
                        fontsize=9, verticalalignment='top', horizontalalignment='left',
                        bbox=props, fontproperties=vs3.vs.MJHfontprop())
            
            # 儲存圖片到專案資料夾
            output_filename = f'single_dimension_continuous_to_categorical_{xheader}_{yheader}.png'
            output_dir = output_dir if LOGger.isinstance_not_empty(output_dir, str) else getattr(mdc, 'report_fd', 'analysis_output')
            if LOGger.isinstance_not_empty(output_dir, str):
                os.makedirs(output_dir, exist_ok=True)
                output_path = getProjectOutputPath(output_dir, output_filename)
                fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            ret['image_path'] = output_path
            ret['analysis_type'] = 'continuous_to_categorical'
            
            return True
            
        except Exception as e:
            LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
            ret['msg'] = f'analyzeContinuousToCategorical error: {str(e)}'
            return False

    def analyzeCategoricalToCategorical(source_data, xheader, yheader, fixed_values=None, ret=None, mdc=None, **kwags):
        """
        非連續型x對非連續型y的分析
        
        Parameters
        ----------
        mdc : object
            模型核心物件
        source_data : pd.DataFrame
            原始資料
        xheader : str
            x軸header名稱
        yheader : str
            y軸header名稱
        ret : dict, optional
            回傳結果字典
            
        Returns
        -------
        bool
            是否成功分析
        """
        ret = ret if isinstance(ret, dict) else {}
        
        try:
            # 設定中文字型
            vs3.plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
            
            # 建立交叉表分析
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
            
            for i, x_category in enumerate(x_categories):
                # 準備預測用的資料
                if mdc is not None:
                    fixed_values = fixed_values if isinstance(fixed_values, dict) else {}
                    predict_data = pd.DataFrame()
                    for col in mdc.xheader:
                        if col == xheader:
                            predict_data[col] = [x_category]
                        elif col in fixed_values:
                            # 使用指定的固定值
                            predict_data[col] = [fixed_values[col]]
                        else:
                            if col in source_data.columns:
                                col_data = source_data[col].dropna()
                                if determineDataType(col_data) == 'continuous':
                                    predict_data[col] = col_data.mean()
                                else:
                                    mode_value = col_data.mode().iloc[0] if len(col_data.mode()) > 0 else col_data.iloc[0]
                                    predict_data[col] = [mode_value]
                            else:
                                predict_data[col] = 0
                    
                    # 執行預測
                    p_npData = mdc.predict(predict_data[mdc.xheader])
                    y_pred_index = mdc.yheader.index(yheader) if yheader in mdc.yheader else 0
                    y_pred = p_npData[0, y_pred_index] if len(p_npData.shape) > 1 else p_npData[0]
                    
                    # 儲存資料
                    ret['model_output']['xdata'].append(str(x_category))
                    ret['model_output']['pdata'].append(str(y_pred))
                
                # 取得實際的y值分布
                actual_y = source_data[source_data[xheader] == x_category][yheader].dropna()
                ret['record_data'][f'category_{i+1}'] = {
                    'x_value': str(x_category),
                    'y_predicted': str(y_pred),
                    'y_actual_distribution': actual_y.value_counts().to_dict()
                }
            
            fig.tight_layout()
            
            # 執行統計檢定並添加到圖表 (添加到第一個子圖)
            if 'statistical_tests' in kwags:
                addStatisticalInfoToPlot(ax1, kwags['statistical_tests'], xheader, yheader)
            else:
                x_type = determineDataType(source_data[xheader])
                y_type = determineDataType(source_data[yheader])
                stats_ret = {}
                if performStatisticalTests(source_data, xheader, yheader, x_type, y_type, ret=stats_ret):
                    addStatisticalInfoToPlot(ax1, stats_ret['statistical_tests'], xheader, yheader)
            
            # 添加固定值 legend（左上角，添加到第一個子圖）
            if 'fixed_values_used' in kwags and kwags['fixed_values_used']:
                legend_text = "其他變數固定值:\n"
                for var_name, var_value in kwags['fixed_values_used'].items():
                    if var_name != xheader:  # 排除分析變數本身
                        legend_text += f"{var_name}: {var_value}\n"
                
                # 在左上角添加文字框
                props = dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8)
                ax1.text(0.02, 0.98, legend_text.strip(), transform=ax1.transAxes, 
                        fontsize=9, verticalalignment='top', horizontalalignment='left',
                        bbox=props, fontproperties=vs3.vs.MJHfontprop())
            
            # 儲存圖片到專案資料夾
            output_filename = f'single_dimension_categorical_to_categorical_{xheader}_{yheader}.png'
            output_dir = output_dir if LOGger.isinstance_not_empty(output_dir, str) else getattr(mdc, 'report_fd', 'analysis_output')
            if LOGger.isinstance_not_empty(output_dir, str):
                os.makedirs(output_dir, exist_ok=True)
                output_path = getProjectOutputPath(output_dir, output_filename)
                fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            ret['image_path'] = output_path
            ret['analysis_type'] = 'categorical_to_categorical'
            ret['crosstab'] = crosstab.to_dict()
            ret['crosstab_percentage'] = crosstab_pct.to_dict()
            
            return True
            
        except Exception as e:
            LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
            ret['msg'] = f'analyzeCategoricalToCategorical error: {str(e)}'
            return False

    def performStatisticalTests(source_data, xheader, yheader, x_type, y_type, ret=None, **kwags):
        """
        對真實資料進行統計檢定分析
        使用 StatisticalAnalysisManager 來管理統計檢定
        
        Parameters
        ----------
        source_data : pd.DataFrame
            原始資料
        xheader : str
            x軸變數名稱
        yheader : str
            y軸變數名稱
        x_type : str
            x軸資料類型 ('continuous' 或 'categorical')
        y_type : str
            y軸資料類型 ('continuous' 或 'categorical')
        ret : dict, optional
            回傳結果字典
            
        Returns
        -------
        bool
            是否成功進行檢定
        """
        ret = ret if isinstance(ret, dict) else {}
        
        try:
            # 創建統計分析管理器
            stat_manager = StatisticalAnalysisManager(alpha=0.05)
            
            # 執行統計分析
            results = stat_manager.perform_analysis(source_data, xheader, yheader, x_type, y_type)
            
            if 'error' in results:
                ret['msg'] = results['error']
                return False
            
            # 將結果存入 ret
            ret['statistical_tests'] = results
            ret['sample_size'] = results.get('sample_size', 0)  # 保持向後相容性
            
            # 統計檢定已由 StatisticalAnalysisManager 處理
            
            return True
            
        except Exception as e:
            LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
            ret['msg'] = f'performStatisticalTests error: {str(e)}'
            return False

    def generateStatisticalReport(statistical_tests, xheader, yheader, PlanID=None, SeqNo=None, fixed_values_used=None, ret=None, mdc=None, **kwags):
        """
        生成統計檢定報告
        
        Parameters
        ----------
        statistical_tests : dict
            統計檢定結果
        xheader : str
            x軸變數名稱
        yheader : str
            y軸變數名稱
        PlanID : int, optional
            計畫ID
        SeqNo : int, optional
            序列號
        fixed_values_used : dict, optional
            實際使用的固定值，格式為 {'變數名': 固定值}
        ret : dict, optional
            回傳結果字典
            
        Returns
        -------
        bool
            是否成功生成報告
        """
        ret = ret if isinstance(ret, dict) else {}
        
        try:
            report_lines = []
            report_lines.append(f"# {xheader} 對 {yheader} 的統計分析報告")
            report_lines.append("=" * 50)
            report_lines.append("")
            
            # 基本資訊
            report_lines.append(f"**分析類型**: {statistical_tests.get('analysis_type', 'Unknown')}")
            report_lines.append(f"**樣本數**: {statistical_tests.get('sample_size', 'Unknown')}")
            
            if PlanID is not None and SeqNo is not None:
                report_lines.append(f"**計畫ID**: {PlanID}")
                report_lines.append(f"**序列號**: {SeqNo}")
            
            # 添加固定值資訊
            if fixed_values_used:
                report_lines.append("")
                report_lines.append("## 其他變數固定值")
                report_lines.append("")
                report_lines.append("在進行單維度分析時，其他非變動分析欄位使用以下固定值：")
                report_lines.append("")
                for var_name, var_value in fixed_values_used.items():
                    if var_name != xheader:  # 排除分析變數本身
                        report_lines.append(f"- **{var_name}**: {var_value}")
            
            report_lines.append("")
            
            # 各項檢定結果
            report_lines.append("## 統計檢定結果")
            report_lines.append("")
            
            for test_name, test_result in statistical_tests.items():
                if test_name in ['analysis_type', 'sample_size', 'summary', 'group_info', 'descriptive_stats']:
                    continue
                    
                if isinstance(test_result, dict) and 'error' not in test_result:
                    report_lines.append(f"### {test_name.replace('_', ' ').title()}")
                    
                    if 'statistic' in test_result:
                        report_lines.append(f"- 統計量: {test_result['statistic']:.4f}")
                    if 'correlation' in test_result:
                        report_lines.append(f"- 相關係數: {test_result['correlation']:.4f}")
                    if 'p_value' in test_result:
                        report_lines.append(f"- p值: {test_result['p_value']:.4f}")
                    if 'significant' in test_result:
                        significance = "顯著" if test_result['significant'] else "不顯著"
                        report_lines.append(f"- 顯著性: {significance} (α = 0.05)")
                    if 'interpretation' in test_result:
                        report_lines.append(f"- 解釋: {test_result['interpretation']}")
                    if 'explanation' in test_result:
                        report_lines.append(f"- 說明: {test_result['explanation']}")
                    
                    report_lines.append("")
            
            # 描述性統計
            if 'descriptive_stats' in statistical_tests:
                report_lines.append("## 描述性統計")
                report_lines.append("")
                for group_name, stats in statistical_tests['descriptive_stats'].items():
                    report_lines.append(f"### {group_name}")
                    report_lines.append(f"- 平均數: {stats['mean']:.4f}")
                    report_lines.append(f"- 中位數: {stats['median']:.4f}")
                    report_lines.append(f"- 標準差: {stats['std']:.4f}")
                    report_lines.append(f"- 最小值: {stats['min']:.4f}")
                    report_lines.append(f"- 最大值: {stats['max']:.4f}")
                    report_lines.append(f"- 樣本數: {stats['count']}")
                    report_lines.append("")
            
            # 總結
            if 'summary' in statistical_tests:
                summary = statistical_tests['summary']
                report_lines.append("## 總結")
                report_lines.append("")
                report_lines.append(f"**結論**: {summary['conclusion']}")
                report_lines.append(f"**顯著檢定數量**: {summary['n_significant']}/{len([k for k in statistical_tests.keys() if k not in ['analysis_type', 'sample_size', 'summary', 'group_info', 'descriptive_stats']])}")
                
                if summary['significant_tests']:
                    report_lines.append(f"**顯著檢定項目**: {', '.join(summary['significant_tests'])}")
                
                report_lines.append("")
            
            # 儲存報告到專案資料夾
            report_content = "\n".join(report_lines)
            
            if PlanID is not None and SeqNo is not None:
                report_filename = f'statistical_report_{xheader}_{yheader}.md'
                report_path = getProjectOutputPath(mdc.report_fd, report_filename)
            else:
                # 向後相容，如果沒有提供 PlanID 和 SeqNo
                report_path = os.path.join('tmp', 'statistical_report.md')
                LOGger.CreateContainer('tmp')
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            ret['report_content'] = report_content
            ret['report_path'] = report_path
            
            return True
            
        except Exception as e:
            LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
            ret['msg'] = f'generateStatisticalReport error: {str(e)}'
            return False

    def generateTwoDimensionStatisticalReport(statistical_tests, interaction_analysis, surface_analysis, xheader1, xheader2, yheader, PlanID=None, SeqNo=None, fixed_values_used=None, ret=None, mdc=None, **kwags):
        """
        生成二維統計檢定報告
        
        Parameters
        ----------
        statistical_tests : dict
            統計檢定結果
        interaction_analysis : dict
            交互作用分析結果
        surface_analysis : dict
            響應面分析結果
        xheader1 : str
            第一個x軸變數名稱
        xheader2 : str
            第二個x軸變數名稱
        yheader : str
            y軸變數名稱
        PlanID : int, optional
            計畫ID
        SeqNo : int, optional
            序列號
        fixed_values_used : dict, optional
            實際使用的固定值，格式為 {'變數名': 固定值}
        ret : dict, optional
            回傳結果字典
            
        Returns
        -------
        bool
            是否成功生成報告
        """
        ret = ret if isinstance(ret, dict) else {}
        
        try:
            report_lines = []
            report_lines.append(f"# {xheader1} 和 {xheader2} 對 {yheader} 的二維統計分析報告")
            report_lines.append("=" * 60)
            report_lines.append("")
            
            # 基本資訊
            report_lines.append(f"**分析類型**: {statistical_tests.get('analysis_type', 'Unknown')}")
            report_lines.append(f"**樣本數**: {statistical_tests.get('sample_size', 'Unknown')}")
            
            if PlanID is not None and SeqNo is not None:
                report_lines.append(f"**計畫ID**: {PlanID}")
                report_lines.append(f"**序列號**: {SeqNo}")
            
            # 添加固定值資訊 🆕
            if fixed_values_used:
                report_lines.append("")
                report_lines.append("## 其他變數固定值")
                report_lines.append("")
                report_lines.append("在進行二維分析時，其他非變動分析欄位使用以下固定值：")
                report_lines.append("")
                for var_name, var_value in fixed_values_used.items():
                    if var_name not in [xheader1, xheader2]:  # 排除分析變數本身
                        report_lines.append(f"- **{var_name}**: {var_value}")
            
            report_lines.append("")
            
            # 統計檢定結果
            report_lines.append("## 統計檢定結果")
            report_lines.append("")
            
            for test_name, test_result in statistical_tests.items():
                if test_name in ['analysis_type', 'sample_size', 'summary']:
                    continue
                    
                if isinstance(test_result, dict) and 'error' not in test_result:
                    report_lines.append(f"### {test_name.replace('_', ' ').title()}")
                    
                    if 'statistic' in test_result:
                        report_lines.append(f"- 統計量: {test_result['statistic']:.4f}")
                    if 'correlation' in test_result:
                        report_lines.append(f"- 相關係數: {test_result['correlation']:.4f}")
                    if 'f_statistic' in test_result:
                        report_lines.append(f"- F統計量: {test_result['f_statistic']:.4f}")
                    if 'r_squared' in test_result:
                        report_lines.append(f"- R²: {test_result['r_squared']:.4f}")
                    if 'p_value' in test_result:
                        report_lines.append(f"- p值: {test_result['p_value']:.4f}")
                    if 'significant' in test_result:
                        significance = "顯著" if test_result['significant'] else "不顯著"
                        report_lines.append(f"- 顯著性: {significance} (α = 0.05)")
                    if 'interpretation' in test_result:
                        report_lines.append(f"- 解釋: {test_result['interpretation']}")
                    if 'explanation' in test_result:
                        report_lines.append(f"- 說明: {test_result['explanation']}")
                    if 'multicollinearity_warning' in test_result and test_result['multicollinearity_warning']:
                        report_lines.append(f"- ⚠️ 警告: 存在多重共線性問題")
                    
                    report_lines.append("")
            
            # 交互作用分析
            if interaction_analysis and 'error' not in interaction_analysis:
                report_lines.append("## 交互作用分析")
                report_lines.append("")
                
                if 'r_squared_with_interaction' in interaction_analysis:
                    report_lines.append(f"- 包含交互作用的R²: {interaction_analysis['r_squared_with_interaction']:.4f}")
                if 'r_squared_without_interaction' in interaction_analysis:
                    report_lines.append(f"- 不含交互作用的R²: {interaction_analysis['r_squared_without_interaction']:.4f}")
                if 'interaction_improvement' in interaction_analysis:
                    report_lines.append(f"- 交互作用改善: {interaction_analysis['interaction_improvement']:.4f}")
                if 'interaction_coefficient' in interaction_analysis:
                    report_lines.append(f"- 交互作用係數: {interaction_analysis['interaction_coefficient']:.4f}")
                if 'interpretation' in interaction_analysis:
                    report_lines.append(f"- 結論: {interaction_analysis['interpretation']}")
                
                report_lines.append("")
            
            # 響應面分析
            if surface_analysis and 'error' not in surface_analysis:
                report_lines.append("## 響應面分析")
                report_lines.append("")
                
                if 'max_point' in surface_analysis:
                    max_pt = surface_analysis['max_point']
                    report_lines.append(f"- 最大值點: ({max_pt['x1']:.4f}, {max_pt['x2']:.4f}) → {max_pt['y']:.4f}")
                if 'min_point' in surface_analysis:
                    min_pt = surface_analysis['min_point']
                    report_lines.append(f"- 最小值點: ({min_pt['x1']:.4f}, {min_pt['x2']:.4f}) → {min_pt['y']:.4f}")
                if 'y_range' in surface_analysis:
                    report_lines.append(f"- Y值範圍: {surface_analysis['y_range']:.4f}")
                if 'surface_roughness' in surface_analysis:
                    report_lines.append(f"- 響應面粗糙度: {surface_analysis['surface_roughness']:.4f}")
                
                    report_lines.append("")
            
            # 總結
            if 'summary' in statistical_tests:
                summary = statistical_tests['summary']
                report_lines.append("## 總結")
                report_lines.append("")
                report_lines.append(f"**結論**: {summary['conclusion']}")
                report_lines.append(f"**顯著檢定數量**: {summary['n_significant']}")
                
                if summary['significant_tests']:
                    report_lines.append(f"**顯著檢定項目**: {', '.join(summary['significant_tests'])}")
                
                report_lines.append("")
            
            # 儲存報告到專案資料夾
            report_content = "\n".join(report_lines)
            
            if PlanID is not None and SeqNo is not None:
                report_filename = f'two_dimension_statistical_report_{xheader1}_{xheader2}_{yheader}.md'
                report_path = getProjectOutputPath(mdc.report_fd, report_filename)
            else:
                # 向後相容，如果沒有提供 PlanID 和 SeqNo
                report_path = os.path.join('tmp', 'two_dimension_statistical_report.md')
                LOGger.CreateContainer('tmp')
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            ret['report_content'] = report_content
            ret['report_path'] = report_path
            
            m_addlog(f'Two dimension statistical report generated: {report_path}', 
                stamps=['PlanID::', PlanID, 'SeqNo::', SeqNo], colora=LOGger.OKGREEN)
            
            return True
            
        except Exception as e:
            LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
            ret['msg'] = f'generateTwoDimensionStatisticalReport error: {str(e)}'
            return False

    def generateRecommendations(statistical_tests, xheader, yheader, analysis_type, **kwags):
        """
        基於統計檢定結果生成建議資訊
        
        Parameters
        ----------
        statistical_tests : dict
            統計檢定結果
        xheader : str
            x軸變數名稱
        yheader : str
            y軸變數名稱
        analysis_type : str
            分析類型
            
        Returns
        -------
        dict
            建議資訊字典
        """
        recommendations = {
            'overall_recommendation': '',
            'statistical_strength': '',
            'effect_size_interpretation': '',
            'practical_suggestions': [],
            'data_quality_notes': [],
            'next_steps': []
        }
        
        try:
            sample_size = statistical_tests.get('sample_size', 0)
            summary = statistical_tests.get('summary', {})
            significant_count = summary.get('n_significant', 0)
            
            # 計算效應量
            max_effect_size = 0.0
            if analysis_type == 'continuous_vs_continuous':
                if 'pearson_correlation' in statistical_tests:
                    max_effect_size = abs(statistical_tests['pearson_correlation'].get('correlation', 0))
                if 'linear_regression' in statistical_tests:
                    r2 = statistical_tests['linear_regression'].get('r_squared', 0)
                    max_effect_size = max(max_effect_size, r2)
            elif analysis_type == 'categorical_vs_categorical':
                if 'cramers_v' in statistical_tests:
                    max_effect_size = statistical_tests['cramers_v'].get('value', 0)
            
            # 整體建議
            if significant_count > 0:
                recommendations['overall_recommendation'] = f'{xheader} 對 {yheader} 有統計上顯著的影響，建議進一步分析其實際應用價值。'
                recommendations['statistical_strength'] = '強' if significant_count >= 2 else '中等'
            else:
                recommendations['overall_recommendation'] = f'{xheader} 對 {yheader} 無統計上顯著影響，可能需要考慮其他因子或增加樣本數。'
                recommendations['statistical_strength'] = '弱'
            
            # 效應量解釋
            if max_effect_size > 0.5:
                recommendations['effect_size_interpretation'] = f'效應量大 ({max_effect_size:.3f})，{xheader} 對 {yheader} 有實質性影響。'
            elif max_effect_size > 0.3:
                recommendations['effect_size_interpretation'] = f'效應量中等 ({max_effect_size:.3f})，{xheader} 對 {yheader} 有中度影響。'
            elif max_effect_size > 0.1:
                recommendations['effect_size_interpretation'] = f'效應量小 ({max_effect_size:.3f})，{xheader} 對 {yheader} 影響較小。'
            else:
                recommendations['effect_size_interpretation'] = f'效應量極小 ({max_effect_size:.3f})，{xheader} 對 {yheader} 幾乎無影響。'
            
            # 實際建議
            if significant_count > 0:
                if analysis_type == 'continuous_vs_continuous':
                    recommendations['practical_suggestions'].append(f'可建立 {xheader} 與 {yheader} 的預測模型')
                    recommendations['practical_suggestions'].append('考慮將此關係應用於製程優化')
                elif analysis_type == 'categorical_vs_continuous':
                    recommendations['practical_suggestions'].append(f'不同 {xheader} 類別對 {yheader} 有顯著差異')
                    recommendations['practical_suggestions'].append('可針對不同類別制定差異化策略')
                elif analysis_type == 'continuous_vs_categorical':
                    recommendations['practical_suggestions'].append(f'{xheader} 可作為 {yheader} 分類的預測指標')
                elif analysis_type == 'categorical_vs_categorical':
                    recommendations['practical_suggestions'].append(f'{xheader} 與 {yheader} 存在關聯性')
                    recommendations['practical_suggestions'].append('可進行交叉分析以了解具體關聯模式')
            else:
                recommendations['practical_suggestions'].append('目前數據顯示無顯著關係，建議：')
                recommendations['practical_suggestions'].append('1. 增加樣本數量以提高檢定力')
                recommendations['practical_suggestions'].append('2. 檢查是否存在非線性關係')
                recommendations['practical_suggestions'].append('3. 考慮其他潛在影響因子')
            
            # 資料品質註記
            if sample_size < 30:
                recommendations['data_quality_notes'].append(f'樣本數較少 (n={sample_size})，結果可靠性有限')
            elif sample_size < 100:
                recommendations['data_quality_notes'].append(f'樣本數適中 (n={sample_size})，結果具參考價值')
            else:
                recommendations['data_quality_notes'].append(f'樣本數充足 (n={sample_size})，結果可靠性高')
            
            # 下一步建議
            if significant_count > 0:
                recommendations['next_steps'].append('進行更深入的因果關係分析')
                recommendations['next_steps'].append('考慮建立預測或分類模型')
                recommendations['next_steps'].append('驗證結果在不同條件下的穩定性')
            else:
                recommendations['next_steps'].append('收集更多資料以提高統計檢定力')
                recommendations['next_steps'].append('探索非線性關係或交互作用')
                recommendations['next_steps'].append('考慮其他可能的影響變數')
            
        except Exception as e:
            LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
            recommendations['overall_recommendation'] = '建議生成過程發生錯誤'
        
        return recommendations

    def singleDimensionAnalysis(PlanID, SeqNo, xheader=None, yheader=None, fixed_values=None, source_data=None, include_model_prediction=True, ret=None, **kwags):
        """
        進行模型上x對y的完整單維度變化評估（包含統計檢定與建議）
        
        Parameters
        ----------
        PlanID : int
            計畫ID
        SeqNo : int
            序列號
        xheader : str, optional
            要用來進行單維度分析的xheader，預設None時取mdc模組的第一個xheader
        yheader : str, optional
            要用來進行單維度分析的yheader，預設None時取mdc模組的第一個yheader
        fixed_values : dict, optional
            指定其他X因子的固定值，格式為 {'變數名': 固定值}
            例如: {'溫度': 25.0, '濕度': 60.0}
            未指定的變數將使用專案資料夾中的資料來填充（連續型用平均值，分類型用眾數）
        source_data : pd.DataFrame, optional
            外部提供的資料來源，如果提供則直接使用此資料進行分析
            如果未提供則從 PlanID/SeqNo 載入專案資料
        include_model_prediction : bool, optional
            是否包含模型預測曲線，預設為 True
            - True: 顯示模型預測曲線和相關資訊
            - False: 只顯示真實資料的統計分析，不包含模型預測
        ret : dict, optional
            回傳結果字典，包含以下額外欄位：
            - statistical_tests: 統計檢定數值結果
            - statistical_summary: 統計檢定摘要
            - recommendations: 建議資訊
            - statistical_report: 統計報告內容
            - statistical_report_path: 統計報告檔案路徑
            - fixed_values_used: 實際使用的固定值
            
        Returns
        -------
        bool
            是否成功進行分析
        """
        ret = ret if isinstance(ret, dict) else {}
        
        try:
            # 初始化模型和資料
            init_ret = {}
            if not initializeModelAndData(PlanID, SeqNo, ret=init_ret, **kwags):
                ret.update(init_ret)
                return False
            
            mdc = init_ret['mdc']
            
            # 使用外部提供的 source_data 或從專案載入的資料
            if source_data is not None:
                # 驗證外部提供的資料格式
                if not isinstance(source_data, pd.DataFrame):
                    ret['msg'] = 'source_data must be a pandas DataFrame'
                    return False
                
                # 記錄使用外部資料
                ret['data_source'] = 'external_provided'
                ret['external_data_shape'] = source_data.shape
            else:
                # 使用從專案載入的資料
                source_data = init_ret['source_data']
                ret['data_source'] = f'project_PlanID_{PlanID}_SeqNo_{SeqNo}'
            
            # 確定xheader和yheader
            if xheader is None:
                if len(mdc.xheader) > 0:
                    xheader = mdc.xheader[0]
                else:
                    ret['msg'] = 'No xheader available in model'
                    return False
            
            if yheader is None:
                if len(mdc.yheader) > 0:
                    yheader = mdc.yheader[0]
                else:
                    ret['msg'] = 'No yheader available in model'
                    return False
            
            # 檢查header是否存在於資料中
            if xheader not in source_data.columns:
                ret['msg'] = f'xheader "{xheader}" not found in source data'
                return False
            
            if yheader not in source_data.columns:
                ret['msg'] = f'yheader "{yheader}" not found in source data'
                return False
            
            # 如果使用外部資料，檢查是否包含模型所需的所有欄位（用於 fixed_values 填充）
            if ret.get('data_source') == 'external_provided':
                missing_headers = []
                for header in mdc.xheader:
                    if header not in source_data.columns:
                        missing_headers.append(header)
                
                if missing_headers:
                    ret['msg'] = f'External source_data missing required model headers: {missing_headers}'
                    ret['missing_headers'] = missing_headers
                    ret['available_headers'] = list(source_data.columns)
                    ret['required_headers'] = mdc.xheader
                    return False
            
            # 檢查header相容性
            compat_ret = {}
            if not checkHeaderCompatibility(mdc, xheader, yheader, ret=compat_ret, **kwags):
                ret.update(compat_ret)
                return False
            
            # 判斷資料類型
            x_type = determineDataType(source_data[xheader])
            y_type = determineDataType(source_data[yheader])
            
            m_addlog(f'Analysis types: {xheader}({x_type}) -> {yheader}({y_type})', 
                stamps=['PlanID::', PlanID, 'SeqNo::', SeqNo], colora=LOGger.OKCYAN)
            
            # 處理固定值參數
            fixed_values = fixed_values if isinstance(fixed_values, dict) else {}
            
            # 驗證固定值中的變數名是否存在於模型的xheader中
            invalid_vars = []
            for var_name in fixed_values.keys():
                if var_name not in mdc.xheader:
                    invalid_vars.append(var_name)
            
            if invalid_vars:
                ret['msg'] = f'固定值中包含無效的變數名: {invalid_vars}。有效的變數名: {mdc.xheader}'
                return False
            
            # 檢查固定值是否包含正在分析的xheader
            if xheader in fixed_values:
                ret['msg'] = f'不能為正在分析的變數 "{xheader}" 設定固定值'
                return False
            
            # 記錄使用的固定值
            ret['fixed_values_used'] = fixed_values.copy()
            
            m_addlog(f'Using fixed values: {fixed_values}', 
                stamps=['PlanID::', PlanID, 'SeqNo::', SeqNo], colora=LOGger.OKCYAN)
            
            # 根據資料類型選擇分析方法
            analysis_ret = {}
            
            if x_type == 'continuous' and y_type == 'continuous':
                # 連續型對連續型
                success = analyzeContinuousToContinuous(source_data, xheader, yheader, fixed_values=fixed_values, include_model_prediction=include_model_prediction, ret=analysis_ret, mdc=mdc, fixed_values_used=ret.get('fixed_values_used'), **kwags)
            elif x_type == 'categorical' and y_type == 'continuous':
                # 非連續型對連續型
                success = analyzeCategoricalToContinuous(source_data, xheader, yheader, fixed_values=fixed_values, include_model_prediction=include_model_prediction, ret=analysis_ret, mdc=mdc, fixed_values_used=ret.get('fixed_values_used'), **kwags)
            elif x_type == 'continuous' and y_type == 'categorical':
                # 連續型對非連續型
                success = analyzeContinuousToCategorical(source_data, xheader, yheader, fixed_values=fixed_values, include_model_prediction=include_model_prediction, ret=analysis_ret, mdc=mdc, fixed_values_used=ret.get('fixed_values_used'), **kwags)
            elif x_type == 'categorical' and y_type == 'categorical':
                # 非連續型對非連續型
                success = analyzeCategoricalToCategorical(source_data, xheader, yheader, fixed_values=fixed_values, include_model_prediction=include_model_prediction, ret=analysis_ret, mdc=mdc, fixed_values_used=ret.get('fixed_values_used'), **kwags)
            else:
                ret['msg'] = f'Unsupported data type combination: {x_type} -> {y_type}'
                return False
            
            if not success:
                ret.update(analysis_ret)
                return False
            
            # 整合結果
            ret.update(analysis_ret)
            ret['xheader'] = xheader
            ret['yheader'] = yheader
            ret['x_type'] = x_type
            ret['y_type'] = y_type
            ret['PlanID'] = PlanID
            ret['SeqNo'] = SeqNo
            ret['excel_files'] = init_ret.get('excel_files', [])
            
            if 'train_data' in init_ret:
                ret['train_data_info'] = {
                    'shape': init_ret['train_data'].shape,
                    'columns': list(init_ret['train_data'].columns)
                }
            
            # 執行統計檢定分析
            stats_ret = {}
            if performStatisticalTests(source_data, xheader, yheader, x_type, y_type, ret=stats_ret, **kwags):
                ret['statistical_tests'] = stats_ret['statistical_tests']
                
                # 生成統計檢定摘要
                summary = stats_ret['statistical_tests'].get('summary', {})
                ret['statistical_summary'] = {
                    'sample_size': stats_ret['statistical_tests'].get('sample_size', 0),
                    'significant_count': summary.get('n_significant', 0),
                    'overall_significant': summary.get('overall_significant', False),
                    'conclusion': summary.get('conclusion', '無法判斷'),
                    'significant_tests': summary.get('significant_tests', [])
                }
                
                # 生成建議資訊
                analysis_type = stats_ret['statistical_tests'].get('analysis_type', '')
                recommendations = generateRecommendations(
                    stats_ret['statistical_tests'], xheader, yheader, analysis_type, **kwags
                )
                ret['recommendations'] = recommendations
                
                # 生成統計報告
                report_ret = {}
                if generateStatisticalReport(stats_ret['statistical_tests'], xheader, yheader, PlanID=PlanID, SeqNo=SeqNo, fixed_values_used=ret.get('fixed_values_used'), ret=report_ret, mdc=mdc, **kwags):
                    ret['statistical_report'] = report_ret['report_content']
                    ret['statistical_report_path'] = report_ret['report_path']
            else:
                ret['statistical_analysis_error'] = stats_ret.get('msg', 'Unknown error')
                ret['statistical_summary'] = {
                    'sample_size': 0,
                    'significant_count': 0,
                    'overall_significant': False,
                    'conclusion': '統計檢定失敗',
                    'significant_tests': []
                }
                ret['recommendations'] = {
                    'overall_recommendation': '由於統計檢定失敗，無法提供可靠建議',
                    'statistical_strength': '無法評估',
                    'effect_size_interpretation': '無法計算',
                    'practical_suggestions': ['請檢查資料品質'],
                    'data_quality_notes': ['統計檢定過程發生錯誤'],
                    'next_steps': ['重新檢查資料格式和完整性']
                }
            
            m_addlog('Single dimension analysis with statistical tests and recommendations completed successfully', 
                stamps=['PlanID::', PlanID, 'SeqNo::', SeqNo], colora=LOGger.OKGREEN)
            
            return True
            
        except Exception as e:
            LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
            ret['msg'] = f'singleDimensionAnalysis error: {str(e)}'
            return False

    # def getProjectOutputPath(mdc.report_fd, filename, subdir='analysis_output'):
    def getProjectOutputPath(output_dir, filename):
        """
        取得專案輸出檔案路徑
        
        Parameters
        ----------
        PlanID : str or int
            計畫ID
        SeqNo : str or int
            序列號
        filename : str
            檔案名稱
        subdir : str, optional
            子目錄名稱，預設為 'analysis_output'
        
        Returns
        -------
        str
            完整的檔案路徑
        """
        # project_dir = os.path.join(m_source_dir, str(PlanID), str(SeqNo))
        # output_dir = os.path.join(project_dir, subdir)
        LOGger.CreateContainer(output_dir)
        return os.path.join(output_dir, filename)

    def twoDimensionAnalysis(PlanID, SeqNo, xheader1=None, xheader2=None, yheader=None, fixed_values=None, source_data=None, include_model_prediction=True, ret=None, **kwags):
        """
        進行模型上兩個x變數對y的二維變異分析
        
        Parameters
        ----------
        PlanID : int
            計畫ID
        SeqNo : int
            序列號
        xheader1 : str, optional
            第一個X軸變數名稱，預設None時使用第一個X變數
        xheader2 : str, optional
            第二個X軸變數名稱，預設None時使用第二個X變數
        yheader : str, optional
            Y軸變數名稱，預設None時使用第一個Y變數
        fixed_values : dict, optional
            指定其他X因子的固定值，格式為 {'變數名': 固定值}
            例如: {'溫度': 25.0, '濕度': 60.0}
            未指定的變數將使用專案資料夾中的資料來填充（連續型用平均值，分類型用眾數）
        source_data : pd.DataFrame, optional
            外部提供的資料來源，如果提供則使用此資料而非專案資料
        include_model_prediction : bool, optional
            是否包含模型預測曲線和響應面，預設為 True
            - True: 顯示模型預測響應面、切片和相關資訊
            - False: 只顯示真實資料的統計分析，不包含模型預測
        ret : dict, optional
            回傳結果字典，包含以下欄位：
            - model_output: 模型預測結果 (3D網格資料)
            - record_data: 真實資料點
            - statistical_tests: 統計檢定結果
            - interaction_analysis: 交互作用分析
            - surface_analysis: 響應面分析
            - fixed_values_used: 實際使用的固定值
            
        Returns
        -------
        bool
            是否成功進行分析
        """
        ret = ret if isinstance(ret, dict) else {}
        
        try:
            # 初始化模型和資料
            init_ret = {}
            if not initializeModelAndData(PlanID, SeqNo, ret=init_ret, **kwags):
                ret.update(init_ret)
                return False
            
            mdc = init_ret['mdc']
            
            # 使用外部資料或專案資料
            if source_data is not None:
                # 驗證外部資料格式
                if not isinstance(source_data, pd.DataFrame):
                    ret['msg'] = 'source_data must be a pandas DataFrame'
                    return False
                ret['data_source'] = 'external_data'
            else:
                source_data = init_ret['source_data']
                ret['data_source'] = f'project_PlanID_{PlanID}_SeqNo_{SeqNo}'
            
            # 確定xheader1, xheader2和yheader
            if xheader1 is None:
                if len(mdc.xheader) > 0:
                    xheader1 = mdc.xheader[0]
                else:
                    ret['msg'] = 'No xheader available in model'
                    return False
            
            if xheader2 is None:
                if len(mdc.xheader) > 1:
                    xheader2 = mdc.xheader[1]
                else:
                    ret['msg'] = 'Need at least 2 xheaders for two-dimension analysis'
                    return False
            
            if yheader is None:
                if len(mdc.yheader) > 0:
                    yheader = mdc.yheader[0]
                else:
                    ret['msg'] = 'No yheader available in model'
                    return False
            
            # 檢查xheader1和xheader2不能相同
            if xheader1 == xheader2:
                ret['msg'] = f'xheader1 and xheader2 cannot be the same: {xheader1}'
                return False
            
            # 檢查header是否存在於資料中
            for header in [xheader1, xheader2, yheader]:
                if header not in source_data.columns:
                    ret['msg'] = f'Header "{header}" not found in source data'
                    return False
            
            # 檢查header相容性
            compat_ret = {}
            for xheader in [xheader1, xheader2]:
                if not checkHeaderCompatibility(mdc, xheader, yheader, ret=compat_ret, **kwags):
                    ret.update(compat_ret)
                    return False
            
            # 處理固定值參數
            fixed_values = fixed_values if isinstance(fixed_values, dict) else {}
            
            # 驗證固定值中的變數名是否存在於模型的xheader中
            invalid_vars = []
            for var_name in fixed_values.keys():
                if var_name not in mdc.xheader:
                    invalid_vars.append(var_name)
            
            if invalid_vars:
                ret['msg'] = f'固定值中包含無效的變數名: {invalid_vars}。有效的變數名: {mdc.xheader}'
                return False
            
            # 檢查固定值是否包含正在分析的xheader
            if xheader1 in fixed_values or xheader2 in fixed_values:
                ret['msg'] = f'不能為正在分析的變數 "{xheader1}" 或 "{xheader2}" 設定固定值'
                return False
            
            # 使用核心邏輯執行分析
            stamps = ['PlanID::', PlanID, 'SeqNo::', SeqNo]
            if not _performTwoDimensionAnalysisCore(mdc, source_data, xheader1, xheader2, yheader, fixed_values, include_model_prediction, ret, stamps, **kwags):
                return False
            
            # 添加專案相關資訊
            ret['PlanID'] = PlanID
            ret['SeqNo'] = SeqNo
            ret['excel_files'] = init_ret.get('excel_files', [])
            
            if 'train_data' in init_ret:
                ret['train_data_info'] = {
                    'shape': init_ret['train_data'].shape,
                    'columns': list(init_ret['train_data'].columns)
                }
            
            # 生成二維統計報告（僅在統計檢定成功時）
            if 'statistical_tests' in ret:
                report_ret = {}
                if generateTwoDimensionStatisticalReport(
                    mdc,
                    ret['statistical_tests'], 
                    ret.get('interaction_analysis', {}),
                    ret.get('surface_analysis', {}),
                    xheader1, xheader2, yheader, 
                    PlanID=PlanID, SeqNo=SeqNo, 
                    fixed_values_used=ret.get('fixed_values_used'), 
                    ret=report_ret, **kwags
                ):
                    ret['statistical_report'] = report_ret['report_content']
                    ret['statistical_report_path'] = report_ret['report_path']
            
            m_addlog('Two dimension analysis completed successfully', 
                stamps=['PlanID::', PlanID, 'SeqNo::', SeqNo], colora=LOGger.OKGREEN)
            
            return True
            
        except Exception as e:
            LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
            ret['msg'] = f'twoDimensionAnalysis error: {str(e)}'
            return False

    def _performTwoDimensionAnalysisCore(mdc, source_data, xheader1, xheader2, yheader, fixed_values, include_model_prediction, ret, stamps, **kwags):
        """
        執行二維分析的核心邏輯（共用函數）
        
        Parameters
        ----------
        mdc : object
            模型核心物件
        source_data : pd.DataFrame
            資料來源
        xheader1 : str
            第一個X軸變數名稱
        xheader2 : str
            第二個X軸變數名稱
        yheader : str
            Y軸變數名稱
        fixed_values : dict
            固定值字典
        include_model_prediction : bool
            是否包含模型預測
        ret : dict
            回傳結果字典
        stamps : list
            日誌標記
        **kwags
            其他參數
            
        Returns
        -------
        bool
            是否成功進行分析
        """
        try:
            # 判斷資料類型
            x1_type = determineDataType(source_data[xheader1])
            x2_type = determineDataType(source_data[xheader2])
            y_type = determineDataType(source_data[yheader])
            
            m_addlog(f'Two-dimension analysis types: {xheader1}({x1_type}) + {xheader2}({x2_type}) -> {yheader}({y_type})', 
                stamps=stamps, colora=LOGger.OKCYAN)
            
            # 記錄使用的固定值
            ret['fixed_values_used'] = fixed_values.copy()
            
            m_addlog(f'Using fixed values: {fixed_values}', 
                stamps=stamps, colora=LOGger.OKCYAN)
            
            # 根據資料類型選擇分析方法
            analysis_ret = {}
            
            if x1_type == 'continuous' and x2_type == 'continuous':
                # 連續型 + 連續型 -> Y
                success = analyzeTwoContinuousToY(mdc, source_data, xheader1, xheader2, yheader, y_type, fixed_values=fixed_values, include_model_prediction=include_model_prediction, ret=analysis_ret, **kwags)
            elif (x1_type == 'categorical' and x2_type == 'continuous') or (x1_type == 'continuous' and x2_type == 'categorical'):
                # 混合型：一個連續型 + 一個類別型 -> Y
                success = analyzeMixedToY(mdc, source_data, xheader1, xheader2, yheader, x1_type, x2_type, y_type, fixed_values=fixed_values, include_model_prediction=include_model_prediction, ret=analysis_ret, **kwags)
            elif x1_type == 'categorical' and x2_type == 'categorical':
                # 類別型 + 類別型 -> Y
                success = analyzeTwoCategoricalToY(mdc, source_data, xheader1, xheader2, yheader, y_type, fixed_values=fixed_values, include_model_prediction=include_model_prediction, ret=analysis_ret, **kwags)
            else:
                ret['msg'] = f'Unsupported data type combination: {x1_type} + {x2_type} -> {y_type}'
                return False
            
            if not success:
                ret.update(analysis_ret)
                return False
            
            # 整合結果
            ret.update(analysis_ret)
            ret['xheader1'] = xheader1
            ret['xheader2'] = xheader2
            ret['yheader'] = yheader
            ret['x1_type'] = x1_type
            ret['x2_type'] = x2_type
            ret['y_type'] = y_type
            ret['analysis_type'] = f'{x1_type}_{x2_type}_to_{y_type}'
            
            # 執行二維統計檢定分析
            stats_ret = {}
            if performTwoDimensionStatisticalTests(source_data, xheader1, xheader2, yheader, x1_type, x2_type, y_type, ret=stats_ret, **kwags):
                ret['statistical_tests'] = stats_ret['statistical_tests']
                ret['interaction_analysis'] = stats_ret.get('interaction_analysis', {})
                ret['surface_analysis'] = stats_ret.get('surface_analysis', {})
            else:
                ret['statistical_analysis_error'] = stats_ret.get('msg', 'Unknown error')
            
            return True
            
        except Exception as e:
            LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
            ret['msg'] = f'_performTwoDimensionAnalysisCore error: {str(e)}'
            return False

    def twoDimensionAnalysisDirectly(data, xheader1, xheader2, yheader, fixed_values=None, include_model_prediction=True, ret=None, **kwags):
        """
            進行兩個x變數對y的二維變異分析（直接使用資料，不需要模型）
            
            Parameters
            ----------
            data : pd.DataFrame
                資料來源
            xheader1 : str
                第一個X軸變數名稱
            xheader2 : str
                第二個X軸變數名稱
            yheader : str
                Y軸變數名稱
            fixed_values : dict, optional
                指定其他X因子的固定值
            include_model_prediction : bool, optional
                是否包含模型預測（此版本不支援模型預測，會自動設為 False）
            ret : dict, optional
                回傳結果字典
                
            Returns
            -------
            bool
                是否成功進行分析
        """
        ret = ret if isinstance(ret, dict) else {}
        
        try:
            # 驗證資料格式
            if not isinstance(data, pd.DataFrame):
                ret['msg'] = 'data must be a pandas DataFrame'
                return False
            
            # 檢查必要的欄位
            if xheader1 is None or xheader2 is None or yheader is None:
                ret['msg'] = 'xheader1, xheader2, and yheader are required'
                return False
            
            # 檢查xheader1和xheader2不能相同
            if xheader1 == xheader2:
                ret['msg'] = f'xheader1 and xheader2 cannot be the same: {xheader1}'
                return False
            
            # 檢查header是否存在於資料中
            for header in [xheader1, xheader2, yheader]:
                if header not in data.columns:
                    ret['msg'] = f'Header "{header}" not found in data'
                    return False
            
            # 此版本不支援模型預測，強制設為 False
            if include_model_prediction and mdc is not None:
                m_addlog('Model prediction is not supported in data-only mode, setting include_model_prediction=False', 
                    stamps=['twoDimensionAnalysisDirectly'], colora=LOGger.WARNING)
                include_model_prediction = False
            
            # 處理固定值參數
            fixed_values = fixed_values if isinstance(fixed_values, dict) else {}
            
            # 創建一個簡單的 mdc 物件用於輸出路徑
            class SimpleMDC:
                def __init__(self):
                    self.xheader = list(data.columns)
                    self.yheader = [yheader]
                    self.report_fd = os.path.join('tmp', 'two_dimension_analysis')
                    LOGger.CreateContainer(self.report_fd)
            
            mdc = SimpleMDC()
            
            # 使用核心邏輯執行分析（核心函數會自動判斷資料類型）
            stamps = ['twoDimensionAnalysisDirectly']
            if not _performTwoDimensionAnalysisCore(mdc, data, xheader1, xheader2, yheader, fixed_values, include_model_prediction, ret, stamps, **kwags):
                return False
            
            m_addlog('Two dimension analysis completed successfully', 
                stamps=stamps, colora=LOGger.OKGREEN)
            
            return True
            
        except Exception as e:
            LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
            ret['msg'] = f'twoDimensionAnalysisDirectly error: {str(e)}'
            return False

    def analyzeTwoContinuousToY(mdc, source_data, xheader1, xheader2, yheader, y_type, fixed_values=None, include_model_prediction=True, ret=None, output_dir=None, **kwags):
        """
        兩個連續型變數對Y的分析（響應面分析）
        """
        ret = ret if isinstance(ret, dict) else {}
        
        try:
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
            
            # 條件性執行模型預測
            if include_model_prediction and mdc is not None:
                # 建立網格
                x1_margin = (x1_max - x1_min) * 0.1
                x2_margin = (x2_max - x2_min) * 0.1
                x1_range = np.linspace(x1_min - x1_margin, x1_max + x1_margin, 20)
                x2_range = np.linspace(x2_min - x2_margin, x2_max + x2_margin, 20)
                X1, X2 = np.meshgrid(x1_range, x2_range)
                
                # 準備預測用的資料
                fixed_values = fixed_values if isinstance(fixed_values, dict) else {}
                predict_data_list = []
                
                x1_flat = X1.ravel()
                x2_flat = X2.ravel()
                for k in range(len(x1_flat)):
                        predict_row = {}
                        for col in mdc.xheader:
                            if col == xheader1:
                                predict_row[col] = x1_flat[k]
                            elif col == xheader2:
                                predict_row[col] = x2_flat[k]
                            elif col in fixed_values:
                                predict_row[col] = fixed_values[col]
                            else:
                                if col in source_data.columns:
                                    col_data = source_data[col].dropna()
                                    if determineDataType(col_data) == 'continuous':
                                        predict_row[col] = col_data.mean()
                                    else:
                                        predict_row[col] = col_data.mode().iloc[0] if len(col_data.mode()) > 0 else col_data.iloc[0]
                                else:
                                    predict_row[col] = 0
                        predict_data_list.append(predict_row)
                
                predict_data = pd.DataFrame(predict_data_list)
                
                # 執行預測
                p_npData = mdc.predict(predict_data[mdc.xheader])
                y_pred_index = mdc.yheader.index(yheader) if yheader in mdc.yheader else 0
                y_pred = p_npData[:, y_pred_index] if len(p_npData.shape) > 1 else p_npData
                
                # 重塑為網格形狀
                Z = y_pred.reshape(X1.shape)
                
                # 3D 響應面
                ax1 = fig.add_subplot(221, projection='3d')
                surf = ax1.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
                
                # 添加真實資料點
                real_data = source_data[[xheader1, xheader2, yheader]].dropna()
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
                contour = ax2.contour(X1, X2, Z, levels=15)
                ax2.clabel(contour, inline=True, fontsize=8)
                contourf = ax2.contourf(X1, X2, Z, levels=15, alpha=0.6, cmap='viridis')
            else:
                # 不包含模型預測時，只顯示真實資料的散點圖
                ax1 = fig.add_subplot(221, projection='3d')
                real_data = source_data[[xheader1, xheader2, yheader]].dropna()
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
            
            # 條件性顯示切片分析
            if include_model_prediction and mdc is not None:
                # 添加真實資料點到等高線圖
                if len(real_data) > 0:
                    ax2.scatter(real_data[xheader1], real_data[xheader2], 
                            c=real_data[yheader], s=30, cmap='viridis', 
                            edgecolors='black', alpha=0.8, label='真實資料')
                
                ax2.set_xlabel(xheader1, fontproperties=vs.MJHfontprop())
                ax2.set_ylabel(xheader2, fontproperties=vs.MJHfontprop())
                ax2.set_title(f'{yheader} 等高線圖', fontproperties=vs.MJHfontprop())
                fig.colorbar(contourf, ax=ax2)
                
                # X1 固定時的切片分析
                ax3 = fig.add_subplot(223)
                mid_x1_idx = len(x1_range) // 2
                x1_fixed_value = x1_range[mid_x1_idx]
                ax3.plot(x2_range, Z[:, mid_x1_idx], 'b-', linewidth=2, 
                        label=f'{xheader1}={x1_fixed_value:.2f}時的預測')
                
                # 添加對應的真實資料
                margin = (x1_max - x1_min) * 0.1
                real_slice1 = real_data[(real_data[xheader1] < x1_max + margin) &
                                        (real_data[xheader1] > x1_min - margin)]
                if len(real_slice1) > 0:
                    ax3.scatter(real_slice1[xheader2], real_slice1[yheader], 
                            c='red', s=30, alpha=0.6, label='真實資料')
                
                ax3.set_xlabel(xheader2, fontproperties=vs.MJHfontprop())
                ax3.set_ylabel(yheader, fontproperties=vs.MJHfontprop())
                ax3.set_title(f'固定 {xheader1} 的切片分析', fontproperties=vs.MJHfontprop())
                ax3.legend(prop=vs.MJHfontprop())
                ax3.grid(True, alpha=0.3)
                
                # X2 固定時的切片分析
                ax4 = fig.add_subplot(224)
                mid_x2_idx = len(x2_range) // 2
                x2_fixed_value = x2_range[mid_x2_idx]
                ax4.plot(x1_range, Z[mid_x2_idx, :], 'g-', linewidth=2, 
                        label=f'{xheader2}={x2_fixed_value:.2f}時的預測')
                
                # 添加對應的真實資料
                margin = (x2_max - x2_min) * 0.1
                real_slice2 = real_data[(real_data[xheader2] < x2_max + margin) &
                                        (real_data[xheader2] > x2_min - margin)]
                if len(real_slice2) > 0:
                    ax4.scatter(real_slice2[xheader1], real_slice2[yheader], 
                            c='red', s=30, alpha=0.6, label='真實資料')
                
                ax4.set_xlabel(xheader1, fontproperties=vs.MJHfontprop())
                ax4.set_ylabel(yheader, fontproperties=vs.MJHfontprop())
                ax4.set_title(f'固定 {xheader2} 的切片分析', fontproperties=vs.MJHfontprop())
                ax4.legend(prop=vs.MJHfontprop())
                ax4.grid(True, alpha=0.3)
            else:
                # 不包含模型預測時，顯示統計分析圖表
                real_data = source_data[[xheader1, xheader2, yheader]].dropna()
                
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
            
            # 添加固定值 legend（左上角） 🆕
            if 'fixed_values_used' in kwags and kwags['fixed_values_used']:
                legend_text = "其他變數固定值:\n"
                for var_name, var_value in kwags['fixed_values_used'].items():
                    if var_name not in [xheader1, xheader2]:  # 排除分析變數本身
                        legend_text += f"{var_name}: {var_value}\n"
                
                # 在第一個子圖（3D圖）的左上角添加文字框
                props = dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8)
                fig.text(0.02, 0.98, legend_text.strip(), 
                        fontsize=9, verticalalignment='top', horizontalalignment='left',
                        bbox=props, fontproperties=vs3.vs.MJHfontprop())
            
            fig.tight_layout()
            
            # 儲存圖片到專案資料夾
            
            output_filename = f'two_dimension_continuous_{xheader1}_{xheader2}_{yheader}.png'
            output_dir = output_dir if LOGger.isinstance_not_empty(output_dir, str) else getattr(mdc, 'report_fd', 'analysis_output')
            if LOGger.isinstance_not_empty(output_dir, str):
                os.makedirs(output_dir, exist_ok=True)
                output_path = getProjectOutputPath(output_dir, output_filename)
                fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # 準備回傳資料
            if include_model_prediction and mdc is not None:
                ret['model_output'] = {
                    'x1_data': x1_range.tolist(),
                    'x2_data': x2_range.tolist(),
                    'surface_data': Z.tolist(),  # 3D網格資料
                    'x1_slice': Z[:, mid_x1_idx].tolist(),  # X1固定時的切片
                    'x2_slice': Z[mid_x2_idx, :].tolist(),  # X2固定時的切片
                    'x1_fixed_value': float(x1_fixed_value),
                    'x2_fixed_value': float(x2_fixed_value)
                }
            else:
                ret['model_output'] = None
            ret['record_data'] = {
                'x1_data': real_data[xheader1].tolist() if len(real_data) > 0 else [],
                'x2_data': real_data[xheader2].tolist() if len(real_data) > 0 else [],
                'y_data': real_data[yheader].tolist() if len(real_data) > 0 else []
            }
            ret['image_path'] = output_path
            ret['analysis_type'] = 'continuous_continuous_to_y'
            
            return True
            
        except Exception as e:
            LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
            ret['msg'] = f'analyzeTwoContinuousToY error: {str(e)}'
            return False

    def performTwoDimensionStatisticalTests(source_data, xheader1, xheader2, yheader, x1_type, x2_type, y_type, ret=None, **kwags):
        """
        執行二維統計檢定分析（重構版）
        """
        ret = ret if isinstance(ret, dict) else {}
        
        try:
            from package.data_analysis import TwoDimensionAnalysisManager
            
            # 使用模組化的二維分析管理器
            analysis_manager = TwoDimensionAnalysisManager(alpha=0.05)
            results = analysis_manager.perform_comprehensive_analysis(
                source_data, xheader1, xheader2, yheader, x1_type, x2_type, y_type
            )
            
            if 'error' in results:
                ret['msg'] = results['error']
                return False
            
            # 更新回傳結果
            ret['statistical_tests'] = results['statistical_tests']
            ret['interaction_analysis'] = results['interaction_analysis']
            ret['surface_analysis'] = results['surface_analysis']
            
            return True
            
        except Exception as e:
            LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
            ret['msg'] = f'performTwoDimensionStatisticalTests error: {str(e)}'
            return False

    def analyzeMixedToY(mdc, source_data, xheader1, xheader2, yheader, x1_type, x2_type, y_type, fixed_values=None, include_model_prediction=True, ret=None, output_dir=None, **kwags):
        """
        混合型分析：一個連續型 + 一個類別型變數對Y的分析
        """
        ret = ret if isinstance(ret, dict) else {}
        
        try:
            # 設定中文字型
            vs3.plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
            
            # 確定哪個是連續型，哪個是類別型
            if x1_type == 'continuous':
                continuous_header = xheader1
                categorical_header = xheader2
            else:
                continuous_header = xheader2
                categorical_header = xheader1
            
            # 取得資料
            continuous_data = source_data[continuous_header].dropna()
            categorical_data = source_data[categorical_header].dropna()
            y_data = source_data[yheader].dropna()
            
            # 取得類別和連續型範圍
            categories = categorical_data.unique()
            cont_min, cont_max = continuous_data.min(), continuous_data.max()
            cont_margin = (cont_max - cont_min) * 0.1
            cont_range = np.linspace(cont_min - cont_margin, cont_max + cont_margin, 30)
            
            # 繪製圖表
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 使用顏色區分類別
            colors = vs2.cm_rainbar(len(categories), c_alpha=0.7)
            
            # 準備回傳資料
            ret['record_data'] = {
                'continuous_data': [],
                'categorical_data': [],
                'y_data': []
            }
            
            # 條件性執行模型預測
            if include_model_prediction and mdc is not None:
                ret['model_output'] = {
                    'continuous_data': cont_range.tolist(),
                    'categories': [str(cat) for cat in categories],
                    'predictions_by_category': {}
                }
            else:
                ret['model_output'] = None
            
            # 條件性繪製預測曲線
            if include_model_prediction and mdc is not None:
                # 為每個類別繪製預測曲線
                for i, category in enumerate(categories):
                    color = colors[i] if i < len(colors) else colors[0]
                
                    # 準備預測資料
                    fixed_values = fixed_values if isinstance(fixed_values, dict) else {}
                    predict_data_list = []
                    
                    for cont_val in cont_range:
                        predict_row = {}
                        for col in mdc.xheader:
                            if col == continuous_header:
                                predict_row[col] = cont_val
                            elif col == categorical_header:
                                predict_row[col] = category
                            elif col in fixed_values:
                                predict_row[col] = fixed_values[col]
                            else:
                                if col in source_data.columns:
                                    col_data = source_data[col].dropna()
                                    if determineDataType(col_data) == 'continuous':
                                        predict_row[col] = col_data.mean()
                                    else:
                                        predict_row[col] = col_data.mode().iloc[0] if len(col_data.mode()) > 0 else col_data.iloc[0]
                                else:
                                    predict_row[col] = 0
                        predict_data_list.append(predict_row)
                    
                    predict_data = pd.DataFrame(predict_data_list)
                    
                    # 執行預測
                    p_npData = mdc.predict(predict_data[mdc.xheader])
                    y_pred_index = mdc.yheader.index(yheader) if yheader in mdc.yheader else 0
                    y_pred = p_npData[:, y_pred_index] if len(p_npData.shape) > 1 else p_npData
                    
                    # 繪製預測曲線
                    ax1.plot(cont_range, y_pred, color=color[:3], linewidth=2, 
                            label=f'{category} 預測')
                    
                    # 儲存預測資料
                    ret['model_output']['predictions_by_category'][str(category)] = y_pred.tolist()
                
                # 添加該類別的真實資料點（不論是否有預測都要顯示）
                category_real = source_data[source_data[categorical_header] == category]
                if len(category_real) > 0:
                    ax1.scatter(category_real[continuous_header], category_real[yheader], 
                            color=color[:3], alpha=0.6, s=30, 
                            label=f'{category} {"真實" if include_model_prediction else "資料"}')
                    
                    # 收集真實資料
                    ret['record_data']['continuous_data'].extend(category_real[continuous_header].tolist())
                    ret['record_data']['categorical_data'].extend([str(category)] * len(category_real))
                    ret['record_data']['y_data'].extend(category_real[yheader].tolist())
            else:
                # 不包含模型預測時，只顯示真實資料
                for i, category in enumerate(categories):
                    color = colors[i] if i < len(colors) else colors[0]
                    
                    # 添加該類別的真實資料點
                    category_real = source_data[source_data[categorical_header] == category]
                    if len(category_real) > 0:
                        ax1.scatter(category_real[continuous_header], category_real[yheader], 
                                color=color[:3], alpha=0.7, s=30, label=f'{category} 資料')
                        
                        # 收集真實資料
                        ret['record_data']['continuous_data'].extend(category_real[continuous_header].tolist())
                        ret['record_data']['categorical_data'].extend([str(category)] * len(category_real))
                        ret['record_data']['y_data'].extend(category_real[yheader].tolist())
            
            ax1.set_xlabel(continuous_header, fontproperties=vs.MJHfontprop())
            ax1.set_ylabel(yheader, fontproperties=vs.MJHfontprop())
            ax1.set_title(f'{continuous_header} 對 {yheader} 的影響（按 {categorical_header} 分類）', 
                        fontproperties=vs.MJHfontprop())
            ax1.legend(prop=vs.MJHfontprop())
            ax1.grid(True, alpha=0.3)
            
            # 類別比較圖（箱線圖）
            category_y_data = []
            category_labels = []
            for category in categories:
                cat_data = source_data[source_data[categorical_header] == category][yheader].dropna()
                if len(cat_data) > 0:
                    category_y_data.append(cat_data.values)
                    category_labels.append(str(category))
            
            if category_y_data:
                ax2.boxplot(category_y_data, labels=category_labels)
                ax2.set_xlabel(categorical_header, fontproperties=vs.MJHfontprop())
                ax2.set_ylabel(yheader, fontproperties=vs.MJHfontprop())
                ax2.set_title(f'{yheader} 按 {categorical_header} 的分布', fontproperties=vs.MJHfontprop())
                ax2.grid(True, alpha=0.3)
            
            # 連續變數的分布圖
            ax3.hist(continuous_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.set_xlabel(continuous_header, fontproperties=vs.MJHfontprop())
            ax3.set_ylabel('頻率', fontproperties=vs.MJHfontprop())
            ax3.set_title(f'{continuous_header} 分布', fontproperties=vs.MJHfontprop())
            ax3.grid(True, alpha=0.3)
            
            # Y變數的分布圖
            ax4.hist(y_data, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            ax4.set_xlabel(yheader, fontproperties=vs.MJHfontprop())
            ax4.set_ylabel('頻率', fontproperties=vs.MJHfontprop())
            ax4.set_title(f'{yheader} 分布', fontproperties=vs.MJHfontprop())
            ax4.grid(True, alpha=0.3)
            
            # 添加固定值 legend（左上角） 🆕
            if 'fixed_values_used' in kwags and kwags['fixed_values_used']:
                legend_text = "其他變數固定值:\n"
                for var_name, var_value in kwags['fixed_values_used'].items():
                    if var_name not in [xheader1, xheader2]:  # 排除分析變數本身
                        legend_text += f"{var_name}: {var_value}\n"
                
                # 在整個圖形的左上角添加文字框
                props = dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8)
                fig.text(0.02, 0.98, legend_text.strip(), 
                        fontsize=9, verticalalignment='top', horizontalalignment='left',
                        bbox=props, fontproperties=vs3.vs.MJHfontprop())
            
            fig.tight_layout()
            
            # 儲存圖片到專案資料夾
            
            output_filename = f'two_dimension_mixed_{xheader1}_{xheader2}_{yheader}.png'
            output_dir = output_dir if LOGger.isinstance_not_empty(output_dir, str) else getattr(mdc, 'report_fd', 'analysis_output')
            if LOGger.isinstance_not_empty(output_dir, str):
                os.makedirs(output_dir, exist_ok=True)
                output_path = getProjectOutputPath(output_dir, output_filename)
                fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            ret['image_path'] = output_path
            ret['analysis_type'] = f'{x1_type}_{x2_type}_to_{y_type}'
            
            return True
            
        except Exception as e:
            LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
            ret['msg'] = f'analyzeMixedToY error: {str(e)}'
            return False

    def analyzeTwoCategoricalToY(mdc, source_data, xheader1, xheader2, yheader, y_type, fixed_values=None, include_model_prediction=True, ret=None, output_dir=None, **kwags):
        """
        兩個類別型變數對Y的分析
        """
        ret = ret if isinstance(ret, dict) else {}
        
        try:
            LOGger.addDebug('source_data', str(source_data.columns))
            LOGger.addDebug('bool:', str('CDATEByS' in source_data.columns))
            # 設定中文字型
            vs3.plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
            
            # 取得類別
            x1_categories = source_data[xheader1].dropna().unique()
            x2_categories = source_data[xheader2].dropna().unique()
            
            # 繪製圖表
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 準備回傳資料
            ret['record_data'] = {
                'x1_data': [],
                'x2_data': [],
                'y_data': [],
                'combination_stats': {}
            }
            
            # 條件性執行模型預測
            if include_model_prediction and mdc is not None:
                ret['model_output'] = {
                    'x1_categories': [str(cat) for cat in x1_categories],
                    'x2_categories': [str(cat) for cat in x2_categories],
                    'predictions_matrix': [],
                    'combination_predictions': {}
                }
                
                # 建立組合預測矩陣
                prediction_matrix = []
                fixed_values = fixed_values if isinstance(fixed_values, dict) else {}
                
                for x1_cat in x1_categories:
                    row_predictions = []
                    for x2_cat in x2_categories:
                        # 準備預測資料
                        predict_row = {}
                        for col in mdc.xheader:
                            if col == xheader1:
                                predict_row[col] = x1_cat
                            elif col == xheader2:
                                predict_row[col] = x2_cat
                            elif col in fixed_values:
                                predict_row[col] = fixed_values[col]
                            else:
                                if col in source_data.columns:
                                    col_data = source_data[col].dropna()
                                    if determineDataType(col_data) == 'continuous':
                                        predict_row[col] = col_data.mean()
                                    else:
                                        predict_row[col] = col_data.mode().iloc[0] if len(col_data.mode()) > 0 else col_data.iloc[0]
                                else:
                                    predict_row[col] = 0
                        
                        predict_data = pd.DataFrame([predict_row])
                        
                        # 執行預測
                        p_npData = mdc.predict(predict_data[mdc.xheader])
                        y_pred_index = mdc.yheader.index(yheader) if yheader in mdc.yheader else 0
                        y_pred = p_npData[0, y_pred_index] if len(p_npData.shape) > 1 else p_npData[0]
                        
                        row_predictions.append(float(y_pred))
                        ret['model_output']['combination_predictions'][f'{x1_cat}_{x2_cat}'] = float(y_pred)
                    
                    prediction_matrix.append(row_predictions)
                
                ret['model_output']['predictions_matrix'] = prediction_matrix
                
                # 繪製預測熱力圖
                sns.heatmap(prediction_matrix, 
                        xticklabels=[str(cat) for cat in x2_categories],
                        yticklabels=[str(cat) for cat in x1_categories],
                        annot=True, fmt='.2f', cmap='viridis', ax=ax1)
                ax1.set_xlabel(xheader2, fontproperties=vs.MJHfontprop())
                ax1.set_ylabel(xheader1, fontproperties=vs.MJHfontprop())
                ax1.set_title(f'{yheader} 預測值熱力圖', fontproperties=vs.MJHfontprop())
            else:
                ret['model_output'] = None
                
                # 不包含模型預測時，只顯示真實資料統計
                ax1.text(0.5, 0.5, '模型預測已關閉\n只顯示統計分析', 
                        transform=ax1.transAxes, ha='center', va='center',
                        fontsize=16, fontproperties=vs.MJHfontprop())
                ax1.set_title(f'{xheader1} + {xheader2} 對 {yheader} 統計分析', fontproperties=vs.MJHfontprop())
            
            # 真實資料的組合分析
            real_combinations = {}
            for x1_cat in x1_categories:
                for x2_cat in x2_categories:
                    combo_data = source_data[
                        (source_data[xheader1] == x1_cat) & 
                        (source_data[xheader2] == x2_cat)
                    ][yheader].dropna()
                    
                    if len(combo_data) > 0:
                        combo_key = f'{x1_cat}_{x2_cat}'
                        real_combinations[combo_key] = {
                            'mean': float(combo_data.mean()),
                            'std': float(combo_data.std()),
                            'count': len(combo_data),
                            'values': combo_data.tolist()
                        }
                        
                        # 收集真實資料
                        combo_source = source_data[
                            (source_data[xheader1] == x1_cat) & 
                            (source_data[xheader2] == x2_cat)
                        ]
                        ret['record_data']['x1_data'].extend([str(x1_cat)] * len(combo_source))
                        ret['record_data']['x2_data'].extend([str(x2_cat)] * len(combo_source))
                        ret['record_data']['y_data'].extend(combo_source[yheader].tolist())
            
            ret['record_data']['combination_stats'] = real_combinations
            
            # 繪製真實資料的平均值熱力圖
            real_matrix = []
            for x1_cat in x1_categories:
                real_row = []
                for x2_cat in x2_categories:
                    combo_key = f'{x1_cat}_{x2_cat}'
                    if combo_key in real_combinations:
                        real_row.append(real_combinations[combo_key]['mean'])
                    else:
                        real_row.append(np.nan)
                real_matrix.append(real_row)
            
            sns.heatmap(real_matrix, 
                    xticklabels=[str(cat) for cat in x2_categories],
                    yticklabels=[str(cat) for cat in x1_categories],
                    annot=True, fmt='.2f', cmap='plasma', ax=ax2)
            ax2.set_xlabel(xheader2, fontproperties=vs.MJHfontprop())
            ax2.set_ylabel(xheader1, fontproperties=vs.MJHfontprop())
            ax2.set_title(f'{yheader} 真實平均值熱力圖', fontproperties=vs.MJHfontprop())
            
            # X1類別的Y值分布
            x1_y_data = []
            x1_labels = []
            for x1_cat in x1_categories:
                cat_data = source_data[source_data[xheader1] == x1_cat][yheader].dropna()
                if len(cat_data) > 0:
                    x1_y_data.append(cat_data.values)
                    x1_labels.append(str(x1_cat))
            
            if x1_y_data:
                ax3.boxplot(x1_y_data, labels=x1_labels)
                ax3.set_xlabel(xheader1, fontproperties=vs.MJHfontprop())
                ax3.set_ylabel(yheader, fontproperties=vs.MJHfontprop())
                ax3.set_title(f'{yheader} 按 {xheader1} 的分布', fontproperties=vs.MJHfontprop())
                ax3.grid(True, alpha=0.3)
            
            # X2類別的Y值分布
            x2_y_data = []
            x2_labels = []
            for x2_cat in x2_categories:
                cat_data = source_data[source_data[xheader2] == x2_cat][yheader].dropna()
                if len(cat_data) > 0:
                    x2_y_data.append(cat_data.values)
                    x2_labels.append(str(x2_cat))
            
            if x2_y_data:
                ax4.boxplot(x2_y_data, labels=x2_labels)
                ax4.set_xlabel(xheader2, fontproperties=vs.MJHfontprop())
                ax4.set_ylabel(yheader, fontproperties=vs.MJHfontprop())
                ax4.set_title(f'{yheader} 按 {xheader2} 的分布', fontproperties=vs.MJHfontprop())
                ax4.grid(True, alpha=0.3)
            
            # 添加固定值 legend（左上角） 🆕
            if 'fixed_values_used' in kwags and kwags['fixed_values_used']:
                legend_text = "其他變數固定值:\n"
                for var_name, var_value in kwags['fixed_values_used'].items():
                    if var_name not in [xheader1, xheader2]:  # 排除分析變數本身
                        legend_text += f"{var_name}: {var_value}\n"
                
                # 在整個圖形的左上角添加文字框
                props = dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8)
                fig.text(0.02, 0.98, legend_text.strip(), 
                        fontsize=9, verticalalignment='top', horizontalalignment='left',
                        bbox=props, fontproperties=vs3.vs.MJHfontprop())
            
            fig.tight_layout()
            
            # 儲存圖片到專案資料夾
            
            output_filename = f'two_dimension_categorical_{xheader1}_{xheader2}_{yheader}.png'
            output_dir = output_dir if LOGger.isinstance_not_empty(output_dir, str) else getattr(mdc, 'report_fd', 'analysis_output')
            if LOGger.isinstance_not_empty(output_dir, str):
                os.makedirs(output_dir, exist_ok=True)
                output_path = getProjectOutputPath(output_dir, output_filename)
                fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            ret['image_path'] = output_path
            ret['analysis_type'] = 'categorical_categorical_to_y'
            
            return True
            
        except Exception as e:
            LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
            ret['msg'] = f'analyzeTwoCategoricalToY error: {str(e)}'
            return False


def updatePKLFileProtocal(source_file, protocol=4, exp_fd='updatePKLP', handler=None, exp_file=None, exp_fn=None, **kwags):
    exp_fd = exp_fd if(LOGger.isinstance_not_empty(exp_fd, str)) else LOGger.execute('exp_fd', handler, default='.', not_found_alarm=False)
    exp_fn = exp_fn if(LOGger.isinstance_not_empty(exp_fn, str)) else LOGger.execute('exp_fn', handler, default=None, not_found_alarm=False)
    if(exp_fn is None): exp_fn = os.path.basename(source_file)
    exp_file = exp_file if(LOGger.isinstance_not_empty(exp_file, str)) else os.path.join(exp_fd, exp_fn)
    if(not os.path.exists(source_file)):
        return False
    with open(source_file, 'rb') as f:
        data = DFP.pickle.load(f)
    if(protocol is None):   protocol = DFP.pickle.HIGHEST_PROTOCOL
    with open(exp_file, 'wb') as f:
        DFP.pickle.dump(data, f, protocol=protocol)
    return True

#%%
def chineseModuleActivate():
    # global m_dictionary
    # m_dictionary.clear()
    # m_dictionary.update({})
    vs3.plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']

def configHandlerPainting(handler, figsize, maxFrames, drawFreq, img_exp_fd, stamps=None, 
                          figLabel='fig', rcFig=None, **kwags):
    handler.img_exp_fd = img_exp_fd
    stamps = stamps if(isinstance(stamps, list)) else []
    handler.figsize = figsize
    handler.drawFreq = drawFreq
    handler.dataCount = 0
    if(rcFig is not None):
        handler.rFig, handler.cFig = rcFig[0], rcFig[1]
        handler.maxFrames = handler.rFig * handler.cFig
    else:
        handler.maxFrames = maxFrames
        nSqrtAxes = int(np.sqrt(maxFrames))
        handler.rFig, handler.cFig = nSqrtAxes, nSqrtAxes+1
    setattr(handler,figLabel,vs3.plt.Figure(figsize=handler.figsize))
    dynFigFileLabel = 'dyn%sFile'%figLabel.title()
    setattr(handler,dynFigFileLabel,os.path.join(img_exp_fd, '%s.jpg'%LOGger.stamp_process('',[*stamps, 'Illustrtion'],'','','','_',for_file=True)))
    setattr(handler,'meta%s'%figLabel.title(), LOGger.mystr())
    handler.xlsx_dont_save_header = ['value','leftValue','upValue','tensor','leftTensor','upTensor']
    handler.df = pd.DataFrame()
    handler.logfile = os.path.join(handler.exp_fd, 'log.txt')
    handler.addlog = LOGger.addloger(logfile=handler.logfile)
    handler.print = LOGger.addloger(logfile='')
    return True

def projectInitial(exp_fd_default='test', config_file='config.json', labels=None, label_encoder_file=None, 
                   binary_inlier_code=0, figsize=(12,12), maxFrames=12, drawFreq=1, rewrite=True, **kwags):
    chineseModuleActivate()
    handler = LOGger.mystr()
    project_buffer = kwags.get('project_buffer', {})
    if(LOGger.isinstance_not_empty(project_buffer.get('exp_fd'), str)):
        handler.exp_fd = dcp(project_buffer['exp_fd'])
    else:
        handler.exp_fd = exp_fd_default
        project_buffer.update({'exp_fd': handler.exp_fd})
    if(rewrite):
        if(os.path.exists(handler.exp_fd) and handler.exp_fd!='.'):
            LOGger.removefile(handler.exp_fd)
    if(not os.path.exists(handler.exp_fd)):
        LOGger.CreateContainer(handler.exp_fd)
    handler.logfile = os.path.join(handler.exp_fd, 'log.txt')
    handler.addlog = LOGger.addloger(logfile=handler.logfile)
    handler.stamps = kwags.get('stamps', [])
    handler.model_fn = None
    
    kwags['stamps'] = handler.stamps
    if(not configHandlerPainting(handler, figsize, maxFrames, drawFreq, handler.exp_fd, **kwags)):
        return False
    return handler

def p0(default_exp_fd='%s_p0'%(os.path.basename(__file__).split('.')[0]), **kwags):
    handler = projectInitial(exp_fd_default=default_exp_fd, **kwags)
    handler.projectName = 'p0'
    project_buffer = kwags.get('project_buffer', {})
    project_buffer['exp_fd'] = handler.exp_fd
    try:
        Data = DFP.import_data(project_buffer['source_file'], sht=project_buffer.get('sht',0))
        xheader = project_buffer.get('xheader')
        if(xheader is None):    
            # for x in Data.columns:
            #     LOGger.addDebug(str(x), type(Data[x]), str(Data[x]))
            xheader = [x for x in Data.columns if not Data[x].map(lambda x:DFP.astype(x,default=np.nan)).isna().any()]
        project_buffer['xheader'] = [x for x in xheader if x not in project_buffer['yheader']]
        otherData = Data[project_buffer['xheader']]
        inspectData = Data[project_buffer['yheader']]
        
        estimatorFile = getattr(handler,'estimatorFile',None)
        mineAlpha = project_buffer.get('mineAlpha', 0.6)
        mineC = project_buffer.get('mineC', 15)
        mine = MINE(alpha=mineAlpha, c=mineC) if(not isinstance(estimatorFile, str)) else loadMINE(handler)
        # kwags.update({'inspectData':inspectData, 'otherData':otherData, 'mine':mine})
        kwags['exp_fd'] = handler.exp_fd
        kwags['mine'] = mine
        if(not factorRelativityAnalysisMIC(inspectData, otherData, handler=handler, **kwags)):
            return False
        handler.estimator = mine
        if(not evaluationScenario(handler=handler, mineAlpha=mineAlpha, mineC=mineC, **kwags)):
            return False
        project_buffer['mineAlpha'] = mineAlpha
        project_buffer['mineC'] = mineC
    except Exception as e:
        LOGger.exception_process(e, logfile='')
        return False
    return True

def p1(default_exp_fd='%s_p1'%(os.path.basename(__file__).split('.')[0]), selectedHeader=None, **kwags):
    handler = projectInitial(exp_fd_default=default_exp_fd, **kwags)
    handler.projectName = 'p1'
    project_buffer = kwags.get('project_buffer', {})
    project_buffer['exp_fd'] = handler.exp_fd
    Data = DFP.import_data(project_buffer['source_file'], sht=project_buffer.get('sht',0))
    if(isinstance(selectedHeader,list)):  Data = Data[selectedHeader]
    kwags['exp_fd'] = handler.exp_fd
    kwags['infrm_default'] = kwags.get('infrm')
    if(not plotDataDistribution(Data, fig=handler.fig, handler=handler, do_stats=True, **kwags)):
        return False
    ret = {}
    if(LOGger.isinstance_not_empty(kwags.get('infrm'), dict)): ret['infrm'] = kwags.get('infrm')
    if(LOGger.isinstance_not_empty(kwags.get('infrms'), dict)): ret['infrms'] = kwags.get('infrms')
    statResFile = os.path.join(handler.exp_fd, 'statRes.json')
    LOGger.CreateFile(statResFile, lambda f:LOGger.save_json(ret, file=f))
    return True

def p2(default_exp_fd='%s_p2'%(os.path.basename(__file__).split('.')[0]), selectedHeader=None, snsHeight=5, **kwags):
    handler = projectInitial(exp_fd_default=default_exp_fd, **kwags)
    handler.projectName = 'p2'
    project_buffer = kwags.get('project_buffer', {})
    project_buffer['exp_fd'] = handler.exp_fd
    Data = DFP.import_data(project_buffer['source_file'], sht=project_buffer.get('sht',0))
    if(isinstance(selectedHeader,list)):  Data = Data[selectedHeader]
    kwags['exp_fd'] = handler.exp_fd
    kwags['infrm_default'] = kwags.get('infrm')
    if(not regressionHeatmapScenario(Data, handler=handler, do_stats=True, height=snsHeight, **kwags)):
        return False
    ret = {}
    if(LOGger.isinstance_not_empty(kwags.get('infrm'), dict)): ret['infrm'] = kwags.get('infrm')
    if(LOGger.isinstance_not_empty(kwags.get('infrms'), dict)): ret['infrms'] = kwags.get('infrms')
    statResFile = os.path.join(handler.exp_fd, 'statRes.json')
    LOGger.CreateFile(statResFile, lambda f:LOGger.save_json(ret, file=f))
    return True

def p3(default_exp_fd='%s_p3'%(os.path.basename(__file__).split('.')[0]), selectedHeader=None, snsHeight=5, 
       mask=None, maskHeader=None, **kwags):
    handler = projectInitial(exp_fd_default=default_exp_fd, **kwags)
    handler.projectName = 'p3'
    project_buffer = kwags.get('project_buffer', {})
    project_buffer['exp_fd'] = handler.exp_fd
    Data = DFP.import_data(project_buffer['source_file'], sht=project_buffer.get('sht',0))
    if(mask is None):
        if(maskHeader is not None):
            mask = Data[maskHeader].map(lambda x:DFP.parse(x)).values
    if(isinstance(selectedHeader,list)):  Data = Data[selectedHeader]
    kwags['exp_fd'] = handler.exp_fd
    kwags['infrm_default'] = kwags.get('infrm')
    if(not plotHistogramScenario(Data, handler=handler, do_stats=True, height=snsHeight, mask=mask, **kwags)):
        return False
    ret = {}
    if(LOGger.isinstance_not_empty(kwags.get('infrm'), dict)): ret['infrm'] = kwags.get('infrm')
    if(LOGger.isinstance_not_empty(kwags.get('infrms'), dict)): ret['infrms'] = kwags.get('infrms')
    statResFile = os.path.join(handler.exp_fd, 'statRes.json')
    LOGger.CreateFile(statResFile, lambda f:LOGger.save_json(ret, file=f))
    return True

def pUPKL(default_exp_fd='.', rewrite=False, **kwags):
    handler = projectInitial(exp_fd_default=default_exp_fd, rewrite=rewrite, **kwags)
    handler.projectName = 'pUPKL'
    project_buffer = kwags.get('project_buffer', {})
    project_buffer['exp_fd'] = handler.exp_fd
    kwags['exp_fd'] = handler.exp_fd
    kwags['infrm_default'] = kwags.get('infrm')
    if(not updatePKLFileProtocal(**kwags)):
        return False
    return True

def method_activation(stg):
    try:
        method = eval(stg)
        return method
    except:
        print('method invalid:%s!!!!'%stg)
        return None

def scenario():
    args_history = LOGger.load_json(json_file) if(os.path.exists(json_file)) else {}
    
    if('xheader_zones' in args_history):
        xheader_zones = LOGger.mydict()
        for k,v in args_history['xheader_zones'].items():
            xheader_zones[k] = activateZone(v, stamps=['yPsr', k], exp_fd='temp')
        args_history['xheader'] = xheader_zones.concatenate()
    if('yheader_zones' in args_history):
        yheader_zones = LOGger.mydict()
        for k,v in args_history['yheader_zones'].items():
            yheader_zones[k] = activateZone(v, stamps=['yPsr', k], exp_fd='temp')
        args_history['yheader'] = yheader_zones.concatenate()
    
    parser = LOGger.myArgParser()
    parser.add_argument("-prmth", "--project_method_stg", type=str, default='p0',
                        help="(p0: 分析每一個feature對於其他features的影響)")
    parser.add_argument("-c", "--defaultConfigFile", type=str, help="(config.json)")
    parser.add_argument("-i", "--source_file", type=str, help="資料來源路徑", default=args_history.get('source_file'))
    parser.add_argument("-ifd", "--source_fd", type=str, help="資料來源路徑資料夾")
    parser.add_argument("-imd", "--source_model", type=str, help="模型名稱")
    parser.add_argument("-msh", "--maskHeader", type=str, help="用來作為分群的欄位名稱", default=args_history.get('maskHeader'))
    parser.add_argument("-snsh", "--snsHeight", type=float, help="sns關係圖的尺寸", default=5)
    parser.add_argument("-vm", "--visualizeMethod", type=str, help="視覺化的方式(report; report_normhist,...)", default='report')
    parser.add_argument("-th", "--selectThreshold", type=float, help="產出選中的特徵需要大於的門檻值", nargs='?')
    parser.add_argument("-sh", "--selectedHeader", type=eval, help="selectedHeader", default=args_history.get('selectedHeader'))
    parser.add_argument("-xh", "--xheader", type=eval, help="xheader()", default=args_history.get('xheader'))
    parser.add_argument("-yh", "--yheader", type=eval, help="yheader()", default=args_history.get('yheader'))
    parser.add_argument("-sv", "--save_types", type=eval, help="save_types(?; ['xlsx', 'pkl'])", nargs='?')
    parser.add_argument("-fs", "--figsize", type=eval, help="figsize`(12,12)`", default=(12,12))
    parser.add_argument("-st", "--stamps", type=eval, help="stamps`?; []`", nargs='?')
    parser.add_argument("-o", "--exp_fd", type=str, help="暫存輸出資料夾(?)", nargs='?')
    parser.add_argument("-op", "--exp_file", type=str, help="輸出的檔案路徑(?)", nargs='?')
    parser.add_argument("-of", "--exp_fn", type=str, help="輸出檔名(?)", nargs='?')
    parser.add_argument("-ptc", "--protocol", type=str, help="protocol(4)", default=4)
    args = parser.parse_args()
    project_buffer = vars(args)
    if(LOGger.isinstance_not_empty(project_buffer.get('defaultConfigFile'),str)):
        if(os.path.isfile(project_buffer.get('defaultConfigFile'))):  
            LOGger.addDebug(project_buffer.get('defaultConfigFile'))
            defaultConfig = LOGger.load_json(project_buffer.get('defaultConfigFile'))
            defaultConfig.update(project_buffer)
            project_buffer = dcp(defaultConfig)
    if(project_buffer['project_method_stg'] in ['p1']):
        visualizeMethod = project_buffer.get('visualizeMethod')
        if(not callable(getattr(vs3, visualizeMethod, None))):
            LOGger.addlog('visualizeMethod invalid:%s!!!!!'%visualizeMethod, colora=LOGger.FAIL, logfile='')
            return 
    print('project_buffer:', LOGger.stamp_process('',project_buffer,':','[',']','\n'))
    # project_buffer['exp_fd'] = None if(project_buffer.get('is_auto_exp_fd')) else project_buffer['exp_fd']
    report = {}
    
    project_method = method_activation(project_buffer.get('project_method_stg'))
    if(project_method==None):
        return
    if(not project_method(project_buffer=project_buffer, report=report, **project_buffer)):
        return
    exp_fd = project_buffer.pop('exp_fd', '.')
    _ = project_buffer.pop('defaultConfigFile', None)
    for k,v in project_buffer.items():
        LOGger.addlog(str(type(v)), logfile=os.path.join(exp_fd, 'log.txt'), stamps=[k])
    LOGger.CreateFile(os.path.join(exp_fd, os.path.basename(json_file)), lambda f:LOGger.save_json(project_buffer, f))
    if(report):
        print(report)
        DFP.project_record_ending(report, sheet_name='main', exp_fd=exp_fd, theme=m_theme)

#%%
# 共用結論生成函式（單/雙維皆可使用）
def generate_conclusion_with_effect(x_names, y_name, n_significant, r_squared):
    """
    根據顯著性與效應量產生結論。
    - x_names: list[str]，可含一或多個自變數名稱
    - y_name: str，因變數名稱
    - n_significant: 顯著檢定數量
    - r_squared: 效應量（可為 None）
    """
    x_names = [str(x) for x in x_names if x]
    x_part = ' 和 '.join(x_names) if x_names else 'X'

    if n_significant == 0:
        return f'{x_part} 對 {y_name} 無顯著影響'

    if r_squared is None:
        return f'{x_part} 對 {y_name} 達統計顯著，但缺乏效應量資訊'

    if r_squared < 0.01:
        return f'{x_part} 與 {y_name} 達統計顯著，但效應量極小（R²≈{r_squared:.3f}），實務解釋力有限'
    elif r_squared < 0.05:
        return f'{x_part} 對 {y_name} 達統計顯著，效應量較小（R²≈{r_squared:.3f}）'
    elif r_squared < 0.15:
        return f'{x_part} 對 {y_name} 有顯著影響，效應量中等（R²≈{r_squared:.3f}）'
    else:
        return f'{x_part} 對 {y_name} 有顯著影響，效應量較大（R²≈{r_squared:.3f}）'


# 統計分析和視覺化管理類別
class StatisticalTestManager:
    """
    統計檢定管理類別
    負責執行各種統計檢定並返回標準化結果
    """
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        from scipy import stats
        self.stats = stats
        
    def perform_continuous_vs_continuous_tests(self, x_data, y_data):
        """執行連續型對連續型的統計檢定"""
        results = {}
        
        # Pearson相關係數
        try:
            r, p = self.stats.pearsonr(x_data, y_data)
            results['pearson_correlation'] = {
                'correlation': float(r),
                'p_value': float(p),
                'significant': bool(p < self.alpha),
                'interpretation': 'strong' if abs(r) > 0.7 else 'moderate' if abs(r) > 0.3 else 'weak',
                'explanation': 'Pearson 假設線性、常態且同質變異；p<0.05 表示線性相關顯著'
            }
        except Exception as e:
            results['pearson_correlation'] = {'error': str(e)}
            
        # Spearman相關係數
        try:
            r, p = self.stats.spearmanr(x_data, y_data)
            results['spearman_correlation'] = {
                'correlation': float(r),
                'p_value': float(p),
                'significant': bool(p < self.alpha),
                'interpretation': 'strong' if abs(r) > 0.7 else 'moderate' if abs(r) > 0.3 else 'weak',
                'explanation': 'Spearman 基於秩次、單調關係；p<0.05 表示單調相關顯著'
            }
        except Exception as e:
            results['spearman_correlation'] = {'error': str(e)}
            
        # Kendall's tau相關係數
        try:
            tau, p = self.stats.kendalltau(x_data, y_data)
            results['kendall_tau'] = {
                'correlation': float(tau),
                'p_value': float(p),
                'significant': bool(p < self.alpha),
                'explanation': 'Kendall tau 比較秩次一致性；p<0.05 表示秩次相關顯著'
            }
        except Exception as e:
            results['kendall_tau'] = {'error': str(e)}
            
        # 線性回歸檢定
        try:
            slope, intercept, r_value, p_value, std_err = self.stats.linregress(x_data, y_data)
            results['linear_regression'] = {
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_value**2),
                'p_value': float(p_value),
                'standard_error': float(std_err),
                'significant': bool(p_value < self.alpha),
                'explanation': '單變量線性回歸假設線性、殘差常態與等變異；p<0.05 斜率顯著，R² 為解釋力'
            }
        except Exception as e:
            results['linear_regression'] = {'error': str(e)}
            
        return results
        
    def perform_categorical_vs_continuous_tests(self, x_data, y_data):
        """執行分類型對連續型的統計檢定"""
        results = {}
        
        # 取得各組資料
        groups = []
        group_names = []
        for category in x_data.unique():
            group_data = y_data[x_data == category].dropna()
            if len(group_data) > 0:
                groups.append(group_data)
                group_names.append(str(category))
                
        results['group_info'] = {
            'n_groups': len(groups),
            'group_names': group_names,
            'group_sizes': [len(g) for g in groups]
        }
        
        if len(groups) >= 2:
            # 兩組比較的檢定
            if len(groups) == 2:
                # Kolmogorov-Smirnov檢定
                try:
                    ks_stat, ks_p = self.stats.ks_2samp(groups[0], groups[1])
                    results['kolmogorov_smirnov'] = {
                        'statistic': float(ks_stat),
                        'p_value': float(ks_p),
                        'significant': bool(ks_p < self.alpha),
                        'interpretation': '兩組分布顯著不同' if ks_p < self.alpha else '兩組分布無顯著差異',
                        'explanation': 'KS 檢定比較兩組分布；p<0.05 表示分布不同，p≥0.05 表示未偵測到差異（獨立樣本）'
                    }
                except Exception as e:
                    results['kolmogorov_smirnov'] = {'error': str(e)}
                    
                # Mann-Whitney U檢定
                try:
                    u_stat, u_p = self.stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                    results['mann_whitney_u'] = {
                        'statistic': float(u_stat),
                        'p_value': float(u_p),
                        'significant': bool(u_p < self.alpha),
                        'interpretation': '兩組中位數顯著不同' if u_p < self.alpha else '兩組中位數無顯著差異',
                        'explanation': 'Mann-Whitney 比較兩組位置/中位數，假設分布形狀相近；p<0.05 表示位置顯著不同'
                    }
                except Exception as e:
                    results['mann_whitney_u'] = {'error': str(e)}
                    
                # t檢定
                try:
                    levene_stat, levene_p = self.stats.levene(groups[0], groups[1])
                    equal_var = levene_p > self.alpha
                    t_stat, t_p = self.stats.ttest_ind(groups[0], groups[1], equal_var=equal_var)
                    results['t_test'] = {
                        'statistic': float(t_stat),
                        'p_value': float(t_p),
                        'significant': bool(t_p < self.alpha),
                        'equal_variance': equal_var,
                        'levene_p': float(levene_p),
                        'interpretation': '兩組平均數顯著不同' if t_p < self.alpha else '兩組平均數無顯著差異',
                        'explanation': 't 檢定假設常態，Levene p>0.05 時視為等變異；p<0.05 表示平均數顯著不同'
                    }
                except Exception as e:
                    results['t_test'] = {'error': str(e)}
            else:
                # 多組比較的檢定
                # Kruskal-Wallis檢定
                try:
                    kw_stat, kw_p = self.stats.kruskal(*groups)
                    results['kruskal_wallis'] = {
                        'statistic': float(kw_stat),
                        'p_value': float(kw_p),
                        'significant': bool(kw_p < self.alpha),
                        'interpretation': '各組中位數顯著不同' if kw_p < self.alpha else '各組中位數無顯著差異',
                        'explanation': 'Kruskal-Wallis 為多組秩次檢定，分布形狀相近；p<0.05 表示至少一組位置不同'
                    }
                except Exception as e:
                    results['kruskal_wallis'] = {'error': str(e)}
                    
                # ANOVA檢定
                try:
                    f_stat, anova_p = self.stats.f_oneway(*groups)
                    results['anova'] = {
                        'f_statistic': float(f_stat),
                        'p_value': float(anova_p),
                        'significant': bool(anova_p < self.alpha),
                        'interpretation': '各組平均數顯著不同' if anova_p < self.alpha else '各組平均數無顯著差異',
                        'explanation': '單因子 ANOVA 假設常態、等變異；p<0.05 表示至少一組平均數不同，未顯著表示未偵測差異'
                    }
                except Exception as e:
                    results['anova'] = {'error': str(e)}
                    
        return results
        
    def perform_continuous_vs_categorical_tests(self, x_data, y_data):
        """執行連續型對分類型的統計檢定"""
        results = {}
        
        # 將分類變數編碼為數值
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_data)
        
        # Point-biserial相關（如果y是二元分類）
        if len(np.unique(y_encoded)) == 2:
            try:
                pb_r, pb_p = self.stats.pointbiserialr(y_encoded, x_data)
                results['point_biserial'] = {
                    'correlation': float(pb_r),
                    'p_value': float(pb_p),
                    'significant': bool(pb_p < self.alpha),
                    'interpretation': '連續變數與二元分類顯著相關' if pb_p < self.alpha else '連續變數與二元分類無顯著相關',
                    'explanation': 'Point-biserial 比較連續與二元分類的線性相關；p<0.05 表示相關顯著'
                }
            except Exception as e:
                results['point_biserial'] = {'error': str(e)}
                
        # Spearman相關（將分類變數視為順序變數）
        try:
            spearman_r, spearman_p = self.stats.spearmanr(x_data, y_encoded)
            results['spearman_correlation'] = {
                'correlation': float(spearman_r),
                'p_value': float(spearman_p),
                'significant': bool(spearman_p < self.alpha),
                'explanation': 'Spearman 基於秩次單調關係；p<0.05 表示單調相關顯著'
            }
        except Exception as e:
            results['spearman_correlation'] = {'error': str(e)}
            
        return results
        
    def perform_categorical_vs_categorical_tests(self, x_data, y_data):
        """執行分類型對分類型的統計檢定"""
        results = {}
        
        # 建立列聯表
        contingency_table = pd.crosstab(x_data, y_data)
        
        # Chi-square獨立性檢定
        try:
            chi2, chi2_p, dof, expected = self.stats.chi2_contingency(contingency_table)
            results['chi_square'] = {
                'chi2_statistic': float(chi2),
                'p_value': float(chi2_p),
                'degrees_of_freedom': int(dof),
                'significant': bool(chi2_p < self.alpha),
                'interpretation': '兩變數顯著相關' if chi2_p < self.alpha else '兩變數無顯著相關',
                'explanation': '卡方獨立性檢定假設期望次數足夠且樣本獨立；p<0.05 表示變數相關'
            }
            
            # Cramér's V（效應量）
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
            results['cramers_v'] = {
                'value': float(cramers_v),
                'interpretation': 'strong' if cramers_v > 0.5 else 'moderate' if cramers_v > 0.3 else 'weak',
                'explanation': 'Cramér’s V 為列聯表效應量，越大關聯越強'
            }
        except Exception as e:
            results['chi_square'] = {'error': str(e)}
            
        # Fisher精確檢定（如果是2x2表格）
        if contingency_table.shape == (2, 2):
            try:
                oddsratio, fisher_p = self.stats.fisher_exact(contingency_table)
                results['fisher_exact'] = {
                    'odds_ratio': float(oddsratio),
                    'p_value': float(fisher_p),
                    'significant': bool(fisher_p < self.alpha),
                    'interpretation': '兩變數顯著相關' if fisher_p < self.alpha else '兩變數無顯著相關',
                    'explanation': 'Fisher 精確檢定用於 2x2 小樣本；p<0.05 表示變數相關'
                }
            except Exception as e:
                results['fisher_exact'] = {'error': str(e)}
                
        return results
        
    def perform_tests(self, x_data, y_data, x_type, y_type):
        """根據資料類型執行相應的統計檢定"""
        analysis_type = f'{x_type}_vs_{y_type}'
        
        if x_type == 'continuous' and y_type == 'continuous':
            results = self.perform_continuous_vs_continuous_tests(x_data, y_data)
        elif x_type == 'categorical' and y_type == 'continuous':
            results = self.perform_categorical_vs_continuous_tests(x_data, y_data)
        elif x_type == 'continuous' and y_type == 'categorical':
            results = self.perform_continuous_vs_categorical_tests(x_data, y_data)
        elif x_type == 'categorical' and y_type == 'categorical':
            results = self.perform_categorical_vs_categorical_tests(x_data, y_data)
        else:
            results = {}
            
        results['analysis_type'] = analysis_type
        results['sample_size'] = len(x_data)
        
        # 計算整體顯著性總結
        significant_tests = []
        for test_name, test_result in results.items():
            if isinstance(test_result, dict) and 'significant' in test_result:
                if test_result['significant']:
                    significant_tests.append(test_name)
                    
        r_squared = None
        if 'linear_regression' in results and isinstance(results['linear_regression'], dict):
            r_squared = results['linear_regression'].get('r_squared', None)

        results['summary'] = {
            'significant_tests': significant_tests,
            'n_significant': len(significant_tests),
            'overall_significant': bool(len(significant_tests) > 0),
            'conclusion': generate_conclusion_with_effect(
                x_names=[getattr(x_data, 'name', 'X')],
                y_name=getattr(y_data, 'name', 'Y'),
                n_significant=len(significant_tests),
                r_squared=r_squared
            )
        }
        
        return results

class ChartAnnotationManager:
    """
    圖表標註管理類別
    負責在圖表上添加統計檢定資訊
    """
    
    def __init__(self):
        pass
        
    def add_statistical_info_to_plot(self, ax, statistical_tests, xheader, yheader):
        """在圖表上添加統計檢定信息"""
        try:
            # 收集統計信息並按重要性分類
            basic_info = []  # 基本信息
            detailed_info = []  # 詳細檢定結果
            
            # 樣本數
            sample_size = statistical_tests.get('sample_size', 0)
            if sample_size > 0:
                basic_info.append(f'樣本數: {sample_size}')
                
            # 根據分析類型收集統計檢定結果
            analysis_type = statistical_tests.get('analysis_type', '')
            significant_count = 0
            min_p_value = 1.0
            max_effect_size = 0.0
            
            if analysis_type == 'continuous_vs_continuous':
                significant_count, min_p_value, max_effect_size, detailed_info = self._process_continuous_vs_continuous(
                    statistical_tests, significant_count, min_p_value, max_effect_size, detailed_info)
                    
            elif analysis_type == 'categorical_vs_continuous':
                significant_count, min_p_value, max_effect_size, detailed_info = self._process_categorical_vs_continuous(
                    statistical_tests, significant_count, min_p_value, max_effect_size, detailed_info)
                    
            elif analysis_type == 'continuous_vs_categorical':
                significant_count, min_p_value, max_effect_size, detailed_info = self._process_continuous_vs_categorical(
                    statistical_tests, significant_count, min_p_value, max_effect_size, detailed_info)
                    
            elif analysis_type == 'categorical_vs_categorical':
                significant_count, min_p_value, max_effect_size, detailed_info = self._process_categorical_vs_categorical(
                    statistical_tests, significant_count, min_p_value, max_effect_size, detailed_info)
                    
            # 生成總結信息
            if significant_count > 0:
                basic_info.append(f'顯著檢定: {significant_count}項 ✓')
                basic_info.append('結論: 顯著影響')
            else:
                basic_info.append('顯著檢定: 0項')
                basic_info.append('結論: 無顯著影響')
                
            # 添加效應量信息
            if max_effect_size > 0:
                if max_effect_size > 0.5:
                    effect_desc = '大'
                elif max_effect_size > 0.3:
                    effect_desc = '中'
                else:
                    effect_desc = '小'
                basic_info.append(f'效應量: {effect_desc} ({max_effect_size:.3f})')
                
            # 合併信息
            info_lines = []
            info_lines.extend(basic_info)
            if basic_info and detailed_info:
                info_lines.append('---')
            info_lines.extend(detailed_info)
            
            # 顯示統計信息
            if info_lines:
                analysis_text = '\n'.join(info_lines)
                props_basic = dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.8)
                ax.text(0.98, 0.02, analysis_text, transform=ax.transAxes, 
                        fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                        bbox=props_basic, fontproperties=vs3.vs.MJHfontprop())
                        
            return True
            
        except Exception as e:
            LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
            return False
            
    def _process_continuous_vs_continuous(self, statistical_tests, significant_count, min_p_value, max_effect_size, detailed_info):
        """處理連續型對連續型的檢定結果"""
        # Pearson相關
        if 'pearson_correlation' in statistical_tests:
            pearson = statistical_tests['pearson_correlation']
            r_val = abs(pearson.get('correlation', 0))
            p_val = pearson.get('p_value', 1)
            is_sig = pearson.get('significant', False)
            
            if is_sig:
                significant_count += 1
            min_p_value = min(min_p_value, p_val)
            max_effect_size = max(max_effect_size, r_val)
            
            sig_mark = '**' if is_sig else ''
            detailed_info.append(f'Pearson r: {pearson.get("correlation", 0):.3f}{sig_mark}')
            detailed_info.append(f'p值: {p_val:.4f}')
            
        # Spearman相關
        if 'spearman_correlation' in statistical_tests:
            spearman = statistical_tests['spearman_correlation']
            r_val = abs(spearman.get('correlation', 0))
            p_val = spearman.get('p_value', 1)
            is_sig = spearman.get('significant', False)
            
            if is_sig:
                significant_count += 1
            min_p_value = min(min_p_value, p_val)
            max_effect_size = max(max_effect_size, r_val)
            
            sig_mark = '**' if is_sig else ''
            detailed_info.append(f'Spearman r: {spearman.get("correlation", 0):.3f}{sig_mark}')
            
        # Kendall's tau相關
        if 'kendall_tau' in statistical_tests:
            kendall = statistical_tests['kendall_tau']
            tau_val = abs(kendall.get('correlation', 0))
            p_val = kendall.get('p_value', 1)
            is_sig = kendall.get('significant', False)
            
            if is_sig:
                significant_count += 1
            min_p_value = min(min_p_value, p_val)
            max_effect_size = max(max_effect_size, tau_val)
            
            sig_mark = '**' if is_sig else ''
            detailed_info.append(f'Kendall τ: {kendall.get("correlation", 0):.3f}{sig_mark}')
            
        # 線性回歸
        if 'linear_regression' in statistical_tests:
            lr = statistical_tests['linear_regression']
            r2 = lr.get('r_squared', 0)
            p_val = lr.get('p_value', 1)
            is_sig = lr.get('significant', False)
            
            if is_sig:
                significant_count += 1
            min_p_value = min(min_p_value, p_val)
            max_effect_size = max(max_effect_size, r2)
            
            sig_mark = '**' if is_sig else ''
            detailed_info.append(f'R²: {r2:.3f}{sig_mark}')
            
        return significant_count, min_p_value, max_effect_size, detailed_info
        
    def _process_categorical_vs_continuous(self, statistical_tests, significant_count, min_p_value, max_effect_size, detailed_info):
        """處理分類型對連續型的檢定結果"""
        # t檢定
        if 't_test' in statistical_tests:
            t_test = statistical_tests['t_test']
            t_val = abs(t_test.get('statistic', 0))
            p_val = t_test.get('p_value', 1)
            is_sig = t_test.get('significant', False)
            
            if is_sig:
                significant_count += 1
            min_p_value = min(min_p_value, p_val)
            
            sig_mark = '**' if is_sig else ''
            detailed_info.append(f't檢定: {t_val:.3f}{sig_mark}')
            detailed_info.append(f'p值: {p_val:.4f}')
            
        # Mann-Whitney U檢定
        if 'mann_whitney_u' in statistical_tests:
            mw = statistical_tests['mann_whitney_u']
            u_val = mw.get('statistic', 0)
            p_val = mw.get('p_value', 1)
            is_sig = mw.get('significant', False)
            
            if is_sig:
                significant_count += 1
            min_p_value = min(min_p_value, p_val)
            
            sig_mark = '**' if is_sig else ''
            detailed_info.append(f'M-W U: {u_val:.0f}{sig_mark}')
            
        # Kolmogorov-Smirnov檢定
        if 'kolmogorov_smirnov' in statistical_tests:
            ks = statistical_tests['kolmogorov_smirnov']
            ks_val = ks.get('statistic', 0)
            p_val = ks.get('p_value', 1)
            is_sig = ks.get('significant', False)
            
            if is_sig:
                significant_count += 1
            min_p_value = min(min_p_value, p_val)
            
            sig_mark = '**' if is_sig else ''
            detailed_info.append(f'K-S: {ks_val:.3f}{sig_mark}')
            
        # ANOVA
        if 'anova' in statistical_tests:
            anova = statistical_tests['anova']
            f_val = anova.get('f_statistic', 0)
            p_val = anova.get('p_value', 1)
            is_sig = anova.get('significant', False)
            
            if is_sig:
                significant_count += 1
            min_p_value = min(min_p_value, p_val)
            
            sig_mark = '**' if is_sig else ''
            detailed_info.append(f'ANOVA F: {f_val:.3f}{sig_mark}')
            detailed_info.append(f'p值: {p_val:.4f}')
            
        # Kruskal-Wallis檢定
        if 'kruskal_wallis' in statistical_tests:
            kw = statistical_tests['kruskal_wallis']
            h_val = kw.get('statistic', 0)
            p_val = kw.get('p_value', 1)
            is_sig = kw.get('significant', False)
            
            if is_sig:
                significant_count += 1
            min_p_value = min(min_p_value, p_val)
            
            sig_mark = '**' if is_sig else ''
            detailed_info.append(f'K-W H: {h_val:.3f}{sig_mark}')
            
        return significant_count, min_p_value, max_effect_size, detailed_info
        
    def _process_continuous_vs_categorical(self, statistical_tests, significant_count, min_p_value, max_effect_size, detailed_info):
        """處理連續型對分類型的檢定結果"""
        # Point-biserial相關
        if 'point_biserial' in statistical_tests:
            pb = statistical_tests['point_biserial']
            r_val = abs(pb.get('correlation', 0))
            p_val = pb.get('p_value', 1)
            is_sig = pb.get('significant', False)
            
            if is_sig:
                significant_count += 1
            min_p_value = min(min_p_value, p_val)
            max_effect_size = max(max_effect_size, r_val)
            
            sig_mark = '**' if is_sig else ''
            detailed_info.append(f'Point-biserial r: {pb.get("correlation", 0):.3f}{sig_mark}')
            detailed_info.append(f'p值: {p_val:.4f}')
            
        # Spearman相關
        if 'spearman_correlation' in statistical_tests:
            spearman = statistical_tests['spearman_correlation']
            r_val = abs(spearman.get('correlation', 0))
            p_val = spearman.get('p_value', 1)
            is_sig = spearman.get('significant', False)
            
            if is_sig:
                significant_count += 1
            min_p_value = min(min_p_value, p_val)
            max_effect_size = max(max_effect_size, r_val)
            
            sig_mark = '**' if is_sig else ''
            detailed_info.append(f'Spearman r: {spearman.get("correlation", 0):.3f}{sig_mark}')
            
        return significant_count, min_p_value, max_effect_size, detailed_info
        
    def _process_categorical_vs_categorical(self, statistical_tests, significant_count, min_p_value, max_effect_size, detailed_info):
        """處理分類型對分類型的檢定結果"""
        # Chi-square檢定
        if 'chi_square' in statistical_tests:
            chi2 = statistical_tests['chi_square']
            chi2_val = chi2.get('chi2_statistic', 0)
            p_val = chi2.get('p_value', 1)
            is_sig = chi2.get('significant', False)
            
            if is_sig:
                significant_count += 1
            min_p_value = min(min_p_value, p_val)
            
            sig_mark = '**' if is_sig else ''
            detailed_info.append(f'χ²: {chi2_val:.3f}{sig_mark}')
            detailed_info.append(f'p值: {p_val:.4f}')
            
        # Fisher精確檢定
        if 'fisher_exact' in statistical_tests:
            fisher = statistical_tests['fisher_exact']
            odds_ratio = fisher.get('odds_ratio', 1)
            p_val = fisher.get('p_value', 1)
            is_sig = fisher.get('significant', False)
            
            if is_sig:
                significant_count += 1
            min_p_value = min(min_p_value, p_val)
            
            sig_mark = '**' if is_sig else ''
            detailed_info.append(f'Fisher: OR={odds_ratio:.3f}{sig_mark}')
            detailed_info.append(f'p值: {p_val:.4f}')
            
        # Cramér's V
        if 'cramers_v' in statistical_tests:
            cv = statistical_tests['cramers_v']
            v_val = cv.get('value', 0)
            max_effect_size = max(max_effect_size, v_val)
            detailed_info.append(f"Cramér's V: {v_val:.3f}")
            
        return significant_count, min_p_value, max_effect_size, detailed_info

class StatisticalAnalysisManager:
    """
    統計分析管理類別
    整合統計檢定和圖表標註功能
    """
    
    def __init__(self, alpha=0.05):
        self.test_manager = StatisticalTestManager(alpha=alpha)
        self.chart_manager = ChartAnnotationManager()
        
    def perform_analysis(self, source_data, xheader, yheader, x_type, y_type):
        """執行完整的統計分析"""
        # 清理資料
        clean_data = source_data[[xheader, yheader]].dropna()
        
        if len(clean_data) < 3:
            return {'error': '資料點數太少，無法進行統計檢定'}
            
        x_data = clean_data[xheader]
        y_data = clean_data[yheader]
        
        # 執行統計檢定
        results = self.test_manager.perform_tests(x_data, y_data, x_type, y_type)
        
        return results
        
    def add_statistical_info_to_plot(self, ax, statistical_tests, xheader, yheader):
        """在圖表上添加統計檢定信息"""
        return self.chart_manager.add_statistical_info_to_plot(ax, statistical_tests, xheader, yheader)
        
    def generate_report(self, statistical_tests, xheader, yheader, PlanID=None, SeqNo=None):
        """生成統計報告"""
        try:
            report_lines = []
            report_lines.append(f"# {xheader} 對 {yheader} 的統計分析報告")
            report_lines.append("")
            report_lines.append(f"**分析時間**: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            if PlanID is not None and SeqNo is not None:
                report_lines.append(f"**計畫ID**: {PlanID}")
                report_lines.append(f"**序號**: {SeqNo}")
                
            report_lines.append(f"**分析類型**: {statistical_tests.get('analysis_type', 'Unknown')}")
            report_lines.append(f"**樣本數**: {statistical_tests.get('sample_size', 0)}")
            report_lines.append("")
            
            # 統計檢定結果
            report_lines.append("## 統計檢定結果")
            report_lines.append("")
            
            for test_name, test_result in statistical_tests.items():
                if test_name in ['analysis_type', 'sample_size', 'summary', 'group_info']:
                    continue
                    
                if isinstance(test_result, dict) and 'error' not in test_result:
                    report_lines.append(f"### {test_name.replace('_', ' ').title()}")
                    
                    if 'statistic' in test_result:
                        report_lines.append(f"- 統計量: {test_result['statistic']:.4f}")
                    if 'correlation' in test_result:
                        report_lines.append(f"- 相關係數: {test_result['correlation']:.4f}")
                    if 'p_value' in test_result:
                        report_lines.append(f"- p值: {test_result['p_value']:.4f}")
                    if 'significant' in test_result:
                        significance = "顯著" if test_result['significant'] else "不顯著"
                        report_lines.append(f"- 顯著性: {significance} (α = 0.05)")
                    if 'interpretation' in test_result:
                        report_lines.append(f"- 解釋: {test_result['interpretation']}")
                        
                    report_lines.append("")
                    
            # 總結
            if 'summary' in statistical_tests:
                summary = statistical_tests['summary']
                report_lines.append("## 總結")
                report_lines.append("")
                report_lines.append(f"**結論**: {summary['conclusion']}")
                report_lines.append(f"**顯著檢定數量**: {summary['n_significant']}")
                
                if summary['significant_tests']:
                    report_lines.append(f"**顯著檢定項目**: {', '.join(summary['significant_tests'])}")
                    
                report_lines.append("")
                
            return "\n".join(report_lines)
            
        except Exception as e:
            LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
            return f"生成報告時發生錯誤: {str(e)}"

class TwoDimensionStatisticalTestManager:
    """二維統計檢定管理器"""
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha
    
    def perform_two_dimension_tests(self, source_data, xheader1, xheader2, yheader, x1_type, x2_type, y_type):
        """執行二維統計檢定分析"""
        try:
            from scipy import stats
            import numpy as np
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            from sklearn.preprocessing import LabelEncoder
            
            # 清理資料，移除NaN值
            clean_data = source_data[[xheader1, xheader2, yheader]].dropna()
            
            if len(clean_data) < 5:
                return {'error': '資料點數太少，無法進行二維統計檢定'}
            
            x1_data = clean_data[xheader1]
            x2_data = clean_data[xheader2]
            y_data = clean_data[yheader]
            
            results = {
                'sample_size': len(clean_data),
                'analysis_type': f'{x1_type}_{x2_type}_to_{y_type}'
            }
            
            # 個別相關性分析
            if x1_type == 'continuous' and y_type == 'continuous':
                try:
                    r1, p1 = stats.pearsonr(x1_data, y_data)
                    results['x1_correlation'] = {
                        'correlation': float(r1),
                        'p_value': float(p1),
                        'significant': bool(p1 < self.alpha),
                        'interpretation': 'strong' if abs(r1) > 0.7 else 'moderate' if abs(r1) > 0.3 else 'weak',
                        'explanation': 'Pearson 線性相關；p<0.05 表示 x1 與 y 線性相關顯著'
                    }
                except Exception as e:
                    results['x1_correlation'] = {'error': str(e)}
            
            if x2_type == 'continuous' and y_type == 'continuous':
                try:
                    r2, p2 = stats.pearsonr(x2_data, y_data)
                    results['x2_correlation'] = {
                        'correlation': float(r2),
                        'p_value': float(p2),
                        'significant': bool(p2 < self.alpha),
                        'interpretation': 'strong' if abs(r2) > 0.7 else 'moderate' if abs(r2) > 0.3 else 'weak',
                        'explanation': 'Pearson 線性相關；p<0.05 表示 x2 與 y 線性相關顯著'
                    }
                except Exception as e:
                    results['x2_correlation'] = {'error': str(e)}
            
            # X1和X2之間的相關性（多重共線性檢查）
            if x1_type == 'continuous' and x2_type == 'continuous':
                try:
                    r12, p12 = stats.pearsonr(x1_data, x2_data)
                    results['x1_x2_correlation'] = {
                        'correlation': float(r12),
                        'p_value': float(p12),
                        'significant': bool(p12 < self.alpha),
                        'multicollinearity_warning': bool(abs(r12) > 0.8),
                        'interpretation': 'high multicollinearity' if abs(r12) > 0.8 else 'acceptable',
                        'explanation': '檢查自變數間相關；高相關（|r|>0.8）提示多重共線性風險'
                    }
                except Exception as e:
                    results['x1_x2_correlation'] = {'error': str(e)}
            
            # 多元回歸分析（如果Y是連續型）
            if y_type == 'continuous':
                try:
                    # 準備X矩陣
                    X_matrix = []
                    if x1_type == 'continuous':
                        X_matrix.append(x1_data.values)
                    else:
                        # 類別型變數需要編碼
                        le = LabelEncoder()
                        X_matrix.append(le.fit_transform(x1_data.values))
                    
                    if x2_type == 'continuous':
                        X_matrix.append(x2_data.values)
                    else:
                        le = LabelEncoder()
                        X_matrix.append(le.fit_transform(x2_data.values))
                    
                    X_matrix = np.column_stack(X_matrix)
                    
                    # 多元線性回歸
                    reg = LinearRegression()
                    reg.fit(X_matrix, y_data)
                    y_pred = reg.predict(X_matrix)
                    
                    r2 = r2_score(y_data, y_pred)
                    
                    results['multiple_regression'] = {
                        'r_squared': float(r2),
                        'coefficients': [float(c) for c in reg.coef_],
                        'intercept': float(reg.intercept_),
                        'feature_names': [xheader1, xheader2],
                        'explanation': '多元線性回歸；p<0.05 表示至少一個自變數對 y 的線性影響顯著，R² 為整體解釋力'
                    }
                    
                    # F檢定
                    n = len(y_data)
                    k = 2  # 兩個預測變數
                    f_stat = (r2 / k) / ((1 - r2) / (n - k - 1))
                    f_p_value = 1 - stats.f.cdf(f_stat, k, n - k - 1)
                    
                    results['f_test'] = {
                        'f_statistic': float(f_stat),
                        'p_value': float(f_p_value),
                        'significant': bool(f_p_value < self.alpha),
                        'interpretation': '模型整體顯著' if f_p_value < self.alpha else '模型整體不顯著',
                        'explanation': '整體 F 檢定：p<0.05 表示模型整體顯著'
                    }
                    
                except Exception as e:
                    results['multiple_regression'] = {'error': str(e)}
            
            return results
            
        except Exception as e:
            LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
            return {'error': f'二維統計檢定錯誤: {str(e)}'}

class TwoDimensionInteractionAnalyzer:
    """二維交互作用分析器"""
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha
    
    def analyze_interaction(self, source_data, xheader1, xheader2, yheader, x1_type, x2_type, y_type, multiple_regression_r2=None):
        """分析交互作用效應"""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            import numpy as np
            
            # 清理資料
            clean_data = source_data[[xheader1, xheader2, yheader]].dropna()
            
            if len(clean_data) < 10:  # 交互作用分析需要更多資料點
                return {'error': '資料點數太少，無法進行交互作用分析'}
            
            results = {}
            
            # 只對連續型變數進行交互作用分析
            if x1_type == 'continuous' and x2_type == 'continuous' and y_type == 'continuous':
                try:
                    x1_data = clean_data[xheader1]
                    x2_data = clean_data[xheader2]
                    y_data = clean_data[yheader]
                    
                    # 計算交互作用項
                    interaction_term = x1_data * x2_data
                    
                    # 包含交互作用的回歸
                    X_with_interaction = np.column_stack([x1_data, x2_data, interaction_term])
                    
                    reg_interaction = LinearRegression()
                    reg_interaction.fit(X_with_interaction, y_data)
                    
                    y_pred_interaction = reg_interaction.predict(X_with_interaction)
                    r2_interaction = r2_score(y_data, y_pred_interaction)
                    
                    # 比較有無交互作用的模型
                    r2_without_interaction = multiple_regression_r2 if multiple_regression_r2 is not None else 0
                    interaction_improvement = r2_interaction - r2_without_interaction
                    
                    results = {
                        'r_squared_with_interaction': float(r2_interaction),
                        'r_squared_without_interaction': float(r2_without_interaction),
                        'interaction_improvement': float(interaction_improvement),
                        'interaction_coefficient': float(reg_interaction.coef_[2]),
                        'interaction_significant': bool(interaction_improvement > 0.01),  # 改善超過1%認為有意義
                        'interpretation': '存在顯著交互作用' if interaction_improvement > 0.01 else '交互作用不顯著'
                    }
                    
                except Exception as e:
                    results = {'error': str(e)}
            else:
                results = {'note': '僅支援連續型變數的交互作用分析'}
            
            return results
            
        except Exception as e:
            LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
            return {'error': f'交互作用分析錯誤: {str(e)}'}

class TwoDimensionSurfaceAnalyzer:
    """二維響應面分析器"""
    
    def __init__(self):
        pass
    
    def analyze_surface(self, source_data, xheader1, xheader2, yheader, x1_type, x2_type, y_type):
        """分析響應面特性"""
        try:
            import numpy as np
            
            # 清理資料
            clean_data = source_data[[xheader1, xheader2, yheader]].dropna()
            
            if len(clean_data) < 5:
                return {'error': '資料點數太少，無法進行響應面分析'}
            
            results = {}
            
            # 只對連續型變數進行響應面分析
            if x1_type == 'continuous' and x2_type == 'continuous' and y_type == 'continuous':
                try:
                    x1_data = clean_data[xheader1]
                    x2_data = clean_data[xheader2]
                    y_data = clean_data[yheader]
                    
                    # 尋找極值點
                    y_max_idx = np.argmax(y_data)
                    y_min_idx = np.argmin(y_data)
                    
                    results = {
                        'max_point': {
                            'x1': float(x1_data.iloc[y_max_idx]),
                            'x2': float(x2_data.iloc[y_max_idx]),
                            'y': float(y_data.iloc[y_max_idx])
                        },
                        'min_point': {
                            'x1': float(x1_data.iloc[y_min_idx]),
                            'x2': float(x2_data.iloc[y_min_idx]),
                            'y': float(y_data.iloc[y_min_idx])
                        },
                        'y_range': float(y_data.max() - y_data.min()),
                        'surface_roughness': float(np.std(y_data)),  # 響應面的粗糙度
                        'x1_range': float(x1_data.max() - x1_data.min()),
                        'x2_range': float(x2_data.max() - x2_data.min())
                    }
                    
                except Exception as e:
                    results = {'error': str(e)}
            else:
                results = {'note': '僅支援連續型變數的響應面分析'}
            
            return results
            
        except Exception as e:
            LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
            return {'error': f'響應面分析錯誤: {str(e)}'}

class TwoDimensionAnalysisManager:
    """二維分析整合管理器"""
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.test_manager = TwoDimensionStatisticalTestManager(alpha)
        self.interaction_analyzer = TwoDimensionInteractionAnalyzer(alpha)
        self.surface_analyzer = TwoDimensionSurfaceAnalyzer()
    
    def perform_comprehensive_analysis(self, source_data, xheader1, xheader2, yheader, x1_type, x2_type, y_type):
        """執行完整的二維分析"""
        try:
            # 統計檢定分析
            statistical_tests = self.test_manager.perform_two_dimension_tests(
                source_data, xheader1, xheader2, yheader, x1_type, x2_type, y_type
            )
            
            # 交互作用分析
            multiple_regression_r2 = None
            if 'multiple_regression' in statistical_tests and 'r_squared' in statistical_tests['multiple_regression']:
                multiple_regression_r2 = statistical_tests['multiple_regression']['r_squared']
            
            interaction_analysis = self.interaction_analyzer.analyze_interaction(
                source_data, xheader1, xheader2, yheader, x1_type, x2_type, y_type, multiple_regression_r2
            )
            
            # 響應面分析
            surface_analysis = self.surface_analyzer.analyze_surface(
                source_data, xheader1, xheader2, yheader, x1_type, x2_type, y_type
            )
            
            # 整體顯著性總結
            significant_tests = []
            for test_name, test_result in statistical_tests.items():
                if isinstance(test_result, dict) and 'significant' in test_result:
                    if test_result['significant']:
                        significant_tests.append(test_name)
            
            statistical_tests['summary'] = {
                'significant_tests': significant_tests,
                'n_significant': len(significant_tests),
                'overall_significant': bool(len(significant_tests) > 0),
                'conclusion': generate_conclusion_with_effect(
                    [xheader1, xheader2], yheader,
                    len(significant_tests),
                    multiple_regression_r2
                )
            }
            
            return {
                'statistical_tests': statistical_tests,
                'interaction_analysis': interaction_analysis,
                'surface_analysis': surface_analysis
            }
            
        except Exception as e:
            LOGger.exception_process(e, logfile=os.path.join('log', 'log_%t.txt'))
            return {'error': f'二維分析錯誤: {str(e)}'}

#%%    
if(__name__=='__main__' ):
    scenario()
