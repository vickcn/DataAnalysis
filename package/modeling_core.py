# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 16:31:18 2021

@author: Ian.Ko
含浸廢品<0.5
廢品最少情況下，含浸速度、擺盪、壓桿刻度、張力、間隙等搭配建議值(建議從G機台開始) 
"""
#import data_collecting as clt
#import Config_parse as con
import abc
import sys
import threading
import dill
import time
#sys.path.append('..')
import os
from package import measureVariance as MV
from package import kerasExplainer as kE
from package import visualization as vs
from package import visualization2 as vs2
from package import visualization3 as vs3
from package import dataframeprocedure as DFP
from package import LOGger
from package.LOGger import CreateContainer, CreateFile, addlog, addloger, show_vector
from package.LOGger import stamp_process, exception_process, for_file_process, abspath, mylist, mystr
from package.LOGger import load_json, save_json, flattern_list, type_string
import pandas as pd
import joblib
from copy import deepcopy as dcp
from scipy.optimize import minimize
from sklearn import svm 
from sklearn import ensemble as ske
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder as LBC
from sklearn.preprocessing import OneHotEncoder as OHC
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from datetime import datetime as dt
from scipy.stats import pointbiserialr
from sklearn.preprocessing import StandardScaler as Stdscr
from sklearn.preprocessing import PowerTransformer as Pwrscr
from sklearn.preprocessing import MinMaxScaler as Mnxscr
from sklearn.model_selection import GridSearchCV
from sklearn import metrics as skm
from sklearn.tree import plot_tree
import keras
from pkg_resources import parse_version
# Keras imports with version check
if parse_version(keras.__version__) >= parse_version('3.0.0'):
    # Keras 3.x
    try:
        from keras import models
        from keras import callbacks
        from keras.callbacks import EarlyStopping
    except ImportError:
        from tensorflow.keras import models
        from tensorflow.keras import callbacks
        from tensorflow.keras.callbacks import EarlyStopping
else:
    # Keras 2.x
    from keras import models
    from keras import callbacks
    from keras.callbacks import EarlyStopping
if parse_version(keras.__version__) >= parse_version('3.0.0'):
    # Keras 3.x
    try:
        from keras.layers import Layer
        from keras.models import Model as Functional
    except ImportError:
        from tensorflow.keras.models import Model as Functional
elif parse_version(keras.__version__) > parse_version('2.8'):
    # Keras 2.8+
    try:
        from keras.engine.functional import Functional
    except ImportError:
        from tensorflow.keras.engine.functional import Functional
else:
    # Keras 2.8 or earlier
    from tensorflow.keras.models import Model as Functional
# Backend and model loading
if parse_version(keras.__version__) >= parse_version('3.0.0'):
    try:
        from keras import backend as bcd
        from keras.saving import load_model, save_model
        from keras.saving import serialize_keras_object, deserialize_keras_object
    except ImportError:
        from tensorflow.keras import backend as bcd
        from tensorflow.keras.models import load_model, save_model
        from tensorflow.keras.models import model_from_json
else:
    from keras import backend as bcd
    from keras.models import model_from_json
if parse_version(keras.__version__) >= parse_version('3.0.0'):
    # Keras 3.x
    try:
        from keras.utils import plot_model
    except ImportError:
        from tensorflow.keras.utils import plot_model
else:
    # Keras 2.x
    try:
        from keras.utils.vis_utils import plot_model
    except ImportError:
        from tensorflow.keras.utils import plot_model
import pickle
from package import algorithms as ALG
from sklearn.metrics import cohen_kappa_score as cohkpa
# Keras activations
if parse_version(keras.__version__) >= parse_version('3.0.0'):
    try:
        from keras import activations
        from keras.activations import relu, elu
    except ImportError:
        from tensorflow.keras import activations
        from tensorflow.keras.activations import relu, elu
else:
    from keras import activations
    from keras.activations import relu, elu
if parse_version(keras.__version__) >= parse_version('3.0.0'):
    # Keras 3.x
    try:
        from keras.utils import get_custom_objects
    except ImportError:
        from tensorflow.keras.utils import get_custom_objects
else:
    # Keras 2.x
    try:
        from keras.utils.generic_utils import get_custom_objects
    except ImportError:
        from tensorflow.keras.utils import get_custom_objects
import focal_loss
from focal_loss import BinaryFocalLoss
import platform as  plf
if(plf.system()=='Linux'):
    try:
        import autosklearn as ask
        from autosklearn.classification import AutoSklearnClassifier
        from autosklearn.regression import AutoSklearnRegressor
    except:
        ask = None
        LOGger.addDebug('No autosklearn!!!')
padCropPadEdge = DFP.padCropPadEdge

threshold_variation_analysis_lock = threading.Lock()
CreateContainer('log')
logfile = 'log\\log_%s.txt'%(dt.now().strftime('%Y%m%d'))
logshield = False
#%%
m_print = LOGger.addloger(logfile='')
m_processKeywords = {}
m_debug = LOGger.myDebuger(stamps=[*os.path.basename(__file__).split('.')[:-1]]) # 'debug.pkl'
m_lossColors = LOGger.mylist([tuple(np.array([12,47,71])/255), (0.2,0.1,0.8,0.5), (0.05,0.05,0.3)])
parseKL = ALG.layer_producer
m_noise_values = [np.nan, None, '', 'nan']
m_coreNameHeader = 'core_name'
def method_activation(stg):
    try:
        method = eval(stg)
        return method
    except:
        print('method invalid:%s!!!!'%stg)
        return None

def preprocessingDefault(data_prop):
    if(data_prop=='categorical'):
        return 'oneHotEncoding'
    elif(data_prop=='binary'):
        return 'encoding'
    return 'normalization'

def lossMethodDefault(data_prop):
    if(data_prop=='categorical'):
        return 'CategoricalCrossentropy' # categorical_crossentropy
    elif(data_prop=='binary'):
        return 'BinaryCrossentropy' # binary_crossentropy
    return 'mse'

def activatorDefault(data_prop):
    if(data_prop=='categorical'):
        return 'softmax'
    elif(data_prop=='binary'):
        return 'sigmoid'
    return 'linear'
    
# def transform_byDataProp(inputs, data_prop, **kwags):
#     if(DFP.isiterable(inputs, exceptions=[str, dict, pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray])):
#         # 應該是資料列的形式，逐項處理
#         inputs_type = type(inputs)
#         return inputs_type(map((lambda x:transform_byDataProp(x,data_prop)), inputs))
#     elif(isinstance(inputs, np.ndarray)):
#         # 應該是資料列的形式，逐項處理
#         if(len(inputs.shape)==1):
#             return np.array(map((lambda x:transform_byDataProp(x,data_prop)), inputs))
#     if(data_prop=='img'):
#         return vs2.cv2imread(inputs, img_tensor_dim=kwags.get('img_tensor_dim',3), 
#                              dont_log_img_shape_error=kwags.get('dont_log_img_shape_error',True))
#     return inputs

#TODO:activateHeaderZoneFromData
def activateHeaderZoneFromData(dic, reference_data, mdc=None, all_categories=None, default_category_value=0, ret=None, **kwags):
    """
    根據source data來經活header

    Parameters
    ----------
    dic : TYPE
        DESCRIPTION.
    reference_data : TYPE
        DESCRIPTION.
    mdc : TYPE, optional
        DESCRIPTION. The default is None.
    all_categories : TYPE, optional
        DESCRIPTION. The default is None.
    default_category_value : TYPE, optional
        DESCRIPTION. The default is 0.
    ret : TYPE, optional
        DESCRIPTION. The default is None.
    **kwags : TYPE
        DESCRIPTION.

    Returns
    -------
    bool
        DESCRIPTION.

    """
    ret = ret if(isinstance(ret, dict)) else {}
    if(not 'data_prop' in dic):
        if(isinstance(all_categories, type(None))):
            default_all_categories = reference_data[[v for v in dic['core'] if v in reference_data]].copy()
            default_all_categories = pd.DataFrame(index=reference_data.index) if(default_all_categories.empty) else default_all_categories
            for col in [v for v in dic['core'] if not v in reference_data.columns]:
                default_all_categories[col] = default_category_value
        elif(isinstance(all_categories, str)):
            default_all_categories = eval(all_categories)
        elif(not DFP.isiterable(all_categories)):
            all_categories = None
        diversity = DFP.uniqueByIndex(tuple(zip(*default_all_categories.T.values)),axis=0).shape[0]
        dic['diversity'] = diversity
        if(default_all_categories.applymap(lambda s:not isinstance(DFP.astype(s), float)).any().any()):
            dic['data_prop'] = 'categorical' if(diversity>2) else 'binary'
        else:
            dic['data_prop'] = 'continuous' if(diversity>2) else 'binary'
        dic['all_categories'] = [DFP.parse(default_all_categories.values[0]), '_unknown_'] if(
            diversity<2 and not dic.get('all_categories', None)) else None
    ret['hz'] = activateZone(dic, **kwags)
    return True

def preprocessorClassMapping(stg, headerZone, score_threshold=None, **kwags):
    if(stg == 'normalization'):
        return myStdscr()
    elif(stg == 'encoding'):
        if(len(headerZone)==1):
            return myLBC(score_thrheshold=score_threshold)
        elif(len(headerZone)>1):
            return MV.HLBC()
        else:
            LOGger.addlog('headerZone is invalid when %s!!!'%stg, headerZone, colora=LOGger.FAIL)
            sys.exit(1)
    elif(stg == 'powertransform'):
        return myPwrscr(method=getattr(stg,'method','yeo-johnson'))
    elif(stg == 'basepowertransform'):
        return myBasePwrscr(method=getattr(stg,'method','yeo-johnson'), baseAxis=getattr(stg,'baseAxis',0))
    else:
        LOGger.addlog('%s is an undefined preprocessing!!!'%stg, colora=LOGger.FAIL)
        sys.exit(1)

def AnnilGroupStrucuteIfNecessary(data, cell_size=None, stamps=None, **kwags):
    if(LOGger.mylist(data.shape).get(1,-1)==1):
        stamps = stamps if(isinstance(stamps, list)) else []
        if(isinstance(data, np.ndarray)):
            if(len(data.shape)!=2+len(cell_size)):
                m_print('flattenCore at data with cell_size failed!!!', 'data.shape', str(data.shape), 'self.cell_size', str(cell_size), 
                        colora=LOGger.FAIL, stamps=stamps)
                return None
        elif(isinstance(data, pd.core.frame.DataFrame)):
            if(DFP.transformByBatch(data, (lambda x:not DFP.isiterable(x))).any()):
                m_print('flattenCore at data with cell_size failed!!! Data structure invalid of type 1!!!', colora=LOGger.FAIL, stamps=stamps)
                return None
            if(DFP.transformByBatch(data, (lambda x:getattr(x[0],'shape',())!=tuple(cell_size))).any()):
                m_print('flattenCore at data with cell_size failed!!! Data structure invalid of type 2!!!', colora=LOGger.FAIL, stamps=stamps)
                return None
        else:
            m_print('flattenCore at data with cell_size failed!!! structure invalid!!!', colora=LOGger.FAIL, stamps=stamps)
            return None
        data = DFP.transformByBatch(getattr(data,'values',data), lambda x:x[0])
    return data

class HEADER_ZONE(LOGger.mylist):
    def __init__(self, core, data_prop='continuous', preprocessing='', preprocessor_file=None, preprocessor_dir='.', 
                 cell_size=None, feaVarAxis=None, reshapeThruMethod='reshapeThruFlatten', activation=None, pad_value=0, exp_fd='.',
                 all_categories=None, stamps=None, hidden_layer_sizes=None, hidden_layer_nns=None, 
                 score_threshold=None, default_value=np.nan, layer_type='Dense', 
                 strideses=None, kernel_sizes=None, maxpool2D_sizes=None, **kwags):
        super().__init__(core)
        self.stamps = stamps if(isinstance(stamps, list)) else []
        self.data_prop = LOGger.mystr(data_prop)
        self.preprocessor = None
        self.referenceData = None
        self.preprocessedData = None
        self.cellColumns = None
        self.preprocessor_file = preprocessor_file
        self.preprocessor_dir = preprocessor_dir
        self.preprocessing = LOGger.transform_dict2class(preprocessing, admit_key_names=[]) if(isinstance(preprocessing, dict)) else (
            preprocessing if(LOGger.isinstance_not_empty(preprocessing, str)) else preprocessingDefault(self.data_prop))
        self.cell_size = cell_size
        self.feaVarAxis = feaVarAxis
        self.activation = LOGger.transform_dict2class(activation, admit_key_names=[]) if(isinstance(activation, dict)) else (
            activation if(activation!='') else activatorDefault(self.data_prop))
        self.pad_value = pad_value
        self.default_value = default_value
        # self.replaced_values = LOGger.mylist(m_noise_values if(not isinstance(replaced_values, list)) else replaced_values)
        self.all_categories = None
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_layer_nns = hidden_layer_nns
        self.layer_type = layer_type
        self.strideses = strideses
        self.kernel_sizes = kernel_sizes
        self.maxpool2D_sizes = maxpool2D_sizes
        self.mainAttrs = ['data_prop','preprocessing','preprocessor_file','all_categories','cell_size','feaVarAxis',
                          'activation','hidden_layer_sizes','pad_value','default_value','hidden_layer_nns','layer_type',
                          'strideses','kernel_sizes','maxpool2D_sizes','clipConfig','cellColumns']
        self.exp_fd = exp_fd
        self.fn = LOGger.stamp_process('',stamps,'','','','_',for_file=1)
        self.score_threshold = score_threshold
        self.mv = correspondingMeasurament(self.data_prop)(exp_fd=exp_fd, data_prop=self.data_prop, stamps=self.stamps, 
                                                           score_threshold=self.score_threshold, **kwags)
        self.figsize = getattr(self.mv,'figsize',(10,10))
        self.preprocessorClass = preprocessorClassMapping(self.preprocessing, self, 
                                                          score_threshold=self.mv.mainAttrs.get('score_threshold'), 
                                                          defaultClass=self.mv.mainAttrs.get('defaultClass'), 
                                                          defaultClassIndex=self.mv.mainAttrs.get('defaultClassIndex'), 
                                                          **kwags)
        if(isinstance(self.preprocessor_file, str)):
            self.preprocessor = joblib.load(os.path.join(os.path.join(self.preprocessor_dir, self.preprocessor_file)))

        self.clipConfig = None
        
    def serializeZone(self):
        return serializeZone(self)
    
    def deserializeZone(self, dic=None, config=None, **kwags):
        return deserializeZone(self, dic, config=config, **kwags)
    
    def flattenCore(self, data, **kwags):
        if(self.cell_size is not None): data = AnnilGroupStrucuteIfNecessary(data, cell_size=self.cell_size, stamps=self.stamps)
        data = getattr(data, 'values', data)
        data = DFP.transformCellsToFlatBatch(data, cell_size=self.cell_size, 
                                             editor=getattr(self,'editor',None), 
                                             pad_value=self.pad_value, 
                                             feaVarAxis=self.feaVarAxis,
                                             **kwags)
        return data
    
    def transform(self, data, **kwags):
        data = transformCells(self, data, **kwags)
        return data
    
    def inverse_transform(self, data, **kwags):
        data = inverse_transformCells(self, data, **kwags)
        return data
    
    def inverse_flattenCore(self, data, **kwags):
        data = DFP.transformFlatBatchToCells(data, cell_size=self.cell_size, 
                                             editor=getattr(self,'editor',None), 
                                             pad_value=self.pad_value, 
                                             feaVarAxis=self.feaVarAxis,
                                             **kwags)
        # if(DFP.isiterable(self.cell_size) and len(data.shape)==1+len(self.cell_size)):
        #     data = np.expand_dims(data, axis=1)
        return data
    
    def fit(self, referenceData, **kwags):
        if(callable(getattr(self,'editor',None))):  referenceData = DFP.transformByBatch(referenceData, self.editor)
        data = self.flattenCore(referenceData)
        return fitPreprocessingStandard(self, referenceData=data, **kwags)
    
    def resetPreprocessor(self, referenceData, **kwags):
        attrs = {}
        attrs.update({'classes_': kwags.get('classes_', getattr(self, 'all_categories'))}) if(isinstance(self.preprocessorClass, LBC)) else None
        self.preprocessor = self.preprocessorClass.fit(referenceData, **attrs)
        return True
    
    def generate_preprocessor_file(self, exp_fd=None, file=None):
        return generate_preprocessor_file(self, exp_fd=None, file=file)
    
    def exportPreprocessor(self, exp_fd=None, file=None):
        if(not exportPreprocessor(self, exp_fd, file)):
            return False
        return True
        
class HEADER_ZONE_INPUT(HEADER_ZONE):
    def __init__(self, core, data_prop='continuous', preprocessing='', preprocessor_file=None, cell_size=None, activation=None, 
                 pad_value=0, all_categories=None, stamps=None, **kwags):
        super().__init__(core, data_prop=data_prop, preprocessing=preprocessing, preprocessor_file=preprocessor_file, cell_size=cell_size, 
                         activation=activation, pad_value=pad_value, all_categories=all_categories, stamps=stamps, **kwags)
        
class HEADER_ZONE_INPUT_FILE(HEADER_ZONE_INPUT): 
    def __init__(self, core, source_fd='.', **kwags):
        self.source_fd = source_fd
        self.fileData = pd.DataFrame()
        super().__init__(core, **kwags)
        self.mainAttrs = self.mainAttrs + ['source_fd']
    
    def editor(self, aData, **kwags):
        # 如果是str path like 就要載入檔案；如果不是，就直接通過
        addlog = LOGger.addloger(logfile='')
        if(LOGger.isinstance_not_empty(aData,str)):
            fullpath = os.path.join(self.source_fd, aData)
            if(os.path.exists(fullpath)):
                if(LOGger.mylist(fullpath.split('.')).get(-1) in ('jpg', 'png', 'bmp')):
                    aData = vs3.cv2imread(fullpath, dont_log_img_shape_error=False)
                else:
                    aData = DFP.import_data(fullpath)
            else:
                addlog("source fullpath doesn't exist: %s"%fullpath, colora=LOGger.Fore.RED)
                sys.exit(1)
        return aData
        
    # def file2img(self, data, crop_method=None, padding_method=None, **kwags):
    #     return file2img(self, data, crop_method=crop_method, padding_method=padding_method, **kwags)
    
    # def transform(self, data, **kwags):
    #     data = self.file2img(data)
    #     new_data = self.preprocessor.transform(data) if(hasattr(self.preprocessor,'transform')) else data
    #     return new_data
    
    # def fit(self, preprocessorClass, referenceData, crop_method=None, padding_method=None, **kwags):
    #     self.fileData = pd.DataFrame(np.array(referenceData), 
    #                                  columns=getattr(referenceData,'columns',np.arange(referenceData.shape[1])),
    #                                  index=getattr(referenceData,'index',np.arange(referenceData.shape[0])))
    #     self.file2img(referenceData, crop_method=crop_method, padding_method=padding_method)
    #     self.preprocessor = preprocessorClass().fit(referenceData)

def correspondingMeasurament(stg):
    if(stg in ['binaray','categorical']):
        return MV.discreteMeasurament
    else:
        return getattr(MV, '%sMeasurament'%stg)

class HEADER_ZONE_OUTPUT(HEADER_ZONE):
    def __init__(self, core, data_prop='continuous', preprocessing='', preprocessor_file=None, cell_size=None, activation=None, 
                 pad_value=0, all_categories=None, stamps=None, lossmethod=None, **kwags):
        super().__init__(core, data_prop=data_prop, preprocessing=preprocessing, preprocessor_file=preprocessor_file, cell_size=cell_size, 
                         activation=activation, pad_value=pad_value, all_categories=all_categories, stamps=stamps, **kwags)
        self.lossmethod = LOGger.transform_dict2class(lossmethod, admit_key_names=[]) if(isinstance(lossmethod, dict)) else (
            lossmethod if(lossmethod!='') else lossMethodDefault(self.data_prop))
        self.mainAttrs = self.mainAttrs + ['lossmethod','default_unfam']
        # self.exp_fd = exp_fd
        # self.figsize = MV.determineFigsize4HeaderZone(data_prop) if(figsize is None) else figsize
        if(not isinstance(getattr(self.data_prop,'export', None), dict)):   self.data_prop.export = self.mv.export()
        
    def evaluation(self, outcome, prediction, **kwags):
        kwags['exp_fd'] = kwags.get('exp_fd', os.path.join(self.exp_fd, 'graph'))
        kwags['exp_fd'] = kwags.get('exp_fd', os.path.join(self.exp_fd, 'graph'))
        # LOGger.addDebug('exp_fd', str(kwags.get('exp_fd','?')))
        if(not dataMeasureVariance4HZ(self, outcome, prediction, **kwags)):
            return False
        return True
    
    def evaluationAndPlot(self, outcome, prediction, **kwags):
        if(not self.evaluation(outcome, prediction, **kwags)):
            return False
        savePlotKwags = {k:v for k,v in kwags.items() if k in ['exp_fd','stamps']}
        self.mv.savePlot(**savePlotKwags)
        return True
    
class HEADER_ZONE_AUTOENCODER(HEADER_ZONE_OUTPUT):
    def __init__(self, core, data_prop='continuous', preprocessing='', preprocessor_file=None, cell_size=None, activation=None, 
                 pad_value=0, all_categories=None, stamps=None, lossmethod=None, layer_type='Conv1D', 
                 strideses=None, kernel_sizes=None, maxpool2D_sizes=None, **kwags):
        super().__init__(core, data_prop=data_prop, preprocessing=preprocessing, preprocessor_file=preprocessor_file, cell_size=cell_size, 
                         activation=activation, pad_value=pad_value, all_categories=all_categories, stamps=stamps, lossmethod=lossmethod, 
                         layer_type=layer_type, strideses=strideses, kernel_sizes=kernel_sizes, maxpool2D_sizes=maxpool2D_sizes, **kwags)

    def inverse_transform(self, data, **kwags):
        data = inverse_transformCells(self, data, **kwags)
        return data
    
    def inverse_flattenCore(self, data, **kwags):
        data = DFP.transformFlatBatchToCells(data, cell_size=self.cell_size, 
                                             editor=getattr(self,'editor',None), 
                                             pad_value=self.pad_value, 
                                             feaVarAxis=self.feaVarAxis,
                                             **kwags)
        return data
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

def activateZone(dic, zoneClassName=None, default_zoneClassName='HEADER_ZONE', preprocessor_dir='.', stamps=None, exp_fd='.', config=None, **kwags):
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
    zoneClassName = zoneClassName if(LOGger.isinstance_not_empty(zoneClassName,str)) else dic.get('zoneClass', default_zoneClassName)
    zoneClass = eval(zoneClassName)
    data_prop = LOGger.transform_dict2class(dic['data_prop'])
    preprocessing = LOGger.transform_dict2class(dic.get('preprocessing'))
    headerZone = zoneClass(dic['core'], data_prop=data_prop, preprocessing=preprocessing, exp_fd=exp_fd, 
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

def generate_preprocessor_file(zone, exp_fd=None, file=None):
    """
    zone has params that are default exp_fd and file. Yet we can still asign custom exp_fd and file settings.

    Parameters
    ----------
    zone : TYPE
        DESCRIPTION.
    exp_fd : TYPE, optional
        DESCRIPTION. The default is None.
    file : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    file : TYPE
        DESCRIPTION.

    """
    exp_fd = exp_fd if(LOGger.isinstance_not_empty(exp_fd, str)) else zone.exp_fd
    file = file if(LOGger.isinstance_not_empty(file, str)) else os.path.join(exp_fd, '%s.pkl'%zone.fn)
    return file

def exportPreprocessor(zone, exp_fd=None, file=None):
    '''
    1st `file` -> 2nd os.path.join(exp_fd,self.fn) -> 3rd XXX

    Parameters
    ----------
    exp_fd : TYPE, optional
        DESCRIPTION. The default is None.
    file : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    bool
        DESCRIPTION.

    '''
    addlog_ = LOGger.addloger(logfile='')
    # 指定輸出路徑，但也有內建
    stamp = zone.fn
    if(LOGger.isinstance_not_empty(file, str)):
        preprocessor_file = file
    elif(LOGger.isinstance_not_empty(exp_fd, str)):
        if(not LOGger.isinstance_not_empty(stamp, str)):
            stamp = 'Psr'
        preprocessor_file = generate_preprocessor_file(zone, exp_fd=exp_fd, file=file)
    else:
        preprocessor_file = os.path.join(zone.exp_fd, 'warehouse', stamp)
    try:
        exp_fd = os.path.dirname(preprocessor_file)
        if(not dump_model(zone.preprocessor, preprocessor_file)):
            addlog_('dump_model error:%s!!!'%file, 'while exp_fd error'%exp_fd, colora=LOGger.FAIL)
            return False
        zone.preprocessor_file = os.path.relpath(preprocessor_file, exp_fd) #這是用來寫在json檔內的設定，要用相對路徑
        LOGger.save_json(zone, os.path.join(exp_fd, '%s.json'%stamp))
    except Exception as e:
        LOGger.exception_process(e,logfile='', stamps=zone.stamps)
        return False
    labels = {}
    if(DFP.isiterable(getattr(zone.preprocessor, 'classes_',None))):
        labels['classes_'] = LOGger.mylist(tuple(zone.preprocessor.classes_)).get_all()
    if(labels): 
        all_categories_fn = '%s.json'%LOGger.stamp_process('',[stamp,'Label'],'','','','_',for_file=True)
        zone.all_categories_file = dcp(all_categories_fn)
        LOGger.save_json(labels, os.path.join(exp_fd, all_categories_fn))
    addlog_('dumped module:%s'%preprocessor_file, stamps=zone.stamps, colora=LOGger.OKGREEN)
    return True

def file2img(hdz, data, editorDefault=None, default_shape=(700,700,3), default_stamps='', default_value=0, **kwags):
    editor = LOGger.execute('editor', hdz, kwags, default=editorDefault, not_found_alarm=False)
    common_shape = getattr(hdz, 'cell_size', default_shape)
    return vs3.file2img(data, editor=editor, common_shape=common_shape, 
                        default_stamps=default_stamps, default_value=default_value, **kwags)

def transformCells(headerZone, data, default_cell_size=None, default_preprocessor=None, default_pad_value=0, default_editor=None, 
                   axisIndex=0, zoneIndex=0, **kwags):
    groupIndex = getattr(headerZone, 'groupIndex', 1)
    cell_size = getattr(headerZone, 'cell_size', default_cell_size)
    preprocessor = getattr(headerZone, 'preprocessor', default_preprocessor)
    pad_value = getattr(headerZone, 'pad_value', default_pad_value)
    editor = getattr(headerZone, 'editor', default_editor)
    if(LOGger.isinstance_not_empty(editor, str)):    editor = eval(editor)
    try:
        if(isinstance(data, pd.core.frame.DataFrame)):
            data = data[headerZone]
        elif(isinstance(data, list)):
            data = list(data)[zoneIndex]
        elif(isinstance(data, np.ndarray)):
            if(len(data.shape)<2):
                m_debug.updateDump({'transformCells':{'data': data, 'cell_size':cell_size}})
            if(data.shape[1]<len(headerZone)):
                m_debug.updateDump({'transformCells':{'data': data, 'cell_size':cell_size}})
            data = data[:,axisIndex:axisIndex + len(headerZone)]
        data = getattr(data,'values',data)
    except Exception as e:
        LOGger.exception_process(e,logfile='')
        m_debug.updateDump({'transformCells':{'data': data}})
    try:
        if(DFP.isiterable(cell_size)):
            # 有cell結構，應該要是獨一欄，把單一欄去除
            data = DFP.transformByBatch(data, method=lambda x:x[0])
        # LOGger.addDebug('a'*50)
        if(data is None):
            return None
        if(callable(editor)):   data = DFP.transformByBatch(data, editor)
        if(preprocessor!=None):
            # 先平坦化(2維化)
            dataFlat = headerZone.flattenCore(data)
            if(isinstance(getattr(headerZone, 'data_prop', None), str)):
                data_prop = getattr(headerZone, 'data_prop')
                if(data_prop=='continuous'):    dataFlat = np.vectorize(DFP.astype_or_remain)(dataFlat)
            dataFlat = preprocessor.transform(dataFlat)
            # LOGger.addDebug('c'*50)
            data = headerZone.inverse_flattenCore(dataFlat)
        elif(DFP.isiterable(cell_size)):
            method = lambda x:DFP.reshapeThruDimensions(x, shape=cell_size, pad_value=pad_value)
            data = DFP.transformByBatch(data, method)
    except Exception as e:
        LOGger.exception_process(e,logfile='')
        m_debug.updateDump({'transformCells':{'preprocessor': preprocessor, 'dataFlat': dataFlat, 'headerZone': headerZone}})
    return np.array(data)

def inverse_transformCells(headerZone, data, default_cell_size=None, default_preprocessor=None, default_pad_value=0, default_editor=None, **kwags):
    groupIndex = getattr(headerZone, 'groupIndex', 1)
    cell_size = getattr(headerZone, 'cell_size', default_cell_size)
    preprocessor = getattr(headerZone, 'preprocessor', default_preprocessor)
    pad_value = getattr(headerZone, 'pad_value', default_pad_value)
    editor = getattr(headerZone, 'editor', default_editor)
    # cell_size = cell_size if(DFP.isiterable(cell_size)) else data.shape[1:] # 想看看這句有必要嗎
    if(preprocessor!=None):
        # 先平坦化(2維化)
        dataFlat = headerZone.flattenCore(data) 
        # 反標準化或反標籤化
        if(preprocessor):   dataFlat = preprocessor.inverse_transform(dataFlat)
        # 打回形成CELLS
        # if(DFP.isiterable(cell_size)):
        #     cell_size_annil_group = tuple(map(int, np.array(data.shape)[np.array([int(i) for i,x in enumerate(data.shape) if i!=groupIndex],dtype=int)]))
        #     data = DFP.reshapeThruFlatten(data, shape=cell_size_annil_group)
        # method = lambda x:DFP.transformFlatToCell(x, cell_size=cell_size, pad_value=pad_value, **kwags)
        # data = DFP.transformByBatch(dataFlat, method)
        data = headerZone.inverse_flattenCore(dataFlat)
    elif(DFP.isiterable(cell_size)):
        if(len(headerZone)!=1):
            print('len(headerZone) = ', len(headerZone), '!=', 1)
            sys.exit(1)
        method = lambda x:DFP.reshapeThruDimensions(x, shape=cell_size, pad_value=pad_value, **kwags)
        data = DFP.transformByBatch(data, method)
    if(callable(editor)):   data = DFP.transformByBatch(data, editor)
    if(DFP.isiterable(cell_size)):
        data = np.expand_dims(data, axis=groupIndex)
    return np.array(data)

# (*tuple(np.array([12,47,71])/255), 0.5) resolSequentialView
def drawLossCurve(ax, *losses, stamps=None, colors=None, default_color=m_lossColors[0], rewrite=True, **kwags):
    colors = m_lossColors
    ret = {}
    lossesResed = DFP.resolSequentialView(*losses, ret=ret)
    ax.clear() if(rewrite) else None
    title = '%s loss curves'%LOGger.stamp_process('',stamps)
    for i,lossCurve in enumerate(lossesResed):
        infrm = {k:v for k,v in ret.get(i, {}).items() if k in ['end','min','nan']}
        label = dcp(LOGger.stamp_process('',infrm,digit=7))
        color = dcp(colors.get(i%len(m_lossColors), default_color))
        ax.plot(range(len(lossCurve)), lossCurve, color=(*(color[:3]), 0.5), label=label)
    ax.set_title(title)
    return True

def saveLossCurve(mdc):
    fig = mdc.lossCurveFig
    stamps = mdc.get_stamps()
    stamp = LOGger.stamp_process('',stamps+['lossCurve'],'','','','_', for_file=True)
    file = os.path.join(mdc.get_graph_exp_fd(), '%s.jpg'%stamp)
    LOGger.CreateFile(file, lambda f:vs3.end(fig, file=f))
    return True

def saveLossCurveWhile(mdc, ret=None):
    if(isinstance(ret, dict)):  ret['success'] = False
    if(not hasattr(mdc,'drawLossStopFlag')):
        mdc.addlog("don't draw loss!!!", colora=LOGger.OKCYAN)
        return False
    mdc.addlog("start draw loss!!!", )
    while(not getattr(mdc,'drawLossStopFlag',True)):
        saveLossCurve(mdc)
        LOGger.addlog('drawLossSleepTime......', getattr(mdc,'drawLossSleepTime',5), colora=LOGger.OKCYAN, logfile='')
        LOGger.time.sleep(getattr(mdc,'drawLossSleepTime',5))
    if(isinstance(ret, dict)):  ret['success'] = True
    mdc.addlog('stop draw loss!!!', colora=LOGger.OKCYAN)
    return True
    
def fitPreprocessingStandard(headerZone, referenceData, **kwags):
    try:
        if(not headerZone.resetPreprocessor(referenceData)):
            return False
    except Exception as e:
        LOGger.exception_process(e,logfile='')
        return False
    return True
    
#%%
#數據工具
#TODO:relu_custom_machine
def relu_custom_machine(threshold=0, support=0, scale=1, omega=1, **kwags):
    def relu_custom(x):
        return bcd.cast_to_floatx(scale*activations.relu(omega*(x-threshold))+support)
    return relu_custom
        
#TODO:selu_custom_machine
def selu_custom_machine(threshold=0, support=0, **kwags):
    def selu_custom(x):
        return bcd.cast_to_floatx(activations.selu(x-threshold)+support)
    return selu_custom
        
#TODO:elu_custom_machine
def elu_custom_machine(threshold=0, support=0, scale=1, omega=1, **kwags):
    def elu_custom(x):
        return bcd.cast_to_floatx(scale*activations.elu(omega*(x-threshold))+support)
    return elu_custom

#TODO:shelu_custom_machine
def shelu_custom_machine(scale=0.5, support=0, threshold=0, **kwags):
    def shelu_custom(x):
        return bcd.cast_to_floatx((scale/0.5)*(
                (activations.elu(x-threshold+0.5)-activations.relu(x-threshold)-
                 activations.elu(threshold-x+0.5)+activations.relu(threshold-x)))+support)
    return shelu_custom

#TODO:sigmoid_custom_machine
def sigmoid_custom_machine(scale=0.5, support=0, threshold=0, **kwags):
    def sigmoid_custom(x):
        return bcd.cast_to_floatx((scale/0.5)*(activations.sigmoid(x-threshold)-0.5)+support)
    return sigmoid_custom

#TODO:zelu_custom_machine
def zelu_custom_machine(scale=0.5, support=0, center=0.5, threshold=0, **kwags):
    def zelu_custom(x):
        return bcd.cast_to_floatx((scale/0.5)*(
                (activations.relu(x-threshold+0.5)-1)-activations.relu(x-threshold-0.5)+0.5)+support)
    return zelu_custom

#TODO:scalering
def scalering(x, scr, axis=0):
    y = x
    if(hasattr(scr, 'n_features_in_')):
        dim = scr.n_features_in_
        X=[0]*axis+[x]+[0]*(dim-(axis+1))
        if(not DFP.isnonnumber(x)):
            y = scr.transform([X])[axis]
    return y

def sigmoid(x, centre=0, k=1, score_min=None, score_max=None):
    if(score_max==None and score_min==None):
        return 1/(1 + np.exp(-k * (x - centre)))
    elif(score_max==None and score_min!=None):
        return np.where(x<=centre, (1/2)*(1 - (x-centre)/(score_min - centre)), 1/(1 + np.exp(-k * (x - centre))))
    elif(score_max!=None and score_min==None):
        return np.where(x>=centre, (1/2)*(1 + (x-centre)/(score_max - centre)), 1/(1 + np.exp(-k * (x - centre))))
    elif(score_max!=None and score_min!=None):
        return np.where(x>=centre, (1/2)*(1 + (x-centre)/(score_max - centre)), (1/2)*(1 - (x-centre)/(score_min - centre)))
#%%"zoneClass": "HEADER_ZONE_INPUT_WITHCELL",
def activate_EIMS(config_file=None, config=None, stamps=None, exp_fd='.', version='version', **kwags):
    mdc = create_EIMS(config_file=config_file, config=config, stamps=stamps, exp_fd=exp_fd, version=version, **kwags)
    if(not mdc.set_config()):
        return None
    return mdc

def create_EIMS(config_file=None, config=None, stamps=None, exp_fd='.', version='version', **kwags):
    if(not LOGger.isinstance_not_empty(config, dict) and not LOGger.isinstance_not_empty(config_file, str)):
        LOGger.addlog('invalid create_EIMS !!!!', logfile='', colora=LOGger.FAIL)
        LOGger.addlog('config', config, logfile='', colora=LOGger.FAIL)
        LOGger.addlog('config_file', config_file, logfile='', colora=LOGger.FAIL)
        sys.exit(1)
    config = config if(LOGger.isinstance_not_empty(config, dict)) else LOGger.load_json(config_file)
    EIMSClassStg = config.get(m_coreNameHeader, 'EIMS_core')
    EIMSClass = eval(EIMSClassStg)
    print('detect EIMSClassStg: %s'%EIMSClassStg)
    mdc = EIMSClass(stamps=stamps, exp_fd=exp_fd, config_file=config_file, version=version, **kwags)
    return mdc

def scipyMinizing(mdc, targets, starts=None, default_start=0, fixHeaders=None, fixData=None, **kwags):
    score_fun = lambda x:mdc.predict(np.array([x]).reshape(-1,len(mdc.xheader)))[0][0]
    return DFP.scipyMinizingTargets(score_fun, targets, starts=starts, default_start=default_start, 
                                    fixHeaders=fixHeaders, fixData=fixData, xheader=mdc.xheader, **kwags)

#TODO:scipy_optimizing
def scipy_local_optimizing(score_fun, X0, method='Nelder-Mead', history=[], stamps=[],
                           bounds=None, options={'disp': True}, ret={}, callback=None, 
                           draw_loss_curve_method=None, **kwags):
    exp_fd = kwags.get('exp_fd', 'test')
    addlog = lambda s, **kwags:LOGger.addlog(s, logfile=os.path.join(exp_fd, 'log.txt'), **kwags)
    try:
        res = minimize((lambda X:mylist([score_fun(X)]).get_all()[-1]), X0, method=method, 
                       callback=(lambda d:callback(
                           d, score_fun, history=history, stamps=stamps, exp_fd=exp_fd, 
                           draw_loss_curve_method=draw_loss_curve_method, 
                           PAW_Standard=kwags.get('PAW_Standard',None))) if(callback) else None, 
                       bounds=bounds, options=options)
    except Exception as e:
        exception_process(e, logfile=os.path.join(exp_fd, 'log.txt'), stamps=['scipy_local_optimizing'])
        addlog('%s優化失敗%s'%(stamp_process('',stamps), stamp_process('',{
            'X0 shape':np.array(X0).shape, 'X0':','.join(list(map(lambda s:DFP.parse(s,2), X0))[:30])})))
        try:
            addlog(' ',stamps={'score_fun(X0)':DFP.parse(score_fun(X0), 4)})
        except Exception as e:
            addlog('score fun與X0不相容!!!')
            exception_process(e, logfile=os.path.join(exp_fd, 'log.txt'))
        addlog(' ', stamps={'bounds shape':str(np.array(bounds).shape)}) if(bounds!=None) else None
        addlog('bound size is not compatible with inputs!!!') if(np.array(bounds).shape[0]!=np.array(X0).shape[0] if(bounds!=None) else False) else None
        return False
    if(res.nit==0):
        addlog('%s迭代失效'%stamp_process('',stamps))
        return False
    ret['res'] = res
    return True

#TODO:explanation
def explanation(outputs, threshold_=0.5, multi_dim_method=lambda d:np.argmax(d, axis=1)):
    """
    Given raw scores and return dominating axis(int)

    Parameters
    ----------
    outputs : TYPE
        DESCRIPTION.
    threshold_ : TYPE, optional
        DESCRIPTION. The default is 0.5.
    multi_dim_method : TYPE, optional
        DESCRIPTION. The default is `lambda d:np.argmax(d, axis=1)`.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    ret = []
    shape = outputs.shape
    if(len(shape)==1):
        ret = list(map(lambda x:1 if(x>threshold_) else 0, tuple(outputs)))
    elif(shape[1]==1):
        outputs = outputs.reshape(-1)
        ret = list(map(lambda x:1 if(x>threshold_) else 0, tuple(outputs)))
    elif(shape[1]>1):
        ret = multi_dim_method(outputs)
    return np.array(ret).reshape(-1,1)

#TODO:explanation_score
def explanation_score(outputs, zone_index=0, multi_dim_method=lambda d:np.max(d, axis=1), return_proba = False,
                      probabilities_method=lambda a:1 - sigmoid(np.array(a), centre=0.5, score_max=1, score_min=0), 
                      **kwags):
    """
    Given raw scores and return dominating score(float)

    Parameters
    ----------
    outputs : TYPE
        DESCRIPTION.
    multi_dim_method : TYPE, optional
        DESCRIPTION. The default is `lambda d:np.max(d, axis=1)`.
    zone_index : TYPE, optional
        DESCRIPTION. The default is 0.
    return_proba : TYPE, optional
        是否把數據都弄成0~1之間?. The default is False.
    probabilities_method : TYPE, optional
        To deteremine how to transform scores to probabilities. The default is 
        `lambda a:1 - sigmoid(np.array(a), centre=0.5, score_max=1, score_min=0)`.
    **kwags : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    ret = []
    output_score = outputs[zone_index] if(isinstance(outputs, list)) else outputs
    if(isinstance(output_score, type(None))):
        return None
    shape = output_score.shape
    if(True if(len(shape)==1) else (shape[1]==1)):
        ret = output_score
    elif(shape[1]>1):
        ret = multi_dim_method(output_score)
    ret = probabilities_method(np.array(ret).reshape(-1, 1)) if(return_proba) else np.array(ret).reshape(-1, 1)
    return ret
#TODO:explanation_confidence
def explanation_confidence(scores, zone_index=-1, threshold=0.5, multi_dim_method=lambda d:np.max(d, axis=1), 
                           probabilities_method=lambda a:1 - sigmoid(np.array(a), centre=0.5, score_max=1, score_min=0), 
                           threshold_sigmoid=0.5, **kwags):
    """
    Given scores and return explanations(True/False).

    Parameters
    ----------
    output : TYPE
        explaining objects.
    zone_index : TYPE, optional
        Available if outputs is type of list. To select the exact zone to be explain. The default is -1.
    multi_dim_method : TYPE, optional
        To transform the selected zone from high dim to one. The default is Euclidean distance method
        `lambda a:np.prod(a, axis=1)>0`.
    threshold : TYPE, optional
        Return True if score greater then threshold. The default is 0.5.
    **kwags : TYPE
        DESCRIPTION.

    Returns
    -------
    outputs : TYPE
        DESCRIPTION.

    """
    kwags.pop('return_proba', None)
    scores = explanation_score(scores, zone_index=zone_index, multi_dim_method=multi_dim_method, return_proba=True,
                               probabilities_method=probabilities_method, **kwags)
    if(isinstance(scores, type(None))):
        return None
    threshold = 0.5 if(np.isnan(DFP.astype(threshold, d_type=float, default=np.nan))) else float(threshold)
    ret = scores>=0.5
    return ret


def padCropPadConstant(cell, cell_size_target, value=0):
    old_shape = cell.shape
    if(len(old_shape)!=len(cell_size_target)):
        return None
    pad_width = []
    for i,sh in enumerate(cell_size_target):
        diff = max(0, sh - old_shape[i])
        pad_width.append(diff)
    cell = np.pad(cell, pad_width=pad_width, mode='constant', constant_values=value)
    slices = []
    for i,sh in enumerate(cell_size_target):
        ubd = max(0, sh)
        slices.append(slice(0, ubd))
    slices = tuple(slices)
    cell = cell[slices]
    return cell

def padInBatch(inputs, cell_size, mode='constant', constant_values=0, **kwags):
    inputs_temps = []
    for i in range(inputs.shape[0]):  
        inputsCell = dcp(inputs[i])
        pad_width = tuple(map((lambda x:(0,x)), np.max(0, np.array(cell_size) - np.array(inputsCell.shape))))
        inputs_temp = np.pad(inputs[i], pad_width=pad_width, mode=mode, constant_values=constant_values)
        inputs_temps.append(dcp(inputs_temp))
    inputs = np.array(inputs_temps)
    return inputs

def cropInBatch(inputs, cell_size, cropMasks=None, **kwags):
    cropMasks = LOGger.mylist(cropMasks if(isinstance(cropMasks, list)) else [])
    inputs_temps = []
    for i in range(inputs.shape[0]):  
        inputsCell = dcp(inputs[i])
        inputs_temp = inputsCell[np.ix_(*[cropMasks.get(j, np.arange(cell_size[j])) for j in range(len(cell_size))])]
        inputs_temps.append(dcp(inputs_temp))
    inputs = np.array(inputs_temps)
    return inputs
    
#%%
def set_header_zones(mdc, header_zones_name, header_zones_config, config_dir='.', config=None, **kwags):
    # stamps = stamps if(isinstance(stamps, list)) else []
    header_zones_name_label = header_zones_name.replace('_zones','')
    for kk,vv in header_zones_config.items():
        # LOGger.addDebug('set_header_zones', kwags.get('figsize'))
        zones_label = 'xPsr' if(header_zones_name_label=='xheader') else ('yPsr' if(header_zones_name_label=='yheader') else 'Psr')
        header_zone = activateZone(vv, default_zoneClassName=mdc.header_zones_types.get(header_zones_name_label, 'HEADER_ZONE'), 
                                   preprocessor_dir=config_dir, stamps=[zones_label, kk], exp_fd=mdc.exp_fd, config=config, **kwags)
        # print('header_zone.exp_fd', getattr(header_zone,'exp_fd', '?'))
        if(not mdc.set_header(header_zones_name_label, kk, header_zone)): #同時新增header_zones跟header
            mdc.addlog('set header failed!!!', stamps=[header_zones_name, kk])
            return False
        # LOGger.addDebug(kk, type(header_zone.data_prop))
    # LOGger.addDebug('getattr(mdc, header_zones_name)', getattr(mdc, header_zones_name).concatenate(dtype=LOGger.mylist))
    mdc.addlog('set_header_zones done:%s'%(','.join(getattr(mdc, header_zones_name).concatenate(dtype=LOGger.mylist)[:10])), 
               stamps=[header_zones_name_label, len(getattr(mdc, header_zones_name_label))])
    return True

def save_header_zones(mdc, header_zones_name_label, header_zones, ret=None):
    header_zones_name = '%s_zones'%header_zones_name_label #-6:_zones
    header_zones_config = ret if(isinstance(ret, dict)) else {}
    for kk,vv in header_zones.items():
        header_zones_config[kk] = vv.serializeZone().copy()
    mdc.addlog('save_header_zones done:%s'%(
        ','.join(getattr(mdc, header_zones_name_label)[:10])), stamps=[header_zones_name_label, len(getattr(mdc, header_zones_name_label))])
    return True

# mystr(v).config_evaluation(eva_bck=eva_bck, **ret)
def set_configAttrs(mdc, stamps, config=None, **kwags):
    config = config if(isinstance(config, dict)) else {}
    stamp = LOGger.stamp_process('',stamps,'','','','_')
    Attrs = getattr(mdc,'%sAttrs'%stamp)
    if(isinstance(Attrs,dict)):
        for k,v in Attrs.items():
            if(not k in config):    
                continue
            thing = dcp(LOGger.transform_dict2class(config[k]) if(isinstance(config[k], dict) or isinstance(config[k], str)) else config[k])
            Attrs[k] = thing
            setattr(mdc, k, thing)
            mdc.addlog(k, Attrs[k], colora=LOGger.OKGREEN, stamps=stamps)
        mdc.addlog('set %sAttrs:%s'%(stamp, ','.join(list(Attrs.keys()))), colora=LOGger.OKGREEN)
    else:
        for k in Attrs:
            if(not k in config):    
                continue
            thing = dcp(LOGger.transform_dict2class(config[k]) if(isinstance(config[k], dict) or isinstance(config[k], str)) else config[k])
            Attrs[k] = thing
            setattr(mdc, k, thing)
            mdc.addlog(k, getattr(mdc, k), colora=LOGger.OKGREEN, stamps=stamps)
        mdc.addlog('set %sAttrs:%s'%(stamps, ','.join(list(map(DFP.parse, Attrs)))), colora=LOGger.OKGREEN)
    return True

def configuring(self, stamps=None, **kwags):
    stamps = stamps if(isinstance(stamps, list)) else []
    stamp = LOGger.stamp_process('',stamps,'','','','_')
    Attrs = dcp(getattr(self,'%sAttrs'%stamp))
    Attrs.update(kwags)
    if(not mdcScenarioConfiguring(self, Attrs, stamps=[stamp])):
        return False
    return True

def checkConfigAvailable(config_file):
    if(not LOGger.isinstance_not_empty(config_file, str)):
        print("config_file err: %s"%config_file)
        return False
    if(not os.path.exists(config_file)):
        print("config_file doesn't exists: %s"%config_file)
        return False
    return True

def loadConfigScenario(handler, eva_bck='$'):
    if(not checkConfigAvailable(handler.config_file)):
        LOGger.addlog('config_file not available:%s'%handler.config_file, colora=LOGger.FAIL, logfile='')
        return False
    handler.config_dir = os.path.dirname(handler.config_file)
    LOGger.addlog('config source_dir:%s'%handler.config_dir, colora=LOGger.OKGREEN, logfile='')
    config = LOGger.load_json(handler.config_file)
    if(isinstance(handler.config, dict)):   
        handler.config.clear()
    else:
        handler.config = {}
    for k,v in config.items():
         v = dcp(LOGger.mystr(v).config_evaluation(eva_bck=eva_bck, **config) if(isinstance(v, str)) else v)
         handler.config[k] = v
    return True

def update_config(mdc, config=None, **kwags):
    """
    一但給了config_file, 就會自動更新config_dir!!!! 
    """
    config = config if(isinstance(config, dict)) else {}
    # LOGger.addDebug('update_config', config.get('model_file','?'))
    set_configAttrs(mdc, 'main', config=config)
    set_configAttrs(mdc, 'fit', config=config)
    set_configAttrs(mdc, 'compile', config=config)
    return True

def set_config(mdc, config_file=None, config_dir=None, config=None, eva_bck='$', **kwags):
    if(not LOGger.isinstance_not_empty(config, dict)):
        if(not loadConfigScenario(mdc, eva_bck=eva_bck)):
            mdc.addlog('loadConfigScenario failed', colora=LOGger.FAIL)
            return False
        mdc.addlog('loadConfigScenario successful', colora=LOGger.OKGREEN)
        config = mdc.config
    else:
        mdc.config = config
    if(config is None):
        mdc.addlog('no config!!!!!!!', colora=LOGger.FAIL)
        return False
    if(not mdc.update_config(config=config, **kwags)):
        return False
    mdc.addlog('mdc.config_dir', mdc.config_dir, colora=LOGger.OKGREEN)
    configModules = dcp(config)
    configModules.update(mdc.mainAttrs) #mdc.mainAttrs在mdc.update_config已經被mdl.config更新過
    if(not set_mdcModules(mdc, config_dir=mdc.config_dir, config=configModules)):
        return False
    if(not set_kerasCustom_config(mdc)):
        mdc.addlog('set_kerasCustom_config failed!!!!!!!')
        return False
    if(LOGger.isinstance_not_empty(getattr(mdc, 'model_file', None),str)):
        loadFile = os.path.join(mdc.config_dir, mdc.model_file)
        if(not mdc.set_model()):
            mdc.addlog('set up model failed:%s'%loadFile, stamps=mdc.get_stamps(), colora=LOGger.FAIL)
            return False
        mdc.addlog('set up model:%s'%loadFile, stamps=mdc.get_stamps(), colora=LOGger.OKCYAN)
    # sys.exit(1)
    return True

def set_config_autoEncoder(mdc, config=None, eva_bck='$', **kwags):
    if(not set_config(mdc, config=config, eva_bck=eva_bck, **kwags)):
        return False
    if(LOGger.isinstance_not_empty(mdc.latentCoresFile, str)):
        latentCoresFile = os.path.join(mdc.config_dir,mdc.latentCoresFile)
        if(os.path.isfile(latentCoresFile)):
            mdc.latentCores = DFP.joblib.load(latentCoresFile)
    if(LOGger.isinstance_not_empty(mdc.latentExplainerFile, str)):
        latentExplainerFile = os.path.join(mdc.config_dir,mdc.latentExplainerFile)
        if(os.path.isfile(latentExplainerFile)):
            mdc.latentExplainer = DFP.joblib.load(latentExplainerFile)
    if(LOGger.isinstance_not_empty(mdc.latentPcaFile, str)):
        latentPcaFile = os.path.join(mdc.config_dir,mdc.latentPcaFile)
        if(os.path.isfile(latentPcaFile)):
            mdc.latentPca = DFP.joblib.load(latentPcaFile)
    return True


#TODO:activate_custom_objects
def set_kerasCustom_config_inZone(mdc, headerZoneItems, stamps=None, ret=None, **kwags):
    stamps = stamps if(isinstance(stamps, list)) else []
    lossmethodsInZones = LOGger.mylist([])
    activationsInZones = LOGger.mylist([])
    for k,v in headerZoneItems:
        lossmethodsInZones.append(getattr(v, 'lossmethod','mse'))
        activationsInZones.append(getattr(v, 'activation', 'relu'))
    LOGger.show_vector(lossmethodsInZones[:20], stamps=['activating lossmethods']+stamps, logfile='', colora=LOGger.OKCYAN) if(lossmethodsInZones) else []
    LOGger.show_vector(activationsInZones[:20], stamps=['activating activations']+stamps, logfile='', colora=LOGger.OKCYAN) if(activationsInZones) else []
    lossmethodsInZonesUniq = DFP.uniqueByIndex(lossmethodsInZones)
    if(np.logical_not(np.array(tuple(map(lambda s,**kwags:init_lossmethods(s,mdc=mdc,**kwags), lossmethodsInZonesUniq)))).any()):
        addlog('detected undefined loss algorithms:%s'%stamp_process('',lossmethodsInZonesUniq,'','','',','), stamps=stamps)
        return False
    activationsInZonesUniq = DFP.uniqueByIndex(activationsInZones)
    if(np.logical_not(np.array(tuple(map((lambda v:init_activations(
            v, scr=(getattr(v, 'preprocessor', None) if(getattr(v, 'preprocessing', 'normalization')=='normalization') else None), 
            axis=getattr(v, 'axis', None), mdc=mdc)), 
            activationsInZonesUniq)))).any()):
        addlog('detected undefined activations:%s'%stamp_process('',activations,'','','',','), stamps=stamps)
        return False
    if(isinstance(ret, dict)):  ret['lossmethodsInZones'] = ret.get('lossmethodsInZones',[]) + lossmethodsInZones
    if(isinstance(ret, dict)):  ret['activationsInZones'] = ret.get('activationsInZones',[]) + activationsInZones
    return True

def set_kerasCustom_config(mdc, **kwags):
    ret = {}
    if(not set_kerasCustom_config_inZone(mdc, mdc.xheader_zones.items(), stamps=['xheader'], 
                                         ret = {} if(not isinstance(mdc, EIMS_AUTOENCODER_core)) else ret,
                                         **kwags)):
        return False
    if(not set_kerasCustom_config_inZone(mdc, getattr(mdc,'yheader_zones',{}).items(), stamps=['yheader'], 
                                         ret = {} if(isinstance(mdc, EIMS_AUTOENCODER_core)) else ret,
                                         **kwags)):
        return False
    if(not set_kerasCustom_config_inZone(mdc, getattr(mdc,'unfam_header_zones',{}).items(), stamps=['unfam_header'], ret = ret, **kwags)):
        return False
    mdc.lossmethods = LOGger.mylist(ret.get('lossmethodsInZones',[]))
    mdc.output_activations = LOGger.mylist(ret.get('activationsInZones',[]))
    return True

def set_mdcModules(mdc, config_dir=None, config=None, **kwags):
    config = config if(isinstance(config, dict)) else {}
    mainAttrs = LOGger.execute('mainAttrs',mdc,config,kwags,default=[],not_found_alarm=False)
    for k,v in config.items():
        if(v==''):
            continue
        # elif(k in mainAttrs):
        #     if(k=='custom_objects'):
        #         if(not set_kerasCustomObjects(v)):
        #             return False
        #             # mdc.addlog('set up %s:%s'%(cst_name, LOGger.stamp_process('',cst_attrs)), stamps=[k])
        #     print('[%s:%s]'%(DFP.parse(k),DFP.parse(v)[:200]))
        #     setattr(mdc, k, dcp(LOGger.transform_dict2class(v) if(isinstance(v, dict) or isinstance(v, str)) else v))
        elif(k[:-5] in mdc.model_stamps and k[-5:]=='_file'):
            setattr(mdc, k, v)
        elif(k in mdc.header_zones_names): #k: xheader_zones, yheader_zones, unfam_header_zones, ....
            if(not set_header_zones(mdc, k, v, config_dir=config_dir, config=config)):
                mdc.addlog('set_header_zones failed!!!', stamps=[k])
                return False
        else:
            continue
        mdc.addlog('set up %s:%s'%(k, type(getattr(mdc, k))), stamps=mdc.get_stamps())
    return True 

def save_config(mdc, exp_fd=None, **kwags):
    exp_fd = exp_fd if(LOGger.isinstance_not_empty(exp_fd, str)) else getattr(mdc, 'exp_fd', '.')
    version = getattr(mdc, 'out_version', getattr(mdc, 'version', '.'))
    config_file = kwags.get('config_file', os.path.join(mdc.get_module_exp_fd(), 'config.json'))
    mainAttrs = LOGger.execute('mainAttrs',mdc,kwags,default=[],not_found_alarm=False)
    mdc.addlog('config exp_fd:%s'%exp_fd)
    mdc.addlog('saving mainAttrs:%s'%(','.join(mainAttrs)))
    
    config = {}
    for k,v in mainAttrs.items():
        attr = getattr(mdc, k, v)
        # if(not getattr(mdc, k, v)):
        #     mdc.addlog('mdc has no mainAttrs: %s'%attr)
        #     return False
        config[k] = LOGger.transform_class2dict(attr)
        mdc.addlog('mainAttrs %s:%s'%(k, config[k]), colora=LOGger.OKGREEN)
    config[m_coreNameHeader]=LOGger.type_string(mdc)
    ret = {}
    for header_zones_name in mdc.header_zones_names:
        ret.clear()
        header_zones_name_label = header_zones_name[:-6] #-6:_zones
        if(not save_header_zones(mdc, header_zones_name_label, getattr(mdc, header_zones_name), ret=ret)):
            return False
        config[header_zones_name] = ret.copy()
    for model_stamp in mdc.model_stamps:
        config['%s_file'%model_stamp] = os.path.relpath(
            LOGger.execute('%s_file'%model_stamp, mdc, kwags, default='-', not_found_alarm=False), mdc.get_module_exp_fd())
    # LOGger.addDebug(str(config))
    CreateFile(config_file, lambda f:LOGger.save_json(config, f))
    return True

def saveModelStandard(mdc, exp_fd, **kwags):
    theme = LOGger.execute('theme', mdc, kwags, default='theme')
    for model_stamp in mdc.model_stamps:
        if(hasattr(mdc, model_stamp)):
            model = getattr(mdc, model_stamp)
            extFileType = getattr(model_stamp, 'extFileType', mdc.modelExpfileType)
            file = os.path.join(mdc.get_module_exp_fd(), '%s.%s'%(
                LOGger.stamp_process('',[theme] + mdc.get_stamps(full=True) + (
                    [model_stamp] if(model_stamp!='model') else []),'','','','_',for_file=True), extFileType))
            try:
                dump_model(model, file)
                setattr(mdc, LOGger.stamp_process('',[model_stamp, 'file'],'','','','_'), LOGger.dcp(file))
            except Exception as e:
                LOGger.exception_process(e,logfile=os.path.join(exp_fd,'log.txt'),stamps=[model_stamp])
                return False
            mdc.addlog('dumped model:%s'%file, stamps=[model_stamp], colora=LOGger.OKCYAN)
        else:
            mdc.addlog('has no model!!!', stamps=[model_stamp], colora='\033[91m')
    return True

def save_models(mdc, exp_fd=None, **kwags):
    theme = LOGger.execute('theme', mdc, kwags, default='theme')
    exp_fd = exp_fd if(LOGger.isinstance_not_empty(exp_fd, str)) else mdc.get_module_exp_fd()
    if(not saveModelStandard(mdc, exp_fd=mdc.get_module_exp_fd(), **kwags)):
        return False
    return True

def save_models_autoEncoder(mdc, exp_fd=None, **kwags):
    theme = LOGger.execute('theme', mdc, kwags, default='theme')
    exp_fd = exp_fd if(LOGger.isinstance_not_empty(exp_fd, str)) else mdc.get_module_exp_fd()
    if(not saveModelStandard(mdc, exp_fd=exp_fd, **kwags)):
        return False
    mdc.latentCoresFile = 'latentCores.pkl'
    LOGger.CreateFile(os.path.join(exp_fd, mdc.latentCoresFile), lambda f:DFP.joblib.dump(mdc.latentCores, f))
    mdc.latentExplainerFile = 'latentExplainer.pkl'
    LOGger.CreateFile(os.path.join(exp_fd, mdc.latentExplainerFile), lambda f:DFP.joblib.dump(mdc.latentExplainer, f))
    mdc.latentPcaFile = 'latentPca.pkl'
    LOGger.CreateFile(os.path.join(exp_fd, mdc.latentPcaFile), lambda f:DFP.joblib.dump(mdc.latentPca, f))
    return True

def save_headerPreprocessors(header_zone, exp_fd=None, file=None, stamps=None, **kwags):
    addlog_ = LOGger.execute('addlog', header_zone, kwags, default=LOGger.addloger(logfile=''), not_found_alarm=False)
    stamps = stamps if(isinstance(stamps, list)) else []
    if(not isinstance(header_zone, HEADER_ZONE)):
        addlog_('not HEADER_ZONE:%s'%type(header_zone), stamps=stamps, colora=LOGger.FAIL)
        return False
    if(getattr(header_zone, 'preprocessor', None) is None):
        addlog_('no preprocessor:%s'%header_zone, stamps=stamps, colora=LOGger.WARNING)
        return True
    if(not header_zone.exportPreprocessor(exp_fd=exp_fd, file=file)):
        addlog_('header_zone.exportPreprocessor failed!!!', stamps=stamps, colora=LOGger.FAIL)
        return False
    return True

def save_mdcHeaderPreprocessors(mdc, exp_fd=None, **kwags):
    addlog_ = LOGger.execute('addlog', mdc, kwags, default=LOGger.addloger(logfile=''), not_found_alarm=False)
    exp_fd = exp_fd if(LOGger.isinstance_not_empty(exp_fd, str)) else mdc.get_module_exp_fd()
    for header_zones_name in mdc.IOheader_zones_names:
        header_zones = getattr(mdc, header_zones_name)
        mdc.addlog('headerZones count:', len(header_zones), stamps=[header_zones_name])
        for k,header_zone in header_zones.items():
            if(not save_headerPreprocessors(header_zone, exp_fd=exp_fd, stamps=[k], addlog=addlog_, file='')):
                return False
    return True

def save_mdcModules(mdc, exp_fd=None):
    """
    mdc.save_models -> save_mdcHeaderPreprocessors -> save_config
    If exp_fd==None, `mdc.save_models`,`save_mdcHeaderPreprocessors` will be save following mdc.get_module_exp_fd(), while save_config will be saved follow mdc.exp_fd

    Parameters
    ----------
    mdc : TYPE
        DESCRIPTION.
    exp_fd : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    bool
        DESCRIPTION.

    """
    if(getattr(mdc, 'model', None) is not None):
        if(not mdc.save_models(exp_fd=exp_fd)):
            return False
    if(not save_mdcHeaderPreprocessors(mdc, exp_fd=exp_fd)):
        return False
    # if(not save_config(mdc, exp_fd=exp_fd)):
    #     return False
    return True

#TODO:init_activations
def init_activations(name, scr=None, axis=0, **kwags):
    if(type(name)==str):
        #keras內建激活函數，無須客製化
        if(name in ['linear', 'relu', 'elu', 'selu', 'sigmoid', 'softmax']):
            return True
    elif(isinstance(name, str)):
        mdc = kwags.get('mdc', None)
        threshold = getattr(name, 'threshold', 0)
        support = getattr(name, 'support', 0)
        scale = getattr(name, 'scale', 0.5)
        centre = getattr(name, 'centre', 0.5)
        addlog = getattr(mdc, 'addlog', kwags.get('addlog', print))
        hypar = getattr(mdc, 'hypar', {})
        stamps = kwags.get('stamps', [])
        activation = None
        if(name=='relu_custom'):
            hypar['activation_threshold_%s'%stamp_process('', stamps, '','','','_')] = threshold
            activation = lambda x:relu(x, threshold=threshold)
        elif(name=='elu_custom'):
            hypar['activation_threshold_%s'%stamp_process('', stamps, '','','','_')] = threshold
            hypar['activation_support_%s'%stamp_process('', stamps, '','','','_')] = support
            activation = elu_custom_machine(threshold, support)
        elif(name in ['sigmoid_custom', 'zelu_custom', 'shelu_custom']):
            hypar['activation_scale_%s'%stamp_process('', stamps, '','','','_')] = scale
            hypar['activation_support_%s'%stamp_process('', stamps, '','','','_')] = support
            hypar['activation_center_%s'%stamp_process('', stamps, '','','','_')] = centre
            hypar['activation_threshold_%s'%stamp_process('', stamps, '','','','_')] = threshold
            scale, support, centre, threshold = tuple(map(lambda v:scalering(v, scr, axis), [scale, support, centre, threshold]))
            if(name=='sigmoid_custom'):
                activation = sigmoid_custom_machine(scale, support, threshold)
            elif(name=='zelu_custom'):
                activation = zelu_custom_machine(scale, support, threshold)
            elif(name=='shelu_custom'):
                activation = shelu_custom_machine(scale, support, threshold)
        if(activation!=None):
            get_custom_objects().update({name:activation})
            if(mdc!=None):
                setattr(mdc, 'custom_objects', {}) if(not hasattr(mdc, 'custom_objects')) else None
                addlog('%s'%pd.DataFrame({'custom activation hypars:':hypar})) if(mdc!=None) else None
                mdc.hypar = hypar
                attrs = {'support':support,'scale':scale,'centre':centre,'threshold':threshold}
                mdc.custom_objects[name] = attrs
        return True
    elif(name==None):
        return True
    return False

#TODO:init_lossmethods
def init_lossmethods(name, **kwags):
    if(type(name)==str):
        if(name in ['rmse', 'mse', 'sparse_categorical_crossentropy', 'categorical_crossentropy',
                    'binary_crossentropy','huber']):
            return True
    elif(isinstance(name, str)):
        mdc = kwags.get('mdc', None)
        addlog = getattr(mdc, 'addlog', kwags.get('addlog', print))
        hypar = getattr(mdc, 'hypar', {})
        stamps = kwags.get('stamps', [])
        gamma = 0
        attrs = {}
        loss_alg = None
        if(name=='focal'):
            gamma = getattr(name, 'gamma', 2)
            attrs['gamma'] = gamma
            hypar['loss_gamma_%s'%stamp_process('', stamps, '','','','_')] = gamma
            loss_alg = BinaryFocalLoss(gamma=gamma)
        elif(name=='normalDist'):
            output_size = getattr(name, 'output_size', 1)
            min_std = getattr(name, 'min_std', 1e-6)
            method = getattr(name, 'method', 'statistical')
            loss_type = getattr(name, 'loss_type', 'rmse')
            attrs['output_size'] = output_size
            attrs['min_std'] = min_std
            attrs['method'] = method
            attrs['loss_type'] = loss_type
            hypar['loss_output_size_%s'%stamp_process('', stamps, '','','','_')] = output_size
            hypar['loss_min_std_%s'%stamp_process('', stamps, '','','','_')] = min_std
            hypar['loss_method_%s'%stamp_process('', stamps, '','','','_')] = method
            loss_alg = ALG.create_normal_distribution_sequence_loss(
                output_size=output_size, min_std=min_std, method=method, loss_type=loss_type)
        elif(name=='normalDistDynamic'):
            min_std = getattr(name, 'min_std', 1e-6)
            method = getattr(name, 'method', 'statistical')
            loss_type = getattr(name, 'loss_type', 'rmse')
            attrs['min_std'] = min_std
            attrs['method'] = method
            attrs['loss_type'] = loss_type
            hypar['loss_min_std_%s'%stamp_process('', stamps, '','','','_')] = min_std
            hypar['loss_method_%s'%stamp_process('', stamps, '','','','_')] = method
            hypar['loss_type_%s'%stamp_process('', stamps, '','','','_')] = loss_type
            loss_alg = ALG.create_normal_distribution_dynamic_loss(
                min_std=min_std, method=method, loss_type=loss_type)
        if(loss_alg!=None):
            get_custom_objects().update({name:loss_alg})
            if(mdc!=None):
                setattr(mdc, 'custom_objects', {}) if(not hasattr(mdc, 'custom_objects')) else None
                addlog('%s'%pd.DataFrame({'custom lossmethod hypars:':hypar})) if(mdc!=None) else None
                mdc.hypar = hypar
                mdc.custom_objects[name] = attrs
        return True
    elif(name==None):
        return True
    return False

#TODO:load_PYOD_model
def load_PYOD_model(file):
    fext = os.path.splitext(file)[1]
    with open(file.replace(fext, '.pkl'), 'rb') as handle:
        model = pickle.load(handle)
    
    json_file = open(file.replace(fext, '.json'), 'r')
    
    loaded_model_json = json_file.read()
    loaded_model_json = loaded_model_json.replace("\"ragged\": false,", " ")
    json_file.close()
    loaded_model_ = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model_.load_weights(file.replace(fext, '.h5'))
    print("Loaded model from disk")
    
    model.model_ = loaded_model_   ## Set the loaded model to the auto encoder instance model
    return model

#TODO:load_model
def load_model(file, **kwags):
    if(file.find('.pyod')>-1):
        return load_PYOD_model(file)
    elif(file.find('.h5')>-1 or file.find('.tf')>-1):
        return models.load_model(file, **kwags)
    else:
        return joblib.load(file)

#TODO:dump_PYOD_model
def dump_PYOD_model(model, file):
    model_json = model.model_.to_json()
    fext = os.path.splitext(file)[1]
    with open(file.replace(fext, '.json'), 'w') as json_file:
      json_file.write(model_json)
    ##serialize weights to HDF5
    model.model_.save_weights(file.replace(fext, '.h5'))
    
    model.model_ = None
    with open(file.replace(fext, '.pkl'), 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

#TODO:dump_model
def dump_model(model, file):
    if(not os.path.isdir(os.path.dirname(file))):   LOGger.CreateContainer(file)
    if(file.find('.pyod')>-1):
        dump_PYOD_model(model, file)
        return True 
    elif(file.find('.h5')>-1 or isinstance(model, Functional)):
        model.save(file)
        return True
    else:
        joblib.dump(model, file)
        return True
    return False

#TODO:create_model_detail_keras_txt
def create_model_detail_keras_txt(model, logfile, deep=True, **kwags):
    print_fn_kwags = {'logfile':logfile}
    print_fn_kwags.update({k:v for (k,v) in kwags.items() if k in ['stamps']})
    addlog = lambda s:LOGger.addlog(s, **print_fn_kwags)
    model.summary(print_fn=addlog)
    if(deep):
        addlog('#####################################################################')
        addlog(str(model.input))
        addlog('---------------------------------------------------------------------')
        for layer in model.layers:
            infrm = {'name':getattr(layer, 'name', ''),
                     'activation':getattr(layer, 'activation', ''),
                     'units':getattr(layer, 'units', ''),
                     'batch_input_shape':getattr(layer, 'batch_input_shape', ''),
                     'trainable':getattr(layer, 'trainable', '')}
            addlog(stamp_process('', infrm))
        addlog('---------------------------------------------------------------------')
        addlog(str(model.output))
        addlog('#####################################################################')

#TODO:EIMS_core transform
def transformKeras(mdc, inputs, **kwags):
    exp_fd = LOGger.execute('exp_fd',mdc,kwags,default='.',not_found_alarm=False)
    addlog_ = LOGger.execute('addlog',mdc,kwags,default=LOGger.addloger(logfile=''),not_found_alarm=False)
    inputsHasZones = isinstance(inputs, list)
    if(inputsHasZones):
        if(len(inputs) != len(mdc.xheader_zones)):
            addlog_('n_xHeaderZones in mdc:', len(mdc.xheader_zones), '!= n_zones in inputs:', len(inputs), colora=LOGger.FAIL)
            mdc.addlog('transformation input invalid', type(inputs), stamps=['transformKeras'], colora=LOGger.FAIL)
            m_debug.updateDump({transformKeras.__name__:{'inputs':inputs, 'inputsHasZones':inputsHasZones, 
                                                         'mdc.xheader_zones.keys':mdc.xheader_zones.keys()}})
            return None
    np_input_zones = mylist()
    for i,(k,v) in enumerate(mdc.xheader_zones.items()):
        stamps = ['transformKeras', k]
        try:
            inputs_transed = v.transform(inputs, stamps=stamps, addlog=addlog_, exp_fd=mdc.exp_fd,
                                         axisIndex=mdc.xheader.index(v[0]), zoneIndex=i)
            if(inputs_transed is None):
                return None
            inputs_transed = mdc.convert4modelCore(inputs_transed)
        except Exception as e:
            mdc.addlog('transformation error', type(inputs.get(i)), stamps=stamps, colora=LOGger.Fore.RED)
            LOGger.exception_process(e, logfile=os.path.join(exp_fd, 'log.txt'))
            m_debug.updateDump({transformKeras.__name__:{'inputs':inputs, 'inputs_transed':inputs_transed, 'i':i, 'v':DFP.parse(v)}})
        np_inputs = np.array(inputs_transed)
        np_input_zones.append(np_inputs)
    return np_input_zones.get()

def generate_unfamMask(mdc, inputs, **kwags):
    unfam_header_zone_index = getattr(mdc, 'unfam_header_zone_index', -1)%len(inputs)
    threshold_unfam = kwags.get('threshold_unfam', getattr(mdc,'threshold_unfam',np.inf))
    zone = inputs[unfam_header_zone_index]
    return zone >= threshold_unfam

#TODO:EIMS_core transform
def inverse_transformKeras(mdc, inputs, use_fam=False, **kwags):
    exp_fd = LOGger.execute('exp_fd',mdc,kwags,default='.',not_found_alarm=False)
    addlog_ = LOGger.execute('addlog',mdc,kwags,default=LOGger.addloger(logfile=''),not_found_alarm=False)
    other_header_zone_indexes = []
    # inputs可能有不同的結構，但以下處理要同化成mylist
    if(isinstance(inputs, list)):
        if(inputs==[]):
            return None
        other_header_zone_indexes = getattr(mdc, 'other_header_zone_index', [mdc.unfam_header_zone_index] if(
            isinstance(getattr(mdc, 'unfam_header_zone_index', None), int)) else [-1])
        N = len(inputs)
        other_header_zone_indexes = [x%N for x in other_header_zone_indexes]
    elif(len(mdc.yheader_zones)>1):
        addlog_('len(mdc.yheader_zones)=%d  but predictions type=%s'%(len(mdc.yheader_zones), type(inputs)))
        dump_file = os.path.join(exp_fd, 'inverse_transformKeras_input_in_zone_error.pkl')
        joblib.dump(inputs, dump_file)
        addlog_('dump_file:', dump_file, stamps=['inverse_transformKeras'], colora=LOGger.FAIL)
        return None
    elif(isinstance(inputs, np.ndarray)):
        inputs = [inputs]
    else:
        dump_file = os.path.join(exp_fd, 'inverse_transform_input_in_zone_error.pkl')
        joblib.dump({'inputs':inputs}, dump_file)
        addlog_('dump_file:', dump_file, stamps=['inverse_transformKeras'], colora=LOGger.Fore.RED)
        return None
    
    dataIndexMask = None
    if(use_fam and getattr(mdc, 'unfam_header', [])):
        # 用熟悉度決定output是否有效
        dataIndexMask = np.logical_not(generate_unfamMask(mdc, inputs, **kwags))
        LOGger.addlog('[notice] use unfam!!', logfile='', colora=LOGger.WARNING)
    
    default_unfam_mdc = kwags.get('default_unfam', getattr(mdc, 'default_unfam', None))
    if(default_unfam_mdc is None):  default_unfam_mdc = np.nan
    column_index = 0
    np_input_zones = mylist()
    for i,(k,v) in enumerate(mdc.yheader_zones.items()):
        if(i in other_header_zone_indexes):
            continue
        input_in_zone = None
        # cell_size = getattr(v, 'cell_size', getattr(mdc, 'cell_size', None))
        if(isinstance(inputs, pd.core.frame.DataFrame)):
            input_in_zone = dcp(inputs[v])
        elif(len(getattr(inputs, 'shape',()))==2):
            input_in_zone = dcp(inputs[:,column_index:column_index+len(v)])
        elif(isinstance(inputs, list)):
            input_in_zone = dcp(inputs[i])
        else:
            joblib.dump(inputs, os.path.join(exp_fd, 'inverse_transform_inputs_error.pkl'))
            mdc.addlog('type error', type(inputs), stamps=['inverse_transformKeras', k], colora=LOGger.Fore.RED)
            return None
        if(np.array(input_in_zone).shape[0]==0):
            addlog('no predictions!!!!', stamps=[k])
            return None
        try:
            input_in_zone = v.inverse_transform(input_in_zone)
        except Exception as e:
            dump_file = os.path.join(exp_fd, 'inverse_transform_input_in_zone_error.pkl')
            addlog_('inverse transformation error', type(input_in_zone), dump_file, stamps=['inverse_transform', k], colora=LOGger.Fore.RED)
            LOGger.exception_process(e, logfile=os.path.join(exp_fd, 'log.txt'))
            joblib.dump({'inputs':inputs}, dump_file)
            return None
        np_inputs = np.array(input_in_zone)
        if(dataIndexMask is not None):
            default_unfam = dcp(kwags.get('default_unfam', getattr(v, 'default_unfam', default_unfam_mdc)))
            default_unfam = np.nan if(default_unfam is None) else default_unfam
            slices = (slice(None),) + (np.newaxis,)*(len(np_inputs)-1)
            dataIndexMaskInZone = dcp(dataIndexMask[slices])
            np_inputs = np.where(dataIndexMaskInZone, np_inputs, default_unfam)
        np_input_zones.append(np_inputs)
        column_index += len(v)
    return np_input_zones.get()

def inverse_transformKerasAutoEncoder(mdc, inputs, use_fam=False, header_zones_name='yheader_zones', **kwags):
    # inputs是predictFront之後的結果
    exp_fd = LOGger.execute('exp_fd',mdc,kwags,default='.',not_found_alarm=False)
    addlog_ = LOGger.execute('addlog',mdc,kwags,default=LOGger.addloger(logfile=''),not_found_alarm=False)
    other_header_zone_indexes = []
    header_zones = getattr(mdc, header_zones_name, LOGger.mydict({}))
    # inputs可能有不同的結構，但以下處理要同化成mylist
    if(isinstance(inputs, list)):
        if(inputs==[]):
            return None
        other_header_zone_indexes = getattr(mdc, 'other_header_zone_index', [mdc.unfam_header_zone_index] if(
            isinstance(getattr(mdc, 'unfam_header_zone_index', None), int)) else [-1])
        N = len(inputs)
        other_header_zone_indexes = [x%N for x in other_header_zone_indexes]
    elif(len(header_zones)>1):
        addlog_('len(mdc.%s)=%d  but predictions type=%s'%(header_zones_name, len(header_zones), type(inputs)))
        dump_file = os.path.join(exp_fd, 'inverse_transformKerasAutoEncoder_input_in_zone_error.pkl')
        joblib.dump(inputs, dump_file)
        addlog_('dump_file:', dump_file, stamps=['inverse_transformKerasAutoEncoder'], colora=LOGger.FAIL)
        return None
    elif(isinstance(inputs, np.ndarray)):
        inputs = [inputs]
    else:
        dump_file = os.path.join(exp_fd, 'inverse_transform_input_in_zone_error.pkl')
        joblib.dump({'inputs':inputs}, dump_file)
        addlog_('dump_file:', dump_file, stamps=['inverse_transformKerasAutoEncoder'], colora=LOGger.Fore.RED)
        return None
    
    dataIndexMask = None
    if(use_fam and getattr(mdc, 'unfam_header', [])):
        # 用熟悉度決定output是否有效
        dataIndexMask = np.logical_not(generate_unfamMask(mdc, inputs, **kwags))
        LOGger.addlog('[notice] use unfam!!', logfile='', colora=LOGger.WARNING)
    
    default_unfam_mdc = kwags.get('default_unfam', getattr(mdc, 'default_unfam', None))
    if(default_unfam_mdc is None):  default_unfam_mdc = np.nan
    column_index = 0
    np_input_zones = mylist()
    for i,(k,v) in enumerate(header_zones.items()):
        if(i in other_header_zone_indexes):
            continue
        input_in_zone = None
        # cell_size = getattr(v, 'cell_size', getattr(mdc, 'cell_size', None))
        if(isinstance(inputs, pd.core.frame.DataFrame)):
            input_in_zone = dcp(inputs[v])
        elif(len(getattr(inputs, 'shape',()))==2):
            input_in_zone = dcp(inputs[:,column_index:column_index+len(v)])
        elif(isinstance(inputs, list)):
            input_in_zone = dcp(inputs[i])
        else:
            joblib.dump(inputs, os.path.join(exp_fd, 'inverse_transform_inputs_error.pkl'))
            mdc.addlog('type error', type(inputs), stamps=['inverse_transformKerasAutoEncoder', k], colora=LOGger.Fore.RED)
            return None
        if(np.array(input_in_zone).shape[0]==0):
            addlog('no predictions!!!!', stamps=[k])
            return None
        try:
            input_in_zone = v.inverse_transform(input_in_zone)
        except Exception as e:
            dump_file = os.path.join(exp_fd, 'inverse_transform_input_in_zone_error.pkl')
            addlog_('inverse transformation error', type(input_in_zone), dump_file, stamps=['inverse_transformKerasAutoEncoder', k], colora=LOGger.Fore.RED)
            LOGger.exception_process(e, logfile=os.path.join(exp_fd, 'log.txt'))
            joblib.dump({'inputs':inputs}, dump_file)
            return None
        np_inputs = np.array(input_in_zone)
        if(dataIndexMask is not None):
            default_unfam = dcp(kwags.get('default_unfam', getattr(v, 'default_unfam', default_unfam_mdc)))
            default_unfam = np.nan if(default_unfam is None) else default_unfam
            slices = (slice(None),) + (np.newaxis,)*(len(np_inputs)-1)
            dataIndexMaskInZone = dcp(dataIndexMask[slices])
            np_inputs = np.where(dataIndexMaskInZone, np_inputs, default_unfam)
        np_input_zones.append(np_inputs)
        column_index += len(v)
    return np_input_zones.get()

def inverse_transformKerasAtReducing(mdc, inputs, use_fam=False, **kwags):
    return inverse_transformKerasAutoEncoder(mdc, inputs, use_fam, header_zones_name='xheader_zones', **kwags)

def inverse_transformKerasAtRegenerating(mdc, inputs, use_fam=False, **kwags):
    return inverse_transformKerasAutoEncoder(mdc, inputs, use_fam, header_zones_name='yheader_zones', **kwags)

#TODO:EIMS_core transformSklearn 未來，面對時序資料，要能夠平面化
def transformSklearn(mdc, inputs, **kwags):
    exp_fd = LOGger.execute('exp_fd',mdc,kwags,default='.',not_found_alarm=False)
    addlog_ = LOGger.execute('addlog',mdc,kwags,default=LOGger.addloger(logfile=''),not_found_alarm=False)
    inputsHasZones = isinstance(inputs, list)
    if(inputsHasZones):
        if(len(inputs) != len(mdc.xheader_zones)):
            addlog_('n_xHeaderZones in mdc:', len(mdc.xheader_zones), '!= n_zones in inputs:', len(inputs), colora=LOGger.FAIL)
            mdc.addlog('transformation input invalid', type(inputs), stamps=['transformKeras'], colora=LOGger.FAIL)
            m_debug.updateDump({transformKeras.__name__:{'inputs':inputs, 'inputsHasZones':inputsHasZones, 
                                                         'mdc.xheader_zones.keys':mdc.xheader_zones.keys()}})
            return None
    np_input_zones = mylist()
    for i,(k,v) in enumerate(mdc.xheader_zones.items()):
        stamps = ['transformKeras', k]
        try:
            inputs_transed = v.transform(inputs, stamps=stamps, addlog=addlog_, exp_fd=mdc.exp_fd,
                                         axisIndex=mdc.xheader.index(v[0]), zoneIndex=i)
            if(inputs_transed is None):
                return None
            inputs_transed = mdc.convert4modelCore(inputs_transed)
        except Exception as e:
            mdc.addlog('transformation error', type(inputs.get(i)), stamps=stamps, colora=LOGger.Fore.RED)
            LOGger.exception_process(e, logfile=os.path.join(exp_fd, 'log.txt'))
            m_debug.updateDump({transformKeras.__name__:{'inputs':inputs, 'inputs_transed':inputs_transed, 'i':i, 'v':DFP.parse(v)}})
        np_inputs = np.array(inputs_transed)
        np_input_zones.append(np_inputs)
    return np_input_zones.get()

#TODO:EIMS_core inverse_transformSklearn 未來，面對時序資料，要能夠平面化
def inverse_transformSklearn(mdc, inputs, **kwags):
    exp_fd = LOGger.execute('exp_fd',mdc,kwags,default='.',not_found_alarm=False)
    addlog_ = LOGger.execute('addlog',mdc,kwags,default=LOGger.addloger(logfile=''),not_found_alarm=False)
    other_header_zone_indexes = []
    # inputs可能有不同的結構，但以下處理要同化成mylist
    if(isinstance(inputs, list)):
        if(inputs==[]):
            return None
        other_header_zone_indexes = getattr(mdc, 'other_header_zone_index', [mdc.unfam_header_zone_index] if(
            isinstance(getattr(mdc, 'unfam_header_zone_index', None), int)) else [-1])
        N = len(inputs)
        other_header_zone_indexes = [x%N for x in other_header_zone_indexes]
    elif(len(mdc.yheader_zones)>1):
        addlog_('len(mdc.yheader_zones)=%d  but predictions type=%s'%(len(mdc.yheader_zones), type(inputs)))
        dump_file = os.path.join(exp_fd, 'inverse_transformKeras_input_in_zone_error.pkl')
        joblib.dump(inputs, dump_file)
        addlog_('dump_file:', dump_file, stamps=['inverse_transformKeras'], colora=LOGger.FAIL)
        return None
    elif(isinstance(inputs, np.ndarray)):
        inputs = [inputs]
    else:
        dump_file = os.path.join(exp_fd, 'inverse_transform_input_in_zone_error.pkl')
        joblib.dump({'inputs':inputs}, dump_file)
        addlog_('dump_file:', dump_file, stamps=['inverse_transformKeras'], colora=LOGger.Fore.RED)
        return None
    
    dataIndexMask = None
    # if(use_fam and getattr(mdc, 'unfam_header', [])):
    #     # 用熟悉度決定output是否有效
    #     dataIndexMask = np.logical_not(generate_unfamMask(mdc, inputs, **kwags))
    #     LOGger.addlog('[notice] use unfam!!', logfile='', colora=LOGger.WARNING)
    
    default_unfam_mdc = kwags.get('default_unfam', getattr(mdc, 'default_unfam', None))
    if(default_unfam_mdc is None):  default_unfam_mdc = np.nan
    column_index = 0
    np_input_zones = mylist()
    for i,(k,v) in enumerate(mdc.yheader_zones.items()):
        if(i in other_header_zone_indexes):
            continue
        input_in_zone = None
        # cell_size = getattr(v, 'cell_size', getattr(mdc, 'cell_size', None))
        if(isinstance(inputs, pd.core.frame.DataFrame)):
            input_in_zone = dcp(inputs[v])
        elif(len(getattr(inputs, 'shape',()))==2):
            input_in_zone = dcp(inputs[:,column_index:column_index+len(v)])
        elif(isinstance(inputs, list)):
            input_in_zone = dcp(inputs[i])
        else:
            joblib.dump(inputs, os.path.join(exp_fd, 'inverse_transform_inputs_error.pkl'))
            mdc.addlog('type error', type(inputs), stamps=['inverse_transformKeras', k], colora=LOGger.Fore.RED)
            return None
        LOGger.addDebug(k, 'input_in_zone', input_in_zone)
        if(np.array(input_in_zone).shape[0]==0):
            addlog('no predictions!!!!', stamps=[k])
            return None
        try:
            input_in_zone = v.inverse_transform(input_in_zone)
        except Exception as e:
            dump_file = os.path.join(exp_fd, 'inverse_transform_input_in_zone_error.pkl')
            addlog_('inverse transformation error', type(input_in_zone), dump_file, stamps=['inverse_transform', k], colora=LOGger.Fore.RED)
            LOGger.exception_process(e, logfile=os.path.join(exp_fd, 'log.txt'))
            joblib.dump({'inputs':inputs}, dump_file)
            return None
        np_inputs = np.array(input_in_zone)
        if(dataIndexMask is not None):
            default_unfam = dcp(kwags.get('default_unfam', getattr(v, 'default_unfam', default_unfam_mdc)))
            default_unfam = np.nan if(default_unfam is None) else default_unfam
            slices = (slice(None),) + (np.newaxis,)*(len(np_inputs)-1)
            dataIndexMaskInZone = dcp(dataIndexMask[slices])
            np_inputs = np.where(dataIndexMaskInZone, np_inputs, default_unfam)
        np_input_zones.append(np_inputs)
        column_index += len(v)
    return np_input_zones.get()

def executeFront(mdc, inputs, coreMethod='predict_core', **kwags):
    """
    transform + coreMethod

    Parameters
    ----------
    mdc : TYPE
        DESCRIPTION.
    inputs : TYPE
        DESCRIPTION.
    **kwags : TYPE
        DESCRIPTION.

    Returns
    -------
    output_raw : TYPE
        DESCRIPTION.

    """
    exp_fd = LOGger.execute('exp_fd',mdc,kwags,default='.',not_found_alarm=False)
    addlog_ = LOGger.execute('addlog',mdc,kwags,default=LOGger.addloger(logfile=''),not_found_alarm=False)
    if(getattr(mdc, 'model', None)==None):
        addlog_('model not set up!!!', stamps=mdc.get_stamps(full=True),logfile='',colora=LOGger.FAIL)
        return None
    inputs_trsed = mdc.transform(inputs)
    if(isinstance(inputs_trsed, type(None))):
        debugFile = os.path.join(exp_fd, 'predict_inputs_trsed_error.pkl')
        joblib.dump({'inputs':inputs, 'inputs_trsed':inputs_trsed}, debugFile)
        addlog_('inputs_trsed None', debugFile, stamps=mdc.get_stamps()+['EIMS core predict'],colora=LOGger.FAIL)
        return None
    try:
        output_raw = getattr(mdc, coreMethod)(inputs_trsed, return_probability=kwags.get('return_probability', False))
    except Exception as e:
        debugFile = os.path.join(exp_fd, 'predict_inputs_trsed_error.pkl')
        addlog_('predict error', type(inputs_trsed), debugFile, stamps=mdc.get_stamps()+['EIMS core predict'],colora=LOGger.FAIL)
        LOGger.exception_process(e, logfile=os.path.join(exp_fd, 'log.txt'))
        joblib.dump({'inputs':inputs, 'inputs_trsed':inputs_trsed}, debugFile)
        return None
    return output_raw

def predictFront(mdc, inputs, **kwags):
    kwags['coreMethod'] = 'predict_core'
    return executeFront(mdc, inputs, **kwags)

def latentScoreFront(mdc, inputs, **kwags):
    kwags['coreMethod'] = 'latentScore_core'
    return executeFront(mdc, inputs, **kwags)

def regenerateFront(mdc, inputs, **kwags):
    kwags['coreMethod'] = 'regenerate_core'
    return executeFront(mdc, inputs, **kwags)

def regenerate(mdc, inputs, use_fam=False, **kwags):
    """
    mdc prediction

    Parameters
    ----------
    mdc : TYPE
        DESCRIPTION.
    inputs : TYPE
        DESCRIPTION.
    use_fam : TYPE, optional
        True:answer selecting by unfam; False:otherwise. The default is False.
    default_unfam : TYPE, optional
        DESCRIPTION. The default is np.nan.
    threshold_unfam : TYPE, optional
        DESCRIPTION. The default is None.
    **kwags : TYPE
        DESCRIPTION.

    Returns
    -------
    outputs : TYPE
        DESCRIPTION.

    """
    outputs = executeFront(mdc, inputs, coreMethod='regenerate_core', **kwags)
    if(isinstance(outputs, type(None))):
        return None
    outputs = mdc.inverse_transform(outputs, use_fam=use_fam, header_zones_name='xheader_zones', **kwags)
    return outputs

#TODO:abc_MODELHOST predict
def predict(mdc, inputs, use_fam=False, **kwags):
    """
    mdc prediction

    Parameters
    ----------
    mdc : TYPE
        DESCRIPTION.
    inputs : TYPE
        DESCRIPTION.
    use_fam : TYPE, optional
        True:answer selecting by unfam; False:otherwise. The default is False.
    default_unfam : TYPE, optional
        DESCRIPTION. The default is np.nan.
    threshold_unfam : TYPE, optional
        DESCRIPTION. The default is None.
    **kwags : TYPE
        DESCRIPTION.

    Returns
    -------
    outputs : TYPE
        DESCRIPTION.

    """
    # # 添加調試代碼
    # m_print = getattr(mdc, 'addlog', LOGger.addloger(logfile=''))
    # m_print(f"[DEBUG] predict 輸入: type={type(inputs)}, shape={getattr(inputs, 'shape', 'N/A')}", stamps=['modeling_core'])
    # if hasattr(inputs, 'dtype'):
    #     m_print(f"[DEBUG] predict 輸入 dtype: {inputs.dtype}", stamps=['modeling_core'])
    # if hasattr(inputs, '__len__') and len(inputs) > 0:
    #     m_print(f"[DEBUG] predict 輸入前3行: {inputs[:3] if hasattr(inputs, '__getitem__') else 'N/A'}", stamps=['modeling_core'])
    
    output_raw = predictFront(mdc, inputs, **kwags)
    if(isinstance(output_raw, type(None))):
        return None
    outputs = mdc.inverse_transform(output_raw, use_fam=use_fam, **kwags)
    return outputs

def predict_score(mdc, inputs, return_proba = False, zone_index=0, 
                  multi_dim_method=lambda d:np.max(d, axis=1), 
                  probabilities_method=lambda a:1 - sigmoid(np.array(a), centre=0.5, score_max=1, score_min=0), 
                  **kwags):
    # 把原生的output_raw呈現出來
    output_raw = predictFront(mdc, inputs, **kwags)
    explanation_axis = getattr(mdc, 'explanation_axis', None)
    #TODO:explanation_score
    outputs = explanation_score(output_raw, zone_index=zone_index, multi_dim_method=multi_dim_method, return_proba=return_proba,
                               probabilities_method=probabilities_method)
    return outputs

def predict_AElatentScore(mdc, inputs, return_proba = False, zone_index=0, 
                  multi_dim_method=lambda d:np.max(d, axis=1), 
                  probabilities_method=lambda a:1 - sigmoid(np.array(a), centre=0.5, score_max=1, score_min=0), 
                  **kwags):
    # 把原生的output_raw呈現出來
    output_raw = latentScoreFront(mdc, inputs, **kwags)
    explanation_axis = getattr(mdc, 'explanation_axis', None)
    #TODO:explanation_score
    outputs = explanation_score(output_raw, zone_index=zone_index, multi_dim_method=multi_dim_method, return_proba=return_proba,
                               probabilities_method=probabilities_method)
    return outputs

def predict_confident(mdc, inputs, zone_index=-1, **kwags):
    # 把原生的output_raw轉成機率
    output_raw = predictFront(mdc, inputs, **kwags)
    outputs = explanation_confidence(output_raw, zone_index=zone_index, **kwags)
    return outputs   

def predict_confidence(mdc, data, **kwags):
    unfam_header_zone_index = getattr(mdc, 'unfam_header_zone_index', -1)
    threshold_unfam = getattr(mdc, 'threshold_unfam', 0.1)
    kwags['zone_index'] = unfam_header_zone_index
    kwags['return_proba'] = True
    kwags['probabilities_method'] = lambda a:np.clip(1 - sigmoid(np.array(a), centre=threshold_unfam, score_min=0.0, score_max=2*threshold_unfam), 0.0, 1.0)
    return predict_score(mdc, data, **kwags)

def predict_AElatentConfidence(mdc, data, **kwags):
    # unfam_header_zone_index = getattr(mdc, 'unfam_header_zone_index', -1)
    threshold_unfam = getattr(mdc, 'threshold_unfam_latentSpace', 0.1)
    kwags['zone_index'] = 0 #unfam_header_zone_index
    kwags['return_proba'] = True
    LOGger.addDebug('threshold_unfam', str(threshold_unfam))
    kwags['probabilities_method'] = lambda a:np.clip(1 - sigmoid(np.array(a), centre=threshold_unfam, score_min=0.0, score_max=2*threshold_unfam), 0.0, 1.0)
    return predict_AElatentScore(mdc, data, **kwags)

def predict_confidenceScore(mdc, data, **kwags):
    unfam_header_zone_index = getattr(mdc, 'unfam_header_zone_index', -1)
    kwags['zone_index'] = unfam_header_zone_index
    kwags['return_proba'] = True
    kwags['probabilities_method'] = lambda a:a
    return predict_score(mdc, data, **kwags)

def callback_process(ware_callbacks, loss_digit=5, callback_freq=10, lossCurve_callback=None, periodic_log_callback=None, **kwags):
    callback_basket = [
            EarlyStopping(**kwags),
            callbacks.ModelCheckpoint(
                filepath=os.path.join(ware_callbacks, 
                        'sc[{loss:.%df}]-crep[{epoch:03d}].h5'%loss_digit),
                save_freq=callback_freq),
            lossCurve_callback,
            periodic_log_callback]
    
    CreateContainer(ware_callbacks)
    return callback_basket

def callback_model(def_sc, ware_callbacks, model=None, custom_objects=None, **kwags):
    addlog = kwags.get('addlog', LOGger.addlog)
    if(os.path.isfile(ware_callbacks)):
        addlog('路徑不存在:%s'%(ware_callbacks))
        return model
    for croot, cdirs, cfiles in os.walk(ware_callbacks, topdown=False):
        for h5file in cfiles:
            that_sc = float(h5file[h5file.find('sc[')+3:h5file.find(']-')])
            if(str(that_sc)!='nan' and that_sc<def_sc):
                addlog('[%.2f→%.2f]更換model:%s'%(
                        def_sc, that_sc, os.path.join(croot, h5file)))
                def_sc = float(that_sc)
                model = models.load_model(
                        os.path.join(croot, h5file), custom_objects=get_custom_objects())
    return model

def cleaning_callbacks(ware_callbacks, sc_thd, save_file_count=5):
    filelist, filescores = [], []
    for croot, cdirs, cfiles in os.walk(ware_callbacks, topdown=False):
        for h5file in cfiles:
            that_sc = float(h5file[h5file.find('sc[')+3:h5file.find(']-')])
            filescores.append(that_sc)
            filelist.append(h5file)
            if(that_sc>=sc_thd):
                os.remove(os.path.join(croot, h5file))
    if(save_file_count!=None):
        afilelist = list(filelist)
        filelist.sort(key = lambda f:filescores[afilelist.index(f)])
        for removing_file in filelist[save_file_count:]:
            if(os.path.isfile(os.path.join(ware_callbacks, removing_file))):
                os.remove(os.path.join(ware_callbacks, removing_file))

def export_model_detail_keras(mdc, exp_fd='', rewrite=True, deep=True, **kwags):
    if(str(type(getattr(mdc, 'model', None))).find('keras')==-1):
        return 
    exp_fd = mdc.get_module_exp_fd()
    fname = '%s.txt'%stamp_process('',['model_summary']+mdc.get_stamps(full=1),'','','','_',for_file=1)
    file = os.path.join(exp_fd, fname) if(rewrite) else DFP.pathrpt(
                                                        os.path.join(exp_fd, fname))
    CreateFile(file, lambda f:create_model_detail_keras_txt(mdc.model, logfile=f, deep=deep))
    return True
        
def export_model_detail_json(mdc, stamps=None, rewrite=True, deep=True, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    exp_fd = mdc.get_module_exp_fd()
    stamps = stamps if(isinstance(stamps, list)) else []
    addlog('export_model_detail_json...', stamps=stamps)
    file = os.path.join(exp_fd, '%s.pkl'%LOGger.stamp_process('', ['model_summary', *stamps],'','','','_'))
    if(not rewrite):    file = DFP.pathrpt(file)
    CreateFile(file, lambda f:(DFP.joblib.dump(mdc.algorithm_params, f)))
    return True
    
#TODO:export_models_leaderboard
def export_models_leaderboard(mdc, stamps=None, rewrite=True, detailed=True, model_object='', deepParams=True, theme='', **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    exp_fd = mdc.exp_fd
    stamps = stamps if(isinstance(stamps, list)) else []
    
    
    # 取得所有訓練過的 estimator 和它們的權重
    model = mdc.model
    WM = model.get_models_with_weights()
    npWM = np.array(WM)
    npWM = npWM[:,[1,0]]
    npWM = npWM[np.argsort(npWM[:,1])[::-1]]
    WMS = dcp(npWM)
    def algorithmParse(x):
        try:
            return LOGger.type_string(getattr(x.named_steps['classifier'],'choice',x.named_steps['classifier']))
        except:
            return DFP.parse(x)
    WMS[:,0] = np.array(tuple(map(algorithmParse, npWM[:,0])))
    fig = vs3.plt.Figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    ax.bar(np.arange(WMS.shape[0]), WMS[:,1], label=LOGger.stamp_process('',['rel', str(model.automl_._metric)],'','','',' '))
    ax.set_xticks(np.arange(WMS.shape[0]))
    ax.set_xticklabels(WMS[:,0], rotation=90)
    file = os.path.join(mdc.get_module_exp_fd(), 'EstiWeightsInEnsamble.jpg')
    CreateFile(file, lambda f:vs3.end(fig, file=f))
    
    paramSerials = {}
    mdc.algorithm_params.clear()
    try:
        for i,pipe in enumerate(npWM[:,0]):
            est = getattr(pipe.named_steps['classifier'],'choice',pipe.named_steps['classifier'])
            key = LOGger.type_string(est)
            estParam = dcp(est.get_params(deep=deepParams))
            estParam['estimatorSerial'] = dcp(key)
            mdc.algorithm_params[i] = dcp(estParam)
            paramSerials[i] = {DFP.parse(k):DFP.parse(v) for k,v in estParam.items()}
        file = os.path.join(mdc.get_module_exp_fd(), 'algorithm_params.pkl')
        CreateFile(file, lambda f:DFP.joblib.dump(mdc.algorithm_params, f))
        DFP.save(pd.DataFrame(paramSerials).T, exp_fd=mdc.get_module_exp_fd(), fn='algorithm_params', save_types=['xlsx'])
    except Exception as e:
        LOGger.exception_process(e,logfile='',stamps=['export_models_leaderboard::algorithm_params'])
    
    addlog('export_model_detail_json...', stamps=stamps)
    pd_leaderboard = mdc.leaderboard
    file = os.path.join(exp_fd, '%s.csv'%stamp_process('',['leaderboard', *stamps],'','','','_'))
    if(not rewrite):    file = DFP.pathrpt(file)
    CreateFile(file, lambda f:pd_leaderboard.to_csv(f))
    show_models = mdc.model.show_models()
    ensamble = {k:{kk:DFP.parse(vv, 4, stg_max_length=15, be_instinct=True) for kk,vv in v.items()} for k,v in show_models.items()}
    try:
        pd_ensamble = pd.DataFrame(ensamble).T
        title = stamp_process('',['leaderboard', *stamps],'','','','_')
        file = os.path.join(exp_fd, '%s.jpg'%stamp_process('',['leaderboard', *stamps],'','','','_'))
        if(not rewrite):    file = DFP.pathrpt(file)
        CreateFile(file, lambda f: vs.matrix_dataframe(pd_ensamble, title=title, file=f, index=pd_ensamble.index, header=pd_ensamble.columns, 
                                             headerhide=False, indexhide=False))
    except Exception as e:
        mdc.addlog('export_models_leaderboard failed!!!', colora=LOGger.FAIL)
        exception_process(e, os.path.join(mdc.exp_fd, 'log.txt'), stamps=['export_models_leaderboard::leaderboard'])
    file = os.path.join('%s.json'%stamp_process('',['leaderboard', *stamps],'','','','_')) 
    if(not rewrite):    file = DFP.pathrpt(file)
    CreateFile(file, lambda f:(save_json(ensamble, file=f)))
    return True

def stackOutputDenseStandard(layer, headerZone, stamps=None, defaultCellSize=None, **kwags):
    stamps = stamps if(isinstance(stamps, list)) else []
    stamp = LOGger.stamp_process('',stamps,'','','','_')
    defaultActivation = getattr(headerZone,'activation','selu')
    default_layer = parseKL('Dense')
    cell_size = getattr(headerZone, 'cell_size', defaultCellSize)
    hidden_layer_sizes = getattr(headerZone, 'hidden_layer_sizes', None)
    hidden_layer_sizes = hidden_layer_sizes if(DFP.isiterable(hidden_layer_sizes)) else []
    hidden_layer_sizes = LOGger.mylist(reversed(hidden_layer_sizes)) #因為是AutoEncoder的output，所以需要反轉
    output_shape = getattr(headerZone, 'output_shape', ((len(headerZone),) if(cell_size is None) else (
            cell_size if(DFP.isiterable(cell_size)) else (cell_size, len(headerZone)))))
    output_flatten_shape = np.product(output_shape)
    for (i,s) in enumerate(hidden_layer_sizes):
        activation = getattr(headerZone,'activation',LOGger.extract(
                getattr(headerZone, 'activations', []), index=i, key=stamp, 
                default=getattr(headerZone, 'activation', 'relu')))
        layer = default_layer(units=output_flatten_shape, activation=activation, 
                            name=stamp_process('',[stamp,i],'','','','_'))(layer)
    LOGger.addDebug('stackOutputDenseStandard layer.shape', layer.shape, stamps=stamps)
    if(len(output_shape)<=1 and layer.shape[1:]!=tuple(output_shape)):
        output_zone = parseKL('Dense')(units=output_shape[0], activation=defaultActivation, 
                                name=stamp_process('',[stamp],'','','','_'))(layer)
    elif(len(output_shape)<=1 and layer.shape[1:]==tuple(output_shape)):
        output_zone = layer
    elif(len(layer.shape)==2 and isinstance(output_shape,int) and layer.shape[1]==output_shape):
        output_zone = layer
    elif(len(layer.shape)==2 and isinstance(output_shape,int) and layer.shape[1]!=output_shape):
        layer = parseKL('Flatten')()(layer)
        layer = parseKL('Dense')(units=output_flatten_shape, activation=defaultActivation,
                                name=stamp_process('',[stamp,'Dense'],'','','','_'))(layer)
        output_zone = parseKL('Reshape')(target_shape=output_shape, 
                                        name=stamp_process('',[stamp,'Reshape'],'','','','_'))(layer)
    elif(len(layer.shape)-1!=len(output_shape)):
        layer = parseKL('Flatten')()(layer)
        output_zone = parseKL('Reshape')(target_shape=output_shape, 
                                        name=stamp_process('',[stamp,'Reshape'],'','','','_'))(layer)
    else:
        m_print('undefined layer.shape:%s'%str(getattr(layer,'shape',type(layer))), 
                'while output_shape:%s'%str(output_shape), stamps=stamps, colora=LOGger.FAIL)
        sys.exit(1)
    return output_zone

def stackOutputConv1dStandard(layer, headerZone, defaultCellSize=None, 
                            stamps=None, hiddenLayerIndex=0, **kwags):
    stamps = stamps if(isinstance(stamps, list)) else []
    stamp = LOGger.stamp_process('',stamps,'','','','_')
    defaultActivation = getattr(headerZone,'activation','selu')
    strideses = getattr(headerZone, 'strideses', None)
    strideses = LOGger.mylist(tuple(strideses) if(DFP.isiterable(strideses) and strideses) else [])
    kernel_sizes = getattr(headerZone, 'kernel_sizes', None)
    kernel_sizes = LOGger.mylist(tuple(kernel_sizes) if(DFP.isiterable(kernel_sizes) and kernel_sizes) else [])
    strideseFlatten = int(np.prod(strideses) if(DFP.isiterable(strideses) and strideses) else 1)
    hidden_layer_sizes = getattr(headerZone, 'hidden_layer_sizes', None)
    LOGger.addDebug('hidden_layer_sizes', hidden_layer_sizes, stamps=stamps)
    hidden_layer_sizes = tuple(hidden_layer_sizes) if(DFP.isiterable(hidden_layer_sizes)) else []
    hidden_layer_sizes = LOGger.mylist(reversed(hidden_layer_sizes)) #因為是AutoEncoder的output，所以需要反轉
    LOGger.addDebug('hidden_layer_sizes', hidden_layer_sizes, stamps=stamps)
    n_hidden_layer_sizes = len(hidden_layer_sizes)
    default_layer = parseKL('Conv1DTranspose')
    cell_size = getattr(headerZone, 'cell_size', defaultCellSize)
    output_shape = getattr(headerZone, 'output_shape', ((len(headerZone),) if(cell_size is None) else (
        cell_size if(DFP.isiterable(cell_size)) else (cell_size, len(headerZone)))))
    output_shape = LOGger.mylist(tuple(output_shape))
    output_flatten_shape = np.product(output_shape)
    LOGger.addDebug('hidden_layer_sizes', hidden_layer_sizes, stamps=stamps)
    i=None
    # preparing the prehead hidden layer
    if(hidden_layer_sizes):
        layer = parseKL('Dense')(
            units=int(output_shape[0]//strideseFlatten) * hidden_layer_sizes[0], activation=defaultActivation, 
            name=stamp_process('',['preOutput_flatten',stamp],'','','','_'))(layer)
        layer = parseKL('Reshape')(
            target_shape=(int(output_shape[0]//strideseFlatten), hidden_layer_sizes[0]), 
            name=stamp_process('',['preOutput_flattenReshape',stamp],'','','','_'))(layer)
        for (i,s) in enumerate(hidden_layer_sizes):
            activation = getattr(headerZone,'activation',LOGger.extract(
                    getattr(headerZone, 'activations', []), index=i, key=stamp, default=defaultActivation))
            layer = default_layer(filters=s, 
                                kernel_size=kernel_sizes.get(i,3), 
                                strides=strideses.get(i,1), 
                                activation=activation, 
                                name=stamp_process('',[stamp, i],'','','','_'))(layer)
    if(len(output_shape)<=1 and layer.shape[1:]==tuple(output_shape)):
        output_zone = layer
    elif(len(output_shape)<=1 and layer.shape[1:]!=tuple(output_shape)):
        layer = parseKL('Flatten')()(layer)
        output_zone = parseKL('Dense')(units=output_shape[0], activation=defaultActivation, 
                                name=stamp_process('',[stamp],'','','','_'))(layer)
    elif(len(output_shape)==2):
        output_zone = parseKL('Conv1D')(filters=output_shape.get(1,1), 
                                    kernel_size=kernel_sizes.get(i,3), #1, #kernel_sizes.get(i,3), 
                                    strides=strideses.get(i,1), 
                                    activation=defaultActivation, 
                                    name=stamp_process('',[stamp],'','','','_'))(layer)
    else:
        m_print('undefined layer.shape:%s'%str(getattr(layer,'shape',type(layer))), 
                'while output_shape:%s'%str(output_shape), stamps=stamps, colora=LOGger.FAIL)
        sys.exit(1)
    return output_zone

def stackOutputLSTMStandard(layer, headerZone, defaultCellSize=None, 
                            stamps=None, hiddenLayerIndex=0, **kwags):
    stamps = stamps if(isinstance(stamps, list)) else []
    stamp = LOGger.stamp_process('',stamps,'','','','_')
    defaultActivation = getattr(headerZone,'activation','selu')
    kernel_sizes = getattr(headerZone, 'kernel_sizes', None)
    kernel_sizes = LOGger.mylist(tuple(kernel_sizes) if(DFP.isiterable(kernel_sizes) and kernel_sizes) else [])
    hidden_layer_sizes = getattr(headerZone, 'hidden_layer_sizes', None)
    LOGger.addDebug('hidden_layer_sizes', hidden_layer_sizes, stamps=stamps)
    hidden_layer_sizes = tuple(hidden_layer_sizes) if(DFP.isiterable(hidden_layer_sizes)) else []
    hidden_layer_sizes = LOGger.mylist(reversed(hidden_layer_sizes)) #因為是AutoEncoder的output，所以需要反轉
    LOGger.addDebug('hidden_layer_sizes', hidden_layer_sizes, stamps=stamps)
    n_hidden_layer_sizes = len(hidden_layer_sizes)
    default_layer = parseKL('LSTM')
    cell_size = getattr(headerZone, 'cell_size', defaultCellSize)
    output_shape = getattr(headerZone, 'output_shape', ((len(headerZone),) if(cell_size is None) else (
        cell_size if(DFP.isiterable(cell_size)) else (cell_size, len(headerZone)))))
    output_shape = LOGger.mylist(tuple(output_shape))
    output_flatten_shape = np.product(output_shape)
    LOGger.addDebug('hidden_layer_sizes', hidden_layer_sizes, stamps=stamps)
    # preparing the prehead hidden layer
    if(hidden_layer_sizes):
        layer = parseKL('RepeatVector')(output_shape[0])(layer)
    for (i,s) in enumerate(hidden_layer_sizes):
        activation = getattr(headerZone,'activation',LOGger.extract(
                getattr(headerZone, 'activations', []), index=i, key=stamp, default=defaultActivation))
        layer = default_layer(s, return_sequences=True, activation=activation,
                            name=stamp_process('',[stamp, i],'','','','_'))(layer)
    if(len(output_shape)==2):
        output_zone = parseKL('TimeDistributed')(parseKL('Dense')(units=output_shape.get(1,1), 
                                    activation=defaultActivation, 
                                    name=stamp_process('',[stamp],'','','','_')))(layer)
    else:
        m_print('undefined layer.shape:%s'%str(getattr(layer,'shape',type(layer))), 
                'while output_shape:%s'%str(output_shape), stamps=stamps, colora=LOGger.FAIL)
        sys.exit(1)
    return output_zone

def stackOutputStandard(layer, headerZone, defaultLayerStyleName='Dense', **kwags):
    if(defaultLayerStyleName=='Conv1D'):
        return stackOutputConv1dStandard(layer, headerZone, **kwags)
    elif(defaultLayerStyleName=='LSTM'):
        return stackOutputLSTMStandard(layer, headerZone, **kwags)
    else:
        return stackOutputDenseStandard(layer, headerZone, **kwags)

def compileStandard(mdc, **kwags):
    addlog_ = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    try:
        layer_index = 0
        keras_input, keras_input_preconcatenate = [], []
        #input layers
        cell_size_mdc = dcp(getattr(mdc, 'cell_size', None))
        for k,v in mdc.xheader_zones.items():
            cell_size = getattr(v, 'cell_size', cell_size_mdc)
            input_shape = getattr(v, 'input_shape', (len(v) if(cell_size is None) else (
               cell_size if(DFP.isiterable(cell_size)) else (cell_size, len(v)))))
            input_zone = parseKL('Input')(shape = input_shape, name=stamp_process('',['input',k],'','','','_'))
            preprocessing = mylist([dcp(getattr(v, 'preprocessing', None))]).get_all()
            if('encoding' in preprocessing):
                latentDim = getattr(v,'classEmbeddingLatentDim',getattr(mdc,'classEmbeddingLatentDim',2))
                embeddingLayer = parseKL('Embedding')(np.prod(len(v.preprocessor.classes_)+1), latentDim, input_length=cell_size)
                input_zone_preconcatenate = embeddingLayer(input_zone)
                v.latentLayerName = dcp(embeddingLayer.name)
                print(v.latentLayerName)
                input_zone_preconcatenate = parseKL('Reshape')(target_shape=(latentDim,))(input_zone_preconcatenate)
            else:
                input_zone_preconcatenate = input_zone
            if(cell_size is not None):
                layer_index_v = 0
                input_zone_preconcatenate, layer_index_v = ALG.stack_layers(
                    layer_index_v, input_zone_preconcatenate, activation=mdc.activation, 
                    default_layer_name = getattr(v, 'layer_type', 'LSTM'),
                    hidden_layer_sizes=getattr(v, 'hidden_layer_sizes', (2,)), 
                    hidden_layer_nns=getattr(v, 'hidden_layer_nns', {}),
                    dropout_rates = getattr(v, 'dropout_rates', None),
                    addlog = addlog_, stamps=['input_preprocess'])
            keras_input.append(input_zone) 
            keras_input_preconcatenate.append(input_zone_preconcatenate)
        addlog_('keras_input:%s'%stamp_process('', stamps = list(map(str, keras_input)), 
                                               stamp_left='\n', stamp_right=''), **kwags)
        mdc.keras_input = mylist(keras_input).get()
        inputs = parseKL('concatenate')(
            keras_input_preconcatenate) if(len(keras_input)>1) else keras_input_preconcatenate[0]
        #hidden layers
        addlog_('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
        layer, layer_index = ALG.stack_layers(layer_index, inputs, activation=mdc.activation, 
                    default_layer_name = getattr(mdc, 'layer_type', 'Dense'),
                    hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                    dropout_rates = getattr(mdc, 'dropout_rates', None),
                    addlog = addlog_)
        if(layer==None):
            return None
        layer_index+=1
        #output layers
        keras_output = []
        lossmethods, output_activations = mylist(), mylist()
        for i,(k,v) in enumerate(list(tuple(mdc.yheader_zones.items()))):
            opt_sz = len(v)
            default_layer = parseKL(getattr(v, 'layer_type', 'Dense'))
            activation = getattr(v,'activation',LOGger.extract(
                    getattr(mdc, 'activations', []), index=i, key=k, default=getattr(mdc, 'activation', 'relu')))
            output_activations.append(activation)
            output_zone = default_layer(units=opt_sz, activation=activation, 
                                    name=stamp_process('',['output',k],'','','','_'))(layer)
            lossmethods.append(v.lossmethod)
            keras_output.append(output_zone)
        for i,(k,v) in enumerate(getattr(mdc, 'unfam_header_zones', {}).items()):
            opt_sz = len(v)
            default_layer = parseKL('Dense')
            activation = getattr(v,'activation',LOGger.extract(
                    getattr(mdc, 'activations', []), index=i, key=k, default=getattr(mdc, 'activation', 'sigmoid')))
            output_activations.append(activation)
            unfam_zone = default_layer(units=opt_sz, activation=activation, 
                                    name=stamp_process('',['unfam',k],'','','','_'))(layer)
            lossmethods.append(v.lossmethod)
            keras_output.append(unfam_zone)
        addlog('keras_output:%s'%stamp_process('', stamps = list(map(str, keras_output)), 
                                                   stamp_left='\n', stamp_right=''), **kwags)
        mdc.lossmethods = lossmethods
        addlog('operating lossmethods:%s'%(str(mdc.lossmethods.get())), **kwags)
        mdc.output_activations = output_activations
        addlog('operating output_activations:%s'%(str(output_activations.get())), **kwags)
        mdc.keras_output = keras_output if(len(keras_output)>1) else keras_output[0]
        model = keras.Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                      name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    except Exception as e:
        exception_process(e, logfile=os.path.join(mdc.exp_fd, 'log.txt'), stamps=mdc.get_stamps() + ['simple_create_model'])
        return None
    mdc.simple_create_model = True
    return model

def compileAutoEncoderStandard(mdc, **kwags):
    addlog_ = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    try:
        layer_index = 0
        keras_input, keras_input_preconcatenate = [], []
        keras_output = []
        lossmethods, output_activations = mylist(), mylist()
        encodedWithUnfam = []
        #input layers
        cell_size_mdc = dcp(getattr(mdc, 'cell_size', None))
        for k,v in mdc.xheader_zones.items():
            cell_size = getattr(v, 'cell_size', cell_size_mdc)
            input_shape = getattr(v, 'input_shape', (len(v) if(cell_size is None) else (
               cell_size if(DFP.isiterable(cell_size)) else (cell_size, len(v)))))
            input_zone = parseKL('Input')(shape = input_shape, name=stamp_process('',['input',k],'','','','_'))
            preprocessing = mylist([dcp(getattr(v, 'preprocessing', None))]).get_all()
            if('encoding' in preprocessing):
                latentDim = getattr(v,'classEmbeddingLatentDim',getattr(mdc,'classEmbeddingLatentDim',2))
                embeddingLayer = parseKL('Embedding')(np.prod(len(v.preprocessor.classes_)+1), latentDim, input_length=cell_size)
                input_zone_preconcatenate = embeddingLayer(input_zone)
                v.latentLayerName = dcp(embeddingLayer.name)
                print(v.latentLayerName)
                input_zone_preconcatenate = parseKL('Reshape')(target_shape=(latentDim,))(input_zone_preconcatenate)
            else:
                input_zone_preconcatenate = input_zone
            if(cell_size is not None):
                layer_index_v = 0
                input_zone_preconcatenate, layer_index_v = ALG.stack_layers(
                    layer_index_v, input_zone_preconcatenate, activation=mdc.activation, 
                    default_layer_name = getattr(v, 'layer_type', 'LSTM'),
                    hidden_layer_sizes=getattr(v, 'hidden_layer_sizes', (2,)), 
                    hidden_layer_nns=getattr(v, 'hidden_layer_nns', {}),
                    dropout_rates = getattr(v, 'dropout_rates', None),
                    maxpool2D_sizes=getattr(v, 'maxpool2D_sizes', None),
                    kernel_sizes=getattr(v, 'kernel_sizes', None),
                    strideses=getattr(v, 'strideses', None),
                    addlog = addlog_, stamps=['input_preprocess'])
            keras_input.append(input_zone) 
            keras_input_preconcatenate.append(input_zone_preconcatenate)
        addlog_('keras_input:%s'%stamp_process('', stamps = list(map(str, keras_input)), 
                                               stamp_left='\n', stamp_right=''), **kwags)
        mdc.keras_input = mylist(keras_input).get()
        inputs = parseKL('concatenate')(
            keras_input_preconcatenate) if(len(keras_input)>1) else keras_input_preconcatenate[0]
        #hidden layers
        addlog_('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
        encoded, layer_index = ALG.stack_layers(layer_index, inputs, activation=mdc.activation, 
                    default_layer_name = getattr(mdc, 'layer_type', 'Dense'),
                    hidden_layer_sizes=mdc.encoder_hidden_layer_sizes, 
                    hidden_layer_nns=mdc.encoder_hidden_layer_nns,
                    dropout_rates = getattr(mdc, 'dropout_rates', None),
                    addlog = addlog_)
        if(encoded==None):
            return None
        encoded, layer_index = ALG.stack_layers(layer_index, encoded, 
                    activation=getattr(mdc, 'latentActivation', 'tanh'),
                    default_layer_name = getattr(mdc, 'layer_type', 'Dense'),
                    hidden_layer_sizes=(len(mdc.yheader),), 
                    dropout_rates = getattr(mdc, 'dropout_rates', None),
                    addlog = addlog_)
        encodedWithUnfam.append(encoded)
        layer_index+=1
        retTemp = {}
        halfDecoded, layer_index = ALG.stack_layers(layer_index, encoded, activation=mdc.activation, 
                    default_layer_name = getattr(mdc, 'layer_type', 'Dense'),
                    hidden_layer_sizes=mdc.decoder_hidden_layer_sizes, 
                    hidden_layer_nns=mdc.decoder_hidden_layer_nns,
                    dropout_rates = getattr(mdc, 'dropout_rates', None),
                    addlog = addlog_, ret=retTemp)
        if(halfDecoded==None):
            return None
        #output layers
        for i,(k,v) in enumerate(list(tuple(mdc.xheader_zones.items()))):
            cell_size = getattr(v, 'cell_size', None)
            defaultLayerStyleName = getattr(v, 'layer_type', 
                                            ('LSTM' if(DFP.isiterable(cell_size)) else 'Dense'))
            LOGger.addDebug('defaultLayerStyleName', defaultLayerStyleName)
            output_zone = stackOutputStandard(halfDecoded, v, defaultLayerStyleName=defaultLayerStyleName, 
                                              stamps = ['output',k])
            if(isinstance(getattr(v, 'clipConfig', None), LOGger.mystr)):
                clipConfig = getattr(v, 'clipConfig')
                clipAxis = getattr(clipConfig, 'axis', None)
                clipMin = getattr(clipConfig, 'min', 0.0) if(getattr(v,'preprocessor',None) is None) else v.preprocessor.transform(
                    np.full((1,v.preprocessor.n_features_in_), getattr(clipConfig, 'min', 0.0)))[0,clipAxis]
                clipMax = getattr(clipConfig, 'max', 1.0) if(getattr(v,'preprocessor',None) is None) else v.preprocessor.transform(
                    np.full((1,v.preprocessor.n_features_in_), getattr(clipConfig, 'max', 1.0)))[0,clipAxis]
                output_zone = parseKL('ClipLayer')(clipMin, clipMax, axis=clipAxis)(output_zone)
            output_activations.append(v.activation)
            lossmethods.append(v.lossmethod)
            keras_output.append(output_zone)
        for i,(k,v) in enumerate(getattr(mdc, 'unfam_header_zones', {}).items()):
            opt_sz = len(v)
            default_layer = parseKL('Dense')
            activation = getattr(v,'activation',LOGger.extract(
                    getattr(mdc, 'activations', []), index=i, key=k, default=getattr(mdc, 'activation', 'sigmoid')))
            output_activations.append(activation)
            unfam_zone = default_layer(units=opt_sz, activation=activation, 
                                    name=stamp_process('',['unfam',k],'','','','_'))(halfDecoded)
            lossmethods.append(v.lossmethod)
            keras_output.append(unfam_zone)    
            encodedWithUnfam.append(unfam_zone)
        addlog('keras_output:%s'%stamp_process('', stamps = list(map(str, keras_output)), 
                                                   stamp_left='\n', stamp_right=''), **kwags)
        mdc.lossmethods = lossmethods
        addlog('operating lossmethods:%s'%(str(mdc.lossmethods.get())), **kwags)
        mdc.output_activations = output_activations
        addlog('operating output_activations:%s'%(str(output_activations.get())), **kwags)
        mdc.keras_output = keras_output if(len(keras_output)>1) else keras_output[0]
        model = keras.Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                            name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
        mdc.encoder = keras.Model(inputs=mdc.keras_input, outputs=encodedWithUnfam, 
                              name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
        decoderInput = parseKL('Input')(shape=len(mdc.yheader))
        LOGger.addDebug('retTemp:%s'%retTemp)
        stack_names = retTemp['stack_names']
        if(stack_names):
            decoderEntranceName = LOGger.mylist(stack_names).get(0, encoded.name)
            inputsTensor = model.get_layer(decoderEntranceName).input
        else:
            inputsTensor = encoded
        mdc.decoder = keras.Model(inputs=inputsTensor, outputs=model.output,
                              name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    except Exception as e:
        exception_process(e, logfile=os.path.join(mdc.exp_fd, 'log.txt'), stamps=mdc.get_stamps() + ['simple_create_model'])
        return None
    mdc.simple_create_model = True
    return model

def compileAutoEncoderCNNLSTM(mdc, **kwags):
    """
    參數化的 CNN+LSTM AutoEncoder 編譯方式，所有結構皆由 config.json 決定。
    支援 CNN+LSTM 雙層處理，使用 stackOutputStandard 進行輸出層處理。
    """
    addlog = kwags.get('addlog', LOGger.addlog)
    try:
        keras_input = []
        keras_input_preconcatenate = []
        # input layers
        cell_size_mdc = dcp(getattr(mdc, 'cell_size', None))
        for k, v in mdc.xheader_zones.items():
            cell_size = getattr(v, 'cell_size', cell_size_mdc)
            activation = getattr(v, 'activation', 'linear')
            input_zone = ALG.Input(shape=cell_size, name=LOGger.stamp_process('', ['input', k], '', '', '', '_'))
            keras_input.append(input_zone)
            
            # 處理 preprocessing
            preprocessing = mylist([dcp(getattr(v, 'preprocessing', None))]).get_all()
            if 'encoding' in preprocessing:
                latentDim = getattr(v, 'classEmbeddingLatentDim', getattr(mdc, 'classEmbeddingLatentDim', 2))
                embeddingLayer = ALG.layer_producer('Embedding')(np.prod(len(v.preprocessor.classes_) + 1), latentDim, input_length=cell_size)
                input_zone_preconcatenate = embeddingLayer(input_zone)
                v.latentLayerName = dcp(embeddingLayer.name)
                print(v.latentLayerName)
                input_zone_preconcatenate = ALG.layer_producer('Reshape')(target_shape=(latentDim,))(input_zone_preconcatenate)
            else:
                input_zone_preconcatenate = input_zone
            
            # 如果有 cell_size，使用 ALG.stack_layers 進行參數化處理
            if cell_size is not None:
                layer_index_v = 0
                # 特殊處理 Pattern 欄位：先 Reshape 增加通道維度
                if k == 'Pattern' and len(cell_size) == 1:
                    input_zone_preconcatenate = ALG.layer_producer('Reshape')((*cell_size, 1))(input_zone_preconcatenate)
                
                # 第一層：CNN 處理（使用專用參數）
                input_zone_preconcatenate, layer_index_v = ALG.stack_layers(
                    layer_index_v, input_zone_preconcatenate, 
                    activation=activation,
                    default_layer_name='Conv1D',  # 固定使用 Conv1D
                    hidden_layer_sizes=getattr(v, 'cnn_hidden_layer_sizes', getattr(v, 'hidden_layer_sizes', (4, 16))),  # CNN 專用參數
                    hidden_layer_nns=getattr(v, 'cnn_hidden_layer_nns', getattr(v, 'hidden_layer_nns', {})),  # CNN 專用配置
                    dropout_rates=getattr(v, 'cnn_dropout_rates', getattr(v, 'dropout_rates', None)),  # CNN 專用 dropout
                    maxpool2D_sizes=getattr(v, 'cnn_maxpool2D_sizes', getattr(v, 'maxpool2D_sizes', None)),  # CNN 專用 maxpool
                    kernel_sizes=getattr(v, 'cnn_kernel_sizes', getattr(v, 'kernel_sizes', None)),  # CNN 專用 kernel_size
                    strideses=getattr(v, 'cnn_strideses', getattr(v, 'strideses', None)),  # CNN 專用 strides
                    addlog=addlog, 
                    stamps=['input_cnn', k]
                )

                # 第二層：LSTM 處理（使用專用參數）
                input_zone_preconcatenate, layer_index_v = ALG.stack_layers(
                    layer_index_v, input_zone_preconcatenate, 
                    activation=activation,
                    default_layer_name='LSTM',  # 固定使用 LSTM
                    hidden_layer_sizes=getattr(v, 'lstm_hidden_layer_sizes', getattr(v, 'hidden_layer_sizes', (20, 40))),  # LSTM 專用參數
                    hidden_layer_nns=getattr(v, 'lstm_hidden_layer_nns', getattr(v, 'hidden_layer_nns', {})),  # LSTM 專用配置
                    dropout_rates=getattr(v, 'lstm_dropout_rates', getattr(v, 'dropout_rates', None)),  # LSTM 專用 dropout
                    maxpool2D_sizes=getattr(v, 'lstm_maxpool2D_sizes', getattr(v, 'maxpool2D_sizes', None)),  # LSTM 專用 maxpool
                    kernel_sizes=getattr(v, 'lstm_kernel_sizes', getattr(v, 'kernel_sizes', None)),  # LSTM 專用 kernel_size
                    strideses=getattr(v, 'lstm_strideses', getattr(v, 'strideses', None)),  # LSTM 專用 strides
                    addlog=addlog, 
                    stamps=['input_lstm', k]
                )
            
            keras_input_preconcatenate.append(input_zone_preconcatenate)
        addlog('keras_input:%s' % LOGger.stamp_process('', stamps=list(map(str, keras_input)), stamp_left='\n', stamp_right=''), **kwags)
        mdc.keras_input = keras_input if len(keras_input) > 1 else keras_input[0]

        # hidden layers
        if len(keras_input_preconcatenate) > 1:
            branch_layer_front = ALG.layer_producer('concatenate')(keras_input_preconcatenate)
        else:
            branch_layer_front = keras_input_preconcatenate[0]
        branch_layer_front, _ = ALG.stack_layers(
            0, branch_layer_front,
            activation=getattr(mdc, 'latentActivation', 'tanh'),
            default_layer_name=getattr(mdc, 'layer_type', 'Dense'),
            hidden_layer_sizes=getattr(mdc, 'encoder_hidden_layer_sizes', (3,)),
            hidden_layer_nns=getattr(mdc, 'encoder_hidden_layer_nns', {}),
            dropout_rates=getattr(mdc, 'dropout_rates', None),
            stamps=['branchEncoder'],
            addlog=addlog
        )
        encoded, _ = ALG.stack_layers(
            0, branch_layer_front,
            activation=getattr(mdc, 'latentActivation', 'tanh'),
            default_layer_name=getattr(mdc, 'layer_type', 'Dense'),
            hidden_layer_sizes=(len(mdc.yheader),),
            dropout_rates=getattr(mdc, 'dropout_rates', None),
            addlog=addlog
        )
        LOGger.addDebug(f'encoded: {encoded.shape}')
        encodedWithUnfam = [encoded]

        # decoder
        input_decoder = ALG.Input(shape=encoded.shape[1:], name=LOGger.stamp_process('', ['input_decoder'], '', '', '', '_'))
        branch_layer_backend, _ = ALG.stack_layers(
            0, input_decoder,
            activation=getattr(mdc, 'activation', 'selu'),
            default_layer_name=getattr(mdc, 'layer_type', 'Dense'),
            hidden_layer_sizes=getattr(mdc, 'encoder_hidden_layer_sizes', (3,)),
            hidden_layer_nns=getattr(mdc, 'encoder_hidden_layer_nns', {}),
            dropout_rates=getattr(mdc, 'dropout_rates', None),
            stamps=['branchDecoder'],
            addlog=addlog
        )
        if branch_layer_backend is None:
            return None

        # output layers - 使用 stackOutputStandard 進行參數化處理
        keras_output = []
        lossmethods, output_activations = LOGger.mylist(), LOGger.mylist()
        for k, v in mdc.xheader_zones.items():
            activation = getattr(v, 'activation', 'selu')
            output_activation = getattr(v, 'output_activation', 'linear')
            cell_size = getattr(v, 'cell_size', cell_size_mdc)
            
            # 根據輸入層的處理方式，選擇對應的輸出層類型
            if k == 'Pattern' and len(cell_size) == 1:
                # Pattern 欄位使用 LSTM 輸出層，對應輸入層的 LSTM 處理
                defaultLayerStyleName = 'LSTM'
            else:
                # 其他欄位根據 cell_size 決定
                defaultLayerStyleName = getattr(v, 'layer_type', ('LSTM' if DFP.isiterable(cell_size) else 'Dense'))
            
            # 使用 stackOutputStandard 進行參數化輸出處理
            output_zone = stackOutputStandard(branch_layer_backend, v, defaultLayerStyleName=defaultLayerStyleName, stamps=['output', k])
            
            # 處理 clip 配置
            if isinstance(getattr(v, 'clipConfig', None), LOGger.mystr):
                clipConfig = getattr(v, 'clipConfig')
                clipAxis = getattr(clipConfig, 'axis', None)
                clipMin = getattr(clipConfig, 'min', 0.0) if (getattr(v, 'preprocessor', None) is None) else v.preprocessor.transform(
                    np.full((1, v.preprocessor.n_features_in_), getattr(clipConfig, 'min', 0.0)))[0, clipAxis]
                clipMax = getattr(clipConfig, 'max', 1.0) if (getattr(v, 'preprocessor', None) is None) else v.preprocessor.transform(
                    np.full((1, v.preprocessor.n_features_in_), getattr(clipConfig, 'max', 1.0)))[0, clipAxis]
                output_zone = ALG.layer_producer('ClipLayer')(clipMin, clipMax, axis=clipAxis)(output_zone)
            
            output_activations.append(v.activation)
            lossmethods.append(v.lossmethod)
            keras_output.append(output_zone)
        
        # unfam header zones
        for i, (k, v) in enumerate(getattr(mdc, 'unfam_header_zones', {}).items()):
            opt_sz = len(v)
            default_layer = ALG.layer_producer('Dense')
            activation = getattr(v, 'activation', LOGger.extract(getattr(mdc, 'activations', []), index=i, key=k, default=getattr(mdc, 'activation', 'sigmoid')))
            output_activations.append(activation)
            unfam_zone = default_layer(units=opt_sz, activation=activation, name=LOGger.stamp_process('', ['unfam', k], '', '', '', '_'))(branch_layer_backend)
            lossmethods.append(v.lossmethod)
            keras_output.append(unfam_zone)
        
        addlog('keras_output:%s' % LOGger.stamp_process('', stamps=list(map(str, keras_output)), stamp_left='\n', stamp_right=''), **kwags)
        mdc.lossmethods = lossmethods
        addlog('operating lossmethods:%s' % (str(mdc.lossmethods.get())), **kwags)
        mdc.output_activations = output_activations
        addlog('operating output_activations:%s' % (str(output_activations.get())), **kwags)
        mdc.keras_output = keras_output if len(keras_output) > 1 else keras_output[0]
        mdc.decoder = ALG.keras.Model(inputs=input_decoder, outputs=mdc.keras_output, name=LOGger.stamp_process('', [*mdc.get_stamps(for_file=False), 'decoder'], '_'))
        modelDecoded = mdc.decoder(encoded)
        for unfam_zone in modelDecoded[len(mdc.xheader_zones):]:
            encodedWithUnfam.append(unfam_zone)
        mdc.encoder = ALG.keras.Model(inputs=mdc.keras_input, outputs=encodedWithUnfam, name=LOGger.stamp_process('', [*mdc.get_stamps(for_file=False), 'encoder'], '_'))
        modelEncoded = mdc.encoder(mdc.keras_input)
        modelDecoded = mdc.decoder(modelEncoded[0])
        model = ALG.keras.Model(inputs=mdc.keras_input, outputs=modelDecoded, name=LOGger.stamp_process(mdc.get_stamps(for_file=False), '', '_'))
    except Exception as e:
        LOGger.exception_process(e, logfile=os.path.join(mdc.exp_fd, 'log.txt'), stamps=mdc.get_stamps() + ['compileAutoEncoderCNNLSTM'])
        return None
    return model

def activateOptimizerSystem(system_name='Adam'):#'SGD'
    return getattr(ALG.tf.keras.optimizers, system_name)

def kerasmodel_logging(mdc, model, print_fn=None):
    model.summary(print_fn=print_fn)
    opt_dict = model.optimizer.get_config() if(hasattr(model.optimizer, 'get_config')) else {}
    opt_seires = pd.Series(opt_dict, name='optimizer config')
    if(mdc!=None):
        mdc.opt_dict = opt_dict
    if(print_fn):
        pd.set_option('display.max_rows', None)
        opt_stg = 'optimizer config:\n%s'%str(opt_seires)
        print_fn(opt_stg)
        pd.set_option('display.max_rows', 0)

def mdcScenarioConfiguring(mdc, config=None, stamps=None, **kwags):
    """
    scenario that mdc set [stamps]-properties by config

    Parameters
    ----------
    mdc : TYPE
        DESCRIPTION.
    config : TYPE, optional
        DESCRIPTION. The default is None.
    stamps : TYPE, optional
        DESCRIPTION. The default is None.
    **kwags : TYPE
        DESCRIPTION.

    Returns
    -------
    bool
        DESCRIPTION.

    """
    stamps = stamps if(isinstance(stamps, list)) else []
    config = config if(isinstance(config, dict)) else {}
    for k,v in config.items():
        setattr(mdc, k, v)
        mdc.addlog('configuring:%s'%DFP.parse(v,digit=4), stamps=stamps+[k], colora=LOGger.OKCYAN)
    return True

def compilingScenario(mdc, need_training=False):
    """
    scenario that mdc get configured before fit.


    Parameters
    ----------
    mdc : TYPE
        DESCRIPTION.
    need_training : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    bool
        DESCRIPTION.

    """
    try:
        if(not mdc.configuring(stamps=['compile'])):
            mdc.addlog('compileConfiguring failed!!!', colora=LOGger.FAIL, stamps=['compilingScenario'])
            return False
        if(not mdc.compiling(need_training=need_training)):
            mdc.addlog('compiling failed!!!', colora=LOGger.FAIL, stamps=['compilingScenario'])
            return False
        mdc.plot_model()
        if(not mdc.configuring(stamps=['fit'])):
            mdc.addlog('fitConfiguring failed!!!', colora=LOGger.FAIL, stamps=['compilingScenario'])
            return False
    except Exception as e:
        LOGger.exception_process(e, logfile=os.path.join(mdc.exp_fd, 'log.txt'), stamps=mdc.get_stamps() + ['compilingScenario'])
        return False
    return True

def compilingKeras(mdc, need_training=False, **kwags):
    addlog_ = LOGger.execute('addlog', mdc, kwags, default=LOGger.addloger(logfile=''), not_found_alarm=False)
    model=None
    addlog_('activating....')
    if(LOGger.isinstance_not_empty(mdc.get_source(), str)):
        print('need_training', need_training)
        model = mdc.model
    if(model==None):
        compileMethodStg = getattr(mdc, 'compileMethod', '')
        model = getattr(ALG, compileMethodStg, compileStandard)(mdc, addlog=addlog_, **getattr(mdc, 'complieParams', {}))
    if(model==None):
        addlog_('模型建置失敗', stamps=mdc.get_stamps())
        return False
    print('modelCompiling:%s'%model)
    if(need_training):
        # if(getattr(getattr(mdc, 'model', None), 'optimizer', None)==None):
        #     if(not activate_custom_objects(mdc)):
        #         addlog('激活自定義訓練物件失敗!!!', stamps=mdc.get_stamps())
        #         return False
        #     addlog('激活自定義訓練物件成功!!!', stamps=mdc.get_stamps())
        mdc.addlog('compiling.......')
        lossmethods_container = mdc.lossmethods.get_all()
        mdc.addlog('lossmethods_container:%s'%lossmethods_container)
        #optimizer
        optimizer = getattr(mdc, 'optimizer', 'Adam')
        optimizer_args = getattr(mdc, 'optimizer_args', {})
        if(isinstance(optimizer, str)):
            optimizer_system = activateOptimizerSystem(optimizer)
            optimizer = optimizer_system(**optimizer_args)
        loss_weights = mylist([getattr(mdc, 'lossweights', kwags.get('lossweights', None))]).get_all().get()
        mdc.addlog('loss_weights', loss_weights) if(loss_weights) else None
        model.compile(optimizer = optimizer, 
                      loss=lossmethods_container.get(), 
                      metrics=mylist([getattr(mdc, 'metrics', kwags.get('metrics', None))]).get_all().get(), 
                      loss_weights=mylist([getattr(mdc, 'lossweights', kwags.get('lossweights', None))]).get_all().get(), 
                      weighted_metrics=mylist([getattr(mdc, 'weighted_metrics', kwags.get('weighted_metrics', None))]).get())
    kerasmodel_logging(mdc, model, print_fn=None)
    mdc.drawLossStopFlag = False
    mdc.model = model
    mdc.export_model_detail()
    return True

def compilingKerasSingleLatent(mdc, need_training=False, **kwags):
    mdc.hidden_layer_sizes = (1,)
    mdc.hidden_layer_nns = {}
    if(not compilingKeras(mdc, need_training=False, **kwags)):
        return False
    return True

def compilingKerasAutoEncoder(mdc, need_training=False, **kwags):
    addlog_ = LOGger.execute('addlog', mdc, kwags, default=LOGger.addloger(logfile=''), not_found_alarm=False)
    model=None
    addlog_('activating....')
    if(LOGger.isinstance_not_empty(mdc.get_source(), str)):
        print('need_training', need_training)
        model = mdc.model
    if(model==None):
        compileMethodStg = getattr(mdc, 'compileMethod', '')
        model = getattr(ALG, compileMethodStg, compileAutoEncoderStandard)(mdc, addlog=addlog_, **getattr(mdc, 'complieParams', {}))
    if(model==None):
        addlog_('模型建置失敗', stamps=mdc.get_stamps())
        return False
    print('modelCompiling:%s'%model)
    if(need_training):
        mdc.addlog('compiling.......')
        lossmethods_container = mdc.lossmethods.get_all()
        mdc.addlog('lossmethods_container:%s'%lossmethods_container)
        #optimizer
        optimizer = getattr(mdc, 'optimizer', 'Adam')
        optimizer_args = getattr(mdc, 'optimizer_args', {})
        if(isinstance(optimizer, str)):
            optimizer_system = activateOptimizerSystem(optimizer)
            optimizer = optimizer_system(**optimizer_args)
        loss_weights = mylist([getattr(mdc, 'lossweights', kwags.get('lossweights', None))]).get_all().get()
        mdc.addlog('loss_weights', loss_weights) if(loss_weights) else None
        model.compile(optimizer = optimizer, 
                      loss=lossmethods_container.get(), 
                      metrics=mylist([getattr(mdc, 'metrics', kwags.get('metrics', None))]).get_all().get(), 
                      loss_weights=mylist([getattr(mdc, 'lossweights', kwags.get('lossweights', None))]).get_all().get(), 
                      weighted_metrics=mylist([getattr(mdc, 'weighted_metrics', kwags.get('weighted_metrics', None))]).get())
    kerasmodel_logging(mdc, model, print_fn=None)
    mdc.drawLossStopFlag = False
    mdc.model = model
    mdc.export_model_detail()
    # sys.exit(1)
    return True

def compilingSklearn(mdc, need_training=False, **kwags):
    addlog_ = LOGger.execute('addlog', mdc, kwags, default=LOGger.addloger(logfile=''), not_found_alarm=False)
    model=None
    addlog_('activating....')
    if(LOGger.isinstance_not_empty(mdc.get_source(), str)):
        return True
    print('modelCompiling:%s'%model)
    if(need_training):
        mdc.addlog('compiling.......')
        if(not hasattr(ske, mdc.algorithmSerial)):
            return False
        mdc.algorithm = getattr(ske, mdc.algorithmSerial)
        cv = mdc.cv
        algorithm_params = mdc.algorithm_params
        algorithm_param_grid = mdc.algorithm_param_grid
        scoring = mdc.scoring
        if(scoring!=None):
            if(not scoring in skm.SCORERS.keys()):
                addlog('SK系統不支援損失函數[%s]，scoring改回為None'%str(scoring))
                scoring = None
            mdc.scoring = scoring
        if(algorithm_param_grid):
            for pr in algorithm_params:
                algorithm_param_grid.update({pr:[algorithm_params[pr]]}) if(not pr in algorithm_param_grid) else None
    return True

def compilingAutoSklearn(mdc, need_training=False, **kwags):
    addlog_ = LOGger.execute('addlog', mdc, kwags, default=LOGger.addloger(logfile=''), not_found_alarm=False)
    model=None
    addlog_('activating....')
    if(LOGger.isinstance_not_empty(mdc.get_source(), str)):
        return True
    print('modelCompiling:%s'%model)
    if(need_training):
        mdc.addlog('compiling.......')
        try:
            mdc.missionSystem = eval(mdc.missionSystemSerial)
        except Exception as e:
            mdc.addlog('mdc.missionSystemSerial invalid:', mdc.missionSystemSerial, colora=LOGger.FAIL)
            return False
        cv = mdc.cv
        missionSystem_params = mdc.missionSystem_params
        missionSystem_param_grid = mdc.missionSystem_param_grid
        scoring = mdc.scoring
        if(scoring!=None):
            if(not scoring in skm.SCORERS.keys()):
                addlog('SK系統不支援損失函數[%s]，scoring改回為None'%str(scoring))
                scoring = None
            mdc.scoring = scoring
        if(missionSystem_param_grid):
            for pr in missionSystem_params:
                missionSystem_param_grid.update({pr:[missionSystem_params[pr]]}) if(not pr in missionSystem_param_grid) else None
    return True

def evaluationEquipBasicData(df):
    """
    equiping cause data, aux data...

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    df_exp : TYPE
        DESCRIPTION.

    """
    if(df is None):
        return pd.DataFrame()
    if(not isinstance(df, pd.core.frame.DataFrame)):
        LOGger.addlog('df need to be a data frame!!!', logfile='', colora=LOGger.FAIL)
        sys.exit(1)
    if(df.empty):
        return df
    df_exp = df.drop(df.columns[np.array([
        df[hd].map(lambda x:(len(x.shape)>1 if(hasattr(x, 'shape')) else False)).any() for hd in df.columns])], axis=1)
    return df_exp

def transformArrays2Tensors(mdc, data, headerZoneItems, stamp, dataCountLbd=0, default_data_value=0):
    # stamps = stamps if(isinstance(stamps, list)) else []
    # stamp = LOGger.stamp_process('',stasmps,'','','','_')
    addlog_ = mdc.addlog
    if(not isinstance(data, pd.core.frame.DataFrame)):
        addlog_('array error:\n', str(data[:2000]), stamps=[stamp])
        return False
    dataZones = LOGger.mylist([])
    dataCountsNote = None
    for k,v in headerZoneItems:
        if(len([c for c in v if not c in data.columns])>0):
            # 為了給出作"異常檢測訓練"給出全零矩陣
            LOGger.show_vector([c for c in v if not c in data.columns], stamps=[stamp, 'data.columns lacks'], 
                               logfile=os.path.join(mdc.exp_fd, 'log.txt'))
            addlog_('using default value[%s]'%DFP.parse(default_data_value), stamps=[stamp, 'data.columns lacks', k])
            otherDim = (1,*(v.cell_size)) if(DFP.isiterable(getattr(v,'cell_size',None))) else (len(v),)
            data_v = np.full((data.shape[0],*otherDim), default_data_value) # pd.DataFrame(np.full((data.shape[0],1), default_data_value), index=data.index)
            addlog_('using default data\n%s'%(getattr(data_v,'values',data_v)[:3,:]), stamps=[stamp, 'data.columns lacks', k])
        else:
            data_v = data[v].copy()
        if(dataCountsNote!=None):
            if(data_v.shape[0]!=dataCountsNote):
                addlog_('array count error:  data_v(%d)'%data_v.shape[0], '!= dataCountsNote(%s)'%dataCountsNote, stamps=[stamp, k])
                return False
        else:
            dataCountsNote = data_v.shape[0]
        try:
            data_v = v.transform(data_v)
        except Exception as e:
            LOGger.exception_process(e,logfile='',stamps=['transformArrays2Tensors', k])
            return False
        if(len(data_v.shape)<2):
            mdc.addlog('data_v shape error', data_v.shape, stamps=['transformArrays2Tensors', k], colora=LOGger.FAIL)
            return False
        if(isinstance(data_v, type(None))):
           mdc.addlog(str(data_v), stamps=[transformArrays2Tensors.__name__, stamp, 'data_encaped None'])
           return False
        data_v_converted = mdc.convert4modelCore(data_v)
        dataZones.append(data_v_converted)
    dataZones = dataZones.get()
    setattr(mdc, LOGger.stamp_process('',[stamp, 'tensor'],'','','','_'), dataZones)
    return True

def fitKerasExceptionScenario(mdc, e, ax, debugPackage=None, failedMemo = 'invalid keras sources', **kwags):
    debugPackage = debugPackage if(isinstance(debugPackage, dict)) else {}
    LOGger.exception_process(e, logfile=os.path.join(mdc.exp_fd, 'log.txt'), stamps=mdc.get_stamps() + ['trainning'])
    addlog_ = kwags.get('addlog', getattr(mdc, 'addlog', m_print))
    m_debug.update(debugPackage)
    m_debug.save()
    m_debug.listen(failedMemo)
    addlog_(failedMemo, stamps=mdc.get_stamps(), colora=LOGger.FAIL)
    if(isinstance(ax, vs3.plt.Axes)):
        ax.text(np.sum(getattr(mdc, 'doneXPosRatio', np.array([0.9,0.1]))*np.array(ax.get_xlim())), 
                np.sum(getattr(mdc, 'doneYPosRatio', np.array([0.9,0.1]))*np.array(ax.get_ylim())), 
                failedMemo, color=(1,0,0))
    saveLossCurve(mdc)
    setattr(mdc, 'drawLossStopFlag', True)
    sys.exit(1)

def supervisingInterruptSignal(targetExtFileName='.interrupt', targetDir='.', operModel=None, **kwags):
    addlog_ = kwags.get('addlog', LOGger.addloger(logfile=''))
    files, _ = DFP.explore_folder(targetDir)
    for file in files:
        if(file.endswith(targetExtFileName)):
            addlog_('收到中斷信號，停止訓練', colora=LOGger.FAIL)
            if(hasattr(operModel, 'stop_training')):    operModel.stop_training = True
            return True
    return False

def fitKeras(mdc, np_X_train, np_y_train, validation_data=None, compileMethod=compileStandard,
             callback_freq = 10, quick_save_best=True, is_self_supervise=False, stamps=None, 
             reinitial_thd=2, storage_thd=1, overfit_maxthd=0.3, epochs=100, batch_size=100, 
             auto_train_max_count=2, break_time=2, save_file_count=5, over_loss_max_ratio=0.3, **kwags):
    addlog_ = getattr(mdc, 'addlog') if(hasattr(mdc, 'addlog')) else kwags.get('addlog', LOGger.addlog)
    interruptFile = os.path.join(mdc.exp_fd, '.interrupt')
    is_valid_data = validation_data is not None
    ax = mdc.lossCurveAx #lossCurveFig
    long_loss_seq, long_valid_loss_seq= [], []
    lossCurvesInCallback = [long_loss_seq, long_valid_loss_seq] if(is_valid_data) else [long_loss_seq]
    lossCurve_callback = LossCurveCallback(ax, *lossCurvesInCallback, plot_interval=10, stamps=mdc.get_stamps())
    
    # 添加定期日志callback
    periodic_log_callback = PeriodicLogCallback(log_interval=1800, log_file=os.path.join(mdc.exp_fd, 'log.txt'), 
                                                stamps=['logLoss', *mdc.get_stamps()])  # 30分钟 = 1800秒
    
    ware_callbacks = os.path.join(mdc.exp_fd, 'warehouse','callbacks',
                                  stamp_process('', mdc.get_stamps(for_file=True),'','','', '_'))
    callback_freq = getattr(mdc, 'callback_freq', callback_freq)
    callback_basket = callback_process(
            ware_callbacks, callback_freq=callback_freq, 
            loss_digit=kwags.get('loss_digit', 6),
            patience=kwags.get('earlystop_patience', 5),
            lossCurve_callback=lossCurve_callback,
            periodic_log_callback=periodic_log_callback) if(callback_freq) else []
    
    batch_size = int((batch_size * np_X_train[0].shape[0])//1) if(DFP.astype(batch_size, default=-1)<1 and DFP.astype(batch_size, default=-1)>0) else batch_size
    batch_size = batch_size if(LOGger.isinstance_not_empty(batch_size, int)) else 100
    fitting_error = False
    best_score = storage_thd
    auto_train_count, loss = 0, []
    model = mdc.model
    is_self_supervise = getattr(mdc, 'is_self_supervise', is_self_supervise)
    thd = threading.Thread(target=saveLossCurveWhile, args=[mdc], daemon=True)
    thd.start()
    while(auto_train_count < auto_train_max_count):
        try:
            thAg = LOGger.myThreadAgent(target_core=(lambda *thargs, **thkwags:supervisingInterruptSignal(
                                operModel=model, targetExtFileName='.interrupt', targetDir=mdc.exp_fd, addlog=mdc.addlog)), 
                                time_waiting=60, immediate_start=True, daemon=True)
            history = model.fit(np_X_train, 
                                np_y_train, 
                                epochs = epochs, 
                                batch_size = batch_size,
                                validation_data = validation_data,
                                callbacks=callback_basket)
            loss = list(history.history['loss'])
            valid_loss = list(history.history.get('val_loss',[]))
            best_score = best_score if(
                    best_score<=min(long_loss_seq)) else min(long_loss_seq)
        except Exception as e:
            debugPackage = {'np_X_train': np_X_train, 'np_y_train':np_y_train,'batch_size':batch_size, 'epochs':epochs, 
                            'model':model, 'lossmethods':mdc.lossmethods, 'keras_output':mdc.keras_output}
            debugPackage.update({'np_X_test':validation_data[0], 'np_y_test':validation_data[1]}) if(validation_data) else None
            fitKerasExceptionScenario(mdc, e, ax, debugPackage=debugPackage, failedMemo = 'invalid keras sources', **kwags)
        finally:
            if(thAg):   thAg.stop()
        if(callback_freq):
            '''清空損失過高的models'''
            cleaning_callbacks(ware_callbacks, storage_thd, save_file_count)
        if(not loss):
            addlog_('[%d]沒有loss數據!!!!'%(auto_train_count), stamps=mdc.get_stamps(), colora=LOGger.FAIL)
            fitKerasExceptionScenario(mdc, e, ax, debugPackage=debugPackage, failedMemo = 'loss missing', **kwags)
        if(np.isnan(np.array(loss) +  (np.array(valid_loss) if(valid_loss) else np.zeros(np.array(loss).shape[0]))).any()):
            nanMask = np.isnan(np.array(loss) +  (np.array(valid_loss) if(valid_loss) else np.zeros(np.array(loss).shape[0])))
            nanCount = np.sum(nanMask)
            totalCount = nanMask.shape[0]
            failedMemo = r'[%d]auto_train_count:%d, nanCount/totalCount = $\frac{%d}{%d}$'%(auto_train_count, len(loss), nanCount, totalCount)
            addlog_(failedMemo, stamps=mdc.get_stamps(), colora=LOGger.FAIL)
            mdc.model = callback_model(reinitial_thd, ware_callbacks, addlog=mdc.addlog, custom_objects=get_custom_objects())
            if(mdc.model is None):
                addlog_('沒有可用的備份模型，停止訓練', stamps=mdc.get_stamps(), colora=LOGger.FAIL)
                fitKerasExceptionScenario(mdc, e, ax, debugPackage=debugPackage, failedMemo = failedMemo, **kwags)
                # return False
            if(fitting_error):
                return False
            fitting_error = True
            auto_train_count += 1
            continue
        else:
            fitting_error = False
        if(not fitting_error):
            drawLossCurve(ax, long_loss_seq, long_valid_loss_seq, stamps=mdc.get_stamps())
        if(sum([1 if(loss[i]>reinitial_thd) else 0 
                for i in range(len(loss))])/len(loss)>over_loss_max_ratio):
            over_loss_ratio = sum([1 if(loss[i]>reinitial_thd) else 0 
                                    for i in range(len(loss))])/len(loss)
            failedMemo = '[reinitial_thd:%s] over_loss_ratio arrived %.2f>%.2f'%(
                                        DFP.parse(reinitial_thd, digit=4), over_loss_ratio, 
                                        over_loss_max_ratio)
            addlog_(failedMemo, colora=LOGger.FAIL)
            ax.text(np.sum(getattr(mdc, 'doneXPosRatio', np.array([0.9,0.1]))*np.array(ax.get_xlim())), 
                    np.sum(getattr(mdc, 'doneYPosRatio', np.array([0.9,0.1]))*np.array(ax.get_ylim())), 
                    failedMemo, color=(1,0,0))
            saveLossCurve(mdc)
            setattr(mdc, 'drawLossStopFlag', True)
            return False
        addlog_('[%d]訓練回數:%d, 最佳訓練損失:%.2f, 最佳試驗損失:%.2f'%(
                                        auto_train_count, len(loss),
                                        min(loss), 
                                        min(valid_loss) if(is_valid_data) else -1))
        addlog_('[%d]訓練回數:%d, 最後訓練損失:%.2f, 最後試驗損失:%.2f'%(
                                        auto_train_count, len(loss),
                                        loss[-1], 
                                        valid_loss[-1] if(is_valid_data) else -1))
        if(is_valid_data):
            valid_loss_min = min(long_valid_loss_seq)
            valid_loss_argmin = np.argmin(valid_loss)
            latest_valid_loss = valid_loss[(valid_loss_argmin+1):]
            overfit_case = np.array(latest_valid_loss)[np.array(latest_valid_loss)>=valid_loss_min]
            if(len(overfit_case)/len(valid_loss) > overfit_maxthd):
                addlog_('[%d]歷史驗證最低損失:%.4f, 本回過擬數量:%d, 本回過擬比例:%.2f(>%.2f). 停止訓練!!!'%(
                            auto_train_count, valid_loss_min, len(overfit_case),
                            len(overfit_case)/len(valid_loss), overfit_maxthd))
                break
        if(quick_save_best):
            def_sc = loss[-1] if(not fitting_error) else reinitial_thd
            mdc.model = callback_model(def_sc, ware_callbacks, model=mdc.model, addlog=mdc.addlog, custom_objects=get_custom_objects())
        if(os.path.exists(interruptFile)):
            addlog_('收到中斷信號，停止訓練', colora=LOGger.FAIL, stamps=mdc.get_stamps())
            break
        auto_train_count+=1
        time.sleep(break_time)
    setattr(mdc, 'drawLossStopFlag', True)
    cleaning_callbacks(ware_callbacks, storage_thd, save_file_count)
    if(not fitting_error):
        drawLossCurve(ax, long_loss_seq, long_valid_loss_seq, stamps=mdc.get_stamps())
        ax.text(np.sum(getattr(mdc, 'doneXPosRatio', np.array([0.1,0.9]))*np.array(ax.get_xlim())), 
                np.sum(getattr(mdc, 'doneYPosRatio', np.array([0.9,0.1]))*np.array(ax.get_ylim())), 'done!!!!!!!', color=(0,0,1))
        saveLossCurve(mdc)
    if(not quick_save_best):
        def_sc = history.history['loss'][-1] if(not fitting_error) else reinitial_thd
        mdc.model = callback_model(def_sc, ware_callbacks, model=mdc.model, addlog=mdc.addlog, custom_objects=get_custom_objects())
    if(model==None):
        ax.text(np.sum(getattr(mdc, 'doneXPosRatio', np.array([0.1,0.9]))*np.array(ax.get_xlim())), 
                np.sum(getattr(mdc, 'doneYPosRatio', np.array([0.9,0.1]))*np.array(ax.get_ylim())), 'model None!!!!!!!', color=(1,0,0))
        saveLossCurve(mdc)
        return False
    mdc.model = model
    mdc.total_epochs = len(long_loss_seq)
    mdc.long_loss_seq = long_loss_seq
    mdc.long_valid_loss_seq = long_valid_loss_seq
    mdc.score = min(long_loss_seq)
    mdc.score_valid = min(long_valid_loss_seq) if(is_valid_data) else -1
    return True


def determineRegionsForAutoEncoder(mdc, np_X_train, **kwags):
    mdc.latentCores = mdc.encoder.predict(np_X_train)[0]
    mdc.latentExplainer = LocalOutlierFactor(n_neighbors=20, novelty=True)
    mdc.latentExplainer.fit(mdc.latentCores)

def fitKerasAutoEncoder(mdc, np_X_train, np_y_train=None, validation_data=None, compileMethod=compileAutoEncoderStandard,
                        callback_freq = 10, quick_save_best=True, is_self_supervise=False, stamps=None, 
                        reinitial_thd=2, storage_thd=1, overfit_maxthd=0.3, epochs=100, batch_size=100, 
                        auto_train_max_count=2, break_time=2, save_file_count=5, over_loss_max_ratio=0.3, **kwags):
    if(np_y_train is None): np_y_train = dcp(np_X_train)
    if(not fitKeras(mdc, np_X_train, np_y_train, validation_data=validation_data, compileMethod=compileMethod,
                    callback_freq = callback_freq, quick_save_best=quick_save_best, is_self_supervise=is_self_supervise, stamps=stamps, 
                    reinitial_thd=reinitial_thd, storage_thd=storage_thd, overfit_maxthd=overfit_maxthd, epochs=epochs, batch_size=batch_size, 
                    auto_train_max_count=auto_train_max_count, break_time=break_time, save_file_count=save_file_count, 
                    over_loss_max_ratio=over_loss_max_ratio, **kwags)):
        return False
    if(determineRegionsForAutoEncoder(mdc, np_X_train)):
        return False
    return True

def drawPgirdScores(mdc, grid_search, ybd=(0.6, 1.1), markersize=5, xlb_rotation=70, showmodes=False, ax=None, fig=None, ret=None, **kwags):
    results = pd.DataFrame(grid_search.cv_results_)
    best = np.argmax(results.mean_test_score.values)
    # yheader = results.columns[(results.columns.str.contains('score'))&
    #                           (np.logical_not(results.columns.str.contains('std')))&
    #                           (np.logical_not(results.columns.str.contains('rank')))&
    #                           (np.logical_not(results.columns.str.contains('time')))]
    if(not isinstance(ax, vs3.plt.Axes)):
        if(not isinstance(fig, vs3.plt.Figure)):
            fig = vs3.plt.Figure(figsize=getattr(mdc, 'pgreidFigsize', (15,8)))
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        
    for i, (_, row) in enumerate(results.iterrows()):
        n_split = len(row.index[row.index.str.contains('split')&np.logical_not(row.index.str.contains('min'))&np.logical_not(row.index.str.contains('max'))])
        scores = row[['split%d_test_score' % j for j in range(n_split)]]
        marker_cv, = ax.plot([i] * n_split, scores, '^', c='gray', markersize=markersize,alpha=.5)
        marker_mean, = ax.plot(i, row.mean_test_score, 'v', c='none', alpha=1,
                                markersize=2*markersize, markeredgecolor='k')
        if i == best:
            marker_best, = ax.plot(i, row.mean_test_score, 'o', c='red',
                                    fillstyle="none", alpha=1, markersize=4*markersize,
                                    markeredgewidth=1)
    grid_par = [vs2.drop(d,'kernel') for d in grid_search.cv_results_['params']] if(not showmodes) else grid_search.cv_results_['params']
    fontsize = markersize*1.5
    # ax.yticks(fontsize=fontsize)
    # ax.xticks(fontsize=fontsize/1.5)
    # ax.xticks(range(len(results)), [str(x).strip("{}").replace("'", "") for x in grid_par], rotation=xlb_rotation)
    ax.set_xticks(np.arange(len(results)))
    xticklabels = ['\n'.join(str(x).strip("{}").replace("'", "").split(',')) for x in grid_par]
    ax.set_xticklabels(xticklabels, rotation=xlb_rotation, fontsize=fontsize)
    # if(DFP.isiterable(xbd)):
    #     ax.set_xlim(*tuple(np.sort(xbd)))
    if(DFP.isiterable(ybd)):
        ax.set_ylim(*tuple(np.sort(ybd)))
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
    ax.set_ylabel("Validation score")
    ax.set_xlabel("Parameter settings")
    ax.legend([marker_cv, marker_mean, marker_best], ["cv score", "mean score", "best parameter setting"], loc=(1.05, 0.4))
    if(isinstance(ret, dict)):
        ret['fig'] = fig
    return True

def plotPgirdScores(mdc, grid_search, fig=None, file=None, stamps=None, **kwags):
    stamps = stamps if(isinstance(stamps, list)) else []
    if(not isinstance(fig, vs3.plt.Figure)):
        fig = vs3.plt.Figure(figsize=getattr(mdc, 'pgreidFigsize', (15,8)))
    fig.clf()
    drawPgirdScores(mdc, grid_search, fig=fig, **kwags)
                    # ybd=(0.6, 1.1), markersize=5, xlb_rotation=90, showmodes=False, fig=fig, **kwags)
    file = file if(LOGger.isinstance_not_empty(file, str)) else os.path.join(
        mdc.get_graph_exp_fd(), '%s.jpg'%LOGger.stamp_process('',['pgirdScores']+stamps,'','','','_',for_file=True))
    LOGger.CreateFile(file, lambda f:vs3.end(fig, file=f))
    return True

def fitSklearn(mdc, np_X_train, np_y_train, validation_data=None, stamps=None, **kwags):
    stamps = stamps if(isinstance(stamps, list)) else []
    algorithm = mdc.algorithm
    algorithm_params = mdc.algorithm_params
    algorithm_param_grid = mdc.algorithm_param_grid
    mdc.grid_search = None
    try:
        if(algorithm_param_grid):
            mdc.addlog('均格搜尋的交叉驗證--GridSearchCV........................', colora=LOGger.OKCYAN)
            mdc.grid_search=GridSearchCV(algorithm(), algorithm_param_grid, scoring=mdc.scoring, cv=mdc.cv) #scoring='accuracy'需要自動對應模型任務, mdc.algorithm要經過compile
            mdc.grid_search.fit(np_X_train, np_y_train)
            """畫圖"""
            plotPgirdScores(mdc, mdc.grid_search, stamps=stamps)
            file = os.path.join(mdc.get_detail_exp_fd(), '%s.csv'%(LOGger.stamp_process('',['params', *mdc.get_stamps(full=True), *stamps],'','','','_')))
            LOGger.CreateFile(file, lambda f: pd.DataFrame(mdc.grid_search.cv_results_).to_csv(f))
            mdc.addlog('Best_parameters:%s'%str(mdc.grid_search.best_params_), colora=LOGger.WARNING)
            max_score = mdc.grid_search.best_score_
            mdc.addlog('Best cross-validation score:%.3f'%max_score, colora=LOGger.WARNING)
            mdc.addlog('Best estimator:%s'%str(mdc.grid_search.best_estimator_), colora=LOGger.WARNING)
            mdc.addlog('-------------------以最佳分數參數訓練資料---------------------------------')
            pba_params = dcp(mdc.grid_search.best_params_)
            pba_params.update({'probability':True})
            algorithm_params.clear()
            algorithm_params.update(pba_params)
            mdc.model = mdc.grid_search.best_estimator_ if(not mdc.probability) else algorithm(**pba_params).fit(np_X_train, np_y_train)
        else:
            algorithm_params.update({'probability':True}) if(mdc.probability) else None
            mdc.model = algorithm(**algorithm_params)
            # LOGger.addDebug(np_X_train, np_y_train)
            mdc.model.fit(np_X_train, np_y_train)
        mdc.export_model_detail()
    except Exception as e:
        LOGger.exception_process(e, logfile=os.path.join(mdc.exp_fd, 'log.txt'), stamps=['fitSklearn'])
        return False
    return True

def fitAutoSklearn(mdc, np_X_train, np_y_train, validation_data=None, stamps=None, deepParams=True, detailed=True, **kwags):
    stamps = stamps if(isinstance(stamps, list)) else []
    missionSystem = mdc.missionSystem
    missionSystem_params = mdc.missionSystem_params
    missionSystem_param_grid = mdc.missionSystem_param_grid
    mdc.grid_search = None
    try:
        if(missionSystem_param_grid):
            mdc.addlog('均格搜尋的交叉驗證--GridSearchCV........................', colora=LOGger.OKCYAN)
            mdc.grid_search=GridSearchCV(missionSystem(), missionSystem_param_grid, scoring=mdc.scoring, cv=mdc.cv) #scoring='accuracy'需要自動對應模型任務, mdc.algorithm要經過compile
            mdc.grid_search.fit(np_X_train, np_y_train)
            """畫圖"""
            plotPgirdScores(mdc, mdc.grid_search, stamps=stamps)
            file = os.path.join(mdc.get_detail_exp_fd(), '%s.csv'%(LOGger.stamp_process('',['params', *mdc.get_stamps(full=True), *stamps],'','','','_')))
            LOGger.CreateFile(file, lambda f: pd.DataFrame(mdc.grid_search.cv_results_).to_csv(f))
            mdc.addlog('Best_parameters:%s'%str(mdc.grid_search.best_params_), colora=LOGger.WARNING)
            max_score = mdc.grid_search.best_score_
            mdc.addlog('Best cross-validation score:%.3f'%max_score, colora=LOGger.WARNING)
            mdc.addlog('Best estimator:%s'%str(mdc.grid_search.best_estimator_), colora=LOGger.WARNING)
            mdc.addlog('-------------------以最佳分數參數訓練資料---------------------------------')
            pba_params = dcp(mdc.grid_search.best_params_)
            pba_params.update({'probability':True})
            missionSystem_params.clear()
            missionSystem_params.update(pba_params)
            mdc.model = mdc.grid_search.best_estimator_ if(not mdc.probability) else missionSystem(**pba_params).fit(np_X_train, np_y_train)
        else:
            missionSystem_params.update({'probability':True}) if(mdc.probability) else None
            mdc.model = missionSystem(**missionSystem_params)
            # LOGger.addDebug(np_X_train, np_y_train)
            mdc.model.fit(np_X_train, np_y_train)
        mdc.leaderboard = mdc.model.leaderboard(detailed=detailed)
    except Exception as e:
        LOGger.exception_process(e, logfile=os.path.join(mdc.exp_fd, 'log.txt'), stamps=['fitAutoSklearn'])
        return False
    export_models_leaderboard(mdc)
    mdc.export_model_detail()
    return True

def predictTestStandard(mdc, X_data, y_data, **kwags):
    exp_fd = mdc.exp_fd
    # addlog_ = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addloger(logfile='')))
    exp_fd = getattr(mdc, 'exp_fd', exp_fd)
    binary_inlier_code = getattr(mdc, 'binary_inlier_code', 1)
    mdcExports = getattr(mdc,'exports',{})
    setattr(mdc,'exports',mdcExports)
    debugFile = getattr(mdc,'debugFile','export.pkl')
    try:
        predictions = mdc.predict(X_data, binary_inlier_code=binary_inlier_code)
    except Exception as e:
        LOGger.exception_process(e, logfile=os.path.join(mdc.exp_fd, 'log.txt'))
        mdc.addlog('predicting error', type(X_data), debugFile, stamps=['predictTestStandard'], colora=LOGger.FAIL)
    else:
        if(isinstance(predictions, type(None))):
            return False
        mdc.addlog('pred shape:', *[str(x.shape) for x in predictions]) if(
            isinstance(predictions, list)) else mdc.addlog('pred shape:', str(predictions.shape))
        mdc.addlog('causes\n', str(MV.shapeOfTensor(X_data)))
        mdc.addlog('predictions\n', str(MV.shapeOfTensor(predictions)))
        mdc.addlog('outcomes\n', str(MV.shapeOfTensor(y_data)))
        mdcExports.update({'predictions':predictions, 'outcomes':y_data})
        success = True
    mdcExports.update({'causes':X_data})
    LOGger.CreateFile(debugFile, lambda f:DFP.joblib.dump(mdcExports, f))
    setattr(mdc,'exports',mdcExports)
    return success

def transformTensor2ListOfZones(p_data):
    if(isinstance(p_data, np.ndarray)):  p_data = LOGger.mylist([p_data.reshape(-1,1) if(len(p_data.shape)==1) else p_data])
    elif((isinstance(p_data, tuple) or (isinstance(p_data, list)) and not isinstance(p_data, LOGger.mylist))):  p_data = LOGger.mylist(p_data)
    return p_data

def draw1Dto1DEvaluation(mdc, xMin=0.0, xMax=1.0, xnBin=100, x_data=None, p_data=None, 
                         facts=(), fig=None, ax=None, stamps=None, tol=np.nan,
                         dataMarker='*', predictedMarker='o', modelCurveColor=(0,0,1),
                         dataColor=(0,0,1,0.3), predictedColor=(0,0,1,0.3), auxColor=(0,0,1,0.3), **kwags):
    if(not isinstance(x_data, np.ndarray)): x_data = np.linspace(xMin, xMax, xnBin)
    x_data = x_data.reshape(-1,1)
    if(not isinstance(p_data, np.ndarray)): p_data = mdc.predict(x_data)
    if(p_data.shape[0]!=x_data.shape[0]): p_data = mdc.predict(x_data)
    x_data = x_data.reshape(-1)
    p_data = p_data.reshape(-1)
    stamps = stamps if(isinstance(stamps, list)) else []
    fig, ax = vs3.pltinitial(fig, ax)
    LOGger.addDebug(id(fig), id(ax), id(ax.get_figure()))
    ax.plot(x_data, p_data, color=modelCurveColor, label=kwags.get('modelCurveLabel', 'model curve'))
    if(facts): 
        if(tol is np.nan):    
            try:
                tol = np.std(np.array(facts[1]).reshape(-1))
            except:
                tol = None
        ax.scatter(*facts, color=dataColor, s=kwags.get('s',20), marker=dataMarker, label=kwags.get('dataLabel', 'data'))
        p_verses_f = mdc.predict(np.array(facts[0]).reshape(-1,1)).reshape(-1)
        ax.scatter(facts[0], p_verses_f, color=predictedColor, s=kwags.get('s',20), 
                   marker=predictedMarker, label=kwags.get('predictedLabel', 'predicted'))
        if(tol is not None):    
            ax.plot(x_data, p_data + tol, color=(1,0,0), ls='--')
            ax.plot(x_data, p_data - tol, color=(1,0,0), ls='--')
        
        for i,x in enumerate(facts[0]):
            ymin = min(p_verses_f[i], facts[1][i])
            ymax = max(p_verses_f[i], facts[1][i])
            ax.vlines(x, ymin=ymin, ymax=ymax, ls='--', color=auxColor)
    return True

def determined_threshold(mdc, scores, contamination=None, total_selected_label=None, **kwags):
    addlog = getattr(mdc, 'addlog') if(hasattr(mdc, 'addlog')) else kwags.get('addlog', LOGger.addlog)
    if(contamination is None):  contamination = getattr(mdc,'contamination',0.1)
    setattr(mdc, 'contamination', min(max(contamination, 0), 0.5))
    addlog("contamination", contamination, colora='\033[92m')
    return MV.determined_threshold(scores, contamination=contamination, total_selected_label=total_selected_label, **kwags)

def drawUnfamScores(mdc, scores, threshold=None, fig=None, ax=None, stamps=None, colorThreshold=(1,0,0,0.3), **kwags):
    threshold = threshold if(threshold is not None) else getattr(mdc,'threshold_unfam',None)
    stamps = stamps if(isinstance(stamps, list)) else []
    fig, ax = vs3.pltinitial(fig, ax)
    if(not MV.drawScores(ax, scores, threshold=threshold, digit=7,
                         colorThreshold=colorThreshold, axst=ax.twinx(), stamps=stamps)):
        return False
    title = kwags.get('title')
    if(title):  ax.set_title(LOGger.stamp_process('',[title, ax.get_title()],'','','',' '))
    return True

def drawUnfamScoresWith1DCauses(mdc, x_data, scores, fig=None, ax=None, stamps=None, threshold=None, 
                                dataColor=(0,0,1,0.3), colorThreshold=(1,0,0,0.3), infrm=None, 
                                tresholdXPosRatio=np.array([1,9])/10, **kwags):
    stamps = stamps if(isinstance(stamps, list)) else []
    fig, ax = vs3.pltinitial(fig, ax)
    if(not MV.drawScoresWith1Dcauses(ax, x_data, scores, threshold=threshold, digit=7, dataColor=dataColor, 
                                     colorThreshold=colorThreshold, stamps=stamps, infrm=infrm, tresholdXPosRatio=tresholdXPosRatio,
                                     xlabel=mdc.xheader[0])):
        return False
    return True

def absUnfamEvaluationScenario(mdc, scoreMethod, confidencifyMethod, X_data, thresholdName='threshold_unfam',
                               handler=None, qualifiedMask=None, stamps=None, determine_threshold=True, ax=None, **kwags):
    if(not mdc.unfam_header_zones):
        return True
    addlog_ = mdc.addlog
    unfamScores = scoreMethod(X_data, zone_index=getattr(mdc, 'unfam_header_zone_index', -1))
    thresholdTemp = getattr(mdc, thresholdName, None)
    if(determine_threshold):    
        thresholdTemp = determined_threshold(mdc, unfamScores, total_selected_label=qualifiedMask, **kwags)
        setattr(mdc, thresholdName, thresholdTemp)
        addlog('mdc.%s:%s'%(thresholdName, getattr(mdc, thresholdName)), stamps=mdc.get_stamps()+['unfam_prop'], colora=LOGger.OKGREEN)
    sta_prop = LOGger.statistics_properties(unfamScores, parse_method=lambda s:LOGger.parse(s, digit=5))
    stamps = stamps if(isinstance(stamps, list)) else []
    addlog_('unfam_stat:\n%s'%str(pd.DataFrame({'unfam_stat':sta_prop})), stamps=stamps+['unfam_stat'], colora=LOGger.OKGREEN)
    
    eval_data = getattr(handler,'eval_data',None)
    if(isinstance(eval_data, pd.core.frame.DataFrame)):
        confidences = confidencifyMethod(X_data)
        eval_data['unfamScores'] = unfamScores.reshape(-1)
        if(not thresholdTemp in [np.inf, -np.inf, np.nan, None]):
            eval_data['unfamMask'] = (unfamScores.reshape(-1) >= thresholdTemp)
        eval_data['confidences'] = confidences.reshape(-1)
    if(not drawUnfamScores(mdc, unfamScores, stamps=stamps, ax=ax, threshold=thresholdTemp, title=kwags.get('title'))):
        addlog_('drawUnfamScores failed!!!', colora=LOGger.FAIL)
    return True

def unfamEvaluationScenario(mdc, X_data, handler=None, qualifiedMask=None, stamps=None, determine_threshold=True, ax=None, **kwags):
    if(not absUnfamEvaluationScenario(mdc, mdc.predict_score, mdc.predict_confidence, X_data, thresholdName='threshold_unfam',
                                    handler=handler, qualifiedMask=qualifiedMask, stamps=stamps, 
                                    determine_threshold=determine_threshold, ax=ax, **kwags)):
        return False
    return True
    
def unfamEvaluationLatentSpace(mdc, X_data, handler=None, qualifiedMask=None, stamps=None, determine_threshold=False, ax=None, **kwags):
    handlerTemp = LOGger.mystr()
    handlerTemp.eval_data = pd.DataFrame(index=X_data.index)

    if(not absUnfamEvaluationScenario(mdc, (lambda x, **kws:latentScoreFront(mdc, x, **kws)), 
                                    (lambda x, **kws:predict_AElatentConfidence(mdc, x, **kws)), 
                                    X_data, thresholdName='threshold_unfam_latentSpace',
                                    handler=handlerTemp, qualifiedMask=qualifiedMask, stamps=stamps, 
                                    determine_threshold=determine_threshold, ax=ax, title='latentSpace', **kwags)):
        return False
    eval_data = getattr(handler,'eval_data',None)
    if(isinstance(eval_data, pd.core.frame.DataFrame)):
        eval_data['latentUnfamScores'] = pd.Series(handlerTemp.eval_data['unfamScores'].values, index=X_data.index)
        if(not mdc.threshold_unfam_latentSpace in [np.inf, -np.inf, np.nan, None]):
            eval_data['latentUnfamMask'] = pd.Series(handlerTemp.eval_data['unfamMask'].values, index=X_data.index)
        eval_data['latentConfidences'] = pd.Series(handlerTemp.eval_data['confidences'].values, index=X_data.index)
    return True

def drawUnfamWith1DcauseScenario(mdc, X_data, handler=None, qualifiedMask=None, stamps=None, determine_threshold=True, 
                                 fig=None, figsize=(8,10), unfamScoreDataColor=(0,1,0,0.3), confiDataColor=(0,0,1,0.3), **kwags):
    addlog_ = mdc.addlog
    if(not mdc.unfam_header_zones):
        mdc.addlog('mdc has no unfam_header_zones!!!', colora=LOGger.WARNING)
        return True
    if(len(X_data.shape)>2 or len(X_data.shape)==0 or (len(X_data.shape)==2 and LOGger.mylist(X_data.shape)[1]!=1)):
        mdc.addlog('X_data shape error', str(X_data.shape), colora=LOGger.FAIL)
        return False
    unfamScores = mdc.predict_score(X_data, zone_index=getattr(mdc, 'unfam_header_zone_index', -1))
    confidences = mdc.predict_confidence(X_data)
    if(determine_threshold):    
        mdc.threshold_unfam = determined_threshold(mdc, unfamScores, total_selected_label=qualifiedMask, **kwags)
        addlog('threshold_unfam:%s'%mdc.threshold_unfam, stamps=mdc.get_stamps()+['unfam_prop'], colora=LOGger.OKGREEN)
    sta_prop = LOGger.statistics_properties(unfamScores, parse_method=lambda s:LOGger.parse(s, digit=5))
    addlog_('unfam_stat:\n%s'%str(pd.DataFrame({'unfam_stat':sta_prop})), stamps=stamps+['unfam_stat'], colora=LOGger.OKGREEN)
    stamps = stamps if(isinstance(stamps, list)) else []
    
    if(isinstance(fig, vs3.plt.Figure)):
        fig.clf()
    else:
        fig = vs3.plt.Figure(figsize=figsize)
    axUnfamScore = fig.add_subplot(1,1,1)
    axConfi = axUnfamScore.twinx()
    x_data = getattr(X_data,'values',X_data).reshape(-1)
    LOGger.addDebug('drawUnfamWith1DcauseScenario', axUnfamScore, axConfi)
    if(not drawUnfamScoresWith1DCauses(mdc, x_data, unfamScores, ax=axUnfamScore, dataColor=unfamScoreDataColor, 
                                       threshold=getattr(mdc,'threshold_unfam',None), colorThreshold=unfamScoreDataColor, stamps=['unfamScore'],
                                       tresholdXPosRatio=np.array([1,9])/10)):
        addlog_('drawUnfamScoresWith1DCauses axUnfamScore failed!!!', colora=LOGger.FAIL)
    if(not drawUnfamScoresWith1DCauses(mdc, x_data, confidences, ax=axConfi, dataColor=confiDataColor,
                                       colorThreshold=confiDataColor, threshold=0.5, stamps=['confidence'],
                                       tresholdXPosRatio=np.array([3,7])/10)):
        addlog_('drawUnfamScoresWith1DCauses axConfi failed!!!', colora=LOGger.FAIL)
    axConfi.set_title('')
    axUnfamScore.set_title('')
    suptitle = LOGger.stamp_process('',[*stamps],'','','',' ')
    fig.suptitle(suptitle)
    return True

def plotUnfamWith1DcauseScenario(mdc, X_data, handler=None, qualifiedMask=None, stamps=None, determine_threshold=True, 
                                 fig=None, figsize=(8,10), exp_fd='.', file='', **kwags):
    stamps = stamps if(isinstance(stamps, list)) else []
    if(isinstance(fig, vs3.plt.Figure)):
        fig.clf()
    else:
        fig = vs3.plt.Figure(figsize=figsize)
    if(not drawUnfamWith1DcauseScenario(mdc, X_data, handler=handler, qualifiedMask=qualifiedMask, stamps=stamps, 
                                        determine_threshold=determine_threshold, fig=fig, figsize=figsize, **kwags)):
        return False
    fn = '%s.jpg'%LOGger.stamp_process('',['UnfamWith1Dcause', *stamps],'','','','_',for_file=True)
    file = file if(LOGger.isinstance_not_empty(file, str)) else os.path.join(exp_fd, fn)
    LOGger.CreateFile(file, lambda f:vs3.end(fig, file=f))
    return True

def predictWithScoresStandard(mdc, X_data, **kwags):
    try:
        output_raw = executeFront(mdc, X_data, **kwags)
        if(isinstance(output_raw, type(None))):
            return None
        p_data = transformTensor2ListOfZones(mdc.inverse_transform(
            output_raw, header_zones_name=kwags.get('header_zones_name', 'yheader_zones')))
        unfam_header_zone_index = getattr(mdc,'unfam_header_zone_index',-1)
        s_data = transformTensor2ListOfZones([*(
            output_raw[unfam_header_zone_index%len(output_raw):(unfam_header_zone_index%len(output_raw))+1])] if(
            getattr(mdc, 'unfam_header', None)) else None)
    except Exception as e:
        LOGger.exception_process(e, stamps=['predictWithScoresStandard'], logfile='')
        return None, None
    return p_data, s_data

def predictWithScoresAE(mdc, X_data, **kwags):
    try:
        output_reduced = predictFront(mdc, X_data, **kwags)
        output_regenerated = regenerateFront(mdc, X_data, **kwags)
        if(output_reduced is None):
            mdc.addlog('reduced None!!!', colora=LOGger.FAIL)
            return None
        if(output_regenerated is None):
            mdc.addlog('regenerated None!!!', colora=LOGger.FAIL)
            return None
        p_data = transformTensor2ListOfZones(mdc.inverse_transform_reduce(output_reduced))
        unfam_header_zone_index = getattr(mdc,'unfam_header_zone_index',-1)
        s_data = transformTensor2ListOfZones([*(
            output_regenerated[unfam_header_zone_index%len(output_regenerated):
                               (unfam_header_zone_index%len(output_regenerated))+1])] if(
            getattr(mdc, 'unfam_header', None)) else None)
    except Exception as e:
        LOGger.exception_process(e, stamps=['predictWithScoresAE'], logfile='')
        return None, None
    return p_data, s_data

def predictWithScoresSklearn(mdc, X_data, **kwags):
    try:
        output_raw = executeFront(mdc, X_data, **kwags)
        if(isinstance(output_raw, type(None))):
            return None
        p_data = transformTensor2ListOfZones(mdc.inverse_transform(output_raw))
        s_data = LOGger.mylist([mdc.predict_score(X_data)])
    except Exception as e:
        LOGger.exception_process(e, stamps=['directPredictionKeras'], logfile='')
        return None, None
    return p_data, s_data

def regenerateWithScores(mdc, X_data, header_zones_name='xheader_zones', **kwags):
    return predictWithScoresStandard(mdc, X_data, header_zones_name='xheader_zones', **kwags)

def predictRegenerateScoring(mdc, X_data, **kwags):
    try:
        output_reduced = predictFront(mdc, X_data, **kwags)
        output_regenerated = regenerateFront(mdc, X_data, **kwags)
        if(output_reduced is None):
            mdc.addlog('reduced None!!!', colora=LOGger.FAIL)
            return None
        if(output_regenerated is None):
            mdc.addlog('regenerated None!!!', colora=LOGger.FAIL)
            return None
        unfam_header_zone_index = getattr(mdc,'unfam_header_zone_index',-1)
        p_data = transformTensor2ListOfZones(mdc.inverse_transform_reduce(output_reduced))
        r_data = transformTensor2ListOfZones(mdc.inverse_transform_regenerate(output_regenerated))
        s_data = transformTensor2ListOfZones([*(
            output_regenerated[unfam_header_zone_index%len(output_regenerated):
                               (unfam_header_zone_index%len(output_regenerated))+1])] if(
            getattr(mdc, 'unfam_header', None)) else None)
    except Exception as e:
        LOGger.exception_process(e, stamps=['predictWithScoresAE'], logfile='')
        return None, None
    return p_data, r_data, s_data

def evaluationStandard(mdc, X_data, y_data, p_data=None, handler=None, ret=None, stamps=None, savePlot=True, **kwags):
    stamps = stamps if(isinstance(stamps, list)) else []
    addlog_ = mdc.addlog
    s_data = LOGger.mylist([])
    if(p_data is None): 
        p_data, s_data = mdc.predictWithScores(X_data)
    if(not isinstance(p_data, LOGger.mylist)):  
        addlog_('p_data not type of mylistOfZones!!! `%s`'%DFP.parse(p_data), stamps=[evaluationStandard.__name__]+mdc.stamps+stamps, colora=LOGger.FAIL)
        return False
    if(not isinstance(s_data, LOGger.mylist)):  
        addlog_('s_data not type of mylistOfZones!!! `%s`'%DFP.parse(s_data), stamps=[evaluationStandard.__name__]+mdc.stamps+stamps, colora=LOGger.FAIL)
        return False
    retTemp = {}
    qualifiedMask = np.full(y_data.shape[0], True)
    colIter = 0
    for i,(k,v) in enumerate(mdc.yheader_zones.items()):
        retTemp.clear()
        outcome = dcp(y_data[v] if(isinstance(y_data, pd.core.frame.DataFrame)) else y_data[:,colIter:colIter+len(v)])
        colIter += len(v)
        prediction = dcp(p_data[i])
        score = dcp(s_data[i])
        method = v.evaluationAndPlot if(savePlot) else v.evaluation
        if(not method(outcome, prediction, np_y_score=score, ret=retTemp, stamps=stamps, **kwags)):
            addlog_('evaluation error!!!', stamps=[evaluationStandard.__name__]+mdc.stamps+stamps+[k], colora=LOGger.FAIL)
            addlog_('outcome', DFP.parse(outcome), stamps=[evaluationStandard.__name__]+mdc.stamps+stamps+[k], colora=LOGger.FAIL)
            addlog_('prediction', DFP.parse(prediction), stamps=[evaluationStandard.__name__]+mdc.stamps+stamps+[k], colora=LOGger.FAIL)
            return False
        # LOGger.addDebug(str(v.data_prop.export))
        if(isinstance(ret, dict)):  ret[k] = dcp(retTemp)
        if('mask' in retTemp):  qualifiedMask = qualifiedMask & dcp(retTemp['mask'])
        if(isinstance(getattr(handler,'eval_data', None), pd.core.frame.DataFrame)): 
            partial_eval_data = dcp(v.data_prop.eval_data)
            if(len(mdc.yheader_zones)>1):   partial_eval_data.columns = list(map((lambda x:LOGger.stamp_process('',[k,x],'','','','_')), partial_eval_data.columns))
            handler.eval_data = dcp(partial_eval_data if(handler.eval_data.empty) else handler.eval_data.join(partial_eval_data, sort=False))
    if(isinstance(ret, dict)):  ret['mask'] = qualifiedMask
    return True

def drawLatentMap(ax, mapping, stamps=None, fontproperties=None, **kwags):
    stamps = stamps if(isinstance(stamps, list)) else []
    stamp = LOGger.stamp_process('',stamps+['classCoordMapping'],'','','','_')
    mappingValues = list(zip(*list(mapping.values())))
    LOGger.addDebug('drawClassesLatentMap stamp', stamp)
    ax.scatter(*mappingValues, label=stamp)
    for k,v in mapping.items():
        if(np.array(v).shape[0]!=2):
            continue
        ax.text(*v, '[%s]:(%s,%s)'%tuple(list(map(DFP.parse, (k,*v)))), fontproperties=fontproperties)
    return True

def drawLatentMapToPlaneView(ax, latent_2d, labels=None, colorDefault=(0,0,1,0.3), colorMask=None, **kwags):
    colorMask = LOGger.mylist(colorMask if(isinstance(colorMask, list)) else [colorDefault]*(latent_2d.shape[0]))
    ax.scatter(latent_2d[:,0], latent_2d[:,1], c=colorMask)
    return True

def sampleRandomPoints(tensor2d, n_points):
    n_points = min(n_points, tensor2d.shape[0])
    if(n_points<1):
        return []
    if(tensor2d.shape[0]==0):
        return []
    kmeans = KMeans(n_clusters=n_points)
    kmeans.fit(tensor2d)
    
    # 對於每個聚類，選擇最接近中心的點
    selected = []
    for center in kmeans.cluster_centers_:
        # 計算所有點到該中心的距離
        distances = np.linalg.norm(tensor2d - center, axis=1)
        # 選擇最近的點
        closest_point_idx = np.argmin(distances)
        selectedTemp = tensor2d[closest_point_idx]
        selectedIdx = np.argmin(np.linalg.norm(tensor2d - selectedTemp, axis=1))
        selected.append(selectedIdx)
    return selected

def drawLatentMapToPlaneViewAndCompareDecoded(
    ax, X_data, p_data, r_data, labels=None, colorDefault=(0,0,0,0.3), colorMask=None, n_display=10, 
    displayScatterSize=40, displayColorDefault=None,
    displayWidth=0.2, displayHeight=0.2, base=0, stamps=None, originalDataColor=(0,0,1,0.3), reconstructedDataColor=(0,1,0,0.3), 
    x_range=None, y_range=None, **kwags):
    addlog_ = kwags.get('addlog', m_print)
    stamps = stamps if(isinstance(stamps, list)) else []
    # 使用t-SNE或UMAP進行降維可視化
    displayIdxes = sampleRandomPoints(p_data, n_display) #list
    displayMask = np.full(p_data.shape[0], False)
    displayMask[displayIdxes] = True
    LOGger.addDebug('drawLatentMapToPlaneViewAndCompareDecoded',str(displayIdxes))
    colorMask = np.array(colorMask if(isinstance(colorMask, list)) else [colorDefault]*(p_data.shape[0]))
    ax.scatter(p_data[np.logical_not(displayMask)][:,0], p_data[np.logical_not(displayMask)][:,1], c=colorDefault)
    displayColorDefault = displayColorDefault if(displayColorDefault is not None) else colorDefault
    ax.scatter(p_data[displayMask][:,0], p_data[displayMask][:,1], edgecolor='red', linewidth=0.5,
               c=displayColorDefault, s=displayScatterSize, marker='*')
    if(x_range is None):
        pw = (np.max(p_data[:,0]) - np.min(p_data[:,0]))*displayWidth
        x_range = (np.min(p_data[:,0]), np.max(p_data[:,0]) + pw)
    if(y_range is None):
        ph = (np.max(p_data[:,1]) - np.min(p_data[:,1]))*displayHeight
        y_range = (np.min(p_data[:,1]), np.max(p_data[:,1]) + ph)
    h = (y_range[1]-y_range[0])*np.clip(displayHeight,0.05,0.5)
    w = (x_range[1]-x_range[0])*np.clip(displayWidth,0.05,0.5)
    for i in displayIdxes:
        p_data_sample = dcp(p_data[i])
        X_data_sample = dcp(X_data[i])
        r_data_sample = dcp(r_data[i])
        rect = [p_data_sample[0]+w*0.1, p_data_sample[1]+h*0.1, w, h]
        sub_ax = ax.inset_axes(rect, transform=ax.transData)
        if(not drawCellIllustration(sub_ax, X_data_sample, stamps=stamps+[i]+[str(tuple(p_data_sample))], 
                                    base=base, dataColor=originalDataColor)):
            addlog_('original data drawCellIllustration failed!!!', colora=LOGger.FAIL, 
                stamps=[drawLatentMapToPlaneViewAndCompareDecoded.__name__]+stamps+[i]+[str(tuple(p_data_sample))])
        if(not drawCellIllustration(sub_ax, r_data_sample, stamps=stamps+[i]+[str(tuple(p_data_sample))], 
                                    base=base, dataColor=reconstructedDataColor)):
            addlog_('reconstructed data drawCellIllustration failed!!!', colora=LOGger.FAIL, 
                stamps=[drawLatentMapToPlaneViewAndCompareDecoded.__name__]+stamps+[i]+[str(tuple(p_data_sample))])
    return True

def drawCellIllustration(ax, data, dataColor=(0,0,0,0.3), stamps=None, valueMin=None, valueMax=None, base=0, **kwags):
    """
    根據數據類型繪製重構結果
    這個函數需要根據具體的數據類型來實現
    """
    stamps = stamps if(isinstance(stamps, list)) else []
    stamp = LOGger.stamp_process('',stamps,'','','','_')
    # 如果是圖像數據
    if(isinstance(data, (list, tuple))):
        if(not drawCellIllustration(ax, np.array(data), dataColor=dataColor, stamps=stamps, valueMax=valueMax, valueMin=valueMin, base=base)):
            return False
        return True
    elif(hasattr(data, 'values')):
        if(not drawCellIllustration(ax, data.values, dataColor=dataColor, stamps=stamps, valueMax=valueMax, valueMin=valueMin, base=base)):
            return False
        return True
    if(np.isnan(np.vectorize(lambda x:DFP.astype(x, default=np.nan))(data)).any()):
        ax.text(0,0,'[%s]invalid num data!!!'%stamp, color='red')
        return False
    valueMin = np.min(data) if(valueMin is None) else valueMin
    valueMax = np.max(data) if(valueMax is None) else valueMax
    if(len(data.shape) > 3):
        ax.text(0,0,'[%s]data shape:%s!!!'%(stamp,str(data.shape)))
    elif(len(data.shape) == 3):
        dataImg = dcp(data)
        dataImg = np.clip(dataImg, valueMin, valueMax)
        dataImg = ((dataImg - valueMin) / (valueMax - valueMin) * 255).astype(np.uint8)
        if dataImg.shape[-1] == 3:
            # RGB 圖像直接顯示
            ax.imshow(dataImg)
        elif dataImg.shape[-1] == 1:
            # 單通道，顯示為灰度圖
            ax.imshow(dataImg.squeeze(), cmap='gray')
        else:
            # 多通道，可以選擇：
            # 1. 取平均值顯示為灰度圖
            ax.imshow(np.mean(dataImg, axis=-1), cmap='gray')
    elif len(data.shape) == 2:
        vs3.report_InAFrame(data, ax=ax, is_end=False, base=base, color_default=dataColor, colors={})
        # ax.scatter(tuple(zip(*data)), color=dataColor, label=stamp)
    # 如果是一維數據
    elif len(data.shape) == 1:
        ax.plot(np.arange(data.shape[0]), data, color=dataColor, label=stamp)
    elif(len(data.shape) < 1):
        ax.text(0,0,'[%s]data:%s!!!'%(stamp,DFP.parse(data,digit=5)))
    # 可以根據需要添加其他類型的可視化
    return True

def drawLatentMapToPlaneGrid(mdc, ax, x_range=(-2,2), y_range=(-2,2), grid_size=(5,5), dataColor=(0,0,1,0.3), dataMarker='o',
                            subplot_size=(0.2,0.2), stamps=None, valueMin=None, valueMax=None, nTo2DimTransformer=None, **kwags):
    """
    在潛在空間的網格點上繪製重構結果
    
    Parameters:
    -----------
    mdc : EIMS_AUTOENCODER_core
        訓練好的自編碼器模型
    ax : matplotlib.axes.Axes
        主圖的軸對象
    x_range : tuple
        x軸的範圍 (min, max)
    y_range : tuple
        y軸的範圍 (min, max)
    grid_size : tuple
        網格的大小 (nx, ny)
    subplot_size : tuple
        子圖的相對大小 (width, height)
    """
    stamps = stamps if(isinstance(stamps, list)) else []
    # 生成網格點
    x = np.linspace(x_range[0], x_range[1], grid_size[0])
    y = np.linspace(y_range[0], y_range[1], grid_size[1])
    xx, yy = np.meshgrid(x, y)
    
    # 繪製主網格
    ax.plot(xx, yy, 'k-', alpha=0.2)
    ax.plot(xx.T, yy.T, 'k-', alpha=0.2)
    
    # 將所有網格點整合成一個批次
    grid_points = np.stack([xx.flatten(), yy.flatten()], axis=1)
    grid_points_oriDim = nTo2DimTransformer.inverse_transform(grid_points) if(nTo2DimTransformer is not None) else grid_points
    ax.scatter(grid_points[:,0], grid_points[:,1], color=dataColor, marker=dataMarker)
    
    h = (y_range[1]-y_range[0])/max(grid_size[1]-1,1)
    w = (x_range[1]-x_range[0])/max(grid_size[0]-1,1)
    # 一次性進行所有點的解碼
    reconstructed_batch = mdc.decoder.predict(grid_points_oriDim)  #TODO: inverse_transform
    reconstructed_batch = mdc.inverse_transform_regenerate(reconstructed_batch)
    insetAxesListTemp = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # 計算子圖的位置
            # LOGger.addDebug(i,j, xx[i,j], yy[i,j], subplot_size[0], subplot_size[1])
            # rect = [xx[i,j], yy[i,j], subplot_size[0], subplot_size[1]]
            rect = [xx[i,j]+w*0.1, yy[i,j]+h*0.1, w*0.5, h*0.5]
            # rect = [i*(1/max(1,grid_size[0]-1)), j*(1/max(1,grid_size[1]-1)), 1/(grid_size[0]*2), 1/(grid_size[1]*2)]
            sub_ax = ax.inset_axes(rect, transform=ax.transData)
            insetAxesListTemp.append((sub_ax, (i,j)))
    insetAxesListTemp2 = []
    for s,inset in enumerate(insetAxesListTemp):
        infrmStgListTemp = []
        for h,(k,v) in enumerate(mdc.xheader_zones.items()):
            cell_size = getattr(v, 'cell_size', None)
            if(DFP.isiterable(cell_size)):  continue
            reconstructed_batch_zone = dcp(reconstructed_batch[h])
            infrm = dict(zip(*(v,dcp(reconstructed_batch_zone[s]))))
            infrmStg = LOGger.stamp_process('',infrm)
            infrmStgListTemp.append(infrmStg)
        infrmStgTotal = LOGger.stamp_process('',infrmStgListTemp,'','','','\n')
        insetAxesListTemp2.append(infrmStgTotal)
    for h,(k,v) in enumerate(mdc.xheader_zones.items()):
        cell_size = getattr(v, 'cell_size', None)
        if(not DFP.isiterable(cell_size)):  continue
        reconstructed_batch_zone = dcp(reconstructed_batch[h])
        if(valueMax is None):   valueMax = np.nanmax(reconstructed_batch_zone)
        if(valueMin is None):   valueMin = np.nanmin(reconstructed_batch_zone)
        # 在每個網格點生成重構圖
        for s,inset in enumerate(insetAxesListTemp):
            sub_ax = inset[0]
            i,j = inset[1]
            reconstructed = dcp(reconstructed_batch_zone[s])
            if(reconstructed.shape[0]==1 and len(reconstructed.shape)>1):  reconstructed = reconstructed[0]
            # 繪製重構結果（需要根據數據類型調整）
            if(not drawCellIllustration(sub_ax, reconstructed, valueMin=valueMin, valueMax=valueMax, stamps=stamps+[i,j], 
                                        base=getattr(mdc, 'cellBaseIndex', 0))):
                mdc.addlog('drawCellIllustration failed!!!', colora=LOGger.FAIL, 
                        stamps=[drawLatentMapToPlaneGrid.__name__]+stamps+[i,j])
        break
    for s,inset in enumerate(insetAxesListTemp):
        infrmStg = dcp(insetAxesListTemp2[s])
        sub_ax = inset[0]
        sub_ax.text(sub_ax.get_xlim()[0], sub_ax.get_ylim()[0], dcp(infrmStg), fontsize=12, ha='left', va='bottom')
        sub_ax.patch.set_alpha(0.5)
    return True

def drawAERegionDyeing(mdc, ax, x_range=None, y_range=None, stamps=None, nTo2DimTransformer=None, **kwags):
    x_min, x_max = tuple(ax.get_xlim()) if(x_range is None) else x_range
    y_min, y_max = tuple(ax.get_ylim()) if(y_range is None) else y_range
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    basePlane = np.c_[xx.ravel(), yy.ravel()]
    baseSpace = nTo2DimTransformer.inverse_transform(basePlane) if(nTo2DimTransformer is not None) else basePlane
    LOGger.addDebug(basePlane, baseSpace)
    Z_lof = -mdc.latentExplainer.score_samples(baseSpace).reshape(xx.shape)
    contour = ax.contourf(xx, yy, Z_lof, cmap=vs3.plt.cm.PuBu, zorder=0)
    fig = ax.get_figure()
    fig.colorbar(contour, ax=ax, label='Anomaly Score')
    return True

def evaluationAERegeneration(mdc, X_data, r_data=None, s_data=None, ret=None, handler=None, stamps=None, 
                            savePlot=True, **kwags):
    stamps = stamps if(isinstance(stamps, list)) else []
    addlog_ = mdc.addlog
    s_data = LOGger.mylist([])
    if(r_data is None): #p_data 在這裡是再制訊號
        r_data, s_data = mdc.regenerateWithScores(X_data)
    if(not isinstance(r_data, LOGger.mylist)):  
        addlog_('r_data not type of mylistOfZones!!! %s'%DFP.parse(r_data), stamps=[evaluationAERegeneration.__name__]+mdc.stamps+stamps, colora=LOGger.FAIL)
        return False
    if(not isinstance(s_data, LOGger.mylist)):  
        addlog_('s_data not type of mylistOfZones!!! %s'%DFP.parse(s_data), stamps=[evaluationAERegeneration.__name__]+mdc.stamps+stamps, colora=LOGger.FAIL)
        return False
    
    addlog_ = mdc.addlog
    retTemp = {}
    qualifiedMask = np.full(X_data.shape[0], True)
    # 重构误差评估
    colIter = 0
    for i,(k,v) in enumerate(mdc.xheader_zones.items()):
        retTemp.clear()
        input_data = dcp(X_data[v] if(isinstance(X_data, pd.core.frame.DataFrame)) else X_data[:,i:i+len(v)])
        np_input_data = DFP.transformByBatch(input_data, (lambda x:x.tolist()))
        regenerated = dcp(r_data[i])
        cell_size = getattr(v, 'cell_size', None)
        # 计算重构误差
        try:
            mse = np.mean((np_input_data - regenerated) ** 2, axis=(tuple(np.arange(len(cell_size)+1)+1) if(DFP.isiterable(cell_size)) else 1))#np.nan #
            mae = np.mean(np.abs(np_input_data - regenerated), axis=(tuple(np.arange(len(cell_size)+1)+1) if(DFP.isiterable(cell_size)) else 1)) #np.nan #
        except Exception as e:
            LOGger.exception_process(e, stamps=[evaluationAERegeneration.__name__]+mdc.stamps+stamps+[k], logfile='')
            m_debug['X_data'] = X_data
            m_debug['r_data'] = r_data
            m_debug['v'] = v
            m_debug['i'] = i
            m_debug.dump()
        # 记录评估结果
        retTemp[k] = {
            'mse': mse,
            'mae': mae,
            'input_data': input_data,
            'reconstruction': regenerated
        }
        # 可视化重构效果
        # original = dcp(X_data[v] if(isinstance(X_data, pd.core.frame.DataFrame)) else X_data[:,colIter:colIter+len(v)])
        colIter += len(v)
        score = dcp(s_data[i])
        method = v.evaluationAndPlot if(savePlot) else v.evaluation
        if(not method(input_data, regenerated, np_y_score=score, ret=retTemp, stamps=stamps, **kwags)):
            addlog_('evaluation error!!!', stamps=[evaluationAERegeneration.__name__]+mdc.stamps+stamps+[k], colora=LOGger.FAIL)
            addlog_('original', DFP.parse(input_data), stamps=[evaluationAERegeneration.__name__]+mdc.stamps+stamps+[k], colora=LOGger.FAIL)
            addlog_('prediction', DFP.parse(regenerated), stamps=[evaluationAERegeneration.__name__]+mdc.stamps+stamps+[k], colora=LOGger.FAIL)
            return False
        if('mask' in retTemp):  qualifiedMask = qualifiedMask & dcp(retTemp['mask'])
        if(isinstance(getattr(handler,'eval_data', None), pd.core.frame.DataFrame)): 
            partial_eval_data = dcp(v.data_prop.eval_data)
            if(len(mdc.xheader_zones)>1):   partial_eval_data.columns = list(map((lambda x:LOGger.stamp_process('',[k,x],'','','','_')), partial_eval_data.columns))
            handler.eval_data = dcp(partial_eval_data if(handler.eval_data.empty) else handler.eval_data.join(partial_eval_data, sort=False))
    if(isinstance(ret, dict)):  ret['mask'] = dcp(qualifiedMask)
    if(isinstance(ret, dict)):  ret['AERegeneration'] = dcp(retTemp)
    return True

def determineLatentPca(p_data, mdc, handler=None, ret=None, **kwags):
    ret = ret if(isinstance(ret, dict)) else {}
    p_data_2d, pca = dcp(p_data), None
    if(len(p_data.shape)==1):
        p_data = p_data.reshape(-1,1)
    elif(len(p_data.shape)!=2):
        mdc.latentPca = pca
        ret['p_data_2d'] = p_data_2d
        return True
    if(p_data.shape[1]!=2):
        pca = PCA(n_components=2)
        p_data_2d = pca.fit_transform(p_data)
        mdc.latentPca = pca
        ret['p_data_2d'] = p_data_2d
        if(isinstance(getattr(handler, 'eval_data', None), pd.core.frame.DataFrame)):    handler.eval_data[['xLatentPCA','yLatentPCA']] = p_data_2d
    return True

def evaluationAELatentSpace(mdc, X_data, p_data=None, r_data=None, s_data=None, ret=None, handler=None, stamps=None, 
                            subaxDisplayHeight=0.2, subaxDisplayWidth=0.2, inlierMask=None, **kwags):
    # encoder_hidden_layer_sizes = mdc.encoder_hidden_layer_sizes
    latentDim = len(mdc.yheader)
    
    stamps = stamps if(isinstance(stamps, list)) else []
    addlog_ = mdc.addlog
    # 潜在空间分析
    try:
        # 获取潜在表示
        if(p_data is None): p_data = mdc.predict(X_data)
        if(r_data is None): r_data = mdc.regenerate(X_data)
        # LOGger.addDebug('evaluationAELatentSpace DEBUGGING...',str(p_data))
        if(isinstance(p_data, list)):   p_data = p_data[0]
        # 计算潜在空间的统计特性
        latent_mean = np.mean(p_data, axis=0)
        latent_std = np.std(p_data, axis=0)
        
        # 记录潜在空间信息
        if(isinstance(ret, dict)):
            ret.update({
                'representations': p_data,
                'mean': latent_mean,
                'std': latent_std
            })
        if(isinstance(getattr(handler, 'eval_data', None), pd.core.frame.DataFrame)):    handler.eval_data[mdc.yheader] = p_data
        retTemp = {}
        if(mdc.latentPca is None):
            if(not determineLatentPca(p_data, mdc, handler=handler, ret=retTemp)):
                return False
            p_data_2d = dcp(retTemp['p_data_2d'])
        else:
            p_data_2d = mdc.latentPca.transform(p_data)
        ph = (np.max(p_data_2d[:,1]) - np.min(p_data_2d[:,1]))*subaxDisplayHeight
        y_range = (np.min(p_data_2d[:,1]), np.max(p_data_2d[:,1]) + ph)
        pw = (np.max(p_data_2d[:,0]) - np.min(p_data_2d[:,0]))*subaxDisplayWidth
        x_range = (np.min(p_data_2d[:,0]), np.max(p_data_2d[:,0]) + pw)
        LOGger.addDebug(str(x_range), str(tuple(map((lambda x:DFP.parse(x,digit=7)), y_range))))
        if(isinstance(getattr(mdc, 'latentAx', None), vs3.plt.Axes)):   
            try:
                # drawAERegionDyeing(mdc, mdc.latentAx, x_range=x_range,y_range=y_range,stamps=stamps, nTo2DimTransformer=pca)
                drawLatentMapToPlaneView(mdc.latentAx, p_data_2d, labels=None, colorDefault=(0,0,1,0.3), colorMask=None, 
                                         x_range=x_range, y_range=y_range, **kwags)
                drawAERegionDyeing(mdc, mdc.latentAx ,stamps=stamps, nTo2DimTransformer=pca)
                mdc.latentAx.set_xlim(x_range)
                mdc.latentAx.set_ylim(y_range)
                mdc.latentAx.set_title(LOGger.stamp_process('',stamps+['latentSpace'],'','','','_'))
            except Exception as e:
                LOGger.exception_process(e, stamps=[evaluationAELatentSpace.__name__]+mdc.stamps+stamps, logfile='')
                m_debug['p_data'] = p_data
                m_debug['pca'] = pca
                m_debug['latentExplainer'] = mdc.latentExplainer
                m_debug.dump()
        if(isinstance(getattr(mdc, 'latentEvalFig', None), vs3.plt.Figure)):   
            try:
                fig = mdc.latentEvalFig
                axindexes = []
                for i,(k,v) in enumerate(mdc.xheader_zones.items()):
                    cell_size = getattr(v, 'cell_size', None)
                    if(not DFP.isiterable(cell_size)):  continue
                    axindexes.append(i)
                n_axes = len(axindexes)
                r = int(np.floor(np.sqrt(n_axes)))+1 if(n_axes>1) else 1
                c = int(np.floor(n_axes/r)+1) if(n_axes>1) else 1
                for i,(k,v) in enumerate(mdc.xheader_zones.items()):
                    if(i not in axindexes):  continue
                    x_tensors = DFP.transformByBatch(X_data[v], (lambda x:x[0]))
                    r_tensors = DFP.transformByBatch(r_data[i], (lambda x:x[0]))
                    ax = fig.add_subplot(r,c,len(vs3.get_frames(fig))+1)
                    LOGger.addDebug(r,c,len(vs3.get_frames(fig))+1)
                    if(inlierMask is not None):
                        LOGger.addDebug('inlierMask', str(inlierMask))
                        outlierMask = np.logical_not(inlierMask)
                        drawLatentMapToPlaneViewAndCompareDecoded(
                        ax, x_tensors[inlierMask], p_data_2d[inlierMask], r_tensors[inlierMask], labels='inlier', colorDefault=(0,0,1,0.3), colorMask=None, 
                        x_range=x_range, y_range=y_range, n_display=min(2,p_data_2d[inlierMask].shape[0]), **kwags)
                        drawLatentMapToPlaneViewAndCompareDecoded(
                        ax, x_tensors[outlierMask], p_data_2d[outlierMask], r_tensors[outlierMask], labels='outlier', colorDefault=(0,0,0,0.3), colorMask=None, 
                        x_range=x_range, y_range=y_range, n_display=min(8,p_data_2d[outlierMask].shape[0]), **kwags)
                    else:
                        drawLatentMapToPlaneViewAndCompareDecoded(
                        ax, x_tensors, p_data_2d, r_tensors, labels=None, colorDefault=(0,0,1,0.3), colorMask=None, 
                        x_range=x_range, y_range=y_range, **kwags)
                    drawAERegionDyeing(mdc, ax,stamps=[*stamps,k], nTo2DimTransformer=pca)
                    # ax.set_xlim(x_range)
                    # ax.set_ylim(y_range)
                    ax.set_title(LOGger.stamp_process('',['latentSpace Comparison', *stamps, k],'','','','_'))
            except Exception as e:
                LOGger.exception_process(e, stamps=[evaluationAELatentSpace.__name__]+mdc.stamps+stamps, logfile='')
                m_debug['p_data'] = p_data
                m_debug['r_data'] = r_data
                m_debug.dump()
        # 可视化潜在空间
        # if(latentDim == 2 and isinstance(getattr(mdc, 'latentGridIllustrationAx', None), vs3.plt.Axes)):
        if(isinstance(getattr(mdc, 'latentGridIllustrationAx', None), vs3.plt.Axes)):
            valueMin = getattr(mdc, 'cellValueMin', None)
            valueMax = getattr(mdc, 'cellValueMax', None)
            x_range_distance = max(np.max(p_data_2d[:,0])-np.min(p_data_2d[:,0]), 1.0)
            y_range_distance = max(np.max(p_data_2d[:,1])-np.min(p_data_2d[:,1]), 1.0)
            x_range = getattr(mdc, 'latentXRange', (np.min(p_data_2d[:,0])-x_range_distance*0.1,np.max(p_data_2d[:,0])+x_range_distance*0.1))
            y_range = getattr(mdc, 'latentYRange', (np.min(p_data_2d[:,1])-y_range_distance*0.1,np.max(p_data_2d[:,1])+y_range_distance*0.1))
            grid_size = getattr(mdc, 'latentGridSize', (5,5))
            x_auto_subplot_size = (x_range[1] - x_range[0])/grid_size[0]
            y_auto_subplot_size = (y_range[1] - y_range[0])/grid_size[1]
            subplot_size = getattr(mdc, 'cellSubplotSize', (x_auto_subplot_size,y_auto_subplot_size))
            drawLatentMapToPlaneGrid(mdc, mdc.latentGridIllustrationAx, x_range=x_range, y_range=y_range, grid_size=grid_size, 
                                    subplot_size=subplot_size, stamps=stamps, valueMin=valueMin, valueMax=valueMax, nTo2DimTransformer=pca, **kwags)
            mdc.latentGridIllustrationAx.set_xlim(x_range)
            mdc.latentGridIllustrationAx.set_ylim(y_range)
            mdc.latentGridIllustrationAx.set_title(LOGger.stamp_process('',stamps+['drawLatentMapToPlaneGrid'],'','','','_'))
    except Exception as e:
        LOGger.exception_process(e, stamps=[evaluationAELatentSpace.__name__]+mdc.stamps+stamps, logfile='')
        return False
    return True

def recyleLatentIllustration(mdc, stamps=None, **kwags):
    axes = getattr(getattr(mdc,'latentFig',None),'axes',[])
    if(axes and np.array([len(ax.get_children())>0 for ax in axes]).all()):
        file=os.path.join(mdc.get_module_exp_fd(), '%s.jpg'%LOGger.stamp_process('',['latentCorrespondence', *stamps],'','','','_'))
        LOGger.CreateFile(file, lambda f:vs3.end(mdc.latentFig, file=f))
        for ax in mdc.latentFig.axes:
            if(ax.get_label().find('colorbar')>-1):  
                ax.remove()
                continue
            for x in ax.get_children():
                if(isinstance(x, vs3.plt.Axes)):  x.remove()
            ax.clear()
    if(getattr(getattr(mdc,'latentEvalFig',None),'axes',[])):
        file = os.path.join(mdc.get_graph_exp_fd(), '%s.jpg'%LOGger.stamp_process('',['latentSpaceComparison', *stamps],'','','','_'))
        LOGger.CreateFile(file, lambda f:vs3.end(mdc.latentEvalFig, file=f))
        for ax in mdc.latentEvalFig.axes:
            try:
                for sub_ax in ax.get_children():
                    if(isinstance(sub_ax, vs3.plt.Axes)):   sub_ax.remove()
            except Exception as e:
                LOGger.exception_process(e, stamps=[evaluationAutoEncoder.__name__]+mdc.stamps+stamps, logfile='')
            ax.clear()
            ax.remove()
    return True

def evaluationAutoEncoder(mdc, X_data, p_data=None, r_data=None, s_data=None, ret=None, handler=None, stamps=None, 
                          determine_threshold=False, qualifiedMask=None, figUnfamScores=None, axAddingLayout=(2,2), **kwags):
    stamps = stamps if(isinstance(stamps, list)) else []
    latentOutlierQualifiedMask = None
    if(p_data is None or r_data is None or s_data is None):
        p_data, r_data, s_data = mdc.predictRegenerateScoring(X_data)
        if(handler is not None):    
            handler.p_data = p_data
            handler.r_data = r_data
            handler.s_data = s_data
        ax = figUnfamScores.add_subplot(*axAddingLayout,len(vs3.get_frames(figUnfamScores))+1) if(
            isinstance(figUnfamScores, vs3.plt.Figure)) else None
        if(not unfamEvaluationLatentSpace(mdc, X_data, handler=handler, qualifiedMask=latentOutlierQualifiedMask, 
                                          stamps=stamps, determine_threshold=determine_threshold, ax=ax, **kwags)):
            return False
        if(isinstance(getattr(handler, 'eval_data', None), pd.core.frame.DataFrame)):    latentOutlierQualifiedMask = handler.eval_data['latentUnfamMask'].values
        # LOGger.addDebug('handler.eval_data', str(handler.eval_data))
        # LOGger.addDebug('latentOutlierQualifiedMask', str(latentOutlierQualifiedMask))
    if(not evaluationAERegeneration(mdc, X_data, r_data=r_data, s_data=s_data, ret=ret, handler=handler, stamps=stamps, **kwags)):
        return False
    if(not evaluationAELatentSpace(mdc, X_data, p_data=p_data, r_data=r_data, s_data=s_data, ret=ret, handler=handler, stamps=stamps, 
                                   inlierMask=latentOutlierQualifiedMask, **kwags)):
        return False
    recyleLatentIllustration(mdc, stamps=stamps, **kwags)
    return True

def activateLatentLayer(layer, input_shape=(1,), latentDim=(2,)):
    # 創建輸入張量
    inputs = keras.Input(shape=(1,))#, dtype=tf.float64)
    outputs = layer(inputs)
    outputs = parseKL('Reshape')(target_shape=latentDim)(outputs)
    return keras.Model(inputs=inputs, outputs=outputs)

def recordClassesLatentForHZ(v, model, ret=None, stamps=None, defaultClasses_=None, **kwags):
    preprocessor = v.preprocessor
    classes_ = getattr(preprocessor,'classes_',defaultClasses_)
    if(not DFP.isiterable(classes_)):
        return False
    classes_ = np.array(classes_)
    if(len(classes_.shape)==0):
        return False
    if(hasattr(preprocessor,'inverse_transform')):  classLabels = preprocessor.transform(classes_)
    layer = model.get_layer(v.latentLayerName)
    
    modelTemp = activateLatentLayer(layer, input_shape=layer.input_shape[1:])
    classCoords = modelTemp.predict(classLabels)
    classCoordMapping = dict(zip(*[classes_, classCoords.tolist()]))
    v.classCoordMapping = classCoordMapping
    if(isinstance(ret, dict)):  
        stamps = stamps if(isinstance(stamps, list)) else []
        stamp = LOGger.stamp_process('',stamps+['classCoordMapping'],'','','','_')
        ret[stamp] = dict(zip(*[classes_, classCoords.tolist()]))
    return True

def drawClassesLatentMap(ax, mapping, stamps=None, fontproperties=None, **kwags):
    stamps = stamps if(isinstance(stamps, list)) else []
    stamp = LOGger.stamp_process('',stamps+['classCoordMapping'],'','','','_')
    mappingValues = list(zip(*list(mapping.values())))
    LOGger.addDebug('drawClassesLatentMap stamp', stamp)
    ax.scatter(*mappingValues, label=stamp)
    for k,v in mapping.items():
        if(np.array(v).shape[0]!=2):
            continue
        ax.text(*v, '[%s]:(%s,%s)'%tuple(list(map(DFP.parse, (k,*v)))), fontproperties=fontproperties)

#fontproperties = 'Microsoft JhengHei'
def recordClassesLatentAndDrawForHZ(v, model, ax, ret=None, stamps=None, defaultClasses_=None, fontproperties=None, **kwags):
    if(not recordClassesLatentForHZ(v, model, ret=ret, stamps=stamps, defaultClasses_=defaultClasses_, **kwags)):
        return False
    drawClassesLatentMap(ax, v.classCoordMapping, stamps=stamps, fontproperties=fontproperties, **kwags)
    return True

def encodingKerasLattenMapping(mdc, fig=None, figsize=(40,40), ret=None, stamps=None, **kwags):
    lattenedZones = {k:v for k,v in mdc.xheader_zones.items() if 
                     v.data_prop in ['discrete', 'binary', 'categorical'] and 
                     LOGger.isinstance_not_empty(getattr(v,'latentLayerName',None),str)}
    if(not LOGger.isinstance_not_empty(lattenedZones, dict)):
        return True
    if(isinstance(fig, vs3.plt.Figure)):
        fig.clf()
    else:
        fig = vs3.plt.Figure(figsize=figsize)
    r = int(np.floor(np.sqrt(len(lattenedZones)))+1)
    c = int(np.floor(len(lattenedZones)/r)+1)
    # colIter = 0
    retTemp = {}
    mdc.fontproperties = 'Microsoft JhengHei'
    stamps = stamps if(isinstance(stamps, list)) else []
    for i,(k,v) in enumerate(lattenedZones.items()):
        retTemp.clear()
        if(v.data_prop in ['discrete', 'binary', 'categorical']):
            ax = fig.add_subplot(r,c,i+1)
            if(not recordClassesLatentAndDrawForHZ(v, model=mdc.model, ax=ax, ret=retTemp, stamps=stamps+[k], 
                                                   fontproperties=getattr(mdc,'fontproperties',None),**kwags)):
                return False
            if(isinstance(ret, dict)):  ret[LOGger.stamp_process('',['x',k],'','','','_')] = dcp(retTemp)
    if(len(vs3.get_frames(fig))>0):
        file = os.path.join(mdc.get_module_exp_fd(), 'latentSpaces.jpg')
        LOGger.CreateFile(file, lambda f:vs3.end(fig, file=f))
    return True

def encodingKerasLattenMapping_target(mdc, fig=None, figsize=(40,40), stamps=None, **kwags):
    addlog = kwags.get('addlog', LOGger.addloger(logfile=''))
    thread_key = kwags['thread_key'] if('thread_key' in kwags) else ''
    addlog('encodingKerasLattenMapping start')
    try:
        encodingKerasLattenMapping(mdc, fig=fig, figsize=figsize, stamps=stamps, **kwags)
        addlog('encodingKerasLattenMapping over')
    except Exception as e:
        LOGger.exception_process(e, stamps=[thread_key], logfile='')

#TODO:encodingKerasLattenMapping_target_threading
def encodingKerasLattenMapping_target_threading(mdc, fig=None, figsize=(40,40), stamps=None, **kwags):
    addlog = kwags.get('addlog', LOGger.addloger(logfile=''))
    kwags.update({'fig':fig, 'figsize':figsize, 'stamps':stamps})
    encodingKerasLattenMapping_target_thread = threading.Thread(target=encodingKerasLattenMapping_target,
                                                           args=(mdc,),
                                                           kwargs=kwags)
    encodingKerasLattenMapping_target_thread.start()

def dataMeasureVariance(mv, outcomes, predictions, stamps=None, **kwags):
    if(not mv.measure(predictions, outcomes, stamps=stamps, **kwags)):
        return False
    return True

def dataMeasureVarianceAndPlot(mv, outcomes, predictions, stamps=None, **kwags):
    if(not dataMeasureVariance(mv, outcomes, predictions, stamps=stamps, **kwags)):
        return False
    savePlotKwags = {k:v for k,v in kwags.items() if k in ['exp_fd']}
    mv.savePlot(**savePlotKwags)
    return True

def dataMeasureVariance4HZ(headerZone, outcomes, predictions, **kwags):
    mv = headerZone.mv
    kwags['stamps'] = kwags.get('stamps', [])+headerZone.stamps
    if(DFP.isiterable(getattr(headerZone.preprocessor,'classes_',None))):  kwags['all_categories'] = headerZone.preprocessor.classes_
    elif(DFP.isiterable(getattr(headerZone,'all_categories',None))): kwags['all_categories'] = headerZone.all_categories 
    # LOGger.addDebug('dataMeasureVariance4HZ', str(outcomes))
    if(not mv.measureDraw4HZ(headerZone, outcomes, predictions, **kwags)):
        return False
    # LOGger.addDebug(headerZone.data_prop)
    # LOGger.addDebug(headerZone.data_prop.eval_data)
    return True

def dataMeasureVarianceAndPlot4HZ(headerZone, outcomes, predictions, **kwags):
    """
    通常都還是從class去調整..

    Parameters
    ----------
    headerZone : TYPE
        DESCRIPTION.
    outcomes : TYPE
        DESCRIPTION.
    predictions : TYPE
        DESCRIPTION.
    **kwags : TYPE
        DESCRIPTION.

    Returns
    -------
    bool
        DESCRIPTION.

    """
    if(not dataMeasureVariance4HZ(headerZone, outcomes, predictions, **kwags)):
        return False
    savePlotKwags = {k:v for k,v in kwags.items() if k in ['exp_fd','stamps']}
    # LOGger.addDebug('savePlotKwags exp_fd', str(savePlotKwags.get('exp_fd','?')))
    mv = headerZone.mv
    mv.savePlot(**savePlotKwags)
    return True

def configuring_EIMS(config, handler=None, ret=None, **kwags):
    mdc = EIMS_core(config=config, **kwags)
    output = mdc
    if(isinstance(handler, LOGger.mystr)):
        handler.mdc = mdc
        output = True
    if(isinstance(ret, dict)):
        ret['mdc'] = mdc
        output = True
    return output
    
def plot_forest(trees, fig=None, figsize=(15,10), fn='plot_forest', exp_fd='.', file=None, filled=True, **kwags):
    n_trees = len(trees)
    r = int(np.sqrt(n_trees)+1)
    c = int(np.sqrt(n_trees)+1)
    if(not isinstance(fig, vs3.plt.Figure)):
        fig = vs3.plt.Figure(figsize=figsize)
    fig.clf()
    # feature_names = feature_names if(isinstance(feature_names, list)) else []
    # class_names = class_names if(isinstance(class_names, list)) else []
    try:
        for i,est in enumerate(trees):
            ax = fig.add_subplot(r,c,i+1)
            plot_tree(est, filled=filled, ax=ax)
            ax.set_title(f"Tree {i}")
        file = file if(LOGger.isinstance_not_empty(file, str)) else os.path.join(exp_fd, '%s.jpg'%fn)
        vs3.end(fig, file=file)
    except Exception as e:
        LOGger.exception_process(e,logfile='')
        # m_debug.update({'fig':fig, 'est':est, 'i':i})
        # m_debug.save()
        return False
    return True

#%%
def scoreLimitationBinaryClassification(x, **kwags):
    return np.clip(x,0,1)

class myLBC(LBC):
    def __init__(self, score_threshold=None, defaultClass='', defaultClassIndex=None, scoreLimitation=None, collaspeThreshold=None, **kwags):
        super().__init__()
        self.collaspeThreshold = collaspeThreshold
        self.score_threshold = score_threshold
        self.scoreLimitation = scoreLimitation if(callable(scoreLimitation)) else scoreLimitationBinaryClassification
        self.defaultClass = defaultClass if(isinstance(defaultClass, str)) else ''
        self.defaultClassIndex = defaultClassIndex
        # self.replaced_values = LOGger.mylist(m_noise_values if(not isinstance(replaced_values, list)) else replaced_values)
        # self.default_value = default_value
        # self.reveal_value = self.replaced_values[0] if(reveal_value is type(None)) else reveal_value
    
    def fit(self, data, classes_=None, uniqMethod=DFP.uniqueByIndex, **kwags):
        if(self.collaspeThreshold is not None):
            defaultClass = self.defaultClass
            defaultClassIndex = self.defaultClassIndex
            data = [*data, defaultClass]
            if(DFP.isiterable(classes_)):
                if(defaultClassIndex is None):
                    classes_ = np.array([*classes_, defaultClass])
                    print(classes_)
                else:
                    defaultClassIndex = defaultClassIndex%len(classes_)
                    classes_ = np.insert(classes_, defaultClassIndex, defaultClass)
        super().fit(data)
        if(DFP.isiterable(classes_)):
            np_classes_ = uniqMethod(classes_)
            print('myLBC fit', np_classes_.shape)
            print('myLBC fit', np_classes_[:10])
            self.classes_ = np.array(np_classes_)
            # super().classes_ = np.array(np_classes_)
        return self
    
    def transform(self, data, **kwags):
        try:
            data = super().transform(data)
        except:
            m_debug.updateDump({self.transform.__name__:{'data':data, '__function__':'myLBC::transform'}})
        return data.reshape(-1,1)
    
    def inverse_transform(self, data, **kwags):
        data = self.collaspe(data, **kwags)
        data = super().inverse_transform(data.reshape(-1))
        return data.reshape(-1,1)
    
    def metric(self, superposition, target, **kwags):
        if(DFP.astype(self.score_threshold, default=np.nan)>=0 and DFP.astype(self.score_threshold, default=np.nan)<=1):
            score_threshold = DFP.astype(self.score_threshold, np.nan)
            return np.abs(np.where(self.scoreLimitation(superposition)<score_threshold, 0, 1) - target)
        return np.abs(superposition - target)
    
    def collaspe(self, data, **kwags):
        return np.argmin(self.metric(data.reshape(-1,1), np.arange(len(self.classes_))), axis=1)
    
class myStdscr(Stdscr):
    def __init__(self, default_value=-1, **kwags):
        super().__init__()
        # self.replaced_values = LOGger.mylist(m_noise_values if(not isinstance(replaced_values, list)) else replaced_values)
        # self.reveal_value = self.replaced_values[0] if(reveal_value is type(None)) else reveal_value
        self.default_value = default_value
        
    def transform(self, data, **kwags):
        try:
            data = super().transform(data)
        except:
            m_debug.update({self.transform.__name__:{'data':data, '__function__':'myStdscr::transform'}})
            m_debug.save()
        return data
    
    def inverse_transform(self, data, **kwags):
        data = super().inverse_transform(data)
        return data

class myPwrscr(Pwrscr):
    """
        yeo-johnson for all; Box-Cox for positive data 
    """
    def __init__(self, default_value=-1, method='yeo-johnson', **kwags):
        super().__init__(method=method)
        # self.replaced_values = LOGger.mylist(m_noise_values if(not isinstance(replaced_values, list)) else replaced_values)
        # self.reveal_value = self.replaced_values[0] if(reveal_value is type(None)) else reveal_value
        self.default_value = default_value
        
    def transform(self, data, **kwags):
        try:
            data = super().transform(data)
        except:
            m_debug.updateDump({self.transform.__name__:{'data':data, '__function__':'myPwrscr::transform'}})
        return data
    
    def inverse_transform(self, data, **kwags):
        data = super().inverse_transform(data)
        return data
    
class myBasePwrscr:
    def __init__(self, method='yeo-johnson', default_value=-1, baseAxis=0, **kwags):
        self.baseScrAlg = myStdscr
        self.featureScrAlg = myPwrscr
        self.method = method
        self.default_value = default_value
        self.baseAxis = baseAxis
        self.baseScr = None
        self.featureScr = None
        
    def fit(self, data, **kwags):
        try:
            baseAxis = self.baseAxis
            np_data = getattr(data, 'values', data)
            np_base = np_data[:,np.array([baseAxis])]
            np_feat = np_data[:,np.array([x for x in np.arange(data.shape[1]) if x != baseAxis])]
            
            self.baseScr = self.baseScrAlg(default_value=self.default_value).fit(np_base)
            self.featureScr = self.featureScrAlg(method=self.method, default_value=self.default_value).fit(np_feat)
        except:
            m_debug.updateDump({self.transform.__name__:{'data':data, '__function__':'myBasePwrscr::transform'}})
        return self
        
    def transform(self, data, **kwags):
        try:
            baseAxis = self.baseAxis
            np_data = getattr(data, 'values', data)
            np_base = np_data[:,np.array([baseAxis])]
            np_feat = np_data[:,np.array([x for x in np.arange(data.shape[1]) if x != baseAxis])]
            
            np_base = self.baseScr.transform(np_base)
            np_feat = self.featureScr.transform(np_feat)
            
            data = np.hstack([np_base, np_feat])
        except:
            m_debug.updateDump({self.transform.__name__:{'data':data, '__function__':'myBasePwrscr::transform'}})
        return data
    
    def inverse_transform(self, data, **kwags):
        baseAxis = self.baseAxis
        np_data = getattr(data, 'values', data)
        np_base = np_data[:,np.array([baseAxis])]
        np_feat = np_data[:,np.array([x for x in np.arange(data.shape[1]) if x != baseAxis])]
        
        np_base = self.baseScr.inverse_transform(np_base)
        np_feat = self.featureScr.inverse_transform(np_feat)
        
        data = np.hstack([np_base, np_feat])
        return data
        

class LossCurveCallback(keras.callbacks.Callback): #tf.keras.callbacks.Callback
    def __init__(self, ax, *losscurves, plot_interval=10, save_path=None, stamps=None, color=None, fig=None, losscurveNames=None):
        """
        

        Parameters
        ----------
        ax : TYPE
            DESCRIPTION.
        losscurves : dict
            DESCRIPTION. The default is {'loss':[], 'val_loss':[]}
        plot_interval : TYPE, optional
            DESCRIPTION. The default is 10.
        save_path : TYPE, optional
            DESCRIPTION. The default is None.
        stamps : TYPE, optional
            DESCRIPTION. The default is None.
        color : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        super().__init__()
        self.stamps = stamps if(isinstance(stamps, list)) else []
        self.fig = fig
        self.ax = ax
        self.plot_interval = plot_interval
        self.save_path = save_path
        self.epochs = []
        losscurves = LOGger.mylist(losscurves)
        if(len(losscurves)==1):
            if(isinstance(losscurves[0], dict)):
                self.losscurves = losscurves
        losscurveNames = losscurveNames if(isinstance(losscurveNames, list)) else ['loss','val_loss']
        self.losscurves = dict(zip(losscurveNames[:max(len(losscurveNames),len(losscurves))], losscurves[:max(len(losscurveNames),len(losscurves))]))
        self.color = color

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs.append(epoch)
        for k,v in self.losscurves.items():
            v.append(logs.get(k))

        if (epoch + 1) % self.plot_interval == 0:
            # 繪製損失曲線
            drawLossCurve(self.ax, *list(self.losscurves.values()), stamps=self.stamps, colors=self.color, rewrite=True)

            # 如果提供了保存路徑，保存圖片
            if self.fig is not None and LOGger.isinstance_not_empty(self.save_path, str):
                os.makedirs(self.save_path, exist_ok=True)
                self.fig.savefig(os.path.join(self.save_path, f'loss_curve_epoch_{epoch + 1}.png'))

class PeriodicLogCallback(keras.callbacks.Callback): #tf.keras.callbacks.Callback
    def __init__(self, log_interval=1800, log_file=None, stamps=None, loss_names=None):
        super(PeriodicLogCallback, self).__init__()
        self.log_interval = log_interval  # 默认1800秒（30分钟）
        self.log_file = log_file
        self.stamps = stamps if(isinstance(stamps, list)) else []
        self.loss_names = loss_names if(isinstance(loss_names, list)) else ['loss', 'val_loss']
        self.epochs = []
        self.losses = {name: [] for name in self.loss_names}
        self.last_log_time = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs.append(epoch)
        
        # 记录所有损失值
        for name in self.loss_names:
            self.losses[name].append(logs.get(name))
        
        # 检查是否达到记录时间间隔
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval if(self.last_log_time is not None) else True:
            
            # 打印当前epoch的损失值
            loss_str = ', '.join([f'{name}: {logs.get(name, ""):.4f}' for name in self.loss_names])
            LOGger.addlog(f'核心訓練進行中.... Epoch {epoch + 1}: {loss_str}', stamps=self.stamps, logfile=self.log_file)
            
            # 更新最后记录时间
            self.last_log_time = current_time

class abc_EIMS_CORE(abc.ABC):
    def __init__(self, stamps=None, exp_fd='test', config_file=None, version='', name='', theme='theme', **kwags):
        self.name = name
        self.theme = theme
        self.stamps = stamps if(isinstance(stamps, list)) else []
        self.exp_fd = exp_fd if(LOGger.isinstance_not_empty(exp_fd, str)) else '.'
        self.logfile = os.path.join(self.exp_fd, 'log.txt')
        self.config_file = config_file
        self.config_dir = os.path.dirname(self.config_file) if(LOGger.isinstance_not_empty(self.config_file, str)) else '.'
        self.config = None
        self.mainAttrs = { 'stamps':[] }  #在set_config跟save_config時一定要用的屬性
        self.fitAttrs = {}
        self.algorithmAttrs = {}
        self.model_stamps = ['model']
        self.debugFile = os.path.join(self.exp_fd, 'export.pkl')
        self.header_zones_names = ['xheader_zones', 'yheader_zones', 'unfam_header_zones']
        self.IOheader_zones_names = ['xheader_zones', 'yheader_zones']
        self.header_zones_types = {'xheader':'HEADER_ZONE_INPUT', 'yheader': 'HEADER_ZONE_OUTPUT', 'unfam_header':'HEADER_ZONE_OUTPUT'}
        
        self.xheader_zones = LOGger.mydict({})
        self.yheader_zones = LOGger.mydict({})
        self.xheader = LOGger.mylist([])
        self.yheader = LOGger.mylist([])
        self.unfam_header_zones = LOGger.mydict({})
        self.unfam_header = LOGger.mylist([])
        self.auxheader = LOGger.mylist([])
        self.model = None
        self.version = version
    
    @abc.abstractmethod
    def predict_core(self, **kwags):
        pass
    
    @abc.abstractmethod
    def compiling(self, **kwags):
        pass
    
    @abc.abstractmethod
    def fit(self, validation_data=None, **kwags):
        pass
    
    @abc.abstractmethod
    def predict(self, **kwags):
        pass
    
    def __str__(self):
        return LOGger.stamp_process('',self.get_stamps(),'','','','_')
    
    def set_header(self, name, key, adding_zone):
        """
        adding activated zones to header
        
        Parameters
        ----------
        name : str
            EX:xheader.
        key : str
            EX:main.
        zone : mylist
            EX:[...].

        Returns
        -------
        None.


        """
        header_zones_concat = getattr(self, name) #mylist
        header_zones = getattr(self, '%s_zones'%name) #mydict
        header_zones[key] = adding_zone
        header_zones_concat = header_zones_concat + adding_zone
        setattr(self, name, header_zones_concat)
        return True
    
    def addlog(self, *log, **kwags):
        kwags.update({'logfile':kwags.get('logfile', self.logfile)})
        LOGger.addlog(*log, **kwags)
        
    def set_fittingAttrs(self):
        for attr in self.fittingAttrs:
            setattr(self, attr, self.fittingConfig[attr])
        
    def set_model(self, compile=False):
        try:
            confg_dir = os.path.dirname(self.config_file)
            for model_stamp in self.model_stamps:
                setattr(self, model_stamp, load_model(
                    os.path.join(confg_dir, getattr(self, '%s_file'%model_stamp)), compile=compile))
        except Exception as e:
            LOGger.exception_process(e,logfile=os.path.join(self.exp_fd, 'log.txt'))
            return False
        return True
    
    def set(self, compile=False):
        if(not self.set_config()):
            return False
        if(not self.set_model(compile=compile)):
            return False
        return True
    
    #TODO:abc_MODELHOST get_stamps
    def get_stamps(self, full=True, for_file=False):
        stamps = (([self.name] if(self.name!='') else []) + self.stamps) if(full) else self.stamps
        stamps = [v for v in stamps if v!='']
        stamps = list(map(lambda s:str(s).replace('<','〈').replace('>','〉'), stamps)) if(for_file) else stamps
        return stamps
    
    def get_module_exp_fd(self):
        version = getattr(self, 'out_version', getattr(self, 'version', '.'))
        return os.path.join(self.exp_fd, 'warehouse', stamp_process('', self.get_stamps(full=True),'','','','_'), version)
    
    def get_exp_fd(self, *layers):
        return os.path.join(self.exp_fd, *layers)
    
    def get_detail_exp_fd(self, *layers):
        return os.path.join(self.exp_fd, 'excel')
    
    def get_graph_exp_fd(self, *layers):
        return os.path.join(self.exp_fd, 'graph')
    
    #TODO:abc_MODELHOST get_source
    def get_source(self):
        if(LOGger.isinstance_not_empty(self.config_file, str)):
            return os.path.dirname(self.config_file)
        elif(getattr(self,'keras_method','')!=''):
             return self.keras_method
        return ''
    
    def create_detail_method(self, *args, **kwags):
        return create_model_detail_keras_txt(*args, **kwags)
    
    #TODO:abc_MODELHOST get_hyperlink
    def get_hyperlink(self, method_detail_file='', **kwags):
        stg = ''
        if(LOGger.isinstance_not_empty(self.config_file,str)):
            if(os.path.exists(self.config_file)):
                title = getattr(self, 'version', '-')
                destination = os.path.dirname(self.config_file)
                stg = LOGger.make_hyperlink(destination, title=title)
        return stg
    
    def configuring(self, stamps=None, **kwags):
        stamps = stamps if(isinstance(stamps, list)) else []
        stamp = LOGger.stamp_process('',stamps,'','','','_')
        Attrs = dcp(getattr(self,'%sAttrs'%stamp))
        Attrs.update(kwags)
        if(not mdcScenarioConfiguring(self, Attrs, stamps=[stamp])):
            return False
        return True
    
    def plot_model(self, **kwags):
        if(self.model is None):
            return True
        return False
    
    def describeModelInOut(self, **kwags):
        self.addlog(self.getStringModelInOut(), **kwags)
    
    def getStringModelInOut(self, **kwags):
        return ''
    
    def predict_score(self, data, **kwags):
        return predict_score(self, data, **kwags)
    
    def predict_confidence(self, data, **kwags):
        return predict_confidence(self, data, **kwags)
    
    def predict_confidenceScore(self, data, **kwags):
        return predict_confidenceScore(self, data, **kwags)
    
    def predictWithScores(self, X_data, **kwags):
        return predictWithScoresStandard(self, X_data, **kwags)
    
    def update_config(self, config, **kwags):
        if(not update_config(self, config)):
            return False
        return True
    
    def set_config(self, **kwags):
        """
        會以<str>self.config_file更新self.config_dir
        """
        kwags['config_file'] = kwags.get('config_file', self.config_file)
        if(not set_config(self, **kwags)):
            return False
        return True
    
    def set_modules(self, **kwags):
        if(not set_mdcModules(self, config_file=self.config_file, **kwags)):
            return False
        return True
    
    def save_config(self, **kwags):
        if(not save_config(self, **kwags)):
            return False
        return True
    
    def save_models(self, exp_fd=None, **kwags):
        if(not save_models(self, **kwags)):
            return False
        return True
    
    def save_modules(self, exp_fd=None, **kwags):
        if(not save_mdcModules(self, exp_fd=exp_fd)):
            return False
        return True
    
    def convert4modelCore(self, array, **kwags):
        return array

    def clear(self):
        model_stamps = self.model_stamps
        for k in model_stamps:
            setattr(self, k, None)
        for k in self.header_zones_names:
            setattr(self, k, LOGger.mydict({}))
            k__ = k.replace('_zones','')
            setattr(self, k__, LOGger.mylist([]))
        return True

class EIMS_core(abc_EIMS_CORE):
    def __init__(self, stamps=None, exp_fd='.', config_file=None, version='version', figsize=(8,7), theme='theme', **kwags):
        super().__init__(stamps=stamps, exp_fd=exp_fd, config_file=config_file, version=version, theme=theme, **kwags)
        self.mainAttrs.update({
            'auxheader': [], 
            'version':'-', 
            'modelFileType':'h5', 
            'threshold_unfam':None, 
            'score_threshold':None,
            'activation':'relu', 
            'lossmethod':'mse', 
            'model_file':None,
            'default_value':0,
            'default_unfam':None})
        self.model = None
        self.modelExpfileType = 'h5'
        self.lossCurveFig = vs3.plt.Figure(figsize=figsize)
        self.lossCurveAx = self.lossCurveFig.add_subplot(1,1,1)
        self.compileMethod = ''
        self.fitAttrs = {'reinitial_thd':200000, 
                         'storage_thd':100000, 
                         'overfit_maxthd':0.3, 
                         'epochs':100, 
                         'batch_size':100,
                         'auto_train_max_count':2, 
                         'break_time':2, 
                         'save_file_count':5, 
                         'over_loss_max_ratio':0.3,
                         'auto_train_max_count': 2,
                         'contamination': 0.1,
                         'initial_model_method': None,
                         'custom_objects': {},
                         'classEmbeddingLatentDim': 2}
        self.compileAttrs = {'hidden_layer_sizes': (1, ),
                             'hidden_layer_nns': None, 
                             'activation': 'relu',
                             'lossmethod': 'mse',
                             'optimizer': 'Adam',
                             'optimizer_args': {},
                             'compileMethod': ''}
    
    def getStringModelInOut(self, **kwags):
        if(self.model is None):
            return ''
        return '[input shape:%s][output shape:%s]'%(str(self.model.input_shape), str(self.model.output_shape))
    
    def plot_model(self, show_shapes=True, show_layer_activations=True, **kwags):
        if(self.model is None):
            return True
        file = os.path.join(self.get_module_exp_fd(), '%s.png'%LOGger.stamp_process('',self.get_stamps(full=True)+['nn_structure'],'','','','_'))
        plot_model(self.model, file, show_shapes=show_shapes, show_layer_activations=show_layer_activations) if(
            parse_version(keras.__version__)>parse_version('2.8')) else plot_model(
            self.model, file, show_shapes=show_shapes, show_layer_names=show_layer_activations)
        return True
    
    def predict_core(self, inputs, **kwags):
        return self.model.predict(inputs)
    
    def convert4modelCore(self, array, **kwags):
        return bcd.cast_to_floatx(array)
    
    def compiling(self, **kwags):
        return compilingKeras(self, **kwags)
    
    def fit(self, x_data, y_data, validation_data=None, **kwags):
        fitAttrs = {k:kwags.get(k, v) for (k,v) in self.fitAttrs.items()}
        if(not fitKeras(self, x_data, y_data, validation_data=validation_data, **fitAttrs)):
            return False
        return True
    
    def export_model_detail(self, exp_fd='', rewrite=True, deep=True, **kwags):
        export_model_detail_keras(self, exp_fd=exp_fd, rewrite=rewrite, deep=deep, **kwags)
    
    def predict(self, data, **kwags):
        return predict(self, data, **kwags)
    
    def transform(self, data, **kwags):
        return transformKeras(self, data, **kwags)
    
    def inverse_transform(self, data, **kwags):
        return inverse_transformKeras(self, data, **kwags)
    
    def evaluation(self, X_data, y_data, p_data=None, **kwags):
        return evaluationStandard(self, X_data, y_data, p_data=p_data, **kwags)
    
    
class EIMS_AUTOENCODER_core(EIMS_core):
    def __init__(self, stamps=None, exp_fd='.', config_file=None, version='version', figsize=(10,10), theme='theme', 
                 latentFigsize=(20,20), latentEvalFigsize=(30,20), **kwags):
        super().__init__(stamps=stamps, exp_fd=exp_fd, config_file=config_file, version=version, theme=theme, 
                         figsize=figsize, **kwags)
        self.encoder = None
        self.decoder = None
        self.model_stamps = ['model', 'encoder', 'decoder']
        self.debugFile = os.path.join(self.exp_fd, 'export.pkl')
        # self.header_zones_names = ['xheader_zones', 'yheader_zones', 'unfam_header_zones']
        self.IOheader_zones_names = ['xheader_zones']
        self.header_zones_types = {'xheader':'HEADER_ZONE_AUTOENCODER', 'yheader':'HEADER_ZONE_OUTPUT', 'unfam_header':'HEADER_ZONE_OUTPUT'}
        self.encoder_hidden_layer_sizes = (10,)
        self.decoder_hidden_layer_sizes = (10,)
        self.encoder_hidden_layer_nns = {}
        self.decoder_hidden_layer_nns = {}
        self.latentCoresFile = None
        self.latentCores = None
        self.latentExplainerFile = None
        self.latentExplainer = None
        self.latentPcaFile = None
        self.latentPca = None
        self.threshold_unfam_latentSpace = None
        self.compileAttrs = {'compileMethod': self.compileMethod,
                             'encoder_hidden_layer_sizes': self.encoder_hidden_layer_sizes,
                             'decoder_hidden_layer_sizes': self.decoder_hidden_layer_sizes,
                             'encoder_hidden_layer_nns': self.encoder_hidden_layer_nns,
                             'decoder_hidden_layer_nns': self.decoder_hidden_layer_nns,
                             'activation': 'relu',
                             'latentActivation': 'tanh',
                             'lossmethod': 'mse',
                             'optimizer': 'Adam',
                             'optimizer_args': {}}
        self.mainAttrs.update({
            'threshold_unfam_latentSpace': self.threshold_unfam_latentSpace,
            'encoder_hidden_layer_sizes': self.encoder_hidden_layer_sizes,
            'decoder_hidden_layer_sizes': self.decoder_hidden_layer_sizes,
            'encoder_hidden_layer_nns': self.encoder_hidden_layer_nns,
            'decoder_hidden_layer_nns': self.decoder_hidden_layer_nns,
            'latentCoresFile': self.latentCoresFile,
            'latentExplainerFile': self.latentExplainerFile,
            'latentPcaFile': self.latentPcaFile
        })
        self.latentFig = vs3.plt.Figure(figsize=latentFigsize) # (figsize=tuple(map(lambda x:x*2, figsize)))
        self.latentAx = self.latentFig.add_subplot(2,1,1)
        self.latentGridIllustrationAx = self.latentFig.add_subplot(2,1,2)
        self.latentEvalFig = vs3.plt.Figure(figsize=latentEvalFigsize)
    
    def getStringModelInOut(self, **kwags):
        if(self.model is None):
            return ''
        return '[input shape:%s][output shape:%s]'%(str(self.encoder.input_shape), str(self.encoder.output_shape))
    
    def plot_model(self, show_shapes=True, show_layer_activations=True, **kwags):
        if(self.model is None):
            return True
        file = os.path.join(self.get_module_exp_fd(), '%s.jpg'%LOGger.stamp_process('',self.get_stamps(full=True)+['nn_structure'],'','','','_'))
        plot_model(self.model, file, show_shapes=show_shapes, show_layer_activations=show_layer_activations) if(
            parse_version(keras.__version__)>parse_version('2.8')) else plot_model(
            self.model, file, show_shapes=show_shapes, show_layer_names=show_layer_activations)
        file = os.path.join(self.get_module_exp_fd(), '%s.jpg'%LOGger.stamp_process('',self.get_stamps(full=True)+['nn_structure','encoder'],'','','','_'))
        plot_model(self.encoder, file, show_shapes=show_shapes, show_layer_activations=show_layer_activations) if(
            parse_version(keras.__version__)>parse_version('2.8')) else plot_model(
            self.encoder, file, show_shapes=show_shapes, show_layer_names=show_layer_activations)
        file = os.path.join(self.get_module_exp_fd(), '%s.jpg'%LOGger.stamp_process('',self.get_stamps(full=True)+['nn_structure','decoder'],'','','','_'))
        plot_model(self.decoder, file, show_shapes=show_shapes, show_layer_activations=show_layer_activations) if(
            parse_version(keras.__version__)>parse_version('2.8')) else plot_model(
            self.decoder, file, show_shapes=show_shapes, show_layer_names=show_layer_activations)
        return True
    
    def predict_core(self, inputs, **kwags):
        ret= self.encoder.predict(inputs)
        # LOGger.addDebug('predict_core', ret)
        return ret

    def predictWithScores(self, X_data, **kwags):
        return predictWithScoresAE(self, X_data, **kwags)
    
    def regenerateWithScores(self, X_data, **kwags):
        return regenerateWithScores(self, X_data, **kwags)
    
    def predictRegenerateScoring(self, X_data, **kwags):
        return predictRegenerateScoring(self, X_data, **kwags)
    
    def convert4modelCore(self, array, **kwags):
        return bcd.cast_to_floatx(array)
    
    def compiling(self, **kwags):
        if(not compilingKerasAutoEncoder(self, **kwags)):
            return False
        return True
    
    def fit(self, x_data, y_data=None, validation_data=None, **kwags):
        if(y_data is None): y_data = dcp(x_data)
        fitAttrs = {k:kwags.get(k, v) for (k,v) in self.fitAttrs.items()}
        if(not fitKerasAutoEncoder(self, x_data, y_data, validation_data=validation_data, **fitAttrs)):
            return False
        return True
    
    def export_model_detail(self, exp_fd='', rewrite=True, deep=True, **kwags):
        export_model_detail_keras(self, exp_fd=exp_fd, rewrite=rewrite, deep=deep, **kwags)
    
    # def predict(self, data, **kwags):
    #     return predict(self, data, **kwags)
    
    # def transform(self, data, **kwags):
    #     return transformKeras(self, data, **kwags)
    
    def inverse_transform(self, data, header_zones_name='yheader_zones', **kwags): # header_zones_name='yheader_zones' 這樣global predict()就不用改
        return inverse_transformKerasAutoEncoder(self, data, header_zones_name=header_zones_name, **kwags)
    
    def inverse_transform_regenerate(self, data, **kwags):
        return self.inverse_transform(data, header_zones_name='xheader_zones', **kwags)
    
    def inverse_transform_reduce(self, data, **kwags):
        return self.inverse_transform(data, header_zones_name='yheader_zones', **kwags)
    
    def regenerate_core(self, inputs, **kwags):
        return self.model.predict(inputs)
        
    def regenerate(self, data, **kwags):
        return regenerate(self, data, **kwags)
    
    def latentScore_core(self, inputs, **kwags):
        return -self.latentExplainer.score_samples(self.encoder.predict(inputs)[0]).reshape(-1,1)

    def collaspe_core(self, inputs, **kwags):
        return self.latentScore_core(inputs) >= self.threshold_unfam_latentSpace

    def evaluation(self, X_data, y_data=None, p_data=None, r_data=None, **kwags):
        # y_data is always None
        return evaluationAutoEncoder(self, X_data, p_data=p_data, r_data=r_data, **kwags)
    
    def set_config(self, **kwags):
        if(not set_config_autoEncoder(self, **kwags)):
            return False
        return True

    def save_models(self, exp_fd=None, **kwags):
        if(not save_models_autoEncoder(self, exp_fd=exp_fd, **kwags)):
            return False
        return True
    
class EIMS_SKLEARN_core(EIMS_core):
    def __init__(self, stamps=None, exp_fd='test', config_file=None, version='version', figsize=(8,7), theme='theme', 
                 algorithmSerial='svm.SVM', **kwags):
        super().__init__(stamps=stamps, exp_fd=exp_fd, config_file=config_file, version=version, theme=theme, **kwags)
        self.header_zones_names = ['xheader_zones', 'yheader_zones']
        self.header_zones_types = {'xheader':'HEADER_ZONE_INPUT', 'yheader': 'HEADER_ZONE_OUTPUT'}
        self.mainAttrs.update({
            'version':'-', 
            'modelFileType':'pkl', 
            'threshold_unfam':None, 
            'score_threshold':None,
            'model_file':None,
            'default_value':0,
            'default_unfam':None})
        self.modelExpfileType = 'pkl'
        self.fitAttrs = {'cv':10,
                         'probability':False}
        self.compileAttrs = {'algorithmSerial':algorithmSerial,
                             'algorithm_param_grid':{},
                             'algorithm_params':{},
                             'scoring': None}
        self.algorithm = None
        self.algorithmSerial = algorithmSerial
    
    def getStringModelInOut(self, **kwags):
        if(self.model is None):
            return ''
        return '[input shape:%s][output shape:1]'%str((None, self.model.n_features_in_))
    
    def plot_model(self, show_shapes=True, show_layer_activations=True, **kwags):
        return True
    
    def predict_core(self, inputs, return_probability=False, **kwags):
        method = self.model.predict if(not return_probability or not hasattr(self.model, 'predict_proba')
                                       or isinstance(self.model, AutoSklearnRegressor)) else self.model.predict_proba
        prediction = method(inputs)
        if(len(prediction.shape)==1): prediction = prediction.reshape(-1,1) #從sklearn轉換過來的的shape可能是(-1,)，為了跟keras的shape一致
        return prediction
    
    def compiling(self, **kwags):
        return compilingSklearn(self, **kwags)
    
    def fit(self, x_data, y_data, validation_data=None, **kwags):
        fitAttrs = {k:kwags.get(k, v) for (k,v) in self.fitAttrs.items()}
        if(not fitSklearn(self, x_data, y_data, validation_data=validation_data, **fitAttrs)):
            return False
        return True
    
    #TODO:MODELHOST_SKLEARN export_model_detail
    def export_model_detail(self, exp_fd='', rewrite=True, deep=True, **kwags):
        model_stamps = getattr(self, 'model_stamps', kwags.get('model_stamps', []))
        if(model_stamps):
            for model_object in model_stamps:
                export_model_detail_json(self, exp_fd=exp_fd, rewrite=rewrite, deep=deep, model_object=model_object, **kwags)
        else:
            export_model_detail_json(self, exp_fd=exp_fd, rewrite=rewrite, deep=deep, **kwags)
        
    #TODO:MODELHOST_SKLEARN export_models_leaderboard
    def export_models_leaderboard(self, exp_fd='', rewrite=True, detailed=True, deep=True, theme='', model_stamps=None, **kwags):
        model_stamps = model_stamps if(isinstance(model_stamps, list)) else [] #是否改成self.model_stamps
        for model_stamp in model_stamps:
            export_models_leaderboard(self, exp_fd=exp_fd, rewrite=rewrite, detailed=detailed, 
                                      model_object=model_stamp, deep=deep, theme=theme, **kwags)
    
    def transform(self, data, **kwags):
        return transformSklearn(self, data, **kwags)
    
    def inverse_transform(self, data, **kwags):
        return inverse_transformSklearn(self, data, **kwags)
    
    # def predict(self, data, **kwags):
    #     return predict(self, data, **kwags)
    
    def predict_score(self, data, multi_dim_method=None, **kwags):
        kwags['return_probability'] = True
        binary_inlier_code = getattr(self, 'binary_inlier_code', 1)
        LOGger.addDebug('binary_inlier_code', binary_inlier_code)
        multi_dim_method=(lambda d:d[:,binary_inlier_code]) if(not callable(multi_dim_method)) else multi_dim_method
        return predict_score(self, data, multi_dim_method=multi_dim_method, **kwags)
    
    def predict_confidence(self, data, **kwags):
        return self.predict_score(data, **kwags)

    def predictWithScores(self, X_data, **kwags):
        return predictWithScoresSklearn(self, X_data, **kwags)

# algorithmSerial == missionSystemSerial嗎?
class EIMS_AUTOSKLEARN_core(EIMS_SKLEARN_core):
    def __init__(self, stamps=None, exp_fd='test', config_file=None, version='version', theme='theme', 
                 algorithmSerial='', missionSystemSerial='AutoSklearnRegressor', **kwags):
        super().__init__(stamps=stamps, exp_fd=exp_fd, config_file=config_file, version=version, theme=theme, **kwags)
        self.mainAttrs.update({
            'version':'-', 
            'modelFileType':'pkl', 
            'threshold_unfam':None, 
            'score_threshold':None,
            'model_file':None,
            'default_value':0,
            'default_unfam':None})
        self.modelExpfileType = 'pkl'
        self.fitAttrs = {'cv':10,
                         'probability':False}
        self.compileAttrs = {'missionSystemSerial': missionSystemSerial, 
                             'missionSystem_param_grid':{},
                             'missionSystem_params':{},
                             'scoring': None}
        self.algorithm = None
        self.algorithmSerial = algorithmSerial
        self.algorithm_params = {}
        self.leaderboard = pd.DataFrame()
        self.missionSystem = None
        self.missionSystemSerial = missionSystemSerial
        
    def compiling(self, **kwags):
        return compilingAutoSklearn(self, **kwags)
    
    def fit(self, x_data, y_data, validation_data=None, **kwags):
        fitAttrs = {k:kwags.get(k, v) for (k,v) in self.fitAttrs.items()}
        if(not fitAutoSklearn(self, x_data, y_data, validation_data=validation_data, **fitAttrs)):
            return False
        return True
    
    def getStringModelInOut(self, **kwags):
        if(self.model is None):
            return ''