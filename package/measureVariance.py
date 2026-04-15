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
from package import visualization3 as vs3
vs = vs3.vs
vs2 = vs3.vs2
DFP = vs3.DFP
import matplotlib.gridspec as gridspec
LOGger = vs3.LOGger
from package.LOGger import CreateContainer, CreateFile, addloger, show_vector
from package.LOGger import stamp_process, exception_process, for_file_process, abspath, mylist, mystr
from package.LOGger import load_json, save_json, flattern_list, type_string
import pandas as pd
import joblib
from copy import deepcopy as dcp
from sklearn.metrics import confusion_matrix
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder as LBC
from sklearn.preprocessing import OneHotEncoder as OHC
import numpy as np
from datetime import datetime as dt
from scipy.stats import pointbiserialr
from sklearn.preprocessing import StandardScaler as Stdscr
from sklearn.preprocessing import MinMaxScaler as Mnxscr
from sklearn import metrics as skm
import pickle
from sklearn.metrics import cohen_kappa_score as cohkpa
import focal_loss
from focal_loss import BinaryFocalLoss
padCropPadEdge = DFP.padCropPadEdge

threshold_variation_analysis_lock = threading.Lock()
CreateContainer('log')
logfile = 'log\\log_%s.txt'%(dt.now().strftime('%Y%m%d'))
logshield = False
addlog = LOGger.addloger(logfile='')
#%%
if(False):
    __file__ = 'measureVariance.py'
json_file = '%s_buffer.json'%os.path.basename(__file__.replace('.py',''))
m_pre_model = os.path.join('pre_model')
#%%
m_debug = LOGger.myDebuger(stamps=[*os.path.basename(__file__).split('.')[:-1]])
m_dictionary = {}
m_dataColor = (5/255,80/255,220/255,0.5)
m_bundaryColor = (255/255,90/255,90/255)
m_lossColors = LOGger.mylist([m_dataColor, (0.2,0.1,0.8,0.5), (0.05,0.05,0.3)])
m_classificationScoreNames = ['rcl','pcs',r'$f_{1}$','fact_counts','pred_counts','fact_ratio']

class HLBC_fit():
    def __init__(self, Y, encoder_system=LBC(), addlog=addlog, default_columns=None):
        np_Y = np.array(Y)
        default_columns = default_columns if(isinstance(default_columns, list)) else range(np_Y.shape[1])
        lbcs, diversities, classes, lbc = DFP.mydict(), DFP.mydict(), DFP.mydict(), None
        if(len(np_Y.shape)>1):
            self.header = getattr(Y, 'columns', default_columns)
            self.n_features_in_ = np_Y.shape[1]
            for i, hd in enumerate(self.header):
                lbc = encoder_system.fit(np_Y[:,i])
                lbcs[hd] = dcp(lbc)
                diversities[hd] = DFP.uniqueByIndex(np_Y[:,i]).shape[0]
                classes[hd] = list(tuple(DFP.uniqueByIndex(np_Y[:,i])))
                addlog('[%s]類別只有1個以下:%d!!!'%(hd, diversities[hd]) if(diversities[hd]<=1) else '')
            self.lbc = lbc
        else:
            self.n_features_in_ = 1
            hd = getattr(Y, 'name', 0)
            self.header = mylist([hd])
            self.lbc = encoder_system.fit(Y)
            diversities[hd] = DFP.uniqueByIndex(Y).shape[0]
            classes[hd] = list(tuple(DFP.uniqueByIndex(Y)))
            lbcs = {hd:self.lbc}
            addlog('[%s]類別只有1個以下:%d!!!'%(hd, diversities[hd]) if(diversities[hd]<=1) else '')
        self.lbcs = lbcs
        self.encoder_system = encoder_system
        self.diversities = diversities
        self.class_sizes = tuple(diversities.values())
        self.classes = classes
        self.classes_ = classes[tuple(classes.keys())[0]]
    
    #為了要送進去model.fit .predict
    def transform(self, Y, return_pandas=False):
        np_Y = np.array(Y).reshape(-1, 1) if(len(np.array(Y).shape)==1) else np.array(Y)
        return_pandas &= str(type(Y)).find('pandas')>-1
        if(self.n_features_in_ != (np.array(Y).shape[1] if(len(np.array(Y).shape)>1) else 1)):
            return {}['Y shape%s; while HLBC shape%s'%(Y.shape, self.n_features_in_)]
        Y_lbced = []
        if(self.n_features_in_>1):
            for i, hd in enumerate(self.header):
                y_lbced = self.lbcs[hd].transform(np_Y[:,i]).reshape(-1,1)
                if(return_pandas):
                    columns = getattr(Y, 'columns', [hd])
                    index = getattr(Y, 'index', range(np_Y.shape[0]))
                    y_lbced = pd.DataFrame(y_lbced, columns=columns, index=index)
                Y_lbced.append(y_lbced)
        else:
            Y_lbced = self.lbc.transform(np_Y.reshape(-1))
        return Y_lbced
    
    #為了方便整理
    def transform_flatten(self, Y):
        np_Y = np.array(Y).reshape(-1, 1) if(len(np.array(Y).shape)==1) else np.array(Y)
        columns = getattr(Y, 'columns', (range(np_Y.shape[1]) if(len(np_Y.shape)>1) else [getattr(Y, 'name', 0)]))
        index = getattr(Y, 'index', range(np_Y.shape[0]))
        ret = np.empty(shape=Y.shape)
        if(self.n_features_in_ != (np.array(Y).shape[1] if(len(np.array(Y).shape)>1) else 1)):
            return {}['Y shape%s; while HLBC shape%s'%(Y.shape, self.n_features_in_)]
        if(self.n_features_in_>1):
            for i, xh in enumerate(self.header):
                ret[:,i] = self.lbcs[xh].transform(np_Y[:,i])
        else:
            ret = self.lbc.transform(Y)
        if(str(type(Y)).find('pandas')>-1):
            ret = pd.DataFrame(ret, columns=columns, index=index) if(self.n_features_in_>1) else pd.Series(ret, name=columns[0], index=index)
        return ret
    
    #為了要接應model.predict Y本該要是np.ndarray or list，要改成list
    def inverse_transform(self, Y, return_pandas=True):
        if(True if(len(np.array(Y).shape)<1) else (np.array(Y).shape[0]==0)):
            return Y
        Y = mylist([Y]) if(not DFP.isiterable(Y)) else mylist(np.transpose(Y).tolist())
        ret = DFP.collection()
        for i, yh in enumerate(self.header):
            return_pandas &= isinstance(Y[i], pd.core.frame.DataFrame)
            y = np.array(Y[i]).reshape(-1) if(DFP.isiterable(Y[i])) else np.array([Y[i]]).reshape(-1)
            try:
                y = np.array(tuple(map(int, y)))
                y_inverse_lbced = self.lbcs[yh].inverse_transform(y).reshape(-1,1)
            except Exception as e:
                exception_process(e, logfile='', stamps=[self.inverse_transform.__name__, yh])
                addlog('y[%s]:%s'%(str(type(y)), str(y)[:200]), logfile='')
                return None
            ret.add(yh = y_inverse_lbced)
        Y_inverse_lbced = dcp(ret.concatenate())
        if(return_pandas):
            columns = self.header
            index = getattr(Y[0], 'index', range(np.array(Y[0]).shape[0])) if(len(Y)>0) else mylist()
            Y_inverse_lbced = pd.DataFrame(Y_inverse_lbced, columns=columns, index=index)
        return Y_inverse_lbced
    
    def inverse_transform_flatten(self, Y):
        np_Y = np.array(Y).reshape(-1, 1) if(len(np.array(Y).shape)==1) else np.array(Y)
        columns = getattr(Y, 'columns', (range(np_Y.shape[1]) if(len(np_Y.shape)>1) else getattr(Y, 'name', [0])))
        index = getattr(Y, 'index', range(np_Y.shape[0]))
        ret = np.empty(shape=Y.shape)
        if(self.n_features_in_ != (np.array(Y).shape[1] if(len(np.array(Y).shape)>1) else 1)):
            return {}['Y shape%s; while HLBC shape%s'%(Y.shape, self.n_features_in_)]
        if(self.n_features_in_>1):
            for i, xh in enumerate(self.header):
                ret[:,i] = self.lbcs[xh].inverse_transform(np_Y[:,i])
        else:
            ret = self.lbc.inverse_transform(Y)
        if(str(type(Y)).find('pandas')>-1):
            ret = pd.DataFrame(ret, columns=columns, index=index) if(self.n_features_in_>1) else pd.Series(ret, name=columns[0], index=index)
        return ret
    
    def get_classes(self, stg_nize=True):
        if(len(self.classes)>1):
            ret = self.classes
            ret = {str(k):str(v) for (k,v) in ret.items()}
            ret = stamp_process('', ret, location=-1, stamp_left='\n', stamp_right='') if(stg_nize) else ret
        else:
            ret = self.classes_
            ret = ','.join(list(map(str, ret))) if(stg_nize) else ret
        return ret
    
#%%
def chineseModuleActivate():
    global m_dictionary
    m_dictionary.clear()
    m_dictionary.update({
        'acc':'準確率',
        'rcl':'召回率',
        'pcs':'精確率',
        '~rcl':'陰性召回率',
        '~pcs':'陰性精確率',
        'pcs_mdn':'中位精確率',
        'rcl_mdn':'中位召回率',
        'pcs_std':'精確率標準差',
        'rcl_std':'召回率標準差',
        'pcs_inlier':'陽性精確率',
        'rcl_inlier':'陽性召回率',
        'pcs_outlier':'陰性精確率',
        'rcl_outlier':'陰性召回率',
        'inlier': '陽性',
        'outlier': '陰性',
        r'$\alpha$':'I型失誤(誤判)率',
        r'$\beta$':'II型失誤(漏判)率',
        r'$f_{1}$':'f分數',
        })
    vs3.plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']

def shapeOfTensor(tensor, noShapeReturnMethod = LOGger.type_string):
    if(isinstance(tensor, list) or isinstance(tensor, tuple)):
        returnType = type(tensor)
        return returnType(map(lambda x:shapeOfTensor(x, noShapeReturnMethod=noShapeReturnMethod), tensor))
    elif(hasattr(tensor, 'shape')):
        return tensor.shape
    return noShapeReturnMethod(tensor)

reg_std_metrics = {}
reg_std_metrics['mse'] = lambda f, p, cell_size=None: mean_squared_error_with_exception_process(f, p, cell_size=cell_size)
# reg_std_metrics['mse'] = lambda f, p: np.mean((f - p)**2)
reg_std_metrics['rmse'] = lambda f, p, cell_size=None: (reg_std_metrics['mse'](f, p, cell_size=cell_size))**(1/2)
reg_std_metrics['r2'] = lambda f, p, cell_size=None: r2_score_with_exception_process(f, p, cell_size=cell_size)
def mean_squared_error_with_exception_process(f, p, cell_size=None):
    try:
        cell_size = cell_size if(DFP.isiterable(cell_size)) else np.array(f).shape[1:]
        if(len(cell_size)>0):
            flat_shape = np.prod(cell_size)
            f = np.array(f).reshape((-1, flat_shape))
            p = np.array(p).reshape((-1, flat_shape))
        return skm.mean_squared_error(f, p)
    except Exception as e:
        exception_process(e, '', stamps=['mean_squared_error'])
        show_vector(np.array(f)[np.array(tuple(map(lambda v: isinstance(v,str), f)))], stamps=['fact str elements'], logfile='')
        show_vector(np.where(np.abs(np.array(f).reshape(-1))==np.inf), stamps=['fact inf elements locs'], logfile='')
        show_vector(np.where(np.array(f).reshape(-1)==np.nan), stamps=['fact nan elements locs'], logfile='')
        show_vector(np.array(p)[np.array(tuple(map(lambda v: isinstance(v,str), p)))], stamps=['pred str elements'], logfile='')
        show_vector(np.where(np.abs(np.array(p).reshape(-1))==np.inf), stamps=['pred inf elements locs'], logfile='')
        show_vector(np.where(np.array(p).reshape(-1)==np.nan), stamps=['pred nan elements locs'], logfile='')
        return np.nan #-1
    
def r2_score_with_exception_process(f, p, cell_size=None):
    try:
        cell_size = cell_size if(DFP.isiterable(cell_size)) else np.array(f).shape[1:]
        if(len(cell_size)>0):
            flat_shape = np.prod(cell_size)
            f = np.array(f).reshape((-1, flat_shape))
            p = np.array(p).reshape((-1, flat_shape))
        return skm.r2_score(f, p)
    except Exception as e:
        exception_process(e, '', stamps=['r2_score'])
        show_vector(np.array(f)[np.array(tuple(map(lambda v: isinstance(v,str), f)))], stamps=['fact str elements'], logfile='')
        show_vector(np.where(np.abs(np.array(f))==np.inf), stamps=['fact inf elements locs'], logfile='')
        show_vector(np.where(np.array(f)==np.nan), stamps=['fact nan elements locs'], logfile='')
        show_vector(np.array(p)[np.array(tuple(map(lambda v: isinstance(v,str), p)))], stamps=['pred str elements'], logfile='')
        show_vector(np.where(np.abs(np.array(p))==np.inf), stamps=['pred inf elements locs'], logfile='')
        show_vector(np.where(np.array(p)==np.nan), stamps=['pred nan elements locs'], logfile='')
        return np.nan #-np.inf
    
#TODO:OKratio
def OKratio(f, p, equal_included=True, pdt_side=0, tol=np.nan):
    if(callalbe(tol) or tol is np.nan or tol is None):
        tol = (np.std(f) if(np.array(f).shape[0]>2) else 1) if(tol==None) else tol
        if(pdt_side>0):
            return np.sum(np.ones(f.shape[0])[
                            p - f >= -tol])/f.shape[0] if(
                            equal_included) else np.sum(np.ones(f.shape[0])[
                            p - f > -tol])/f.shape[0]
        elif(pdt_side<0):
            return np.sum(np.ones(f.shape[0])[
                            p - f <= tol])/f.shape[0] if(
                            equal_included) else np.sum(np.ones(f.shape[0])[
                            p - f < tol])/f.shape[0]
        elif(pdt_side==0):
            return np.sum(np.ones(f.shape[0])[
                    np.abs(p - f) <= tol])/f.shape[0] if(
                            equal_included) else np.sum(np.ones(f.shape[0])[
                    np.abs(p - f) < tol])/f.shape[0]
        else:
            {}['error pdt side:%s'%pdt_side]
    else:
        #for example...... tol = lambda a,b: np.abs(a-b)<=a*0.03
        return np.sum(np.ones(f.shape[0])[
                            tol(f, p)])/f.shape[0]

def shift_confu_matrix_by_bounding_boxes(np_y_fact, np_y_pred, bounding_data_fact, bounding_data_pred, cfM, all_categories=[], 
                                         negative_class=None, iou_threshold = 0.2, **kwags):
    # print('.............................')
    dtype = kwags.get('dtype', lambda x:x)
    all_categories = all_categories if(np.array(all_categories).shape[0]>0) else list(tuple(DFP.uniqueByIndex(np_y_fact)))
    all_categories = list(map(dtype, all_categories))
    negative_class = all_categories[0] if(isinstance(negative_class, type(None))) else negative_class
    # main_columns = [c for c in range(np_y_fact.shape[1]) if not c in bounding_columns_index]
    dim = np.array(bounding_data_fact).shape[1]
    # print('0')
    if(dim!=2):
        print('尚未設計完成!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        return None
    # bounding_data_fact[:,0]是匡列起點，[:,1]是匡列長度，所以np.sum(bounding_data, axis=1)是匡列終點
    # print('1')
    print('shift_confu_matrix_by_bounding_boxes')
    bounding_data_fact = np.array(bounding_data_fact)
    bounding_data_pred = np.array(bounding_data_pred)
    A_to_find_centre = np.array([[1],[1/2]])
    y_fact_bounding_centre = np.matmul(bounding_data_fact, A_to_find_centre)
    y_pred_bounding_centre = np.matmul(bounding_data_pred, A_to_find_centre)
    A_to_find_radius = np.array([[0],[1/2]])
    y_fact_radius = np.matmul(bounding_data_fact, A_to_find_radius)
    y_pred_radius = np.matmul(bounding_data_pred, A_to_find_radius)
    minor_diff = np.min(np.vstack([(np.sum(bounding_data_fact, axis=1) - 
                                 np.array(bounding_data_pred)[:,0]), 
                                 np.sum(bounding_data_pred, axis=1) - 
                                 np.array(bounding_data_fact)[:,0]]), axis=0)
    intersection = np.max(np.vstack([np.zeros(shape=(bounding_data_pred.shape[0],)), np.min(
                       np.vstack([(np.sum(bounding_data_fact, axis=1) - 
                                 np.array(bounding_data_pred)[:,0]), 
                                 np.sum(bounding_data_pred, axis=1) - 
                                 np.array(bounding_data_fact)[:,0]]), axis=0)]), axis=0)
    union = np.where(
        (np.abs(y_fact_bounding_centre - y_pred_bounding_centre)>=(y_fact_radius + y_pred_radius)).reshape(-1),
        1, np.max(np.hstack([bounding_data_fact, bounding_data_pred]), axis=1) - np.min(
                       np.hstack([bounding_data_fact, bounding_data_pred]), axis=1))
    # sys.exit(1)
    iou = intersection/union
    for _cls in [c for c in all_categories if c!=negative_class]:
        shift = np.sum((np_y_pred==_cls).reshape(-1) & (iou <= iou_threshold))
        cls_index = all_categories.index(_cls)
        ncs_index = all_categories.index(negative_class)
        cfM[cls_index,cls_index] = cfM[cls_index,cls_index] - shift
        cfM[cls_index,ncs_index] = cfM[cls_index,ncs_index] + shift
    return cfM

def classConjuction(array):
    return np.apply_along_axis((lambda x:', '.join(list(map(DFP.parse, x)))), 1, array)

#TODO:sumup_confusion_matrix
def sumup_confusion_matrix(np_y_fact=np.where(np.random.random(10)>0.5,'NG','OK'), 
                           np_y_pred=np.where(np.random.random(10)>0.5,'NG','OK'), 
                           all_categories=None, stamps=None, **kwags):
    addlog_ = LOGger.addloger(logfile='')
    dtype = kwags.get('dtype', lambda x:x)
    stamps = stamps if(isinstance(stamps, list)) else []
    all_categories = list(tuple((DFP.uniqueByIndex(
        all_categories if(np.array(all_categories).shape[0]>0 if(DFP.isiterable(all_categories)) else False)  else np_y_fact))))
    all_categories = list(map(dtype, all_categories))
    class_size = len(all_categories)
    bounding_columns_index = kwags.get('bounding_columns_index', None)
    
    main_columns_index = None if(isinstance(bounding_columns_index, type(None))) else [
        c for c in range(np_y_fact.shape[1]) if not c in bounding_columns_index]
    cfM = confusion_matrix(list(map(dtype, np_y_fact)), list(map(dtype, np_y_pred)), labels=all_categories) if(
        isinstance(main_columns_index, type(None))) else confusion_matrix(list(map(dtype, np_y_fact[:,main_columns_index])), 
                                                                          list(map(dtype, np_y_pred[:,main_columns_index])), 
                                                                          labels=all_categories)
    if(not isinstance(kwags.get('bounding_data_fact', None), type(None)) and 
       not isinstance(kwags.get('bounding_data_pred', None), type(None))):
        bounding_data_fact = kwags.pop('bounding_data_fact')
        bounding_data_pred = kwags.pop('bounding_data_pred')
        cfM = shift_confu_matrix_by_bounding_boxes(np_y_fact, np_y_pred, bounding_data_fact, bounding_data_pred, cfM, 
                                                   all_categories=[], negative_class=None,
                                                   iou_threshold = 0.2, **kwags)
        # sys.exit(1)
        stamps = stamps + ['OT']
    addlog_('ordered class name list:%s'%(stamp_process('...' if(class_size>10) else '', 
                                                         all_categories[:10], '','','',',')))
    log = 'confusion matrix:%s'%str(cfM[:5,:5]).replace(']\n','...],').replace(']]','...],...]')
    addlog_(log)
    stamps = [v for v in stamps if v!='']+[''] if(stamps) else []
    return cfM

#TODO:sumup_confusion_rates
def sumup_confusion_rates(cfM, all_categories=None, header_title=None, f_beta=1,
                          xrot=90, beta=1, kappa=np.nan, max_class_size=500,
                          binary_inlier_code = 1, stamps=None, addlog=addlog, **kwags):
    all_categories = list(tuple((DFP.uniqueByIndex(all_categories)) if(
        np.array(all_categories).shape[0]>0 if(DFP.isiterable(all_categories)) else False)  else np.arange(cfM.shape[0])))
    class_size = len(all_categories)
    
    recalls = [((cfM[k,k]/sum(list(cfM[k,:]))) if(
            sum(list(cfM[k,:]))>0) else np.nan) for k in range(min(class_size,max_class_size))]
    precisions = [((cfM[k,k]/sum(list(cfM[:,k]))) if(
            sum(list(cfM[:,k]))>0) else np.nan) for k in range(min(class_size,max_class_size))]
    stamps = [v for v in stamps if v!='']+[''] if(stamps) else []
    
    clf_eva_class = pd.DataFrame(
            [recalls, precisions], index=['rcl','pcs']).T
    fscn = r'$f_{%s}$'%('%d'%beta if(int(beta)==beta) else '%.2f'%beta)
    clf_eva_class[fscn] = (
            1+beta**2)*clf_eva_class['pcs']*clf_eva_class['rcl']/(
            beta**2*clf_eva_class['pcs']+clf_eva_class['rcl'])
    clf_eva_class = clf_eva_class.fillna(0).round(2)
    clf_eva_class = (pd.DataFrame(all_categories[:max_class_size], columns=[header_title if(LOGger.isinstance_not_empty(header_title,str)) else 'main'],
                    index=range(min(class_size,max_class_size))).join(clf_eva_class))
    fact_counts = np.sum(cfM, axis=1)
    clf_eva_class['fact_counts'] = fact_counts[:max_class_size]
    pred_counts = np.sum(cfM, axis=0)
    clf_eva_class['pred_counts'] = pred_counts[:max_class_size]
    fact_ratio = fact_counts/np.sum(fact_counts)
    clf_eva_class['fact_ratio'] = fact_ratio[:max_class_size]
    if(isinstance(kwags.get('missed_label_loc', None), int)):
        missed_label_loc = kwags['missed_label_loc']
        missed_anti_recall = [((cfM[k,missed_label_loc]/sum(list(cfM[k,:]))) if(
                sum(list(cfM[k,:]))>0) else np.nan) for k in range(min(class_size,max_class_size))]
        clf_eva_class['missed_anti_recall'] = missed_anti_recall
        crossed_mistake_anti_recall = [((((sum(list(cfM[k,:]))) - cfM[k,k] - cfM[k,missed_label_loc])/sum(list(cfM[k,:]))) if(
                sum(list(cfM[k,:]))>0) else np.nan) for k in range(min(class_size,max_class_size))]
        clf_eva_class['crossed_mistake_anti_recall'] = crossed_mistake_anti_recall
    clf_eva_class = clf_eva_class.sort_values(['pcs', fscn], ascending=False) #先排pcs, 再排f-score
    clf_eva_class = clf_eva_class[(clf_eva_class['fact_counts']>0)|(clf_eva_class['pred_counts']>0)]
    clf_eva_class_for_graph = clf_eva_class.copy()
    for hd in clf_eva_class: #把有非數字的欄位剔除
        if(not np.array(tuple(map(DFP.isnonnumber, clf_eva_class[hd]))).any()):
            clf_eva_class_for_graph_hd = clf_eva_class[hd].map(float)
            if(not (np.array(clf_eva_class_for_graph_hd)>1).any() and not (np.array(clf_eva_class_for_graph_hd)<0).any()):
                continue
        clf_eva_class_for_graph.pop(hd)
    clf_eva_class_for_graph.applymap(lambda s:DFP.astype(s, default=0))
    acc = sum([cfM[k,k] for k in range(class_size)])/cfM.sum()
    mdn_pcs = np.median(precisions)
    ret = {'acc':acc, 'pcs_mdn':mdn_pcs, 'MultiClassesEvaluation':clf_eva_class}
    if(class_size<=2):
        if(binary_inlier_code>=len(all_categories)):
            addlog('binary_inlier_code[%d]>len(all_categories)[%d]!!!!'%(binary_inlier_code, len(all_categories)), colora=LOGger.FAIL)
            return ret
        binary_outlier_code = 1 - binary_inlier_code
        precision = precisions[binary_inlier_code]
        recall = recalls[binary_inlier_code]
        e_type1 = 1 - precision
        e_type2 = 1 - recall
        f_score = (1+beta**2)*precision*recall/(beta**2*precision+recall)
        table = pd.DataFrame([['%s:%s %%'%(m_dictionary.get('pcs','pcs'), DFP.parse(precision*100, digit=3)), 
                               '%s:%s %%'%(m_dictionary.get('rcl','rcl'), DFP.parse(recall*100, digit=3))], 
                              ['%s:%s %%'%(m_dictionary.get('~pcs','~pcs'), DFP.parse(precisions[binary_outlier_code]*100, digit=3)), 
                               '%s:%s %%'%(m_dictionary.get('~rcl','~rcl'), DFP.parse(recalls[binary_outlier_code]*100, digit=3))], 
                              ['%s:%s %%'%(m_dictionary.get(r'$\alpha$',r'$\alpha$'), DFP.parse(100*e_type1, digit=3)), 
                               '%s:%s %%'%(m_dictionary.get(r'$\beta$',r'$\beta$'), DFP.parse(100*e_type2, digit=3))],
                              ['%s:%s %%'%(m_dictionary.get('acc','acc'), DFP.parse(100*acc, digit=3)),
                               '%s:%s %%'%(m_dictionary.get(r'$f_{1}$',r'$f_{1}$'), DFP.parse(f_beta*100, digit=3))]])
        ret.update({'pcs':precision, 'rcl':recall, 'e_type1':e_type1, 'e_type2':e_type2, 'f-score':f_score, 'BinaryClassesEvaluation':table, 'f_beta':f_beta})
    return ret

#TODO:sumup_confusion_matrix_rates
def sumup_confusion_matrix_rates(np_y_fact, np_y_pred, all_categories=None, f_beta=1, binary_inlier_code=1, stamps=None, **kwags):
    cfM = sumup_confusion_matrix(np_y_fact, np_y_pred, all_categories=all_categories, stamps=stamps, **kwags)
    try:
        kappa = cohkpa(np_y_fact, np_y_pred)
    except Exception as e:
        exception_process(
            e, os.path.join('log', 'log_%t.txt'), stamps=['sumup_confusion_matrix_rates:cohkpa']+stamps)
        kappa = np.nan
        addlog('cohkpa failed!!!! checking...')
        addlog('np_y_fact:%s'%str(np_y_fact)[:200])
        addlog('shape:%s'%str(np_y_fact.shape))
        addlog('np_y_pred:%s'%str(np_y_pred)[:200])
        addlog('shape:%s'%str(np_y_pred.shape))
    ret = sumup_confusion_rates(cfM, all_categories, f_beta=f_beta, kappa=kappa, binary_inlier_code=binary_inlier_code, stamps=stamps, **kwags)
    ret.update({'cfM':cfM, 'kappa':kappa})
    return ret

def saveNumData(data, file=None, stamps=None, exp_fd='test', fn='fn', **kwags):
    stamps = stamps if(isinstance(stamps, list)) else []
    if(len(data.index)>10000):
        file = file if(LOGger.isinstance_not_empty(file, str)) else os.path.join(exp_fd, '%s.pkl'%LOGger.stamp_process('',stamps,'','','','_',for_file=True))
        LOGger.CreateFile(file, lambda f: data.to_pickle(f))
    else:
        file = file if(LOGger.isinstance_not_empty(file, str)) else os.path.join(exp_fd, '%s.csv'%LOGger.stamp_process('',stamps,'','','','_',for_file=True))
        LOGger.CreateFile(file, lambda f: data.to_csv(f))
    return True

#TODO:set_cfmtx_metrics
def set_cfmtx_metrics(n_classes, fbeta=1, binary_inlier_code=1):
    binary_outlier_code = 1 - binary_inlier_code
    cfmtx_metrics= {}
    cfmtx_metrics['acc'] = lambda m: np.sum(
            [m[k,k] for k in range(n_classes)])/np.sum(m)
    stgbeta = '%d'%fbeta if(fbeta//1==fbeta) else '%.2f'%fbeta
    if(n_classes>2):
        cfmtx_metrics['rcl_mdn'] = lambda m: np.median([m[k,k]/sum(
            list(m[k,:])) for k in range(n_classes)])
        cfmtx_metrics['rcl_std'] = lambda m: np.std([m[k,k]/sum(
            list(m[k,:])) for k in range(n_classes)])
        cfmtx_metrics['pcs_mdn'] = lambda m: np.median([m[k,k]/sum(
            list(m[:,k])) for k in range(n_classes)])
        cfmtx_metrics['pcs_std'] = lambda m: np.std([m[k,k]/sum(
            list(m[:,k])) for k in range(n_classes)])
        cfmtx_metrics[r'$f_{%s}$-score_mdn'%stgbeta] = lambda m:np.median([(
                    1+fbeta**2)*(m[k,k]/sum(
                list(m[:,binary_inlier_code])))*(m[k,k]/sum(
                list(m[k,:])) if(sum(list(m[k,:]))>0) else 0)/(fbeta**2*(m[
                k,k]/sum(list(m[:,k])))+(m[k,k]/sum(list(m[k,:])) if(
                sum(list(m[k,:]))>0) else 0)) for k in range(n_classes)])
    else:
        cfmtx_metrics['rcl_inlier'] = lambda m: m[
                binary_inlier_code,binary_inlier_code]/sum(
                list(m[binary_inlier_code,:])) if(
                sum(list(m[binary_inlier_code,:]))>0) else 0
        cfmtx_metrics['pcs_inlier'] = lambda m: m[
                binary_inlier_code,binary_inlier_code]/sum(
                list(m[:,binary_inlier_code]))
        cfmtx_metrics[r'$\alpha$'] = lambda m: m[
                binary_outlier_code,binary_inlier_code]/sum(
                list(m[:,binary_inlier_code])) if(
                sum(list(m[binary_inlier_code,:]))>0) else 0
        cfmtx_metrics[r'$\beta$'] = lambda m: m[
                binary_inlier_code,binary_outlier_code]/sum(
                list(m[:,binary_outlier_code])) if(
                sum(list(m[binary_outlier_code,:]))>0) else 0
        cfmtx_metrics['rcl_outlier'] = lambda m: m[
            binary_outlier_code,binary_outlier_code]/sum(
            list(m[binary_outlier_code,:])) if(
            sum(list(m[binary_outlier_code,:]))>0) else 0
        cfmtx_metrics['pcs_outlier'] = lambda m: m[
            binary_outlier_code,binary_outlier_code]/sum(
            list(m[:,binary_outlier_code]))
        cfmtx_metrics[r'f$_{%s}$-score'%stgbeta] = lambda m:(
                    1+fbeta**2)*(m[
                binary_inlier_code,binary_inlier_code]/sum(
                list(m[:,binary_inlier_code])))*(m[
                binary_inlier_code,binary_inlier_code]/sum(
                list(m[binary_inlier_code,:])) if(
                sum(list(m[binary_inlier_code,:]))>0) else 0)/(fbeta**2*(m[
                binary_inlier_code,binary_inlier_code]/sum(
                list(m[:,binary_inlier_code])))+(m[
                binary_inlier_code,binary_inlier_code]/sum(
                list(m[binary_inlier_code,:])) if(
                sum(list(m[binary_inlier_code,:]))>0) else 0))
    return cfmtx_metrics

def sumup_regression_norms(y_real, y_pred, tol=np.nan, names=(list(tuple(reg_std_metrics.keys()))+['OKR']), 
                           xdim=0, equal_included=False, pdt_side=0,
                           additional_norms=None, additional_kwagses={}, **kwags):
    y_pred = np.array(y_pred)
    y_real = np.array(y_real)
    tol = np.std(y_real) if(np.isnan(DFP.astype(tol, default=np.nan)) and not callable(tol)) else tol
    ret = {k: reg_std_metrics[k](y_real, y_pred) for k in names if k in reg_std_metrics}
    y_real_axis1dim = y_real.shape[1] if(len(y_real.shape)>1) else 1
    pred_real_offset = np.linalg.norm((y_pred - y_real).reshape(-1,y_real_axis1dim), axis=1) if(
                y_real_axis1dim>1) else (y_pred - y_real).reshape(-1)
    if('r2' in names and xdim!=0):
        ret['adr2'] = 1-((1-ret['r2'])*(y_real.shape[0]-1)/(y_real.shape[0]-xdim-2))
    if('OKR' in names):
        if(not np.isnan(DFP.astype(tol, d_type=float, default=np.nan))):
            if(pdt_side>0):
                if(y_real_axis1dim==1):
                    ret['OKR'] = np.sum(np.ones(y_real.shape[0])[
                                    pred_real_offset >= -tol])/y_real.shape[0] if(
                                    equal_included) else np.sum(np.ones(y_real.shape[0])[
                                    pred_real_offset > -tol])/y_real.shape[0]
                else:
                    ret['OKR'] = np.nan
            elif(pdt_side<0):
                if(y_real_axis1dim==1):
                    ret['OKR'] = np.sum(np.ones(y_real.shape[0])[
                                pred_real_offset <= tol])/y_real.shape[0] if(
                                equal_included) else np.sum(np.ones(y_real.shape[0])[
                                pred_real_offset < tol])/y_real.shape[0]
                else:
                    ret['OKR'] = np.nan
            elif(pdt_side==0):
                ret['OKR'] = np.sum(np.ones(y_real.shape[0])[
                        np.abs(pred_real_offset) <= tol])/y_real.shape[0] if(
                                equal_included) else np.sum(np.ones(y_real.shape[0])[
                        np.abs(pred_real_offset) < tol])/y_real.shape[0]
            else:
                {}['error pdt side:%s'%pdt_side]
        else:
            #for example...... tol = lambda a,b: np.abs(a-b)<=a*0.03
            ret['OKR'] = np.sum(np.ones(y_real.shape[0])[tol(y_real, y_pred)])/y_real.shape[0]
    if(isinstance(additional_norms, dict)):
        for k,f in additional_norms.items():
            additional_kwags = additional_kwagses.get(k, {})
            ret[k] = f(y_real, y_pred, **additional_kwags)
    if(callable(tol)):
        ret['mask'] = tol(y_real, y_pred)
    else:
        if(pdt_side>0):
            if(y_real_axis1dim==1):
                ret['mask'] = (pred_real_offset >= -tol) if(equal_included) else (pred_real_offset > -tol)
            else:
                ret['mask'] = np.full(y_real.shape[0], np.nan)
        elif(pdt_side<0):
            if(y_real_axis1dim==1):
                ret['mask'] = (pred_real_offset <= tol) if(equal_included) else (pred_real_offset < tol)
            else:
                ret['mask'] = np.full(y_real.shape[0], np.nan)
        else:
            ret['mask'] = (np.abs(pred_real_offset) <= tol) if(equal_included) else (np.abs(pred_real_offset) < tol)
    ret['tol'] = tol if(tol>0) else LOGger.mystr('-')
    ret['size'] = np.array(y_real).shape[0]
    return ret

def parseMeasureExport(export):
    if(not isinstance(export, dict)):
        return {}
    infrm_plot = {}
    if('miou' in export):   infrm_plot['miou'] = dcp('%.2f%%'%(export['miou']*100))
    # if('count' in export):   infrm_plot[r'$n$'] = export.pop('count')
    if('count' in export):   infrm_plot[r'$n$'] = dcp(export['count'])
    if('tol' in export):   
        tol = dcp(export['tol'])
        infrm_plot[r'$\delta$'] = 'non-constant' if(callable(tol)) else dcp('%s'%DFP.parse(tol, digit=4))
    if('OKR' in export):   infrm_plot['OKR'] = dcp('%s %%'%DFP.parse(export['OKR']*100, digit=4))
    if('rmse' in export):   infrm_plot['rmse'] = dcp(export['rmse'])
    return infrm_plot

def determined_threshold(dataReference, contamination=0.1, total_selected_label=None, **kwags):
    total_selected_label = np.full(dataReference.shape[0], True) if(total_selected_label is None) else total_selected_label
    #如果進容差的資料太少，就還是以全部train data來調比例
    if(dict(zip(*np.unique(total_selected_label, return_counts=1))).get(True, 0)<4):
        total_selected_label = np.full(dataReference.shape[0], True)
    data_good = dataReference.reshape(-1)[total_selected_label]
    return np.percentile(data_good, (1-contamination)*100, axis=0)
    
def compute_threshold_variation_stream_target(metric, score_array, thrshs, np_y_data, 
                                              classfication_method=None, comparisonMetrics=None, return_values={}, 
                                              ascending_better=True, **kwags):
    addlog = kwags.get('addlog', LOGger.addlog)
    thread_key = kwags['thread_key'] if('thread_key' in kwags) else ''
    addlog('%scompute_threshold_variation_stream start'%(
                    '[%s]'%thread_key if(thread_key) else ''))
    stream, maxdot, evaluations = compute_threshold_variation_stream(metric, score_array, thrshs, np_y_data, 
                                                                     classfication_method, comparisonMetrics, 
                                                                     ascending_better=ascending_better, **kwags)
    return_values['stream'] = stream
    return_values.update({'maxdot':maxdot}) if(type(maxdot)!=type(None)) else None
    return_values['evaluations'] = evaluations
    addlog('maxdot:%s, stream:\n%s'%(str(maxdot), str(stream)[:200]), stamps=[thread_key])
    addlog('compute_threshold_variation_stream over', stamps=[thread_key])
    
#TODO:compute_threshold_variation_stream
def compute_threshold_variation_stream(metric, score_array, thrshs, np_y_data, classfication_method=None, comparisonMetrics=None, ascending_better=True, **kwags):
    """
    

    Parameters
    ----------
    metric : TYPE
        metric for confusion targets. Ex: pcs, rcl, acc, ....
    score_array : TYPE
        superpositions produces from causes thru a model.
    thrshs : array of float((0,1))
        test for superpositions transforming to collasped state.
    np_y_data : array of int({0,1})
        of collasped states.
    classfication_method : TYPE, optional
        DESCRIPTION. The default is None.
    otherMetrics : TYPE, optional
        DESCRIPTION. The default is [].
    ascending_better : TYPE, optional
        DESCRIPTION. The default is True.
    **kwags : TYPE
        DESCRIPTION.

    Returns
    -------
    stream : TYPE
        DESCRIPTION.
    maxdot : TYPE
        DESCRIPTION.
    evaluations : TYPE
        DESCRIPTION.

    """
    score_array = np.array(score_array).reshape(-1)
    np_y_data = np.array(np_y_data).reshape(-1)
    stream = []
    #應用門檻的方式，沒有給定就是以超過門檻定義為[1]類-異常
    classfication_method = (lambda score_array, thrsh: score_array>thrsh) if(classfication_method is None) else classfication_method
    # np_y_data_flat = np_y_data.reshape(-1)
    for thrsh in thrshs:
        np_y_csped_byTh = np.array(list(map(int, classfication_method(score_array, thrsh))))
        try:
            cfM = confusion_matrix(np_y_data, np_y_csped_byTh.reshape(-1))
        except:
            cfM = confusion_matrix(np.array(tuple(map(lambda x:bool(x), np_y_data))), 
                                   np.array(tuple(map(lambda x:bool(x), np_y_csped_byTh.reshape(-1)))))
        stream += [metric(cfM)]
    try:
        best_thrsh = thrshs[np.argsort(stream)[-1]] if(ascending_better) else thrshs[np.argsort(stream)[0]]
        best_evaluation = np.max(stream) if(ascending_better) else np.min(stream)
        maxdot = (best_thrsh, best_evaluation)
    except:
        maxdot = None
    #在這個指標最佳的情況下，其他指標的狀況
    np_y_csped_byBestTh = np.array(list(map(int, classfication_method(score_array, best_thrsh))))
    cfM = confusion_matrix(np_y_data, np_y_csped_byBestTh)
    comparisonMetrics = [] if(not isinstance(comparisonMetrics , list)) else comparisonMetrics
    evaluations=[mtc(cfM) for mtc in comparisonMetrics]
    return stream, maxdot, evaluations

#TODO:threshold_variation_analysis
#edge_plank=(0.1,-0.5,0.85,0.95)
#score_method= model.decision_function
#score_method= model.predict_proba
def threshold_variation_analysis(fig, 
                                 np_y_data=np.where(np.random.random(10)>0.5,1,0), 
                                 np_s_data=np.random.random(10), 
                                 preprocessor=None, thrsh_range=(), n_sample=100,
                                 topic='', binary_inlier_code=1, classification_method=None, f_beta=1, stamps=None, thresholds=[0.5], 
                                 tresholdYPosRatio=np.array([1,9])/10, colorThreshold=(1,0,0,0.3), all_categories=None, 
                                 file=None, 
                                 **kwags):
    stamps = stamps if(isinstance(stamps, list)) else []
    addlog = kwags.get('addlog', LOGger.addloger(logfile=''))
    np_s_data = np.array(np_s_data)
    sc_min = np.amin(np_s_data) if(thrsh_range==()) else thrsh_range[0]
    sc_max = np.amax(np_s_data) if(thrsh_range==()) else thrsh_range[1]
    h = (sc_max - sc_min)/10
    thrshs = np.linspace(sc_min+h, sc_max-h, n_sample)
    n_classes = len(list(preprocessor.classes_)) if(preprocessor is not None) else len(DFP.uniqueByIndex(np_y_data))
    np_y_data = np_y_data if(preprocessor is None) else preprocessor.transform(np_y_data)
    
    cfmtx_metrics = set_cfmtx_metrics(n_classes, fbeta=f_beta, binary_inlier_code=binary_inlier_code)
    thds, return_value_basket=[], {}
    for k, f in cfmtx_metrics.items():
        return_value_basket[k] = {}
        thd = threading.Thread(target=compute_threshold_variation_stream_target, 
                              args=(f, np_s_data, thrshs, np_y_data, classification_method, 
                                    list(tuple(cfmtx_metrics.values()))), 
                              kwargs={'thread_key':k, 'return_values':return_value_basket[k], 'addlog':addlog,
                                      'ascending_better':(sum(list(map(lambda s:k.find(s), ['alpha','beta','誤'])))<=0)})
        thd.start()
        thds.append(thd)
    for thd in thds:
        thd.join()
    if(isinstance(fig, vs3.plt.Figure)):
        fig.clf()
        outer_grid = gridspec.GridSpec(2, 1, height_ratios=[2,1], figure=fig)
        # 門檻分析曲線
        axCurves = fig.add_subplot(outer_grid[0])
        if(n_classes==2):
            axDistribution = axCurves.twinx()
            vs3.normhist(np_s_data[np_y_data==(binary_inlier_code if(preprocessor is None) else preprocessor.classes_[binary_inlier_code])].reshape(-1), 
                         ax=axDistribution, color=(0,0,1,0.3), stamps=[''])
            vs3.normhist(np_s_data[np_y_data!=(binary_inlier_code if(preprocessor is None) else preprocessor.classes_[1-binary_inlier_code])].reshape(-1), 
                         ax=axDistribution, color=(0,0.05,0.3), stamps=[''])
            axDistribution.legend()
        colors = dict(zip(*[cfmtx_metrics, vs2.cm_rainbar(len(cfmtx_metrics), c_alpha=0.5)]))
        df_evaluations = pd.DataFrame()
        for j, (k, f) in enumerate(cfmtx_metrics.items()):
            return_values = return_value_basket[k]
            maxdot = dcp(return_values.get('maxdot', None))
            if(type(maxdot)==type(None)):
                addlog('[%s]maxdot failed:%s'%(k, str(maxdot)[:200]), colora=LOGger.FAIL)
                continue
            try:
                list(map((lambda x:x+0), np.array(maxdot)))
            except:
                addlog('[%s]偵測到非數字:\n%s'%(k, str(maxdot)[:200]), colora=LOGger.FAIL)
                continue
            stream = dcp(return_values['stream'])
            stream = DFP.astype(stream, d_type=(lambda x:x+0), default=0)
            evaluations = dcp(return_values['evaluations'])
            df_evaluations[k] = pd.Series(evaluations, index=cfmtx_metrics.keys())
            
            axCurves.plot(tuple(thrshs), stream, label=m_dictionary.get(k,k), color=colors.get(k, colors.get(j)))
            axCurves.text(*maxdot, '(%s)'%LOGger.stamp_process('',maxdot,'','','',', ',digit=4), color=(*(colors.get(k, colors.get(j))[:3]), 1))
        for threshold in thresholds:
            axCurves.axvline(threshold, ls='--', color=colorThreshold)
            axCurves.text(threshold, np.sum(tresholdYPosRatio*np.array(axCurves.get_ylim())), DFP.parse(threshold, digit=4))
        axCurves.set_xlabel('scores')
        axCurves.set_ylabel('scoreEvaluationTargets')
        all_categories = LOGger.mylist(tuple((DFP.uniqueByIndex(all_categories if(
            np.array(all_categories).shape[0]>0 if(DFP.isiterable(all_categories)) else False) else np_y_data))))
        title = LOGger.stamp_process('',['%s:%s'%(m_dictionary.get('inlier','inlier'), DFP.parse(all_categories[binary_inlier_code]))])
        axCurves.set_title(title)
        
        axTable = fig.add_subplot(outer_grid[1])
        vs.matrix_dataframe(df_evaluations.round(4), title=title, file=f, 
                            index=df_evaluations.index.map(lambda x:m_dictionary.get(x,x)), 
                            header=df_evaluations.index.map(lambda x:m_dictionary.get(x,x)), 
                            headerhide=False, indexhide=False, ax=axTable)
        if(LOGger.isinstance_not_empty(file,str)):
            LOGger.CreateFile(file, lambda f:vs3.end(fig, file=f))
    return True
#TODO:threshold_variation_analysis_target
def threshold_variation_analysis_target(fig, 
                                        np_y_data=np.where(np.random.random(10)>0.5,1,0), 
                                        np_s_data=np.random.random(10), **kwags):
    addlog = kwags.get('addlog', LOGger.addloger(logfile=''))
    threshold_variation_analysis_lock.acquire()
    thread_key = kwags['thread_key'] if('thread_key' in kwags) else ''
    addlog('%sthreshold_variation_analysis start', stamps=[thread_key])
    try:
        threshold_variation_analysis(fig, np_y_data, np_s_data, **kwags)
        addlog('threshold_variation_analysis over', stamps=[thread_key])
    except Exception as e:
        LOGger.exception_process(e, stamps=[thread_key], logfile='')
    threshold_variation_analysis_lock.release()

#TODO:threshold_variation_analysis_target_threading
def threshold_variation_analysis_target_threading(fig=None, 
                                                  np_y_data=np.where(np.random.random(10)>0.5,1,0), 
                                                  np_s_data=np.random.random(10), 
                                                  figsize=(20,20), 
                                                  ret=None, file=None, **kwags):
    addlog = kwags.get('addlog', LOGger.addlog)
    kwags['file'] = file
    fig_ = fig if(isinstance(fig, vs3.plt.Figure)) else vs3.plt.Figure(figsize=figsize)
    threshold_variation_analysis_thread = threading.Thread(target=threshold_variation_analysis_target,
                                                           args=(fig_, np_y_data, np_s_data),
                                                           kwargs=kwags)
    addlog('threshold_variation_analysis start...')
    threshold_variation_analysis_thread.start()
    if(isinstance(ret,dict)):   ret['threshold_variation_analysis_thread'] = threshold_variation_analysis_thread
    if(fig is None):
        return fig_
    return True
        

def dataMeasureVariance(headerZone, outcomes, predictions):
    return False

def determineFigsize4HeaderZone(data_prop):
    if(data_prop=='continuous'):
        return (15,20)
    elif(data_prop in ['discrete','categorical','binary']):
        return (15,20)
    else:
        return (10,10)

def plotFromDataMeasureVarianceEnd(DMVed, figSource=None, file=None, stamps=None, exp_fd='.', **kwags):
    if(not DMVed):  return False
    figOutput = DMVed if(isinstance(DMVed, vs3.plt.Figure)) else figSource
    stamps = stamps if(isinstance(stamps, list)) else []
    fn = LOGger.stamp_process('',stamps,'','','','_',for_file=True)
    file = file if(LOGger.isinstance_not_empty(file,str)) else os.path.join(exp_fd, '%s.jpg'%fn)
    LOGger.CreateFile(file, lambda f:vs3.end(figOutput, file=f))
    return True

def continuousDataMeasureVariance(y_fact=np.random.random(10,), np_y_pred=np.random.random(10,), 
                                  additional_norms=None, tol=np.nan, handler=None, ret=None, stamps=None, cell_size=None, **kwags):
    """
    compute the contunuous comparison properties of y_fact and np_y_pred

    Parameters
    ----------
    y_fact : TYPE
        DESCRIPTION.
    np_y_pred : TYPE
        DESCRIPTION.
    additional_norms : TYPE, optional
        DESCRIPTION. The default is None.
    tol : TYPE, optional
        DESCRIPTION. The default is None.
    handler : TYPE, optional
        DESCRIPTION. The default is None.
    ret : TYPE, optional
        DESCRIPTION. The default is None.
    stamps : TYPE, optional
        DESCRIPTION. The default is None.
    cell_size : TYPE, optional
        used if len of data shape > 1 for flatten. The default is None.
    **kwags : TYPE
        DESCRIPTION.

    Returns
    -------
    bool
        DESCRIPTION.

    """
    try:
        np_y_fact = np.array(getattr(y_fact, 'values', y_fact).tolist())
        np_y_pred = np.array(getattr(np_y_pred, 'values', np_y_pred).tolist())
    except Exception as e:
        LOGger.exception_process(e, logfile='')
        return False
    if(np_y_fact.shape!=np_y_pred.shape):
        LOGger.addlog('dataMeasureVariance ERR: np_y_fact.shape!=np_y_pred.shape', np_y_fact.shape, np_y_pred.shape, colora='\x1b[31m', logfile='', stamps=stamps)
        return False
    if(y_fact.shape[0]<2):
        LOGger.addlog('data too poor!!! Count:%d'%y_fact.shape[0], stamps=stamps, colora='\x1b[31m')
        return False
    stamps = stamps if(isinstance(stamps, list)) else []
    axis1dim = np_y_fact.shape[1] if(len(np_y_fact.shape)>1) else 1
    if(len(y_fact.shape)>1):
        if(np.product(y_fact.shape[1:])>1):
            if(not isinstance(ret, dict)):  ret = {}
            eval_data = pd.DataFrame(index=getattr(y_fact,'index',np.arange(y_fact.shape[0])))
            np_y_fact_avg = DFP.transformGroupofCellsToFlatBatch(np_y_fact, cell_size=cell_size) if(
                DFP.isiterable(cell_size)) else DFP.transformCellsToFlatBatch(np_y_fact)
            np_y_pred_avg = DFP.transformGroupofCellsToFlatBatch(np_y_pred, cell_size=cell_size) if(
                DFP.isiterable(cell_size)) else DFP.transformCellsToFlatBatch(np_y_pred)
            if(not DFP.isiterable(cell_size) and (len(y_fact.shape)>2 or len(np_y_pred.shape)>2)):
                print('cell_size', cell_size, 'y_fact.shape', y_fact.shape, 'np_y_pred.shape', np_y_pred.shape, 'incompatible!!!!')
                return True
            main_regression_targets = sumup_regression_norms(np_y_fact_avg, np_y_pred_avg, additional_norms=additional_norms, tol=tol, **kwags)
            LOGger.addDebug('main_regression_targets', str(main_regression_targets))
            np_y_abserr = np.linalg.norm((np_y_fact - np_y_pred).reshape(-1, axis1dim), axis=1).reshape(-1)
            eval_data['abserr'] = np.array([np_y_abserr]).transpose()
            ret['main'] = dcp(main_regression_targets)
            retTemp = {}
            if(not DFP.isiterable(cell_size)):  #cell_size的話還要拆解內部結構，太龐大了，算main就好
                for i,k in enumerate(getattr(y_fact, 'columns', np.arange(y_fact.shape[1]))):
                    retTemp.clear()
                    if(not continuousDataMeasureVariance(getattr(y_fact, 'values', y_fact)[:,i], getattr(np_y_pred, 'values', np_y_pred)[:,i], 
                                                        ret=retTemp, additional_norms=additional_norms, tol=tol, stamps=stamps+[k])):
                        LOGger.addlog('continuousDataMeasureVariance failed', stamps=stamps+[k], colora='\x1b[31m')
                        continue
                    eval_data[list(map((lambda x:LOGger.stamp_process('',[k, x],'','','','_')), ['fct','prd','err','abserr']))] = dcp(retTemp['np_evaluation'])
                    ret[k] = dcp(retTemp)
            else:
                print(stamps, 'cell_size', cell_size)
            if(handler!=None):   handler.eval_data = dcp(eval_data)
            if(handler!=None):   handler.export = dcp(ret)
            return True
        else:
            if hasattr(y_fact, 'index'):
                y_fact = pd.Series(y_fact.values.reshape(-1), index=y_fact.index, name=getattr(y_fact,'columns',[getattr(y_fact,'name',None)])[0])
            else:
                y_fact =np.array(y_fact).reshape(-1)
            np_y_pred = np_y_pred.reshape(-1)
            if(not continuousDataMeasureVariance(y_fact, np_y_pred, additional_norms=additional_norms, tol=tol, handler=handler, ret=ret, stamps=stamps, cell_size=cell_size, **kwags)):
                LOGger.addlog('continuousDataMeasureVariance failed', stamps=stamps+['main'], colora=LOGger.FAIL)
                return False
            return True
    LOGger.addDebug('sumup_regression_norms tol', tol)
    regression_targets = sumup_regression_norms(np_y_fact, np_y_pred, additional_norms=additional_norms, tol=tol, **kwags)
    LOGger.addDebug('regression_targets', str(regression_targets))
    LOGger.addDebug('stamps', str(stamps))
    np_y_err = (np_y_fact - np_y_pred).reshape(-1)
    np_y_abserr = np.linalg.norm((np_y_fact - np_y_pred).reshape(-1, axis1dim), axis=1).reshape(-1)
    np_evaluation = np.array([np_y_fact, np_y_pred, np_y_err, np_y_abserr]).transpose()
    ret = ret if(isinstance(ret, dict)) else {}
    ret.update(regression_targets)
    ret['count'] = np_y_fact.shape[0]
    ret['np_evaluation'] = np_evaluation
    ret['np_y_err'] = np_y_err
    ret['mask'] = dcp(regression_targets['mask'])
    if(handler!=None):
        handler.eval_data = pd.DataFrame(np_evaluation, columns=list(map((lambda x:LOGger.stamp_process('',stamps+[x],'','','','_')), ['fct','prd','err','abserr'])),
                                        index=getattr(y_fact,'index',np.arange(y_fact.shape[0])))
        handler.export = dcp(ret)
    return True

def drawContinuousDataMeasureVariance(y_fact=np.random.random(10,), np_y_pred=np.random.random(10,), 
                                      tol=np.nan, export=None, fig=None, ax=None, figsize=(10,7), stamps=None, 
                                      mask=None, cell_size=None, layoutAdding=(1,1,1), **kwags):
    try:
        np_y_fact = np.array(getattr(y_fact, 'values', y_fact).tolist())
        np_y_pred = np.array(getattr(np_y_pred, 'values', np_y_pred).tolist())
    except Exception as e:
        LOGger.exception_process(e, logfile='')
        return False
    if(np_y_fact.shape!=np_y_pred.shape):
        return False
    if(y_fact.shape[0]<2):
        return True
    export = export if(isinstance(export, dict)) else {}
    if(tol is None):    tol = export.get('tol')
    if(not isinstance(fig, vs3.plt.Figure)):
        fig = vs3.plt.Figure(figsize=figsize)
        if(not drawContinuousDataMeasureVariance(y_fact=y_fact, np_y_pred=np_y_pred, 
                                                 tol=tol, export=export, fig=fig, ax=ax, stamps=stamps, 
                                                 mask=mask, cell_size=cell_size, layoutAdding=layoutAdding, **kwags)):
            return None
        return fig
    else:
        fig.clf()
    if(len(y_fact.shape)>1):
        cell_size = cell_size if(DFP.isiterable(cell_size)) else None
        np_y_factFlat = DFP.transformGroupofCellsToFlatBatch(np_y_fact, cell_size=cell_size) if(
            DFP.isiterable(cell_size)) else DFP.transformCellsToFlatBatch(np_y_fact)
        np_y_predFlat = DFP.transformGroupofCellsToFlatBatch(np_y_pred, cell_size=cell_size) if(
            DFP.isiterable(cell_size)) else DFP.transformCellsToFlatBatch(np_y_pred)
        np_y_fact_avg = np.linalg.norm(np_y_factFlat, axis=1)
        np_y_pred_avg = np.linalg.norm(np_y_predFlat, axis=1)
        if(y_fact.shape[1]==1 or DFP.isiterable(cell_size)):#cell_size的話還要拆解內部結構，太龐大了，算main就好
            if(not isinstance(ax, vs3.plt.Axes)):
                ax = fig.add_subplot(*layoutAdding)
            if(not drawContinuousDataMeasureVariance(np_y_fact_avg, np_y_pred_avg, tol=tol, ax=ax, figsize=figsize, 
                                                     export=LOGger.mylist(export.values()).get(0), stamps=stamps, **kwags)):
                return False
            return True
        else:
            stamps = stamps if(isinstance(stamps, list)) else []
            outer_grid = gridspec.GridSpec(2, 1, figure=fig)  # 整个大图
            axMain = fig.add_subplot(outer_grid[0])
            totalDrawingSuccess = drawContinuousDataMeasureVariance(
                np_y_fact_avg, np_y_pred_avg, tol=tol, export=export.get('main'), ax=axMain, figsize=figsize, stamps=stamps, **kwags)
            # 在大图中细分为 3x3 的子图
            dim = y_fact.shape[1]
            q = int(np.sqrt(dim)) + 1
            columns = LOGger.mylist(getattr(y_fact,'columns',tuple(map(DFP.parse, np.arange(y_fact.shape[1])))))
            inner_grid = gridspec.GridSpecFromSubplotSpec(q, q, subplot_spec=outer_grid[1])
            k = 0
            subDrawingSuccess = {}
            for i in range(q):
                for j in range(q):
                    if(k>=np_y_fact.shape[1]):
                        break
                    axBi = fig.add_subplot(inner_grid[i, j])
                    try:
                        subDrawingSuccess[k] = drawContinuousDataMeasureVariance(
                            np_y_fact[:,k], np_y_pred[:,k], tol=LOGger.extract(tol, index=k), 
                            export=export.get(columns.get(k,k)), ax=axBi, stamps=stamps+[columns.get(k,k)], **kwags)
                        axBi.set_title(columns.get(k,k))
                    except Exception as e:
                        LOGger.exception_process(e,logfile='', stamps=[drawContinuousDataMeasureVariance.__name__, columns.get(k,k)])
                    k += 1
            suptitle = LOGger.stamp_process('',stamps,'','','',' ')
            success = (bool(totalDrawingSuccess) * np.prod(list(map(bool, subDrawingSuccess.values())))) == 1
            fig.suptitle(suptitle)
            return success
    scatter_color = kwags.get('scatter_color', m_dataColor)
    bundary_color = kwags.get('bundary_color', m_bundaryColor)
    exportPlot = parseMeasureExport(export)
    label = kwags.get('label', LOGger.stamp_process('',exportPlot,digit=4))
    if(not isinstance(ax, vs3.plt.Axes)):
        ax = fig.add_subplot(*layoutAdding)
    ax.scatter(np_y_fact, np_y_pred, color=scatter_color, label=label)
    
    if(tol is np.nan or tol is None or (not callable(tol) and DFP.astype(tol, default=0)<=0)):
        tol = np.std(np_y_fact.reshape(-1))
    amin = np.min(np_y_fact)
    amax = np.max(np_y_fact)
    if(not callable(tol)):
        ax.plot(np.linspace(amin, amax, 100), np.linspace(amin, amax, 100), ls='--', color=bundary_color)
        ax.plot(np.linspace(amin, amax, 100)+tol, np.linspace(amin, amax, 100), ls='--', color=bundary_color)
        ax.plot(np.linspace(amin, amax, 100)-tol, np.linspace(amin, amax, 100), ls='--', color=bundary_color)
    else:
        ax.plot(np.linspace(amin, amax, 100), np.linspace(amin, amax, 100), ls='--', color=bundary_color)
    title = LOGger.stamp_process('',stamps,'','','',' ')
    ax.set_xlabel('fact')
    ax.set_ylabel('pred')
    ax.set_title(title)
    return True

def continuousDataMeasureVarianceAndDraw(y_fact=np.random.random(10,), np_y_pred=np.random.random(10,), 
                                         additional_norms=None, tol=np.nan, handler=None, fig=None, ax=None, stamps=None, cell_size=None, 
                                         scatter_color=m_dataColor, **kwags):
    data_prop = handler if(isinstance(handler, LOGger.mystr)) else LOGger.mystr()
    if(not continuousDataMeasureVariance(y_fact, np_y_pred, additional_norms=additional_norms, tol=tol, handler=data_prop, 
                                         stamps=stamps, cell_size=cell_size, **kwags)):
        return False
    print('data_prop.export.keys()', data_prop.export.keys())
    return drawContinuousDataMeasureVariance(y_fact, np_y_pred, export=data_prop.export, fig=fig, ax=ax, tol=tol, 
                                             stamps=stamps, cell_size=cell_size, scatter_color=scatter_color)

def continuousDataMeasureVarianceAndDraw4HZ(headerZone, y_fact=np.random.random(10,), np_y_pred=np.random.random(10,), 
                                            additional_norms=None, tol=np.nan, handler=None, fig=None, ax=None, **kwags):
    data_prop = getattr(headerZone, 'data_prop', handler if(isinstance(handler, LOGger.mystr)) else LOGger.mystr())
    cell_size = getattr(headerZone, 'cell_size', None)
    stamps = getattr(headerZone, 'stamps', [])
    return continuousDataMeasureVarianceAndDraw(y_fact, np_y_pred, additional_norms=additional_norms, tol=tol, 
                                                handler=data_prop, fig=fig, ax=ax, stamps=stamps, cell_size=cell_size, **kwags)

def continuousDataMeasureVarianceAndPlot(y_fact=np.random.random(10,), np_y_pred=np.random.random(10,), 
                                         additional_norms=None, tol=np.nan, handler=None, fig=None, stamps=None, cell_size=None, 
                                         exp_fd='.', file=None, figsize=(10,7), **kwags):
    DMVed = continuousDataMeasureVarianceAndDraw(y_fact=y_fact, np_y_pred=np_y_pred, additional_norms=additional_norms, tol=tol, 
                                                      handler=handler, fig=fig, stamps=stamps, cell_size=cell_size, figsize=figsize, **kwags)
    stamps = stamps if(isinstance(stamps, list)) else []
    if(not plotFromDataMeasureVarianceEnd(
            DMVed, figSource=fig, file=file, stamps=stamps+['continuousDataMeasureVariance'], exp_fd=exp_fd, **kwags)):
        return False
    return True

def parseDeepCells(all_categories, dataSource, **kwags):
    return tuple(map((lambda x:', '.join(list(map(DFP.parse, x))) if(DFP.isiterable(x)) else x), 
                     (all_categories if(DFP.isiterable(all_categories)) else dataSource)))

def discreteDataMeasureVariance(y_fact=np.where((np.random.random(10)>0.5),'NG','OK'), 
                                np_y_pred=np.where((np.random.random(10)>0.5),'NG','OK'), 
                                all_categories=None, binary_inlier_code=1, f_beta=1, 
                                stamps=None, cell_size=None, ret=None, handler=None, **kwags):
    try:
        np_y_fact = np.array(getattr(y_fact, 'values', y_fact).tolist())
        np_y_pred = np.array(getattr(np_y_pred, 'values', np_y_pred).tolist())
    except Exception as e:
        LOGger.exception_process(e, logfile='')
        return False
    if(np_y_fact.shape!=np_y_pred.shape):
        LOGger.addlog('dataMeasureVariance ERR: np_y_fact.shape!=np_y_pred.shape', np_y_fact.shape, np_y_pred.shape, colora='\x1b[31m', logfile='', stamps=stamps)
        return False
    if(y_fact.shape[0]<2):
        print('data too poor!!! Count:%d'%y_fact.shape[0])
        return False
    stamps = stamps if(isinstance(stamps, list)) else []
    if(len(y_fact.shape)>1):
        try:
            # 多維度的意思，是各維度中有不同就視為不同
            np_y_fact_avg = DFP.transformGroupofCellsToFlatBatch(np_y_fact, cell_size=cell_size) if(
                DFP.isiterable(cell_size)) else DFP.transformCellsToFlatBatch(np_y_fact)
            np_y_pred_avg = DFP.transformGroupofCellsToFlatBatch(np_y_pred, cell_size=cell_size) if(
                DFP.isiterable(cell_size)) else DFP.transformCellsToFlatBatch(np_y_pred)
            np_y_fact_avg = np.apply_along_axis((lambda x:', '.join(list(map(DFP.parse, x)))), 1, np_y_fact_avg)
            y_fact_avg = pd.Series(np_y_fact_avg, index=getattr(y_fact,'index',np.arange(y_fact.shape[0])))

            np_y_pred_avg = np.apply_along_axis((lambda x:', '.join(list(map(DFP.parse, x)))), 1, np_y_pred_avg)
            all_categories = DFP.uniqueByIndex(parseDeepCells(all_categories, dataSource=np_y_fact_avg))
            if(isinstance(ret, dict)):
                ret['count'] = np_y_fact_avg.shape[0]
                ret['np_y_fact_avg'] = np_y_fact_avg
                ret['np_y_pred_avg'] = np_y_pred_avg
                ret['all_categories'] = all_categories
                ret['mapping'] = dict(zip(all_categories, tuple(range(np.array(all_categories).shape[0]))))
            if(not discreteDataMeasureVariance(y_fact_avg, np_y_pred_avg, ret=ret, all_categories=all_categories, 
                                               stamps=stamps, cell_size=cell_size, handler=handler, binary_inlier_code=binary_inlier_code)):
                return False
        except Exception as e:
            LOGger.exception_process(e,logfile='')
            m_debug.updateDump({'discreteDataMeasureVariance':{'np_y_fact':np_y_fact, 'np_y_pred':np_y_pred, 'all_categories':all_categories}})
            return False
        return True
    ret = ret if(isinstance(ret, dict)) else {}
    ret.update(sumup_confusion_matrix_rates(np_y_fact, np_y_pred, all_categories=all_categories, 
                                            f_beta=f_beta, binary_inlier_code=binary_inlier_code, stamps=stamps, **kwags))
    np_evaluation = np.array([np_y_fact, np_y_pred]).transpose()
    ret['count'] = np_y_fact.shape[0]
    ret['np_evaluation'] = np_evaluation
    ret['mask'] = (np_y_fact==np_y_pred)
    if(handler!=None):
        handler.eval_data = pd.DataFrame(np_evaluation, columns=list(map((lambda x:LOGger.stamp_process('',stamps+[x],'','','','_')), ['fct','prd'])),
                                        index=getattr(y_fact,'index',np.arange(y_fact.shape[0])))
        handler.export = dcp(ret)
    return True

def drawDiscreteDataConfusionMatrix(cfM=np.arange(4).reshape(2,2), all_categories=None, binary_inlier_code=1, f_beta=1, export=None, fig=None, ax=None, figsize=(6,4), stamps=None, 
                                    mask=None, cell_size=None, layoutAdding=(1,1,1), **kwags):
    if(not DFP.isiterable(all_categories)):    all_categories = LOGger.mylist(tuple(getattr(cfM, 'class_names', range(cfM.shape[0])), [0,1]))
    if(not isinstance(ax, vs3.plt.Axes)):
        if(not isinstance(fig, vs3.plt.Figure)):
            fig = vs3.plt.Figure(figsize=figsize)
            drawDiscreteDataConfusionMatrix(cfM, all_categories=all_categories, binary_inlier_code=binary_inlier_code, 
                                            f_beta=f_beta, export=export, fig=fig, stamps=stamps, mask=mask, cell_size=cell_size, 
                                            layoutAdding=layoutAdding, **kwags)
            return fig
        ax = fig.add_subplot(*layoutAdding)
    try:
        fig, ax = vs.matrix_dataframe(cfM[:10,:10], 
                                      index=tuple(map((lambda x:m_dictionary.get(x,x)),all_categories[:10])), 
                                      header=tuple(map((lambda x:m_dictionary.get(x,x)),all_categories[:10])), 
                                      headerhide=False, indexhide=False, ax=ax)
        title = LOGger.stamp_process('',{m_dictionary.get('inlier','inlier'):all_categories[binary_inlier_code]})
        ax.set_title(title)
    except Exception as e:
        addlog_ = LOGger.addloger(logfile='', colora='\x1b[31m')
        vs3.ax_errmsg(ax, e, stamps=[__file__, drawDiscreteDataConfusionMatrix.__name__])
        addlog_('cfM', cfM)
        addlog_('all_categories', all_categories)
    # export = export if(isinstance(export, dict)) else {}
    title = LOGger.stamp_process('',stamps,'','','',' ')
    ax.set_xlabel('pred')
    ax.yaxis.set_label_position("right")
    ylabel = ax.set_ylabel('fact')
    ylabel.set_rotation(-90)
    ax.set_title(title)
    return True

def drawDiscreteDataMeasureScore(cfM=np.arange(4).reshape(2,2), all_categories=None, export=None, fig=None, ax=None, layoutAdding=(1,1,1), stamps=None, title=None, 
                                 xrot=-45, binary_inlier_code=1, scoreNames=None, countingNames=None, figsize=(6,8), **kwags):
    all_categories = list(tuple((DFP.uniqueByIndex(all_categories)) if(
        np.array(all_categories).shape[0]>0 if(DFP.isiterable(all_categories)) else False)  else np.arange(cfM.shape[0])))
    class_size = len(all_categories)
    if(not isinstance(ax, vs3.plt.Axes)):
        if(not isinstance(fig, vs3.plt.Figure)):
            fig = vs3.plt.Figure(figsize=figsize)
            drawDiscreteDataMeasureScore(cfM, all_categories=all_categories, export=export, fig=fig, 
                                         layoutAdding=layoutAdding, stamps=stamps, title=title, 
                                         xrot=xrot, binary_inlier_code=binary_inlier_code, scoreNames=scoreNames, 
                                         countingNames=countingNames, **kwags)
            return fig
        ax = fig.add_subplot(*layoutAdding)
    stamps = stamps if(isinstance(stamps, list)) else []
    title = title if(LOGger.isinstance_not_empty(title, str)) else LOGger.stamp_process('',stamps,'','','',' ')
    export = export if(isinstance(export, dict)) else {}
    if(class_size>2):
        clf_eva_class = export['MultiClassesEvaluation']
        clf_eva_class_for_graph = clf_eva_class.copy()
        try:
            clf_eva_class_for_graph.set_index('main', inplace=True)
            scoreNames = scoreNames if(isinstance(scoreNames, list)) else ['rcl','pcs',r'$f_{1}$']
            clf_eva_class_for_graph[scoreNames].plot.bar(ax=ax, title=title, rot=xrot)
            
            # 设置每个条形的透明度
            alpha_array = kwags.get('alpha_array')
            alpha_array = alpha_array if(DFP.isiterable(alpha_array)) else clf_eva_class_for_graph['fact_counts'].values
            # 将 alpha_array 归一化到 (0, 1)
            normalized_alpha = alpha_array / np.max(alpha_array)
            for i, (bar_group, alpha) in enumerate(zip(ax.patches, np.tile(normalized_alpha, len(scoreNames)))):
                bar_group.set_alpha(alpha)
                if(i<np.array(alpha_array).shape[0]):
                    # 获取条形的顶部位置
                    x = bar_group.get_x() + bar_group.get_width() / 2
                    y = bar_group.get_height()
                    
                    # 显示 alpha_array 的值
                    ax.text(x, y, DFP.parse(alpha_array[i]), ha='center', va='bottom', fontsize=9)
        except Exception as e:
            print('title', title)
            vs3.ax_errmsg(ax, e, stamps=[drawDiscreteDataMeasureScore.__name__, 'MultiClasses'])
        title = LOGger.stamp_process('',{'n_categories':len(all_categories)})
    else:
        if(binary_inlier_code>=len(all_categories)):
            addlog('binary_inlier_code[%d]>len(all_categories)[%d]!!!!'%(binary_inlier_code, len(all_categories)))
            return False
        
        if('BinaryClassesEvaluation' in export):
            table = export['BinaryClassesEvaluation']
        else:
            table = sumup_confusion_rates(cfM, all_categories=all_categories, xrot=xrot,
                                          binary_inlier_code = binary_inlier_code, stamps=stamps, **kwags)['BinaryClassesEvaluation']
            export['BinaryClassesEvaluation'] = table
        table.index = table.index.map(lambda x:m_dictionary.get(x,x))
        table.columns = table.columns.map(lambda x:m_dictionary.get(x,x))
        try:
            vs.matrix_dataframe(table, title=title, headerhide=True, indexhide=True, ax=ax)
        except Exception as e:
            vs3.ax_errmsg(ax, e, stamps=[drawDiscreteDataMeasureScore.__name__, 'BinaryClasses'])
        title = LOGger.stamp_process('',{m_dictionary.get('inlier','inlier'):all_categories[binary_inlier_code]})
    ax.set_title(title)
    return True

def drawDiscreteDataMeasureVariance(cfM=np.arange(4).reshape(2,2), all_categories=None, binary_inlier_code=1, f_beta=1, export=None, 
                                    fig=None, figsize=(6,8), stamps=None, mask=None, cell_size=None, 
                                    layoutAdding=(1,1,1), max_class_size=20, xrot=-45, **kwags):
    all_categories = list(tuple((DFP.uniqueByIndex(all_categories)) if(
        np.array(all_categories).shape[0]>0 if(DFP.isiterable(all_categories)) else False)  else np.arange(cfM.shape[0])))
    class_size = len(all_categories)
    export = export if(isinstance(export, dict)) else {}
    stamps = stamps if(isinstance(stamps, list)) else []
    LOGger.addDebug(type(fig), id(fig))
    if(not isinstance(fig, vs3.plt.Figure)):
        fig = vs3.plt.Figure(figsize=figsize)
        if(not drawDiscreteDataMeasureVariance(cfM, all_categories=all_categories, 
                                               binary_inlier_code=binary_inlier_code, f_beta=f_beta, export=export, 
                                               fig=fig, stamps=stamps, mask=mask, cell_size=cell_size, layoutAdding=layoutAdding, 
                                               max_class_size=max_class_size, xrot=xrot, **kwags)):
            return None
        return fig
    else:
        fig.clf()
    outer_grid = gridspec.GridSpec(2, 1, height_ratios=[2,1], figure=fig)
    # 混淆矩陣
    axConfMatrix = fig.add_subplot(outer_grid[1])
    drawDiscreteDataConfusionMatrix(cfM, all_categories=all_categories, binary_inlier_code=binary_inlier_code, f_beta=f_beta, 
                                    export=export, ax=axConfMatrix, stamps=stamps+['cfM'], mask=mask, cell_size=cell_size, **kwags)
    # 混淆指標
    axMain = fig.add_subplot(outer_grid[0])
    drawDiscreteDataMeasureScore(cfM, all_categories=all_categories, export=export, ax=axMain, layoutAdding=layoutAdding, stamps=stamps+['score'], 
                                 binary_inlier_code=binary_inlier_code, f_beta=f_beta, xrot=xrot, **kwags)
    return True

def discreteDataMeasureVarianceAndDraw(y_fact=np.where((np.random.random(10)>0.5),'NG','OK'), 
                                       np_y_pred=np.where((np.random.random(10)>0.5),'NG','OK'), 
                                       all_categories=None, binary_inlier_code=1, f_beta=1, handler=None, 
                                       fig=None, figsize=(6,8), stamps=None, cell_size=None, layoutAdding=(1,1,1), ret=None, **kwags):
    data_prop = handler if(isinstance(handler, LOGger.mystr)) else LOGger.mystr()
    if(not isinstance(ret, dict)):  ret = {}
    if(not discreteDataMeasureVariance(y_fact, np_y_pred, all_categories=all_categories, binary_inlier_code=binary_inlier_code, f_beta=f_beta, 
                                       stamps=stamps, cell_size=cell_size, ret=ret, handler=data_prop, **kwags)):
        return False
    print('data_prop.export.keys()', data_prop.export.keys())
    return drawDiscreteDataMeasureVariance(ret['cfM'], all_categories=ret.get('all_categories', all_categories), 
                                          binary_inlier_code=binary_inlier_code, f_beta=f_beta, 
                                          export=data_prop.export, fig=fig, figsize=figsize,  
                                          stamps=stamps, cell_size=cell_size, layoutAdding=layoutAdding)

def discreteDataMeasureVarianceAndPlot4HZ(headerZone, 
                                          y_fact=np.where((np.random.random(10)>0.5),'NG','OK'), 
                                          np_y_pred=np.where((np.random.random(10)>0.5),'NG','OK'), 
                                          all_categories=None, binary_inlier_code=1, f_beta=1, handler=None, 
                                          fig=None, figsize=(6,8), stamps=None, cell_size=None, layoutAdding=(1,1,1), ret=None, **kwags):
    data_prop = getattr(headerZone, 'data_prop', handler if(isinstance(handler, LOGger.mystr)) else LOGger.mystr())
    cell_size = getattr(headerZone, 'cell_size', None)
    stamps = getattr(headerZone, 'stamps', [])
    return discreteDataMeasureVarianceAndDraw(y_fact, np_y_pred, all_categories=all_categories, binary_inlier_code=binary_inlier_code, f_beta=f_beta, 
                                             handler=data_prop, fig=fig, figsize=figsize, 
                                             stamps=stamps, cell_size=cell_size, layoutAdding=layoutAdding, ret=ret, **kwags)

def discreteDataMeasureVarianceAndPlot(y_fact=np.where((np.random.random(10)>0.5),'NG','OK'), 
                                       np_y_pred=np.where((np.random.random(10)>0.5),'NG','OK'), 
                                       all_categories=None, binary_inlier_code=1, f_beta=1, handler=None, 
                                       fig=None, figsize=(6,8), stamps=None, cell_size=None, layoutAdding=(1,1,1), ret=None, 
                                       exp_fd='.', file=None, **kwags):
    data_prop = handler if(isinstance(handler, LOGger.mystr)) else LOGger.mystr()
    DMVed = discreteDataMeasureVarianceAndDraw(y_fact, np_y_pred, all_categories=all_categories, 
                                              binary_inlier_code=binary_inlier_code, f_beta=f_beta, 
                                              handler=data_prop, fig=fig, stamps=stamps, cell_size=cell_size, 
                                              layoutAdding=layoutAdding, ret=ret, figsize=figsize, **kwags)
    stamps = stamps if(isinstance(stamps, list)) else []
    if(not plotFromDataMeasureVarianceEnd(
            DMVed, figSource=fig, file=file, stamps=stamps+['discreteDataMeasureVariance'], exp_fd=exp_fd, **kwags)):
        return False
    return True

def drawScores(ax, np_y_score=np.random.random(10), inlier_code=1, dataColor=(0,0,1,0.3),
               threshold=None, preprocessor=None, stamps=None, infrm=None, 
               tresholdYPosRatio=np.array([1,9])/10, axst=None, colorThreshold=(1,0,0,0.3), digit=4, **kwags):
    try:
        np_y_score = np_y_score.reshape(-1)
        infrm = infrm if(isinstance(infrm, dict)) else {}
        stamps = stamps if(isinstance(stamps, list)) else []
        vs3.normhist(np_y_score, ax, axst=axst, color=dataColor, stamps=stamps)
        if(threshold is not None):  
            infrm['th'] = threshold
            ax.axvline(threshold, ls='--', color=colorThreshold)
            ax.text(threshold, np.sum(tresholdYPosRatio*np.array(ax.get_ylim())), DFP.parse(threshold, digit=digit))
        title = LOGger.stamp_process('',['score distribution', LOGger.stamp_process('',infrm, digit=digit)],'','','',' ')
        ax.set_title(title)
        ax.set_ylabel('probability density')
        ax.set_xlabel('scores')
    except Exception as e:
        exception_process(e, logfile='', stamps=['MV.drawUnfamScores']+stamps)
        return False
    return True

def drawScoresWith1Dcauses(ax, np_x_data, np_y_score=np.random.random(10), inlier_code=1, dataColor=(0,0,1,0.3), threshold=None, 
                           stamps=None, infrm=None, tresholdXPosRatio=np.array([1,9])/10, colorThreshold=(1,0,0,0.3), digit=4, **kwags):
    try:
        np_x_data = np_x_data.reshape(-1)
        np_y_score = np_y_score.reshape(-1)
        infrm = infrm if(isinstance(infrm, dict)) else {}
        stamps = stamps if(isinstance(stamps, list)) else []
        stamp = LOGger.stamp_process('',stamps,'','','','_')
        infrmStg = LOGger.stamp_process('',infrm)
        stg = LOGger.stamp_process('',[stamp, infrmStg],'','','',':')
        ax.scatter(np_x_data, np_y_score, color=dataColor, s=kwags.get('s',20), label=stg)
        if(threshold is not None):  
            infrm['th'] = threshold
            ax.axhline(threshold, ls='--', color=colorThreshold)
            ax.text(np.sum(tresholdXPosRatio*np.array(ax.get_xlim())), threshold, DFP.parse(threshold, digit=digit))
        title = LOGger.stamp_process('',['score distribution', LOGger.stamp_process('',infrm, digit=digit)],'','','',' ')
        ax.set_title(title)
        ax.set_ylabel('scores')
        ax.set_xlabel(kwags.get('xlabel', '1d x'))
    except Exception as e:
        exception_process(e, logfile='', stamps=['MV.drawScoresWith1Dcauses']+stamps)
        return False
    return True

def drawBinaryClassesScores(ax, 
                            np_y_fact=np.where((np.random.random(10)>0.5),'NG','OK'), 
                            np_y_score=np.random.random(10)>0.5, 
                            inlier_code=1, threshold=None, preprocessor=None, stamps=None, infrm=None, 
                            tresholdYPosRatio=np.array([1,9])/10, axst=None, **kwags):
    try:
        infrm = infrm if(isinstance(infrm, dict)) else {}
        stamps = stamps if(isinstance(stamps, list)) else []
        try:
            select_mask = (np_y_fact+0)
        except:
            select_mask = dcp(np_y_fact)
        np_y_score = np_y_score.reshape(-1)
        np_inlier = np_y_score[np_y_fact==preprocessor.classes_[inlier_code]] if(preprocessor) else np_y_score[select_mask==inlier_code]
        np_outlier = np_y_score[np_y_fact==preprocessor.classes_[1-inlier_code]] if(preprocessor) else np_y_score[select_mask==1-inlier_code]
        inlier_header = (preprocessor.classes_[inlier_code] if(len(preprocessor.classes_)>inlier_code) else 'inlier') if(preprocessor) else 'inlier'
        outlier_header = (preprocessor.classes_[1-inlier_code] if(len(preprocessor.classes_)>1-inlier_code) else 'outlier') if(preprocessor) else 'outlier'
        # np_score = np.hstack([np_inlier, np_outlier])
        stamps = stamps if(isinstance(stamps, list)) else []
        
        vs3.normhist(np_inlier, ax, axst=axst, color=(0,0,1,0.3), stamps=[inlier_header])
        vs3.normhist(np_outlier, ax, axst=axst, color=(0,0.05,0.2,0.3), stamps=[outlier_header])
        
        if(threshold is not None):  
            infrm['th'] = threshold
            ax.axvline(threshold, ls='--')
            ax.text(threshold, np.sum(tresholdYPosRatio*np.array(ax.get_ylim())), DFP.parse(threshold, digit=4))
        title = LOGger.stamp_process('',['score distribution', LOGger.stamp_process('',infrm, digit=4)],'','','',' ')
        ax.set_title(title)
        ax.set_ylabel('')
    except Exception as e:
        exception_process(e, logfile='', stamps=['MV.drawUnfamScores']+stamps)
        return False
    return True

def plot_nn(layer, fig, r=1, c=3, headerlabels=None, scr_input=None, scr_output=None,
            edge_plank=(None,None,None,None,-0.45,None),stamps=None):
    stamps = stamps if(isinstance(stamps, list)) else []
    #W
    name = layer.name
    if(name.find('unfam')>-1):
        return False
    if(getattr(layer,'get_weights',None)==None):
        print(name,'no get_weights method')
        return False
    somethings = layer.get_weights()
    if(len(somethings)<2):
        print(name,'lack weights',str(somethings)[:1000])
        return False
    weights, biases = somethings[:2]
    title = LOGger.stamp_process('',[name,'weights'],'','','',' ')
    ax = fig.add_subplot(r,c,len(fig.axes)+1)
    
    ax.set_title(title)
    if(len(weights.shape)>2 or len(weights.shape)<1):
        ax.text(0,0,'len(weights.shape)=%s , len(weights.shape)=%s'%(str(weights.shape), str(weights.shape)))
    else:
        print(title, weights.shape)
        headerlabels = headerlabels if(isinstance(headerlabels, list)) else []
        if(len(headerlabels)>0 and len(headerlabels)==weights.shape[0]):
            ax.set_xticks(range(len(headerlabels)))
            ax.set_xticklabels(headerlabels, rotation=90)
        weights =  np.transpose(weights)
        if(scr_input):
            weights = scr_input.inverse_transform(weights)
        # # 在 Axes 对象上绘制热力图
        # cax = ax.imshow(weights, cmap='viridis', interpolation='nearest')
        # # 添加颜色条
        # cbar = fig.colorbar(cax, ax=ax)
        vs3.simphist(weights, ax=ax)
        
        
    
    #B
    ax = fig.add_subplot(r,c,len(fig.axes)+1)
    print(title,'biases', 'len(fig.axes)', len(fig.axes))
    ax.set_title(LOGger.stamp_process('',[name,'biases'],'','','',' '))
    if(len(biases.shape)>2 or len(biases.shape)<1):
        ax.text(0,0,'len(biases.shape)=%s , len(biases.shape)=%s'%(str(biases.shape), str(biases.shape)))
    else:
        print(title, biases.shape)
        if(scr_output):
            biases = scr_output.inverse_transform(biases)
        biases = biases.reshape(-1, 1) if(len(biases.shape)<2) else biases
        # 在 Axes 对象上绘制热力图
        cax = ax.imshow(biases, cmap='viridis', interpolation='nearest')
        # 添加颜色条
        cbar = fig.colorbar(cax, ax=ax)
        
    fig.subplots_adjust(*edge_plank)
    # fig.tight_layout()
    return True
        
def plotsave_nn(layer, r=1, c=3, headerlabels=None, scr_input=None, scr_output=None,
            edge_plank=(None,None,None,None,-0.45,None), file='', exp_fd='.', stamps=None, **kwags):
    stamps = stamps if(isinstance(stamps, list)) else []
    print(kwags)
    fig = vs2.plt.Figure(**kwags)
    if(not plot_nn(layer, fig, r=r, c=c, headerlabels=headerlabels, scr_input=scr_input, scr_output=scr_output,
            edge_plank=edge_plank, stamps=stamps)):
        return False
    file = os.path.join(exp_fd, '%s.jpg'%LOGger.stamp_process('',['nn_weight']+stamps,'','','','_',for_file=1))
    fig.tight_layout()
    fig.savefig(file)
    return True
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
def measureDrawMVScenario(mv, y_fact, np_y_pred, fig=None, ax=None, layoutAdding=(1,1,1), stamps=None, ret=None, **kwags):
    handler = mv.data_prop
    stamps=mv.stamps + (stamps if(isinstance(stamps, list)) else [])
    mainAttrs = dcp(mv.mainAttrs)
    mainAttrs.update({k:v for k,v in kwags.items() if k in mv.mainAttrs})
    if(not isinstance(ret, dict)):  ret = {}
    # LOGger.addDebug('measureDraw', str(y_fact))
    if(not mv.measure(y_fact, np_y_pred, handler=handler, stamps=stamps, ret=ret, **mainAttrs)):
        return False
    print('data_prop.export.keys()', handler.export.keys())
    drawAttrs = dcp(mv.drawAttrs)
    drawAttrs.update({k:mv.mainAttrs.get(k,v) for k,v in kwags.items() if k in mv.drawAttrs})
    if('np_y_score' in kwags):  drawAttrs.update({'np_y_score':kwags['np_y_score']})
    if('figsizeThresholdAnalysis' in kwags):  drawAttrs.update({'figsizeThresholdAnalysis':kwags['figsizeThresholdAnalysis']})
    fig = fig if(isinstance(fig, vs3.plt.Figure)) else mv.fig
    if(not mv.draw(y_fact, np_y_pred, export=handler.export, stamps=stamps, ax=ax, cfM=ret.get('cfM'), 
                     all_categories=ret.get('all_categories'), 
                     preprocessor=kwags.get('preprocessor'), 
                     **drawAttrs)):
        return False
    if('handler' in kwags): 
        kwags['handler'].eval_data = handler.eval_data
        kwags['handler'].export = handler.export
    return True

def savePlotMVScenario(mv, stamps=None, **kwags):
    stamps = stamps if(isinstance(stamps, list)) else mv.stamps
    fn = LOGger.stamp_process('',[mv.data_prop, 'measurament']+mv.stamps+stamps,'','','','_',for_file=True)
    exp_fd = kwags.get('exp_fd', getattr(mv,'exp_fd','.'))
    file = kwags.get('file', os.path.join(exp_fd, '%s.jpg'%fn))
    vs3.end(mv.fig, file=file)
    mv.fig.clf()
    return True

class abc_Measurament(abc.ABC):
    def __init__(self, exp_fd='.', stamps=None, figsize=None, data_prop=None, **kwags):
        self.data_prop = LOGger.mystr(data_prop if(LOGger.isinstance_not_empty(data_prop, str)) else 'continuous') if(not isinstance(data_prop, LOGger.mystr)) else data_prop
        self.stamps = stamps if(isinstance(stamps, list)) else []
        self.mainAttrs = {}
        self.drawAttrs = {}
        self.evalAttrs = {} #本來是想裝eval要釋出的性質，但好像不實用
        # LOGger.addDebug('abc_Measurament figsize', figsize)
        self.figsize = figsize
        self.fig = vs3.plt.Figure(figsize=self.figsize)
        # LOGger.addDebug(self.stamps, id(self.fig))
        self.exp_fd = exp_fd
    
    @abc.abstractmethod
    def measure(self, **kwags):
        pass
    
    @abc.abstractmethod
    def draw(self, **kwags):
        pass
    
    def export(self, **kwags):
        return getattr(self.data_prop, 'export', {})
    
    def measureDraw(self, y_fact, np_y_pred, fig=None, ax=None, layoutAdding=(1,1,1), stamps=None, ret=None, **kwags):
        if(not measureDrawMVScenario(self, y_fact, np_y_pred, fig=fig, ax=ax, 
                                     layoutAdding=layoutAdding, stamps=stamps, ret=ret, **kwags)):
            return False
        return True
    
    def measureDraw4HZ(self, headerZone, y_fact, np_y_pred, ax=None, stamps=None, **kwags):
        handler = kwags.get('handler', getattr(headerZone,'data_prop', getattr(self,'data_prop',LOGger.mystr())))
        cell_size = getattr(headerZone, 'cell_size', None)
        kwags['preprocessor'] = kwags.get('preprocessor', getattr(headerZone, 'preprocessor', None))
        if(not self.measureDraw(y_fact, np_y_pred, handler=handler, ax=ax, stamps=stamps, cell_size=cell_size, **kwags)):
            return False
        headerZone.data_prop.eval_data = handler.eval_data
        headerZone.data_prop.export = handler.export
        return True
    
    def savePlot(self, stamps=None, **kwags):
        if(not savePlotMVScenario(self, stamps=stamps, **kwags)):
            return False
        return True
    
    def updateMainAttrs(self, **kwags):
        self.mainAttrs.update({k:v for k,v in kwags.items() if k in self.mainAttrs})
        
    def updateDrawAttrs(self, **kwags):
        self.drawAttrs.update({k:v for k,v in kwags.items() if k in self.drawAttrs})
        
    def updateEvalAttrs(self, **kwags):
        self.evalAttrs.update({k:v for k,v in kwags.items() if k in self.drawAttrs})

class continuousMeasurament(abc_Measurament):
    def __init__(self, stamps=None, exp_fd='.', figsize=(15,20), tol=np.nan, additional_norms=None, data_prop='continuous', **kwags):
        super().__init__(stamps=stamps, exp_fd=exp_fd, data_prop=data_prop, figsize=figsize, **kwags)
        self.mainAttrs.update({'tol':tol, 'additional_norms':additional_norms, 'cell_size':None})
        self.drawAttrs.update({'tol':tol, 'additional_norms':additional_norms, 'cell_size':None})
        
    def measure(self, y_fact=np.random.random(10), np_y_pred=np.random.random(10), stamps=None, ret=None, **kwags):
        stamps=self.stamps + (stamps if(isinstance(stamps, list)) else [])
        attrs = {}
        attrs.update(self.drawAttrs)
        attrs.update(self.mainAttrs)
        attrs.update(kwags)
        return continuousDataMeasureVariance(y_fact, np_y_pred, stamps=stamps, ret=ret, **attrs)
    
    def draw(self, y_fact=np.random.random(10), np_y_pred=np.random.random(10), fig=None, ax=None, stamps=None, **kwags):
        stamps=self.stamps + (stamps if(isinstance(stamps, list)) else [])
        attrs = {}
        attrs.update(self.mainAttrs)
        attrs.update(kwags)
        fig = fig if(isinstance(fig, vs3.plt.Figure)) else self.fig
        return drawContinuousDataMeasureVariance(y_fact, np_y_pred, fig=fig, ax=ax, stamps=stamps, **attrs)
    
class discreteMeasurament(abc_Measurament):
    def __init__(self, stamps=None, all_categories=None, exp_fd='.', figsize=(6,8), binary_inlier_code=1, f_beta=1, 
                 xrot=-45, scoreNames=None, countingNames=None, max_class_size=20, data_prop='discrete', 
                 score_threshold=None, defaultClass='', defaultClassIndex=None, figsizeThresholdAnalysis=(12,15), **kwags):
        super().__init__(stamps=stamps, exp_fd=exp_fd, data_prop=data_prop, figsize=figsize, **kwags)
        self.mainAttrs.update({'all_categories':all_categories, 'binary_inlier_code':binary_inlier_code, 'f_beta':f_beta, 
                               'score_threshold':score_threshold, 'defaultClass':defaultClass, 'defaultClassIndex':defaultClassIndex})
        self.figsizeThresholdAnalysis = figsizeThresholdAnalysis
        self.figThresholdAnalysis = vs3.plt.Figure(figsize=self.figsizeThresholdAnalysis)
        self.drawAttrs.update({'xrot':xrot, 'scoreNames':scoreNames, 'countingNames':countingNames, 'max_class_size':max_class_size,
                               'binary_inlier_code':binary_inlier_code, 'f_beta':f_beta})
        
    def measure(self, y_fact=np.where(np.random.random(10)>0.5,1,0), np_y_pred=np.where(np.random.random(10)>0.5,1,0), stamps=None, **kwags):
        stamps=self.stamps + (stamps if(isinstance(stamps, list)) else [])
        attrs = {}
        attrs.update(self.mainAttrs)
        attrs.update(kwags)
        return discreteDataMeasureVariance(y_fact, np_y_pred, stamps=stamps, **attrs)
    
    def draw(self, y_fact=np.where(np.random.random(10)>0.5,1,0), np_y_pred=np.where(np.random.random(10)>0.5,1,0), 
             figThresholdAnalysis=None, fig=None, ax=None, cfM=None, stamps=None, preprocessor=None, np_y_score=None, **kwags):
        addlog_ = LOGger.addloger(logfile='')
        stamps=self.stamps + (stamps if(isinstance(stamps, list)) else [])
        if(not isinstance(cfM, np.ndarray)):
            addlog_('draw cfM error!!! %s'%DFP.parse(cfM), stamps=stamps, colora=LOGger.FAIL)
            return False
        if(len(cfM.shape)!=2):
            addlog_('draw cfM error!!! %s'%str(cfM.shape), stamps=stamps, colora=LOGger.FAIL)
            return False
        if(cfM.shape[0]<2):
            addlog_('draw cfM error!!! %s'%str(cfM.shape), stamps=stamps, colora=LOGger.FAIL)
            return False
        if(cfM.shape[0]!=cfM.shape[1]):
            addlog_('draw cfM error!!! %s'%str(cfM.shape), stamps=stamps, colora=LOGger.FAIL)
            return False
        attrs = {}
        attrs.update(self.drawAttrs)
        attrs.update(self.mainAttrs)
        attrs.update(kwags)
        fig = fig if(isinstance(fig, vs3.plt.Figure)) else self.fig
        if(not drawDiscreteDataMeasureVariance(cfM, fig=fig, stamps=stamps, **attrs)):
            return False
        
        if(DFP.isiterable(np_y_score)):
            y_fact_csped = preprocessor.transform(y_fact) if(hasattr(preprocessor, 'transform')) else y_fact
            score_threshold = attrs['score_threshold']
            score_threshold = DFP.astype(score_threshold, d_type=float) if(DFP.astype(score_threshold, d_type=float) is not None) else 0.5
            attrs['thresholds'] = [score_threshold]
            figThresholdAnalysis = figThresholdAnalysis if(isinstance(figThresholdAnalysis, vs3.plt.Figure)) else self.figThresholdAnalysis
            attrs['all_categories'] = kwags.get('all_categories', getattr(preprocessor, 'classes_', [0,1]))
            exp_fd = kwags.get('exp_fd', os.path.join(getattr(self,'exp_fd','.'), 'graph'))
            fn = LOGger.stamp_process('',[self.data_prop, 'measurament', 'thresholdAnalysis']+stamps,'','','','_',for_file=True)
            file = kwags.get('file', os.path.join(exp_fd, '%s.jpg'%fn))
            threshold_variation_analysis_target_threading(figThresholdAnalysis, y_fact_csped, np_y_score, 
                                                          stamps = stamps, thread_key = 'thread_key',
                                                          addlog=kwags.get('addlog', LOGger.addloger(logfile='')), file=file, **attrs)
        return True
    
    def drawDiscreteDataConfusionMatrix(self, cfM, export=None, fig=None, ax=None, figsize=(6,4), stamps=None, **kwags):
        stamps=self.stamps + (stamps if(isinstance(stamps, list)) else [])
        attrs = {}
        attrs.update(self.drawAttrs)
        attrs.update(self.mainAttrs)
        attrs.update(kwags)
        return drawDiscreteDataConfusionMatrix(cfM, export=export, fig=fig, figsize=figsize, stamps=stamps, **attrs)
    
    def drawDiscreteDataMeasureScore(self, cfM, export=None, fig=None, ax=None, stamps=None, figsize=(10,10), **kwags):
        stamps=self.stamps + (stamps if(isinstance(stamps, list)) else [])
        attrs = {}
        attrs.update(self.drawAttrs)
        attrs.update(self.mainAttrs)
        attrs.update(kwags)
        return drawDiscreteDataMeasureScore(cfM, export=export, fig=fig, figsize=figsize, stamps=stamps, **attrs)

class binaryMeasurament(discreteMeasurament):
    def __init__(self, all_categories=None, **kwags):
        all_categories = LOGger.mylist(tuple(all_categories[:2]) if(DFP.isiterable(all_categories)) else ['OK','NG'])
        super().__init__(all_categories=all_categories, **kwags)
    
class categoricalMeasurament(discreteMeasurament):
    def __init__(self, all_categories, **kwags):
        super().__init__(all_categories=all_categories, **kwags)
        self.mainAttrs.update({'class_base':list(tuple(np.eye(len(all_categories))))})
###############################################################################################################################################    
###############################################################################################################################################
###############################################################################################################################################
def projectInitial(exp_fd_default='.', **kwags):
    handler = LOGger.mystr()
    project_buffer = kwags.get('project_buffer', {})
    if(LOGger.isinstance_not_empty(project_buffer.get('exp_fd'), str)):
        handler.exp_fd = dcp(project_buffer['exp_fd'])
    else:
        handler.exp_fd = exp_fd_default
        project_buffer.update({'exp_fd': handler.exp_fd})
    if(os.path.exists(handler.exp_fd) and handler.exp_fd!='.'):
        LOGger.removefile(handler.exp_fd)
    if(not os.path.exists(handler.exp_fd)):
        LOGger.CreateContainer(handler.exp_fd)
    handler.logfile = os.path.join(handler.exp_fd, 'log.txt')
    handler.addlog = LOGger.addloger(logfile=handler.logfile)
    handler.exports = {}
    return handler

def p0(config_file, default_exp_fd='%s_p0'%(os.path.basename(__file__).split('.')[0]), **kwags):
    """
    

    Parameters
    ----------
    config_file : TYPE
        DESCRIPTION.
    default_exp_fd : TYPE, optional
        DESCRIPTION. The default is '%s_p0'%(os.path.basename(__file__).split('.')[0]).
    **kwags : TYPE
        DESCRIPTION.

    Returns
    -------
    bool
        DESCRIPTION.

    """
    handler = projectInitial(exp_fd_default=default_exp_fd, **kwags)
    handler.projectName = 'p0'
    project_buffer = kwags.get('project_buffer', {})
    project_buffer['exp_fd'] = handler.exp_fd
    try:
        print('config_file', config_file)
        if(not testDrawContinuousDataMeasureVariance(handler=handler)):
            return False
    except Exception as e:
        LOGger.exception_process(e, logfile='')
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
    args_history = {}
    args_history.update(LOGger.load_json(json_file) if(os.path.exists(json_file)) else {})
    
    parser = LOGger.myArgParser()
    parser.add_argument("-prmth", "--project_method_stg", type=str, help="指定程序方式\
                        (p0:testDrawContinuousDataMeasureVariance; \
                         ", default='p0')
    parser.add_argument("-o", "--exp_fd", type=str, help="結果的存儲方式", default=None)
    
    args = parser.parse_args()
    project_buffer = vars(args)
    LOGger.addlog('params:\n%s'%LOGger.stamp_process('',project_buffer,':','[',']','\n'), logfile='')
       
    project_method = method_activation(project_buffer.get('project_method_stg'))
    if(project_method==None):
        return
    if(not project_method(**project_buffer, project_buffer=project_buffer)):
        return
    exp_fd = args.exp_fd
    LOGger.save_json(project_buffer, os.path.join(exp_fd, json_file))

if(__name__=='__main__'):
    scenario()