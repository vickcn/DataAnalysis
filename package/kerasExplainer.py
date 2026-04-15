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
from . import visualization3 as vs3
vs = vs3.vs
vs2 = vs3.vs2
DFP = vs3.DFP
import matplotlib.gridspec as gridspec
LOGger = vs3.LOGger
from .LOGger import CreateContainer, CreateFile, addloger, show_vector
from .LOGger import stamp_process, exception_process, for_file_process, abspath, mylist, mystr
from .LOGger import load_json, save_json, flattern_list, type_string
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
m_debugFile = '%s.pkl'%LOGger.stamp_process('',[*(os.path.basename(__file__).split('.')[:-1]),'debug'],'','','','_')
m_debug = LOGger.mydict({})
m_debug.dump = lambda :DFP.joblib.dump(m_debug, m_debugFile)# 'debug.pkl'
m_dataColor = (5/255,80/255,220/255,0.5)
m_bundaryColor = (255/255,90/255,90/255)
m_lossColors = LOGger.mylist([m_dataColor, (0.2,0.1,0.8,0.5), (0.05,0.05,0.3)])
m_classificationScoreNames = ['rcl','pcs',r'$f_{1}$','fact_counts','pred_counts','fact_ratio']

class EXPLAINER(LOGger.mystr):
    def __init__(self, model, stamps=None):
        super().__init__()
        self.stamps = stamps if(isinstance(stamps, list)) else []
        self.model = model  # 初始化 model 屬性

    # def __str__(self):
    #     stamp = LOGger.stamp_process('',self.stamps,'','','','_')
    #     return stamp  # 返回 stamp 作為字符串表示
        
    
#%%
#name: '1th-layer'
def lastUniftLayerReport(ke, name):
    # last layer只有一個的時候有效
    model = ke.model
    layer = model.get_layer(name)
    
def create_Ab_grid(fig):
    outer_grid = gridspec.GridSpec(1, 2, width_ratios=[2,1], figure=fig)
    # 門檻分析曲線
    ax_A = fig.add_subplot(outer_grid[0])
    ax_b = fig.add_subplot(outer_grid[1])
    return ax_A, ax_b
    
#weightLabels = get_weightLabelsFromHZs(mdc.xheader_zones)
def get_weightLabelsFromHZs(headerZones, defaultClassEmbeddingLatentDim=2):
    outputs = []
    for k,v in headerZones.items():
        if(v.preprocessing=='encoding'):
            latentDim = getattr(v,'classEmbeddingLatentDim',defaultClassEmbeddingLatentDim)
            v = [LOGger.stamp_process('',[x,i],'','','','_') for i in range(latentDim) for x in v]
        outputs += v
    return outputs

def drawWeightsBias(layer, fig=None, figsize=(10,10), barThickness=0.8, weightLabels=None, 
                    BiasXPosRatio=np.array([1,1])/2, BiasYPosRatio=np.array([1,1])/2, 
                    biasPreprocessor=None, **kwags):
    # Dense好像才可以
    w, b = tuple(layer.weights)
    wV, bV = w.numpy(), b.numpy()
    wV_flat = wV.reshape(-1)
    
    if(isinstance(fig, vs3.plt.Figure)):    
        fig.clf()
    else:
        fig = vs3.plt.Figure(figsize=figsize)
    
    ax_A, ax_b=create_Ab_grid(fig)
    
    
    vline = 0 if(0<=np.nanmax(wV_flat) and 0>=np.nanmin(wV_flat)) else np.nanmedian(wV_flat)
    colors = None
    if(np.unique(wV_flat).shape[0]>2):
        sigma = np.nanstd(wV_flat)
        colors=[(0,0,1,0.3) if(abs(x-vline)<sigma) else (0.8,0.8,0,0.6) for x in wV_flat]
    ax_A.barh(np.arange(wV_flat.shape[0]), wV_flat, label='weights', height=barThickness, color=colors)
    ax_A.axvline(vline, ls='--', color=(0,0,0,0.3))
    ax_A.axvline(vline+sigma, ls='--', color=(1,0,0,0.3), label=r'$+\sigma$')
    ax_A.axvline(vline-sigma, ls='--', color=(1,0,0,0.3), label=r'$-\sigma$')
    if(DFP.isiterable(weightLabels)):
        if(np.array(weightLabels).shape[0]==wV_flat.shape[0]):
            ax_A.set_yticks(np.arange(wV_flat.shape[0]), weightLabels)
    else:
        ax_A.set_yticks(np.arange(wV_flat.shape[0]), np.arange(wV_flat.shape[0]))
    
    ax_A.set_ylabel('index')
    ax_A.set_xlabel('weights')
    
    if(hasattr(biasPreprocessor, 'inverse_transform')):
        bV = biasPreprocessor.inverse_transform(bV.reshape(1,1))
    ax_b.imshow(bV.reshape(1,1), cmap='coolwarm', aspect='auto')
    ax_b.text(np.sum(BiasXPosRatio*np.array(ax_b.get_xlim())), np.sum(BiasYPosRatio*np.array(ax_b.get_ylim())), 
              'bias:%s'%(DFP.parse(bV.reshape(-1)[0], digit=4)), color=(1,1,1))
    ax_b.tick_params(axis='y', left=False, right=False, labelleft=False, labelright=False)
    ax_b.tick_params(axis='x', left=False, right=False, labelleft=False, labelright=False)
    fig.suptitle(LOGger.stamp_process('',['drawWeightsBias', layer.name],'','','',' '))
    
def plotWeightsBias(layer, file=None, fn='saveWeightsBias', exp_fd='.', fig=None, figsize=(10,10), barThickness=0.8, weightLabels=None, 
                    BiasXPosRatio=np.array([1,1])/2, BiasYPosRatio=np.array([1,1])/2, **kwags):
    if(isinstance(fig, vs3.plt.Figure)):    
        fig.clf()
    else:
        fig = vs3.plt.Figure(figsize=figsize)
    drawWeightsBias(layer, fig=fig, figsize=figsize, barThickness=barThickness, weightLabels=weightLabels, 
                    BiasXPosRatio=BiasXPosRatio, BiasYPosRatio=BiasYPosRatio, **kwags)
    file = file if(LOGger.isinstance_not_empty(file,str)) else os.path.join(exp_fd, '%s.jpg'%fn)
    LOGger.CreateFile(file, lambda f:vs3.end(fig, file=f))
        
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