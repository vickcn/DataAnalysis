# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 16:22:44 2021

@author: ian.ko
"""
import platform
from package import visualization2 as vs2
vs = vs2.vs
DFP = vs.DFP
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
#from matplotlib import cm
import scipy
import pandas as pd
import matplotlib.image as mpimg # mpimg 用於讀取圖片
from matplotlib import font_manager as fm #圖片字型
from copy import copy as cp
from copy import deepcopy as dcp
import seaborn as sns
import joblib
import random as rdm
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d
import mpl_toolkits
from datetime import datetime as dt
from sklearn.preprocessing import MinMaxScaler as Mmr
import matplotlib.dates as mdates
import threading
from multiprocessing import Event
import queue
import time
import matplotlib.lines as mlines
customization_curve_fit = DFP.customization_curve_fit
m_winsound_import_succeed=False
try:
    import winsound
    m_winsound_import_succeed = True
except:
    print('import winsound failed!!!')
LOGger = DFP.LOGger
from package.LOGger import exception_process, stamp_process, CreateFile, CreateContainer, ConfigAgent, instances_method_process, myFileRemover
from package.LOGger import mylist, mystr, type_string, is_timestamp, isinstance_not_empty, addloger, execute, save, gatelog, make_hyperlink, removefile
cv2 = LOGger.mystr('cv2')
cv2.CHAIN_APPROX_NONE = None
cv2.RETR_EXTERNA = None
cv2.MARKER_CROSS = None
cv2.RETR_EXTERNAL = None
m_cv2_import_succeed=False
try:
    import cv2
    m_cv2_import_succeed = True
except:
    print('import cv2 failed!!!')
#%%
showlog = 5
empty_print = 0
edge_plank=(0,0,1,1)
default_axis_values=0
default_range=(0,1)

m_figsize=(15,8)
file=''
title=''
title_fontsize = 20
mode='d'
ctr_tol=0, 
xtkrot=0, 
is_key_label=True

fontfile = mystr('/usr/share/fonts/truetype/test/msjh.ttc').path_sep_correcting()
fm.fontManager.addfont(fontfile) if(os.path.exists(fontfile)) else None
common_method_inputs = {
        'figsize':m_figsize, 'file':file, 'title':title,
        'title_fontsize':title_fontsize, 'mode':mode, 
        'ctr_tol':ctr_tol, 'xtkrot':xtkrot}
m_plt_Figure_keywords=['figsize','dpi','facecolor','edgecolor','linewidth','frameon']
m_myFigure_keywords=['figsize','dpi','facecolor','edgecolor','linewidth','frameon']

paint_keys = ['file', 'title', 'edge_plank', 'mode', 'figsize', 'cook', 'layout',
              'default_range', 'default_axis_value','linewidth']
print_keys = ['sep', 'print_sep', 'end', 'flush']
basic_keys = ['label', 'color', 'ls']
Line2D_keys = ['label', 'color', 'ls']
implot_keys = ['cmap', 'extent', 'make_cmap_nodes_', 'make_cmap_colors_']
implot_core_keys = ['cmap', 'extent']
curveplot_keys = basic_keys
scatterplot_keys = basic_keys + ['s','cmap','c', 'norm','marker']
nrmdsplot_keys = basic_keys + ['analysis_alpha', 'histtype', 'density', 'stacked']
nrmdsplot_core_keys = basic_keys + ['histtype', 'density', 'stacked']
vline_keys = ['vline', 'ymin', 'ymax', 'alpha']
vline_keys += ['vline_%s'%v for v in vline_keys+ basic_keys if v!='vline']
xticks_keys = ['xticks']
yticks_keys = ['yticks']
xticklabel_keys = ['xticklabel', 'rotation', 'fontsize', 'fontname', 'fontproperties']
xticklabel_keys += ['xticklabel_%s'%v for v in xticklabel_keys if v!='xticklabel']
yticklabel_keys = ['yticklabel', 'rotation', 'fontsize', 'fontname', 'fontproperties']
yticklabel_keys += ['yticklabel_%s'%v for v in xticklabel_keys if v!='yticklabel']
xlabel_keys = ['xlabel', 'rotation']
xlabel_keys += ['xlabel_%s'%v for v in xticklabel_keys if v!='xlabel']
ylabel_keys = ['ylabel', 'rotation']
ylabel_keys += ['ylabel_%s'%v for v in xticklabel_keys if v!='ylabel']
annotation_keys = ['xytext', 'xycoords']
axtitle_keys = ['axtitle_', 'axtitle_fontsize', 'axtitle_fontname']
legend_keys = ['prop']
suptitle_keys = ['font', 'fontsize', 'fontname', 'fontproperties']
textplot_keys = ['font', 'fontsize', 'fontname', 'fontproperties']
m_markers = ['o','v','^','<','>','s','D','d','p','h','H','8','P','X','+','x']
#%%
m_uniq_thru_set=lambda tensor, **kwags:sorted(list(set(tensor)), **kwags)
m_uniq_thru_np=lambda tensor, **kwags:np.unique(tensor, **kwags)
m_print = LOGger.addloger(logfile='')
label_callback_key='labels'

class plotPreset:
    def __init__(self, figsize=m_figsize, exp_fd='.', fn='fn', file=None, stamps=None):
        self.figsize = figsize
        self.exp_fd = exp_fd
        self.fn = fn
        self.file = file
        self.stamps = stamps if(isinstance(stamps, list)) else []
        
m_pltPre = plotPreset()

def loadc_from_pkl(pklfile, configfile=None, ret=None, **kwags):
    fig, config = None, LOGger.mystr()
    try:
        ret = ret if(isinstance(ret, dict)) else {}
        _fig = joblib.load(pklfile)
        print('axes', len(_fig.axes))
        configfile = configfile if(configfile!='') else '%s_fc.pkl'%('.'.join(pklfile.split('.')[:-1]))
        if(isinstance(configfile, str)):
            config = joblib.load(configfile)
            for k in m_plt_Figure_keywords:
                kwags.get(k, getattr(config, k, None))
        fig = myFig(_fig, **kwags)
        ret['fig'] = fig
        ret['config'] = config
    except Exception as e:
        LOGger.exception_process(e,logfile='')
        return False
    return True

def savec_as_pkl(fig, stamps, config=LOGger.mystr(), configfile=None, axsfile=None):
    try:
        if(type(fig)==myFig):
            for k in m_plt_Figure_keywords:
                setattr(config, k, getattr(fig, k, None))
        fn = LOGger.stamp_process('',stamps,'','','','_')
        _fig = getattr(fig,'fig',fig)
        joblib.dump(_fig, '%s.pkl'%fn)
        axs = {}
        for ax in getattr(fig,'myaxes',[]):
            key = LOGger.stamp_process('',[ax.get_subplotspec().rowspan.start, ax.get_subplotspec().colspan.start],'','','','-')
            axs[key] = ax.relations
        axsfile = axsfile if(LOGger.isinstance_not_empty(axsfile, str)) else '%s_axs.pkl'%fn
        joblib.dump(axs, axsfile)
        configfile = configfile if(LOGger.isinstance_not_empty(configfile, str)) else '%s_fc.pkl'%fn
        joblib.dump(config, configfile)
    except Exception as e:
        LOGger.exception_process(e,logfile='')
        return False
    return True

def method_activation(stg):
    try:
        method = eval(stg)
        return method
    except:
        print('method invalid:%s!!!!'%stg)
        return None

def MJHfontprop():
    if(platform.system().lower()=='windows'):
        prop = fm.FontProperties(family='Microsoft JhengHei')
    else:
        fontfile = mystr("/usr/share/fonts/truetype/test/msjh.ttc").path_sep_correcting()
        if(not os.path.exists(fontfile)):
            prop = fm.FontProperties(family='DejaVu Sans')
        prop = fm.FontProperties(fname=fontfile)
    return prop
#TODO:printer
def printer(*logs, common_log='', anno_sep=':', **kwags):
    kwags['sep'] = kwags['print_sep'] if('print_sep' in kwags) else (
            kwags['sep'] if('sep' in kwags) else ' ')
    showlevel = kwags['showlevel'] if('showlevel' in kwags) else 1
    if(showlog>=showlevel):
        logs = logs if(empty_print) else tuple([log for log in logs if log!=''])
        logs = ('%s%s%s'%(common_log, anno_sep, v) for v in logs) if(common_log) else logs
        if('emphasis' in kwags):
            emphasis = (kwags['emphasis'])*40
            print(emphasis)
        print(*logs, **{k:kwags[k] for k in kwags if k in print_keys})
        if('emphasis' in kwags):
            print(emphasis)
#TODO:data_infrm
def data_infrm(x, operate_in='', name='', row=0, max_showchar=200):
    printer('%s%s%sshape:%s'%('[%s]'%operate_in if(operate_in) else '', 
                              '[%d]'%row if(row) else '', 
                              ' %s '%name if(name) else '',
                              str(np.array(x).shape)))
    printer('%s%s%s:%s'%('[%s]'%operate_in if(operate_in) else '', 
                              '[%d]'%row if(row) else '', 
                              ' %s '%name if(name) else '',
                              str(np.array(x))[:max_showchar]),
            showlevel=2)
#TODO:showvariable
def showvariable(**kwags):
    printer(*['%s:%s'%(v_name, kwags[v_name]) for v_name in kwags])
#TODO:isnotaxis
def isnotaxis(*args, nonempty=1, same_size=1):
    if(args==()):
        return True
    common_size = None
    for i, arg in enumerate(args):
        np_arg = dcp(np.array(arg))
        printer('[isnotaxis][%d]np_arg:\n%s'%(i, str(arg)[:200]), showlevel=4)
        if(str(np_arg.dtype).find('datetime')>-1):
            return False
        if(str(np_arg.dtype).find('int')==-1 and str(np_arg.dtype).find('float')==-1):
            data_infrm(arg, operate_in='isaxis:not number', name='%d'%i)
            return True
        #dtype顯示元素皆純數字的情況下，shape=()表示純數字!!
        if((np_arg.shape[0]==0 if(len(np_arg.shape)>0) else False) if(nonempty) else False):
            data_infrm(arg, operate_in='isaxis:empty', name='%d'%i)
            return True
        if(np.array(arg).shape!=common_size if(common_size!=None) else False):
            data_infrm(arg, operate_in='isaxis:inconsistent with common size %s'%common_size, 
                       name='%d'%i)
            return True
        common_size = dcp(np_arg.shape) if(common_size==None) else common_size
    return False

############################################################################################
#TODO:scenario
############################################################################################
def astype(subject, d_type=int, default=None):
    try:
        new_subject = d_type(subject)
    except:
        new_subject = default if(default!='same') else subject
    return new_subject
#TODO:get_return
def get_return(method=np.min, *args, default_value=None, **kwags):
    ret = default_value
    try:
        ret = method(*args, **kwags) if(type(method(*args, **kwags))==dict) else [
                *LOGger.get_all_values(method(*args, **kwags), decompose_layer_counts=1, only_numbers=0)]
    except:
        pass
    return (tuple(ret) if(np.array(ret).shape[0]>1) else ret[0]) if(LOGger.isiterable(ret)) else ret
#TODO:plot_set_figszie
def plot_set_figsize(all_xvalues=[], all_yvalues=[], figsize=None, 
                     plot_set_figsize_on=False, **kwags):
    if(not plot_set_figsize_on):
        return figsize
    printer('[plot_set_figsize] assigned figure size=%s'%str(figsize), showlevel=2)
    if(figsize!=None):
        return figsize
    mxalue_count = max(len(all_xvalues), 5)
    myalue_count = max(len(all_yvalues), 1)
    xtkrot = kwags.get('xtkrot', 0)
    friendly_height = (2 + min(myalue_count*0.5,4) + 7*np.sin(xtkrot))
    return ((min(mxalue_count*0.2,6)+4.5, friendly_height) if(
            figsize==None) else figsize)
#TODO:plot_set_axis_range
def plot_set_axis_range(ax, *args, xalue_lim=None, yalue_lim=None, vlines=[], 
                        cook='xy', default_axis_value=0, default_range=(1,0), 
                        plot_set_axis_range_on=True, **kwags):
    if(not plot_set_axis_range_on):
        return {}
    vlines = [float(vlines[k]['value']) for k in vlines] if(
            type(vlines)==dict) else vlines
    if(cook=='xy'):
        do_args = [list(set(LOGger.get_all_values(v))) for v in args[:2]] #X, Y內容分開算
        #資料內容在幾行之後
    elif(cook in ['a','x','y']):
        do_args = [list(set(LOGger.get_all_values(args)))]*2 #所有數據算在一起
        #資料內容在幾行之後
    else:
        {}['cook error:%s'%cook]
    printer('[plot_set_axis_range] cook:%s'%cook, showlevel=2)
    
    # if(isnotaxis(do_args[0])):
    #     printer('[axis:x][plot_set_axis_range] not axis...', showlevel=3)
    #     {}['plot_set_axis_range stop']
    # printer('[plot_set_axis_range]x values:%s'%str(do_args[0])[:200], showlevel=4)
    X = LOGger.get_all_values(do_args[0]) + LOGger.get_all_values(vlines) if(not isnotaxis(do_args[0])) else []
    
    # if(isnotaxis(do_args[1])):
    #     printer('[axis:y][plot_set_axis_range] not axis...', showlevel=3)
    #     {}['plot_set_axis_range stop']
    # printer('[plot_set_axis_range]y values:%s'%str(do_args[1])[:200], showlevel=4)
    Y = LOGger.get_all_values(do_args[1]) if(not isnotaxis(do_args[1])) else []
    xmax, xmin, ymax, ymin = get_return(np.amax, X, default_value = None), get_return(
                                        np.amin, X, default_value = None), get_return(
                                        np.amax, Y, default_value = None), get_return(
                                                np.amin, Y, default_value = None)
    printer('(xmax, xmin, ymax, ymin)=%s'%(','.join(list(map(lambda v:DFP.parse(v, digit=3), [xmax, xmin, ymax, ymin])))), showlevel=3)
    amax, amin = get_return(max, xmax, ymax, default_value = None), get_return(min, xmin, ymin, default_value = None)
    printer('(amax, amin)=%s'%(','.join(list(map(lambda v:DFP.parse(v, digit=3), [amax, amin])))), showlevel=4)
    hx, hy = 0, 0
    if(not None in [xmax, xmin]):
        hx=(xmax - xmin)/10
        xalue_lim = xalue_lim if(xalue_lim!=None) else (xmin-hx, xmax+hx)
        ax.set_xlim(*xalue_lim)
        printer('xrange:(%.2f, %.2f) with h=%.2f'%(xmin, xmax, hx), showlevel=2)
    if(not None in [ymax, ymin]):
        hy=(ymax - ymin)/10
        yalue_lim = yalue_lim if(yalue_lim!=None) else (ymin-hy, ymax+hy)
        ax.set_ylim(*yalue_lim)
        printer('yrange:(%.2f, %.2f) with h=%.2f'%(ymin, ymax, hy), showlevel=2)
    return {'xymxn':(xmin, xmax, ymin, ymax), 
                'amxn':(amin, amax), 'h':(hx, hy)}

def textInfrms(ax, infrm, xStart=0, nItemsInALine=6, ItemStgUbd=20, n_lineLbd=15, textAttr=None, digit=4, **kwags):
    if(infrm is None):
        return True
    infrmSimple = {k:v for k,v in infrm.items() if(not (isinstance(v, dict) or DFP.isiterable(v) or hasattr(v, 'shape')))}
    textAttr = textAttr if(isinstance(textAttr, dict)) else {}
    n_item = len(infrmSimple)
    n_line = int(n_item/nItemsInALine) + 1
    n_line = max(n_line, n_lineLbd)
    yL, yU  = ax.get_ylim()
    if(isinstance(infrmSimple, dict)):
        infrmKeys = list(infrmSimple.keys()) 
    elif(isinstance(infrmSimple, list)):
        infrmKeys = LOGger.mylist(infrmSimple)
    else:
        print('invalid infrm format:%s'%type(infrmSimple))
        return False
    for i,y in enumerate(np.linspace(yL, yU, n_line)):
        infrmTemp = dcp({k:infrmSimple[k] for k in infrmKeys[nItemsInALine*i:nItemsInALine*(i+1)]} if(
            isinstance(infrmSimple, dict)) else infrmKeys[nItemsInALine*i:nItemsInALine*(i+1)])
        if(not infrmTemp):
            continue
        stgTemp = dcp(LOGger.stamp_process('',infrmTemp,max_len=ItemStgUbd, digit=digit))
        ax.text(xStart, y,stgTemp, **textAttr)
    return True

def texts(ax, stg, xStart=0, lengthALineUbd=40, n_lineLbd=15, textAttr=None, **kwags):
    textAttr = textAttr if(isinstance(textAttr, dict)) else {}
    n_line = int(len(stg)/lengthALineUbd) + 1
    n_line = max(n_line, n_lineLbd)
    yL, yU  = ax.get_ylim()
    for i,y in enumerate(np.linspace(yL, yU, n_line)):
        stgTemp = dcp(stg[lengthALineUbd*i:lengthALineUbd*(i+1)])
        if(not stgTemp):
            continue
        ax.text(xStart, y, stgTemp, **textAttr)
    return True
        
#TODO:ax_msg
def ax_errmsg(ax, e, center=(1.2,2), xlim=(1,5), ylim=(1,5), msg_ubd=1000,
              family='Consolas', fontsize = 10, function_key='', newline_count=5, layoutpos=(1,1,1), fig=None, **kwags):
    pltinitial(fig, ax)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    msg = str(e)
    msglist = msg.split(' ')
    newmsg = ''
    for i in range(0, len(msglist), newline_count):
        a_stg = ' '.join(msglist[i:(i+newline_count)])
        newmsg += (a_stg + '\n')
    newmsg = '%s[line:%d]err:%s'%(
            ('[%s]'%function_key if(function_key) else ''), e.__traceback__.tb_lineno, newmsg[:-1])
    printer(newmsg, emphasis='=')
    kwags.update({'family':family, 'fontsize':fontsize})
    ax.text(*center, newmsg[:msg_ubd], **({k.replace('text_',''):v for k, v in kwags.items() if k in textplot_keys}))
    
def get_axes(fig, consider_twinxed=True):
    axes = mylist()
    if consider_twinxed:
        # print('get_axes', getattr(fig, 'myaxes', fig.axes))
        axes = mylist(getattr(fig, 'myaxes', fig.axes))
        return axes
    for ax in getattr(fig, 'myaxes', fig.axes):
        if ax._sharex != None:
            continue
        if ax._sharey != None:
            continue
        axes.append(ax)
    return axes

def get_frames(fig):
    return get_axes(fig, consider_twinxed=False)

def get_ylim(fig, ax):
    all_y_data = np.array([]).reshape(0,2)
    if(fig!=None):
        for ax in get_axes(fig):
            appending_y_data = get_ylim(None, ax).reshape(1,2)
            all_y_data = dcp(appending_y_data if(all_y_data.shape[0]==0) else np.append(all_y_data, appending_y_data))
    appending_y_data = np.array([np.nanmin([ax.get_ylim()[0], *(all_y_data[:,0])]), np.nanmax([ax.get_ylim()[1], *(all_y_data[:,1])])]).reshape(1,2)
    all_y_data = dcp(appending_y_data if(all_y_data.shape[0]==0) else np.append(all_y_data, appending_y_data))
    return np.array([np.nanmin(all_y_data[:,0]), np.nanmax(all_y_data[:,1])]).reshape(-1)

def get_ylim_margin(fig, ax, hratio=0.1):
    ylim = get_ylim(fig, ax)
    output = np.array([*ylim, abs(ylim[1] - ylim[0])*hratio])
    return output

def get_xlim(fig, ax):
    all_x_data = np.array([]).reshape(0,2)
    if(fig!=None):
        for ax in get_axes(fig):
            appending_x_data = get_xlim(None, ax).reshape(1,2)
            all_x_data = dcp(appending_x_data if(all_x_data.shape[0]==0) else np.append(all_x_data, appending_x_data))
    appending_x_data = np.array([np.nanmin([ax.get_xlim()[0], *(all_x_data[:,0])]), np.nanmax([ax.get_xlim()[1], *(all_x_data[:,1])])]).reshape(1,2)
    all_x_data = dcp(appending_x_data if(all_x_data.shape[0]==0) else np.append(all_x_data, appending_x_data))
    return np.array([np.nanmin(all_x_data[:,0]), np.nanmax(all_x_data[:,1])]).reshape(-1)

def get_xlim_margin(fig, ax, hratio=0.1):
    xlim = get_xlim(fig, ax)
    output = np.array([*xlim, abs(xlim[1] - xlim[0])*hratio])
    return output

def get_lim(fig, ax):
    all_data = np.array([]).reshape(0,2)
    if(fig!=None):
        for ax in get_axes(fig):
            appending_data = get_lim(None, ax).reshape(1,2)
            all_data = dcp(appending_data if(all_data.shape[0]==0) else np.append(all_data, appending_data))
    appending_data = np.array([np.nanmin([ax.get_xlim()[0], ax.get_ylim()[0], *all_data]), 
                               np.nanmax([ax.get_xlim()[1], ax.get_xlim()[1], *all_data])]).reshape(1,2)
    all_data = dcp(appending_data if(all_data.shape[0]==0) else np.append(all_data, appending_data))
    return np.array([np.nanmin(all_data[:,0]), np.nanmax(all_data[:,1])]).reshape(-1)

def get_lim_margin(fig, ax, hratio=0.1):
    lim = get_lim(fig, ax)
    output = np.array([*lim, abs(lim[1] - lim[0])*hratio])
    return output

def drawon(ax, relation, mCA=None, doStandardLabel=True, **kwags):
    """
    

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    relation : TYPE
        Could be iterable, then operate by terms.
    mColorAgent : myColorAgent
        colorPeriod, colorDefault, colorSelections, ...

    Returns
    -------
    TYPE
        0=all failed; 1=all sucess; other num represents successful ratio.

    """
    if(DFP.isiterable(relation)):
        if(len(relation)==0):
            return True
        if(not isinstance(mCA, myColorAgent)):  mCA = myColorAgent(period=kwags.get('colorPeriod', len(relation)), 
                                                                   default=kwags.get('colorDefault', None), 
                                                                   selections=kwags.get('colorSelections', None),
                                                                   alpha=kwags.get('colorAlpha', 0.3))
        success = 0
        for i,relation_i in enumerate(relation):
            # print('relation_i',relation_i)
            relation_i.kwags.update({'color':mCA.call(i)})
            if(doStandardLabel):
                relation_i.kwags.update({'label':relation_i.kwags.get('label', i)})
            print(relation_i.kwags)
            success = success + drawon(ax, relation_i)
        return success / len(relation)
    plot_method_stg = dcp(relation.plot_method_stg)
    # print(plot_method_stg)
    if(hasattr(ax, plot_method_stg)):
        plot_method_global = getattr(ax, plot_method_stg)
        plot_method = lambda *args, **kwags: plot_method_global(*args, **kwags)
        print('ax', plot_method_global, 'draw....')
    else:
        try:
            plot_method_global = eval(plot_method_stg)
            plot_method = lambda da, **kwags: plot_method_global(da, ax, **kwags)
            print(plot_method, 'draw....')
        except Exception as e:
            LOGger.exception_process(e, logfile='')
            return False
    return autoplot(plot_method, relation.export(), relation.dtype, relation.dshape, **relation.kwags)
            
def is_fig_completed(fig, layout=None, stamps=None, **kwags):
    layout = getattr(fig, 'layout', layout)
    stamps = stamps if(isinstance(stamps, list)) else []
    if(DFP.isiterable(layout)):
        n_axes = len(get_frames(fig))
        layout_product = np.product(layout[:2])
        if(n_axes>=layout_product):
            LOGger.addlog('full of axes!!![%s>=%s]'%(n_axes, layout_product), logfile='', stamps=stamps)
            return True
    return False

def end(fig, do_axdraw=False, file=None, exp_fd_default='.', fn_default='fn', save_type_default='jpg', title='', **kwags):
    if(fig):
        for ax in get_axes(fig):
            if(do_axdraw and isinstance(ax, myAxes)):
                ax.drawon()
            handles, labels = ax.get_legend_handles_labels()
            filtered_handles = [x for i,x in enumerate(handles) if LOGger.isinstance_not_empty(labels[i], str)]
            filtered_labels = [x for x in labels if LOGger.isinstance_not_empty(x, str)]
            # 只在有實際標籤時才輸出日誌
            if LOGger.isinstance_not_empty(filtered_labels,list):
                # LOGger.addDebug('filtered_labels', filtered_labels)
                ax.legend(filtered_handles, filtered_labels)
        fig.suptitle(title) if(LOGger.isinstance_not_empty(title, str)) else None
        fig.tight_layout()
    if(isinstance(file, str)):
        exp_fd = LOGger.execute('exp_fd', kwags, fig, default=exp_fd_default, not_found_alarm=False)
        fn = LOGger.execute('fn', kwags, fig, default=fn_default, not_found_alarm=False)
        save_type = LOGger.execute('save_type', kwags, fig, default=save_type_default, not_found_alarm=False)
        if(file==''):
            file = os.path.join(exp_fd, '%s.%s'%(fn, save_type))
        fig.savefig(file)
    # else:
    #     getattr(fig, 'fig', fig).show()

def autoplot(method, data, dtype, dshape, slice_select=slice(None, 2), show_kwags=False, **kwags):
    """
    

    Parameters
    ----------
    method : TYPE
        plot methof.
    data : TYPE
        len(data)==2 or dshape==(2,) ---> x-y correlation
        len(data)==1 ---> 1 block
        len(data)>2 ---> higher dim down to 2 dim
    dtype : TYPE
        DESCRIPTION.
    dshape : TYPE
        DESCRIPTION.
    slice_select : TYPE, optional
        DESCRIPTION. The default is slice(None, 2).
    show_kwags : TYPE, optional
        DESCRIPTION. The default is False.
    **kwags : TYPE
        DESCRIPTION.

    Returns
    -------
    bool
        DESCRIPTION.

    """
    try:
        if(issubclass(dtype, list) or issubclass(dtype, tuple)):
            if(len(data)==2 or dshape==(2,)):
                method(*data, **kwags)
            elif(len(data)==1):
                for num in data:
                    method(num, **kwags)
            elif(len(data)>2):
                method(*(data[slice_select]), **kwags)
        elif(issubclass(dtype, np.ndarray)):
            if(len(dshape)==2 and 2 in [*dshape]):
                columnChannelIndex = np.where(np.array(dshape)==2)[0][0]
                if(2 in [*dshape]):
                    method(*tuple(data), **kwags) if(columnChannelIndex==0) else method(*tuple(zip(*data)), **kwags)
                else:
                    for num in data:
                        method(num, **kwags)
            elif(len(dshape)==1):
                # print(method)
                method(data, **kwags)
                # for num in data:
                #     method(num, **kwags)
            else:
                return False
        elif(issubclass(dtype, pd.core.frame.DataFrame) or issubclass(dtype, pd.core.series.Series)):
            return autoplot(method, data.values, np.ndarray, dshape, slice_select=slice_select, show_kwags=show_kwags, **kwags)
        elif(dshape==None):
            for num in data:
                method(num, **kwags)
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=[autoplot.__name__])
        return False
    return True

def export_plt(core, slice_select=slice(None, 2)):
    '''
        for matplotlib not opencv!!!
    '''
    if(isinstance(LOGger.mylist(core).get(0), type(None))):
        return core
    dtype = type(core[0])
    dshape = np.array(core[0]).shape
    if(issubclass(dtype, list) or issubclass(dtype, tuple)):
        if(len(core)==2):
            return core
        elif(dshape==(2,)):
            # core裡面還是list, tuple，dshape又是(2,)，代表是平面點集
            return list(zip(*core))
        elif(len(core)==1):
            return core[0]
        elif(len(core)>2):
            return core[slice_select]        
    elif(issubclass(dtype, np.ndarray)):
        if(len(dshape)==2 and 2 in [*dshape]):
            if(dshape[0]==2):
                return list(tuple(core))
            elif(dshape[1]==2):
                return list(zip(*core))
            # 當成圖像
            return core[0]
        elif(len(dshape)==1):
            return core[0] #list(tuple(core))
        return None
    elif(issubclass(dtype, pd.core.frame.DataFrame) or issubclass(dtype, pd.core.series.Series)):
        print('core', type(core[0]))
        return core[0]
    elif(dshape==None):
        return core[0]

def subplots(r,c,a,stamps=None, **kwags):
    print({k:v for k,v in kwags.items() if k in m_plt_Figure_keywords})
    _fig = plt.Figure(**{k:v for k,v in kwags.items() if k in m_plt_Figure_keywords})
    fig = myFig(_fig)
    ax = fig.add_subplot(r,c,a,stamps=stamps)
    # print('ax', type(ax))
    return fig, ax

def set_aims(ax, aims, ls='--', color=None):
    if(not DFP.isiterable(aims)):
        return
    if(len(np.array(aims).shape)<1):
        return
    if(np.array(aims).shape[1]!=2):
        return
    aimsT = list(zip(*aims))
    ax.vlines(aimsT[0], -1, 2, ls=ls, color=color)
    ax.hlines(aimsT[1], 0, 100, ls=ls, color=color)

def myhist(data, ax, color=(0.1,0.05,0.4), alpha=0.5, num_bins=100, histtype = 'barstacked', density = True, base=None, scatter_size=5, 
         marker=None, label=None, **kwags):
    try:
        if(DFP.isiterable(base)):
            if(np.array(base).shape[0]!=data.shape[0]):
                base = None
        if(isinstance(base, type(None))):
            label = label.strip('_')
            ax.hist(data, num_bins, density=density, stacked=True, histtype=('bar' if(histtype==None) else histtype), color=tuple(list(color)[:3])+(alpha,),
                    label=label)
        else:
            ax.scatter(base, data, color=tuple(list(color)[:3])+(alpha,), s=scatter_size, marker=marker, label=label)
    except Exception as e:
        LOGger.exception_process(e,logfile='')
    
def regreiou(n_y_fact, n_y_pred, ax, colorFact=(0,0,1), colorPred=(0,1,0), linewidth=5, 
             alpha=0.3, axtitle='', **kwags):
    try:
        n = n_y_fact.shape[0]
        for i,x in enumerate(range(n)):
            ax.plot([x,x], [n_y_fact[x,0],n_y_fact[x,1]],color=colorFact,linewidth=linewidth, alpha=alpha,label='fact' if(i==0) else None)
            ax.plot([x,x], [n_y_pred[x,0],n_y_pred[x,1]],color=colorPred,linewidth=5, alpha=alpha,label='pred' if(i==0) else None)
        if(LOGger.isinstance_not_empty(axtitle, str)):
            ax.set_title(axtitle)
    except Exception as e:
        LOGger.exception_process(e,logfile='')
        
def regre(n_y_fact, n_y_pred, ax, tol=None, color=(5/255,80/255,220/255,0.5), linewidth=5, 
             alpha=0.3, label=None, axtitle='', scattersize=10,  bundary_color = (255/255,90/255,90/255), **kwags):
    try:
        all_values = np.hstack([n_y_fact, n_y_pred])
        amin, amax = np.min(all_values), np.max(all_values)
        ax.scatter(n_y_fact, n_y_pred, color=color, linewidth=linewidth, alpha=alpha, label=label, s=scattersize)
        ax.plot(np.linspace(amin, amax, 100), np.linspace(amin, amax, 100), ls='--', color=bundary_color)
        if(tol==None):
            tol = np.std(np.array(n_y_fact).reshape(-1))
        ax.plot(np.linspace(amin, amax, 100)+tol, np.linspace(amin, amax, 100), ls='--', color=bundary_color)
        ax.plot(np.linspace(amin, amax, 100)-tol, np.linspace(amin, amax, 100), ls='--', color=bundary_color)
        if(LOGger.isinstance_not_empty(axtitle, str)):
            ax.set_title(axtitle)
    except Exception as e:
        LOGger.exception_process(e,logfile='')
    
def add_data(data, ax, plot_method_stg, **kwags):
    """
    
    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.
    plot_method_stg : TYPE
        normhist, plot, scatter.
    **kwags : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    rel = myRelation(data, plot_method_stg=plot_method_stg, **kwags)
    if(not hasattr(rel, 'plot_method_stg')):
        return False
    ax.add(rel)    
    return True

def checkDataShape(data, criterion=(lambda x:len(np.array(x).shape)!=1)):
    if(not DFP.isiterable(data)):
        print('not iterable', str(data)[:1000])
        return False
    if(criterion!=None):
        if(criterion(data)):
            print('len(np.array(data).shape)', len(np.array(data).shape))
            return False
    return True

def simphist(data, ax, axst=None, infrm=None, diversity_lbd=2, stats_method=None, memo='', color=(0.1,0.05,0.4), alpha=0.5, 
             num_bins=100, histtype = 'barstacked', density = True, xmax=None, xmin=None, stamps=None, ret=None, 
             threshold_color=None, do_stats=True, n_sigma=3, base=None, yticks_st=None, scatter_size=5, marker=None, 
             do_consistency_validation=False, tol=None, **kwags):
    if(not checkDataShape(data, criterion=(lambda x:len(np.array(x).shape)!=1))):
        return False
    stamps = stamps if(isinstance(stamps, list) or isinstance(stamps, dict)) else []
    stamp = LOGger.stamp_process('',stamps)
    ret = ret if(isinstance(ret, dict)) else {}
    infrm = infrm if(isinstance(infrm, dict)) else {}
    infrm_plot = dcp(infrm if(isinstance(infrm, dict)) else {})
    try:
        diversity = len(list(set(data)))
        usl, lsl = np.nan, np.nan
        if(diversity<diversity_lbd):
            print('diversity %d<%d'%(diversity, diversity_lbd))
        else:
            if(do_stats):
                DFP.normfit(data, n_sigma=n_sigma, ret=infrm)
                stamp = LOGger.stamp_process('',stamps,'','','','_', digit=4)
                ret.update({stamp:infrm})
            if(do_consistency_validation):
                if(DFP.isiterable(base)):
                    if(np.array(base).shape[0]==np.array(data).shape[0]):
                        infrm.update(DFP.sumup_regression_norms(base, data, tol=tol))
            infrm_plot = dcp(infrm)
            if(infrm):
                if('count' in infrm_plot):    infrm_plot['$n$'] = DFP.parse(infrm_plot.pop('count'))
                elif('size' in infrm_plot):    infrm_plot['$n$'] = DFP.parse(infrm_plot.pop('size'))
                if('mse' in infrm_plot):   infrm_plot['mse'] = DFP.parse(infrm_plot.pop('mse'), digit=4)
                if('rmse' in infrm_plot):   infrm_plot['rmse'] = DFP.parse(infrm_plot.pop('rmse'), digit=4)
                if('r2' in infrm_plot):   infrm_plot['$r^2$'] = DFP.parse(infrm_plot.pop('r2'), digit=4)
                if('OKR' in infrm_plot):   infrm_plot['OKR'] = DFP.parse(infrm_plot.pop('OKR'), digit=4)
                if('tol' in infrm_plot):   infrm_plot['$\epsilon$'] = DFP.parse((infrm_plot.pop('tol') if(
                    not np.isnan(DFP.astype(infrm_plot['tol'], default=np.nan))) else LOGger.type_string(infrm_plot.pop('tol'))) , digit=4)
                infrm_plot.pop('W', np.nan)
                infrm_plot.pop('p', np.nan)
                mu = infrm_plot.pop('mean', np.nan)
                std = infrm_plot.pop('std', np.nan)
                infrm_plot['$\mu$'] = DFP.parse(mu)
                infrm_plot['$\sigma$'] = DFP.parse(std)
                usl = infrm_plot.pop('norm_upper', np.nan)
                lsl = infrm_plot.pop('norm_lower', np.nan)
        label, label_try = '', 0
        if(LOGger.isinstance_not_empty(infrm_plot, dict) or LOGger.isinstance_not_empty(stamps, list)):
            while(label_try<3):
                try:
                    label = dcp(r'%s'%LOGger.stamp_process('',[LOGger.stamp_process(
                    '',[stamp, LOGger.stamp_process('',infrm_plot,'=','','',', ',digit=4)],'','','',':'), memo],'','','','\n'))
                except Exception as e:
                    LOGger.exception_process(e,logfile='',stamps=[simphist.__name__]+stamps+[label_try])
                    time.sleep(1)
                else:
                    break
                label_try += 1
        myhist(data, ax, color=color, alpha=alpha, num_bins=num_bins, histtype=histtype, density=density, base=base, 
             infrm=infrm_plot, scatter_size=scatter_size, marker=marker, label=label)
        if(not isinstance(base,type(None))):
            np_base = np.array(base)
            axst_ = axst if(axst) else ax
            axst_.set_yticks(yticks_st) if(DFP.isiterable(yticks_st)) else None
            xmax = np_base.max() if(xmax==None) else xmax
            xmin = np_base.min() if(xmin==None) else xmin
            x_ = np.linspace(xmin, xmax, num_bins)
            threshold_color = threshold_color[:3] if(DFP.isiterable(threshold_color)) else color
            if(infrm):
                if(not DFP.parse(infrm_plot.get('$\mu$')) in ['None', 'nan']):
                    axst_.plot(x_, np.full(x_.shape[0], mu), '--', color=threshold_color[:3], label=r'$\mu$:%s'%infrm_plot.get('$\mu$',''))
                if(not DFP.parse(lsl) in ['None', 'nan']):
                    axst_.plot(x_, np.full(x_.shape[0], lsl), '-', color=threshold_color[:3], label=r'$lsl$:%s'%DFP.parse(lsl))
                if(not DFP.parse(usl) in ['None', 'nan']):
                    axst_.plot(x_, np.full(x_.shape[0], usl), '-', color=threshold_color[:3], label=r'$usl$:%s'%DFP.parse(usl))
                if(not DFP.parse(infrm_plot.get('OKR')) in ['None', 'nan']):
                    # axst_.plot(x_, np.full(x_.shape[0], mu), '--', color=threshold_color[:3], label=r'$\mu$:%s'%infrm_plot.get('$\mu$',''))
                    draw_consistency_auxlines(None, axst_, tol=tol, color = threshold_color[:3])
                    
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=[normhist.__name__]+stamps)
        return False
    return True 
   
def normhist(data, ax, axst=None, infrm=None, diversity_lbd=2, stats_method=None, memo='', color=(0.1,0.05,0.4), alpha=0.5, 
             num_bins=100, histtype = 'barstacked', density = True, xmax=None, xmin=None, stamps=None, ret=None, 
             threshold_color=None, do_stats=True, n_sigma=3, **kwags):
    if(not checkDataShape(data, criterion=(lambda x:len(np.array(x).shape)!=1))):
        return False
    stamps = stamps if(isinstance(stamps, list) or isinstance(stamps, dict)) else []
    stamp = LOGger.stamp_process('',stamps)
    ret = ret if(isinstance(ret, dict)) else {}
    infrm = infrm if(isinstance(infrm, dict)) else {}
    infrm_plot = {}
    try:
        mu, std = np.nan, np.nan
        diversity = len(list(set(data)))
        if(diversity<diversity_lbd):
            print('diversity %d<%d'%(diversity, diversity_lbd))
        else:
            if(do_stats):
                DFP.normfit(data, n_sigma=n_sigma, ret=infrm)
                stamp = LOGger.stamp_process('',stamps,'','','','_')
                ret.update({stamp:infrm})
            infrm_plot = dcp(infrm)
            if('count' in infrm_plot):    infrm_plot['$n$'] = DFP.parse(infrm_plot.pop('count'))
            mu = infrm_plot.pop('mean', np.nan)
            std = infrm_plot.pop('std', np.nan)
            infrm_plot['$\mu$'] = DFP.parse(mu)
            infrm_plot['$\sigma$'] = DFP.parse(std)
            if('norm_upper' in infrm_plot):    infrm_plot['$usl$'] = DFP.parse(infrm_plot.pop('norm_upper'))
            if('norm_lower' in infrm_plot):    infrm_plot['$lsl$'] = DFP.parse(infrm_plot.pop('norm_lower'))
            if('p' in infrm_plot):    infrm_plot['$p$'] = DFP.parse(infrm_plot.pop('p'))
            if('W' in infrm_plot):    infrm_plot['$W$'] = DFP.parse(infrm_plot.pop('W'))
            if('E' in infrm_plot):    infrm_plot['$E$'] = DFP.parse(infrm_plot.pop('E'))
        label, label_try = '', 0
        while(label_try<3):
            try:
                label = dcp(r'%s'%LOGger.stamp_process('',[LOGger.stamp_process(
                '',[stamp, LOGger.stamp_process('',infrm_plot,'=','','',', ')],'','','',':'), memo],'','','','\n'))
            except Exception as e:
                LOGger.exception_process(e,logfile='',stamps=[normhist.__name__]+stamps+[label_try])
                time.sleep(1)
            else:
                break
            label_try += 1
        myhist(data, ax, color=color, alpha=alpha, num_bins=num_bins, histtype=histtype, density=density, label=label)
        if(axst):
            xmax = data.max() if(xmax==None) else xmax
            xmin = data.min() if(xmin==None) else xmin
            if(not np.isnan(mu) and not np.isnan(std)):
                x_ = np.linspace(xmin, xmax, num_bins)
                y = scipy.stats.norm.pdf(x_,mu,std)
                axst.plot(x_, y, '--', color=color[:3])
            threshold_color = threshold_color[:3] if(DFP.isiterable(threshold_color)) else color
            if(DFP.astype(infrm_plot.get('$lsl$'))!=None):
                threshold = DFP.astype(infrm_plot['$lsl$'])
                axst.axvline(threshold, color=threshold_color, ls='-.')
            if(DFP.astype(infrm_plot.get('$usl$'))!=None):
                threshold = DFP.astype(infrm_plot['$usl$'])
                axst.axvline(threshold, color=threshold_color, ls='-.')
            axst.set_ylim(bottom=0)
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=[normhist.__name__]+stamps)
        return False
    return True
    
def add_chi2hist(data, ax, **kwags):
    """
    
    Parameters
    ----------
    data : TYPE
        should be 1d data
    **kwags : TYPE
        infrm=None, diversity_lbd=2, stats_method=None, memo='', color=(0.1,0.05,0.4), alpha=0.5, 
        num_bins=100, histtype = 'barstacked', density = True, xmax=None, xmin=None, stamps=None, ret=None, 
        threshold_color=None, do_stats=True, alpha_conf=0.05, title='Histogram and chi2dstrb', xlb='', ylb='probability density', **kwags

    Returns
    -------
    None.
    
    """
    rel = myRelation(data, plot_method_stg='chi2hist', **kwags)
    ax.add(rel)
    
def chi2hist(data, ax, axst=None, infrm=None, diversity_lbd=2, stats_method=None, memo='', color=(0.1,0.05,0.4), alpha=0.5, 
             num_bins=100, histtype = 'barstacked', density = True, xmax=None, xmin=None, stamps=None, ret=None, 
             threshold_color=None, do_stats=True, df=None, alpha_conf=0.05, xlb='', ylb='probability density', **kwags):
    if(not checkDataShape(data, criterion=(lambda x:len(np.array(x).shape)!=1))):
        return False
    stamps = stamps if(isinstance(stamps, list) or isinstance(stamps, dict)) else []
    stamp = LOGger.stamp_process('',stamps)
    ret = ret if(isinstance(ret, dict)) else {}
    infrm = infrm if(isinstance(infrm, dict)) else {}
    
    try:
        diversity = len(list(set(data)))
        if(diversity<diversity_lbd):
            print('diversity %d<%d'%(diversity, diversity_lbd))
        else:
            if(do_stats):
                infrm.clear()
                DFP.chi2fit(data, alpha=alpha_conf, ret=infrm)
            infrm_plot = dcp(infrm)
            if('count' in infrm_plot):    infrm_plot['$n$'] = DFP.parse(infrm_plot.pop('count'))
            if('df' in infrm_plot):    
                df_infrm = infrm_plot.pop('df')
                if(df==None):   df=df_infrm
                infrm_plot['$\delta$'] = DFP.parse(df)
            if('loc' in infrm_plot):    infrm_plot['$x_0$'] = DFP.parse(infrm_plot.pop('loc'))
            if('scale' in infrm_plot):    infrm_plot['$\lambda$'] = DFP.parse(infrm_plot.pop('scale'))
            if('alpha' in infrm_plot):    infrm_plot['$\\alpha$'] = DFP.parse(infrm_plot.pop('alpha'))
            if('p' in infrm_plot):    infrm_plot['$p$'] = DFP.parse(infrm_plot.pop('p'))
            if('W' in infrm_plot):    infrm_plot['$W$'] = DFP.parse(infrm_plot.pop('W'))
            if('chi2_upper' in infrm_plot):   infrm_plot.pop('chi2_upper')
            if('chi2_upper_adjusted' in infrm_plot):    infrm_plot['$ucl$'] = DFP.parse(infrm_plot.pop('chi2_upper_adjusted'))#('chi2_upper_adjusted'))
        myhist(data, ax, color=color, alpha=alpha, num_bins=num_bins, histtype=histtype, density=density)
        if(axst):
            xmax = data.max() if(xmax==None) else xmax
            xmin = data.min() if(xmin==None) else xmin
            x_ = np.linspace(xmin, xmax, num_bins)
            y = scipy.stats.chi2.pdf(x_, df=df)
            axst.plot(x_, y, '--', color=color[:3], label=r'%s'%(LOGger.stamp_process(
                '',[LOGger.stamp_process('',[stamp, LOGger.stamp_process('',infrm_plot,'=','','',', ')],'','','',':'),   memo],'','','','\n')))
            threshold_color = threshold_color[:3] if(DFP.isiterable(threshold_color)) else color
            if(DFP.astype(infrm_plot.get('$ucl$'))!=None):
                threshold = DFP.astype(infrm_plot['$ucl$'])
                axst.axvline(threshold, color=threshold_color, ls='-.')
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=[chi2hist.__name__]+stamps)
        print('Did you set df?? now df=%s'%DFP.parse(df))
        return False
    return True

def regressioniou(fact, pred, ax, axst=None, infrm=None, diversity_lbd=2, stats_method=None, memo='', 
                  colorFact=(0,0,1), colorPred=(0,1,0), linewidth=5, alpha=0.3, 
                  num_bins=100, histtype = 'barstacked', density = True, xmax=None, xmin=None, stamps=None, ret=None, 
                  threshold_color=None, do_stats=True, n_sigma=3, **kwags):
    if(not checkDataShape(fact, criterion=(lambda x:np.array(x).shape[1]!=2 if(len(np.array(x).shape)==2) else True))):
        return False
    if(not checkDataShape(pred, criterion=(lambda x:np.array(x).shape[1]!=2 if(len(np.array(x).shape)==2) else True))):
        return False
    stamps = stamps if(isinstance(stamps, list) or isinstance(stamps, dict)) else []
    stamp = LOGger.stamp_process('',stamps)
    ret = ret if(isinstance(ret, dict)) else {}
    infrm = infrm if(isinstance(infrm, dict)) else {}
    try:
        if(do_stats):
            infrm['iou'] = DFP.iou(fact, pred, egn=kwags.get('egn',np), get_zeros_method=kwags.get('get_zeros_method',np))
            infrm['miou'] = np.median(infrm['iou'])
            infrm['count'] = fact.shape[0] if(hasattr(fact, 'shape')) else len(fact)
            stamp = LOGger.stamp_process('',stamps,'','','','_')
            ret.update({stamp:infrm})
        infrm_plot = dcp(infrm)
        infrm_plot.pop('iou', None)
        infrm_plot['miou'] = '%.2f%%'%(infrm['miou']*100)
        infrm_plot['n'] = infrm_plot.pop('count', np.nan)
        label, label_try = '', 0
        while(label_try<3):
            try:
                label = dcp(r'%s'%LOGger.stamp_process('',[LOGger.stamp_process(
                '',[stamp, LOGger.stamp_process('',infrm_plot,'=','','',', ')],'','','',':'), memo],'','','','\n'))
            except Exception as e:
                LOGger.exception_process(e,logfile='',stamps=[normhist.__name__]+stamps+[label_try])
                time.sleep(1)
            else:
                break
            label_try += 1
        regreiou(fact, pred, ax, colorFact=colorFact, colorPred=colorPred, linewidth=linewidth, 
                 alpha=alpha, axtitle=label)
        # if(axst):
        #     xmax = data.max() if(xmax==None) else xmax
        #     xmin = data.min() if(xmin==None) else xmin
        #     x_ = np.linspace(xmin, xmax, num_bins)
        #     y = scipy.stats.norm.pdf(x_,mu,std)
        #     axst.plot(x_, y, '--', color=color[:3])
        #     threshold_color = threshold_color[:3] if(DFP.isiterable(threshold_color)) else color
        #     if(DFP.astype(infrm_plot.get('$lsl$'))!=None):
        #         threshold = DFP.astype(infrm_plot['$lsl$'])
        #         axst.axvline(threshold, color=threshold_color, ls='-.')
        #     if(DFP.astype(infrm_plot.get('$usl$'))!=None):
        #         threshold = DFP.astype(infrm_plot['$usl$'])
        #         axst.axvline(threshold, color=threshold_color, ls='-.')
        #     axst.set_ylim(bottom=0)
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=[normhist.__name__]+stamps)
        return False
    return True

def pltinitial(fig=None, ax=None, r=1,c=1,a=1,stamps=None,figsize=m_figsize,**kwags):
    """
    ax != None ---> Nothing changes!!!
    fig != None ---> types follow

    Parameters
    ----------
    fig : TYPE, optional
        DESCRIPTION. The default is None.
    ax : TYPE, optional
        DESCRIPTION. The default is None.
    r : TYPE, optional
        DESCRIPTION. The default is 1.
    c : TYPE, optional
        DESCRIPTION. The default is 1.
    a : TYPE, optional
        DESCRIPTION. The default is 1.
    stamps : TYPE, optional
        DESCRIPTION. The default is None.
    **kwags : TYPE
        DESCRIPTION.

    Returns
    -------
    fig : TYPE
        maybe of type plt.Figure or myFig.
    ax : TYPE
        maybe of type plt.Axes or myAxes.

    """
    if(ax is None):
        # if(not isinstance(fig, myFig)):
        if(fig is None):
            # fig, ax = subplots(r,c,a,**kwags)
            fig = plt.Figure(figsize=figsize)
        if(isinstance(fig, plt.Figure)):
            ax = fig.add_subplot(r,c,a)
        elif(isinstance(fig, myFig)):
            #myFig
            ax = fig.add_subplot(r,c,a,stamps=stamps)
            print('\n',id(fig.fig), id(ax.ax), len(get_frames(fig)), r,c,a,'\n')
    figRet = ax.get_figure() if(isinstance(ax, plt.Axes)) else fig
    return figRet, ax

def twinx(ax, right_outward=0):
    axtw = getattr(ax,'ax',ax).twinx()
    axtw.spines['right'].set_position(('outward', right_outward))
    return axtw

def nanDataProcess(data, handler=None, **kwags):
    data_nonallna = dcp(data[[v for v in data.columns if (np.logical_not(np.isnan(data[v].values)).any())]] if(
        hasattr(data, 'columns')) else data[:,np.logical_not(np.isnan(data).any(axis=0))])
    n_frame = data_nonallna.shape[1]
    if(n_frame==0):
        print('data_nonallna.shape[1]==0!!!')
        return  False
    if(handler is not None):
        handler.n_frame = n_frame
        handler.data_nonallna = data_nonallna
    return True

def FromIntToMask(base, refernceData, **kwags):
    data_shape = refernceData.shape
    if(data_shape[1]==0):
        return None, refernceData, 0
    base_axis = base%(data_shape[1])
    base = np.array(refernceData)[:,base_axis]
    data_seperate_from_base = np.delete(np.array(refernceData), base_axis, 1) if(isinstance(refernceData, np.ndarray)) else refernceData.drop(
        refernceData.columns[base_axis], axis=1)
    n_frame_seperate_from_base = refernceData.shape[1]
    return base, data_seperate_from_base, n_frame_seperate_from_base

def report(data, fig=None, ax=None, file='', is_end=True, mask=None, 
           title='', axtitle='', color=None, colors=None, color_default=None, infrms=None, infrm_default=None, 
           do_stats=True, diversity_lbd=2, init_kwags=None, class_uniquifying_method=m_uniq_thru_set, 
           axst=None, yticks_st=None, stamps=None, base=None, scatter_size=5, do_consistency_validation=False, tol=None, **kwags):
    stamps = stamps if(isinstance(stamps, list)) else []
    infrms =  infrms if(isinstance(infrms, dict)) else {}
    data_shape = np.array(data).shape
    handler = LOGger.mystr()
    ismulti_mask = False
    if(DFP.isiterable(mask)):
        if(len(mask.shape)>1):
            ismulti_mask = mask.shape[0]==data.shape[0]
    if(len(data_shape)>1):
        if(not nanDataProcess(data, handler=handler, **kwags)):
            return False
        n_frame = handler.n_frame
        data_nonallna = handler.data_nonallna
        if(isinstance(base, int)):  base, data, n_frame = FromIntToMask(base, data_nonallna)
        fig, ax = pltinitial(fig, ax, r=int(n_frame//int(np.sqrt(n_frame))+1), c=int(np.sqrt(n_frame)), **kwags)
        for ic, nc in enumerate(getattr(data, 'columns', range(data.shape[1]))):
            data_ic = np.array(data)[:,ic]
            mask_ic = dcp(mask)
            if(ismulti_mask):
                mask_ic = np.array(mask)[:,ic] if(ic<n_frame) else None
            # print('fig.axes', len(fig.axes))
            if(not report(data_ic, mask=mask_ic, color=color, colors=colors, color_default=color_default, file=None, 
                   axtitle=getattr(data, 'columns', range(1,data.shape[1]+1))[ic], 
                   infrms=infrms.get(nc,{}), infrm_default=infrm_default, do_stats=do_stats, diversity_lbd=diversity_lbd, 
                   is_end=False, fig=getattr(fig,'fig',fig), ax=(getattr(ax,'ax',ax) if(ic==0) else None), 
                   init_kwags=init_kwags, class_uniquifying_method=class_uniquifying_method, stamps=stamps+[nc], 
                   axst=None, yticks_st=yticks_st, r=int(n_frame//int(np.sqrt(n_frame))+1), c=int(np.sqrt(n_frame)), a=ic+1, base=base, 
                   scatter_size=scatter_size, do_consistency_validation=do_consistency_validation, tol=tol, title=title, **kwags)):
                return False
        end(getattr(fig,'fig',fig), do_axdraw=True, file=file, title=title, **kwags) if(is_end) else None
        return True
    fig, ax = pltinitial(fig, ax, **kwags)
    if(not isinstance(mask, np.ndarray)):
        mask = np.full(data.shape[0], True)
    else:
        if(mask.shape[0]!=data.shape[0]):
            mask = np.full(data.shape[0], True)
    # print("getattr(mask, 'values', mask)", getattr(mask, 'values', mask))
    classes = class_uniquifying_method(getattr(mask, 'values', mask))
    if(do_stats):
        (ax if(axst==None) else axst).set_yticks(yticks_st) if(DFP.isiterable(yticks_st)) else None
    ax.set_title(axtitle) if(axtitle or axtitle==0) else None
    if(DFP.isiterable(base)):
        if(np.array(base).shape[0]!=data.shape[0]):
            print("np.array(base).shape[0]!=data.shape[0]!!!")
            base = None
    infrm_default =  infrm_default if(isinstance(infrm_default, dict)) else {}
    if(isinstance(color, type(None))):
        if(DFP.isiterable(colors)):
            colors = mylist(colors)
            colors = {m:colors.get(i, color_default) for i,m in enumerate(classes)}
        elif(not isinstance(colors, dict)):
            colors = dict(zip(*[classes, vs2.cm_rainbar(len(classes))]))
    else:
        colors = {k:color for k in classes}
    ret=kwags.get('ret',{})
    ret['stat'] = ret.get('stat', {})
    for m in classes:
        data_m = dcp(np.array(data)[mask==m])
        data_m = np.array(tuple(map(lambda v:DFP.astype(v,default=np.nan), data_m)))
        if(DFP.isiterable(base)):
            base_m = dcp(np.array(base)[mask==m])
            base_m = np.array(tuple(map(lambda v:DFP.astype(v,default=np.nan), base_m)))
        else:
            base_m = None
        diversity = len(list(set(getattr(data_m,'values',data_m))))
        if(not np.logical_not(np.isnan(getattr(data_m,'values',data_m))).any()):
            continue
        data_m, base_m = LOGger.selectionSynch(data_m, base_m, method=lambda x:x[np.logical_not(np.isnan(getattr(data_m,'values',data_m)))])
        data_m, base_m = LOGger.selectionSynch(data_m, base_m, method=lambda x:x[np.abs(getattr(data_m,'values',data_m))<np.inf])
        ret_stat = {}
        simphist(data_m, ax, axst=(axst if(diversity>diversity_lbd) else None), 
                 infrm=dcp(infrms.get(m, infrm_default)), color=dcp(colors.get(m, color_default)),
                 diversity_lbd=diversity_lbd, stamps=stamps+([m] if(len(classes)!=1) else []), do_stats=do_stats, ret=ret_stat, 
                 base=base_m, scatter_size=scatter_size, do_consistency_validation=do_consistency_validation, tol=tol)
        ret['stat'].update(ret_stat)
    end(fig, do_axdraw=True, file=file, title=title, **kwags) if(is_end) else None    
    return True
    
def report_InAFrame(data, fig=None, ax=None, file='', is_end=True, mask=None, 
                    axtitle='', colors=None, color_default=None, infrms=None, infrm_default=None, 
                    do_stats=False, diversity_lbd=2, init_kwags=None, class_uniquifying_method=m_uniq_thru_set, 
                    axst=None, yticks_st=None, stamps=None, base=None, scatter_size=5, 
                    marker=None, markers=None, do_consistency_validation=False, tol=None, **kwags):
    stamps = stamps if(isinstance(stamps, list)) else []
    infrms =  infrms if(isinstance(infrms, dict)) else {}
    handler = LOGger.mystr()
    if(len(data.shape)>1):
        if(not nanDataProcess(data, handler=handler, **kwags)):
            return False
        data_nonallna = handler.data_nonallna
        ismulti_mask = False
        if(DFP.isiterable(mask)):
            if(len(mask.shape)>1):
                ismulti_mask = mask.shape[0]==data.shape[0]
        if(isinstance(base, int)):  base, data_nonallna, _ = FromIntToMask(base, data_nonallna)
        header = getattr(data_nonallna, 'columns', range(data_nonallna.shape[1]))
        dominateLen = max(len(header), len(m_markers))
        markers = markers if(isinstance(markers, dict)) else dict(tuple(zip(*(header[:dominateLen], LOGger.myCycle(m_markers)[:dominateLen]))))
        fig, ax = pltinitial(fig, ax, **kwags)
        for ic, nc in enumerate(header):
            data_ic = np.array(data_nonallna)[:,ic]
            mask_ic = dcp(mask)
            if(ismulti_mask):
                mask_ic = np.array(mask)[:,ic]
            axc=(ax if(ic==0) else twinx(ax, 30*(ic-1)))
            report_InAFrame(data_ic, mask=mask_ic, colors=colors, color_default=color_default, file=None, 
                            axtitle=getattr(data_nonallna, 'columns', range(data_nonallna.shape[1]))[ic], 
                            infrms=infrms.get(nc,{}), infrm_default=infrm_default, do_stats=do_stats, diversity_lbd=diversity_lbd, 
                            is_end=False, fig=None, ax=axc, 
                            init_kwags=init_kwags, class_uniquifying_method=class_uniquifying_method, stamps=stamps+[nc], 
                            axst=axst, yticks_st=yticks_st, scatter_size=scatter_size,
                            marker=dcp(markers.get(nc, markers.get(ic, m_markers[ic%len(m_markers)]))), 
                            base=base, **kwags)
        end(getattr(fig,'fig',fig), do_axdraw=True, file=file, title=title, **kwags) if(is_end) else None
        return True
    fig, ax = pltinitial(fig, ax, **kwags)
    if(not isinstance(mask, np.ndarray)):
        mask = np.full(data.shape[0], True)
    else:
        if(mask.shape[0]!=data.shape[0]):
            mask = np.full(data.shape[0], True)
    classes = class_uniquifying_method(getattr(mask, 'values', mask))
    if(do_stats):
        (axst if(axst!=None) else ax).set_yticks(yticks_st) if(DFP.isiterable(yticks_st)) else None
    # ax.set_title(axtitle) if(axtitle) else None
    if(DFP.isiterable(base)):
        if(np.array(base).shape[0]!=data.shape[0]):
            base = None
    infrm_default =  infrm_default if(isinstance(infrm_default, dict)) else {}
    colors =  colors if(isinstance(colors, dict)) else dict(zip(*[classes, vs2.cm_rainbar(len(classes))]))
    ret=kwags.get('ret',{})
    ret['stat'] = ret.get('stat', {})
    for m in classes:
        data_m = dcp(np.array(data)[mask==m])
        data_m = np.array(tuple(map(lambda v:DFP.astype(v,default=np.nan), data_m)))
        # data_m = np.array(tuple(map(lambda v:DFP.astype(v,default=np.nan), data_m))) if(
        #     str(data_m.dtype).find('int')==-1 and str(data_m.dtype).find('float')==-1) else data_m
        diversity = len(list(set(getattr(data_m,'values',data_m))))
        if(not np.logical_not(np.isnan(getattr(data_m,'values',data_m))).any()):
            continue
        data_m = data_m[np.logical_not(np.isnan(getattr(data_m,'values',data_m)))]
        data_m = data_m[np.abs(getattr(data_m,'values',data_m))<np.inf]
        ret_stat = {}
        if(isinstance(ax,myAxes)):
            if(isinstance(base, type(None))):
                add_data(data_m, ax=ax, plot_method_stg='hist', axst=(axst if(diversity>diversity_lbd) else None), 
                         color=dcp(colors.get(m, color_default)), stamps=stamps+([m] if(len(classes)!=1) else []), 
                         infrm=dcp(infrms.get(m, infrm_default)), do_stats=do_stats, ret=ret_stat)
            else:
                add_data(data_m, ax=ax, plot_method_stg='scatter', axst=(axst if(diversity>diversity_lbd) else None), 
                         color=dcp(colors.get(m, color_default)), stamps=stamps+([m] if(len(classes)!=1) else []), 
                         infrm=dcp(infrms.get(m, infrm_default)), do_stats=do_stats, ret=ret_stat, marker=marker)
        elif(isinstance(ax,plt.Axes)):
            simphist(data_m, ax, axst=(axst if(diversity>diversity_lbd) else None), 
                     infrm=dcp(infrms.get(m, infrm_default)), color=dcp(colors.get(m, color_default)),
                     diversity_lbd=diversity_lbd, stamps=stamps+([m] if(len(classes)!=1) else []), do_stats=do_stats, ret=ret_stat, 
                     base=base, scatter_size=scatter_size, marker=marker, do_consistency_validation=do_consistency_validation, tol=tol)
        ret['stat'].update(ret_stat)
    end(fig, do_axdraw=True, file=file, title=title, **kwags) if(is_end) else None
    return True

def report_normhist(data=np.random.normal(size=10000), mask=None, color=None, colors=None, color_default=(0.1,0.05,0.4), file='', 
                    xlb='', ylb='probability density', axtitle='', title='Histogram and normdstrb', infrms=None, 
                    infrm_default=None, do_stats=True, diversity_lbd=2, is_end=True, fig=None, ax=None, init_kwags=None, 
                    class_uniquifying_method=m_uniq_thru_set, axst=None, yticks_st=None, stamps=None, **kwags):
    """
    
    Parameters
    ----------
    data : TYPE, optional
        DESCRIPTION. The default is np.random.normal(size=10000).
    mask : TYPE, optional
        DESCRIPTION. The default is None.
    colors : TYPE, optional
        DESCRIPTION. The default is None.
    color_default : TYPE, optional
        DESCRIPTION. The default is (0.1,0.05,0.4).
    file : TYPE, optional
        DESCRIPTION. The default is ''.
    **kwags : TYPE
        for subplots.

    Returns
    -------
    None.

    """
    infrms =  infrms if(isinstance(infrms, dict)) else {}
    stamps = stamps if(isinstance(stamps, list)) else []
    if(len(data.shape)>1):
        data_nonallna = data[[v for v in data.columns if (np.logical_not(np.isnan(getattr(data[v],'values',data[v]))).any())]].copy()
        n_frame = data_nonallna.shape[1]
        fig, ax = pltinitial(fig, ax, r=int(n_frame//int(np.sqrt(n_frame))+1), c=int(np.sqrt(n_frame)), **kwags)
        for ic, nc in enumerate(getattr(data_nonallna, 'columns', range(data_nonallna.shape[1]))):
            inrfmStamp = dcp(LOGger.stamp_process('',[nc,ic],'','','','_') if(nc in infrms) else nc)
            infrms.update({inrfmStamp:{}})
            data_ic = np.array(data_nonallna)[:,ic]
            report_normhist(data_ic, mask=mask, colors=colors, color_default=color_default, file=None, 
                            xlb=xlb, ylb=ylb, axtitle=getattr(data_nonallna, 'columns', range(data_nonallna.shape[1]))[ic], 
                            infrms=infrms.get(inrfmStamp,{}), infrm_default=infrm_default, do_stats=do_stats, diversity_lbd=diversity_lbd, 
                            is_end=False, fig=getattr(fig,'fig',fig), ax=(getattr(ax,'ax',ax) if(ic==0) else None), 
                            init_kwags=init_kwags, class_uniquifying_method=class_uniquifying_method, stamps=stamps+[nc], 
                            axst=None, yticks_st=yticks_st, r=int(n_frame//int(np.sqrt(n_frame))+1), c=int(np.sqrt(n_frame)), a=ic+1, **kwags)
        end(getattr(fig,'fig',fig), do_axdraw=True, file=file, title=title, **kwags) if(is_end) else None
        return 
    fig, ax = pltinitial(fig, ax, **kwags)
    if(not isinstance(mask, np.ndarray)):
        mask = np.full(data.shape[0], True)
    else:
        if(mask.shape[0]!=data.shape[0]):
            mask = np.full(data.shape[0], True)
    classes = class_uniquifying_method(getattr(mask, 'values', mask))
    if(axst==None):
        axst = ax.twinx()
        axst.set_xlabel(xlb)
        axst.set_ylabel(ylb)
        axst.set_yticks(yticks_st) if(DFP.isiterable(yticks_st)) else None
    ax.set_title(axtitle) if(axtitle) else None
    if(isinstance(color, type(None))):
        if(DFP.isiterable(colors)):
            colors = mylist(colors)
            colors = {m:colors.get(i, color_default) for i,m in enumerate(classes)}
        elif(not isinstance(colors, dict)):
            colors = dict(zip(*[classes, vs2.cm_rainbar(len(classes))]))
    else:
        colors = {k:color for k in classes}
    ret=kwags.get('ret',{})
    ret['stat'] = ret.get('stat', {})
    for m in classes:
        data_m = dcp(np.array(data)[mask==m])
        if(not np.logical_not(np.isnan(getattr(data_m,'values',data_m))).any()):
            continue
        data_m = data_m[np.logical_not(np.isnan(getattr(data_m,'values',data_m)))]
        data_m = data_m[np.abs(getattr(data_m,'values',data_m))<np.inf]
        diversity = len(list(set(getattr(data_m,'values',data_m))))
        ret_stat = {}
        if(isinstance(ax,myAxes)):
            add_data(data_m, ax=ax, plot_method_stg='normhist', axst=(axst if(diversity>diversity_lbd) else None), 
                     color=dcp(colors.get(m, color_default)), stamps=stamps+([m] if(len(classes)!=1) else []), 
                     infrm=dcp(infrms.get(m, infrm_default)), do_stats=do_stats, ret=ret_stat)
        elif(isinstance(ax,plt.Axes)):
            normhist(data_m, ax, axst=(axst if(diversity>diversity_lbd) else None), 
                     infrm=dcp(infrms.get(m, infrm_default)), color=dcp(colors.get(m, color_default)),
                     diversity_lbd=diversity_lbd, stamps=stamps+([m] if(len(classes)!=1) else []), do_stats=do_stats, ret=ret_stat)
        ret['stat'].update(ret_stat)
    end(fig, file=file, title=title, **kwags) if(is_end) else None
    
def report_chi2hist(data=np.random.chisquare(df=2,size=10000), mask=None, colors=None, color_default=(0.1,0.05,0.4), file='', 
                    dfs=None, df_default=2, xlb='', ylb='probability density', axtitle='', 
                    title='Histogram and chi2dstrb', infrms=None, infrm_default=None, do_stats=True, 
                    diversity_lbd=3, is_end=True, fig=None, ax=None, init_kwags=None, 
                    class_uniquifying_method=m_uniq_thru_set, axst=None, xticks=None, yticks_st=None, stamps=None, **kwags):
    """
    

    Parameters
    ----------
    data : TYPE, optional
        DESCRIPTION. The default is np.random.chisquare(df=2,size=10000).
    mask : TYPE, optional
        DESCRIPTION. The default is None.
    colors : TYPE, optional
        DESCRIPTION. The default is None.
    color_default : TYPE, optional
        DESCRIPTION. The default is (0.1,0.05,0.4).
    file : TYPE, optional
        DESCRIPTION. The default is ''.
    dfs : TYPE, optional
        DESCRIPTION. The default is None.
    df_default : TYPE, optional
        DESCRIPTION. The default is 2(for default data) / None(for real data).
    **kwags : TYPE
        for subplots.

    Returns
    -------
    None.

    """
    fig, ax = pltinitial(fig, ax, **kwags)
    if(not isinstance(mask, np.ndarray)):
        mask = np.full(data.shape[0], True)
    else:
        if(mask.shape[0]!=data.shape[0]):
            mask = np.full(data.shape[0], True)
    classes = class_uniquifying_method(getattr(mask, 'values', mask))
    if(axst==None):
        axst = ax.twinx()
        axst.set_xlabel(xlb)
        axst.set_ylabel(ylb)
        axst.set_yticks(yticks_st) if(DFP.isiterable(yticks_st)) else None
        axst.set_title(axtitle) if(axtitle) else None
    infrms =  infrms if(isinstance(infrms, dict)) else {}
    stamps = stamps if(isinstance(stamps, list)) else []
    colors =  colors if(isinstance(colors, dict)) else dict(zip(*[classes, vs2.cm_rainbar(len(classes))]))
    dfs =  dfs if(isinstance(dfs, dict)) else dict(zip(*[classes, np.full(len(classes), df_default)]))
    for m in classes:
        data_m = dcp(np.array(data)[mask==m])
        diversity = len(list(set(getattr(data_m,'values',data_m))))
        if(isinstance(ax,myAxes)):
            add_data(data_m, ax, 'chi2hist', axst=(axst if(diversity>diversity_lbd) else None), 
                     color=colors.get(m, color_default), df=dfs.get(m, df_default), stamps=stamps+([m] if(len(classes)!=1) else []),
                     infrm=infrms.get(m, infrm_default), do_stats=do_stats)
        elif(isinstance(ax,plt.Axes)):
            normhist(data_m, ax, axst=(axst if(diversity>diversity_lbd) else None), 
                     infrm=dcp(infrms.get(m, infrm_default)), color=dcp(colors.get(m, color_default)),
                     diversity_lbd=diversity_lbd, stamps=([m] if(len(classes)!=1) else []), do_stats=do_stats)
    end(fig, do_axdraw=True, file=file, title=title, **kwags) if(is_end) else None
    
def report_regressioniou(n_y_fact, n_y_pred, fig=None, ax=None, layout=(1,1), axIndex=1, is_end=True, 
                         colorFact=(0,0,1), colorPred=(0,1,0), linewidth=5, alpha=0.3, do_stats=True,
                         file=None, title='RegressionIOU', axtitle='', infrm=None, **kwags):
    if(not isinstance(n_y_fact, np.ndarray) or not isinstance(n_y_pred, np.ndarray)):
        print('type error!!!', 'n_y_fact', type(n_y_fact), 'n_y_pred', type(n_y_pred))
        return False
    if(len(n_y_fact.shape)!=2):
        print('n_y_fact shape', n_y_fact.shape)
        return False
    if(n_y_fact.shape[1]!=2):
        print('n_y_fact shape', n_y_fact.shape)
        return False
    if(n_y_fact.shape[1]!=n_y_pred.shape[1]):
        print('n_y_fact shape != n_y_pred shape', n_y_fact.shape, n_y_pred.shape)
        return False
    if((np.diff(n_y_fact, axis=1)<0).any() or (np.diff(n_y_pred, axis=1)<0).any()):
        print('Not ascending!!!', (np.diff(n_y_fact, axis=1)<0).any(), (np.diff(n_y_pred, axis=1)<0).any())
        return False
    fig, ax = pltinitial(fig=fig, ax=ax, r=layout[0], c=layout[1], a=axIndex, figsize=figsize)
    if(not regressioniou(n_y_fact, n_y_pred, ax=ax, infrm=infrm, colorFact=colorFact, colorPred=colorPred, 
                  linewidth=linewidth, alpha=alpha, do_stats=do_stats)):
        return False
    end(fig, do_axdraw=True, file=file, title=title, **kwags) if(is_end) else None
    return True

def draw_consistency_auxlines(fig, ax, tol=None, color = (255/255,90/255,90/255), label=None, **kwags):
    if(ax!=None):
        m,M,h = get_lim_margin(None, ax)
        ax.plot([m-h, M+h], [m-h, M+h], ls='--', color=color, label=label)
        if(not np.isnan(DFP.astype(tol, default=np.nan))):
            ax.plot([m-h, M+h], [m-tol-h, M-tol+h], ls='--', color=color)
            ax.plot([m-h, M+h], [m+tol-h, M+tol+h], ls='--', color=color)
    elif(fig!=None):
        for ax in get_axes(fig):
            if(not draw_consistency_auxlines(None, ax, tol=tol, **kwags)):
                return False
    return True
        
def drawRegressionHeatmap(df, stamps=None, handler=None, ret=None, mask=None, maskColumnName='mask', 
                          height=3, diag_kind='auto', columns=None, **kwags):
    if(mask is not None):
        if(mask.shape[0]!=df.shape[0]):
            LOGger.addlog('shape inconsistence!!', mask.shape[0], df.shape[0], colora=LOGger.FAIL)
            return False
        df[maskColumnName] = mask
    stamps = stamps if(isinstance(stamps, list)) else []
    if(not isinstance(columns,list)):    columns = LOGger.mylist(df.columns)
    snsed = vs.sns.pairplot(df, hue=(maskColumnName if(mask is not None) else None),
                          kind='reg', height=height, vars=columns, 
                          diag_kind=diag_kind)
    fig = snsed.fig
    if(handler is not None): 
        handler.fig = fig
        handler.snsed = snsed
        
    if(isinstance(ret, dict)):  
        ret['fig'] = fig
        ret['snsed'] = snsed
    return True
    
def drawRegressionCorrmap(df, stamps=None, handler=None, ret=None, mask=None, maskColumnName='mask', 
                          height=3, diag_kind='auto', columns=None, corr=None, suptitle='', **kwags):
    if(mask is not None):
        if(mask.shape[0]!=df.shape[0]):
            LOGger.addlog('shape inconsistence!!', mask.shape[0], df.shape[0], colora=LOGger.FAIL)
            return False
        df[maskColumnName] = mask
    stamps = stamps if(isinstance(stamps, list)) else []
    if(not isinstance(columns,list)):    columns = LOGger.mylist(df.columns)
    snsed = vs.sns.pairplot(df, hue=(maskColumnName if(mask is not None) else None),
                          kind='reg', height=height, vars=columns, 
                          diag_kind=diag_kind)
    if(hasattr(corr,'shape')):
        facecolorDyeMethod = kwags.get('facecolorDyeMethod')
        if(LOGger.mylist(corr.shape).get(1,0)>0):
            for i, row_var in enumerate(getattr(corr,'columns',range(corr.shape[1]))):
                for j, col_var in enumerate(getattr(corr,'columns',range(corr.shape[1]))):
                    ax = snsed.axes[i, j]
                    ax.set_xlabel('')
                    ax.set_ylabel('')
                    # 顯示被隱藏的刻度
                    ax.tick_params(axis='both', which='both', bottom=True, top=False, left=True, right=False)
                    ax.tick_params(axis='both', which='major', labelsize=8)  # 可以調整字體大小
                    
                    # 顯示刻度標籤
                    ax.xaxis.set_tick_params(labelbottom=True)
                    ax.yaxis.set_tick_params(labelleft=True)
                    
                    # 從現有的ax獲取數據範圍並設置合適的刻度
                    x_min, x_max = ax.get_xlim()
                    y_min, y_max = ax.get_ylim()
                    
                    # 讓matplotlib自動選擇合適的刻度
                    ax.xaxis.set_major_locator(plt.AutoLocator())
                    ax.yaxis.set_major_locator(plt.AutoLocator())
                    
                    # 重新設置範圍
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)
                    if i == j:
                        label = DFP.parse(df.columns[i])#,stg_max_length=10)
                        ax.set_title(label)
                        continue
                    label = LOGger.stamp_process('',[df.columns[j], df.columns[i]],'','','','<->')#,max_len=5)
                    r =  np.clip((corr.loc[col_var, row_var] if(hasattr(corr, 'iloc')) else corr[i,j]), a_max=1.0, a_min=0.0)
                    infrm = dcp(LOGger.stamp_process('',{'corr':r},digit=5))
                    title = ax.set_title(infrm)
                    title.set_bbox(dict(
                        facecolor=(facecolorDyeMethod(r) if(callable(facecolorDyeMethod)) else None),
                        edgecolor='none'
                    ))
                    ax.legend([dcp(label)])
    fig = snsed.fig
    fig.suptitle(suptitle)
    if(handler is not None): 
        handler.fig = fig
        handler.snsed = snsed
        
    if(isinstance(ret, dict)):  
        ret['fig'] = fig
        ret['snsed'] = snsed
    return True
    
def drawHeatmap(df, stamps=None, handler=None, ret=None, mask=None, maskColumnName='mask', 
                          height=3, diag_kind='auto', columns=None, **kwags):
    if(mask is not None):
        if(mask.shape[0]!=df.shape[0]):
            LOGger.addlog('shape inconsistence!!', mask.shape[0], df.shape[0], colora=LOGger.FAIL)
            return False
        df[maskColumnName] = mask
    stamps = stamps if(isinstance(stamps, list)) else []
    if(not isinstance(columns,list)):    columns = LOGger.mylist(df.columns)
    
    cat_cols += [col for col in df.columns if df[col].nunique() < 10 and df[col].dtype in ['int64', 'int32']]
    def customMethod(x, y, **kwargs):
        ax = plt.gca()
        x_is_cat = x.name in cat_cols
        y_is_cat = y.name in cat_cols
    
        # 兩個都連續：用回歸線
        if not x_is_cat and not y_is_cat:
            sns.regplot(x=x, y=y, ax=ax, scatter_kws={"s": 10}, line_kws={"color": "red"}, **kwargs)
        # 一個類別，一個連續
        elif x_is_cat and not y_is_cat:
            sns.stripplot(x=x, y=y, ax=ax, **kwargs)
        elif not x_is_cat and y_is_cat:
            sns.stripplot(x=y, y=x, ax=ax, orient="h", **kwargs)
        else:
            # 兩個都是類別變數 → 熱度圖式點圖
            ctab = pd.crosstab(x, y)
            sns.heatmap(ctab, annot=True, fmt='d', cbar=False, ax=ax)
    snsed = sns.PairGrid(df)
    snsed.map_lower(customMethod)
    snsed.map_diag(sns.histplot)
    snsed.map_upper(customMethod)
    fig = snsed.fig
    if(handler is not None): 
        handler.fig = fig
        handler.snsed = snsed
        
    if(isinstance(ret, dict)):  
        ret['fig'] = fig
        ret['snsed'] = snsed
    return True
    
    
#%%
class myColorAgent:
    def __init__(self, period=None, default=None, selections=None, alpha=0.3, **kwags):
        self.period = period
        self.default = default
        self.selections = selections
        self.alpha = alpha
        if(not DFP.isiterable(self.selections)):
            if(isinstance(self.period, int)):
                if(self.period>0):
                    print('colorPeriod', self.period)
                    self.selections = vs2.cm_rainbar(self.period)
        else:
            if(np.array(self.selections).shape[0]>0):
                self.period = np.array(self.selections).shape[0]
        
    def call(self, index=0):
        if(not isinstance(self.period, int)):
            return self.default
        if(DFP.isiterable(self.selections) and self.period>0):
            solidColor = self.selections[index%self.period]
        else:
            solidColor = self.default
        if(DFP.isiterable(solidColor)):
            return (*(solidColor[:3]), self.alpha) if(np.array(solidColor).shape[0]>=3) else solidColor
        return self.default
    
    
class myRelation:
    def __init__(self, *core, plot_method_stg='plot', **kwags):
        """
        

        Parameters
        ----------
        *core : TYPE
            DESCRIPTION.
        plot_method_stg : TYPE, optional
            original methods. The default is 'plot'.
        **kwags : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if(not isinstance_not_empty(core, tuple)):
            print('core error!!!!', type(core))
            print('`%s`'%str(core)[:200])
            return
        if(len(set([type(elem) for elem in core]))!=1):
            print('rels types inconsistance!!!!')
            print(*list(set([type(elem) for elem in core]))[:10])
            return
        if(len(set([getattr(np.array(elem), 'shape', None) if(DFP.isiterable(elem)) else None for elem in core]))!=1):
            print('rels shape error!!!!')
            print(*list(set([getattr(np.array(elem), 'shape', None) if(DFP.isiterable(elem)) else None for elem in core]))[:10])
            return
        self.dtype = type(core[0])
        self.dshape = getattr(np.array(core[0]), 'shape', None) if(DFP.isiterable(core[0])) else None
        self.core = core #Ex: [(x1,y1),(x2,y2)], v3, h4...
        self.plot_method_stg = plot_method_stg if(isinstance(plot_method_stg, str)) else 'plot'
        self.kwags = kwags
        
    def export(self, slice_select=slice(None, 2)):
        '''
            for matplotlib not opencv!!!
        '''
        return export_plt(self.core, slice_select=slice_select)

class myAxes:
    def __init__(self, ax, myfigure=None, stamps=None, **kwags):
        self.ax = ax
        self.relations = []
        self.stamps = stamps if(isinstance(stamps, list)) else []
        self.myfigure = myfigure
            
    def twinx(self, stamps=None, **kwargs):
        _axtw = self.ax.twinx()
        stamps = getattr(self, 'stamps', stamps)
        stamps = stamps if(isinstance(stamps, list)) else []
        axtw = myAxes(_axtw, stamps=stamps, myfigure=self.myfigure, **kwargs)
        if(isinstance(self.myfigure, myFig)):
            self.myfigure.myaxes.append(axtw)
        return axtw
    
    # def normhist(self, data, **kwags):
    #     keywords = ['infrm','diversity_lbd','stats_method','memo','color','alpha','num_bins','histtype','density',
    #                 'xmax','xmin','stamps','ret','threshold_color','do_stats','n_sigma']
    #     for k in keywords:
    #         variable = LOGger.execute(k, kwags, self, default=None, not_found_alarm=False)
    #         if(not isinstance(variable, type(None))):
    #             kwags[k] = variable
    #     normhist(data, self.ax, **kwags)
        
    def add(self, relation, **kwags):
        relation.kwags.update(kwags)
        self.relations.append(relation)
        
    def __getattr__(self, name):
        # 获取self.ax对象上的属性或方法
        attr = getattr(self.ax, name)

        if callable(attr):
            # 如果是方法，定义一个函数来转发调用
            def method(*args, **kwargs):
                return attr(*args, **kwargs)
            return method
        else:
            # 如果是属性，直接返回属性值
            return attr
        
    def drawon(self):
        drawon(self, self.relations)

#讓輸出的資訊跟著fig本身
class myFig:
    def __init__(self, fig=None, edge_plank=None, suptitle_stg='', stamps=None, fn='fn', exp_fd='.', save_type='jpg', layout=None, axes=None, **kwags):
        # super().__init__()
        self.fig = plt.Figure(**{k:kwags[k] for k in m_plt_Figure_keywords if k in kwags}) if(
            not isinstance(fig, plt.Figure)) else fig
        self.fig.subplots_adjust(*edge_plank) if(DFP.isiterable(edge_plank)) else None
        self.stamps = stamps if(isinstance(stamps, list)) else []
        self.exp_fd = exp_fd
        self.save_type = save_type
        self.suptitle_stg = suptitle_stg
        self.fn = fn if(isinstance_not_empty(fn, str)) else stamp_process('',[self.suptitle]+self.stamps,'','','','_',for_file=1)
        self.layout = layout
        self.myaxes = []
    
    def __getattr__(self, name):
        # 获取self.ax对象上的属性或方法
        attr = getattr(self.fig, name)

        if callable(attr):
            # 如果是方法，定义一个函数来转发调用
            def method(*args, **kwargs):
                return attr(*args, **kwargs)
            return method
        else:
            # 如果是属性，直接返回属性值
            return attr
    
    def suptitle(self, stg=''):
        self.fig.suptitle(self.suptitle_stg if(LOGger.isinstance_not_empty(self.suptitle_stg, str)) else stg)
    
    #讓輸出的資訊跟著fig本身
    def savefig(self, file=None):
        file = file if(isinstance_not_empty(file, str)) else os.path.join(self.exp_fd, '%s.%s'%(self.fn, self.save_type))
        self.fig.savefig(file)
    
    def set_expfile(self, suptitle=None, stamps=None, fn=None, exp_fd=None, save_type=None):
        if(isinstance(stamps, list)):   self.stamps = stamps
        if(isinstance(exp_fd, str)):   self.exp_fd = exp_fd
        if(isinstance(save_type, str)):   self.save_type = save_type
        if(isinstance(suptitle, str)):   self.suptitle = suptitle
        self.fn = fn if(isinstance_not_empty(fn, str)) else stamp_process('',[self.suptitle]+self.stamps,'','','','_',for_file=1)
    
    def is_completed(self):
        return is_fig_completed(self.fig, stamps=[self.suptitle]+self.stamps)
    
    def add_subplot(self, r, c, a, stamps=None, **kwargs):
        if(self.is_completed()):
            return None
        _ax = self.fig.add_subplot(r, c, a)
        ax = myAxes(_ax, myfigure=self, stamps=stamps, **kwargs)
        self.myaxes.append(ax)
        return ax
    
    def get_file(self):
        return os.path.join(self.exp_fd, self.fn)
    
    def drawon(self, **kwags):
        for ax in self.axes:
            ax.drawon() 
    
    def end(self, do_axdraw=False, **kwags):
        end(self, do_axdraw=do_axdraw, **kwags)
        
class myPainter:
    def __init__(self, fig=None, theme='theme', stamps=None, exp_fd='.', n_axes=None, layout=(1,1), save_type='jpg', 
                 fig_kwags=None, figsize=(10,10), imgfile=None, **kwags):
        self.layout = tuple(map(int, ((np.sqrt(n_axes), (n_axes/np.sqrt(n_axes))+1) if(
            n_axes>0 if(isinstance(n_axes, int)) else False) else layout)))
        self.theme = theme
        self.stamps = stamps if(isinstance(stamps, list)) else []
        self.exp_fd = exp_fd
        self.save_type = save_type if(save_type in ['jpg','png']) else 'jpg'
        self.fig_kwags = fig_kwags if(isinstance(fig_kwags, dict)) else {}
        if(figsize is not None):   self.fig_kwags['figsize'] = figsize
        self.fig = fig if(isinstance(fig, plt.Figure)) else plt.Figure(**self.fig_kwags)
        self.expCount = 0
        self.file = None
        self.updateExpFile()
        
    def updateExpFile(self):
        fn = LOGger.stamp_process('',[self.theme, *self.stamps, (self.expCount if(self.expCount) else '')],'','','','_',for_file=True)
        self.file = DFP.pathrpt(os.path.join(self.exp_fd, '%s.%s'%(fn, self.save_type)))
        self.expCount += 1
    
    def new(self, file='', layout=None, fig_kwags=None, ret=None, **kwags):
        if(len(get_axes(self.fig))>0):
            if(not self.save(file=file, ret=ret)):
                return False
        layout = layout if(DFP.isiterable(layout)) else self.layout
        fig_kwags = fig_kwags if(isinstance(fig_kwags, dict)) else self.fig_kwags
        self.fig = plt.Figure(**fig_kwags)
        return True
        
    def save(self, file='', ret=None, **kwags):
        file = self.file if(file=='') else file
        end(self.fig, file=file, **kwags)
        if(isinstance(ret, dict)):
            ret['imgfile'] = file
        self.updateExpFile()
        return True
            
    def add(self, file='', ret=None, **kwags):
        if(is_fig_completed(self.fig, layout=self.layout, stamps=self.stamps, **kwags)):
            if(not self.new(file=file, ret=ret)):
                return False
        self.ax = self.fig.add_subplot(*self.layout, len(get_frames(self.fig))+1)
        return True
        
    


#%%
def cv2DataImgFit(data):
    a, b = np.min(getattr(data,'values',data)), np.max(getattr(data,'values',data))
    return DFP.linsptrsfmMachine(a,b,0,255,int,0,255)

def cv2DataImg(data):
    trsfm = cv2DataImgFit(data)
    values = getattr(data, 'values', data)
    values = trsfm(values)
    return values

def cv2imresize(img, scale_factor=1, axis=0):
    if(not m_cv2_import_succeed):
        return None
    if(not DFP.isiterable(scale_factor)):
        scale_factor = DFP.astype(scale_factor, default=1)
        scale_factor = [scale_factor, scale_factor] 
    elif(np.array(scale_factor).shape[0]==1):
        np_scale_factor = np.ones(2)
        np_scale_factor[axis] = scale_factor
        scale_factor = list(tuple(np_scale_factor))
    if(len(img.shape)==3):
        image_height, image_width, _ = img.shape
    else:
        image_height, image_width = img.shape
    new_width = int(image_width * scale_factor[0])
    new_height = int(image_height * scale_factor[1])
    # 縮小圖像
    try:
        img = img.astype(np.uint8)
    except:
        return None
    return cv2.resize(img, (new_width, new_height))

def imgConfiguration(img, target_shape):
    # 確保 arr 是 (a, b, 1) 或 (a, b, 3) 的形狀
    if img.ndim < 2 or img.ndim > 3:
        raise ValueError("img.ndim:%s"%img.ndim)
    imgNdim = dcp(img).astype(np.uint8)
    if len(imgNdim.shape) == 2 and len(target_shape)==3:  
        imgNdim = cv2.cvtColor(imgNdim, cv2.COLOR_GRAY2BGR) if(
            LOGger.mylist(target_shape).get(2,0)!=3) else np.expand_dims(imgNdim, axis=2)
    elif len(imgNdim.shape) == 3 and len(target_shape)==2:  
        imgNdim = cv2.cvtColor(imgNdim, cv2.COLOR_BGR2GRAY)
    elif len(imgNdim.shape) == 3 and len(target_shape) == 3: 
        imgNdim = imgNdim[:,:,:target_shape[2]]
    elif len(imgNdim.shape) == 2 and len(target_shape) == 2: 
        pass
    else:
        print(str(img)[:1000])
        raise ValueError("ERR: target_shape %s with img shape %s"%(str(target_shape), str(img.shape)))
    return imgNdim

def imgResize(img, target_shape=(10,10,3), default_value=np.nan):
    imgNdim = imgConfiguration(img, target_shape=target_shape)
    imgNew = cv2.resize(imgNdim, target_shape[:2])
    if(len(imgNew.shape)==2 and len(target_shape)>2 and LOGger.mylist(target_shape).get(2,0)!=3):
        imgNew = np.expand_dims(imgNew, axis=2)
    return imgNew

def imgCoinSizing(img, target_shape=(10,10,3), default_value=np.nan):
    """
    將形狀為 (a,b) 或 (a,b,1) 或 (a,b,3) 的 numpy array 調整為指定的 (A,B,1) 或 (A,B,3)。
    
    :param arr: 輸入的 numpy array。
    :param target_shape: 目標形狀 (A, B, 1 或 3)。
    :return: 調整後並裁剪到正確形狀的 numpy array。
    """
    
    # target_channels = target_shape[2] if(len(target_shape)>2) else 1
    imgNdim = imgConfiguration(img, target_shape=target_shape)
    for channelIndex in range(len(imgNdim.shape[:2])):
        imgNdimImgShape = (imgNdim.shape[1], imgNdim.shape[0])
        if(target_shape[channelIndex] >= imgNdimImgShape[channelIndex]):
            continue
        resRatio = target_shape[channelIndex]/imgNdimImgShape[channelIndex]
        imgNdim = cv2.resize(imgNdim, tuple(map((lambda x:int(x*resRatio)), imgNdimImgShape)))
    temp = np.pad(imgNdim.shape[:3], (0, 3 - min(3, len(imgNdim.shape))), constant_values=(-1,-1))
    a, b, c = tuple(np.where(temp==-1, np.nan, temp))
    temp = np.pad(target_shape[:3], (0, 3 - min(3, len(target_shape))), constant_values=(-1,-1))
    A, B, C = tuple(np.where(temp==-1, np.nan, temp))
    result = np.full(target_shape, default_value, dtype=imgNdim.dtype)
    
    # 找到需要填充的範圍
    min_a = int(np.nanmin((a, A)))
    min_b = int(np.nanmin((b, B)))
    min_c = int(np.nanmin((c, C, 0)))
    # 填充或裁剪 arr 到 result 中
    if(len(result.shape)==3 and len(imgNdim.shape)==3):
        result[:min_a, :min_b, :min_c] = imgNdim[:min_a, :min_b, :min_c]
    elif(len(result.shape)==2 and len(imgNdim.shape)==2):
        result[:min_a, :min_b] = imgNdim[:min_a, :min_b]
    return result

def file2img(data, editor=None, common_shape=(700,700,3), default_stamps=None, default_value=np.nan, source_fd='.', **kwags):
    '''
    

    Parameters
    ----------
    data : TYPE
        shape (-1,) dtype object(ndarray) str.
    editor : TYPE, optional
        DESCRIPTION. The default is None.
    common_shape : TYPE, optional
        DESCRIPTION. The default is (700,700,3).
    default_stamps : TYPE, optional
        DESCRIPTION. The default is None.
    default_value : TYPE, optional
        DESCRIPTION. The default is np.nan.
    source_fd : TYPE, optional
        DESCRIPTION. The default is '.'.
    **kwags : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    addlog_ = LOGger.execute('addlog', kwags, default=LOGger.addloger(logfile=''), not_found_alarm=False)
    data_temp = []
    default_stamps = default_stamps if(isinstance(default_stamps, list)) else []
    stamps = LOGger.execute('stamps', kwags, default=default_stamps, not_found_alarm=False)
    if(DFP.isiterable(common_shape)):
        if(len(common_shape)<2):
            return None
        elif(len(common_shape)>2):
            common_shape = (*common_shape[:2], (common_shape[2] if(common_shape[2] in (1,3)) else 1))
    if(isinstance(editor, str)):
        editor = method_activation(editor)
    if(hasattr(editor, '__call__') and DFP.isiterable(common_shape)):
        dcp_editor = dcp(editor)
        editor = lambda x:dcp_editor(x,target_shape=common_shape,default_value=default_value)
    dataFor = dcp(getattr(data, 'values', data))
    for i,x in enumerate(dataFor):
        img = dcp(np.full(common_shape, default_value))
        if(LOGger.isinstance_not_empty(x,str)):
            fullpath = os.path.join(source_fd, x)
            if(os.path.exists(fullpath)):
                if(fullpath[-4:] in ('.png','.jpg')):
                    img = cv2imread(fullpath, **kwags)
        elif(isinstance(x,np.ndarray)):
            img = dcp(x)
        if(hasattr(editor, '__call__')):
            img = editor(img)
        if(common_shape):
            if(not hasattr(img, 'shape')):
                img = np.full(common_shape, default_value)
            elif(tuple(img.shape)!=tuple(common_shape)):
                addlog_('shape error:', img.shape, 'while common shape:', common_shape, stamps=stamps)
                img = np.full(common_shape, default_value)
        data_temp.append(img)
    return np.array(data_temp)

def findcontrastcolor(img,pl=None,pu=None,pr=None,pd=None,gray_backgroud_default=None):
    channel = 1 if(len(img.shape)<2) else img.shape[2]
    img = cv2imread(img)
    h,w=img.shape[:2]
    if(set(list(map(type, [pl,pu,pr,pd])))=={int}):
        pl,pu,pr,pd=max(pl,0),max(pu,0),min(pr,w),min(pd,h)
        img = dcp(img[pu:pd, pl:pr])
        # print(pl,pu,pr,pd)
        # print(np.median(img, axis=(0,1)))
    mdn = np.median(img, axis=(0,1))
    if(np.linalg.norm(mdn - np.full(channel, 127.5)) < 63.75):
        return [0,0,255] if(gray_backgroud_default==None) else gray_backgroud_default
    else:
        return 255 - mdn

def cv2imshow(img, scale_factor=1, img_tensor_dim=3, scatters=None, 
              color=None, r=10, colors=None, msize=20, msizes=None, 
              stype=(cv2.MARKER_CROSS if(cv2!=None) else None), img_msk=None,
              stypes=None, th=2, is_label_coord=True, font_size=0.5, text_plank=10, waitKey=0, 
              data_type=np.uint8, **kwags):
    if(not m_cv2_import_succeed):
        return
    text_plank = max(font_size, text_plank)
    img_show = dcp(cv2imread(img, img_tensor_dim=img_tensor_dim))
    img_msk = cv2imread(img, img_tensor_dim=img_tensor_dim)
    img_show = cv2bitwise_and(img_show, img_msk, dont_log=True)
    if(isinstance(img_show, type(None))):   return
    scatters = mylist(scatters if(isinstance(scatters, list)) else [])
    colors = mylist(colors if(isinstance(colors, list)) else [])
    stypes = mylist(stypes if(isinstance(stypes, list)) else [])
    msizes = mylist(msizes if(isinstance(msizes, list)) else [])
    for i,scatter in enumerate(scatters):
        color_ = dcp(colors.get(i,color))
        color_ = findcontrastcolor(img,int(scatter[0]-r),int(scatter[1]-r),int(scatter[0]+r),int(scatter[1]+r)) if(
            isinstance(color_, type(None))) else color_
        cv2.drawMarker(img_show, scatter, color=color_, markerType=stypes.get(i,stype), 
                       markerSize=msizes.get(i,msize))
        if(is_label_coord):
            x_text = scatter[0] if(scatter[0]>0) else text_plank
            y_text = int(scatter[1]-text_plank) if(int(scatter[1]-text_plank)>0) else int(scatter[1]+text_plank)
            stg = '(%s,%s)'%(DFP.parse(scatter[0]), DFP.parse(scatter[1]))
            cv2.putText(img_show, stg, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_size, 
                        color_, 1, cv2.LINE_AA)
    if(scale_factor!=1):    img_show = cv2imresize(img_show, scale_factor=scale_factor)
    window_title=kwags.get('window_title', 'Image')
    try:
        img_show = img_show.astype(data_type)
    except:
        print('convert to %s failed!!!'%str(data_type))
        return 
    cv2.imshow(window_title, img_show)
    cv2.waitKey(waitKey) if(isinstance(waitKey, int)) else None
    
def cv2get_mgrayscale(img, img_msk=None, img_tensor_dim=3, **kwags):
    img = dcp(cv2imread(img, img_tensor_dim=img_tensor_dim))
    img_msk = dcp(cv2imread(img_msk, img_tensor_dim=img_tensor_dim))
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if(not isinstance(img_msk, type(None))):
        non_black_pixels_mask = np.where(np.all(img_msk != [0,0,0], axis=-1))
        selected_pixels = img_g[non_black_pixels_mask]
        gray = np.sum(selected_pixels)
    else:
        gray = np.sum(img_g)
    return gray
    
def cv2VideoEdition(file, wait_key=0, break_alphbet='q', **kwags):
    if(not m_cv2_import_succeed):
        return
    if(not os.path.isfile(file)):
        print('file error:%s'%file)
        return
    cap = cv2.VideoCapture(file)
    while(True):
        ret, fr = cap.read()
        if(not ret or fr==[]):
            break
        cv2imshow(fr, **kwags)
        key = cv2.waitKey(wait_key)
        if(key==ord(break_alphbet)):
            break
    cap.release()
    cv2.destroyAllWindows()

def cv2imread(img, img_tensor_dim=3, dont_log_img_shape_error=True, **kwags):
    if(not m_cv2_import_succeed):
        return None
    addlog = kwags.get('addlog', addloger(logfile=kwags.get('logfile', '')))
    img = dcp(img)
    if(isinstance_not_empty(img, str)):
        if(os.path.exists(img)):
            return cv2.imread(img)
        else:
            addlog("img file doesn't exists:%s"%img)
            return None
    try:
        img = np.array(img).astype(np.uint8)
    except:
        addlog('img convert error!!! type:%s, value:\n%s'%(type(img), str(img)[:200]))
        return None
    if(len(img.shape) in [img_tensor_dim, img_tensor_dim - 1] if(isinstance(img_tensor_dim, int)) else True):
        return img
    else:
        addlog('img shape error: %s<%s !!!'%(img.shape, DFP.parse(img_tensor_dim-1))) if(not dont_log_img_shape_error) else None
        return None
        
def cv2bitwise_and(img1, img2, dont_log=False, **kwags):
    if(not m_cv2_import_succeed):
        return None
    img1 = dcp(cv2imread(img1, **kwags))
    img2 = dcp(cv2imread(img2, **kwags))
    if(isinstance(img1, type(None))):
        return None
    if(isinstance(img2, type(None))):
        return img1
    h, w = img2.shape[:2]    
    img = cv2.bitwise_and(cv2.resize(img1, (w,h), interpolation=cv2.INTER_AREA), img2)
    return img

def cv2_contourcentre(ct, mm=None):
    if(not m_cv2_import_succeed):
        return None
    cx, cy = None, None
    mm = cv2.moments(ct) if(not isinstance(mm, dict)) else mm
    #print(mm)  # mm是字典類型
    # 取得中心點
    if mm['m00']:
        cx = mm['m10'] / mm['m00']
        cy = mm['m01'] / mm['m00']
    return cx, cy

def cv2moments(ct, ret=None, only_convex_pass=True, **kwags):
    old_ret = dcp(ret)
    if(not m_cv2_import_succeed):
        return False if(isinstance(old_ret, dict)) else {}
    ret = ret if(isinstance(ret, dict)) else {}
    kwags['click'].update({cv2.moments.__name__: dt.now()}) if('click' in kwags) else None
    mm = cv2.moments(ct)
    kwags['click'].update({cv2.moments.__name__: (dt.now() - kwags['click'][cv2.moments.__name__]).total_seconds()}) if(
        'click' in kwags) else None
    # 取得中心點
    cx, cy = cv2_contourcentre(ct, mm)
    if(cx==None or cy==None):
        return False if(isinstance(old_ret, dict)) else {}
    ret.update({'cx':cx, 'cy':cy})
    if(not cv2.isContourConvex(ct) if(only_convex_pass) else False):
        return False if(isinstance(old_ret, dict)) else {}
    if(isinstance(old_ret, dict)):
        ret.update({'mm':mm})
        return True                   
    return mm

def cv2draw_ending(img, is_end=True, file='', dont_log=False, not_end_ret=None, wait=True, waittime_ubd=10, **kwags):
    addlog = kwags.get('addlog', addloger(logfile=kwags.get('logfile','')))
    if(is_end):
        if(isinstance_not_empty(file, str)):
            img = cv2imresize(img, scale_factor=kwags.get('scale_factor',1))
            kwags['click'].update({'ending_saveimg': dt.now()}) if('click' in kwags) else None
            if(wait):
                file_, _ext = '.'.join(file.split('.')[:-1]), file.split('.')[-1]
                startTime = dt.now()
                CreateFile('%s_.%s'%(file_, _ext), lambda f:cv2.imwrite(filename=f, img=img))
                while(not os.path.exists('%s_.%s'%(file_, _ext)) and (dt.now() - startTime).total_seconds()<waittime_ubd):
                    continue
                if(os.path.exists(file)):
                    removefile(file)
                os.rename('%s_.%s'%(file_, _ext), file)
            else:
                CreateFile(file, lambda f:cv2.imwrite(filename=f, img=img))
            kwags['click'].update({'ending_saveimg':(dt.now() - kwags['click']['ending_saveimg']).total_seconds()}) if(
                'click' in kwags) else None
        else:
            try:
                cv2imshow(img, **kwags)
            except:
                addlog('ending imshow failed!!!') if(not dont_log) else None
        return 1
    return 0

def cv2immoments(img, threshold_lbd=80, threshold_ubd=240, kernel_size=3, dilate_iterations=3, erode_iterations=2, 
                 color_transplant_circle=(), ret=None, stamps=None, only_convex_pass=True,dont_log=True,is_end=False,
                 file='',dont_showimg=False,judge_color_default=(122, 0, 255),**kwags):
    if(not m_cv2_import_succeed):
        return None
    stamps = stamps if(isinstance(stamps, list)) else []
    ret = ret if(isinstance(ret, dict)) else {}
    img_new = dcp(cv2imread(img))
    edged = cv2imedging(img, threshold_lbd=threshold_lbd, threshold_ubd=threshold_ubd, kernel_size=kernel_size, 
                        dilate_iterations=dilate_iterations, erode_iterations=erode_iterations,**kwags)
    (contours, _) = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    tr_index = 0
    mms = []
    for i,ct in enumerate(contours):
        ret_ct = {}
        mm = cv2.moments(ct)
        cx, cy = cv2_contourcentre(ct, mm)
        cv2.drawContours(img_new, ct, -1, (0, 255, 0), -1)
        if(not cv2transplant(img_new, (cx, cy), r=0, d=1,  judge_color_default=judge_color_default,
                             color_annocircle=color_transplant_circle, ret=ret_ct, tr_stamps=stamps+[tr_index])):
            continue
        img_new = ret_ct['img']
        mms.append(mm)
        tr_index += 1
    ret.update({'mms':mms}) if(is_end) else None
    outcome = cv2draw_ending(img_new, is_end=is_end, file=file, dont_log=dont_log, **kwags)
    return mms if(outcome==0) else (True if(outcome>0) else False)

def cv2imbound(img, threshold_lbd=80, threshold_ubd=240, kernel_size=3, dilate_iterations=3, erode_iterations=2, 
               color_transplant_circle=(), ret=None, bounding_method='standard',judge_color_default=(122, 0, 255),
               stamps=None, only_convex_pass=False,dont_log=True,is_end=False,file='',dont_showimg=False,**kwags):
    if(not m_cv2_import_succeed):
        return None
    addlog = kwags.get('addlog', addloger(logfile=kwags.get('logfile','')))
    stamps = stamps if(isinstance(stamps, list)) else []
    ret = ret if(isinstance(ret, dict)) else {}
    img_new = dcp(cv2imread(img))
    edged = cv2imedging(img, threshold_lbd=threshold_lbd, threshold_ubd=threshold_ubd, kernel_size=kernel_size, 
                        dilate_iterations=dilate_iterations, erode_iterations=erode_iterations,**kwags)
    (contours, _) = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    tr_index = 0
    mms, Bboxes, minAreaboxes = [], [], []
    for i,ct in enumerate(contours):
        ret_ct = {}
        if(not cv2moments(ct, ret=ret_ct, only_convex_pass=only_convex_pass)):
            addlog('contour not convex!!!!', stamps=[
                DFP.parse(ret_ct.get('cx','cx?')), DFP.parse(ret_ct.get('cy','cy?'))]) if(
                not dont_log) else None
            continue
        mm = ret_ct['mm']
        if(np.array(tuple(map(lambda x:DFP.astype(x, default=None)==None, (ret_ct['cx'], ret_ct['cy'])))).any()):
            continue
        cv2.drawContours(img_new, ct, -1, (0, 255, 0), -1)
        if(not cv2transplant(img_new, ct=ct, color_annocircle=color_transplant_circle, 
                             bounding_method=bounding_method, judge_color_default=judge_color_default,
                             ret=ret_ct, tr_stamps=stamps+[tr_index], **kwags)):
            continue
        img_new = ret_ct['img']
        Bboxes.append(ret_ct.get('Bbox')) if('Bbox' in ret_ct) else None
        minAreaboxes.append(ret_ct.get('minAreabox')) if('minAreabox' in ret_ct) else None
        mms.append(mm)
        tr_index += 1
    ret.update({'mms':mms}) if(is_end) else None
    ret.update({'Bboxes':Bboxes}) if(is_end and len(Bboxes)>0) else None
    ret.update({'minAreaboxes':minAreaboxes}) if(is_end and len(minAreaboxes)>0) else None
    ret.update({'img':img_new})
    outcome = cv2draw_ending(img_new, is_end=is_end, file=file, dont_log=dont_log, **kwags)
    if(outcome==0):
        return (Bboxes if(len(Bboxes)>0) else minAreaboxes), mms
    else:
        return True if(outcome>0) else False

def cv2mark_from_one_to_another_along_axis(img, img_todraw=None, target=None, backgroud=None, 
                                           target_todraw=None, threshold_colordist=None, threshold_loc=5, 
                                           axis=0, channel=3, selected_points=None, **kwags):
    if(axis==1):
        img_t = img.transpose((1,0,2))
        img_todraw = img_todraw.transpose((1,0,2)) if(isinstance(img_todraw, np.ndarray)) else None
        img_todraw = cv2mark_from_one_to_another_along_axis(
            img_t, img_todraw, target=target, target_todraw=target_todraw, 
            threshold_colordist=threshold_colordist, threshold_loc=threshold_loc,
            backgroud=backgroud, axis=0, channel=channel, selected_points=selected_points)
        if(isinstance(img, np.ndarray)):
            img_todraw = img_todraw.transpose((1,0,2))
        return img_todraw
    elif(axis!=0):
        return None
    addlog = kwags.get('addlog', addloger(logfile=kwags.get('logfile','')))
    backgroud = backgroud[:channel] if(DFP.isiterable(backgroud)) else np.median(img, axis=(0,1))
    target = target[:channel] if(DFP.isiterable(target)) else [0,0,255]
    target_todraw = target_todraw[:channel] if(DFP.isiterable(target_todraw)) else findcontrastcolor(img, backgroud)
    img_todraw = img_todraw if(isinstance(img_todraw, np.ndarray)) else np.full(img.shape, backgroud)
    need_selected_points = True if(isinstance(selected_points, list)) else False
    
    log_counter = kwags.get('log_counter', {})
    distances_at_x_0 = np.sqrt(np.sum((img[:, 0, :].astype(np.float32) - target) ** 2, axis=-1))
    closet_distance = np.min(distances_at_x_0)
    y_candidates = np.where(distances_at_x_0==closet_distance)[0]
    print('y_candidates', y_candidates)
    for y_pre in y_candidates:
        selected_points_y = []
        img_todraw[y_pre, 0, :] = target_todraw
        selected_points_y.append((0,y_pre) if(axis==0) else (y_pre,0)) if(need_selected_points) else None
        for x in range(1, img.shape[1]):
            # 提取當前行
            row = img[max(0,y_pre - threshold_loc):min(y_pre + threshold_loc,img.shape[0]), x, :]
            distances = np.sqrt(np.sum((row.astype(np.float32) - target) ** 2, axis=-1))
            min_distance = np.min(distances) 
            y_cand_locs = np.where(min_distance == distances)[0] #假設有threshold_loc=10，且選到100,101,102,105,120可達最小誤差1
            y_in_seg = int(np.median(y_cand_locs))
            y = y_in_seg + max(0,y_pre - threshold_loc)
            if(threshold_colordist):
                if(distances[y_in_seg]>threshold_colordist):
                    addlog('colordist too large:%s(>%s) pos:%d,%d'%(
                        DFP.parse(distances[y_in_seg]), DFP.parse(threshold_colordist), x, y), 
                        log_counter=log_counter, log_counter_stamp='debug')
                    continue
            # 儲存最接近目標顏色的座標
            img_todraw[y, x, :] = target_todraw
            selected_points_y.append((x,y) if(axis==0) else (y,x)) if(need_selected_points) else None
            y_pre = dcp(y)
        selected_points.append(selected_points_y) if(need_selected_points) else None
    return img_todraw

def cv2transplant_anno_feature_scenario(rect, bw, bh, anno_features=None, bounding_method=None, ret=None, **kwags):
    ret = ret if(isinstance(ret, dict)) else {}
    if(isinstance(anno_features, str)):
        if(anno_features=='standard' and bounding_method=='standard'):
            anno_features = ['area','bboxarea','bboxabr','bboxab']
        elif(anno_features=='standard' and bounding_method=='minarea'):
            anno_features = ['area','minrabr','minrab','angle']
        elif(anno_features=='all' and isinstance_not_empty(bounding_method)):
            anno_features = ['bboxabr','bboxab','minarea','minrabr','angle']
    anno_features = {v:None for v in anno_features} if(isinstance(anno_features, list)) else anno_features
    anno_features = anno_features if(isinstance(anno_features, dict)) else {}
    for k,v in kwags.items():
        if(k in anno_features):
            anno_features.update({'area':v})
    bh,bw = max(rect[1][0], rect[1][1]), min(rect[1][0], rect[1][1])
    if('minrabr' in anno_features):
        minrabr = bh/bw if(bw!=0) else np.inf
        anno_features.update({'minrabr':DFP.parse(minrabr)})
        ret.update({'minrabr':minrabr})
    if('minrab' in anno_features):
        minrab = (bh, bw)
        anno_features.update({'minrab':','.join(tuple(map(lambda x:DFP.parse(x,digit=0), minrab)))})
    if('angle' in anno_features):
        angle = (180-rect[2]) if(rect[1][0] > rect[1][1]) else 90 - rect[2]
        anno_features.update({'angle':DFP.parse(angle)})
        ret.update({'angle':angle})
    if('bboxabr' in anno_features):
        bboxabr = bh/bw if(bw!=0) else np.inf
        anno_features.update({'bboxabr':DFP.parse(bboxabr)})
        ret.update({'bboxabr':bboxabr})
    if('bboxab' in anno_features):
        bboxab = (bh,bw)
        anno_features.update({'bboxab':','.join(tuple(map(lambda x:DFP.parse(x,digit=0), bboxab)))})
    return anno_features

def cv2transplant_judge1(bw, bh, angle, bw_lbd=20, bh_lbd=200, mrw_lbd=12, mrh_lbd=200, angle_central=90, angle_thrsh=20,
                         method='standard', bounding_method='standard', **kwags):
    if(bounding_method.lower()=='standard'):
        return bw>bw_lbd and bh>bh_lbd
    if(bounding_method.lower()=='minarea'):
        return bh>mrh_lbd and bw>mrw_lbd and np.abs(angle-angle_central) < angle_thrsh

def cv2transplant_color_leveling(highlight_color=(0, 0, 255), dimlight_color=(190,40,0), 
                                 bw_lbd=20, bh_lbd=200, mrw_lbd=12, mrh_lbd=200, angle_thrsh=20, ret=None, 
                                 mrw_lbd_dr=None, mrh_lbd_dr=None, 
                                 method='standard', bounding_method='standard', judge_method=cv2transplant_judge1, **kwags):
    ret = ret if(isinstance(ret, dict)) else {}
    if(judge_method == None):
        return True
    if(bounding_method.lower()=='standard'):
        if('Bbox' in kwags):
            Bbox = kwags['Bbox']
            bx, by, bw, bh = tuple(Bbox)
            inputs = kwags.get('judge_inputs',np.array([[bx, by, bw, bh, 0]]))
            judgement = judge_method(inputs=inputs, bw=bw, bh=bh, angle='angle', bw_lbd=bw_lbd, bh_lbd=bh_lbd, 
                                     bounding_method=bounding_method, **kwags)
            ret.update({'judgement':judgement})
            ret.update({'color':highlight_color if(judgement) else dimlight_color})
            return True
    elif(bounding_method.lower()=='minarea'):
        if('minArect' in kwags):
            minArect = kwags['minArect']
            bh,bw = max(minArect[1][0], minArect[1][1]), min(minArect[1][0], minArect[1][1])
            bx, by = 0,0
            angle = kwags.get('angle', minArect[2])
            inputs = kwags.get('judge_inputs',np.array([[bx, by, bw, bh, angle]]))
            judgement = judge_method(inputs=inputs, bw=bw, bh=bh, mrw_lbd=mrw_lbd, mrh_lbd=mrh_lbd, angle_thrsh=angle_thrsh, 
                                     bounding_method=bounding_method, **kwags)
            ret.update({'judgement':judgement})
            ret.update({'judgement_draw':judge_method(
                inputs=inputs, bw=bw, bh=bh, mrw_lbd=mrw_lbd_dr, mrh_lbd=mrh_lbd_dr, angle_thrsh=angle_thrsh, 
                bounding_method=bounding_method, **kwags) if(mrw_lbd_dr!=None and mrh_lbd_dr!=None) else judgement})
            ret.update({'color':highlight_color if(ret.get('judgement_draw')) else dimlight_color})
            return True
    return False

def cv2crop_and_pad(image, x, y, w, h, pad_color=[0, 0, 0],d=1):
    if(not m_cv2_import_succeed):
        return
    # 获取原图的高度和宽度
    H, W, _ = image.shape
    
    # 初始化输出图像，用pad_color填充
    h_ = int(np.ceil(h/d))
    w_ = int(np.ceil(w/d))
    output = np.full((h_, w_, 3), pad_color, dtype=np.uint8)
    x_end_ = min(x+w_, W)
    y_end_ = min(y+h_, H)
    
    # 计算源图像和目标位置的重叠区域
    x_start = max(x, 0)
    y_start = max(y, 0)
    x_end = min(x+w, W)
    y_end = min(y+h, H)
    
    # 计算在输出图像中的起始位置
    out_x_start = x_start - x
    out_y_start = y_start - y
    
    # 进行裁剪和填充操作
    try:
        output[out_y_start:out_y_start+(y_end_-y_start), out_x_start:out_x_start+(x_end_-x_start)] = image[y_start:y_end:d, x_start:x_end:d]
        return output
    except:
        print('crop output')
        print(output.shape)
        print(out_y_start,out_y_start+(y_end_-y_start), out_x_start,out_x_start+(x_end_-x_start))
        print('crop source img')
        print(image.shape)
        print(y_start,y_end, x_start,x_end, d)

#TODO:rect_fit_in_bigone
def rect_fit_in_bigone():
    pass
    
def scale_lineartransform(x, xlbd=0, xubd=201, ylbd=0, yubd=100, reverse=False):
    return int(yubd*(min(1,max(0,((1-(x-xlbd)/(xubd-xlbd)) if(reverse) else (x-xlbd)/(xubd-xlbd))+xlbd)))) + ylbd

def cv2transplant(img, centre=None, r=10, d=1, ct=None, bounding_method='', bgr_sample=None, 
                  color_annocircle=None, judge_color_default=(122, 0, 255), judge_method=cv2transplant_judge1, 
                  ret=None, th=2, font_size=1.2, do_fit_ellipse=False,
                  tr_stamps=None, img_sample=None, bgr_need_flatten=False, image_transpose=True, 
                  anno_stg=None, anno_features=None, anno_font_size=0.5, drawing_mode='', 
                  tr_use_thread=False, show_every_contours=True, **kwags):
    if(not m_cv2_import_succeed):
        return False
    ret = ret if(isinstance(ret, dict)) else {}
    img = cv2imread(img)
    judge_color = dcp(judge_color_default)
    color_annocircle = (122, 122, 255) if(
        len(color_annocircle)<3 if(isinstance(color_annocircle, tuple)) else False) else color_annocircle
    img_sample = img_sample if(not isinstance(img_sample, type(None))) else dcp(img)
    img_sample_t = img_sample.transpose(1,0,2) if(image_transpose) else img_sample
    w,h = img_sample_t.shape[:2]
    if(not isinstance(centre, type(None))):
        cx, cy = tuple(centre[:2])
    else:
        ret_mm = {}
        if(not cv2moments(ct, ret=ret_mm)):
            return False
        cx, cy = ret_mm['cx'], ret_mm['cy']
    Ea,Eb,Ec,Ed = int(cx-r),int(cx+r),int(cy-r),int(cy+r)
    if(isinstance(ct, type(None))):
        if(cx-r<0):
            cx = r
        elif(cx+r>=w):
            cx = w - r - 1
        if(cy-r<0):
            cy = r
        elif(cy+r>=h):
            cy = h - r - 1
        ea,eb,ec,ed = int(cx-r),int(cx+r),int(cy-r),int(cy+r)
    else:
        bx, by, bw, bh = cv2.boundingRect(ct)
        if(bx<0):
            ea = 0
            eb = bw
        elif(bx>w):
            eb = w
            ea = w - bw
        else:
            ea = bx
            eb = bx + bw
        if(by<0):
            ec = 0
            ed = bh
        elif(by>h):
            ed = h
            ec = h - bh
        else:
            ec = by
            ed = by + bh
        # ea,eb,ec,ed = bx, bx + bw, by, by + bh
        rect = cv2.minAreaRect(ct)
        box = cv2.boxPoints(rect)
        ret.update({'Bbox':[bx, by, bw, bh] if(bounding_method.lower()=='standard') else [*rect[0], *rect[1]]})
        ret.update({'minAreabox':box})
        ret.update({'minArect':rect})
        anno_features = cv2transplant_anno_feature_scenario(
            rect, bw, bh, anno_features=anno_features, bounding_method=bounding_method, ret=ret, **kwags)
        cv2transplant_color_leveling(bounding_method=bounding_method, ret=ret, **ret, **kwags)
        judge_color = ret.get('color', judge_color_default)
        kwags.update({'box_mina':box})
    if(isinstance(bgr_sample, list)):
        a_sample = cv2crop_and_pad(img_sample,Ea,Ec,Eb-Ea,Ed-Ec,d=d)
        a_sample = a_sample.transpose(1,0,2) if(image_transpose) else a_sample
        a_sample_gray = cv2.cvtColor(a_sample, cv2.COLOR_BGR2GRAY)
        ret.update({'brightness':np.median(a_sample_gray)})
        bgr_sample.append(a_sample if(not bgr_need_flatten) else a_sample.flatten())
    if(kwags.get('ImageServant')!=None):
        params = {k:v for k,v in locals().items() if not k in [cv2transplant.__name__,'kwags'] and k.find('img')==-1}
        params.update({k:v for k,v in kwags.items() if not k in ['ImageServant']})
        params.update({'color':params.pop('judge_color', judge_color)})
        params.update({'box_mina':params.pop('box', None)})
        source_stamp = dcp(kwags['source_stamp'])
        kwags['ImageServant'].add_auxparams(params=params, source_stamp=source_stamp)
        return True
    return cv2draw_auximage(img, ea, eb, ec, ed, centre=centre, r=r, d=d, ct=ct, ret=ret, 
                  bounding_method=bounding_method, anno_features = anno_features, 
                  color_annocircle=color_annocircle, color_default=judge_color_default, 
                  drawing_mode=drawing_mode, th=th, tr_stamps=tr_stamps, 
                  font_size=font_size, anno_font_size=anno_font_size, anno_stg=anno_stg, **kwags)
        
def cv2draw_auximage(img, ea, eb, ec, ed, centre=None, r=20, d=1, ct=None, ret=None, bounding_method='', 
                  box_mina=None, anno_features = None, color_annocircle=None, color_default=(122, 0, 255), 
                  drawing_mode='', th=2, tr_stamps=None, font_size=0.5, anno_font_size=0.5, anno_stg=None, 
                  show_every_contours=True, **kwags):
    if(not m_cv2_import_succeed):
        return False
    tr_stamps = tr_stamps if(isinstance(tr_stamps, list)) else []
    tr_stamp = stamp_process('',tr_stamps,'','','',' ')
    anno_features = anno_features if(isinstance(anno_features, dict)) else {}
    anno_stg = anno_stg if(isinstance(anno_stg, str)) else ''
    ret = ret if(isinstance(ret, dict)) else {}
    color = color_default
    h,w = img.shape[:2]
    cv2.drawContours(img, ct, -1, (0, 255, 0), -1) if(show_every_contours) else None
    if(isinstance(ct, type(None))):
        if(d==1):
            cv2.rectangle(img, (ea-1, ec-1), (eb+1, ed+1), color_annocircle, thickness=th) if(
                len(color_annocircle)>2 if(DFP.isiterable(color_annocircle)) else False) else None
        else:
            cx, cy = tuple(centre[:2])
            for i in range(-r,r,d):
                for j in range(-r,r,d):
                    x=int(cx+i)
                    y=int(cy+j)
                    if x < 0 or y < 0: 
                        continue
                        # raise IndexError # check negative index
                    cv2.circle(img, (x, y), 1, color_annocircle, cv2.FILLED) if(
                        len(color_annocircle)>2 if(DFP.isiterable(color_annocircle)) else False) else None
    elif(bounding_method.lower()=='standard'):
        if(ret['judgement_draw'] if(drawing_mode=='thrift') else True):
            cv2.rectangle(img, (ea-1, ec+1), (eb+1, ed-1), color, thickness=th) if(
                len(color)>2 if(DFP.isiterable(color)) else False) else None
    elif(bounding_method.lower()=='minarea'):
        box_int = np.int0(box_mina)
        if(ret['judgement_draw'] if(drawing_mode=='thrift') else True):
            cv2.drawContours(img, [box_int], 0, color, th) if(
                len(color)>2 if(DFP.isiterable(color)) else False) else None
    text_plank = max(font_size*20, (r if(isinstance(ct, type(None))) else 0))
    if(drawing_mode!='thrift' and isinstance_not_empty(tr_stamp, str)):
        x_text = ea
        y_text = int(ed+text_plank) if(int(ed+text_plank)<h) else int(ec-text_plank)
        cv2.putText(img, tr_stamp, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_size, 
                    (int(255), int(255), int(255)), 1, cv2.LINE_AA)
    if(anno_features):
        anno_feature_stg = stamp_process('',anno_features)
        anno_stg = stamp_process('',[anno_stg, anno_feature_stg],'','','','\n')
    if(ret.get('judgement_draw',False) and isinstance_not_empty(anno_stg, str)):
        x_text = ea
        y_text = int(ed+2*text_plank) if(int(ed+2*text_plank)<h) else int(ec-2*text_plank)
        cv2.putText(img, anno_stg, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, anno_font_size, 
                    (int(255), int(255), int(255)), 1, cv2.LINE_AA)
    ret.update({'img':img})
    return True

def cv2imedging(img, img_msk=None, threshold_lbd=80, threshold_ubd=240, kernel_size=3, dilate_iterations=3, erode_iterations=2,
                guassian_blur_kernal=None, is_equalizing=False, dilate_then_erode=True, **kwags):
    if(not m_cv2_import_succeed):
        return None
    img_msked = dcp(cv2bitwise_and(img,img_msk))
    if(isinstance(DFP.astype(guassian_blur_kernal), float)):
        guassian_blur_kernal = int(DFP.astype(guassian_blur_kernal)//1)
        guassian_blur_kernal = (guassian_blur_kernal,guassian_blur_kernal)
    guassian_blur_kernal = guassian_blur_kernal[:2] if(DFP.isiterable(guassian_blur_kernal)) else None
    kwags['click'].update({cv2imedging.__name__: dt.now()}) if('click' in kwags) else None
    img_blurred = cv2.GaussianBlur(img_msked, guassian_blur_kernal, 0) if(not isinstance(guassian_blur_kernal, type(None))) else img_msked
    img_gray = cv2.cvtColor(img_blurred, cv2.COLOR_RGB2GRAY) # 轉灰階
    img_gray_eqd = cv2.equalizeHist(img_gray) if(is_equalizing) else img_gray
    edged = cv2.Canny(img_gray_eqd,threshold_lbd,threshold_ubd,apertureSize = kernel_size)
    #閉合
    if(dilate_then_erode):
        edged = cv2.dilate(edged, None, iterations=dilate_iterations) if(isinstance_not_empty(dilate_iterations, int)) else edged
        edged = cv2.erode(edged, None, iterations=erode_iterations) if(isinstance_not_empty(erode_iterations, int)) else edged
    else:
        edged = cv2.erode(edged, None, iterations=erode_iterations) if(isinstance_not_empty(erode_iterations, int)) else edged
        edged = cv2.dilate(edged, None, iterations=dilate_iterations) if(isinstance_not_empty(dilate_iterations, int)) else edged
    kwags['click'].update({cv2imedging.__name__: (dt.now() - kwags['click'][cv2imedging.__name__]).total_seconds()}) if(
        'click' in kwags) else None
    return edged

def cv2imthredging(img, img_msk=None, threshold=80, threshold_ubd=240, dilate_iterations=3, erode_iterations=2,
                guassian_blur_kernal=None, is_equalizing=False, dilate_then_erode=True, **kwags):
    if(not m_cv2_import_succeed):
        return None
    img_msked = dcp(cv2bitwise_and(img,img_msk))
    if(isinstance(DFP.astype(guassian_blur_kernal), float)):
        guassian_blur_kernal = int(DFP.astype(guassian_blur_kernal)//1)
        guassian_blur_kernal = (guassian_blur_kernal,guassian_blur_kernal)
    guassian_blur_kernal = guassian_blur_kernal[:2] if(DFP.isiterable(guassian_blur_kernal)) else None
    kwags['click'].update({cv2imthredging.__name__: dt.now()}) if('click' in kwags) else None
    img_blurred = cv2.GaussianBlur(img_msked, guassian_blur_kernal, 0) if(not isinstance(guassian_blur_kernal, type(None))) else img_msked
    img_gray = cv2.cvtColor(img_blurred, cv2.COLOR_RGB2GRAY) # 轉灰階
    img_gray_eqd = cv2.equalizeHist(img_gray) if(is_equalizing) else img_gray
    #閉合
    if(dilate_then_erode):
        edged = cv2.dilate(img_gray_eqd, None, iterations=dilate_iterations) if(isinstance_not_empty(dilate_iterations, int)) else img_gray_eqd
        edged = cv2.erode(img_gray_eqd, None, iterations=erode_iterations) if(isinstance_not_empty(erode_iterations, int)) else img_gray_eqd
    else:
        edged = cv2.erode(img_gray_eqd, None, iterations=erode_iterations) if(isinstance_not_empty(erode_iterations, int)) else img_gray_eqd
        edged = cv2.dilate(img_gray_eqd, None, iterations=dilate_iterations) if(isinstance_not_empty(dilate_iterations, int)) else img_gray_eqd
    _, edged = cv2.threshold(img_gray_eqd, threshold, 255, cv2.THRESH_BINARY)
    kwags['click'].update({cv2imthredging.__name__: (dt.now() - kwags['click'][cv2imthredging.__name__]).total_seconds()}) if(
        'click' in kwags) else None
    return edged

def cv2findContours_drawing(img, contours, bgr_sample=None, r=20, d=1, stamps=None, ret=None, file='', 
                            bgr_need_flatten = False, color_transplant_circle=None, only_convex_pass=False, 
                            dont_log=True, dont_showimg=False, do_fit_ellipse=False, color_ellipse=None, 
                            th_ellipse=2, draw_box_bound=True, save_and_show=False, ret_keys=None, is_fast_judgement=True,
                            judge_color_default=(122, 0, 255), judge_method=cv2transplant_judge1, ct_area_lbd=2500, 
                            img_raw=None, **kwags):
    addlog = kwags.get('addlog', addloger(logfile=kwags.get('logfile','')))
    stamps = stamps if(isinstance(stamps, list)) else []
    stamp = stamp_process('', stamps, '','','',' ')
    ret = ret if(isinstance(ret, dict)) else {}
    ret_keys = ret_keys if(isinstance(ret_keys, list)) else [
        'ct', 'centre', 'brightness', 'area', 'ellipse_fit_feature','Bbox','minAreabox','angle']
    now = kwags.get('now', dt.now())
    img_msked_aux = dcp(img)
    #list很慢要改善
    centres, brightnesses, contours_ret, areas, ellipse_fit_features, bbox_features, mina_features, angle_features, tr_index=[], [], [], [], [], [], [], [], 0
    judgements = []
    kwags['click'].update({'contours_saveimg': 0}) if(
        not 'contours_saveimg' in kwags['click'] if('click' in kwags) else False) else None
    kwags['click'].update({'contours_scenario': dt.now()}) if('click' in kwags) else None
    ct_area_lbd = max(0, DFP.astype(ct_area_lbd, default=0))
    for i,ct in enumerate(contours):
        area = cv2.contourArea(ct)
        if(ct_area_lbd):
            if(area<=ct_area_lbd):
                continue
        ret_ct = {}
        if(not cv2moments(ct, ret=ret_ct, only_convex_pass=only_convex_pass, **ret_ct)):
            addlog('contour not convex!!!!', stamps=[DFP.parse(ret_ct['cx']), DFP.parse(ret_ct['cy'])]) if(
                not dont_log) else None
            continue
        centre = (ret_ct['cx'], ret_ct['cy'])
        if(np.array(tuple(map(lambda x:DFP.astype(x, default=None)==None, centre))).any()):
            continue
        kwags.update({'ct':ct}) if(draw_box_bound) else kwags.update({'ct':None})
        kwags.update({'area':area})
        if(not cv2transplant(img_msked_aux, centre, r=r, d=d, bgr_sample=bgr_sample, 
                             color_annocircle=color_transplant_circle, ret=ret_ct, 
                             tr_stamps=stamps+[tr_index], source_stamp=stamp,
                             img_sample=img, bgr_need_flatten=bgr_need_flatten,
                             judge_method=judge_method, judge_color_default=judge_color_default,
                             **kwags)):
            continue
        judgements.append(ret_ct.get('judgement'))
        if(kwags.get('ImageServant')==None):    img_msked_aux = ret_ct['img']
        # if(is_fast_judgement):
        #     if(ret_ct.get('judgement', False)):
        #         break
        if(do_fit_ellipse):
            if(len(ct)>5):
                ellipse = cv2.fitEllipse(ct)
                # ellipse: centre_x, centre_y, length_major_axis, length_minor_axis, angle
                ellipse_fit_features.append(mylist(ellipse).get_all())
                color_ellipse = color_ellipse if(
                    np.array(color_ellipse).shape[0]>2 if(DFP.isiterable(color_ellipse)) else False) else (255,200,60)
                cv2.ellipse(img_msked_aux, ellipse, color_ellipse, th_ellipse) if(isinstance_not_empty(th_ellipse, int)) else None
            else:
                ellipse_fit_features.append((np.nan, np.nan, np.nan, np.nan, np.nan))
        (brightnesses.append(ret_ct['brightness']) if('brightness' in ret_ct) else None) if('brightness' in ret_keys) else None
        centres.append(centre) if('centre' in ret_keys) else None
        areas.append(area if(ret_ct.get('judgement')) else 0) if('area' in ret_keys) else None
        contours_ret.append(ct) if('ct' in ret_keys) else None
        (bbox_features.append(ret_ct['Bbox']) if(isinstance_not_empty(ret_ct.get('Bbox'), list)) else None) if('Bbox' in ret_keys) else None
        (mina_features.append(ret_ct['minAreabox']) if(not isinstance(ret_ct.get('minAreabox'), type(None))) else None) if('minAreabox' in ret_keys) else None
        (angle_features.append(ret_ct.get('angle',np.nan)) if(isinstance_not_empty(ret_ct.get('angle'), list)) else None) if('angle' in ret_keys) else None
        tr_index += 1
    judgement = np.array(judgements).any()
    ret.update({'area':areas})
    ret.update({'judgement':judgement})
    if(kwags.get('AttrAgent')!=None):
        picfile_stg, picfile_stg_raw = '', ''
        if(kwags.get('ImageServant')!=None):
            judgement_stg = 'NG' if(judgement) else ''
            fn = stamp_process('',[stamp, judgement_stg],'','','','_',for_file=True)
            picfile = os.path.join(kwags['ImageServant'].exp_fd, now.strftime('%Y%m%d'), '%s.jpg'%fn)
            picfile_stg = make_hyperlink(picfile, kwags['AttrAgent'].exp_fd, supervise_root=kwags.get('supervise_root'))
            fn = stamp_process('',[stamp, judgement_stg, 'RAW'],'','','','_',for_file=True)
            picfile_raw = os.path.join(kwags['ImageServant'].exp_fd, now.strftime('%Y%m%d'), '%s.jpg'%fn)
            picfile_stg_raw = make_hyperlink(picfile_raw, os.path.join(kwags['AttrAgent'].exp_fd,'.'), supervise_root=kwags.get('supervise_root'))
        kwags['AttrAgent'].update(stamp, 'runtime', now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], 'judgement', judgement,
                                  'picfile', picfile_stg, 'picfile_raw', picfile_stg_raw)
    if(kwags.get('ImageServant')!=None):
        kwags['ImageServant'].add_auxparams(dcp(judgement), source_stamp=stamp, stamp='judgement')
        kwags['ImageServant'].add_auxparams(kwags.get('memo_stamps'), source_stamp=stamp, stamp='memo_stamps') if(kwags.get('memo_stamps')) else None
        kwags['ImageServant'].add_img(img_raw, tm=now, stamp=stamp)
        kwags['click'].update({'contours_scenario': (dt.now() - kwags['click']['contours_scenario']).total_seconds()}) if(
            'click' in kwags) else None
        return
    ret.update({'judgements':judgements}) if(isinstance_not_empty(judgements, list)) else None
    ret.update({'ellipse_fit_feature':ellipse_fit_features}) if(isinstance_not_empty(ellipse_fit_features, list)) else None
    ret.update({'brightness':brightnesses}) if(isinstance_not_empty(brightnesses, list)) else None
    ret.update({'centres':centres})
    ret.update({'contours':contours_ret}) if(isinstance_not_empty(contours_ret, list)) else None
    ret.update({'img':img_msked_aux}) if(kwags.get('ImageServant')==None) else None
    ret.update({'Bbox':bbox_features}) if(isinstance_not_empty(bbox_features, list)) else None
    if(isinstance_not_empty(mina_features, list)):
        mina_features = np.array([[box, *DFP.edgelength_of_rectlike(box)[:2]] for box in mina_features])
        ret.update({'minbox':mina_features})
        ret.update({'bounding_angle':angle_features}) if(isinstance_not_empty(angle_features, list)) else None
    kwags['click'].update({'contours_scenario': (dt.now() - kwags['click']['contours_scenario']).total_seconds()}) if(
        'click' in kwags) else None
    if(isinstance_not_empty(file, str)):
        kwags['click'].update({'contours_saveimg': dt.now()}) if(
            'click' in kwags) else None
        ext_file = file[file.rfind('.'):]
        file = file.replace(ext_file,'_NG%s'%ext_file) if(ret.get('judgement', False)) else file
        CreateFile(file, lambda f:cv2.imwrite(filename=f, img=img_msked_aux))
        kwags['click'].update({'contours_saveimg': (dt.now() - kwags['click']['contours_saveimg']).total_seconds()}) if(
            'click' in kwags) else None
        cv2imshow(img_msked_aux, **kwags) if(save_and_show) else None
    else:
        cv2imshow(img_msked_aux, **kwags) if(not dont_showimg) else None

def cv2findContours(img, maskimg=None, threshold_lbd=80, threshold_ubd=240, kernel_size=3, dilate_iterations=3, erode_iterations=2, 
                    is_end=True, ret=None, bgr_sample=None, r=20, d=1, dont_log=True, dont_showimg=False, bgr_need_flatten=False,
                    stamps=None, only_convex_pass=False, color_transplant_circle=None, file='', do_fit_ellipse=False, 
                    color_ellipse=None, th_ellipse=2, draw_box_bound=True, draw_use_thread=False, threshold_bd=60, 
                    edging_method='thredging', mode=cv2.CHAIN_APPROX_NONE, method=cv2.RETR_EXTERNAL, canvas=None, **kwags):
    if(not m_cv2_import_succeed):
        return False if(is_end) else None
    stamps = stamps if(isinstance(stamps, list)) else []
    ret = ret if(isinstance(ret, dict)) else {}
    img_msked = dcp(cv2bitwise_and(img,maskimg))
    if(edging_method=='canny'):
        edged = cv2imedging(img_msked, threshold_lbd=threshold_lbd, threshold_ubd=threshold_ubd, kernel_size=kernel_size, 
                            dilate_iterations=dilate_iterations, erode_iterations=erode_iterations,**kwags)
    else:
        edged = cv2imthredging(img_msked, threshold=threshold_bd, 
                               dilate_iterations=dilate_iterations, erode_iterations=erode_iterations,**kwags)
    
    kwags['click'].update({cv2.findContours.__name__: dt.now()}) if('click' in kwags) else None
    (contours, _) = cv2.findContours(edged, mode=mode, method=method)
    kwags['click'].update({cv2.findContours.__name__: (dt.now() - kwags['click'][cv2.findContours.__name__]).total_seconds()}) if(
        'click' in kwags) else None
    print('no contours be found!!') if(len(contours)==0) else None
    if(isinstance(canvas, np.ndarray)):
        if(canvas.shape==img.shape):
            img_msked = canvas
    cv2findContours_drawing(img_msked, contours, bgr_sample=bgr_sample, r=r, d=d, stamps=stamps, ret=ret, file=file, 
                            bgr_need_flatten = bgr_need_flatten, color_transplant_circle=color_transplant_circle, 
                            only_convex_pass=only_convex_pass, dont_log=dont_log, dont_showimg=dont_showimg, 
                            do_fit_ellipse=do_fit_ellipse, color_ellipse=color_ellipse, img_raw=img,
                            th_ellipse=th_ellipse, draw_box_bound=draw_box_bound, **kwags)
    if(not is_end):
        return contours
    return True

def cv2clusterColors(img, contours, eps=0.14, min_samples=10, is_end=False, maskColor=(0,0,0), deleteColors=None, ret=None, 
                     dont_log=True, file=None, **kwags):
    colors = np.zeros((0,3))
    pos = np.zeros((0,2))
    if(isinstance(maskColor, type(None))):
        imgColorsUniq, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
        if(imgColorsUniq.shape[0]==256):
            maskColor = imgColorsUniq[np.argmin(counts)]
        elif(not (0,0,0) in imgColorsUniq):
            maskColor = (0,0,0)
        else:
            maskColor = np.median(255 - imgColorsUniq.astype(np.uint8), axis=0)
        print('maskColor', maskColor)
    canvas = np.full(img.shape, np.array(maskColor).astype(np.uint8))
    # 創建一個結構元素（kernel），可以調整大小來改變效果
    kernel = np.ones((2, 2), np.uint8)
    deleteColors = deleteColors if(isinstance(deleteColors, list)) else [255 - np.array(maskColor)]
    print('contours %d'%len(contours))
    for i, contour in enumerate(contours):
        coords = dcp(contour[:,0,:])
        canvas_temp = np.full(img.shape, np.array(maskColor).astype(np.uint8))
        
        canvas_temp[coords[:,1]%canvas_temp.shape[0], coords[:,0]%canvas_temp.shape[1], :] = 255 - np.array(maskColor)    #初始化輪廓
        # 使用膨脹操作來擴大輪廓
        canvas_dilated = cv2.dilate(canvas_temp, kernel, iterations=1)
        # 或者使用閉操作來填補輪廓內的洞
        canvas_closed = cv2.morphologyEx(canvas_dilated, cv2.MORPH_CLOSE, kernel)
        
        canvas_msked = np.where(np.all(canvas_closed==maskColor, axis=-1)[:, :, np.newaxis], canvas_closed, img)
        
        condition = np.logical_not(np.all(canvas_msked==maskColor, axis=-1)) #dcp(np.sqrt(np.sum((canvas_msked-np.full(canvas_msked.shape, maskColor))**2, axis=2))>0)
        for deleteColor in deleteColors:
            condition &= np.sqrt(np.sum((canvas_msked-np.full(
                canvas_msked.shape, np.array(deleteColor).astype(np.uint8)))**2, axis=2))>10
        pos_adding = np.array(tuple(zip(*np.where(condition))))
        if(pos_adding.shape[0]==0):
            continue
        colors_adding = canvas_msked[condition,:]
        pos = np.append(pos, pos_adding, axis=0)
        colors = np.append(colors, colors_adding, axis=0)
    colorsUniq = np.unique(colors, axis=0)
    if(colorsUniq.shape[0]>10**5):
        print('too many colors: %s'%colorsUniq.shape[0])
        outcome = cv2draw_ending(canvas, is_end=is_end, file=None, dont_log=dont_log, **kwags)
        if(outcome==0):
            return None, None
        else:
            return False
    
    # 重新塑形 T 變為 (a*b, 3)
    img_flat = img.reshape(-1, 3)
    mask, colorClassSet = tuple(DFP.dbscaning(colorsUniq, eps=eps, min_samples=min_samples).values())
    colorClassMap = {}
    for colorClass in colorClassSet[:-1]:#[:1]:
        colorsInClass = colorsUniq[mask==colorClass].astype(np.uint8)
        colorsInClassMedian = np.nanmedian(colorsInClass, axis=0).astype(np.uint8)
        # 使用 np.isin 比較每個 (3,) 值是否在 L 中
        # axis=1 表示只在每個 (3,) 向量內部進行比較
        matches = np.isin(img_flat.view([('', img_flat.dtype)] * 3), colorsInClass.view([('', colorsInClass.dtype)] * 3))
        # 獲得匹配位置的索引，並將其轉換回 (a, b) 的二維索引
        matched_indices = np.argwhere(matches.reshape(img.shape[:2]))
        colorClassMap[LOGger.stamp_process('',colorsInClassMedian,'','','',',',for_file=True)] = matched_indices
        canvas[matched_indices[:,0], matched_indices[:,1], :] = colorsInClassMedian
    outcome = cv2draw_ending(canvas, is_end=is_end, file=file, dont_log=dont_log, **kwags)
    if(outcome==0):
        return canvas, colorClassMap
    ret = ret if(isinstance(ret, dict)) else {}
    ret['img'] = canvas
    ret['colorClassMap'] = colorClassMap
    return True if(outcome>0) else False

def cv2findContoursClusterColors(img, mode=cv2.CHAIN_APPROX_NONE, method=cv2.RETR_EXTERNAL, is_end=False,
                                 eps=0.14, min_samples=10, maskColor=(0,0,0), deleteColors=None, infrm=None, 
                                 dont_log=True, file='', ret=None, **kwags):
    kwags['canvas'] = np.zeros(img.shape)
    contours = cv2findContours(img, is_end=False, mode=mode, method=method, ret=ret, **kwags)
    return cv2clusterColors(img, contours, eps=0.14, min_samples=10, is_end=is_end, maskColor=maskColor, deleteColors=deleteColors, 
                            infrm=infrm, dont_log=dont_log, file=file, ret=ret, **kwags)

# def cv2fitellipse(img, centre, major, minor, angle, file=None, is_end=True, dont_showimg=False, **kwags):
#     if(not m_cv2_import_succeed):
#         return None
#     img_ret = cv2imread(img)

#     (h, w) = img_ret.shape[:2]
#     center = (w / 2, h / 2)
#     M = cv2.getRotationMatrix2D(center, angle, resize_rate)
#     img_ret = cv2.warpAffine(img_ret, M, (w, h))

#     if(is_end):
#         if(isinstance_not_empty(file, str)):
#             CreateFile(file, lambda f:cv2.imwrite(filename=f, img=img_ret))
#         else:
#             cv2imshow(img_ret, **kwags) if(not dont_showimg) else None
#     else:
#         return img_ret
    
def cv2rotate(img, angle=90, resize_rate=1, file=None, is_end=True, dont_showimg=False, centre=None, **kwags):
    if(not m_cv2_import_succeed):
        return None
    img_ret = cv2imread(img)

    (h, w) = img_ret.shape[:2]
    center = centre[:2] if(DFP.isiterable(centre)) else (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, int(angle), resize_rate)
    img_ret = cv2.warpAffine(img_ret, M, (w, h))

    if(is_end):
        if(isinstance_not_empty(file, str)):
            CreateFile(file, lambda f:cv2.imwrite(filename=f, img=img_ret))
        else:
            cv2imshow(img_ret, **kwags) if(not dont_showimg) else None
    else:
        return img_ret
    
def cv2imexport_masking(img, img_msk, stamps=None, exp_fd='.'):
    imgfile_stamp = stamp_process('',os.path.basename(img).split('.')[:-1],'','','','_',for_file=1) if(
        isinstance_not_empty(img, str)) else 'source'
    exp_fd = exp_fd if(isinstance(exp_fd, str)) else '.'
    stamps = stamps if(isinstance(stamps, list)) else ['masked']
    
    new_img = dcp(cv2bitwise_and(img,img_msk))
    stamp = stamp_process('',[imgfile_stamp]+stamps,'','','','_',for_file=1)
    file = os.path.join(exp_fd, '%s.png'%stamp)
    CreateFile(file, lambda f:cv2.imwrite(filename=f, img=new_img))
    
def cv2imexport_edging(img,img_msk=None,threshold_lbd=80, threshold_ubd=240, kernel_size=3, 
                       dilate_iterations=3, erode_iterations=2,exp_fd='.',stamps=None,**kwags):
    imgfile_stamp = stamp_process('',os.path.basename(img).split('.')[:-1],'','','','_',for_file=1) if(
        isinstance_not_empty(img, str)) else 'source'
    exp_fd = exp_fd if(isinstance(exp_fd, str)) else '.'
    stamps = stamps if(isinstance(stamps, list)) else ['masked']
    edged = cv2imedging(img, img_msk=img_msk, threshold_lbd=threshold_lbd, threshold_ubd=threshold_ubd, kernel_size=kernel_size, 
                        dilate_iterations=dilate_iterations, erode_iterations=erode_iterations,**kwags)
    stamp = stamp_process('',[imgfile_stamp]+stamps,'','','','_',for_file=1)
    file = os.path.join(exp_fd, '%s.png'%stamp)
    CreateFile(file, lambda f:cv2.imwrite(filename=f, img=edged))
    
def cv2imshow_edging(img,img_msk=None,threshold_lbd=80, threshold_ubd=240, kernel_size=3, 
                       dilate_iterations=3, erode_iterations=2,exp_fd='.',stamps=None,**kwags):
    imgfile_stamp = stamp_process('',os.path.basename(img).split('.')[:-1],'','','','_',for_file=1) if(
        isinstance_not_empty(img, str)) else 'source'
    exp_fd = exp_fd if(isinstance(exp_fd, str)) else '.'
    stamps = stamps if(isinstance(stamps, list)) else ['masked']
    edged = cv2imedging(img, img_msk=img_msk, threshold_lbd=threshold_lbd, threshold_ubd=threshold_ubd, kernel_size=kernel_size, 
                        dilate_iterations=dilate_iterations, erode_iterations=erode_iterations,**kwags)
    stamp = stamp_process('',[imgfile_stamp]+stamps,'','','','_',for_file=1)
    cv2imshow(edged, window_title=stamp, **kwags)
   
def cv2imshow_thredging(img,img_msk=None,threshold=80, kernel_size=3, 
                       dilate_iterations=3, erode_iterations=2,exp_fd='.',stamps=None,**kwags):
    imgfile_stamp = stamp_process('',os.path.basename(img).split('.')[:-1],'','','','_',for_file=1) if(
        isinstance_not_empty(img, str)) else 'source'
    exp_fd = exp_fd if(isinstance(exp_fd, str)) else '.'
    stamps = stamps if(isinstance(stamps, list)) else ['masked']
    edged = cv2imthredging(img, img_msk=img_msk, threshold=threshold, kernel_size=kernel_size, 
                        dilate_iterations=dilate_iterations, erode_iterations=erode_iterations,**kwags)
    stamp = stamp_process('',[imgfile_stamp]+stamps,'','','','_',for_file=1)
    cv2imshow(edged, window_title=stamp, **kwags)
    
def cv2imexport_edging_batch(img,img_msk=None,threshold_lbds=None, threshold_ubds=None, kernel_size=3, 
                             dilate_iterations=3, erode_iterations=2,exp_fd='.',stamps=None,**kwags):
    imgfile_stamp = stamp_process('',os.path.basename(img).split('.')[:-1],'','','','_',for_file=1) if(
        isinstance_not_empty(img, str)) else 'source'
    exp_fd = exp_fd if(isinstance(exp_fd, str)) else '.'
    stamps = stamps if(isinstance(stamps, list)) else ['masked']
    threshold_lbds = threshold_lbds if(DFP.isiterable(threshold_lbds)) else np.arange(80,240,50)
    threshold_ubds = threshold_ubds if(DFP.isiterable(threshold_ubds)) else [np.max(threshold_lbds)]
    for threshold_lbd in threshold_lbds:
        for threshold_ubd in threshold_ubds:
            stamps_inloop=[DFP.parse(threshold_lbd), DFP.parse(threshold_ubd)]
            cv2imexport_edging(img,img_msk=img_msk,threshold_lbd=threshold_lbd, threshold_ubd=threshold_ubd, 
                               kernel_size=kernel_size, dilate_iterations=dilate_iterations, 
                               erode_iterations=erode_iterations,exp_fd=exp_fd,stamps=[imgfile_stamp]+stamps+stamps_inloop,**kwags)
            
def cv2contour_curverture(ct, with_coord=True):
    if(not m_cv2_import_succeed):
        return None
    # if(isinstance(ct, type(None))):
    
    ct = np.array(ct).reshape(-1, 2).astype(np.float32)
    
    # 重新取樣輪廓點
    length = cv2.arcLength(ct, closed=True)
    # ct = cv2.approxPolyDP(ct, 0.02*length, closed=True)
    ct = cv2.approxPolyDP(ct, 0.002*length, closed=True)
    ct = ct.reshape(-1, 2)
    # 計算第一導數和第二導數
    dx = np.gradient(ct[:, 0])
    dy = np.gradient(ct[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # 計算曲率
    k = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    return np.hstack([k.reshape(-1,1), ct]) if(with_coord) else k

def cv2warning_imshow(img, judgement, window_title='warning', waitKey=0, **kwags):
    cv2imshow(img, waitKey=waitKey, window_title=window_title, **kwags) if(judgement) else None
    
def cv2VideoLive(source_file, img_msk_file=None, exp_fd='.', waitKey=1, gray_ubd=None,
                 target_extfiletype=None, dont_log=True, is_autoremovefile=False,
                 autoremove_period_seconds=24*60*60, autoremove_sleep_period_seconds=5*60, frame_filter=None, 
                 config_json_file=None, config_keys=None, autoremove_fd=None, window_title='live', **kwags):
    try:
        fr, cg_sv, cap = None, None, None
        if(is_autoremovefile and (os.path.isdir(autoremove_fd) if(isinstance_not_empty(autoremove_fd, str)) else False)):
            target_extfiletype = target_extfiletype if(isinstance(target_extfiletype, list)) else ['png','jpg']
            fr = myFileRemover(exp_fd, period_seconds=autoremove_period_seconds, target_extfiletype=target_extfiletype,
                               sleep_period_seconds=autoremove_sleep_period_seconds)
        livekwags = dcp(kwags)
        livekwags.update({'waitKey':waitKey})
        livekwags.update({'gray_ubd':gray_ubd})
        if(os.path.isfile(config_json_file) if(config_json_file) else False):
            cg_sv = ConfigAgent(config_json_file, config_keys, livekwags, stamps=['main'], renew_passtime=60)
        img_msk = cv2imread(img_msk_file)
        livekwags.update({'img_msk':img_msk})
        cap = myVideoCapture(source_file)
        log_counter = {}
        while(True if(dont_log) else livekwags.update({'fr_index':livekwags.get('fr_index',0)+1})==None):
            frame_read = cap.read()
            cv2imshow(frame_read, window_title=window_title, **livekwags)
            if(gray_ubd):
                gray_value = cv2get_mgrayscale(frame_read, img_msk=livekwags.get('img_msk',img_msk))
                if(not gatelog(lambda **kkwags:gray_value>livekwags.get('gray_ubd', np.inf),  log_counter=log_counter, 
                               log_counter_stamp='filter interrupts', 
                               stop_message='gray value %s(>=%s)'%(DFP.parse(gray_value), DFP.parse(livekwags.get('gray_ubd', np.inf))),
                               pass_message=getattr(frame_filter,'pass_message', None),
                               img = frame_read, img_msk=img_msk, fr_index=livekwags.get('fr_index',0))):
                    time.sleep(5)
                    continue
    except Exception as e:
        exception_process(e, logfile='', stamps=[cv2VideoLive.__name__])
    finally:
        instances_method_process(cap, cg_sv, fr, method_name='stop')
    
def cv2ContourDetection(source_file, img_msk_file, dont_log=False, log_per_fr=10, dont_log_img_shape_error=True, 
                       waitKey=1, exp_fd='image', saveimg_fr_freq = 0, stamps=None, dont_showimg=True, dont_showimg_aux=True,
                       click_table_save_types=['csv'], click_table_exp_fd='.', save_and_show=False, consequence_seconds=3,
                       monitor_count=4, img_buffer_ubd=4, wait_judgement_timeout=3, is_warning_saveimg=False, 
                       draw_use_thread=True, scale_factor=0.4, is_warning_sound=False, 
                       servants_config_keys=None, attr_config_keys=None,
                       config_json_file='', warning_saveimg_passtime=3*60, jumpup_when_warning=True, 
                       frame_filter=None, is_attr_exp=True, attr_save_types=['csv'], 
                       is_autoremovefile=False, autoremove_period_seconds=24*60*60, autoremove_sleep_period_seconds=5*60,
                       target_extfiletype=None, **kwags):
    if(not m_cv2_import_succeed):
        return
    img_msk = cv2imread(img_msk_file)
    addlog = execute('addlog', kwags, default=addloger(logfile=kwags.get('logfile','')), not_found_alarm=False)
    CreateContainer(exp_fd)
    if(not source_file.find('://')>-1):
        if(not os.path.isfile(source_file)):
            addlog('no source:%s'%source_file)
            return
    click_table = kwags.get('click_table', None)
    ag_click = DFP.myAttributeAgent(
        buffer=click_table, stamps=['click'], first_pop_index=0, cleaning_waitng_time=5*60,
        exp_fd=click_table_exp_fd, save_types=click_table_save_types, operate_method=lambda p:np.sum(p, axis=1)) if(
            click_table!=None) else None
    ag = DFP.myAttributeAgent(
        stamps=['main'], first_pop_index=0, cleaning_waitng_time=12*60*60,
        exp_fd=exp_fd, save_types=attr_save_types, rewrite=False) if(is_attr_exp) else None
    cap = myVideoCapture(source_file)
    name = dt.now().strftime('%Y%m%d-%H%M%S')
    addlog('scale_factor:%.2f'%scale_factor)
    vm = myVisualMonitor(name, monitor_count=monitor_count, scale_factor=scale_factor, is_warning_sound=is_warning_sound,
                         img_buffer_ubd=img_buffer_ubd, consequence_seconds=consequence_seconds, warning_saveimg_passtime=warning_saveimg_passtime,
                         is_warning_saveimg=is_warning_saveimg, exp_fd=exp_fd, click_table=click_table, 
                         jumpup_when_warning=jumpup_when_warning)
    ims = myImageServant(name, exp_fd=exp_fd, click_table=click_table, VisualMonitor=vm,
                         saveimg_fr_freq=saveimg_fr_freq, is_auximg=True) if(draw_use_thread) else None
    config_json_file, cg_sv, cg = (config_json_file if(isinstance(config_json_file, str)) else ''), None, None
    if(config_json_file):
        servants_config_keys = servants_config_keys if(isinstance(servants_config_keys, list)) else [
            'consequence_seconds','scale_factor','is_warning_sound','saveimg_fr_freq','warning_saveimg_passtime',
            'jumpup_when_warning']
        cg_sv = ConfigAgent(config_json_file, servants_config_keys, vm, ims, stamps=['sv'], renew_passtime=60)
        attr_config_keys = attr_config_keys if(isinstance(attr_config_keys, list)) else [
            'dilate_iterations','erode_iterations','scale_factor','wait_judgement_timeout','angle_central',
            'bounding_method','anno_features','draw_box_bound','bw_lbd','bh_lbd','mrw_lbd','mrh_lbd',
            'threshold_ubd','threshold_lbd','ct_area_lbd']
        cg = ConfigAgent(config_json_file, attr_config_keys, kwags, renew_passtime=60, log_when_renew=True, stamps=['main'])
    fr = None
    if(is_autoremovefile):
        target_extfiletype = target_extfiletype if(isinstance(target_extfiletype, list)) else ['png','jpg']
        fr = myFileRemover(exp_fd, period_seconds=autoremove_period_seconds, target_extfiletype=target_extfiletype,
                           sleep_period_seconds=autoremove_sleep_period_seconds)
    # fps = cap.cap.get(cv2.CAP_PROP_FPS)
    kwags.update({'scale_factor':scale_factor})
    # kwags.update({'fr_index':0}) if(not dont_log) else None
    livekwags = dcp(kwags)
    livekwags.update({'waitKey':kwags.pop('waitKey_live', 1)})
    stamps = stamps if(isinstance(stamps, list)) else []
    file_stamps = stamps + ['monitored']
    try:
        log_counter, fr_index = {}, 0
        while(True):
            fr_index += 1
            frame_read = cap.read()
            cv2imshow(frame_read, window_title='live', **livekwags) if(not dont_showimg) else None
            if(frame_filter):
                frame_filter_criterion = getattr(frame_filter, 'criterion', frame_filter)
                if(not gatelog(frame_filter_criterion, log_counter=log_counter, log_counter_stamp='filter interrupts', 
                               stop_message=getattr(frame_filter,'stop_message', None),
                               pass_message=getattr(frame_filter,'pass_message', None),
                               img = frame_read, img_msk=img_msk, fr_index=fr_index)):
                    time.sleep(5)
                    continue
            now = dt.now()
            frame_stamp = stamp_process('',file_stamps+[now.strftime('%H%M%S-%f')[:-4]],'','','','_',for_file=True)
            # 獲取目前的幀編號
            # frame_number = int(cap.cap.get(cv2.CAP_PROP_POS_FRAMES))
            # time_in_video = frame_number / fps
            if(not isinstance(click_table, type(None))):
                kwags['click'] = {}
                ag_click.update(frame_stamp, package=kwags['click'])
            kwags.update({'file':''})
            #TODO:saveimg_fr_freq要default=0
            if(fr_index%saveimg_fr_freq==1 if(saveimg_fr_freq>0) else False):
                fn = stamp_process('',file_stamps+[now.strftime('%H%M%S-%f')[:-4]],'','','','_',for_file=1)
                exp_file = os.path.join(exp_fd, now.strftime('%Y%m%d'), '%s.jpg'%fn)
                kwags.update({'file':exp_file})
            ret = {'judgement':None}
            cv2findContours(frame_read, img_msk, ret=ret, dont_log_img_shape_error=dont_log_img_shape_error, 
                            waitKey=waitKey, dont_showimg=dont_showimg_aux, save_and_show=save_and_show,
                            stamps=[frame_stamp], ImageServant=ims, ret_keys=['judgement'], now=now, AttrAgent=ag, **kwags)
            if(ims==None):
                if(not gatelog((lambda **kwags:isinstance(kwags['ret'].get('judgement'), type(None))), log_counter=log_counter,
                               log_counter_stamp='wait judgement timeout', 
                               stop_message='wait judgement timeout[%.4f]...'%wait_judgement_timeout, 
                               delay_timeout=wait_judgement_timeout, ret=ret)):
                    continue
                if(ret['judgement']):
                    if(not gatelog((lambda **kwags:isinstance(kwags['ret'].get('img'), type(None))), log_counter=log_counter,
                                   log_counter_stamp='wait judgement img timeout', 
                                   stop_message='wait judgement img timeout[%.4f]...'%wait_judgement_timeout, 
                                   delay_timeout=wait_judgement_timeout, ret=ret)):
                        continue
                    vm.add_img(ret['img'], tm=now, stamp=frame_stamp, img_raw=frame_read)
            if('click' in kwags and ag_click!=None):
                if(fr_index%log_per_fr==1):
                    addlog('time report:%.3f\n%s'%(sum(list(kwags['click'].values())), 
                                                   stamp_process('',kwags['click'],adjoint_sep='\n')), stamps=[])
    except Exception as e:
        exception_process(e, logfile='', stamps=[cv2ContourDetection.__name__])
    finally:
        instances_method_process(cap, ims, vm, cg, cg_sv, ag, ag_click, fr, method_name='stop')
        if(ag!=None):
            ag.save(ag.exp_fd, ag.save_types, rewrite = False)
        if(ag_click!=None):
            ag_click.save(ag_click.exp_fd, ag_click.save_types, rewrite = False)

def draw_curve_fit(img, xdata, ydata, method=(lambda x,a,b,c,d:a*x**3+b*x**2+c*x+d), p0=None, ret=None,
                   color_p_fit=[255,150,0], color_f=[0,0,255], fontsize=0.3, color_text=[255,255,255], n_pt_fit_show=1000, 
                   pt_radius=5, is_end=True, file='', dont_log=True, **kwags):
    ret = ret if(isinstance(ret, dict)) else {}
    if(customization_curve_fit(method, xdata, ydata, p0=None, ret=ret)):
        executor = ret['executor']
        xps_fit = list(map(int, np.linspace(0, img.shape[1]-1, n_pt_fit_show)))
        yps_fit = list(map(lambda x:int(min(max(0,np.ceil(executor(x))), img.shape[0]-1)), xps_fit))
        marks = list(zip(*[xdata, ydata]))
        marks_fit = list(zip(*[xps_fit, yps_fit]))
        for mk in marks:
            cv2.circle(img, mk, color=color_f, radius=pt_radius, thickness=-1)
        for mk in marks_fit:
            cv2.circle(img, mk, color=color_p_fit, radius=pt_radius, thickness=-1)
    cv2draw_ending(img, is_end=is_end, file=file, dont_log=dont_log, **kwags)

def cv2getblur(img):
    # 使用拉普拉斯算子
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return laplacian_var
       
def cv2getblur_at(img, pt, augmented_px=50, oversized_method='cut', augmented_pxs=None, default=np.nan):
    H, W = img.shape[:2]
    if(DFP.isiterable(augmented_pxs)):
        augmented_pxs = mylist(augmented_pxs)
    a,b,c,d = (pt[0] - augmented_pxs.get(0, augmented_px), pt[0] + augmented_pxs.get(1, augmented_px), 
               pt[1] - augmented_pxs.get(2, augmented_px), pt[1] + augmented_pxs.get(3, augmented_px))
    if(oversized_method=='cut'):
        a = max(0, min(a, W))
        b = max(0, min(b, W))
        c = max(0, min(c, H))
        d = max(0, min(d, H))
    try:
        img_opr = img[c:d, a:b, :]
    except:
        return default
    return cv2getblur(img_opr)

def cv2baseInfo(img, showCrosshair=False, font_size=0.3, parse_digit=3, 
                reply_on_img=True, color_frame=[0,0,255], th_frame=1, color_text=[0,0,0], 
                img_org=None, **kwags):
    roi = cv2.selectROI(img, showCrosshair=showCrosshair)
    try:
        # 根據選擇的區域切割圖片
        img_opr = img_org if(not isinstance(img_org, type(None))) else img
        cropped_img = img_opr[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2]),:]
        
        # 計算框框中的像素值
        x,y = int(roi[0]), int(roi[1])
        h,w = cropped_img.shape[:2]
        mean_val = np.mean(cropped_img, axis=(0, 1))
        blur = cv2getblur(cropped_img)
        
        replys = ["(vx,vy)=(%s,%s); Pixels (w,h): %sx%s"%(DFP.parse(x, digit=parse_digit), DFP.parse(y, digit=parse_digit),
            DFP.parse(w, digit=parse_digit), DFP.parse(h, digit=parse_digit)),
                  "Mean pixel values (B, G, R): (%s,%s,%s)"%tuple(map(lambda x:DFP.parse(x, digit=parse_digit), mean_val)),
                  "blur value: %s"%(DFP.parse(blur, digit=parse_digit))]
        if(reply_on_img):
            n_replys = len(replys)
            for i,reply in enumerate(replys):
                cv2.putText(img, reply, (roi[0], roi[1] + (i - n_replys)*10), cv2.FONT_HERSHEY_SIMPLEX, font_size, 
                            list(map(lambda x:int(np.uint8(x)), color_text)), 1, cv2.LINE_AA)
            cv2.rectangle(img, (int(roi[0])-1, int(roi[1])-1), (int(roi[0]+roi[2])+1, int(roi[1]+roi[3])+1), 
                          color=color_frame, thickness=th_frame)
        else:
            for reply in replys:
                print(reply)
        
    except Exception as e:
        exception_process(e, logfile='')
        
def cv2getblur_comparison(img, axis=1, r=0.5, ret=None, **kwags):
    if(axis==0):
        img_t = img.transpose(1,0,2)
        return cv2getblur_comparison(img_t, axis=1, r=r, ret=ret)
    H, W = img.shape[:2]
    r = r if(abs(r-0.5)<=0.5) else 0.5
    dn_cropped_img = img[int(H*r):H,:,:]
    up_cropped_img = img[0:int(H*r),:,:]
    
    up_blur = cv2getblur(up_cropped_img)
    dn_blur = cv2getblur(dn_cropped_img)
    blur_diff = up_blur - dn_blur
    
    if(not isinstance(ret , dict)):
        return blur_diff
    
    ret['up_blur'] = up_blur
    ret['dn_blur'] = dn_blur
    return blur_diff

def cv2getblur_comparison_at(img, pt, augmented_px=50, axis=1, r=0.5, ret=None, augmented_pxs=None, oversized_method='cut', default=np.nan,
                             **kwags):
    H, W = img.shape[:2]
    if(DFP.isiterable(augmented_pxs)):
        augmented_pxs = mylist(augmented_pxs)
    else:
        augmented_pxs = mylist()
    a,b,c,d = (pt[0] - augmented_pxs.get(0, augmented_px), pt[0] + augmented_pxs.get(1, augmented_px), 
               pt[1] - augmented_pxs.get(2, augmented_px), pt[1] + augmented_pxs.get(3, augmented_px))
    if(oversized_method=='cut'):
        a = max(0, min(a, W))
        b = max(0, min(b, W))
        c = max(0, min(c, H))
        d = max(0, min(d, H))
    try:
        img_opr = img[c:d, a:b, :]
    except:
        return default
    # print(c,d)
    return cv2getblur_comparison(img_opr, axis=axis, r=r, ret=ret)
        
def cv2updownhalf_blur_comparison(img, showCrosshair=False, font_size=0.3, parse_digit=3, 
                                  reply_on_img=True, color_frame=[0,0,255], th_frame=1, color_text=[0,0,0], 
                                  img_org=None, **kwags):
    roi = cv2.selectROI(img, showCrosshair=showCrosshair)
    try:
        # 根據選擇的區域切割圖片
        img_opr = img_org if(not isinstance(img_org, type(None))) else img
        dn_cropped_img = img_opr[int((2*roi[1]+roi[3])/2):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2]),:]
        up_cropped_img = img_opr[int(roi[1]):int((2*roi[1]+roi[3])/2), int(roi[0]):int(roi[0]+roi[2]),:]
        
        up_blur = cv2getblur(up_cropped_img)
        dn_blur = cv2getblur(dn_cropped_img)
        blur_diff = up_blur - dn_blur
        reply = "blur value: U[%s] - D[%s] = %s"%(
            DFP.parse(up_blur, digit=parse_digit), DFP.parse(dn_blur, digit=parse_digit), DFP.parse(blur_diff, digit=parse_digit))
        if(reply_on_img):
            cv2.putText(img, reply, (roi[0], roi[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, font_size, 
                        list(map(lambda x:int(np.uint8(x)), color_text)), 1, cv2.LINE_AA)
            cv2.rectangle(img, (int(roi[0])-1, int(roi[1])-1), (int(roi[0]+roi[2])+1, int((2*roi[1]+roi[3])/2)+1), 
                          color=color_frame, thickness=th_frame)
            cv2.rectangle(img, (int(roi[0])-1, int((2*roi[1]+roi[3])/2)-1), (int(roi[0]+roi[2])+1, int(roi[1]+roi[3])+1), 
                          color=color_frame, thickness=th_frame)
        else:
            print(reply)
        
        
    except Exception as e:
        exception_process(e, logfile='')

def cv2find_sharpenest_edge(img, axis=0, d=100, y_lbd=1400, y_ubd=1800, selected_points=None, select_threshold=700, 
                            color_p_fit=[255,150,0], color_f=[0,0,255], fontsize=0.3, color_text=[255,255,255], n_pt_fit_show=1000, 
                            method_curve_fit=(lambda x,a,b,c,d:a*x**3+b*x**2+c*x+d), is_dbscan_and_find_max_group=False, **kwags):
    selected_points = selected_points if(isinstance(selected_points, list)) else []
    selected_points_x = []
    for x in range(0, img.shape[1], d):
        blue_diff_abs = np.abs(tuple(map(lambda y:cv2getblur_comparison_at(img, (int((2*x+d)/2), y)), np.arange(y_lbd,y_ubd))))
        if(np.max(blue_diff_abs)<=select_threshold):
            continue
        y_pos = np.argmax(blue_diff_abs)
        selected_points.append(y_pos + y_lbd)
        selected_points_x.append(int((2*x+d)/2))
    
    if(is_dbscan_and_find_max_group):
        label_mask, label_set = tuple(DFP.dbscaning(pd.DataFrame(selected_points)).values())
        values, counts = np.unique(label_mask, return_counts=1)
        seleted_group = values[np.argmax(counts)]
        selected_points_x = list(tuple(np.array(selected_points_x)[label_mask == seleted_group]))
        selected_points = list(tuple(np.array(selected_points)[label_mask == seleted_group]))
    
    img_aux = dcp(img)
    draw_curve_fit(
        img_aux, selected_points_x, selected_points, method_curve_fit, p0=None, ret=None,
        color_p_fit=color_p_fit, color_f=color_f, fontsize=fontsize, color_text=color_text, n_pt_fit_show=n_pt_fit_show, **kwags)
    

def cv2selectROI(img, recursion_max=1, break_key=27, showCrosshair=False, scenario=cv2baseInfo, **kwags):
    # 讀取圖片
    img = cv2imread(img)
    if(kwags.get('scale_factor',None)):
        scale_factor = kwags.get('scale_factor',1)
        img = cv2imresize(img, scale_factor=scale_factor)
    # 顯示圖片並使用滑鼠選擇框框區域
    img_org = dcp(img)
    while(recursion_max>0 if(isinstance(recursion_max, int)) else True):
        scenario(img, showCrosshair=showCrosshair, img_org=img_org, **kwags)
        key = cv2.waitKey(0)
        # print(key)
        if(key==break_key):
            break
        elif(key==ord('c')):
            img = dcp(img_org)
        if(isinstance(recursion_max, int)):
            recursion_max = recursion_max - 1
        # img = dcp(img_org)
    cv2.destroyAllWindows()

# 定义一个函数来显示图表
def cv2show_chart_onbaord_byevent(base_data, base_img, x, y, param, **kwags):
    cell_shape = LOGger.execute('cell_shape', param, kwags, not_found_alarm=False, default=(100,30))
    # print('x', x, 'cell', *cell_shape, 'base axis_1', base_data.shape[1])
    rows_per_page = LOGger.execute('rows_per_page', param, kwags, not_found_alarm=False, default=10)
    start_row = param.start_row
    row = y // cell_shape[1] + start_row
    start_row = min(max(0, start_row), base_data.shape[0] - rows_per_page)
    if start_row <= row < start_row + rows_per_page and 1 * cell_shape[0] + 10 <= x < 2 * cell_shape[0] + 10:
        base_img[:,:,:] = 255
        cv2.putText(base_img, DFP.parse(base_data.index[row]), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        th = threading.Thread(target=cv2show_chart_onbaord_byevent_doing, args=[base_data, base_img, x, y, param],
                              kwargs=kwags)
        th.start()
    
    
def cv2show_chart_onbaord_byevent_doing(base_data, base_img, x, y, param, rows_per_page=20, cell_shape=(100,30), 
                                        figsize=(10,10), dpi=100):
    start_row = param.start_row
    start_row = min(max(0, start_row), base_data.shape[0] - rows_per_page)
    plot_method = param.plot_method
    rows_per_page = LOGger.execute('rows_per_page', param, not_found_alarm=False, default=rows_per_page)
    cell_shape = LOGger.execute('cell_shape', param, not_found_alarm=False, default=cell_shape)
    figsize = LOGger.execute('figsize', param, not_found_alarm=False, default=figsize)
    dpi = LOGger.execute('dpi', param, not_found_alarm=False, default=dpi)
    
    row = y // cell_shape[1] + start_row
    col = x // cell_shape[0] - 1
    if start_row <= row < start_row + rows_per_page and 1 * cell_shape[0] + 10 <= x < 2 * cell_shape[0] + 10:
        fig, ax = pltinitial(fig=type(None), figsize=figsize, dpi=dpi)
        if(plot_method!=None):
            if(not plot_method(data=base_data, r=row, c=col, fig=fig, ax=ax)):
                return 
        fig.tight_layout()
        try:
            param.fig = fig
            canvas = FigureCanvas(fig)
            canvas.draw()
            buf = canvas.buffer_rgba()
            img = np.asarray(buf)
            cv2.imwrite(filename='cv2.png', img=img)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            base_img[:,:,:] = cv2.resize(img, base_img.shape[:2])
        except Exception as e:
            print('base_img', base_img.shape)
            LOGger.exception_process(e,logfile='')
                
# 定义一个函数来显示图表
def cv2roll_table_byevent(base_data, base_img, flags, param, rows_per_page=20, font_color=(255,255,255), **kwags):
    """
    

    Parameters
    ----------
    base_data : TYPE
        DESCRIPTION.
    base_img : TYPE
        DESCRIPTION.
    flags : TYPE
        DESCRIPTION.
    param : TYPE
        DESCRIPTION.
    rows_per_page : TYPE, optional
        DESCRIPTION. The default is 10.
    font_color : TYPE, optional
        DESCRIPTION. The default is (255,255,255).
    **kwags : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    start_row = param.start_row
    # board_img = param.board_img
    rows_per_page = LOGger.execute('rows_per_page', param, not_found_alarm=False, default=rows_per_page)
    font_color = LOGger.execute('font_color', param, not_found_alarm=False, default=font_color)
    # if event == cv2.EVENT_MOUSEWHEEL:
    base_img[:,:,:] = 0
    if flags > 0 and start_row > 0:  # 向上滚动
        start_row -= 1
    elif flags < 0 and start_row + rows_per_page < base_data.shape[0]:  # 向下滚动
        start_row += 1
    start_row = min(max(0, start_row), base_data.shape[0] - rows_per_page)
    cell_width = int(base_img.shape[1]//4)
    cell_height = int(base_img.shape[0]//20)
    draw_index_method = (lambda i,**kkwags:cv2.putText(
            base_img, str(start_row + i), (rows_per_page, (i+1) * cell_height - rows_per_page), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1)) if(getattr(param,'draw_index_method', None)==None) else param.draw_index_method
    draw_cell_method = (lambda i,j,**kkwags:cv2.putText(
            base_img, "(%s %s)"%(start_row + i,j), ((j+1) * cell_width + 10, (i+1) * cell_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1)) if(getattr(param,'draw_cell_method', None)==None) else param.draw_cell_method
    for i in range(rows_per_page):
        if start_row + i >= base_data.shape[0]:
            break
        draw_index_method(i,img=base_img,data=base_data,cw=cell_width,ch=cell_height,start_row=start_row,rows_per_page=rows_per_page,font_color=font_color)
        for j in range(base_data.shape[1]):
            draw_cell_method(i,j,img=base_img,data=base_data,cw=cell_width,ch=cell_height,start_row=start_row,font_color=font_color)
    setattr(param,'start_row',start_row) if(param.start_row != start_row) else None

def cv2survey_sequential_data_event(base_data, event, x, y, flags, event_param, board_img=None, table_img=None, 
                                    mouse_wheel_visual_param=None, mouse_lclick_visual_param=None):
    if event == cv2.EVENT_MOUSEWHEEL:
        # print('click', mouse_lclick_visual_param.start_row)
        # print('wheel', mouse_wheel_visual_param.start_row)
        cv2roll_table_byevent(base_data, table_img, flags, param=mouse_wheel_visual_param)
        setattr(mouse_lclick_visual_param,'start_row',mouse_wheel_visual_param.start_row) if(
            mouse_lclick_visual_param.start_row!=mouse_wheel_visual_param.start_row) else None
    elif(event == cv2.EVENT_LBUTTONDOWN):#cv2.EVENT_MOUSEMOVE):
        cv2show_chart_onbaord_byevent(base_data, board_img, x, y, param=mouse_lclick_visual_param)

def cv2survey_sequential_data(data, plot_method=None, rows_per_page=20, main_shape=(300,600), font_color=(255,255,255), 
                              cell_shape=None, board_shape=None, stamps=None, base=None, figsize=(5,5), 
                              draw_index_method=None, draw_cell_method=None, exp_fd='.'):
    """
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    plot_method : TYPE, optional
        DESCRIPTION. The default is         (data,r,c,fig,ax) |-> report(data.iloc[r,c],base=base,is_end=False,fig=fig,ax=ax)
    rows_per_page : TYPE, optional
        DESCRIPTION. The default is 10.
    main_shape : TYPE, optional
        DESCRIPTION. The default is (300,600).
    font_color : TYPE, optional
        DESCRIPTION. The default is (255,255,255).
    cell_shape : TYPE, optional
        DESCRIPTION. The default is None.
    board_shape : TYPE, optional
        DESCRIPTION. The default is None.
    stamps : TYPE, optional
        DESCRIPTION. The default is None.
    base : TYPE, optional
        DESCRIPTION. The default is None.
    figsize : TYPE, optional
        DESCRIPTION. The default is (5,5).
    draw_index_method : TYPE, optional
        DESCRIPTION. The default is         (i;data,img,cw,ch,start_row,rows_per_page,font_color;**kwags) |-> cv2.putText(
                base_img, str(start_row + i), (rows_per_page, (i+1) * cell_height - rows_per_page), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1)
    draw_cell_method : TYPE, optional
        DESCRIPTION. The default is         (i,j;data,img,cw,ch,start_row,font_color;**kwags) |-> cv2.putText(
                base_img, "(%s %s)"%(start_row + i,j), ((j+1) * cell_width + 10, (i+1) * cell_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1)

    Returns
    -------
    None.

    """
    
    stamps = (stamps if(isinstance(stamps, list)) else [])+['Main']
    stamp = LOGger.stamp_process('',stamps,'','','',' ',for_file=1)
    
    # 创建一个窗口用于显示表格
    total_img = np.zeros((*(main_shape[:2]), 3), dtype=np.uint8)
    table_img = total_img[:,:int(total_img.shape[1]//2),:]
    board_img = total_img[:,int(total_img.shape[1]//2):,:]
    cell_width = int(table_img.shape[1]//4)
    cell_height = int(table_img.shape[0]//20)
    # print('cell', cell_width, cell_height)
    
    mouse_wheel_visual_param, mouse_lclick_visual_param = LOGger.mystr(), LOGger.mystr()
    mouse_lclick_visual_param.start_row = 0
    mouse_lclick_visual_param.plot_method = (lambda data,r,c,fig,ax,**kkwags:report(data.iloc[r,c],base=base,is_end=False,fig=fig,ax=ax)) if(
        plot_method==None) else plot_method
    mouse_lclick_visual_param.figsize=figsize
    mouse_lclick_visual_param.cell_shape = (cell_width, cell_height)
    mouse_lclick_visual_param.fig = None
    mouse_wheel_visual_param.draw_cell_method = draw_cell_method
    mouse_wheel_visual_param.draw_index_method = draw_index_method
    mouse_wheel_visual_param.start_row = 0
    mouse_wheel_visual_param.board_img = board_img
    mouse_wheel_visual_param.rows_per_page = rows_per_page
    mouse_wheel_visual_param.font_color =font_color
    cv2roll_table_byevent(data, table_img, flags=0, param=mouse_wheel_visual_param)
    
    
    # 设置 OpenCV 窗口和鼠标回调函数
    cv2.namedWindow(stamp)
    cv2.setMouseCallback(stamp, lambda e,x,y,f,p:cv2survey_sequential_data_event(
        data,e,x,y,f,p,board_img=board_img,table_img=table_img,
        mouse_wheel_visual_param=mouse_wheel_visual_param,
        mouse_lclick_visual_param=mouse_lclick_visual_param))
    
    # def handle_keys(key, mouse_wheel_visual_param):
    #     start_row = mouse_wheel_visual_param.start_row
    #     if key == 2621440:  # Ctrl + Home key (Windows)
    #         start_row = 0
    #     elif key == 2621480:  # Ctrl + End key (Windows)
    #         start_row = max(0, data.shape[0] - rows_per_page)
    #     mouse_wheel_visual_param.start_row = start_row
    
    while True:
        cv2.imshow(stamp, total_img)
        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # 按下 ESC 键退出
            break
        if key == ord('a'): 
            mouse_wheel_visual_param.start_row = 0
            mouse_lclick_visual_param.start_row = 0
            cv2roll_table_byevent(data, table_img, flags=0, param=mouse_wheel_visual_param)
        elif key == ord('z'): 
            mouse_wheel_visual_param.start_row = data.shape[0] - rows_per_page
            mouse_lclick_visual_param.start_row = data.shape[0] - rows_per_page
            cv2roll_table_byevent(data, table_img, flags=0, param=mouse_wheel_visual_param)
        elif key == ord('s') and mouse_wheel_visual_param.start_row - rows_per_page>=0: 
            mouse_wheel_visual_param.start_row = mouse_wheel_visual_param.start_row - rows_per_page
            mouse_lclick_visual_param.start_row = mouse_wheel_visual_param.start_row
            cv2roll_table_byevent(data, table_img, flags=0, param=mouse_wheel_visual_param)
        elif key == ord('x') and mouse_wheel_visual_param.start_row + rows_per_page <= data.shape[0] - rows_per_page: 
            mouse_wheel_visual_param.start_row = data.shape[0] - rows_per_page if(
                mouse_wheel_visual_param.start_row + rows_per_page > data.shape[0] - rows_per_page) else (
                    mouse_wheel_visual_param.start_row + rows_per_page)
            mouse_lclick_visual_param.start_row = mouse_wheel_visual_param.start_row
            cv2roll_table_byevent(data, table_img, flags=0, param=mouse_wheel_visual_param)
        elif key == ord('d') and getattr(mouse_lclick_visual_param, 'fig', None)!=None: 
            jpgfile = DFP.pathrpt(os.path.join(exp_fd, 'survey.jpg'))
            CreateFile(jpgfile, lambda f:mouse_lclick_visual_param.fig.savefig(f))
        # if key == ord('z'): 
        #     mouse_wheel_visual_param.start_row = 0
        #     cv2roll_table_byevent(data, table_img, flags=0, param=mouse_wheel_visual_param)
        
    cv2.destroyAllWindows()
    


#%%
class mySeqWindow:
    def __init__(self, base_data, main_shape, base=None, rows_per_page=10, figsize=(5,5), font_color=(255,255,255), stamps=None):
        self.stamps = (stamps if(isinstance(stamps, list)) else [])+['Main']
        self.base_data = base_data
        self.main_shape = main_shape
        self.total_img = np.zeros((*(main_shape[:2]), 3), dtype=np.uint8)
        self.table_img = self.total_img[:,:int(self.total_img.shape[1]//2),:]
        self.board_img = self.total_img[:,int(self.total_img.shape[1]//2):,:]
        self.base = base
        self.font_color = font_color
        
        self.cell_width = int(self.table_img.shape[1]//3)
        self.cell_height = int(self.table_img.shape[0]//20)
        self.mouse_wheel_visual_param, self.mouse_lclick_visual_param = LOGger.mystr(), LOGger.mystr()
        self.mouse_lclick_visual_param.start_row = 0
        self.mouse_lclick_visual_param.cell_shape = (self.cell_width, self.cell_height) #20240617長寬格式無須交換，因為data frame跟圖形無關
        self.mouse_lclick_visual_param.plot_method = lambda data,fig,ax,**kkwags:report(data,base=base,is_end=False,fig=fig,ax=ax)
        self.mouse_lclick_visual_param.figsize=figsize
        self.mouse_wheel_visual_param.start_row = 0
        self.mouse_wheel_visual_param.board_img = self.board_img
        self.mouse_wheel_visual_param.rows_per_page = rows_per_page
        self.mouse_wheel_visual_param.font_color = self.font_color
        
    def show(self):
        stamp = LOGger.stamp_process('',self.stamps,'','','',' ',for_file=1)
        cv2.namedWindow(stamp)
        cv2.setMouseCallback(stamp, self.cv2survey_sequential_data_event)
        # total_img[:,int(total_img.shape[1]//2):,:] = board_img
        
        while True:
            cv2.imshow(stamp, self.base_img)
            if cv2.waitKey(10) & 0xFF == 27:  # 按下 ESC 键退出
                break
            
        cv2.destroyAllWindows()
    
    def cv2survey_sequential_data_event(self, event, x, y, flags, event_param):
        cv2survey_sequential_data_event(self.base_data, event, x, y, flags, event_param, 
                                        board_img=self.board_img, table_img=self.table_img, 
                                        mouse_wheel_visual_param=self.mouse_wheel_visual_param, 
                                        mouse_lclick_visual_param=self.mouse_lclick_visual_param,
                                        draw_cell_method=self.draw_cell)
    
    def draw_table(self, i, j, start_row=0):
        cv2.putText(self.base_img, str(start_row + i), (self.rows_per_page, (i+1) * self.cell_height - self.rows_per_page), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.font_color, 1)
        for j in range(self.base_data.shape[1]):
            cv2.putText(self.base_img, f"({start_row + i},{j})", ((j+1) * self.cell_width + 10, (i+1) * self.cell_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.font_color, 1)
            
    def draw_cell(self, i, j):
        cv2.putText(self.base_img, "(%s %s)"%(self.start_row + i,j), ((j+1) * self.cell_width + 10, (i+1) * self.cell_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.font_color, 1)

class myWindow:
    def __init__(self, stamps=None, shape=(600,600,3)):
        self.stop_flag = False
        self.img = np.zeros(shape)
        self.stamps = stamps if(isinstance(stamps, list)) else ['']
        self.t = threading.Thread(target=self.show)
        self.t.daemon = True
        
    def handle_signal(self,signum, frame):
        self.stop_flag.set()
    
    def show(self):
        title = LOGger.stamp_process('',self.stamps,'','','',' ')
        while not self.stop_flag:
            cv2.imshow(title, self.img)
            key = cv2.waitKey(10) & 0xFF
            if key == 27:  # 按下 ESC 键退出
                self.stop()
    
    def start(self):
        self.t.start()
            
    def stop(self):
        self.stop_flag = True
        self.t.join()

class myWindows:
    def __init__(self, window_stamps=None, default_window_shape=(600,600,3)):
        self.windows = {k:myWindow(stamps=[k],shape=default_window_shape) for k in window_stamps} if(isinstance(window_stamps, list)) else {}
        
    def start(self):
        for k,wd in self.windows.items():
            wd.start()
            
    def stop(self):
        for k,wd in self.windows.items():
            wd.stop_flag = True
            wd.t.join()

class myCv2Canvas:
    def __init__(self, shape, layout, initial_value=0, channel=3, img_father=None):
        self.initial_value = initial_value
        self.img = np.ones((*(shape[:2]), channel), dtype=np.int32)*initial_value if(isinstance(img_father, type(None))) else img_father
        self.axes = {}
        self.shape = self.img.shape
        self.layout = layout
        yIndexes = range(0,self.shape[0]+1,int(self.shape[0]/layout[1]))
        xIndexes = range(0,self.shape[1]+1,int(self.shape[1]/layout[0]))
        # print(tuple(yIndexes)[:-1], tuple(xIndexes)[:-1])
        for i,idx in enumerate(yIndexes[:-1]):
            for j,jdx in enumerate(xIndexes[:-1]):
                self.axes[(j,i)] = self.img[idx:yIndexes[i+1],jdx:xIndexes[j+1],:]
                
    def get(self, i=0, j=0):
        return self.axes[(i,j)]
    
    def clear(self):
        self.img[:] = np.ones(self.shape, dtype=np.int32)*self.initial_value
        

class myVideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        self.stop_flag = False
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    #ctrl+c啟動正常結束程序
    # def handle_signal(self,signum, frame):
    #   print('handle_signal deteced!!!!')
    #   self.stop_flag.set()
        
    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while not self.stop_flag:  
            ret, frame = self.cap.read()
            #print(ret)
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()
    
    def stop(self):
        self.stop_flag = True
        self.cap.release()
    
class myVideoCaptureSemiManual(myVideoCapture):
    def __init__(self, name):
          self.pause_flag = False
          self.current_frame = None
          super().__init__(name)
        
    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while not self.stop_flag:
            if not self.pause_flag:
                ret, frame = self.cap.read()
                if not ret:
                    break
                if not self.q.empty():
                    try:
                        self.q.get_nowait()  # discard previous (unprocessed) frame
                    except queue.Empty:
                        pass
                self.current_frame = frame
                self.q.put(frame)
            # else:
            #     if self.current_frame is not None:
            #         self.q.put(self.current_frame)
                  
    def read(self):
          # print('read....')
          if not self.pause_flag:
               # print('not pause....')
              return self.q.get()
          else:
              return self.current_frame
    
    def pause(self):
        self.pause_flag = not self.pause_flag

    def jump(self, frames):
        if self.cap.isOpened():
            self.pause()
            current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            print('j1')
            new_pos = current_pos + frames
            new_pos = max(0, min(new_pos, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))
            print('j2')
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            print('j3')
            self.pause()
            print('j4')
    
class myImageServant():
    def __init__(self, name, img_buffer_ubd=10, exp_fd='image', time_sleep=1, click_table=None, is_auximg=False,
                 VisualMonitor=None, saveimg_fr_freq=0, method=cv2draw_auximage, window_shape=(1920,1080), 
                 scale_factor=0.4, **kwags):
        self.name = name
        self.time_sleep = time_sleep
        self.window_shape = window_shape
        self.img_buffer = []
        self.is_auximg = is_auximg
        self.scale_factor = scale_factor
        self.stop_flag = False
        self.img_live = None
        self.img_buffer_ubd = img_buffer_ubd if(
            img_buffer_ubd>0 if(isinstance(img_buffer_ubd, int)) else False) else self.monitor_count
        self.click_table = click_table
        self.params_table = {}
        self.exp_fd = exp_fd
        self.saveimg_fr_freq = saveimg_fr_freq
        self.VisualMonitor = VisualMonitor
        self.addlog = execute('addlog', kwags, default=addloger(logfile=execute('logfile', kwags, default='', not_found_alarm=False)), not_found_alarm=False)
        self.method = method
        self.s = threading.Thread(target=self._serving)
        self.s.daemon = True
        self.s.start()
    
    def handle_signal(self,signum, frame):
        #self.adgProcLogger.debug('%s(%d) handling signal %r' ,self.__netLabel,self.pid, signum) #type(self).__name__
        self.stop_flag.set()
        
    def add_img(self, img, tm=None, stamp=''):
        if(len(self.img_buffer)>=self.img_buffer_ubd):
            self.img_buffer.pop(0)
        tm = tm if(isinstance(tm, dt)) else dt.now()
        stamp = stamp if(isinstance(stamp, str)) else ''
        self.img_buffer.append([tm, img, stamp])
    
    def add_auxparams(self, params, source_stamp, stamp=None, **kwags):
        params_table = self.params_table
        index = len(params_table.get(source_stamp,{}))
        stamp = stamp if(isinstance(stamp, str)) else index
        if(not source_stamp in params_table):
            params_table.update({source_stamp:{}})
        params_table[source_stamp].update({stamp:params})
    
    def aux_img_process(self, ret=None):
          ret = ret if(isinstance(ret, dict)) else {}
          _, img_raw, stamp = self.img_buffer[0]
          img = dcp(img_raw+0)
          params_list = self.params_table.get(stamp, {})
          for params in {k:v for k,v in params_list.items() if (k>=0 if(isinstance(k,int)) else False)}.values():
              self.method(img, **params)
              if('ret' in params):    img = params['ret']['img']
          if(isinstance_not_empty(params_list.get('memo_stamps'), list)):
              memo_stamps = params_list['memo_stamps']
              memo_stamp = stamp_process('',memo_stamps,'','','',' ')
              cv2.putText(img, memo_stamp, (10, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, 
                          (int(255), int(255), int(255)), 5, cv2.LINE_AA)
          return img
    
    def _serving(self, **kwags):
        try:
            vm = self.VisualMonitor
            index = 0
            saveimg_fr_freq = self.saveimg_fr_freq
            click = None
            while(not self.stop_flag):
                if(len(self.img_buffer)==0):
                    time.sleep(self.time_sleep)
                    continue
                tm, img_raw, stamp = self.img_buffer[0]
                judgement_stg = ''
                if(self.is_auximg):
                    img = self.aux_img_process()
                    if(self.params_table.get(stamp, {}).get('judgement', False)):
                        judgement_stg = 'NG'
                        if(vm!=None):
                            vm.add_img(img, tm=tm, stamp=stamp, img_raw=img_raw)
                            self.img_buffer = self.img_buffer[1:]
                            self.params_table.pop(stamp, None)
                            index += 1
                            continue
                self.img_buffer = self.img_buffer[1:]
                self.params_table.pop(stamp, None)
                index += 1
                if(index%saveimg_fr_freq!=1 if(saveimg_fr_freq>0) else True):
                    continue
                fn = stamp_process('',[stamp, judgement_stg, 'RAW'],'','','','_',for_file=True)
                file = os.path.join(self.exp_fd, tm.strftime('%Y%m%d'), '%s.jpg'%fn)
                ta = dt.now() if(not isinstance(self.click_table, type(None))) else None
                CreateFile(file, lambda f:cv2.imwrite(filename=f, img=img_raw))
                click = self.click_table.get(stamp, {})
                click.update({'contours_saveimg':(dt.now() - ta).total_seconds()}) if(
                    not isinstance(self.click_table, type(None))) else None
                if((img!=img_raw).any()):
                    fn = stamp_process('',[stamp, judgement_stg],'','','','_',for_file=True)
                    file = os.path.join(self.exp_fd, tm.strftime('%Y%m%d'), '%s.jpg'%fn)
                    ta = dt.now() if(not isinstance(self.click_table, type(None))) else None
                    CreateFile(file, lambda f:cv2.imwrite(filename=f, img=img))
                    click.update({'contours_saveimg':(dt.now() - ta).total_seconds()}) if(
                        not isinstance(self.click_table, type(None))) else None
        except Exception as e:
            exception_process(e, logfile='', stamps=[self.name, self._serving.__name__])
    
    def _showlive(self):
        while(not self.stop_flag):
              img_live = dcp(np.zeros((*tuple(reversed(self.window_shape[:2])),3)) if(isinstance(self.img_live, type(None))) else self.img_live)
              cv2warning_imshow(img_live, True, 'main', scale_factor=self.scale_factor, waitKey=1)
    
    def stop(self):
        self.stop_flag = True
        cv2.destroyAllWindows()

class myVisualMonitor:
    def __init__(self, name, monitor_count=4, consequence_seconds=3, img_buffer_ubd=None,
                 scale_factor=0.3, is_warning_saveimg=False, exp_fd='image', click_table=None, warning_saveimg_passtime=3*60, 
                 is_warning_sound=False, frequency=1000, duration=1000, jumpup_when_warning=True, warning_window_shape=(1920,1080), 
                 **kwags):
        self.name = name
        self.img_buffer = []
        self.saveimg_buffer = []
        self.img_raw_buffer = []
        self.warning_window_shape = warning_window_shape
        self.monitor_count = monitor_count
        self.click_table = click_table
        self.scale_factor = scale_factor
        self.is_warning_saveimg = is_warning_saveimg
        self.exp_fd = exp_fd
        self.img_buffer_ubd = img_buffer_ubd if(img_buffer_ubd>0 if(isinstance(img_buffer_ubd, int)) else False) else self.monitor_count
        self.consequence_seconds = consequence_seconds
        self.is_warning_sound = is_warning_sound
        self.frequency = frequency
        self.duration = duration
        self.warning_saveimg_passtime = warning_saveimg_passtime
        self.jumpup_when_warning = jumpup_when_warning
        self.addlog = execute('addlog', kwags, default=addloger(logfile=execute('logfile', kwags, default='', not_found_alarm=False)), not_found_alarm=False)
        self.stop_flag = False
        self.t = threading.Thread(target=self._monitor)
        self.t.daemon = True
        self.t.start()
        self.w = None
        self.warning = False
        self.ws = threading.Thread(target=self._warning_saveimg)
        self.ws.daemon = True
        self.ws.start()
        self.dr = None

        #ctrl+c啟動正常結束程序
    def handle_signal(self,signum, frame):
        #self.adgProcLogger.debug('%s(%d) handling signal %r' ,self.__netLabel,self.pid, signum) #type(self).__name__
        self.stop_flag.set()
          
    def add_img(self, img, tm=None, stamp='', img_raw=None):
        if(len(self.img_buffer)>=self.img_buffer_ubd):
            self.img_buffer.pop(0)
        tm = tm if(isinstance(tm, dt)) else dt.now()
        stamp = stamp if(isinstance(stamp, str)) else ''
        self.img_buffer.append([tm, img, stamp])
        self.saveimg_buffer.append([tm, img, stamp]) if(self.is_warning_saveimg) else None
        self.img_raw_buffer.append([tm, img_raw, stamp]) if(not isinstance(img_raw, type(None))) else None
          
    def _monitor(self):
        try:
            log_counter = {}
            start_time = dt.now()
            while(not self.stop_flag):
                if((dt.now() - start_time).total_seconds()<1):
                    continue
                start_time = dt.now()
                if(len(self.img_buffer)<self.monitor_count):
                    continue
                pd_img_buffer = pd.DataFrame(self.img_buffer, columns=['tm','img','stamp'])
                DFP.calibration_tm(pd_img_buffer, 'tm')
                tmline = np.array(pd_img_buffer['tm'])
                tmline_oldest = tmline[:self.monitor_count]
                imgs_oldest = dcp(np.array(pd_img_buffer['img'].iloc[:self.monitor_count]))
                stamp_list = dcp(list(tuple(np.array(pd_img_buffer['stamp'].iloc[:self.monitor_count]))))
                if(not (np.abs(np.diff(tmline_oldest)) > self.consequence_seconds).any()):
                    if((not self.warning) and self.jumpup_when_warning):
                        self.warning = True
                        self.w = threading.Thread(target=self._warning, args=[imgs_oldest], kwargs={'stamp_list':stamp_list})
                        self.w.daemon = True
                        self.w.start()
                    self.addlog('tmline related to warning!!!', stamps=[self.name], log_counter=log_counter, log_counter_stamp='tmline irrelated', 
                                reset_log_counter=True, log_when_unreset=True)
                else:
                    self.addlog('tmline irrelated:%s'%(','.join(list(map(DFP.parse, tmline_oldest)))), stamps=[self.name],
                           log_counter=log_counter, log_counter_stamp='tmline irrelated')
                self.img_buffer = self.img_buffer[1:]
        except Exception as e:
            exception_process(e, logfile='', stamps=[self.name, self._monitor.__name__])
                      
    def _warning_saveimg(self, **kwags):
        try:
            start_time = dt.now()
            while(not self.stop_flag):
                if((dt.now() - start_time).total_seconds()<1):
                    continue
                start_time = dt.now()
                if(len(self.saveimg_buffer)==0):
                    continue
                warning_saveimg_timer = getattr(self ,'warning_saveimg_timer', dt.min)
                if((dt.now() - warning_saveimg_timer).total_seconds()>=self.warning_saveimg_passtime):
                    is_saved_this_time = False
                    pd_img_buffer = pd.DataFrame(self.saveimg_buffer, columns=['tm','img','stamp'])
                    img_raw_buffer = dcp(self.img_raw_buffer)
                    for i,img in enumerate(pd_img_buffer['img'].values):
                        stamp = DFP.parse(pd_img_buffer['stamp'].iloc[i])
                        tm = pd_img_buffer['tm'].iloc[i]
                        fn = stamp_process('',[stamp, 'NG'],'','','','_',for_file=True)
                        file = os.path.join(self.exp_fd, tm.strftime('%Y%m%d'), '%s.jpg'%fn)
                        if(not os.path.exists(file)):
                            if((dt.now() - warning_saveimg_timer).total_seconds()>=self.warning_saveimg_passtime):
                                is_saved_this_time = True
                                ta = dt.now() if(not isinstance(self.click_table, type(None))) else None
                                CreateFile(file, lambda f:cv2.imwrite(filename=f, img=img))
                                self.click_table.get(stamp, {}).update({'contours_saveimg':(dt.now() - ta).total_seconds()}) if(
                                    not isinstance(self.click_table, type(None))) else None
                                self.saveimg_buffer = self.saveimg_buffer[1:]
                        if(not img_raw_buffer):
                            continue
                        stamp = img_raw_buffer[i][2]
                        fn = stamp_process('',[stamp, 'NG','RAW'],'','','','_',for_file=True)
                        file = os.path.join(self.exp_fd, tm.strftime('%Y%m%d'), '%s.jpg'%fn)
                        if(not os.path.exists(file)):
                            if((dt.now() - warning_saveimg_timer).total_seconds()>=self.warning_saveimg_passtime):
                                CreateFile(file, lambda f:cv2.imwrite(filename=f, img=img_raw_buffer[i][1]))
                                self.img_raw_buffer = self.img_raw_buffer[1:]
                    setattr(self, 'warning_saveimg_timer', dt.now()) if(is_saved_this_time) else None
        except Exception as e:
            exception_process(e, logfile='', stamps=[self.name, self._warning_saveimg.__name__])
            
    def initial_windows(self, **kwags):
          for i in range(self.monitor_count):
              cv2warning_imshow(np.zeros((*self.warning_window_shape[:2],3)), True, 'warning_%s'%i, 
                                scale_factor=self.scale_factor, waitKey=1)
    
    def _warning(self, imgs, stamp_list=None, **kwags):
        # self._warning0(imgs, stamp_list=stamp_list, **kwags)
        stamp_list = mylist(stamp_list if(isinstance(stamp_list, list)) else [])
        for i,img in enumerate(imgs):
            cv2warning_imshow(img, True, 'warning_%s'%i, scale_factor=self.scale_factor, waitKey=1)
        ta = dt.now()
        while(True):
            if((dt.now() - ta).total_seconds()<1):
                continue
            ta = dt.now()
            if(self.is_warning_sound and m_winsound_import_succeed):
                winsound.Beep(self.frequency, self.duration)
            if(cv2.waitKey(1)==ord(' ')):
                break
      
    def _warning0(self, imgs, stamp_list=None, **kwags):
        stamp_list = mylist(stamp_list if(isinstance(stamp_list, list)) else [])
        for i,img in enumerate(imgs):
            cv2warning_imshow(img, True, 'warning_%s'%stamp_list.get(i,i), scale_factor=self.scale_factor, waitKey=1)
        ta = dt.now()
        while(True):
            if((dt.now() - ta).total_seconds()<1):
                continue
            ta = dt.now()
            if(self.is_warning_sound and m_winsound_import_succeed):
                winsound.Beep(self.frequency, self.duration)
            if(cv2.waitKey(1)==ord(' ')):
                break
        for i,img in enumerate(imgs):
             cv2.destroyWindow('warning_%s'%stamp_list.get(i,i))
        self.warning = False
        
    def stop(self):
        self.stop_flag = True
        cv2.destroyAllWindows()

class myPloter:
    def __init__(self, drawMethod, args=None, kwargs=None, name='', figsize=(15,8), exp_fd='.', fn='plotted', stamps=None, 
                 fig=None, **kwags):
        self.name = name
        self.stamps = stamps if(isinstance(stamps, list)) else []
        self.addlog = execute('addlog', kwags, default=addloger(logfile=execute('logfile', kwags, default='', not_found_alarm=False)), not_found_alarm=False)
        self.drawMethod = drawMethod
        self.figsize = figsize
        self.exp_fd = exp_fd
        self.fn = fn
        self.args = args if(DFP.isiterable(args)) else []
        self.kwargs = kwargs if(isinstance(kwargs, dict)) else {}
        self.fig = plt.Figure(figsize=self.figsize)

    def plot(self, args=None, kwargs=None, stamps=None, fig=None, ax=None, file=None, exp_fd=None, ret=None, **kwags):
        stamps = stamps if(isinstance(stamps, list)) else []
        if(isinstance(ax, plt.Axes)):
            if(isinstance(fig, plt.Figure)):
                m_print('ignore fig!!!!!', colora=LOGger.FAIL)
            ax.clear()
            fig = ax.get_figure()
        else:
            if(not isinstance(fig, plt.Figure)):    fig = self.fig
            fig.clf()
            ax = fig.add_subplot(1,1,1)
        args = args if(DFP.isiterable(args)) else self.args
        kwargs = kwargs if(isinstance(kwargs, dict)) else self.kwargs
        ret = ret if(isinstance(ret, dict)) else {}
        ret['success'] = self.drawMethod(*args, ax=ax, stamps=stamps, **kwargs)
        exp_fd = exp_fd if(LOGger.isinstance_not_empty(exp_fd, str)) else self.exp_fd
        file = file if(LOGger.isinstance_not_empty(file, str)) else os.path.join(
            exp_fd, '%s.jpg'%LOGger.stamp_process('',[self.fn]+self.stamps+stamps,'','','','_',for_file=True))
        m_print('plot saved at %s'%os.path.dirname(file), colora=LOGger.OKGREEN)
        LOGger.CreateFile(file, lambda f:end(fig, file=f))
        return True
    
def doPlot(drawMethod, stamps=None, fig=None, exp_fd=None, fn='plotted', file=None, **kwags):
    pltr = myPloter(drawMethod, stamps=stamps, **kwags)
    pltr.plot(fig=fig, exp_fd=exp_fd, fn=fn, file=file)
    return True

