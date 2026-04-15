# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 16:22:44 2021

@author: ian.ko
"""
import platform
from package import visualization as vs
import os
import sys
DFP = vs.DFP
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
#from matplotlib import cm
import scipy
import pandas as pd
import matplotlib.image as mpimg # mpimg 用於讀取圖片
from matplotlib import font_manager as fm #圖片字型
from copy import copy as cp
from copy import deepcopy as dcp
#import seaborn as sns

import random as rdm
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d
from datetime import datetime as dt
from sklearn.preprocessing import MinMaxScaler as Mmr
import matplotlib.dates as mdates
import threading
from multiprocessing import Event
import queue
import time
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

figsize=(15,8)
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
        'figsize':figsize, 'file':file, 'title':title,
        'title_fontsize':title_fontsize, 'mode':mode, 
        'ctr_tol':ctr_tol, 'xtkrot':xtkrot}
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
#%%
label_callback_key='labels'
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
#TODO:ands???
#def ands(*args, procedures=[], procedure=(lambda *args: True), ini_ret=True):
#    ret = True
#    for i, arg in enumerate(args):
#        if(isiterable(arg)):
#            ret &= procedures[i](*arg) if(i<len(procedures)) else procedure(*arg)
#        else:
#            ret &= procedures[i](arg) if(i<len(procedures)) else arg
#    return ret
#TODO:isiterable
def isiterable(a, exceptions=['str']):
    if(sum([(str(type(a)).find(ecpn)>-1)+0 for ecpn in exceptions])>0):
        return False
    try:
        iter(a)
        return True
    except:
        return False
#TODO:extract
def extract(container, index=0, key='', default=None):
    try:
        if(isiterable(container)):
            shape = np.array(container).shape
            printer('[extract][index:%s][container shape:%s]'%(str(index)[:200], shape), 
                    showlevel=5)
            return np.array(container[np.array(index)]) if(isiterable(index)) else (
                    container[index] if(index<shape[0] or -index<=shape[0]) else default)
        elif(type(container)==dict):
            printer('[extract][key:%s][container length:%d]'%(key, len(container)), 
                    showlevel=5)
            return container[key] if(key in container) else default
    except:
        pass
    return default
#TODO:get_all_values
def get_all_values(*args, only_numbers=1, decompose_layer_counts=np.inf, decompose_layer_counts_lbd=0, **kwags):
    values = []
    for arg in args:
        if(isiterable(arg) and decompose_layer_counts>decompose_layer_counts_lbd):
            values += get_all_values(*arg, only_numbers=only_numbers, decompose_layer_counts=decompose_layer_counts-1)
        else:
            values += ([arg] if(
                    str(type(arg)).find('float')>-1 or 
                    str(type(arg)).find('int')>-1) else []) if(only_numbers) else [arg]
    return values
#TODO:get_return
def get_return(method=np.min, *args, default_value=None, **kwags):
    ret = default_value
    try:
        ret = method(*args, **kwags) if(type(method(*args, **kwags))==dict) else [
                *get_all_values(method(*args, **kwags), decompose_layer_counts=1, only_numbers=0)]
    except:
        pass
    return (tuple(ret) if(np.array(ret).shape[0]>1) else ret[0]) if(isiterable(ret)) else ret
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
        do_args = [list(set(get_all_values(v))) for v in args[:2]] #X, Y內容分開算
        #資料內容在幾行之後
    elif(cook in ['a','x','y']):
        do_args = [list(set(get_all_values(args)))]*2 #所有數據算在一起
        #資料內容在幾行之後
    else:
        {}['cook error:%s'%cook]
    printer('[plot_set_axis_range] cook:%s'%cook, showlevel=2)
    
    # if(isnotaxis(do_args[0])):
    #     printer('[axis:x][plot_set_axis_range] not axis...', showlevel=3)
    #     {}['plot_set_axis_range stop']
    # printer('[plot_set_axis_range]x values:%s'%str(do_args[0])[:200], showlevel=4)
    X = get_all_values(do_args[0]) + get_all_values(vlines) if(not isnotaxis(do_args[0])) else []
    
    # if(isnotaxis(do_args[1])):
    #     printer('[axis:y][plot_set_axis_range] not axis...', showlevel=3)
    #     {}['plot_set_axis_range stop']
    # printer('[plot_set_axis_range]y values:%s'%str(do_args[1])[:200], showlevel=4)
    Y = get_all_values(do_args[1]) if(not isnotaxis(do_args[1])) else []
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
#TODO:add_subplot
def add_subplot(fig, fig_axes_start_index=-1, rowsize=1, colsize=1):
    n_fig_axes= len(fig.axes)
    printer('n_fig_axes_index=%s'%n_fig_axes, showlevel=2)
    if((fig_axes_start_index>=n_fig_axes or fig_axes_start_index<0) if(
                fig_axes_start_index!=None) else True):
        printer('rsz, csz, n_fig_axes_index+1=%s'%str((rowsize, colsize, n_fig_axes+1))[:200], showlevel=2)
        fig_axes_start_index = n_fig_axes if(n_fig_axes<rowsize*colsize) else 0
        return fig.add_subplot(rowsize, colsize, fig_axes_start_index+1)
    else:
        return fig.axes[fig_axes_start_index]
#TODO:initial_figure
def initial_figure(figsize=None, edge_plank=edge_plank, n_axes=1, layout=(), 
                   plot_set_figsize_on=0,
                   all_xvalues=[], all_yvalues=[], fig=None, **kwags):
    try:
        colsize, rowsize = 1, 1
        if(fig==None):
            figsize = plot_set_figsize(all_xvalues, all_yvalues,
                                       figsize, plot_set_figsize_on=plot_set_figsize_on, **kwags)
            n_axes += (len(fig.axes) if(fig!=None) else 0)
            printer('[initial_figure] n_axes=%d'%n_axes, showlevel=3)
            colsize = int(np.sqrt(n_axes)//1 if(
                    layout==()) else layout[1])
            rowsize = int((n_axes//colsize)+1 if(
                    layout==()) else layout[0])
            fig = plt.figure(figsize=figsize)
            printer('[initial_figure] figure size=(%.2f,%.2f)'%tuple(fig.get_size_inches()))
            printer('colsize, rowsize=(%s,%s)'%(colsize, rowsize), showlevel=3)
            fig.subplots_adjust(*edge_plank)
        return fig, colsize, rowsize
    except Exception as e:
        ax_errmsg(None, e, function_key='initial_figure')
        return None, None, None
#TODO:newcanvas
def newcanvas(figsize=None, suptitle='', edge_plank=edge_plank, **kwags):
    fig = initial_figure(figsize, edge_plank)[0]
    if(suptitle):
        fig.suptitle(suptitle, **({k.replace('suptitle_',''):v for k, v in kwags.items() if k in suptitle_keys}))
#    ax = fig.add_subplot(2,2,1)
#    ax.set_title('asdfg')
#    ax = fig.add_subplot(2,2,2)
#    ax.set_title('gh')
#    fig.show()
    return fig
#TODO:separate_kwags
def separate_kwags(kwags, index, key):
    do_kwags = dcp(kwags)
    for k in kwags:
        if(k[-1]=='s' and k!='s' and k[-2:]!='ss' and k!='ls' and k!='linestyles' and k[:-8]!='xycoords' and
           k[-3:]!='_ls'):
            content = extract(kwags[k], index=index, key=key)
            if(content!=None if(not isiterable(content)) else True):
                do_kwags[k[:-1]] = dcp(content)
            do_kwags.pop(k)
    return do_kwags
############################################################################################
#TODO:tools
############################################################################################
def tplize(obj):
    if(type(obj)==str):
        tp = (obj,) 
    else:
        try:
           tp = tuple(obj)
        except TypeError:
           tp = tuple([obj])
    return tp

def drop(dic, key):
    ddic = dic.copy()
    if(len(tplize(key))>1):
        for k in key:
            ddic = drop(ddic, k)
    else:
        if(key in ddic.keys()):
            ddic.pop(key)
    return ddic

def calculate_gaussian_kl_divergence(m1,m2,v1,v2):
    try:
        return np.log(v2 / v1) + ((v1**2)+(m1 - m2)**2)/(2*(v2**2)) - 0.5
    except:
        return 10*8

class my_rainbar():
    def __init__(self, n=5, a=0, b=1, c_alpha=1, default_color=np.array([0,0,0,1])):
        self.n = n
        self.a = a
        self.b = b
        self.c_alpha = c_alpha
        cmap = cm.rainbow(np.linspace(a, b, n))
        cmap[:,3] = np.array([c_alpha]*n)
        self.cmap = cmap
        self.default_color = default_color
    
    def get(self, i=0, c_alpha=None):
        if((i>=0 and i<np.array(self.cmap).shape[0]) if(isinstance(i, int)) else False):
            color = self.cmap[i]
        else:
            color = self.default_color
        color = color[:,c_alpha] if(not DFP.isnonnumber(c_alpha)) else color
        return color

def cm_rainbar(_n=5, _a=0, _b=1, _i=None, c_alpha=1):
    if(_i==None):
        ret = cm.rainbow(np.linspace(_a, _b, _n))
        ret[:,3] = np.array([c_alpha]*_n)
        return ret
    elif(_i>=0 and _i<_n):
        return (*tuple(cm.rainbow(np.linspace(_a, _b, _n)[_i][:3])), c_alpha)
    
############################################################################################
#TODO:plotting
############################################################################################
#TODO:has_twin
def has_twin(ax):
    for other_ax in ax.figure.axes:
        if other_ax is ax:
            continue
        if other_ax.bbox.bounds == ax.bbox.bounds:
            return True
    return False
#TODO:get_twin
def get_twin(ax, twin_index=0):
    if(not has_twin(ax)):
        return ax.twinx()
    other_ax_figure_axes = [v for v in ax.figure.axes if v!=ax]
    return other_ax_figure_axes[twin_index] if(twin_index<len(other_ax_figure_axes)) else ax.twinx()
#TODO:ax_msg
def ax_errmsg(ax, e, center=(1.2,2), xlim=(1,5), ylim=(1,5), 
              family='Consolas', fontsize = 10, function_key='', newline_count=5, **kwags):
    if(ax==None):
        fig, ax = plt.subplots()
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
    ax.text(*center, newmsg[:200], **({k.replace('text_',''):v for k, v in kwags.items() if k in textplot_keys}))
#TODO:ax_set_axtitle
def ax_set_axtitle(ax, **kwags):
    printer('[ax_set_axtitle]kwags:\n%s'%str(
            {(k if(k=='axtitle') else k.replace('axtitle_', '')):kwags[k]
             for k in kwags if k in axtitle_keys}), showlevel=4)
    if('axtitle' in kwags):
        axtitle = kwags['axtitle']
        ax.set_title(axtitle, **({(k if(k=='axtitle') else k.replace('axtitle_', '')):kwags[k]
                for k in kwags if k in axtitle_keys}))
#TODO:ax_set_annotation
def ax_set_annotation(ax, **kwags):
    printer('[ax_set_annotation]kwags:\n%s'%str(
            {(k if(k=='annotation') else k.replace('annotation_', '')):kwags[k]
             for k in kwags if k in annotation_keys}), showlevel=4)
    annotation = kwags['annotation'] if('annotation' in kwags) else None
    if(type(annotation)!=dict):
        return
    for anno_crd, anno in annotation.items():
        ax.annotate(anno, anno_crd, 
                    **({(k if(k=='annotation') else k.replace('annotation_', '')):kwags[k]
                for k in kwags if k in annotation_keys})) if('annotation' in kwags) else None
#TODO:ax_set_ticks
def ax_set_ticks(ax, **kwags):
    printer('[ax_set_ticks]kwags:\n%s'%str(
            {(k if(k=='xticks') else k.replace('xticks_', '')):kwags[k]
             for k in kwags if k in xticks_keys + yticks_keys}), showlevel=4)
    ax.set_xticks(**({(k if(k=='xticks') else k.replace('xticks_', '')):kwags[k]
                      for k in kwags if k in xticks_keys})) if('xticks' in kwags) else None
    ax.set_yticks(**({(k if(k=='yticks') else k.replace('yticks_', '')):kwags[k] 
                      for k in kwags if k in yticks_keys})) if('yticks' in kwags) else None
#TODO:ax_set_labels
def ax_set_labels(ax, **kwags):
    printer('[ax_set_labels]kwags:\n%s'%str(
            {(k if(k=='xlabel') else k.replace('xlabel_', '')):kwags[k]
             for k in kwags if k in xlabel_keys}), showlevel=4)
    ax.set_xlabel(**({(k if(k=='xlabel') else k.replace('xlabel_', '')):kwags[k]
                      for k in kwags if k in xlabel_keys})) if('xlabel' in kwags) else None
    ax.set_ylabel(**({(k if(k=='ylabel') else k.replace('ylabel_', '')):kwags[k] 
                      for k in kwags if k in ylabel_keys})) if('ylabel' in kwags) else None
    ax.set_xticklabels(**({('labels' if(k=='xticklabel') else k.replace('xticklabel_', '')):
        kwags[k] for k in kwags if k in xticklabel_keys})) if('xticklabel' in kwags) else None
    ax.set_yticklabels(**({('labels' if(k=='yticklabel') else k.replace('yticklabel_', '')):
        kwags[k] for k in kwags if k in yticklabel_keys})) if('yticklabel' in kwags) else None
    ax.tick_params(axis='x', **{k.replace('xticklabel_', ''):v for k,v in kwags.items() if k in xticklabel_keys})
    ax.tick_params(axis='y', **{k.replace('yticklabel_', ''):v for k,v in kwags.items() if k in yticklabel_keys})
#TODO:ax_set_vlines
#vline=[[1]]
#vline_label
def ax_set_vlines(ax, **kwags):
    printer('[ax_set_vlines]kwags:\n%s'%str(
            {(k if(k=='vline') else k.replace('vline_', '')):kwags[k]
             for k in kwags if k in vline_keys}), showlevel=4)
    vline = kwags.pop('vline', None)
    if(type(vline)!=list and type(vline)!=dict):
        return
    vline_show_key = kwags.pop('vlines_show_key', False)
    vline = {k:vline[k] for k in range(len(vline))} if(isinstance(vline, list)) else vline
    vline_labels = kwags['vline_labels'] if('vline_labels' in kwags) else (
            {0:kwags['vline_label']} if('vline_label' in kwags) else {})
    for i, (k, v) in enumerate(vline.items()):
        vls = get_all_values(v)
        vline_label = vline_labels[k] if(k in vline_labels) else ''
        vline_label = '[%s]%s'%(k, vline_label) if(vline_show_key) else ''
        if(vline_label):
            kwags['label'] = dcp(vline_label)
        for i in vls:
            ax.axvline(v, **({(k if(k=='vline') else k.replace('vline_', '')):kwags[k]
                      for k in kwags if k in vline_keys})) if(
                      vline!={}) else None
#TODO:make_cmap
#nodes = [0, scr.transform([[0]])[0][0], 1]
def make_cmap(colors=[(0,0,0.8), (1,1,1), (0.8,0,0)], nodes=[-1,0,1000], scr=None, name='custom', 
              bad_value='k', over_value=-1, under_value=0, single_value_method=None, isshow=False,
              c_alpha = 1, call_name='', find_names=False, is_discrete=False):
    """
    

    Parameters
    ----------
    colors : TYPE, optional
        DESCRIPTION. The default is [(0,0,0.8), (1,1,1), (0.8,0,0)].
    nodes : TYPE, optional
        DESCRIPTION. The default is [-1,0,1000]. #輸出的物件一定會平移至0~1
    scr : TYPE, optional
        DESCRIPTION. The default is None.
    name : TYPE, optional
        DESCRIPTION. The default is 'custom'.
    bad_value : TYPE, optional
        DESCRIPTION. The default is 'k'.
    over_value : TYPE, optional
        DESCRIPTION. The default is -1.
    under_value : TYPE, optional
        DESCRIPTION. The default is 0.
    single_value_method : TYPE, optional
        DESCRIPTION. The default is None.
    isshow : TYPE, optional
        DESCRIPTION. The default is False.
    c_alpha : TYPE, optional
        DESCRIPTION. The default is 1.
    call_name : TYPE, optional
        DESCRIPTION. The default is ''.
    find_names : TYPE, optional
        DESCRIPTION. The default is False.
    is_discrete : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    cmap_custom : TYPE
        DESCRIPTION.

    """
    if(find_names):
        print(','.join(['dark_bg']))
        return 
    if(call_name in ['dark_bg']):
        if(call_name=='dark_bg'):
            cmap_custom = make_cmap([(0.2,0.05,0.4,c_alpha),(0.05,0.2,0.3,c_alpha),(0.2,0.4,0.05,c_alpha)], 
                                    [0,0.5,1], name=call_name)
    elif(call_name in ['',None]):
        if(np.array(colors).shape[0]!=np.array(nodes).shape[0] and not is_discrete):
            printer('[make cmap]color nodes count[%d] != data nodes count[%d]'%(
                                np.array(colors).shape[0], np.array(nodes).shape[0]))
            return None
        scr = Mmr((0.1,0.9)).fit(np.array(nodes).reshape(-1,1)) if(scr==None) else scr
        nodes = list(tuple(scr.transform(np.array(nodes).reshape(-1,1)).ravel()))
        printer(nodes, emphasis='-', showlevel=3)
        nodes = [0] + nodes[1:-1] + [1] #LinearSegmentedColormap一定要從0到1結尾
        printer(nodes, emphasis='-', showlevel=3)
        colors = list(map((lambda t:(*t[:3], (c_alpha if(len(t)<4) else t[3]))), colors))
        cmap_custom = mcolors.LinearSegmentedColormap.from_list(
            'cmap_%s'%name, list(zip(nodes, colors))) if(not is_discrete) else mcolors.ListedColormap(
                colors, name='cmap_%s'%name)
        cmap_custom.set_bad(bad_value) if(np.array(bad_value).reshape(-1).shape in [(3,),(4,)] or 
                                          isinstance(bad_value, str)) else None
        cmap_custom.set_over(over_value) if(np.array(over_value).reshape(-1).shape in [(3,),(4,)] or 
                                          isinstance(over_value, str)) else (
            mylist(colors).get(over_value, None) if(isinstance(over_value, int)) else None)
        cmap_custom.set_under(under_value) if(np.array(under_value).reshape(-1).shape in [(3,),(4,)] or 
                                          isinstance(under_value, str)) else (
            mylist(colors).get(under_value, None) if(isinstance(under_value, int)) else None)
        if(single_value_method!=None):
            #如果要對應的數據只有一個值，那要給什麼顏色
            # Eg: single_value_method = lambda v,**kwags: tuple(np.clip(
            #     np.array([0,0,1]) + v*(np.array([1,0,0]) - np.array([0,0,1])), 0, 1)
            single_value_method = (lambda v,**kwags: tuple(np.clip(
                np.array(colors[0]) + v*(np.array(colors[-1]) - np.array(colors[0])), 0, 1))) if(
                    single_value_method=='prob_standard') else single_value_method
            cmap_custom.single_value_method = single_value_method
    else:
        print('call_name %s is undefined'%call_name)
        return None
    if(isshow):
        show_cmap(cmap_custom)
    return cmap_custom

#TODO:get_n_graph
def get_n_graph(*args, mode):
    printer('[get_n_graph]args:\n%s'%str(args)[:200], showlevel=5)
    if(mode in ['x', 'd', 'y', 'r']):
        n_graph = max([len(arg) for arg in args])
    if(mode in ['s']):#'s' for single dimension working
        n_graph = len(args[0][0])
    else:
        n_graph = np.prod([len(arg) for arg in args])
    return n_graph
#TODO:multigraph_process
def multigraph_process(*args, plot_method=None, mode='d', keys={}, 
                       default_axis_values=0, default_range=(0,1), 
                       showmask=[], fig=None, **kwags):
    try:
        if(str(type(plot_method)).find('function')==-1):
            printer('[multigraph_process]plot method error:%s'%plot_method)
            return
        printer('[multigraph_process]args shape:%s'%(str(np.array(args).shape)), showlevel=1)
        printer('[multigraph_process]args:\n%s'%(str([str(v)[:100] for v in args])[:200]), showlevel=5)
        defaults = [default_axis_values]*len(args)
        printer('[multigraph_process]colors in kwags:%s'%('colors' in kwags), showlevel=4)
        printer('[multigraph_process]labels in kwags:%s'%('labels' in kwags), showlevel=4)
        printer('[multigraph_process]fig_axes_start_index in kwags %s'%(
                    'fig_axes_start_index' in kwags), showlevel=4)
        #TODO:[multigraph_process]mode d,x,y,a set colors
        if('colors' in kwags):
            n_graph = get_n_graph(args, mode=mode)
            c_alpha = kwags['c_alpha'] if('c_alpha' in kwags) else 1
            c_alpha = max(min(c_alpha, 1), 0)
            kwags['colors'] = [v for v in cm_rainbar(
                    _n=n_graph)] if(kwags['colors']=='rainbow'
                    ) else kwags['colors']
            kwags['colors'] = [(*v[:3], c_alpha) for v in kwags['colors']] if(
                    'c_alpha' in kwags) else kwags['colors']
            printer('colors:%s'%str(kwags['colors'])[:200], showlevel=4)
        if(mode in ['s','s:x', 's:y'] and not 'cook' in kwags):
            kwags['cook'] = 'x'
        printer('[multigraph]args:\n%s'%(str(args)[:200]), showlevel=5)
        paint = Paint(plot_method=plot_method, **kwags)
        index = 0
        printer('[multigraph]mode:%s....'%(mode), showlevel=2)
        printer('-------------------------------------------------', showlevel=2)
        if(mode in ['d', 'x', 'y', 'a']):
            for i, x in enumerate(args[0] if(len(args)>0) else [[defaults[0]]]):
                for j, y in enumerate(args[1] if(len(args)>1) else [[defaults[1]]]):
                    if((i!=j) if(mode=='d') else False):
                        continue
                    if((i!=0) if(mode=='x') else False):
                        continue
                    if((j!=0) if(mode=='y') else False):
                        continue
                    if(np.array(x).shape!=np.array(y).shape):
                        data_infrm(x, operate_in='multigraph', name='x')
                        data_infrm(y, operate_in='multigraph', name='y')
                        continue
                    key = dcp((keys[(i,j)] if((i,j) in keys) else (i, j)) if(
                            type(keys)==dict) else (
                              (keys[index] if(index<len(keys)) else (i, j)) if(
                                isinstance(keys, list)) else (i, j)))
                    if(key not in showmask if(showmask) else False):
                        continue
                    #這裡要定義同一個座標系裡不同系統的圖形的property
                    do_kwags = separate_kwags(kwags, index=index, key=key)
                    printer('[multigraph][(%d,%d)][%s][%d]adding....'%(i, j, key, index), showlevel=3)
                    printer('[multigraph]do_kwags:%s'%str(do_kwags)[:200+max(0, showlog-5)*50], 
                            showlevel=5)
                    printer('[multigraph]x:\n%s'%(str(x)[:200]), showlevel=4)
                    printer('[multigraph]y:\n%s'%(str(y)[:200]), showlevel=4)
                    plot_method = do_kwags.pop('method', plot_method)
                    paint.add(x, y, method=plot_method, key=0, graph_key=key, **do_kwags)
                    index += 1
                    printer('-------------------------------------------------', showlevel=2)
    #                {False:0}[index>1]
        elif(mode in ['s'] or mode.find('s:')>-1):
            args = args[0] #只取第一維度
            for index, arg in enumerate(args):
                printer('[multigraph][mode:s][index:%d]arg:\n%s...'%(index, str(arg)[:200]))
                key = dcp(keys[index] if(index in keys) else index)
                if(key not in showmask if(showmask) else False):
                    continue
                #這裡要定義同一個座標系裡不同系統的圖形的property
                do_kwags = separate_kwags(kwags, index=index, key=key)
                printer('[multigraph][mode:s]do_kwags:%s'%str(do_kwags)[:200], showlevel=5)
                printer('[multigraph][mode:s][%d-%d][%s]graph adding....'%(index, index, key), showlevel=3)
                printer('[multigraph][mode:s]a:\n%s'%(str(arg)[:200]), showlevel=4)
                paint.add(arg, method=plot_method, key=0, graph_key=(key, index), **do_kwags)
                printer('-------------------------------------------------', showlevel=2)
        printer('[multigraph][%s] paint.add done'%mode, showlevel=3)
        paint_kwags = dcp(kwags)
        paint_kwags.update({k:locals()[k] for k in paint_keys if k in locals()})
        paint_kwags.update({k:kwags[k] for k in paint_keys if k in kwags})
        printer('[multigraph] paint_kwags:\n%s'%(','.join(list(paint_kwags.keys()))))
        paint.draw(fig=fig, is_adding_ax=False, **paint_kwags)
    except Exception as e:
        ax = fig.add_subplot(1,1,1) if(fig!=None) else None
        ax_errmsg(ax, e, function_key='multigraph_process')
        return None
#TODO:multiXYgraph_process
def multiXYgraph_process(X, Y, plot_method, mode='d', keys={}, file='', title='',
                         default_axis_values=0, default_range=(0,1), 
                         showmask=[], fig=None, **kwags):
    printer('[multiXYgraph_process]colors in kwags:%s'%('colors' in kwags), showlevel=4)
    printer('[multiXYgraph_process]fig_axes_start_index in kwags %s'%(
                'fig_axes_start_index' in kwags), showlevel=4)
    #TODO:set colors
    if('colors' in kwags):
        n_graph = len(get_all_values(X))
        c_alpha = kwags['c_alpha'] if('c_alpha' in kwags) else 1
        c_alpha = max(min(c_alpha, 1), 0)
        kwags['colors'] = [v for v in cm_rainbar(
                _n=n_graph)] if(kwags['colors']=='rainbow'
                ) else kwags['colors']
        kwags['colors'] = [(*v[:3], c_alpha) for v in kwags['colors']] if(
                'c_alpha' in kwags) else kwags['colors']
        printer(str(kwags['colors'])[:200], showlevel=4)
    paint = Paint(plot_method=plot_method, **kwags)
    index = 0
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            if((i!=j) if(mode=='d') else False):
                continue
            if((i!=0) if(mode=='x') else False):
                continue
            if((j!=0) if(mode=='y') else False):
                continue
            if(np.array(x).shape!=np.array(y).shape):
                data_infrm(x, operate_in='multiXYgraph', name='x')
                data_infrm(y, operate_in='multiXYgraph', name='y')
                continue
            key = dcp((keys[(i,j)] if((i,j) in keys) else (i, j)) if(
                    type(keys)==dict) else (
                      (keys[index] if(index<len(keys)) else (i, j)) if(
                        isinstance(keys, list)) else (i, j)))
            if(key not in showmask if(showmask) else False):
                continue
            printer('[(%d,%d)][%s][%d]graph adding....'%(i, j, key, index), showlevel=2)
            do_kwags = dcp(kwags)
            for k in kwags:
                if(k[-1]=='s' and k!='s' and k[-2:]!='ss' and k!='ls' and k!='linestyles'):
                    content = extract(kwags[k], index=index, key=key)
                    if(content!=None if(not isiterable(content)) else True):
                        do_kwags[k[:-1]] = dcp(content)
                    do_kwags.pop(k)
            paint.add(x, y, method=plot_method, key=0, graph_key=key, **do_kwags)
            index += 1
    printer('[multiXYgraph] paint.add done', showlevel=3)
    paint_kwags = dcp(kwags)
    paint_kwags.update({k:locals()[k] for k in paint_keys if k in locals()})
    printer('[multiXYgraph] paint_kwags:\n%s'%(','.join(list(paint_kwags.keys()))))
    paint.draw(fig=fig, is_adding_ax=False, **paint_kwags)
#TODO:draw_ending
def draw_ending(fig, is_end=True, file='', title='', title_fontsize=title_fontsize, showimg=False, **kwags):
    if(is_end):
        printer('[draw_ending] fig id:%s'%id(fig), showlevel=3)
        if(title!=''):
            printer('[draw_ending] title:%s'%title, showlevel=5)
            fig.suptitle(title, **({k.replace('suptitle_',''):v for k, v in kwags.items() if k in suptitle_keys})) if(
                hasattr(fig, 'suptitle')) else None
        for ax in fig.axes:
            ax.legend(**({k:kwags[k] for k in kwags if k in legend_keys}))
        if(isinstance_not_empty(file, str)):
            printer('[draw_ending] file:%s'%file, showlevel=3)
            fig.savefig(file)
        fig.show() if(showimg) else None
#        plt.ioff()
#        return None
#    else:
#        return fig
def F(plot_method, plot_method_keys):
    return lambda X,Y,**kwags:xyplot(X, Y, plot_method=plot_method, plot_method_keys=plot_method_keys, 
        **{k:v for (k,v) in kwags.items() if 
                 not k in ['plot_method','plot_method_keys']})


#TODO:xyplot
def xyplot(X, Y, mode='d', 
           plot_method=lambda X, Y, axx, **kwags: axx.plot(X, Y, **kwags), 
           plot_method_keys = basic_keys, default_range=(0,1), 
           showmask=[], keys={}, title='', file='', 
           is_ax_set_labels=1, fig=None, ax=None, **kwags):
    try:
        fig, ncols, nrows = initial_figure(fig=fig, **kwags)
        printer('[xyplot][X, Y] shape:%s'%(str(np.array([X, Y]).shape)), showlevel=1)
        printer('[X, Y]:%s'%str([X, Y])[:200], showlevel=4)
        if(sum([1 if((isiterable(arg[0]) if(len(arg)>0) else 0) if(
                isiterable(arg)) else 0) else 0 for arg in [X, Y]])==len([X, Y])):
            printer('[xyplot] multi graph...', showlevel=3)
            #TODO:[xyplot]需要multigraph_process
            multigraph_process(X, Y, 
                               plot_method=F(plot_method, plot_method_keys),
                               mode=mode, keys=keys, default_range=default_range, 
                               file=file, title=title, fig=fig, **kwags)
            return
        printer('[xyplot] single graph... with ax:%s'%ax, showlevel=3)
    #    if(ax==None):
    #        printer('ax None', showlevel=2)
    #        return None
        if(ax==None):
            ax = add_subplot(fig, rowsize=nrows, colsize=ncols)
        printer('[xyplot] kwags:\n%s...'%(','.join([str(v) for i,v in enumerate(
                                kwags.keys()) if i<(showlog*5)])), showlevel=3)
        if(is_timestamp(X)):
            datetimeformat = kwags.get('datetimeformat', '%Y-%m-%d %H:%M:%S.%f')
            printer('[xyplot]x為時間資料!!!!!!!!!!')
            ax.xaxis.set_major_formatter(mdates.DateFormatter(datetimeformat))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plot_method(X, Y, ax=ax, **({k:kwags[k] for k in kwags if k in plot_method_keys}))
        ax_set_labels(ax, **kwags) if(is_ax_set_labels) else None
        ax_set_annotation(ax, **{k:kwags[k] for k in kwags if type(k)==str}) if(is_ax_set_labels) else None
        ax_set_vlines(ax, **{k:kwags[k] for k in kwags if type(k)==str})if(is_ax_set_labels) else None
        #如果是從Paint進來的，不需要回傳fig；但如果是外部來的，會需要回傳
        return
    except Exception as e:
        ax_errmsg(ax, e, function_key='xyplot')
#TODO:vlinesplot_core
def vlinesplot_core(*args, ax=None, label_at_locs=[0], **kwags):
    try:
        printer('vlinesplot_core:%s'%str(id(ax)), emphasis='-')
        args = args[0]
        label_at_locs = range(len(args)) if(label_at_locs == None) else label_at_locs
#        print('[vlinesplot_core]%s'%str(args)[:200])
        for i, arg in enumerate(args):
            do_kwags = dcp(kwags)
            do_kwags['label'] = (kwags['label'] if('label' in kwags) else None) if(i in label_at_locs) else None
            ax.axvline(arg, **({k:do_kwags[k] for k in kwags if k in basic_keys}))
    except Exception as e:
        ax_errmsg(ax, e, function_key='vlinesplot_core')
#TODO:sgplot(single plot)
def sgplot(D, mode='s', 
           plot_method=vlinesplot_core, 
           plot_method_keys = basic_keys, default_range=(0,1), 
           showmask=[], keys={}, title='', file='', 
           is_ax_set_labels=1, fig=None, ax=None, **kwags):
    try:
        fig = initial_figure(fig=fig, **kwags)[0]
        printer(str(D)[:200], emphasis='-', showlevel=5)
        if(str(type(D[0])).find('pandas.core.frame')==-1):
#        if(True):
            printer('[sgplot]D shape:%s'%(str(np.array([D]).shape)), showlevel=1)
            if(sum([1 if((isiterable(arg[0]) if(len(arg)>0) else 0) if(
                    isiterable(arg)) else 0) else 0 for arg in [D]])==len([D])):
                printer('[sgplot] multi graph...', showlevel=3)
                #TODO:[sgplot]需要multigraph_process
                multigraph_process(D, 
                                   plot_method=lambda D, ax=None, plot_method=plot_method, **kwags:sgplot(
                                   D, ax=ax, plot_method=plot_method, plot_method_keys=plot_method_keys, **kwags), 
                                   mode=mode, keys=keys, default_range=default_range, 
                                   file=file, title=title, fig=fig, **kwags)
                return
        printer('[sgplot] single graph... with ax:%s'%ax, showlevel=3)
        if(ax==None):
            ax = add_subplot(fig)
        printer('[sgplot] kwags:\n%s...'%(','.join([str(v) for i,v in enumerate(
                                kwags.keys()) if i<(showlog*5)])), showlevel=3)
        plot_method(D, ax=ax, **({k:kwags[k] for k in kwags if k in plot_method_keys}))
        ax_set_labels(ax, **kwags) if(is_ax_set_labels) else None
        ax_set_annotation(ax, **{k:kwags[k] for k in kwags if type(k)==str})if(is_ax_set_labels) else None
        ax_set_vlines(ax, **{k:kwags[k] for k in kwags if type(k)==str})if(is_ax_set_labels) else None
        #如果是從Paint進來的，不需要回傳fig；但如果是外部來的，會需要回傳
        return
    except Exception as e:
        ax_errmsg(ax, e, function_key='sgplot')
#TODO:curveplot
def curveplot(X, Y, mode='d', showmask=[], keys={}, title='', file='', 
              is_ax_set_labels=1, fig=None, ax=None, **kwags):
    fig = initial_figure(fig=fig, **kwags)[0]
    plot_method=lambda *args, ax=None, **kwags: ax.plot(*args, **kwags)
    xyplot(X, Y, mode=mode, plot_method=plot_method, plot_method_keys=curveplot_keys,
                  showmask=showmask, keys=keys, title=title, file=file, 
                  is_ax_set_labels=is_ax_set_labels, fig=fig, ax=ax, **kwags)
    return 
#TODO:scatterplot
def scatterplot(X, Y, mode='d', showmask=[], keys={}, title='', file='', 
              is_ax_set_labels=1, fig=None, ax=None, **kwags):
    fig = initial_figure(fig=fig, **kwags)[0]
    plot_method=lambda *args, ax=None, **kwags: ax.scatter(*args, **kwags)
    #解決在scatterplot函式中，參數color不能與c並存的問題
    if(mylist(np.array(kwags.get('c', [])).shape).get(0, 0)>0 and 
        isinstance(kwags.get('cmap', None), matplotlib.colors.LinearSegmentedColormap)):
        kwags.pop('color', None)
        if(np.unique(kwags['c']).shape[0]==1):
            # #TODO:處理single value for cmap
            cmap = kwags['cmap']
            the_c = cmap.single_value_method(kwags['c'][0]) if(hasattr(cmap, 'single_value_method')) else None
            kwags['c'] = the_c if(the_c!=None) else kwags['c']
    xyplot(X, Y, mode=mode, plot_method=plot_method, plot_method_keys=scatterplot_keys,
                  showmask=showmask, keys=keys, title=title, file=file, 
                  is_ax_set_labels=is_ax_set_labels, fig=fig, ax=ax, **kwags)
    return
#TODO:vlinesplot
def vlinesplot(*args, showmask=[], keys={}, title='', file='', 
              is_ax_set_labels=1, fig=None, ax=None, **kwags):
    fig = initial_figure(fig=fig, **kwags)[0]
    sgplot(*args, plot_method=vlinesplot_core,
                  showmask=showmask, keys=keys, title=title, file=file, 
                  is_ax_set_labels=is_ax_set_labels, fig=fig, ax=ax, **kwags)
    return
#TODO:nrmdsplot_core
def nrmdsplot_core(arg, ax=None, num_bins=100, xmax=None, xmin=None, showinfrm=True, analysis_alpha=0.3, **kwags):
    try:
        diversity = len(list(set(tuple(arg))))
        allvalues = get_all_values(tuple(arg))
        xmax, xmin = (np.max(allvalues) if(xmax==None) else xmax), (np.min(allvalues) if(xmin==None) else xmin)
        x_ = np.linspace(xmin, xmax, num_bins)
        (mu, sigma) = scipy.stats.norm.fit(arg)
#            print(arg, mu, sigma, sep='||')
        print(mu, sigma, sep='||')
        if(showinfrm):
            outer_label = kwags['label'] if('label' in kwags) else None
            kwags['label'] = r'size: %d, $\mu$:%.3f, $\sigma$:%.3f%s'%(
                    np.array(arg).shape[0], mu, sigma, ('\n%s'%outer_label if(outer_label) else '')) 
        printer('nrmdsplot_core:%s'%str(id(ax)), emphasis='-')
        n, bins, patches = ax.hist(arg, **kwags)
        if(diversity>=2):
            axtw = get_twin(ax, 0)
            y = scipy.stats.norm.pdf(x_, mu, sigma)
            print(np.max(y), np.min(y), sep='||')
            axtw.plot(x_, y, '--', color=((*kwags['color'][:3], analysis_alpha) if(
                    DFP.isiterable(kwags.get('color',None))) else (0,0,0,analysis_alpha)))
        ax.autoscale(enable=True, axis='y')
        printer('nrmdsplot_core done!', emphasis='-')
    except Exception as e:
        ax_errmsg(ax, e, 'nrmdsplot_core')
#TODO:nrmdsplot
def nrmdsplot(*args, showmask=[], keys={}, title='', file='',
              is_ax_set_labels=1, fig=None, ax=None, **kwags):
    fig = initial_figure(fig=fig, **kwags)[0]
    sgplot(*args, plot_method=nrmdsplot_core,
                 showmask=showmask, keys=keys, plot_method_keys=nrmdsplot_keys, title=title, file=file, 
                 is_ax_set_labels=is_ax_set_labels, fig=fig, ax=ax, **kwags)
    return
#TODO:show_cmap
def show_cmap(cmap, norm=None, extend=None):
    if norm is None:
        norm = mcolors.Normalize(vmin=0, vmax=cmap.N)
    im = cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    fig.colorbar(im, cax=ax, orientation='horizontal')
    plt.show()
#TODO:implot_core
def implot_core(arg, ax=None, **kwags):
    try:
        arg = arg[0]
        scr = Mmr((0.1,0.9)).fit(np.array(arg).reshape(-1,1))
#        str_type = str(type(arg))
        arg = pd.DataFrame(arg)
        arg = arg.applymap(lambda x:scr.transform([[x]])[0][0])
        printer(str(arg), emphasis='-')
        if('make_cmap_colors_' in kwags and 'make_cmap_nodes_' in kwags):
            printer('[implot_core]make cmap', showlevel=3)
            colors = kwags.pop('make_cmap_colors_')
            nodes = kwags.pop('make_cmap_nodes_')
            cmap_custom = make_cmap(colors, nodes, scr=scr)
            kwags['cmap'] = cmap_custom
#        if(not 'extent' in kwags):
#            if(not np.array(list(map(DFP.isnonnumber, arg.columns)).any()):
#                xaxis = list(map(float, arg.columns))
#                yaxis = list(map(float, arg.columns))
#                extent= (xaxis[0], xaxis), max(yaxis), min(yaxis)
#            arg.columns
#        fig, ax = plt.subplots()
#        fig = plt.figure()
#        ax = fig.add_subplot(1,1,1)
        ax.imshow(arg, **({k:v for (k,v) in kwags.items() if k in implot_core_keys}))
        printer('implot_core done!', emphasis='-')
    except Exception as e:
        ax_errmsg(ax, e, 'implot_core')
#TODO:implot
def implot(*args, showmask=[], keys={}, title='', file='',
              is_ax_set_labels=1, fig=None, ax=None, **kwags):
    fig = initial_figure(fig=fig, **kwags)[0]
    sgplot(*args, plot_method=implot_core,
                 showmask=showmask, keys=keys, plot_method_keys=implot_keys,
                 title=title, file=file, 
                 is_ax_set_labels=is_ax_set_labels, fig=fig, ax=ax, 
                 layout=(1,1), plot_set_axis_range_on=False, **kwags)
    return
#TODO:regression_validation_plot
def regression_validation_plot(X, Y, tol=None, mode='d', showmask=[], keys={}, title='', file='', 
              label=None, is_ax_set_labels=1, scatter_color=(5/255,80/255,220/255,0.5), 
              bundary_color = (255/255,90/255,90/255), fig=None, ax=None, **kwags):
    all_X, all_Y, all_v = get_all_values(X), get_all_values(Y), get_all_values([X, Y])
    amin, amax = min(all_v), max(all_v)
    fig = initial_figure(all_xvalues=all_X, all_yvalues=all_Y, **kwags)[0] #figsize, plot_set_figsize_on, all_xvalues, all_yvalues
    ax = fig.add_subplot(1, 1, 1)
    
    scatterplot(X[0], Y[0], fig=fig, ax=ax, color=scatter_color, label=label)
    
    if(tol==None):
        tol = np.std(np.array(Y).reshape(-1))
    curveplot(np.linspace(amin, amax, 100), np.linspace(amin, amax, 100), fig=fig, ax=ax, ls='--', color=bundary_color)
    curveplot(np.linspace(amin, amax, 100)+tol, np.linspace(amin, amax, 100), fig=fig, ax=ax, ls='--', color=bundary_color)
    curveplot(np.linspace(amin, amax, 100)-tol, np.linspace(amin, amax, 100), fig=fig, ax=ax, ls='--', color=bundary_color, 
                  xlabel=kwags.get('xlabel', None), ylabel=kwags.get('ylabel',None))
    if(label):
        ax.legend(**({k:kwags[k] for k in kwags if k in legend_keys}))
    if(title):
        fig.suptitle(title, **({k.replace('suptitle_',''):v for k, v in kwags.items() if k in suptitle_keys}))
    if(file):
        fig.savefig(file)
    
#%%
# if(False):
# #%%
# #TODO:test
#     colors = ['darkorange', 'gold', 'lawngreen', 'lightseagreen']
#     colors = [(0,0,0.8), (1,1,1), (0.8,0,0)]
#     cmap = mcolors.ListedColormap(colors)
#     show_cmap(cmap)
    
#     nodes = [0, 0.5 , 1]
#     cmap_mywarmcool = mcolors.LinearSegmentedColormap.from_list(
#         'mywarmcool', list(zip(nodes, colors)))
#     show_cmap(cmap1)
    
#     fig, ax = plt.subplots(figsize=(10,5))
#     curveplot([[1,2,3,4,5]], [[1,2,3,4,5]], labels=['line'], color=(0,0.4,0.4,0.5), vlines=[[2.5]])
#     curveplot([[1,2,3,4,5],[2,3,4,5]], [[1,2,3,4,5],[3,3,3,3]], colors='rainbow')
#     # vs2.curveplot([[1,2,3,4,5]], [[1,2,3,4,5], [6,7,8,9,10]], 
#     #           mode='x', labels=['a','b'], colors=[(0,0.4,0.4,0.3), (0.2,0.4,0)], 
#     #           ylabel='ylb', xlabel='xlb', rotation=270, xlabel_rotation=70)
# #    curveplot([[1,2,3,4,5]], [[1,2,3,4,5], [6,7,8,9,10]], fig=fig, fig_axes_start_index=0,
# #              mode='x', labels=['a','b'], colors=[(0,0.4,0.4,0.3), (0.2,0.4,0)], 
# #              c_alpha=0.6)
#     curveplot([[1,2,3,4,5]], [[1,2,3,4,5], [6,7,8,9,10]],
#               mode='x', labels=['a','b'], colors=[(0,0.4,0.4,0.3), (0.2,0.4,0)], 
#               c_alpha=0.1)
#     curveplot([np.linspace(0,10,100)], [np.linspace(0,10,100), 10-np.linspace(0,10,100)], mode='x', 
#                  annotations=[{(5, 5):'center', (2, 2):'hi'}])
#     scatterplot([np.linspace(0,10,100)], [np.linspace(0,10,100), 10-np.linspace(0,10,100)], mode='x', labels=['a','b'])
#     scatterplot([np.linspace(0,10,100)], [np.linspace(0,10,100), 10-np.linspace(0,10,100)], mode='x', keys=['a','b'])
#     scatterplot([np.linspace(0,10,100)], [np.linspace(0,10,100)], figsize=(8, 5), file='test.jpg')
    
#     vs2.regression_validation_plot(np.linspace(0,10,100), np.linspace(0,10,100), xlabel='x', ylabel='y', title='test',
#                                    label='hi', edge_plank=(0,0,1,0.92))
    
    
#     ####
#     fig = vs2.initial_figure()[0]
#     ax = fig.add_subplot(1, 1, 1)
#     vs2.scatterplot(np.linspace(0,10,100), np.linspace(0,10,100), fig=fig, ax=ax, label='hi')
#     vs2.curveplot(np.linspace(0,10,100)+1, np.linspace(0,10,100), fig=fig, ax=ax)
#     vs2.curveplot(np.linspace(0,10,100)-1, np.linspace(0,10,100), fig=fig, ax=ax, ls='--', xlabel='x')
#     vs2.vlinesplot([1,2,3], fig=fig, ax=ax, color='b')
#     ax.legend()
#     ####
#     xyplot([[1,2,3,4,5]], [[1,2,3,4,5], [6,7,8,9,10]], mode='x')
#     #xyplot([1,2,3,4,5], [6,7,8,9,10], ax=0, label='hi',
#     #       plot_method=lambda *args, ax=None, **kwags: ax.scatter(*args, **kwags))
#     #xyplot([[1,2,3,4,5]], [[1,2,3,4,5], [6,7,8,9,10]], label='hi', mode='x',
#     #       plot_method=lambda *args, ax=None, **kwags: ax.scatter(*args, **kwags))
#     #xyplot([1,2,3,4,5], [6,7,8,9,10], ax=ax)
# #    xyplot([1,2,3,4,5], [6,7,8,9,10], ax=0)
# #    print(1)
    
#     fig = plt.figure(figsize=(3,4))
#     sgplot([[1,2,3,4,5]], fig=fig, fig_axes_start_index=-1)
#     sgplot([[1,2,3,4,5]])
#     sgplot([[1,2,3,4,5]], colors=[(0,0,1), (0,1,0), (1,0,0)]) #因為1,2,...,5被框在同一個[]裡，所以只套用第一個顏色
#     sgplot([[1],[2],[3],[4],[5]], colors='rainbow', labels=['p','e','a','c','e'], ls='--')
#     sgplot([pd.Series([1,1,1.5,1.2,1.3,1.5,2,1,3,4,5])], plot_method=nrmdsplot_core) #XX!!!!!
#     nrmdsplot([pd.Series([1,1,1.5,1.2,1.3,1.5,2,1,3,4,5]), pd.Series([5,5,4,4,4,1,4,4,1,1])], 
#                colors=[(0,0,0.8, 0.3), (0.8,0,0, 0.3)], analysis_alpha=1)
    
#     fig, ax = plt.subplots()
#     nrmdsplot([1,1,1,2,2,3,4,5], color=(0,0,0.8, 0.3), fig=fig, ax=ax, analysis_alpha=1)
#     nrmdsplot([3,3,3,10], color=(0.8,0,0, 0.3), fig=fig, ax=ax, analysis_alpha=1)
#     vlinesplot([3], label='line', fig=fig, ax=ax)
#     draw_ending(fig)
    
#     paint = Paint(plot_method=nrmdsplot)
#     paint.add([1,1,1,2,2,3,4,5], color=(0,0,0.8, 0.3), key=0)
#     paint.add([3,3,3,10], color=(0.8,0,0, 0.3), key=0)
#     paint.add([3], label='line', color=(0,0,0), ls='--', method=vlinesplot, key=0)
#     paint.draw()
    
    
#     paint = Paint(plot_method=nrmdsplot)
#     paint.add([1,1,1,2,2,3,4,5], color=(0,0,0.8, 0.3), key=0)
#     paint.add([3,3,3,10], color=(0.8,0,0, 0.3), key=0)
#     paint.add([3,3,3,10], color=(0.8,0,0, 0.3), key=1)
#     paint.add([3,3,3,10], color=(0.8,0,0, 0.3), key=2)
#     paint.add([3,3,3,10], color=(0.8,0,0, 0.3), key=3)
#     paint.add([3], label='line', color=(0,0,0), ls='--', method=vlinesplot, key=0)
#     paint.draw(figsize=(10,7), layout=(2,2), edge_plank=(0.1,0.1,0.9,0.9,0.2))
    
    
    
#     paint = Paint(plot_method=curveplot)
#     paint.add([1,2,3,4,5,6,7,8], [1,1,1,2,2,3,4,5], color=(0,0.3,0.8), key=0)
# #    paint.add([3,3,3,10], color=(0.8,0,0, 0.3), key=0)
#     paint.add([3], label='a', color=(0,0,0.8,0.3), ls='--', method=vlinesplot, key=0)
# #    paint.add([1,2,3,4,5,6,7,8], [5,1,1,4,1,1,3,2], color=(0.3,0.05,0.1, 0.3), key=0)
# #    paint.add([3,3,3,10], color=(0.8,0,0, 0.3), key=0)
#     paint.add([6.8], label='b', color=(0.4,0.05,0.01,0.3), ls='--', method=vlinesplot, key=0)
#     paint.draw()
    
    
    
    
    
#     vlinesplot([[3], [4,5]], labels=['line', 'line2'])
    
    
    
    
#     extentx = [0,1,2]
#     extenty = [0,1,2]
#     extent = (sorted(extentx)[0], sorted(extentx)[-1], sorted(extenty)[0], sorted(extenty)[1])
#     colors = [(0,0,0.8), (1,1,1), (0.8,0,0)]
    
#     df_hctable = pd.DataFrame([[1,-1, 12], [-10, 4, 40], [-13, 34, 0.2]], columns=extentx, index=extenty)
#     nodes = [df_hctable.max().max(), 0, df_hctable.min().min()]
# #    implot([[df_hctable]], extent=extent, make_cmap_colors_=colors, make_cmap_nodes_=nodes, 
# #           edge_plank=(0,0.15,0,0.85), file='test1.jpg') #加了wdge_plank就不行了??
#     implot([[df_hctable]], extent=extent, make_cmap_colors_=colors, make_cmap_nodes_=nodes)
    
#     paint = vs2.Paint(plot_method=vs2.implot)
#     for i in range(2):
#         paint.add([df_hctable], key=i)
#     paint.draw(plot_set_axis_range=0, extent=extent, make_cmap_colors_=colors, make_cmap_nodes_=nodes)
    
#     vs2.implot([[df_hctable]], extent=extent, make_cmap_colors_=colors, make_cmap_nodes_=nodes)
        
        
        
#     fig = vs2.initial_figure(figsize=(15,10))[0]
#     for i in range(2):
#         ax = fig.add_subplot(2,1,i+1)
#         vs2.implot([[df_hctable]], extent=extent, make_cmap_colors_=colors, 
#                make_cmap_nodes_=[df_hctable.min().min(), 0, df_hctable.max().max()], fig=fig, ax=ax)
#         1
    
# if(False):
#     xyplot([[1,2,3,4,5]], mode='s', plot_method=lambda *args, ax=None, **kwags: ax.axvline(*args, **kwags))
#     scatterplot([np.linspace(0,10,100)], [np.linspace(0,10,100)], figsize=(8, 5), vlines=[1])
#     xyplot([[1,2,3,4,5], [6, 7]], colors=['b','r'], labels=['try'], mode='s', 
#            plot_method=lambda *args, ax=None, **kwags: ax.axvline(*args, **kwags))
#     vlinesplot([1, 2], [3, 4], colors=['b', 'r'], label='try')
    
# if(False):
#     annotation = {(1,3.5):'hi'}
#     fig, ax = plt.subplots(figsize=(10,5))
#     vs2.vlinesplot([1, 2], [3, 4], colors=['b', 'r'], fig=fig, ax=ax, ls='--')
#     # vs2.curveplot([np.linspace(0, 4, 100)], [[3.5]*100, [3]*100], mode='x')
#     vs2.ax_set_annotation(ax, annotation={(1.5,0.5):'hi'})
#     print(1)
    
#%%
############################################################################################
class Coord():
    def __init__(self, plot_method, *args, start_key=None, showmask=[],
                 vlines={}, kwags_vlines={}, **kwags):
        start_key = 0 if(start_key==None) else start_key
        graphs = {}
        if(not isnotaxis(*args)):
            graphs[start_key]= dcp((kwags.get('method', plot_method), *args))
        self.graphs = dcp(graphs)
        self.plot_method = dcp(plot_method)
        self.kwags = dcp(kwags)
        if(len(np.array(vlines).shape)>0):
            dct_vlines = {v:vlines[v] for v in vlines}
            vlines = dct_vlines
        self.vlines = dcp(vlines) if(type(vlines)==dict) else {}
        self.kwags_vlines = dcp(kwags_vlines)
        self.showmask = showmask
    #TODO:Coord.add
    def add(self, *args, method=None, key=None, **kwags):
        if(isnotaxis(*args)):
            printer('Coord().add fail')
            return
        printer('Coord.add:\n%s'%str([v for v in self.graphs]), emphasis='-')
        key = dcp(np.max([v for v in self.graphs if not DFP.isnonnumber(v)] + [-1])+1 if(key==None) else key)
        printer('Coord().add key=%s'%str(key), showlevel=2)
        printer('Coord().add kwags:\n%s'%','.join(list(map(str, [v for v in kwags][:10]))), showlevel=4)
        self.graphs[key] = dcp((method, *args))
        self.kwags[key] = dcp(kwags)
        printer(str(self.graphs.keys()), showlevel=4)
    #TODO:Coord.get_all_values
    def get_all_values(self, axises=[]):
        values = []
        for axis in axises:
            for graph_index, (graph_key, (method, *args)) in enumerate(self.graphs.items()):
                do_axis = dcp(extract(axises, index=graph_index, key=graph_key, default=axis))
                do_args = dcp(extract(args, index=do_axis, default=args))
                try:
                    if(isiterable(do_args)):
                        values += get_all_values(do_args)
                    else:
                        values += get_all_values(do_args)
                except Exception as e:
                    print('[Coord.get_all_values]error:%s'%e)
                    print('[graph:%s][axis:%s]arg:%s'%(
                            graph_key, do_axis, str(do_args)[:200]))
                    {}['stop']
                printer('[Coord.get_all_values][graph:%s][axis:%s]arg:\n%s'%(
                            graph_key, str(do_axis)[:200], str(do_args)[:200]), showlevel=4)
        return values
    #TODO:Coord.draw
    def draw(self, ax, graph_kwags={}, vlines={}, kwags_vlines={}, fig=None, **kwags):
        try:
            ta = dt.now()
            do_kwags = dcp(self.kwags)
            
            do_kwags.update(kwags)
            common_kwags = dcp(do_kwags)
            for k in do_kwags:
                if(k in self.graphs):
                    common_kwags.pop(k)
            printer('[Coord.draw]common_kwags:\n%s'%(','.join([str(v) for i,v in enumerate(
                                common_kwags.keys()) if i<(showlog*5)])), showlevel=3)
            #do graphs
            index=0
            printer('graph keys:%s'%str(self.graphs.keys())[:200], showlevel=4)
            for (graph_key, (method, *args)) in self.graphs.items():
                printer('[index:%d][graph key:%s]Coord drawing....'%(
                        index, str(graph_key)), showlevel=(2+index//5))
                do_graph_kwags = dcp(common_kwags)
                do_graph_kwags.update({'color':extract(kwags.get('colors',mylist([])), index, graph_key, None)})
                do_graph_kwags.update(do_kwags.get(graph_key, {}))
                #TODO:[Coord.draw]only_head_label when mode s
                (do_graph_kwags.pop('label', None) if(graph_key[1]>0) else None) if(
                        'only_head_label' in do_graph_kwags) else None
                method_iter = dcp(self.plot_method if(method==None) else method)
                printer('[Coord.draw]do_graph_kwags:\n%s...'%(','.join([str(v) for i,v in enumerate(
                                do_graph_kwags.keys()) if i<(showlog*5)])), 
                                showlevel=3+index//5)
                printer('%s'%(str(args)[:200]), showlevel=6, emphasis='-')
                if(do_kwags.get(graph_key, {}).get('show_graph_key', do_kwags.get('show_graph_key', True))):
                    do_graph_kwags.update({'label':stamp_process(do_graph_kwags.get('label',''),[graph_key])})
                #這裡method是自定義的...plot一類，且arg理當只有一層iter系統，所以不需要回傳fig
                do_graph_kwags.pop('return_fig', None)
                method_iter(*args, ax=ax, fig=fig, is_ax_set_labels=0, return_fig=False, **do_graph_kwags)
                index += 1
            printer('[Coord draw] ax_set_labels...', showlevel=4)
            printer('[Coord.draw]do_kwags:\n%s'%(','.join([str(v) for i,v in enumerate(
                                do_kwags.keys()) if i<(showlog*5)])), showlevel=3)
            ax_set_labels(ax, **{k:do_kwags[k] for k in do_kwags if type(k)==str})
            ax_set_annotation(ax, **{k:do_kwags[k] for k in do_kwags if type(k)==str})
            ax_set_vlines(ax, **{k:do_kwags[k] for k in do_kwags if type(k)==str})
            printer('[Coord.draw]spending time:%.2f(s)'%((dt.now() - ta).total_seconds()), showlevel=3)
        except Exception as e:
            ax_errmsg(ax, e, function_key='Coord.draw')
    #TODO:Coord.export_as_data_frame
    def export_as_data_frame(self, graph_sel_axis=1, data_spec_length=None, data_index=None, data_index_stamps=None,
                             exclude_method=None, the_common_axis=0, the_common_axis_key='base', ax=None, **kwags):
        try:
            data_spec_length = None if(data_index!=None) else data_spec_length
            df = pd.DataFrame()
            for k,(m,*v) in self.graphs.items():
                if(exclude_method(k,m,v,**kwags) if(exclude_method) else False):
                    continue
                if(np.array(v[graph_sel_axis]).shape[0]==data_spec_length if(isinstance(data_spec_length, int)) else False):
                    continue
                stream = list(tuple(np.array(v[graph_sel_axis]).reshape(-1)))
                if(df.empty):
                    if(the_common_axis>=0):
                        the_common_stream = v[the_common_axis]
                        data_index = data_index if(DFP.isiterable(data_index)) else range(len(the_common_stream))
                        data_index = ['%s'%stamp_process('',list(map(str, data_index_stamps))+['#%s'%i],'','','','_') 
                                      for i in data_index] if(DFP.isiterable(data_index_stamps)) else data_index
                        df = pd.DataFrame(the_common_stream, index=data_index, columns=[the_common_axis_key])
                        df[k] = stream
                    else:
                        data_index = data_index if(DFP.isiterable(data_index)) else range(len(stream))
                        data_index = ['%s'%stamp_process('',list(map(str, data_index_stamps))+['#%s'%i],'','','','_') 
                                      for i in data_index] if(DFP.isiterable(data_index_stamps)) else data_index
                        df = pd.DataFrame(stream, index=data_index, columns=[k])
                else:
                    if(len(stream)!=len(df.index)):
                        printer("[%s] new stream with shape [%d] doesn't match the determined ones [%d]"%(
                            k, len(stream), len(df.index)), showlevel=1)
                        continue
                    df[k] = v[graph_sel_axis]
            return df
        except Exception as e:
            ax_errmsg(ax, e, function_key='Coord.export_as_data_frame')
        
#TODO:Paint
class Paint():
    def __init__(self, plot_method, *args, start_key=0, figsize=None, 
                 title='', file='', edge_plank=(0,0,1,1), 
                 showmask=[], showkeys={}, **kwags):
        if(type(plot_method)==str):
#            invert_op = globals()[plot_method]
            invert_op = exec(plot_method)
            if not callable(invert_op):
                {}['[%s] not callable'%plot_method]
            else:
                plot_method = dcp(invert_op)
        self.plot_method = plot_method
        coords = {}
        printer(str(args)[:200], showlevel=3)
        if(not isnotaxis(*args)):
            coords[start_key]= dcp((plot_method, *args))
        self.coords = coords
        self.figsize = figsize
        self.title = title
        self.file = file
        self.edge_plank = edge_plank
        for k in kwags:
            setattr(self, k, dcp(kwags[k])) if(k in Line2D_keys) else None
        self.showmask = showmask
        self.showkeys = showkeys
    #TODO:Paint.concatenate
    def concatenate(self, pnt2, copy=True, concatenate_in_coords=True, stamps_for_coords=None, stamps_for_graphs=None,
                    replace_coords=False, replace_graphs=False, **kwags):
        stamps_for_coords = stamps_for_coords if(isinstance(stamps_for_coords, list)) else []
        stamps_for_graphs = stamps_for_graphs if(isinstance(stamps_for_graphs, list)) else []
        pnt_end = dcp(self) if(copy) else self
        if(concatenate_in_coords):
            for k,v in pnt2.coords.items():
                if(k in self.coords):
                    pnt2_graphs = {stamp_process(
                        '',[k_]+stamps_for_graphs, '','','','_'):v_ for (k_,v_) in v.items()} if(
                            stamps_for_graphs) else v.graphs.copy()
                    pnt_end.coords[k].graphs.update(pnt2_graphs)
                else:
                    pnt_end.coords.update({k:v})
        else:
            if(stamps_for_coords):   pnt2_coords = {stamp_process(
                    '',[k_]+stamps_for_coords, '','','','_'):v_ for (k_,v_) in pnt2.coords.items()}
            pnt_end.coords.update(pnt2_coords)
        for k in kwags:
            setattr(pnt_end, k, dcp(kwags[k])) if(k in Line2D_keys) else None
        if(copy):   return pnt_end
    #TODO:Paint.add
    def add(self, *args, method=None, key=None, graph_key=None, **kwags):
        do_kwags = {k:getattr(self,k) for k in Line2D_keys if hasattr(self, k)}
        do_kwags.update(kwags)
        printer('[Paint.add]:%s'%str(args)[:200], showlevel=3)
        printer(str(args[1])[:200] if(len(args)>1) else '', showlevel=4)
        printer(str(method)[:200], showlevel=4)
        # if(isnotaxis(*args)):
        #     printer('Paint().add fail')
        #     return
        if(type(method)==str):
#            invert_op = globals()[plot_method]
            invert_op = exec(method)
            if not callable(invert_op):
                {}['[%s] not callable'%method]
            else:
                method = dcp(invert_op)
        key = dcp(np.max([v for v in self.coords if not DFP.isnonnumber(v)] + [-1]) if(key==None) else key)
        if(key in self.coords):
            crd = self.coords[key]
            printer('[Paint.add][coord key:%s][graph key:%s]'%(str(key), str(graph_key)), showlevel=3)
            crd.add(*args, method=method, key=graph_key, **kwags)
        else:
            method = self.plot_method if(method==None) else method
            sup_method = kwags.get('sup_method', method)
            kwags.update({'method':method})
            printer('[Paint.add][coord key:%s][graph key:%s]'%(str(key), str(graph_key)), showlevel=3)
            printer('[Paint.add]kwags:%s...'%(str(kwags)[:200]), showlevel=4)
            self.coords[key] = dcp(Coord(sup_method, *args, start_key=graph_key, **kwags))
    #TODO:Paint.get_all_values()
    def get_all_values(self, axis=None, axises=[], **kwags):
        values = []
        for crd_key, crd in self.coords.items():
            for graph_index, (graph_key, (method, *args)) in enumerate(crd.graphs.items()):
#                {}['']
                do_axis = dcp(extract(axises, index=graph_index, key=graph_key, default=axis))
                do_args = dcp(extract(args, index=do_axis, default=args))
                try:
                    if(isiterable(do_args)):
                        values += get_all_values(do_args)
                    else:
                        values += get_all_values(do_args)
                except Exception as e:
                    print('[Paint.get_all_values]error:%s'%e)
                    print('[coord:%s][graph:%s][axis:%s]arg:%s'%(
                            crd_key, graph_key, do_axis, str(do_args)[:200]))
                    {}['stop']
                printer('[Paint.get_all_values][coord:%s][graph:%s][axis:%s]arg:\n%s'%(
                            crd_key, graph_key, str(do_axis)[:200], str(do_args)[:200]), showlevel=4)
        return values
    #TODO:Paint.draw
    def draw(self, file='', title='', edge_plank=(0,0,1,1), layout=(), fig=None, 
             fig_axes_start_index=-1, is_end=True, is_adding_ax=True, **kwags):
        try:
            ax = None
            ta = dt.now()
            title = title if(title) else self.title
            file = file if(file) else self.file
            edge_plank = edge_plank if(edge_plank) else self.edge_plank
            printer(('[Paint.draw]mode:%s'%kwags['mode'] if('mode' in kwags) else ''), showlevel=3)
#            all_xvalues = self.get_all_values(1) if(
#                    mode in ['s', 's:x'] and mode!='s:y') else self.get_all_values(0)
            all_xvalues = self.get_all_values(0)
            printer('all_xvalues:\n%s'%str(all_xvalues)[:200], showlevel=5)
            all_yvalues = self.get_all_values(0) if(mode in ['s:y']) else self.get_all_values(1)
            printer('all_yvalues:\n%s'%str(all_yvalues)[:200], showlevel=5)
            figsize = self.figsize if(kwags['figsize']==() if('figsize' in kwags) else True) else kwags['figsize']
            is_default_fig = dcp(fig!=None)
            printer('is_default_fig:%s'%str(is_default_fig), showlevel=3)
            showmask = kwags.get('showmask', getattr(self, 'showmask', []))
            showkeys = kwags.get('showkeys', getattr(self ,'showkeys', []))
            coords = {k:self.coords[k] for (i,k) in enumerate(self.coords.keys()) if (i in showmask or k in showkeys)} if(
                showmask or showkeys) else self.coords.copy()
            fig, colsize, rowsize = initial_figure(figsize, edge_plank, 
                               n_axes=len(coords), layout=layout, 
                               all_xvalues=all_xvalues, all_yvalues=all_yvalues, 
                               fig=fig, kwags=kwags)
            do_kwags = {}
            for k in Line2D_keys:
                do_kwags.update({k:getattr(self, k)}) if(hasattr(self, k)) else None
            for index, (key, crd) in enumerate(coords.items()):
                # if((not index in showmask) if(showmask!=[]) else False):
                #     continue
                # if((not key in showkeys) if(showkeys!={}) else False):
                #     continue
                printer('[%d][%s]drawing....'%(index, str(key)), showlevel=3, emphasis='-')
                printer('fig_axes_start_index=%s'%str(fig_axes_start_index), showlevel=4)
                ax = fig.add_subplot(rowsize, colsize, index+1) if(
                        not is_default_fig) else (add_subplot(fig, fig_axes_start_index, rowsize, colsize) if(
                            len(fig.axes)==0 or is_adding_ax) else fig.axes[0])
                printer('[Paint.draw]number of fig.axes=%d'%len(fig.axes), showlevel=3)
                printer('[Paint.draw]fig.axes:\n%s'%(str(fig.axes)[:200]), showlevel=4)
                #TODO:定義ax的xy極限
                all_xvalues = crd.get_all_values([0])
                all_yvalues = crd.get_all_values([1])
                plot_set_axis_range(ax, all_xvalues, all_yvalues,
                                    **kwags)
                crd.draw(ax, fig=fig, s=kwags.get('s', None),
                         colors=kwags.get('colors',{}).get(key, []) if(len(coords)>1) else kwags.get('colors',[]))
                (kwags.update({'axtitle':kwags['axtitles'][index]}) if(index<len(kwags['axtitles'])) else None) if(
                        'axtitles' in kwags) else kwags.pop('axtitle', None)
                ax_set_axtitle(ax, **kwags)
#                ax.legend(**({k:kwags[k] for k in kwags if k in legend_keys}))
            printer('[Paint.draw]spending time:%.2f(s)'%((dt.now() - ta).total_seconds()), showlevel=2)
            return draw_ending(fig, is_end=is_end, file=file, title=title, prop=kwags.get('prop',{}),
                               fontname=kwags.get('suptitle_fontname',kwags.get('fontname', None)),
                               fontproperties=kwags.get('fontproperties', None))
        except Exception as e:
            ax_errmsg(ax, e, function_key='Paint.draw') if(ax!=None) else exception_process(e, logfile='', stamps=['Paint.draw'])
    #TODO:Paint.export_as_data_frame
    def export_as_data_frame(self, concatenate_axis=1, exclude_method_among_coords=None,
                             same_column_name_keep='first', same_row_name_keep='first',
                             graph_sel_axis=1, data_spec_length=None, data_index=None, data_index_stamps=None,
                             exclude_method_among_graphs=None, the_common_axis=0, the_common_axis_key='base', 
                             ax=None, **kwags):
        try:
            coll = DFP.collection()
            for k,c in self.coords.items():
                if(exclude_method_among_coords(k,c,**kwags) if(exclude_method_among_coords) else False):
                    continue
                the_common_axis_in_coord = getattr(c, 'the_common_axis', the_common_axis)
                the_common_axis_key_in_coord = getattr(c, 'the_common_axis_key', the_common_axis_key)
                df = c.export_as_data_frame(graph_sel_axis=graph_sel_axis, data_spec_length=data_spec_length, 
                                            data_index=data_index, data_index_stamps=data_index_stamps,
                                            exclude_method=exclude_method_among_graphs,
                                            the_common_axis=the_common_axis, the_common_axis_key=the_common_axis_key)
                coll.add(data=df, item_key=k)
            df_concatenate = coll.concatenate(axis=concatenate_axis)
            if(same_column_name_keep!=None):
                df_concatenate = pd.DataFrame(np.array(df_concatenate)[:,np.array(list(map(
                    lambda v:list(df_concatenate.columns).index(v), df_concatenate.columns[
                    np.logical_not(df_concatenate.columns.duplicated(keep=same_column_name_keep))]))).reshape(-1)],
                    columns = df_concatenate.columns[
                        np.logical_not(df_concatenate.columns.duplicated(keep=same_column_name_keep))], 
                    index=df_concatenate.index)
            return df_concatenate
        except Exception as e:
            ax_errmsg(ax, e, function_key='Paint.export_as_data_frame')
#%% 
##############################################################################
#TODO:plot_data_process
def plot_data_process(data, main_col=None, root=[], plot_method=curveplot, 
                      isreturn=0, annotation=None, **kwags):
    ret = None
    str_type = str(type(data))
    if(str_type.find('ndarray')>-1):
        pd_data = pd.DataFrame(data)
        return plot_data_process(
                pd_data, main_col=main_col, root=root, plot_method=plot_method, 
                isreturn=isreturn, **kwags)
    if(str_type.find('pandas')>-1):
        plot_data = pd.DataFrame(data).copy()
        if(np.array(root).shape[0]!=data.shape[0]):
            plot_data = pd.DataFrame(data).copy()
            plot_data = plot_data.reset_index().copy() if(main_col=='index') else plot_data.copy()
            root = plot_data.pop(main_col if(main_col!=None) else plot_data.columns[0])
            printer(str(list(root))[:200], showlevel=3)
        fig, ax = plt.subplots(figsize=kwags.get('figszie',(10,5)))
        kwags.update({'fig':fig}) if(not 'fig' in kwags) else None
        kwags.update({'ax':ax}) if(not 'ax' in kwags) else None
        ret = plot_method(
        [list(root)], 
        [list(plot_data[col]) for col in plot_data.columns], 
        mode='x', subkeys=[v for v in plot_data.columns], **kwags)
        vlines = kwags.get('vlines', None)
        vlinesplot(vlines, fig=fig, ax=ax, annotation=annotation, ls='--') if(
            not isinstance(vlines, type(None))) else None
        if(isreturn):
            return ret
        
def gridplot_data_process(data, main_col=None, root=None, plot_method=curveplot, isreturn=0, 
                          labels_among_coord=None, display_statistics=None, 
                          stamps_for_coords=None, stamps_for_graphs=None, **kwags):
    stamps_for_coords = stamps_for_coords if(isinstance(stamps_for_coords, list)) else []
    stamps_for_graphs = stamps_for_graphs if(isinstance(stamps_for_graphs, list)) else []
    display_statistics = display_statistics if(isinstance(display_statistics, dict)) else {}
    labels_among_coord = labels_among_coord if(isinstance(labels_among_coord, list)) else []
    root = root if(DFP.isiterable(root)) else []
    if(isinstance(data, np.ndarray)):
        pd_data = pd.DataFrame(data)
        return gridplot_data_process(
               pd_data, main_col=main_col, root=root, plot_method=plot_method, 
               isreturn=isreturn, **kwags)
    if(isinstance(data, pd.core.frame.DataFrame)):
        plot_data = pd.DataFrame(data).copy()
        if(np.array(root).shape[0]!=data.shape[0]):
            plot_data = pd.DataFrame(data).copy()
            plot_data = plot_data.reset_index().copy() if(main_col=='index') else plot_data.copy()
            main_col = main_col if(main_col!=None) else plot_data.columns[0]
            printer('main_col:%s'%str(main_col)[:10], showlevel=1)
            root = plot_data.pop(main_col)
            printer(str(list(root))[:200], showlevel=3)
        paint = Paint(plot_method=plot_method, **kwags)
        for i, col in enumerate(plot_data.columns):
            stamp_for_coords = stamp_process('',stamps_for_coords+[col],'','','','_')
            stamp_for_graphs = stamp_process('',stamps_for_graphs+[col],'','','','_')
            #TODO:[gridplot_data_process] paint.add
            label = ''
            label = extract(labels_among_coord, i, col) if(labels_among_coord) else (
                    kwags.get('label', ''))
            display_statistics_stamps = {}
            for k,f in display_statistics.items():
                try:
                    display_statistics_stamps.update({k:DFP.parse(f(plot_data[col]))})
                except Exception as e:
                    exception_process(e, logfile='', stamps=[gridplot_data_process.__name__, col])
            label = label + stamp_process('\n',display_statistics_stamps,location=-1) 
            kwags.update(dcp({'label':dcp(label)}))
            label = kwags.pop('label', None)
            paint.add(list(root), list(plot_data[col]), 
                       method=plot_method, key=stamp_for_coords, graph_key=stamp_for_graphs, label=dcp(label), **kwags)
        paint_kwags = dcp(kwags)
        paint_kwags.update({k:locals()[k] for k in paint_keys if k in locals()})
        if(isreturn):
            paint.kwags = paint_kwags
            return paint
        paint.draw(**paint_kwags)
        
#TODO:export_data_chart
def export_data_chart(data, exp_fd, stamps=None, purpose='', **kwags):
    stamps = [] if(stamps==None) else stamps
    stamps = [purpose] + stamps
    title = stamp_process('', stamps, '','','',' ')
    file = os.path.join(exp_fd, '%s.jpg'%stamp_process('',stamps,'','','','_'))
    CreateFile(file, lambda f:vs.plot_data_process_with_grid(
        data, file=f, title=title, **kwags))
#%%
##############################################################################
#TODO:test2
# if(False):
    # data = df_runcard.copy()
    # plot_data_process(data[['mt', 'speed', 'gap1L']].astype(float), figsize=(5, 3))
    # print(1)
    # for i in range(2):
    #     gridplot_data_process(data[['mt', 'speed', 'gap1L']].astype(float), file='test.jpg')
    # print(1)
    
    
    # columns = ['p50','clockwise','ti_mel','spin_spdtm','vac_bef_rod','gassing_spdtm',
    #            'ti_over_heat','centrifugal_on','spin_max',
    #            'spin_gassing_ovlptm','rod_to_gassing_tm','NGR']
    # normdistb_fit(D_[columns], mask=[], mask_col=None, figsize=(15,10), histtype = 'bar',
    #               edge_plank=(0.1,-0.2,0.9,0.94,0.4,0.3), labels=columns, 
    #               axtitles=list(map(lambda i:'[%d]'%i, range(len(columns)))),
    #               title='earlier', file='later.jpg')
    
    # ftr_name = 'spin_max'
    # normdistb_fit(A_[ftr_name], mask=A_['NG'], mask_col=None, histtype = 'bar',
    #               edge_plank=(0.1,0.1,0.9,0.9,0.4), title=ftr_name)

#%%
def drawparams(grid_search, file='', xbd=(), ybd=(0.6, 1.1), figsize=None, edge_plank=edge_plank,
               markersize=5, xlb_rotation=90, showmodes=False, **kwags):
    results = pd.DataFrame(grid_search.cv_results_)
    best = np.argmax(results.mean_test_score.values)
    all_xvalues = get_all_values(results.index)
    yheader = results.columns[(results.columns.str.contains('score'))&(
                               np.logical_not(results.columns.str.contains('std')))&(
                                       np.logical_not(results.columns.str.contains('rank')))&(
                                       np.logical_not(results.columns.str.contains('time')))]
    all_yvalues = get_all_values(np.array(results[yheader]).reshape(1,-1))
    fig, colsize, rowsize = initial_figure(figsize, edge_plank, 
                           all_xvalues=all_xvalues, 
                           all_yvalues=all_yvalues, plot_set_figsize_on=0, **kwags)
    ax = fig.add_subplot(rowsize, colsize, 1)
#    if(figsize==None):
#        pfy = len(grid_search.best_params_)*8
#
#        pfx = (len(grid_search.cv_results_['params'])*len(grid_search.cv_results_['params'][0])*
#               len(str(grid_search.cv_results_['params'][0])))/20
#        figsize = (pfx, pfy)
#    xbd = (-1, len(results)) if xbd==() else xbd
    plot_set_axis_range(ax, all_xvalues, all_yvalues, xvalue_lim=xbd, ylimit=ybd, **kwags)
    #marker_best = 0
    for i, (_, row) in enumerate(results.iterrows()):
        n_split = len(row.index[row.index.str.contains('split')])
        scores = row[['split%d_test_score' % j for j in range(n_split)]]
        marker_cv, = ax.plot([i] * n_split, scores, '^', c='gray', markersize=markersize,alpha=.5)
        marker_mean, = ax.plot(i, row.mean_test_score, 'v', c='none', alpha=1,
                                markersize=2*markersize, markeredgecolor='k')
        if i == best:
            marker_best, = ax.plot(i, row.mean_test_score, 'o', c='red',
                                    fillstyle="none", alpha=1, markersize=4*markersize,
                                    markeredgewidth=1)
    grid_par = [drop(d,'kernel') for d in grid_search.cv_results_['params']] if(
            not showmodes) else grid_search.cv_results_['params']
    fontsize = markersize*1.5
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize/1.5)
    plt.xticks(range(len(results)), [str(x).strip("{}").replace("'", "") for x
                                     in grid_par],rotation=xlb_rotation)
    
    ax.set_ylabel("Validation score")
    ax.set_xlabel("Parameter settings")
    ax.legend([marker_cv, marker_mean, marker_best],
               ["cv score", "mean score", "best parameter setting"],
               loc=(1.05, .4),
               **({k:kwags[k] for k in kwags if k in legend_keys}))
    #plt.legend([marker_cv, marker_mean],
    #           ["cv accuracy", "mean accuracy"],
    #           loc=(1.05, .4))
    if(file!=''):
        fig.savefig(file)
    fig.show()
    plt.ioff()
#TODO:normdistb_fit bar barstacked
def normdistb_fit(data, mask=[], mask_col=0, title='', xlb='', ylb='probability density', 
                  vlines=[], vlines_labels=[], file='', num_bins = 100, 
                  class_colors=[(0,0,0.8, 0.3), (0.8,0,0,0.3)], color=None, vline_color=None,
                  histtype = 'barstacked', density = True, figsize=None, layout=(),
                  histalpha = 0.5, stacked=True, xmax=None, xmin=None, axindex=0,
                  default_value=0, is_end=True, paint=None, fig=None, ax=None, 
                  edge_plank=edge_plank, axtitles=[], **kwags):
    np_data = np.array(data)
    shape = np_data.shape
    if(len(shape)<1):
        return paint
    if(len(shape)==1):
        mask = np.array([True]*shape[0]) if(np.array(mask).shape[0]==0) else mask
        class_colors = mylist([v for v in cm_rainbar(_n=len(set(mask)))] if(class_colors=='rainbow') else class_colors)
        if(np.array(mask).shape[0]!=shape[0]):
            return
        if(np.array(list(map(DFP.isnonnumber, data))).any()):
            printer('偵測到空值，進行%s:\n%s...'%(('排除' if(default_value==None) else '補值[%s]'%default_value), 
                      str(','.join(list(tuple(data.columns[
                      np.array(list(map(DFP.isnonnumber, data))).any()])))[:20])), showlevel=1)
            np_data = np.where(np.array(list(map(DFP.isnonnumber, data))), 
                            np.array([default_value]*np_data.shape[0]), np_data) if(default_value!=None) else np_data[
                                    np.logical_not(np.array(list(map(DFP.isnonnumber, data))))]
        new_paint = Paint(plot_method=nrmdsplot) if(paint==None) else paint
        np_mask = np.array(mask)
        np_mask_ = dcp(np_mask if(default_value!=None) else np_mask[
                                    np.logical_not(np.array(list(map(DFP.isnonnumber, data))))])
        classes = sorted(list(set(tuple(mask))))
        for i, classid in enumerate(classes):
            idx = np.array(np_mask_==classid)
            x = np_data[idx]
            new_paint.add(list(tuple(x)), color=class_colors.get(i),y=axindex)
        new_paint.add(vlines, color=vline_color, ls='--', label=vlines_labels, method=vlinesplot, key=axindex,)
        return new_paint.draw(file=file, title=title, histtype=histtype, histalpha=histalpha, layout=layout,
                              density=density, stacked=stacked, figsize=figsize, edge_plank=edge_plank) if(
                is_end) else (new_paint if(paint==None) else None)
    if(len(shape)>1):
        if(not data.applymap(DFP.isnonnumber).any().any()):
            columns = list(data.columns)
        else:
            printer('偵測到空值，進行排除:\n%s...'%str(','.join(list(tuple(data.columns[
                    data.applymap(DFP.isnonnumber).any()])))[:20]))
            columns = data.columns[np.logical_not(
                    data.applymap(DFP.isnonnumber).any())]
        mask_coln = (columns.pop(mask_col) if(mask_col!=None) else None)
        mask = (list(map(bool, data[mask_coln])) if(mask_coln!=None) else []) if(
                np.array(mask).shape[0]==0) else mask
        ftr_num = len(columns)
        printer('沒有features? 不需要分類mask的話，要設 mask_col=None') if(ftr_num==0) else None
        class_colors = [v for v in cm_rainbar(_n=ftr_num)] if(class_colors=='rainbow') else class_colors
        new_paint = Paint(plot_method=nrmdsplot) if(paint==None) else paint
        for i in range(ftr_num):
            xmax, xmin = np.max(data[columns[i]]), np.min(data[columns[i]])
            normdistb_fit(data[columns[i]], mask=mask, mask_col=None, title=title, xlb=xlb, ylb=ylb, 
                  vlines=vlines, vlines_labels=vlines_labels, file=file, num_bins = num_bins, 
                  class_colors=class_colors, vline_color=vline_color, 
                  histtype = histtype, density = density, figsize=figsize, 
                  histalpha = histalpha, xmax=xmax, xmin=xmin,axindex=i,
                  default_value=default_value, is_end=False, paint=new_paint, **kwags)
        axtitles = columns if(axtitles=='columns') else axtitles
        return new_paint.draw(file=file, title=title, histtype=histtype, histalpha=histalpha, layout=layout,
                              density=density, stacked=stacked, figsize=figsize, edge_plank=edge_plank,
                              axtitles=axtitles) if(
                is_end) else (new_paint if(paint==None) else None)
            
#        paint = Paint(plot_method=nrmdsplot)
#        paint.add([1,1,1,2,2,3,4,5], color=(0,0,0.8, 0.3), key=0)
#        paint.add([3,3,3,10], color=(0.8,0,0, 0.3), key=0)
#        paint.add([3,3,3,10], color=(0.8,0,0, 0.3), key=1)
#        paint.add([3,3,3,10], color=(0.8,0,0, 0.3), key=2)
#        paint.add([3,3,3,10], color=(0.8,0,0, 0.3), key=3)
#        paint.add([3], label='line', color=(0,0,0), ls='--', method=vlinesplot, key=0)
#        paint.draw(figsize=(10,7), layout=(2,2), edge_plank=(0.1,0.1,0.9,0.9,0.2))
#TODO:normdistb_fit_old
def normdistb_fit_old(data, classes, title='', xlb='', ylb='probability density', 
                  vlines=[], file='', isevaluate = False, num_bins = 100, 
                  histtype = 'barstacked', density = True, 
                  histalpha = 0.5, xmax=None, xmin=None):
    evaluate = {classid:{} for classid in classes}
    data = np.array(data)
    fig, ax = plt.subplots()
    axtw = ax.twinx()
    xmax = data.max() if(xmax==None) else xmax
    xmin = data.min() if(xmin==None) else xmin
    x_ = np.linspace(xmin, xmax, num_bins)
    for classid in classes:
        idx = np.array(classes[classid]['mask'])
        x = data[idx]
        diversity = len(list(set(x)))
        if(diversity<2):
            printer('資料種類%d太少，不常態擬合!!'%diversity)
        else:
            (mu, sigma) = scipy.stats.norm.fit(x)
        evaluate[classid]['mean'] = float(mu) if(diversity>=2) else (list(set(x))[0] if(diversity>0) else 0)
        evaluate[classid]['std'] = float(sigma) if(diversity>=2) else -1
        n, bins, patches = ax.hist(x, num_bins, 
                                   density=density, 
                                   stacked=True, 
                                   histtype='bar' if(histtype==None) else histtype,
                                   color=tuple(list(classes[classid]['color'])[:3])+(histalpha,))
        if(diversity>=2):
            y = scipy.stats.norm.pdf(x_,mu,sigma)
            axtw.plot(x_, y, '--', color=classes[classid]['color'],
                      label='%s:$\mu$=%.2f, $\sigma$=%.2f'%(classid, mu,sigma))
        axtw.set_xlabel(xlb)
        axtw.set_ylabel(ylb)
    if(len(classes)==2):
        if(diversity>=2):
            compareid = list(classes.keys())
            evaluate['entropy']=(calculate_gaussian_kl_divergence(evaluate[compareid[0]]['mean'],
                                                                 evaluate[compareid[1]]['mean'],

                                                                 evaluate[compareid[0]]['std'],
                                                                 evaluate[compareid[1]]['std']))
        else:
            evaluate['entropy'] = -1
    #TODO:分布中的關鍵垂直線
    for vline in vlines:
        axtw.axvline(vline)
    axtw.legend(**({k:kwags[k] for k in kwags if k in legend_keys}))
    axtw.set_title('Histogram of %s\n%s'%(
        title, ('KL divergence:%.2f'%(evaluate['entropy']) if(isevaluate and diversity>=2) else '')))
    fig.tight_layout()
    if(file!=''):
        fig.savefig(file)
    plt.show()
    if(isevaluate):
        return evaluate

def get_blackpoint_mask(img, channel=3, pattern=None):
    channel = channel if(not DFP.isiterable(pattern)) else np.array(pattern).shape[0]
    if(np.array(img).shape[1]//channel!=img.shape[1]/channel):
        return None
    M = DFP.diagonal_semibelt_matrix(channel,int(np.array(img).shape[1]//channel),-1,
                                     pattern=pattern)
    return (np.matmul(img, M)==0).T.any()
    
def piclog_process(img, stamps=None, exp_fd='IO_pics', a_index=None, 
                   layout=None, frame_height=160, frame_width=160, frame_thickness = 1, 
                   frame_color=(40,5,0), channel_counts=3, **kwags):
    stamps = stamps if(isinstance(stamps, list)) else []
    exp_fd = exp_fd if(isinstance(exp_fd, str)) else 'IO_tables'
    layout = layout[:2] if(DFP.isiterable(layout)) else (5,5)
    total_size = (layout[0]*frame_width, layout[1]*frame_height)
    file = os.path.join(exp_fd,'%s.pkl'%stamp_process('',[exp_fd]+stamps,'','','','_',for_file=1))
    if(os.path.exists(file)):
        try:
            base_img = cv2.imread(file)
        except Exception as e:
            exception_process(e, logfile=kwags.get('logfile', os.path.join('log','log_%t.txt')))
            return False
    if(np.array(base_img).shape[0]<total_size[0]):
        base_img = np.concatenate([base_img, np.zeros(
            (total_size[0]-np.array(base_img).shape[0], np.array(base_img).shape[1], channel_counts))])
    if(np.array(base_img).shape[1]<total_size[1]):
        base_img = np.concatenate([base_img, np.zeros(
            (np.array(base_img).shape[0], total_size[1]-np.array(base_img).shape[1], channel_counts))], axis=1)
    
    #設計空畫框
    for i in range(0, layout[0]*frame_width, frame_width):
        for j in range(0, layout[1]*frame_height, frame_height):
            cv2.rectangle(base_img, (i, j), (i+frame_width, j+frame_height), frame_color, frame_thickness)
            
    #TODO:cv2嵌入...
    cv2imshow(img, window_title='illustration')
    pic_history = None
    if(os.path.exists(file)):
        try:
            pic_history = cv2.imread(file)
        except Exception as e:
            exception_process(e, logfile=kwags.get('logfile', os.path.join('log','log_%t.txt')))
            return False
    if(not isiterable(img)):
        print('illegal data type:%s'%type_string(img))
        return False
    np_img = np.array(img)

def draw_fft(ydata, xdata=None, ret=None, n_fft_anchor=100, 
             fit_line_color=(1,0,0,0.5), data_color=(0,0,1,0.5),s_data=None,
             fig=None, ax=None, is_end=True, **kwags):
    ret = ret if(isinstance(ret, dict)) else {}
    if(not DFP.fft_scenario(ydata, xdata=xdata, ret=ret)):
        return False if(is_end) else (None, None)
    n_fft_anchor = max(n_fft_anchor, np.array(ydata).shape[0])
    x_uni = ret['x_uni'] if(not isinstance(ret['x_uni'], type(None))) else np.arange(0, np.array(ydata).shape[0])
    
    fft_function = ret['fft_function']
    x_fft = np.linspace(x_uni[0],x_uni[-1],n_fft_anchor)
    y_fft = fft_function(x_fft)
    if(fig==None):
        fig,r,cc=initial_figure(figsize=kwags.get('figsize',None), edge_plank=kwags.get('edge_plank', edge_plank))
    ax = add_subplot(fig) if(ax==None) else ax
    scatterplot(x_fft,y_fft,fig=fig,ax=ax,color=fit_line_color)
    # xdata = xdata if(not isinstance(xdata, type(None))) else np.arange(0, np.array(ydata).shape[0])
    # scatterplot(xdata,ydata,fig=fig,ax=ax,color=data_color,s=s_data)
    if(is_end):
        file = kwags.get('file',None)
        if(isinstance_not_empty(file, str)):
            CreateFile(file, lambda f:fig.savefig(f))
        fig.show()
    return True

def draw_ifft(ydata, xdata=None, ret=None, n_ifft_anchor=100, 
             fit_line_color=(1,0,0,0.5), data_color=(0,0,1,0.5),s_data=None,
             fig=None, ax=None, is_end=True, **kwags):
    ret = ret if(isinstance(ret, dict)) else {}
    if(not DFP.fft_scenario(ydata, xdata=xdata, ret=ret)):
        return False if(is_end) else (None, None)
    n_ifft_anchor = max(n_ifft_anchor, np.array(ydata).shape[0])
    x_uni = ret['x_uni'] if(not isinstance(ret['x_uni'], type(None))) else np.arange(0, np.array(ydata).shape[0])
    
    ifft_function = ret['ifft_function']
    x_ifft = np.linspace(x_uni[0],x_uni[-1],n_ifft_anchor)
    y_ifft = ifft_function(x_ifft)
    # print(str(y_ifft))
    # sys.exit(1)
    if(fig==None):
        fig,r,cc=initial_figure(figsize=kwags.get('figsize',None), edge_plank=kwags.get('edge_plank', edge_plank))
    ax = add_subplot(fig) if(ax==None) else ax
    curveplot(x_ifft,y_ifft,fig=fig,ax=ax,color=fit_line_color)
    xdata = xdata if(not isinstance(xdata, type(None))) else np.arange(0, np.array(ydata).shape[0])
    scatterplot(xdata,ydata,fig=fig,ax=ax,color=data_color,s=s_data)
    if(is_end):
        file = kwags.get('file',None)
        if(isinstance_not_empty(file, str)):
            CreateFile(file, lambda f:fig.savefig(f))
        fig.show()
    return True

def plot_frequency_spectrum(data, stamps=None, ret=None, ret_keys=None, is_end=False, spot_s=None, **kwags):
    exp_fd = execute('exp_fd', kwags, default='test', not_found_alarm=False)
    ret = ret if(isinstance(ret, dict)) else {}
    ret_keys = ret_keys if(isinstance(ret_keys, list)) else []
    stamps = (stamps if(isinstance(stamps, list)) else [])
    stamp = stamp_process('', stamps, '','','',' ')
    key = execute('key', kwags, default=stamp, not_found_alarm=False)
    graph_key = execute('graph_key', kwags, default='', not_found_alarm=False)
    color = execute('color', kwags, not_found_alarm=False)
    title = kwags.pop('title',None)
    title = stamp if(not isinstance_not_empty(title, str)) else title
    file = kwags.pop('file',None)
    file = (os.path.join(exp_fd, '%s.png'%title) if(file=='') else file) if(isinstance(file, str)) else None
    X_freq= DFP.fftfreq(len(data))
    X_FFT= DFP.fft(data)
    spectrum = X_FFT
    if(is_end):
        CreateFile(file, lambda f:scatterplot([list(tuple(X_freq))], [list(tuple(np.abs(X_FFT)))], title=title, file=f, **kwags))
    else:
        pnt = execute('pnt', ret, ret_keys, kwags, default=None, not_found_alarm=False)
        pnt = pnt if(isinstance(pnt, Paint)) else Paint(plot_method=scatterplot)
        pnt.add(list(tuple(X_freq)), list(tuple(np.abs(X_FFT))), key=key, graph_key=graph_key, color=color, s=spot_s)
        ret.update({'pnt':pnt})
    for k in ret_keys:
        ret.update({k:execute(k,locals(),kwags,globals())})
        
def plot_frequency_spectrum_uni(ydata, xdata=None, stamps=None, ret=None, ret_keys=None, is_end=False, spot_s=None, 
                      n_fft_anchor=100, **kwags):
    exp_fd = execute('exp_fd', kwags, globals())
    ret = ret if(isinstance(ret, dict)) else {}
    ret_keys = ret_keys if(isinstance(ret_keys, list)) else []
    stamps = (stamps if(isinstance(stamps, list)) else [])
    stamp = stamp_process('', stamps, '','','',' ')
    key = execute('key', kwags, default=stamp)
    graph_key = execute('graph_key', kwags, default='')
    color = execute('color', kwags)
    title = kwags.pop('title',None)
    title = stamp if(not isinstance_not_empty(title, str)) else title
    file = kwags.pop('file',None)
    file = (os.path.join(exp_fd, '%s.png'%title) if(file=='') else file) if(isinstance(file, str)) else None
    if(not DFP.fft_scenario(ydata, xdata=xdata, ret=ret)):
        return False if(is_end) else (None, None)
    n_fft_anchor = max(n_fft_anchor, np.array(ydata).shape[0])
    x_uni = ret['x_uni'] if(not isinstance(ret['x_uni'], type(None))) else np.arange(0, np.array(ydata).shape[0])
    
    fft_function = ret['fft_function']
    x_fft = np.linspace(x_uni[0],x_uni[-1],n_fft_anchor)
    y_fft = fft_function(x_fft)
    b_freq= DFP.fftfreq(np.array(ydata).shape[0])
    if(is_end):
        CreateFile(file, lambda f:scatterplot([list(tuple(b_freq))], [list(tuple(np.abs(y_fft)))], title=title, file=f, **kwags))
    else:
        pnt = execute('pnt', ret, ret_keys, kwags, Paint(plot_method=scatterplot))
        pnt = pnt if(isinstance(pnt, Paint)) else Paint(plot_method=scatterplot)
        pnt.add(list(tuple(b_freq)), list(tuple(np.abs(y_fft))), key=key, graph_key=graph_key, color=color, s=spot_s)
        ret.update({'pnt':pnt})
    for k in ret_keys:
        ret.update({k:execute(k,locals(),kwags,globals())})

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
    if(image_width*image_height==0):
        return None
    new_width = int(image_width * scale_factor[0])
    new_height = int(image_height * scale_factor[1])
    # 縮小圖像
    try:
        img = img.astype(np.uint8)
    except:
        return None
    return cv2.resize(img, (new_width, new_height))

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
        addlog('img convert error!!! type:%s(dtype:%s...), value:\n%s'%(type(img), type(mylist(img).get(0,None)), str(img)[:200]))
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
    img_msked = dcp(cv2bitwise_and(img,img_msk)) if(not isinstance(img_msk, type(None))) else img
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
    img_msked = dcp(cv2bitwise_and(img,img_msk)) if(not isinstance(img_msk, type(None))) else img
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
                    edging_method='thredging', **kwags):
    if(not m_cv2_import_succeed):
        return False if(is_end) else None
    stamps = stamps if(isinstance(stamps, list)) else []
    ret = ret if(isinstance(ret, dict)) else {}
    img_msked = dcp(cv2bitwise_and(img,maskimg)) if(not isinstance(maskimg, type(None))) else img
    if(edging_method=='canny'):
        edged = cv2imedging(img_msked, threshold_lbd=threshold_lbd, threshold_ubd=threshold_ubd, kernel_size=kernel_size, 
                            dilate_iterations=dilate_iterations, erode_iterations=erode_iterations,**kwags)
    else:
        edged = cv2imthredging(img_msked, threshold=threshold_bd, 
                               dilate_iterations=dilate_iterations, erode_iterations=erode_iterations,**kwags)
    
    kwags['click'].update({cv2.findContours.__name__: dt.now()}) if('click' in kwags) else None
    (contours, _) = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    kwags['click'].update({cv2.findContours.__name__: (dt.now() - kwags['click'][cv2.findContours.__name__]).total_seconds()}) if(
        'click' in kwags) else None
    print('no contours be found!!') if(len(contours)==0) else None
    cv2findContours_drawing(img_msked, contours, bgr_sample=bgr_sample, r=r, d=d, stamps=stamps, ret=ret, file=file, 
                            bgr_need_flatten = bgr_need_flatten, color_transplant_circle=color_transplant_circle, 
                            only_convex_pass=only_convex_pass, dont_log=dont_log, dont_showimg=dont_showimg, 
                            do_fit_ellipse=do_fit_ellipse, color_ellipse=color_ellipse, img_raw=img,
                            th_ellipse=th_ellipse, draw_box_bound=draw_box_bound, **kwags)
    if(not is_end):
        return contours
    return True

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
        img_msk = cv2imread(img_msk_file) if(img_msk_file) else None
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



#%%
class myVideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    self.stop_flag = False
    self.t = threading.Thread(target=self._reader)
    self.t.daemon = True
    self.t.start()

    #ctrl+c啟動正常結束程序
  def handle_signal(self,signum, frame):
    #self.adgProcLogger.debug('%s(%d) handling signal %r' ,self.__netLabel,self.pid, signum) #type(self).__name__
    self.stop_flag.set()
        
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
        """
        
        
        Parameters
        ----------
        img : TYPE
        完全沒處理的原圖
        tm : TYPE, optional
        DESCRIPTION. The default is None.
        stamp : TYPE, optional
        DESCRIPTION. The default is ''.
        
        Returns
        -------
        None.
              
        """
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
        if('ret' in params):    img = params['ret']['img']
        if(isinstance_not_empty(params_list.get('memo_stamps'), list)):
            memo_stamps = params_list['memo_stamps']
            memo_stamp = stamp_process('',memo_stamps,'','','',' ')
            cv2.putText(img, memo_stamp, (10, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, 
                        (int(255), int(255), int(255)), 5, cv2.LINE_AA)
        # self.method(img, **params_package)
        # if('ret' in params_package):    img = params_package['ret']['img']
        # if(isinstance_not_empty(params_package.get('memo_stamps'), list)):
        #     memo_stamps = params_package['memo_stamps']
        #     memo_stamp = stamp_process('',memo_stamps,'','','',' ')
        #     cv2.putText(img, memo_stamp, (10, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, 
        #                 (int(255), int(255), int(255)), 5, cv2.LINE_AA)
        return img
    
    def saveSubdirList(self, **kwags):
        stamps = [self.exp_fd]
        if('tm' in kwags):
            tm = kwags['tm']
            stamps.append(tm.strftime('%Y%m%d'))
        return stamps
    
    def nameRawImg(self, *stamps, **kwags): #tm.strftime('%Y%m%d')
        fn = stamp_process('',list(stamps)+['RAW'],'','','','_',for_file=True)
        return os.path.join(*(self.saveSubdirList(**kwags)), '%s.jpg'%fn)
    
    def nameAuxImg(self, *stamps, **kwags): #tm.strftime('%Y%m%d')
        fn = stamp_process('',list(stamps),'','','','_',for_file=True)
        return os.path.join(*(self.saveSubdirList(**kwags)), '%s.jpg'%fn)

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
                judgement = self.params_table.get(stamp, {}).get('judgement', False)
                img, judgement_stg = None, ''
                if(self.is_auximg):
                    img = self.aux_img_process()
                    if(judgement):
                        judgement_stg = 'NG'
                        if(vm!=None):
                            #有vm交給vm打印
                            vm.add_img(img, tm=tm, stamp=stamp, img_raw=img_raw)
                            self.img_buffer = self.img_buffer[1:]
                            self.params_table.pop(stamp, None)
                            index += 1
                            continue
                self.img_live = dcp(img if(isinstance(img, np.ndarray)) else img_raw)
                self.img_buffer = self.img_buffer[1:]
                self.params_table.pop(stamp, None)
                index += 1
                if(index%saveimg_fr_freq!=1 if(saveimg_fr_freq>0) else True):
                    continue
                #原圖打印
                ta = dt.now() if(not isinstance(self.click_table, type(None))) else None
                CreateFile(self.nameRawImg(stamp, tm=tm), lambda f:cv2.imwrite(filename=f, img=img_raw))
                click = self.click_table.get(stamp, {})
                click.update({'contours_saveimg':(dt.now() - ta).total_seconds()}) if(
                    not isinstance(self.click_table, type(None))) else None
                if(not isinstance(img, np.ndarray)):
                    continue
                if(img.shape==img_raw.shape):
                    if(not (img!=img_raw).any()):
                        continue
                #處理過後的圖img，不管N不NG，跟img_raw不一樣就要打印
                ta = dt.now() if(not isinstance(self.click_table, type(None))) else None
                CreateFile(self.nameAuxImg(stamp, tm=tm), lambda f:cv2.imwrite(filename=f, img=img))
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
    
    def saveSubdirList(self, **kwags):
        stamps = [self.exp_fd]
        if('tm' in kwags):
            tm = kwags['tm']
            stamps.append(tm.strftime('%Y%m%d'))
        return stamps
    
    def nameRawImg(self, *stamps, **kwags): #tm.strftime('%Y%m%d')
        fn = stamp_process('',list(stamps)+['NG','RAW'],'','','','_',for_file=True)
        return os.path.join(*(self.saveSubdirList(**kwags)), '%s.jpg'%fn)
    
    def nameAuxImg(self, *stamps, **kwags): #tm.strftime('%Y%m%d')
        fn = stamp_process('',list(stamps)+['NG'],'','','','_',for_file=True)
        return os.path.join(*(self.saveSubdirList(**kwags)), '%s.jpg'%fn)
    
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
        # 輪巡並存圖
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
                        file = self.nameAuxImg(stamp, tm=tm)
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
                        # 如果需要描繪NG的原圖...
                        stamp = img_raw_buffer[i][2]
                        fn = stamp_process('',[stamp, 'NG','RAW'],'','','','_',for_file=True)
                        tm = pd_img_buffer['tm'].iloc[i]
                        file = self.nameRawImg(stamp, tm=tm)
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