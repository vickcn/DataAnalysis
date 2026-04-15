# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 16:22:44 2021

@author: ian.ko
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#from matplotlib import cm
import scipy
import pandas as pd
import matplotlib.image as mpimg # mpimg 用於讀取圖片
from matplotlib import font_manager as fm
from copy import copy as c
from copy import deepcopy as dcp
import seaborn as sns
import random as rdm
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d
from package import dataframeprocedure as DFP
LOGger = DFP.LOGger
exception_process = LOGger.exception_process
mystr = LOGger.mystr
import platform
import os
showlog = 0
empty_print = 0
edge_plank=(0,0,1,1)

figsize=(15,8)
file=''
title=''
titlefontsize = 20
mode='d'
ctr_tol=0, 
xtkrot=0, 
is_key_label=True
common_method_inputs = {
        'figsize':figsize, 'file':file, 'title':title,
        'titlefontsize':titlefontsize, 'mode':mode, 
        'ctr_tol':ctr_tol, 'xtkrot':xtkrot}

label_callback_key='labels'

fontfile = mystr('/usr/share/fonts/truetype/test/msjh.ttc').path_sep_correcting()
fm.fontManager.addfont(fontfile) if(os.path.exists(fontfile)) else None
def MJHfontprop():
    if(platform.system().lower()=='windows'):
        prop = fm.FontProperties(family='Microsoft JhengHei')
    else:
        fontfile = mystr("/usr/share/fonts/truetype/test/msjh.ttc").path_sep_correcting()
        if(not os.path.exists(fontfile)):
            prop = fm.FontProperties(family='DejaVu Sans')
        prop = fm.FontProperties(fname=fontfile)
    return prop

def printer(*logs, level=1, **kwags):
    if(showlog>=level):
        logs = logs if(empty_print) else tuple([log for log in logs if log!=''])
        print(*logs)

def clean_common_params(params, clean_keys=['file']):
            for key in clean_keys:
                params.pop('file', None)
            return params

if(False):
    # 建立 3D 圖形
    fig = plt.figure(figsize=(20,10))
    ax = fig.gca(projection='3d')
    ax.set_zlim3d(0, 1)
    # 產生測試資料
    X = np.arange(0, 1.4, 0.25)
    Y = np.arange(0, 1.4, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = 2*(X*Y)/(X+Y)
#    X, Y, Z = axes3d.get_test_data(0.05)
    
    # 繪製 3D 曲面圖形
#    ax.plot_surface(X, Y, Z, cmap='seismic')
#    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
#    ax.plot_wireframe(X, Y, Z, rcount=10, ccount=10)
    
    x1 = np.array([0.98])
    y1 = np.array([0.65])
    z1 = 2*x1*y1/(x1+y1)
    ax.scatter(x1, y1, z1, 
               cmap='Reds', marker='o', label='My Points 1')
    x2 = np.array([0.97])
    y2 = np.array([0.76])
    z2 = 2*x2*y2/(x2+y2)
    ax.scatter(x2, y2, z2, 
               cmap='Reds', marker='o', label='My Points 2')
    ax.legend()
    # 顯示圖形
    plt.show()
    
#f=lambda x,y:2*x*y/(x+y)
#surfacewireplot_(
#            f, xrange=(0, 1.2, 0.25), yrange=(0, 1.2, 0.25), zlim=(0,1),
#            figsize=(20,10))
def surfacewireplot_(f, xrange=(0, 1, 0.25), yrange=(0, 1, 0.25), zlim=(-1,1), 
                   stride=(1,1), X=[], Y=[], 
                   xlb='', ylb='', title='', axtitle='', file='',
                   figsize=(), xalue_lim=(), yalue_lim=(), 
                   c_alpha=1, mode='d', loc=0, edge_plank=(0,0,1,1),
                   canvas=None, showlog=0, subkeys={}, multi_mode='',
                   showmask=[], labels={}, colors='rainbow', vlines=[],
                   set_axis_range=1, 
                   **kwags):
    x = np.array(X) if(np.array(X).shape[0]>0) else np.arange(*xrange)
    y = np.array(Y) if(np.array(Y).shape[0]>0) else np.arange(*yrange)
    label_package={}
    label_package['xlabel'] = kwags.pop('xlabel') if(
                'xlabel' in kwags) else {}
    label_package['xtickslabel'] = kwags.pop('xtickslabel') if(
            'xtickslabel' in kwags) else {}
    label_package['ylabel'] = kwags.pop('ylabel') if(
            'ylabel' in kwags) else {}
    label_package['ytickslabel'] = kwags.pop('ytickslabel') if(
            'ytickslabel' in kwags) else {}
    if(np.array(x[0]).shape!=() and np.array(y[0]).shape!=()):
        subkwags = kwags['subkwags'] if('subkwags' in kwags) else {}
        printer('0:\n%s'%(str(colors)[:200]))
        fig, ax = multigraph_process(x, y, plot_method=surfacewireplot_, subkeys=subkeys, 
                           labels=labels, colors=colors, c_alpha=c_alpha,
                           vlines = vlines, edge_plank=edge_plank,
                           showmask=showmask, mode=mode, loc=loc, 
                           figsize=figsize, kwags=kwags, subkwags=subkwags,
                           canvas=canvas)
    else:
        if(True if((x.shape in {(), (0,), (1,)}) or 
                   (y.shape in {(), (0,), (1,)})) else (
                   x.shape[0]==0 or y.shape[0]==0)):
            printer("data size didn't match!! x:%d, y:%d"%(x.shape[0], y.shape[0]))
            printer('x:%s'%str(x)[:200])
            printer('y:%s'%str(y)[:200])
            return canvas
        x, y = np.meshgrid(x, y)
        z = f(x, y)
        kwags = clean_kwags(kwags)
        figsize = plot_set_figszie(x, y, kwags, figsize)
        fig = plt.figure(figsize=figsize) if(canvas==None) else canvas[0]
        ax = fig.gca(projection='3d') if(canvas==None) else canvas[1]
        ax.set_zlim3d(*zlim)
        printer('figure size=(%.2f,%.2f)'%(figsize[0], figsize[1]))
        if(set_axis_range):
            ax = plot_set_axis_range(ax, x, y, xrange[:2], yrange[:2])[0]
        kwags_stride = {'rstride':stride[0], 'cstride':stride[1]}
        ax.plot_wireframe(x, y, z, **kwags_stride)
#    printer('label_package:%s'%label_package)
    return plot_detail_process(
                fig, ax, kwags, 
                axtitle = axtitle, title=title, file=file, 
                **label_package)
    
def plot_start(*args, kwags={}, on=True):
    label_package={}
    label_package['xlabel'] = kwags.pop('xlabel') if(
                'xlabel' in kwags) else {}
    label_package['xtickslabel'] = kwags.pop('xtickslabel') if(
            'xtickslabel' in kwags) else {}
    label_package['ylabel'] = kwags.pop('ylabel') if(
            'ylabel' in kwags) else {}
    label_package['ytickslabel'] = kwags.pop('ytickslabel') if(
            'ytickslabel' in kwags) else {}
    if(not on):
        return label_package
    new_args = []
    for arg in args:
        new_args += [np.array(arg)]
    return tuple(new_args + [label_package])
    
#fig.subplots_adjust(bottom=edge_plank[1], top=edge_plank[0])
def plot_condition_judge(x_shape, y_shape, ii, jj, loc=0, mode='', 
                         showmask=[], **kwags):
    ret = (x_shape!=y_shape or x_shape[0]==0 or y_shape[0]==0)
    printer('[x-y-mode:%s][i:%d][j:%d][loc:%d][x-y-shape:%s]'%(
            mode, ii, jj, loc, ret), level=3)
    if(showmask):
        key = kwags['key'] if('key' in kwags) else (ii, jj)
        ret &= (not key in showmask)
    else:
        if(mode=='d'):
            ret |= (ii!=jj)
        if(mode=='x'):
            ret |= (ii!=loc)
        if(mode=='y'):
            ret |= (jj!=loc)
    return ret

def check_inputs_shapes(x, y):
    if(True if((x.shape in {(), (0,), (1,)}) or 
               (y.shape in {(), (0,), (1,)})) else (
            x.shape!=y.shape or x.shape[0]==0 or y.shape[0]==0)):
        printer("data size didn't match!! x:%d, y:%d"%(x.shape[0], y.shape[0]))
        printer('x:%s'%str(x)[:200])
        printer('y:%s'%str(y)[:200])
        return True
    return False

def make_ctr_tol(ax, mxn, ctr_tol, area_alpha=0.2):
    (amax, amin), (hx, hy) = mxn['amxn'], mxn['h']
    if(str(type(ctr_tol)).find('function')>-1):
        dx = np.linspace(amin-hx, amax+hx, 500)
        dy = np.linspace(amin-hy, amax+hy, 500)
        dx, dy = np.meshgrid(dx, dy)
        z = ctr_tol(dx.flatten(), dy.flatten()) + 0
        z = z.reshape(dx.shape)
        ax.contourf(dx, dy, z, alpha=area_alpha)
        t_ = np.linspace(amin-hx, amax+hx, 500)
        plt.plot(t_, t_, '--',color=(255/255,90/255,90/255))
    else:
        if(ctr_tol>0):
            t_ = np.linspace(amin-hx, amax+hx, 500)
            ax.plot(t_, t_, '--',color=(255/255,90/255,90/255))
            ax.plot(t_, [x - ctr_tol for x in t_], '--',color=(255/255,90/255,90/255))
            ax.plot(t_, [x + ctr_tol for x in t_], '--',color=(255/255,90/255,90/255))
    return ax

def plot_set_figszie(X, Y, kwags, figsize=(), on=True):
    if(not on):
        return None
    if(np.array(X[0])!=()):
        X = [v for x in X for v in list(x)]
    if(np.array(Y[0])!=()):
        Y = [v for y in Y for v in list(y)]
    mxalue_count = len(X)
    myalue_count = len(Y)
    xtkrot = kwags['xtkrot'] if('xtkrot' in kwags) else 0
    friendly_height = (2 + min(myalue_count*0.5,4) + 7*np.sin(xtkrot))
    return (min(mxalue_count*0.2,6)+4.5, friendly_height) if(
            figsize==()) else tuple(figsize)

def astype(subject, d_type=int, default=None):
    try:
        new_subject = d_type(subject)
    except:
        new_subject = default if(default!='same') else subject
    return new_subject

def dict_astype(old_dict, axis='keys', d_type=int):
    new_dict = {}
    for k in old_dict:
        if(axis.find('key')>-1):
            subject = dcp(k)
            new_k = astype(subject, d_type=d_type)
            if(new_k==None):
                continue
            new_dict[new_k] = old_dict[k]
        elif(axis.find('value')>-1):
            subject = dcp(old_dict[k])
            new_v = astype(subject, d_type=d_type)
            if(new_v==None):
                continue
            new_dict[k] = new_v
        else:
            new_dict[k] = old_dict[k]
    return new_dict

def set_grid_layout(plots, layout=(), figsize=(), row_ratio=3.5, col_ratio=5):
        colsize = np.sqrt(len(plots))//1 if(
                layout==()) else layout[1]
        rowsize = (len(plots)//colsize)+1 if(
                layout==()) else layout[0]
        figsize = (colsize*col_ratio, rowsize*row_ratio) if(
                figsize==()) else figsize
        return colsize, rowsize, figsize
            
def plot_set_axis_range(ax, X, Y, xalue_lim=None, yalue_lim=None, vlines=[], on=True,
                        **kwags):
    if(not on):
        return ax, {}
    vlines = [float(vlines[k]['value']) for k in vlines] if(
            type(vlines)==dict) else vlines
    if(np.array(X[0])!=()):
        X = [v for x in X for v in list(x)] + vlines
    if(np.array(Y[0])!=()):
        Y = [v for y in Y for v in list(y)]
    xmax, xmin = np.amax(X), np.amin(X)
    printer(xmax, xmin)
    ymax, ymin = np.amax(Y), np.amin(Y)
#    {}[('%d,'*4)%(xmax, xmin, ymax, ymin)]
    amax, amin = max(xmax, ymax), min(xmin, ymin)
    hx, hy=(xmax - xmin)/10, (ymax - ymin)/10
    xalue_lim = xalue_lim if(xalue_lim!=None) else (xmin-hx, xmax+hx)
    ax.set_ylim(*xalue_lim)
    yalue_lim = yalue_lim if(yalue_lim!=None) else (ymin-hy, ymax+hy)
    ax.set_ylim(*yalue_lim)
    return ax, {'xymxn':(xmin, xmax, ymin, ymax), 
                'amxn':(amin, amax), 'h':(hx, hy)}

def plot_detail_process(fig, ax, kwags={},
                    xlabel={}, xtickslabel={},
                    ylabel={}, ytickslabel={}, 
                    title='', axtitle='', file='', 
                    dotannos={}, **kwkwags):
    #TODO:canvas and (fig, ax)
    canvas = (fig, ax)
    if(dotannos):
        for dotanno in dotannos:
            ax.annotate(dotanno, dotannos[dotanno])
    if(xlabel):
        if((not 'subject' in xlabel) if(type(xlabel)==dict) else True):
            {}['let xlabel be a dict and set subject!!! wrong xlabel:%s'%str(xlabel)]
        subject = xlabel.pop('subject')
        ax.set_xlabel(subject, **xlabel)
    if(xtickslabel):
        subject = xtickslabel.pop('subject')
        ax.set_xticklabels(subject, **xtickslabel)
    if(ylabel):
        subject = ylabel.pop('subject')
        ax.set_ylabel(subject, **ylabel)
    if(ytickslabel):
        subject = ytickslabel.pop('subject')
        ax.set_yticklabels(subject, **ytickslabel)
    if(not 'dont_legend' in kwkwags):
        ax.legend()
    if(not axtitle==''):
        axtitle_kwags = kwags['axtitle_kwags'] if(
                'axtitle_kwags' in kwags) else  {}
        ax.set_title(axtitle, **axtitle_kwags)
    if(title!=''):
        title_kwags = kwags['title_kwags'] if(
                'title_kwags' in kwags) else  {}
        fig.suptitle(title, **title_kwags)
    if(not file==''):
        fig.savefig(file)
    if(canvas==None):
        fig.show()
        plt.ioff()
        return None
    else:
        return fig, ax

class PLANEMETRICS():
    def __init__(self):
        return 
        
    def circle(self, p=2):
        if(p<=0):
            {}['p-error:%.2f'%p]
        return (lambda P,Q: ((P[0]-Q[0])**p+(P[1]-Q[1])**p)**(1/p))
    
    def rectangle(self, abr=1, theta=0):
        return (lambda P,Q: np.max(np.array(
                [abr*np.abs(P[0]-Q[0]),np.abs(P[1]-Q[1])]), axis=0))
        
    def diamond(self, abr=1, theta=0):
        return (lambda P,Q: abr*np.abs(P[0]-Q[0])+np.abs(P[1]-Q[1]))
plm = PLANEMETRICS()

def binary_classify_seqential_data(data, field_sz, xheaders, yheaders, subplot_method,
                                   title='', file='',
                                   showmask = 'full',
                                   subplot_showmask = 'diagonal',
                                   binary_symbol=['OK','NG'], 
                                   population_data_T=pd.DataFrame(),
                                   xarray_method = None, 
                                   xarray_method_inputs={},
                                   yarray_method = None, 
                                   yarray_method_inputs={},
                                   separate_method=None, 
                                   separate_method_inputs={},
                                   colors_method = None, colors_method_inputs={},
                                   label_method=None, label_method_inputs={},
                                   filter_method = None, filter_method_inputs={},
                                   showlog=6):
    if(np.array(binary_symbol).shape[0]!=2):
        {}['symbol shape error:%s'%str(np.array(binary_symbol).shape)]
    if(showmask=='full'):
        showmask = [(xn,yn) for xn in xheaders for yn in yheaders]
    elif(showmask=='diagonal' and xheaders==yheaders):
        showmask = [(k,k) for k in xheaders]
    gd = GRAPHDATA()
    if(population_data_T.empty):
        data_T = data.T.copy()
    for xn in xheaders:
        for yn in yheaders:
            if((xn, yn) in showmask):
                subtitle = yn if(len(xheaders)==1) else (xn, yn)
                for i in range(0, len(data.index), field_sz):
                    indexes_now = data.index[i:i+field_sz].copy()
                    data_a_field = population_data_T[indexes_now].T.copy() if(
                            not population_data_T.empty) else data_T[indexes_now].T.copy()
                    if(filter_method):
                        filter_method_inputs.update({'data_a_field':data_a_field,
                                                     'subtitle':subtitle})
                        if(filter_method(**filter_method_inputs)):
                            continue
                    separate_method_inputs.update({'data_a_field':data_a_field.copy(),
                                                   'binary_symbol':binary_symbol})
                    isgood = separate_method(**separate_method_inputs)
                    xarray = [v for v in (data_a_field.index.copy() if(
                            xn=='index') else data_a_field[xn].copy())]
                    if(xarray_method):
                        xarray_method_inputs.update({'array':xarray})
                        xarray = xarray_method(**xarray_method_inputs)
                    yarray = [v for v in (data_a_field.index.copy() if(
                            yn=='index') else data_a_field[yn].copy())]
                    if(yarray_method):
                        yarray_method_inputs.update({'array':yarray})
                        yarray = yarray_method(**yarray_method_inputs)
                    showlog = gd.add_data([{(isgood, int(i/field_sz)):xarray}, 
                                           {(isgood, int(i/field_sz)):yarray}], 
                                key=subtitle, new_frame={}, showlog=showlog)
                if(subplot_showmask=='diagonal'):
                    subplot_showmask = [(k,k) for k in gd.X[subtitle].keys()]
                elif(subplot_showmask=='full'):
                    subplot_showmask = [(
                            i,j) for i in gd.X[subtitle].keys() for j in gd.Y[subtitle]]
                colors = None
                if(colors_method!=None):
                    colors_method_inputs.update({'binary_symbol':dcp(binary_symbol),
                                                 'subtitle':dcp(subtitle)})
                    colors = colors_method(**colors_method_inputs)
                label = None
                if(label_method!=None):
                    label_method_inputs.update({'data_a_subtitle':dcp(gd.X[subtitle]),
                                                'binary_symbol':dcp(binary_symbol),
                                                'subtitle':dcp(subtitle)})
                    label = label_method(**label_method_inputs)
                gd.set_properties(pn='method', value=dcp(subplot_method), key = subtitle)
                gd.set_properties(pn='label', value=dcp(label), key = subtitle)
                gd.set_properties(pn='showmask', value=dcp(subplot_showmask), key = subtitle)
                gd.set_properties(pn='colors', value=dcp(colors), key = subtitle)
    gd.gridplot(file=file, title = title, titlefontsize=15,
                edge_plank=(0.05,0,0.93,1)) 

def gridplot(plots, layout=(), file='', title='', titlefontsize = 20,
             figsize=(), edge_plank=(0,0,1,1), canvas=None, **kwags):
    colsize = int(np.sqrt(len(plots))//1) if(
            layout==()) else layout[1]
    rowsize = int((len(plots)//colsize))+1 if(
            layout==()) else layout[0]
    figsize = (colsize*5, rowsize*3.5) if(
            figsize==()) else figsize
    printer(len(plots), colsize, rowsize)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(*edge_plank)
    
    for index, plot in enumerate(plots):
        ax = plt.subplot(rowsize, colsize, index+1)
        method = plots[plot]['method']
        printer('[%s][%s]:%s||%s'%(index, plot, ax, fig))
        fig, ax = method(*tuple(plots[plot]['inputs']), canvas=(fig, ax),
                         **dict(plots[plot]['params'] if(
                                 'params' in plots[plot]) else {}))
    if(title!=''):
        fig.suptitle(title, fontsize=titlefontsize)
    if(file!=''):
        fig.savefig(file)
    if(canvas):
        return fig, ax
    else:
        fig.show()
        plt.ioff()

def gridplot_(plots, layout=(), plot_method=None, 
              figsize=(), canvas=None, **kwags):
    method_inputs = dcp(common_method_inputs)
    label_package = plot_start(kwags=kwags)[2]
    colsize, rowsize, figsize = set_grid_layout(plots, layout=layout, figsize=figsize)
    printer(len(plots), colsize, rowsize)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(*edge_plank)
    
    for index, plot in enumerate(plots):
        ax = plt.subplot(rowsize, colsize, index+1)
        method = plots[plot]['method'] if(
                'method' in plots[plot]) else dcp(plot_method)
        printer('[%s][%s]:%s||%s'%(index, plot, ax, fig))
        method_inputs.update(dict(plots[plot]['inputs'] if(
                        'inputs' in plots[plot]) else {}))
        method_inputs = clean_common_params(method_inputs)
        fig, ax = method(canvas=(fig, ax), **method_inputs)
        
    return plot_detail_process(
                fig, ax, kwags, 
                axtitle = '', title=title, file=file, 
                **label_package)

#c_alpha=0.2
#plots_keys = ['gap1L_goal','gap1R_goal', 
#              'gap2L_goal', 'gap2R_goal']
#plots = {}
#for i, plot in enumerate(plots_keys):
#    plots[plot] = {}
#    plots[plot]['method'] = scatterplot_
#    plots[plot]['inputs'] = (list(D['%s_fct'%plot].astype(float)), 
#                             list(D['%s_pdt'%plot].astype(float)))
#    plots[plot]['params'] = {'label':'%d'%i,
#                             'color':(*tuple(cm.rainbow(np.linspace(
#                                            0,1,len(plots_keys)))[i][:3]), c_alpha)}
#coverplot(plots, label_callback_key='label')
def coverplot(plots, mode='d', layout=(), file='', title='',
         ctr_tol=0, xtkrot=0, titlefontsize = 20, plot_method=None, 
         is_key_label=True, label_callback_key='labels', 
         figsize=(), edge_plank=(0,0,1,1), 
         showmask=[], canvas=None, showlog=0, **kwags):
    all_xvalues = [v for k in plots.keys() for v in plots[
                                k]['inputs'][0]]
    all_yvalues = [v for k in plots.keys() for v in plots[
                                k]['inputs'][1]]
    all_values = all_xvalues + all_yvalues
    mmax, mmin = np.amax(all_values), np.amin(all_values)
    printer('max:%s'%(str(np.amax(all_values).round(2))[:100]))
    printer('min:%s'%(str(np.amin(all_values).round(2))[:100]))
    mxalue_count = len(
            set(tuple(np.array(all_xvalues).reshape(-1))))
    myalue_count = len(
            set(tuple(np.array(all_yvalues).reshape(-1))))
    friendly_height = (2 + min(myalue_count*0.5,4) + 7*np.sin(xtkrot))
    colsize = min(mxalue_count*0.2,6)+4 if(
            layout==()) else layout[1]
    rowsize = friendly_height if(
            layout==()) else layout[0]
    figsize = (colsize, rowsize) if(
            figsize==()) else figsize
    printer(len(plots), colsize, rowsize)
    fig, ax = plt.subplots(figsize=figsize) if(
                canvas==None) else (canvas[0], canvas[1]) #figsize=(5,3.5)
    fig.subplots_adjust(**edge_plank)
    h=(mmax - mmin)/10
    if(str(type(ctr_tol)).find('function')>-1):
        dx = np.linspace(mmin-h, mmax+h, 500)
        dy = np.linspace(mmin-h, mmax+h, 500)
        dx, dy = np.meshgrid(dx, dy)
        z = ctr_tol(dx.flatten(), dy.flatten()) + 0
        z = z.reshape(dx.shape)
        plt.contourf(dx, dy, z, alpha=0.2)
        t_ = np.linspace(mmin-h, mmax+h, 500)
        plt.plot(t_, t_, '--',color=(255/255,90/255,90/255))
    else:
        if(ctr_tol>0):
            t_ = np.linspace(mmin-h, mmax+h, 500)
            plt.plot(t_, t_, '--',color=(255/255,90/255,90/255))
            plt.plot(t_, [x - ctr_tol for x in t_], '--',color=(255/255,90/255,90/255))
            plt.plot(t_, [x + ctr_tol for x in t_], '--',color=(255/255,90/255,90/255))
    for index, plot in enumerate(plots):
        if(showmask):
            if(not plot in showmask):
                continue
        if(showlog>0):
            printer('[%d][%s]drawing....'%(index, str(plot)))
        method = dcp(plot_method if(
                not 'method' in plots[plot]) else plots[plot]['method'])
        method = (lambda canvas, **kwags: (canvas[0], canvas[1])) if(
                method==None) else method
        if(showlog>0):
            printer('\nplot method!!\n')
        if(showlog>0):
            printer('[%s][%s]:%s||%s'%(index, plot, ax, fig))
        method_params = plots[plot]['params'] if(
                                 'params' in plots[plot]) else {}
        if(is_key_label):
            key_label = str(plot)
            assigned_labels = dcp(method_params[label_callback_key]) if(
                    label_callback_key in method_params) else ''
            method_params.update({label_callback_key:(
            '[%s]%s'%(key_label, str(assigned_labels)) if(
                type(assigned_labels)!=dict
                    ) else {k:'[%s]%s'%(key_label, assigned_labels[k]
                        ) for k in assigned_labels.keys()})})
        try:
            fig, ax = method(*tuple(plots[plot]['inputs']), canvas=(fig, ax),
                         **dict(method_params))
        except Exception as e:
            printer('[%d][%s]method:%s'%(
                    index, plot, str(method)))
            printer('method_params:\n%s'%(
                    str(method_params)[:100]))
            {}['error:%s'%e]
    ax.legend()
    if(title!=''):
        fig.suptitle(title, fontsize=titlefontsize)
    if(file!=''):
        fig.savefig(file)
    if(canvas):
        return fig, ax
    else:
        fig.show()
        plt.ioff()
        return None
    
def multifigcurveplot(X, Y, pictures=(), title='', labels='', vlines=[], layout=(),
                      figsize=(), titlefontsize = 20, file='', edge_plank=(0,0,1,1),
                      **kwags):
    #pictures = list(set(xnames)-{'RODT','bckswg','frtswg','mdlswg'})
    colsize = np.sqrt(len(pictures))//1 if(
            layout==()) else layout[1]
    rowsize = (len(pictures)//colsize)+1 if(
            layout==()) else layout[0]
    plt.subplots_adjust(*edge_plank)
    figsize = (colsize*5, rowsize*3.5) if(
            figsize==()) else figsize
    plt.figure(figsize=figsize)
    if(type(vlines)!=dict):
        vlineslist = list(vlines)
        vlines = {-1:vlineslist}
    for index, feature in enumerate(pictures):
        x = X if(type(X)!=dict) else (
                X[feature] if(feature in X.keys()) else None)
        y = Y if(type(Y)!=dict) else (
                Y[feature] if(feature in Y.keys()) else None)
        if(x==None or y==None):
            continue
        
        label = str(labels) if(type(labels)!=dict) else (
                labels[feature] if(feature in labels.keys()) else '')
        vline = vlines[feature] if(feature in vlines.keys()) else vlines
        
        
        plt.subplot(rowsize, colsize, index+1)
        plt.title('%s'%feature)
        plt.plot(x, y, label=label)
        ls, color = '--', (0/255,0/255,0/255, 0.3)
        if(not -1 in vline.keys()):
            for vn in vline:
                ls, color = '--', (0/255,0/255,0/255, 0.3)
                vL = list(vline[vn]['values']) if(
                        'values' in vline[vn].keys()) else []
                if('color' in vline[vn].keys()):
                    color = vline[vn]['color']
                if('ls' in vline[vn].keys()):
                    ls = vline[vn]['ls']
                for vl in vL:
                    plt.axvline(x=vl, ls=ls, color=color)
        else:
            vL = list(vline[-1])
            for vl in vL:
                plt.axvline(x=vl, ls=ls, color=color)
        if(not label==''):
            plt.legend()
    if(title!=''):
        plt.suptitle(title, fontsize=titlefontsize)
    if(file!=''):
        plt.savefig(file)
    plt.show()
    plt.ioff()

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
    
def drawparams(grid_search, file='', xbd=(), ybd=(0.6, 1.1), figsize=(), 
               markersize=5, xlb_rotation=90, showmodes=False):
    results = pd.DataFrame(grid_search.cv_results_)
    best = np.argmax(results.mean_test_score.values)
    if(figsize==()):
        pfy = len(grid_search.best_params_)*8

        pfx = (len(grid_search.cv_results_['params'])*len(grid_search.cv_results_['params'][0])*
               len(str(grid_search.cv_results_['params'][0])))/20
        figsize = (pfx, pfy)
    plt.figure(figsize = figsize)
    xbd = (-1, len(results)) if xbd==() else xbd
    plt.xlim(xbd[0], xbd[1])
    plt.ylim(ybd[0], ybd[1])
    #marker_best = 0
    for i, (_, row) in enumerate(results.iterrows()):
        scores = row[['split%d_test_score' % j for j in range(5)]]
        marker_cv, = plt.plot([i] * 5, scores, '^', c='gray', markersize=markersize,alpha=.5)
        marker_mean, = plt.plot(i, row.mean_test_score, 'v', c='none', alpha=1,
                                markersize=2*markersize, markeredgecolor='k')
        if i == best:
            marker_best, = plt.plot(i, row.mean_test_score, 'o', c='red',
                                    fillstyle="none", alpha=1, markersize=4*markersize,
                                    markeredgewidth=1)
    if(not showmodes):
        
        grid_par = [drop(d,'kernel') for d in grid_search.cv_results_['params']]
    else:
        grid_par = grid_search.cv_results_['params']
    fontsize = markersize*1.5
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize/1.5)
    plt.xticks(range(len(results)), [str(x).strip("{}").replace("'", "") for x
                                     in grid_par],rotation=xlb_rotation)
    plt.ylabel("Validation score")
    plt.xlabel("Parameter settings")
    plt.legend([marker_cv, marker_mean, marker_best],
               ["cv score", "mean score", "best parameter setting"],
               loc=(1.05, .4))
    #plt.legend([marker_cv, marker_mean],
    #           ["cv accuracy", "mean accuracy"],
    #           loc=(1.05, .4))
    if(file!=''):
        plt.savefig(file)
    plt.show()
    plt.ioff()

def plot_neighborhoods(cores, r=1, axis_edge = (-5,20,20,-5), 
              metric_method=None, delta = 0.025, showcore=False,
              color = None, figsize=(5,5), file='', xtkrot=0,
              labels='', multiline=True, showmask=None,
              canvas=None, showlog=0, **kwags):
    cores = np.array(cores).reshape(-1,2)
    if(cores.shape[1]!=2):
        {}['core error:%s'%str(c)]
    if(type(r)!=list):
        r = np.array([r]*cores.shape[0])
    r = np.array(r)
    if(r.shape[0]!=cores.shape[0]):
        {}['size error:r.shape=%s, cores.shape=%s'%(
                str(r.shape), str(cores.shape))]
    if(metric_method==None):
        def metric(X, Y):
            b=1
            for i in range(r.shape[0]):
                b *= plm.circle(2)(cores[i], (X,Y))-r[i]
            return b
    else:
        def metric(X, Y):
            b=1
            for i in range(r.shape[0]):
                b *= metric_method(cores[i], (X,Y))-r[i]
            return b
    if(canvas!=None):
        fig, ax = canvas
    else:
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot()
    xrange = np.arange(axis_edge[0], axis_edge[2], delta)
    yrange = np.arange(axis_edge[3], axis_edge[1], delta)
    X, Y = np.meshgrid(xrange, yrange)
    ax.contour(X, Y, metric(X,Y), [0], colors=color)
    if(showcore):
        ax.scatter(cores[:,0], cores[:,1])
    if(not file==''):
        fig.savefig(file)
    if(canvas):
        return fig, ax
    fig.show()

def draw_GPC_pb_2d(dfxx, visualize_ftrs, dfyy, gpc, dfpp=[], 
                   section_method='median', grid_size=100, 
                   title='', subtitles={}, file=''):
    dfxx = pd.DataFrame(dfxx)
    dfyy = pd.DataFrame(dfyy)
    if(dfpp!=[]):
        dfpp = pd.DataFrame(dfpp)
    dfx_visual =  dfxx[visualize_ftrs]
    
    #plt.figure(figsize=(len(dfyy)* 2, 1 * 2))
    plt.subplots_adjust(bottom=.2, top=.95)
    
    x1max = np.mat(dfx_visual).transpose()[0].max()
    x1min = np.mat(dfx_visual).transpose()[0].min()
    x2max = np.mat(dfx_visual).transpose()[1].max()
    x2min = np.mat(dfx_visual).transpose()[1].min()
    """tmax = np.mat(dfx_visual).max()
    tmin = np.mat(dfx_visual).min()"""
    xx1 = np.linspace(x1min, x1max, grid_size)
    xx2 = np.linspace(x2min, x2max, grid_size).T
    xx = {attr:None for attr in visualize_ftrs}
    xx[visualize_ftrs[0]], xx[visualize_ftrs[1]] = np.meshgrid(xx1, xx2)
    
    n_fill = grid_size**2
    xxgrid = np.zeros(n_fill)
    for col in dfxx.columns:
        if(col in visualize_ftrs):
            add_array = xx[col].ravel()
        else:
            if(section_method=='median'):
                add_array = np.array([np.median(np.array(dfxx[col]))]*n_fill)
            elif(str(type(section_method))=='function'):
                add_array = np.array(section_method(np.array(dfxx[col]), {'col':col}))
        xxgrid = np.c_[xxgrid, add_array.ravel()]
    xxgrid = np.array(xxgrid[:,1:])
    probgrid = gpc.predict_proba(xxgrid)
    classes = ['OK','NG'] if (len(dfyy.columns)==1) else list(dfyy.columns)
    n_classes = len(classes)
    for index, k in enumerate(classes):
        plt.subplot(1, n_classes, index+1)
        plt.title("Class %d" % index) if(subtitles=={}) else subtitles[k]
        imshow_handle = plt.imshow(probgrid[:, index].reshape(grid_size, grid_size),
                                   extent=(x1min, x1max, x2min, x2max), origin='lower')
        plt.xticks(())
        plt.yticks(())
        
        idx = (np.array(dfyy).ravel() == index)
        if(idx.any()):
            plt.scatter(np.array(np.mat(dfx_visual)[idx, 0]), 
                        np.array(np.mat(dfx_visual)[idx, 1]), 
                        marker='o', c='w', edgecolor='k')
        if(str(type(dfpp))[8:-2]=='pandas.core.frame.DataFrame'):
            idxp = (np.array(dfpp).ravel() == index)
            if(idxp.any()):
                plt.scatter(np.array(np.mat(dfx_visual)[idxp, 0]), 
                            np.array(np.mat(dfx_visual)[idxp, 1]), 
                            marker='o', c=(1,1,1,0.5), edgecolor=(0,0,0,0.5))
    ax = plt.axes([0.15, 0.04, 0.7, 0.05])
    if(title!=''):
        plt.title(title)
    plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')
    if(file!=''):
        plt.savefig(file)

def calculate_gaussian_kl_divergence(m1,m2,v1,v2):
    try:
        return np.log(v2 / v1) + ((v1**2)+(m1 - m2)**2)/(2*(v2**2)) - 0.5
    except:
        return 10*8

def plot_scatter_matrix(df0, vrbs, pairing_header='all', clf_header=[],
                        file='', corner=True, height=3, kind=None,
                        diag_kind='auto', handler=None, **kwags):
        pairing_header = [c for c in df0.columns if not c in clf_header] if(
                pairing_header=='all') else pairing_header
        df = df0[pairing_header].copy()
        if(clf_header!=[]):
            df['class'] = df0[clf_header].T.apply(
                    lambda x: ','.join(x.dropna().astype('str')))
        fig = sns.pairplot(df, hue=('class' if(clf_header!=[]) else None),
                           kind=kind, height=height, vars=vrbs, 
                           diag_kind=diag_kind).fig
        if(file!=''):
            fig.savefig(file)
        if(handler is not None):    handler.fig = fig
        return True

def normdistb_fit(data, classes=None, title='', xlb='', ylb='probability density', 
                  vlines=[], file='', isevaluate = False, num_bins = 100, 
                  histtype = 'barstacked', density = True, 
                  histalpha = 0.5, xmax=None, xmin=None, df=2, n_sigma=2, infrms=None, do_stats=True):
    classes = classes if(isinstance(classes, dict)) else {0:{'mask':np.full(np.array(data).shape[0], True)}}
    data = np.array(data)
    fig, ax = plt.subplots(figsize=(10,5))
    axtw = ax.twinx()
    xmax = data.max() if(xmax==None) else xmax
    xmin = data.min() if(xmin==None) else xmin
    x_ = np.linspace(xmin, xmax, num_bins)
    infrms = infrms if(isinstance(infrms, dict)) else {}
    for classid in classes:
        idx = np.array(classes[classid]['mask'])
        x = data[idx]
        diversity = len(list(set(x)))
        infrm = infrms.get(classid, {})
        if(diversity<2):
            printer('資料種類%d太少，不常態擬合!!'%diversity)
        else:
            if(do_stats):
                infrm = {}
                DFP.normfit(x, n_sigma=n_sigma, ret=infrm)
            if('count' in infrm):    infrm['$n$'] = DFP.parse(infrm.pop('count'))
            mu = infrm.pop('mean')
            std = infrm.pop('std')
            infrm['$\mu$'] = DFP.parse(mu)
            infrm['$\sigma$'] = DFP.parse(std)
            if('norm_upper' in infrm):    infrm['$usl$'] = DFP.parse(infrm.pop('norm_upper'))
            if('norm_lower' in infrm):    infrm['$lsl$'] = DFP.parse(infrm.pop('norm_lower'))
            if('p' in infrm):    infrm['$p$'] = DFP.parse(infrm.pop('p'))
            if('W' in infrm):    infrm['$W$'] = DFP.parse(infrm.pop('W'))
        infrms[classid] = infrm
        class_color = dcp(classes[classid].get('color', (0.1,0.05,0.4)))
        n, bins, patches = ax.hist(x, num_bins, 
                                   density=density, 
                                   stacked=True, 
                                   histtype='bar' if(histtype==None) else histtype,
                                   color=tuple(list(class_color)[:3])+(histalpha,))
        if(diversity>=2):
            y = scipy.stats.norm.pdf(x_,mu,std)
            axtw.plot(x_, y, '--', color=class_color, label=r'%s:%s'%(classid, LOGger.stamp_process('',infrm,'=','','',', ')))
            if(DFP.astype(infrm.get('$lsl$'))!=None):
                threshold = DFP.astype(infrm['$lsl$'])
                threshold_color = dcp(classes[classid].get('threshold_color', class_color))
                axtw.axvline(threshold, color=threshold_color, ls='-.')
            if(DFP.astype(infrm.get('$usl$'))!=None):
                threshold = DFP.astype(infrm['$usl$'])
                threshold_color = dcp(classes[classid].get('threshold_color', class_color))
                axtw.axvline(threshold, color=threshold_color, ls='-.')
    axtw.set_xlabel(xlb)
    axtw.set_ylabel(ylb)
    axtw.legend()
    axtw.set_title(r'Histogram and normdstrb of %s'%(title))
    fig.tight_layout()
    if(file!=''):
        fig.savefig(file)
    else:
        fig.show()

def chi2distb_fit(data, classes=None, title='', xlb='', ylb='probability density', 
                  vlines=[], file='', isevaluate = False, num_bins = 100, 
                  histtype = 'barstacked', density = True, 
                  histalpha = 0.5, xmax=None, xmin=None, df=2, alpha=0.05, infrms=None, do_stats=True):
    classes = classes if(isinstance(classes, dict)) else {0:{'mask':np.full(np.array(data).shape[0], True)}}
    data = np.array(data)
    fig, ax = plt.subplots(figsize=(10,5))
    axtw = ax.twinx()
    xmax = data.max() if(xmax==None) else xmax
    xmin = data.min() if(xmin==None) else xmin
    x_ = np.linspace(xmin, xmax, num_bins)
    infrms = infrms if(isinstance(infrms, dict)) else {}
    for classid in classes:
        idx = np.array(classes[classid]['mask'])
        x = data[idx]
        diversity = len(list(set(x)))
        infrm = infrms.get(classid, {})
        if(diversity<2):
            printer('資料種類%d太少，不常態擬合!!'%diversity)
        else:
            if(do_stats):
                infrm = {}
                DFP.chi2fit(x, alpha=alpha, ret=infrm)
            if('count' in infrm):    infrm['$n$'] = DFP.parse(infrm.pop('count'))
            if('df' in infrm):    
                df = infrm.pop('df')
                infrm['$\delta$'] = DFP.parse(df)
            if('loc' in infrm):    infrm['$x_0$'] = DFP.parse(infrm.pop('loc'))
            if('scale' in infrm):    infrm['$\lambda$'] = DFP.parse(infrm.pop('scale'))
            if('chi2_upper_adjusted' in infrm):    infrm['$ucl$'] = DFP.parse(infrm.pop('chi2_upper_adjusted'))
            if('p' in infrm):    infrm['$p$'] = DFP.parse(infrm.pop('p'))
            if('W' in infrm):    infrm['$W$'] = DFP.parse(infrm.pop('W'))
        infrms[classid] = infrm
        class_color = classes[classid].get('color', (0.1,0.05,0.4))
        n, bins, patches = ax.hist(x, num_bins, 
                                   density=density, 
                                   stacked=True, 
                                   histtype='bar' if(histtype==None) else histtype,
                                   color=tuple(list(class_color)[:3])+(histalpha,))
        if(diversity>=2):
            y = scipy.stats.chi2.pdf(x_, df=df)
            axtw.plot(x_, y, '--', color=class_color, label=r'%s:%s'%(classid, LOGger.stamp_process('',infrm,'=','','',', ')))
            if(DFP.astype(infrm.get('$ucl$'))!=None):
                threshold = DFP.astype(infrm['$ucl$'])
                threshold_color = classes[classid].get('threshold_color', class_color)
                axtw.axvline(threshold, color=threshold_color, ls='-.')
    axtw.set_xlabel(xlb)
    axtw.set_ylabel(ylb)
    axtw.legend()
    axtw.set_title(r'Histogram and chi2dstrb of %s'%(title))
    fig.tight_layout()
    if(file!=''):
        fig.savefig(file)
    else:
        fig.show()

def plot_countour(x,y,z,cmap=None):
    # define grid.
    xi = np.linspace(-2.1,2.1,100)
    yi = np.linspace(-2.1,2.1,100)
    ## grid the data.
    zi = scipy.interpolate.griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    # contour the gridded data, plotting dots at the randomly spaced data points.
    CS = plt.contour(xi,yi,zi,6,linewidths=0.5,colors='k')
    #CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
    CS = plt.contourf(xi,yi,zi,6,cmap=(cmap if(cmap!=None) else cm.Greys_r))
    #plt.colorbar() # draw colorbar
    # plot data points.
    #plt.scatter(x,y,marker='o',c='b',s=5)
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.title('griddata test (%d points)' % len(z))
    plt.show()
    
#TODO:matrix_dataframe
def matrix_dataframe(mtx, title='', file='', index=None, header=None, bbox=[0, 0, 1, 1],
                     headerhide=True, indexhide=True, reshape=(), phase=0, ax=None, 
                     col_width=2, size=(), font_size=13, fontproperties=None, fontname=None):
    if(indexhide==True):
        index=None
    if(headerhide==True):
        header=None
    dmtx = pd.DataFrame.from_dict(mtx) if(type(mtx)==dict) else mtx
    npmtx = np.array(dmtx)
    df = pd.DataFrame(npmtx, 
                      index=index if(DFP.isiterable(index)) else dmtx.index, 
                      columns=header if(DFP.isiterable(header)) else dmtx.columns)
    if(np.array([list(reshape)]).shape==(1,2)):
#        headerhide, indexhide = True, True
        a=reshape[0]
        b=reshape[1]
        printer(a,b,len(np.array(df).ravel()),np.array(df).shape[0])
        if(a*b>=len(np.array(df).ravel()) and np.array(df).shape[0]==1):
            ddf = (pd.DataFrame(np.array(df.columns)).T)[range(b)]
            for k in range(a*2-1):
                K=k//2
                if(k%2==1):
                    ddf = ddf.append(pd.DataFrame(
                            np.mat(df.columns[b*(K+1):b*(K+2)] if(
                                    len(np.array(df.columns[b*(K+1):b*(K+2)]))==b) else(
                                            list(df.columns[b*(K+1):b*(K+2)])+['']*(
                                                    b-len(np.array(df.columns[b*(K+1):b*(K+2)]))))) , 
                            index=[k+1], columns=list(range(b))))
                else:
                    ddf = ddf.append(pd.DataFrame(np.mat(
                                    [df[col][list(df.index)[0]] for col in list(df.columns)[b*K:b*(K+1)]]+
                                    ([] if(len(np.array(df.columns[b*K:b*(K+1)]))==b) else(
                                         ['']*(
                                            b-len(np.array(df.columns[b*K:b*(K+1)])))))),
                                    index = [k+1], columns=range(b)))
            df = ddf.copy()
    fig, ax = render_mpl_table(df, header_columns=0, col_width=col_width, bbox=bbox,
                               columnhide = headerhide, phase=phase, fontproperties=fontproperties, fontname=fontname,
                               indexhide = indexhide, size=size, font_size=font_size, ax=ax)
    if(LOGger.isinstance_not_empty(title, str)):
        ax.set_title(title, fontproperties=(fontproperties if(fontproperties) else fontname))
    if(LOGger.isinstance_not_empty(file, str)):
        fig.savefig(file)
    return fig, ax

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=13,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0, phase=0,
                     ax=None, index=False, columnhide=False, indexhide=False, size=(), 
                     **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height]) if(size==()) else size
        fig, ax = plt.subplots(figsize=size)
    mpl_table = ax.table(cellText=data.values, bbox=bbox, loc='right', 
                         colLabels=data.columns if(columnhide==False) else None, 
                         rowLabels=data.index if(indexhide==False) else None)
    fontproperties = kwargs.get('fontproperties', None) 
    fontproperties = fontproperties if(fontproperties) else kwargs.get('fontname', None)
    # print(fontproperties)
    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if(phase==0):
            if(columnhide == False and (k[0] == 0 or k[1] < header_columns)):
                cell.set_text_props(weight='bold', color='w', 
                                    fontproperties=fontproperties)
                cell.set_facecolor(header_color)
            else:
                cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
        else:
            if k[0]%phase == 0 or k[1] < header_columns:
                cell.set_text_props(weight='bold', color='w', 
                                    fontproperties=fontproperties)
                cell.set_facecolor(header_color)
            else:
                cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    # 隐藏边框（spines）
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax.get_figure(), ax

def showdigit(digit_data, x_pixels, y_pixels, file='', 
              cmap=matplotlib.cm.binary, interpolation='nearest'):
    some_digit_image = digit_data.reshape(x_pixels, y_pixels)
    plt.imshow(some_digit_image, cmap=cmap, interpolation=interpolation)
    plt.axis("off")
    if(not file==''):
        plt.savefig(file)
    plt.ioff()
    plt.show()

def subscrtb(ob, contain_str=False):
    try:
        oob = ob[0]
    except:
        oob = ob
    return (not oob==ob) and (contain_str or type(ob)!=str)

def drawhistogram(seq, subseq=None, file='', xlb='', ylb='', title='', xtkrot=0, figsize=(), 
                  maxmin=(), hratio=1, barwidth=0.6, color='#0504aa'):
    n = len(seq)
    if(type(seq)==list):
        n, bins, patches = plt.hist(x=seq, bins='auto', color=color,
                            alpha=0.7, rwidth=barwidth)
        plt.grid(axis='y', alpha=0.75)
        #plt.text(23, 45, r'$\mu=15, b=3$')
        maxfreq = n.max()
        # Set a clean upper y-axis limit.
        plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    elif(type(seq)==dict):
        #pos = np.arange(len(seq.keys()))
        adseq={str(k):float(seq[k]) if (type(seq[k])==int or type(seq[k])==float or type(seq[k])==str
                 ) else (len(seq[k]) if (subscrtb(seq[k])) else seq[k]) for k in seq.keys()}
        # gives histogram aspect to the bar diagram
        friendly_height = (len(set(list(adseq.values())))*0.02+15)*hratio
        yvaluemax = np.array(list(adseq.values())).max() if (len(maxmin)<2) else maxmin[1]
        yvaluemin = np.array(list(adseq.values())).min() if (len(maxmin)<2) else maxmin[0]
        figsz = (n*0.2+4, friendly_height) if(figsize==()) else tuple(figsize)
        f, ax = plt.subplots(figsize=figsz) # set the size that you'd like (width, height)
        if(len(maxmin)<2):
            ystd = np.std(np.array(list(adseq.values())))
            yunit = ystd // 10 if ystd>=10 else (int(ystd) if ystd>1 else 1)
            plt.ylim(ymax = np.ceil(yvaluemax / yunit) * yunit if yvaluemax % yunit else yvaluemax + yunit,
                     ymin = (yvaluemin // yunit) * yunit if yvaluemin % yunit else yvaluemin - yunit)
        else:
            plt.ylim(maxmin[0], maxmin[1])
        if(type(color)==dict):
            for k in adseq.keys():
                cur_color = '#0504aa'
                for c in color.keys():
                    if(k.find(c)>-1):
                        cur_color = str(color[c])
                plt.bar([k], [adseq[k]], barwidth, color=cur_color)
        else:
            plt.bar(adseq.keys(), adseq.values(), barwidth, color=color)
    plt.title(title)
    plt.xlabel(xlb)
    plt.ylabel(ylb)
    plt.xticks(rotation=xtkrot)
    if(not file==''):
        plt.savefig(file)
    plt.ioff()
    plt.show()
    
def vlinesplot_(vlines, is_key_label = False, 
                xalue_lim=(), yalue_lim=(), figsize=(), 
                colors='rainbow', c_alpha=1, 
                set_axis_range=0, canvas=None, **kwags):
    float_format = '%.1f' if('float_format' in kwags) else '%s'
    vlines = dict(vlines) if(type(vlines)==dict) else {
            k:{'value':v} for k,v in enumerate(vlines)}
    vline_values = [float(vlines[k]['value']) for k in vlines]
    figsize = plot_set_figszie(
            vline_values, [0], kwags, figsize)
    fig, ax = plt.subplots(figsize=figsize) if(
                canvas==None) else (canvas[0], canvas[1]) #figsize=(5,3.5)
    if(set_axis_range):
        fig.subplots_adjust(*edge_plank)
        ax, mxn = plot_set_axis_range(ax, vline_values, [0], 
                xalue_lim, yalue_lim, vlines)
        printer(ax.get_xlim())
        printer(ax.get_ylim())
    if(colors=='rainbow'):
        colors = [v for v in cm_rainbar(
                _n=len(vlines), c_alpha=c_alpha)]
    if(c_alpha!=None):
        c_alpha = max(min(c_alpha, 1), 0)
        colors = [(*tuple(v)[:3], c_alpha) for v in colors]
    for i, k in enumerate(vlines):
        vline_property = dict(vlines[k]) if(k in vlines) else {}
        value = vline_property.pop('value')
        vline_property['ls'] = dcp(vline_property['ls']) if('ls' in vline_property) else (
                dcp(vline_property['ls']) if('ls' in vline_property) else '--')
        vline_property['color'] = dcp(vline_property['color']) if('color' in vline_property) else (
                dcp(vline_property['color']) if('color' in vline_property) else colors[i])
        key_label = '[%s]'%float_format if(is_key_label) else ''
        assigned_label = dcp(vline_property['label']) if('label' in vline_property) else (
                dcp(vline_property['label']) if('label' in vline_property) else '')
        vline_property['label'] = None if(
                key_label=='' and assigned_label=='') else '%s%s'%(key_label, assigned_label)
        ax.axvline(x=value, **vline_property)
    return fig, ax

def imshowplot_(img, figsize=(), title='', file='', canvas=None, **kwags):
    kwags = clean_kwags(kwags)
    fig, ax = plt.subplots(figsize=figsize) if(
                canvas==None) else (canvas[0], canvas[1])
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    return plot_detail_process(
                fig, ax, kwags, 
                axtitle = title, title='', file=file, 
                dont_legend=1)
    
def histogramplot_(*arg, xlb='', ylb='', title='', axtitle='', file='',
                       figsize=(), xalue_lim=(), yalue_lim=(), 
                       c_alpha=1, mode='d', loc=0, edge_plank=(0,0,1,1),
                       canvas=None, showlog=0, subkeys={}, multi_mode='',
                       showmask=[], labels={}, colors='rainbow', vlines=[],
                       set_axis_range=1, 
                       **kwags):
    plot_started = [*plot_start(*arg, kwags)]
    np_args, label_package = plot_started[:-1], plot_started[-1]
    if(np.array(x[0]).shape!=() and np.array(y[0]).shape!=()):
        subkwags = kwags['subkwags'] if('subkwags' in kwags) else {}
        fig, ax = multigraph_process(X, Y, plot_method=curveplot_, subkeys=subkeys, 
                           labels=labels, colors=colors, c_alpha=c_alpha,
                           vlines = vlines, edge_plank=edge_plank,
                           showmask=showmask, mode=mode, loc=loc, 
                           figsize=figsize, kwags=kwags, subkwags=subkwags,
                           canvas=canvas)
    else:
        if(check_inputs_shapes(x, y)):
            return canvas
        kwags = clean_kwags(kwags)
        figsize = plot_set_figszie(X, Y, kwags, figsize)
        fig, ax = plt.subplots(figsize=figsize) if(
                    canvas==None) else (canvas[0], canvas[1]) #figsize=(5,3.5)
        printer('figure size=(%.2f,%.2f)'%(figsize[0], figsize[1]))
        if(set_axis_range):
            ax = plot_set_axis_range(X, Y, ax, xalue_lim, yalue_lim)[0]
        if(len(set(x))==1):
            xv = float(x[0])
            ax.axvline(x=xv, ls='--', **kwags)
        ax.plot(x, y, **kwags)
#    printer('label_package:%s'%label_package)
    return plot_detail_process(
                fig, ax, kwags, 
                axtitle = axtitle, title=title, file=file, 
                **label_package)
    
#curveplot_(X, Y, label='hi')
#curveplot_(X, Y, label='hi', **{
#        'xlabel':{'subject':'xlb', 'rotation':90}, 
#        'xtickslabel':{'subject':[150, 220], 'rotation':90}})
#curveplot_([X,Y], [Y], subkwags={'label':'hi'})
#curveplot_([X,Y], [Y], subkeys={(0,0):'a', (1,0):'b'})
#curveplot_([X,Y], [Y], subkeys=['a','b'])
#curveplot_([X,Y], [Y], labels=['a','b'])
#curveplot_([X,Y], [Y], subkeys=['a','b'], colors=[(1,0,0),(0,0,1)], c_alpha=0.2)
#curveplot_([X,Y], [Y], subkwags={'label':'hi', str((1,0)):{'label':'hello'}})
#curveplot_([X,Y], [Y], subkeys=['a','b'], xlabel={'subject':'xlb', 'rotation':90})
def curveplot_(X, Y, xlb='', ylb='', title='', axtitle='', file='',
                       figsize=(), xalue_lim=(), yalue_lim=(), 
                       c_alpha=1, darker_rate=1, mode='d', loc=0, edge_plank=(0,0,1,1),
                       canvas=None, showlog=0, subkeys={}, multi_mode='',
                       showmask=[], labels={}, colors='rainbow', vlines=[],
                       maxanno_digit='' , dotannos={}, annoyvalue_method=None, 
                       set_axis_range=1, **kwags):
    dotannos_ = dcp(dotannos)
    x, y, label_package = plot_start(X, Y, kwags=kwags)
    if(np.array(x[0]).shape!=() and np.array(y[0]).shape!=()):
        subkwags = kwags['subkwags'] if('subkwags' in kwags) else {}
        fig, ax = multigraph_process(X, Y, plot_method=curveplot_, subkeys=subkeys, 
                           labels=labels, colors=colors, maxanno_digit=maxanno_digit,
                           c_alpha=c_alpha, darker_rate=darker_rate,
                           vlines = vlines, edge_plank=edge_plank,
                           showmask=showmask, mode=mode, loc=loc,
                           kwags=kwags, subkwags=subkwags,
                           canvas=canvas)
    else:
        printer('[curveplot_...]', level=2)
        if(check_inputs_shapes(x, y)):
            return canvas
        if(maxanno_digit.count('%')==2 if(type(maxanno_digit)==str) else False):
            #Ex: maxanno_digit = '%.2f, %.2f'
            mx, my = np.array(x)[np.argsort(y)[-1]], np.max(y)
            printer(('[curveplot_] max value dot:%s)'%maxanno_digit)%(mx, my))
            dotannos_.update({('%s'%(maxanno_digit))%(mx, my):(mx, my)})
        kwags = dcp(clean_kwags(kwags))
        figsize = plot_set_figszie(X, Y, kwags, figsize)
        fig, ax = plt.subplots(figsize=figsize) if(
                    canvas==None) else (canvas[0], canvas[1]) #figsize=(5,3.5)
        if(canvas==None):
            fig.subplots_adjust(*edge_plank)
        printer('figure size=(%.2f,%.2f)'%(figsize[0], figsize[1]))
        if(set_axis_range):
            ax = plot_set_axis_range(ax, X, Y, xalue_lim, yalue_lim)[0]
        if(len(set(x))==1):
            xv = float(x[0])
            ax.axvline(x=xv, ls='--', **kwags)
        ax.plot(x, y, **kwags)
#    printer('label_package:%s'%label_package)
    return plot_detail_process(
                fig, ax, kwags, 
                axtitle = axtitle, title=title, file=file, dotannos=dotannos_,
                **label_package)

def curveplot(X, Y, colors={}, cmap='coolwarm', title='', labels='', twin_axs={},
              xlb='', xtkrot=0, ylb='', colorlb='', vlines=[], xticks={}, dotannos={},
              array_start_end=(), file='', maxmin=(), figsize=(), rgn='', 
              rgndeg=1, showlog=0, rgnlinedensity=500, multiline=True, 
              showmask=None, canvas=None):
    X = {i:X[i] for i in (range(len(X)) if type(X)!=dict else X.keys())}
    Y = {i:Y[i] for i in (range(len(Y)) if type(Y)!=dict else Y.keys())}
    if(multiline==True):
        n = len(set('_'.join(['_'.join(['%.4f'%x if(
                type(x)!=str) else x for x in X[k]]) for k in X.keys()]).split('_')))
        m = len(set('_'.join(['_'.join(['%.4f'%y for y in Y[k]]) for k in Y.keys()]).split('_')))
        friendly_height = (2 + min(m*0.5,4) + 7*np.sin(xtkrot))
        figsz = (min(n*0.2,6)+4, friendly_height) if(figsize==()) else tuple(figsize)
        printer('figure size=(%.2f,%.2f)'%(figsz[0], figsz[1]))
        fig = plt.figure(figsize=figsz) if(canvas==None) else canvas[0]
        ax = fig.add_subplot() if(canvas==None) else canvas[1]
        if(twin_axs!={}):
            ax2 = ax.twinx()
    index=0            
    for xk in X.keys():
        for yk in Y.keys():
            x = np.array(X[xk])[array_start_end[0]:array_start_end[1]] if(
                    np.array(array_start_end).shape==(2,)) else np.array(X[xk])
            y = np.array(Y[yk])[array_start_end[0]:array_start_end[1]] if(
                    np.array(array_start_end).shape==(2,)) else np.array(Y[yk])
            execute = ((len(x)==len(y)) and (
                    ((xk,yk) in showmask) if(showmask!=None) else True))
#            {}['%s'%showmask]
            if(execute):
                if(showlog>0):
                    printer('[(%s,%s)] drawing....'%(xk,yk))
                    showlog -= 1
                lb = (labels[(xk,yk)] if((xk,yk) in labels.keys()) else (
                        None)) if(type(labels)==dict) else ((labels[index]
                        if(index<np.array(labels).shape[0]) else '')
                        if(len(np.array(labels).shape)>0) else str(labels))
                if(multiline==False):
                    n, m = len(x), len(set(list(y)))
                    friendly_height = (2 + min(m*0.5,4) + 7*np.sin(xtkrot))
    
                    figsz = (min(n*0.2,6)+4, friendly_height) if(
                            figsize==()) else tuple(figsize)
                    printer('[%s]figure size=(%.2f,%.2f)'%(str((xk,yk)), figsz[0], figsz[1]))
                    fig = plt.figure(figsize=figsz)
                    ax = fig.add_subplot()
                if(len(maxmin)>=2):
                    mx = np.array(y).max()
                    mn = np.array(y).min()
                    d = len(set(list(np.array(y).flatten())))*(2/3)
                    h=(mx-mn)/d
                    ax.set_ylim(maxmin[0]-h, maxmin[1]+h)
                    printer('[%s]邊界空隙:%.2f'%(str((xk,yk)), h))
                    if(len(maxmin)==4):
                        mx = np.array(x).max()
                        mn = np.array(x).min()
                        d = len(set(list(np.array(x).flatten())))*(2/3)
                        h=(mx-mn)/d
                        ax.set_xlim(maxmin[2]-h, maxmin[3]+h)
                plt.title(title)
                if(type(vlines)==dict):
                    for vL in vlines:
                        for vl in vlines[vL]['values']:
                            plt.axvline(x = vl, 
                                        ls = '--' if(
                                                'ls' not in vlines[vL].keys(
                                                        )) else vlines[vL]['ls'], 
                                        color = (0,0,0,0.3) if(
                                                'color' not in vlines[vL].keys(
                                                        )) else vlines[vL]['color'])
                else:
                    vL = list(vlines)
                    if(vL!=[]):
                        for vl in vL:
                            plt.axvline(x=vl, ls='--', color=(0/255,0/255,0/255, 0.3))
                if(not xlb==''):
                    plt.xlabel(xlb)
                if(not xlb==''):
                    plt.ylabel(ylb)
                if((xk,yk) in twin_axs.keys()):
                    if('xlim' in twin_axs[(xk,yk)].keys()):
                        ax2.set_xlim(twin_axs[(xk,yk)]['xlim'][0],twin_axs[(xk,yk)]['xlim'][1])
                    if('ylim' in twin_axs[(xk,yk)].keys()):
                        ax2.set_ylim(twin_axs[(xk,yk)]['ylim'][0],twin_axs[(xk,yk)]['ylim'][1])
                    ax2.plot(x, y, label=lb, color = None if(
                            type(colors)!=dict) else (
                                    None if((xk,yk) not in colors.keys()) else(
                                            colors[(xk,yk)])))
                else:
                    ax.plot(x, y, label=lb, color = None if(
                            type(colors)!=dict) else (
                                    None if((xk,yk) not in colors.keys()) else(
                                            colors[(xk,yk)])))
                if(multiline==False):
                    if(xticks!={}):
                        xtk_val = range(1,len(x)+1)
                        if((xk,yk) in xticks.keys()):
                            xtk_val = xticks[(xk,yk)]['values'] if(
                                    'values' in xticks[(xk,yk)]) else xtk_val
                        plt.xticks(x, xtk_val, rotation=xtkrot)
                    else:
                        plt.xticks(rotation=xtkrot)
                    if(not lb=='' or not rgn==''):
                        plt.legend()
                    if(type(file)==dict):
                        if((xk,yk) in file.keys()):
                            plt.savefig(file[(xk,yk)])
                index+=1
    if(dotannos):
        for dotanno in dotannos:
            plt.annotate(dotanno, dotannos[dotanno])
    if(multiline==True):
        if(xticks!={}):
            xgap = xticks['gaps']
            xtk_val = range(1,len(xgap)+1)
            if('values' in xticks.keys()):
                xtk_val = xticks['values'] if(
                        'values' in xticks.keys()) else xtk_val
            plt.xticks(xgap, xtk_val, rotation=xtkrot)
        else:
            plt.xticks(rotation=xtkrot)
        if(not labels=='' or not rgn==''):
            plt.legend()
    if(file!=''):
        plt.savefig(file)
    if(canvas==None):
        plt.ioff()
        plt.show()
    else:
        return fig, ax

def set_colors(*args, colors, mode, c_alpha=None, on=True, **kwags):
    if(not on):
        return None
    if(colors.find('rainbow')>-1):
        colors = [v for v in cm_rainbar(
                _n=(max([np.array(v).shape[0] for v in args]) if(
                        mode in ['d','x','y']) else np.prod(
                                [np.array(v).shape[0] for v in args])))]
    if('darker_rate' in kwags):
        darker_rate = float(kwags['darker_rate'])
        colors = [tuple(np.array(v)/darker_rate) for v in colors]
    if(c_alpha!=None):
        c_alpha = max(min(c_alpha, 1), 0)
        colors = [(*tuple(v)[:3], c_alpha) for v in colors]
    return colors

def multigraph_process_XYcook(X, Y, plot_method, fig, ax, loc=0, showmask=[], 
                              subkeys={}, subkwags={}, mode='d',
                              labels={}, colors={}, maxanno_digit='',
                              label_callback_key='label', 
                              color_callback_key='color'):
    printer('[multigraph_process_XYcook...]')
    index=0
    for i, xx in enumerate(X):
        for j, yy in enumerate(Y):
            if(plot_condition_judge(
                    np.array(xx).shape, np.array(yy).shape, i, j, 
                    loc=loc, mode=mode, showmask=showmask)):
                continue
            if(type(subkeys)==dict):
                key = dcp(subkeys[(i,j)] if(
                        (i,j) in subkeys) else str((i,j)))
            if(str(type(subkeys)).find('list')>-1 or 
               str(type(subkeys)).find('ndarray')>-1):
                key = dcp(subkeys[index] if(
                        index<np.array(subkeys).shape[0]) else str((i,j)))
            printer('[multigraph_process_XYcook...] key:%s'%key)
            subkwags_key = dcp((subkwags[key] if(
                                key in subkwags) else subkwags))
            label, color = (subkwags_key['label'] if(
                    'label' in subkwags_key) else '',
                    subkwags_key['color'] if(
                    'color' in subkwags_key) else None)
            if(type(labels)==dict):
                label = dcp('[%s]%s'%(key, labels[key]) if(
                        key in labels) else '[%s]%s'%(key, label))
            if(str(type(labels)).find('list')>-1 or 
               str(type(labels)).find('ndarray')>-1):
                label = dcp('[%s]%s'%(key, labels[index]) if(
                        index<np.array(labels).shape[0]) else '[%s]%s'%(key, label))
#            printer('2:\n%s'%(str(colors)[:200]))
            if(type(colors)==dict):
                color = tuple(dcp(colors[key] if(
                        key in colors) else color))
            if(str(type(colors)).find('list')>-1 or 
               str(type(colors)).find('ndarray')>-1):
                color = tuple(dcp(colors[index] if(
                        index<np.array(colors).shape[0]) else color))
            subkwags_key.update({label_callback_key:label, 
                                 color_callback_key:color,
                                 'maxanno_digit':maxanno_digit})
#            printer('%s\n%s'%(key, str(subkwags_key)))
            fig, ax = plot_method(
                    xx, yy, set_axis_range=0, canvas=(fig, ax), **subkwags_key)
#            printer(str(subkwags))
            index+=1
    return fig, ax

def multigraph_process_seqcook(plot_method, fig, ax, *args, showmask=[], 
                              subkeys={}, subkwags={}, 
                              labels={}, colors={}, maxanno_digit='',
                              label_callback_key='label', 
                              color_callback_key='color'):
    printer('[multigraph_process_seqcook...]')
    index=0
    for i, tt in enumerate(args):
        if(plot_condition_judge(1, 1, 1, showmask=showmask, key=i)):
            continue
        if(type(subkeys)==dict):
            key = dcp(subkeys[i] if(
                    i in subkeys) else str(i))
        if(str(type(subkeys)).find('list')>-1 or 
           str(type(subkeys)).find('ndarray')>-1):
            key = dcp(subkeys[index] if(
                    index<np.array(subkeys).shape[0]) else str(i))
        subkwags_key = dcp((subkwags[key] if(
                            key in subkwags) else subkwags))
        label, color = (subkwags_key['label'] if(
                'label' in subkwags_key) else '',
                subkwags_key['color'] if(
                'color' in subkwags_key) else None)
        if(type(labels)==dict):
            label = dcp('[%s]%s'%(key, labels[key]) if(
                    key in labels) else '[%s]%s'%(key, label))
        if(str(type(labels)).find('list')>-1 or 
           str(type(labels)).find('ndarray')>-1):
            label = dcp('[%s]%s'%(key, labels[index]) if(
                    index<np.array(labels).shape[0]) else '[%s]%s'%(key, label))
#            printer('2:\n%s'%(str(colors)[:200]))
        if(type(colors)==dict):
            color = tuple(dcp(colors[key] if(
                    key in colors) else color))
        if(str(type(colors)).find('list')>-1 or 
           str(type(colors)).find('ndarray')>-1):
            color = tuple(dcp(colors[index] if(
                    index<np.array(colors).shape[0]) else color))
        subkwags_key.update({label_callback_key:label, 
                             color_callback_key:color})
#            printer('%s\n%s'%(key, str(subkwags_key)))
        fig, ax = plot_method(
                *args, set_axis_range=0, canvas=(fig, ax), **subkwags_key)
#            printer(str(subkwags))
        index+=1
    return fig, ax

def multigraph_process(*args, plot_method=None, subkeys={},
                       labels=[], colors='rainbow', c_alpha=1, darker_rate=1,
                       vlines=[], edge_plank=(0,0,1,1),
                       maxanno_digit='',
                       showmask=[], mode='d', loc=0, 
                       xalue_lim=(), yalue_lim=(),
                       figsize=(), kwags={}, subkwags={},
                       canvas=None,
                       label_callback_key='label', 
                       color_callback_key='color',
                       plot_set_figszie_on=True,
                       plot_set_axis_range_on=True,
                       table_type='xycook'):
    printer('[multigraph_process...]')
    figsize = plot_set_figszie(*args, kwags, figsize, on=plot_set_figszie_on)
    fig, ax = plt.subplots(figsize=figsize) if(
                canvas==None) else (canvas[0], canvas[1]) #figsize=(5,3.5)
    if(canvas==None):
        printer('[multigraph_process] edge_plank:%s'%str(edge_plank))
        fig.subplots_adjust(*edge_plank)
    ax, mxn = plot_set_axis_range(ax, *tuple(args[:2] if(
                                    table_type=='xycook') else [[], []]),
                                  xalue_lim, yalue_lim, vlines, 
                                  on=(plot_set_axis_range_on and table_type=='xycook'))
    printer(ax.get_xlim())
    printer(ax.get_ylim())
    colors = set_colors(*args, colors=colors, mode=mode, 
                        c_alpha=c_alpha, darker_rate=darker_rate)
#    printer('1:\n%s'%(str(colors)[:200]))
    if(table_type=='xycook'):
        fig, ax = multigraph_process_XYcook(*(args[:2]), plot_method, fig, ax,
                                        loc=loc, showmask=showmask, mode=mode,
                                        subkeys=subkeys, subkwags=subkwags, 
                                        labels=labels, colors=colors, 
                                        maxanno_digit=maxanno_digit,
                                        label_callback_key=label_callback_key, 
                                        color_callback_key=color_callback_key)
    elif(table_type=='seqcook'):
        fig, ax = multigraph_process_XYcook(*args, plot_method, fig, ax,
                                        showmask=showmask, 
                                        subkeys=subkeys, subkwags=subkwags, 
                                        labels=labels, colors=colors, 
                                        maxanno_digit=maxanno_digit,
                                        label_callback_key=label_callback_key, 
                                        color_callback_key=color_callback_key)
    if(vlines):
        (fig, ax) = vlinesplot_(vlines, canvas=(fig, ax))
    return fig, ax

def cm_rainbar(_n=5, _a=0, _b=1, _i=None, c_alpha=1):
    if(_i==None):
        ret = cm.rainbow(np.linspace(_a, _b, _n))
        ret[:,3] = np.array([c_alpha]*_n)
        return ret
    elif(_i>=0 and _i<_n):
        return (*tuple(cm.rainbow(np.linspace(_a, _b, _n)[_i][:3])), c_alpha)

def clean_kwags(kwags):
    ik=0
    while(ik<len(kwags)):
        k = tuple(kwags.keys())[ik]
        if(str(k).find('(')>-1 or str(k).find(')')>-1):
            kwags.pop(k)
        ik+=1
    return kwags

#scatterplot_(X, Y, label='hi')
#scatterplot_(X, Y, label='hi', **{
#        'xlabel':{'subject':'xlb', 'rotation':90}, 
#        'xtickslabel':{'subject':[150, 220], 'rotation':90}})
#scatterplot_([X,Y], [Y], subkwags={'label':'hi'})
#scatterplot_([X,Y], [Y], subkeys={(0,0):'a', (1,0):'b'})
#scatterplot_([X,Y], [Y], subkeys=['a','b'])
#scatterplot_([X,Y], [Y], labels=['a','b'])
#scatterplot_([X,Y], [Y], colors=[(1,0,0),(0,0,1)], c_alpha=0.2)
#scatterplot_([X,Y], [Y], subkwags={'label':'hi', str((1,0)):{'label':'hello'}})
#scatterplot_([X,Y], [Y], subkeys=['a','b'], xlabel={'subject':'xlb', 'rotation':90})
def scatterplot_(X, Y, xlb='', ylb='', title='', axtitle='', file='', 
                       figsize=(), xalue_lim=(), yalue_lim=(), 
                       c_alpha=1, darker_rate=1, mode='d', loc=0,
                       canvas=None, showlog=0, subkeys={}, multi_mode='',
                       showmask=[], labels={}, colors='rainbow', set_axis_range=1, 
                       edge_plank=(0,0,1,1),
                       **kwags):
    x, y, label_package = plot_start(X, Y, kwags=kwags)
    if(np.array(x[0]).shape!=() and np.array(y[0]).shape!=()):
        subkwags = kwags['subkwags'] if('subkwags' in kwags) else {}
        fig, ax = multigraph_process(X, Y, plot_method=scatterplot_, subkeys=subkeys, 
                           labels=labels, colors=colors, 
                           c_alpha=c_alpha, darker_rate=darker_rate, 
                           showmask=showmask, mode=mode, loc=loc, edge_plank=edge_plank,
                           figsize=figsize, kwags=kwags, subkwags=subkwags,
                           canvas=canvas)
    else:
        if(check_inputs_shapes(x, y)):
            return canvas
        kwags = clean_kwags(kwags)
        figsize = plot_set_figszie(X, Y, kwags, figsize)
        fig, ax = plt.subplots(figsize=figsize) if(
                    canvas==None) else (canvas[0], canvas[1]) #figsize=(5,3.5)
        printer('figure size=(%.2f,%.2f)'%(figsize[0], figsize[1]))
        if(set_axis_range):
            ax = plot_set_axis_range(ax, X, Y, xalue_lim, yalue_lim)[0]
        ax.scatter(x, y, **kwags)
#    printer('label_package:%s'%label_package)
    return plot_detail_process(
                fig, ax, kwags, file=file, axtitle=axtitle, title=title,
                **label_package)

def scatterplot(X, Y, colors=[], cmap='coolwarm', title='', labels='', figsize=(),
                xlb='', xtkrot=0, ylb='', colorlb='', vlines=[], file='', maxmin=(), 
                rgn='', rgndeg=1, rgnlinedensity=500, ctr_tol=0, canvas=None, frame_axis='',
                **kwags):
#    printer('\nxlb:%s\nylb:%s'%(xlb, ylb))
    rgn_done = False
    X = {i:X[i] for i in (range(len(X)) if type(X)!=dict else X.keys())}
    Y = {i:Y[i] for i in (range(len(Y)) if type(Y)!=dict else Y.keys())}
    n = len(set('_'.join(['_'.join(['%.4f'%x if(
            type(x)!=str) else x for x in X[k]]) for k in X.keys()]).split('_')))
    m = len(set('_'.join(['_'.join(['%.4f'%y for y in Y[k]]) for k in Y.keys()]).split('_')))
    friendly_height = (2 + min(m*0.5,4) + 7*np.sin(xtkrot))
    figsz = (min(n*0.2,6)+4.5, friendly_height) if(figsize==()) else tuple(figsize)
    printer('figure size=(%.2f,%.2f)'%(figsz[0], figsz[1]))
    if(len(maxmin)>=2):
        fig = plt.figure(figsize=figsz) if(canvas==None) else canvas[0]
        ax = fig.add_subplot() if(canvas==None) else canvas[1]
        ymx = np.matrix(Y).max()
        ymn = np.matrix(Y).min()
        h=(ymx-ymn)/10
        ax.set_ylim(maxmin[0]-h, maxmin[1]+h)
    else:
        fig = plt.figure(figsize=figsz) if(canvas==None) else canvas[0]
        ax = fig.add_subplot() if(canvas==None) else canvas[1]
    m = [v for v in X.values()]+[v for v in Y.values()] if(frame_axis==''
         ) else ([v for v in X.values()] if(frame_axis=='x'
                ) else [v for v in Y.values()])
    m = [v for w in m for v in w]
    printer('all data shape:%s'%str(np.asarray(m).shape))
    printer('all data like:%s'%(str(np.array(m))[:100]))
    mmax, mmin = np.amax(m), np.amin(m)
    printer('max:%s'%(str(np.amax(m).round(2))[:100]))
    printer('min:%s'%(str(np.amin(m).round(2))[:100]))
    h=(mmax - mmin)/10
    if(str(type(ctr_tol)).find('function')>-1):
        dx = np.linspace(mmin-h, mmax+h, 500)
        dy = np.linspace(mmin-h, mmax+h, 500)
        dx, dy = np.meshgrid(dx, dy)
        z = ctr_tol(dx.flatten(), dy.flatten()) + 0
        z = z.reshape(dx.shape)
        plt.contourf(dx, dy, z, alpha=0.2)
        t_ = np.linspace(mmin-h, mmax+h, 500)
        plt.plot(t_, t_, '--',color=(255/255,90/255,90/255))
    else:
        if(ctr_tol>0):
            t_ = np.linspace(mmin-h, mmax+h, 500)
            plt.plot(t_, t_, '--',color=(255/255,90/255,90/255))
            plt.plot(t_, [x - ctr_tol for x in t_], '--',color=(255/255,90/255,90/255))
            plt.plot(t_, [x + ctr_tol for x in t_], '--',color=(255/255,90/255,90/255))
    for ix in X.keys():
        for iy in Y.keys():
            x = np.array(X[ix])
            y = np.array(Y[iy])
            if(x.shape!=y.shape or x.shape[0]==0 or y.shape[0]==0):
                continue
            curx_maxmin = (maxmin[2], maxmin[3]) if(len(maxmin)>=4) else(
                    (x.min(), x.max()))
            if(not title==''):
                plt.title(title)
            lb = (labels[(ix,iy)] if((ix,iy) in labels.keys()) else (
                    'key:%s'%str((ix,iy)))) if(type(labels)==dict) else str(labels)
            if(type(colors)==dict):
                c=(colors[(ix,iy)] if((ix,iy) in colors.keys()) else None
                                        if(type(colors)==dict) else str(colors))
                plt.scatter(x ,y , c=c, label=lb)
            elif(type(colors)==list and colors!=[]):
                plt.scatter(x ,y , c=colors, cmap=cmap, label=lb)
                plt.colorbar().set_label(colorlb, rotation=270)
            else:
                plt.scatter(x ,y, label=lb)
            if(not rgn==''):
                m, b = np.polyfit(x, y, rgndeg)
                mx_r = np.cov(np.array(x),np.array(y))
                corr = mx_r[0,1]/np.sqrt(mx_r[0,0]*mx_r[1,1])
                plt.plot(x, m*x + b, rgn, label='rgn line m:%.4f, b:%.4f \n corr:%.2f'%(m,b,corr))
                rgn_done = True
            if(type(vlines)==dict):
                for vL in vlines:
                    for vl in vlines[vL]['values']:
                        plt.axvline(x = vl, 
                                    ls = '--' if(
                            'ls' not in vlines[vL].keys()) else vlines[vL]['ls'], 
                                    color = (0,0,0,0.3) if(
                            'color' not in vlines[vL].keys()) else vlines[vL]['color'])
            else:
                vL = list(vlines)
                if(vL!=[]):
                    for vl in vL:
                        plt.axvline(x=vl, ls='--', color=(0/255,0/255,0/255, 0.3))
            if(not xlb==''):
                plt.xlabel(xlb)
            if(not ylb==''):
                plt.ylabel(ylb)    
            if(not labels=='' or not rgn==''):
                plt.legend()
    if(not file==''):
        plt.savefig(file)
    if(rgn_done):
        if(canvas==None):
            plt.ioff()
            plt.show()
            return m, b, corr
        else:
            return (fig, ax), (m, b, corr)
    else:
        if(canvas==None):
            plt.ioff()
            plt.show()
        else:
            return fig, ax

def scatterplot_with_axis_labels(xheader, yheader):
    return (lambda **kwags:scatterplot(
            xlb=xheader, ylb=yheader, **kwags))

#def data_norms_method(pdt, fct, tol):
#    normed = norms(pdt, fct, tol=tol)
#    return '\nOKR:%.2f\nrmse:%.2f'%(normed['OKR'], normed['rmse'])
#
#xheader = 'gap1L_goal_fct'
#yheader = 'gap1L_goal_pdt'
#mask_reference_column_name = 'vscst'
#title = 'test data'
#file = os.path.join(exp_fd, 'test_data_%s_%s.jpg')%(xheader, yheader)
#colors = [(0,0.05,0.4,0.2), (0.8,0.6,0.2,0.2), (0,0.7,0.1,0.2)]
#plot_by_class_in_data(data, xheader, yheader, 
#                      mask_reference_column_name = 'vscst',
#                      title=title, file=file, colors=colors,
#                      plot_method=(lambda **kwags:scatterplot(
#                              xlb=xheader, ylb=yheader, **kwags)),
#                      ctr_tol=main_tol, figsize=(8,4), 
#                      data_norms_method = data_norms_method,
#                      edge_plank=(0.9,0.15,0,1))
def plot_by_class_in_data(data, xheader, yheader, mask_reference_column_name='',
                          title='', file='', labels='', colors=[], 
                          plot_method = scatterplot, data_norms_method=None,
                          ctr_tol=0, figsize=(), reference_mask=[],
                          mask_reference_column_names=[], refer_method=None,
                          canvas=None, **kwags):
    mask_reference_column = (data[mask_reference_column_name] if(
                mask_reference_column_name!='') else data.index) if(
                np.array(reference_mask).shape[0]!=np.array(
                        data.index).shape[0]) else np.array(reference_mask)
#    {}['%d||%d'%(np.array(reference_mask).shape[0], 
#                 np.array(data.index)[0])]
    set_refer = list(set(mask_reference_column))
    printer('set refer:%s'%(str(set_refer)[:200]))
    gd = GRAPHDATA()
    cm_rainbow = cm.rainbow(np.linspace(0, 1, len(set_refer))) if(not 
            'c_alpha' in kwags) else [
                (*tuple(v)[:3], kwags['c_alpha']) for v in cm.rainbow(np.linspace(
                        0, 1, len(set_refer)))]
    printer(cm_rainbow)
    for index, refer in enumerate(set_refer):
        data_refer = data[refer_method(refer, mask_reference_column)] if(
                refer_method!=None) else data[mask_reference_column==refer]
        norms_stg = ''
        if(data_norms_method):
            norms_stg = data_norms_method(list(data_refer[yheader]), 
                                           list(data_refer[xheader]),
                                           tol=ctr_tol)
        xx = [v for v in data_refer[xheader]]
        yy = [v for v in data_refer[yheader]]
        gd.add_data([xx, yy], key=refer)
        labels_refer = dcp(('%s%s'%(str(labels[refer]), norms_stg) if(refer in labels) else '') if(
                type(labels)==dict) else ('%s%s'%(str(labels[index]), norms_stg) if(
                        index<len(labels)) else '') if(
                        type(labels)==list) else '%s%s'%(str(labels), norms_stg))
        gd.set_properties('labels', value=labels_refer, key=refer)
        colors_refer = dcp({(i,i):cm_rainbow[i] for i in range(
                len(cm_rainbow))} if(colors==[]) else (
                ({(0,0):colors[refer]} if(refer in colors) else ()) if(
                type(colors)==dict) else (({(0,0):colors[index]} if(
                        index<len(colors)) else ()) if(
                        type(colors)==list) else {(0,0):colors})))
        printer(colors)
        {}['%s'%str(colors_refer)]
        gd.set_properties('colors', value=colors_refer, key=refer)
    return gd.classesplot_inone(plot_method = plot_method, 
                             ctr_tol = ctr_tol, figsize=figsize, 
                             title=title, file=file, canvas=canvas, 
                             data_norms_method=data_norms_method,
                             **kwags)

#plots_columns=['gap1L_goal', 'gap1R_goal',
#            'gap2L_goal', 'gap2R_goal',
#            'gap3L_goal', 'gap3R_goal']
#kwags_common_in_grid = {'mask_reference_column_name':'vscst',
#                      'reference_mask':reference_mask,
#                      'data_norms_method':data_norms_method,
#                      'colors':colors, 'ctr_tol':main_tol, 
#                      'figsize':(8,4),
#                      'edge_plank':(0.9,0.15,0,1)}
#vs.plot_regression_data_analysis(data, plots_columns, title=title, file='',
#                              figsize = (13,16),
#                              kwags_common_in_grid=kwags_common_in_grid)
def plot_regression_data_analysis(data, plots_columns, 
                                  common_xheader='_fct', 
                                  common_yheader='_pdt', 
                                  title='', file='', figsize=(),
                                  kwags_common_in_grid={}, 
                                  kwags_in_grid_by_key={}, **kwags):
    plots = {dcp(k):{} for k in plots_columns}
    for k in plots:
        xheader, yheader = dcp('%s%s'%(k,common_xheader)), dcp('%s%s'%(k, common_yheader))
        plots[k]['method'] = plot_by_class_in_data
        plots[k]['inputs'] = dcp((data, xheader, yheader))
        plot_method = dcp(scatterplot_with_axis_labels(xheader, yheader))
        plots[k]['params'] = dcp({'plot_method':plot_method})
        plots[k]['params'].update(kwags_common_in_grid)
        if(k in kwags_in_grid_by_key):
            plots[k]['params'].update(kwags_in_grid_by_key[k])
    gridplot(plots, title=title, file=file, figsize=figsize, **kwags)

class GRAPHDATA():
    def __init__(self, plot_method=None):
        self.X = {}
        self.Y = {}
        self.P = {}
        self.plot_method = plot_method
        self.A = {0:self.X, 1:self.Y, 'P':self.P}
    
    def show_X(self):
        for k,v in self.X.items():
            print('%s:%s'%(k,str(v)[:200]))
            
    def show_Y(self):
        for k,v in self.Y.items():
            print('%s:%s'%(k,str(v)[:200]))
            
    def show_P(self):
        for k,v in self.P.items():
            print('%s:%s'%(k,str(v)[:200]))
    
    def rename_frame(self, old_axis_name, new_axis_name):
        self.A[new_axis_name] = self.A.pop(old_axis_name)
    
    def set_frame(self, axis=-1, key=None, new_frame=[]):
        if(not axis in self.A.keys()):
            for a in self.A.keys():
                self.set_frame(axis=a, key=key, new_frame=new_frame)
        else:
            table = self.A[axis]
#            printer('  [0][axis:%s]table=%s'%(axis, str(table)[:50]))
#            printer('  new_frame:%s'%(str(new_frame)[:50]))
            key = key if(key!=None) else len(table)
            table[key] = dcp(new_frame)
#            printer('  [1][axis:%s]table=%s'%(axis, str(table)[:50]))
            
    def add_data(self, array, axis=-1, key=None, new_frame=[], showlog=12):
        if(not axis in self.A.keys() and np.array(array).shape[0]>=2):
            for a in self.A.keys():
                if(a!='P'):
                    showlog = self.add_data(
                        array[a], axis=a, key=key, new_frame=new_frame, 
                        showlog = showlog)
            showlog = self.add_data(
                    None, axis='P', key=key, new_frame=new_frame, showlog=showlog)
        else:
            if(axis!='P'):
                table = self.A[axis]
#                printer('[0][axis:%s]X=%s'%(axis, str(self.X)[:50]))
#                printer('[0][axis:%s]table=%s'%(axis, str(table)[:50]))
                key = key if(key!=None) else len(table)
#                printer('[a][axis:%s]table=%s'%(axis, str(table)[:50]))
                if(not key in table.keys()):
                    self.set_frame(axis = axis, key = key, new_frame=new_frame)
#                printer('[1][axis:%s]X=%s'%(axis, str(self.X)[:50]))
#                printer('[1][axis:%s]table=%s'%(axis, str(table)[:50]))
                if(type(table[key])==list):
                    table[key].append(array)
                elif(type(table[key])==dict):
                    table[key].update(array)
                else:
                    {}['key[%s] table type error:%s'%(str(key), str(type(table[key])))]
#                printer('[2][axis:%s]X=%s'%(axis, str(self.X)[:50]))
#                printer('[2][axis:%s]table=%s'%(axis, str(table)[:50]))
            else:
                table = self.P
                key = key if(key!=None) else len(table)
                table[key] = {}
            if(len(tuple(table[key]))%500==1 and showlog>0):
                printer('[%d]axis:%s, key:%s, data:%s'%(
                        showlog, axis, str(key), str(table[key])[:50] + (
                                '...' if(len(str(table[key]))>=50) else '')))
                showlog -= 1
        return showlog
    def clean_data(self, axis=-1, key=None):
            if(not axis in self.A.keys()):
                for a in self.A.keys():
                    self.clean_data(axis=a, key=key)
            else:
                if(key==None):
                    keys = (list(self.X.keys()) if(axis==0) else list(self.Y.keys())) if(
                            axis!='P') else list(self.P.keys())
#                    printer('keys:%s...'%(('%s,'*5)%(tuple(keys)[:5])+'...'))
                    for tb_key in keys:
                        self.clean_data(axis=axis, key=tb_key)
                else:
                    table = self.A[axis]
                    table.pop(key)
    
    def show_data(self, axis=-1, key=None, top_num=3, return_information=False):
        if(not axis in self.A.keys()):
                for a in self.A.keys():
                    self.show_data(axis=a, key=key, 
                                   top_num=top_num, 
                                   return_information=return_information)
        else:
            table = self.A[axis]
            if(key==None):
                for tb_key in [v for v in table][:top_num]:
                    self.show_data(axis=axis, key=tb_key, 
                                   top_num=top_num, 
                                   return_information=return_information)
                if(len(table)>top_num):
                    printer('...')
                    printer('[axis:%s]table size:%d'%(axis, len(table)))
                    printer('-----------------------------')
            else:
                if(not key in table.keys()):
                    printer('[axis:%s]there is no such key:%s'%(axis, key))
                    return
                data = table[key]
                printer('[key:%s]data shape:%s'%(
                        key, str(np.array(tuple(data)).shape)))
    
    def set_properties(self, pn, value, key=None):
        if(key==None):
            for k in self.X:
                self.set_properties(pn=pn, value=value, key=k)
        else:
            if(not key in self.P.keys()):
                self.P[key] = {}
            self.P[key][pn] = dcp(value)
    
    def classesplot_inone(self, x_axis=0, y_axis=1, layout=(), ctr_tol=0,
             file='', title='', titlefontsize = 20, plot_method=None, 
             is_key_label=True, figsize=(), edge_plank=(0,0,1,1), 
             showmask=None, canvas=None, showlog=0, **kwags):
        xtable = self.A[x_axis] #x_axis=0, xtable = gd.A[x_axis]
        ytable = self.A[y_axis] #y_axis=1, ytable = gd.A[y_axis]
        colsize = np.sqrt(len(xtable))//1 if(
                layout==()) else layout[1]
        rowsize = (len(xtable)//colsize)+1 if(
                layout==()) else layout[0]
        figsize = (colsize*5, rowsize*3.5) if(
                figsize==()) else figsize
        printer(len(xtable), colsize, rowsize)
        fig, ax = plt.subplots(figsize=figsize) if(
                    canvas==None) else (canvas[0], canvas[1]) #figsize=(5,3.5)
        fig.subplots_adjust(*edge_plank)
        all_values = [v for k in xtable.keys() for ls in xtable[k] for v in ls] + [
                      v for k in ytable.keys() for ls in ytable[k] for v in ls]
        mmax, mmin = np.amax(all_values), np.amin(all_values)
        printer('max:%s'%(str(np.amax(all_values).round(2))[:100]))
        printer('min:%s'%(str(np.amin(all_values).round(2))[:100]))
        h=(mmax - mmin)/10
        if(str(type(ctr_tol)).find('function')>-1):
            dx = np.linspace(mmin-h, mmax+h, 500)
            dy = np.linspace(mmin-h, mmax+h, 500)
            dx, dy = np.meshgrid(dx, dy)
            z = ctr_tol(dx.flatten(), dy.flatten()) + 0
            z = z.reshape(dx.shape)
            plt.contourf(dx, dy, z, alpha=0.2)
            t_ = np.linspace(mmin-h, mmax+h, 500)
            plt.plot(t_, t_, '--',color=(255/255,90/255,90/255))
        else:
            if(ctr_tol>0):
                t_ = np.linspace(mmin-h, mmax+h, 500)
                plt.plot(t_, t_, '--',color=(255/255,90/255,90/255))
                plt.plot(t_, [x - ctr_tol for x in t_], '--',color=(255/255,90/255,90/255))
                plt.plot(t_, [x + ctr_tol for x in t_], '--',color=(255/255,90/255,90/255))
        for index, key in enumerate(self.X):
            if(showmask!=None and not key in showmask):
                continue
            if(showlog>0):
                printer('[%d][%s]drawing....'%(index, str(key)))
            method = lambda canvas, **kwags: (canvas[0], canvas[1])
            if(plot_method!=None):
                method = dcp(plot_method)
                printer('\nplot method!!\n')
            else:
                if(key in self.P):
                    if('method' in self.P[key]):
                        method = self.P[key]['method']
                else:
                    if(self.plot_methd!=None):
                        method = dcp(self.plot_methd)
            method_inputs = {'X':xtable[key], 'Y':ytable[key]} #key='paper_rpm'
            method_inputs.update(
                    {str(k):self.P[key][k] 
                    for k in self.P[key].keys() if(k!='method')})
            method_inputs.update({'canvas':(fig, ax)})
            if(is_key_label):
                key_label = str(key)
                assigned_labels = dcp(method_inputs['labels']) if(
                        'labels' in method_inputs) else ''
                method_inputs.update({'labels':(
                '[%s]%s'%(key_label, str(assigned_labels)) if(
                    type(assigned_labels)!=dict
                        ) else {k:'[%s]%s'%(key_label, assigned_labels[k]
                            ) for k in assigned_labels.keys()})})
            try:
                fig, ax = method(**method_inputs)
            except Exception as e:
                printer('[%d][%s]method:%s'%(
                        index, key, str(method)))
                printer('method_inputs:\n%s'%(
                        str(method_inputs)[:100]))
                {}['error:%s'%e]
        if(title!=''):
            fig.suptitle(title, fontsize=titlefontsize)
        if(file!=''):
            fig.savefig(file)
        if(canvas):
            return fig, ax
        else:
            fig.show()
            plt.ioff()
            return None
                
    def gridplot(self, x_axis=0, y_axis=1, layout=(), file='', title='', titlefontsize = 20,
             plot_method=None, is_key_label=True, figsize=(), edge_plank=(0,0,1,1), showmask=None, 
             canvas=None, showlog=0, **kwags):
        try:
            xtable = self.A[x_axis] #x_axis=0, xtable = gd.A[x_axis]
            ytable = self.A[y_axis] #y_axis=1, ytable = gd.A[y_axis]
            colsize = int(np.sqrt(len(xtable))//1 if(
                    layout==()) else layout[1])
            rowsize = int((len(xtable)//colsize)+1 if(
                    layout==()) else layout[0])
#            colsize = int((np.sqrt(len(xtable)) if(
#                    layout==()) else layout[1])//1)
#            rowsize = int(((len(xtable)//colsize)+1 if(
#                    layout==()) else layout[0])//1)
            figsize = (colsize*5, rowsize*3.5) if(
                    figsize==()) else figsize
            printer(len(xtable), colsize, rowsize)
            fig = plt.figure(figsize=figsize) #figsize=(5,3)
            printer('[GRAPHDATA gridplot] edge_plank:%s'%str(edge_plank))
            fig.subplots_adjust(*edge_plank)
            for index, key in enumerate(self.X):
                if(showmask!=None and not key in showmask):
                    continue
                if(showlog>0):
                    printer('[%d][%s]drawing....'%(index, str(key)))
                ax = plt.subplot(rowsize, colsize, index+1)
                method = lambda canvas, **kwags: (canvas[0], canvas[1])
                if(plot_method!=None):
                    method = dcp(plot_method)
                else:
                    if(key in self.P):
                        if('method' in self.P[key]):
                            method = self.P[key]['method']
                    else:
                        if(self.plot_methd!=None):
                            method = dcp(self.plot_methd)
                method_inputs = {'X':xtable[key], 'Y':ytable[key]} #key='paper_rpm'
                method_inputs.update(
                        {str(k):self.P[key][k] 
                        for k in self.P[key].keys() if(k!='method')})
                method_inputs.update({'canvas':(fig,ax)})
                if(is_key_label):
                    key_label = str(key)
                    assigned_labels = dcp(method_inputs['labels']) if(
                            'labels' in method_inputs) else ''
                    method_inputs.update({'labels':(
                    '[%s]%s'%(key_label, str(assigned_labels)) if(
                        type(assigned_labels)!=dict
                            ) else {k:'[%s]%s'%(key_label, assigned_labels[k]
                                ) for k in assigned_labels.keys()})})
                try:
                    fig, ax = method(**method_inputs)
                except Exception as e:
                    printer('[GRAPHDATA gridplot][%d][%s]method:%s'%(
                            index, key, str(method)))
                    printer('[GRAPHDATA gridplot]method_inputs:\n%s'%(
                            str(method_inputs)[:1000]))
                    method_inputs.pop('X')
                    method_inputs.pop('Y')
                    printer('[GRAPHDATA gridplot]method_inputs:\n%s'%(
                            str(method_inputs)[:1000]))
                    {}['error:%s'%e]
            if(title!=''):
                fig.suptitle(title, fontsize=titlefontsize)
            if(file!=''):
                fig.savefig(file)
            if(canvas):
                return fig, ax
            else:
                fig.show()
                plt.ioff()
                return None
        except Exception as e:
            exception_process(e, logfile=os.path.join('log', 'log.txt'), stamps=['GRAPHDATA'])
    
    def export_as_data_frame(self, x_axis=0, y_axis=1, x_axis_name='x', 
                             index='', mask=None, **kwags):
        xtable = self.A[x_axis] if(x_axis!=None) else {}
        ytable = self.A[y_axis]
        ptabel = self.P
        df_return = pd.DataFrame()
        if(xtable!={} and x_axis_name!=None):#x_axis_name!=None
            common_key = [k for k in xtable.keys()][0]
            df_return[x_axis_name] = list(xtable[common_key][0])
        for key in ytable:
            if(mask!=None):
                if(not key in mask):
                    continue
            df_return[key] = list(ytable[key][0])
        df_return = pd.DataFrame(
                np.mat(df_return), columns = df_return.columns,
                index=['%s#%d'%(index, i) for i in range(len(df_return.index))])
        return df_return
    
#interpolation_method=DFP.interpolation(method=lambda t:a*t**2+b*t+c)         
#interpolation_method=DFP.interpolation(method='linear')
#xxx = [0,1,2,3]
#yyy = [1,5,10,12]
#show_interpolation_graph(xxx, yyy, interpolation_method=DFP.interpolation)
def show_interpolation_graph(xxx, yyy, interpolation_method, x=None, file='', 
                             n_linsppts = 100, figsize=(5,5), canvas=None, **kwags):
    a = np.min(xxx)
    b = np.max(xxx)
    interpolation_production = interpolation_method(xxx, yyy, **kwags)
    printer(type(interpolation_production))
    f = interpolation_production['executor']
    x = [v for v in np.linspace(a, b, n_linsppts)]
    y = [f(v) for v in np.linspace(a, b, n_linsppts)]
    fig, ax = canvas if(canvas) else plt.subplots(figsize=figsize)
    fig, ax = curveplot([x], [y], canvas=(fig, ax))
    fig, ax = scatterplot([xxx], [yyy], frame_axis='x', canvas=(fig, ax))
    if(file and canvas==None):
        fig.savefig(file)
    else:
        if(canvas):
            return fig, ax
        else:
            fig.show()
            plt.ioff()
    return

#show_polyinterpolation(xxx, yyy)
#d=9
#anchor_interpolation_method = lambda data, **kwags:DFP.data_polyinterpolation(
#        data, n_anchor=d, **kwags)
#show_polyinterpolation(X, Y, anchor_interpolation_method)
def show_polyinterpolation(X, Y, anchor_interpolation_method, title='', file='',
                           curveplot_labels='', n_linsppts = 100, figsize=(10,6), 
                           curveplot_colors=[], canvas=None,  **kwags):
    if(np.array(X).shape[0]!=np.array(Y).shape[0]):
        return
    fig, ax = canvas if(canvas) else plt.subplots(figsize=figsize)
    fig, ax = curveplot(
            [X], [Y], labels=curveplot_labels, 
            colors=curveplot_colors, canvas=(fig, ax))
    XY = pd.DataFrame([X, Y]).T
    A, B = tuple(anchor_interpolation_method(XY, **kwags).T.values)
    fig, ax = scatterplot([[v for v in A]], [[v for v in B]], 
                          colors={(0,0):(0,0,0,0.5)}, canvas=(fig, ax))
    if(title):
        fig.suptitle(title)
    if(canvas):
        return fig, ax
    else:
        if(file):
            fig.savefig(file)
        fig.show()
        plt.ioff()

def color_map_r2b(k, n, reverse=False, **kwags):
    g = kwags['g'] if('g' in kwags) else 0
    a = kwags['a'] if('a' in kwags) else 1
    ret = (1-k/n, g, k/n, a) if(reverse) else (k/n, g, 1-k/n, a)
    if((np.array(ret)>1).any()):
        return None
    return ret

#file = DFP.pathrpt(os.path.join('scatter_matrix', 'dbscaning.jpg'))
#title = 'pairly_features_show_dbscan'
#pairly_features_show_dbscan(
#                data, label_mask, g_color_map=0.2, a_color_map=0.2,
#                assigned_pairs = [('speed', 'paper_rpm')],
#                title = title,
#                edge_plank=(0.92, 0.05, 0, 1), file=file)
def pairly_features_show_dbscan(data, label_mask, columns='full', 
                                            choose_number=4, **kwags):
    columns = data.columns if(columns=='full') else columns
    if(str(type(label_mask)).find('ndarray')==-1):
        label_mask = data[label_mask] if(
                label_mask in data.columns) else label_mask
    label_set = list(set(label_mask))
    kwags_color_map = {'g':kwags['g_color_map'] if('g_color_map' in kwags) else 0, 
                       'a':kwags['a_color_map'] if('a_color_map' in kwags) else 1}
    color_map = kwags['color_map'] if('color_map' in kwags
                     ) else (lambda k:color_map_r2b(
                             k, len(label_set), **kwags_color_map))
    gd = GRAPHDATA()
    if('assigned_pairs' in kwags):
        assigned_pairs = kwags['assigned_pairs']
        for assigned_pair in assigned_pairs:
            for label in label_set:
                x = [v for v in data[assigned_pair[0]][label_mask==label]]
                y = [v for v in data[assigned_pair[1]][label_mask==label]]
                pars_stg = '%s-%s'%assigned_pair
                gd.add_data([x, y], key=pars_stg)
                value = dcp({(k,k):color_map(k) for k in range(len(label_set))})
                gd.set_properties(pn='colors', value=value, key=pars_stg)
                value = dcp({(k,k):'%s[%d]'%(pars_stg, label_set[k]
                        ) for k in range(len(label_set))})
                gd.set_properties(pn='labels', value=value, key=pars_stg)
    pairs = list({(v, w) for w in columns for v in columns if v!=w})
    chosen_pairs = rdm.sample(pairs, choose_number)
    for chosen_pair in chosen_pairs:
        pars_stg = '-'.join([str(v) for v in chosen_pair])
        for label in label_set:
            x = [v for v in data[chosen_pair[0]][label_mask==label]]
            y = [v for v in data[chosen_pair[1]][label_mask==label]]
            gd.add_data([x, y], key='%s'%(pars_stg))
        value = dcp({(k,k):color_map(k) for k in range(len(label_set))})
        gd.set_properties(pn='colors', value=value, key='%s'%(pars_stg))
        value = dcp({(k,k):'%s[%d]'%(pars_stg, label_set[k]
                ) for k in range(len(label_set))})
        gd.set_properties(pn='labels', value=value, key='%s'%(pars_stg))
    gd.gridplot(plot_method=scatterplot, is_key_label=0, **kwags)
    
def plot_data_process(data, main_col=None, root=[], plot_method=curveplot_, 
                      **kwags):
    str_type = str(type(data))
    if(str_type.find('ndarray')>-1):
        pd_data = pd.DataFrame(data)
        return plot_data_process(
    pd_data, main_col=main_col, root=root, plot_method=plot_method, **kwags)
    if(str_type.find('pandas')>-1):
        plot_data = pd.DataFrame(data).copy()
        if(np.array(root).shape[0]!=data.shape[0]):
            plot_data = pd.DataFrame(data).copy()
            plot_data = plot_data.reset_index().copy() if(main_col=='index') else plot_data.copy()
            root = plot_data.pop(main_col if(main_col!=None) else plot_data.columns[0])
            printer(str(list(root))[:200])
        return plot_method(
        [list(root)], 
        [list(plot_data[col]) for col in plot_data.columns], 
        mode='x', subkeys=[v for v in plot_data.columns], **kwags)

#plot_data_process_with_grid(data, main_col=None, xheaders='full', 
#                                plot_method=curveplot, **kwags)
def plot_data_process_with_grid(data, main_col=None, xheaders='full', 
                                plot_method=curveplot, is_key_label=1, **kwags):
        gd = GRAPHDATA()
        xheaders = list(data.columns if(xheaders=='full') else xheaders)
        data_ = data.copy()
        if(main_col!=None):
            root = list(data_.pop(main_col))
            xheaders.pop(list(xheaders).index(main_col))
        else:
            root = list(data_.index)
        for xn in xheaders:
            xarray = list(data_[xn].astype(float))
            gd.add_data([root, xarray], key = xn)
        gd.gridplot(plot_method=plot_method, is_key_label=is_key_label, 
                    **kwags)
#%%