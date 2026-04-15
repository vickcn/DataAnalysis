# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 16:31:18 2021

@author: anna.chou
含浸廢品<0.5
廢品最少情況下，含浸速度、擺盪、壓桿刻度、張力、間隙等搭配建議值(建議從G機台開始) 
"""
#import data_collecting as clt
#import Config_parse as con
import abc
import sys
import threading

import os
from package import visualization2 as vs2
vs = vs2.vs
DFP = vs2.DFP
LOGger = vs2.LOGger
#import server_data_scienario as svd
from package.LOGger import CreateContainer, CreateFile, addlog, stamp_process, exception_process, mylist

import keras
from pkg_resources import parse_version

# Keras imports with version check
if parse_version(keras.__version__) >= parse_version('3.0.0'):
    # Keras 3.x
    try:
        from keras.models import Sequential, Model
        from keras.layers import Concatenate, concatenate, Layer, Input
        from keras import backend as bcd
        from keras.utils import plot_model
        from keras.optimizers import Adam, SGD
        from keras import callbacks
        from keras import models
        from keras.callbacks import EarlyStopping
    except ImportError:
        from tensorflow.keras.models import Sequential, Model
        from tensorflow.keras.layers import Concatenate, concatenate, Layer, Input
        from tensorflow.keras import backend as bcd
        from tensorflow.keras.utils import plot_model
        from tensorflow.keras.optimizers import Adam, SGD
        from tensorflow.keras import callbacks
        from tensorflow.keras import models
        from tensorflow.keras.callbacks import EarlyStopping
else:
    # Keras 2.x
    from keras.models import Sequential, Model
    from keras.layers import Concatenate, concatenate, Layer, Input
    from keras import backend as bcd
    from keras.utils.vis_utils import plot_model
    from keras.optimizers import Adam, SGD
    from keras import callbacks
    from keras import models
    from keras.callbacks import EarlyStopping
import tensorflow as tf

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
from keras import backend as K
sys.path.append('..')
import pandas as pd
import joblib
from copy import deepcopy as dcp
from sklearn.metrics import confusion_matrix
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder as LBC
import numpy as np
from datetime import datetime as dt
from sklearn.preprocessing import StandardScaler as Stdscr
from sklearn.preprocessing import MinMaxScaler as Mnxscr
from sklearn import metrics as skm
from keras import models
from keras import backend as bcd #转换为张量
from keras.models import model_from_json
import pickle
from sklearn.metrics import cohen_kappa_score as cohkpa

logfile = 'log\\log_%s.txt'%(dt.now().strftime('%Y%m%d'))
logshield = False
layer_producer = lambda nn_type, **kwags: getattr(keras.layers, nn_type, get_custom_objects().get(nn_type))
#%%
class ClipLayer(Layer):
    def __init__(self, min_value, max_value, axis=None, **kwargs):
        """
        參數:
            min_value: 最小值
            max_value: 最大值
            axis: 要裁剪的軸。如果為None，則對所有元素進行裁剪
        """
        super(ClipLayer, self).__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value
        self.axis = axis

    def call(self, inputs):
        if self.axis is None:
            # 原來的行為：對所有元素進行裁剪
            return tf.clip_by_value(inputs, self.min_value, self.max_value)
        else:
            # 只對指定axis進行裁剪
            shape = tf.shape(inputs)
            # 創建布爾遮罩
            mask = tf.range(shape[self.axis]) < shape[self.axis]
            # 擴展維度以匹配輸入形狀
            for _ in range(self.axis):
                mask = tf.expand_dims(mask, 0)
            for _ in range(len(inputs.shape) - self.axis - 1):
                mask = tf.expand_dims(mask, -1)
            
            # 只在指定axis上應用裁剪
            clipped = tf.where(
                mask,
                tf.clip_by_value(inputs, self.min_value, self.max_value),
                inputs
            )
            return clipped

    def get_config(self):
        config = super(ClipLayer, self).get_config()
        config.update({
            "min_value": self.min_value,
            "max_value": self.max_value,
            "axis": self.axis
        })
        return config

class SequenceDenseLayer(Layer):
    def __init__(self, shape, activation='linear', **kwargs):
        super(SequenceDenseLayer, self).__init__(**kwargs)
        self.shape = shape
        units = np.prod(shape)
        self.dense = layer_producer('Dense')(units, activation=activation)  # 创建一个 Dense 层
        self.activation = activation

    def call(self, inputs):
        # Dense 层生成的输出 (batch_size, units)
        x = self.dense(inputs)
        
        # 假设 Dense 输出的 shape 是 (batch_size, 40)，将其 reshape 为 (batch_size, 20)
        x_reshaped = tf.reshape(x, (-1, *self.shape)) if(len(self.shape)>1) else tf.reshape(x, (-1, *self.shape, 1))
        return x_reshaped
    
    def get_config(self):
        config = super(SequenceDenseLayer, self).get_config()
        config.update({
            "shape": self.shape,
            "activation": self.activation
        })
        return config

class SortLayer(keras.layers.Layer):
    def call(self, inputs):
        return tf.sort(inputs, axis=-1)
    
get_custom_objects()['SequenceDenseLayer']=SequenceDenseLayer
get_custom_objects()['SortLayer']=SortLayer
get_custom_objects()['ClipLayer']=ClipLayer

def standard_input_setting(mdc, **kwags):
    try:
        keras_input, keras_input_preconcatenate = [], []
        #input layers
        cell_size = dcp(getattr(mdc, 'cell_size', None))
        for k,v in mdc.xheader.mylist_grp.items():
            if(cell_size==None):
                input_shape = len(v) 
            else:
                if(DFP.isiterable(cell_size)):
                    input_shape = (*cell_size, len(v))
                else:
                    input_shape = (cell_size, len(v))
            input_zone = Input(shape = input_shape, name=stamp_process('',['input',k],'','','','_'))
            preprocessing = LOGger.mylist([dcp(getattr(v, 'preprocessing', None))]).get_all()
            if('encoding' in preprocessing):
                input_zone_preconcatenate = layer_producer('Embedding')(
                    np.prod(v.preprocessor.class_sizes), len(v), input_length=cell_size)(input_zone)
                input_zone_preconcatenate = layer_producer('Reshape')(target_shape=(len(v),))(input_zone_preconcatenate)
            else:
                input_zone_preconcatenate = input_zone
            keras_input.append(input_zone) 
            keras_input_preconcatenate.append(input_zone_preconcatenate)
        addlog('keras_input:%s'%stamp_process('', stamps = list(map(str, keras_input)), 
                                               stamp_left='\n', stamp_right=''), **kwags)
            
        mdc.keras_input = LOGger.mylist(keras_input).get()
        mdc.inputs_layer = concatenate(
            keras_input_preconcatenate) if(len(keras_input)>1) else keras_input_preconcatenate[0]
        return True
    except Exception as e:
        exception_process(e, logilfe=os.path.join(mdc.exp_fd, 'log.txt'))
        return False

if(True): #myLayerBuilder
    class myCoreLayerBuilder:
        def __init__(self, name='', **kwags):
            self.name = name
            self.paramDefaults = {
                'name':name, 
                'activation':None
                }
            self.layerSystem = None
            
        def build(self, **kwags):
            LOGger.addDebug(LOGger.type_string(self))
            if(hasattr(keras.layers, LOGger.type_string(self).replace("my","").replace("LayerBuilder",""))):
                self.layerSystem = getattr(keras.layers, LOGger.type_string(self).replace("my","").replace("LayerBuilder",""))
            else:
                self.layerSystem = keras.layers.Dense
                self.paramDefaults['units'] = 1
            
            layerBuildKwags = dcp(self.paramDefaults)
            layerBuildKwags.update({k:v for k,v in kwags.items() if k in self.paramDefaults})
            if(not self.defaultConfiguring(layerParam=layerBuildKwags, **kwags)):
                return None
            LOGger.addDebug(self.name)
            LOGger.addDebug(self.layerSystem)
            return self.layerSystem(**layerBuildKwags)
        
        def defaultConfiguring(self, layerParam=None, **kwags):
            return True
        
    class myDenceLayerBuilder(myCoreLayerBuilder):
        def __init__(self, units=1, use_bias=True, **kwags):
            super().__init__(**kwags)
            self.paramDefaults['units'] = units
            self.paramDefaults['use_bias'] = use_bias
            
    class myLSTMLayerBuilder(myDenceLayerBuilder):
        def __init__(self, units=1, use_bias=True, dropout=0.0, return_sequences=True, **kwags):
            super().__init__(units=units, use_bias=use_bias, **kwags)
            self.paramDefaults['dropout'] = dropout
            self.paramDefaults['return_sequences'] = return_sequences
            
    class myConv1DLayerBuilder(myCoreLayerBuilder):
        def __init__(self, filters=1, kernel_size=None, strides=1, padding='valid', use_bias=True, **kwags):
            super().__init__(**kwags)
            self.paramDefaults['filters'] = filters
            self.paramDefaults['kernel_size'] = kernel_size
            self.paramDefaults['strides'] = strides
            self.paramDefaults['padding'] = padding
            self.paramDefaults['use_bias'] = use_bias
            
        def defaultConfiguring(self, layer=None, layerParam=None, **kwags):
            layerParam = layerParam if(isinstance(layerParam, dict)) else {}
            if(layerParam['kernel_size'] is None):  layerParam['kernel_size'] = LOGger.extract(getattr(layer,'shape', ()), 1, default=9)
            return True
            
    class myConv2DLayerBuilder(myConv1DLayerBuilder):
        def __init__(self, filters=1, kernel_size=(3,3), strides=(1,1), padding='valid', use_bias=True, **kwags):
            super().__init__(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias, **kwags)

def stack_layer_configuration(default_layer_name, ret, layer_index=0, return_sequences=True, previous_layer=None,
                            hdn_lyr_size=1, kernel_size=None, strides=None, stack_names=None, stamps=None, **kwags):
    default_layer = layer_producer(default_layer_name)
    kwags_layer = {}
    if(default_layer_name=='LSTM'):
        kwags_layer.update({'units':int(hdn_lyr_size//1)})
        # kwags_layer.update({'return_sequences':True} if(i<len(tuple(hidden_layer_sizes))-1) else {
        #     'return_sequences':kwags.get('default_outreturn_sequences', False)})
        kwags_layer.update({'return_sequences':return_sequences})
    elif(default_layer_name=='Conv2D'):
        kwags_layer.update({'filters':int(hdn_lyr_size//1)})
        kernel_size = tuple(kernel_size if(DFP.isiterable(kernel_size)) else (3,3))
        kwags_layer.update({'kernel_size':kernel_size})
    elif(default_layer_name=='Conv1D'):
        kwags_layer.update({'filters':int(hdn_lyr_size//1)})
        kernel_size = kernel_size if(isinstance(kernel_size, int)) else 9
        kwags_layer.update({'kernel_size':kernel_size})
        strides = strides if(isinstance(strides, int)) else 1
        kwags_layer.update({'strides':strides})
    else:
        kwags_layer.update({'units':int(hdn_lyr_size//1)})
    name = dcp(LOGger.stamp_process('',[*stamps, '%dth-layer'%(layer_index+1)],'','','','_'))
    if(isinstance(stack_names, list)): stack_names.append(dcp(name))
    LOGger.addDebug('name:%s'%str(name), stamps=['stack_layers'])
    ret['name'] = name
    ret['kwags_layer'] = kwags_layer
    ret['default_layer'] = default_layer
    return True

def stack_layers(layer_index, start, hidden_layer_sizes=(), hidden_layer_nns={},
                 activation='relu', default_layer_name='Dense', dropout_rates=None, 
                 maxpool2D_sizes=None, kernel_sizes=None, strideses=None, 
                 stamps=None, ret=None, outputFlatten=True, cell_size=None, **kwags):
    stamps = stamps if(isinstance(stamps, list)) else []
    addlog = kwags.get('addlog', LOGger.addlog)
    try:
        stack_names = []
        strideseFlatten = int(np.prod(strideses) if(DFP.isiterable(strideses) and strideses) else 1)
        layer = start
        if(LOGger.isinstance_not_empty(hidden_layer_nns, dict)):
            enumeration = [(i,k,v) for i,(k,v) in enumerate(hidden_layer_nns.items())]
            for (i,k,v) in enumeration:
                addlog('##hdn_lyr_nn method - layer%d'%(layer_index+1))
                nn_type = str(v.get('type', default_layer_name))
                kwags = dcp(v)
                name = '%dth-layer'%(layer_index+1) if(isinstance(k, int)) else LOGger.stamp_process(
                    '', [*stamps, k], '','','','_')
                kwags.update({'name':dcp(name)})
                stack_names.append(dcp(name))
                if(nn_type=='LSTM'):
                    if(len(layer.shape)==2):
                        layer = layer_producer('RepeatVector')(cell_size[0])(layer)
                    if(i == -1%len(hidden_layer_nns)):
                        kwags['return_sequences'] = False
                    else:
                        iNext,kNext,vNext = enumeration[i+1]
                        if(vNext.get('type', default_layer_name) == 'Dense'):
                            kwags['return_sequences'] = False
                addlog('type:%s with param:%s'%(nn_type, stamp_process('', kwags, ':','[',']')))
                layer = eval('my%sLayerBuilder'%nn_type)().build(**kwags)(layer)
                layer_index += 1
        elif(DFP.isiterable(hidden_layer_sizes)):
            if(not isinstance(dropout_rates, dict) and not isinstance(dropout_rates, list)):
                dropout_rates = LOGger.mylist([])
            print('create dropout_rates:%s'%str(dropout_rates))
            maxpool2D_sizes = (LOGger.mylist(maxpool2D_sizes if(
                len(maxpool2D_sizes)!=0) else [None]*(len(hidden_layer_sizes)-1)+[(2,2)])) if(
                DFP.isiterable(maxpool2D_sizes)) else LOGger.mylist([])
            print('create maxpool_sizes:%s'%str(maxpool2D_sizes))
            if(not isinstance(kernel_sizes, dict) and not isinstance(kernel_sizes, list)):
                kernel_sizes = []
            if(not isinstance(strideses, dict) and not isinstance(strideses, list)):
                strideses = []
            print('create kernel_sizes:%s'%str(kernel_sizes))
            n_hidden_layer_sizes = len(tuple(hidden_layer_sizes))
            retTemp = {}
            for i, hdn_lyr_size in enumerate(list(hidden_layer_sizes)):
                retTemp.clear()
                print('hdn_lyr_sz method - layer%d:'%(layer_index+1))
                return_sequences = True if(i%n_hidden_layer_sizes<n_hidden_layer_sizes-1) else kwags.get(
                    'default_outreturn_sequences', False)
                kernel_size = LOGger.extract(kernel_sizes, i, default=kwags.get('kernel_sizeDefault', None))
                strides = LOGger.extract(strideses, i, default=kwags.get('stridesDefault', None))
                if(not stack_layer_configuration(default_layer_name, ret=retTemp, layer_index=layer_index, 
                                    return_sequences=return_sequences, hdn_lyr_size=hdn_lyr_size, 
                                    kernel_size=kernel_size, strides=strides, stack_names=stack_names, stamps=stamps, 
                                    previous_layer=layer, **kwags)):
                    return None, 0
                layer = retTemp['default_layer'](activation=str(activation), name=retTemp['name'], **retTemp['kwags_layer'])(layer)
                # layer = default_layer(activation=str(activation), name=name, **kwags_layer)(layer)
                print('layer.shape:%s'%str(layer.shape))
                maxpool2D_config = dcp(LOGger.extract(maxpool2D_sizes, key=retTemp['name'], index=i, default={}))
                maxpool2D_config = (maxpool2D_config if(len(maxpool2D_config)>1) else (2,2)) if(
                    DFP.isiterable(maxpool2D_config)) else None
                maxpool2D_config = {'pool_size':tuple(map(DFP.astype, maxpool2D_config[:2]))} if(
                    len(maxpool2D_config)>1 if(DFP.isiterable(maxpool2D_config)) else False) else maxpool2D_config
                if(LOGger.isinstance_not_empty(maxpool2D_config, dict)):
                    layer = layer_producer('MaxPooling2D')(**maxpool2D_config)(layer)
                    print('maxpool2D_config:%s'%str(maxpool2D_config))
                dropout_config = dcp(LOGger.extract(dropout_rates, key=retTemp['name'], index=i, default={}))
                dropout_config = {'rate':float(dropout_config)} if(
                    not DFP.isnonnumber(dropout_config)) else dropout_config
                if(LOGger.isinstance_not_empty(dropout_config, dict)):
                    dropout_config.update({'name':LOGger.stamp_process('',[*stamps, '%dth-dropout'%(layer_index+1)],'','','','_')})
                    default_layer = layer_producer('Dropout')
                    layer = default_layer(**dropout_config)(layer)
                if(outputFlatten):
                    if(len(layer.shape)>2 and i%n_hidden_layer_sizes==n_hidden_layer_sizes-1):
                        layer = layer_producer('Flatten')()(layer)
                print('layer.shape::%s'%str(layer.shape))
                layer_index += 1
        if(isinstance(ret, dict)):  
            ret['stack_names'] = stack_names
            LOGger.addDebug('stack_names:%s'%str(ret['stack_names']))
    except Exception as e:
        exp_fd = kwags.get('exp_fd', 'log')
        exception_process(e, logfile=os.path.join(exp_fd, 'log.txt'), stamps=[stack_layers.__name__])
        return None, 0
    return layer, layer_index

def standard_output_setting(mdc, layer, **kwags):
    keras_output = []
    lossmethods, output_activations = LOGger.mylist(), LOGger.mylist()
    for i,(k,v) in enumerate(mdc.yheader.mylist_grp.items()):
        opt_sz = len(v)
        default_layer = layer_producer('Dense')
        activation = getattr(v,'activation',LOGger.extract(
                getattr(mdc, 'activations', []), index=i, key=k, default=getattr(mdc, 'activation', 'relu')))
        output_activations.append(activation)
        output_zone = default_layer(units=opt_sz, activation=activation, 
                                name=stamp_process('',['output',k],'','','','_'))(layer)
        lossmethods.append(v.lossmethod)
        keras_output.append(output_zone)
    unfam_layer_stacking(mdc, layer, keras_output) if(
        not isinstance(getattr(mdc, 'unfam_header', None), type(None))) else None 
    addlog('keras_output:%s'%stamp_process('', stamps = list(map(str, keras_output)), 
                                               stamp_left='\n', stamp_right=''), **kwags)
    mdc.lossmethods = lossmethods
    addlog('operating lossmethods:%s'%(str(mdc.lossmethods.get())), **kwags)
    mdc.output_activations = output_activations
    addlog('operating output_activations:%s'%(str(output_activations.get())), **kwags)
    mdc.keras_output = keras_output if(len(keras_output)>1) else keras_output[0]
    mdc.model_form = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    return True
    
def unfam_layer_stacking(mdc, layer_input, layer_output_mains, lossmethods=None, output_activations=None):
    lossmethods = lossmethods if(isinstance(lossmethods, list)) else []
    output_activations = output_activations if(isinstance(output_activations, list)) else ['sigmoid']
    mdc.keras_output = mylist(layer_output_mains)
    print('output_activations', output_activations)
    for i,(k,v) in enumerate(mdc.unfam_header_collector.mylist_grp.items()):
        output_shape = len(v)
        default_layer = layer_producer('Dense')
        activation = LOGger.extract(output_activations, index=i, key=k, defualt='sigmoid')
        unfamed = default_layer(units=output_shape, activation=activation, 
                                name=stamp_process('',['unfam',k],'','','','_'))(layer_input)
        mdc.keras_output += [unfamed]
        lossmethod = LOGger.extract(lossmethods, index=i, key=k, default='binary_crossentropy')
        mdc.lossmethods += [lossmethod]
        mdc.output_activations += [activation]
    mdc.keras_output = mdc.keras_output.get()
    return True

def stack_layers_v2(layer_index, start, hidden_layer_sizes=(), hidden_layer_nns={},
                    activation='relu', default_layer_name='Dense', dropout_rates=None, 
                    maxpool2D_sizes=None, kernel_sizes=None, strideses=None, 
                    stamps=None, ret=None, outputFlatten=True, cell_size=None, **kwags):
    stamps = stamps if(isinstance(stamps, list)) else []
    addlog = kwags.get('addlog', LOGger.addlog)
    try:
        stack_names = []
        strideseFlatten = int(np.prod(strideses) if(DFP.isiterable(strideses) and strideses) else 1)
        layer = start
        if(LOGger.isinstance_not_empty(hidden_layer_nns, dict)):
            enumeration = [(i,k,v) for i,(k,v) in enumerate(hidden_layer_nns.items())]
            for (i,k,v) in enumeration:
                addlog('##hdn_lyr_nn method - layer%d'%(layer_index+1))
                nn_type = str(v.get('type', default_layer_name))
                kwags = dcp(v)
                name = '%dth-layer'%(layer_index+1) if(isinstance(k, int)) else LOGger.stamp_process(
                    '', [*stamps, k], '','','','_')
                kwags.update({'name':dcp(name)})
                stack_names.append(dcp(name))
                if(nn_type=='LSTM'):
                    if(len(layer.shape)==2):
                        layer = layer_producer('RepeatVector')(cell_size[0])(layer)
                    if(i == -1%len(hidden_layer_nns)):
                        kwags['return_sequences'] = False
                    else:
                        iNext,kNext,vNext = enumeration[i+1]
                        if(vNext.get('type', default_layer_name) == 'Dense'):
                            kwags['return_sequences'] = False
                # elif(nn_type=='Conv1D'):
                    # if(True):
                    #     layer = layer_producer('Dense')(
                    #         units=int(cell_size[0]//strideseFlatten) * hidden_layer_sizes[0], activation=activation, 
                    #         name=stamp_process('',['preOutput_flatten',k],'','','','_'))(layer)
                    #     layer = layer_producer('Reshape')(
                    #         target_shape=(int(cell_size[0]//strideseFlatten), hidden_layer_sizes[0]), 
                    #         name=stamp_process('',['preOutput_flattenReshape',k],'','','','_'))(layer)
                addlog('type:%s with param:%s'%(nn_type, stamp_process('', kwags, ':','[',']')))
                layer = eval('my%sLayerBuilder'%nn_type)().build(**kwags)(layer)
                layer_index += 1
        elif(DFP.isiterable(hidden_layer_sizes)):
            if(not isinstance(dropout_rates, dict) and not isinstance(dropout_rates, list)):
                dropout_rates = LOGger.mylist([])
            print('create dropout_rates:%s'%str(dropout_rates))
            maxpool2D_sizes = (LOGger.mylist(maxpool2D_sizes if(
                len(maxpool2D_sizes)!=0) else [None]*(len(hidden_layer_sizes)-1)+[(2,2)])) if(
                DFP.isiterable(maxpool2D_sizes)) else LOGger.mylist([])
            print('create maxpool_sizes:%s'%str(maxpool2D_sizes))
            if(not isinstance(kernel_sizes, dict) and not isinstance(kernel_sizes, list)):
                kernel_sizes = []
            if(not isinstance(strideses, dict) and not isinstance(strideses, list)):
                strideses = []
            print('create kernel_sizes:%s'%str(kernel_sizes))
            n_hidden_layer_sizes = len(tuple(hidden_layer_sizes))
            retTemp = {}
            for i, hdn_lyr_size in enumerate(list(hidden_layer_sizes)):
                retTemp.clear()
                print('hdn_lyr_sz method - layer%d:'%(layer_index+1))
                return_sequences = True if(i%n_hidden_layer_sizes<n_hidden_layer_sizes-1) else kwags.get(
                    'default_outreturn_sequences', False)
                kernel_size = LOGger.extract(kernel_sizes, i, default=kwags.get('kernel_sizeDefault', None))
                strides = LOGger.extract(strideses, i, default=kwags.get('stridesDefault', None))
                if(not stack_layer_configuration(default_layer_name, ret=retTemp, layer_index=layer_index, 
                                    return_sequences=return_sequences, hdn_lyr_size=hdn_lyr_size, 
                                    kernel_size=kernel_size, strides=strides, stack_names=stack_names, stamps=stamps, 
                                    previous_layer=layer, **kwags)):
                    return None, 0
                layer = retTemp['default_layer'](activation=str(activation), name=retTemp['name'], **retTemp['kwags_layer'])(layer)
                # layer = default_layer(activation=str(activation), name=name, **kwags_layer)(layer)
                print('layer.shape:%s'%str(layer.shape))
                maxpool2D_config = dcp(LOGger.extract(maxpool2D_sizes, key=retTemp['name'], index=i, default={}))
                maxpool2D_config = (maxpool2D_config if(len(maxpool2D_config)>1) else (2,2)) if(
                    DFP.isiterable(maxpool2D_config)) else None
                maxpool2D_config = {'pool_size':tuple(map(DFP.astype, maxpool2D_config[:2]))} if(
                    len(maxpool2D_config)>1 if(DFP.isiterable(maxpool2D_config)) else False) else maxpool2D_config
                if(LOGger.isinstance_not_empty(maxpool2D_config, dict)):
                    layer = layer_producer('MaxPooling2D')(**maxpool2D_config)(layer)
                    print('maxpool2D_config:%s'%str(maxpool2D_config))
                dropout_config = dcp(LOGger.extract(dropout_rates, key=retTemp['name'], index=i, default={}))
                dropout_config = {'rate':float(dropout_config)} if(
                    not DFP.isnonnumber(dropout_config)) else dropout_config
                if(LOGger.isinstance_not_empty(dropout_config, dict)):
                    dropout_config.update({'name':LOGger.stamp_process('',[*stamps, '%dth-dropout'%(layer_index+1)],'','','','_')})
                    default_layer = layer_producer('Dropout')
                    layer = default_layer(**dropout_config)(layer)
                if(outputFlatten):
                    if(len(layer.shape)>2 and i%n_hidden_layer_sizes==n_hidden_layer_sizes-1):
                        layer = layer_producer('Flatten')()(layer)
                print('layer.shape::%s'%str(layer.shape))
                layer_index += 1
        if(isinstance(ret, dict)):  
            ret['stack_names'] = stack_names
            LOGger.addDebug('stack_names:%s'%str(ret['stack_names']))
    except Exception as e:
        exp_fd = kwags.get('exp_fd', 'log')
        exception_process(e, logfile=os.path.join(exp_fd, 'log.txt'), stamps=[stack_layers.__name__])
        return None, 0
    return layer, layer_index

#%%
def standard_sampling_constructor(latent_dim):
    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                  mean=0., stddev=0.1)
        return z_mean + K.exp(z_log_sigma) * epsilon
    return sampling

def building_ACWPred_v6(mdc, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    
    
    #INPUT
    virtual_weight_zones, input_LSTM_zones, input_temp_zones  = [], [], []
    #LSTM層
    for al in [1,2,3,5,6,7,8]:
        LSTM_layered = None
        input_LSTM_zones_AL = []
        for f in ['tp','rh']:
            k = LOGger.stamp_process('',['AL%d'%al, f],'','','','_',for_file=1)
            if(k in mdc.xheader_collector.mylist_grp):
            # if(True):
                header_zone = mdc.xheader_collector.mylist_grp[k]
                cell_size = getattr(header_zone, 'cell_size', None)
                input_shape = cell_size if(cell_size!=None) else len(header_zone)
                input_main_zone = Input(shape = input_shape, name=stamp_process('',['input',k],'','','','_'))
                input_LSTM_zones_AL.append(input_main_zone)
                input_LSTM_zones.append(input_main_zone) #for entrance
        if(input_LSTM_zones_AL!=[]):
            merged_LSTM_AL = layer_producer('concatenate')(input_LSTM_zones_AL)
            layer_index = 0
            mdc.addlog('mdc.hidden_layer_sizes:%s'%str(mdc.hidden_layer_sizes), stamps=['LSTM'])
            LSTM_layered, layer_index = stack_layers(layer_index, merged_LSTM_AL, activation=mdc.activation, 
                                                     default_layer_name = 'LSTM',
                                                     hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                                                     cell_size=cell_size, 
                                                     dropout_rates = getattr(mdc, 'dropout_rates', None),
                                                     addlog = addlog, stamps=['AL%d'%al])
            
        if('AL%d_weight'%al in mdc.xheader_collector.mylist_grp):
            header_zone = mdc.xheader_collector.mylist_grp['AL%d_weight'%al]
            input_shape = len(header_zone)
            input_temp_zone = Input(shape = input_shape, name=stamp_process('',['input','AL%d_weight'%al],'','','','_'))
            input_temp_zones.append(input_temp_zone) #for entrance
            LSTM_layered = layer_producer('concatenate')([input_temp_zone, LSTM_layered]) if(LSTM_layered!=None) else input_temp_zone
            LSTM_layered = layer_producer('Dense')(units=1, activation='linear',
                                                   name=stamp_process('',['input','AL%d_virtual_weight'%al],'','','','_'))(LSTM_layered)
        if(LSTM_layered!=None):
            virtual_weight_zones.append(LSTM_layered)
    
    # sys.exit(1)
    mdc.keras_input = input_temp_zones + input_LSTM_zones
    merged_LSTM = layer_producer('concatenate')(virtual_weight_zones)
    
    #OUTPUT
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'main'
    header_zone = mdc.yheader_collector.mylist_grp[header_zone_key]
    output_shape = header_zone.cell_size[-1]
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'linear')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_main = default_layer(units=1, activation=activation,
                                 name=stamp_process('',['output','decode'],'','','','_'))(merged_LSTM)
    output_main = default_layer(units=output_shape, activation=getattr(mdc, 'output_activation', 'linear'),
                                 name=stamp_process('',['output','main'],'','','','_'))(decoded_main)
    output_main = layer_producer('SortLayer')()(output_main)
    
    #UNFAM
    header_zone_key = 'main'
    header_zone = mdc.unfam_header_collector.mylist_grp[header_zone_key]
    
    default_layer = layer_producer('Dense')
    unfamed = default_layer(units=1, activation='sigmoid',
                            name=stamp_process('',['unfam',header_zone_key],'','','','_'))(decoded_main)
    lossmethods.append('BinaryCrossentropy')
    activations.append('sigmoid')
    
    mdc.keras_output = [output_main, unfamed]
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    return model

def building_ACWPred_v5(mdc, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    
    
    #INPUT
    virtual_weight_zones, input_LSTM_zones, input_temp_zones  = [], [], []
    #LSTM層
    for al in [1,2,3,5,6,7,8]:
        LSTM_layered = None
        input_LSTM_zones_AL = []
        for f in ['tp','rh']:
            k = LOGger.stamp_process('',['AL%d'%al, f],'','','','_',for_file=1)
            if(k in mdc.xheader_collector.mylist_grp):
            # if(True):
                header_zone = mdc.xheader_collector.mylist_grp[k]
                cell_size = getattr(header_zone, 'cell_size', None)
                input_shape = cell_size if(cell_size!=None) else len(header_zone)
                input_main_zone = Input(shape = input_shape, name=stamp_process('',['input',k],'','','','_'))
                input_LSTM_zones_AL.append(input_main_zone)
                input_LSTM_zones.append(input_main_zone) #for entrance
        if(input_LSTM_zones_AL!=[]):
            merged_LSTM_AL = layer_producer('concatenate')(input_LSTM_zones_AL)
            temp_shape = (cell_size[0], 2*cell_size[1])
            merged_LSTM_AL = layer_producer('Reshape')((np.prod(temp_shape), 1))(merged_LSTM_AL)
            merged_LSTM_AL = layer_producer('Conv1D')(1, kernel_size=1, use_bias=True, padding='same')(merged_LSTM_AL)
            merged_LSTM_AL = layer_producer('Reshape')(temp_shape)(merged_LSTM_AL)
            layer_index = 0
            mdc.addlog('mdc.hidden_layer_sizes:%s'%str(mdc.hidden_layer_sizes), stamps=['Conv1D'])
            LSTM_layered, layer_index = stack_layers(layer_index, merged_LSTM_AL, activation=mdc.activation, 
                                                     default_layer_name = 'Conv1D',
                                                     hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                                                     cell_size=cell_size, 
                                                     dropout_rates = getattr(mdc, 'dropout_rates', None),
                                                     addlog = addlog, stamps=['AL%d'%al])
            LSTM_layered = layer_producer('Flatten')()(LSTM_layered)
        if('AL%d_weight'%al in mdc.xheader_collector.mylist_grp):
            header_zone = mdc.xheader_collector.mylist_grp['AL%d_weight'%al]
            input_shape = len(header_zone)
            input_temp_zone = Input(shape = input_shape, name=stamp_process('',['input','AL%d_weight'%al],'','','','_'))
            input_temp_zones.append(input_temp_zone) #for entrance
            LSTM_layered = layer_producer('concatenate')([input_temp_zone, LSTM_layered]) if(LSTM_layered!=None) else input_temp_zone
            LSTM_layered = layer_producer('Dense')(units=1, activation='linear',
                                                   name=stamp_process('',['input','AL%d_virtual_weight'%al],'','','','_'))(LSTM_layered)
        if(LSTM_layered!=None):
            virtual_weight_zones.append(LSTM_layered)
    
    # sys.exit(1)
    mdc.keras_input = input_temp_zones + input_LSTM_zones
    merged_LSTM = layer_producer('concatenate')(virtual_weight_zones)
    
    #OUTPUT
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'main'
    header_zone = mdc.yheader_collector.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'linear')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_main = default_layer(units=1, activation=activation,
                                 name=stamp_process('',['output','decode'],'','','','_'))(merged_LSTM)
    output_main = default_layer(units=output_shape, activation=getattr(mdc, 'output_activation', 'linear'),
                                 name=stamp_process('',['output','main'],'','','','_'))(decoded_main)
    
    #UNFAM
    header_zone_key = 'main'
    header_zone = mdc.unfam_header_collector.mylist_grp[header_zone_key]
    
    default_layer = layer_producer('Dense')
    unfamed = default_layer(units=1, activation='sigmoid',
                            name=stamp_process('',['unfam',header_zone_key],'','','','_'))(decoded_main)
    lossmethods.append('BinaryCrossentropy')
    activations.append('sigmoid')
    
    mdc.keras_output = [output_main, unfamed]
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    return model

def building_ACWPred_v4(mdc, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    
    
    #INPUT
    virtual_weight_zones, input_LSTM_zones, input_temp_zones  = [], [], []
    #LSTM層
    for al in [1,2,3,5,6,7,8]:
        LSTM_layered = None
        input_LSTM_zones_AL = []
        for f in ['tp','rh']:
            k = LOGger.stamp_process('',['AL%d'%al, f],'','','','_',for_file=1)
            if(k in mdc.xheader_collector.mylist_grp):
            # if(True):
                header_zone = mdc.xheader_collector.mylist_grp[k]
                cell_size = getattr(header_zone, 'cell_size', None)
                input_shape = cell_size if(cell_size!=None) else len(header_zone)
                input_main_zone = Input(shape = input_shape, name=stamp_process('',['input',k],'','','','_'))
                input_LSTM_zones_AL.append(input_main_zone)
                input_LSTM_zones.append(input_main_zone) #for entrance
        if(input_LSTM_zones_AL!=[]):
            merged_LSTM_AL = layer_producer('concatenate')(input_LSTM_zones_AL)
            layer_index = 0
            mdc.addlog('mdc.hidden_layer_sizes:%s'%str(mdc.hidden_layer_sizes), stamps=['Conv1D'])
            LSTM_layered, layer_index = stack_layers(layer_index, merged_LSTM_AL, activation=mdc.activation, 
                                                     default_layer_name = 'Conv1D',
                                                     hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                                                     cell_size=cell_size, 
                                                     dropout_rates = getattr(mdc, 'dropout_rates', None),
                                                     addlog = addlog, stamps=['AL%d'%al])
            LSTM_layered = layer_producer('Flatten')()(LSTM_layered)
        if('AL%d_weight'%al in mdc.xheader_collector.mylist_grp):
            header_zone = mdc.xheader_collector.mylist_grp['AL%d_weight'%al]
            input_shape = len(header_zone)
            input_temp_zone = Input(shape = input_shape, name=stamp_process('',['input','AL%d_weight'%al],'','','','_'))
            input_temp_zones.append(input_temp_zone) #for entrance
            LSTM_layered = layer_producer('concatenate')([input_temp_zone, LSTM_layered]) if(LSTM_layered!=None) else input_temp_zone
            LSTM_layered = layer_producer('Dense')(units=1, activation='linear',
                                                   name=stamp_process('',['input','AL%d_virtual_weight'%al],'','','','_'))(LSTM_layered)
        if(LSTM_layered!=None):
            virtual_weight_zones.append(LSTM_layered)
    
    # sys.exit(1)
    mdc.keras_input = input_temp_zones + input_LSTM_zones
    merged_LSTM = layer_producer('concatenate')(virtual_weight_zones)
    
    #OUTPUT
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'main'
    header_zone = mdc.yheader_collector.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'linear')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_main = default_layer(units=1, activation=activation,
                                 name=stamp_process('',['output','decode'],'','','','_'))(merged_LSTM)
    output_main = default_layer(units=output_shape, activation=getattr(mdc, 'output_activation', 'linear'),
                                 name=stamp_process('',['output','main'],'','','','_'))(decoded_main)
    
    #UNFAM
    header_zone_key = 'main'
    header_zone = mdc.unfam_header_collector.mylist_grp[header_zone_key]
    
    default_layer = layer_producer('Dense')
    unfamed = default_layer(units=1, activation='sigmoid',
                            name=stamp_process('',['unfam',header_zone_key],'','','','_'))(decoded_main)
    lossmethods.append('BinaryCrossentropy')
    activations.append('sigmoid')
    
    mdc.keras_output = [output_main, unfamed]
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    return model

def building_ACWPred_v3(mdc, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    
    
    #INPUT
    virtual_weight_zones, input_LSTM_zones, input_temp_zones  = [], [], []
    #LSTM層
    for al in [1,2,3,5,6,7,8]:
        LSTM_layered = None
        input_LSTM_zones_AL = []
        for f in ['tp','rh']:
            k = LOGger.stamp_process('',['AL%d'%al, f],'','','','_',for_file=1)
            if(k in mdc.xheader_collector.mylist_grp):
            # if(True):
                header_zone = mdc.xheader_collector.mylist_grp[k]
                cell_size = getattr(header_zone, 'cell_size', None)
                input_shape = cell_size if(cell_size!=None) else len(header_zone)
                input_main_zone = Input(shape = input_shape, name=stamp_process('',['input',k],'','','','_'))
                input_LSTM_zones_AL.append(input_main_zone)
                input_LSTM_zones.append(input_main_zone) #for entrance
        if(input_LSTM_zones_AL!=[]):
            merged_LSTM_AL = layer_producer('concatenate')(input_LSTM_zones_AL)
            layer_index = 0
            mdc.addlog('mdc.hidden_layer_sizes:%s'%str(mdc.hidden_layer_sizes), stamps=['LSTM'])
            LSTM_layered, layer_index = stack_layers(layer_index, merged_LSTM_AL, activation=mdc.activation, 
                                                     default_layer_name = 'LSTM',
                                                     hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                                                     cell_size=cell_size, 
                                                     dropout_rates = getattr(mdc, 'dropout_rates', None),
                                                     addlog = addlog, stamps=['AL%d'%al])
            
        if('AL%d_weight'%al in mdc.xheader_collector.mylist_grp):
            header_zone = mdc.xheader_collector.mylist_grp['AL%d_weight'%al]
            input_shape = len(header_zone)
            input_temp_zone = Input(shape = input_shape, name=stamp_process('',['input','AL%d_weight'%al],'','','','_'))
            input_temp_zones.append(input_temp_zone) #for entrance
            LSTM_layered = layer_producer('concatenate')([input_temp_zone, LSTM_layered]) if(LSTM_layered!=None) else input_temp_zone
            LSTM_layered = layer_producer('Dense')(units=1, activation='linear',
                                                   name=stamp_process('',['input','AL%d_virtual_weight'%al],'','','','_'))(LSTM_layered)
        if(LSTM_layered!=None):
            virtual_weight_zones.append(LSTM_layered)
    
    # sys.exit(1)
    mdc.keras_input = input_temp_zones + input_LSTM_zones
    merged_LSTM = layer_producer('concatenate')(virtual_weight_zones)
    
    #OUTPUT
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'main'
    header_zone = mdc.yheader_collector.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'linear')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_main = default_layer(units=1, activation=activation,
                                 name=stamp_process('',['output','decode'],'','','','_'))(merged_LSTM)
    output_main = default_layer(units=output_shape, activation=getattr(mdc, 'output_activation', 'linear'),
                                 name=stamp_process('',['output','main'],'','','','_'))(decoded_main)
    
    #UNFAM
    header_zone_key = 'main'
    header_zone = mdc.unfam_header_collector.mylist_grp[header_zone_key]
    
    default_layer = layer_producer('Dense')
    unfamed = default_layer(units=1, activation='sigmoid',
                            name=stamp_process('',['unfam',header_zone_key],'','','','_'))(decoded_main)
    lossmethods.append('BinaryCrossentropy')
    activations.append('sigmoid')
    
    mdc.keras_output = [output_main, unfamed]
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    return model

def building_ACWPred_v2(mdc, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    
    
    #INPUT
    k = LOGger.stamp_process('',['weight'],'','','','_',for_file=1)
    header_zone = mdc.xheader_collector.mylist_grp[k]
    input_shape = len(header_zone)
    input_weight_zone = Input(shape = input_shape, name=stamp_process('',['input',k],'','','','_'))
    default_layer = layer_producer('Dense')
    weight_zone = default_layer(units=1, activation='linear', name=stamp_process('',['input',k,'1st'],'','','','_'))(input_weight_zone)
    virtual_weight_zones, input_LSTM_zones  = [], []
    #LSTM層
    for al in [1,2,3,5,6,7,8]:
        input_LSTM_zones_AL = []
        for f in ['tp','rh']:
            k = LOGger.stamp_process('',['AL%d'%al, f],'','','','_',for_file=1)
            header_zone = mdc.xheader_collector.mylist_grp[k]
            cell_size = getattr(header_zone, 'cell_size', None)
            input_shape = cell_size if(cell_size!=None) else len(header_zone)
            input_main_zone = Input(shape = input_shape, name=stamp_process('',['input',k],'','','','_'))
            input_LSTM_zones_AL.append(input_main_zone)
            input_LSTM_zones.append(input_main_zone) #for entrance
        merged_LSTM_AL = layer_producer('concatenate')(input_LSTM_zones_AL)
        layer_index = 0
        mdc.addlog('mdc.hidden_layer_sizes:%s'%str(mdc.hidden_layer_sizes), stamps=['LSTM'])
        LSTM_layered, layer_index = stack_layers(layer_index, merged_LSTM_AL, activation=mdc.activation, 
                                                      default_layer_name = 'LSTM',
                                                      hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                                                      cell_size=cell_size, 
                                                      dropout_rates = getattr(mdc, 'dropout_rates', None),
                                                      addlog = addlog, stamps=['AL%d'%al])
        virtual_weight_zones.append(LSTM_layered)
    
    mdc.keras_input = [input_weight_zone]+input_LSTM_zones
    merged_LSTM = layer_producer('concatenate')([weight_zone]+virtual_weight_zones)
    
    #OUTPUT
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'main'
    header_zone = mdc.yheader_collector.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'linear')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_main = default_layer(units=output_shape, activation=activation,
                                 name=stamp_process('',['output','decode'],'','','','_'))(merged_LSTM)
    output_main = default_layer(units=output_shape, activation=getattr(mdc, 'output_activation', 'linear'),
                                 name=stamp_process('',['output','main'],'','','','_'))(decoded_main)
    
    #UNFAM
    header_zone_key = 'main'
    header_zone = mdc.unfam_header_collector.mylist_grp[header_zone_key]
    
    default_layer = layer_producer('Dense')
    unfamed = default_layer(units=1, activation='sigmoid',
                            name=stamp_process('',['unfam',header_zone_key],'','','','_'))(decoded_main)
    lossmethods.append('BinaryCrossentropy')
    activations.append('sigmoid')
    
    mdc.keras_output = [output_main, unfamed]
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    return model

def building_ACWPred_v1(mdc, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    
    
    #INPUT
    virtual_weight_zones, input_LSTM_zones  = [], []
    #LSTM層
    for al in [1,2,3,5,6,7,8]:
        input_LSTM_zones_AL = []
        for f in ['tp','rh']:
            k = LOGger.stamp_process('',['AL%d'%al, f],'','','','_',for_file=1)
            header_zone = mdc.xheader_collector.mylist_grp[k]
            cell_size = getattr(header_zone, 'cell_size', None)
            input_shape = cell_size if(cell_size!=None) else len(header_zone)
            input_main_zone = Input(shape = input_shape, name=stamp_process('',['input',k],'','','','_'))
            input_LSTM_zones_AL.append(input_main_zone)
            input_LSTM_zones.append(input_main_zone) #for entrance
        merged_LSTM_AL = layer_producer('concatenate')(input_LSTM_zones_AL)
        layer_index = 0
        mdc.addlog('mdc.hidden_layer_sizes:%s'%str(mdc.hidden_layer_sizes), stamps=['LSTM'])
        LSTM_layered, layer_index = stack_layers(layer_index, merged_LSTM_AL, activation=mdc.activation, 
                                                      default_layer_name = 'LSTM',
                                                      hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                                                      cell_size=cell_size, 
                                                      dropout_rates = getattr(mdc, 'dropout_rates', None),
                                                      addlog = addlog, stamps=['AL%d'%al])
        virtual_weight_zones.append(LSTM_layered)
    
    mdc.keras_input = input_LSTM_zones
    merged_LSTM = layer_producer('concatenate')(virtual_weight_zones)
    
    #OUTPUT
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'main'
    header_zone = mdc.yheader_collector.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'linear')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_main = default_layer(units=output_shape, activation=activation,
                                 name=stamp_process('',['output','decode'],'','','','_'))(merged_LSTM)
    output_main = default_layer(units=output_shape, activation=getattr(mdc, 'output_activation', 'linear'),
                                 name=stamp_process('',['output','main'],'','','','_'))(decoded_main)
    
    #UNFAM
    header_zone_key = 'main'
    header_zone = mdc.unfam_header_collector.mylist_grp[header_zone_key]
    
    default_layer = layer_producer('Dense')
    unfamed = default_layer(units=1, activation='sigmoid',
                            name=stamp_process('',['unfam',header_zone_key],'','','','_'))(decoded_main)
    lossmethods.append('BinaryCrossentropy')
    activations.append('sigmoid')
    
    mdc.keras_output = [output_main, unfamed]
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    return model

def building_ACWPred_v0(mdc, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    
    
    #INPUT
    virtual_weight_zones, input_LSTM_zones = [], []
    #LSTM層
    for al in [1,2,3,5,6,7,8]:
        input_LSTM_zones_AL = []
        for f in ['tp','rh']:
            k = LOGger.stamp_process('',['AL%d'%al, f],'','','','_',for_file=1)
            header_zone = mdc.xheader_collector.mylist_grp[k]
            cell_size = getattr(header_zone, 'cell_size', None)
            input_shape = cell_size if(cell_size!=None) else len(header_zone)
            input_main_zone = Input(shape = input_shape, name=stamp_process('',['input',k],'','','','_'))
            input_LSTM_zones_AL.append(input_main_zone)
            input_LSTM_zones.append(input_main_zone) #for entrance
        merged_LSTM_AL = layer_producer('concatenate')(input_LSTM_zones_AL)
        layer_index = 0
        mdc.addlog('mdc.hidden_layer_sizes:%s'%str(mdc.hidden_layer_sizes), stamps=['LSTM'])
        LSTM_layered, layer_index = stack_layers(layer_index, merged_LSTM_AL, activation=mdc.activation, 
                                                      default_layer_name = 'LSTM',
                                                      hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                                                      cell_size=cell_size, 
                                                      dropout_rates = getattr(mdc, 'dropout_rates', None),
                                                      addlog = addlog, stamps=['AL%d'%al])
        
        merged_LSTM = layer_producer('concatenate')([virtual_weight_zones[-1], LSTM_layered]) if(al>1) else LSTM_layered
        output_AL = layer_producer('Dense')(units=1, activation=mdc.activation,
                                            name=stamp_process('',['output','AL%d'%al],'','','','_'))(merged_LSTM)
        virtual_weight_zones.append(output_AL)
    
    
    mdc.keras_input = input_LSTM_zones
    
    #OUTPUT
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'main'
    header_zone = mdc.yheader_collector.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'linear')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_main = default_layer(units=output_shape, activation=activation,
                                 name=stamp_process('',['output','decode'],'','','','_'))(merged_LSTM)
    output_main = default_layer(units=output_shape, activation=getattr(mdc, 'output_activation', 'linear'),
                                 name=stamp_process('',['output','main'],'','','','_'))(decoded_main)
    
    #UNFAM
    header_zone_key = 'main'
    header_zone = mdc.unfam_header_collector.mylist_grp[header_zone_key]
    
    default_layer = layer_producer('Dense')
    unfamed = default_layer(units=1, activation='sigmoid',
                            name=stamp_process('',['unfam',header_zone_key],'','','','_'))(decoded_main)
    lossmethods.append('BinaryCrossentropy')
    activations.append('sigmoid')
    
    mdc.keras_output = [output_main, unfamed]
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    return model

def building_dryingPred_v9(mdc, latent_dim = 2, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    
    
    
    k='main'
    header_zone = mdc.xheader_collector.mylist_grp[k]
    cell_size = getattr(header_zone, 'cell_size', None)
    input_shape = cell_size if(cell_size!=None) else len(header_zone)
    input_zone = Input(shape = input_shape, name=stamp_process('',[k],'','','','_'))
    layer_index = 0
    # mdc.addlog('mdc.hidden_layer_sizes:%s'%str(mdc.hidden_layer_sizes), stamps=['LSTM'])
    hidden_layered, layer_index = stack_layers(layer_index, input_zone, activation=mdc.activation, 
                                               default_layer_name = 'Dense',
                                               hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                                               cell_size=None, 
                                               dropout_rates = getattr(mdc, 'dropout_rates', None),
                                               addlog = addlog)
    
    k='tm'
    header_zone = mdc.xheader_collector.mylist_grp[k]
    cell_size = getattr(header_zone, 'cell_size', None)
    input_shape = cell_size if(cell_size!=None) else len(header_zone)
    input_tm_zone = Input(shape = input_shape, name=stamp_process('',[k],'','','','_'))
    
    mdc.keras_input = [input_zone, input_tm_zone]
    
    #OUTPUT
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'main'
    header_zone = mdc.yheader_collector.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'linear')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    
    
    decoded_main = default_layer(units=100, activation=mdc.activation,
                                 name=stamp_process('',['output','decode'],'','','','_'))(hidden_layered)
    
    
    
    
    merged1 = layer_producer('concatenate')([input_tm_zone, decoded_main])
    output_zone = layer_producer('ConditionalLogarithmOutputLayer')(units=output_shape, activation=activation, axis=0,
                                                                  name=stamp_process('',['output','nearzero'],'','','','_'))(merged1)
    
    #UNFAM
    header_zone_key = 'main'
    header_zone = mdc.unfam_header_collector.mylist_grp[header_zone_key]
    
    default_layer = layer_producer('Dense')
    unfamed = default_layer(units=1, activation='sigmoid',
                            name=stamp_process('',['unfam',header_zone_key],'','','','_'))(merged1)
    lossmethods.append('BinaryCrossentropy')
    activations.append('sigmoid')
    
    mdc.keras_output = [output_zone, unfamed]
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    return model

def building_dryingPred_v8(mdc, latent_dim = 2, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    
    
    
    k='main'
    header_zone = mdc.xheader_collector.mylist_grp[k]
    cell_size = getattr(header_zone, 'cell_size', None)
    input_shape = cell_size if(cell_size!=None) else len(header_zone)
    input_zone = Input(shape = input_shape, name=stamp_process('',[k],'','','','_'))
    layer_index = 0
    # mdc.addlog('mdc.hidden_layer_sizes:%s'%str(mdc.hidden_layer_sizes), stamps=['LSTM'])
    hidden_layered, layer_index = stack_layers(layer_index, input_zone, activation=mdc.activation, 
                                               default_layer_name = 'Dense',
                                               hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                                               cell_size=None, 
                                               dropout_rates = getattr(mdc, 'dropout_rates', None),
                                               addlog = addlog)
    
    k='tm'
    header_zone = mdc.xheader_collector.mylist_grp[k]
    cell_size = getattr(header_zone, 'cell_size', None)
    input_shape = cell_size if(cell_size!=None) else len(header_zone)
    input_tm_zone = Input(shape = input_shape, name=stamp_process('',[k],'','','','_'))
    
    mdc.keras_input = [input_zone, input_tm_zone]
    
    #OUTPUT
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'main'
    header_zone = mdc.yheader_collector.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'linear')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    
    
    decoded_main = default_layer(units=100, activation=mdc.activation,
                                 name=stamp_process('',['output','decode'],'','','','_'))(hidden_layered)
    
    
    
    
    merged1 = layer_producer('concatenate')([input_tm_zone, decoded_main])
    output_zone = layer_producer('ConditionalZeroFlatOutputLayer')(units=output_shape, activation=activation, axis=0,
                                                                  name=stamp_process('',['output','nearzero'],'','','','_'))(merged1)
    
    #UNFAM
    header_zone_key = 'main'
    header_zone = mdc.unfam_header_collector.mylist_grp[header_zone_key]
    
    default_layer = layer_producer('Dense')
    unfamed = default_layer(units=1, activation='sigmoid',
                            name=stamp_process('',['unfam',header_zone_key],'','','','_'))(merged1)
    lossmethods.append('BinaryCrossentropy')
    activations.append('sigmoid')
    
    mdc.keras_output = [output_zone, unfamed]
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    return model

def building_dryingPred_v7(mdc, latent_dim = 2, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    
    
    
    k='main'
    header_zone = mdc.xheader_collector.mylist_grp[k]
    cell_size = getattr(header_zone, 'cell_size', None)
    input_shape = cell_size if(cell_size!=None) else len(header_zone)
    input_zone = Input(shape = input_shape, name=stamp_process('',[k],'','','','_'))
    layer_index = 0
    # mdc.addlog('mdc.hidden_layer_sizes:%s'%str(mdc.hidden_layer_sizes), stamps=['LSTM'])
    hidden_layered, layer_index = stack_layers(layer_index, input_zone, activation=mdc.activation, 
                                               default_layer_name = 'Dense',
                                               hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                                               cell_size=None, 
                                               dropout_rates = getattr(mdc, 'dropout_rates', None),
                                               addlog = addlog)
    
    k='tm'
    header_zone = mdc.xheader_collector.mylist_grp[k]
    cell_size = getattr(header_zone, 'cell_size', None)
    input_shape = cell_size if(cell_size!=None) else len(header_zone)
    input_tm_zone = Input(shape = input_shape, name=stamp_process('',[k],'','','','_'))
    
    mdc.keras_input = [input_zone, input_tm_zone]
    
    #OUTPUT
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'main'
    header_zone = mdc.yheader_collector.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'linear')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    
    
    decoded_main = default_layer(units=100, activation=mdc.activation,
                                 name=stamp_process('',['output','decode'],'','','','_'))(hidden_layered)
    
    
    
    
    merged1 = layer_producer('concatenate')([input_tm_zone, decoded_main])
    output_zone = layer_producer('ConditionalLinear1OutputLayer')(units=output_shape, activation=activation, axis=0,
                                                                  name=stamp_process('',['output','nearzero'],'','','','_'))(merged1)
    
    #UNFAM
    header_zone_key = 'main'
    header_zone = mdc.unfam_header_collector.mylist_grp[header_zone_key]
    
    default_layer = layer_producer('Dense')
    unfamed = default_layer(units=1, activation='sigmoid',
                            name=stamp_process('',['unfam',header_zone_key],'','','','_'))(merged1)
    lossmethods.append('BinaryCrossentropy')
    activations.append('sigmoid')
    
    mdc.keras_output = [output_zone, unfamed]
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    return model

def building_dryingPred_v6(mdc, latent_dim = 2, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    
    
    
    k='main'
    header_zone = mdc.xheader_collector.mylist_grp[k]
    cell_size = getattr(header_zone, 'cell_size', None)
    input_shape = cell_size if(cell_size!=None) else len(header_zone)
    input_zone = Input(shape = input_shape, name=stamp_process('',[k],'','','','_'))
    layer_index = 0
    # mdc.addlog('mdc.hidden_layer_sizes:%s'%str(mdc.hidden_layer_sizes), stamps=['LSTM'])
    hidden_layered, layer_index = stack_layers(layer_index, input_zone, activation=mdc.activation, 
                                               default_layer_name = 'Dense',
                                               hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                                               cell_size=None, 
                                               dropout_rates = getattr(mdc, 'dropout_rates', None),
                                               addlog = addlog)
    
    k='tm'
    header_zone = mdc.xheader_collector.mylist_grp[k]
    cell_size = getattr(header_zone, 'cell_size', None)
    input_shape = cell_size if(cell_size!=None) else len(header_zone)
    input_tm_zone = Input(shape = input_shape, name=stamp_process('',[k],'','','','_'))
    
    mdc.keras_input = [input_zone, input_tm_zone]
    
    #OUTPUT
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'main'
    header_zone = mdc.yheader_collector.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'linear')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    
    
    decoded_main = default_layer(units=100, activation=mdc.activation,
                                 name=stamp_process('',['output','decode'],'','','','_'))(hidden_layered)
    
    
    
    
    merged1 = layer_producer('concatenate')([input_tm_zone, decoded_main])
    output_zone = layer_producer('ConditionalLinearCoefOutputLayer')(units=output_shape, activation=activation, axis=0,
                                                   name=stamp_process('',['output','nearzero'],'','','','_'))(merged1)
    
    #UNFAM
    header_zone_key = 'main'
    header_zone = mdc.unfam_header_collector.mylist_grp[header_zone_key]
    
    default_layer = layer_producer('Dense')
    unfamed = default_layer(units=1, activation='sigmoid',
                            name=stamp_process('',['unfam',header_zone_key],'','','','_'))(merged1)
    lossmethods.append('BinaryCrossentropy')
    activations.append('sigmoid')
    
    mdc.keras_output = [output_zone, unfamed]
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    return model

def building_dryingPred_v5(mdc, latent_dim = 2, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    
    
    
    k='main'
    header_zone = mdc.xheader_collector.mylist_grp[k]
    cell_size = getattr(header_zone, 'cell_size', None)
    input_shape = cell_size if(cell_size!=None) else len(header_zone)
    input_zone = Input(shape = input_shape, name=stamp_process('',[k],'','','','_'))
    layer_index = 0
    # mdc.addlog('mdc.hidden_layer_sizes:%s'%str(mdc.hidden_layer_sizes), stamps=['LSTM'])
    hidden_layered, layer_index = stack_layers(layer_index, input_zone, activation=mdc.activation, 
                                               default_layer_name = 'Dense',
                                               hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                                               cell_size=None, 
                                               dropout_rates = getattr(mdc, 'dropout_rates', None),
                                               addlog = addlog)
    
    k='tm'
    header_zone = mdc.xheader_collector.mylist_grp[k]
    cell_size = getattr(header_zone, 'cell_size', None)
    input_shape = cell_size if(cell_size!=None) else len(header_zone)
    input_tm_zone = Input(shape = input_shape, name=stamp_process('',[k],'','','','_'))
    
    mdc.keras_input = [input_zone, input_tm_zone]
    
    #OUTPUT
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'main'
    header_zone = mdc.yheader_collector.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'linear')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    
    
    decoded_main = default_layer(units=10, activation=mdc.activation,
                                 name=stamp_process('',['output','decode'],'','','','_'))(hidden_layered)
    
    
    
    
    merged1 = layer_producer('concatenate')([input_tm_zone, decoded_main])
    output_zone = layer_producer('ConditionalLinearOutputLayer')(units=output_shape, activation=activation, axis=0,
                                                   name=stamp_process('',['output','nearzero'],'','','','_'))(merged1)
    
    #UNFAM
    header_zone_key = 'main'
    header_zone = mdc.unfam_header_collector.mylist_grp[header_zone_key]
    
    default_layer = layer_producer('Dense')
    unfamed = default_layer(units=1, activation='sigmoid',
                            name=stamp_process('',['unfam',header_zone_key],'','','','_'))(merged1)
    lossmethods.append('BinaryCrossentropy')
    activations.append('sigmoid')
    
    mdc.keras_output = [output_zone, unfamed]
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    return model

def building_dryingPred_v4(mdc, latent_dim = 2, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    
    
    
    k='main'
    header_zone = mdc.xheader_collector.mylist_grp[k]
    cell_size = getattr(header_zone, 'cell_size', None)
    input_shape = cell_size if(cell_size!=None) else len(header_zone)
    input_zone = Input(shape = input_shape, name=stamp_process('',[k],'','','','_'))
    layer_index = 0
    # mdc.addlog('mdc.hidden_layer_sizes:%s'%str(mdc.hidden_layer_sizes), stamps=['LSTM'])
    hidden_layered, layer_index = stack_layers(layer_index, input_zone, activation=mdc.activation, 
                                               default_layer_name = 'Dense',
                                               hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                                               cell_size=None, 
                                               dropout_rates = getattr(mdc, 'dropout_rates', None),
                                               addlog = addlog)
    mdc.keras_input = input_zone
    
    #OUTPUT
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'main'
    header_zone = mdc.yheader_collector.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'linear')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    
    decoded_main = default_layer(units=output_shape, activation=mdc.activation,
                                 name=stamp_process('',['output','decode'],'','','','_'))(hidden_layered)
    
    merged = layer_producer('concatenate')([input_zone, decoded_main])
    output_zone = layer_producer('ConditionalLinearOutputLayer')(units=output_shape, activation=activation, axis=1,
                                                   name=stamp_process('',['output','nearzero'],'','','','_'))(merged)
    
    #UNFAM
    header_zone_key = 'main'
    header_zone = mdc.unfam_header_collector.mylist_grp[header_zone_key]
    
    default_layer = layer_producer('Dense')
    unfamed = default_layer(units=1, activation='sigmoid',
                            name=stamp_process('',['unfam',header_zone_key],'','','','_'))(decoded_main)
    lossmethods.append('BinaryCrossentropy')
    activations.append('sigmoid')
    
    mdc.keras_output = [output_zone, unfamed]
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    return model

def building_dryingPred_v3(mdc, latent_dim = 2, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    
    
    
    #LSTM層
    k='weightCurve'
    header_zone = mdc.xheader_collector.mylist_grp[k]
    cell_size = getattr(header_zone, 'cell_size', None)
    input_shape = cell_size if(cell_size!=None) else len(header_zone)
    input_LSTM_zone = Input(shape = input_shape, name=stamp_process('',[k],'','','','_'))
    layer_index = 0
    mdc.addlog('mdc.hidden_layer_sizes:%s'%str(mdc.hidden_layer_sizes), stamps=['LSTM'])
    main_LSTM_layered, layer_index = stack_layers(layer_index, input_LSTM_zone, activation=mdc.activation, 
                                                  default_layer_name = 'LSTM',
                                                  hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                                                  cell_size=getattr(mdc, 'cell_size', kwags.get('cell_size', None)), 
                                                  dropout_rates = getattr(mdc, 'dropout_rates', None),
                                                  addlog = addlog)
    #浸漿量
    k='enviroment'
    header_zone = mdc.xheader_collector.mylist_grp[k]
    input_shape = len(header_zone)
    input_main_zone = Input(shape = input_shape, name=stamp_process('',[k],'','','','_'))
    merged = layer_producer('concatenate')([input_main_zone, main_LSTM_layered])
    
    mdc.keras_input = [input_main_zone, input_LSTM_zone]
    
    #OUTPUT
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'main'
    header_zone = mdc.yheader_collector.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'linear')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    print('merged', merged)
    decoded_main = default_layer(units=output_shape, activation=mdc.activation,
                                 name=stamp_process('',['output','decode'],'','','','_'))(merged)
    # print('ma', header_zone.ma, 'mb', header_zone.mb)
    output_main = layer_producer('SequenceDenseLayer')(header_zone.cell_size, activation=activation, 
                                                         name=stamp_process('',['output','main'],'','','','_'))(decoded_main)
    # output_main = layer_producer('DryingRatioClipLayer')(ma=header_zone.ma, mb=header_zone.mb)(
    #     [input_main_zone, input_LSTM_zone, decoded_main])
    
    #UNFAM
    header_zone_key = 'main'
    header_zone = mdc.unfam_header_collector.mylist_grp[header_zone_key]
    
    default_layer = layer_producer('Dense')
    unfamed = default_layer(units=1, activation='sigmoid',
                            name=stamp_process('',['unfam',header_zone_key],'','','','_'))(decoded_main)
    lossmethods.append('BinaryCrossentropy')
    activations.append('sigmoid')
    
    mdc.keras_output = [output_main, unfamed]
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    return model

def building_dryingPred_v2(mdc, latent_dim = 2, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    
    
    
    #LSTM層
    k='weightCurve'
    header_zone = mdc.xheader_collector.mylist_grp[k]
    cell_size = getattr(header_zone, 'cell_size', None)
    input_shape = cell_size if(cell_size!=None) else len(header_zone)
    input_LSTM_zone = Input(shape = input_shape, name=stamp_process('',[k],'','','','_'))
    layer_index = 0
    mdc.addlog('mdc.hidden_layer_sizes:%s'%str(mdc.hidden_layer_sizes), stamps=['LSTM'])
    main_LSTM_layered, layer_index = stack_layers(layer_index, input_LSTM_zone, activation=mdc.activation, 
                                                  default_layer_name = 'LSTM',
                                                  hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                                                  cell_size=getattr(mdc, 'cell_size', kwags.get('cell_size', None)), 
                                                  dropout_rates = getattr(mdc, 'dropout_rates', None),
                                                  addlog = addlog)
    #浸漿量
    k='enviroment'
    header_zone = mdc.xheader_collector.mylist_grp[k]
    input_shape = len(header_zone)
    input_main_zone = Input(shape = input_shape, name=stamp_process('',[k],'','','','_'))
    merged = layer_producer('concatenate')([input_main_zone, main_LSTM_layered])
    
    mdc.keras_input = [input_main_zone, input_LSTM_zone]
    
    #OUTPUT
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'main'
    header_zone = mdc.yheader_collector.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'linear')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    print('merged', merged)
    decoded_main = default_layer(units=output_shape, activation=mdc.activation,
                                 name=stamp_process('',['output','decode'],'','','','_'))(merged)
    # print('ma', header_zone.ma, 'mb', header_zone.mb)
    output_main = layer_producer('DryingRatioClipLayer')(ma=header_zone.ma, mb=header_zone.mb)(
        [input_main_zone, input_LSTM_zone, decoded_main])
    
    #UNFAM
    header_zone_key = 'main'
    header_zone = mdc.unfam_header_collector.mylist_grp[header_zone_key]
    
    default_layer = layer_producer('Dense')
    unfamed = default_layer(units=1, activation='sigmoid',
                            name=stamp_process('',['unfam',header_zone_key],'','','','_'))(decoded_main)
    lossmethods.append('BinaryCrossentropy')
    activations.append('sigmoid')
    
    mdc.keras_output = [output_main, unfamed]
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    return model

def building_dryingPred_v1(mdc, latent_dim = 2, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    
    
    
    #LSTM層
    k='weightCurve'
    header_zone = mdc.xheader_collector.mylist_grp[k]
    cell_size = getattr(header_zone, 'cell_size', None)
    input_shape = cell_size if(cell_size!=None) else len(header_zone)
    input_LSTM_zone = Input(shape = input_shape, name=stamp_process('',[k],'','','','_'))
    layer_index = 0
    mdc.addlog('mdc.hidden_layer_sizes:%s'%str(mdc.hidden_layer_sizes), stamps=['LSTM'])
    main_LSTM_layered, layer_index = stack_layers(layer_index, input_LSTM_zone, activation=mdc.activation, 
                                                  default_layer_name = 'LSTM',
                                                  hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                                                  cell_size=getattr(mdc, 'cell_size', kwags.get('cell_size', None)), 
                                                  dropout_rates = getattr(mdc, 'dropout_rates', None),
                                                  addlog = addlog)
    #浸漿量
    k='enviroment'
    header_zone = mdc.xheader_collector.mylist_grp[k]
    input_shape = len(header_zone)
    input_main_zone = Input(shape = input_shape, name=stamp_process('',[k],'','','','_'))
    merged = layer_producer('concatenate')([input_main_zone, main_LSTM_layered])
    
    mdc.keras_input = [input_main_zone, input_LSTM_zone]
    
    #OUTPUT
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'main'
    header_zone = mdc.yheader_collector.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'linear')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    print('merged', merged)
    decoded_main = default_layer(units=100, activation=mdc.activation,
                                 name=stamp_process('',['output','decode'],'','','','_'))(merged)
    output_main = default_layer(units=output_shape, activation=activation,
                                 name=stamp_process('',['output','main'],'','','','_'))(decoded_main)
    
    #UNFAM
    header_zone_key = 'main'
    header_zone = mdc.unfam_header_collector.mylist_grp[header_zone_key]
    
    default_layer = layer_producer('Dense')
    unfamed = default_layer(units=1, activation='sigmoid',
                            name=stamp_process('',['unfam',header_zone_key],'','','','_'))(decoded_main)
    lossmethods.append('BinaryCrossentropy')
    activations.append('sigmoid')
    
    mdc.keras_output = [output_main, unfamed]
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    return model

def building_dryingPred_v0(mdc, latent_dim = 2, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    
    
    #INPUT
    input_main_zones, input_LSTM_zones, main_layereds = [], [], []
    #LSTM層
    for k in ['tp','rh']:
        header_zone = mdc.xheader_collector.mylist_grp[k]
        cell_size = getattr(header_zone, 'cell_size', None)
        input_shape = cell_size if(cell_size!=None) else len(header_zone)
        input_main_zone = Input(shape = input_shape, name=stamp_process('',['input',k],'','','','_'))
        input_LSTM_zones.append(input_main_zone)
    merged_LSTM = layer_producer('concatenate')(input_LSTM_zones)
    layer_index = 0
    mdc.addlog('mdc.hidden_layer_sizes:%s'%str(mdc.hidden_layer_sizes), stamps=['LSTM'])
    main_LSTM_layered, layer_index = stack_layers(layer_index, merged_LSTM, activation=mdc.activation, 
                                                  default_layer_name = 'LSTM',
                                                  hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                                                  cell_size=getattr(mdc, 'cell_size', kwags.get('cell_size', None)), 
                                                  dropout_rates = getattr(mdc, 'dropout_rates', None),
                                                  addlog = addlog)
    #浸漿量
    k='impreg'
    header_zone = mdc.xheader_collector.mylist_grp[k]
    input_shape = len(header_zone)
    input_main_zone = Input(shape = input_shape, name=stamp_process('',['input',k],'','','','_'))
    merged = layer_producer('concatenate')([input_main_zone, main_LSTM_layered])
    
    mdc.keras_input = [input_main_zone] + input_LSTM_zones
    
    #OUTPUT
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'main'
    header_zone = mdc.yheader_collector.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'linear')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_main = default_layer(units=output_shape, activation=activation,
                                 name=stamp_process('',['output','decode'],'','','','_'))(merged)
    output_main = default_layer(units=output_shape, activation=getattr(mdc, 'output_activation', 'linear'),
                                 name=stamp_process('',['output','main'],'','','','_'))(decoded_main)
    
    #UNFAM
    header_zone_key = 'main'
    header_zone = mdc.unfam_header_collector.mylist_grp[header_zone_key]
    
    default_layer = layer_producer('Dense')
    unfamed = default_layer(units=1, activation='sigmoid',
                            name=stamp_process('',['unfam',header_zone_key],'','','','_'))(decoded_main)
    lossmethods.append('BinaryCrossentropy')
    activations.append('sigmoid')
    
    mdc.keras_output = [output_main, unfamed]
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    return model

def sg_find_middle_edge_v0(mdc, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    
    header_zone_key = 'main'
    header_zone = mdc.xheader_collector.mylist_grp[header_zone_key]
    input_shape = getattr(header_zone, 'cell_size')
    input_main_zone = Input(shape=input_shape, name=stamp_process('',['input',header_zone_key],'','','','_'))
    mdc.keras_input = input_main_zone
    
    # 搭建卷積層
    x = layer_producer('Conv2D')(32, (3, 3), activation='selu')(input_main_zone)
    x = layer_producer('MaxPooling2D')((2, 2))(x)
    x = layer_producer('Conv2D')(64, (3, 3), activation='selu')(x)
    x = layer_producer('MaxPooling2D')((2, 2))(x)
    x = layer_producer('Conv2D')(128, (3, 3), activation='selu')(x)
    x = layer_producer('MaxPooling2D')((2, 2))(x)
    
    # 展平層和全連接層
    x = layer_producer('Flatten')()(x)
    pre_output_main = layer_producer('Dense')(128, activation='selu')(x)
    
    
    header_zone_key = 'main'
    header_zone = mdc.yheader_collector.mylist_grp[header_zone_key]
    activations, lossmethods = mylist(), mylist()
    output_main_activation = 'linear'#header_zone.get('activation', 'linear')
    output_main = layer_producer('Dense')(input_shape[1], activation=output_main_activation)(pre_output_main)
    lossmethods.append('mse')
    mdc.addlog('y activation:%s'%output_main_activation)
    activations.append(output_main_activation)
    
    header_zone_key = 'main'
    header_zone = mdc.unfam_header_collector.mylist_grp[header_zone_key]
    
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    unfamed = default_layer(units=output_shape, activation='sigmoid',
                            name=stamp_process('',['unfam',header_zone_key],'','','','_'))(pre_output_main)
    lossmethods.append('BinaryCrossentropy')
    activations.append('sigmoid')
    
    mdc.keras_output = [output_main, unfamed]
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    return model
    

def impreg_NGfiber_v0(mdc, latent_dim = 2, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    
    
    cell_size = getattr(mdc, 'cell_size', None)
    header_zone_key = 'main'
    header_zone = mdc.xheader_collector.mylist_grp[header_zone_key]
    input_shape = getattr(header_zone, 'input_shape')
    
    input_main_zone = Input(shape = input_shape, name=stamp_process('',['input',header_zone_key],'','','','_'))
    mdc.keras_input = input_main_zone
    
    layer_index = 0
    mdc.addlog('mdc.hidden_layer_sizes:%s'%str(mdc.hidden_layer_sizes))
    
    main_layered, layer_index = stack_layers(layer_index, input_main_zone, activation=mdc.activation, 
                default_layer_name = 'Conv1D', hidden_layer_sizes=mdc.hidden_layer_sizes, 
                dropout_rates = getattr(mdc, 'dropout_rates', None),kernel_sizes=getattr(mdc, 'kernel_sizes', None),
                hidden_layer_nns=mdc.hidden_layer_nns, maxpool2D_sizes = None, addlog = addlog)
    main_layered = layer_producer('Flatten')()(main_layered)
    
    
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'main'
    header_zone = mdc.yheader_collector.mylist_grp[header_zone_key]
    output_shape = 1
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'sigmoid')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, 
            default=getattr(mdc, 'lossmethod', 'binary_crossentropy')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_main = default_layer(units=output_shape, activation=activation,
                                 name=stamp_process('',['output',header_zone_key],'','','','_'))(main_layered)
    
    
    unfam_layer_stacking(mdc, main_layered, [decoded_main]) if(
        not isinstance(getattr(mdc, 'unfam_header', None), type(None))) else None
    
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    return model

def building_vibrafreq_v0(mdc, latent_dim = 2, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    
    
    cell_size = getattr(mdc, 'cell_size', None)
    header_zone_key = 'main'
    header_zone = mdc.xheader_collector.mylist_grp[header_zone_key]
    input_shape = getattr(header_zone, 'input_shape')
    
    input_main_zone = Input(shape = input_shape, name=stamp_process('',['input',header_zone_key],'','','','_'))
    mdc.keras_input = input_main_zone
    
    layer_index = 0
    mdc.addlog('mdc.hidden_layer_sizes:%s'%str(mdc.hidden_layer_sizes))
    
    main_layered, layer_index = stack_layers(layer_index, input_main_zone, activation=mdc.activation, 
                default_layer_name = 'Conv1D', hidden_layer_sizes=mdc.hidden_layer_sizes, 
                dropout_rates = getattr(mdc, 'dropout_rates', None),kernel_sizes=getattr(mdc, 'kernel_sizes', None),
                hidden_layer_nns=mdc.hidden_layer_nns, maxpool2D_sizes = None, addlog = addlog)
    main_layered = layer_producer('Flatten')()(main_layered)
    
    
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'main'
    header_zone = mdc.yheader_collector.mylist_grp[header_zone_key]
    output_shape = 1
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'sigmoid')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, 
            default=getattr(mdc, 'lossmethod', 'binary_crossentropy')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_main = default_layer(units=output_shape, activation=activation,
                                 name=stamp_process('',['output',header_zone_key],'','','','_'))(main_layered)
    
    
    unfam_layer_stacking(mdc, main_layered, [decoded_main]) if(
        not isinstance(getattr(mdc, 'unfam_header', None), type(None))) else None
    
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    return model

def building_fibermonitor_v0(mdc, latent_dim = 2, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    
    
    cell_size = getattr(mdc, 'cell_size', None)
    header_zone_key = 'main'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = getattr(header_zone, 'input_shape')
    
    input_main_zone = Input(shape = input_shape, name=stamp_process('',['input',header_zone_key],'','','','_'))
    layer_index = 0
    mdc.addlog('mdc.hidden_layer_sizes:%s'%str(mdc.hidden_layer_sizes))
    
    main_layered, layer_index = stack_layers(layer_index, input_main_zone, activation=mdc.activation, 
                default_layer_name = 'Conv2D', hidden_layer_sizes=mdc.hidden_layer_sizes, 
                dropout_rates = getattr(mdc, 'dropout_rates', None), kernel_sizes=getattr(mdc, 'kernel_sizes', None),
                hidden_layer_nns=mdc.hidden_layer_nns, maxpool2D_sizes = [], addlog = addlog)
    
    mdc.keras_input = input_main_zone
    
    
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'main'
    header_zone = mdc.yheader.mylist_grp[header_zone_key]
    output_shape = 1
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'sigmoid')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, 
            default=getattr(mdc, 'lossmethod', 'binary_crossentropy')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_main = default_layer(units=output_shape, activation=activation,
                                 name=stamp_process('',['output',header_zone_key],'','','','_'))(main_layered)
    
    
    unfam_layer_stacking(mdc, main_layered, [decoded_main]) if(
        not isinstance(getattr(mdc, 'unfam_header', None), type(None))) else None
    
    # header_zone_key = 'main'
    # header_zone = mdc.unfam_header.mylist_grp[header_zone_key]
    # output_shape = len(header_zone)
    # default_layer = layer_producer('Dense')
    # unfamed = default_layer(units=output_shape, activation='sigmoid',
    #                             name=stamp_process('',['unfam',header_zone_key],'','','','_'))(main_layered)
    # mdc.keras_output = [decoded_main, unfamed]
    
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    # mdc.lossmethods = lossmethods
    # mdc.output_activations = activations
    return model

def building_vibration_v0(mdc, latent_dim = 2, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    
    
    cell_size = getattr(mdc, 'cell_size', None)
    header_zone_key = 'main'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = (cell_size, len(header_zone)) if(isinstance(cell_size, int)) else len(header_zone)
    input_main_zone = Input(shape = input_shape, name=stamp_process('',['input',header_zone_key],'','','','_'))
    layer_index = 0
    mdc.addlog('mdc.hidden_layer_sizes:%s'%str(mdc.hidden_layer_sizes))
    main_layered, layer_index = stack_layers(layer_index, input_main_zone, activation=mdc.activation, 
                default_layer_name = ('Dense' if(getattr(mdc, 'cell_size', None)==None) else 'LSTM'),
                hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                cell_size=getattr(mdc, 'cell_size', kwags.get('cell_size', None)), 
                dropout_rates = getattr(mdc, 'dropout_rates', None),
                addlog = addlog)
    
    mdc.keras_input = input_main_zone
    
    
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'main'
    header_zone = mdc.yheader.mylist_grp[header_zone_key]
    output_shape = len(header_zone.preprocessor.classes_)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'sigmoid')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'binary_crossentropy')))
    # activation = 'sigmoid'
    # lossmethod = 'binary_crossentropy'
    activation = 'softmax'
    lossmethod = 'sparse_categorical_crossentropy'
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_main = default_layer(units=output_shape, activation=activation,
                                 name=stamp_process('',['output',header_zone_key],'','','','_'))(main_layered)
    
    mdc.keras_output = decoded_main
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    return model

def building_impreg_defect_detection_v0(mdc, latent_dim = 2, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    
    
    cell_size = getattr(mdc, 'cell_size', None)
    header_zone_key = 'main'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = (cell_size, len(header_zone)) if(isinstance(cell_size, int)) else len(header_zone)
    input_main_zone = Input(shape = input_shape, name=stamp_process('',['input',header_zone_key],'','','','_'))
    layer_index = 0
    main_layered, layer_index = stack_layers(layer_index, input_main_zone, activation=mdc.activation, 
                default_layer_name = ('Dense' if(getattr(mdc, 'cell_size', None)==None) else 'LSTM'),
                hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                cell_size=getattr(mdc, 'cell_size', kwags.get('cell_size', None)), 
                dropout_rates = getattr(mdc, 'dropout_rates', None),
                addlog = addlog)
    
    header_zone_key = 'condition'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = len(header_zone)
    input_cond_zone = Input(shape = input_shape, name=stamp_process('',['input','condition'],'','','','_'))
    
    merged = layer_producer('concatenate')([main_layered, input_cond_zone])
    
    mdc.keras_input = [input_main_zone, input_cond_zone]
    
    
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'main'
    header_zone = mdc.yheader.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'sigmoid')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'binary_crossentropy')))
    activation = 'sigmoid'
    lossmethod = 'binary_crossentropy'
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_main = default_layer(units=output_shape, activation=activation,
                                 name=stamp_process('',['output',header_zone_key],'','','','_'))(merged)
    
    
    merged_2 = layer_producer('concatenate')([decoded_main, merged])
    
    
    
    header_zone_key = 'loc'
    header_zone = mdc.yheader.mylist_grp[header_zone_key]
    output_shape = len(header_zone.all_categories)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'softmax')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'sparse_categorical_crossentropy')))
    activation = 'softmax'
    lossmethod = 'sparse_categorical_crossentropy'
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_boundingbox_start = default_layer(units=output_shape, activation=activation,
                                name=stamp_process('',['output',header_zone_key],'','','','_'))(merged_2)
    
    header_zone_key = 'lgn'
    header_zone = mdc.yheader.mylist_grp[header_zone_key]
    output_shape = len(header_zone.all_categories)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'softmax')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'sparse_categorical_crossentropy')))
    activation = 'softmax'
    lossmethod = 'sparse_categorical_crossentropy'
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_boundingbox_lgn = default_layer(units=output_shape, activation=activation,
                                            name=stamp_process('',['output',header_zone_key],'','','','_'))(merged_2)
    
    
    mdc.keras_output = [decoded_main, decoded_boundingbox_start, decoded_boundingbox_lgn]
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    return model

def building_impreg_coating_v0(mdc, latent_dim = 2, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    
    cell_size = getattr(mdc, 'cell_size', None)
    header_zone_key = 'main'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = (cell_size, len(header_zone)) if(isinstance(cell_size, int)) else len(header_zone)
    input_main_zone = Input(shape = input_shape, name=stamp_process('',['input',header_zone_key],'','','','_'))
    layer_index = 0
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    main_layered, layer_index = stack_layers(layer_index, input_main_zone, activation=mdc.activation, 
                default_layer_name = ('Dense' if(getattr(mdc, 'cell_size', None)==None) else 'LSTM'),
                hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                cell_size=getattr(mdc, 'cell_size', kwags.get('cell_size', None)), 
                dropout_rates = getattr(mdc, 'dropout_rates', None),
                addlog = addlog)
    
    header_zone_key = 'condition'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = len(header_zone)
    input_cond_zone = Input(shape = input_shape, name=stamp_process('',['input','condition'],'','','','_'))
    
    merged = layer_producer('concatenate')([main_layered, input_cond_zone])
    
    mdc.keras_input = [input_main_zone, input_cond_zone]
    
    
    activations, lossmethods = mylist(), mylist()
    
    
    header_zone_key = 'bypos'
    header_zone = mdc.yheader.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'relu')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_bypos = default_layer(units=output_shape, activation=getattr(mdc, 'output_activation', 'linear'),
                                name=stamp_process('',['output',header_zone_key],'','','','_'))(merged)
    
    header_zone_key = 'main'
    header_zone = mdc.yheader.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'relu')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_main = default_layer(units=output_shape, activation=getattr(mdc, 'output_activation', 'linear'),
                                name=stamp_process('',['output',header_zone_key],'','','','_'))(decoded_bypos)
    
    
    
    
    
    mdc.keras_output = [decoded_main, decoded_bypos]
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    
    
    # header_zone_key = 'main'
    # decoded_main = mdc.autoencoder.get_layer(stamp_process('',['output',header_zone_key],'','','','_'))(encoded_input)
    # mdc.decoder = Model(encoded_input, decoded_main, name='decoder')
    mdc.model = model
    return model

def building_impreg_coating_v1(mdc, latent_dim = 2, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    
    cell_size = getattr(mdc, 'cell_size', None)
    header_zone_key = 'main'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = (cell_size, len(header_zone)) if(isinstance(cell_size, int)) else len(header_zone)
    input_main_zone = Input(shape = input_shape, name=stamp_process('',['input',header_zone_key],'','','','_'))
    layer_index = 0
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    main_layered, layer_index = stack_layers(layer_index, input_main_zone, activation=mdc.activation, 
                default_layer_name = ('Dense' if(getattr(mdc, 'cell_size', None)==None) else 'LSTM'),
                hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                cell_size=getattr(mdc, 'cell_size', kwags.get('cell_size', None)), 
                dropout_rates = getattr(mdc, 'dropout_rates', None),
                addlog = addlog)
    
    header_zone_key = 'condition'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = len(header_zone)
    input_cond_zone = Input(shape = input_shape, name=stamp_process('',['input','condition'],'','','','_'))
    
    merged = layer_producer('concatenate')([main_layered, input_cond_zone])
    
    mdc.keras_input = [input_main_zone, input_cond_zone]
    
    
    activations, lossmethods = mylist(), mylist()
    
    
    header_zone_key = 'bypos'
    header_zone = mdc.yheader.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'relu')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_bypos = default_layer(units=output_shape, activation=getattr(mdc, 'output_activation', 'linear'),
                                name=stamp_process('',['output',header_zone_key],'','','','_'))(merged)
    
    header_zone_key = 'main'
    header_zone = mdc.yheader.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'relu')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_main = default_layer(units=output_shape, activation=getattr(mdc, 'output_activation', 'linear'),
                                name=stamp_process('',['output',header_zone_key],'','','','_'))(merged)
    
    
    
    
    
    mdc.keras_output = [decoded_main, decoded_bypos]
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    
    
    # header_zone_key = 'main'
    # decoded_main = mdc.autoencoder.get_layer(stamp_process('',['output',header_zone_key],'','','','_'))(encoded_input)
    # mdc.decoder = Model(encoded_input, decoded_main, name='decoder')
    mdc.model = model
    return model

def building_impreg_temptime_uRT_ab_v0(mdc, latent_dim = 1, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    layer_index = 0
    standard_input_setting(mdc, **kwags)
    inputs_layer = mdc.inputs_layer
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    layer, layer_index = stack_layers(layer_index, inputs_layer, activation=mdc.activation, 
                default_layer_name = ('Dense' if(getattr(mdc, 'cell_size', None)==None) else 'LSTM'),
                hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                cell_size=getattr(mdc, 'cell_size', kwags.get('cell_size', None)), 
                dropout_rates = getattr(mdc, 'dropout_rates', None),
                default_outreturn_sequences = True,
                addlog = addlog)
    if(layer==None):
        return None
    layer_index += 1
    encoded = layer_producer('LSTM')(units = int(latent_dim//1), activation='linear', name='coding-layer',
                                   return_sequences=False)(layer)
    mdc.encoder = Model(inputs_layer, encoded, name='encoder')
    encoded_input = Input(shape=(int(latent_dim//1),))
    layer_index += 1
    keras_output = mylist()
    lossmethods, output_activations = LOGger.mylist(), LOGger.mylist()
    cell_size = dcp(getattr(mdc, 'cell_size', None))
    for i,(k,v) in enumerate(mdc.xheader.mylist_grp.items()):
        if(cell_size==None):
            decode_output_shape = len(v) 
        else:
            if(DFP.isiterable(cell_size)):
                decode_output_shape = (*cell_size, len(v))
            else:
                decode_output_shape = (cell_size, len(v))
                
        default_layer = layer_producer('LSTM')
        activation = getattr(v,'activation',LOGger.extract(
                getattr(mdc, 'activations', []), index=i, key=k, default=getattr(mdc, 'activation', 'relu')))
        output_activations.append(activation)
        decoder_input_layer = layer_producer('RepeatVector')(decode_output_shape[0])(encoded)
        output_zone = default_layer(units=decode_output_shape[1], activation=activation, return_sequences=True, 
                                    name=stamp_process('',['output',k],'','','','_'))(
                                        decoder_input_layer)
        lossmethods.append(v.lossmethod)
        keras_output.append(output_zone)
    addlog('keras_output:%s'%stamp_process('', stamps = list(map(str, keras_output)), 
                                               stamp_left='\n', stamp_right=''), **kwags)
    mdc.lossmethods = lossmethods
    addlog('operating lossmethods:%s'%(str(mdc.lossmethods.get())), **kwags)
    mdc.output_activations = output_activations
    addlog('operating output_activations:%s'%(str(output_activations.get())), **kwags)
    mdc.keras_output = keras_output if(len(keras_output)>1) else keras_output[0]
    mdc.model_form = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    decoded = None
    for l in mdc.model_form.layers[-2:]:    decoded = l(encoded_input if(decoded==None) else decoded)
    mdc.decoder = Model(encoded_input, decoded, name='decoder')
    
    return mdc.model_form


def building_impreg_temptime_uRT_v0(mdc, latent_dim = 2, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    
    cell_size = getattr(mdc, 'cell_size', None)
    header_zone_key = 'main'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = (cell_size, len(header_zone)) if(isinstance(cell_size, int)) else len(header_zone)
    input_main_zone = Input(shape = input_shape, name=stamp_process('',['input',header_zone_key],'','','','_'))
    layer_index = 0
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    main_layered, layer_index = stack_layers(layer_index, input_main_zone, activation=mdc.activation, 
                default_layer_name = ('Dense' if(getattr(mdc, 'cell_size', None)==None) else 'LSTM'),
                hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                cell_size=getattr(mdc, 'cell_size', kwags.get('cell_size', None)), 
                dropout_rates = getattr(mdc, 'dropout_rates', None),
                addlog = addlog)
    
    header_zone_key = 'condition'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = len(header_zone)
    input_cond_zone = Input(shape = input_shape, name=stamp_process('',['input','condition'],'','','','_'))
    
    merged = layer_producer('concatenate')([main_layered, input_cond_zone])
    # print("getattr(mdc, 'output_activation', 'sigmoid'):%s"%getattr(mdc, 'output_activation', 'sigmoid'))
    # encoded = layer_producer('Dense')(
    #     units = latent_dim, activation=getattr(mdc, 'latent_activation', 'linear'), name='coding-layer')(merged)
    
    mdc.keras_input = [input_main_zone, input_cond_zone]
    # mdc.encoder = Model(mdc.keras_input, encoded, name='encoder')
    # encoded_input = Input(shape=(int(latent_dim//1),))
    
    
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'main'
    header_zone = mdc.yheader.mylist_grp[header_zone_key]
    decoded_main = merged
    
    
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'relu')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_main = default_layer(units=output_shape, activation=getattr(mdc, 'output_activation', 'linear'),
                                name=stamp_process('',['output',header_zone_key],'','','','_'))(decoded_main)
    mdc.keras_output = decoded_main
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    
    
    # header_zone_key = 'main'
    # decoded_main = mdc.autoencoder.get_layer(stamp_process('',['output',header_zone_key],'','','','_'))(encoded_input)
    # mdc.decoder = Model(encoded_input, decoded_main, name='decoder')
    mdc.model = model
    return model

def building_impreg_temptime_uRT_v1(mdc, latent_dim = 2, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    
    cell_size = getattr(mdc, 'cell_size', None)
    header_zone_key = 'main'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = (cell_size, len(header_zone)) if(isinstance(cell_size, int)) else len(header_zone)
    input_main_zone = Input(shape = input_shape, name=stamp_process('',['input',header_zone_key],'','','','_'))
    layer_index = 0
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    main_layered, layer_index = stack_layers(layer_index, input_main_zone, activation=mdc.activation, 
                default_layer_name = ('Dense' if(getattr(mdc, 'cell_size', None)==None) else 'LSTM'),
                hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                cell_size=getattr(mdc, 'cell_size', kwags.get('cell_size', None)), 
                dropout_rates = getattr(mdc, 'dropout_rates', None),
                addlog = addlog)
    
    header_zone_key = 'condition'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = len(header_zone)
    input_cond_zone = Input(shape = input_shape, name=stamp_process('',['input','condition'],'','','','_'))
    
    merged = layer_producer('concatenate')([main_layered, input_cond_zone])
    # print("getattr(mdc, 'output_activation', 'sigmoid'):%s"%getattr(mdc, 'output_activation', 'sigmoid'))
    # encoded = layer_producer('Dense')(
    #     units = latent_dim, activation=getattr(mdc, 'latent_activation', 'linear'), name='coding-layer')(merged)
    
    mdc.keras_input = [input_main_zone, input_cond_zone]
    # mdc.encoder = Model(mdc.keras_input, encoded, name='encoder')
    # encoded_input = Input(shape=(int(latent_dim//1),))
    
    decoded_state = merged
    activations, lossmethods = mylist(), mylist()
        
    header_zone_key = 'state'
    header_zone = mdc.yheader.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'relu')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_state = default_layer(units=output_shape, activation=getattr(mdc, 'state_output_activation', 'sigmoid'),
                                name=stamp_process('',['output',header_zone_key],'','','','_'))(decoded_state)
    
    header_zone_key = 'main'
    header_zone = mdc.yheader.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'relu')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    
    decoded_main = default_layer(units=output_shape, activation=getattr(mdc, 'output_activation', 'linear'),
                                name=stamp_process('',['output',header_zone_key],'','','','_'))(decoded_state)
    
    
    
    
    mdc.keras_output = [decoded_main, decoded_state]
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    
    
    # header_zone_key = 'main'
    # decoded_main = mdc.autoencoder.get_layer(stamp_process('',['output',header_zone_key],'','','','_'))(encoded_input)
    # mdc.decoder = Model(encoded_input, decoded_main, name='decoder')
    mdc.model = model
    return model

def building_impreg_temptime_uRT_v2(mdc, latent_dim = 2, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    
    cell_size = getattr(mdc, 'cell_size', None)
    header_zone_key = 'main'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = (cell_size, len(header_zone)) if(isinstance(cell_size, int)) else len(header_zone)
    input_main_zone = Input(shape = input_shape, name=stamp_process('',['input',header_zone_key],'','','','_'))
    layer_index = 0
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    main_layered, layer_index = stack_layers(layer_index, input_main_zone, activation=mdc.activation, 
                default_layer_name = ('Dense' if(getattr(mdc, 'cell_size', None)==None) else 'LSTM'),
                hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                cell_size=getattr(mdc, 'cell_size', kwags.get('cell_size', None)), 
                dropout_rates = getattr(mdc, 'dropout_rates', None),
                addlog = addlog)
    
    header_zone_key = 'condition'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = len(header_zone)
    input_cond_zone = Input(shape = input_shape, name=stamp_process('',['input','condition'],'','','','_'))
    
    merged = layer_producer('concatenate')([main_layered, input_cond_zone])
    # print("getattr(mdc, 'output_activation', 'sigmoid'):%s"%getattr(mdc, 'output_activation', 'sigmoid'))
    # encoded = layer_producer('Dense')(
    #     units = latent_dim, activation=getattr(mdc, 'latent_activation', 'linear'), name='coding-layer')(merged)
    
    mdc.keras_input = [input_main_zone, input_cond_zone]
    # mdc.encoder = Model(mdc.keras_input, encoded, name='encoder')
    # encoded_input = Input(shape=(int(latent_dim//1),))
    
    decoded_state = merged
    activations, lossmethods = mylist(), mylist()
        
    header_zone_key = 'state'
    header_zone = mdc.yheader.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'relu')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    mdc.addlog("getattr(mdc, 'state_output_activation', 'sigmoid'):%s"%str(getattr(mdc, 'state_output_activation', 'sigmoid')))
    decoded_state = default_layer(units=output_shape, activation=getattr(mdc, 'state_output_activation', 'sigmoid'),
                                name=stamp_process('',['output',header_zone_key],'','','','_'))(decoded_state)
    
    merged_with_state = layer_producer('concatenate')([merged, decoded_state])
    
    header_zone_key = 'main'
    header_zone = mdc.yheader.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'relu')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    
    decoded_main = default_layer(units=output_shape, activation=getattr(mdc, 'output_activation', 'linear'),
                                name=stamp_process('',['output',header_zone_key],'','','','_'))(merged_with_state)
    
    
    
    
    mdc.keras_output = [decoded_main, decoded_state]
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    
    
    # header_zone_key = 'main'
    # decoded_main = mdc.autoencoder.get_layer(stamp_process('',['output',header_zone_key],'','','','_'))(encoded_input)
    # mdc.decoder = Model(encoded_input, decoded_main, name='decoder')
    mdc.model = model
    return model

def building_impreg_temptime_uRT_v3(mdc, latent_dim = 2, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    
    cell_size = getattr(mdc, 'cell_size', None)
    header_zone_key = 'main'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = (cell_size, len(header_zone)) if(isinstance(cell_size, int)) else len(header_zone)
    input_main_zone = Input(shape = input_shape, name=stamp_process('',['input',header_zone_key],'','','','_'))
    layer_index = 0
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    main_layered, layer_index = stack_layers(layer_index, input_main_zone, activation=getattr(mdc ,'activation', 'selu'),
                default_layer_name = ('Dense' if(getattr(mdc, 'cell_size', None)==None) else 'LSTM'),
                hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                cell_size=getattr(mdc, 'cell_size', kwags.get('cell_size', None)), 
                dropout_rates = getattr(mdc, 'dropout_rates', None),
                addlog = addlog)
    
    header_zone_key = 'condition'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = len(header_zone)
    input_cond_zone = Input(shape = input_shape, name=stamp_process('',['input','condition'],'','','','_'))
    
    merged = layer_producer('concatenate')([main_layered, input_cond_zone])
    # print("getattr(mdc, 'output_activation', 'sigmoid'):%s"%getattr(mdc, 'output_activation', 'sigmoid'))
    # encoded = layer_producer('Dense')(
    #     units = latent_dim, activation=getattr(mdc, 'latent_activation', 'linear'), name='coding-layer')(merged)
    
    mdc.keras_input = [input_main_zone, input_cond_zone]
    # mdc.encoder = Model(mdc.keras_input, encoded, name='encoder')
    # encoded_input = Input(shape=(int(latent_dim//1),))
    
    
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'main'
    header_zone = mdc.yheader.mylist_grp[header_zone_key]
    decoded_main = merged
    
    
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'relu')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_main = default_layer(units=output_shape, activation=getattr(mdc, 'output_activation', 'linear'),
                                name=stamp_process('',['output',header_zone_key],'','','','_'))(decoded_main)
    
    
    header_zone_key = 'main'
    header_zone = mdc.unfam_header.mylist_grp[header_zone_key]
    # unfamed = layer_producer('concatenate')([merged_for_encoded, input_target_zone]) #lstm不能直接concatenate，必須降維
    
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    unfamed = default_layer(units=output_shape, activation='sigmoid',
                                name=stamp_process('',['unfam',header_zone_key],'','','','_'))(merged)
    
    
    
    mdc.keras_output = [decoded_main, unfamed]
    
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    
    
    
    mdc.model = model
    return model

def building_impreg_temptime_uRT_v4(mdc, latent_dim = 2, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    
    cell_size = getattr(mdc, 'cell_size', None)
    header_zone_key = 'main'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = (cell_size, len(header_zone)) if(isinstance(cell_size, int)) else len(header_zone)
    input_main_zone = Input(shape = input_shape, name=stamp_process('',['input',header_zone_key],'','','','_'))
    layer_index = 0
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    main_layered, layer_index = stack_layers(layer_index, input_main_zone, activation=mdc.activation, 
                default_layer_name = ('Dense' if(getattr(mdc, 'cell_size', None)==None) else 'LSTM'),
                hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                cell_size=getattr(mdc, 'cell_size', kwags.get('cell_size', None)), 
                dropout_rates = getattr(mdc, 'dropout_rates', None),
                addlog = addlog)
    
    header_zone_key = 'condition'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = len(header_zone)
    input_cond_zone = Input(shape = input_shape, name=stamp_process('',['input','condition'],'','','','_'))
    
    merged = layer_producer('concatenate')([main_layered, input_cond_zone])
    # print("getattr(mdc, 'output_activation', 'sigmoid'):%s"%getattr(mdc, 'output_activation', 'sigmoid'))
    # encoded = layer_producer('Dense')(
    #     units = latent_dim, activation=getattr(mdc, 'latent_activation', 'linear'), name='coding-layer')(merged)
    
    mdc.keras_input = [input_main_zone, input_cond_zone]
    # mdc.encoder = Model(mdc.keras_input, encoded, name='encoder')
    # encoded_input = Input(shape=(int(latent_dim//1),))
    
    
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'pv_sv_seperated'
    header_zone = mdc.yheader.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'sigmoid')))
    decoded_pv_sv_seperated = default_layer(units=output_shape, activation=activation,
                                            name=stamp_process('',['output',header_zone_key],'','','','_'))(merged)
    
    header_zone_key = 'cst_osc'
    header_zone = mdc.yheader.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'sigmoid')))
    decoded_cst_osc = default_layer(units=output_shape, activation=activation,
                                            name=stamp_process('',['output',header_zone_key],'','','','_'))(merged)
    
    Merged = layer_producer('concatenate')([merged, decoded_pv_sv_seperated, decoded_cst_osc])
    
    
    
    
    header_zone_key = 'main'
    header_zone = mdc.yheader.mylist_grp[header_zone_key]
    
    
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'linear')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_main = default_layer(units=output_shape, activation=getattr(mdc, 'output_activation', 'linear'),
                                name=stamp_process('',['output',header_zone_key],'','','','_'))(Merged)
    
    
    header_zone_key = 'high_risk'
    header_zone = mdc.yheader.mylist_grp[header_zone_key]
    
    
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'sigmoid')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'BinaryCrossentropy')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_high_risk = default_layer(units=output_shape, activation=activation,
                                name=stamp_process('',['output',header_zone_key],'','','','_'))(Merged)
    
    
    header_zone_key = 'pv_sv_seperated'
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'sigmoid')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'BinaryCrossentropy')))
    addlog('activation:%s, lossmethod:%s'%(activation, lossmethod), stamps=[header_zone_key])
    activations.append(activation)
    lossmethods.append(lossmethod)
    
    header_zone_key = 'cst_osc'
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'sigmoid')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'BinaryCrossentropy')))
    addlog('activation:%s, lossmethod:%s'%(activation, lossmethod), stamps=[header_zone_key])
    activations.append(activation)
    lossmethods.append(lossmethod)
    
    header_zone_key = 'main'
    header_zone = mdc.unfam_header.mylist_grp[header_zone_key]
    # unfamed = layer_producer('concatenate')([merged_for_encoded, input_target_zone]) #lstm不能直接concatenate，必須降維
    
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'BinaryCrossentropy')))
    lossmethods.append(lossmethod)
    addlog('activation:%s, lossmethod:%s'%(activation, lossmethod), stamps=[header_zone_key])
    unfamed = default_layer(units=output_shape, activation='sigmoid',
                                name=stamp_process('',['unfam',header_zone_key],'','','','_'))(Merged)
    
    
    
    mdc.keras_output = [decoded_main, decoded_high_risk, decoded_pv_sv_seperated, decoded_cst_osc, unfamed]
    
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    
    mdc.model = model
    return model

def building_impreg_temptime_uRT_v5(mdc, latent_dim = 2, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    
    cell_size = getattr(mdc, 'cell_size', None)
    header_zone_key = 'main'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = (cell_size, len(header_zone)) if(isinstance(cell_size, int)) else len(header_zone)
    input_main_zone = Input(shape = input_shape, name=stamp_process('',['input',header_zone_key],'','','','_'))
    layer_index = 0
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    main_layered, layer_index = stack_layers(layer_index, input_main_zone, activation=mdc.activation, 
                default_layer_name = ('Dense' if(getattr(mdc, 'cell_size', None)==None) else 'LSTM'),
                hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                cell_size=getattr(mdc, 'cell_size', kwags.get('cell_size', None)), 
                dropout_rates = getattr(mdc, 'dropout_rates', None),
                addlog = addlog)
    
    header_zone_key = 'condition'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = len(header_zone)
    input_cond_zone = Input(shape = input_shape, name=stamp_process('',['input','condition'],'','','','_'))
    
    merged = layer_producer('concatenate')([main_layered, input_cond_zone])
    # print("getattr(mdc, 'output_activation', 'sigmoid'):%s"%getattr(mdc, 'output_activation', 'sigmoid'))
    # encoded = layer_producer('Dense')(
    #     units = latent_dim, activation=getattr(mdc, 'latent_activation', 'linear'), name='coding-layer')(merged)
    
    mdc.keras_input = [input_main_zone, input_cond_zone]
    # mdc.encoder = Model(mdc.keras_input, encoded, name='encoder')
    # encoded_input = Input(shape=(int(latent_dim//1),))
    
    
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'pv_sv_seperated'
    header_zone = mdc.yheader.mylist_grp[header_zone_key]
    
    
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'sigmoid')))
    decoded_pv_sv_seperated = default_layer(units=output_shape, activation=activation,
                                            name=stamp_process('',['output',header_zone_key],'','','','_'))(merged)
    
    Merged = layer_producer('concatenate')([merged, decoded_pv_sv_seperated])
    
    
    
    
    header_zone_key = 'main'
    header_zone = mdc.yheader.mylist_grp[header_zone_key]
    
    
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'linear')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_main = default_layer(units=output_shape, activation=getattr(mdc, 'output_activation', 'linear'),
                                name=stamp_process('',['output',header_zone_key],'','','','_'))(Merged)
    
    
        
    
    header_zone_key = 'pv_sv_seperated'
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'sigmoid')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'BinaryCrossentropy')))
    addlog('activation:%s, lossmethod:%s'%(activation, lossmethod), stamps=[header_zone_key])
    activations.append(activation)
    lossmethods.append(lossmethod)
    
    header_zone_key = 'main'
    header_zone = mdc.unfam_header.mylist_grp[header_zone_key]
    # unfamed = layer_producer('concatenate')([merged_for_encoded, input_target_zone]) #lstm不能直接concatenate，必須降維
    
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'BinaryCrossentropy')))
    lossmethods.append(lossmethod)
    unfamed = default_layer(units=output_shape, activation='sigmoid',
                                name=stamp_process('',['unfam',header_zone_key],'','','','_'))(Merged)
    
    
    
    mdc.keras_output = [decoded_main, decoded_pv_sv_seperated, unfamed]
    
    model = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    
    mdc.model = model
    return model



def building_impreg_temptime_uRT_ab_v4(mdc, latent_dim = 1, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    
    cell_size = getattr(mdc, 'cell_size', None)
    header_zone_key = 'main'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = (cell_size, len(header_zone)) if(isinstance(cell_size, int)) else len(header_zone)
    input_main_zone = Input(shape = input_shape, name=stamp_process('',['input',header_zone_key],'','','','_'))
    layer_index = 0
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    main_layered, layer_index = stack_layers(layer_index, input_main_zone, activation=mdc.activation, 
                default_layer_name = ('Dense' if(getattr(mdc, 'cell_size', None)==None) else 'LSTM'),
                hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                cell_size=getattr(mdc, 'cell_size', kwags.get('cell_size', None)), 
                dropout_rates = getattr(mdc, 'dropout_rates', None),
                addlog = addlog)
    
    header_zone_key = 'condition'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = len(header_zone)
    input_cond_zone = Input(shape = input_shape, name=stamp_process('',['input','condition'],'','','','_'))
    
    
    header_zone_key = 'target'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = len(header_zone)
    input_target_zone = Input(shape = input_shape, name=stamp_process('',['input','target'],'','','','_'))
    
    mdc.keras_input = [input_main_zone, input_cond_zone, input_target_zone]
    
    merged_for_encoded = layer_producer('concatenate')([main_layered, input_cond_zone])
    encoded = layer_producer('Dense')(
        units = latent_dim, activation=getattr(mdc, 'latent_activation', 'linear'), name='coding-layer')(merged_for_encoded)
    
    
    mdc.encoder = Model(mdc.keras_input, encoded, name='encoder')
    # encoded_input = Input(shape=(int(latent_dim//1),))
    
    
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'main'
    header_zone = mdc.yheader.mylist_grp[header_zone_key]
    decoded_combine = layer_producer('concatenate')([encoded, input_target_zone])
    
    
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'relu')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_combine = default_layer(units=output_shape, activation=getattr(mdc, 'output_activation', 'sigmoid'),
                                name=stamp_process('',['output',header_zone_key],'','','','_'))(decoded_combine)
    
    
    header_zone_key = 'main'
    header_zone = mdc.unfam_header.mylist_grp[header_zone_key]
    unfamed = layer_producer('concatenate')([merged_for_encoded, input_target_zone]) #lstm不能直接concatenate，必須降維
    
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    unfamed = default_layer(units=output_shape, activation='sigmoid',
                                name=stamp_process('',['unfam',header_zone_key],'','','','_'))(unfamed)
    
    
    
    mdc.keras_output = [decoded_combine, unfamed]
    mdc.autoencoder = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    
    
    header_zone_key = 'main'
    
    return mdc.autoencoder

def building_impreg_temptime_uRT_ab_v3(mdc, latent_dim = 2, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    
    cell_size = getattr(mdc, 'cell_size', None)
    header_zone_key = 'main'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = (cell_size, len(header_zone)) if(isinstance(cell_size, int)) else len(header_zone)
    input_main_zone = Input(shape = input_shape, name=stamp_process('',['input',header_zone_key],'','','','_'))
    layer_index = 0
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    main_layered, layer_index = stack_layers(layer_index, input_main_zone, activation=mdc.activation, 
                default_layer_name = ('Dense' if(getattr(mdc, 'cell_size', None)==None) else 'LSTM'),
                hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                cell_size=getattr(mdc, 'cell_size', kwags.get('cell_size', None)), 
                dropout_rates = getattr(mdc, 'dropout_rates', None),
                addlog = addlog)
    
    header_zone_key = 'condition'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = len(header_zone)
    input_cond_zone = Input(shape = input_shape, name=stamp_process('',['input','condition'],'','','','_'))
    
    merged = layer_producer('concatenate')([main_layered, input_cond_zone])
    # print("getattr(mdc, 'output_activation', 'sigmoid'):%s"%getattr(mdc, 'output_activation', 'sigmoid'))
    encoded = layer_producer('Dense')(
        units = latent_dim, activation=getattr(mdc, 'latent_activation', 'linear'), name='coding-layer')(merged)
    
    mdc.keras_input = [input_main_zone, input_cond_zone]
    mdc.encoder = Model(mdc.keras_input, encoded, name='encoder')
    encoded_input = Input(shape=(int(latent_dim//1),))
    
    
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'main'
    header_zone = mdc.yheader.mylist_grp[header_zone_key]
    decoded_main = encoded
    
    
    output_shape = len(header_zone)
    default_layer = layer_producer('Dense')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'relu')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_main = default_layer(units=output_shape, activation=getattr(mdc, 'output_activation', 'sigmoid'),
                                name=stamp_process('',['output',header_zone_key],'','','','_'))(decoded_main)
    mdc.keras_output = decoded_main
    mdc.autoencoder = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    
    
    header_zone_key = 'main'
    decoded_main = mdc.autoencoder.get_layer(stamp_process('',['output',header_zone_key],'','','','_'))(encoded_input)
    mdc.decoder = Model(encoded_input, decoded_main, name='decoder')
    
    return mdc.autoencoder

def building_impreg_temptime_uRT_ab_v2(mdc, latent_dim = 1, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    
    cell_size = getattr(mdc, 'cell_size', None)
    header_zone_key = 'main'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = (cell_size, len(header_zone)) if(isinstance(cell_size, int)) else len(header_zone)
    input_main_zone = Input(shape = input_shape, name=stamp_process('',['input',header_zone_key],'','','','_'))
    layer_index = 0
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    main_layered, layer_index = stack_layers(layer_index, input_main_zone, activation=mdc.activation, 
                default_layer_name = ('Dense' if(getattr(mdc, 'cell_size', None)==None) else 'LSTM'),
                hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                cell_size=getattr(mdc, 'cell_size', kwags.get('cell_size', None)), 
                dropout_rates = getattr(mdc, 'dropout_rates', None),
                addlog = addlog)
    
    header_zone_key = 'condition'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = len(header_zone)
    input_cond_zone = Input(shape = input_shape, name=stamp_process('',['input','condition'],'','','','_'))
    
    merged = layer_producer('concatenate')([main_layered, input_cond_zone])
    print("getattr(mdc, 'output_activation', 'sigmoid'):%s"%getattr(mdc, 'output_activation', 'sigmoid'))
    encoded = layer_producer('Dense')(
        units = latent_dim, activation=getattr(mdc, 'output_activation', 'sigmoid'), name='coding-layer')(merged)
    
    mdc.keras_input = [input_main_zone, input_cond_zone]
    mdc.encoder = Model(mdc.keras_input, encoded, name='encoder')
    encoded_input = Input(shape=(int(latent_dim//1),))
    
    
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'main'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    decoded_main = encoded
    if(isinstance(cell_size, int)): 
        decoded_main = layer_producer('RepeatVector')(cell_size, name='repeat')(decoded_main)
        decoded_main = layer_producer('TimeDistributed')(
            layer_producer('Dense')(1), input_shape=(cell_size, 1), name='timedistributed')(decoded_main)
    output_shape = (cell_size, len(header_zone)) if(isinstance(cell_size, int)) else len(header_zone)
    default_layer = layer_producer('LSTM')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'relu')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_main = default_layer(units=output_shape[1], activation=activation, return_sequences=True, 
                                name=stamp_process('',['output',header_zone_key],'','','','_'))(decoded_main)
    
    
    header_zone_key = 'condition'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'relu')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_cond = layer_producer('Dense')(units = output_shape, activation=activation, 
                                           name=stamp_process('',['output',header_zone_key],'','','','_'))(encoded)
    
    mdc.keras_output = [decoded_main, decoded_cond]
    mdc.autoencoder = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    
    
    header_zone_key = 'main'
    decoded_main = mdc.autoencoder.get_layer(stamp_process('',['output',header_zone_key],'','','','_'))(
        mdc.autoencoder.get_layer('timedistributed')(
            mdc.autoencoder.get_layer('repeat')(encoded_input)))
    header_zone_key = 'condition'
    decoded_cond = mdc.autoencoder.get_layer(stamp_process('',['output',header_zone_key],'','','','_'))(encoded_input)
    mdc.decoder = Model(encoded_input, [decoded_main, decoded_cond], name='decoder')
    
    return mdc.autoencoder

def building_impreg_temptime_uRT_ab_v1(mdc, latent_dim = 1, **kwags):
    addlog = getattr(mdc, 'addlog', kwags.get('addlog', LOGger.addlog))
    
    cell_size = getattr(mdc, 'cell_size', None)
    header_zone_key = 'main'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = (cell_size, len(header_zone)) if(isinstance(cell_size, int)) else len(header_zone)
    input_main_zone = Input(shape = input_shape, name=stamp_process('',['input',header_zone_key],'','','','_'))
    layer_index = 0
    addlog('dropout_rates:%s'%str(mdc.dropout_rates)) if(isinstance(getattr(mdc, 'dropout_rates', None), list)) else None
    main_layered, layer_index = stack_layers(layer_index, input_main_zone, activation=mdc.activation, 
                default_layer_name = ('Dense' if(getattr(mdc, 'cell_size', None)==None) else 'LSTM'),
                hidden_layer_sizes=mdc.hidden_layer_sizes, hidden_layer_nns=mdc.hidden_layer_nns,
                cell_size=getattr(mdc, 'cell_size', kwags.get('cell_size', None)), 
                dropout_rates = getattr(mdc, 'dropout_rates', None),
                addlog = addlog)
    
    header_zone_key = 'condition'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    input_shape = len(header_zone)
    input_cond_zone = Input(shape = input_shape, name=stamp_process('',['input','condition'],'','','','_'))
    
    merged = layer_producer('concatenate')([main_layered, input_cond_zone])
    print("getattr(mdc, 'output_activation', 'sigmoid'):%s"%getattr(mdc, 'output_activation', 'sigmoid'))
    encoded = layer_producer('Dense')(
        units = latent_dim, activation=getattr(mdc, 'output_activation', 'sigmoid'), name='coding-layer')(merged)
    
    mdc.keras_input = [input_main_zone, input_cond_zone]
    mdc.encoder = Model(mdc.keras_input, encoded, name='encoder')
    encoded_input = Input(shape=(int(latent_dim//1),))
    
    
    activations, lossmethods = mylist(), mylist()
    
    header_zone_key = 'main'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    decoded_main = layer_producer('RepeatVector')(cell_size, name='repeat')(encoded)
    output_shape = (cell_size, len(header_zone)) if(isinstance(cell_size, int)) else len(header_zone)
    default_layer = layer_producer('LSTM')
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'relu')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_main = default_layer(units=output_shape[1], activation=activation, return_sequences=True, 
                                name=stamp_process('',['output',header_zone_key],'','','','_'))(decoded_main)
    
    
    header_zone_key = 'condition'
    header_zone = mdc.xheader.mylist_grp[header_zone_key]
    output_shape = len(header_zone)
    activation = getattr(header_zone,'activation',LOGger.extract(
            getattr(mdc, 'activations', []), index=0, key=header_zone_key, default=getattr(mdc, 'activation', 'relu')))
    lossmethod = getattr(header_zone,'lossmethod',LOGger.extract(
            getattr(mdc, 'lossmethods', []), index=0, key=header_zone_key, default=getattr(mdc, 'lossmethod', 'mse')))
    activations.append(activation)
    lossmethods.append(lossmethod)
    decoded_cond = layer_producer('Dense')(units = output_shape, activation=activation, 
                                           name=stamp_process('',['output',header_zone_key],'','','','_'))(encoded)
    
    mdc.keras_output = [decoded_main, decoded_cond]
    mdc.model_form = Model(inputs=mdc.keras_input, outputs=mdc.keras_output, 
                  name=stamp_process('', mdc.get_stamps(for_file=False),'', '', '', '_'))
    
    mdc.lossmethods = lossmethods
    mdc.output_activations = activations
    
    
    header_zone_key = 'main'
    decoded_main = mdc.model_form.get_layer(stamp_process('',['output',header_zone_key],'','','','_'))(
        mdc.model_form.get_layer('repeat')(encoded_input))
    header_zone_key = 'condition'
    decoded_cond = mdc.model_form.get_layer(stamp_process('',['output',header_zone_key],'','','','_'))(encoded_input)
    mdc.decoder = Model(encoded_input, [decoded_main, decoded_cond], name='decoder')
    
    return mdc.model_form
    
    
    