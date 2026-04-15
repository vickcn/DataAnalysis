# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 11:56:56 2023
@author: anna.chou
getRFC
"""
import json
from suds.client import Client
from suds.transport.https import HttpAuthenticated
from suds.sudsobject import asdict
from datetime import date
from dateutil.relativedelta import relativedelta
import xmltodict
#import AGPLogger

class rRFC:
    def __init__(self,url,username,password):
        """
        --> url:string WS連結
        --> username:string 帳號
        --> password:string 密碼
        """
        t = HttpAuthenticated(username=username,password=password,timeout=600)
        self.client=Client(url,transport=t)
        
    def get_response(self,method):
        return getattr(self.client.service,method)
        
    def recursive_dict(self,d):
        out = {}
        #for k, v in asdict(d).iteritems():  #python 2.x
        for k, v in asdict(d).items():
            if hasattr(v, '__keylist__'):
                out[k] = self.recursive_dict(v)
            elif isinstance(v, list):
                out[k] = []
                for item in v:
                    if hasattr(item, '__keylist__'):
                        out[k].append(self.recursive_dict(item))
                    else:
                        out[k].append(item)
            else:
                out[k] = v
        return out
    
    def get_methods_name(self):
        """
        取得所有方法 get_methods_name
        <-- method_list:list 所有RFC
        """
        method_list = []
        for i in self.client.wsdl.services[0].ports[0].methods:
            method_list.append(i)
        return method_list
    
    def get_method_parm(self,method_name):
        """
        取得方法的参数 get_method_parm
        --> method_name:string RFC名稱
        <-- params_list:list 參數內容
        """
        method=self.client.wsdl.services[0].ports[0].methods[method_name]
        input_parames=method.binding.input
        params_list = []
        for i in input_parames.param_defs(method):
            params_list.append((i[1].name,i[1].type[0]))
        #params=input_parames.param_defs(method)[0]
        #return params[1].name,params[1].type[0]
        return params_list
    
    def get_parm_item(self,table_name):
        """
        取得table內的參數 get_parm_item
        --> table_name:string 表格名稱
        """
        item_name = table_name[7:]
        ltable = self.client.factory.create(table_name)
        #print(ltable)
        litem = self.client.factory.create(item_name)
        #print(litem)
        out = {}
        for k, v in asdict(ltable).items():
            #print(k,v)
            out[k] = [self.recursive_dict(litem)]
        return out
    
    def get_parm_data(self,method_name):
        """
        參數內格式轉成字串 get_parm_data
        --> method_name:string RFC名稱
        """
        importdata=""
        for item in self.get_method_parm(method_name):
            #print(item)
            if item[1].upper().find('TABLEOF') > -1: #有table 往下展
                importdata += "\"" + item[0] + "\":" + json.dumps(self.get_parm_item(item[1]))+","
                #print(item[0],":",json.dumps(RFC.get_parm_item(item[1])))
            else:
                importdata += "\"" + item[0] + "\":null,"
                #print(item[0],":null")
        importdata = "{" + importdata[:-1] + "}"
        return importdata
    
    def get_response_data(self, method_name, importdata):
        """
        時間日期轉換 get_response_data
        --> method_name:string RFC名稱
        --> importdata:string 傳入資料
        <-- responseData:list 回傳資料
        """
        fun = self.get_response(method_name)
        '''當天'''
        if importdata.find("$getToday") > -1:
            today = date.today().strftime("%Y%m%d")
            importdata = importdata.replace("$getToday", str(today))
        ''' 年月日轉換 '''
        if importdata.find("$getDate") > -1:
            #把所有$getDate置換
            m = importdata.find("$getDate")
            while (m>-1):
                DateStart = m + 9
                DateEnd = importdata.find(")", DateStart)
                DateValue = importdata[DateStart:DateEnd].split(",")
                newDate = date.today() + relativedelta(years=int(DateValue[0]), months=int(DateValue[1]), days=int(DateValue[2]))
                importdata = importdata[:m] + str(newDate.strftime("%Y%m%d")) + importdata[DateEnd+1:]
                m = importdata.find("$getDate")
        jimportdata = json.loads(importdata)
        response = fun(**jimportdata)
        responseData = self.recursive_dict(response)
        return responseData

    def get_output(self, method_name, importdata):
        """
        回傳資料 get_output
        --> method_name:string RFC名稱
        --> importdata:string 傳入資料
        <-- :list 回傳資料
        """
        #AGPLogger.addlog(method_name)
        #AGPLogger.addlog(importdata)
        responseData = self.get_response_data(method_name, importdata)
        #時間日期轉換
        if(responseData.get("XmlStr")): #xml
            #AGPLogger.addlog('XmlStr')
            xoutputdata = xmltodict.parse(responseData.get("XmlStr"))
            #AGPLogger.addlog('xoutputdata')
            if len(xoutputdata['ROOT']['item']) > 0:
                output = xoutputdata['ROOT']['item'] #dict
                return output
        elif len(responseData['OutTab']) > 0:
            return responseData['OutTab']['item']
        else: #沒值
            return ""

    def get_response_xmltodata(self, method_name, importdata):
        """
        xml轉換 get_response_xmltodata
        --> method_name:string RFC名稱
        --> importdata:string 傳入資料
        """
        fun = self.get_response(method_name)
        ximportdata = xmltodict.parse(importdata)
        response = fun(**ximportdata)
        responseData = self.recursive_dict(response)
        return responseData