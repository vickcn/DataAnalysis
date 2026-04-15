# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:08:51 2019
連接Modbus
@author: anna.chou
20230218
-新增字串讀取功能
20241201
-適配 pymodbus 3.6.2 版本
20241201
-自動檢測 pymodbus 版本並適配
"""
import os
import sys
import numpy as np
import traceback
import struct
import binascii

# 檢測 pymodbus 版本並導入相應模組
try:
    import pymodbus
    pymodbus_version = pymodbus.__version__
    print(f"檢測到 pymodbus 版本: {pymodbus_version}")
    
    # 根據版本選擇導入方式
    if pymodbus_version.startswith('2.'):
        # pymodbus 2.x 版本
        from pymodbus.client.sync import ModbusTcpClient as ModbusClient
        print("使用 pymodbus 2.x 語法")
        IS_PYMODBUS_V2 = True
    else:
        # pymodbus 3.x 版本
        from pymodbus.client import ModbusTcpClient as ModbusClient
        print("使用 pymodbus 3.x 語法")
        IS_PYMODBUS_V2 = False
        
    from pymodbus.register_read_message import ReadHoldingRegistersResponse
    from pymodbus.payload import BinaryPayloadDecoder
    from pymodbus.constants import Endian
    
except ImportError as e:
    print(f"無法導入 pymodbus: {e}")
    sys.exit(1)
except Exception as e:
    print(f"版本檢測錯誤: {e}")
    # 預設使用 3.x 語法
    from pymodbus.client import ModbusTcpClient as ModbusClient
    from pymodbus.register_read_message import ReadHoldingRegistersResponse
    from pymodbus.payload import BinaryPayloadDecoder
    from pymodbus.constants import Endian
    IS_PYMODBUS_V2 = False
    print("預設使用 pymodbus 3.x 語法")
###########################################################################
def stamp_process(stg='', stamps=[], stamp_sep=':', stamp_left='[', stamp_right=']', 
                  adjoint_sep='', outer_stamp_left='', outer_stamp_right='', location=1, 
                  exceptions = [''], annih_when_stamps_empty=True, for_file=False, **kwags):
    stamp = ''
    if(isinstance(stamps, dict)):
        stamp = adjoint_sep.join(list(map(lambda t:('%s%s%s%s%s'%(stamp_left,
                str(t[0]), stamp_sep, str(t[1]), stamp_right) if(t[1]!='') else ''), tuple(stamps.items()))))
    elif((stamps!='') if(isinstance(stamps, str)) else False):
        stamp = '%s%s%s'%(stamp_left, stamps, stamp_right)
    elif((np.array(stamps).shape[0]>0) if(len(np.array(stamps).shape)>0) else False):
        stamps = [v for v in stamps if not v in exceptions]
        stamp = adjoint_sep.join(list(map(lambda s:('%s%s%s'%(stamp_left, s, stamp_right) if(s!='') else ''), stamps)))
    if(annih_when_stamps_empty and stamp==''):
        return ''
    stamp = outer_stamp_left + stamp + outer_stamp_right if(stamp!='') else ''
    stg = (stamp + stg if(location>0) else stg + stamp) if(location!=0) else stg
    # stg = for_file_process(stg) if(for_file) else stg
    return stg

def exception_process(e, logfile=os.path.join('log_%t.txt'), stamps=None, **kwags):
    addlog = kwags.get('addlog', lambda s, **kwags:print(s))
    stamps = stamps if(isinstance(stamps, list)) else []
    exc_type, exc_obj, ex_stack = sys.exc_info()
    ex_stamps = {'lineno':'%d'%e.__traceback__.tb_lineno,
                 'name':'%s'%e.__traceback__.tb_frame.f_code.co_name,
                 'type':'%s'%exc_type}
    msg = '[ERROR]%s\n%s%s\n%s'%(e.__traceback__.tb_frame.f_code.co_filename,
        stamp_process('',ex_stamps), stamp_process('', stamps), '錯誤訊息:\n%s\n'%str(e))
    addlog('--------------------------------------------------', logfile=logfile, **kwags)
    for stack in traceback.extract_tb(ex_stack):
        addlog(str(stack)[:200], logfile=logfile, **kwags)
    addlog('--------------------------------------------------', logfile=logfile, **kwags)
    kwags.update({'annih_when_stamps_empty':False})
    addlog(msg, logfile=logfile, **kwags)

def modbus_read_methods(modbus):
    return {float:modbus.ReadModbus_f,
            str:modbus.ReadModbus_s,
            int:modbus.ReadModbus_i}

class Modbus:
    def __init__(self, host, port, stamps=None, log_counter_ubd=1):
        self.stamps = stamps if(isinstance(stamps, list)) else []
        self.log_counter = {}
        self.log_counter_ubd = log_counter_ubd
        self.host = host
        self.port = port
        
        # 根據版本選擇客戶端創建方式
        if IS_PYMODBUS_V2:
            # pymodbus 2.x 版本
            self.client = ModbusClient(host, port, timeout=1)
        else:
            # pymodbus 3.x 版本
            self.client = ModbusClient(host, port, timeout=1)
            
        self.status_connect = self.client.connect()
        
    def connect(self):
        try:
            if IS_PYMODBUS_V2:
                # pymodbus 2.x 版本的連接方式
                self.status_connect = self.client.connect()
            else:
                # pymodbus 3.x 版本的連接方式
                self.status_connect = self.client.connect()
        except Exception as e:
            exception_process(e, logfile='', stamps=self.stamps+[self.log_counter.get('err',0)]) if(
                self.log_counter.get('err',0)<self.log_counter_ubd) else None
            self.log_counter.update({'err':self.log_counter.get('err',0)+1})
            return False
        self.log_counter.update({'err':0})
        return True

    #讀取接口的資料_float
    def ReadModbus_f(self, start_addr, count):
        rr = self.client.read_holding_registers(start_addr, count, unit=1)
        condition = (isinstance(rr, ReadHoldingRegistersResponse) if(IS_PYMODBUS_V2) else (rr is not None and not rr.isError()))
        if(condition):
            value = []
            for i in range(count//2):
                get0 = "{:0>4X}".format(rr.registers[2*i])
                get1 = "{:0>4X}".format(rr.registers[2*i+1])
                value.append(struct.unpack('>f', binascii.unhexlify(get0+get1))[0])
            return value
        else:
            return []

    #寫回接口的資料_float
    def WriteModbus_f(self, start_addr, value):
        s = struct.Struct('>f')
        packed_data = binascii.hexlify(s.pack(value))
        rq = self.client.write_registers(start_addr, [int(packed_data[0:4], 16), int(packed_data[4:], 16)], unit=1)
        
        # 根據版本選擇錯誤檢查方式
        condition = (rq.isError() if(IS_PYMODBUS_V2) else (rq is None or rq.isError()))
        return 'NG' if(condition) else 'OK'

    #讀取接口的資料_float_r
    def ReadModbus_f_r(self, start_addr, count):
        rr = self.client.read_holding_registers(start_addr, count, unit=1)
        print(rr)
        
        condition = (isinstance(rr, ReadHoldingRegistersResponse) if(IS_PYMODBUS_V2) else (rr is not None and not rr.isError()))
        if(condition):
            value = []
            for i in range(count//2):
                get0 = "{:0>4X}".format(rr.registers[2*i])
                get1 = "{:0>4X}".format(rr.registers[2*i+1])
                value.append(struct.unpack('>f', binascii.unhexlify(get1+get0))[0])
            return value
        else:
            return []

    #寫回接口的資料_float_r
    def WriteModbus_f_r(self, start_addr, value):
        s = struct.Struct('>f')
        packed_data = binascii.hexlify(s.pack(*value))
        rq = self.client.write_registers(start_addr, [int(packed_data[4:], 16), int(packed_data[0:4], 16)], unit=1)
        
        condition = (rq.isError() == True if(IS_PYMODBUS_V2) else (rq is None or rq.isError()))
        if(condition):
            return 'NG'
        return 'OK'

    #讀取接口的資料_整數
    def ReadModbus_i(self, start_addr, count):
        rr = self.client.read_holding_registers(start_addr, count, unit=1)
        
        condition = (isinstance(rr, ReadHoldingRegistersResponse) if(IS_PYMODBUS_V2) else (rr is not None and not rr.isError()))
        if(condition):
            byteorder = (Endian.Big if(IS_PYMODBUS_V2) else Endian.BIG)
            decoder = BinaryPayloadDecoder.fromRegisters(rr.registers, byteorder=byteorder)
            if count == 1:
                return decoder.decode_16bit_int()
            elif count == 2:
                return decoder.decode_32bit_int()
            elif count == 4:
                return decoder.decode_64bit_int()
        return 0

    #寫回接口的資料_整數
    def WriteModbus_i(self, start_addr, value):
        rq = self.client.write_registers(start_addr, value, unit=1)
        
        condition = (rq.isError() == True if(IS_PYMODBUS_V2) else (rq is None or rq.isError()))
        if(condition):
            return 'NG'
        return 'OK'
    
    #讀取接口的資料_字串
    def ReadModbus_s(self, start_addr, count=20, decode_count=None, strip_chr='\x00'):
        rr = self.client.read_holding_registers(start_addr, count, unit=1)
        
        condition = (isinstance(rr, ReadHoldingRegistersResponse) if(IS_PYMODBUS_V2) else (rr is not None and not rr.isError()))
        if(condition):
            decode_count = count if(not isinstance(decode_count, int)) else decode_count
            if(IS_PYMODBUS_V2):
                stg = BinaryPayloadDecoder.fromRegisters(rr.registers).decode_string(decode_count).decode()
                stg = stg.strip(strip_chr) if(isinstance(strip_chr, str)) else stg
                return stg
            else:
                try:
                    decoder = BinaryPayloadDecoder.fromRegisters(rr.registers)
                    stg = decoder.decode_string(decode_count).decode()
                    stg = stg.strip(strip_chr) if(isinstance(strip_chr, str)) else stg
                    return stg
                except Exception as e:
                    print(f"字串解碼錯誤: {e}")
                    return ''
        return ''

    #寫回接口的資料_字串
    # def WriteModbus_s(self, start_addr, value):
    #     return 'NG'
    #     rq = self.client.write_registers(start_addr, value, unit=1)
    #     if rq.isError() == True:
    #         return 'NG'
    #     return 'OK'
    
    
###########################################################################