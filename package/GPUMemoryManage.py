import sys
import gc
import psutil
import os
# 設定環境變數
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['TF_GPU_MEMORY_FRACTION'] = '1'
import tensorflow as tf
from package import LOGger
m_print = LOGger.addloger(logfile='')

class GPUMemoryManager:
    def __init__(self, addlog=None):
        self.addlog = addlog if(callable(addlog)) else m_print
        # self.memory_limit_mb = memory_limit_mb
        # self.setup_gpu()
    
    # def setup_gpu(self):
    #     # 設定 GPU 記憶體限制
    #     gpus = tf.config.experimental.list_physical_devices('GPU')
    #     if gpus:
    #         try:
    #             # 設定動態記憶體分配
    #             for gpu in gpus:
    #                 tf.config.experimental.set_memory_growth(gpu, True)
                
    #             # 設定記憶體限制
    #             tf.config.experimental.set_virtual_device_configuration(
    #                 gpus[0],
    #                 [tf.config.experimental.VirtualDeviceConfiguration(
    #                     memory_limit=self.memory_limit_mb
    #                 )]
    #             )
                
    #         except RuntimeError as e:
    #             LOGger.exception_process(e, logfile='')
    #             m_print(f"GPU 設定錯誤: {e}", colora=LOGger.FAIL)
    
    def clear_gpu_memory(self):
        try:
            # 清理 TensorFlow 的記憶體
            tf.keras.backend.clear_session()
            
            # 清理 Python 的記憶體
            gc.collect()
            
            # 清理 CUDA 快取
            # if tf.config.list_physical_devices('GPU'):
            #     tf.config.experimental.set_memory_growth(
            #         tf.config.list_physical_devices('GPU')[0], 
            #         True
            #     )
        except Exception as e:
            self.addlog(f"清理 GPU 記憶體失敗: {e}", colora=LOGger.FAIL)
            return False
        return True
    
    def get_gpu_memory_usage(self):
        try:
            # 獲取 GPU 記憶體使用情況
            gpu = tf.config.list_physical_devices('GPU')[0]
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            
            return {
                'current': memory_info['current'] / 1024 / 1024,  # MB
                'peak': memory_info['peak'] / 1024 / 1024,  # MB
                # 'limit': self.memory_limit_mb
            }
        except Exception as e:
            self.addlog(f"無法獲取 GPU 記憶體使用情況: {e}", colora=LOGger.FAIL)
            return None
        
    def show_gpu_memory_usage(self, stamps=None):
        # 獲取 GPU 記憶體使用情況
        memory_usage = self.get_gpu_memory_usage()
        if memory_usage:
            self.addlog(f"當前 GPU 記憶體使用情況: {memory_usage}", colora=LOGger.OKGREEN, stamps=stamps)

#%%
def gpuResourceInit(gpu, memory_limit=2000, stamps=None):
    stamps = stamps if(isinstance(stamps, list)) else []
    if(memory_limit is None):
        m_print("記憶體限制為 None，跳過 GPU 記憶體限制設置", colora=LOGger.WARNING, stamps=[gpu.name, *stamps])
        return True
    try:
        # 如果還沒設定過，這裡會是空的
        configs = tf.config.experimental.get_virtual_device_configuration(gpu)
        if configs:
           m_print(f"GPU {gpu.name} 已經設定過虛擬設備配置", "memory limit:%s"%configs[0].memory_limit, colora=LOGger.WARNING, stamps=stamps)
           gpuConfigBackupFile = LOGger.stamp_process('',['gpuConfig', *stamps], '','','','_',for_file=True)
           LOGger.CreateFile(gpuConfigBackupFile, lambda f:LOGger.save_json(configs, f))
           return True 
        # 只分配 60% 的 GPU 記憶體
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])  # 單位：MB
        m_print(f"GPU {gpu.name} 記憶體限制設置為 {memory_limit} MB", colora=LOGger.OKGREEN, stamps=stamps)
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=stamps)
        return False
    return True

def gpusResourceInit(default_memory_limit=2000, stamps=None, mask=None):
    stamps = stamps if(isinstance(stamps, list)) else []
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for i,gpu in enumerate(gpus):
                if LOGger.isiterable(mask):
                    if not gpu.name in mask:
                        continue
                # 設定 GPU 記憶體增長
                if(not gpuResourceInit(gpu, memory_limit=default_memory_limit, stamps=[*stamps, i, gpu.name])):
                    return False
                m_print(f"GPU {gpu.name} 初始化成功", colora=LOGger.OKGREEN, stamps=[*stamps, i, gpu.name])
        else:
            m_print("沒有可用的 GPU", colora=LOGger.FAIL, stamps=stamps)
            return False
    except Exception as e:
        LOGger.exception_process(e, logfile='', stamps=stamps)
        return False
    return True

def testGPU():
    try:
        # 检查可用的物理设备
        physical_devices = tf.config.list_physical_devices()
        print("可用的物理设备:", physical_devices)

        # 检查可用的GPU
        gpus = tf.config.list_physical_devices('GPU')
        print("可用的GPU:", gpus)

        # 检查是否使用CUDA构建
        print("是否使用CUDA构建:", tf.test.is_built_with_cuda())

        # 检查是否使用GPU
        print("是否使用GPU:", tf.test.is_gpu_available())

        # 创建一个简单的操作来测试
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print(c)
    except Exception as e:
        m_print(f"GPU 測試失敗: {e}", colora=LOGger.FAIL)
        return False
    return True

def testGPUManage():
    # 測試 GPU 記憶體管理
    gpu_manager = GPUMemoryManager()
    
    # 獲取 GPU 記憶體使用情況
    gpu_manager.show_gpu_memory_usage()
    # 測試清理 GPU 記憶體
    if(not gpu_manager.clear_gpu_memory()):
        return False
    # 獲取 GPU 記憶體使用情況
    gpu_manager.show_gpu_memory_usage()
    return True

def scenario():
    if(not testGPU()):
        m_print("GPU 測試失敗", colora=LOGger.FAIL)
        return False
    if(not testGPUManage()):
        m_print("GPU 記憶體管理測試失敗", colora=LOGger.FAIL)
        return False    
    return True

if __name__ == "__main__":
    scenario()