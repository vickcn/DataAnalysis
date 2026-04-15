import os
import sys
import asyncio

from package import data_analysis as ds
from package.data_analysis import *
from package import distribution_test as DT
from package import EIMSDataProc as EDP
from package import LOGger
from src.discrete_aggregator import aggregate_discrete_data
import concurrent.futures
dt = LOGger.dt
plt = vs3.plt

m_print = LOGger.addloger(logfile='')
m_logfile = os.path.join('log', 'DataAnalysis.log')
m_addlog = LOGger.addloger(logfile=m_logfile)

def autoExportSet(exp_fd=None, source_file=None, default=os.path.join('tmp', 'micCorr'), **kwargs):
    if LOGger.isinstance_not_empty(exp_fd, str):
        return exp_fd
    elif exp_fd == '[auto]':
        if LOGger.isinstance_not_empty(source_file, str):
            return os.path.join(os.path.dirname(source_file), 'micCorr')
    return default

def DataParsingFromFile(data_exp_file, ret=None, sht=None, selectHeader=None, dataClassAsgin=None, **kwargs):
    # 讀取原始資料
    raw_data = DFP.import_data(data_exp_file, sht=sht)
    
    # 添加 dataClassAsgin 的 log 記錄
    m_addlog(f'接收到的 dataClassAsgin: {dataClassAsgin}', stamps=['DataParsingFromFile'])
    m_addlog(f'dataClassAsgin 類型: {type(dataClassAsgin)}', stamps=['DataParsingFromFile'])
    
    # 使用 EIMSDataSetCore 處理資料
    df, dfInfo = DataParsing(raw_data, dataClassAsgin=dataClassAsgin, selectHeader=selectHeader)
    if(df is None or dfInfo is None):
        return False
    if isinstance(ret, dict):
        ret['matrix'] = df
        ret['dfInfo'] = dfInfo
    return True

def DataParsing(data, dataClassAsgin=None, stamps=None, selectHeader=None, **kwargs):
    # 使用 EIMSDataSetCore 處理資料
    try:
        if isinstance(selectHeader, list):
            m_addlog(f"接收到的 selectHeader: {','.join(selectHeader[:10])}", stamps=['DataParsing'], colora=LOGger.OKGREEN)
            data = data[selectHeader].copy()
        df, dfInfo, _ = EDP.EIMSDataSetCore(data, dataClassAsgin=dataClassAsgin)
    except Exception as e:
        stamps = stamps if isinstance(stamps, list) else []
        LOGger.exception_process(e, logfile='', stamps=['DataParsing', *stamps])
        m_addlog('EIMSDataSetCore 失敗', stamps=['DataParsing', *stamps], colora=LOGger.FAIL)
        return None, None
    return df, dfInfo

def PlotCorrelation(matrix, method='mic', exp_fd=os.path.join('micCorr'), stamps=None, ret=None, **kwargs):
    if(not ds.plotCorrelation(matrix, stamps=stamps, handler=None, exp_fd=exp_fd, file=None, numColor=None, mask=None, height=5, ret=ret, **kwargs)):
        return False
    return True

def DataParsingAndPlotCorrelation(data_exp_file, ret=None, sht=None, exp_fd=None, selectHeader=None, stamps=None, method='mic', isAggregate=True, dataClassAsgin=None, **kwargs):
    if(not DataParsingFromFile(data_exp_file, ret=ret, sht=sht, selectHeader=selectHeader, dataClassAsgin=dataClassAsgin, **kwargs)):
        return False
    matrix = ret['matrix']
    stamps = stamps if isinstance(stamps, list) else []
    
    # 先確定輸出目錄，以便聚合映射關係可以保存到相同位置
    exp_fd = autoExportSet(exp_fd, source_file=data_exp_file)
    
    if isAggregate:
        # 應用離散數據聚合功能，減少類別數量以提升相關性分析效率
        try:
            # 檢查是否在事件循環中運行
            try:
                # 如果已經在事件循環中，直接等待協程
                loop = asyncio.get_running_loop()
                # 在已運行的事件循環中，需要使用不同的方法
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, aggregate_discrete_data(matrix, exp_fd=exp_fd, **kwargs))
                    aggregated_matrix, aggregation_results = future.result()
            except RuntimeError:
                # 沒有運行的事件循環，可以使用 asyncio.run()
                aggregated_matrix, aggregation_results = asyncio.run(
                    aggregate_discrete_data(matrix, exp_fd=exp_fd, **kwargs)
                )
            
            # 更新 ret 中的數據
            ret['matrix'] = aggregated_matrix
            ret['original_matrix'] = matrix  # 保留原始數據
            ret['aggregation_results'] = aggregation_results
            
            # 記錄聚合結果
            aggregated_count = len([r for r in aggregation_results if r['aggregated']])
            if aggregated_count > 0:
                m_addlog(f"✓ 離散數據聚合完成：處理了 {aggregated_count} 個欄位，提升相關性分析效率", 
                        stamps=['DataParsingAndPlotCorrelation', *stamps], colora=LOGger.OKCYAN)
            
            matrix = aggregated_matrix  # 使用聚合後的數據進行相關性分析
        except Exception as e:
            LOGger.exception_process(e, logfile='', stamps=['DataParsingAndPlotCorrelation', *stamps])
            m_addlog(f"⚠ 離散數據聚合失敗，使用原始數據進行分析", stamps=['DataParsingAndPlotCorrelation', *stamps], colora=LOGger.WARNING)
            # 聚合失敗時繼續使用原始數據，不中斷流程
            # 如果聚合失敗，exp_fd 可能還沒設定，需要再次設定
            if exp_fd is None:
                exp_fd = autoExportSet(exp_fd, source_file=data_exp_file)
        
    if(not PlotCorrelation(matrix, method=method, exp_fd=exp_fd, stamps=None, ret=ret, **kwargs)):
        return False
    return True

def CalculateCorrelation(matrix, method='mic', exp_fd=os.path.join('micCorr'), stamps=None, ret=None, **kwargs):
    if(not ds.saveCorrelation(matrix, method, exp_fd, stamps, ret=ret, **kwargs)):
        return False
    return True

def analyzeContinuousToContinuous(data, xheader, yheader, fixed_values=None, include_model_prediction=True, ret=None, output_dir=None, **kwargs):
    if(not ds.analyzeContinuousToContinuous(data, xheader, yheader, fixed_values, include_model_prediction, ret=ret, output_dir=output_dir, **kwargs)):
        return False
    return True

def analyzeCategoricalToContinuous(data, xheader, yheader, fixed_values=None, include_model_prediction=True, ret=None, output_dir=None, **kwargs):
    if(not ds.analyzeCategoricalToContinuous(data, xheader, yheader, fixed_values, include_model_prediction, ret=ret, output_dir=output_dir, **kwargs)):
        return False
    return True

def analyzeContinuousToCategorical(data, xheader, yheader, fixed_values=None, ret=None, output_dir=None, **kwargs):
    if(not ds.analyzeContinuousToCategorical(data, xheader, yheader, fixed_values, ret=ret, output_dir=output_dir, **kwargs)):
        return False
    return True

def analyzeCategoricalToCategorical(data, xheader, yheader, fixed_values=None, ret=None, output_dir=None, **kwargs):
    if(not ds.analyzeCategoricalToCategorical(data, xheader, yheader, fixed_values, ret=ret, output_dir=output_dir, **kwargs)):
        return False
    return True

def twoDimensionAnalysis(data, xheader1, xheader2, yheader, fixed_values=None, include_model_prediction=True, ret=None, output_dir=None, **kwargs):
    if(not ds.twoDimensionAnalysisDirectly(data, xheader1, xheader2, yheader, fixed_values, include_model_prediction, ret=ret, output_dir=output_dir, **kwargs)):
        return False
    return True

def analyzeTwoContinuousToY(data, xheader1, xheader2, yheader, y_type, fixed_values=None, include_model_prediction=True, ret=None, output_dir=None, **kwargs):
    if(not ds.analyzeTwoContinuousToY(data, xheader1, xheader2, yheader, y_type, fixed_values, include_model_prediction, ret=ret, output_dir=output_dir, **kwargs)):
        return False
    return True

def analyzeMixedToY(data, xheader1, xheader2, yheader, x1_type, x2_type, y_type, fixed_values=None, include_model_prediction=True, ret=None, output_dir=None, **kwargs):
    if(not ds.analyzeMixedToY(data, xheader1, xheader2, yheader, x1_type, x2_type, y_type, fixed_values, include_model_prediction, ret=ret, output_dir=output_dir, **kwargs)):
        return False
    return True

def analyzeTwoCategoricalToY(data, xheader1, xheader2, yheader, y_type, fixed_values=None, include_model_prediction=True, ret=None, output_dir=None, **kwargs):
    if(not ds.analyzeTwoCategoricalToY(data, xheader1, xheader2, yheader, y_type, fixed_values, include_model_prediction, ret=ret, output_dir=output_dir, **kwargs)):
        return False
    return True

def standardTests(handler, **kwargs):
    print("=" * 60)
    print("Excel 資料分布檢定分析")
    print("=" * 60)
    print(f"檔案路徑: {handler.file_path}")
    print(f"分析欄位: {handler.column}")
    if handler.group_by:
        print(f"分群欄位: {handler.group_by}")
    if handler.sheet:
        print(f"工作表: {handler.sheet}")
    print(f"顯著性水準: {handler.alpha}")
    print(f"輸出目錄: {handler.output_dir}")
    print()
    
    # 載入資料
    print("正在載入資料...")
    data = DT.load_excel_data(handler.file_path, handler.sheet)
    
    if data is None:
        print("載入資料失敗，程式結束")
        return
    
    # 驗證欄位
    if not DT.validate_column(data, handler.column):
        return
    
    if handler.group_by and not DT.validate_column(data, handler.group_by):
        return
    
    # 開始分析
    print("開始進行分布檢定分析...")
    results = {}
    
    # 計算描述性統計
    print("計算描述性統計...")
    if handler.group_by:
        descriptive_stats = {}
        grouped = data.groupby(handler.group_by)[handler.column]
        for name, group_data in grouped:
            descriptive_stats[str(name)] = DT.calculate_descriptive_statistics(group_data)['overall']
        results['descriptive_stats'] = descriptive_stats
    else:
        results['descriptive_stats'] = DT.calculate_descriptive_statistics(data[handler.column])
    
    # 常態性檢定
    print("執行常態性檢定...")
    if handler.group_by:
        normality_tests = {}
        grouped = data.groupby(handler.group_by)[handler.column]
        for name, group_data in grouped:
            normality_tests[str(name)] = DT.perform_normality_tests(group_data, handler.alpha)
        results['normality_tests'] = normality_tests
    else:
        results['normality_tests'] = {'overall': DT.perform_normality_tests(data[handler.column], handler.alpha)}
    
    # 群組比較檢定
    if handler.group_by:
        print("執行群組比較檢定...")
        grouped_data = {}
        for name, group_data in data.groupby(handler.group_by)[handler.column]:
            grouped_data[str(name)] = group_data
        
        results['group_comparison'] = DT.perform_group_comparison_tests(grouped_data, handler.alpha)
    
    # 創建圖表
    print("生成分布圖表...")
    plot_files = DT.create_comprehensive_distribution_plot(data, handler.column, handler.group_by, results, handler.output_dir)
    results['plot_files'] = plot_files
    
    # 生成報告
    print("生成分析報告...")
    report_path = DT.generate_report(results, handler.column, handler.group_by, handler.output_dir)
    results['report_path'] = report_path
    
    # 輸出結果摘要
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    
    print(f"生成的檔案:")
    print(f"- 分析報告: {report_path}")
    for plot_file in plot_files:
        print(f"- 圖表: {plot_file}")
    
    # 輸出主要結果摘要
    print(f"\n主要結果摘要:")
    
    if 'descriptive_stats' in results:
        if handler.group_by:
            print(f"群組數量: {len(results['descriptive_stats'])}")
            for group_name, stats in results['descriptive_stats'].items():
                print(f"  {group_name}: 平均={stats['mean']:.4f}, 標準差={stats['std']:.4f}, 樣本數={stats['count']}")
        else:
            stats = results['descriptive_stats']['overall']
            print(f"整體統計: 平均={stats['mean']:.4f}, 標準差={stats['std']:.4f}, 樣本數={stats['count']}")
    
    # 常態性檢定結果摘要
    if 'normality_tests' in results:
        print(f"\n常態性檢定結果:")
        for group_name, tests in results['normality_tests'].items():
            significant_tests = []
            for test_name, test_result in tests.items():
                if 'significant' in test_result and test_result['significant']:
                    significant_tests.append(test_name)
            
            if significant_tests:
                print(f"  {group_name}: 拒絕常態分布假設 ({', '.join(significant_tests)})")
            else:
                print(f"  {group_name}: 無法拒絕常態分布假設")
    
    # 群組比較結果摘要
    if 'group_comparison' in results:
        print(f"\n群組比較檢定結果:")
        group_tests = results['group_comparison']
        
        significant_tests = []
        for test_name, test_result in group_tests.items():
            if isinstance(test_result, dict) and 'significant' in test_result and test_result['significant']:
                significant_tests.append(test_name)
        
        if significant_tests:
            print(f"  群組間存在顯著差異 ({', '.join(significant_tests)})")
        else:
            print(f"  群組間無顯著差異")
    
    print(f"\n所有結果已儲存至: {handler.output_dir}")
    
    # 使用說明
    print(f"\n使用說明:")
    print(f"1. 查看詳細分析報告: {results['report_path']}")
    print(f"2. 查看分布圖表: {', '.join([os.path.basename(f) for f in results['plot_files']])}")
    
    # 建議下一步操作
    print(f"\n建議下一步操作:")
    if handler.group_by is None:
        print(f"- 嘗試使用 --group_by 參數指定分群欄位進行群組比較")
    if not handler.use_api:
        print(f"- 使用 --use_api 參數獲得更詳細的統計分析")
    print(f"- 根據常態性檢定結果選擇適當的統計方法")
    print(f"- 考慮進行更深入的資料探索和視覺化分析")

# ============================================================================
# API 版本的分析函數（使用 API 調用 MDC 模型，不影響原有函數）
# ============================================================================

# 導入 API 分析模組
try:
    from package import api_analysis as api_ds
    API_ANALYSIS_AVAILABLE = True
except ImportError:
    API_ANALYSIS_AVAILABLE = False
    m_addlog("警告：無法導入 api_analysis 模組，API 版本函數將不可用", stamps=['DataAnalysis'], colora=LOGger.WARNING)

def analyzeContinuousToContinuous_via_api(data, xheader, yheader, fixed_values=None, include_model_prediction=True, ret=None, output_dir=None, api_url="http://10.1.3.236:5678", model_name="ACAngle", version="v0-0-2-0", timeout=30, **kwargs):
    """
    連續型x對連續型y的分析（API 版本）
    
    使用 API 調用 MDC 模型進行預測，不需要本地載入 MDC 物件
    
    Parameters
    ----------
    data : pd.DataFrame
        原始資料
    xheader : str
        x軸header名稱
    yheader : str
        y軸header名稱
    fixed_values : dict, optional
        指定其他X因子的固定值
    include_model_prediction : bool
        是否包含模型預測
    ret : dict, optional
        回傳結果字典
    output_dir : str, optional
        輸出目錄
    api_url : str
        API 服務器 URL（預設: http://10.1.3.236:5678）
    model_name : str
        模型名稱（預設: ACAngle）
    version : str
        模型版本（預設: v0-0-2-0）
    timeout : int
        API 請求超時時間（秒，預設: 30）
    **kwargs
        其他參數
        
    Returns
    -------
    bool
        是否成功分析
    """
    if not API_ANALYSIS_AVAILABLE:
        m_addlog("錯誤：api_analysis 模組不可用，無法使用 API 版本", stamps=['DataAnalysis'], colora=LOGger.FAIL)
        return False
    
    if(not api_ds.analyzeContinuousToContinuous(data, xheader, yheader, fixed_values, include_model_prediction, ret=ret, output_dir=output_dir, api_url=api_url, model_name=model_name, version=version, timeout=timeout, **kwargs)):
        return False
    return True

def analyzeCategoricalToContinuous_via_api(data, xheader, yheader, fixed_values=None, include_model_prediction=True, ret=None, output_dir=None, api_url="http://10.1.3.236:5678", model_name="ACAngle", version="v0-0-2-0", timeout=30, **kwargs):
    """
    非連續型x對連續型y的分析（API 版本）
    """
    if not API_ANALYSIS_AVAILABLE:
        m_addlog("錯誤：api_analysis 模組不可用，無法使用 API 版本", stamps=['DataAnalysis'], colora=LOGger.FAIL)
        return False
    
    if(not api_ds.analyzeCategoricalToContinuous(data, xheader, yheader, fixed_values, include_model_prediction, ret=ret, output_dir=output_dir, api_url=api_url, model_name=model_name, version=version, timeout=timeout, **kwargs)):
        return False
    return True

def analyzeContinuousToCategorical_via_api(data, xheader, yheader, fixed_values=None, ret=None, output_dir=None, api_url="http://10.1.3.236:5678", model_name="ACAngle", version="v0-0-2-0", timeout=30, **kwargs):
    """
    連續型x對非連續型y的分析（API 版本）
    
    注意：目前 api_analysis 尚未實現此函數，此函數會回退到本地模式
    """
    m_addlog("警告：analyzeContinuousToCategorical_via_api 尚未實現，回退到本地模式", stamps=['DataAnalysis'], colora=LOGger.WARNING)
    # 回退到本地模式
    if(not ds.analyzeContinuousToCategorical(data, xheader, yheader, fixed_values, ret=ret, output_dir=output_dir, **kwargs)):
        return False
    return True

def analyzeCategoricalToCategorical_via_api(data, xheader, yheader, fixed_values=None, ret=None, output_dir=None, api_url="http://10.1.3.236:5678", model_name="ACAngle", version="v0-0-2-0", timeout=30, **kwargs):
    """
    非連續型x對非連續型y的分析（API 版本）
    """
    if not API_ANALYSIS_AVAILABLE:
        m_addlog("錯誤：api_analysis 模組不可用，無法使用 API 版本", stamps=['DataAnalysis'], colora=LOGger.FAIL)
        return False
    
    if(not api_ds.analyzeCategoricalToCategorical(data, xheader, yheader, fixed_values, ret=ret, output_dir=output_dir, api_url=api_url, model_name=model_name, version=version, timeout=timeout, **kwargs)):
        return False
    return True

def analyzeTwoContinuousToY_via_api(data, xheader1, xheader2, yheader, y_type, fixed_values=None, include_model_prediction=True, ret=None, output_dir=None, api_url="http://10.1.3.236:5678", model_name="ACAngle", version="v0-0-2-0", timeout=30, **kwargs):
    """
    兩個連續型x對y的分析（API 版本）
    """
    if not API_ANALYSIS_AVAILABLE:
        m_addlog("錯誤：api_analysis 模組不可用，無法使用 API 版本", stamps=['DataAnalysis'], colora=LOGger.FAIL)
        return False
    
    if(not api_ds.analyzeTwoContinuousToY(data, xheader1, xheader2, yheader, y_type, fixed_values, include_model_prediction, ret=ret, output_dir=output_dir, api_url=api_url, model_name=model_name, version=version, timeout=timeout, **kwargs)):
        return False
    return True

def analyzeMixedToY_via_api(data, xheader1, xheader2, yheader, x1_type, x2_type, y_type, fixed_values=None, include_model_prediction=True, ret=None, output_dir=None, api_url="http://10.1.3.236:5678", model_name="ACAngle", version="v0-0-2-0", timeout=30, **kwargs):
    """
    混合型x對y的分析（API 版本）
    """
    if not API_ANALYSIS_AVAILABLE:
        m_addlog("錯誤：api_analysis 模組不可用，無法使用 API 版本", stamps=['DataAnalysis'], colora=LOGger.FAIL)
        return False
    
    if(not api_ds.analyzeMixedToY(data, xheader1, xheader2, yheader, x1_type, x2_type, y_type, fixed_values, include_model_prediction, ret=ret, output_dir=output_dir, api_url=api_url, model_name=model_name, version=version, timeout=timeout, **kwargs)):
        return False
    return True

def analyzeTwoCategoricalToY_via_api(data, xheader1, xheader2, yheader, y_type, fixed_values=None, include_model_prediction=True, ret=None, output_dir=None, api_url="http://10.1.3.236:5678", model_name="ACAngle", version="v0-0-2-0", timeout=30, **kwargs):
    """
    兩個非連續型x對y的分析（API 版本）
    """
    if not API_ANALYSIS_AVAILABLE:
        m_addlog("錯誤：api_analysis 模組不可用，無法使用 API 版本", stamps=['DataAnalysis'], colora=LOGger.FAIL)
        return False
    
    if(not api_ds.analyzeTwoCategoricalToY(data, xheader1, xheader2, yheader, y_type, fixed_values, include_model_prediction, ret=ret, output_dir=output_dir, api_url=api_url, model_name=model_name, version=version, timeout=timeout, **kwargs)):
        return False
    return True

if __name__ == '__main__':
    pass



