# 離散型數據聚合功能使用指南

## 概述

離散型數據聚合功能使用 LLM 智能聚合過於瑣碎的離散數據，將大量類別聚合成少數有意義的類別，提升數據分析效率和相關性計算的準確性。

## 快速開始

### 1. 確保 TextProcessor 服務運行

```bash
# 在 C:\ML_HOME\TextProcessor 目錄下
python main.py
```

### 2. 使用 API 端點

#### 方法 A: 僅聚合離散數據

```bash
curl -X POST "http://localhost:6030/discrete-aggregation" \
  -H "Content-Type: application/json" \
  -d '{
    "filePath": "path/to/your/data.csv",
    "discrete_columns": ["國家", "交通方式", "時間戳記"],
    "max_categories": 8,
    "llm_provider": "remote",
    "llm_model": "remote8b",
    "ret": {}
  }'
```

#### 方法 B: 聚合 + 相關性分析

```bash
curl -X POST "http://localhost:6030/discrete-aggregation-with-analysis" \
  -H "Content-Type: application/json" \
  -d '{
    "filePath": "path/to/your/data.csv", 
    "discrete_columns": ["國家", "交通方式"],
    "max_categories": 8,
    "use_fast_correlation": false,
    "n_jobs": 4,
    "ret": {}
  }'
```

### 3. Python 程式調用

```python
import asyncio
from src.discrete_aggregator import aggregate_discrete_data
import pandas as pd

async def main():
    # 載入數據
    df = pd.read_csv('your_data.csv')
    
    # 執行聚合
    df_aggregated, results = await aggregate_discrete_data(
        df,
        columns=['國家', '交通方式', '時間戳記'],
        max_categories=8
    )
    
    # 查看結果
    for result in results:
        if result['aggregated']:
            print(f"{result['column_name']}: {result['original_categories']} -> {result['final_categories']} 類別")

# 執行
asyncio.run(main())
```

## 聚合策略說明

### 時間數據聚合
**範例輸入:**
```
2023-01-01 09:30:00, 2023-01-01 14:20:00, 2023-01-01 19:45:00, ...
```

**聚合結果:**
```json
{
  "2023-01-01 09:30:00": "上午時段",
  "2023-01-01 14:20:00": "下午時段", 
  "2023-01-01 19:45:00": "晚上時段"
}
```

### 地理數據聚合
**範例輸入:**
```
中國, 日本, 韓國, 美國, 加拿大, 德國, 法國, ...
```

**聚合結果:**
```json
{
  "中國": "東亞",
  "日本": "東亞",
  "韓國": "東亞",
  "美國": "北美洲",
  "加拿大": "北美洲",
  "德國": "歐洲",
  "法國": "歐洲"
}
```

### 交通方式聚合
**範例輸入:**
```
公車, 捷運, 計程車, 自行車, 步行, 機車, 汽車, ...
```

**聚合結果:**
```json
{
  "公車": "公共交通",
  "捷運": "公共交通",
  "計程車": "公共交通",
  "自行車": "非機動交通",
  "步行": "非機動交通",
  "機車": "私人交通",
  "汽車": "私人交通"
}
```

## 參數配置

### 核心參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `filePath` | string | - | 數據文件路徑 |
| `discrete_columns` | array | null | 要聚合的欄位列表（null=自動檢測） |
| `max_categories` | integer | 10 | 聚合後最大類別數 |
| `textprocessor_url` | string | "http://localhost:6017" | TextProcessor 服務 URL |
| `llm_provider` | string | "remote" | LLM 提供者 |
| `llm_model` | string | "remote8b" | LLM 模型別名 |

### 分析參數（僅限 with-analysis 端點）

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `use_fast_correlation` | boolean | false | 是否使用快速 Pearson 相關性 |
| `n_jobs` | integer | null | 並行計算進程數（null=自動） |

## 回應格式

### 聚合端點回應

```json
{
  "success": true,
  "message": "離散數據聚合完成，處理了 3 個欄位",
  "aggregated_data": [
    {
      "國家": "東亞",
      "交通方式": "公共交通",
      "時間戳記": "上午時段",
      "年齡": 25,
      "收入": 50000
    }
  ],
  "aggregation_results": [
    {
      "column_name": "國家",
      "data_type": "geographic",
      "aggregated": true,
      "mapping": {
        "中國": "東亞",
        "日本": "東亞"
      },
      "original_categories": 15,
      "final_categories": 6,
      "timestamp": "2023-10-30T18:30:00"
    }
  ],
  "ret": {}
}
```

### 聚合+分析端點回應

```json
{
  "success": true,
  "message": "離散數據聚合和相關性分析完成。聚合了 3 個欄位，分析了 5 個欄位",
  "aggregated_data": [...],
  "analysis_data": [...],
  "aggregation_results": [...],
  "correlation_analysis": true,
  "analysis_columns": ["國家", "交通方式", "年齡", "收入", "評分"],
  "ret": {}
}
```

## 日誌監控

### 日誌位置
- **主日誌**: `log/discrete_aggregator_YYYYMMDD.txt`
- **JSON 日誌**: `jsonlog/YYYYMMDD/aggregation_HHMMSS_columnname.json`

### 日誌範例

**主日誌:**
```
2023-10-30 18:30:15 [INFO] 分析欄位 國家: 唯一值數量=15, 數據類型=geographic
2023-10-30 18:30:20 [WARNING] 開始聚合欄位 國家, 類型: geographic
2023-10-30 18:30:25 [SUCCESS] LLM 成功生成映射關係，包含 15 個映射
2023-10-30 18:30:26 [SUCCESS] 聚合完成: 國家 (15 -> 6 類別)
```

**JSON 日誌:**
```json
{
  "column_name": "國家",
  "data_type": "geographic", 
  "aggregated": true,
  "mapping": {
    "中國": "東亞",
    "日本": "東亞",
    "美國": "北美洲"
  },
  "original_categories": 15,
  "final_categories": 6,
  "timestamp": "2023-10-30T18:30:26.123456"
}
```

## 故障排除

### 常見錯誤

#### 1. TextProcessor 連接失敗
```json
{
  "success": false,
  "message": "處理失敗: LLM 調用失敗: HTTP 500"
}
```
**解決方法:**
- 確認 TextProcessor 服務正在運行 (`http://localhost:6017/health`)
- 檢查防火牆設置
- 驗證 LLM 模型配置

#### 2. 文件載入失敗
```json
{
  "success": false,
  "message": "數據載入失敗"
}
```
**解決方法:**
- 檢查文件路徑是否正確
- 確認文件格式支援 (CSV, Excel)
- 驗證文件權限

#### 3. 聚合失敗回退
```
[WARNING] 使用回退聚合策略
```
**說明:**
- LLM 調用失敗時自動使用基於頻率的聚合
- 結果仍然有效，但可能不如 LLM 聚合精確

### 性能優化

#### 小數據集 (< 1000 行)
```json
{
  "max_categories": 8,
  "use_fast_correlation": false
}
```

#### 中等數據集 (1000-10000 行)
```json
{
  "max_categories": 10,
  "n_jobs": 4,
  "use_fast_correlation": false
}
```

#### 大數據集 (> 10000 行)
```json
{
  "max_categories": 8,
  "n_jobs": -1,
  "use_fast_correlation": true
}
```

## 最佳實踐

### 1. 欄位選擇
- 優先聚合類別數 > 10 的欄位
- 避免聚合已經有意義的少量類別
- 考慮業務邏輯的重要性

### 2. 類別數設置
- 時間數據: 6-8 個類別
- 地理數據: 8-12 個類別  
- 交通方式: 4-6 個類別
- 一般類別: 8-10 個類別

### 3. LLM 選擇
- **OpenAI GPT-4**: 最佳聚合質量，但成本較高
- **Remote LLM**: 平衡質量和成本
- **回退策略**: 基於頻率，速度最快

### 4. 工作流程建議
1. 先執行聚合端點查看結果
2. 調整 `max_categories` 參數
3. 使用聚合+分析端點生成最終結果
4. 檢查日誌確認聚合質量

## 範例場景

### 場景 1: 客戶分析
**原始數據:**
- 國家: 50+ 個國家
- 城市: 200+ 個城市  
- 職業: 100+ 種職業

**聚合後:**
- 地區: 8 個大洲/地區
- 城市類型: 6 個等級
- 職業類別: 10 個大類

### 場景 2: 交通分析
**原始數據:**
- 交通工具: 20+ 種
- 時間: 精確到分鐘
- 路線: 100+ 條路線

**聚合後:**
- 交通類型: 5 個大類
- 時間段: 6 個時段
- 路線區域: 8 個區域

### 場景 3: 產品分析
**原始數據:**
- 產品型號: 500+ 個
- 品牌: 50+ 個
- 類別: 200+ 個

**聚合後:**
- 產品系列: 10 個系列
- 品牌等級: 6 個等級
- 主要類別: 8 個大類

通過離散數據聚合，可以顯著提升相關性分析的效果，讓數據模式更加清晰可見。
