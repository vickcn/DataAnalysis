# 離散型數據聚合功能

## 概述

本模組實現了智能的離散型數據聚合功能，使用 LLM 將過於瑣碎的離散數據聚合成較少的有意義類別，以提升數據分析效率。

## 功能特點

### 🤖 智能聚合
- 使用 LLM 根據語義相似性聚合數據
- 支援多種數據類型的專門聚合策略
- 自動檢測數據類型並選擇最適合的聚合方法

### 📊 數據類型支援
- **時間數據**: 聚合成時間段（上午、下午、晚上等）
- **地理數據**: 按洲、地區聚合
- **交通方式**: 按交通類型聚合
- **一般類別**: 基於語義相似性聚合

### ⚡ 性能優化
- 支援並行處理
- 智能回退機制
- 可配置的聚合參數

### 📝 完整日誌
- 詳細的聚合過程記錄
- JSON 格式的結構化日誌
- 時間戳記錄

## 安裝和依賴

### 必要依賴
```bash
pip install pandas numpy requests asyncio
```

### TextProcessor 服務
需要運行 TextProcessor 服務來提供 LLM 支援：
```bash
# 在 C:\ML_HOME\TextProcessor 目錄下
python main.py
```

## 使用方法

### 基本使用

```python
import asyncio
from src.discrete_aggregator import aggregate_discrete_data
import pandas as pd

# 載入數據
df = pd.read_csv('your_data.csv')

# 執行聚合
df_aggregated, results = await aggregate_discrete_data(
    df,
    columns=['country', 'transportation', 'datetime'],  # 指定要聚合的欄位
    max_categories=8,  # 聚合後最大類別數
    textprocessor_url="http://localhost:6017",
    llm_provider="remote",
    llm_model="remote8b"
)

print(f"聚合完成！")
for result in results:
    if result['aggregated']:
        print(f"{result['column_name']}: {result['original_categories']} -> {result['final_categories']} 類別")
```

### 進階使用

```python
from src.discrete_aggregator import DiscreteAggregator

# 創建聚合器實例
aggregator = DiscreteAggregator(
    textprocessor_url="http://localhost:6017",
    llm_provider="openai",  # 使用 OpenAI
    llm_model="gpt4o_chat",
    max_categories=10,
    min_frequency=2
)

# 分析單一欄位
analysis = aggregator.analyze_column(df['country'], 'country')
print(f"分析結果: {analysis}")

# 聚合單一欄位
result = await aggregator.aggregate_column(df['country'], 'country', analysis)
print(f"聚合結果: {result}")

# 應用聚合到整個 DataFrame
df_aggregated = aggregator.apply_aggregation(df, [result])
```

### 與相關性分析整合

```python
from src.integration_example import analyze_with_aggregation

# 完整的聚合 + 分析流程
results = await analyze_with_aggregation(
    file_path='your_data.csv',
    discrete_columns=['country', 'transportation', 'category'],
    max_categories=8,
    use_fast_correlation=False,  # 使用 MIC 處理離散數據
    n_jobs=4  # 並行計算
)

if results:
    print("分析完成！")
    print(f"聚合欄位: {len([r for r in results['aggregation_results'] if r['aggregated']])} 個")
```

## API 整合

### 添加到現有 API

```python
from fastapi import FastAPI
from src.api_integration import add_discrete_aggregation_endpoints

app = FastAPI()

# 添加離散數據聚合端點
add_discrete_aggregation_endpoints(app)
```

### API 端點

#### 1. `/discrete-aggregation` (POST)
僅執行離散數據聚合

**請求範例:**
```json
{
  "filePath": "path/to/data.csv",
  "sheet": 0,
  "discrete_columns": ["country", "transportation", "category"],
  "max_categories": 8,
  "textprocessor_url": "http://localhost:6017",
  "llm_provider": "remote",
  "llm_model": "remote8b",
  "ret": {}
}
```

#### 2. `/discrete-aggregation-with-analysis` (POST)
執行聚合 + 相關性分析

**請求範例:**
```json
{
  "filePath": "path/to/data.csv",
  "discrete_columns": ["country", "transportation"],
  "max_categories": 8,
  "use_fast_correlation": false,
  "n_jobs": 4,
  "ret": {}
}
```

## 配置選項

### DiscreteAggregator 參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `textprocessor_url` | str | "http://localhost:6017" | TextProcessor 服務 URL |
| `llm_provider` | str | "remote" | LLM 提供者 |
| `llm_model` | str | "remote8b" | LLM 模型別名 |
| `max_categories` | int | 10 | 聚合後最大類別數 |
| `min_frequency` | int | 1 | 保留類別的最小頻率 |

### 聚合策略

#### 時間數據聚合
- 自動識別時間格式
- 聚合成時間段（上午、下午、晚上等）
- 考慮工作日/週末、季節等

#### 地理數據聚合  
- 按洲聚合（亞洲、歐洲、北美洲等）
- 按地區聚合（華北、華南等）
- 考慮文化和經濟相似性

#### 交通方式聚合
- 按交通類型（公共交通、私人交通等）
- 按運輸方式（陸運、海運、空運）
- 最多聚合成 6 個類別

#### 一般類別聚合
- 基於語義相似性
- 保留高頻重要類別
- 合併低頻相似類別

## 日誌和監控

### 日誌文件
- 主日誌: `log/discrete_aggregator_YYYYMMDD.txt`
- JSON 日誌: `jsonlog/YYYYMMDD/aggregation_HHMMSS_columnname.json`

### 日誌內容
- 聚合過程詳情
- LLM 調用記錄
- 錯誤和警告信息
- 性能統計

## 故障排除

### 常見問題

#### 1. TextProcessor 連接失敗
```
錯誤: LLM 調用失敗: HTTP 500
```
**解決方法:**
- 確認 TextProcessor 服務正在運行
- 檢查 URL 和端口設置
- 驗證 LLM 模型配置

#### 2. JSON 解析失敗
```
錯誤: JSON 解析失敗: Expecting property name enclosed in double quotes
```
**解決方法:**
- 檢查 LLM 輸出格式
- 調整 temperature 參數降低隨機性
- 使用回退聚合策略

#### 3. 記憶體不足
```
錯誤: MemoryError
```
**解決方法:**
- 減少 max_categories 數量
- 分批處理大型數據集
- 使用 use_fast_correlation=True

### 性能優化建議

1. **小數據集** (< 1000 行): 使用預設設置
2. **中等數據集** (1000-10000 行): 設置 `n_jobs=4`
3. **大數據集** (> 10000 行): 
   - 設置 `n_jobs=-1`
   - 使用 `use_fast_correlation=True`
   - 考慮分批處理

## 範例

### 完整範例
參見 `integration_example.py` 中的完整使用範例。

### 測試
```bash
# 執行測試
python src/test_discrete_aggregator.py

# 執行整合範例
python src/integration_example.py
```

## 版本歷史

- **v1.0.0**: 初始版本，支援基本聚合功能
- 支援時間、地理、交通、一般類別聚合
- 整合 TextProcessor LLM 服務
- 完整的日誌和監控系統
