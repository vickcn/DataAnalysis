# DataAnalysis API 服務說明

> 本文件請使用 **UTF-8** 編碼讀寫（建議 UTF-8 without BOM），避免繁體中文顯示異常。

## 1) 服務用途

`DataAnalysis` 提供資料解析、相關性分析、單/雙變數分析、標準檢定、不穩定度分析、檔案轉換與評估報告等功能，並以 FastAPI 對外提供 REST API。

---

## 2) 啟動方式

安裝相依套件：

```bash
pip install -r requirements.txt
```

啟動 API：

```bash
python api_server.py
```

或使用啟動腳本：

```bash
python start_server.py
```

預設位址（依 `config.json` 而定）：

- API Base: `http://10.1.3.127:6030`
- Swagger: `http://10.1.3.127:6030/docs`
- ReDoc: `http://10.1.3.127:6030/redoc`

---

## 3) 主要設定（config.json）

- `Host_IP`：API 綁定 IP
- `Host_Port`：API 綁定埠號
- `source_dir`：預設資料來源目錄
- `referenceDirs`：備援搜尋目錄清單
- `gpuMemoLimit`：GPU 記憶體限制（若流程會用到）

---

## 4) API 一覽

### 4.1 基礎與瀏覽

- **GET** `/`：服務根路由（回傳首頁或基本服務資訊）
- **GET** `/config`：回傳服務設定
- **GET** `/files`：列目錄內容（`directory` 可選）
- **GET** `/search-file`：依檔名關鍵字搜尋（`filename` 必填）

### 4.2 前端頁面與靜態資源

- **GET** `/index.html`
- **GET** `/main.html`
- **GET** `/heteroscedastic.html`
- **GET** `/app.css`
- **GET** `/app.js`

### 4.3 資料解析與相關性

- **POST** `/data-parsing`
- **POST** `/dppc`
- **POST** `/plot-correlation`（非阻塞佇列：回 `job_id`；每任務獨立子目錄 + `job.json`；`n_jobs` 可選；可選 `spill_to_disk` / `spill_backend`（目前 `pkl`）/ `spill_keep_files`：MIC 前欄位分檔至 `.../spill/`，MIC 後還原繪圖，預設結束刪 spill 目錄）
- **GET** `/plot-correlation/status/{job_id}`、`GET` `/plot-correlation/jobs`
- **POST** `/plot-correlation/jobs/{job_id}/cancel`
- **POST** `/calculate-correlation`

### 4.4 單變數對單變數分析

- **POST** `/analyze-continuous-to-continuous`
- **POST** `/analyze-categorical-to-continuous`
- **POST** `/analyze-continuous-to-categorical`
- **POST** `/analyze-categorical-to-categorical`

### 4.5 單變數對單變數（via API 模型版本）

- **POST** `/analyze-continuous-to-continuous-via-api`
- **POST** `/analyze-categorical-to-continuous-via-api`
- **POST** `/analyze-continuous-to-categorical-via-api`
- **POST** `/analyze-categorical-to-categorical-via-api`

### 4.6 雙自變數分析

- **POST** `/two-dimension-analysis`
- **POST** `/analyze-two-continuous-to-y`
- **POST** `/analyze-two-categorical-to-y`
- **POST** `/analyze-mixed-to-y`

### 4.7 雙自變數（via API 模型版本）

- **POST** `/analyze-two-continuous-to-y-via-api`
- **POST** `/analyze-two-categorical-to-y-via-api`
- **POST** `/analyze-mixed-to-y-via-api`

### 4.8 統計檢定

- **POST** `/standard-tests`

### 4.9 不穩定度分析

- **POST** `/instability/compute`
- **POST** `/instability/save`
- **POST** `/instability/plot`

### 4.10 視覺化資料點

- **POST** `/data/points`

### 4.11 評估任務

- **POST** `/evaluationPlanTask`
- **POST** `/evaluationPlanTaskLLM`

### 4.12 檔案工具

- **POST** `/file/preview`
- **POST** `/file/convert-to-pkl`
- **POST** `/file/convert-pkl-protocol`

---

## 5) 常用請求欄位（重點）

### 5.1 共用欄位（多數分析 API）

- `filePath`（必填）：檔案路徑
- `sheet`（可選，預設 `0`）：Excel 工作表索引
- `output_dir`（可選）：輸出資料夾

### 5.2 單變數分析

- `xheader`、`yheader`（必填）
- `fixed_values`（可選）：固定條件過濾
- `include_model_prediction`（可選，預設 `true`）

### 5.3 雙自變數分析

- `xheader1`、`xheader2`、`yheader`（必填）
- `x1_type`、`x2_type`、`y_type`（混合分析必填）

### 5.4 via-api 版本額外欄位

- `api_url`（可選，預設 `http://10.1.3.236:5678`）
- `model_name`（可選，預設 `ACAngle`）
- `version`（可選，預設 `v0-0-2-0`）
- `timeout`（可選，預設 `30` 秒）

### 5.5 `/data/points`

- `columns`（必填）：輸出座標欄位清單（2D 或 3D）
- `fixed_values`（可選）：過濾條件
- `sample_n`（可選，預設 `2000`）
- `seed`（可選，預設 `7`）
- `include_extra_columns`（可選）：附加欄位，供前端 hover/click 顯示

### 5.6 `/instability/*`

- `x_col`、`y_col`（必填）
- `layers`（可選）
- `output_dir`（`save` / `plot` 可選；未給時走服務預設路徑）
- `ret.plotly_sample_n`（可選，預設 `3000`）：後端產生 Plotly 點數抽樣上限
- `ret.plotly_seed`（可選，預設 `7`）：後端 Plotly 抽樣隨機種子
- 回應 `res.plotly`：由後端直接給 `data/layout/config/meta`，可直接餵給 Plotly 前端渲染

### 5.7 `/evaluationPlanTask`

- `plan_id`（必填，正整數）
- `seq_no`（可選，預設 `1`）
- `stamps`（可選）
- `output_dir`（可選）

### 5.8 `/evaluationPlanTaskLLM`

- 含 `/evaluationPlanTask` 所有欄位，另有：
  - `provider`（可選，預設由系統設定）
  - `model`（可選，預設 `remote8b`）
  - `user_prompt_template`（可選）
  - `system_prompt`（可選）
  - `prompt_temperature`（可選）
  - `personality`（可選）
  - `enable_llm_analysis`（可選，預設 `true`）
  - `fH_prompt_alias_level`（可選，預設 `3`）

### 5.9 檔案工具

- `/file/preview`：`filePath`、`sheet`、`preview_rows`（預設 `5`）
- `/file/convert-to-pkl`：`filePath`、`sheet`、`protocol`（預設 `4`）、`output_dir`、`output_path`
- `/file/convert-pkl-protocol`：`filePath`、`protocol`（預設 `4`）、`output_dir`、`output_path`

---

## 6) 快速範例

### 6.1 Python

```python
import requests

base = "http://10.1.3.127:6030"

res = requests.post(
    f"{base}/analyze-continuous-to-continuous",
    json={
        "filePath": "data.xlsx",
        "sheet": 0,
        "xheader": "temperature",
        "yheader": "pressure"
    },
    timeout=120
)
print(res.json())
```

### 6.2 cURL

```bash
curl -X POST "http://10.1.3.127:6030/file/preview" \
  -H "Content-Type: application/json" \
  -d "{\"filePath\":\"data.xlsx\",\"sheet\":0,\"preview_rows\":5}"
```

---

## 7) 常見問題

1. **找不到檔案**
   - 若 `filePath` 非絕對路徑，服務會先以 `source_dir` 搜尋，再到 `referenceDirs` 搜尋。

2. **`/files` 無法瀏覽指定目錄**
   - 該端點會限制在可瀏覽根目錄下，超出範圍會回傳 400。

3. **中文亂碼**
   - 請確認檔案與編輯器都使用 UTF-8，並避免以 ANSI/Big5 另存覆蓋。
