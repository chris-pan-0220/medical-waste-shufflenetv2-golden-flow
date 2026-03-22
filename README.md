# in-use — NanoShuffleNetV2 INT8 ROM v4 驗證工作區

## 概述

此工作區包含 NanoShuffleNetV2_10k INT8 量化模型的 **ROM v4 (Split-ROM Architecture)**
端到端驗證流程。每個 Processing Unit (PU) 擁有 3 份獨立 ROM（Header / Weight / Bias），
共 3 PU × 3 = **9 個 ROM 實體**，允許並行讀取。

核心腳本 `verify_romv4.py` 從 9 個 ROM COE 檔讀取參數，以 RTL 等位寬的定點算術
模擬硬體推理路徑，並與 PyTorch INT8 模型 (QNNPACK) 的推理結果比對。

## 本版更新

- 完成此 repo 的 first-commit 基線整理，納入 ROM v4 驗證腳本、COE 檔、debug JSON、模型權重與設計文件。
- 更新 `workspace/generate_rom_v4.py`：將原本尚未完成、先以 `0` 佔位的記憶體流向欄位，改為依資料路徑手動填入正確的 `rd_mem` / `wr_mem`。
- `workspace/generate_rom_v4.py` 內各個 `# check:` 註解位置，現在明確標註 layer 間資料在哪個 buffer 讀取、寫回哪個目的 buffer，例如 `gb1`、`gb2`、`ib` 之間的切換。

## 快速開始

```bash
# 端到端驗證 (ROM v4 vs PyTorch INT8)
cd in-use/workspace
uv run verify_romv4.py --config config.toml

# 限制批次數 (debug 用)
uv run verify_romv4.py --config config.toml --max_batches 5

# 從 INT8 state dict 重新產生 9 個 ROM COE
uv run generate_rom_v4.py --workdir .

# 產生 Verilog 單通道量化管線測試向量
uv run gen_pwconv_testvec.py                           # 掃描 q_x=0..255
uv run gen_pwconv_testvec.py --range "0,140,255" --hex  # 指定值 + 十六進制
uv run gen_pwconv_testvec.py --mem-full testvec.mem     # 含所有中間訊號的 .mem 檔
```

### 配置檔

所有參數集中在 `workspace/config.toml`：

```toml
[paths]
workdir   = "."
data_root = "../datasets/selfbuilt_dataset_integrate"

[rom_v4]
pw_header    = "PWConv_header.coe"
pw_weight    = "PWConv_weight.coe"
pw_bias      = "PWConv_bias.coe"
dw_header    = "DWConv_header.coe"
dw_weight    = "DWConv_weight.coe"
dw_bias      = "DWConv_bias.coe"
gapfc_header = "GAP_FC_header.coe"
gapfc_weight = "GAP_FC_weight.coe"
gapfc_bias   = "GAP_FC_bias.coe"

[eval]
batch_size  = 64
img_size    = 128
seed        = 0
num_workers = 4
# max_batches = 10       # 取消註解以限制批次數 (debug 用)
```

> 所有路徑皆相對於 config.toml 所在的目錄。
> 版控中提供 `workspace/config.toml.example` 作為範本；請在本機維護自己的 `workspace/config.toml`。

## 檔案說明

### 腳本

| 檔案 | 說明 |
|---|---|
| `workspace/verify_romv4.py` | **主要驗證腳本** — ROM v4 端到端驗證，內嵌所有 RTL 算術核心 |
| `workspace/generate_rom_v4.py` | 從 INT8 state dict 產生 9 個 ROM COE 檔 + debug JSON，並在 header 中寫入各 layer 的 `rd_mem` / `wr_mem` buffer 流向設定 |
| `workspace/gen_pwconv_testvec.py` | 為 Verilog PWConv 單通道量化管線產生測試向量 |
| `workspace/nano_shufflenet_v2_05_10k.py` | 模型架構定義 (`NanoShuffleNetV2_10k`) |

### ROM v4 參數檔 (9 個 ROM — runtime 輸入)

| 檔案 | 格式 | 目標 PU | 說明 |
|---|---|---|---|
| `workspace/PWConv_header.coe` | .coe (u8) | PWConvUnit | Header (layer config, zero-points, requant) |
| `workspace/PWConv_weight.coe` | .coe (word) | PWConvUnit | 權重 (4-channel packed 32-bit words) |
| `workspace/PWConv_bias.coe` | .coe (word) | PWConvUnit | 偏置 (4-channel packed 64-bit words) |
| `workspace/DWConv_header.coe` | .coe (u8) | DWConvUnit | Header |
| `workspace/DWConv_weight.coe` | .coe (word) | DWConvUnit | 權重 (72-bit words, 3×3 kernel packed) |
| `workspace/DWConv_bias.coe` | .coe (word) | DWConvUnit | 偏置 (16-bit words) |
| `workspace/GAP_FC_header.coe` | .coe (u8) | GAP+FC | Header |
| `workspace/GAP_FC_weight.coe` | .coe (word) | GAP+FC | FC 權重 (byte-level) |
| `workspace/GAP_FC_bias.coe` | .coe (word) | GAP+FC | FC 偏置 (16-bit words) |

### ROM v4 Debug JSON (產生腳本附帶輸出)

| 檔案 | 說明 |
|---|---|
| `workspace/PWConv_header.json` | PWConv header 解析後的可讀格式 |
| `workspace/PWConv_weight.json` | PWConv 權重明細 |
| `workspace/PWConv_bias.json` | PWConv 偏置明細 |
| `workspace/DWConv_header.json` | DWConv header 解析後的可讀格式 |
| `workspace/DWConv_weight.json` | DWConv 權重明細 |
| `workspace/DWConv_bias.json` | DWConv 偏置明細 |
| `workspace/GAP_FC_header.json` | GAP+FC header 解析後的可讀格式 |
| `workspace/GAP_FC_weight.json` | GAP+FC 權重明細 |
| `workspace/GAP_FC_bias.json` | GAP+FC 偏置明細 |

### 模型權重

| 檔案 | 說明 |
|---|---|
| `workspace/nano_shufflenet_int8.pt` | INT8 量化模型 state dict |

### 設計參考文件

| 檔案 | 說明 |
|---|---|
| `workspace/ROM_v4.md` | ROM v4 (Split-ROM) 格式規格書 |
| `workspace/ROM_v4_design_notes.md` | ROM v4 設計決策補充說明 |
| `workspace/PWConvUnit-design.md` | PWConv 硬體單元設計文件 |

### 資料集

| 路徑 | 說明 |
|---|---|
| `datasets/selfbuilt_dataset_integrate` | 驗證用影像資料集 (ImageFolder 格式) |

## RTL 定點算術參數 (硬編碼)

| 常數 | 值 | 說明 |
|---|---|---|
| `N_FRAC` | 15 | 定點乘法器小數位數 |
| `S_PRE` | 0 | 預位移位數 (legacy, 未使用) |
| `ACC_BITS` | 22 | 有號累加器位寬 |
| `A_BITS` | 22 | 乘法器運算元位寬 |
| `M_BITS` | 16 | 無號乘法器位寬 |
| `BIAS_BITS` | 16 | 有號偏置位寬 |
| `POST_BITS` | 16 | requant 後位寬 |
