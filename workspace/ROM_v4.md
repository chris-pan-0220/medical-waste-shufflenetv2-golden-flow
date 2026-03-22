# Parameter ROM Specification (v4)

------------------------------------------------------------------------

# 0. 架構總覽：分散式獨立 ROM (Split-ROM Architecture)

每個硬體運算單元 (PU) 維護三份獨立的 ROM：

| ROM 類型 | 內容 | 定址模式 | 說明 |
|:---------|:-----|:---------|:-----|
| **Header ROM** | 控制 / 維度 / 量化參數 | byte-addressable | 每層讀取一次，以 linked-list 串接 |
| **Weight ROM** | int8 權重 | word-addressable（word 大小依 PU） | AGU 依 dataflow 計算 word address 讀取 |
| **Bias ROM** | int16 偏差值 | word-addressable（word 大小依 PU） | AGU 依 dataflow 計算 word address 讀取 |

三個 PU 共 **9 個 ROM 實體**：

| # | ROM 實體名稱 | 定址模式 | Word 位寬 | 說明 |
|:-:|:------------|:---------|:---------:|:-----|
| 1 | PWConvUnit Header ROM | byte | 8 bits | 控制參數，含 concat-align |
| 2 | PWConvUnit Weight ROM | word | P×8 bits | P = 輸出通道並行度（見 PWConvUnit-design.md） |
| 3 | PWConvUnit Bias ROM | word | **P×16 bits** | P 個 int16 bias，配合 P 路並行 output channel |
| 4 | DWConvUnit Header ROM | byte | 8 bits | 控制參數，含 stride |
| 5 | DWConvUnit Weight ROM | word | **72 bits** | 9 bytes = 一個 3×3 kernel，FPGA 原生位寬 |
| 6 | DWConvUnit Bias ROM | word | 16 bits | int16 bias |
| 7 | GAP_FC Header ROM | byte | 8 bits | 控制參數 |
| 8 | GAP_FC Weight ROM | word | TBD | 待 GAP_FC_Unit RTL 設計確定 |
| 9 | GAP_FC Bias ROM | word | 16 bits | int16 bias |

```
┌─── PWConvUnit ───────────────────────────┐
│  Header ROM ──► Weight ROM ──► Bias ROM  │
│  (byte-addr)    (P×8-bit)     (P×16-bit) │
└──────────────────────────────────────────┘

┌─── DWConvUnit ───────────────────────────┐
│  Header ROM ──► Weight ROM ──► Bias ROM  │
│  (byte-addr)    (72-bit)      (16-bit)   │
└──────────────────────────────────────────┘

┌─── GAP_FC ───────────────────────────────┐
│  Header ROM ──► Weight ROM ──► Bias ROM  │
│  (byte-addr)    (TBD)         (16-bit)   │
└──────────────────────────────────────────┘

每份 ROM 獨立 address space，各自從 address 0 開始。
```

`layer_type` 對照：

| `layer_type` | 值 | 所屬 PU ROM 群組 | 硬體運算單元 | 說明 |
|:------------:|:--:|:---------------:|:----------:|------|
| PWConv       | 0  | PWConvUnit      | PWConvUnit  | Pointwise 1×1 Convolution |
| DWConv       | 1  | DWConvUnit      | DWConvUnit  | Depthwise 3×3 Convolution（groups = C_out） |
| NormalConv   | 2  | DWConvUnit      | DWConvUnit  | Standard 3×3 Convolution（groups = 1） |
| GAP_FC       | 3  | GAP_FC          | GAP_FC_Unit | Global Average Pooling + Fully Connected |

> 設計特點、設計決策與更改紀錄詳見 [ROM_v4_design_notes.md](ROM_v4_design_notes.md)。

------------------------------------------------------------------------

# 1. 共用概念

## 1.1 Linked-List 串接

每個 PU 的 Header ROM 各自以 linked-list 串接該 PU 內所有 layer：

-   各 Header ROM 的第一個 layer 從 **byte address 0** 開始
-   `next_layer_base_addr` 指向同一份 Header ROM 內下一層的 byte address
-   最後一層填 `LAST_LAYER (0xFFFF)`

```
Header ROM（某一 PU）：

+---------------------------+   ← byte address 0
| Layer 0 Header            |
+---------------------------+   ← next_layer_base_addr of Layer 0
| Layer 1 Header            |
+---------------------------+   ← next_layer_base_addr of Layer 1
| ...                       |
+---------------------------+
| Layer N-1 Header          |   ← next_layer_base_addr = LAST_LAYER
+---------------------------+
```

Weight ROM 與 Bias ROM 沒有 linked-list 結構——各層的資料位置由 Header ROM 中的 `weight_base` / `bias_base` 直接指定。

## 1.2 位址模式

| ROM 類型 | 位址單位 | `base` / `addr` 含義 |
|:---------|:--------|:--------------------|
| Header ROM | byte | `next_layer_base_addr` = 下一層 header 起始 byte address |
| Weight ROM | word（大小依 PU） | `weight_base` = 該層第一個 weight word 的 word address |
| Bias ROM | 16-bit word（PWConvUnit 為 P×16-bit word） | `bias_base` = 該層第一個 bias word 的 word address |

AGU 運作模式：

-   **Header ROM**：讀取一層 header 時，從 `layer_base` 開始逐 byte 讀取，長度 = `HEADER_SIZE`（PU-specific，固定值）
-   **Weight ROM**：以 `weight_base` 為基準，AGU 依 dataflow 邏輯計算每次所需的 word address（例如根據當前 c_out、c_in 等索引），一次讀取一個完整 word
-   **Bias ROM**：以 `bias_base` 為基準，AGU 依 dataflow 邏輯計算所需的 word address，一次讀取一個 bias word（PWConvUnit 每個 word 含 P 個 int16 bias，配合 P 路並行）

> **注意**：Weight / Bias ROM 的讀取順序由各 PU 的 dataflow 決定，不一定是純粹的位址遞增。
> 例如 DWConvUnit 可能先輸出 output feature map c0 的 row0，再輸出 c1 的 row0……
> AGU 根據 dataflow 的迴圈結構動態計算 word address。

## 1.3 layer_type 行為對照

| 行為 | PWConv (0) | DWConv (1) | NormalConv (2) | GAP_FC (3) |
|------|:----------:|:----------:|:--------------:|:----------:|
| 運算單元 | PWConvUnit | DWConvUnit | DWConvUnit | GAP_FC_Unit |
| Kernel | 1×1 | 3×3 | 3×3 | N/A (FC = matrix mult) |
| Groups | 1 | C_out | 1 | 1 |
| ReLU | on | off | on | off |
| Padding | 0 | 1 | 1 | N/A |

`layer_type` 在各 PU 中：

-   **DWConvUnit Header ROM**：用於區分 DWConv (1) / NormalConv (2)，影響 groups 與 ReLU 行為
-   **PWConvUnit Header ROM**：永遠為 0（sanity check 用）
-   **GAP_FC Header ROM**：永遠為 3（sanity check 用）

## 1.4 rd_mem / wr_mem

記憶體選擇，指定硬體讀寫哪一塊 BRAM：

| 值 | 意義 |
|:--:|------|
| 0  | 無寫入目標（僅 `wr_mem`，用於 GAP_FC 輸出 → 外部介面如 LED） |
| 1  | Global BRAM 1 |
| 2  | Global BRAM 2 |
| 3  | Intermediate BRAM |

> **注意**：`rd_mem` / `wr_mem` 為 **RTL 設計參數**，golden sample Python 不讀取此欄位。
> ROM 產生器（compile 工具）從外部人工編輯的設定檔讀取這兩個欄位的值（見 Appendix B）。

## 1.5 Shape 語義

| 欄位 | PWConv | DWConv | NormalConv | GAP_FC |
|------|:------:|:------:|:----------:|:------:|
| `C_out` | 輸出通道數 | 通道數（= C_in） | 輸出通道數 | FC 輸出維度（num_classes） |
| `C_in` | 輸入通道數 | 通道數（= C_out） | 輸入通道數 | FC 輸入維度 = GAP 通道數 |
| `H` | feature map 高度（input=output） | 輸入 feature map 高度 | 輸入 feature map 高度 | GAP 輸入空間高度 |
| `W` | feature map 寬度（input=output） | 輸入 feature map 寬度 | 輸入 feature map 寬度 | GAP 輸入空間寬度 |

## 1.6 量化公式

### m_requant

$$m_{\text{requant}} = \text{round}(M \cdot 2^{N})$$

其中 $M = \dfrac{s_x \cdot s_w}{s_y}$，$N = 15$（`N_FRAC`）。以 `M_BITS = 16` 位無號數儲存。適用所有 layer type。

### m_align（僅 PWConv、且 `concat_align==1`）

$$m_{\text{align}} = \text{round}\!\left(\dfrac{s_2}{s_1} \cdot 2^{N}\right)$$

其中 $s_2$ = branch2 output scale，$s_1$ = branch1 output scale。以 `M_BITS = 16` 位無號數儲存。

### Bias 離線計算

$$b = \text{round}\!\left(\frac{b_{f32}}{s_x \cdot s_w}\right)$$

以 `BIAS_BITS = 16` 位有號數（int16）儲存。適用所有具備 bias 的 layer type。

------------------------------------------------------------------------

# 2. PWConvUnit ROM 群組

## 2.1 Header ROM

```
PWCONV_HEADER_SIZE = 26
```

| Offset | Size (bytes) | Name | Type | 說明 |
|:------:|:------------:|------|:----:|------|
| +0  | 2 | `next_layer_base_addr` | u16 LE | 下一層 header 的 byte address；最後一層 = `LAST_LAYER` |
| +2  | 1 | `layer_type`           | u8     | 固定 = 0（PWConv；sanity check 用） |
| +3  | 2 | `weight_base`          | u16 LE | 該層第一個 weight word 在 PWConvUnit Weight ROM 的 word address |
| +5  | 2 | `bias_base`            | u16 LE | 該層第一個 bias word 在 PWConvUnit Bias ROM 的 word address |
| +7  | 2 | `C_out`                | u16 LE | 輸出通道數 |
| +9  | 2 | `C_in`                 | u16 LE | 輸入通道數 |
| +11 | 1 | `H`                    | u8     | 輸入 feature map 高度（= 輸出，PWConv stride 永遠 = 1） |
| +12 | 1 | `W`                    | u8     | 輸入 feature map 寬度（= 輸出） |
| +13 | 1 | `shuffle_mode`         | u8     | 0 = no shuffle；1 = channel shuffle 寫出 |
| +14 | 1 | `branch`               | u8     | 0 = branch1；1 = branch2 |
| +15 | 1 | `concat_align`         | u8     | 1 = 執行 concat-align；0 = 不執行 |
| +16 | 1 | `rd_mem`               | u8     | 讀取 BRAM 目標（RTL 設計參數） |
| +17 | 1 | `wr_mem`               | u8     | 寫入 BRAM 目標（RTL 設計參數） |
| +18 | 1 | `z_x`                  | u8     | 輸入 zero point |
| +19 | 1 | `z_y`                  | u8     | 輸出 zero point |
| +20 | 2 | `m_requant`            | u16 LE | Re-quantization 乘數 |
| +22 | 1 | `z1_align`             | u8     | Concat-align 目標 zero point（branch1 域） |
| +23 | 1 | `z2_align`             | u8     | Concat-align 來源 zero point（branch2 域） |
| +24 | 2 | `m_align`              | u16 LE | Concat-align 乘數 |

### 欄位補充

#### shuffle_mode / branch / concat_align

詳見 PWConvUnit-design.md § 7.6：

| shuffle_mode | branch | out_c' 公式                    | 使用場景 |
|:------------:|:------:|-------------------------------|----------|
| 0            | —      | `out_c' = c_out`              | 非最後 PWConv（中間結果） |
| 1            | 0      | `out_c' = 2 * c_out`          | stride-2 block branch1 最後 PW |
| 1            | 1      | `out_c' = 2 * (c_out - C_out/2) + 1` | stride-2/stride-1 block branch2 最後 PW |

#### z1_align / z2_align / m_align

-   當 `concat_align == 1` 時，硬體在 PWConv requant + ReLU 之後自動執行 concat-align（見 Appendix A）
-   當 `concat_align == 0` 時，硬體忽略這三個欄位（ROM 中為 don't-care）

> **備註**：邏輯上 `z2_align` 等於同一層的 `z_y`（PWConv 輸出 zero point），
> 但仍獨立儲存以簡化硬體路由——align 單元可直接從 ROM 讀取，不需從 requant 管線回傳 `z_y`。

## 2.2 Weight ROM

-   **Word 位寬**：P × 8 bits（P = PWConvUnit 輸出通道並行度，見 PWConvUnit-design.md）
-   元素型別：int8（對稱量化 $z_w = 0$）

### Data Layout（Group-by-P）

每個 word 包含 P 個 weight，對應同一 `c_in` 下 P 個連續 `c_out` 的權重：

```
word_addr = weight_base + c_out_group * C_in + c_in
```

其中 `c_out_group = c_out / P`，`p = c_out % P`。

Word 內第 `p` 個 byte（bits `[p*8+7 : p*8]`）= weight[c_out_group * P + p][c_in]。

| 項目 | 值 |
|------|:---|
| 每層 word 數 | `(C_out / P) × C_in` |
| 每層 weight 總量 | `C_out × C_in` bytes |

### 跨層排列

Weight ROM 中各層資料依序排列，無 padding：

```
PWConvUnit Weight ROM:

  word 0 ~ (layer0_words - 1)      ← Layer 0 weights
  word layer0_words ~ ...           ← Layer 1 weights
  ...
```

各層 `weight_base` 值由 compiler 累計前面所有層的 word 數得出。

## 2.3 Bias ROM

-   **Word 位寬**：P × 16 bits（P 個 int16 bias，配合 P 路並行 output channel 同時讀取）
-   元素型別：int16 LE

每個 word 包含 P 個 bias，對應同一個 `c_out_group` 下 P 個連續 output channel 的 bias：

```
word_addr = bias_base + c_out_group
```

其中 `c_out_group = c_out / P`。Word 內第 `p` 個 int16（bits `[p*16+15 : p*16]`）= bias[c_out_group * P + p]。

| 項目 | 值 |
|------|:---|
| 每層 word 數 | `C_out / P` |
| 每層 bias 總量 | `C_out` 個 int16 |

Bias ROM 中各層資料依序排列，無 padding。各層 `bias_base` 值由 compiler 累計前面所有層的 `C_out / P` 得出。

------------------------------------------------------------------------

# 3. DWConvUnit ROM 群組

## 3.1 Header ROM

```
DWCONV_HEADER_SIZE = 20
```

| Offset | Size (bytes) | Name | Type | 說明 |
|:------:|:------------:|------|:----:|------|
| +0  | 2 | `next_layer_base_addr` | u16 LE | 下一層 header 的 byte address；最後一層 = `LAST_LAYER` |
| +2  | 1 | `layer_type`           | u8     | 1 = DWConv，2 = NormalConv |
| +3  | 2 | `weight_base`          | u16 LE | 該層第一個 weight word 在 DWConvUnit Weight ROM 的 word address |
| +5  | 2 | `bias_base`            | u16 LE | 該層第一個 bias word 在 DWConvUnit Bias ROM 的 word address |
| +7  | 2 | `C_out`                | u16 LE | 輸出通道數 |
| +9  | 2 | `C_in`                 | u16 LE | 輸入通道數 |
| +11 | 1 | `H`                    | u8     | 輸入 feature map 高度 |
| +12 | 1 | `W`                    | u8     | 輸入 feature map 寬度 |
| +13 | 1 | `stride`               | u8     | 卷積步幅（1 或 2） |
| +14 | 1 | `rd_mem`               | u8     | 讀取 BRAM 目標（RTL 設計參數） |
| +15 | 1 | `wr_mem`               | u8     | 寫入 BRAM 目標（RTL 設計參數） |
| +16 | 1 | `z_x`                  | u8     | 輸入 zero point |
| +17 | 1 | `z_y`                  | u8     | 輸出 zero point |
| +18 | 2 | `m_requant`            | u16 LE | Re-quantization 乘數 |

### 欄位補充

#### layer_type

-   `1 (DWConv)`：Depthwise 卷積，groups = C_out，ReLU off
-   `2 (NormalConv)`：Standard 卷積，groups = 1，ReLU on

此欄位是 DWConvUnit 最關鍵的 dispatch 欄位——同一個硬體單元根據 `layer_type` 決定 groups 與 ReLU 行為。

#### stride

-   `1` 或 `2`
-   輸出空間尺寸：`H_out = H / stride`，`W_out = W / stride`

## 3.2 Weight ROM (72-bit FPGA 原生位寬)

-   **Word 位寬**：**72 bits**（9 bytes）
-   元素型別：int8（對稱量化 $z_w = 0$）
-   每個 word 恰好儲存一個完整的 **3×3 kernel**

### FPGA BRAM 原生位寬說明

現代 FPGA 的 Block RAM（如 Xilinx 7-series / UltraScale 的 RAMB36E1）在物理結構上原生支援 9-bit / 18-bit / 36-bit / 72-bit 的資料位寬（原本設計用來包含 byte parity check bit）。

在 SystemVerilog 中直接宣告 / 例化一個 72-bit 寬度的 ROM：

```systemverilog
logic [71:0] dw_weight_rom [0:DEPTH-1];
```

如此一來：
-   每個 3×3 kernel（9 bytes = 72 bits）剛好是一個 BRAM word
-   100% 空間利用率，零 padding 浪費
-   AGU 依 dataflow 計算 word address，每次讀取即為一個完整 3×3 kernel

### DWConv Data Layout

DWConv（groups = C_out = C_in）：每個 channel 一個 3×3 kernel。

```
word_addr = weight_base + c_out
```

| 項目 | 值 |
|------|:---|
| 每層 word 數 | `C_out` |
| 每層 weight 總量 | `C_out × 9` bytes |

### NormalConv Data Layout

NormalConv（groups = 1）：每個 (c_out, c_in) 組合一個 3×3 kernel。

```
word_addr = weight_base + c_out * C_in + c_in
```

| 項目 | 值 |
|------|:---|
| 每層 word 數 | `C_out × C_in` |
| 每層 weight 總量 | `C_out × C_in × 9` bytes |

### Word 內 Byte 排列

每個 72-bit word 中 9 個 int8 weight 的排列（row-major）：

```
bits[7:0]   = w[0][0]   (kh=0, kw=0)
bits[15:8]  = w[0][1]   (kh=0, kw=1)
bits[23:16] = w[0][2]   (kh=0, kw=2)
bits[31:24] = w[1][0]   (kh=1, kw=0)
bits[39:32] = w[1][1]   (kh=1, kw=1)
bits[47:40] = w[1][2]   (kh=1, kw=2)
bits[55:48] = w[2][0]   (kh=2, kw=0)
bits[63:56] = w[2][1]   (kh=2, kw=1)
bits[71:64] = w[2][2]   (kh=2, kw=2)
```

即 byte index = `kh * 3 + kw`，maps to bits `[(kh*3+kw+1)*8-1 : (kh*3+kw)*8]`。

> **備註**：此 word 內 byte 排列（row-major）與 word 間的排列（c_out 為外層）
> 即為 golden sample 與最終 runtime 共用的 memory layout。
> AGU 根據 dataflow 的迴圈結構在 runtime 計算所需的 word address。

### 跨層排列

DWConvUnit Weight ROM 中各層資料依序排列，無 padding。layer 的排列順序即為 DWConvUnit 內的執行順序。

## 3.3 Bias ROM

-   **Word 位寬**：16 bits
-   元素型別：int16 LE

```
word_addr = bias_base + c_out
```

| 項目 | 值 |
|------|:---|
| 每層 word 數 | `C_out` |

> **備註**：Bias 的排列（`c_out = 0, 1, 2, ...` 循序）即為 golden sample 與最終 runtime 共用的 memory layout。
> AGU 根據 dataflow 的迴圈結構在 runtime 計算所需的 word address。

------------------------------------------------------------------------

# 4. GAP_FC ROM 群組

## 4.1 Header ROM

```
GAPFC_HEADER_SIZE = 19
```

| Offset | Size (bytes) | Name | Type | 說明 |
|:------:|:------------:|------|:----:|------|
| +0  | 2 | `next_layer_base_addr` | u16 LE | 下一層 header 的 byte address；最後一層 = `LAST_LAYER` |
| +2  | 1 | `layer_type`           | u8     | 固定 = 3（GAP_FC；sanity check 用） |
| +3  | 2 | `weight_base`          | u16 LE | 該層第一個 weight word 在 GAP_FC Weight ROM 的 word address |
| +5  | 2 | `bias_base`            | u16 LE | 該層第一個 bias word 在 GAP_FC Bias ROM 的 word address |
| +7  | 2 | `C_out`                | u16 LE | FC 輸出維度（num_classes） |
| +9  | 2 | `C_in`                 | u16 LE | FC 輸入維度 = GAP 通道數 |
| +11 | 1 | `H`                    | u8     | GAP 輸入空間高度 |
| +12 | 1 | `W`                    | u8     | GAP 輸入空間寬度 |
| +13 | 1 | `rd_mem`               | u8     | 讀取 BRAM 目標（RTL 設計參數） |
| +14 | 1 | `wr_mem`               | u8     | 寫入 BRAM 目標（RTL 設計參數；0 = 無寫入，輸出至外部介面） |
| +15 | 1 | `z_x`                  | u8     | 輸入 zero point（GAP 輸入 ZP = FC 輸入 ZP） |
| +16 | 1 | `z_y`                  | u8     | 輸出 zero point（FC 輸出 ZP） |
| +17 | 2 | `m_requant`            | u16 LE | FC re-quantization 乘數 |

### 欄位補充

-   **GAP_FC 的 z_x 共用**：GAP（AdaptiveAvgPool2d）保持輸入的 quantization parameters 不變，故 FC 的 z_x 等於 GAP 的 z_x，只需一個 z_x 欄位。
-   **GAP 的 H×W 除法**：可由 header 中的 H、W 推導。本模型為 8×8 = 64 = 2^6，硬體可直接右移 6 位元。

## 4.2 Weight ROM

-   元素型別：int8（對稱量化 $z_w = 0$）
-   **Word 位寬**：**TBD**（待 GAP_FC_Unit RTL 設計確定）

### Data Layout

FC 邏輯上等同 PWConv 的線性化：

```
logical_index = C_in * c_out + c_in
```

| 項目 | 值 |
|------|:---|
| GAP weight | 無 |
| FC weight 數量 | `C_out × C_in` |

> **備註**：此邏輯索引即為 golden sample 與最終 runtime 共用的 memory layout。
> GAP_FC Weight ROM 的 word 位寬待 GAP_FC_Unit RTL 設計確定後決定。
> AGU 根據 dataflow 的迴圈結構在 runtime 計算所需的 word address。

## 4.3 Bias ROM

-   **Word 位寬**：16 bits
-   元素型別：int16 LE

```
word_addr = bias_base + c_out
```

| 項目 | 值 |
|------|:---|
| 每層 word 數 | `C_out` |

> **備註**：Bias 排列（`c_out = 0, 1, 2, ...` 循序）即為 golden sample 與最終 runtime 共用的 memory layout。
> AGU 根據 dataflow 的迴圈結構在 runtime 計算所需的 word address。

------------------------------------------------------------------------

# 5. Endianness 規則

## Header ROM（所有 PU 共用規則）

Header ROM 為 byte-addressable，多 byte 欄位以 little-endian 儲存：

| 資料 | 型別 | Endianness |
|------|------|------------|
| `next_layer_base_addr` | u16 | little-endian |
| `weight_base` | u16 | little-endian |
| `bias_base` | u16 | little-endian |
| `C_out` | u16 | little-endian |
| `C_in` | u16 | little-endian |
| `m_requant` | u16 | little-endian |
| `m_align`（僅 PWConvUnit） | u16 | little-endian |
| 所有 u8 欄位 | u8 | —（1 byte） |

## Weight ROM

Weight ROM 為 word-addressable，每個 word 以**整體**讀出：

| PU | Word 位寬 | 內部 byte 排列 |
|----|:---------:|:--------------|
| PWConvUnit | P×8 bits | byte[p] = bits[(p+1)*8-1 : p*8]，p = c_out % P |
| DWConvUnit | 72 bits | byte[kh*3+kw] = bits[(kh*3+kw+1)*8-1 : (kh*3+kw)*8] |
| GAP_FC | TBD | TBD |

每個 weight 元素為 **int8**（1 byte），無 endianness 問題。

## Bias ROM

| PU | Word 位寬 | 內部排列 |
|----|:---------:|:------------|
| PWConvUnit | P×16 bits | int16[p] = bits[(p+1)*16-1 : p*16]，p = c_out % P |
| DWConvUnit | 16 bits | 單一 int16 |
| GAP_FC | 16 bits | 單一 int16 |

每個 bias 元素為 **int16**，以 little-endian 儲存。

------------------------------------------------------------------------

# 6. 終止與保留值

```
LAST_LAYER = 0xFFFF     ← next_layer_base_addr 的終止標記（各 Header ROM 各自獨立）
```

-   各 PU Header ROM 的第一層從 byte address `0` 開始
-   硬體讀到 `next_layer_base_addr == LAST_LAYER` 即知該 PU 內所有層已處理完畢
-   Weight ROM 與 Bias ROM 無終止標記——由 Header ROM 提供的 `weight_base` / `bias_base` + shape 推算長度

> **注意**：v4 取消了 `NOT_APPLICABLE_U8 (0xFF)` 與 `NOT_APPLICABLE_U16 (0xFFFF)` 的填充規則。
> 各 PU Header 已客製化，不存在「不適用的欄位」。
> `0xFFFF` 僅保留作為 `LAST_LAYER` 終止標記。

------------------------------------------------------------------------

------------------------------------------------------------------------

# Appendix A — Concat-Align（PWConvUnit 專屬）

Concat-align 參數（`z1_align`、`z2_align`、`m_align`）整合進 PWConvUnit Header ROM，
由硬體根據 `concat_align` 欄位決定是否執行 re-alignment，流程為：

1. PWConv 1×1 conv → requant → ReLU → 得到 $q_y$（uint8）
2. 若 `concat_align == 1`：

   $$d = q_y - z_{2,\text{align}}$$
   $$p = \text{wrap\_sint}(d \times m_{\text{align}},\; 9 + M_{\text{BITS}})$$
   $$u = p \ggg N_{\text{FRAC}}$$
   $$q_{\text{out}} = \text{sat\_u8}(u + z_{1,\text{align}})$$

3. Channel shuffle write（依 `shuffle_mode` / `branch` 決定 output address）

當 `concat_align == 0` 時，步驟 2 跳過，直接以 $q_y$ 進入步驟 3。

> **注意**：`concat_align` 與 `branch` 欄位**獨立**。並非所有 `branch==1` 的層都需要 concat-align，
> 只有 branch2 的**最後一個 PWConv**（即 `*.branch2.5`）才需要。

------------------------------------------------------------------------

# Appendix B — rd_mem / wr_mem 外部設定檔

`rd_mem` 和 `wr_mem` 為 RTL 設計參數，不從模型檔案中推導，而是由人工編輯一份外部設定檔提供。

ROM 產生器（compile 工具）在建構 ROM 時：
1. 從模型檔案讀取所有模型參數（weight, bias, scale, zero_point, shape 等）
2. 從外部設定檔讀取每一層的 `rd_mem` / `wr_mem` 值
3. 分別寫入各 PU 的三份 ROM（Header ROM、Weight ROM、Bias ROM）

### 設定檔格式建議

使用 TOML 或 JSON，以層名稱為 key：

```toml
# mem_config.toml — BRAM 讀寫配置（人工編輯）
# val: 0=none, 1=global_bram_1, 2=global_bram_2, 3=intermediate_bram

[layers]

[layers."conv1.0"]
rd_mem = 1
wr_mem = 2

[layers."stages.0.0.branch1.0"]
rd_mem = 2
wr_mem = 3

[layers."stages.0.0.branch1.2"]
rd_mem = 3
wr_mem = 1

# ... 其餘各層 ...

[layers."gap_fc"]
rd_mem = 1
wr_mem = 0    # 輸出直接控制 LED，無 BRAM 寫入
```

> **注意**：設定檔中的層名稱必須與 ROM 產生器內部的層名稱列表完全對應。

------------------------------------------------------------------------

# Appendix C — Stride-1 Block 的 Branch1 Passthrough

在 stride-1 block 中，branch1（`x1`）是直接 passthrough（取前半 channels），沒有對應的卷積運算。
若最終 output 需要做 channel shuffle 寫出，`x1` 的 shuffle 寫入需由其他機制處理（如 DMA / 獨立搬移單元），
**不應**在任何 ROM 中為 branch1 passthrough 建立 layer entry。

------------------------------------------------------------------------

# Appendix D — 執行順序與 ROM 分配參考（NanoShuffleNetV2_10k）

以 `stage_repeats = [2, 2, 2]` 為例，全部 layer 的全域執行順序如下。
各 PU ROM 群組中 layer 的排列順序即為該 PU 內 layer 在此表中出現的相對順序。

## D.1 全域執行順序

| 序號 | Layer 名稱 | layer_type | 所屬 PU | C_out | C_in | H | W | stride | 備註 |
|:----:|-----------|:----------:|:------:|:-----:|:----:|:-:|:-:|:------:|------|
| 0 | conv1.0 | NormalConv | DWConvUnit | 16 | 3 | 128 | 128 | 2 | 首層標準卷積 |
| 1 | stages.0.0.branch1.0 | DWConv | DWConvUnit | 16 | 16 | 64 | 64 | 2 | stride-2 block |
| 2 | stages.0.0.branch1.2 | PWConv | PWConvUnit | 12 | 16 | 32 | 32 | — | shuffle_mode=1, branch=0 |
| 3 | stages.0.0.branch2.0 | PWConv | PWConvUnit | 12 | 16 | 64 | 64 | — | |
| 4 | stages.0.0.branch2.3 | DWConv | DWConvUnit | 12 | 12 | 64 | 64 | 2 | |
| 5 | stages.0.0.branch2.5 | PWConv | PWConvUnit | 12 | 12 | 32 | 32 | — | shuffle_mode=1, branch=1, concat_align=1 |
| 6 | stages.0.1.branch2.0 | PWConv | PWConvUnit | 12 | 12 | 32 | 32 | — | stride-1 block |
| 7 | stages.0.1.branch2.3 | DWConv | DWConvUnit | 12 | 12 | 32 | 32 | 1 | |
| 8 | stages.0.1.branch2.5 | PWConv | PWConvUnit | 12 | 12 | 32 | 32 | — | shuffle_mode=1, branch=1, concat_align=1 |
| 9 | stages.1.0.branch1.0 | DWConv | DWConvUnit | 24 | 24 | 32 | 32 | 2 | stride-2 block |
| 10 | stages.1.0.branch1.2 | PWConv | PWConvUnit | 16 | 24 | 16 | 16 | — | shuffle_mode=1, branch=0 |
| 11 | stages.1.0.branch2.0 | PWConv | PWConvUnit | 16 | 24 | 32 | 32 | — | |
| 12 | stages.1.0.branch2.3 | DWConv | DWConvUnit | 16 | 16 | 32 | 32 | 2 | |
| 13 | stages.1.0.branch2.5 | PWConv | PWConvUnit | 16 | 16 | 16 | 16 | — | shuffle_mode=1, branch=1, concat_align=1 |
| 14 | stages.1.1.branch2.0 | PWConv | PWConvUnit | 16 | 16 | 16 | 16 | — | stride-1 block |
| 15 | stages.1.1.branch2.3 | DWConv | DWConvUnit | 16 | 16 | 16 | 16 | 1 | |
| 16 | stages.1.1.branch2.5 | PWConv | PWConvUnit | 16 | 16 | 16 | 16 | — | shuffle_mode=1, branch=1, concat_align=1 |
| 17 | stages.2.0.branch1.0 | DWConv | DWConvUnit | 32 | 32 | 16 | 16 | 2 | stride-2 block |
| 18 | stages.2.0.branch1.2 | PWConv | PWConvUnit | 32 | 32 | 8 | 8 | — | shuffle_mode=1, branch=0 |
| 19 | stages.2.0.branch2.0 | PWConv | PWConvUnit | 32 | 32 | 16 | 16 | — | |
| 20 | stages.2.0.branch2.3 | DWConv | DWConvUnit | 32 | 32 | 16 | 16 | 2 | |
| 21 | stages.2.0.branch2.5 | PWConv | PWConvUnit | 32 | 32 | 8 | 8 | — | shuffle_mode=1, branch=1, concat_align=1 |
| 22 | stages.2.1.branch2.0 | PWConv | PWConvUnit | 32 | 32 | 8 | 8 | — | stride-1 block |
| 23 | stages.2.1.branch2.3 | DWConv | DWConvUnit | 32 | 32 | 8 | 8 | 1 | |
| 24 | stages.2.1.branch2.5 | PWConv | PWConvUnit | 32 | 32 | 8 | 8 | — | shuffle_mode=1, branch=1, concat_align=1 |
| 25 | gap_fc | GAP_FC | GAP_FC | 14 | 64 | 8 | 8 | — | GAP + FC 合併，wr_mem=0 |

## D.2 各 PU ROM 內容

### PWConvUnit ROM（15 layers）

Header ROM 內 layer 排列順序：

| ROM 序號 | 全域序號 | Layer 名稱 |
|:--------:|:-------:|-----------|
| 0 | 2 | stages.0.0.branch1.2 |
| 1 | 3 | stages.0.0.branch2.0 |
| 2 | 5 | stages.0.0.branch2.5 |
| 3 | 6 | stages.0.1.branch2.0 |
| 4 | 8 | stages.0.1.branch2.5 |
| 5 | 10 | stages.1.0.branch1.2 |
| 6 | 11 | stages.1.0.branch2.0 |
| 7 | 13 | stages.1.0.branch2.5 |
| 8 | 14 | stages.1.1.branch2.0 |
| 9 | 16 | stages.1.1.branch2.5 |
| 10 | 18 | stages.2.0.branch1.2 |
| 11 | 19 | stages.2.0.branch2.0 |
| 12 | 21 | stages.2.0.branch2.5 |
| 13 | 22 | stages.2.1.branch2.0 |
| 14 | 24 | stages.2.1.branch2.5 |

### DWConvUnit ROM（10 layers）

Header ROM 內 layer 排列順序：

| ROM 序號 | 全域序號 | Layer 名稱 | layer_type |
|:--------:|:-------:|-----------|:----------:|
| 0 | 0 | conv1.0 | NormalConv (2) |
| 1 | 1 | stages.0.0.branch1.0 | DWConv (1) |
| 2 | 4 | stages.0.0.branch2.3 | DWConv (1) |
| 3 | 7 | stages.0.1.branch2.3 | DWConv (1) |
| 4 | 9 | stages.1.0.branch1.0 | DWConv (1) |
| 5 | 12 | stages.1.0.branch2.3 | DWConv (1) |
| 6 | 15 | stages.1.1.branch2.3 | DWConv (1) |
| 7 | 17 | stages.2.0.branch1.0 | DWConv (1) |
| 8 | 20 | stages.2.0.branch2.3 | DWConv (1) |
| 9 | 23 | stages.2.1.branch2.3 | DWConv (1) |

### GAP_FC ROM（1 layer）

| ROM 序號 | 全域序號 | Layer 名稱 |
|:--------:|:-------:|-----------|
| 0 | 25 | gap_fc |
