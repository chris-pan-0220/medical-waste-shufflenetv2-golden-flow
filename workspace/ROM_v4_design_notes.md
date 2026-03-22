# ROM v4 設計特點與備註

本檔案為 [ROM_v4.md](ROM_v4.md) 的補充說明，記錄設計決策與架構特點。

------------------------------------------------------------------------

## 設計特點

-   **三個 PU × 三種 ROM = 9 個 ROM 實體**，允許 Header / Weight / Bias 並行讀取
-   **Per-PU 客製化 Header**，各 PU 僅儲存必要欄位
-   **無 Address Lookup Table**（各 Header ROM 以 `next_layer_base_addr` 逐層跳轉）
-   **無 global header / version / magic number**
-   Header ROM 為 byte address；Weight / Bias ROM 為 word address
-   Weight / Bias ROM 各自獨立，可分開控制 memory bandwidth

------------------------------------------------------------------------

## 設計保證

-   **Deterministic**：各 PU Header ROM 按執行順序逐層串接，無需查表
-   **Deterministic weight / bias address 計算**：由 `weight_base` / `bias_base` 直接提供，AGU 依 dataflow 計算 word address
-   **無 param_len / magic / version 欄位**
-   **Fully ROM-inferable**
-   **並行讀取**：Header / Weight / Bias 分離，可在同一 cycle 存取不同 ROM
-   **獨立 bandwidth 控制**：Weight / Bias / Header ROM 各自獨立，可分開控制 memory bandwidth
-   **零 padding 浪費**：DWConvUnit Weight ROM 利用 72-bit FPGA 原生位寬
-   **Per-PU 客製化 Header**：消除不適用欄位的空間浪費

------------------------------------------------------------------------

## 設計決策與備註

1. **Split-ROM 動機**：將 Header / Weight / Bias 分離為獨立記憶體，允許硬體在同一 cycle 內並行讀取控制參數與資料，簡化 port arbitration。
2. **FPGA BRAM 原生位寬**：Xilinx 7-series / UltraScale 的 RAMB36E1 原生支援 9 / 18 / 36 / 72-bit 位寬（含 parity bits）。DWConvUnit 的 3×3 kernel 恰好 9 bytes = 72 bits，可完美對齊一個 BRAM word，零 padding 浪費。
3. **Header ROM byte-addressable**：Header 每層只讀取一次，不是 throughput bottleneck，故維持 byte-addressable 以簡化異質欄位存取。
4. **Weight / Bias ROM word-addressable**：AGU 以 `weight_base` / `bias_base` 為基準，依 dataflow 計算 word address，不需 byte-level 移位或跨邊界拼接。
5. **`layer_type` 保留於所有 PU**：雖然 PWConvUnit / GAP_FC 的 `layer_type` 值固定，仍保留以維持 sanity check 與除錯便利。
6. **Concat-align 欄位保留在 PWConvUnit Header**：僅部分 PWConv 層使用（`concat_align=1`），其餘層該區為 don't-care；但保留固定大小 header 以避免 variable-length header 的 AGU 複雜度。
7. **GAP_FC 通常只有 1 層**：但架構仍以 linked-list 支援多層。
8. **ReLU / Padding / Kernel size**：全部由 `layer_type` 隱式決定，硬編碼於 RTL，不存入 ROM。

------------------------------------------------------------------------

## 需要同步至其他 spec 的內容

| 變更 | 目標 spec | 說明 |
|------|-----------|------|
| `stride` 讀取邏輯 | DWConvUnit-design.md（待建立） | DWConvUnit 從 Header ROM 讀取 stride 欄位（1 或 2），決定卷積步幅。padding 固定為 1，硬編碼。 |
| DWConvUnit 服務 NormalConv | DWConvUnit-design.md | DWConvUnit 同時處理 DWConv（groups=C_out）和 NormalConv（groups=1），由 `layer_type` 區分。 |
| ReLU 行為由 `layer_type` 決定 | PWConvUnit-design.md / DWConvUnit-design.md | PWConv → ReLU on；DWConv → ReLU off；NormalConv → ReLU on；GAP_FC → ReLU off。硬編碼，無需 header 欄位。 |
| `rd_mem` / `wr_mem` 硬體控制 | Top-level / Scheduler design | 執行控制器根據 `rd_mem` / `wr_mem` 選擇 BRAM port。GAP_FC 的 `wr_mem=0` 表示輸出直接控制 LED 或其他外部介面。 |
| GAP 空間尺寸 | GAP_FC_Unit-design.md（待建立） | GAP 除以 H×W 做平均。本模型中 H=W=8 → 除以 64 → 可用 >>6 實作。 |
| Kernel / Groups 由 `layer_type` 決定 | DWConvUnit-design.md | DWConv → 3×3, groups=C_out；NormalConv → 3×3, groups=1；PWConv → 1×1, groups=1；GAP_FC → N/A。硬體隱式推導，不存 ROM。 |
| 72-bit Weight ROM BRAM 配置 | DWConvUnit-design.md | DWConvUnit Weight ROM 使用 72-bit 寬度的 Block RAM，需在 RTL 中明確例化 RAMB36E1 / RAMB18E1 並配置 `READ_WIDTH = 72`。 |
| PWConvUnit Weight ROM 位寬 | PWConvUnit-design.md | PWConvUnit Weight ROM 位寬 = P×8 bits，P 為 PWConvUnit 的輸出通道並行度設計參數。 |
| PWConvUnit Bias ROM 位寬 | PWConvUnit-design.md | PWConvUnit Bias ROM 位寬 = P×16 bits，一次讀取 P 個 int16 bias，配合 P 路並行 output channel。 |

------------------------------------------------------------------------

## 更改紀錄（v3 → v4）

| # | 變更 | 說明 |
|---|------|------|
| 1 | **Split-ROM Architecture**：每個 PU 拆為 Header ROM + Weight ROM + Bias ROM | v3 中每個 PU（PWConvUnit / DWConvUnit / GAP_FC）各有 1 份 linked-list ROM（header + fixed params + weight + bias 串在一起）。v4 將每個 PU 的 ROM 拆為三個獨立記憶體區塊，共 **9 個 ROM 實體**，允許硬體在同一 cycle 並行讀取。 |
| 2 | **Per-PU 客製化 Header** | 各 PU 的 Header ROM 僅包含該 PU 所需欄位，消除不適用欄位的 `0xFF` 填充浪費。PWConvUnit Header 26 bytes、DWConvUnit Header 20 bytes、GAP_FC Header 19 bytes。 |
| 3 | **`w_offset` / `b_offset` → `weight_base` / `bias_base`** | 原本為相對 `layer_base` 的偏移，改為各自 Weight / Bias ROM 內的**絕對位址**（word address），AGU 可直接以此為基準計算目標位址，省去 `layer_base + offset` 加法器延遲。 |
| 4 | **固定參數併入 Header** | v3 中獨立的 Fixed Params 區段（z_x, z_y, m_requant, z1_align, z2_align, m_align，共 8 bytes）全部併入 Header ROM，各 PU 僅保留所需欄位。不再有獨立的 `param_base`。 |
| 5 | **Weight ROM FPGA BRAM 原生位寬最佳化** | DWConvUnit Weight ROM 採用 **72-bit** 位寬（9 bytes = 一個完整 3×3 kernel），利用 FPGA Block RAM 原生 9-bit parity 支援，達到 100% 空間利用率與最簡化 AGU。PWConvUnit Weight ROM 位寬為 P×8 bits（見 PWConvUnit-design.md）。GAP_FC Weight ROM 位寬待定。 |
| 6 | **Weight / Bias ROM word-addressable** | Weight ROM 與 Bias ROM 改為 word-addressable（word 大小依 PU 而異），AGU 依 dataflow 計算 word address 即可取得所需資料，不需 byte-level 移位或跨邊界拼接邏輯。Header ROM 維持 byte-addressable。 |
