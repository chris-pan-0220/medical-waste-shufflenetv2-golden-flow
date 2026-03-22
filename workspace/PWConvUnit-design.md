# PWConvUnit Design

執行 **Pointwise (1×1) Convolution** 的硬體加速模組，並包含 INT8 quantization, re-quantization, ReLU。

本文件的計算流程與位寬完全對應 `golden_full_int8_compare_rtlbits_fromcoe_u8.py` 中的 RTL 常數定義。

=> note: 改為新版的 verify_xxx.py

---

## 1. 參數定義

| 符號 | 型別 | 位寬 | 說明 |
|------|------|------|------|
| $q_x$ | uint8 | 8 | 量化輸入 activation |
| $z_x$ | uint8 | 8 | 輸入 zero point |
| $q_w$ | int8 | 8 | 量化權重（對稱量化，$z_w = 0$，省略） |
| $b$ | int16 | 16 | 量化 bias：$b = \text{round}\!\Big(\dfrac{b_{f32}}{s_x \cdot s_w}\Big)$，以 `BIAS_BITS` 位有號數儲存 |
| $m$ | uint16 | 16 | Re-quantization 乘數：$m = \text{round}(M \cdot 2^{N})$，以 `M_BITS` 位無號數儲存 |
| $z_y$ | uint8 | 8 | 輸出 zero point |
| $q_y$ | uint8 | 8 | Re-quantized 輸出 |

其中：

- $M = \dfrac{s_x \cdot s_w}{s_y}$，real-valued re-quantization scale ratio
- $N = 15$，fixed-point 小數位數（`N_FRAC`）
- $s_x, s_w, s_y$ 分別為輸入、權重、輸出的量化 scale（離線計算，不存於硬體中）

=> note: 建議新增欄位說明哪些參數是 runtime 得到，offline 計算得到。
- quant/re-quant的相關參數都是offline, weight & bias 也是 offline

---

## 2. RTL 設計常數

| 常數 | 值 | 說明 |
|------|----|------|
| `ACC_BITS` | 22 | MAC 累加器位寬（signed） |
| `BIAS_BITS` | 16 | Bias 儲存位寬（signed） |
| `M_BITS` | 16 | 乘數 $m$ 儲存位寬（unsigned） |
| `N_FRAC` | 15 | Fixed-point 小數位數 |
| `POST_BITS` | 16 | Post-shift 後位寬（signed），加 $z_y$ 之前 |

=> note: verify 檔案當中, A_BITS 似乎是多餘的，因為 S_PRE 已經沒有作用被預設為 0？

---

## 3. 計算流程（Online Datapath）

### Step 1 — Input Zero-Point Subtraction

$$d_x = q_x - z_x$$

- $q_x \in [0, 255]$，$z_x \in [0, 255]$
- $d_x \in [-255, 255]$，以 **int9** 儲存（`wrap_sint(·, 9)`）

### Step 2 — Weight

$$d_w = q_w$$

- 對稱量化，$z_w = 0$，免減
- $d_w \in [-128, 127]$，以 **int8** 儲存（`wrap_sint(·, 8)`）

### Step 3 — Multiply-Accumulate (MAC)

$$\text{acc} = \sum_{i=0}^{C_{in}-1} d_x[i] \cdot d_w[i]$$

- 每個乘積 $d_x \cdot d_w$ 為 int17
- 累加結果以 **int22** 儲存（`wrap_sint(·, ACC_BITS)`）

### Step 4 — Add Bias

$$a = \text{wrap}\!\big(\text{acc} + b,\;\; 22\big)$$

- $b$ 先 `wrap_sint(·, BIAS_BITS=16)` 確保 int16 範圍
- 相加後以 `add_wrap_sint(·, ACC_BITS=22)` 包回 **int22**

### Step 5 — Fixed-Point Multiply

$$p = a \times m$$

- $a$: int22（signed），$m$: uint16（unsigned，視為非負整數）
- 乘積位寬 = $ACC\_BITS + M\_BITS = 22 + 16 = 38$
- 以 **int38** 儲存（`wrap_sint(·, 38)`）

### Step 6 — Post-Shift（Re-quantization）

$$y = p \ggg N = p \ggg 15$$

- 使用 Shift-and-Round Module（$\ggg$，見第 5 節）右移 $N = 15$ 位
- 結果以 **int16** 儲存（`rshift_round_away_sint(p, 15, out_bits=POST_BITS=16)`）

### Step 7 — Add Zero-Point & Saturation

$$q_y = \text{clip}(y + z_y,\;\; 0,\;\; 255)$$

- $y$: int16，$z_y$: uint8
- 飽和到 **uint8** 範圍（`sat_uint(·, 8)`）

### Step 8 — ReLU（Always On）

$$q_y = \max(q_y,\;\; z_y)$$

- 量化 ReLU：將低於 zero point 的值 clamp 到 $z_y$
- 若 $z_y = 0$，ReLU 等價於無操作（因 $q_y$ 已經是 uint8 $\geq 0$）
- 經由 golden code 驗證，**本模型中所有 PWConv 層皆固定啟用 ReLU**，硬體一律執行此步驟

---

## 4. 位寬流追蹤（Summary）

| 階段 | 運算 | 位寬 | 型別 |
|------|------|------|------|
| $d_x$ | $q_x - z_x$ | 9 | signed |
| $d_w$ | $q_w$ | 8 | signed |
| product | $d_x \cdot d_w$ | 17 | signed |
| acc | $\Sigma$ product | 22 | signed（wrap） |
| $a$ | acc $+$ bias | 22 | signed（wrap） |
| $p$ | $a \times m$ | 38 | signed（wrap） |
| $y$ | $p \ggg 15$ | 16 | signed（wrap） |
| $q_y$ | clip($y + z_y$) | 8 | unsigned（sat） |
| ReLU | $\max(q_y, z_y)$ | 8 | unsigned |

---

## 5. Shift-and-Round Module（$\ggg$）

帶符號四捨五入右移，採用 **round-to-nearest, ties-away-from-zero** 策略：

$$a \ggg k = \text{sgn}(a) \cdot \left\lfloor \frac{|a| + 2^{k-1}}{2^k} \right\rfloor$$

對應 golden code 中的 `rshift_round_away_sint(x, sh, out_bits)`。

硬體實作步驟：

1. 取絕對值 $|a|$
2. 加 rounding bias $2^{k-1}$
3. 算術右移 $k$ 位（即 $\gg k$）
4. 還原符號
5. （若指定 `out_bits`）以 `wrap_sint` 截斷至目標位寬

=> note: 檢查到此，底下等待處理

---

## 6. Dataflow（Input Streaming 與 Output-Channel Tiling）

本節描述 **Pointwise Conv（1×1）** 的資料流假設。對每個空間位置 $(h, w)$，輸出為：

$$\text{acc}[c_{out}, h, w] = \sum_{c_{in}=0}^{C_{in}-1} d_x[c_{in}, h, w] \cdot d_w[c_{out}, c_{in}]$$

其中 $d_x = q_x - z_x$（int9）、$d_w = q_w$（int8），累加器以 `ACC_BITS=22` 位有號數 wrap 儲存。

### 6.1 迴圈/串流順序

令並行度 $P$（同時計算的 output channels 數）。將 $C_{out}$ 以 tile 切分：

$$t = 0, 1, \dots, \frac{C_{out}}{P}-1 \qquad c_{out} = t\cdot P + p,\; p\in[0, P-1]$$

你目前描述的 input streaming / 計算順序可形式化為：

1. `H`（逐列處理）：固定 $h$
2. `C_out/P`（逐個 output-channel tile）：固定 $t$
3. `C_in`（逐個輸入通道累加）：固定 $c_{in}$
4. `W`（一列內由左到右串流）：$w = 0 \dots W-1$

也就是：

```text
for h in 0..H-1:
	for t in 0..(C_out/P)-1:
		psum[p][w] = 0  for p=0..P-1, w=0..W-1
		for cin in 0..C_in-1:
			stream x = d_x[cin, h, w] for w=0..W-1
			in parallel for p in 0..P-1:
				psum[p][w] += x * d_w[t*P+p, cin]
		# 完成 tile t 的 C_in 累加後，得到 P 個 output channels 的 acc
		# 後續接 bias + requant + (optional) ReLU，寫回 ofmap
```

### 6.2 舉例說明（沿用你原本例子）

若 $C_{in}=4$、$C_{out}=8$、$P=4$，則 $t \in \{0, 1\}$：

- 當 $h=0, t=0$：依序串流 $c_{in}=0..3$ 的整列資料（$w=0..W-1$），完成後得到 $c_{out}=0..3$ 的 psum
- 當 $h=0, t=1$：**重新讀取同一列同一組輸入通道資料**（$c_{in}=0..3$，$w=0..W-1$），但改用 $c_{out}=4..7$ 的 weights，完成後得到第二組 psum
- $t=0,1$ 都完成後，才進入 $h=1$

> 以上 dataflow 的本質是「**以 output-channel tile 為單位重用 weights**」，代價是同一列 ifmap 需要為不同 $t$ 重複讀取/重播；實作上通常需要 ifmap line buffer 或能夠重送該列資料。

---

## 7. 硬體實作描述（以 P=4 為例）

### 7.1 平行 MAC 與權重配置

- 並行度：$P=4$
- 每個 clock 取 1 筆輸入 $d_x$，broadcast 到 $P$ 個乘法器
- 第 $p$ 個乘法器固定對應 tile 內的第 $p$ 個 output channel（$c_{out}=t\cdot P + p$），並持有對應的 weight $d_w[c_{out}, c_{in}]$

### 7.2 PSUM 儲存體（Line PSUM Array）

- 每個 output channel（在同一 tile 內）各自有一條 psum line buffer：共 $P$ 條
- psum 的位寬對應 `ACC_BITS=22`（signed, two's-complement, wrap）
- 若設計上支援最大列寬 $W_{max}=32$，則 psum buffer 的容量約為：

$$P \times W_{max} \times ACC\_BITS = 4 \times 32 \times 22\text{ bits}$$

你原本提到以 LUT 合成 psum array，這裡可視為該 buffer 的實作選擇。

### 7.3 後處理管線（Bias / Requant / ReLU）

- 當某個 tile 的 $C_{in}$ 累加完成後，會得到 $P$ 路 psum（對應 $P$ 個 output channels）
- 每一路 psum 依序執行本文件第 3 節 Step 4～Step 8：
	- `+ bias`（wrap 回 int22）
	- `× m`（wrap 至 int38）
	- `>>> 15`（shift-and-round，wrap 至 int16）
	- `+ z_y` 並 `sat` 成 uint8
	- ReLU：$\max(q_y, z_y)$（固定啟用，非 optional）

### 7.4 Output Buffering（降低全域記憶體非連續寫入）

- 因為同一個 clock 同時產生 $P$ 個不同 $c_{out}$ 的輸出，直接寫入 global BRAM 可能造成位址不連續、寫入效率不佳
- 因此先將各路輸出暫存於 per-channel 的 output line buffer
- 累積到可合併寫入的粒度（例如同一個 $c_{out}$ 的連續 4 個 $w$ 位置）後，再以 burst/對齊的方式寫回

---

## 7. Address Generation Unit (AGU) 與 Address 計算規格

本節定義 PWConvUnit 的 counter 範圍、進位條件，以及 input/weight/bias/output 的線性位址公式。
對應 Verilog 中的 AGU counter 與各 memory port address。

### 7.1 Counter 定義與範圍

- 常數：並行度 `P = 4`
- `w`：`0 .. W-1`
- `c_in`：`0 .. C_in-1`
- `c_out_group`（輸出通道 tile index）：`0 .. (C_out/P)-1`
- `p`（group 內 lane index）：`0 .. P-1`  
  > `p` 通常是並行 lane，不一定用 counter 逐一跑；也可同 clock 同時產生 `p=0..P-1` 的位址。

定義：
- `c_out = c_out_group * P + p`

### 7.2 Dataflow 與 Counter 進位（H → C_out/P → C_in → W）

本模組採用以下巢狀順序（`w` 為最快）：

```text
for h in 0..H-1:
  for c_out_group in 0..(C_out/P)-1:
    for c_in in 0..C_in-1:
      for w in 0..W-1:
        compute P outputs in parallel (p=0..P-1)
```

對應的 carry / increment 條件（以 valid input 為前提）：

- 每來一個 input element：`w++`
- 若 `w == W-1`：`w ← 0`，且 `c_in++`
- 若 `w == W-1 && c_in == C_in-1`：`w ← 0`，`c_in ← 0`，且 `c_out_group++`
- 若 `w == W-1 && c_in == C_in-1 && c_out_group == (C_out/P)-1`：前三者 reset，且 `h++`
- stop condition：
  - `h == H-1 && c_out_group == (C_out/P)-1 && c_in == C_in-1 && w == W-1`

> 注意：`c_in` 的上界一定是 `C_in-1`，不可誤用 `C_out-1`，否則會造成提早進位/提前結束而導致位址錯誤。

### 7.3 Input Address（CHW layout）

假設 input feature map 為 **CHW** 線性排列，且每元素 1 byte（uint8）：

- `in_addr = c_in * (H*W) + h * W + w`

### 7.4 Weight Address（Local ROM layout）

本公式假設 weight ROM 以 **group-by-P** 排列（最適合 P=4 並行）：

- Layout：`W[c_out_group][c_in][p]`，每元素 1 byte（int8）
- `w_addr = weight_base + c_out_group * (C_in * P) + c_in * P + p`

> 若你的 ROM 不是此 layout，而是傳統 `W[c_out][c_in]`（以 c_out 為主序），則位址需改為：
> `w_addr = weight_base + c_out * C_in + c_in`，其中 `c_out = c_out_group*P + p`。

### 7.5 Bias Address（Local ROM layout）

bias 以 output channel 為主序，每元素 2 bytes（int16）或依實作（此處僅定義 index）：

- `b_addr = bias_base + c_out_group * P + p`

### 7.6 Output Address（CHW layout + optional shuffle）

假設 output feature map 為 CHW 線性排列：

- `out_addr = out_c' * (H*W) + h * W + w`

其中 `out_c'` 由 shuffle_mode / branch_index 決定：

- `shuffle_mode = 0`（no shuffle）  
  - `out_c' = c_out`

- `shuffle_mode = 1`（shuffle, two-branch merge）
  - `branch_index = 0`（branch1）  
    - `out_c' = 2 * c_out`
  - `branch_index = 1`（branch2，通常對應後半 channels）  
    - `out_c' = 2 * (c_out - C_out/2) + 1`

約束（建議在上層保證）：
- `C_out % P == 0`
- shuffle 時 `C_out` 需為偶數（存在 `C_out/2`）
- branch2 使用時通常要求 `c_out ∈ [C_out/2, C_out-1]`，否則 `out_c'` 會落到不預期範圍

---

## 每一個 layer 的 ofmap 大小
- 目的：估算 global BRAM 與 intermediate buffer（CHW, element 個數）
- 依據：`nano_shufflenet_v2_05_10k.py`（`stage_out_channels=[16,24,32,64]`，`stage_repeats=[2,2,2]`）

命名對照（方便你把手寫 stage 編號映射回模型）：
- `stage2.1` ≈ `stages.0.0`（stride=2 block）
- `stage2.2` ≈ `stages.0.1`（stride=1 block）
- `stage3.1` ≈ `stages.1.0`（stride=2 block）
- `stage3.2` ≈ `stages.1.1`（stride=1 block）
- `stage4.1` ≈ `stages.2.0`（stride=2 block）
- `stage4.2` ≈ `stages.2.1`（stride=1 block）

> 以下數值已依模型結構核對：stride=2 block 會把空間大小減半；每個 stage 的輸出 channel 依序為 24/32/64。

### CHW 尺寸與 element 數（C×H×W）

| layer / node | CHW | elements |
|---|---:|---:|
| input image | 3×128×128 | 49,152 |
| conv1 output | 16×64×64 | 65,536 |
| stage2.1.branch1.DW (stride2) | 16×32×32 | 16,384 |
| stage2.1.branch1.PW | 12×32×32 | 12,288 |
| stage2.1.branch2.PW1 (stride1) | 12×64×64 | 49,152 |
| stage2.1.branch2.DW (stride2) | 12×32×32 | 12,288 |
| stage2.1.branch2.PW2 | 12×32×32 | 12,288 |
| stage2.1 (cat+shuffle) | 24×32×32 | 24,576 |
| stage2.2.branch1 (=x1) | 12×32×32 | 12,288 |
| stage2.2.branch2.PW1 | 12×32×32 | 12,288 |
| stage2.2.branch2.DW | 12×32×32 | 12,288 |
| stage2.2.branch2.PW2 | 12×32×32 | 12,288 |
| stage2.2 (cat+shuffle) | 24×32×32 | 24,576 |
| stage3.1.branch1.DW (stride2) | 24×16×16 | 6,144 |
| stage3.1.branch1.PW | 16×16×16 | 4,096 |
| stage3.1.branch2.PW1 (stride1) | 16×32×32 | 16,384 |
| stage3.1.branch2.DW (stride2) | 16×16×16 | 4,096 |
| stage3.1.branch2.PW2 | 16×16×16 | 4,096 |
| stage3.1 (cat+shuffle) | 32×16×16 | 8,192 |
| stage3.2.branch1 (=x1) | 16×16×16 | 4,096 |
| stage3.2.branch2.PW1 | 16×16×16 | 4,096 |
| stage3.2.branch2.DW | 16×16×16 | 4,096 |
| stage3.2.branch2.PW2 | 16×16×16 | 4,096 |
| stage3.2 (cat+shuffle) | 32×16×16 | 8,192 |
| stage4.1.branch1.DW (stride2) | 32×8×8 | 2,048 |
| stage4.1.branch1.PW | 32×8×8 | 2,048 |
| stage4.1.branch2.PW1 (stride1) | 32×16×16 | 8,192 |
| stage4.1.branch2.DW (stride2) | 32×8×8 | 2,048 |
| stage4.1.branch2.PW2 | 32×8×8 | 2,048 |
| stage4.1 (cat+shuffle) | 64×8×8 | 4,096 |
| stage4.2.branch1 (=x1) | 32×8×8 | 2,048 |
| stage4.2.branch2.PW1 | 32×8×8 | 2,048 |
| stage4.2.branch2.DW | 32×8×8 | 2,048 |
| stage4.2.branch2.PW2 | 32×8×8 | 2,048 |
| stage4.2 (cat+shuffle) | 64×8×8 | 4,096 |
| GAP output | 64×1×1 | 64 |
| FC output | 14 | 14 |

### 最大 ofmap layer

以 element 數（也等同 uint8 byte 數）來看，最大的是：

- **conv1 output：16×64×64 = 65,536 elements**

其次是：
- input image：3×128×128 = 49,152
- stage2.1.branch2.PW1：12×64×64 = 49,152


