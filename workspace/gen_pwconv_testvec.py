# /// script
# requires-python = ">=3.10"
# ///
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_pwconv_testvec.py — 為 Verilog PWConv 單通道量化管線產生測試向量。

對應 verify_romv4.py Section 2/3/4 的 RTL 定寬運算邏輯，
逐步追蹤 Verilog 管線每一級的預期值。

Layer:  stages.0.0.branch1.2  (第一個 PWConv)
Config: output_ch=0, input_ch=0  (weight=11, bias=406)

Verilog pipeline:
  1. in_shifted  = q_x - z_x                     (int9)
  2. product     = in_shifted * weight            (int17)
     psum_raw    = Σ product                      (int22, 單通道 = product)
  3. psum_biased = psum_raw + bias                (int22)
  4. psum_scaled = psum_biased * m_requant         (int38)
  5. psum_shifted= rshift_round_away(psum_scaled)  (int16, RSH_N=15)
  6. q_y         = clip(psum_shifted + z_y, 0,255) (uint8)
  7. q_y_relu    = max(q_y, z_y)                   (uint8)

Usage:
  uv run gen_pwconv_testvec.py                           # 預設掃描所有 q_x=0..255
  uv run gen_pwconv_testvec.py --range "0,1,50,140,255"  # 指定值
  uv run gen_pwconv_testvec.py --hex                     # 十六進制輸出
  uv run gen_pwconv_testvec.py --csv out.csv             # 輸出 CSV
  uv run gen_pwconv_testvec.py --mem-full testvec.mem    # 含所有中間訊號的 .mem 檔
"""

import argparse
import csv
import sys

# =====================================================================
# 來自 verify_romv4.py Section 2 — RTL Fixed-Width Constants
# =====================================================================
N_FRAC    = 15       # requant 右移量
S_PRE     = 0        # pre-shift (此層為 0)
ACC_BITS  = 22
A_BITS    = 22
M_BITS    = 16
BIAS_BITS = 16
POST_BITS = 16

# =====================================================================
# 來自 ROM 資料 (stages.0.0.branch1.2, ch0)
# =====================================================================
WEIGHT    = 11       # int8  — localparam signed [7:0] weight = 11
BIAS      = 406      # int16 — localparam signed [15:0] bias = 406
Z_X       = 140      # uint8 — 輸入 zero point
Z_Y       = 0        # uint8 — 輸出 zero point
M_REQUANT = 436      # uint16 — re-quantization 乘數
RSH_N     = N_FRAC - S_PRE   # = 15, 對應 Verilog localparam RSH_N = 15


# =====================================================================
# 來自 verify_romv4.py Section 3 — RTL Fixed-Width Helpers
# =====================================================================
def _mask(bits: int) -> int:
    return (1 << bits) - 1


def wrap_sint(x: int, bits: int) -> int:
    """模擬有號定寬截斷 (two's complement wrap)。"""
    m = _mask(bits)
    u = x & m
    sign = 1 << (bits - 1)
    return (u ^ sign) - sign


def sat_uint(x: int, bits: int) -> int:
    """無號飽和截斷 (clamp to [0, 2^bits - 1])。"""
    if x < 0:
        return 0
    mx = (1 << bits) - 1
    return min(x, mx)


def rshift_round_away_sint(x: int, sh: int, out_bits: int | None = None) -> int:
    """
    有號 round-to-nearest, ties-away-from-zero 右移。
    y = sgn(x) * ((|x| + 2^(sh-1)) >> sh)
    對應 Verilog rshift_round_away module。
    """
    assert sh >= 0
    if sh == 0:
        y = x
    else:
        ax = abs(x)
        y_mag = (ax + (1 << (sh - 1))) >> sh
        y = -y_mag if x < 0 else y_mag
    if out_bits is not None:
        y = wrap_sint(y, out_bits)
    return y


# =====================================================================
# 逐級計算管線
# =====================================================================
def compute_pipeline(q_x: int) -> dict:
    """
    給定 q_x (uint8)，回傳每一級管線的預期值。
    與 Verilog 完全對應。
    """
    assert 0 <= q_x <= 255

    # ---- Step 1: Zero-point Compensation ----
    # in_shifted(int9) = q_x(uint8) - z_x(uint8)
    in_shifted = wrap_sint(q_x - Z_X, 9)

    # ---- Step 2: MAC (single channel: psum_raw = product) ----
    # product(int17) = in_shifted(int9) * weight(int8)
    product = wrap_sint(in_shifted * WEIGHT, 17)

    # psum_raw(int22) = Σ product  (只有一個通道，所以 = product)
    psum_raw = wrap_sint(product, ACC_BITS)

    # ---- Step 3: Bias Addition ----
    # psum_biased(int22) = psum_raw(int22) + bias(int16)
    bias_wrapped = wrap_sint(BIAS, BIAS_BITS)
    psum_biased = wrap_sint(psum_raw + bias_wrapped, ACC_BITS)

    # ---- Step 4: Requantization Scaling ----
    # psum_scaled(int38) = psum_biased(int22) * m_requant(uint16)
    m_u = M_REQUANT & _mask(M_BITS)
    psum_scaled = wrap_sint(psum_biased * m_u, A_BITS + M_BITS)  # 38-bit

    # ---- Step 5: Shift & Round ----
    # psum_shifted(int16) = rshift_round_away(psum_scaled, RSH_N)
    psum_shifted = rshift_round_away_sint(psum_scaled, RSH_N, out_bits=POST_BITS)

    # ---- Step 6: Zero-point Addition & Clipping ----
    # q_y(uint8) = clip(psum_shifted + z_y, 0, 255)
    psum_shifted_add_z_y = psum_shifted + Z_Y
    q_y = sat_uint(psum_shifted_add_z_y, 8)

    # ---- Step 7: ReLU ----
    # q_y_relu(uint8) = max(q_y, z_y)
    q_y_relu = max(q_y, Z_Y)

    return {
        "q_x":                  q_x,
        "in_shifted":           in_shifted,
        "product":              product,
        "psum_raw":             psum_raw,
        "psum_biased":          psum_biased,
        "psum_scaled":          psum_scaled,
        "psum_shifted":         psum_shifted,
        "psum_shifted_add_z_y": psum_shifted_add_z_y,
        "q_y":                  q_y,
        "q_y_relu":             q_y_relu,
    }


# =====================================================================
# 格式化輸出
# =====================================================================
FIELDS = [
    "q_x", "in_shifted", "product", "psum_raw", "psum_biased",
    "psum_scaled", "psum_shifted", "psum_shifted_add_z_y", "q_y", "q_y_relu",
]

# Verilog 各訊號位寬 (用於十六進制格式化)
BIT_WIDTHS = {
    "q_x":                  8,
    "in_shifted":           9,
    "product":              17,
    "psum_raw":             22,
    "psum_biased":          22,
    "psum_scaled":          38,
    "psum_shifted":         16,
    "psum_shifted_add_z_y": 16,
    "q_y":                  8,
    "q_y_relu":             8,
}


def to_hex(val: int, bits: int) -> str:
    """將有號整數轉為 Verilog-style 十六進制 (two's complement)。"""
    m = _mask(bits)
    u = val & m
    hex_digits = (bits + 3) // 4
    return f"{u:0{hex_digits}X}"


def print_table(results: list[dict], use_hex: bool = False):
    """以表格格式輸出到 stdout。"""
    sep = "=" * 120
    print(sep)
    print(f"Parameters: weight={WEIGHT}, bias={BIAS}, z_x={Z_X}, z_y={Z_Y}, "
          f"m_requant={M_REQUANT}, RSH_N={RSH_N}")
    print(sep)
    hdr = " | ".join(f"{f:>22s}" for f in FIELDS)
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if use_hex:
            cols = [to_hex(r[f], BIT_WIDTHS[f]).rjust(22) for f in FIELDS]
        else:
            cols = [str(r[f]).rjust(22) for f in FIELDS]
        print(" | ".join(cols))
    print(sep)


def write_csv(results: list[dict], path: str, use_hex: bool = False):
    """寫入 CSV 檔案。"""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        if use_hex:
            for r in results:
                row = {k: to_hex(r[k], BIT_WIDTHS[k]) for k in FIELDS}
                w.writerow(row)
        else:
            for r in results:
                w.writerow({k: r[k] for k in FIELDS})
    print(f"[INFO] Wrote {len(results)} test vectors to {path}")


def write_verilog_mem(results: list[dict], path: str):
    """
    產生 Verilog $readmemh 可讀取的 .mem 檔。
    每行: q_x(hex) q_y_relu(hex)
    """
    with open(path, "w") as f:
        f.write("// q_x  q_y_relu  (generated by gen_pwconv_testvec.py)\n")
        f.write(f"// weight={WEIGHT}, bias={BIAS}, z_x={Z_X}, z_y={Z_Y}, "
                f"m_requant={M_REQUANT}, RSH_N={RSH_N}\n")
        for r in results:
            qx_h  = to_hex(r["q_x"], 8)
            qy_h  = to_hex(r["q_y_relu"], 8)
            f.write(f"{qx_h} {qy_h}\n")
    print(f"[INFO] Wrote {len(results)} test vectors to {path}")


def write_full_verilog_mem(results: list[dict], path: str):
    """
    產生含所有中間訊號的 .mem 檔。
    每行格式: q_x in_shifted product psum_raw psum_biased psum_scaled psum_shifted q_y q_y_relu
    """
    with open(path, "w") as f:
        f.write("// " + " ".join(FIELDS) + "\n")
        f.write(f"// weight={WEIGHT}, bias={BIAS}, z_x={Z_X}, z_y={Z_Y}, "
                f"m_requant={M_REQUANT}, RSH_N={RSH_N}\n")
        for r in results:
            hexvals = [to_hex(r[fld], BIT_WIDTHS[fld]) for fld in FIELDS]
            f.write(" ".join(hexvals) + "\n")
    print(f"[INFO] Wrote {len(results)} full test vectors to {path}")


# =====================================================================
# Main
# =====================================================================
def main():
    ap = argparse.ArgumentParser(
        description="產生 Verilog PWConv 單通道量化管線的測試向量")
    ap.add_argument("--csv", type=str, default=None,
                    help="輸出 CSV 檔案路徑")
    ap.add_argument("--mem", type=str, default=None,
                    help="輸出 Verilog .mem 檔 (I/O only)")
    ap.add_argument("--mem-full", type=str, default=None,
                    help="輸出 Verilog .mem 檔 (含所有中間訊號)")
    ap.add_argument("--hex", action="store_true",
                    help="以十六進制顯示 (two's complement)")
    ap.add_argument("--range", type=str, default="0-255",
                    help="q_x 掃描範圍, e.g. '0-255' or '100,128,200,255'")
    args = ap.parse_args()

    # 解析掃描範圍
    if "," in args.range:
        q_vals = [int(v.strip()) for v in args.range.split(",")]
    elif "-" in args.range:
        lo, hi = args.range.split("-")
        q_vals = list(range(int(lo), int(hi) + 1))
    else:
        q_vals = [int(args.range)]

    # 計算所有測試向量
    results = [compute_pipeline(qx) for qx in q_vals]

    # 輸出
    print_table(results, use_hex=args.hex)

    if args.csv:
        write_csv(results, args.csv, use_hex=args.hex)

    if args.mem:
        write_verilog_mem(results, args.mem)

    if args.mem_full:
        write_full_verilog_mem(results, args.mem_full)

    # 統計摘要
    outputs = [r["q_y_relu"] for r in results]
    print(f"\n[Summary] q_x range: {q_vals[0]}..{q_vals[-1]} "
          f"({len(q_vals)} vectors)")
    print(f"          q_y_relu range: {min(outputs)}..{max(outputs)}")


if __name__ == "__main__":
    main()
