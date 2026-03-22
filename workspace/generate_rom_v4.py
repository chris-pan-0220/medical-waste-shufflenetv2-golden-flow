#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate ROM v4 (Split-ROM Architecture) from INT8 state dict.

Follows ROM_v4.md spec:
  9 ROM files = 3 PU × 3 (Header / Weight / Bias):
    PWConv_header.coe, PWConv_weight.coe, PWConv_bias.coe
    DWConv_header.coe, DWConv_weight.coe, DWConv_bias.coe
    GAP_FC_header.coe, GAP_FC_weight.coe, GAP_FC_bias.coe
  + rom_v4_debug.json

Architecture: NanoShuffleNetV2_10k (quantized INT8, QNNPACK backend)

Usage:
  python generate_rom_v4.py --workdir .
"""

import argparse
import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
import torch

# =====================================================================
# ROM v4 constants (from ROM_v4.md)
# =====================================================================
P = 4                       # PWConvUnit output-channel parallelism
PWCONV_HEADER_SIZE = 26     # bytes per PWConv header entry
DWCONV_HEADER_SIZE = 20     # bytes per DWConv header entry
GAPFC_HEADER_SIZE  = 19     # bytes per GAP_FC header entry
LAST_LAYER = 0xFFFF

N_FRAC    = 15
M_BITS    = 16
BIAS_BITS = 16

# layer_type values
LT_PWCONV     = 0
LT_DWCONV     = 1
LT_NORMALCONV = 2
LT_GAPFC      = 3


# =====================================================================
# Model / Quant Helpers (shared with v2 generator)
# =====================================================================
def import_module(py_path: Path, mod_name: str = "ref"):
    spec = importlib.util.spec_from_file_location(mod_name, str(py_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def set_engine_qnnpack():
    if "qnnpack" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "qnnpack"


def build_quantized_model(ref_mod, num_classes, int8_sd_path):
    set_engine_qnnpack()
    skel = ref_mod.NanoShuffleNetV2_10k(num_classes=num_classes)
    skel.eval()
    skel.fuse_model()
    skel.qconfig = torch.ao.quantization.get_default_qat_qconfig("qnnpack")
    skel.train()
    torch.ao.quantization.prepare_qat(skel, inplace=True)
    skel.eval()
    qmodel = torch.ao.quantization.convert(skel, inplace=False)
    sd = torch.load(int8_sd_path, map_location="cpu", weights_only=True)
    qmodel.load_state_dict(sd, strict=True)
    qmodel.eval()
    return qmodel, sd


def get_scale_zp(sd, prefix):
    s = float(sd[prefix + ".scale"].item())
    z = int(sd[prefix + ".zero_point"].item())
    return s, z


def get_conv_params(sd, prefix):
    """Return (w_int8_np, b_f32_np, s_w, z_w, s_y, z_y, shape)."""
    w_q = sd[prefix + ".weight"]
    b_f32 = sd[prefix + ".bias"]
    s_y, z_y = get_scale_zp(sd, prefix)
    s_w = float(w_q.q_scale())
    z_w = int(w_q.q_zero_point())
    w_int = w_q.int_repr().numpy()  # int8 ndarray
    return w_int, b_f32.numpy(), s_w, z_w, s_y, z_y, w_int.shape


def get_fc_params(sd):
    packed = sd["fc._packed_params._packed_params"]
    w_q, b_f32 = packed[0], packed[1]
    s_y, z_y = get_scale_zp(sd, "fc")
    s_w = float(w_q.q_scale())
    z_w = int(w_q.q_zero_point())
    w_int = w_q.int_repr().numpy()
    return w_int, b_f32.detach().numpy(), s_w, z_w, s_y, z_y, w_int.shape


def compute_m_requant(s_x, s_w, s_y):
    M = (s_x * s_w) / s_y
    m = int(round(M * (2 ** N_FRAC)))
    m = m & ((1 << M_BITS) - 1)
    return m


def compute_m_align(s_in, s_out):
    alpha = s_in / s_out
    m = int(round(alpha * (2 ** N_FRAC)))
    m = m & ((1 << M_BITS) - 1)
    return m


def compute_bias_int16(b_f32, s_x, s_w):
    b_int = np.round(b_f32.astype(np.float64) / (s_x * s_w)).astype(np.int64)
    # wrap to int16
    b_int = ((b_int + 0x8000) & 0xFFFF) - 0x8000
    return b_int.astype(np.int16)


# =====================================================================
# Build All Layer Lists (PWConv, DWConv, GAP_FC)
# =====================================================================
def build_all_layer_lists(sd, stage_repeats):
    """
    Walk the NanoShuffleNetV2_10k network in execution order and build
    three lists of layer dicts — one for each PU ROM group.

    Returns: (pw_layers, dw_layers, gapfc_layers)
    """
    pw_layers = []
    dw_layers = []
    gapfc_layers = []

    # --- Initial quant params ---
    s_x = float(sd["quant.scale"].item())
    z_x = int(sd["quant.zero_point"].item())

    # Track spatial dimensions (H, W of current activation)
    h, w = 128, 128  # input image size

    # === conv1.0 — NormalConv → DWConvUnit ROM ===
    w_int, b_f32, s_w, z_w, s_conv1, z_conv1, shape = get_conv_params(sd, "conv1.0")
    m_req = compute_m_requant(s_x, s_w, s_conv1)
    b_int16 = compute_bias_int16(b_f32, s_x, s_w)
    C_out, C_in, kH, kW = shape

    # check: camera input is in gb2

    dw_layers.append({
        "name": "conv1.0",
        "layer_type": LT_NORMALCONV,
        "C_out": int(C_out), "C_in": int(C_in),
        "H": h, "W": w, "stride": 2,
        "rd_mem": 1, "wr_mem": 0, # check: gb2 -> gb1
        "z_x": int(z_x), "z_y": int(z_conv1),
        "m_requant": m_req,
        "weight_int8": w_int,       # [C_out, C_in, 3, 3]
        "bias_int16": b_int16,
        "s_x": s_x, "s_w": s_w, "s_y": s_conv1,
    })

    s_x, z_x = s_conv1, z_conv1
    h, w = h // 2, w // 2       # conv1 stride=2: 128 → 64

    # === Stages ===
    for si, reps in enumerate(stage_repeats):
        for bi in range(reps):
            prefix = f"stages.{si}.{bi}"

            if bi == 0:
                # ──────── stride-2 block ────────
                # branch1.0 (DW, stride=2)
                w_int, b_f32, s_w_dw, z_w_dw, s_b1_dw, z_b1_dw, shape = \
                    get_conv_params(sd, f"{prefix}.branch1.0")
                m_req_dw = compute_m_requant(s_x, s_w_dw, s_b1_dw)
                b_int16_dw = compute_bias_int16(b_f32, s_x, s_w_dw)
                C_out_dw = int(shape[0])

                dw_layers.append({
                    "name": f"{prefix}.branch1.0",
                    "layer_type": LT_DWCONV,
                    "C_out": C_out_dw, "C_in": C_out_dw,
                    "H": h, "W": w, "stride": 2,
                    "rd_mem": 0, "wr_mem": 2, # check: gb1 -> ib
                    "z_x": int(z_x), "z_y": int(z_b1_dw),
                    "m_requant": m_req_dw,
                    "weight_int8": w_int,   # [C_out, 1, 3, 3]
                    "bias_int16": b_int16_dw,
                    "s_x": s_x, "s_w": s_w_dw, "s_y": s_b1_dw,
                })

                h2, w2 = h // 2, w // 2     # after stride-2

                # branch1.2 (PW) — shuffle_mode=1, branch=0
                w_int, b_f32, s_w_pw, _, s_b1_pw, z_b1_pw, shape = \
                    get_conv_params(sd, f"{prefix}.branch1.2")
                m_req = compute_m_requant(s_b1_dw, s_w_pw, s_b1_pw)
                b_int16 = compute_bias_int16(b_f32, s_b1_dw, s_w_pw)
                C_out_pw, C_in_pw = int(shape[0]), int(shape[1])

                pw_layers.append({
                    "name": f"{prefix}.branch1.2",
                    "C_out": C_out_pw, "C_in": C_in_pw,
                    "H": h2, "W": w2,
                    "shuffle_mode": 1, "branch": 0, "concat_align": 0,
                    "rd_mem": 2, "wr_mem": 1, # check: ib -> gb2
                    "z_x": int(z_b1_dw), "z_y": int(z_b1_pw),
                    "m_requant": m_req,
                    "z1_align": 0, "z2_align": 0, "m_align": 0,
                    "weight_int8": w_int.reshape(C_out_pw, C_in_pw),
                    "bias_int16": b_int16,
                    "s_x": s_b1_dw, "s_w": s_w_pw, "s_y": s_b1_pw,
                })

                # branch2.0 (PW) — shuffle_mode=0, branch=1
                w_int, b_f32, s_w_pw, _, s_b2_0, z_b2_0, shape = \
                    get_conv_params(sd, f"{prefix}.branch2.0")
                m_req = compute_m_requant(s_x, s_w_pw, s_b2_0)
                b_int16 = compute_bias_int16(b_f32, s_x, s_w_pw)
                C_out_pw, C_in_pw = int(shape[0]), int(shape[1])

                pw_layers.append({
                    "name": f"{prefix}.branch2.0",
                    "C_out": C_out_pw, "C_in": C_in_pw,
                    "H": h, "W": w,
                    "shuffle_mode": 0, "branch": 1, "concat_align": 0,
                    "rd_mem": 0, "wr_mem": 2, # check: gb1 -> ib
                    "z_x": int(z_x), "z_y": int(z_b2_0),
                    "m_requant": m_req,
                    "z1_align": 0, "z2_align": 0, "m_align": 0,
                    "weight_int8": w_int.reshape(C_out_pw, C_in_pw),
                    "bias_int16": b_int16,
                    "s_x": s_x, "s_w": s_w_pw, "s_y": s_b2_0,
                })

                # branch2.3 (DW, stride=2)
                w_int, b_f32, s_w_dw, _, s_b2_dw, z_b2_dw, shape = \
                    get_conv_params(sd, f"{prefix}.branch2.3")
                m_req_dw = compute_m_requant(s_b2_0, s_w_dw, s_b2_dw)
                b_int16_dw = compute_bias_int16(b_f32, s_b2_0, s_w_dw)
                C_out_dw = int(shape[0])

                dw_layers.append({
                    "name": f"{prefix}.branch2.3",
                    "layer_type": LT_DWCONV,
                    "C_out": C_out_dw, "C_in": C_out_dw,
                    "H": h, "W": w, "stride": 2,
                    "rd_mem": 2, "wr_mem": 0, # check: ib -> gb1
                    "z_x": int(z_b2_0), "z_y": int(z_b2_dw),
                    "m_requant": m_req_dw,
                    "weight_int8": w_int,
                    "bias_int16": b_int16_dw,
                    "s_x": s_b2_0, "s_w": s_w_dw, "s_y": s_b2_dw,
                })

                # branch2.5 (PW) — shuffle_mode=1, branch=1, concat_align=1
                w_int, b_f32, s_w_pw, _, s_b2_5, z_b2_5, shape = \
                    get_conv_params(sd, f"{prefix}.branch2.5")
                m_req = compute_m_requant(s_b2_dw, s_w_pw, s_b2_5)
                b_int16 = compute_bias_int16(b_f32, s_b2_dw, s_w_pw)
                C_out_pw, C_in_pw = int(shape[0]), int(shape[1])
                m_align = compute_m_align(s_b2_5, s_b1_pw)

                pw_layers.append({
                    "name": f"{prefix}.branch2.5",
                    "C_out": C_out_pw, "C_in": C_in_pw,
                    "H": h2, "W": w2,
                    "shuffle_mode": 1, "branch": 1, "concat_align": 1,
                    "rd_mem": 0, "wr_mem": 1, # check: gb1 -> gb2
                    "z_x": int(z_b2_dw), "z_y": int(z_b2_5),
                    "m_requant": m_req,
                    "z1_align": int(z_b1_pw), "z2_align": int(z_b2_5),
                    "m_align": m_align,
                    "weight_int8": w_int.reshape(C_out_pw, C_in_pw),
                    "bias_int16": b_int16,
                    "s_x": s_b2_dw, "s_w": s_w_pw, "s_y": s_b2_5,
                    "s_align_in": s_b2_5, "s_align_out": s_b1_pw,
                })

                # After cat+shuffle → output domain = branch1 domain
                s_x, z_x = s_b1_pw, z_b1_pw
                h, w = h2, w2

            else:
                # passthrough block: gb2 -> gb1

                # ──────── stride-1 block ────────
                s_x1, z_x1 = s_x, z_x

                # branch2.0 (PW) — shuffle_mode=0, branch=1
                w_int, b_f32, s_w_pw, _, s_b2_0, z_b2_0, shape = \
                    get_conv_params(sd, f"{prefix}.branch2.0")
                m_req = compute_m_requant(s_x, s_w_pw, s_b2_0)
                b_int16 = compute_bias_int16(b_f32, s_x, s_w_pw)
                C_out_pw, C_in_pw = int(shape[0]), int(shape[1])

                pw_layers.append({
                    "name": f"{prefix}.branch2.0",
                    "C_out": C_out_pw, "C_in": C_in_pw,
                    "H": h, "W": w,
                    "shuffle_mode": 0, "branch": 1, "concat_align": 0,
                    "rd_mem": 1, "wr_mem": 2, # check: gb2 -> ib
                    "z_x": int(z_x), "z_y": int(z_b2_0),
                    "m_requant": m_req,
                    "z1_align": 0, "z2_align": 0, "m_align": 0,
                    "weight_int8": w_int.reshape(C_out_pw, C_in_pw),
                    "bias_int16": b_int16,
                    "s_x": s_x, "s_w": s_w_pw, "s_y": s_b2_0,
                })

                # branch2.3 (DW, stride=1)
                w_int, b_f32, s_w_dw, _, s_b2_dw, z_b2_dw, shape = \
                    get_conv_params(sd, f"{prefix}.branch2.3")
                m_req_dw = compute_m_requant(s_b2_0, s_w_dw, s_b2_dw)
                b_int16_dw = compute_bias_int16(b_f32, s_b2_0, s_w_dw)
                C_out_dw = int(shape[0])

                dw_layers.append({
                    "name": f"{prefix}.branch2.3",
                    "layer_type": LT_DWCONV,
                    "C_out": C_out_dw, "C_in": C_out_dw,
                    "H": h, "W": w, "stride": 1,
                    "rd_mem": 2, "wr_mem": 1, # check: ib -> gb2
                    "z_x": int(z_b2_0), "z_y": int(z_b2_dw),
                    "m_requant": m_req_dw,
                    "weight_int8": w_int,
                    "bias_int16": b_int16_dw,
                    "s_x": s_b2_0, "s_w": s_w_dw, "s_y": s_b2_dw,
                })

                # branch2.5 (PW) — shuffle_mode=1, branch=1, concat_align=1
                w_int, b_f32, s_w_pw, _, s_b2_5, z_b2_5, shape = \
                    get_conv_params(sd, f"{prefix}.branch2.5")
                m_req = compute_m_requant(s_b2_dw, s_w_pw, s_b2_5)
                b_int16 = compute_bias_int16(b_f32, s_b2_dw, s_w_pw)
                C_out_pw, C_in_pw = int(shape[0]), int(shape[1])
                m_align = compute_m_align(s_b2_5, s_x1)

                pw_layers.append({
                    "name": f"{prefix}.branch2.5",
                    "C_out": C_out_pw, "C_in": C_in_pw,
                    "H": h, "W": w,
                    "shuffle_mode": 1, "branch": 1, "concat_align": 1,
                    "rd_mem": 1, "wr_mem": 0, # check: gb2 -> gb1
                    "z_x": int(z_b2_dw), "z_y": int(z_b2_5),
                    "m_requant": m_req,
                    "z1_align": int(z_x1), "z2_align": int(z_b2_5),
                    "m_align": m_align,
                    "weight_int8": w_int.reshape(C_out_pw, C_in_pw),
                    "bias_int16": b_int16,
                    "s_x": s_b2_dw, "s_w": s_w_pw, "s_y": s_b2_5,
                    "s_align_in": s_b2_5, "s_align_out": s_x1,
                })

                # After cat+shuffle → output domain = branch1 (passthrough) domain
                s_x, z_x = s_x1, z_x1

    # === GAP_FC ===
    w_int_fc, b_f32_fc, s_w_fc, z_w_fc, s_fc, z_fc, shape_fc = get_fc_params(sd)
    m_req_fc = compute_m_requant(s_x, s_w_fc, s_fc)
    b_int16_fc = compute_bias_int16(b_f32_fc, s_x, s_w_fc)
    C_out_fc, C_in_fc = int(shape_fc[0]), int(shape_fc[1])

    gapfc_layers.append({
        "name": "gap_fc",
        "layer_type": LT_GAPFC,
        "C_out": C_out_fc, "C_in": C_in_fc,
        "H": h, "W": w,       # should be 8, 8
        "rd_mem": 0, "wr_mem": 0, # check: gb1 -> LED output
        "z_x": int(z_x), "z_y": int(z_fc),
        "m_requant": m_req_fc,
        "weight_int8": w_int_fc,    # [C_out, C_in]
        "bias_int16": b_int16_fc,
        "s_x": s_x, "s_w": s_w_fc, "s_y": s_fc,
    })

    return pw_layers, dw_layers, gapfc_layers


# =====================================================================
# Pack PWConvUnit ROMs
# =====================================================================
def pack_pw_header(layer, next_addr, weight_base, bias_base):
    """Pack one PWConv header (26 bytes) per ROM_v4.md §2.1."""
    hdr = bytearray(PWCONV_HEADER_SIZE)
    struct.pack_into("<H", hdr,  0, next_addr   & 0xFFFF)  # +0  next_layer_base_addr
    hdr[2] = LT_PWCONV                                     # +2  layer_type
    struct.pack_into("<H", hdr,  3, weight_base & 0xFFFF)  # +3  weight_base
    struct.pack_into("<H", hdr,  5, bias_base   & 0xFFFF)  # +5  bias_base
    struct.pack_into("<H", hdr,  7, layer["C_out"] & 0xFFFF)
    struct.pack_into("<H", hdr,  9, layer["C_in"]  & 0xFFFF)
    hdr[11] = layer["H"] & 0xFF
    hdr[12] = layer["W"] & 0xFF
    hdr[13] = layer["shuffle_mode"] & 0xFF
    hdr[14] = layer["branch"]       & 0xFF
    hdr[15] = layer["concat_align"] & 0xFF
    hdr[16] = layer["rd_mem"]       & 0xFF
    hdr[17] = layer["wr_mem"]       & 0xFF
    hdr[18] = layer["z_x"]         & 0xFF
    hdr[19] = layer["z_y"]         & 0xFF
    struct.pack_into("<H", hdr, 20, layer["m_requant"] & 0xFFFF)
    hdr[22] = layer["z1_align"]    & 0xFF
    hdr[23] = layer["z2_align"]    & 0xFF
    struct.pack_into("<H", hdr, 24, layer["m_align"]   & 0xFFFF)
    return bytes(hdr)


def pack_pw_weight_words(layer):
    """
    Pack weight[C_out, C_in] as P-grouped words.
    Word width = P × 8 bits.
    word_addr = c_out_group * C_in + c_in,
    byte[p] = weight[c_out_group*P + p][c_in].
    """
    C_out, C_in = layer["C_out"], layer["C_in"]
    w = layer["weight_int8"].reshape(C_out, C_in)
    assert C_out % P == 0, f"C_out={C_out} not divisible by P={P}"
    n_groups = C_out // P

    words = []
    for g in range(n_groups):
        for cin in range(C_in):
            word_bytes = bytearray(P)
            for p in range(P):
                word_bytes[p] = int(w[g * P + p, cin]) & 0xFF
            words.append(bytes(word_bytes))
    return words


def pack_pw_bias_words(layer):
    """
    Pack bias[C_out] as P-grouped words.
    Word width = P × 16 bits.
    word_addr = c_out_group,
    int16[p] = bias[c_out_group*P + p].
    """
    C_out = layer["C_out"]
    b = layer["bias_int16"].flatten()
    assert C_out % P == 0
    n_groups = C_out // P

    words = []
    for g in range(n_groups):
        word_bytes = bytearray(P * 2)
        for p in range(P):
            struct.pack_into("<h", word_bytes, p * 2, int(b[g * P + p]))
        words.append(bytes(word_bytes))
    return words


def build_pw_roms(pw_layers):
    """Build PWConv Header / Weight / Bias ROM data."""
    header_bytes = bytearray()
    weight_words = []
    bias_words = []
    weight_word_offset = 0
    bias_word_offset = 0
    debug_layers = []

    for i, layer in enumerate(pw_layers):
        w_base = weight_word_offset
        b_base = bias_word_offset

        w_words = pack_pw_weight_words(layer)
        b_words = pack_pw_bias_words(layer)

        weight_words.extend(w_words)
        bias_words.extend(b_words)
        weight_word_offset += len(w_words)
        bias_word_offset += len(b_words)

        # next_layer_base_addr
        if i < len(pw_layers) - 1:
            next_addr = (i + 1) * PWCONV_HEADER_SIZE
        else:
            next_addr = LAST_LAYER

        hdr = pack_pw_header(layer, next_addr, w_base, b_base)
        header_bytes.extend(hdr)

        debug_layers.append({
            "index": i,
            "name": layer["name"],
            "header_base": i * PWCONV_HEADER_SIZE,
            "next_layer_base_addr": next_addr,
            "weight_base": w_base,
            "bias_base": b_base,
            "C_out": layer["C_out"], "C_in": layer["C_in"],
            "H": layer["H"], "W": layer["W"],
            "shuffle_mode": layer["shuffle_mode"],
            "branch": layer["branch"],
            "concat_align": layer["concat_align"],
            "z_x": layer["z_x"], "z_y": layer["z_y"],
            "m_requant": layer["m_requant"],
            "z1_align": layer["z1_align"], "z2_align": layer["z2_align"],
            "m_align": layer["m_align"],
            "weight_words": len(w_words),
            "bias_words": len(b_words),
            "offline_scales": {
                "s_x": layer.get("s_x"),
                "s_w": layer.get("s_w"),
                "s_y": layer.get("s_y"),
                "s_align_in": layer.get("s_align_in"),
                "s_align_out": layer.get("s_align_out"),
            },
        })

    return bytes(header_bytes), weight_words, bias_words, debug_layers


# =====================================================================
# Pack DWConvUnit ROMs
# =====================================================================
def pack_dw_header(layer, next_addr, weight_base, bias_base):
    """Pack one DWConv header (20 bytes) per ROM_v4.md §3.1."""
    hdr = bytearray(DWCONV_HEADER_SIZE)
    struct.pack_into("<H", hdr,  0, next_addr    & 0xFFFF)
    hdr[2] = layer["layer_type"]  & 0xFF          # 1=DWConv, 2=NormalConv
    struct.pack_into("<H", hdr,  3, weight_base  & 0xFFFF)
    struct.pack_into("<H", hdr,  5, bias_base    & 0xFFFF)
    struct.pack_into("<H", hdr,  7, layer["C_out"] & 0xFFFF)
    struct.pack_into("<H", hdr,  9, layer["C_in"]  & 0xFFFF)
    hdr[11] = layer["H"]      & 0xFF
    hdr[12] = layer["W"]      & 0xFF
    hdr[13] = layer["stride"]  & 0xFF
    hdr[14] = layer["rd_mem"]  & 0xFF
    hdr[15] = layer["wr_mem"]  & 0xFF
    hdr[16] = layer["z_x"]    & 0xFF
    hdr[17] = layer["z_y"]    & 0xFF
    struct.pack_into("<H", hdr, 18, layer["m_requant"] & 0xFFFF)
    return bytes(hdr)


def pack_dw_weight_words(layer):
    """
    Pack DW/NormalConv weights as 72-bit (9-byte) words.
    DWConv:     word per channel  →  C_out words
    NormalConv: word per (c_out, c_in) → C_out × C_in words
    Byte order within word: row-major 3×3 kernel.
    """
    lt = layer["layer_type"]
    w = layer["weight_int8"]  # [C_out, 1, 3, 3] or [C_out, C_in, 3, 3]
    words = []

    if lt == LT_DWCONV:
        C_out = w.shape[0]
        for co in range(C_out):
            kernel = w[co, 0].flatten()  # [9]
            word_bytes = bytearray(9)
            for k in range(9):
                word_bytes[k] = int(kernel[k]) & 0xFF
            words.append(bytes(word_bytes))
    elif lt == LT_NORMALCONV:
        C_out, C_in = w.shape[0], w.shape[1]
        for co in range(C_out):
            for ci in range(C_in):
                kernel = w[co, ci].flatten()  # [9]
                word_bytes = bytearray(9)
                for k in range(9):
                    word_bytes[k] = int(kernel[k]) & 0xFF
                words.append(bytes(word_bytes))
    else:
        raise ValueError(f"Unexpected layer_type={lt} in DW weight packing")

    return words


def pack_dw_bias_words(layer):
    """Pack bias as 16-bit words (1 int16 per word)."""
    b = layer["bias_int16"].flatten()
    words = []
    for co in range(len(b)):
        words.append(struct.pack("<h", int(b[co])))
    return words


def build_dw_roms(dw_layers):
    """Build DWConv Header / Weight / Bias ROM data."""
    header_bytes = bytearray()
    weight_words = []
    bias_words = []
    weight_word_offset = 0
    bias_word_offset = 0
    debug_layers = []

    for i, layer in enumerate(dw_layers):
        w_base = weight_word_offset
        b_base = bias_word_offset

        w_words = pack_dw_weight_words(layer)
        b_words = pack_dw_bias_words(layer)

        weight_words.extend(w_words)
        bias_words.extend(b_words)
        weight_word_offset += len(w_words)
        bias_word_offset += len(b_words)

        if i < len(dw_layers) - 1:
            next_addr = (i + 1) * DWCONV_HEADER_SIZE
        else:
            next_addr = LAST_LAYER

        hdr = pack_dw_header(layer, next_addr, w_base, b_base)
        header_bytes.extend(hdr)

        debug_layers.append({
            "index": i,
            "name": layer["name"],
            "layer_type": layer["layer_type"],
            "header_base": i * DWCONV_HEADER_SIZE,
            "next_layer_base_addr": next_addr,
            "weight_base": w_base,
            "bias_base": b_base,
            "C_out": layer["C_out"], "C_in": layer["C_in"],
            "H": layer["H"], "W": layer["W"],
            "stride": layer["stride"],
            "z_x": layer["z_x"], "z_y": layer["z_y"],
            "m_requant": layer["m_requant"],
            "weight_words": len(w_words),
            "bias_words": len(b_words),
            "offline_scales": {
                "s_x": layer.get("s_x"),
                "s_w": layer.get("s_w"),
                "s_y": layer.get("s_y"),
            },
        })

    return bytes(header_bytes), weight_words, bias_words, debug_layers


# =====================================================================
# Pack GAP_FC ROMs
# =====================================================================
def pack_gapfc_header(layer, next_addr, weight_base, bias_base):
    """Pack one GAP_FC header (19 bytes) per ROM_v4.md §4.1."""
    hdr = bytearray(GAPFC_HEADER_SIZE)
    struct.pack_into("<H", hdr,  0, next_addr    & 0xFFFF)
    hdr[2] = LT_GAPFC                            # layer_type = 3
    struct.pack_into("<H", hdr,  3, weight_base  & 0xFFFF)
    struct.pack_into("<H", hdr,  5, bias_base    & 0xFFFF)
    struct.pack_into("<H", hdr,  7, layer["C_out"] & 0xFFFF)
    struct.pack_into("<H", hdr,  9, layer["C_in"]  & 0xFFFF)
    hdr[11] = layer["H"]      & 0xFF
    hdr[12] = layer["W"]      & 0xFF
    hdr[13] = layer["rd_mem"] & 0xFF
    hdr[14] = layer["wr_mem"] & 0xFF
    hdr[15] = layer["z_x"]   & 0xFF
    hdr[16] = layer["z_y"]   & 0xFF
    struct.pack_into("<H", hdr, 17, layer["m_requant"] & 0xFFFF)
    return bytes(hdr)


def pack_gapfc_weight_words(layer):
    """
    Pack FC weight as byte-level words (word width = 8 bits, TBD).
    logical_index = c_out * C_in + c_in
    """
    w = layer["weight_int8"]   # [C_out, C_in]
    C_out, C_in = w.shape[0], w.shape[1]
    words = []
    for co in range(C_out):
        for ci in range(C_in):
            words.append(bytes([int(w[co, ci]) & 0xFF]))
    return words


def pack_gapfc_bias_words(layer):
    """Pack FC bias as 16-bit words."""
    b = layer["bias_int16"].flatten()
    words = []
    for co in range(len(b)):
        words.append(struct.pack("<h", int(b[co])))
    return words


def build_gapfc_roms(gapfc_layers):
    """Build GAP_FC Header / Weight / Bias ROM data."""
    header_bytes = bytearray()
    weight_words = []
    bias_words = []
    weight_word_offset = 0
    bias_word_offset = 0
    debug_layers = []

    for i, layer in enumerate(gapfc_layers):
        w_base = weight_word_offset
        b_base = bias_word_offset

        w_words = pack_gapfc_weight_words(layer)
        b_words = pack_gapfc_bias_words(layer)

        weight_words.extend(w_words)
        bias_words.extend(b_words)
        weight_word_offset += len(w_words)
        bias_word_offset += len(b_words)

        if i < len(gapfc_layers) - 1:
            next_addr = (i + 1) * GAPFC_HEADER_SIZE
        else:
            next_addr = LAST_LAYER

        hdr = pack_gapfc_header(layer, next_addr, w_base, b_base)
        header_bytes.extend(hdr)

        debug_layers.append({
            "index": i,
            "name": layer["name"],
            "layer_type": layer["layer_type"],
            "header_base": i * GAPFC_HEADER_SIZE,
            "next_layer_base_addr": next_addr,
            "weight_base": w_base,
            "bias_base": b_base,
            "C_out": layer["C_out"], "C_in": layer["C_in"],
            "H": layer["H"], "W": layer["W"],
            "z_x": layer["z_x"], "z_y": layer["z_y"],
            "m_requant": layer["m_requant"],
            "weight_words": len(w_words),
            "bias_words": len(b_words),
            "offline_scales": {
                "s_x": layer.get("s_x"),
                "s_w": layer.get("s_w"),
                "s_y": layer.get("s_y"),
            },
        })

    return bytes(header_bytes), weight_words, bias_words, debug_layers


# =====================================================================
# COE Writers
# =====================================================================
def write_coe_bytes(data_bytes: bytes, path: str):
    """Write byte-level COE (radix=16, one byte per line)."""
    lines = ["memory_initialization_radix=16;", "memory_initialization_vector="]
    n = len(data_bytes)
    for i, b in enumerate(data_bytes):
        suffix = ";" if i == n - 1 else ","
        lines.append(f"{b:02x}{suffix}")
    Path(path).write_text("\n".join(lines) + "\n")


def write_coe_words(word_bytes_list: list[bytes], path: str):
    """
    Write word-level COE (radix=16, one word per line).
    Each entry in word_bytes_list is a bytes object representing one word.
    Hex output: MSB first (leftmost = highest bit index).
    """
    lines = ["memory_initialization_radix=16;", "memory_initialization_vector="]
    n = len(word_bytes_list)
    for i, word in enumerate(word_bytes_list):
        # Reverse byte order: byte[0] at lowest bits → rightmost in hex
        hex_str = "".join(f"{b:02x}" for b in reversed(word))
        suffix = ";" if i == n - 1 else ","
        lines.append(f"{hex_str}{suffix}")
    Path(path).write_text("\n".join(lines) + "\n")


# =====================================================================
# Debug JSON
# =====================================================================
def write_debug_json(pw_debug, dw_debug, gapfc_debug,
                     pw_hdr_bytes, pw_w_words, pw_b_words,
                     dw_hdr_bytes, dw_w_words, dw_b_words,
                     gf_hdr_bytes, gf_w_words, gf_b_words,
                     path: str):
    obj = {
        "spec": "ROM_v4.md",
        "format": "Split-ROM Architecture, 3 PU × 3 ROM",
        "constants": {
            "P": P,
            "PWCONV_HEADER_SIZE": PWCONV_HEADER_SIZE,
            "DWCONV_HEADER_SIZE": DWCONV_HEADER_SIZE,
            "GAPFC_HEADER_SIZE": GAPFC_HEADER_SIZE,
            "LAST_LAYER": f"0x{LAST_LAYER:04X}",
            "N_FRAC": N_FRAC,
            "M_BITS": M_BITS,
            "BIAS_BITS": BIAS_BITS,
        },
        "pwconv": {
            "header_rom_bytes": len(pw_hdr_bytes),
            "weight_rom_words": len(pw_w_words),
            "bias_rom_words": len(pw_b_words),
            "num_layers": len(pw_debug),
            "layers": pw_debug,
        },
        "dwconv": {
            "header_rom_bytes": len(dw_hdr_bytes),
            "weight_rom_words": len(dw_w_words),
            "bias_rom_words": len(dw_b_words),
            "num_layers": len(dw_debug),
            "layers": dw_debug,
        },
        "gapfc": {
            "header_rom_bytes": len(gf_hdr_bytes),
            "weight_rom_words": len(gf_w_words),
            "bias_rom_words": len(gf_b_words),
            "num_layers": len(gapfc_debug),
            "layers": gapfc_debug,
        },
    }
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


# =====================================================================
# Per-COE JSON Debug Writers
# =====================================================================

def _s8(b: int) -> int:
    """Unsigned byte → signed int8."""
    b = int(b) & 0xFF
    return b - 256 if b >= 128 else b


def _u16_le(data, off: int) -> int:
    """Read u16 little-endian from bytes/bytearray."""
    return int(data[off]) | (int(data[off + 1]) << 8)


def _s16_from_word(word_bytes: bytes) -> int:
    """Read int16 LE from a 2-byte word."""
    v = int(word_bytes[0]) | (int(word_bytes[1]) << 8)
    return v - 0x10000 if v >= 0x8000 else v


def _word_hex_msb(word_bytes: bytes) -> str:
    """Word bytes → hex string MSB-first (matches COE output)."""
    return "".join(f"{b:02x}" for b in reversed(word_bytes))


# ---- PWConv ----

def write_json_pw_header(header_bytes: bytes, debug_layers: list, path: str):
    """Decode PWConv header ROM into per-layer JSON."""
    layers = []
    for dl in debug_layers:
        a = dl["header_base"]
        h = header_bytes
        layers.append({
            "layer_index": dl["index"],
            "name":        dl["name"],
            "byte_offset": a,
            "raw_hex":     h[a:a + PWCONV_HEADER_SIZE].hex(),
            "fields": {
                "next_layer_base_addr": {"offset": 0,  "size": 2, "type": "u16 LE", "hex": f"0x{_u16_le(h,a+0):04X}",  "value": _u16_le(h, a + 0)},
                "layer_type":           {"offset": 2,  "size": 1, "type": "u8",     "hex": f"0x{h[a+2]:02X}",          "value": int(h[a + 2])},
                "weight_base":          {"offset": 3,  "size": 2, "type": "u16 LE", "hex": f"0x{_u16_le(h,a+3):04X}",  "value": _u16_le(h, a + 3)},
                "bias_base":            {"offset": 5,  "size": 2, "type": "u16 LE", "hex": f"0x{_u16_le(h,a+5):04X}",  "value": _u16_le(h, a + 5)},
                "C_out":                {"offset": 7,  "size": 2, "type": "u16 LE", "hex": f"0x{_u16_le(h,a+7):04X}",  "value": _u16_le(h, a + 7)},
                "C_in":                 {"offset": 9,  "size": 2, "type": "u16 LE", "hex": f"0x{_u16_le(h,a+9):04X}",  "value": _u16_le(h, a + 9)},
                "H":                    {"offset": 11, "size": 1, "type": "u8",     "hex": f"0x{h[a+11]:02X}",         "value": int(h[a + 11])},
                "W":                    {"offset": 12, "size": 1, "type": "u8",     "hex": f"0x{h[a+12]:02X}",         "value": int(h[a + 12])},
                "shuffle_mode":         {"offset": 13, "size": 1, "type": "u8",     "hex": f"0x{h[a+13]:02X}",         "value": int(h[a + 13])},
                "branch":               {"offset": 14, "size": 1, "type": "u8",     "hex": f"0x{h[a+14]:02X}",         "value": int(h[a + 14])},
                "concat_align":         {"offset": 15, "size": 1, "type": "u8",     "hex": f"0x{h[a+15]:02X}",         "value": int(h[a + 15])},
                "rd_mem":               {"offset": 16, "size": 1, "type": "u8",     "hex": f"0x{h[a+16]:02X}",         "value": int(h[a + 16])},
                "wr_mem":               {"offset": 17, "size": 1, "type": "u8",     "hex": f"0x{h[a+17]:02X}",         "value": int(h[a + 17])},
                "z_x":                  {"offset": 18, "size": 1, "type": "u8",     "hex": f"0x{h[a+18]:02X}",         "value": int(h[a + 18])},
                "z_y":                  {"offset": 19, "size": 1, "type": "u8",     "hex": f"0x{h[a+19]:02X}",         "value": int(h[a + 19])},
                "m_requant":            {"offset": 20, "size": 2, "type": "u16 LE", "hex": f"0x{_u16_le(h,a+20):04X}", "value": _u16_le(h, a + 20)},
                "z1_align":             {"offset": 22, "size": 1, "type": "u8",     "hex": f"0x{h[a+22]:02X}",         "value": int(h[a + 22])},
                "z2_align":             {"offset": 23, "size": 1, "type": "u8",     "hex": f"0x{h[a+23]:02X}",         "value": int(h[a + 23])},
                "m_align":              {"offset": 24, "size": 2, "type": "u16 LE", "hex": f"0x{_u16_le(h,a+24):04X}", "value": _u16_le(h, a + 24)},
            },
        })
    obj = {
        "rom_name": "PWConv_header",
        "spec": "ROM_v4.md §2.1",
        "addressing": "byte",
        "entry_size_bytes": PWCONV_HEADER_SIZE,
        "total_bytes": len(header_bytes),
        "num_layers": len(layers),
        "layers": layers,
    }
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def write_json_pw_weight(weight_words: list[bytes], debug_layers: list, path: str):
    """Decode PWConv weight words into per-layer JSON with int8 values."""
    layers = []
    for dl in debug_layers:
        w_base  = dl["weight_base"]
        n_words = dl["weight_words"]
        C_out   = dl["C_out"]
        C_in    = dl["C_in"]
        words_data = []
        for wi in range(n_words):
            word = weight_words[w_base + wi]
            c_out_group = wi // C_in
            c_in        = wi % C_in
            values = [_s8(word[p]) for p in range(P)]
            words_data.append({
                "word_addr":    w_base + wi,
                "local_index":  wi,
                "c_out_group":  c_out_group,
                "c_in":         c_in,
                "hex":          _word_hex_msb(word),
                "int8_values":  values,
            })
        layers.append({
            "layer_index":  dl["index"],
            "name":         dl["name"],
            "weight_base":  w_base,
            "num_words":    n_words,
            "C_out": C_out, "C_in": C_in,
            "words": words_data,
        })
    obj = {
        "rom_name": "PWConv_weight",
        "spec": "ROM_v4.md §2.2",
        "word_width_bits": P * 8,
        "total_words": len(weight_words),
        "P": P,
        "num_layers": len(layers),
        "layers": layers,
    }
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def write_json_pw_bias(bias_words: list[bytes], debug_layers: list, path: str):
    """Decode PWConv bias words into per-layer JSON with int16 values."""
    layers = []
    for dl in debug_layers:
        b_base  = dl["bias_base"]
        n_words = dl["bias_words"]
        C_out   = dl["C_out"]
        words_data = []
        for wi in range(n_words):
            word = bias_words[b_base + wi]
            c_out_group = wi
            values = []
            for p in range(P):
                v = int(word[p * 2]) | (int(word[p * 2 + 1]) << 8)
                if v >= 0x8000:
                    v -= 0x10000
                values.append(v)
            words_data.append({
                "word_addr":     b_base + wi,
                "local_index":   wi,
                "c_out_group":   c_out_group,
                "hex":           _word_hex_msb(word),
                "int16_values":  values,
            })
        layers.append({
            "layer_index":  dl["index"],
            "name":         dl["name"],
            "bias_base":    b_base,
            "num_words":    n_words,
            "C_out":        C_out,
            "words":        words_data,
        })
    obj = {
        "rom_name": "PWConv_bias",
        "spec": "ROM_v4.md §2.3",
        "word_width_bits": P * 16,
        "total_words": len(bias_words),
        "P": P,
        "num_layers": len(layers),
        "layers": layers,
    }
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


# ---- DWConv ----

def write_json_dw_header(header_bytes: bytes, debug_layers: list, path: str):
    """Decode DWConv header ROM into per-layer JSON."""
    layers = []
    for dl in debug_layers:
        a = dl["header_base"]
        h = header_bytes
        lt_val = int(h[a + 2])
        lt_name = {LT_DWCONV: "DWConv", LT_NORMALCONV: "NormalConv"}.get(lt_val, f"unknown({lt_val})")
        layers.append({
            "layer_index": dl["index"],
            "name":        dl["name"],
            "byte_offset": a,
            "raw_hex":     h[a:a + DWCONV_HEADER_SIZE].hex(),
            "fields": {
                "next_layer_base_addr": {"offset": 0,  "size": 2, "type": "u16 LE", "hex": f"0x{_u16_le(h,a+0):04X}",  "value": _u16_le(h, a + 0)},
                "layer_type":           {"offset": 2,  "size": 1, "type": "u8",     "hex": f"0x{h[a+2]:02X}",          "value": lt_val, "name": lt_name},
                "weight_base":          {"offset": 3,  "size": 2, "type": "u16 LE", "hex": f"0x{_u16_le(h,a+3):04X}",  "value": _u16_le(h, a + 3)},
                "bias_base":            {"offset": 5,  "size": 2, "type": "u16 LE", "hex": f"0x{_u16_le(h,a+5):04X}",  "value": _u16_le(h, a + 5)},
                "C_out":                {"offset": 7,  "size": 2, "type": "u16 LE", "hex": f"0x{_u16_le(h,a+7):04X}",  "value": _u16_le(h, a + 7)},
                "C_in":                 {"offset": 9,  "size": 2, "type": "u16 LE", "hex": f"0x{_u16_le(h,a+9):04X}",  "value": _u16_le(h, a + 9)},
                "H":                    {"offset": 11, "size": 1, "type": "u8",     "hex": f"0x{h[a+11]:02X}",         "value": int(h[a + 11])},
                "W":                    {"offset": 12, "size": 1, "type": "u8",     "hex": f"0x{h[a+12]:02X}",         "value": int(h[a + 12])},
                "stride":               {"offset": 13, "size": 1, "type": "u8",     "hex": f"0x{h[a+13]:02X}",         "value": int(h[a + 13])},
                "rd_mem":               {"offset": 14, "size": 1, "type": "u8",     "hex": f"0x{h[a+14]:02X}",         "value": int(h[a + 14])},
                "wr_mem":               {"offset": 15, "size": 1, "type": "u8",     "hex": f"0x{h[a+15]:02X}",         "value": int(h[a + 15])},
                "z_x":                  {"offset": 16, "size": 1, "type": "u8",     "hex": f"0x{h[a+16]:02X}",         "value": int(h[a + 16])},
                "z_y":                  {"offset": 17, "size": 1, "type": "u8",     "hex": f"0x{h[a+17]:02X}",         "value": int(h[a + 17])},
                "m_requant":            {"offset": 18, "size": 2, "type": "u16 LE", "hex": f"0x{_u16_le(h,a+18):04X}", "value": _u16_le(h, a + 18)},
            },
        })
    obj = {
        "rom_name": "DWConv_header",
        "spec": "ROM_v4.md §3.1",
        "addressing": "byte",
        "entry_size_bytes": DWCONV_HEADER_SIZE,
        "total_bytes": len(header_bytes),
        "num_layers": len(layers),
        "layers": layers,
    }
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def write_json_dw_weight(weight_words: list[bytes], debug_layers: list, path: str):
    """Decode DWConv weight words (72-bit) into per-layer JSON with 3×3 kernels."""
    layers = []
    for dl in debug_layers:
        w_base  = dl["weight_base"]
        n_words = dl["weight_words"]
        lt      = dl["layer_type"]
        C_out   = dl["C_out"]
        C_in    = dl["C_in"]
        words_data = []
        for wi in range(n_words):
            word = weight_words[w_base + wi]
            # Decode 9 int8 values as 3×3 kernel
            kernel_flat = [_s8(word[k]) for k in range(9)]
            kernel_3x3 = [kernel_flat[r*3:(r+1)*3] for r in range(3)]
            entry = {
                "word_addr":   w_base + wi,
                "local_index": wi,
                "hex":         _word_hex_msb(word),
                "kernel_3x3":  kernel_3x3,
            }
            if lt == LT_DWCONV:
                entry["c_out"] = wi
            elif lt == LT_NORMALCONV:
                entry["c_out"] = wi // C_in
                entry["c_in"]  = wi % C_in
            words_data.append(entry)
        lt_name = {LT_DWCONV: "DWConv", LT_NORMALCONV: "NormalConv"}.get(lt, f"unknown({lt})")
        layers.append({
            "layer_index":  dl["index"],
            "name":         dl["name"],
            "layer_type":   lt,
            "layer_type_name": lt_name,
            "weight_base":  w_base,
            "num_words":    n_words,
            "C_out": C_out, "C_in": C_in,
            "words": words_data,
        })
    obj = {
        "rom_name": "DWConv_weight",
        "spec": "ROM_v4.md §3.2",
        "word_width_bits": 72,
        "total_words": len(weight_words),
        "num_layers": len(layers),
        "layers": layers,
    }
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def write_json_dw_bias(bias_words: list[bytes], debug_layers: list, path: str):
    """Decode DWConv bias words (16-bit) into per-layer JSON."""
    layers = []
    for dl in debug_layers:
        b_base  = dl["bias_base"]
        n_words = dl["bias_words"]
        C_out   = dl["C_out"]
        words_data = []
        for wi in range(n_words):
            word = bias_words[b_base + wi]
            words_data.append({
                "word_addr":    b_base + wi,
                "local_index":  wi,
                "c_out":        wi,
                "hex":          _word_hex_msb(word),
                "int16_value":  _s16_from_word(word),
            })
        layers.append({
            "layer_index":  dl["index"],
            "name":         dl["name"],
            "bias_base":    b_base,
            "num_words":    n_words,
            "C_out":        C_out,
            "words":        words_data,
        })
    obj = {
        "rom_name": "DWConv_bias",
        "spec": "ROM_v4.md §3.3",
        "word_width_bits": 16,
        "total_words": len(bias_words),
        "num_layers": len(layers),
        "layers": layers,
    }
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


# ---- GAP_FC ----

def write_json_gapfc_header(header_bytes: bytes, debug_layers: list, path: str):
    """Decode GAP_FC header ROM into per-layer JSON."""
    layers = []
    for dl in debug_layers:
        a = dl["header_base"]
        h = header_bytes
        layers.append({
            "layer_index": dl["index"],
            "name":        dl["name"],
            "byte_offset": a,
            "raw_hex":     h[a:a + GAPFC_HEADER_SIZE].hex(),
            "fields": {
                "next_layer_base_addr": {"offset": 0,  "size": 2, "type": "u16 LE", "hex": f"0x{_u16_le(h,a+0):04X}",  "value": _u16_le(h, a + 0)},
                "layer_type":           {"offset": 2,  "size": 1, "type": "u8",     "hex": f"0x{h[a+2]:02X}",          "value": int(h[a + 2])},
                "weight_base":          {"offset": 3,  "size": 2, "type": "u16 LE", "hex": f"0x{_u16_le(h,a+3):04X}",  "value": _u16_le(h, a + 3)},
                "bias_base":            {"offset": 5,  "size": 2, "type": "u16 LE", "hex": f"0x{_u16_le(h,a+5):04X}",  "value": _u16_le(h, a + 5)},
                "C_out":                {"offset": 7,  "size": 2, "type": "u16 LE", "hex": f"0x{_u16_le(h,a+7):04X}",  "value": _u16_le(h, a + 7)},
                "C_in":                 {"offset": 9,  "size": 2, "type": "u16 LE", "hex": f"0x{_u16_le(h,a+9):04X}",  "value": _u16_le(h, a + 9)},
                "H":                    {"offset": 11, "size": 1, "type": "u8",     "hex": f"0x{h[a+11]:02X}",         "value": int(h[a + 11])},
                "W":                    {"offset": 12, "size": 1, "type": "u8",     "hex": f"0x{h[a+12]:02X}",         "value": int(h[a + 12])},
                "rd_mem":               {"offset": 13, "size": 1, "type": "u8",     "hex": f"0x{h[a+13]:02X}",         "value": int(h[a + 13])},
                "wr_mem":               {"offset": 14, "size": 1, "type": "u8",     "hex": f"0x{h[a+14]:02X}",         "value": int(h[a + 14])},
                "z_x":                  {"offset": 15, "size": 1, "type": "u8",     "hex": f"0x{h[a+15]:02X}",         "value": int(h[a + 15])},
                "z_y":                  {"offset": 16, "size": 1, "type": "u8",     "hex": f"0x{h[a+16]:02X}",         "value": int(h[a + 16])},
                "m_requant":            {"offset": 17, "size": 2, "type": "u16 LE", "hex": f"0x{_u16_le(h,a+17):04X}", "value": _u16_le(h, a + 17)},
            },
        })
    obj = {
        "rom_name": "GAP_FC_header",
        "spec": "ROM_v4.md §4.1",
        "addressing": "byte",
        "entry_size_bytes": GAPFC_HEADER_SIZE,
        "total_bytes": len(header_bytes),
        "num_layers": len(layers),
        "layers": layers,
    }
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def write_json_gapfc_weight(weight_words: list[bytes], debug_layers: list, path: str):
    """Decode GAP_FC weight words (8-bit) into per-layer JSON."""
    layers = []
    for dl in debug_layers:
        w_base  = dl["weight_base"]
        n_words = dl["weight_words"]
        C_out   = dl["C_out"]
        C_in    = dl["C_in"]
        words_data = []
        for wi in range(n_words):
            word = weight_words[w_base + wi]
            c_out = wi // C_in
            c_in  = wi % C_in
            words_data.append({
                "word_addr":    w_base + wi,
                "local_index":  wi,
                "c_out":        c_out,
                "c_in":         c_in,
                "hex":          _word_hex_msb(word),
                "int8_value":   _s8(word[0]),
            })
        layers.append({
            "layer_index":  dl["index"],
            "name":         dl["name"],
            "weight_base":  w_base,
            "num_words":    n_words,
            "C_out": C_out, "C_in": C_in,
            "words": words_data,
        })
    obj = {
        "rom_name": "GAP_FC_weight",
        "spec": "ROM_v4.md §4.2",
        "word_width_bits": 8,
        "total_words": len(weight_words),
        "num_layers": len(layers),
        "layers": layers,
    }
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def write_json_gapfc_bias(bias_words: list[bytes], debug_layers: list, path: str):
    """Decode GAP_FC bias words (16-bit) into per-layer JSON."""
    layers = []
    for dl in debug_layers:
        b_base  = dl["bias_base"]
        n_words = dl["bias_words"]
        C_out   = dl["C_out"]
        words_data = []
        for wi in range(n_words):
            word = bias_words[b_base + wi]
            words_data.append({
                "word_addr":    b_base + wi,
                "local_index":  wi,
                "c_out":        wi,
                "hex":          _word_hex_msb(word),
                "int16_value":  _s16_from_word(word),
            })
        layers.append({
            "layer_index":  dl["index"],
            "name":         dl["name"],
            "bias_base":    b_base,
            "num_words":    n_words,
            "C_out":        C_out,
            "words":        words_data,
        })
    obj = {
        "rom_name": "GAP_FC_bias",
        "spec": "ROM_v4.md §4.3",
        "word_width_bits": 16,
        "total_words": len(bias_words),
        "num_layers": len(layers),
        "layers": layers,
    }
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


# =====================================================================
# Main
# =====================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Generate ROM v4 (Split-ROM) from INT8 state dict"
    )
    ap.add_argument("--workdir", type=str, required=True,
                    help="Directory with nano_shufflenet_v2_05_10k.py and nano_shufflenet_int8.pt")
    ap.add_argument("--outdir", type=str, default=None,
                    help="Output directory (default: same as workdir)")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    outdir  = Path(args.outdir) if args.outdir else workdir
    ref_py       = workdir / "nano_shufflenet_v2_05_10k.py"
    int8_sd_path = workdir / "nano_shufflenet_int8.pt"

    if not ref_py.exists():
        raise FileNotFoundError(f"Missing {ref_py}")
    if not int8_sd_path.exists():
        raise FileNotFoundError(f"Missing {int8_sd_path}")

    # Import model module
    ref_mod = import_module(ref_py, "nano_ref")

    # Determine num_classes
    labels_json = workdir / "labels.json"
    if labels_json.exists():
        with open(labels_json) as f:
            num_classes = len(json.load(f))
    else:
        num_classes = 14  # fallback

    print(f"[INFO] num_classes = {num_classes}")

    # Build quantized model & load state dict
    qmodel, sd = build_quantized_model(ref_mod, num_classes, int8_sd_path)

    stage_repeats = list(
        ref_mod.NanoShuffleNetV2_10k(num_classes=num_classes).stage_repeats
    )
    print(f"[INFO] stage_repeats = {stage_repeats}")

    # ── Build all layer lists ──
    pw_layers, dw_layers, gapfc_layers = build_all_layer_lists(sd, stage_repeats)
    print(f"[INFO] PWConv layers:  {len(pw_layers)}")
    print(f"[INFO] DWConv layers:  {len(dw_layers)}")
    print(f"[INFO] GAP_FC layers:  {len(gapfc_layers)}")

    # Print summary
    for i, l in enumerate(pw_layers):
        print(f"  PW[{i:2d}] {l['name']:40s} C_out={l['C_out']:3d} C_in={l['C_in']:3d} "
              f"H={l['H']:3d} W={l['W']:3d} shuf={l['shuffle_mode']} br={l['branch']} "
              f"cat_align={l['concat_align']}  m_req={l['m_requant']:5d} "
              f"zx={l['z_x']:3d} zy={l['z_y']:3d}")
    for i, l in enumerate(dw_layers):
        lt_str = "NConv" if l["layer_type"] == LT_NORMALCONV else "DWConv"
        print(f"  DW[{i:2d}] {l['name']:40s} {lt_str:5s} C_out={l['C_out']:3d} C_in={l['C_in']:3d} "
              f"H={l['H']:3d} W={l['W']:3d} s={l['stride']}  m_req={l['m_requant']:5d} "
              f"zx={l['z_x']:3d} zy={l['z_y']:3d}")
    for i, l in enumerate(gapfc_layers):
        print(f"  GF[{i:2d}] {l['name']:40s} C_out={l['C_out']:3d} C_in={l['C_in']:3d} "
              f"H={l['H']:3d} W={l['W']:3d}  m_req={l['m_requant']:5d} "
              f"zx={l['z_x']:3d} zy={l['z_y']:3d}")

    # ── Build ROMs ──
    pw_hdr, pw_w, pw_b, pw_dbg = build_pw_roms(pw_layers)
    dw_hdr, dw_w, dw_b, dw_dbg = build_dw_roms(dw_layers)
    gf_hdr, gf_w, gf_b, gf_dbg = build_gapfc_roms(gapfc_layers)

    print(f"\n[INFO] PWConv  — header={len(pw_hdr):5d} B, weight={len(pw_w):5d} words, bias={len(pw_b):4d} words")
    print(f"[INFO] DWConv  — header={len(dw_hdr):5d} B, weight={len(dw_w):5d} words, bias={len(dw_b):4d} words")
    print(f"[INFO] GAP_FC  — header={len(gf_hdr):5d} B, weight={len(gf_w):5d} words, bias={len(gf_b):4d} words")

    # ── Write COE files ──
    outdir.mkdir(parents=True, exist_ok=True)

    write_coe_bytes(pw_hdr, str(outdir / "PWConv_header.coe"))
    write_coe_words(pw_w,   str(outdir / "PWConv_weight.coe"))
    write_coe_words(pw_b,   str(outdir / "PWConv_bias.coe"))

    write_coe_bytes(dw_hdr, str(outdir / "DWConv_header.coe"))
    write_coe_words(dw_w,   str(outdir / "DWConv_weight.coe"))
    write_coe_words(dw_b,   str(outdir / "DWConv_bias.coe"))

    write_coe_bytes(gf_hdr, str(outdir / "GAP_FC_header.coe"))
    write_coe_words(gf_w,   str(outdir / "GAP_FC_weight.coe"))
    write_coe_words(gf_b,   str(outdir / "GAP_FC_bias.coe"))

    for name in ["PWConv_header", "PWConv_weight", "PWConv_bias",
                  "DWConv_header", "DWConv_weight", "DWConv_bias",
                  "GAP_FC_header", "GAP_FC_weight", "GAP_FC_bias"]:
        print(f"[OK] Written: {outdir / name}.coe")

    # ── Write debug JSON ──
    json_path = str(outdir / "rom_v4_debug.json")
    write_debug_json(pw_dbg, dw_dbg, gf_dbg,
                     pw_hdr, pw_w, pw_b,
                     dw_hdr, dw_w, dw_b,
                     gf_hdr, gf_w, gf_b,
                     json_path)
    print(f"[OK] Written: {json_path}")

    # ── Write per-COE JSON debug files ──
    write_json_pw_header(pw_hdr, pw_dbg, str(outdir / "PWConv_header.json"))
    write_json_pw_weight(pw_w,   pw_dbg, str(outdir / "PWConv_weight.json"))
    write_json_pw_bias(pw_b,     pw_dbg, str(outdir / "PWConv_bias.json"))

    write_json_dw_header(dw_hdr, dw_dbg, str(outdir / "DWConv_header.json"))
    write_json_dw_weight(dw_w,   dw_dbg, str(outdir / "DWConv_weight.json"))
    write_json_dw_bias(dw_b,     dw_dbg, str(outdir / "DWConv_bias.json"))

    write_json_gapfc_header(gf_hdr, gf_dbg, str(outdir / "GAP_FC_header.json"))
    write_json_gapfc_weight(gf_w,   gf_dbg, str(outdir / "GAP_FC_weight.json"))
    write_json_gapfc_bias(gf_b,     gf_dbg, str(outdir / "GAP_FC_bias.json"))

    for name in ["PWConv_header", "PWConv_weight", "PWConv_bias",
                  "DWConv_header", "DWConv_weight", "DWConv_bias",
                  "GAP_FC_header", "GAP_FC_weight", "GAP_FC_bias"]:
        print(f"[OK] Written: {outdir / name}.json")


if __name__ == "__main__":
    main()
