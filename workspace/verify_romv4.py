#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_romv4.py — ROM v4 (Split-ROM Architecture) end-to-end verification.

Verifies the 9 v4 ROM COE files against the PyTorch INT8 quantized model.
All RTL-bitwidth-aware arithmetic kernels are embedded directly.

Architecture: NanoShuffleNetV2_10k (quantized INT8, QNNPACK backend)

Usage:
  uv run verify_romv4.py --config config.toml
"""

import argparse
import re
import tomllib
from pathlib import Path
import importlib.util

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets


# =====================================================================
# Section 1 — ROM v4 Constants
# =====================================================================
P = 4                       # PWConvUnit output-channel parallelism
PWCONV_HEADER_SIZE = 26
DWCONV_HEADER_SIZE = 20
GAPFC_HEADER_SIZE  = 19
LAST_LAYER = 0xFFFF

LT_PWCONV     = 0
LT_DWCONV     = 1
LT_NORMALCONV = 2
LT_GAPFC      = 3


# =====================================================================
# Section 2 — RTL Fixed-Width Constants
# =====================================================================
N_FRAC    = 15
S_PRE     = 0
ACC_BITS  = 22
A_BITS    = 22
M_BITS    = 16
BIAS_BITS = 16
POST_BITS = 16


# =====================================================================
# Section 3 — RTL Fixed-Width Helpers
# =====================================================================
def _mask(bits: int) -> int:
    return (1 << bits) - 1


def wrap_sint(x: torch.Tensor, bits: int) -> torch.Tensor:
    x = x.to(torch.int64)
    m = _mask(bits)
    u = x & m
    sign = 1 << (bits - 1)
    return (u ^ sign) - sign


def sat_uint(x: torch.Tensor, bits: int) -> torch.Tensor:
    x = x.to(torch.int64)
    return torch.clamp(x, 0, (1 << bits) - 1)


def rshift_round_away_sint(
    x: torch.Tensor, sh: int, out_bits: int | None = None
) -> torch.Tensor:
    assert sh >= 0
    x = x.to(torch.int64)
    if sh == 0:
        y = x
    else:
        ax = torch.abs(x)
        y_mag = (ax + (1 << (sh - 1))) >> sh
        y = torch.where(x < 0, -y_mag, y_mag)
    if out_bits is not None:
        y = wrap_sint(y, out_bits)
    return y


def add_wrap_sint(
    a: torch.Tensor, b: torch.Tensor, bits: int
) -> torch.Tensor:
    return wrap_sint(a.to(torch.int64) + b.to(torch.int64), bits)


# =====================================================================
# Section 4 — RTL Arithmetic Kernels
# =====================================================================
def requant_from_acc_hw(
    acc: torch.Tensor, m_u: torch.Tensor, z_y: int, relu: bool
) -> torch.Tensor:
    a = wrap_sint(acc, ACC_BITS)
    a_pre = rshift_round_away_sint(a, S_PRE, out_bits=A_BITS)
    m_u = m_u.to(torch.int64)
    p = a_pre.to(torch.int64) * m_u
    p = wrap_sint(p, A_BITS + M_BITS)
    k = N_FRAC - S_PRE
    y = rshift_round_away_sint(p, k, out_bits=POST_BITS)
    qy_u = sat_uint(y + int(z_y), 8).to(torch.uint8)
    if relu:
        z = int(z_y)
        if z != 0:
            qy_u = torch.maximum(qy_u, torch.tensor(z, dtype=torch.uint8))
    return qy_u


def conv2d_u8_i8_hw(
    qx_u8, zx, w_i8_int32, b_int16_s64, zy, m_u16,
    stride, padding, groups, relu,
):
    assert qx_u8.dtype == torch.uint8
    x = wrap_sint(qx_u8.to(torch.int64) - int(zx), 9).to(torch.int32)
    w = wrap_sint(w_i8_int32.to(torch.int64), 8).to(torch.int32)
    acc = F.conv2d(
        x, w, bias=None, stride=stride, padding=padding, groups=groups
    ).to(torch.int64)
    acc = wrap_sint(acc, ACC_BITS)
    b_int = wrap_sint(b_int16_s64.to(torch.int64), BIAS_BITS)
    acc = add_wrap_sint(acc, b_int.view(1, -1, 1, 1), ACC_BITS)
    m = torch.tensor(
        int(m_u16) & _mask(M_BITS), dtype=torch.int64, device=acc.device
    )
    return requant_from_acc_hw(acc, m, int(zy), relu=relu)


def linear_u8_i8_hw(qx_u8, zx, w_i8_int32, b_int16_s64, zy, m_u16):
    assert qx_u8.dtype == torch.uint8
    x = wrap_sint(qx_u8.to(torch.int64) - int(zx), 9).to(torch.int32)
    w = wrap_sint(w_i8_int32.to(torch.int64), 8).to(torch.int32)
    acc = torch.einsum(
        "ni,oi->no", x.to(torch.int64), w.to(torch.int64)
    )
    acc = wrap_sint(acc, ACC_BITS)
    b_int = wrap_sint(b_int16_s64.to(torch.int64), BIAS_BITS)
    acc = add_wrap_sint(acc, b_int.view(1, -1), ACC_BITS)
    m = torch.tensor(
        int(m_u16) & _mask(M_BITS), dtype=torch.int64, device=acc.device
    )
    return requant_from_acc_hw(acc, m, int(zy), relu=False)


def avgpool8x8_u8_hw(qx_u8: torch.Tensor, z_x: int) -> torch.Tensor:
    assert qx_u8.shape[2] == 8 and qx_u8.shape[3] == 8
    d = wrap_sint(qx_u8.to(torch.int64) - int(z_x), 9)
    S = d.sum(dim=(2, 3), keepdim=True).to(torch.int64)
    S = wrap_sint(S, 16)
    A = rshift_round_away_sint(S, 6, out_bits=16)
    return sat_uint(A + int(z_x), 8).to(torch.uint8)


def channel_shuffle_u8(qx_u8: torch.Tensor, groups: int = 2) -> torch.Tensor:
    N, C, H, W = qx_u8.shape
    assert C % groups == 0
    x = qx_u8.view(N, groups, C // groups, H, W)
    x = x.transpose(1, 2).contiguous()
    return x.view(N, C, H, W)


def align_u8_to_u8_hw(
    q_u8: torch.Tensor, z_in: int, z_out: int, m_align: int
) -> torch.Tensor:
    d = wrap_sint(q_u8.to(torch.int64) - int(z_in), 9)
    m = torch.tensor(
        int(m_align) & _mask(M_BITS), dtype=torch.int64, device=d.device
    )
    p = wrap_sint(d * m, 9 + M_BITS)
    u = rshift_round_away_sint(p, N_FRAC, out_bits=16)
    return sat_uint(u + int(z_out), 8).to(torch.uint8)


# =====================================================================
# Section 5 — Engine / Import / Build Utilities
# =====================================================================
def set_engine_qnnpack():
    if "qnnpack" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "qnnpack"


def import_ref_module(ref_py: Path):
    spec = importlib.util.spec_from_file_location("nano_ref", str(ref_py))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_pytorch_int8_model(ref_mod, num_classes, int8_sd_path: Path):
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


def build_val_loader(
    data_root: Path, img_size: int, batch_size: int,
    seed: int, num_workers: int, ref_mod,
):
    _, val_tf = ref_mod.get_transforms(img_size=img_size)
    full_ds = datasets.ImageFolder(root=str(data_root))
    targets = [s[1] for s in full_ds.samples]

    from sklearn.model_selection import train_test_split
    _, val_idx = train_test_split(
        np.arange(len(targets)),
        test_size=0.2,
        stratify=targets,
        random_state=seed,
    )

    val_ds = datasets.ImageFolder(root=str(data_root), transform=val_tf)
    val_subset = Subset(val_ds, val_idx)
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return val_loader, full_ds.classes


# =====================================================================
# Section 6 — ROM v4 I/O Helpers
# =====================================================================
def read_coe_u8(path: Path) -> np.ndarray:
    """Read byte-level COE file (radix=16) as u8 array."""
    txt = Path(path).read_text().strip()
    m = re.search(
        r"memory_initialization_vector\s*=\s*(.*?)\s*;",
        txt, flags=re.S | re.I,
    )
    if not m:
        raise ValueError(f"Cannot parse COE file: {path}")
    body = m.group(1)
    tokens = [t.strip() for t in body.replace("\n", "").split(",") if t.strip()]
    return np.array([int(t, 16) & 0xFF for t in tokens], dtype=np.uint8)


def read_coe_words(path: Path) -> list[int]:
    """Read word-level COE file (radix=16). Returns list of ints."""
    txt = Path(path).read_text().strip()
    m = re.search(
        r"memory_initialization_vector\s*=\s*(.*?)\s*;",
        txt, flags=re.S | re.I,
    )
    if not m:
        raise ValueError(f"Cannot parse COE file: {path}")
    body = m.group(1)
    tokens = [t.strip() for t in body.replace("\n", "").split(",") if t.strip()]
    return [int(t, 16) for t in tokens]


def rd_u16_le(mem, addr):
    return (int(mem[addr + 1]) << 8) | int(mem[addr])


# =====================================================================
# Section 7 — V4 ROM Readers
# =====================================================================
class PWConvROM_V4:
    """Read PWConvUnit v4 split ROM (Header + Weight + Bias)."""

    def __init__(self, header_u8: np.ndarray,
                 weight_words: list[int],
                 bias_words: list[int]):
        self.header = header_u8
        self.weight_words = weight_words
        self.bias_words = bias_words
        self.layers: list[dict] = []
        self._parse_all()
        self._cursor = 0

    def _parse_all(self):
        addr = 0
        while addr + PWCONV_HEADER_SIZE <= len(self.header):
            layer = self._parse_one(addr)
            self.layers.append(layer)
            if layer["next_layer_base_addr"] == LAST_LAYER:
                break
            addr = layer["next_layer_base_addr"]

    def _parse_one(self, addr: int) -> dict:
        h = self.header
        nxt   = rd_u16_le(h, addr + 0)
        lt    = int(h[addr + 2])
        w_base = rd_u16_le(h, addr + 3)
        b_base = rd_u16_le(h, addr + 5)
        C_out = rd_u16_le(h, addr + 7)
        C_in  = rd_u16_le(h, addr + 9)
        H     = int(h[addr + 11])
        W     = int(h[addr + 12])
        shuf  = int(h[addr + 13])
        branch = int(h[addr + 14])
        cat_al = int(h[addr + 15])
        # rd_mem, wr_mem at +16, +17 — skipped (not used by golden)
        zx    = int(h[addr + 18])
        zy    = int(h[addr + 19])
        mreq  = rd_u16_le(h, addr + 20)
        z1    = int(h[addr + 22])
        z2    = int(h[addr + 23])
        ma    = rd_u16_le(h, addr + 24)

        # --- Extract weight tensor ---
        assert C_out % P == 0
        n_groups = C_out // P
        w_arr = np.zeros((C_out, C_in), dtype=np.int8)
        for g in range(n_groups):
            for ci in range(C_in):
                word = self.weight_words[w_base + g * C_in + ci]
                for p in range(P):
                    val = (word >> (p * 8)) & 0xFF
                    if val >= 128:
                        val -= 256
                    w_arr[g * P + p, ci] = val

        w_t = torch.from_numpy(w_arr).to(torch.int32).unsqueeze(-1).unsqueeze(-1)

        # --- Extract bias tensor ---
        b_arr = np.zeros(C_out, dtype=np.int16)
        for g in range(n_groups):
            word = self.bias_words[b_base + g]
            for p in range(P):
                u16 = (word >> (p * 16)) & 0xFFFF
                if u16 >= 0x8000:
                    u16 -= 0x10000
                b_arr[g * P + p] = u16

        b_t = torch.from_numpy(b_arr.astype(np.int16)).to(torch.int64)

        return {
            "next_layer_base_addr": nxt,
            "layer_type": lt,
            "C_out": C_out, "C_in": C_in,
            "H": H, "W": W,
            "shuffle_mode": shuf, "branch": branch, "concat_align": cat_al,
            "zx": zx, "zy": zy, "mreq": mreq,
            "z1_align": z1, "z2_align": z2, "m_align": ma,
            "w": w_t, "b": b_t,
        }

    def pop(self) -> dict:
        if self._cursor >= len(self.layers):
            raise IndexError("No more PWConv layers in ROM v4")
        layer = self.layers[self._cursor]
        self._cursor += 1
        return layer

    def reset(self):
        self._cursor = 0


class DWConvROM_V4:
    """Read DWConvUnit v4 split ROM (Header + Weight + Bias)."""

    def __init__(self, header_u8: np.ndarray,
                 weight_words: list[int],
                 bias_words: list[int]):
        self.header = header_u8
        self.weight_words = weight_words
        self.bias_words = bias_words
        self.layers: list[dict] = []
        self._parse_all()
        self._cursor = 0

    def _parse_all(self):
        addr = 0
        while addr + DWCONV_HEADER_SIZE <= len(self.header):
            layer = self._parse_one(addr)
            self.layers.append(layer)
            if layer["next_layer_base_addr"] == LAST_LAYER:
                break
            addr = layer["next_layer_base_addr"]

    def _parse_one(self, addr: int) -> dict:
        h = self.header
        nxt   = rd_u16_le(h, addr + 0)
        lt    = int(h[addr + 2])
        w_base = rd_u16_le(h, addr + 3)
        b_base = rd_u16_le(h, addr + 5)
        C_out = rd_u16_le(h, addr + 7)
        C_in  = rd_u16_le(h, addr + 9)
        H     = int(h[addr + 11])
        W     = int(h[addr + 12])
        stride = int(h[addr + 13])
        # rd_mem, wr_mem at +14, +15 — skipped
        zx    = int(h[addr + 16])
        zy    = int(h[addr + 17])
        mreq  = rd_u16_le(h, addr + 18)

        # --- Determine groups, relu from layer_type ---
        if lt == LT_NORMALCONV:
            groups = 1
            relu = True
        elif lt == LT_DWCONV:
            groups = C_out
            relu = False
        else:
            raise ValueError(f"Unexpected layer_type={lt} in DWConv ROM")

        # --- Extract weight tensor (72-bit words = 9 bytes) ---
        if lt == LT_DWCONV:
            # [C_out, 1, 3, 3]
            w_arr = np.zeros((C_out, 1, 3, 3), dtype=np.int8)
            for co in range(C_out):
                word = self.weight_words[w_base + co]
                for k in range(9):
                    val = (word >> (k * 8)) & 0xFF
                    if val >= 128:
                        val -= 256
                    w_arr[co, 0, k // 3, k % 3] = val
        else:
            # NormalConv: [C_out, C_in, 3, 3]
            w_arr = np.zeros((C_out, C_in, 3, 3), dtype=np.int8)
            for co in range(C_out):
                for ci in range(C_in):
                    word = self.weight_words[w_base + co * C_in + ci]
                    for k in range(9):
                        val = (word >> (k * 8)) & 0xFF
                        if val >= 128:
                            val -= 256
                        w_arr[co, ci, k // 3, k % 3] = val

        w_t = torch.from_numpy(w_arr).to(torch.int32)

        # --- Extract bias (16-bit words) ---
        b_arr = np.zeros(C_out, dtype=np.int16)
        for co in range(C_out):
            word = self.bias_words[b_base + co]
            if word >= 0x8000:
                word -= 0x10000
            b_arr[co] = word

        b_t = torch.from_numpy(b_arr.astype(np.int16)).to(torch.int64)

        return {
            "next_layer_base_addr": nxt,
            "layer_type": lt,
            "C_out": C_out, "C_in": C_in,
            "H": H, "W": W, "stride": stride,
            "groups": groups, "relu": relu,
            "zx": zx, "zy": zy, "mreq": mreq,
            "w": w_t, "b": b_t,
        }

    def pop(self) -> dict:
        if self._cursor >= len(self.layers):
            raise IndexError("No more DWConv layers in ROM v4")
        layer = self.layers[self._cursor]
        self._cursor += 1
        return layer

    def reset(self):
        self._cursor = 0


class GAPFC_ROM_V4:
    """Read GAP_FC v4 split ROM (Header + Weight + Bias)."""

    def __init__(self, header_u8: np.ndarray,
                 weight_words: list[int],
                 bias_words: list[int]):
        self.header = header_u8
        self.weight_words = weight_words
        self.bias_words = bias_words
        self.layers: list[dict] = []
        self._parse_all()
        self._cursor = 0

    def _parse_all(self):
        addr = 0
        while addr + GAPFC_HEADER_SIZE <= len(self.header):
            layer = self._parse_one(addr)
            self.layers.append(layer)
            if layer["next_layer_base_addr"] == LAST_LAYER:
                break
            addr = layer["next_layer_base_addr"]

    def _parse_one(self, addr: int) -> dict:
        h = self.header
        nxt   = rd_u16_le(h, addr + 0)
        lt    = int(h[addr + 2])
        w_base = rd_u16_le(h, addr + 3)
        b_base = rd_u16_le(h, addr + 5)
        C_out = rd_u16_le(h, addr + 7)
        C_in  = rd_u16_le(h, addr + 9)
        H     = int(h[addr + 11])
        W     = int(h[addr + 12])
        # rd_mem, wr_mem at +13, +14 — skipped
        zx    = int(h[addr + 15])
        zy    = int(h[addr + 16])
        mreq  = rd_u16_le(h, addr + 17)

        # --- Extract FC weight (byte-level words) ---
        w_arr = np.zeros((C_out, C_in), dtype=np.int8)
        for co in range(C_out):
            for ci in range(C_in):
                val = self.weight_words[w_base + co * C_in + ci]
                val = val & 0xFF
                if val >= 128:
                    val -= 256
                w_arr[co, ci] = val

        w_t = torch.from_numpy(w_arr).to(torch.int32)

        # --- Extract FC bias (16-bit words) ---
        b_arr = np.zeros(C_out, dtype=np.int16)
        for co in range(C_out):
            word = self.bias_words[b_base + co]
            if word >= 0x8000:
                word -= 0x10000
            b_arr[co] = word

        b_t = torch.from_numpy(b_arr.astype(np.int16)).to(torch.int64)

        return {
            "next_layer_base_addr": nxt,
            "layer_type": lt,
            "C_out": C_out, "C_in": C_in,
            "H": H, "W": W,
            "zx": zx, "zy": zy, "mreq": mreq,
            "w": w_t, "b": b_t,
        }

    def pop(self) -> dict:
        if self._cursor >= len(self.layers):
            raise IndexError("No more GAP_FC layers in ROM v4")
        layer = self.layers[self._cursor]
        self._cursor += 1
        return layer

    def reset(self):
        self._cursor = 0


# =====================================================================
# Section 8 — Forward Pass (V4 Split ROM)
# =====================================================================
@torch.no_grad()
def forward_from_rom(
    qx0_u8: torch.Tensor,
    pw_rom: PWConvROM_V4,
    dw_rom: DWConvROM_V4,
    gapfc_rom: GAPFC_ROM_V4,
    stage_repeats: list[int],
    verbose: bool = False,
) -> torch.Tensor:
    """
    Run the full network forward pass using only ROM data.
    Layer order is implicit: each ROM is popped in execution order.
    """
    pw_rom.reset()
    dw_rom.reset()
    gapfc_rom.reset()

    # --- conv1.0  (NormalConv in DWConvUnit ROM) ---
    dw = dw_rom.pop()
    assert dw["layer_type"] == LT_NORMALCONV, (
        f"Expected NormalConv, got layer_type={dw['layer_type']}"
    )
    q = conv2d_u8_i8_hw(
        qx0_u8, dw["zx"], dw["w"], dw["b"], dw["zy"], dw["mreq"],
        stride=dw["stride"], padding=1,
        groups=dw["groups"], relu=dw["relu"],
    )
    if verbose:
        print(f"[verify] conv1.0: {tuple(q.shape)}")

    # --- Stages ---
    for sid, reps in enumerate(stage_repeats):
        for bi in range(reps):
            if bi == 0:
                # ========== stride-2 block ==========
                # branch1.0  DW
                dw = dw_rom.pop()
                q_b1 = conv2d_u8_i8_hw(
                    q, dw["zx"], dw["w"], dw["b"], dw["zy"], dw["mreq"],
                    stride=dw["stride"], padding=1,
                    groups=dw["groups"], relu=dw["relu"],
                )

                # branch1.2  PW
                pw = pw_rom.pop()
                q_b1 = conv2d_u8_i8_hw(
                    q_b1, pw["zx"], pw["w"], pw["b"], pw["zy"], pw["mreq"],
                    stride=1, padding=0, groups=1, relu=True,
                )

                # branch2.0  PW
                pw = pw_rom.pop()
                q_b2 = conv2d_u8_i8_hw(
                    q, pw["zx"], pw["w"], pw["b"], pw["zy"], pw["mreq"],
                    stride=1, padding=0, groups=1, relu=True,
                )

                # branch2.3  DW
                dw = dw_rom.pop()
                q_b2 = conv2d_u8_i8_hw(
                    q_b2, dw["zx"], dw["w"], dw["b"], dw["zy"], dw["mreq"],
                    stride=dw["stride"], padding=1,
                    groups=dw["groups"], relu=dw["relu"],
                )

                # branch2.5  PW (with concat-align)
                pw = pw_rom.pop()
                q_b2 = conv2d_u8_i8_hw(
                    q_b2, pw["zx"], pw["w"], pw["b"], pw["zy"], pw["mreq"],
                    stride=1, padding=0, groups=1, relu=True,
                )

                assert pw["concat_align"] == 1, (
                    f"Expected concat_align=1 for stride-2 branch2.5"
                )
                q_b2a = align_u8_to_u8_hw(
                    q_b2, pw["z2_align"], pw["z1_align"], pw["m_align"],
                )
                q = torch.cat([q_b1, q_b2a], dim=1)
                q = channel_shuffle_u8(q, groups=2)

            else:
                # ========== stride-1 block ==========
                c = q.shape[1]
                q_x1 = q[:, : c // 2, :, :]
                q_x2 = q[:, c // 2 :, :, :]

                # branch2.0  PW
                pw = pw_rom.pop()
                q_b2 = conv2d_u8_i8_hw(
                    q_x2, pw["zx"], pw["w"], pw["b"], pw["zy"], pw["mreq"],
                    stride=1, padding=0, groups=1, relu=True,
                )

                # branch2.3  DW
                dw = dw_rom.pop()
                q_b2 = conv2d_u8_i8_hw(
                    q_b2, dw["zx"], dw["w"], dw["b"], dw["zy"], dw["mreq"],
                    stride=dw["stride"], padding=1,
                    groups=dw["groups"], relu=dw["relu"],
                )

                # branch2.5  PW (with concat-align)
                pw = pw_rom.pop()
                q_b2 = conv2d_u8_i8_hw(
                    q_b2, pw["zx"], pw["w"], pw["b"], pw["zy"], pw["mreq"],
                    stride=1, padding=0, groups=1, relu=True,
                )

                assert pw["concat_align"] == 1, (
                    "Expected concat_align=1 for stride-1 branch2.5"
                )
                q_b2a = align_u8_to_u8_hw(
                    q_b2, pw["z2_align"], pw["z1_align"], pw["m_align"],
                )
                q = torch.cat([q_x1, q_b2a], dim=1)
                q = channel_shuffle_u8(q, groups=2)

            if verbose:
                print(f"[verify] stages.{sid}.{bi}: {tuple(q.shape)}")

    # --- GAP + FC ---
    gf = gapfc_rom.pop()
    q = avgpool8x8_u8_hw(q, gf["zx"])
    q = q.view(q.shape[0], -1)  # flatten spatial

    q = linear_u8_i8_hw(q, gf["zx"], gf["w"], gf["b"], gf["zy"], gf["mreq"])

    logits = (q.to(torch.float32) - float(gf["zy"])) * 1.0
    return logits


# =====================================================================
# Section 9 — Config Loader + CLI
# =====================================================================
def load_config(toml_path: Path) -> dict:
    with open(toml_path, "rb") as f:
        raw = tomllib.load(f)

    base = toml_path.resolve().parent
    paths  = raw.get("paths", {})
    rom_v4 = raw.get("rom_v4", {})
    ev     = raw.get("eval", {})

    def resolve(p: str) -> str:
        return str((base / p).resolve())

    cfg = {
        "workdir":   resolve(paths.get("workdir", ".")),
        "data_root": resolve(paths.get("data_root", "")),
        # v4 ROMs
        "pw_header":    resolve(rom_v4.get("pw_header", "")),
        "pw_weight":    resolve(rom_v4.get("pw_weight", "")),
        "pw_bias":      resolve(rom_v4.get("pw_bias", "")),
        "dw_header":    resolve(rom_v4.get("dw_header", "")),
        "dw_weight":    resolve(rom_v4.get("dw_weight", "")),
        "dw_bias":      resolve(rom_v4.get("dw_bias", "")),
        "gapfc_header": resolve(rom_v4.get("gapfc_header", "")),
        "gapfc_weight": resolve(rom_v4.get("gapfc_weight", "")),
        "gapfc_bias":   resolve(rom_v4.get("gapfc_bias", "")),
        # eval
        "batch_size":  int(ev.get("batch_size", 64)),
        "img_size":    int(ev.get("img_size", 128)),
        "seed":        int(ev.get("seed", 0)),
        "num_workers": int(ev.get("num_workers", 4)),
        "max_batches": ev.get("max_batches", None),
    }
    if cfg["max_batches"] is not None:
        cfg["max_batches"] = int(cfg["max_batches"])
    return cfg


def parse_args() -> dict:
    ap = argparse.ArgumentParser(
        description="Verify ROM v4 (Split-ROM) against PyTorch INT8 model",
    )
    ap.add_argument("--config", type=str, default=None,
                    help="Path to config.toml")
    ap.add_argument("--workdir",    type=str, default=None)
    ap.add_argument("--data_root",  type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--img_size",   type=int, default=None)
    ap.add_argument("--seed",       type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--pw_header",    type=str, default=None)
    ap.add_argument("--pw_weight",    type=str, default=None)
    ap.add_argument("--pw_bias",      type=str, default=None)
    ap.add_argument("--dw_header",    type=str, default=None)
    ap.add_argument("--dw_weight",    type=str, default=None)
    ap.add_argument("--dw_bias",      type=str, default=None)
    ap.add_argument("--gapfc_header", type=str, default=None)
    ap.add_argument("--gapfc_weight", type=str, default=None)
    ap.add_argument("--gapfc_bias",   type=str, default=None)

    cli = ap.parse_args()

    if cli.config is not None:
        cfg = load_config(Path(cli.config))
    else:
        cfg = {}

    for key in [
        "workdir", "data_root", "batch_size", "img_size", "seed",
        "num_workers", "max_batches",
        "pw_header", "pw_weight", "pw_bias",
        "dw_header", "dw_weight", "dw_bias",
        "gapfc_header", "gapfc_weight", "gapfc_bias",
    ]:
        cli_val = getattr(cli, key, None)
        if cli_val is not None:
            cfg[key] = cli_val

    required = [
        "workdir", "data_root",
        "pw_header", "pw_weight", "pw_bias",
        "dw_header", "dw_weight", "dw_bias",
        "gapfc_header", "gapfc_weight", "gapfc_bias",
    ]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        ap.error(
            f"Missing required parameters: {missing}. "
            "Provide them via --config or CLI flags."
        )

    cfg.setdefault("batch_size", 64)
    cfg.setdefault("img_size", 128)
    cfg.setdefault("seed", 0)
    cfg.setdefault("num_workers", 4)
    cfg.setdefault("max_batches", None)

    return cfg


# =====================================================================
# Section 10 — Main
# =====================================================================
def main():
    cfg = parse_args()

    workdir = Path(cfg["workdir"])
    ref_py       = workdir / "nano_shufflenet_v2_05_10k.py"
    int8_sd_path = workdir / "nano_shufflenet_int8.pt"
    if not ref_py.exists():
        raise FileNotFoundError(f"Missing {ref_py}")
    if not int8_sd_path.exists():
        raise FileNotFoundError(f"Missing {int8_sd_path}")

    set_engine_qnnpack()
    ref_mod = import_ref_module(ref_py)

    val_loader, class_names = build_val_loader(
        Path(cfg["data_root"]),
        cfg["img_size"], cfg["batch_size"],
        cfg["seed"], cfg["num_workers"], ref_mod,
    )
    num_classes = len(class_names)
    print(f"[INFO] classes={num_classes}  {class_names}")

    stage_repeats = list(
        ref_mod.NanoShuffleNetV2_10k(num_classes=num_classes).stage_repeats
    )
    print(f"[INFO] stage_repeats={stage_repeats}")

    qmodel, sd = build_pytorch_int8_model(ref_mod, num_classes, int8_sd_path)
    qmodel.eval()

    s_in = float(sd["quant.scale"].item())
    z_in = int(sd["quant.zero_point"].item())

    # ----- Load v4 ROMs -----
    print("[INFO] Loading ROM v4 files ...")

    pw_rom = PWConvROM_V4(
        read_coe_u8(Path(cfg["pw_header"])),
        read_coe_words(Path(cfg["pw_weight"])),
        read_coe_words(Path(cfg["pw_bias"])),
    )
    print(f"  PWConv ROM v4: {len(pw_rom.layers)} layers")

    dw_rom = DWConvROM_V4(
        read_coe_u8(Path(cfg["dw_header"])),
        read_coe_words(Path(cfg["dw_weight"])),
        read_coe_words(Path(cfg["dw_bias"])),
    )
    print(f"  DWConv ROM v4: {len(dw_rom.layers)} layers")

    gapfc_rom = GAPFC_ROM_V4(
        read_coe_u8(Path(cfg["gapfc_header"])),
        read_coe_words(Path(cfg["gapfc_weight"])),
        read_coe_words(Path(cfg["gapfc_bias"])),
    )
    print(f"  GAP_FC ROM v4: {len(gapfc_rom.layers)} layers")

    # ----- Evaluation loop -----
    total = 0
    correct_q = 0
    correct_g = 0
    top1_mismatch = 0
    max_batches = cfg["max_batches"]

    for bi, (x, y) in enumerate(val_loader):
        if max_batches is not None and bi >= max_batches:
            break

        x, y = x.cpu(), y.cpu()

        # PyTorch INT8 reference
        out_q = qmodel(x)
        pred_q = out_q.argmax(dim=1)

        # Golden ROM-based forward
        qx = torch.quantize_per_tensor(x, s_in, z_in, torch.quint8)
        out_g = forward_from_rom(
            qx.int_repr(), pw_rom, dw_rom, gapfc_rom,
            stage_repeats, verbose=False,
        )
        pred_g = out_g.argmax(dim=1)

        total += y.numel()
        correct_q += (pred_q == y).sum().item()
        correct_g += (pred_g == y).sum().item()
        top1_mismatch += (pred_q != pred_g).sum().item()

        if bi % 20 == 0:
            print(
                f"[batch {bi}]  "
                f"pytorch_int8_acc={100 * correct_q / total:.3f}%  "
                f"golden_romv4_acc={100 * correct_g / total:.3f}%  "
                f"top1_mismatch={100 * top1_mismatch / total:.3f}%"
            )

    print("\n=========== Result ===========")
    print(f"acc_pytorch_int8:  {100.0 * correct_q / total:.3f}%")
    print(f"acc_golden_romv4:  {100.0 * correct_g / total:.3f}%")
    print(f"top1_mismatch:     {100.0 * top1_mismatch / total:.3f}%")
    print(f"total_samples:     {total}")


if __name__ == "__main__":
    main()
