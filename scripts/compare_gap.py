#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_gap.py
  - 置換HSE-ML (Operator) と モノリスML (Field) の "交換寄与ギャップ" を同一データで比較
  - 前提: HDF5 に未占有バンドが含まれていること（--add-unocc >= 1）
  - 擬似SCFを 0〜S step 回し、その都度の gap と gap_err を出力
  - 任意で stats_npz を渡すと、phi を訓練時と同じ標準化で入力できる（精度安定化に有効）
"""
import os, sys, argparse
import numpy as np
import h5py, torch
import torch.nn.functional as F

sys.path.append("ml")
from model_unet import SRHFOperatorUNet
from model_monolith import MonolithFieldNet

def gram_schmidt(psi):  # psi: (B,2,Nx,Ny,Nz)
    B, _, Nx, Ny, Nz = psi.shape
    def inner(a, b):
        ar, ai = a[:,0], a[:,1]; br, bi = b[:,0], b[:,1]
        return (ar*br + ai*bi).sum(dim=(1,2,3), keepdim=True)
    out = []
    for i in range(B):
        v = psi[i:i+1].clone()
        for u in out:
            coeff = inner(u, v) / (inner(u, u) + 1e-30)
            v = v - coeff * u
        norm = torch.sqrt(inner(v, v) + 1e-30)
        v = v / norm.view(1,1,1,1,1)
        out.append(v)
    return torch.cat(out, dim=0)

def band_expectation(psi, vpsi, dv):
    pr, pi = psi[:,0], psi[:,1]
    vr, vi = vpsi[:,0], vpsi[:,1]
    return (pr*vr + pi*vi).sum(dim=(1,2,3)) * dv  # (B,)

def maybe_norm_phi(phi, stats_npz):
    if stats_npz is None or not os.path.exists(stats_npz):
        return phi
    s = np.load(stats_npz)
    fm, fs = s["phi_mean"], s["phi_std"]  # (5,)
    fm = fm.reshape(1,5,1,1,1); fs = (fs.reshape(1,5,1,1,1) + 1e-12)
    return (phi - fm) / fs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/dumps/Si_prim_gamma.h5")
    ap.add_argument("--ckpt_A", default="results/ckpts/srhf_unet.pt", help="Operator-ML")
    ap.add_argument("--ckpt_C", default="results/ckpts/monolith_fieldnet.pt", help="Monolith-ML")
    ap.add_argument("--stats", default="data/dumps/stats_patches.npz", help="phi用の標準化統計 (任意)")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # 入力
    with h5py.File(args.h5, "r") as f:
        psi_r = np.array(f["psi_real"])
        psi_i = np.array(f["psi_imag"])
        vx_r  = np.array(f["vxpsi_real"])   # 参照（交換寄与）
        vx_i  = np.array(f["vxpsi_imag"])
        rho   = np.array(f["rho"])
        grad  = np.array(f["grad_rho"])
        omega = float(np.array(f["omega"]))
        cell  = np.array(f["cell"]); grid = np.array(f["grid_shape"])
        vol   = abs(np.linalg.det(cell)); dv = float(vol / (grid[0]*grid[1]*grid[2]))
        occ   = np.array(f["occ"])  # (B,)

    B, Nx, Ny, Nz = psi_r.shape
    # LUMO が存在するかチェック
    if not (occ < 1e-8).any():
        print("[ERROR] HDF5 に未占有バンドが含まれていません。dump 時に --add-unocc を指定してください。")
        sys.exit(1)

    # HOMO/LUMO index
    homo_idx = np.where(occ > 1e-8)[0].max()
    lumo_idx = homo_idx + 1
    if lumo_idx >= B:
        print("[ERROR] LUMO index out of range. --add-unocc を増やしてください。")
        sys.exit(1)

    # 特徴 φ とテンソル整形（全域＝1パッチ前提）
    grad_mag = np.sqrt(np.maximum(1e-30, (grad**2).sum(axis=0)))
    phi = np.stack([rho, grad_mag, grad[0], grad[1], grad[2]], axis=0)[None, ...]  # (1,5,Nx,Ny,Nz)
    phi = maybe_norm_phi(phi, args.stats).astype(np.float32)
    omg = np.array([[omega]], dtype=np.float32)

    psi = np.stack([psi_r, psi_i], axis=1).astype(np.float32)      # (B,2,Nx,Ny,Nz)
    vref = np.stack([vx_r, vx_i], axis=1).astype(np.float32)

    psi_t  = torch.from_numpy(psi)
    vref_t = torch.from_numpy(vref)
    phi_t  = torch.from_numpy(phi)
    omg_t  = torch.from_numpy(omg)
    psi_t  = gram_schmidt(psi_t)
    dv_t   = torch.tensor(dv, dtype=torch.float32)

    # モデル読��込み
    A = SRHFOperatorUNet(in_ch_phi=5, in_ch_psi=2, base=32).to(args.device)
    A.load_state_dict(torch.load(args.ckpt_A, map_location=args.device), strict=False); A.eval()
    C = MonolithFieldNet(in_ch_phi=5, base=32).to(args.device)
    C.load_state_dict(torch.load(args.ckpt_C, map_location=args.device), strict=False); C.eval()

    def apply_model(M, psi_b):
        Bnow = psi_b.shape[0]
        with torch.no_grad():
            vpsi = M(psi_b.to(args.device),
                     phi_t.expand(Bnow, -1, -1, -1, -1).to(args.device),
                     omg_t.expand(Bnow, -1).to(args.device)).cpu()
        return vpsi

    def gap_from(lam):
        return (lam[lumo_idx] - lam[homo_idx]).item()

    def report(step, psi_now):
        out = {}
        for name, M in [("A", A), ("C", C)]:
            vpred = apply_model(M, psi_now)
            lam_pred = band_expectation(psi_now, vpred, dv_t)  # (B,)
            lam_ref  = band_expectation(psi_now, vref_t, dv_t)
            gap_pred = gap_from(lam_pred)
            gap_ref  = gap_from(lam_ref)
            gap_err  = abs(gap_pred - gap_ref)
            out[name] = (gap_pred, gap_err)
        print(f"[step {step}] exch-gap  A: {out['A'][0]:+.6e} (err={out['A'][1]:.6e})"
              f"   C: {out['C'][0]:+.6e} (err={out['C'][1]:.6e})   ref={gap_from(band_expectation(psi_now, vref_t, dv_t)):+.6e}")

    # 0-step
    report(0, psi_t)

    # 擬似SCF
    alpha = args.alpha
    psi_cur = psi_t.clone()
    for s in range(1, args.steps+1):
        # A で更新（※どちらで更新するかは方針次第。ここではAで更新）
        vpsi_A = apply_model(A, psi_cur)
        psi_cur = gram_schmidt(psi_cur - alpha * vpsi_A)
        report(s, psi_cur)

if __name__ == "__main__":
    main()
