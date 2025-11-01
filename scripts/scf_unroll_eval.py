#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scf_unroll_eval.py  (band metrics)
  - 学習済みモデルで v_x^SR ψ を推定し、擬似SCFを 1〜3 step 回す
  - 各 step で：
      * "交換寄与の準位列" λ_b = <ψ_b | v_x^SR ψ_b> を pred/ref で計算
      * 帯域MAE = mean_b |λ_b^pred - λ_b^ref|
      * もし未占有が含まれていれば、交換寄与ギャップ Δ_gap = λ_LUMO - λ_HOMO の誤差も出力
  - 直交セル・Γ点、小〜中グリッド前提（パッチ=全域）。必要に応じてタイル化は拡張可。
"""
import os, sys, argparse
import numpy as np
import h5py, torch
import torch.nn.functional as F

sys.path.append("ml")
from model_unet import SRHFOperatorUNet

def gram_schmidt(psi):  # psi: (B,2,Nx,Ny,Nz)
    B, _, Nx, Ny, Nz = psi.shape
    def inner(a, b):
        ar, ai = a[:,0], a[:,1]; br, bi = b[:,0], b[:,1]
        return (ar*br + ai*bi).sum(dim=(1,2,3), keepdim=True)  # (B,1,1,1)
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
    """λ_b = <ψ_b | vψ_b> を (B,) で返す"""
    pr, pi = psi[:,0], psi[:,1]
    vr, vi = vpsi[:,0], vpsi[:,1]
    lam = (pr*vr + pi*vi).sum(dim=(1,2,3)) * dv
    return lam  # (B,)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/dumps/Si_prim_gamma.h5")
    ap.add_argument("--ckpt", default="results/ckpts/srhf_unet.pt")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # --- 入力 ---
    with h5py.File(args.h5, "r") as f:
        psi_r = np.array(f["psi_real"])         # (B,Nx,Ny,Nz)
        psi_i = np.array(f["psi_imag"])
        vx_r  = np.array(f["vxpsi_real"])       # (B,Nx,Ny,Nz)
        vx_i  = np.array(f["vxpsi_imag"])
        rho   = np.array(f["rho"])
        grad  = np.array(f["grad_rho"])         # (3,Nx,Ny,Nz)
        omega = float(np.array(f["omega"]))
        cell  = np.array(f["cell"])
        grid  = np.array(f["grid_shape"])
        vol   = abs(np.linalg.det(cell))
        dv    = float(vol / (grid[0]*grid[1]*grid[2]))
        occ   = np.array(f["occ"])              # (B,) 1 or 2 typically

    B, Nx, Ny, Nz = psi_r.shape
    has_unocc = (occ < 1e-8).any()  # 後述のダンプ拡張を入れた場合のみ True になり得る

    # φ = [rho, |grad|, grad_x, grad_y, grad_z]
    grad_mag = np.sqrt(np.maximum(1e-30, (grad**2).sum(axis=0)))
    phi = np.stack([rho, grad_mag, grad[0], grad[1], grad[2]], axis=0)  # (5,Nx,Ny,Nz)

    print("[B]", B) # Why 6?
    # テンソル化
    psi = torch.from_numpy(np.stack([psi_r, psi_i], axis=1).astype(np.float32))   # (B,2,Nx,Ny,Nz)
    vref = torch.from_numpy(np.stack([vx_r, vx_i], axis=1).astype(np.float32))    # (B,2,Nx,Ny,Nz)
    phi_t = torch.from_numpy(phi[None, ...])                                      # (1,5,Nx,Ny,Nz)
    omg_t = torch.tensor([[omega]], dtype=torch.float32)
    psi = gram_schmidt(psi)  # 直交・正規化
    dv_t = torch.tensor(dv, dtype=torch.float32)

    # モデル
    model = SRHFOperatorUNet(in_ch_phi=5, in_ch_psi=2, base=32).to(args.device)
    sd = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    def apply_model(psi_b):  # (B,2,Nx,Ny,Nz) -> (B,2,Nx,Ny,Nz)
        Bnow = psi_b.shape[0]
        phi_rep = phi_t.expand(Bnow, -1, -1, -1, -1).to(args.device)
        omg_rep = omg_t.expand(Bnow, -1).to(args.device)
        with torch.no_grad():
            vpsi = model(psi_b.to(args.device), phi_rep, omg_rep).cpu()
        return vpsi

    def report(step, psi_now):
        vpsi_pred = apply_model(psi_now)
        # 帯域MAE（交換寄与の準位列）
        lam_pred = band_expectation(psi_now, vpsi_pred, dv_t)     # (B,)
        lam_ref  = band_expectation(psi_now, vref,      dv_t)     # ref は dump のラベル
        band_mae = torch.mean(torch.abs(lam_pred - lam_ref)).item()
        # “交換寄与ギャップ”の誤差（未占有が無ければスキップ）
        gap_msg = "gap: n/a (no unoccupied bands in dump)"
        if has_unocc:
            # HOMO = occ>0 の最大 index, LUMO = その次の未占有最小 index
            occ_t = torch.from_numpy(occ)
            homo_idx = torch.nonzero(occ_t > 1e-8).max().item()
            lumo_idx = (homo_idx + 1) if (homo_idx + 1) < B else None
            if lumo_idx is not None and (occ[lumo_idx] < 1e-8):
                gap_pred = (lam_pred[lumo_idx] - lam_pred[homo_idx]).item()
                gap_ref  = (lam_ref [lumo_idx] - lam_ref [homo_idx]).item()
                gap_err  = abs(gap_pred - gap_ref)
                gap_msg  = f"gap_err(exch)={gap_err:.6e}  (pred={gap_pred:.6e}, ref={gap_ref:.6e})"
        print(f"[step {step}] band_MAE(exch)={band_mae:.6e}   {gap_msg}")

    # step 0（初期）
    report(0, psi)

    # 擬似SCF
    alpha = args.alpha
    psi_cur = psi.clone()
    for s in range(1, args.steps+1):
        vpsi = apply_model(psi_cur)
        psi_cur = psi_cur - alpha * vpsi          # 前進オイラー型
        psi_cur = gram_schmidt(psi_cur)           # 直交化
        report(s, psi_cur)

if __name__ == "__main__":
    main()
