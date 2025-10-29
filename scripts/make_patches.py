#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_patches.py
  - dump_gpaw_hse_sr.py で作った HDF5 から 3D パッチを生成し、学習用 npz を作る
  - φ(r) は [rho, |grad_rho|, grad_x, grad_y, grad_z] の 5ch
  - ψ と (v_x^SR ψ) は [real, imag] の 2ch

使い方例:
  python scripts/make_patches.py \
    --inputs data/dumps/Si_prim_gamma.h5 \
    --out-train data/dumps/train_patches.npz \
    --out-val   data/dumps/val_patches.npz \
    --patch 32 --stride 16 --val-frac 0.1
"""
import os
import sys
import argparse
import glob
import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description='Create 3D patches for operator learning.')
    p.add_argument('--inputs', nargs='+', required=True,
                   help='Input HDF5 files (one or more). Glob OK if shell expands.')
    p.add_argument('--out-train', required=True, help='Output npz for training.')
    p.add_argument('--out-val', required=True, help='Output npz for validation.')
    p.add_argument('--patch', type=int, default=32, help='Patch size P (cubic P x P x P).')
    p.add_argument('--stride', type=int, default=16, help='Stride for sliding window.')
    p.add_argument('--val-frac', type=float, default=0.1, help='Validation fraction (0-1).')
    p.add_argument('--seed', type=int, default=42, help='Random seed for split.')
    return p.parse_args()


def slide_indices(L, P, S):
    """長さ L からパッチサイズ P・ストライド S で開始インデックス列を作る。端は L-P を必ず含める。"""
    if L <= P:
        return [0]
    idx = list(range(0, L - P + 1, S))
    if idx[-1] != L - P:
        idx.append(L - P)
    return idx


def main():
    args = parse_args()
    np.random.seed(args.seed)

    X_psi_list, X_phi_list, Y_vpsi_list, OMG_list, DV_list = [], [], [], [], []

    # 入力ファイルをループ
    for path in args.inputs:
        if not os.path.exists(path):
            print(f"[WARN] missing: {path}")
            continue
        with h5py.File(path, 'r') as f:
            # 取り出し
            psi_r = np.array(f['psi_real'])  # (B,Nx,Ny,Nz)
            psi_i = np.array(f['psi_imag'])
            vx_r  = np.array(f['vxpsi_real'])
            vx_i  = np.array(f['vxpsi_imag'])
            rho   = np.array(f['rho'])       # (Nx,Ny,Nz)
            grad  = np.array(f['grad_rho'])  # (3,Nx,Ny,Nz)
            omega = float(np.array(f['omega']))

            cell = np.array(f['cell'])
            grid_shape = np.array(f['grid_shape'])
            volume = abs(np.linalg.det(cell))
            dv = float(volume / (grid_shape[0]*grid_shape[1]*grid_shape[2]))

            # φ(r): 5ch = [rho, |grad|, grad_x, grad_y, grad_z]
            grad_mag = np.sqrt(np.maximum(1e-30, (grad**2).sum(axis=0)))
            phi = np.stack([rho, grad_mag, grad[0], grad[1], grad[2]], axis=0)  # (5,Nx,Ny,Nz)

            B, Nx, Ny, Nz = psi_r.shape
            P = min(args.patch, Nx, Ny, Nz)   # グリッドが小さければ自動で縮小
            S = args.stride if (Nx >= P and Ny >= P and Nz >= P) else P

            xs = slide_indices(Nx, P, S)
            ys = slide_indices(Ny, P, S)
            zs = slide_indices(Nz, P, S)

            # バンド×パッチでサンプル化
            for b in range(B):
                psi2 = np.stack([psi_r[b], psi_i[b]], axis=0)      # (2,Nx,Ny,Nz)
                vps2 = np.stack([vx_r[b],  vx_i[b]],  axis=0)      # (2,Nx,Ny,Nz)
                for ix in xs:
                    for iy in ys:
                        for iz in zs:
                            sl = (slice(ix, ix+P), slice(iy, iy+P), slice(iz, iz+P))
                            X_psi_list.append(psi2[:, sl[0], sl[1], sl[2]])  # (2,P,P,P)
                            X_phi_list.append(phi[:, sl[0], sl[1], sl[2]])   # (5,P,P,P)
                            Y_vpsi_list.append(vps2[:, sl[0], sl[1], sl[2]]) # (2,P,P,P)
                            OMG_list.append([omega])
                            DV_list.append([dv])

        print(f"[OK] {path} -> bands={B}, grid=({Nx},{Ny},{Nz}), patch={P}, "
              f"count={len(xs)*len(ys)*len(zs)} per band")

    # 連結
    if len(X_psi_list) == 0:
        print("[ERROR] No samples created. Check inputs.")
        sys.exit(1)

    X_psi  = np.stack(X_psi_list, axis=0).astype(np.float32)   # (N,2,P,P,P)
    X_phi  = np.stack(X_phi_list, axis=0).astype(np.float32)   # (N,5,P,P,P)
    Y_vpsi = np.stack(Y_vpsi_list, axis=0).astype(np.float32)  # (N,2,P,P,P)
    OMG    = np.array(OMG_list, dtype=np.float32)              # (N,1)
    DV     = np.array(DV_list, dtype=np.float32)

    # シャッフル & 分割
    N = X_psi.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    n_val = max(1, int(N * args.val_frac))
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]

    def save_npz(path, ids):
        np.savez_compressed(
            path,
            X_psi=X_psi[ids],
            X_phi=X_phi[ids],
            y_vpsi=Y_vpsi[ids],
            omega=OMG[ids],
            dv=DV[ids]
        )
        print(f"[SAVE] {path}: N={len(ids)}")

    os.makedirs(os.path.dirname(args.out_train), exist_ok=True)
    save_npz(args.out_train, tr_idx)
    save_npz(args.out_val,   val_idx)

    # サマリ
    P = X_psi.shape[-1]
    print("--- Patch summary ---")
    print(f"N_total={N}, patch={P}, train={len(tr_idx)}, val={len(val_idx)}")
    print("Channels: X_psi=2, X_phi=5, y_vpsi=2, omega=1")
    print("Done.")


if __name__ == '__main__':
    main()

