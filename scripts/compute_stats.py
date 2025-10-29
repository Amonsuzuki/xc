#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_stats.py
  - train_patches.npz から ψ, φ, y_vpsi のチャネル毎 mean/std を計算して保存
"""
import argparse
import numpy as np
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train-npz', required=True, help='data/dumps/train_patches.npz')
    p.add_argument('--out-stats', required=True, help='data/dumps/stats_patches.npz')
    return p.parse_args()

def ch_mean_std(x):  # x: (N,C,P,P,P)
    # (N, C, P, P, P) → (C,)
    axes = (0,2,3,4)
    mean = x.mean(axis=axes)
    std  = x.std(axis=axes) + 1e-12
    return mean.astype(np.float32), std.astype(np.float32)

def main():
    args = parse_args()
    z = np.load(args.train_npz)
    psi = z['X_psi']   # (N,2, P,P,P)
    phi = z['X_phi']   # (N,5, P,P,P)
    y   = z['y_vpsi']  # (N,2, P,P,P)

    #psi_mean, psi_std = ch_mean_std(psi)
    phi_mean, phi_std = ch_mean_std(phi)
    psi_mean = np.zeros(2, dtype=np.float32); psi_std = np.ones(2, dtype=np.float32)
    y_mean   = np.zeros(2, dtype=np.float32); y_std   = np.ones(2, dtype=np.float32)    
    #y_mean,   y_std   = ch_mean_std(y)

    os.makedirs(os.path.dirname(args.out_stats), exist_ok=True)
    np.savez(args.out_stats,
             psi_mean=psi_mean, psi_std=psi_std,
             phi_mean=phi_mean, phi_std=phi_std,
             y_mean=y_mean,     y_std=y_std)
    print("[SAVE]", args.out_stats)
    print("psi_mean/std:", psi_mean, psi_std)
    print("phi_mean/std:", phi_mean, phi_std)
    print("y_mean/std  :", y_mean,   y_std)

if __name__ == '__main__':
    main()

