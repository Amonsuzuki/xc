#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dump_gpaw_hse_sr.py  (A方式・演算子学習用データダンプ: vxpsi はダミー)

概要:
  - 入力構造（ASE対応: [[.traj]], .xyz, .cif など）から GPAW+HSE06 で単点計算を実行
  - 占有軌道 ψ_{n,k}(r) を実空間グリッドで取得
  - 短距離HF交換の作用結果 (v_x^SR ψ) は、[[まずは]]ダミーのゼロ配列を出力
  - 密度 ρ と ∇ρ などの局所特徴も書き出し
  - HDF5 1ファイルにまとめて保存

使い方例:
  python scripts/dump_gpaw_hse_sr.py \
      --input data/raw/Si.traj \
      --output data/dumps/Si_prim_gamma.h5 \
      --omega 0.11 --h 0.25 --k 1 1 1

メモ:
  - まずは Γ点・小セルでOK（--k 1 1 1）
  - vxpsi_* はダミーなので学習コードの動作確認が主目的
      - 後で (v_x^SR ψ) 実装に差し替える
"""
import os
import sys
import argparse
import time
import numpy as np
import h5py

from ase.io import read
from gpaw import GPAW, FermiDirac

def build_kgrid_orthorhombic(cell, grid_shape):
    """直交セルを仮定して k グリッド(角周波数)を作る。cell は (3,3) で各行が格子ベクトル。
       grid_shape=(Nx,Ny,Nz)。戻り値は kx, ky, kz の 3D 配列。"""
    a, b, c = np.linalg.norm(cell[0]), np.linalg.norm(cell[1]), np.linalg.norm(cell[2])
    Nx, Ny, Nz = grid_shape
    kx_1d = 2*np.pi*np.fft.fftfreq(Nx, d=a/Nx)
    ky_1d = 2*np.pi*np.fft.fftfreq(Ny, d=b/Ny)
    kz_1d = 2*np.pi*np.fft.fftfreq(Nz, d=c/Nz)
    kx, ky, kz = np.meshgrid(kx_1d, ky_1d, kz_1d, indexing='ij')
    return kx, ky, kz

def screened_coulomb_Kk(kx, ky, kz, omega):
    """FT{ erfc(ω r)/r } = 4π/k^2 * (1 - exp(-k^2/(4ω^2)))、k=0 は極限で π/ω^2。"""
    k2 = kx*kx + ky*ky + kz*kz
    Kk = np.empty_like(k2, dtype=np.float64)
    mask = (k2 > 0)
    Kk[mask] = 4*np.pi/k2[mask] * (1.0 - np.exp(-k2[mask]/(4.0*omega*omega)))
    Kk[~mask] = np.pi/(omega*omega)  # k=0 の極限
    return Kk



def parse_args():
    p = argparse.ArgumentParser(description='Dump GPAW HSE(SR) data for operator learning (vxpsi).')
    p.add_argument('--input', required=True, help='Input structure file (ASE-readable: .traj, .xyz, .cif, ...)')
    p.add_argument('--output', required=True, help='Output HDF5 path, e.g., data/dumps/sample.h5')
    p.add_argument('--omega', type=float, default=0.11, help='HSE06 screening parameter omega (1/Å-ish).')
    p.add_argument('--h', type=float, default=0.25, help='Real-space grid spacing (Å).')
    p.add_argument('--k', type=int, nargs=3, default=[1, 1, 1], help='k-point grid, e.g., --k 1 1 1 (Gamma-only MVP)')
    p.add_argument('--smearing', type=float, default=0.01, help='Fermi smearing width (eV).')
    p.add_argument('--txt', default=None, help='GPAW text log file (e.g., results/gpaw.log). Default: None (silent).')
    p.add_argument('--spin', type=int, default=0, help='Spin channel for dump (0: up/non-magnetic, 1: down).')
    p.add_argument('--occ-thresh', type=float, default=1e-6, help='Occupation threshold to pick occupied bands.')
    p.add_argument('--pad', action='store_true', help='Pad wavefunctions to full cell shape (recommended).')
    p.add_argument('--sr-mode', choices=['dummy','self','occ'], default='occ',
                   help='SR-HF action mode: dummy/self/occ')
    p.add_argument('--add-unocc', type=int, default=0,
               help='占有に加えて未占有バンドをこの本数だけ追加（HOMOの次から）')
    p.add_argument('--nbands', type=int, default=None,
               help='GPAW の nbands を明示（占有+未占有が入るだけの本数を指定）')
    return p.parse_args()


def grad3_periodic(f: np.ndarray):
    """中心差分 + 周期境界で ∇f を計算（各軸ごとに np.roll）"""
    gx = np.roll(f, -1, axis=0) - np.roll(f,  1, axis=0)
    gy = np.roll(f, -1, axis=1) - np.roll(f,  1, axis=1)
    gz = np.roll(f, -1, axis=2) - np.roll(f,  1, axis=2)
    return np.stack([gx, gy, gz], axis=0) * 0.5


def get_grid_shape(calc: GPAW):
    """GPAW のグリッド形状 (Nx,Ny,Nz) を取得"""
    # できるだけバージョン差に頑健に
    gd = None
    if hasattr(calc, 'get_grid_descriptor'):
        gd = calc.get_grid_descriptor()
    elif hasattr(calc, 'wfs') and hasattr(calc.wfs, 'gd'):
        gd = calc.wfs.gd
    if gd is None:
        raise RuntimeError('Failed to get grid descriptor. GPAW API changed?')
    # 属性名の差異へ対処
    if hasattr(gd, 'N_c'):
        return tuple(int(n) for n in gd.N_c)
    if hasattr(gd, 'shape'):
        return tuple(int(n) for n in gd.shape)
    raise RuntimeError('Unknown grid descriptor format (no N_c/shape).')


def get_occupations(calc: GPAW, kpt_index=0, spin=0):
    """
    占有数ベクトルを取得。GPAW バージョンでAPI差があるため、複数パターンを試す。
    戻り値: occ (nbands,), nbands
    """
    occ = None
    # 新しめAPI: get_occupation_numbers(kpt=, spin=)
    try:
        occ = calc.get_occupation_numbers(kpt=kpt_index, spin=spin)
    except TypeError:
        # 旧API: 引数なしで返る（形状が nbands or (nbands, nkpts) or (nbands, nkpts, nspins)）
        occ = calc.get_occupation_numbers()
        if occ is None:
            raise
        occ = np.array(occ)

        if occ.ndim == 1:
            pass
        elif occ.ndim == 2:
            occ = occ[:, kpt_index]
        elif occ.ndim == 3:
            occ = occ[:, kpt_index, spin]
        else:
            raise RuntimeError(f'Unexpected occupation array shape: {occ.shape}')

    occ = np.array(occ, dtype=float).ravel()
    nbands = calc.get_number_of_bands()
    if occ.shape[0] != nbands:
        # 一部バージョンで get_number_of_bands(kpt,spin) が必要な場合も -> フォールバック
        nbands = occ.shape[0]
    return occ, nbands


def get_wavefunction(calc: GPAW, band: int, kpt_index=0, spin=0, pad=True):
    """
    ψ_{band,kpt,spin}(r) の実空間配列を取得。
    Gamma/非Gamma で dtype が実/複素のことがあるが、実部/虚部に分解して扱う。
    """
    # 推奨: 擬波動関数で十分（MVP）。必要なら all-electron へ差し替え可。
    psi = calc.get_pseudo_wave_function(band=band, kpt=kpt_index, spin=spin, pad=pad)
    psi = np.asarray(psi)
    if np.iscomplexobj(psi):
        psi_r = psi.real.astype(np.float32, copy=False)
        psi_i = psi.imag.astype(np.float32, copy=False)
    else:
        psi_r = psi.astype(np.float32, copy=False)
        psi_i = np.zeros_like(psi_r, dtype=np.float32)
    return psi_r, psi_i


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 構造を読み込み
    atoms = read(args.input)

    # HSE06 設定と GPAW 計算器
    kpts = {'size': tuple(args.k)}
    calc = GPAW(mode={'name': 'pw', 'ecut': 400},
                xc='HSE06',
                kpts=kpts,
                occupations=FermiDirac(args.smearing),
                h=args.h,
                txt=args.txt,
                nbands=args.nbands)
    atoms.calc = calc

    # 単点計算を実行（軽めの収束でOK）
    t0 = time.time()
    e_total = atoms.get_potential_energy()
    print("[e_total]", e_total)
    t1 = time.time()
    print("[time]", t1 - t0)

    # グリッド形状・格子
    grid_shape = get_grid_shape(calc)  # (Nx,Ny,Nz)
    print("[grid_shape]", grid_shape)

    # 2 Si for 1 cell
    cell = atoms.cell.array.astype(np.float64) # unit vector
    print("[cell]", cell)
    volume = abs(np.linalg.det(cell))
    print("[volume]", volume)
    dv = volume / (grid_shape[0]*grid_shape[1]*grid_shape[2])

    # 密度（全スピン和；必要に応じてspinを分けて取得）
    # per grid point
    try:
        rho = calc.get_all_electron_density(gridrefinement=1)
    except TypeError:
        # バージョン違いへ対応
        rho = calc.get_all_electron_density()
    rho = np.asarray(rho, dtype=np.float32)
    print("[rho]", rho.shape)

    # ∇ρ（中心差分・周期境界）
    grad_rho = grad3_periodic(rho).astype(np.float32)  # (3,Nx,Ny,Nz)
    print("[grad_rho]", grad_rho.shape)

    # 占有数とバンド選択（占有のみ）
    spin = args.spin
    kpt_index = 0  # MVP: Γ点
    occ, nbands = get_occupations(calc, kpt_index=kpt_index, spin=spin)
    print("[occ]", occ)
    print("[nbands]", nbands)
    # nbands = 8 for Si
    """
    occ_mask = occ > args.occ_thresh
    band_ids = np.where(occ_mask)[0]
    if band_ids.size == 0:
        print('[WARN] No occupied bands found over threshold. Falling back to lowest occupied guess.')
        # 最低1本は出す
        band_ids = np.array([int(np.argmax(occ))], dtype=np.int32)
    """

    # identify how many orbits are fullfilled
    occ_mask = occ > args.occ_thresh
    print("[occ_mask]", occ_mask)
    # get index of filled orbits
    occ_ids = np.where(occ_mask)[0]
    print("[occ_ids]", occ_ids)
    if occ_ids.size == 0:
        occ_ids = np.array([int(np.argmax(occ))], dtype=np.int32)

    band_ids = occ_ids.tolist()
    print("[band_ids]", band_ids)
    if args.add_unocc > 0:
        start = occ_ids.max() + 1
        extra = list(range(start, min(start + args.add_unocc, calc.get_number_of_bands())))
        band_ids += extra
    band_ids = np.array(band_ids, dtype=np.int32)


    # ψ を全占有バンド分ロード（1回だけ）
    psi_r_all, psi_i_all = [], []
    # for number of filled orbits
    for b in band_ids:
        pr, pi = get_wavefunction(calc, band=int(b), kpt_index=kpt_index, spin=spin, pad=args.pad)
        psi_r_all.append(pr.astype(np.float32))
        psi_i_all.append(pi.astype(np.float32))
    # concat all orbits
    psi_r_all = np.stack(psi_r_all, axis=0)  # (B,Nx,Ny,Nz)
    psi_i_all = np.stack(psi_i_all, axis=0)
    #print(psi_r_all)
    #print(psi_i_all)

    psi_real = psi_r_all.copy()
    psi_imag = psi_i_all.copy()
    vxpsi_real = np.zeros_like(psi_real, dtype=np.float32)
    vxpsi_imag = np.zeros_like(psi_imag, dtype=np.float32)

    
    #psi_r_list, psi_i_list = [], []
    #vx_r_list,  vx_i_list  = [], []

    # 体積と k カーネル（1回だけ作る）
    # grid coordinates
    kx, ky, kz = build_kgrid_orthorhombic(cell, grid_shape) # (16, 16, 16, 3)

    # short-range screened Coulomb potential
    # (16, 16, 16)
    Kk = screened_coulomb_Kk(kx, ky, kz, args.omega)
    #print(Kk)

    """
    for bi, b in enumerate(band_ids):
        psi_r, psi_i = get_wavefunction(calc, band=int(b), kpt_index=kpt_index, spin=spin, pad=args.pad)
        psi = psi_r.astype(np.float64) + 1j*psi_i.astype(np.float64)

        # --- 自己項のみ（j=b）で φ_bb = (ψ_b* ψ_b) * w_SR を計算 ---
        chi = np.conj(psi) * psi                      # χ_bb(r) = |ψ_b(r)|^2
        Chi_k = np.fft.fftn(chi)                      # FFT
        phi = np.fft.ifftn(Chi_k * Kk).real * dv      # 逆FFT＋体積要素で近似積分スケール

        # v_x^SR ψ_b ≈ - ψ_b * φ_bb
        vxpsi = - psi * phi

        # 返却用（float32, 実/虚）
        psi_r_list.append(psi_r.astype(np.float32))
        psi_i_list.append(psi_i.astype(np.float32))
        vx_r_list.append(vxpsi.real.astype(np.float32))
        vx_i_list.append(vxpsi.imag.astype(np.float32))

    psi_real = np.stack(psi_r_list, axis=0)  # (B,Nx,Ny,Nz)
    psi_imag = np.stack(psi_i_list, axis=0)
    vxpsi_real = np.stack(vx_r_list, axis=0)
    vxpsi_imag = np.stack(vx_i_list, axis=0)
    """

    # 占有重み：スピン非分極系では occ=2 のことが多いので「各スピン1」の近似で min(occ,1.0) を採用
    # mapping from 0~2.0 to 0~1.0
    occ_for_sum = np.minimum(occ[band_ids], 1.0).astype(np.float64)
    print("[occ_for_sum]", occ_for_sum)

    # 作用計算
    t2 = time.time()
    B = len(band_ids)
    for bi in range(B):
        # ψ_b
        psi_b = psi_r_all[bi].astype(np.float64) + 1j*psi_i_all[bi].astype(np.float64)

        if args.sr_mode == 'dummy':
            vxpsi = np.zeros_like(psi_b, dtype=np.complex128)

        elif args.sr_mode == 'self':
            # 自己項 j=b のみ
            chi = np.conj(psi_b) * psi_b                              # χ_bb = |ψ_b|^2, existence pissibility
            phi = np.fft.ifftn(np.fft.fftn(chi) * Kk).real * dv       # φ_bb = χ_bb * w_SR
            vxpsi = - psi_b * phi

        else: # default='occ'
            # 占有和： v_x ψ_b = - Σ_j w_j ψ_j ( (ψ_j* ψ_b) * w_SR )
            acc = np.zeros_like(psi_b, dtype=np.complex128)
            for j in range(B):
                wj = occ_for_sum[j]   # 0〜1
                if wj <= 0.0: 
                    continue
                psi_j = psi_r_all[j].astype(np.float64) + 1j*psi_i_all[j].astype(np.float64)
                chi_bj = np.conj(psi_j) * psi_b # existence possibillity if psi_j = psi_b, it is pair density
                # pair density indicates "How much orbits overlap"
                # The bigger it is, the more they overlap
                phi_bj = np.fft.ifftn(np.fft.fftn(chi_bj) * Kk).real * dv # it should include imaginary, maybe
                # if recognize chi_bj as electron density, phi_bj is screened coulomb potential ("screened by omega!")

                # non-local exchange between b and j
                acc += wj * (psi_j * phi_bj)
            # non-local exchange between b and j which summed up
            vxpsi = -acc

        # 保存（float32）
        vxpsi_real[bi] = vxpsi.real.astype(np.float32)
        vxpsi_imag[bi] = vxpsi.imag.astype(np.float32)

    t3 = time.time()



    # HDF5 へ保存
    with h5py.File(args.output, 'w') as f:
        f.create_dataset('cell', data=cell)
        f.create_dataset('grid_shape', data=np.array(grid_shape, dtype=np.int32))
        f.create_dataset('omega', data=np.float32(args.omega))
        f.create_dataset('kpoint', data=np.array([0.0, 0.0, 0.0], dtype=np.float64))
        f.create_dataset('bands', data=np.array(band_ids, dtype=np.int32))
        #f.create_dataset('bands', data=band_ids.astype(np.int32))
        f.create_dataset('occ', data=occ[band_ids].astype(np.float32))
        f.create_dataset('psi_real', data=psi_real, compression='gzip')
        f.create_dataset('psi_imag', data=psi_imag, compression='gzip')
        f.create_dataset('vxpsi_real', data=vxpsi_real, compression='gzip')
        f.create_dataset('vxpsi_imag', data=vxpsi_imag, compression='gzip')
        f.create_dataset('rho', data=rho, compression='gzip')
        f.create_dataset('grad_rho', data=grad_rho, compression='gzip')
        # τ はGPAWのAPI差が大きいのでMVPでは省略（必要なら後で追加）
        f.attrs['energy_total_eV'] = float(e_total)
        f.attrs['sr_mode'] = args.sr_mode

    # サマリ
    print('--- Dump summary ---')
    print('input      :', args.input)
    print('output     :', args.output)
    print('omega      :', args.omega)
    print('grid_shape :', grid_shape)
    #print('k-points   :', tuple(args.k), '(MVP uses kpt_index=0 only)')
    print('k-points   :', tuple(args.k), '(Γのみ対応)')
    #print('bands(out) :', band_ids.tolist())
    print('bands(out) :', list(map(int, band_ids)))
    print('E_total(eV):', float(e_total))
    print('sr-mode    :', args.sr_mode)
    print('times [s]  : SCF={:.2f}, build={:.2f}, action={:.2f}, total={:.2f}'.format(
        t1-t0, (t2-t1), (t3-t2), (t3-t0)))
    print('Done.')


if __name__ == '__main__':
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
 
