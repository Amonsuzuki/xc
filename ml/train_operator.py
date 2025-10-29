import os, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from dataset import PatchDataset
from model_unet import SRHFOperatorUNet

def seed_everything(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def denorm(x, m, s):
    # x: (B,2,P,P,P), m/s: (2,1,1,1)
    return x * s + m

def energy_on_patch(psi, vpsi, dv):
    # Re⟨psi | vpsi⟩ ≈ sum_r (psi_r*vpsi_r + psi_i*vpsi_i) * dv
    pr, pi = psi[:,0], psi[:,1]
    vr, vi = vpsi[:,0], vpsi[:,1]
    e = (pr*vr + pi*vi).sum(dim=(1,2,3)) * dv.view(-1)  # (B,)
    return e

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(0)

    train_npz = "data/dumps/train_patches.npz"
    val_npz   = "data/dumps/val_patches.npz"
    stats_npz = "data/dumps/stats_patches.npz"  # compute_stats.py で作成

    # データ
    train_ds = PatchDataset(train_npz, stats_path=stats_npz, normalize=True)
    val_ds   = PatchDataset(val_npz,   stats_path=stats_npz, normalize=True)
    train_ld = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    val_ld   = DataLoader(val_ds,   batch_size=2, shuffle=False, num_workers=0)

    # モデル
    model = SRHFOperatorUNet(in_ch_phi=5, in_ch_psi=2, base=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    l2 = nn.MSELoss()
    l1 = nn.SmoothL1Loss()
    clip_grad = 1.0

    # 逆変換の定数（GPUへ）
    pm = train_ds._pm_t.to(device)  # (2,1,1,1)
    ps = train_ds._ps_t.to(device)
    ym = train_ds._ym_t.to(device)
    ys = train_ds._ys_t.to(device)

    E_WEIGHT = 0.1  # エネルギー整合ロスの重み（調整ポイント）

    def val_epoch(ep):
        model.eval()
        mse_sum_std, mse_sum_den, n = 0.0, 0.0, 0
        with torch.no_grad():
            for psi, phi, omg, y, dv, extra in val_ld:
                psi  = psi.to(device);  phi = phi.to(device)
                omg  = omg.to(device);  y   = y.to(device);  dv = dv.to(device)
                pred = model(psi, phi, omg)

                # 標準化空間の MSE（従来）
                mse_sum_std += l2(pred, y).item() * psi.size(0)

                # --- 非正規化の MSE を追加で計算 ---
                psi_den  = denorm(psi, pm, ps)
                pred_den = denorm(pred, ym, ys)
                y_den    = denorm(y,    ym, ys)
                # チャネル/空間平均の MSE（1サンプルごとに）
                diff2 = (pred_den - y_den) ** 2
                mse_sample = diff2.mean(dim=(1,2,3,4))  # (B,)
                mse_sum_den += mse_sample.sum().item()

                n += psi.size(0)

        print(f"[{ep:02d}] val_op_MSE(std) ={mse_sum_std/max(1,n):.4e}  "
              f"val_op_MSE(denorm)={mse_sum_den/max(1,n):.4e}")

    # 学習ループ
    EPOCHS = 100
    for ep in range(EPOCHS):
        model.train()
        for psi, phi, omg, y, dv, extra in train_ld:
            psi, phi, omg, y, dv = psi.to(device), phi.to(device), omg.to(device), y.to(device), dv.to(device)
            opt.zero_grad()

            pred = model(psi, phi, omg)
            loss_op = l2(pred, y)

            # --- エネルギー整合（正規化を戻して計算） ---
            psi_den = denorm(psi, pm, ps)      # (B,2,P,P,P)
            pred_den = denorm(pred, ym, ys)
            y_den    = denorm(y,    ym, ys)

            e_pred = energy_on_patch(psi_den, pred_den, dv)  # (B,)
            e_true = energy_on_patch(psi_den, y_den,    dv)  # (B,)
            loss_E = l1(e_pred, e_true)

            loss = loss_op + E_WEIGHT * loss_E
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()

        val_epoch(ep)

    # （任意）チェックポイント保存
    os.makedirs("results/ckpts", exist_ok=True)
    torch.save(model.state_dict(), "results/ckpts/srhf_unet.pt")
    print("[SAVE] results/ckpts/srhf_unet.pt")

if __name__ == "__main__":
    main()

