import torch, torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from dataset import PatchDataset
from model_unet import SRHFOperatorUNet
from model_monolith import MonolithFieldNet

def denorm(x, m, s): return x * s + m

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_npz   = "data/dumps/val_patches.npz"
    stats_npz = "data/dumps/stats_patches.npz"

    ds  = PatchDataset(val_npz, stats_path=stats_npz, normalize=True)
    ld  = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

    # 置換HSE-ML（演算子学習）
    A = SRHFOperatorUNet(in_ch_phi=5, in_ch_psi=2, base=32).to(device)
    A.load_state_dict(torch.load("results/ckpts/srhf_unet.pt", map_location=device), strict=False)
    A.eval()

    # モノリスML
    C = MonolithFieldNet(in_ch_phi=5, base=32).to(device)
    C.load_state_dict(torch.load("results/ckpts/monolith_fieldnet.pt", map_location=device), strict=False)
    C.eval()

    pm = ds._pm_t.to(device); ps = ds._ps_t.to(device)
    ym = ds._ym_t.to(device); ys = ds._ys_t.to(device)

    def eval_model(M):
        mse_std, mse_den, n = 0.0, 0.0, 0
        with torch.no_grad():
            for psi, phi, omg, y, dv, extra in ld:
                psi, phi, omg, y, dv = psi.to(device), phi.to(device), omg.to(device), y.to(device), dv.to(device)
                pred = M(psi, phi, omg)
                mse_std += F.mse_loss(pred, y, reduction="mean").item() * psi.size(0)
                # 非正規化
                psi_den  = denorm(psi, pm, ps)
                pred_den = denorm(pred, ym, ys)
                y_den    = denorm(y,    ym, ys)
                diff2 = (pred_den - y_den) ** 2
                mse_den += diff2.mean(dim=(1,2,3,4)).sum().item()
                n += psi.size(0)
        return mse_std/max(1,n), mse_den/max(1,n)

    a_std, a_den = eval_model(A)
    c_std, c_den = eval_model(C)

    print("== Validation (same data) ==")
    print(f"A: Operator-ML     MSE(std)={a_std:.4e}  MSE(denorm)={a_den:.4e}")
    print(f"C: Monolith Field  MSE(std)={c_std:.4e}  MSE(denorm)={c_den:.4e}")

if __name__ == "__main__":
    main()

