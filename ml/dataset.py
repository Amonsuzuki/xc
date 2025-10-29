import numpy as np, torch
from torch.utils.data import Dataset

class PatchDataset(Dataset):
    def __init__(self, npz_path, stats_path=None, normalize=True):
        z = np.load(npz_path)
        self.psi = z["X_psi"].astype(np.float32)      # (N,2,P,P,P)
        self.phi = z["X_phi"].astype(np.float32)      # (N,5,P,P,P)
        self.y   = z["y_vpsi"].astype(np.float32)     # (N,2,P,P,P)
        self.omega = z["omega"].astype(np.float32)    # (N,1)
        self.dv = z["dv"].astype(np.float32)
        
        self.normalize = normalize and (stats_path is not None)
        if self.normalize:
            s = np.load(stats_path)
            #self.psi_mean, self.psi_std = s["psi_mean"], s["psi_std"]
            self.psi_mean = np.zeros(2, np.float32); self.psi_std = np.ones(2, np.float32)
            self.y_mean   = np.zeros(2, np.float32); self.y_std   = np.ones(2, np.float32)
            self.phi_mean, self.phi_std = s["phi_mean"], s["phi_std"]
            #self.y_mean, self.y_std = s["y_mean"], s["y_std"]

            def bs(v): return v.reshape(1, -1, 1, 1, 1).astype(np.float32)
            pm, ps = bs(self.psi_mean), bs(self.psi_std)
            fm, fs = bs(self.phi_mean), bs(self.phi_std)
            ym, ys = bs(self.y_mean), bs(self.y_std)

            #self.psi = (self.psi - pm) / ps
            self.phi = (self.phi - fm) / fs
            #self.y = (self.y - ym) / ys

            self._pm_t = torch.from_numpy(pm[0])
            self._ps_t = torch.from_numpy(ps[0])
            self._ym_t = torch.from_numpy(ym[0])
            self._ys_t = torch.from_numpy(ys[0])
        else:
            self._pm_t = self._ps_t = self._ym_t = self._ys_t = None


    def __len__(self): return self.psi.shape[0]
    def __getitem__(self, i):
        psi = torch.from_numpy(self.psi[i])
        phi = torch.from_numpy(self.phi[i])
        omg = torch.from_numpy(self.omega[i])
        y = torch.from_numpy(self.y[i])
        dv = torch.from_numpy(self.dv[i])
        extra = {
                "psi_mean": self._pm_t, "psi_std": self._ps_t,
                "y_mean": self._ym_t, "y_std": self._ys_t,
        }
        return psi, phi, omg, y, dv, extra

