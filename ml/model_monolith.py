import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.SiLU(),
        )
    def forward(self, x): return self.net(x)

class UNet3D_ScalarOut(nn.Module):
    def __init__(self, in_ch, base=32):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.down1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(base, base*2))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(base*2, base*4))
        self.up1 = nn.ConvTranspose3d(base*4, base*2, 2, stride=2)
        self.conv1 = DoubleConv(base*4, base*2)
        self.up2 = nn.ConvTranspose3d(base*2, base, 2, stride=2)
        self.conv2 = DoubleConv(base*2, base)
        self.outc = nn.Conv3d(base, 1, 1)  # 1ch のスカラー場 W(r)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        u1 = self.up1(x3)
        c1 = self.conv1(torch.cat([u1, x2], dim=1))
        u2 = self.up2(c1)
        c2 = self.conv2(torch.cat([u2, x1], dim=1))
        return self.outc(c2)

class MonolithFieldNet(nn.Module):
    """
    φ(r), ω -> W(r) を推定し、vψ_pred = W(r) * ψ を返す“モノリスML”。
    """
    def __init__(self, in_ch_phi=5, base=32):
        super().__init__()
        self.omega_embed = nn.Sequential(
            nn.Linear(1, base), nn.SiLU(), nn.Linear(base, 1)  # 1ch に射影
        )
        self.unet = UNet3D_ScalarOut(in_ch=in_ch_phi + 1, base=base)

    def forward(self, psi, phi, omega):
        """
        psi: (B,2,P,P,P), phi: (B,5,P,P,P), omega: (B,1)
        return: (B,2,P,P,P)  ・・・ vψ_pred
        """
        B, _, P, _, _ = psi.shape
        omg = self.omega_embed(omega).to(psi.dtype).view(B,1,1,1,1).expand(B,1,P,P,P)
        x = torch.cat([phi, omg], dim=1)                 # (B,6,P,P,P)
        W = self.unet(x)                                 # (B,1,P,P,P)
        vpsi_pred = W * psi                              # 各点で実/虚に同じ W を掛ける
        return vpsi_pred

