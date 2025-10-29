import torch
import torch.nn as nn
import torch.nn.functional as F

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

class UNet3D(nn.Module):
    def __init__(self, in_ch, out_ch, base=32):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.down1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(base, base*2))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(base*2, base*4))
        self.up1 = nn.ConvTranspose3d(base*4, base*2, 2, stride=2)
        self.conv1 = DoubleConv(base*4, base*2)
        self.up2 = nn.ConvTranspose3d(base*2, base, 2, stride=2)
        self.conv2 = DoubleConv(base*2, base)
        self.outc = nn.Conv3d(base, out_ch, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        u1 = self.up1(x3)
        c1 = self.conv1(torch.cat([u1, x2], dim=1))
        u2 = self.up2(c1)
        c2 = self.conv2(torch.cat([u2, x1], dim=1))
        return self.outc(c2)

class SRHFOperatorUNet(nn.Module):
    def __init__(self, in_ch_phi=5, in_ch_psi=2, base=32):
        super().__init__()
        self.omega_embed = nn.Sequential(nn.Linear(1, base), nn.SiLU(), nn.Linear(base, 1))
        self.unet = UNet3D(in_ch=in_ch_phi+in_ch_psi+1, out_ch=2, base=base)

    def forward(self, psi, phi, omega):
        # psi: (B,2,P,P,P), phi: (B,C,P,P,P), omega: (B,1)
        B, _, P, _, _ = psi.shape
        omg = self.omega_embed(omega).view(B, -1, 1, 1, 1).expand(B, 1, P, P, P)
        x = torch.cat([psi, phi, omg], dim=1)
        out = self.unet(x)  # (B,2,P,P,P) → (v_x^SR ψ)(r)
        return out

