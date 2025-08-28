import torch, torch.nn as nn, torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(c_in, c_out)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.up   = nn.ConvTranspose2d(c_in, c_in//2, 2, stride=2)
        self.conv = DoubleConv(c_in, c_out)

    def forward(self, x, skip):
        x = self.up(x)
        dy, dx = x.size(2) - skip.size(2), x.size(3) - skip.size(3)
        if dy > 0 or dx > 0:  # przycinanie środkiem
            x = x[:, :, dy//2 : x.size(2)-(dy-dy//2), dx//2 : x.size(3)-(dx-dx//2)]
        elif dy < 0 or dx < 0:  # pad, gdy za mały
            x = F.pad(x, [-dx//2, -(dx-(-dx//2)), -dy//2, -(dy-(-dy//2))])
        return self.conv(torch.cat([skip, x], dim=1))

class UNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=6, base=64):
        super().__init__()
        self.e1 = DoubleConv(in_ch, base)
        self.d1 = Down(base, base*2)
        self.d2 = Down(base*2, base*4)
        self.d3 = Down(base*4, base*8)
        self.b  = DoubleConv(base*8, base*16)
        self.u3 = Up(base*16, base*8)
        self.u2 = Up(base*8,  base*4)
        self.u1 = Up(base*4,  base*2)
        self.u0 = Up(base*2,  base)
        self.out = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        s1 = self.e1(x); s2 = self.d1(s1); s3 = self.d2(s2); s4 = self.d3(s3)
        x  = self.b(s4)
        x  = self.u3(x, s4); x = self.u2(x, s3); x = self.u1(x, s2); x = self.u0(x, s1)
        return self.out(x)