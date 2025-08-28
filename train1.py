import os, time, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import UNet
from dataset import SegDataset
from losses import dice_loss

cfg = dict(
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    train_img='data/train/images', train_msk='data/train/masks',
    val_img='data/val/images',     val_msk='data/val/masks',
    epochs=15, batch=4, lr=1e-4, num_classes=6, ckpt_out='best.pt'
)

def main():
    tr = SegDataset(cfg['train_img'], cfg['train_msk'], aug=True)
    va = SegDataset(cfg['val_img'],   cfg['val_msk'],   aug=False)
    tl = DataLoader(tr, batch_size=cfg['batch'], shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=cfg['batch'])

    model = UNet(in_ch=3, num_classes=cfg['num_classes']).to(cfg['device'])
    opt   = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-5)
    sch   = CosineAnnealingLR(opt, T_max=cfg['epochs'], eta_min=1e-6)
    scaler= GradScaler()

    best = 1e9
    for ep in range(1, cfg['epochs']+1):
        model.train(); t0=time.time(); tr_loss=0.0
        for x,y in tl:
            x,y = x.to(cfg['device']), y.to(cfg['device'])
            opt.zero_grad()
            with autocast():
                logits = model(x)
                loss = 0.5*F.cross_entropy(logits, y) + 0.5*dice_loss(logits, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tr_loss += loss.item()
        sch.step()

        model.eval(); va_loss=0.0
        with torch.no_grad():
            for x,y in vl:
                x,y = x.to(cfg['device']), y.to(cfg['device'])
                with autocast():
                    logits = model(x)
                    va_loss += (0.5*F.cross_entropy(logits, y) + 0.5*dice_loss(logits, y)).item()

        tr_loss /= len(tl); va_loss /= len(vl)
        print(f"Ep {ep:02d} | train {tr_loss:.4f} | val {va_loss:.4f} | {time.time()-t0:.1f}s")
        if va_loss < best:
            best = va_loss
            torch.save(model.state_dict(), cfg['ckpt_out'])
            print(f"saved → {cfg['ckpt_out']}")

if __name__ == "__main__":
    main()