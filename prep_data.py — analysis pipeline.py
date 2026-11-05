# prep_data.py — build the analysis assets for the Streamlit app from raw DESI DR9 cutouts.
#
# Inputs : build/cutouts_grz.npy   (N,3,96,96) flux in nanomaggies/pixel
#          build/meta_sel.json     object metadata (ra, dec, type, mags, colors)
# Output : data/desi_dr9_sample.npz  (everything the app needs, offline)
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
torch.set_num_threads(2)
RNG = np.random.default_rng(42)
imgs = np.load("build/cutouts_grz.npy")            # (N,3,96,96) nanomaggies/pixel
meta = json.load(open("build/meta_sel.json"))
N = len(meta)
types = np.array([m["type"] for m in meta])
is_star = types == "PSF"
is_gal = ~is_star
gr = np.array([m["gr"] for m in meta])

# ----------------------------------------------------------------------
# 1) RGB rendering (Lupton-style asinh stretch, LS convention g=B,r=G,z=R)
# ----------------------------------------------------------------------
def band_sigma(x):
    """Background noise of a band: median per-image MAD."""
    mads = np.median(np.abs(x - np.median(x, axis=(1, 2), keepdims=True)), axis=(1, 2))
    return float(np.median(mads))

sig = np.array([band_sigma(imgs[:, b]) for b in range(3)])   # g,r,z noise
soft = 3.0 * sig                                              # asinh softening

def to_rgb(cube):
    """cube (...,3,H,W) nanomaggies -> uint8 RGB (...,H,W,3). g->B, r->G, z->R."""
    q = np.asarray(cube, dtype=np.float64) / soft.reshape((1, 3, 1, 1))  # (...,3,H,W)
    v = np.arcsinh(q)
    # normalise each band by its 99.9th percentile across the sample
    hi = np.percentile(v.reshape(-1, 3), 99.9, axis=0)
    v = np.clip(v / hi.reshape((1, 3, 1, 1)), 0, 1)
    rgb = np.stack([v[..., 2, :, :], v[..., 1, :, :], v[..., 0, :, :]], axis=-1)  # R=z G=r B=g
    return (rgb * 255).astype(np.uint8)

rgb = to_rgb(imgs)

# ----------------------------------------------------------------------
# 2) Structural / data-quality statistics per object and per band
# ----------------------------------------------------------------------
# surface brightness proxy: mean flux in central 5x5 px (1.3 arcsec) per band
core = imgs[:, :, 45:51, 45:51].mean(axis=(2, 3))
# per-band background noise of THIS cutout (MAD)
noise = np.median(np.abs(imgs - np.median(imgs, axis=(2, 3), keepdims=True)), axis=(2, 3))
snr = core / np.maximum(noise, 1e-9)

# ----------------------------------------------------------------------
# 3) Small CNN: star (PSF) vs galaxy (extended), trained with augmentation
# ----------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, padding=2)   # feature maps shown in app
        self.conv2 = nn.Conv2d(8, 16, 5, padding=2)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 12 * 12, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return self.head(x)

def augment(x):
    """Random flips + 90-degree rotations (sky images have no preferred orientation)."""
    b = x.shape[0]
    for i in range(b):
        k = int(torch.randint(0, 4, (1,)))
        x[i] = torch.rot90(x[i], k, dims=(1, 2))
        if torch.rand(1) < 0.5:
            x[i] = torch.flip(x[i], dims=(1,))
        if torch.rand(1) < 0.5:
            x[i] = torch.flip(x[i], dims=(2,))
    return x

# normalise inputs: asinh-stretched, single channel scale shared
X = torch.tensor(np.arcsinh(imgs / soft.reshape(1, 3, 1, 1)), dtype=torch.float32)
y = torch.tensor(is_star.astype(np.int64))

model = Net()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
losses = []
for epoch in range(250):
    xb = augment(X.clone())
    logits = model(xb)
    loss = F.cross_entropy(logits, y)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(float(loss))
model.eval()
with torch.no_grad():
    preds = model(X).argmax(1).numpy()
    probs = torch.softmax(model(X), 1).numpy()
    acc = (preds == y.numpy()).mean()
print(f"train accuracy {acc:.2f} (loss {losses[-1]:.3f})")

# leave-one-out CV for an honest generalisation number
correct = 0
loocv_preds = []
for i in range(N):
    idx = np.arange(N) != i
    m = Net(); o = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(120):
        xb = augment(X[idx].clone()); lb = y[idx]
        o.zero_grad(); F.cross_entropy(m(xb), lb).backward(); o.step()
    m.eval()
    with torch.no_grad():
        pred = m(X[i:i+1]).argmax(1).item()
    loocv_preds.append(pred)
    correct += int(pred == int(y[i]))
    if (i + 1) % 10 == 0:
        print(f"  loocv {i+1}/{N}")
print(f"leave-one-out accuracy {correct}/{N} = {correct/N:.2f}")
loocv_acc = correct / N

# ----------------------------------------------------------------------
# 4) Saliency: |d logit(galaxy) / d input| (vanilla gradients), smoothed
# ----------------------------------------------------------------------
Xg = X.clone().requires_grad_(True)
logits = model(Xg)
score = logits[:, 1].sum()          # galaxy logit
score.backward()
sal = Xg.grad.abs().sum(1).numpy()  # (N,96,96)
# light smoothing + normalise to [0,1] per object
def smooth(a, k=5):
    ker = np.ones((k, k)) / (k * k)
    from numpy.fft import rfft2, irfft2
    return np.real(irfft2(rfft2(a) * rfft2(ker, a.shape), a.shape))
sal = np.stack([smooth(s) for s in sal])
sal = sal / np.maximum(sal.max(axis=(1, 2), keepdims=True), 1e-12)

# feature maps of conv1 for every object (8 maps, 96x96)
with torch.no_grad():
    fm = F.relu(model.conv1(X)).numpy()          # (N,8,96,96)
fm = fm / np.maximum(fm.max(axis=(2, 3), keepdims=True), 1e-9)

# ----------------------------------------------------------------------
# 5) Package everything for the app
# ----------------------------------------------------------------------
np.savez_compressed(
    "data/desi_dr9_sample.npz",
    imgs=imgs.astype(np.float32), rgb=rgb,
    types=types, is_star=is_star, gr=gr,
    ra=np.array([m["ra"] for m in meta]), dec=np.array([m["dec"] for m in meta]),
    g=np.array([m["g"] for m in meta]), r=np.array([m["r"] for m in meta]),
    z=np.array([m["z"] for m in meta]), objid=np.array([m["objid"] for m in meta]),
    core=core, noise=noise, snr=snr, sig=sig,
    saliency=sal.astype(np.float32), featmaps=fm.astype(np.float32),
    losses=np.array(losses), preds=preds, probs=probs,
    loocv_preds=np.array(loocv_preds), loocv_acc=loocv_acc,
    train_acc=float(acc),
)
json.dump(dict(train_acc=float(acc), loocv_acc=float(loocv_acc), final_loss=losses[-1],
               n_gal=int((~is_star).sum()), n_star=int(is_star.sum())),
          open("build/metrics.json", "w"), indent=1)
torch.save(model.state_dict(), "build/cnn_stargaraxy.pt")
print("saved data/desi_dr9_sample.npz + build/metrics.json")
