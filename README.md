# DESI Legacy Surveys DR9 — Image Browser (Week 8)

Streamlit image browser built on real **DESI Legacy Imaging Surveys DR9** data
(DECaLS, brick `1501p022`, COSMOS field): 60 sky cutouts, 96×96 px at 0.262″/px,
each a 3-channel image in the **g, r, z** broad bands (calibrated nanomaggies/pixel) —
48 galaxies (REX/EXP/DEV/SER) spanning the full g−r colour range plus 12 stars (PSF).

## Run it

```bash
pip install streamlit numpy pandas matplotlib pillow scikit-learn
streamlit run app_desi_browser.py
```

Everything the app needs is bundled in `data/desi_dr9_sample.npz` — no network required.

## What each tab shows (rubric mapping)

### 1. Montage 
A 6×10 grid of RGB composites (g→blue, r→green, z→red, asinh stretch) **sorted by
g−r colour from blue to red**, with tile borders colour-coded the same way. It reveals
the galaxy colour bimodality — the *blue cloud* of star-forming disks vs the *red
sequence* of old quiescent ellipticals — plus stars (point-like, white) and real
survey artifacts: a few tiles contain chance-aligned neighbours, and the brightest
stars show saturated cores and bleed halos. Sortable by colour, magnitude or type.

### 2. Structure: channels & intensity 
- Per-band pixel-intensity distributions over all 60 cutouts (blank-sky peak + object tail)
- Background noise per band: **g ≈ 0.0019, r ≈ 0.0036, z ≈ 0.0090 nanomaggies/px** —
  the z band is ≈5× noisier than g (a real data-quality finding: z is the shallowest
  band and the near-IR sky is brighter, so z-based colours are the noisiest measurements)
- Core SNR per band, stars vs galaxies (log scale)
- Per-object radial intensity profiles in each band (PSF-like vs extended)

### 3. Saliency & feature maps 
A small CNN (3 conv layers, ~50k params, flip/rotation augmentation) separates stars
from galaxies: **leave-one-out CV accuracy 55/60 = 92%** (train 100%; the 5 misses are
compact REX galaxies barely resolved beyond the seeing disc and faint stars).
Attribution = **gradient × input** w.r.t. the galaxy logit:

- Stars: compact blob inside the seeing disc — 50% of attribution within **r ≈ 1.7″**
- Galaxies: attribution follows the light (disk major axis on inclined disks) —
  50% only by **r ≈ 5.3″**
- ⇒ the model learned the *morphological* cue (compact vs extended), not colour or brightness
- First-layer feature maps (8 filters) act as edge/blob detectors; responses are
  object-driven, near-zero on blank sky

## Files

| File | Purpose |
|---|---|
| `app_desi_browser.py` | the Streamlit app (4 tabs) |
| `data/desi_dr9_sample.npz` | cutouts + metadata + stats + saliency + feature maps |

## Data provenance

- Tractor catalog: [DESI Legacy Surveys DR9, brick 1501p022](https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/south/tractor/150/tractor-1501p022.fits)
- Cutouts: [legacysurvey.org viewer cutout service](https://www.legacysurvey.org/viewer) (`cutout.fits`, layer `ls-dr9`, 0.262″/px)
- Survey description: [Dey et al. 2019, AJ 157, 168](https://ui.adsabs.harvard.edu/abs/2019AJ....157..168D/abstract)
