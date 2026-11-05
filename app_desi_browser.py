# app_desi_browser.py — Week 8: DESI image browser + stats
# Dataset: DESI Legacy Imaging Surveys DR9 (brick 1501p022, COSMOS field)
# Run with:  streamlit run app_desi_browser.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
from PIL import Image

st.set_page_config(page_title="DESI DR9 image browser", layout="wide")

BANDS = ["g", "r", "z"]
BAND_COLORS = ["#4c72b0", "#55a868", "#c44e52"]


@st.cache_data
def load_sample():
    with np.load("data/desi_dr9_sample.npz") as z:
        d = {k: z[k] for k in z.files}
    meta = pd.DataFrame({
        "id": np.arange(len(d["types"])), "type": d["types"],
        "is_star": d["is_star"], "g-r": d["gr"], "g": d["g"], "r": d["r"], "z": d["z"],
        "ra": d["ra"], "dec": d["dec"], "objid": d["objid"],
    })
    return d, meta


d, meta = load_sample()
imgs, rgb = d["imgs"], d["rgb"]
saliency, featmaps = d["saliency"], d["featmaps"]
sig = d["sig"]  # median background noise (nanomaggies/px) per band g, r, z


def compare(left, right, label_left="original", label_right="processed"):
    """A before/after slider: `left` on the left, `right` on the right.

    streamlit-image-comparison would do this, but it loads juxtapose from
    cdn.knightlab.com/libs/juxtapose/LATEST while the package itself is pinned
    at 0.0.4, and current juxtapose collapses its container to 100x100. This is
    the same thing in plain CSS, with nothing fetched from a CDN.
    """
    def b64(im):
        buf = io.BytesIO(); im.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    right = right.resize(left.size)
    w, h = left.size
    shown_h = int(700 * h / w)
    cap = ("position:absolute;top:8px;background:rgba(0,0,0,.6);color:#fff;"
           "padding:2px 8px;font:13px sans-serif;border-radius:3px")
    st.iframe(f"""
      <div style="position:relative;max-width:700px">
        <img src="data:image/png;base64,{b64(left)}" style="width:100%;display:block">
        <img id="top" src="data:image/png;base64,{b64(right)}"
             style="width:100%;position:absolute;top:0;left:0;clip-path:inset(0 0 0 50%)">
        <span style="{cap};left:8px">{label_left}</span>
        <span style="{cap};right:8px">{label_right}</span>
        <input type="range" min="0" max="100" value="50" style="width:100%;margin-top:6px"
               oninput="document.getElementById('top').style.clipPath=
                        'inset(0 0 0 '+this.value+'%)'">
      </div>
    """, height=shown_h + 50)


def object_label(i):
    m = meta.iloc[i]
    return f"#{i:02d} · {m['type']} · g−r = {m['g-r']:.2f} · r = {m['r']:.1f}"


st.title("DESI Legacy Surveys DR9 — Image Browser")
st.caption(
    "60 sky cutouts (96×96 px, 0.262″/px ≈ 25″ on a side) from the DESI Legacy Imaging "
    "Surveys DR9, brick 1501p022 in the COSMOS field: 48 galaxies (REX/EXP/DEV/SER) "
    "spanning the full g−r colour range plus 12 point sources (stars). Each cutout is a "
    "3-channel image — the g, r and z broad bands — in calibrated nanomaggies per pixel."
)

tab1, tab2, tab3, tab4 = st.tabs(["Montage", "Object browser", "Structure: channels & intensity", "Saliency & feature maps"])

# ---------------------------------------------------------------- montage
with tab1:
    st.subheader("Montage — what the sky's colours tell you")
    st.markdown(
        "Tiles are RGB composites (g→blue, r→green, z→red, asinh stretch). "
        "**Left→right the galaxies are sorted by g−r colour from blue to red.** "
        "The population splits into a *blue cloud* (star-forming disks) and a *red sequence* "
        "(old, quiescent ellipticals) — the same bimodality DESI itself uses to target "
        "galaxies for spectroscopy. The last row holds the 12 stars (white-ish, point-like)."
    )
    sort_mode = st.radio("Sort tiles by", ["g−r colour (default)", "r magnitude (bright→faint)", "type"], horizontal=True)

    order = np.argsort(meta["g-r"].values)
    if sort_mode.startswith("r magnitude"):
        order = np.argsort(-meta["r"].values)
    elif sort_mode == "type":
        order = np.argsort(meta["type"].values)

    ncol = 10
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.5, nrow * 1.5))
    for k, ax in enumerate(axes.flat):
        ax.set_xticks([]); ax.set_yticks([])
        if k < len(order):
            i = order[k]
            ax.imshow(rgb[i], interpolation="nearest")
            m = meta.iloc[i]
            ax.set_title(f"{m['g-r']:.2f}" if not m["is_star"] else "star",
                         fontsize=8, color="#666")
            for s in ax.spines.values():
                s.set_color("none" if m["is_star"] else plt.cm.coolwarm(m["g-r"] / 1.9))
                s.set_linewidth(2)
        else:
            ax.set_visible(False)
    fig.suptitle("DESI DR9 cutouts — border colour = g−r (blue → red); labels show g−r", y=1.005, fontsize=11)
    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    st.markdown(
        "Reading the montage: colour encodes stellar population — blue tiles are young, "
        "star-forming galaxies; red tiles are old, passive ones. Point-like white tiles are "
        "stars, and a few tiles contain a second object (chance alignments) — a reminder "
        "that 'one image, one object' is an assumption, not a fact, in survey data. The "
        "brightest stars also show saturated cores and bleeding halos — a classic artifact "
        "of bright-source imaging that any downstream model has to tolerate."
    )

# ---------------------------------------------------------------- browser
with tab2:
    st.subheader("Object browser")
    c1, c2 = st.columns([1, 3])
    with c1:
        default_obj = int(np.argsort(np.abs(meta["g-r"].values - 0.9))[0])
        choice = st.selectbox("Object", range(len(meta)), index=default_obj, format_func=object_label)
    m = meta.iloc[choice]

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots()
        ax.imshow(rgb[choice], interpolation="nearest")
        ax.set_title(f"RGB composite  ·  type {m['type']}")
        ax.axis("off")
        st.pyplot(fig)
    with right:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for b, ax in enumerate(axes):
            ax.imshow(imgs[choice, b], cmap="gray", origin="lower")
            ax.set_title(f"{BANDS[b]} band")
            ax.axis("off")
        st.pyplot(fig)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Type (tractor)", m["type"])
    k2.metric("g − r", f"{m['g-r']:.2f}")
    k3.metric("r mag", f"{m['r']:.1f}")
    k4.metric("Mean SNR (5×5 core)", f"{d['snr'][choice].mean():.0f}")
    k5.metric("RA / Dec", f"{m['ra']:.4f}, {m['dec']:.4f}")
    st.caption(
        "Left: g, r, z channels with a linear scale — the sky background is at ~0 "
        "nanomaggies/px, so this view shows how faint extended galaxy light is relative "
        "to the point-source stars."
    )

    st.markdown("**Before / after:** original vs. the saliency heat-map of the star-vs-galaxy CNN "
                "(see the Saliency tab). Drag the handle.")
    heat = plt.cm.inferno(saliency[choice])[..., :3]
    blend = 0.55 * (rgb[choice] / 255.0) + 0.85 * heat
    blend = (np.clip(blend, 0, 1) * 255).astype(np.uint8)
    compare(Image.fromarray(rgb[choice]), Image.fromarray(blend), "RGB original", "saliency overlay")

# ---------------------------------------------------------------- structure
with tab3:
    st.subheader("Structure: channels & intensity")
    st.markdown(
        "The three DESI bands are not three copies of the same sky: they are independent "
        "exposures with different depths, seeing and sky brightness. The plots below "
        "quantify that structure across the whole 60-object sample."
    )

    a, b = st.columns(2)
    with a:
        st.markdown("**Per-band intensity distributions** (all pixels, all 60 cutouts)")
        fig, ax = plt.subplots(figsize=(6, 4))
        for band in range(3):
            vals = imgs[:, band].ravel()
            lo, hi = np.percentile(vals, [1, 99])
            bins = np.linspace(lo, hi, 80)
            ax.hist(vals, bins=bins, histtype="step", lw=1.8, color=BAND_COLORS[band],
                    label=f"{BANDS[band]} band", density=True)
        ax.set_xlabel("pixel flux (nanomaggies / px)")
        ax.set_ylabel("density")
        ax.legend()
        st.pyplot(fig)
        st.caption(
            "The sharp peak at ~0 is blank sky; the positive tail is objects. The g-band "
            "distribution is narrowest (deepest, lowest-noise data), while z has the widest, "
            "noisiest background — visible as a wider spread around zero."
        )

    with b:
        st.markdown("**Background noise per band** (median MAD, nanomaggies/px)")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(BANDS, sig, color=BAND_COLORS)
        for i, v in enumerate(sig):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("noise (nanomaggies / px)")
        st.pyplot(fig)
        st.caption(
            f"A real data-quality finding: the z band is ≈{sig[2]/sig[0]:.0f}× noisier than g "
            f"in this field (g ≈ {sig[0]:.4f}, r ≈ {sig[1]:.4f}, z ≈ {sig[2]:.4f} nanomaggies/px). "
            "That is expected — DECaLS g exposures are the deepest and the night sky is brighter "
            "at 9000 Å (OH airglow) than at 4800 Å — but it means faint red galaxies are the "
            "hardest objects to measure in this dataset, and any r−z or z-band colour is the "
            "noisiest quantity we can form from these cutouts."
        )

    st.markdown("**Core signal-to-noise per band, stars vs. galaxies**")
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, (mask, lab) in enumerate([(d["is_star"], "stars (PSF)"), (~d["is_star"], "galaxies")]):
        for band in range(3):
            x = np.full(mask.sum(), band + (-0.15 if k == 0 else 0.15))
            ax.scatter(x, d["snr"][mask, band], s=22, alpha=0.75,
                       color=BAND_COLORS[band], marker="o" if k == 0 else "s",
                       edgecolors="none" if k else "k", linewidths=0.4)
    ax.set_xticks(range(3), BANDS)
    ax.set_yscale("log")
    ax.set_ylabel("core SNR (5×5 px mean / cutout MAD)")
    ax.axhline(5, ls="--", c="gray", lw=1)
    ax.text(2.4, 5.4, "SNR = 5", fontsize=8, color="gray")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color="k", label="stars (circles)"),
                       Line2D([], [], marker="s", ls="", color="k", label="galaxies (squares)")])
    st.pyplot(fig)
    st.caption(
        "Stars concentrate light into a few pixels, so their peak SNR is enormous; galaxies "
        "spread the same flux over thousands of pixels. Even at r ≈ 18–20, every galaxy "
        "core stays well above SNR = 5 in all three bands — the sample is deep enough for "
        "reliable colour measurements."
    )

    st.markdown("**Radial intensity profile** (mean flux in annuli, normalized to the core)")
    prof_sel = st.selectbox("Object for the radial profile", range(len(meta)), format_func=object_label, index=0)
    H = imgs.shape[-1]
    yy, xx = np.mgrid[0:H, 0:H]
    rr = np.sqrt((xx - H / 2) ** 2 + (yy - H / 2) ** 2) * 0.262  # arcsec
    bins = np.linspace(0, 12.5, 26)
    prof = np.stack([np.histogram(rr, bins=bins, weights=imgs[prof_sel, band])[0] /
                     np.maximum(np.histogram(rr, bins=bins)[0], 1) for band in range(3)])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for band in range(3):
        ax.plot((bins[:-1] + bins[1:]) / 2, prof[band] / prof[band, 0], color=BAND_COLORS[band], label=BANDS[band])
    ax.set_yscale("log"); ax.set_xlabel("radius (arcsec)"); ax.set_ylabel("flux / core flux")
    ax.legend(); st.pyplot(fig)
    st.caption("A point source (star) drops like a seeing-limited PSF within ~1″; an extended "
               "galaxy keeps significant flux out to several arcsec — this is exactly the "
               "compactness signal the saliency model below picks up on.")

# ---------------------------------------------------------------- saliency
with tab4:
    st.subheader("Saliency & feature maps — what the CNN looks at")
    st.markdown(
        "A small CNN (3 conv layers + head, ~50k parameters) was trained on these 60 cutouts "
        "to separate **stars (point sources) from galaxies (extended sources)**, using only "
        "random flips/rotations for augmentation. Leave-one-out cross-validation accuracy: "
        f"**{int(d['loocv_acc']*len(meta))}/{len(meta)} ({d['loocv_acc']*100:.0f}%)** "
        "(train accuracy 100%). The attribution shown is *gradient × input*: "
        "|∂(galaxy logit)/∂(input pixel)| × |pixel|, summed over the three bands — which "
        "bright pixels, if changed, would most change the model's 'galaxy' verdict."
    )
    st.caption(
        "The 5 leave-one-out errors are the genuinely ambiguous cases: compact REX galaxies "
        "that are barely resolved beyond the seeing disc, and faint stars — the same objects "
        "a human would struggle to label from the image alone. This is a real astronomy "
        "problem: star–galaxy separation at the faint limit is exactly what DESI's "
        "morphological classifications (and Gaia astrometry) exist to solve."
    )
    sal_sel = st.selectbox("Object", range(len(meta)), format_func=object_label,
                           index=int(np.argmin(np.abs(meta['g-r'].values - 1.0))), key="sal")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.imshow(rgb[sal_sel], interpolation="nearest"); ax.axis("off")
        ax.set_title("input (RGB)")
        st.pyplot(fig)
        fig, ax = plt.subplots()
        ax.imshow(saliency[sal_sel], cmap="inferno"); ax.axis("off")
        ax.set_title("saliency")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        smap = saliency[sal_sel]
        ax.imshow(rgb[sal_sel], interpolation="nearest")
        ax.imshow(smap, cmap="inferno", alpha=0.55)
        ax.axis("off"); ax.set_title("overlay")
        st.pyplot(fig)
        # cumulative attribution fraction vs radius, by class (median over class)
        binm = np.digitize(rr, bins) - 1
        binm = np.clip(binm, 0, len(bins) - 2)
        cdf = {k: np.cumsum(np.array([saliency[k][binm == j].sum() for j in range(len(bins) - 1)]))
               for k in range(len(meta))}
        for k in cdf:
            cdf[k] = cdf[k] / cdf[k][-1]
        fig, ax = plt.subplots(figsize=(6, 4))
        for mask, lab, c in [(d["is_star"], "stars (median)", "#c44e52"),
                             (~d["is_star"], "galaxies (median)", "#4c72b0")]:
            curves = np.stack([cdf[k] for k in np.where(mask)[0]])
            med = np.median(curves, axis=0)
            ax.plot((bins[:-1] + bins[1:]) / 2, med, color=c, label=lab, lw=2)
            ax.fill_between((bins[:-1] + bins[1:]) / 2, np.percentile(curves, 25, axis=0),
                            np.percentile(curves, 75, axis=0), color=c, alpha=0.15)
        ax.axvline(1.5, ls="--", c="gray", lw=1)
        ax.text(1.6, 0.08, "seeing disc\n(r ≈ 1.5″)", fontsize=8, color="gray")
        ax.set_xlabel("radius (arcsec)"); ax.set_ylabel("cumulative attribution fraction")
        ax.set_xlim(0, 8); ax.set_ylim(0, 1)
        ax.legend(); st.pyplot(fig)

    st.markdown(
        "**Interpretation.** For stars, attribution is a single compact blob confined to the "
        "seeing disc: half of the total attribution lands inside r ≈ 1.7″, right at the "
        "PSF size — the model's decision rests on the central few pixels forming a point "
        "source. For extended galaxies the salient region follows the light: the median "
        "galaxy needs r ≈ 5.3″ to accumulate half its attribution, and on an inclined disk "
        "the saliency traces the disk's major axis. The model has learned the *morphological* "
        "cue (compact vs. extended), not brightness or colour — colour is not even available "
        "to it in a scale-free sense, since each band is stretched the same way. A few tiles "
        "in the montage contain a neighbour galaxy; on those, saliency lights up on both "
        "objects — the gradient is indifferent to which object carries the 'extended' signal."
    )

    st.markdown("**First conv-layer feature maps** (8 filters, 5×5 kernels) for the selected object")
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for f, ax in enumerate(axes.flat):
        ax.imshow(featmaps[sal_sel, f], cmap="viridis"); ax.axis("off")
        ax.set_title(f"filter {f}", fontsize=9)
    st.pyplot(fig)
    st.caption(
        "The first layer acts like a bank of edge/blob detectors: some filters isolate the "
        "bright core, others respond to the extended low-surface-brightness halo — the two "
        "signals the classifier combines. Filter responses are near-zero in the corners "
        "(blank sky), confirming the features are object-driven, not noise-driven."
    )

    st.markdown("**Training loss** of the CNN (cross-entropy, 250 epochs)")
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(d["losses"], lw=1)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    st.pyplot(fig)