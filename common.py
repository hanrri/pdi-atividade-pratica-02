import numpy as np, time, json, os, textwrap
from PIL import Image, ImageOps
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12})
DPI = 110

BASE = os.path.dirname(os.path.abspath(__file__))
FOTOS = os.path.join(BASE, "fotos")
FIGS = os.path.join(BASE, "figs")
RES = os.path.join(BASE, "res")
os.makedirs(FIGS, exist_ok=True); os.makedirs(RES, exist_ok=True)

def ler_cinza(nome, lado=1024):
    im = ImageOps.exif_transpose(Image.open(f"{FOTOS}/{nome}")).convert("L")
    if lado:
        im.thumbnail((lado, lado), Image.LANCZOS)
    return np.asarray(im, dtype=np.float64)

def ler_rgb(nome, lado=1024):
    im = ImageOps.exif_transpose(Image.open(f"{FOTOS}/{nome}")).convert("RGB")
    if lado:
        im.thumbnail((lado, lado), Image.LANCZOS)
    return np.asarray(im)

def psnr(ref, img):
    return peak_signal_noise_ratio(np.clip(ref, 0, 255), np.clip(img, 0, 255), data_range=255)

def ssim(ref, img):
    return structural_similarity(np.clip(ref, 0, 255), np.clip(img, 0, 255), data_range=255)

def u8(a):
    return np.clip(np.round(a), 0, 255).astype(np.uint8)

def salvar_json(obj, nome):
    with open(f"{RES}/{nome}", "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def mostrar(ax, img, titulo, cmap="gray", **kw):
    ax.imshow(img, cmap=cmap, **kw)
    ax.set_title("\n".join(textwrap.fill(l, 28) for l in titulo.split("\n")), fontsize=13)
    ax.axis("off")
