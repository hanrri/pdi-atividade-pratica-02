import numpy as np, cv2
from skimage.color import rgb2lab
from skimage.filters import threshold_multiotsu
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
from common import *
from a1_rrl import rrl, achar_area_lisa

LADO = 512
R_TH = 12
AREA_MIN = 40

def el(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

from skimage.morphology import disk as _disk

def disco(r):
    return _disk(r).astype(np.uint8)

def carregar():
    rgb = ler_rgb("20261105_133734.jpg", LADO)
    gray = ler_cinza("20261105_133734.jpg", LADO)
    a = rgb2lab(rgb)[..., 1]
    return rgb, gray, a

def reduz_ruido(a):
    pos, _ = achar_area_lisa(a, 5, a.shape[0] - 7, 5, a.shape[1] - 5)
    var = float(np.var(a[pos[0]:pos[0] + 32, pos[1]:pos[1] + 32]))
    return rrl(a, var, 7), var, pos

def tophat(img, r=R_TH):
    return cv2.morphologyEx(img.astype(np.float32), cv2.MORPH_TOPHAT, el(r)).astype(np.float64)

def binariza(th):
    t = threshold_multiotsu(th, classes=3)[0]
    return th > t, float(t)

def abrir(bw, r=2):
    return cv2.morphologyEx(bw.astype(np.uint8), cv2.MORPH_OPEN, el(r)).astype(bool)

def fechar(bw, r=2):
    return cv2.morphologyEx(bw.astype(np.uint8), cv2.MORPH_CLOSE, el(r)).astype(bool)

def executar(a, ordem="padrao"):
    out = {}
    if ordem == "tophat_antes_ruido":
        th = tophat(a); an, var, pos = reduz_ruido(th)
        th = np.maximum(an, 0)
    else:
        an, var, pos = reduz_ruido(a); th = tophat(an)
    out.update(an=an, var=var, th=th)
    bw_raw, t = binariza(th); out.update(bw_raw=bw_raw, t=t)
    if ordem == "preencher_antes_borda":
        bw_c = clear_border(binary_fill_holes(bw_raw)); bw_f = bw_c
    elif ordem == "abrir_antes":
        bw_c = clear_border(abrir(bw_raw)); bw_f = binary_fill_holes(bw_c)
    else:
        bw_c = clear_border(bw_raw); bw_f = binary_fill_holes(bw_c)
    out.update(bw_clear=bw_c, bw_fill=bw_f)
    fin = bw_f if ordem == "abrir_antes" else abrir(bw_f)
    fin = remove_small_objects(fin, max_size=AREA_MIN - 1)
    out["final"] = fin
    return out
