import numpy as np, cv2
from scipy import ndimage as ndi
from skimage.morphology import (skeletonize, thin, convex_hull_object, reconstruction, disk)
from skimage.measure import label
from skimage.segmentation import clear_border
from pipeline import *

def pontas(sk):
    viz = ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - sk
    return sk & (viz == 1)

def podar(sk, n):
    s = sk.copy()
    for _ in range(n):
        s &= ~pontas(s)
    return s

def dilat_geod(F, G, k):
    cruz = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    D = F.copy()
    for _ in range(k):
        D = cv2.dilate(D.astype(np.uint8), cruz).astype(bool) & G
    return D

if __name__ == "__main__":
    rgb, gray, a = carregar()
    r = executar(a); mask = r["final"]; raw = r["bw_raw"]
    g8 = u8(gray); res = {}

    ses = {"quadrado 5x5": np.ones((5, 5), np.uint8), "disco r=3": el(3), "linha 1x15": np.ones((1, 15), np.uint8)}
    ops = {"erosão": cv2.MORPH_ERODE, "dilatação": cv2.MORPH_DILATE, "abertura": cv2.MORPH_OPEN, "fechamento": cv2.MORPH_CLOSE}
    tab = []
    fig, ax = plt.subplots(3, 5, figsize=(17, 13))
    for i, (sn, se) in enumerate(ses.items()):
        mostrar(ax[i, 0], mask, f"binária original\nSE: {sn}")
        for j, (on, op) in enumerate(ops.items()):
            o = cv2.morphologyEx(mask.astype(np.uint8), op, se).astype(bool)
            mostrar(ax[i, j + 1], o, f"{on} (área={o.sum()}, objs={label(o).max()})")
            tab.append({"SE": sn, "op": on, "area": int(o.sum()), "objetos": int(label(o).max())})
    plt.tight_layout(); plt.savefig(f"{FIGS}/b1_binaria.png", dpi=70); plt.close()
    fig, ax = plt.subplots(3, 5, figsize=(17, 20))
    for i, (sn, se) in enumerate(ses.items()):
        mostrar(ax[i, 0], g8, f"cinza original\nSE: {sn}")
        for j, (on, op) in enumerate(ops.items()):
            mostrar(ax[i, j + 1], cv2.morphologyEx(g8, op, se), on)
    plt.tight_layout(); plt.savefig(f"{FIGS}/b1_cinza.png", dpi=70); plt.close()
    res["b1"] = tab

    kern = np.array([[-1, -1, 0], [-1, 1, 1], [0, 1, 1]], np.int8)
    hm = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_HITMISS, kern).astype(bool)
    fronteira = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    preenchida = ndi.binary_fill_holes(raw)
    buracos = preenchida & ~raw
    rot, ncomp = ndi.label(mask, structure=np.ones((3, 3)))
    hull = convex_hull_object(mask)
    fino = thin(mask); esq = skeletonize(mask); poda = podar(esq, 5)
    res["b2"] = {"cantos_hit_or_miss": int(hm.sum()), "pixels_fronteira": int(fronteira.sum()),
                 "pixels_buracos_preenchidos": int(buracos.sum()), "componentes": int(ncomp),
                 "area_mascara": int(mask.sum()), "area_fecho_convexo": int(hull.sum()),
                 "pixels_afinamento": int(fino.sum()), "pixels_esqueleto": int(esq.sum()), "pixels_esqueleto_podado": int(poda.sum()),
                 "pontas_esqueleto": int(pontas(esq).sum()), "pontas_apos_poda": int(pontas(poda).sum())}
    fig, ax = plt.subplots(2, 4, figsize=(17, 14))
    mostrar(ax[0, 0], mask, "máscara binária")
    mostrar(ax[0, 1], cv2.dilate(hm.astype(np.uint8), np.ones((5, 5), np.uint8)), f"hit-or-miss (cantos): {hm.sum()}")
    mostrar(ax[0, 2], fronteira, "extração de fronteira")
    mostrar(ax[0, 3], np.dstack([raw, preenchida, preenchida & ~raw]).astype(float), "preenchimento (verde/azul = buracos)")
    mostrar(ax[1, 0], rot, f"componentes conexos: {ncomp}", cmap="nipy_spectral")
    mostrar(ax[1, 1], hull | mask, "fecho convexo (por objeto)")
    mostrar(ax[1, 2], fino, "afinamento")
    mostrar(ax[1, 3], np.dstack([esq, poda, np.zeros_like(esq)]).astype(float), "esqueleto (vermelho) e podado (amarelo)")
    plt.tight_layout(); plt.savefig(f"{FIGS}/b2_binarios.png", dpi=70); plt.close()

    marcador = cv2.erode(mask.astype(np.uint8), el(3)).astype(bool)
    d1, d5, d15 = (dilat_geod(marcador, mask, k) for k in (1, 5, 15))
    rec = reconstruction(marcador.astype(np.uint8), mask.astype(np.uint8), method="dilation").astype(bool)
    k = 0; D = marcador.copy()
    while True:
        D2 = dilat_geod(D, mask, 1); k += 1
        if (D2 == D).all(): break
        D = D2
    abert_comum = abrir(mask, 3)
    borda = np.zeros_like(raw); borda[0, :] = borda[-1, :] = borda[:, 0] = borda[:, -1] = True
    marc_b = raw & borda
    tocam = reconstruction(marc_b.astype(np.uint8), raw.astype(np.uint8), method="dilation").astype(bool)
    sem_borda = raw & ~tocam
    ab_g = cv2.morphologyEx(g8, cv2.MORPH_OPEN, el(8))
    ab_rec = reconstruction(ab_g.astype(float), g8.astype(float), method="dilation")
    res["b3"] = {"iteracoes_ate_convergir": k, "area_marcador": int(marcador.sum()), "area_reconstruida": int(rec.sum()),
                 "area_mascara": int(mask.sum()), "area_abertura_comum": int(abert_comum.sum()),
                 "objetos_borda_removidos_px": int(tocam.sum()), "objetos_antes_borda": int(label(raw).max()),
                 "objetos_depois_borda": int(label(sem_borda).max())}
    fig, ax = plt.subplots(2, 4, figsize=(17, 14))
    mostrar(ax[0, 0], marcador, "marcador F (erosão r=3)"); mostrar(ax[0, 1], d1, "dilatação geodésica k=1")
    mostrar(ax[0, 2], d5, "k=5"); mostrar(ax[0, 3], rec, f"reconstrução (convergiu em k={k})")
    mostrar(ax[1, 0], abert_comum, "abertura comum (disco 3)"); mostrar(ax[1, 1], tocam, "objetos que tocam a borda")
    mostrar(ax[1, 2], ab_g, "cinza: abertura comum r=8"); mostrar(ax[1, 3], ab_rec, "cinza: abertura por reconstrução")
    plt.tight_layout(); plt.savefig(f"{FIGS}/b3_reconstrucao.png", dpi=70); plt.close()

    grad = cv2.morphologyEx(g8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    th_g = cv2.morphologyEx(g8, cv2.MORPH_TOPHAT, el(15)); bh_g = cv2.morphologyEx(g8, cv2.MORPH_BLACKHAT, el(15))
    asf = g8.copy()
    for rr in (1, 2, 3):
        asf = cv2.morphologyEx(cv2.morphologyEx(asf, cv2.MORPH_OPEN, el(rr)), cv2.MORPH_CLOSE, el(rr))
    raios = list(range(0, 26))
    area_b = [int(cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, disco(rr)).sum()) if rr > 0 else int(mask.sum()) for rr in raios]
    soma_g = [float(cv2.morphologyEx(g8, cv2.MORPH_OPEN, disco(rr)).sum()) if rr > 0 else float(g8.sum()) for rr in raios]
    curva_b = np.array(area_b) / area_b[0]; pec_b = -np.diff(curva_b)
    curva_g = np.array(soma_g) / soma_g[0]; pec_g = -np.diff(curva_g)
    res["b4"] = {"raios": raios, "curva_binaria": curva_b.tolist(), "pecstrum_binario": pec_b.tolist(),
                 "curva_cinza": curva_g.tolist(), "pecstrum_cinza": pec_g.tolist(),
                 "raio_pico_pecstrum_binario": int(np.argmax(pec_b) + 1), "raio_pico_pecstrum_cinza": int(np.argmax(pec_g) + 1)}
    fig, ax = plt.subplots(2, 4, figsize=(17, 14))
    mostrar(ax[0, 0], grad, "gradiente morfológico (3x3)"); mostrar(ax[0, 1], th_g, "top-hat (disco r=15)")
    mostrar(ax[0, 2], bh_g, "bottom-hat (disco r=15)"); mostrar(ax[0, 3], asf, "filtro sequencial alternado (r=1..3)")
    ax[1, 0].plot(raios, curva_b, "o-"); ax[1, 0].set_title("granulometria binária (área restante)", fontsize=9); ax[1, 0].set_xlabel("raio do disco")
    ax[1, 1].bar(raios[1:], pec_b); ax[1, 1].set_title("espectro de padrões (binário)", fontsize=9); ax[1, 1].set_xlabel("raio")
    ax[1, 2].plot(raios, curva_g, "o-"); ax[1, 2].set_title("granulometria em cinza", fontsize=9); ax[1, 2].set_xlabel("raio do disco")
    ax[1, 3].bar(raios[1:], pec_g); ax[1, 3].set_title("espectro de padrões (cinza)", fontsize=9); ax[1, 3].set_xlabel("raio")
    plt.tight_layout(); plt.savefig(f"{FIGS}/b4_cinza_granulometria.png", dpi=70); plt.close()
    salvar_json(res, "b.json")
    for k_, v_ in res.items():
        if k_ != "b1": print(k_, {a_: b_ for a_, b_ in v_.items() if not isinstance(b_, list)})
