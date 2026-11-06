import numpy as np, cv2
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from skimage.segmentation import clear_border, find_boundaries
from scipy.ndimage import binary_fill_holes, median_filter
from pipeline import *

def jaccard(a, b):
    u = (a | b).sum(); return float((a & b).sum() / u) if u else 1.0

if __name__ == "__main__":
    rgb, gray, a = carregar()
    g8 = u8(gray); res = {}
    gd = median_filter(g8, 5)
    falha = np.maximum(cv2.morphologyEx(gd, cv2.MORPH_TOPHAT, el(R_TH)), cv2.morphologyEx(gd, cv2.MORPH_BLACKHAT, el(R_TH)))
    bw_falha = falha > threshold_otsu(falha)
    res["falha_cinza"] = {"fracao_branca": float(bw_falha.mean()), "componentes": int(label(clear_border(bw_falha)).max())}
    r = executar(a); fin = r["final"]
    lab = label(fin, connectivity=2); props = regionprops(lab)
    res["pipeline"] = {"var_ruido_a": r["var"], "limiar_multiotsu": r["t"], "raio_tophat": R_TH, "area_min": AREA_MIN,
                       "componentes_final": int(lab.max()),
                       "areas": sorted([int(p.area) for p in props], reverse=True),
                       "objs_apos_binarizar": int(label(r["bw_raw"]).max()), "objs_apos_borda": int(label(r["bw_clear"]).max()),
                       "objs_apos_preencher": int(label(r["bw_fill"]).max())}
    raios = list(range(0, 21))
    area = [int(fin.sum()) if rr == 0 else int(cv2.morphologyEx(fin.astype(np.uint8), cv2.MORPH_OPEN, disco(rr)).sum()) for rr in raios]
    curva = np.array(area) / area[0]; pec = -np.diff(curva)
    esq = skeletonize(fin)
    res["pipeline"].update({"curva_gran": curva.tolist(), "pec": pec.tolist(), "pixels_esqueleto": int(esq.sum()),
                            "raio_dominante": int(np.argmax(pec) + 1)})
    over = rgb.copy(); fr = find_boundaries(lab, mode="outer"); over[fr] = (0, 255, 0)
    over[esq] = (255, 0, 0)
    fig, ax = plt.subplots(2, 4, figsize=(17, 14))
    mostrar(ax[0, 0], rgb, "1) foto original (fundo complexo)"); mostrar(ax[0, 1], a, "canal a* (CIELab)", vmin=-5, vmax=25)
    mostrar(ax[0, 2], r["an"], "2) após RRL 7x7", vmin=-5, vmax=25); mostrar(ax[0, 3], r["th"], f"3) top-hat (disco r={R_TH})", vmax=25)
    mostrar(ax[1, 0], r["bw_raw"], f"4) binarização multi-Otsu (t={r['t']:.1f})"); mostrar(ax[1, 1], r["bw_clear"], "5) remoção de objetos da borda")
    mostrar(ax[1, 2], r["bw_fill"], "6) preenchimento de cavidades"); mostrar(ax[1, 3], over, f"7) resultado: {lab.max()} componentes\n(contorno verde, esqueleto vermelho)")
    plt.tight_layout(); plt.savefig(f"{FIGS}/c_pipeline.png", dpi=DPI); plt.close()
    fig, ax = plt.subplots(1, 3, figsize=(16, 7))
    mostrar(ax[0], falha, "falha: top-hat+bottom-hat no cinza", vmax=120); mostrar(ax[1], bw_falha, f"Otsu: {100*bw_falha.mean():.0f}% de pixels brancos")
    ax[2].plot(raios, curva, "o-"); ax[2].set_title("granulometria da máscara final", fontsize=13); ax[2].set_xlabel("raio do disco"); ax[2].set_ylabel("área restante (fração)")
    plt.tight_layout(); plt.savefig(f"{FIGS}/c_falha_gran.png", dpi=DPI); plt.close()
    var_ord = {}
    for nome, o in (("padrão (RRL→top-hat→bin→borda→preenche→abre)", "padrao"), ("top-hat antes do RRL", "tophat_antes_ruido"),
                    ("preenche antes de remover borda", "preencher_antes_borda"), ("abre antes de borda/preenche", "abrir_antes")):
        rr = executar(a, o)["final"]
        var_ord[nome] = {"componentes": int(label(rr, connectivity=2).max()), "area": int(rr.sum()), "jaccard_vs_padrao": jaccard(rr, fin),
                         "pixels_diferentes": int((rr ^ fin).sum())}
    base = r["bw_fill"]
    for rad in (2, 4):
        af = fechar(abrir(base, rad), rad); fa = abrir(fechar(base, rad), rad)
        var_ord[f"abertura∘fechamento vs fechamento∘abertura (r={rad})"] = {
            "componentes_AF": int(label(af, connectivity=2).max()), "componentes_FA": int(label(fa, connectivity=2).max()),
            "area_AF": int(af.sum()), "area_FA": int(fa.sum()), "pixels_diferentes": int((af ^ fa).sum()), "jaccard": jaccard(af, fa)}
    marrom = [(190, 60), (297, 72), (408, 120), (648, 90), (75, 210), (188, 215), (490, 215), (685, 400), (392, 480), (90, 590), (157, 740), (540, 720)]
    claras = [(110, 300), (255, 545), (492, 525), (480, 640), (303, 690), (290, 890), (208, 960), (745, 665)]
    esc = LADO / 1000
    def achou(p, R=14):
        x, y = int(p[0] * esc), int(p[1] * esc)
        return bool(fin[max(0, y - R):y + R, max(0, x - R):x + R].any())
    cent = [(int(px * esc), int(py * esc)) for px, py in marrom + claras]
    sem_peca = 0
    for k in range(1, lab.max() + 1):
        ys, xs = np.nonzero(lab == k)
        if not any(((xs - cx) ** 2 + (ys - cy) ** 2 <= 18 ** 2).any() for cx, cy in cent): sem_peca += 1
    res["contagem_manual"] = {"marrons": len(marrom), "claras": len(claras), "marrons_detectadas": sum(achou(p) for p in marrom),
                              "claras_detectadas": sum(achou(p) for p in claras), "componentes_sem_peca": sem_peca}
    res["ordem"] = var_ord
    salvar_json(res, "c.json")
    print(json.dumps({k: v for k, v in res.items()}, indent=1, ensure_ascii=False)[:4000])
