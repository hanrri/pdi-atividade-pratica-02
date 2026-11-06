import numpy as np
from matplotlib.patches import Circle
from common import *
from PIL import Image

fig, ax = plt.subplots(1, 5, figsize=(17, 6.5))
for a_, (nome, t) in zip(ax, [("20261105_142335.jpg", "a) ruído periódico\n(monitor)"), ("20261105_134410.jpg", "b) alta luz\n(ISO 100, flash)"),
                              ("20261105_134340.jpg", "b) baixa luz\n(ISO 1600, 1/20 s)"), (None, "c) sal e pimenta\n(d = 0,25, sintético)"),
                              ("20261105_133734.jpg", "d) fundo complexo\n(tabuleiro)")]):
    img = np.asarray(Image.open(f"{RES}/sal_pimenta_d25.png")) if nome is None else ler_rgb(nome, 600)
    a_.imshow(img, cmap="gray" if nome is None else None); a_.set_title(t, fontsize=13); a_.axis("off")
plt.tight_layout(); plt.savefig(f"{FIGS}/fotos.png", dpi=DPI); plt.close()

d = np.load(f"{RES}/a1_imgs.npz"); a1 = json.load(open(f"{RES}/a1.json"))
cr = (slice(380, 640), slice(190, 560))
fig, ax = plt.subplots(2, 4, figsize=(17, 8.5))
for i, pref in enumerate(("real", "ctrl")):
    base = d[f"{pref}_ruidosa"]
    imgs = [("clara" , d["clara"]), ("ruidosa", base), ("média 7x7", d[f"{pref}_media7"]), ("RRL 7x7", d[f"{pref}_RRL7"])]
    tit = {"real": "foto escura (ISO 1600) normalizada", "ctrl": "clara + ruído gaussiano σ=25"}[pref]
    for j, (t, im) in enumerate(imgs):
        mostrar(ax[i, j], im[cr], (t if j else "referência (foto clara)") + (f"\n{tit}" if j == 1 else ""), vmin=0, vmax=255)
plt.tight_layout(); plt.savefig(f"{FIGS}/a1_comparacao.png", dpi=DPI); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ks = [3, 5, 7, 11, 15]
for m, c in (("media", "C0"), ("RRL", "C3")):
    ax[0].plot(ks, [r["psnr"] for r in a1["ctrl"]["tabela"] if r["metodo"] == m], "o-", color=c, label=m)
    ax[1].plot(ks, [r["ssim"] for r in a1["ctrl"]["tabela"] if r["metodo"] == m], "o-", color=c, label=m)
ax[0].axhline(a1["ctrl"]["psnr_ruidosa"], color="gray", ls="--", label="ruidosa"); ax[1].axhline(a1["ctrl"]["ssim_ruidosa"], color="gray", ls="--", label="ruidosa")
ax[0].set_xlabel("janela (px)"); ax[0].set_ylabel("PSNR (dB)"); ax[1].set_xlabel("janela (px)"); ax[1].set_ylabel("SSIM")
ax[0].legend(); ax[0].set_title("ruído gaussiano controlado", fontsize=13); ax[1].set_title("ruído gaussiano controlado", fontsize=13)
plt.tight_layout(); plt.savefig(f"{FIGS}/a1_curvas.png", dpi=DPI); plt.close()

d2 = np.load(f"{RES}/a2_imgs.npz")
cr2 = (slice(330, 600), slice(150, 620))
fig, ax = plt.subplots(3, 4, figsize=(17, 11))
for i, dens in enumerate((0.1, 0.25, 0.4)):
    for j, (k, t) in enumerate((("ruid", "ruidosa"), ("mediana_fixa_7x7_", "mediana fixa 7x7"), ("adaptativa_Smax=7_", "adaptativa Smax=7"), ("ref", "referência"))):
        im = d2["ref"] if k == "ref" else d2[f"{k}{dens}"] if k != "ruid" else d2[f"ruid{dens}"]
        mostrar(ax[i, j], im[cr2], f"{t}  (d={dens})" if k != "ref" else "referência", vmin=0, vmax=255)
plt.tight_layout(); plt.savefig(f"{FIGS}/a2_comparacao.png", dpi=DPI); plt.close()

d3 = np.load(f"{RES}/a3a4_imgs.npz"); r3 = json.load(open(f"{RES}/a3a4.json"))
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
mostrar(ax[0], d3["g"], "foto do monitor (cinza)", vmin=0, vmax=255)
ax[1].imshow(d3["S"], cmap="gray"); ax[1].set_title("espectro de Fourier (log(1+|F|))", fontsize=13); ax[1].axis("off")
M, N = d3["S"].shape
for (pu, pv) in r3["picos"]:
    for s in (1, -1):
        ax[1].add_patch(Circle((N // 2 + s * pu, M // 2 + s * pv), 20, fill=False, color="r", lw=1.2))
plt.tight_layout(); plt.savefig(f"{FIGS}/a3_espectro.png", dpi=DPI); plt.close()
sl = (slice(330, 630), slice(40, 420))
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
for a_, n in zip(ax.ravel(), ["g", "BR_ideal", "BR_Butterworth", "BR_Gaussiano", "Notch_simples", "Notch_ótimo"]):
    mostrar(a_, d3[n][sl], {"g": "original"}.get(n, n.replace("_", " ")), vmin=0, vmax=255)
plt.tight_layout(); plt.savefig(f"{FIGS}/a3_filtros.png", dpi=DPI); plt.close()
fig, ax = plt.subplots(2, 4, figsize=(17, 9))
cr3 = (slice(80, 330), slice(120, 520))
mostrar(ax[0, 0], d3["eta"], "padrão de ruído η(x,y) extraído\n(passa-notch gaussiano)", vmin=-30, vmax=30)
mostrar(ax[0, 1], d3["w"], "fator w(x,y) (janela 41x41)", vmin=-2, vmax=4, cmap="coolwarm")
mostrar(ax[0, 2], d3["Notch_simples"], "notch simples", vmin=0, vmax=255); mostrar(ax[0, 3], d3["Notch_ótimo"], "notch ótimo", vmin=0, vmax=255)
for j, nome in enumerate(("estacionario", "amplitude variavel")):
    ruid, simp, otm = (d3[f"ctrl_{nome}_{i}"] for i in range(3))
    c = r3["controlado"][nome]
    mostrar(ax[1, 2 * j], ruid[cr3], f"{nome}: ruidosa\nPSNR={c['ruidosa'][0]:.1f}", vmin=0, vmax=255)
    mostrar(ax[1, 2 * j + 1], otm[cr3], f"notch ótimo\nPSNR={c['notch otimo'][0]:.1f}  (simples {c['notch simples'][0]:.1f})", vmin=0, vmax=255)
plt.tight_layout(); plt.savefig(f"{FIGS}/a4_otimo.png", dpi=DPI); plt.close()
print("ok")
