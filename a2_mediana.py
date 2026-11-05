import numpy as np
from scipy.ndimage import median_filter, minimum_filter, maximum_filter
from common import *
from PIL import Image

def mediana_adaptativa(g, smax=7):
    g = g.astype(np.float64)
    saida = g.copy()
    pendente = np.ones(g.shape, bool)
    for k in range(3, smax + 1, 2):
        zmed = median_filter(g, size=k, mode="reflect")
        zmin = minimum_filter(g, size=k, mode="reflect")
        zmax = maximum_filter(g, size=k, mode="reflect")
        nivelA_ok = (zmed > zmin) & (zmed < zmax)
        resolve = pendente & nivelA_ok
        central_ok = (g > zmin) & (g < zmax)
        saida[resolve & central_ok] = g[resolve & central_ok]
        saida[resolve & ~central_ok] = zmed[resolve & ~central_ok]
        pendente &= ~nivelA_ok
        ultima = zmed
    saida[pendente] = ultima[pendente]
    return saida

if __name__ == "__main__":
    ref = np.asarray(Image.open(f"{RES}/referencia_limpa.png"), dtype=np.float64)
    res = {"tabela": []}
    imgs = {"ref": ref}
    for d in (0.10, 0.25, 0.40):
        ruid = np.asarray(Image.open(f"{RES}/sal_pimenta_d{int(d*100):02d}.png"), dtype=np.float64)
        imgs[f"ruid{d}"] = ruid
        linha = {"densidade": d, "psnr_ruidosa": psnr(ref, ruid), "ssim_ruidosa": ssim(ref, ruid)}
        metodos = {"mediana 3x3": lambda x: median_filter(x, size=3, mode="reflect"),
                   "mediana fixa 7x7": lambda x: median_filter(x, size=7, mode="reflect"),
                   "adaptativa Smax=7": lambda x: mediana_adaptativa(x, 7),
                   "adaptativa Smax=5": lambda x: mediana_adaptativa(x, 5)}
        for nome, fun in metodos.items():
            ts = []
            for _ in range(3):
                t0 = time.perf_counter(); f = fun(ruid); ts.append(time.perf_counter() - t0)
            t = float(np.median(ts))
            imgs[f"{nome}_{d}"] = f
            res["tabela"].append({"densidade": d, "metodo": nome, "psnr": psnr(ref, f), "ssim": ssim(ref, f), "tempo_s": t})
            print(f"d={d} {nome:20s} PSNR {psnr(ref,f):6.2f}  SSIM {ssim(ref,f):.3f}  t={t*1000:7.1f} ms")
        res.setdefault("ruidosas", []).append(linha)
    salvar_json(res, "a2.json")
    np.savez_compressed(f"{RES}/a2_imgs.npz", **{k.replace(" ", "_"): v for k, v in imgs.items()})
