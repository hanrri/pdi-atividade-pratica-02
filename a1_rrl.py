import numpy as np
from scipy.ndimage import uniform_filter
from common import *

def media_local(g, k):
    return uniform_filter(g, size=k, mode="reflect")

def var_local(g, k):
    m = media_local(g, k)
    return np.maximum(uniform_filter(g * g, size=k, mode="reflect") - m * m, 0)

def rrl(g, var_ruido, k):
    mL, vL = media_local(g, k), var_local(g, k)
    razao = np.where(vL > 0, var_ruido / np.maximum(vL, 1e-12), 1.0)
    razao = np.minimum(razao, 1.0)
    return g - razao * (g - mL)

def achar_area_lisa(g, y0, y1, x0, x1, t=32):
    melhor, pos = 1e18, None
    yy, xx = np.mgrid[0:t, 0:t]
    A = np.c_[xx.ravel(), yy.ravel(), np.ones(t * t)]
    for y in range(y0, y1 - t, 4):
        for x in range(x0, x1 - t, 4):
            p = g[y:y + t, x:x + t].ravel()
            c, *_ = np.linalg.lstsq(A, p, rcond=None)
            v = np.var(p - A @ c)
            if v < melhor:
                melhor, pos = v, (y, x)
    return pos, melhor

def estima_var(g, pos, t=32):
    y, x = pos
    return achar_area_lisa.__globals__["np"].var(g[y:y + t, x:x + t])

if __name__ == "__main__":
    clara = ler_cinza("20261105_134410.jpg")
    escura = ler_cinza("20261105_134340.jpg")
    a = clara.std() / escura.std()
    b = clara.mean() - a * escura.mean()
    esc_n = np.clip(a * escura + b, 0, 255)
    res = {"ganho_a": a, "offset_b": b}

    pos, v = achar_area_lisa(esc_n, 5, 115, 5, 760)
    var_est = np.var(esc_n[pos[0]:pos[0] + 32, pos[1]:pos[1] + 32])
    print("area lisa em", pos, "var(detrend)=%.1f  var(patch)=%.1f  sigma=%.2f" % (v, var_est, var_est ** .5))
    res["real"] = {"area_lisa_yx": pos, "var_ruido": var_est, "sigma": var_est ** .5,
                   "psnr_ruidosa": psnr(clara, esc_n), "ssim_ruidosa": ssim(clara, esc_n)}
    pos2, _ = achar_area_lisa(esc_n, 130, 500, 5, 760)
    res["real"]["area2_yx"] = pos2
    def std_area(img, p):
        return float(np.std(img[p[0]:p[0] + 32, p[1]:p[1] + 32]))
    def nitidez(img):
        from scipy.ndimage import sobel
        gm = np.hypot(sobel(img, 0), sobel(img, 1))
        return float(np.percentile(gm, 99))
    res["real"]["std_area_ruidosa"] = std_area(esc_n, pos)
    res["real"]["nitidez_ruidosa"] = nitidez(esc_n)
    res["real"]["tabela"] = []
    imgs = {"ruidosa": esc_n}
    for k in (3, 5, 7, 11, 15):
        for nome, f in (("media", media_local(esc_n, k)), ("RRL", rrl(esc_n, var_est, k))):
            res["real"]["tabela"].append({"metodo": nome, "janela": k, "psnr": psnr(clara, f),
                "ssim": ssim(clara, f), "std_area_lisa": std_area(f, pos), "nitidez": nitidez(f)})
            imgs[f"{nome}{k}"] = f
    rng = np.random.default_rng(1)
    sig = 25.0
    ruid = np.clip(clara + rng.normal(0, sig, clara.shape), 0, 255)
    pos_c, _ = achar_area_lisa(clara, 5, 115, 5, 760)
    var_c = np.var(ruid[pos_c[0]:pos_c[0] + 32, pos_c[1]:pos_c[1] + 32])
    res["ctrl"] = {"sigma_real": sig, "area_lisa_yx": pos_c, "var_estimada": var_c, "sigma_estimado": var_c ** .5,
                   "psnr_ruidosa": psnr(clara, ruid), "ssim_ruidosa": ssim(clara, ruid), "tabela": []}
    imgs_c = {"ruidosa": ruid}
    for k in (3, 5, 7, 11, 15):
        for nome, f in (("media", media_local(ruid, k)), ("RRL", rrl(ruid, var_c, k))):
            res["ctrl"]["tabela"].append({"metodo": nome, "janela": k, "psnr": psnr(clara, f), "ssim": ssim(clara, f)})
            imgs_c[f"{nome}{k}"] = f
    salvar_json(res, "a1.json")
    np.savez_compressed(f"{RES}/a1_imgs.npz", clara=clara, **{"real_" + k: v for k, v in imgs.items()},
                        **{"ctrl_" + k: v for k, v in imgs_c.items()})
    for ex in ("real", "ctrl"):
        print(ex, "ruidosa PSNR %.2f SSIM %.3f" % (res[ex]["psnr_ruidosa"], res[ex]["ssim_ruidosa"]))
        for r in res[ex]["tabela"]:
            print("  ", r["metodo"], r["janela"], "PSNR %.2f SSIM %.3f" % (r["psnr"], r["ssim"]),
                  ("std %.2f nit %.1f" % (r["std_area_lisa"], r["nitidez"])) if "nitidez" in r else "")
    print("estimativa ctrl: sigma=%.2f (real 25)" % (var_c ** .5), "nit ruidosa real", res["real"]["nitidez_ruidosa"])
