import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter, maximum_filter
from common import *
from PIL import Image

def grade(M, N):
    v, u = np.mgrid[:M, :N]
    return u - N // 2, v - M // 2

def espectro(g):
    return np.fft.fftshift(np.fft.fft2(g))

def volta(F):
    return np.real(np.fft.ifft2(np.fft.ifftshift(F)))

def br_ideal(M, N, R0, W):
    u, v = grade(M, N); D = np.hypot(u, v)
    return np.where(np.abs(D - R0) <= W / 2, 0.0, 1.0)

def br_butter(M, N, R0, W, n=2):
    u, v = grade(M, N); D = np.maximum(np.hypot(u, v), 1e-6)
    return 1.0 / (1.0 + ((D * W) / (np.abs(D ** 2 - R0 ** 2) + 1e-6)) ** (2 * n))

def br_gauss(M, N, R0, W):
    u, v = grade(M, N); D = np.maximum(np.hypot(u, v), 1e-6)
    return 1.0 - np.exp(-0.5 * ((D ** 2 - R0 ** 2) / (D * W)) ** 2)

def notch_ideal(M, N, picos, r):
    u, v = grade(M, N); H = np.ones((M, N))
    for (pu, pv) in picos:
        for s in (1, -1):
            H[np.hypot(u - s * pu, v - s * pv) <= r] = 0.0
    return H

def notch_gauss_passa(M, N, picos, sigma):
    u, v = grade(M, N); H = np.zeros((M, N))
    for (pu, pv) in picos:
        for s in (1, -1):
            H = np.maximum(H, np.exp(-((u - s * pu) ** 2 + (v - s * pv) ** 2) / (2 * sigma ** 2)))
    return H

def notch_otimo(g, picos, sigma=10, janela=41):
    M, N = g.shape
    eta = volta(espectro(g) * notch_gauss_passa(M, N, picos, sigma))
    mg, me = uniform_filter(g, janela), uniform_filter(eta, janela)
    mge, mee = uniform_filter(g * eta, janela), uniform_filter(eta * eta, janela)
    w = (mge - mg * me) / np.maximum(mee - me * me, 1e-9)
    return g - w * eta, eta, w

def energia_lobos(F, picos, r):
    u, v = grade(*F.shape); mask = np.zeros(F.shape, bool)
    for (pu, pv) in picos:
        for s in (1, -1):
            mask |= np.hypot(u - s * pu, v - s * pv) <= r
    return float((np.abs(F[mask]) ** 2).sum() / (np.abs(F) ** 2).sum())

if __name__ == "__main__":
    g = ler_cinza("20261105_142335.jpg", 1024)
    M, N = g.shape
    F = espectro(g); S = np.log1p(np.abs(F))
    picos = [(114, -6), (241, -8)]
    R_N = 20
    R0 = float(np.hypot(*picos[0])); W = 45.0
    res = {"picos": picos, "R0": R0, "W": W, "R_notch": R_N}
    filtros = {"BR ideal": br_ideal(M, N, R0, W), "BR Butterworth": br_butter(M, N, R0, W),
               "BR Gaussiano": br_gauss(M, N, R0, W), "Notch simples": notch_ideal(M, N, picos, R_N)}
    sai = {}
    res["energia_lobos_original"] = energia_lobos(F, picos, R_N)
    for nome, H in filtros.items():
        f = volta(F * H); sai[nome] = f
        res.setdefault("filtros", {})[nome] = {"energia_lobos": energia_lobos(F * H, picos, R_N),
                                                "dif_media_abs": float(np.abs(f - g).mean())}
    f_ot, eta, w = notch_otimo(g, picos)
    sai["Notch ótimo"] = f_ot
    res["filtros"]["Notch ótimo"] = {"energia_lobos": energia_lobos(espectro(f_ot), picos, R_N),
                                      "dif_media_abs": float(np.abs(f_ot - g).mean()),
                                      "w_medio": float(w.mean()), "w_min": float(w.min()), "w_max": float(w.max())}
    for k, v in res["filtros"].items(): print(k, v)
    print("energia original", res["energia_lobos_original"])
    ref = np.asarray(Image.open(f"{RES}/referencia_limpa.png"), dtype=np.float64)
    Mr, Nr = ref.shape; y, x = np.mgrid[:Mr, :Nr]
    fu, fv = 60, 40
    base = np.sin(2 * np.pi * (fu * x / Nr + fv * y / Mr)) + 0.6 * np.sin(2 * np.pi * (-30 * x / Nr + 90 * y / Mr))
    ctrl = {}
    for nome, amp in (("estacionario", np.ones_like(ref)), ("amplitude variavel", 0.4 + 1.2 * x / Nr)):
        ruid = np.clip(ref + 25 * amp * base, 0, 255)
        Fr = espectro(ruid)
        pk = [(fu, fv), (-30, 90)]
        Hn = notch_ideal(Mr, Nr, pk, 6)
        f_simples = volta(Fr * Hn)
        f_ot, _, _ = notch_otimo(ruid, pk, sigma=4, janela=61)
        ctrl[nome] = {"ruidosa": (psnr(ref, ruid), ssim(ref, ruid)),
                      "notch simples": (psnr(ref, f_simples), ssim(ref, f_simples)),
                      "notch otimo": (psnr(ref, f_ot), ssim(ref, f_ot))}
        sai["ctrl_" + nome] = (ruid, f_simples, f_ot)
        print(nome, ctrl[nome])
    res["controlado"] = ctrl
    salvar_json(res, "a3a4.json")
    np.savez_compressed(f"{RES}/a3a4_imgs.npz", g=g, S=S, w=w, eta=eta,
                        **{k.replace(" ", "_"): v for k, v in sai.items() if not k.startswith("ctrl_")},
                        **{k + "_" + str(i): a for k, t in sai.items() if k.startswith("ctrl_") for i, a in enumerate(t)},
                        ref=ref)
