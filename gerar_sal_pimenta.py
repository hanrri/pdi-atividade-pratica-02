import sys
import numpy as np
from PIL import Image, ImageOps

def sal_pimenta(img, densidade, rng):
    out = img.copy()
    mask = rng.random(img.shape)
    out[mask < densidade / 2] = 0
    out[(mask >= densidade / 2) & (mask < densidade)] = 255
    return out

if __name__ == "__main__":
    caminho = sys.argv[1]
    lado = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    im = ImageOps.exif_transpose(Image.open(caminho)).convert("L")
    im.thumbnail((lado, lado), Image.LANCZOS)
    ref = np.asarray(im, dtype=np.uint8)
    Image.fromarray(ref).save("referencia_limpa.png")
    rng = np.random.default_rng(42)
    for d in (0.10, 0.25, 0.40):
        ruidosa = sal_pimenta(ref, d, rng)
        Image.fromarray(ruidosa).save(f"sal_pimenta_d{int(d*100):02d}.png")
        print(f"d={d}: fração real corrompida = {(ruidosa != ref).mean():.3f}")
