# Atividade Prática 02 — Processamento Digital de Imagens

Universidade Federal do Piauí (UFPI) · Ciência da Computação · Período 2026.2
Aluno: Hanrri Kleber Rocha da Silva

Código da atividade prática 02: restauração de imagens (filtros espaciais e de
frequência) e processamento morfológico aplicados a fotografias feitas pelo
próprio aluno. A metodologia e a análise dos resultados estão no relatório em PDF.

## O que tem aqui

| Arquivo | Parte do trabalho |
|---|---|
| `a1_rrl.py` | A1: filtro adaptativo de redução de ruído local (RRL) contra o filtro de média |
| `a2_mediana.py` | A2: mediana adaptativa contra mediana fixa 7×7 (ruído sal e pimenta) |
| `a3_a4_frequencia.py` | A3 e A4 (bônus): espectro de Fourier, rejeita-banda, notch simples e notch ótimo |
| `b_morfologia.py` | B: erosão, dilatação, abertura, fechamento, reconstrução, granulometria etc. |
| `c_pipeline.py` | C: pipeline de segmentação do tabuleiro e estudo de ordem das operações |
| `figuras_a.py` | Gera as figuras das partes 0 e A |
| `gerar_sal_pimenta.py` | Gera as imagens com ruído sal e pimenta |
| `common.py`, `pipeline.py` | Funções usadas pelos outros scripts |

```
.
├── fotos/    fotografias originais (.jpg)
├── res/      imagens com ruído sal e pimenta e resultados gerados
├── figs/     figuras geradas (criada ao rodar)
└── *.py
```

## Requisitos

- Python 3.11 ou mais novo (testado com 3.13)
- As bibliotecas do `requirements.txt`: NumPy, SciPy, scikit-image (0.26 ou mais novo),
  OpenCV, Matplotlib e Pillow

## Instalação

Clone o repositório e entre na pasta:

```bash
git clone https://github.com/hanrri/pdi-atividade-pratica-02.git
cd pdi-atividade-pratica-02
```

Também dá para usar o botão *Code → Download ZIP* do GitHub, extrair e entrar na pasta extraída.

Crie e ative um ambiente virtual.

Linux ou macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Quando o ambiente estiver ativo, o terminal mostra `(.venv)` no começo da linha.
Em um terminal novo, é preciso ativar de novo antes de rodar os scripts.

## Como rodar

Com o terminal dentro da pasta do projeto e o ambiente ativo, rode os scripts nesta ordem:

```bash
python a1_rrl.py
python a2_mediana.py
python a3_a4_frequencia.py
python b_morfologia.py
python c_pipeline.py
python figuras_a.py
```

Ou tudo de uma vez (para no primeiro erro):

```bash
python a1_rrl.py && python a2_mediana.py && python a3_a4_frequencia.py && python b_morfologia.py && python c_pipeline.py && python figuras_a.py
```

O `figuras_a.py` deve ser o último, porque usa arquivos gerados pelos três primeiros.
No Linux, se o comando `python` não existir, use `python3`. O conjunto leva
cerca de um a dois minutos.

## O que é gerado

- No terminal: PSNR, SSIM, tempos de execução e contagens.
- `res/`: arquivos `.json` com todos os números e três `.npz` com imagens intermediárias
  (cerca de 190 MB no total, podem ser apagados depois).
- `figs/`: as 15 figuras usadas no relatório.

Os tempos da mediana (A2) variam conforme a máquina. Os demais valores devem ser
iguais aos do relatório, pois as sementes aleatórias são fixas.

## Ruído sal e pimenta

As imagens com ruído já estão em `res/` (densidades 0,10, 0,25 e 0,40, a partir de
`fotos/20261105_134410.jpg`). Para gerá-las de novo (os arquivos são gravados em `res/`):

```bash
python gerar_sal_pimenta.py fotos/20261105_134410.jpg 1024
```

## Problemas comuns

- `ModuleNotFoundError`: o ambiente virtual não está ativo ou faltou o `pip install -r requirements.txt`.
- `FileNotFoundError` em `fotos/...`: a foto não está na pasta `fotos/` ou o nome foi alterado.
- `FileNotFoundError` em `res/...`: os scripts foram rodados fora de ordem.
- Erro com `max_size` em `remove_small_objects`: scikit-image desatualizado; rode `pip install -U scikit-image`.
