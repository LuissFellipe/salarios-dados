"""Estilo dos gráficos: a mesma paleta validada das páginas do portfólio
(categóricas 1–3 passam no validador de daltonismo em todos os pares),
linhas finas, grade recessiva, texto em tinta neutra."""

import matplotlib as mpl
import matplotlib.pyplot as plt

AZUL, LARANJA, VERDE = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA_2, MUDO, GRADE, EIXO, FUNDO = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
NIVEIS = {"Júnior": AZUL, "Pleno": LARANJA, "Sênior": VERDE}


def estilo():
    mpl.rcParams.update({
        "figure.facecolor": FUNDO, "axes.facecolor": FUNDO, "savefig.facecolor": FUNDO,
        "figure.dpi": 110, "font.family": "sans-serif", "font.size": 10,
        "text.color": TINTA, "axes.labelcolor": TINTA_2, "xtick.color": MUDO, "ytick.color": TINTA_2,
        "axes.edgecolor": EIXO, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRADE, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "lines.linewidth": 2,
        # "R$ 7 mil → R$ 10 mil" tem dois cifrões: sem isto, vira fórmula
        "text.parse_math": False,
    })


def reais(v, _=None):
    """R$ 15,5 mil. Uma casa decimal: arredondar para o milhar transforma
    15.490 × 15.520 em "15 mil × 16 mil", uma diferença que não existe."""
    if v >= 1000:
        return "R$ " + f"{v / 1000:.1f}".replace(".", ",") + " mil"
    return f"R$ {v:.0f}"


def reais_eixo(v, _=None):
    """Para marcas de eixo, que caem em números redondos."""
    return f"R$ {v / 1000:.0f} mil" if v >= 1000 else f"R$ {v:.0f}"


def eixo_reais(ax, eixo="x"):
    fmt = mpl.ticker.FuncFormatter(reais_eixo)
    (ax.xaxis if eixo == "x" else ax.yaxis).set_major_formatter(fmt)
