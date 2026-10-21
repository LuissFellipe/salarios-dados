"""O que a harmonização promete, conferido contra os CSVs brutos.

Cada teste compara com o bruto, não com um número decorado: se a Data
Hackers republicar uma edição, o teste continua valendo.
"""

import numpy as np
import pandas as pd
import pytest

from salarios import harmonizar as h
from salarios.edicoes import BRUTOS, EDICOES


@pytest.fixture(scope="module")
def df():
    return h.harmonizar()


@pytest.fixture(scope="module")
def brutos():
    return {a: pd.read_csv(BRUTOS / e["arquivo"], usecols=list(e["colunas"].values()), low_memory=False)
            for a, e in EDICOES.items()}


def col(brutos, ano, campo):
    return brutos[ano][EDICOES[ano]["colunas"][campo]]


def test_nenhuma_resposta_some(df, brutos):
    for ano in EDICOES:
        assert (df["ano"] == ano).sum() == len(brutos[ano])


def test_valor_desconhecido_interrompe():
    with pytest.raises(h.ValorDesconhecido):
        h.mapear(pd.Series(["Masculino", "Não-binário"]), h.GENERO, "genero", 2099)
    with pytest.raises(h.ValorDesconhecido):
        h.faixa_em_reais("de R$ 50.001/mês a R$ 60.000/mês ")
    with pytest.raises(h.ValorDesconhecido):
        h.faixa_em_reais("de R$ 9.000/mês a R$ 8.000/mês")


@pytest.mark.parametrize("campo,mapa,saida", [
    ("genero", h.GENERO, "genero"), ("ensino", h.ENSINO, "ensino"), ("situacao", h.SITUACAO, "situacao"),
    ("porte", h.PORTE, "porte"), ("modelo", h.MODELO, "modelo"), ("mudar", h.MUDAR, "mudar"),
])
def test_so_vira_vazio_o_que_foi_mandado_virar(df, brutos, campo, mapa, saida):
    """Vazio na saída = vazio no bruto + valores que o mapa manda esvaziar."""
    for ano, ed in EDICOES.items():
        if campo not in ed["colunas"]:
            continue
        b = col(brutos, ano, campo)
        esvaziados = b.map(lambda v: pd.notna(v) and mapa.get(v) == h.VAZIO).sum()
        assert df.loc[df["ano"] == ano, saida].isna().sum() == b.isna().sum() + esvaziados, (ano, campo)


def test_faixas(df, brutos):
    validas = df.dropna(subset=["sal_min"])
    assert (validas["sal_min"] < validas["sal_max"]).all()
    assert (validas["sal_min"] >= 0).all()
    # topo aberto: 2019 em R$ 25.001, de 2021 em diante em R$ 40.001
    topo = validas[np.isinf(validas["sal_max"])].groupby("ano")["sal_min"].unique()
    assert list(topo[2019]) == [25001.0]
    assert all(list(topo[a]) == [40001.0] for a in [2021, 2022, 2023, 2024, 2025])
    # só a faixa ambígua de 2025 vira vazio; o resto do bruto preenchido sobrevive
    for ano in EDICOES:
        b = col(brutos, ano, "faixa")
        perdidas = (b == "de R$ 25.001/mês a R$ 3000/mês").sum()
        assert df.loc[df["ano"] == ano, "sal_min"].notna().sum() == b.notna().sum() - perdidas


def test_faixas_formam_escada(df):
    """Em cada ano, as faixas encostam uma na outra (máximo + 1 = próximo mínimo)."""
    for ano, g in df.dropna(subset=["sal_min"]).groupby("ano"):
        f = g[["sal_min", "sal_max"]].drop_duplicates().sort_values("sal_min").to_numpy()
        for (lo1, hi1), (lo2, _) in zip(f, f[1:]):
            assert lo2 == hi1 + 1 or (lo1 == 0 and lo2 == 1001), (ano, hi1, lo2)


def test_gestor(df, brutos):
    eh = df["gestor"].fillna(False).astype(bool)
    assert (df.loc[eh, "familia"] == "Gestor").all()
    assert (df.loc[~eh, "familia"] != "Gestor").all()
    assert df.loc[eh, "gestor_nivel"].notna().all()
    # 2019: os 29 gestores que também deram cargo mantêm o cargo
    g19 = df[(df["ano"] == 2019) & eh]
    assert (g19["cargo"] != "Gestor").sum() == 29


def test_cargo_so_vazio_quando_bruto_vazio(df, brutos):
    for ano in EDICOES:
        d = df[df["ano"] == ano]
        eh = d["gestor"].fillna(False).astype(bool)
        assert d.loc[~eh, "familia"].isna().sum() == col(brutos, ano, "cargo")[~eh.to_numpy()].isna().sum()


def test_nivel_e_experiencia(df):
    assert df.loc[df["ano"] == 2019, "nivel"].isna().all()       # 2019 não perguntou
    assert set(df["nivel_comparavel"].dropna()) == {"Júnior", "Pleno", "Sênior"}
    assert df["exp_anos"].dropna().between(0, 12).all()


def test_uf_e_regiao(df):
    ufs = set(df["uf"].dropna()) - {"Exterior"}
    assert ufs <= h.UFS
    com_uf = df[df["uf"].isin(h.UFS)]
    assert (com_uf["regiao"] == com_uf["uf"].map(h.REGIAO_DA_UF)).all()


def test_motivo_salario_so_para_insatisfeitos(df):
    """A pergunta do motivo só aparece para quem disse não estar satisfeito, e
    quase todo insatisfeito responde: em 2023, 15 de 1.333 deixaram em branco
    (nas outras edições, nenhum)."""
    r = df[df["ano"] >= 2021]
    respondeu = r["motivo_salario"].notna()
    insatisfeito = (r["satisfeito"] == False).fillna(False)  # noqa: E712
    assert (r.loc[respondeu, "satisfeito"] == False).all()  # noqa: E712
    em_branco = (insatisfeito & ~respondeu).groupby(r["ano"]).sum()
    assert em_branco.to_dict() == {2021: 0, 2022: 0, 2023: 15, 2024: 0, 2025: 0}
    assert df.loc[df["ano"] == 2019, "motivo_salario"].isna().all()
