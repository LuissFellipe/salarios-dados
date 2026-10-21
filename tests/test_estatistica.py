"""As ferramentas de faixa, testadas onde a resposta é conhecida: salários
simulados de uma lognormal, agrupados nas faixas do questionário."""

import numpy as np
import pandas as pd
import pytest

from salarios.estatistica import RegressaoIntervalar, ic_bootstrap, mediana_agrupada, quantil_agrupado

# as 14 faixas do questionário de 2021 em diante
CORTES = [0, 1000, 2000, 3000, 4000, 6000, 8000, 12000, 16000, 20000, 25000, 30000, 40000, np.inf]


def agrupar(salarios):
    i = np.searchsorted(CORTES, salarios, side="right") - 1
    lo = np.array([CORTES[k] + (1 if k else 0) for k in i], float)
    hi = np.array([CORTES[k + 1] for k in i], float)
    return lo, hi


def test_mediana_agrupada_perto_da_verdadeira():
    rng = np.random.default_rng(1)
    s = rng.lognormal(np.log(9000), 0.6, 20000)
    lo, hi = agrupar(s)
    verdadeira = np.median(s)
    estimada = mediana_agrupada(lo, hi)
    assert abs(estimada - verdadeira) / verdadeira < 0.03
    # e o atalho errado: mediana dos centros de faixa cai num degrau
    centros = np.where(np.isinf(hi), lo, (lo + hi) / 2)
    assert np.median(centros) in set(centros)


def test_quantil_no_topo_aberto_e_nan():
    lo = np.array([40001.0] * 6 + [30001.0] * 4)
    hi = np.array([np.inf] * 6 + [40000.0] * 4)
    assert np.isnan(quantil_agrupado(lo, hi, 0.5))
    assert 30001 <= quantil_agrupado(lo, hi, 0.2) <= 40000


def test_ic_bootstrap_cobre_a_verdade():
    rng = np.random.default_rng(2)
    s = rng.lognormal(np.log(7000), 0.5, 3000)
    lo, hi = agrupar(s)
    a, b = ic_bootstrap(lo, hi, n=300)
    assert a < np.median(s) < b


def test_regressao_intervalar_recupera_os_efeitos():
    rng = np.random.default_rng(3)
    n = 6000
    d = pd.DataFrame({
        "nivel": rng.choice(["Júnior", "Pleno", "Sênior"], n),
        "remoto": rng.integers(0, 2, n),
        "exp": rng.uniform(0, 12, n),
    })
    verdade = {"C(nivel)[T.Pleno]": 0.45, "C(nivel)[T.Sênior]": 0.85, "remoto": 0.10, "exp": 0.04}
    mu = (np.log(4500) + verdade["C(nivel)[T.Pleno]"] * (d.nivel == "Pleno")
          + verdade["C(nivel)[T.Sênior]"] * (d.nivel == "Sênior") + verdade["remoto"] * d.remoto + verdade["exp"] * d.exp)
    s = np.exp(mu + rng.normal(0, 0.35, n))
    d["sal_min"], d["sal_max"] = agrupar(s)
    m = RegressaoIntervalar("C(nivel) + remoto + exp", d).ajustar()
    c = m.coeficientes()
    for termo, beta in verdade.items():
        assert c.loc[termo, "ic_baixo"] < beta < c.loc[termo, "ic_alto"], (termo, c.loc[termo].to_dict())
    assert abs(m.sigma - 0.35) < 0.03


def test_regressao_no_centro_da_faixa_erra_mais():
    """O motivo de ter a regressão por intervalo: MQO no centro da faixa
    puxa o efeito para baixo quando muita gente está no topo aberto."""
    rng = np.random.default_rng(4)
    n = 6000
    x = rng.integers(0, 2, n)
    s = np.exp(np.log(20000) + 0.6 * x + rng.normal(0, 0.4, n))     # muita gente acima de 40 mil
    lo, hi = agrupar(s)
    d = pd.DataFrame({"x": x, "sal_min": lo, "sal_max": hi})
    certo = RegressaoIntervalar("x", d).ajustar().coeficientes().loc["x", "beta"]
    centro = np.log(np.where(np.isinf(hi), lo, (lo + hi) / 2))
    mqo = np.polyfit(x, centro, 1)[0]
    assert abs(certo - 0.6) < 0.05
    assert abs(mqo - 0.6) > abs(certo - 0.6) + 0.1


def test_regressao_aceita_categoria_sem_ninguem():
    """Recorte de uma categórica do pandas mantém as categorias vazias."""
    rng = np.random.default_rng(5)
    n = 2000
    d = pd.DataFrame({"g": pd.Categorical(rng.choice(["a", "b"], n), categories=["a", "b", "nunca"])})
    s = np.exp(np.log(6000) + 0.3 * (d.g == "b") + rng.normal(0, 0.3, n))
    d["sal_min"], d["sal_max"] = agrupar(s)
    c = RegressaoIntervalar("C(g)", d).ajustar().coeficientes()
    assert "C(g)[T.nunca]" not in c.index
    assert c.loc["C(g)[T.b]", "ic_baixo"] < 0.3 < c.loc["C(g)[T.b]", "ic_alto"]


def test_regressao_recusa_termos_redundantes():
    """Dois termos que marcam as mesmas pessoas não são identificáveis."""
    rng = np.random.default_rng(6)
    n = 1000
    g = rng.choice(["gestor", "outro"], n)
    d = pd.DataFrame({"familia": g, "nivel": np.where(g == "gestor", "gestor", rng.choice(["jr", "sr"], n))})
    d["sal_min"], d["sal_max"] = agrupar(rng.lognormal(np.log(8000), 0.4, n))
    with pytest.raises(ValueError, match="posto"):
        RegressaoIntervalar("C(familia) + C(nivel)", d)


def test_residuo_generalizado_acompanha_o_desvio_verdadeiro():
    rng = np.random.default_rng(7)
    n = 5000
    x = rng.integers(0, 2, n)
    eps = rng.normal(0, 0.4, n)
    d = pd.DataFrame({"x": x})
    d["sal_min"], d["sal_max"] = agrupar(np.exp(np.log(8000) + 0.5 * x + eps))
    r = RegressaoIntervalar("x", d).ajustar().residuo_generalizado()
    assert np.corrcoef(r, eps)[0, 1] > 0.85
    assert abs(r.mean()) < 0.02            # em média, ninguém ganha acima do próprio perfil
