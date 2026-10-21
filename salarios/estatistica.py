"""Estatística para salário em faixa.

O questionário não pergunta o salário, pergunta a faixa ("de R$ 8.001 a R$
12.000"). Duas consequências:

- **Mediana agrupada.** A mediana de dado em classes: acha a faixa onde a
  frequência acumulada cruza 50% e interpola linearmente dentro dela. Usar o
  centro da faixa de cada pessoa e tirar a mediana disso dá sempre um centro
  de faixa — degraus de 4 mil reais em vez de um número.
- **Regressão por intervalo.** log(salário) = Xβ + ε, ε ~ Normal(0, σ²), e a
  verossimilhança de cada pessoa é a probabilidade de o salário cair na faixa
  dela: Φ((log máx − Xβ)/σ) − Φ((log mín − Xβ)/σ). A faixa aberta do topo
  (máximo infinito) e a de baixo (mínimo zero) entram sem truque. exp(β) − 1 é
  o efeito percentual de cada variável com as outras fixas.
"""

import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy import optimize, stats


# ── mediana e quantis de dado agrupado ────────────────────────────────────

def quantil_agrupado(lo: np.ndarray, hi: np.ndarray, q: float = 0.5) -> float:
    """Quantil de respostas em faixa, por interpolação linear dentro da faixa.

    Devolve NaN se o quantil cair na faixa aberta do topo (não há como
    interpolar até o infinito) — quem chama decide como reportar ("acima de")."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    ok = ~np.isnan(lo)
    lo, hi = lo[ok], hi[ok]
    if len(lo) == 0:
        return np.nan
    faixas = pd.DataFrame({"lo": lo, "hi": hi}).value_counts().sort_index()
    acum = faixas.cumsum() / faixas.sum()
    alvo_n = q
    anterior = 0.0
    for (flo, fhi), frac in acum.items():
        if frac >= alvo_n:
            if np.isinf(fhi):
                return np.nan
            dentro = (alvo_n - anterior) / (frac - anterior)
            return flo + dentro * (fhi - flo)
        anterior = frac
    return np.nan


def mediana_agrupada(lo, hi) -> float:
    return quantil_agrupado(lo, hi, 0.5)


def ic_bootstrap(lo, hi, estatistica=mediana_agrupada, n: int = 1000, semente: int = 0, nivel: float = 0.95):
    """Intervalo de confiança por reamostragem das respostas."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    ok = ~np.isnan(lo)
    lo, hi = lo[ok], hi[ok]
    rng = np.random.default_rng(semente)
    reps = []
    for _ in range(n):
        i = rng.integers(0, len(lo), len(lo))
        reps.append(estatistica(lo[i], hi[i]))
    reps = np.array(reps)
    a = (1 - nivel) / 2
    return float(np.nanquantile(reps, a)), float(np.nanquantile(reps, 1 - a))


# ── regressão por intervalo ───────────────────────────────────────────────

class RegressaoIntervalar:
    """log(salário) ~ fórmula, com salário observado só como faixa [mín, máx].

        m = RegressaoIntervalar("C(nivel) + exp_anos", df).ajustar()
        m.efeitos()      # efeito percentual, com intervalo de 95%
    """

    def __init__(self, formula: str, dados: pd.DataFrame, lo: str = "sal_min", hi: str = "sal_max"):
        d = dados.dropna(subset=[lo, hi]).copy()
        # categórica do pandas guarda as categorias que o recorte não tem
        # ("Gestor" numa amostra sem gestores); o patsy faria delas colunas de
        # zeros e a matriz não teria inversa
        d = sem_categorias_vazias(d)
        self.X = dmatrix(formula, d, return_type="dataframe")
        posto = np.linalg.matrix_rank(self.X.to_numpy())
        if posto < self.X.shape[1]:
            # colinearidade perfeita: algum termo é combinação exata de outros
            # (ex.: família "Gestor" e nível "Gestor" marcando as mesmas
            # pessoas). O otimizador acha um mínimo, mas os coeficientes
            # envolvidos não têm sentido, e a hessiana numérica às vezes nem
            # avisa
            raise ValueError(f"matriz sem posto completo ({posto} de {self.X.shape[1]} colunas): "
                             "há termos redundantes na fórmula")
        d = d.loc[self.X.index]
        self.indice = d.index
        self.n = len(d)
        with np.errstate(divide="ignore"):
            self.llo = np.log(d[lo].to_numpy(float))          # log(0) = −∞: faixa de baixo
        self.lhi = np.log(d[hi].to_numpy(float))              # log(∞) = ∞: faixa do topo
        self.nomes = list(self.X.columns)

    def _nll(self, theta):
        beta, log_sigma = theta[:-1], theta[-1]
        sigma = np.exp(log_sigma)
        mu = self.X.to_numpy() @ beta
        p = stats.norm.cdf((self.lhi - mu) / sigma) - stats.norm.cdf((self.llo - mu) / sigma)
        return -np.sum(np.log(np.clip(p, 1e-300, None)))

    def ajustar(self):
        # partida: MQO no centro da faixa (faixas abertas no limite finito)
        centro = np.where(np.isinf(self.lhi), self.llo, np.where(np.isinf(self.llo), self.lhi,
                                                                 (self.llo + self.lhi) / 2))
        b0, *_ = np.linalg.lstsq(self.X.to_numpy(), centro, rcond=None)
        theta0 = np.append(b0, np.log(np.std(centro - self.X.to_numpy() @ b0) + 1e-6))
        r = optimize.minimize(self._nll, theta0, method="BFGS")
        if not r.success and r.status != 2:      # 2 = perda de precisão perto do ótimo, aceitável
            raise RuntimeError(f"não convergiu: {r.message}")
        self.theta = r.x
        self.cov = self._hessiana_inversa(r.x)
        self.sigma = float(np.exp(r.x[-1]))
        return self

    def _hessiana_inversa(self, x, eps=1e-4):
        """Hessiana numérica por diferenças centrais (a do BFGS é aproximação
        grosseira para erro-padrão)."""
        k = len(x)
        H = np.zeros((k, k))
        for i in range(k):
            for j in range(i, k):
                ei, ej = np.eye(k)[i] * eps, np.eye(k)[j] * eps
                H[i, j] = H[j, i] = (self._nll(x + ei + ej) - self._nll(x + ei - ej)
                                     - self._nll(x - ei + ej) + self._nll(x - ei - ej)) / (4 * eps * eps)
        return np.linalg.inv(H)

    def coeficientes(self) -> pd.DataFrame:
        ep = np.sqrt(np.diag(self.cov))[:-1]
        b = self.theta[:-1]
        return pd.DataFrame({"beta": b, "ep": ep, "ic_baixo": b - 1.96 * ep, "ic_alto": b + 1.96 * ep},
                            index=self.nomes)

    def efeitos(self) -> pd.DataFrame:
        """Efeito percentual de cada termo (exp(β) − 1), sem o intercepto."""
        c = self.coeficientes().drop(index="Intercept", errors="ignore")
        return pd.DataFrame({"efeito": np.expm1(c["beta"]), "ic_baixo": np.expm1(c["ic_baixo"]),
                             "ic_alto": np.expm1(c["ic_alto"])})

    def residuo_generalizado(self) -> pd.Series:
        """Quanto cada pessoa ganha acima ou abaixo do esperado para o perfil
        dela, em log: E[ε | salário dentro da faixa]. Como só a faixa é
        conhecida, é a média de uma normal truncada nos limites dela:
        σ · (φ(a) − φ(b)) / (Φ(b) − Φ(a)), com a e b os limites padronizados.
        exp(resíduo) − 1 lê como "% acima do perfil"."""
        mu = self.X.to_numpy() @ self.theta[:-1]
        a, b = (self.llo - mu) / self.sigma, (self.lhi - mu) / self.sigma
        massa = np.clip(stats.norm.cdf(b) - stats.norm.cdf(a), 1e-300, None)
        return pd.Series(self.sigma * (stats.norm.pdf(a) - stats.norm.pdf(b)) / massa, index=self.indice)


def sem_categorias_vazias(dados: pd.DataFrame) -> pd.DataFrame:
    """Tira das colunas categóricas as categorias que o recorte não tem. O
    patsy (statsmodels incluído) cria uma coluna para cada categoria, vazia ou
    não, e a coluna de zeros deixa a matriz sem inversa."""
    return dados.apply(lambda c: c.cat.remove_unused_categories() if c.dtype.name == "category" else c)
