"""Exporta os agregados que a página em site/ desenha.

Recalcula a partir de dados/tratados/respostas.parquet com as mesmas
especificações dos cadernos, para a página não depender de número copiado
à mão. Só sai agregado (medianas, efeitos, proporções): os dados da pesquisa
são CC BY-NC-SA 4.0, e o derivado publicado segue a mesma licença.

    uv run python -m salarios.site
"""

import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .edicoes import RAIZ
from .estatistica import RegressaoIntervalar, ic_bootstrap, mediana_agrupada, sem_categorias_vazias
from .ipca import fator

SAIDA = RAIZ / "site/data/resultados.json"
# Seis casas: a página arredonda para uma casa percentual. Gravar com quatro
# arredondava duas vezes — 5,2486% virava 0,0525 e a página mostrava 5,3%,
# contra 5,2% nos cadernos.
FAMILIAS = ["Engenheiro de ML", "Cientista de Dados", "Engenheiro / Arquiteto de Dados",
            "Desenvolvedor / Eng. de Software", "Analista de BI / Analytics Engineer",
            "Analista de Dados", "Analista de Negócios"]
NIVEIS = ["Júnior", "Pleno", "Sênior"]
VINCULOS = ["CLT", "PJ / Empreendedor", "Servidor público", "No Brasil, para empresa de fora", "Freelancer", "Estágio"]
REF = {"modelo": "Presencial", "uf_grupo": "SP", "ensino": "Graduação", "genero_b": "Masculino",
       "porte": "101–500", "situacao": "CLT", "exp_faixa": "1 a 3 anos"}


def base():
    df = pd.read_parquet(RAIZ / "dados/tratados/respostas.parquet")
    br = df[(df["situacao"] != "Fora do Brasil") & df["sal_min"].notna()].copy()
    f = br["ano"].map(fator)
    br["sal_min"], br["sal_max"] = br["sal_min"] * f, br["sal_max"] * f
    return df, br


def mediana_ic(g, n=500):
    if len(g) < 20:
        return None
    a, b = ic_bootstrap(g["sal_min"].to_numpy(), g["sal_max"].to_numpy(), n=n)
    return {"mediana": round(mediana_agrupada(g["sal_min"], g["sal_max"])), "ic": [round(a), round(b)], "n": len(g)}


def termo(v):
    return f"C({v}, Treatment('{REF[v]}'))" if v in REF else f"C({v})"


def amostra_modelo(br):
    d = br[(br["ano"] >= 2021) & br["familia"].notna() & (br["familia"] != "Gestor") & br["situacao"].isin(VINCULOS)].copy()
    d["ano_c"] = d["ano"].astype(str)
    d["genero_b"] = d["genero"].astype(str).where(d["genero"].isin(["Masculino", "Feminino"]))
    ufs = [u for u, n in d["uf"].value_counts().items() if n >= 150 and u != "Exterior"]
    d["uf_grupo"] = d["uf"].astype(str).where(d["uf"].isin(ufs), "outros estados")
    return sem_categorias_vazias(d.dropna(subset=["exp_faixa", "ensino", "modelo", "porte", "genero_b", "nivel_comparavel"]))


def efeito(m, var, nivel):
    r = m.efeitos().loc[f"{termo(var)}[T.{nivel}]"]
    return {"efeito": round(float(r["efeito"]), 6), "ic": [round(float(r["ic_baixo"]), 6), round(float(r["ic_alto"]), 6)]}


ROTULOS = {"familia": "Cargo", "nivel_comparavel": "Nível", "exp_faixa": "Experiência em dados", "modelo": "Modelo de trabalho",
           "uf_grupo": "Estado", "ensino": "Escolaridade", "genero_b": "Gênero", "porte": "Porte da empresa (pessoas)",
           "situacao": "Vínculo"}
# grupos do gráfico de fatores na versão completa; ano, cargo, nível e
# experiência têm gráfico próprio
GRUPOS_FATORES = {"modelo": "Modelo de trabalho (vs. presencial)", "ensino": "Escolaridade (vs. graduação)",
                  "uf_grupo": "Estado (vs. SP)", "genero_b": "Gênero (vs. homens)",
                  "porte": "Porte da empresa (vs. 101–500 pessoas)", "situacao": "Vínculo (vs. CLT)"}


def chave(nome):
    """'C(modelo, Treatment('Presencial'))[T.Remoto]' → ('modelo', 'Remoto')."""
    if nome == "Intercept":
        return None, None
    var = nome[2:].split(",")[0].split(")")[0]
    return var, nome.split("[T.", 1)[1][:-1]


def todos_os_efeitos(m):
    e = m.efeitos()
    return [{"grupo": GRUPOS_FATORES[v], "var": v, "nivel": n, "efeito": round(float(r["efeito"]), 6),
             "ic": [round(float(r["ic_baixo"]), 6), round(float(r["ic_alto"]), 6)]}
            for nome, r in e.iterrows() if (v := chave(nome)[0]) in GRUPOS_FATORES for n in [chave(nome)[1]]]


def simulador(m, d, variaveis):
    """O que a página precisa para prever log(salário) = xβ de um perfil
    qualquer, com o intervalo de xβ por x'Σx. Só agregado: coeficientes e
    covariância, nenhuma resposta."""
    k = len(m.nomes)
    vars_ = []
    for v in variaveis:
        com_coluna = {chave(n)[1] for n in m.nomes if chave(n)[0] == v}
        presentes = [str(c) for c in d[v].cat.categories] if hasattr(d[v], "cat") else sorted(d[v].dropna().unique())
        ref = [c for c in presentes if c not in com_coluna]
        assert len(ref) == 1, (v, ref)
        contagem = d[v].astype(str).value_counts()
        vars_.append({"var": v, "rotulo": ROTULOS[v], "ref": ref[0],
                      "niveis": [{"nivel": c, "n": int(contagem.get(c, 0))} for c in presentes]})
    return {
        "colunas": ["Intercept" if n == "Intercept" else "=".join(chave(n)) for n in m.nomes],
        "beta": [float(f"{b:.8g}") for b in m.theta[:-1]],
        "cov": [[float(f"{m.cov[i, j]:.6g}") for j in range(k)] for i in range(k)],
        "sigma": round(m.sigma, 6), "n": m.n, "variaveis": vars_,
        "fixo": {"ano_c": "2025"},
    }


def exportar():
    df, br = base()
    r25 = br[br["ano"] == 2025]

    # cargo × nível em cada edição com nível (2021 em diante): alimenta o
    # gráfico de cargos com filtro de ano e a evolução de cada cargo
    cargos_ano = [{"familia": fam, "nivel": nv, "ano": int(a), **m}
                  for a in range(2021, 2026) for fam in FAMILIAS for nv in NIVEIS
                  if (m := mediana_ic(br[(br["ano"] == a) & (br["familia"] == fam) & (br["nivel_comparavel"] == nv)]))]
    cargos = [{k: v for k, v in c.items() if k != "ano"} for c in cargos_ano if c["ano"] == 2025]

    com_nivel = br[br["nivel_comparavel"].notna()]
    evolucao_nivel = [{"nivel": nv, "ano": int(a), **mediana_ic(g)}
                      for nv in NIVEIS for a, g in com_nivel[com_nivel["nivel_comparavel"] == nv].groupby("ano")]
    mediana_geral = {int(a): round(mediana_agrupada(g["sal_min"], g["sal_max"])) for a, g in br.groupby("ano")}
    senior_pct = (com_nivel.groupby("ano")["nivel_comparavel"].apply(lambda s: (s == "Sênior").mean())).round(4)

    d = amostra_modelo(br)
    controles = ["ano_c", "familia", "nivel_comparavel", "exp_faixa", "modelo", "uf_grupo", "ensino", "genero_b", "porte", "situacao"]
    completo = RegressaoIntervalar(" + ".join(termo(v) for v in controles), d).ajustar()
    sem_nivel = RegressaoIntervalar(" + ".join(termo(v) for v in controles if v != "nivel_comparavel"), d).ajustar()

    # efeito do ano: a mesma amostra e a mesma especificação principal do
    # caderno 03 — não a amostra do 04, que filtra vínculo e mais campos e dá
    # outro número (−6,3% em vez de −6,6%)
    d03 = br[(br["ano"] >= 2021) & br["familia"].notna() & (br["familia"] != "Gestor")].dropna(subset=["exp_anos", "ensino"]).copy()
    d03["uf_grupo"] = d03["uf"].astype(str).where(d03["uf"].isin(["SP", "MG", "RJ", "PR", "RS", "SC", "DF"]), "outros")
    d03["ano_c"] = d03["ano"].astype(str)
    m03 = RegressaoIntervalar("C(ano_c) + C(familia) + C(nivel_comparavel) + exp_anos + I(exp_anos**2)"
                              " + C(uf_grupo) + C(ensino)", d03).ajustar()
    e03 = m03.efeitos()
    anos = [{"ano": 2021, "efeito": 0.0, "ic": [0.0, 0.0]}] + [
        {"ano": a, "efeito": round(float(e03.loc[f"C(ano_c)[T.{a}]", "efeito"]), 6),
         "ic": [round(float(e03.loc[f"C(ano_c)[T.{a}]", "ic_baixo"]), 6), round(float(e03.loc[f"C(ano_c)[T.{a}]", "ic_alto"]), 6)]}
        for a in [2022, 2023, 2024, 2025]]

    fatores = {
        "Modelo de trabalho (vs. presencial)": [("modelo", n) for n in ["Remoto", "Híbrido flexível", "Híbrido com dias fixos"]],
        "Escolaridade (vs. graduação)": [("ensino", n) for n in ["Doutorado", "Mestrado", "Pós-graduação (lato sensu)", "Sem graduação"]],
        "Estado (vs. SP)": [("uf_grupo", n) for n in ["DF", "SC", "PR", "MG", "RJ", "RS", "BA", "PE"]],
        "Gênero (vs. homens)": [("genero_b", "Feminino")],
    }
    fatores = [{"grupo": g, "nivel": n, **efeito(completo, v, n)} for g, itens in fatores.items() for v, n in itens]

    exp_ordem = ["Sem experiência", "Menos de 1 ano", "1 a 3 anos", "3 a 6 anos", "6 a 10 anos", "Mais de 10 anos"]
    experiencia = {nome: [0.0 if n == "1 a 3 anos" else efeito(m, "exp_faixa", n)["efeito"] for n in exp_ordem]
                   for nome, m in [("total", sem_nivel), ("dentro_do_nivel", completo)]}

    remoto_nivel, fatores_todos, n_por_nivel = {}, {"Todos": todos_os_efeitos(completo)}, {"Todos": completo.n}
    for nv in NIVEIS:
        m = RegressaoIntervalar(" + ".join(termo(v) for v in controles if v != "nivel_comparavel"),
                                d[d["nivel_comparavel"] == nv]).ajustar()
        remoto_nivel[nv] = efeito(m, "modelo", "Remoto")
        fatores_todos[nv], n_por_nivel[nv] = todos_os_efeitos(m), m.n
    sim = simulador(completo, d, [v for v in controles if v != "ano_c"])

    d["relativo"] = completo.residuo_generalizado()
    d["quinto"] = pd.qcut(d["relativo"], 5, labels=False) + 1
    d["insatisfeito"] = (d["satisfeito"] == False).astype(float).where(d["satisfeito"].notna())  # noqa: E712
    d["buscando"] = d["mudar"].isin(["Buscando (Brasil ou exterior)", "Buscando só no exterior"]).astype(float).where(d["mudar"].notna())
    ins = d[d["insatisfeito"] == 1].dropna(subset=["motivo_salario"])
    quintos = [{"quinto": int(q), "distancia": round(float(np.expm1(g["relativo"].median())), 6),
                "insatisfeito": round(float(g["insatisfeito"].mean()), 6), "buscando": round(float(g["buscando"].mean()), 6),
                "cita_salario": round(float(ins[ins["quinto"] == q]["motivo_salario"].astype(float).mean()), 6)}
               for q, g in d.groupby("quinto")]
    d["dez_pct_abaixo"] = -d["relativo"] / np.log(1.1)
    lg = smf.logit("insatisfeito ~ dez_pct_abaixo + C(nivel_comparavel) + C(ano_c) + C(familia) + C(modelo)",
                   d.dropna(subset=["insatisfeito"])).fit(disp=0)
    cita = ins.groupby("motivo_salario")["relativo"]

    modelo_trab = pd.crosstab(df[df["situacao"] != "Fora do Brasil"]["ano"], df["modelo"], normalize="index")

    dados = {
        "fonte": "State of Data Brazil (Data Hackers + Bain & Company), CC BY-NC-SA 4.0",
        "respostas": int(len(df)), "edicoes": sorted(int(a) for a in df["ano"].unique()),
        "mediana_2025": round(mediana_agrupada(r25["sal_min"], r25["sal_max"])), "respostas_2025": int(len(r25)),
        "cargos_2025": cargos, "cargos_ano": cargos_ano,
        "ipca": {int(a): round(fator(int(a)), 6) for a in sorted(df["ano"].unique())},
        "mediana_geral": mediana_geral, "respostas_ano": {int(a): int(n) for a, n in br.groupby("ano").size().items()},
        "senior_pct": {int(k): float(v) for k, v in senior_pct.items()},
        "evolucao_nivel": evolucao_nivel, "efeito_ano": anos, "n_modelo_ano": m03.n,
        "fatores": fatores, "n_modelo": completo.n,
        "experiencia": {"faixas": exp_ordem, **experiencia},
        "remoto_por_nivel": remoto_nivel, "fatores_todos": fatores_todos, "n_fatores": n_por_nivel,
        "simulador": sim,
        "remoto_pct": {int(a): round(float(v), 6) for a, v in modelo_trab["Remoto"].items()},
        "quintos": quintos,
        "insatisfacao_por_10pct": round(float(lg.get_margeff().summary_frame().loc["dez_pct_abaixo", "dy/dx"]), 6),
        "cita_salario_abaixo_do_perfil": round(float((ins[ins["motivo_salario"] == True]["relativo"] < 0).mean()), 6),  # noqa: E712
        "cita_salario_distancia": round(float(np.expm1(cita.median()[True])), 6),
    }
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(json.dumps(dados, ensure_ascii=False, indent=1))
    print(f"{SAIDA.relative_to(RAIZ)}: {SAIDA.stat().st_size / 1024:.0f} KB")
    return dados


if __name__ == "__main__":
    exportar()
