"""Seis edições do State of Data → uma tabela, uma linha por resposta.

Regra: todo valor bruto passa por um mapeamento explícito. Valor que nenhuma
tabela conhece interrompe a harmonização em vez de virar vazio: em pesquisa
que muda de questionário todo ano, o vazio silencioso é como uma categoria
nova some da análise sem ninguém perceber.

    uv run python -m salarios.harmonizar     # grava dados/tratados/respostas.parquet
"""

import re

import numpy as np
import pandas as pd

from .edicoes import BRUTOS, EDICOES, RAIZ

SAIDA = RAIZ / "dados/tratados"
VAZIO = "__vazio__"   # marca, nos mapas, o valor bruto que vira NaN de propósito


class ValorDesconhecido(ValueError):
    pass


def mapear(serie: pd.Series, mapa: dict, campo: str, ano: int) -> pd.Series:
    brutos = set(serie.dropna().unique())
    faltam = sorted(map(str, brutos - mapa.keys()))
    if faltam:
        raise ValorDesconhecido(f"{ano} · {campo}: sem mapeamento para {faltam}")
    return serie.map(lambda v: np.nan if pd.isna(v) or mapa[v] == VAZIO else mapa[v])


# ── faixa salarial ────────────────────────────────────────────────────────

RE_FAIXA = re.compile(r"de R\$ ([\d.]+)/mês a R\$ ([\d.]+)/mês")
# Erros de digitação do próprio questionário, uma resposta cada.
FAIXAS_CORRIGIDAS = {
    # o teto (2.000) bate com a faixa de R$ 1.001 a R$ 2.000; falta um dígito no piso
    "de R$ 101/mês a R$ 2.000/mês": (1001, 2000),
    # 25.001 a 3.000 não é faixa: pode ser 25–30 mil ou 2–3 mil. Fica vazio.
    "de R$ 25.001/mês a R$ 3000/mês": None,
}


def faixa_em_reais(rotulo) -> tuple[float, float]:
    """(mínimo, máximo) em R$/mês. Faixa aberta no topo tem máximo infinito."""
    if pd.isna(rotulo):
        return (np.nan, np.nan)
    if rotulo in FAIXAS_CORRIGIDAS:
        corr = FAIXAS_CORRIGIDAS[rotulo]
        return corr if corr else (np.nan, np.nan)
    if rotulo == "Menos de R$ 1.000/mês":
        return (0.0, 1000.0)
    if m := re.fullmatch(r"Acima de R\$ ([\d.]+)/mês", rotulo):
        return (float(m.group(1).replace(".", "")), np.inf)
    if m := RE_FAIXA.fullmatch(rotulo):
        lo, hi = (float(x.replace(".", "")) for x in m.groups())
        if hi <= lo:
            raise ValorDesconhecido(f"faixa invertida: {rotulo!r}")
        return (lo, hi)
    raise ValorDesconhecido(f"faixa salarial desconhecida: {rotulo!r}")


# ── cargo ─────────────────────────────────────────────────────────────────

# rótulo bruto → (cargo fino, família comparável entre anos)
# Famílias juntam o que alguma edição juntou: BI com Analytics Engineer (2021),
# engenheiro com arquiteto de dados (2022–2023), dev com analista de sistemas
# (2022 em diante).
CARGOS = {
    # 2019
    "Data Scientist/Cientista de Dados": ("Cientista de Dados", "Cientista de Dados"),
    "Data Analyst/Analista de Dados": ("Analista de Dados", "Analista de Dados"),
    "Business Intelligence/Analista de BI": ("Analista de BI", "Analista de BI / Analytics Engineer"),
    "Data Engineer/Engenheiro de Dados": ("Engenheiro de Dados", "Engenheiro / Arquiteto de Dados"),
    "Business Analyst/Analista de Negócios": ("Analista de Negócios", "Analista de Negócios"),
    "Engenheiro de Machine Learning": ("Engenheiro de ML", "Engenheiro de ML"),
    "Engenheiro": ("Outras Engenharias", "Outros"),
    "Outras": ("Outro", "Outros"),
    # 2021 em diante
    "Cientista de Dados/Data Scientist": ("Cientista de Dados", "Cientista de Dados"),
    "Analista de Dados/Data Analyst": ("Analista de Dados", "Analista de Dados"),
    "Analista de BI/BI Analyst/Analytics Engineer": ("Analista de BI / Analytics Engineer", "Analista de BI / Analytics Engineer"),
    "Analista de BI/BI Analyst": ("Analista de BI", "Analista de BI / Analytics Engineer"),
    "Analytics Engineer": ("Analytics Engineer", "Analista de BI / Analytics Engineer"),
    "Engenheiro de Dados/Data Engineer": ("Engenheiro de Dados", "Engenheiro / Arquiteto de Dados"),
    "Engenheiro de Dados/Arquiteto de Dados/Data Engineer/Data Architect": ("Engenheiro / Arquiteto de Dados", "Engenheiro / Arquiteto de Dados"),
    "Engenheiro de Dados/Data Engineer/Data Architect": ("Engenheiro de Dados", "Engenheiro / Arquiteto de Dados"),
    "Arquiteto de Dados/Data Architect": ("Arquiteto de Dados", "Engenheiro / Arquiteto de Dados"),
    "Arquiteto de Dados": ("Arquiteto de Dados", "Engenheiro / Arquiteto de Dados"),
    "Arquiteto de dados": ("Arquiteto de Dados", "Engenheiro / Arquiteto de Dados"),
    "Engenheiro de Machine Learning/ML Engineer": ("Engenheiro de ML", "Engenheiro de ML"),
    "Engenheiro de Machine Learning/ML Engineer/AI Engineer": ("Engenheiro de ML", "Engenheiro de ML"),
    "Analista de Negócios/Business Analyst": ("Analista de Negócios", "Analista de Negócios"),
    "Desenvolvedor ou Engenheiro de Software": ("Desenvolvedor / Eng. de Software", "Desenvolvedor / Eng. de Software"),
    "Analista de Sistemas/Analista de TI": ("Analista de Sistemas", "Desenvolvedor / Eng. de Software"),
    "Desenvolvedor/ Engenheiro de Software/ Analista de Sistemas": ("Desenvolvedor / Eng. de Software", "Desenvolvedor / Eng. de Software"),
    "Product Manager": ("Product Manager", "Product Manager"),
    "Product Manager/ Product Owner (PM/APM/DPM/GPM/PO)": ("Product Manager", "Product Manager"),
    "Data Product Manager/ Product Manager (PM/APM/DPM/GPM/PO)": ("Product Manager", "Product Manager"),
    "Estatístico": ("Estatístico", "Estatístico"),
    "DBA/Administrador de Banco de Dados": ("DBA", "DBA"),
    "Analista de Inteligência de Mercado": ("Inteligência de Mercado", "Inteligência de Mercado"),
    "Analista de Inteligência de Mercado/Market Intelligence": ("Inteligência de Mercado", "Inteligência de Mercado"),
    "Analista de Marketing": ("Analista de Marketing", "Outros"),
    "Economista": ("Economista", "Outros"),
    "Professor": ("Professor / Pesquisador", "Outros"),
    "Professor/Pesquisador": ("Professor / Pesquisador", "Outros"),
    "Analista Administrativo": ("Analista Administrativo", "Outros"),
    "Técnico": ("Suporte / Técnico", "Outros"),
    "Suporte Técnico": ("Suporte / Técnico", "Outros"),
    "Analista de Suporte/Analista Técnico": ("Suporte / Técnico", "Outros"),
    "Outras Engenharias (não inclui dev)": ("Outras Engenharias", "Outros"),
    "Outro": ("Outro", "Outros"),
    "Outra Opção": ("Outro", "Outros"),
}

GESTOR_NIVEL = {
    # 2019
    "Coordenador": "Supervisor/Coordenador", "Supervisor": "Supervisor/Coordenador",
    "Gerente": "Gerente/Head/Diretor/VP", "Head": "Gerente/Head/Diretor/VP", "Diretor": "Gerente/Head/Diretor/VP",
    "Team Leader/Tech Leader": "Team Leader/Tech Leader", "C-level (CDO, CIO, CTO)": "Sócio ou C-level",
    "Outras": "Outros",
    # 2021 em diante; 2021 junta gerente, head, diretor e VP numa opção só
    "Gerente/Head/Diretor/VP": "Gerente/Head/Diretor/VP", "Gerente/Head": "Gerente/Head/Diretor/VP",
    "Diretor/VP": "Gerente/Head/Diretor/VP", "Supervisor/Coordenador": "Supervisor/Coordenador",
    "Sócio ou C-level (CEO, CDO, CIO, CTO etc)": "Sócio ou C-level",
}
# 2022 teve "Outra" com texto livre (13 respostas: "LIder", "MEI", "Chapter Lead"...)
GESTOR_TEXTO_LIVRE = "Outros"

NIVEL = {"Júnior": "Júnior", "Pleno": "Pleno", "Sênior": "Sênior", "Especialista/Staff+": "Especialista/Staff+"}

# ── demais campos ─────────────────────────────────────────────────────────

GENERO = {"Masculino": "Masculino", "Feminino": "Feminino", "Outro": "Outro", "Prefiro não informar": VAZIO}

ENSINO = {
    "Não tenho graduação formal": "Sem graduação", "Estudante de Graduação": "Graduação em curso",
    "Graduação/Bacharelado": "Graduação", "Pós-graduação": "Pós-graduação (lato sensu)",
    "Mestrado": "Mestrado", "Doutorado ou Phd": "Doutorado", "Prefiro não informar": VAZIO,
}

SITUACAO = {
    "Empregado (CTL)": "CLT", "Empregado (CLT)": "CLT",
    "Empreendedor ou Empregado (CNPJ)": "PJ / Empreendedor", "Servidor Público": "Servidor público",
    "Estagiário": "Estágio", "Freelancer": "Freelancer",
    "Trabalho na área Acadêmica/Pesquisador": "Academia / Pesquisa",
    "Vivo no Brasil e trabalho remoto para empresa de fora do Brasil (PJ)": "No Brasil, para empresa de fora",
    "Vivo no Brasil e trabalho remoto para empresa de fora do Brasil": "No Brasil, para empresa de fora",
    "Vivo fora do Brasil e trabalho para empresa de fora do Brasil": "Fora do Brasil",
    "Desempregado, buscando recolocação": "Desempregado", "Desempregado e não estou buscando recolocação": "Desempregado",
    "Somente Estudante (graduação)": "Só estudante", "Somente Estudante (pós-graduação)": "Só estudante",
    "Prefiro não dizer": VAZIO, "Prefiro não informar": VAZIO,
}

PORTE = {
    "de 1 a 5": "1–10", "de 6 a 10": "1–10", "de 11 a 50": "11–100", "de 51 a 100": "11–100",
    "de 101 a 500": "101–500", "de 501 a 1000": "501–3.000", "de 501 a 1.000": "501–3.000",
    "de 1001 a 3000": "501–3.000", "de 1.001 a 3.000": "501–3.000",
    "Acima de 3000": "Acima de 3.000", "Acima de 3.000": "Acima de 3.000",
    # 2 respostas (2023, 2025): 501 a 1.000 ou 51 a 100?
    "de 501 a 100": VAZIO,
}

# anos aproximados (centro da faixa) e faixa comum. As faixas do questionário
# mudam todo ano (2–3, 3–4, 4–5, 4–6, 5–6), então a faixa comum tem fronteiras
# que algumas edições cruzam; o número aproximado é o que entra na regressão.
EXPERIENCIA = {
    "Não tenho experiência na área de dados": (0.0, "Sem experiência"),
    "Menos de 1 ano": (0.5, "Menos de 1 ano"),
    "de 1 a 2 anos": (1.5, "1 a 3 anos"), "de 2 a 3 anos": (2.5, "1 a 3 anos"),
    "de 3 a 4 anos": (3.5, "3 a 6 anos"), "de 4 a 5 anos": (4.5, "3 a 6 anos"),
    "de 4 a 6 anos": (5.0, "3 a 6 anos"), "de 5 a 6 anos": (5.5, "3 a 6 anos"),
    "de 6 a 10 anos": (8.0, "6 a 10 anos"), "de 7 a 10 anos": (8.5, "6 a 10 anos"),
    "Mais de 10 anos": (12.0, "Mais de 10 anos"),
}

MODELO = {
    "Modelo 100% remoto": "Remoto",
    "Modelo híbrido flexível (o funcionário tem liberdade para escolher quando estar no escritório presencialmente)": "Híbrido flexível",
    "Modelo híbrido com dias fixos de trabalho presencial": "Híbrido com dias fixos",
    "Modelo 100% presencial": "Presencial",
}

MUDAR = {
    "Não estou buscando e não pretendo mudar de emprego nos próximos 6 meses": "Não pretende mudar",
    "Não estou buscando, mas me considero aberto a outras oportunidades": "Aberto a propostas",
    "Estou em busca de oportunidades dentro ou fora do Brasil": "Buscando (Brasil ou exterior)",
    "Estou em busca de oportunidades, mas apenas fora do Brasil": "Buscando só no exterior",
}

UFS = {"AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG", "PA", "PB", "PR",
       "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO"}
REGIAO_DA_UF = {
    **dict.fromkeys(["AC", "AP", "AM", "PA", "RO", "RR", "TO"], "Norte"),
    **dict.fromkeys(["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"], "Nordeste"),
    **dict.fromkeys(["DF", "GO", "MT", "MS"], "Centro-Oeste"),
    **dict.fromkeys(["ES", "MG", "RJ", "SP"], "Sudeste"),
    **dict.fromkeys(["PR", "RS", "SC"], "Sul"),
}
REGIAO = {"Sudeste": "Sudeste", "Sul": "Sul", "Nordeste": "Nordeste", "Centro-oeste": "Centro-Oeste",
          "Norte": "Norte", "Exterior": "Exterior",
          "Região Sudeste": "Sudeste", "Região Sul": "Sul"}


def uf(valor):
    """2019 escreve "São Paulo (SP)"; as outras, só a sigla."""
    if pd.isna(valor):
        return np.nan
    if valor == "Exterior":
        return "Exterior"
    sigla = m.group(1) if (m := re.search(r"\(([A-Z]{2})\)$", valor)) else valor
    if sigla not in UFS:
        raise ValorDesconhecido(f"UF desconhecida: {valor!r}")
    return sigla


def booleano(valor):
    if pd.isna(valor):
        return pd.NA
    if valor in (True, 1, 1.0, "True"):
        return True
    if valor in (False, 0, 0.0, "False"):
        return False
    raise ValorDesconhecido(f"booleano desconhecido: {valor!r}")


# ── montagem ─────────────────────────────────────────────────────────────

def harmonizar_edicao(ano: int) -> pd.DataFrame:
    ed = EDICOES[ano]
    cols = ed["colunas"]
    bruto = pd.read_csv(BRUTOS / ed["arquivo"], usecols=list(cols.values()), low_memory=False)
    b = {campo: bruto[col] for campo, col in cols.items()}
    n = len(bruto)
    vazio = pd.Series([np.nan] * n)
    out = pd.DataFrame({"ano": ano, "resposta": range(n)})

    out["idade"] = pd.to_numeric(b["idade"], errors="raise")
    out["genero"] = mapear(b["genero"], GENERO, "genero", ano)
    out["uf"] = b["uf"].map(uf)
    regiao = mapear(b["regiao"], REGIAO, "regiao", ano)
    # a região sai da UF quando ela existe; o campo região do questionário só
    # cobre quem a anonimização deixou sem UF
    out["regiao"] = out["uf"].map(REGIAO_DA_UF).fillna(regiao)
    out.loc[out["uf"] == "Exterior", "regiao"] = "Exterior"
    out["ensino"] = mapear(b["ensino"], ENSINO, "ensino", ano)
    out["situacao"] = mapear(b["situacao"], SITUACAO, "situacao", ano)
    out["porte"] = mapear(b["porte"], PORTE, "porte", ano)
    out["gestor"] = b["gestor"].map(booleano).astype("boolean")

    cargo_bruto = b["cargo"]
    fino_familia = mapear(cargo_bruto, {k: k for k in CARGOS}, "cargo", ano)
    out["cargo"] = fino_familia.map(lambda v: CARGOS[v][0] if pd.notna(v) else np.nan)
    out["familia"] = fino_familia.map(lambda v: CARGOS[v][1] if pd.notna(v) else np.nan)
    gnivel = b["cargo_gestor"].map(
        lambda v: np.nan if pd.isna(v) else GESTOR_NIVEL.get(v, GESTOR_TEXTO_LIVRE if ano == 2022 else None))
    if gnivel.isna().sum() != b["cargo_gestor"].isna().sum():
        desconhecidos = sorted(set(b["cargo_gestor"].dropna()) - GESTOR_NIVEL.keys())
        raise ValorDesconhecido(f"{ano} · cargo_gestor: {desconhecidos}")
    out["gestor_nivel"] = gnivel
    # gestor vira a família "Gestor". De 2021 em diante ele não responde
    # "cargo atual", e o cargo fica "Gestor"; em 2019, 29 gestores
    # responderam as duas perguntas, e o cargo que deram fica
    eh_gestor = out["gestor"].fillna(False).astype(bool)
    out.loc[eh_gestor & out["cargo"].isna(), "cargo"] = "Gestor"
    out.loc[eh_gestor, "familia"] = "Gestor"

    out["nivel"] = mapear(b["nivel"], NIVEL, "nivel", ano) if "nivel" in b else vazio
    out["nivel_comparavel"] = out["nivel"].replace({"Especialista/Staff+": "Sênior"})

    lim = b["faixa"].map(faixa_em_reais)
    out["faixa"] = b["faixa"]
    out["sal_min"] = lim.map(lambda t: t[0])
    out["sal_max"] = lim.map(lambda t: t[1])

    exp = mapear(b["exp_dados"], {k: k for k in EXPERIENCIA}, "exp_dados", ano)
    out["exp_anos"] = exp.map(lambda v: EXPERIENCIA[v][0] if pd.notna(v) else np.nan)
    out["exp_faixa"] = exp.map(lambda v: EXPERIENCIA[v][1] if pd.notna(v) else np.nan)

    out["satisfeito"] = (b["satisfeito"].map(booleano) if "satisfeito" in b else pd.Series([pd.NA] * n)).astype("boolean")
    out["mudar"] = mapear(b["mudar"], MUDAR, "mudar", ano) if "mudar" in b else vazio
    out["motivo_salario"] = (b["motivo_salario"].map(booleano) if "motivo_salario" in b
                             else pd.Series([pd.NA] * n)).astype("boolean")
    out["modelo"] = mapear(b["modelo"], MODELO, "modelo", ano) if "modelo" in b else vazio
    return out


CATEGORIAS = ["genero", "uf", "regiao", "ensino", "situacao", "porte", "cargo", "familia", "gestor_nivel",
              "nivel", "nivel_comparavel", "faixa", "exp_faixa", "mudar", "modelo"]


def harmonizar() -> pd.DataFrame:
    df = pd.concat([harmonizar_edicao(a) for a in EDICOES], ignore_index=True)
    for c in CATEGORIAS:
        df[c] = df[c].astype("category")
    return df


def main():
    df = harmonizar()
    SAIDA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SAIDA / "respostas.parquet", index=False)
    print(f"{len(df):,} respostas · {df.shape[1]} colunas → dados/tratados/respostas.parquet")
    print(df.groupby("ano").size().to_string())


if __name__ == "__main__":
    main()
