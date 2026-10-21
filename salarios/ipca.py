"""IPCA do IBGE (SIDRA, tabela 1737, número-índice) para levar salários a reais
de uma mesma data.

Referência de cada edição: novembro do ano. De 2021 a 2025 a coleta foi de
meados de outubro a meados de dezembro (descrição de cada dataset no
Kaggle); a de 2019 não informa o período e segue a mesma suposição.
Os valores saem em reais de novembro de 2025, o mês central da coleta mais
recente.
"""

import json
import urllib.request

from .edicoes import RAIZ

URL = "https://apisidra.ibge.gov.br/values/t/1737/n1/all/v/2266/p/201901-202612"
CACHE = RAIZ / "dados/brutos/ipca_1737.json"
MES_REFERENCIA = 11
BASE = (2025, 11)


def indices() -> dict[tuple[int, int], float]:
    if not CACHE.exists():
        with urllib.request.urlopen(URL, timeout=60) as r:
            CACHE.write_bytes(r.read())
    linhas = json.loads(CACHE.read_text())[1:]       # a primeira linha é o cabeçalho
    return {(int(l["D3C"][:4]), int(l["D3C"][4:])): float(l["V"]) for l in linhas if l["V"] not in ("...", "-")}


def fator(ano: int) -> float:
    """Multiplica um valor de novembro do `ano` para reais de novembro de 2025."""
    i = indices()
    return i[BASE] / i[(ano, MES_REFERENCIA)]
