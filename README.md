# salarios-dados

Quanto ganha quem trabalha com dados no Brasil, e do que isso depende. Análise
em Jupyter das seis edições do **State of Data Brazil** (Data Hackers + Bain &
Company), de 2019 a 2025-26: cerca de 22,7 mil respostas.

**Página com os resultados:** https://luissfellipe.github.io/salarios-dados/

## Cadernos

| | Pergunta | Estado |
|---|---|---|
| [01 · Os dados](notebooks/01_dados.ipynb) | De onde vêm, como seis questionários viram uma tabela, quem responde | pronto |
| [02 · Cargo e nível](notebooks/02_cargo_e_nivel.ipynb) | Quanto ganha cada cargo, em cada nível (2025) | pronto |
| [03 · Evolução](notebooks/03_evolucao.ipynb) | O salário real subiu ou caiu desde 2021? | pronto |
| [04 · Fatores](notebooks/04_fatores.ipynb) | Experiência, remoto, estado, pós-graduação e gênero, com o resto controlado | pronto |
| [05 · Satisfação](notebooks/05_satisfacao.ipynb) | Quem ganha abaixo do próprio perfil está saindo? | pronto |
| [06 · Conclusões](notebooks/06_conclusoes.ipynb) | As oito respostas numa página | pronto |

Os `.py` em `notebooks/` são a fonte (formato jupytext, diff legível); os
`.ipynb` são gerados e executados a partir deles.

## Principais resultados (reais de novembro de 2025)

- Mediana geral de quem mora no Brasil: **R$ 10,6 mil**.
- **Cientista e engenheiro de dados sêniores empatam em R$ 15,5 mil**; analista
  de dados sênior, R$ 10,8 mil. A distância abre com a senioridade.
- Do júnior ao sênior a mediana multiplica por 2,7 a 3,5 (nas famílias com mais respostas).
- "Especialista/Staff+", opção nova de 2025, tem mediana 42% acima de Sênior.
- O 100% remoto caiu de 56,8% (2021) para 40,2% (2025) entre quem responde.
- **O salário real do mesmo perfil caiu de 6% a 8% entre 2021 e 2025** (quem
  não é gestor, com cargo, nível, experiência, estado e escolaridade fixos;
  principal: −6,6%, IC −9,1% a −3,9%). A mediana bruta subiu 20% no mesmo
  período porque a amostra ficou mais sênior (sêniores: 30,8% → 48,0%).
- Por nível, em termos reais: **júnior −14%**, pleno −8%, sênior +6%.
- Com o resto fixo (não gestores, 2021–2025): **remoto +23%** sobre presencial
  (+14% no júnior, +35% no sênior); **Minas −8,6%** em relação a São Paulo;
  mestrado +13,5% e doutorado +16,3% sobre graduação; **mulheres −5,1%**
  (bruto −11,7%), maior entre sêniores (−8,2%).
- Experiência: +88% de "1 a 3" para "mais de 10 anos", mas a curva achata
  depois dos 6 anos.
- Quem ganha abaixo do próprio perfil quer sair: no quinto mais abaixo, 44%
  estão insatisfeitos e 49% procuram emprego (22% e 31% no quinto mais
  acima); 76% de quem reclama do salário está mesmo abaixo do perfil.

## Método

- **Harmonização** (`salarios/harmonizar.py`): a coluna de cada campo é escolhida
  pelo texto da pergunta, porque a letra muda de um ano para o outro; todo valor
  bruto passa por um mapa explícito, e valor desconhecido interrompe a carga.
- **Salário em faixa:** a pesquisa dá a faixa, não o valor. Mediana agrupada
  (interpolação dentro da faixa) e regressão por intervalo (log-normal com
  verossimilhança da faixa), em `salarios/estatistica.py`.
- **Inflação:** IPCA do IBGE (SIDRA, tabela 1737), novembro de cada ano levado a
  novembro de 2025.
- **Testes:** 23, em `tests/`. Os da harmonização conferem contra os CSVs
  brutos; os de estatística usam salários simulados, em que a resposta é
  conhecida. Sete defeitos foram plantados de propósito no código (mapeamento
  frouxo, faixa apagada, cargo de gestor sobrescrito, mediana sem interpolar,
  regressão sem o piso da faixa, categoria vazia, termos redundantes) e cada um
  foi pego por algum teste.
- **Salário relativo:** resíduo generalizado da regressão por intervalo — o
  desvio esperado de cada pessoa em relação ao próprio perfil, dada a faixa.

## Rodar

```
uv sync
mkdir -p dados/brutos && cd dados/brutos
for d in pesquisa-data-hackers-2019 state-of-data-2021 state-of-data-2022 \
         state-of-data-brazil-2023 state-of-data-brazil-20242025 state-of-data-brazil-2025-2026; do
  uvx kaggle datasets download datahackers/$d -p $d --unzip
done
cd ../..
uv run python -m salarios.harmonizar
uv run python -m ipykernel install --sys-prefix --name salarios-dados
uv run pytest
uv run python -m salarios.site      # site/data/resultados.json para a página
cd notebooks && uv run jupytext --set-kernel salarios-dados --to notebook --execute 0*.py
```

## Limites

- **Amostra autosselecionada:** pesquisa online da comunidade Data Hackers.
  Descreve quem responde, não o mercado inteiro. São Paulo é 40% das respostas.
- **A amostra muda de perfil:** idade mediana de 29 para 32 anos e pós-graduação
  de 44% para 57% entre 2019 e 2025.
- **Faixa, não valor:** o topo é aberto (acima de R$ 40.001; em 2019, de
  R$ 25.001), e é onde a faixa diz menos.

## Licença dos dados

State of Data Brazil, Data Hackers e Bain & Company, sob
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Os
dados não estão neste repositório; o que é derivado deles (tabelas e gráficos
dos cadernos) segue a mesma licença.
