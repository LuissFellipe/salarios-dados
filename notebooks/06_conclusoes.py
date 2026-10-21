# %% [markdown]
# # 06 · Conclusões
#
# As oito perguntas do projeto, respondidas com os números dos cadernos 02 a
# 05. Valores em reais de novembro de 2025. A base é o State of Data Brazil
# (Data Hackers + Bain & Company, CC BY-NC-SA 4.0): **quem responde a
# pesquisa**, não o mercado inteiro.
#
# ## 1. Quanto ganha cada cargo, em cada nível? ([02](02_cargo_e_nivel.ipynb))
#
# Em 2025, a mediana de quem mora no Brasil é **R$ 10,6 mil**. No nível
# sênior, **cientista e engenheiro de dados empatam em R$ 15,5 mil**;
# desenvolvedor, R$ 14,8 mil; analista de BI ou Analytics Engineer, R$ 12,2
# mil; analista de dados, R$ 10,8 mil. Engenheiro de ML sênior tem a maior
# mediana (R$ 20,3 mil), com amostra pequena. Do júnior ao sênior a mediana
# multiplica por 2,7 a 3,5 nas famílias com mais respostas. Quem marcou "Especialista/Staff+" ganha 42% mais
# que quem marcou Sênior; gerente, head ou diretor, R$ 23,1 mil.
#
# ## 2. O salário real subiu ou caiu? ([03](03_evolucao.ipynb))
#
# **Caiu para o mesmo perfil.** A mediana bruta subiu 20% entre 2021 e 2025,
# mas porque a amostra ficou mais sênior (de 30,8% para 48,0% de sêniores).
# Com cargo, nível, experiência, estado e escolaridade fixos, quem não é
# gestor ganhava em 2025 **de 6% a 8% menos em poder de compra** que em 2021
# (principal: −6,6%, IC −9,1% a −3,9%, em cinco especificações). Por nível:
# **júnior −14%**, pleno −8%, sênior +6%. Para gestores não há efeito claro.
#
# ## 3. Quanto vale a experiência? ([04](04_fatores.ipynb))
#
# Sem controlar o nível, ter mais de 10 anos de dados paga 88% a mais que ter
# de 1 a 3 — **mas a curva achata depois dos 6 anos**: de "6 a 10" para "mais
# de 10" são só 6% a mais. Dentro do mesmo nível, a experiência vale 31%; a
# maior parte do ganho vem pela promoção.
#
# ## 4. Remoto paga mais? ([04](04_fatores.ipynb))
#
# **Sim, 23% a mais que presencial**, com o resto fixo; híbrido flexível paga
# como remoto (+25%), híbrido com dias fixos fica no meio (+15%). O número é
# estável nos cinco anos e cresce com a senioridade: +14% no júnior, +35% no
# sênior. É associação: vaga remota tende a estar em empresa que paga mais.
# E o remoto está encolhendo: de 56,8% para 40,2% de quem responde, entre 2021
# e 2025.
#
# ## 5. Quanto paga o estado? Minas × São Paulo ([04](04_fatores.ipynb))
#
# **Minas Gerais paga 8,6% menos que São Paulo** com o resto fixo, no mesmo
# patamar de Rio (−8,9%) e Paraná (−8,3%). O DF empata com SP. No Nordeste, de
# −9% a −17%. A pesquisa publica só a UF: cidades não dá para separar.
#
# ## 6. Pós-graduação aumenta o salário? ([04](04_fatores.ipynb))
#
# Em relação a quem tem graduação: **mestrado +13,5%, doutorado +16,3%**, pós
# lato sensu +5,2%. Sem graduação, −10,9%.
#
# ## 7. Existe diferença entre homens e mulheres? ([04](04_fatores.ipynb))
#
# **Sim: mulheres ganham 5,1% menos** que homens iguais em todo o resto (IC
# −6,6% a −3,7%); a diferença bruta é de 11,7%. Ela cresce quando se controla a
# escolaridade, porque as mulheres da amostra têm mais mestrado ou doutorado
# (18,7% contra 13,6%). É maior entre sêniores (−8,2%). Como o modelo fixa o
# nível, ele não mede desigualdade nas promoções, só entre quem já chegou ao
# mesmo lugar.
#
# ## 8. Quem ganha menos que o mercado está saindo? ([05](05_satisfacao.ipynb))
#
# Quem ganha abaixo do próprio perfil está mais insatisfeito e procura mais
# emprego: no quinto mais abaixo (mediana de −35%), **44% estão insatisfeitos e
# 49% procuram emprego**, contra 22% e 31% no quinto mais acima. Cada 10%
# abaixo do perfil soma 1,9 ponto de insatisfação e 1,65 ponto de procura.
# **E quem reclama do salário, em geral, tem razão:** 76% dos que citam o
# salário como motivo estão de fato abaixo do perfil. Procurar emprego só no
# exterior é mais comum entre quem já ganha acima do perfil.
#
# ## Para quem está começando
#
# Juntando as respostas: o júnior é quem mais perdeu poder de compra desde
# 2021 (−14%), e o primeiro salto — de júnior a pleno — é o maior da carreira
# (1,6× a 2,1×). A distância entre engenharia ou ciência de dados e análise
# cresce com a senioridade: R$ 0,8 mil entre engenheiro e analista juniores,
# R$ 4,7 mil entre os sêniores. O remoto paga mais em todos os níveis, e o
# prêmio cresce com a senioridade.
#
# ## Limites, de uma vez
#
# - **Amostra autosselecionada** da comunidade Data Hackers; São Paulo é 40%
#   das respostas, e quem responde ficou mais velho e mais estudado ao longo
#   dos anos.
# - **Faixa, não valor.** O topo é aberto (acima de R$ 40.001) e está
#   enchendo: 3,6% das respostas em 2025.
# - **Associação, não causa.** Os controles tiram o que a pesquisa mede;
#   empresa, habilidade e rede de contatos ficam de fora.
# - **O questionário muda.** A harmonização (caderno 01) documenta cada
#   decisão; a opção "salário" como motivo de insatisfação mudou de texto em
#   2024, e a edição de 2019 não perguntou o nível.
#
# ## Método, em uma linha cada
#
# - Seis edições harmonizadas pelo texto da pergunta; valor desconhecido
#   interrompe a carga (`salarios/harmonizar.py`).
# - Mediana agrupada e regressão por intervalo log-normal, testadas em dados
#   simulados (`salarios/estatistica.py`).
# - IPCA do IBGE, novembro de cada ano levado a novembro de 2025.
# - 23 testes; sete defeitos plantados de propósito, todos pegos.
