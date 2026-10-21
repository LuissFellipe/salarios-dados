"""Que coluna de cada edição responde a cada campo harmonizado.

Mapeado pelo texto da pergunta e conferido ano a ano, nunca pela letra: a
pergunta "Atualmente qual a sua forma de trabalho?" é P2_q em 2021, P2_p em
2022, P2_r em 2023, 2.r em 2024 e 2.q em 2025. Os nomes de 2021 a 2023 são
tuplas escritas como texto, com um espaço dentro: "('P2_h ', 'Faixa salarial')".

"motivo_salario" é uma das opções de "qual o motivo da sua insatisfação",
perguntada só a quem se disse insatisfeito. O texto da opção mudou: até 2023,
"Salário atual não corresponde ao mercado"; de 2024 em diante,
"Remuneração/Salário".
"""

from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
BRUTOS = RAIZ / "dados/brutos"

EDICOES = {
    2019: {
        "arquivo": "pesquisa-data-hackers-2019/datahackers-survey-2019-anonymous-responses.csv",
        "colunas": {
            "idade": "('P1', 'age')", "genero": "('P2', 'gender')", "uf": "('P5', 'living_state')",
            "regiao": "('D1', 'living_macroregion')", "ensino": "('P8', 'degreee_level')",
            "formacao": "('D3', 'anonymized_degree_area')", "situacao": "('P10', 'job_situation')",
            "setor": "('D4', 'anonymized_market_sector')", "porte": "('P12', 'workers_number')",
            "gestor": "('P13', 'manager')", "cargo_gestor": "('D5', 'anonymized_manager_level')",
            "cargo": "('D6', 'anonymized_role')", "faixa": "('P16', 'salary_range')",
            "exp_dados": "('P17', 'time_experience_data_science')",
        },
    },
    2021: {
        "arquivo": "state-of-data-2021/State of Data 2021 - Dataset - Pgina1.csv",
        "colunas": {
            "idade": "('P1_a ', 'Idade')", "genero": "('P1_b ', 'Genero')", "uf": "('P1_e_a ', 'uf onde mora')",
            "regiao": "('P1_e_b ', 'Regiao onde mora')", "ensino": "('P1_h ', 'Nivel de Ensino')",
            "formacao": "('P1_i ', 'Área de Formação')", "situacao": "('P2_a ', 'Qual sua situação atual de trabalho?')",
            "setor": "('P2_b ', 'Setor')", "porte": "('P2_c ', 'Numero de Funcionarios')",
            "gestor": "('P2_d ', 'Gestor?')", "cargo_gestor": "('P2_e ', 'Cargo como Gestor')",
            "cargo": "('P2_f ', 'Cargo Atual')", "nivel": "('P2_g ', 'Nivel')", "faixa": "('P2_h ', 'Faixa salarial')",
            "exp_dados": "('P2_i ', 'Quanto tempo de experiência na área de dados você tem?')",
            "satisfeito": "('P2_k ', 'Você está satisfeito na sua empresa atual?')",
            "mudar": "('P2_n ', 'Você pretende mudar de emprego nos próximos 6 meses?')",
            "motivo_salario": "('P2_l_b ', 'Salário atual não corresponde ao mercado')",
            "modelo": "('P2_q ', 'Atualmente qual a sua forma de trabalho?')",
        },
    },
    2022: {
        "arquivo": "state-of-data-2022/State_of_data_2022.csv",
        "colunas": {
            "idade": "('P1_a ', 'Idade')", "genero": "('P1_b ', 'Genero')", "uf": "('P1_i_1 ', 'uf onde mora')",
            "regiao": "('P1_i_2 ', 'Regiao onde mora')", "ensino": "('P1_l ', 'Nivel de Ensino')",
            "formacao": "('P1_m ', 'Área de Formação')", "situacao": "('P2_a ', 'Qual sua situação atual de trabalho?')",
            "setor": "('P2_b ', 'Setor')", "porte": "('P2_c ', 'Numero de Funcionarios')",
            "gestor": "('P2_d ', 'Gestor?')", "cargo_gestor": "('P2_e ', 'Cargo como Gestor')",
            "cargo": "('P2_f ', 'Cargo Atual')", "nivel": "('P2_g ', 'Nivel')", "faixa": "('P2_h ', 'Faixa salarial')",
            "exp_dados": "('P2_i ', 'Quanto tempo de experiência na área de dados você tem?')",
            "satisfeito": "('P2_k ', 'Você está satisfeito na sua empresa atual?')",
            "mudar": "('P2_n ', 'Você pretende mudar de emprego nos próximos 6 meses?')",
            "motivo_salario": "('P2_l_2 ', 'Salário atual não corresponde ao mercado')",
            "modelo": "('P2_p ', 'Atualmente qual a sua forma de trabalho?')",
        },
    },
    2023: {
        "arquivo": "state-of-data-brazil-2023/State_of_data_BR_2023_Kaggle - df_survey_2023.csv",
        "colunas": {
            "idade": "('P1_a ', 'Idade')", "genero": "('P1_b ', 'Genero')", "uf": "('P1_i_1 ', 'uf onde mora')",
            "regiao": "('P1_i_2 ', 'Regiao onde mora')", "ensino": "('P1_l ', 'Nivel de Ensino')",
            "formacao": "('P1_m ', 'Área de Formação')", "situacao": "('P2_a ', 'Qual sua situação atual de trabalho?')",
            "setor": "('P2_b ', 'Setor')", "porte": "('P2_c ', 'Numero de Funcionarios')",
            "gestor": "('P2_d ', 'Gestor?')", "cargo_gestor": "('P2_e ', 'Cargo como Gestor')",
            "cargo": "('P2_f ', 'Cargo Atual')", "nivel": "('P2_g ', 'Nivel')", "faixa": "('P2_h ', 'Faixa salarial')",
            "exp_dados": "('P2_i ', 'Quanto tempo de experiência na área de dados você tem?')",
            "satisfeito": "('P2_k ', 'Você está satisfeito na sua empresa atual?')",
            "mudar": "('P2_n ', 'Você pretende mudar de emprego nos próximos 6 meses?')",
            "motivo_salario": "('P2_l_2 ', 'Salário atual não corresponde ao mercado')",
            "modelo": "('P2_r ', 'Atualmente qual a sua forma de trabalho?')",
        },
    },
    2024: {
        "arquivo": "state-of-data-brazil-20242025/Final Dataset - State of Data 2024 - Kaggle - df_survey_2024.csv",
        "colunas": {
            "idade": "1.a_idade", "genero": "1.b_genero", "uf": "1.i.1_uf_onde_mora", "regiao": "1.i.2_regiao_onde_mora",
            "ensino": "1.l_nivel_de_ensino", "formacao": "1.m_área_de_formação", "situacao": "2.a_situação_de_trabalho",
            "setor": "2.b_setor", "porte": "2.c_numero_de_funcionarios", "gestor": "2.d_atua_como_gestor",
            "cargo_gestor": "2.e_cargo_como_gestor", "cargo": "2.f_cargo_atual", "nivel": "2.g_nivel",
            "faixa": "2.h_faixa_salarial", "exp_dados": "2.i_tempo_de_experiencia_em_dados",
            "satisfeito": "2.k_satisfeito_atualmente", "mudar": "2.n_planos_de_mudar_de_emprego_6m",
            "motivo_salario": "2.l.1_Remuneração/Salário",
            "modelo": "2.r_modelo_de_trabalho_atual",
        },
    },
    2025: {
        "arquivo": "state-of-data-brazil-2025-2026/Final Dataset - State of Data 2025-2026 - Kaggle.csv",
        "colunas": {
            "idade": "1.a_idade", "genero": "1.b_genero", "uf": "1.i.1_uf_onde_mora", "regiao": "1.i.2_regiao_onde_mora",
            "ensino": "1.l_nivel_de_ensino", "formacao": "1.m_área_de_formação", "situacao": "2.a_situação_de_trabalho",
            "setor": "2.b_setor", "porte": "2.c_numero_de_funcionarios", "gestor": "2.d_atua_como_gestor",
            "cargo_gestor": "2.e_cargo_como_gestor", "cargo": "2.f_cargo_atual", "nivel": "2.g_nivel",
            "faixa": "2.h_faixa_salarial", "exp_dados": "2.i_tempo_de_experiencia_em_dados",
            "satisfeito": "2.k_satisfeito_atualmente", "mudar": "2.n_planos_de_mudar_de_emprego_6m",
            "motivo_salario": "2.l.1_Remuneração/Salário",
            "modelo": "2.q_modelo_de_trabalho_atual",
        },
    },
}
