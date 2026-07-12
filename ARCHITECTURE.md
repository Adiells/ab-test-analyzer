# Arquitetura do Projeto

## Overview

O **AB Test Analyzer** é uma plataforma analítica AI-Native desenvolvida para automatizar de ponta a ponta a análise de experimentos A/B (como testes de cashback) conduzidos por equipes de Growth e Produto.

A aplicação recebe dados brutos, executa uma pipeline de análise altamente estruturada e produz uma recomendação de negócio e um relatório executivo fundamentado. O projeto foi projetado com uma clara separação de responsabilidades: cálculos estatísticos e matemáticos são estritamente determinísticos, enquanto a Inteligência Artificial é empregada na interface de interação, na contextualização de intenções e na tradução das descobertas estatísticas em relatórios de negócios claros.

---

## Objetivos Arquiteturais

* **Automação de Ponta a Ponta**: Reduzir o tempo gasto em análises manuais, processando os dados brutos até a recomendação final de forma automatizada.
* **Separabilidade de Negócio e IA**: Garantir que as decisões de negócio e cálculos de métricas sejam 100% reprodutíveis e determinísticos, sem risco de alucinações da IA.
- **Interface Inteligente (AI-Native)**: Permitir que usuários acessem e analisem experimentos utilizando linguagem natural por meio de chat interativo.
* **Persistência Consolidada**: Registrar automaticamente cada análise no Google Sheets (com contingência em arquivo CSV local).
* **Modularidade e Reutilização**: Facilitar a adição de novos modelos estatísticos, novas fontes de dados ou a substituição do modelo LLM sem impactos no núcleo do sistema.

---

## Arquitetura Geral do Sistema

O sistema é dividido em três camadas principais:
1. **Camada de Interface**: Traduz as intenções do usuário (linguagem natural ou chamadas de ferramentas) na seleção dos dados.
2. **Orquestrador e Pipeline**: O fluxo de processamento de dados brutas até a estruturação das decisões.
3. **Serviços de Integração e Saída**: Geração de relatórios, plotagem de gráficos e persistência de histórico.

```text
                  +-----------------------------------------+
                  |                 Usuário                 |
                  +---------------------+-------------------+
                                        | (Linguagem Natural)
                                        ▼
                  +-----------------------------------------+
                  |         Camada de Interface AI          |
                  |                                         |
                  |  [teste-ab] (CLI Chat / cli_chat.py)    |
                  |       L Memória de Sessão & Gemini      |
                  |                                         |
                  |  [ai_interface.py] (External Tool API)  |
                  +---------------------+-------------------+
                                        | (Dataset Resolvido)
                                        ▼
                  +-----------------------------------------+
                  |        Pipeline Orquestradora           |
                  |        (Orchestrator.run())             |
                  +---------------------+-------------------+
                                        |
     ┌──────────────────┬───────────────┼───────────────┬──────────────────┐
     |                  |               |               |                  |
     ▼                  ▼               ▼               ▼                  ▼
+--------+         +--------+      +---------+     +---------+        +---------+
|Ingestão|         |Valida- |      |Pré-pro- |     |Cálculo  |        |Análise  |
|Normali-|         |ção e   |      |cessamen-|     |de       |        |Estatís- |
|zação   |         |Sanidade|      |to       |     |Métricas |        |tica     |
+--------+         +--------+      +---------+     +---------+        +---------+
     |                  |               |               |                  |
     └──────────────────┴───────────────┼───────────────┴──────────────────┘
                                        ▼
                  +-----------------------------------------+
                  |             Motor de Decisão            |
                  |             (Regras Hardcoded)          |
                  +---------------------+-------------------+
                                        | (Decisões Estruturadas)
                                        ▼
                  +-----------------------------------------+
                  |         Redação do Relatório            |
                  |    (Narrativa IA + Plots + Markdown)    |
                  +---------------------+-------------------+
                                        |
                   ┌────────────────────┴────────────────────┐
                   ▼                                         ▼
     +---------------------------+             +---------------------------+
     | Planilha do Google Sheets |             | Relatórios e Plots Locais |
     | (Ou Contingência CSV local)|             | (Markdown + Gráficos PNG) |
     +---------------------------+             +---------------------------+
```

---

## Detalhamento das Etapas da Pipeline

### 1. Ingestão e Normalização
Lê o dataset de entrada (suporta arquivos CSV) e padroniza a formatação do conteúdo básico (nomes de colunas, remoção de espaços e caixa de caracteres).

### 2. Validação
Executa testes sanitários rígidos no dataset normalizado:
* Verifica presença de colunas obrigatórias.
* Garante consistência de tipos.
* Detecta anomalias graves (ex: valores monetários negativos, múltiplos parceiros por arquivo ou volume nulo de compradores).

### 3. Pré-processamento e Enriquecimento
Converte campos monetários para o formato numérico operacional, realiza formatações temporais e cria colunas derivadas necessárias para as métricas financeiras (ex: Receita Líquida derivada da diferença entre comissão ganha e cashback distribuído).

### 4. Métricas de Negócio
Computa de forma determinística os indicadores de performance agrupados por variante:
* GMV Total (Volume de vendas)
* Receita Líquida Total
* Compradores Totais
* Margens e taxas financeiras (Margem Líquida, Taxa de Cashback e Taxa de Comissão)

### 5. Análise Estatística (SciPy)
Analisa a variabilidade dos dados e aplica testes de hipóteses de acordo com o design do experimento:
* Teste de Normalidade (Shapiro-Wilk) nas amostras.
* Teste de Comparação de Médias para 2 grupos (Teste T Independente).
* Teste Omnibus para múltiplos grupos (ANOVA de uma via).
* Comparações em Pares corrigidas pelo método de Bonferroni.
* Cálculo do Tamanho do Efeito (Cohen's d).

### 6. Motor de Decisão
Aplica regras estáticas e limites de significância configurados (alfa padrão = 0.05) para emitir uma recomendação definitiva de negócio. O motor classifica o teste entre:
* `scale_treatment`: Escalar variante vencedora para 100% do tráfego.
* `keep_control`: Manter controle (sem efeito prático).
* `collect_more_data`: Coletar mais dados (sem significância estatística, mas com indício de efeito).
* `inconclusive`: Inconclusivo (evidência insuficiente).

### 7. Redação e Visualizações
Gera os artefatos de entrega:
* Redige gráficos comparativos em PNG das variantes na pasta `output/plots/`.
* Combina as tabelas numéricas, testes estatísticos e a narrativa contextualizada em um relatório executivo final em Markdown.

### 8. Persistência e Registro
Publica a linha consolidada do teste na planilha ativa do Google Sheets. Caso falhe ou não existam credenciais válidas configuradas, desvia a execução para gravação no arquivo CSV local de contingência (`output/experiment_log.csv`).

---

## Organização do Código-Fonte

O código-fonte está estruturado de acordo com o padrão de separação de domínios:

```text
src/
├── core/                        # Regras e lógica analítica pura
│   ├── ingestion/               # Leitura e parsing de fontes de dados
│   ├── normalization/           # Higienização de dados
│   ├── validation/              # Garantias de qualidade de dados
│   ├── preprocessing/           # Transformações operacionais de dados
│   ├── metrics/                 # Métricas de negócio determinísticas
│   ├── statistics/              # Modelagem estatística
│   └── decision/                # Decisão lógica estruturada
├── integrations/                # Comunicação externa (Google Sheets)
├── llm/                         # Prompting e geração de narrativa de IA
├── reporting/                   # Visualizações e formatação de Markdown
└── cli_chat.py                  # Camada conversacional inteligente (Interface)
```

---

## Papel da Inteligência Artificial (LLM)

No **AB Test Analyzer**, o Large Language Model (Gemini) possui um papel estrito e bem delimitado, mantendo a confiabilidade analítica da plataforma:

1. **Roteamento de Intenções e Resolução Conversacional**:
   * O LLM analisa as requisições de linguagem natural e o contexto da sessão (histórico de chat) para determinar ações (ex: analisar dataset, sair, pedir ajuda).
   * Resolve referências contextuais (ex: *"analise o segundo"*, *"mostre o teste anterior"*) pareando-as com os datasets reais descobertos na pasta `input/`.
   * Redige de forma amigável as perguntas de esclarecimento e desambiguação para o usuário.
2. **Geração de Relatórios Executivos**:
   * A IA recebe os resultados estatísticos consolidados e as métricas determinísticas prontas.
   * Sua função é puramente traduzir estes dados em uma explicação de negócios em português, estruturando os resultados de forma clara para stakeholders não técnicos.

**A IA nunca calcula métricas de negócio, médias ou valores de p-valor, garantindo zero alucinações numéricas.**

---

## Escalabilidade e Extensibilidade

A modularidade do sistema permite fácil evolução em diversas direções:
* **Novos formatos de dados**: Substituição simplificada ou extensão do módulo de ingestão para ler bancos de dados SQL, APIs JSON ou arquivos de dados parquet.
* **Novos algoritmos estatísticos**: Inclusão de testes não paramétricos (como Mann-Whitney U) ou modelos Bayesianos estendendo o módulo `core/statistics`.
* **Outros canais de Integração**: Adição de novos exportadores na pasta `integrations` para plataformas como Slack, Discord, Notion ou e-mail corporativo.
* **Substituição de LLM**: A interface com LLM é isolada sob um serviço único, facilitando a troca por outros provedores ou modelos locais sem impactar a pipeline analítica.
