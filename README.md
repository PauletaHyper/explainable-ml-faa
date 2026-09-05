# Explainable ML — Burnout & Desempenho Académico

Projeto de Machine Learning (Fundamentos de Aprendizagem Automática, FCUL) que combina **modelação preditiva** com **explicabilidade de modelos**, aplicado a dados de hábitos de estudo e bem-estar de estudantes.

## Problemas abordados

1. **Classificação — Nível de exaustão (burnout):** prever o nível de exaustão de um estudante (Very Low → Very High) com base em atividades extracurriculares e hábitos de vida.
2. **Regressão — Desempenho académico:** prever a nota final de exame com base em hábitos de estudo e saúde.

**Stakeholders identificados:** gabinetes de apoio psicológico e bem-estar (deteção precoce de risco de burnout) e serviços de tutoria académica (sinalização de estudantes em risco de notas baixas).

## Pipeline

- Análise exploratória e tratamento de dados em falta (`student_records_full.csv` / `student_records_missing.csv`)
- Pré-processamento: normalização (StandardScaler / MinMaxScaler / RobustScaler) e imputação de valores em falta
- Treino de modelos de classificação e regressão
- **Explicabilidade** (notebook dedicado `FAA2526-TP09-Explainability.ipynb`):
  - Regras de decisão de Decision Trees (modelo interpretável por natureza)
  - **LIME** — explicações locais por instância via modelo substituto
  - **SHAP** — atribuição de contribuição por feature (Shapley values)
  - Comparação entre os três métodos e discussão de divergências

## Estrutura

```
FAA_23.ipynb                        → Notebook principal: EDA, pré-processamento e modelos
FAA2526-TP09-Explainability.ipynb   → Explicabilidade (Decision Tree rules, LIME, SHAP)
Dados_Projeto/                      → Datasets do projeto
main.py                             → Definição do problema
```

## Stack

Python · Jupyter · scikit-learn · SHAP · LIME · pandas
