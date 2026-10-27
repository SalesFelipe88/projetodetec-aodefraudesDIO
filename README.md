Detecção de Fraudes em Cartões de Crédito

Projeto de Machine Learning para identificar transações potencialmente fraudulentas em cartões de crédito, desenvolvido em Python como parte de um desafio prático de Ciência de Dados.

O foco é avaliar modelos em uma base altamente desbalanceada, na qual transações fraudulentas representam uma pequena parcela dos registros. Por isso, a acurácia não é suficiente para avaliar o desempenho: precisão, recall e F1-score da classe fraude são métricas centrais.

## Objetivos

- Explorar e preparar os dados de transações.
- Investigar o desbalanceamento entre transações legítimas e fraudulentas.
- Treinar e comparar Regressão Logística, Random Forest e XGBoost.
- Avaliar os modelos com métricas de classificação e matrizes de confusão.
- Investigar diferentes limiares de decisão.
- Interpretar previsões usando SHAP.
- Documentar limitações e possibilidades de melhoria.

## Dataset

A base contém as colunas `Time`, `Amount`, `Class` e `V1` a `V28`. As variáveis `V1` a `V28` foram transformadas por PCA para proteger informações sensíveis.

O arquivo CSV é carregado diretamente pela URL de origem, sem ser armazenado neste repositório:

[Dataset de transações de cartão de crédito](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv)

A base original possui aproximadamente 284.807 transações e 492 fraudes (cerca de 0,17%). Os resultados efetivos são conferidos no notebook durante a exploração dos dados.

## Tecnologias

- Python
- Pandas e NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- SHAP
- Google Colab / Jupyter Notebook

## Etapas do projeto

1. **Exploração dos dados:** inspeção das dimensões, colunas, tipos, valores ausentes e distribuição da classe-alvo.
2. **Preparação:** criação da variável `LogAmount` com `numpy.log1p`, separação entre variáveis preditoras (`X`) e alvo (`y`), e divisão estratificada dos dados.
3. **Padronização:** uso de `StandardScaler` para a Regressão Logística, ajustando o scaler somente nos dados de treino.
4. **Treinamento:** comparação entre Regressão Logística com pesos de classe, Random Forest e XGBoost com ajuste de peso para a classe minoritária.
5. **Avaliação:** análise de precisão, recall, F1-score, matrizes de confusão e curvas ROC e Precision-Recall.
6. **Explicabilidade:** uso de SHAP para avaliar a importância global das variáveis e contribuições em previsões individuais.

## Resultados exploratórios iniciais

A tabela abaixo registra os resultados do primeiro experimento, usando o limiar padrão de 0,5. Eles são resultados exploratórios e não devem ser confundidos com a avaliação final após a seleção do limiar em validação.

| Modelo | Precisão — fraude | Recall — fraude | F1-score — fraude | Falsos positivos | Fraudes perdidas |
|---|---:|---:|---:|---:|---:|
| Regressão Logística | 6,03% | 91,84% | 0,1131 | 1.403 | 8 |
| Random Forest | 96,00% | 73,47% | 0,8324 | 3 | 26 |
| XGBoost | 78,50% | 85,71% | 0,8195 | 23 | 14 |

### Interpretação

- **Regressão Logística:** apresentou alto recall, mas gerou muitos falsos positivos.
- **Random Forest:** obteve a maior precisão entre os experimentos iniciais e poucos falsos positivos, mas deixou passar mais fraudes.
- **XGBoost:** apresentou um equilíbrio entre precisão e recall, detectando mais fraudes que o Random Forest nesse experimento.

A escolha do modelo depende do custo operacional de investigar alertas e do impacto de deixar fraudes passarem. A acurácia isolada pode ser enganosa em conjuntos altamente desbalanceados.

## Seleção do limiar e avaliação final

O notebook inclui experimentos com diferentes limiares de decisão. Para uma avaliação mais rigorosa, o limiar deve ser escolhido usando um conjunto de validação, sem otimização baseada no conjunto de teste. O teste deve permanecer reservado para a avaliação final.

**Preencher após executar a avaliação final do notebook:**

- Limiar escolhido: `[preencher com o resultado]`
- Precisão da classe fraude: `[preencher]`
- Recall da classe fraude: `[preencher]`
- F1-score da classe fraude: `[preencher]`
- ROC-AUC: `[preencher]`
- Average Precision: `[preencher]`

## Explicabilidade com SHAP

A análise SHAP do XGBoost destacou `V14`, `V4`, `V10` e `V12` entre as variáveis com maior impacto global na amostra analisada. O gráfico beeswarm mostra a distribuição e a direção das contribuições, enquanto o gráfico waterfall detalha a contribuição das variáveis para uma previsão individual.

Como `V1` a `V28` foram transformadas por PCA, sua importância no modelo não permite atribuir diretamente um significado comercial a cada variável. SHAP explica o comportamento do modelo, mas não comprova causalidade nem determina, por si só, que uma transação é fraudulenta.

## Como executar

1. Abra o notebook `deteccao-fraudes-cartao.ipynb` no Google Colab ou Jupyter.
2. Execute as células em ordem.
3. Confirme que o CSV é carregado pela URL e que todas as células terminam sem erros.
4. Salve o notebook com as saídas, tabelas e gráficos.
5. Confira se os resultados descritos neste README correspondem à versão final executada.

As bibliotecas podem ser instaladas no Colab com:

```python
%pip install -q xgboost shap
```

## Limitações e próximos passos

- Comparar undersampling e oversampling sem contaminar validação ou teste.
- Ampliar a busca de hiperparâmetros com validação cruzada.
- Comparar critérios alternativos para escolha do limiar, considerando custos de falsos positivos e falsos negativos.
- Investigar variáveis comportamentais temporais caso estejam disponíveis dados apropriados.
- Documentar e reproduzir a avaliação final com as métricas efetivamente obtidas.

## Autor

**Felipe Sales**

Projeto de estudo para prática de Python, análise de dados e Machine Learning aplicado à prevenção a fraudes.

