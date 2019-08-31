# Time-Series-Prediction-Algorithms
Algoritmos de Predição de Séries Temporais

Este repositório contém a implementação de três algoritmos de predição de séries temporais na linguagem de programação Python, versão 3. São eles: Rede Neural Recorrente, Auto-Regressive Integrated Moving Average (ARIMA) e Support Vector Regression (SVR).

A comparação dos algoritmos foi apresentada no seguinte artigo: 

S. Oliveira, J. Kniess, R. Parpinelli e W. Castañeda, “Predição de Séries Temporais em Internet das Coisas com Redes Neurais Recorrentes”, In: 50o Simpósio Brasileiro de Pesquisa Operacional (SBPO), 2018. (OLIVEIRA et al., 2018).

Link: http://proceedings.science/sbpo/papers/predicao-de-series-temporais-em-internet-das-coisas-com-redes-neurais-recorrentes?lang=pt-br.

Para executar cada algoritmo, é necessário atentar para as bibliotecas necessárias, como: TensorFlow, Pandas, Numpy, MatPlotLib, dentre outras. Além disso, o caminho do dataset deve ser passado como parâmetro logo após a chamada do arquivo. Por exemplo: "python RNN.py DATASETS/1.txt".

Especificamente no caso da Rede Neural Recorrente, são realizadas 10 execuções para que seja computada a média e desvio padrão das métricas avaliadas (Erro Absoluto Médio, Coeficiente de Determinação e Tempo de Execução). Os resultados são exportados para os arquivos em formato JSON. 
