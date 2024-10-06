# Rede Neural Perceptron Multicamadas (Multi-Layer Perceptron - MLP)

import mindspore as ms  # Biblioteca principal para construção de redes neurais e machine learning
import numpy as np  # Biblioteca para manipulação de arrays e geração de dados numéricos
from mindspore import nn, Model, Tensor, context  # Importa nn para definir a rede neural, Model para configurar o modelo, Tensor como estrutura de dados e context para definir o ambiente de execução
from mindspore.train.callback import Callback  # Importa a classe base para definir callbacks que monitoram o treinamento
from mindspore import dataset as ds  # Módulo para criar e manipular datasets no formato MindSpore
import matplotlib.pyplot as plt  # Biblioteca para visualização e criação de gráficos

# Configuração do ambiente de execução para MindSpore
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")  # Define o modo gráfico para execução mais eficiente e usa a CPU como dispositivo

# Função para gerar dados de círculos para classificação
def generate_circule_data(n_samples=500):
    np.random.seed(0)  # Define a semente para que os resultados sejam reproduzíveis
    radius = np.random.rand(n_samples) * 2  # Gera raios aleatórios entre 0 e 2
    angle = np.random.rand(n_samples) * 2 * np.pi  # Gera ângulos aleatórios entre 0 e 2π (círculo completo)
    x = radius * np.cos(angle)  # Calcula a coordenada X a partir do raio e ângulo
    y = radius * np.sin(angle)  # Calcula a coordenada Y a partir do raio e ângulo

    labels = (radius > 1).astype(int)  # Define as classes: 0 se raio <= 1, 1 se raio > 1
    data = np.column_stack((x, y))  # Combina as coordenadas X e Y em um array de 2 colunas
    return data, labels  # Retorna os dados (coordenadas) e os rótulos (classes)

# Gera 500 pontos de dados e seus rótulos
data, labels = generate_circule_data(500)

# Visualização dos dados gerados em um gráfico de dispersão
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='coolwarm')  # Plota os pontos coloridos de acordo com suas classes
plt.xlabel("X-axis")  # Rótulo para o eixo X
plt.ylabel("Y-axis")  # Rótulo para o eixo Y
plt.title("Conjunto de dados de treinamento")  # Título do gráfico
plt.show()  # Exibe o gráfico com os dados gerados

# Definição da rede neural simples usando MindSpore
class SimpleNN(nn.Cell):  # Define a classe da rede neural herdando de nn.Cell

    def __init__(self):
        super(SimpleNN, self).__init__()  # Inicializa a classe base nn.Cell
        self.fc1 = nn.Dense(2, 4)  # Primeira camada totalmente conectada (2 entradas -> 4 saídas)
        self.relu = nn.ReLU()  # Função de ativação ReLU (retorna 0 para valores negativos e valor original para positivos)
        self.fc2 = nn.Dense(4, 2)  # Segunda camada totalmente conectada (4 entradas -> 2 saídas correspondentes às duas classes)

    def construct(self, x):  # Define como os dados passam pela rede
        x = self.relu(self.fc1(x))  # Aplica a primeira camada e a função de ativação ReLU
        return self.fc2(x)  # Retorna a saída da segunda camada (logits)

# Instancia a rede neural
net = SimpleNN()

# Prepara o dataset para treinamento com os dados e rótulos gerados
train_dataset = ds.NumpySlicesDataset({"data": data.astype(np.float32), "label": labels.astype(np.int32)}, shuffle=True).batch(16)
# Converte o conjunto de dados para um formato MindSpore, faz shuffle nos dados e agrupa em batches de 16 amostras

# Definir a função de perda e o otimizador
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')  # Função de perda para classificação (softmax + entropia cruzada)
optimizer = nn.Adam(net.trainable_params(), learning_rate=0.01)  # Otimizador Adam com taxa de aprendizado de 0.01

# Configuração do modelo com rede neural, função de perda e otimizador
model = Model(network=net, loss_fn=loss_fn, optimizer=optimizer, metrics={'Accuracy': nn.Accuracy()})

# Definição de um Callback para monitorar a perda durante o treinamento
class LossMonitor(Callback):

    def __init__(self):
        super(LossMonitor, self).__init__()  # Inicializa a classe base Callback
        self.losses = []  # Cria uma lista para armazenar as perdas

    def set_end(self, run_context):  # Método que é chamado no final de cada etapa
        loss = run_context.original_args().net_outputs.asnumpy()  # Obtém a perda atual como valor NumPy
        self.losses.append(loss)  # Adiciona a perda à lista
        print(f"Step {run_context.original_args().cur_step_num}, Loss: {loss:.4f}")  # Imprime a perda atual no console

# Instancia o monitor de perdas
loss_monitor = LossMonitor()

# Inicia o treinamento do modelo
print('Iniciando treinamento')  # Exibe mensagem no console indicando o início do treinamento
model.train(epoch=10, train_dataset=train_dataset, callbacks=[loss_monitor], dataset_sink_mode=False)  
# Treina o modelo por 10 épocas com o dataset preparado, usando o LossMonitor como callback

# Função para plotar as fronteiras de decisão aprendidas pela rede
def plot_decision_boundary(model, data, labels, title=""):
    # Definir os limites do gráfico
    x_min, x_max = data[:, 0].min() - 0.5, data[:, 0].max() + 0.5
    y_min, y_max = data[:, 1].min() - 0.5, data[:, 1].max() + 0.5
    # Cria uma malha de pontos no espaço 2D
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    mesh_data = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)  # Combina as coordenadas em uma única matriz
    predictions = model.predict(Tensor(mesh_data))  # Realiza predições para cada ponto da malha usando o modelo treinado
    pred_classes = np.argmax(predictions.asnumpy(), axis=1)  # Converte os logits para classes (0 ou 1)
    # Plota as regiões correspondentes a cada classe
    plt.contourf(xx, yy, pred_classes.reshape(xx.shape), cmap='coolwarm', alpha=0.7)  
    # Plota os pontos de dados reais sobre as fronteiras de decisão
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='coolwarm')  
    plt.title(title)  # Define o título do gráfico
    plt.show()  # Exibe o gráfico

# Plotar as fronteiras de decisão após o treinamento do modelo
plot_decision_boundary(model, data, labels, title="Fronteira de Decisão Após Treinamento")
