import mindspore as ms
from mindspore import nn, context
from mindspore.train.serialization import save_checkpoint

# Configurar o ambiente de execução para CPU
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# Definir uma rede neural simples
class SimpleNet(nn.Cell):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Dense(784, 128)  # Camada totalmente conectada com 784 entradas e 128 saídas
        self.relu = nn.ReLU()          # Função de ativação ReLU
        self.fc2 = nn.Dense(128, 10)   # Camada totalmente conectada com 128 entradas e 10 saídas

    def construct(self, x):
        x = self.relu(self.fc1(x))     # Aplicar a camada 1 e a ReLU
        return self.fc2(x)             # Aplicar a camada 2 para gerar a saída

# Instanciar a rede
net = SimpleNet()

# Treinar a rede (simulação de treino sem dados reais para simplificação)
# Aqui, vamos apenas inicializar a rede sem dados para criar o modelo básico.

# Salvar o modelo em um arquivo de checkpoint
save_checkpoint(net, "simple_model.ckpt")

print("Modelo salvo com sucesso no arquivo 'simple_model.ckpt'.")