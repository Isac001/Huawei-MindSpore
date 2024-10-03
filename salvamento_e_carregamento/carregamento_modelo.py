import mindspore as ms
from mindspore import nn, Tensor, context
from mindspore.train.serialization import load_checkpoint

# Configurar o ambiente de execução para CPU
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# Definir a mesma arquitetura da rede neural usada no salvamento
class SimpleNet(nn.Cell):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Dense(784, 128)  # Camada totalmente conectada com 784 entradas e 128 saídas
        self.relu = nn.ReLU()          # Função de ativação ReLU
        self.fc2 = nn.Dense(128, 10)   # Camada totalmente conectada com 128 entradas e 10 saídas

    def construct(self, x):
        x = self.relu(self.fc1(x))     # Aplicar a camada 1 e a ReLU
        return self.fc2(x)             # Aplicar a camada 2 para gerar a saída

# Criar um novo objeto da rede neural
loaded_net = SimpleNet()

# Carregar os parâmetros salvos no arquivo de checkpoint
load_checkpoint("simple_model.ckpt", net=loaded_net)

print("Modelo carregado com sucesso a partir do arquivo 'simple_model.ckpt'.")

# Realizar uma inferência com um exemplo de entrada aleatório
sample_input = Tensor(ms.numpy.randn(1, 784).astype(ms.float32))  # Entrada com formato (1, 784)
output = loaded_net(sample_input)  # Passar o dado pelo modelo carregado

# Exibir a saída da inferência
print("Resultado da inferência com o modelo carregado:", output.asnumpy())