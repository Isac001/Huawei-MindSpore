import mindspore as ms
from mindspore import nn, Tensor, context, Model
from mindspore.train.callback import Callback
from mindspore import dataset as ds
import matplotlib.pyplot as plt
import numpy as np

# Configurar o ambiente de execução
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# 1. Construção da Rede Neural
class SimpleNet(nn.Cell):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Dense(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(128, 10)

    def construct(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

net = SimpleNet()

# 2. Geração de Dados de Treinamento e Teste
def generate_data(n_samples=1000, test_split=0.2):
    np.random.seed(0)
    data = np.random.randn(n_samples, 784).astype(np.float32)
    labels = (np.random.rand(n_samples) * 10).astype(np.int32)
    split_index = int(n_samples * (1 - test_split))
    train_data, test_data = data[:split_index], data[split_index:]
    train_labels, test_labels = labels[:split_index], labels[split_index:]
    return train_data, train_labels, test_data, test_labels

train_data, train_labels, test_data, test_labels = generate_data()

# Criar dataset MindSpore
train_dataset = ds.NumpySlicesDataset({"data": train_data, "label": train_labels}, shuffle=True).batch(32)
test_dataset = ds.NumpySlicesDataset({"data": test_data, "label": test_labels}, shuffle=False).batch(32)

# 3. Definição de Perda e Otimizador
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
net_opt = nn.Adam(net.trainable_params(), learning_rate=0.01)

# 4. Configuração do Modelo
model = Model(network=net, loss_fn=net_loss, optimizer=net_opt, metrics={"Accuracy": nn.Accuracy()})

# 5. Callback Personalizado para Monitorar Perdas
class LossHistory(Callback):
    """Callback para armazenar a perda durante o treinamento."""
    def __init__(self):
        super(LossHistory, self).__init__()
        self.losses = []

    def step_end(self, run_context):
        """Método chamado ao final de cada etapa."""
        cb_params = run_context.original_args()
        current_loss = cb_params.net_outputs.asnumpy()
        self.losses.append(current_loss)
        print(f"Step {cb_params.cur_step_num}, Loss: {current_loss}")

# Instanciar o monitor de perdas personalizado
loss_history = LossHistory()

# 6. Treinamento
print("Iniciando o treinamento...")
model.train(epoch=10, train_dataset=train_dataset, callbacks=[loss_history], dataset_sink_mode=False)

# 7. Avaliação
print("Avaliando no conjunto de teste...")
acc = model.eval(test_dataset, dataset_sink_mode=False)
print(f"Acurácia no conjunto de teste: {acc['Accuracy']:.4f}")

# 8. Visualização do Processo
# Vamos plotar as perdas registradas durante o treinamento
plt.figure(figsize=(10, 5))
plt.plot(loss_history.losses, label='Loss over Epochs')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training Loss over Time')
plt.legend()
plt.grid()
plt.show()

# 9. Inferência e Visualização de Saídas de Teste
sample_data = test_data[:5]
sample_labels = test_labels[:5]

# Converter os dados para o formato Tensor
sample_data_tensor = Tensor(sample_data, ms.float32)

# Realizar inferência usando a rede treinada
predictions = model.predict(sample_data_tensor)
predicted_labels = np.argmax(predictions.asnumpy(), axis=1)

# Exibir as previsões no terminal
print("Resultados de Inferência para Amostras de Teste:")
for i in range(len(sample_data)):
    print(f"Exemplo {i + 1}: Valor Real: {sample_labels[i]}, Predição: {predicted_labels[i]}")