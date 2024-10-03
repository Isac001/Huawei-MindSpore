# Importando submódulo de conjunto de dados p/ carregamento
import mindspore.dataset as ds

# Caminho absoluto de onde está os dados
dataset_dir = "C:/Users/isacD/mind_spore/mnist"

# Variavel que carrega o conjunto de dados MNIST p/ treinamento
mnist_train = ds.MnistDataset(dataset_dir=dataset_dir, usage="train")

# Exibição de número de amostras no conjunto de dados
print(f"Número de amostras no conjunto de treinamento é: {mnist_train.get_dataset_size()}")