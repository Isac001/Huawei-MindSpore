# Importações dos submódulos de dataset e o matplot p/ plot de representações visuais.
import mindspore.dataset.vision as vision
import mindspore.dataset as ds
import matplotlib.pyplot as plt

# Variável contendo o caminho absoluto p/ o diretorio com os dados.
dataset_dir = "C:/Users/isacD/mind_spore/mnist"

# Lista de transformações que serão aplicadas em cada imagem no conjunto de dados
basic_tranform = [
    vision.Resize((32, 32)), # Redimensiona cada imagem p/ o padrão 32x32 (Trinta e dois pixels por Trinta e dois pixels)
    vision.Normalize((0.5,), (0.5,)), # Normalização da imagem, onde o (0.5,) é a média e o desvio padrão
    vision.HWC2CHW() # Converte a imagem no formtado [Altura, Largura, Canais] (HWC) que é o formato preferido por muitos frame de DP p/ processamento eficiente
]

# Variável que carrega o conjunto de dados MNIST p/ treinamento, o "usage=" é onde é definido como conjunto de treino
mnist_train = ds.MnistDataset(dataset_dir=dataset_dir, usage="train")

# Variável que aplica as tranformações especificadas anteriormente a cada amostra de dados
mnist_train = mnist_train.map(operations=basic_tranform) # O 'map' é usado para aplicar operações em cada elemento de um Dataset no MindSpore.

''' Laço 'for' p/ integrar sobre o conjunto de dados. 
    Onde se cria um interador para percorrer o conjunto de dados (mnist_train.create_dict_iterator),
    Com o 'output_numpy=True' retornando os dados como arrays do NumPy em vez de tensores para facilitar o plot/visualização com o matplotlib
    E o 'num_epoches=1' definindo o número de épocas que os dados vão passar no laço, no caso 1 (uma) '''
for data in mnist_train.create_dict_iterator(output_numpy=True, num_epochs=1):

    # Visualização de imagem

    # Linnha onde pega a imagem tranformada, seleciona o primeiro canal e define o mapeamento de cores como 'gray'/'cinca', exibindo a imagem em escala cinza
    plt.imshow(data['image'][0], cmap='gray')

    # Linha que define o título do gráfico com base no rótulo da imagem (valor de 0 a 9)
    plt.title(f"Rótulo: {data['label']}")

    # Linha que exibe a imagem na tela usando o matplot
    plt.show()

    # Linha que quebra o loop após a exibir a primeira imagem
    break