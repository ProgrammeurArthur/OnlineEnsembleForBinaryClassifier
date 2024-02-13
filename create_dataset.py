import pandas as pd
from river.datasets import synth
import os

# Criar listas para armazenar os dados
data_list = []
target_list = []

# Gerar os dados do conjunto de dados de conceito de derivação
dataset = synth.ConceptDriftStream(
    stream=synth.SEA(seed=42, variant=0),
    drift_stream=synth.SEA(seed=42, variant=1),
    seed=1, position=250000, width=2000
)

# Coletar os dados do dataset e salvá-los nas listas
for x, y in dataset.take(500000):
    data_list.append(x)
    target_list.append(y)

# Criar um DataFrame com os dados
df = pd.DataFrame(data_list)
df['target'] = target_list

#criar diretorio dentro do riverdata
dir_path_bd='/home/arthur/river_data'
dataset='dataset_Conceptdrift'
dir_path_model = os.path.join(dir_path_bd, dataset)
os.makedirs(dir_path_model, exist_ok=True)

# Salvar o DataFrame em um arquivo CSV
csv_file = os.path.join(dir_path_model, f'{dataset}.csv')
df.to_csv(csv_file, index=False)