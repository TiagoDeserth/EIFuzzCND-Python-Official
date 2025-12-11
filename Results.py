import os
import pandas as pd
import numpy as np
from typing import List
import csv
from Evaluation.ResultsForExample import ResultsForExample
from Output.HandlesFiles import HandlesFiles
from Output.LineChart import plot_line_chart
from scipy.io import arff
from DebugLogger import DebugLogger
from ClassMapper import ClassMapper

DIVISOR = 1000

def count_lines_in_csv(file_path: str) -> int:
    df = pd.read_csv(file_path, encoding="latin1")
    return len(df) -1

'''
def check_last_column(path_to_file: str) -> List[float]:

    mapper = ClassMapper()
    num_classes_treino = mapper.get_num_classes()

    df = pd.read_csv(path_to_file, encoding="latin1")
    last_column = df.iloc[:, -1].dropna().unique()

    # !last_column_values = list(last_column.astype(str))

    # !result = [float(i) for i in range(len(last_column_values))]

    #DebugLogger.log(f"[DEBUG check_last_column] Classes no treino: {result}")
    #DebugLogger.log(f"[DEBUG check_last_column] Total: {len(result)} classes")

    #return []
    #! return result

    return sorted(list(last_column))
'''

def check_last_column(dataset: str, caminho: str) -> List[float]:
    '''
    Carrega as classes do arquivo de treino ARFF e retorna os índices
    mapeados consistentemente com o ClassMapper.
    '''

    train_path = os.path.join(caminho, f"{dataset}-train.arff")
    
    mapper = ClassMapper()
    class_mapping = mapper.get_mapping()
    
    # Carregar treino
    data_arff, meta = arff.loadarff(train_path)
    df = pd.DataFrame(data_arff)
    class_col = df.columns[-1]
    
    # Aplicar mapeamento
    df[class_col] = df[class_col].apply(
        lambda x: class_mapping.get(x.decode() if isinstance(x, bytes) else x, -1.0)
    )
    
    # Retornar classes únicas do treino
    classes_treino = sorted(df[class_col].dropna().unique())
    
    DebugLogger.log(f"[DEBUG check_last_column] Classes no treino (mapeadas): {classes_treino}")
    
    return classes_treino

'''
def check_last_column(path_to_file: str) -> List[float]:
    df = pd.read_csv(path_to_file, encoding="latin1")
    last_col = df.iloc[:, -1].dropna()

    # pega valores únicos na ordem em que aparecem
    unique_vals = list(dict.fromkeys(last_col))

    # cria um mapa label_original -> índice numérico
    label_to_index = {val: float(i) for i, val in enumerate(unique_vals)}

    # devolve só a lista de índices das classes de treino
    result = list(label_to_index.values())

    DebugLogger.log("[DEBUG check_last_column] mapa treino label->idx:", label_to_index)
    DebugLogger.log("[DEBUG check_last_column] classes treino (idx):", result)
    #return result
    #return []
'''

'''
def store_lines(path_to_file: str, distinct_values: List[float]) -> List[int]:
    df = pd.read_csv(path_to_file, encoding="latin1")
    seen_values = set()
    line_values = []

    DebugLogger.log(f"[DEBUG store_lines] Classe conhecidas (treino): {distinct_values}")
    DebugLogger.log(f"[DEBUG store_lines] Total de linhas no results: {len(df)}")

    for idx, value in enumerate(df.iloc[:, 1], start = 1):
        if idx == 1:
            continue
        if value not in seen_values and value not in distinct_values:
            seen_values.add(value)
            line_values.append((idx - 1) // DIVISOR)
    return line_values
'''

def store_lines(path_to_file: str, distinct_values: List[float]) -> List[int]:
    seen_values = set()
    line_values = []
    
    DebugLogger.log(f"[DEBUG store_lines] Classes conhecidas (treino): {distinct_values}")
    
    line_counter = 0
    with open(path_to_file, 'r', encoding='latin1') as f:
        reader = csv.reader(f)
        
        for row in reader:
            line_counter += 1
            
            if line_counter == 1: # Pula header (exatamente como Java)
                #DebugLogger.log("HEADER:", row)  
                continue
            
            value = float(row[1])  # Coluna 1 = Rotulo Verdadeiro
            #DebugLogger.log(f"linha {line_counter}: row[1] = {value}")

            if line_counter <= 10:
                DebugLogger.log(f"[DEBUG linha {line_counter}] classe={value}, está no treino? {value in distinct_values}")
            
            if value not in seen_values and value not in distinct_values:
                seen_values.add(value)
                momento = (line_counter - 1) // DIVISOR
                line_values.append(momento)
                DebugLogger.log(f"[DEBUG store_lines] Nova classe {value} na linha {line_counter}, momento {momento}")
    
    DebugLogger.log(f"[DEBUG store_lines] Total de novas classes detectadas: {len(line_values)}")
    DebugLogger.log(f"[DEBUG store_lines] Momentos das novas classes: {line_values}")

    return line_values

def main():
    current = os.getcwd()
    dataset = "kdd"
    latencia = ["10000000"]
    percented_labeled = ["1.0"]
    results_eifuzzcnd = {}

    mapper = ClassMapper()
    caminho_dataset = os.path.join(current, "datasets", dataset, "")
    mapper.initialize(caminho_dataset, dataset)

    for lat in latencia:
        print(f"🔄 Processando latência {lat}...")
        for percent in percented_labeled:
            print(f"📊 Processando porcentagem rotulada {percent}...")
            caminho_train = os.path.join(current, "datasets", dataset, f"{dataset}-train.arff")
            caminho_resultados = os.path.join(current, "datasets", dataset, "graphics_data", f"{dataset}{lat}-{percent}-EIFuzzCND-Python-results.csv")
            caminho_novidades = os.path.join(current, "datasets", dataset, "graphics_data", f"{dataset}{lat}-{percent}-EIFuzzCND-Python-novelties.csv")
            caminho_acuracia = os.path.join(current, "datasets", dataset, "graphics_data", f"{dataset}{lat}-{percent}-EIFuzzCND-Python-acuracia.csv")

            # !classes_treinamento = check_last_column(caminho_train)

            classes_treinamento = check_last_column(dataset, caminho_dataset)

            count_results = count_lines_in_csv(caminho_resultados)
            count_novelties = count_lines_in_csv(caminho_novidades)
            count_acuracias = count_lines_in_csv(caminho_acuracia)

            novas_classes = store_lines(caminho_resultados, classes_treinamento)
            results_eifuzzcnd[0] = HandlesFiles.loadResults(caminho_resultados, count_results)

            novidades = HandlesFiles.loadNovelties(caminho_novidades, count_novelties)

            precisoes_fuzzcnd = []
            recalls_fuzzcnd = []
            f1scores_fuzzcnd = []
            unknown_rate = []
            acuracias_fuzzcnd = []
            unk_r_fuzzcnd = []

            HandlesFiles.loadMetrics(caminho_acuracia,
                                      count_acuracias,
                                      acuracias_fuzzcnd,
                                      precisoes_fuzzcnd,
                                      recalls_fuzzcnd,
                                      f1scores_fuzzcnd,
                                      unk_r_fuzzcnd,
                                      unknown_rate
                                      )

            #DebugLogger.log("len(accuracy)=", len(acuracias_fuzzcnd), "sample:", acuracias_fuzzcnd[:5])
            #DebugLogger.log("len(unknown) =", len(unknown_rate), "sample:", unknown_rate[:5])

            metricas_fuzzcnd = [acuracias_fuzzcnd, unknown_rate]
            rotulos = ["Accuracy", "unknownRate"]

            #DebugLogger.log(f"[DEBUG main] novas_classes que vão pro gráfico: {novas_classes}")
            #DebugLogger.log(f"[DEBUG main] Total de linhas tracejadas: {len(novas_classes)}")
 
            chart = plot_line_chart(lat, lat, metricas_fuzzcnd, rotulos, novidades, novas_classes, dataset, percent)
            print("✅ Gráfico criado com sucesso:", chart)
            print("🔧 Tipo de objeto retornado:", type(chart))
            chart.show()

if __name__ == "__main__":
    main()

