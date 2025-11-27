import os
import pandas as pd
import numpy as np
from typing import List
import csv

from Evaluation.ResultsForExample import ResultsForExample
from Output.HandlesFiles import HandlesFiles
from Output.LineChart import plot_line_chart

from DebugLogger import DebugLogger

DIVISOR = 1000

def count_lines_in_csv(file_path: str) -> int:
    df = pd.read_csv(file_path, encoding="latin1")
    return len(df) -1

def check_last_column(path_to_file: str) -> List[float]:
    df = pd.read_csv(path_to_file, encoding="latin1")
    last_column = df.iloc[:, -1].dropna().unique()
    last_column_values = list(last_column.astype(str))

    result = [float(i) for i in range(len(last_column_values))]
    #DebugLogger.log(f"[DEBUG check_last_column] Classes no treino: {result}")
    #DebugLogger.log(f"[DEBUG check_last_column] Total: {len(result)} classes")

    #return []
    return result
    #return [float(i) for i in range(len(last_column_values))]

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
    
    print(f"[DEBUG store_lines] Classes conhecidas (treino): {distinct_values}")
    
    line_counter = 0
    with open(path_to_file, 'r', encoding='latin1') as f:
        reader = csv.reader(f)
        
        for row in reader:
            line_counter += 1
            
            if line_counter == 1:  # Pula header (exatamente como Java)
                continue
            
            value = float(row[1])  # Coluna 1 = Rotulo Verdadeiro
            
            if value not in seen_values and value not in distinct_values:
                seen_values.add(value)
                momento = (line_counter - 1) // DIVISOR
                line_values.append(momento)
                print(f"[DEBUG store_lines] Nova classe {value} na linha {line_counter}, momento {momento}")
    
    #print(f"[DEBUG store_lines] Total de novas classes detectadas: {len(line_values)}")
    #print(f"[DEBUG store_lines] Momentos das novas classes: {line_values}")

    return line_values

def main():
    current = os.getcwd()
    dataset = "kdd"
    latencia = ["10000000"]
    percented_labeled = ["1.0"]
    results_eifuzzcnd = {}

    for lat in latencia:
        print(f"🔄 Processando latência {lat}...")
        for percent in percented_labeled:
            print(f"📊 Processando porcentagem rotulada {percent}...")
            caminho_train = os.path.join(current, "datasets", dataset, f"{dataset}-train.csv")
            caminho_resultados = os.path.join(current, "datasets", dataset, "graphics_data", f"{dataset}{lat}-{percent}-EIFuzzCND-Python-results.csv")
            caminho_novidades = os.path.join(current, "datasets", dataset, "graphics_data", f"{dataset}{lat}-{percent}-EIFuzzCND-Python-novelties.csv")
            caminho_acuracia = os.path.join(current, "datasets", dataset, "graphics_data", f"{dataset}{lat}-{percent}-EIFuzzCND-Python-acuracia.csv")

            classes_treinamento = check_last_column(caminho_train)
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

            print("len(accuracy)=", len(acuracias_fuzzcnd), "sample:", acuracias_fuzzcnd[:5])
            print("len(unknown) =", len(unknown_rate), "sample:", unknown_rate[:5])

            metricas_fuzzcnd = [acuracias_fuzzcnd, unknown_rate]
            rotulos = ["Accuracy", "unknownRate"]

            DebugLogger.log(f"[DEBUG main] novas_classes que vão pro gráfico: {novas_classes}")
            DebugLogger.log(f"[DEBUG main] Total de linhas tracejadas: {len(novas_classes)}")
 
            chart = plot_line_chart(lat, lat, metricas_fuzzcnd, rotulos, novidades, novas_classes, dataset, percent)
            print("✅ Gráfico criado com sucesso:", chart)
            print("🔧 Tipo de objeto retornado:", type(chart))
            chart.show()

if __name__ == "__main__":
    main()

