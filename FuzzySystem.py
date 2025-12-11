import os
import pandas as pd
import numpy as np
from scipy.io import arff
from Phases.OfflinePhase import OfflinePhase
from Phases.OnlinePhase import OnlinePhase
from typing import List
from DebugLogger import DebugLogger
import random
from ClassMapper import ClassMapper

random.seed(42)
np.random.seed(42)

def main():
    DebugLogger.init()

    try:
        dataset = "kdd"
        caminho = os.path.join(os.getcwd(), "datasets", dataset, "")

        # * Testes 10/12 - ADD
        mapper = ClassMapper()
        mapper.initialize(caminho, dataset)
        class_mapping = mapper.get_mapping()

        # *Parâmetros
        fuzzyfication: float = 2.0 
        alpha: float = 2.0
        theta: float = 1.0
        K: int = 8
        kShort: int = 8  # *Número de clusters
        T: int = 80
        minWeightOffline: int = 0
        minWeightOnline: int = 30
        latencia: List[int] = [10000000]  # *2000, 5000, 10000, 10000000
        tChunk: int = 2000
        ts: int = 200

        phi: float = 0.2
        percentLabeled: List[float] = [1.0]

        # ? NEW: tamanho da Janela (buffer) para updates em lote
        # ? é possível experimentar tamanho 1 para o comportamento que já temos, onde o modelo é atualizado a cada exemplo
        # ? Mas, para o teste real, é possível testar valores maiores (como 20, 50) para agrupar updates
        # ? windowSize: int = 1

        # Carrega dataset em ARFF (equivalente ao Java/Weka)
        train_path = os.path.join(caminho, dataset + "-train.arff")
        DebugLogger.log(f"Tentando carregar: {train_path}")
        # !data_arff, meta = arff.loadarff(train_path)
        # !df = pd.DataFrame(data_arff)

        # Separa atributos (X) e classe (y)
        # !X = df.iloc[:, :-1].astype(float).values

        # Força conversão do rótulo nominal (bytes) -> string -> float
        # !y = pd.to_numeric(df.iloc[:, -1].astype(str), errors="coerce").astype(np.float64).values

        # Junta de volta no formato [features..., classValue]
        #data = np.column_stack([X, y])
        # !data = np.column_stack([X, y]).astype(np.float64)

        # *Modificações para transformação de labels nominais para numéricos (float)
        data_arff, meta = arff.loadarff(train_path)
        df = pd.DataFrame(data_arff)
        #! class_col = df.columns[-1]
        # !df[class_col] = df[class_col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
        # !classes_uniques = df[class_col].unique()
        # !class_for_index = {classe: idx for idx, classe in enumerate(classes_uniques)}
        # !df[class_col] = df[class_col].map(class_for_index).astype(float)
        # !X = df.iloc[:, :-1].astype(float).values
        # !y = df.iloc[:, -1].values
        # !data = np.column_stack([X, y]).astype(np.float64)
        class_col = df.columns[-1]

        ''' Testes 10/12/2025
        nominal_values = meta[class_col][1]
        class_for_index = {val.decode() if isinstance(val, bytes) else val: idx
                           for idx, val in enumerate(nominal_values)}
        index_for_class = {idx: val for val, idx in class_for_index.items()}

        df[class_col] = df[class_col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
        df[class_col] = df[class_col].map(class_for_index).astype(float)
        '''

        # ? ---> MODIFICADO: Usar mapeamento centralizado <---
        df[class_col] = df[class_col].apply(
            lambda x: class_mapping.get(x.decode() if isinstance(x, bytes) else x, -1.0)
        )
        
        X = df.iloc[:, :-1].astype(float).values
        y = df.iloc[:, -1].values
        data = np.column_stack([X, y]).astype(np.float64)

        print("Primeiros 10 exemplos (classe) [OfflinePhase]: ", [row[-1] for row in data[:10]])
        print("Últimos 10 exemplos (classe) [OfflinePhase]: ", [row[-1] for row in data[-10:]])

        #print("Primeiras linhas processadas:")
        #print(data[:5])  # debug: veja se agora está [f1, f2, ..., class]

        for lat in latencia:
            for labeled in percentLabeled:
                condicaoSatisfeita: bool = False
                while not condicaoSatisfeita:
                    # *Fase offline
                    offlinePhase = OfflinePhase(
                        dataset,
                        caminho,
                        fuzzyfication,
                        alpha,
                        theta,
                        K,
                        minWeightOffline)
                    supervisedModel = offlinePhase.inicializar(data)

                    # *Fase online
                    onlinePhase = OnlinePhase(
                        caminho,
                        supervisedModel,
                        lat,
                        tChunk,
                        T,
                        kShort,
                        phi,
                        ts,
                        minWeightOnline,
                        labeled,
                        #windowSize # ? Novo parâmetro para a janela deslizante
                    ) 
                    onlinePhase.initialize(dataset)

                    if onlinePhase.getTamConfusion() > 999:
                        continue
                    else:
                        condicaoSatisfeita = True
                        break

    finally:
        DebugLogger.close()

if __name__ == "__main__":
    main()

