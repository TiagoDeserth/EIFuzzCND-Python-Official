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
        dataset = "rbf"
        caminho = os.path.join(os.getcwd(), "datasets", dataset, "")

        # * Testes 10/12 - ADD
        mapper = ClassMapper()
        mapper.initialize(caminho, dataset)
        class_mapping = mapper.get_mapping()

        # *Parâmetros
        fuzzyfication: float = 2.0 
        alpha: float = 2.0
        theta: float = 1.0
        K: int = 4
        kShort: int = 4  # *Número de clusters
        T: int = 40
        minWeightOffline: int = 0
        minWeightOnline: int = 15
        latencia: List[int] = [10000000]  # *2000, 5000, 10000, 10000000
        tChunk: int = 2000
        ts: int = 200

        phi: float = 0.2
        percentLabeled: List[float] = [1.0]

        # ? NEW: tamanho da Janela (buffer) para updates em lote
        # ? é possível experimentar tamanho 1 para o comportamento que já temos, onde o modelo é atualizado a cada exemplo
        # ? Mas, para o teste real, é possível testar valores maiores (como 20, 50) para agrupar updates
        windowSize: int = 1

        # Carrega dataset em ARFF (equivalente ao Java/Weka)
        train_path = os.path.join(caminho, dataset + "-train.arff")
        DebugLogger.log(f"Tentando carregar: {train_path}")

        # *Modificações para transformação de labels nominais para numéricos (float)
        data_arff, meta = arff.loadarff(train_path)
        df = pd.DataFrame(data_arff)
        class_col = df.columns[-1]

        # ? ---> MODIFICADO: Usar mapeamento centralizado <---
        df[class_col] = df[class_col].apply(
            lambda x: class_mapping.get(x.decode() if isinstance(x, bytes) else x, -1.0)
        )
        
        X = df.iloc[:, :-1].astype(float).values
        y = df.iloc[:, -1].values
        data = np.column_stack([X, y]).astype(np.float64)

        #DebugLogger.log("Primeiros 10 exemplos (classe) [OfflinePhase]: ", [row[-1] for row in data[:10]])
        #DebugLogger.log("Últimos 10 exemplos (classe) [OfflinePhase]: ", [row[-1] for row in data[-10:]])

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
                        windowSize # ? Novo parâmetro para a janela deslizante
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

