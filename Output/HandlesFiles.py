# HandlesFiles.py
import os
import csv
from typing import List
from Structs.Example import Example
from Evaluation.ResultsForExample import ResultsForExample

class HandlesFiles:

    @staticmethod
    def salvaNovidades(novidades: List[float], arquivo: str, latencia: int, percentLabeled: float):
        current = os.path.abspath(".")
        path = os.path.join(current, "datasets", arquivo, "graphics_data",
                            f"{arquivo}{latencia}-{percentLabeled}-EIFuzzCND-Python-novelties.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Linha", "Novidade"])
            for i, val in enumerate(novidades):
                writer.writerow([i, val])

    @staticmethod
    def salvaResultados(examples: List[Example], arquivo: str, latencia: int, percentLabeled: float):
        current = os.path.abspath(".")
        path = os.path.join(current, "datasets", arquivo, "graphics_data",
                            f"{arquivo}{latencia}-{percentLabeled}-EIFuzzCND-Python-results.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Linha", "Rotulo Verdadeiro", "Rotulo Classificado"])
            for i, ex in enumerate(examples, start=1):
                rotClass = "unknown" if ex.getRotuloClassificado() == -1.0 else ex.getRotuloClassificado()
                writer.writerow([i, ex.getRotuloVerdadeiro(), rotClass])

    @staticmethod
    def salvaMetrics(tempo: int, acuracia: float, precision: float, recall: float, f1Score: float,
                     dataset: str, latencia: int, percentLabeled: float,
                     unkMen: float, unknownRate: float, append: bool = False):
        current = os.path.abspath(".")
        path = os.path.join(current, "datasets", dataset, "graphics_data",
                            f"{dataset}{latencia}-{percentLabeled}-EIFuzzCND-Python-acuracia.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        write_header = not append or not os.path.isfile(path)

        with open(path, "a" if append else "w", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Tempo", "Acuracia", "Precision", "Recall", "F1-Score", "unkMen", "unknownRate"])
            writer.writerow([tempo, acuracia, precision, recall, f1Score, unkMen, unknownRate])

    @staticmethod
    def loadResults(caminho: str, numAnalises: int) -> List[ResultsForExample]:
        results = []
        try:
            with open(caminho, "r") as f:
                reader = csv.reader(f)
                next(reader)  # pular cabeçalho
                for _ in range(numAnalises):
                    row = next(reader)
                    trueLabel, predLabel = row[1], row[2]
                    results.append(ResultsForExample(trueLabel, predLabel))
        except FileNotFoundError:
            print(f"loadResults - Não foi possível abrir o arquivo: {caminho}")
            exit(1)
        return results

    @staticmethod
    def loadNovelties(caminho: str, numAnalises: int) -> List[float]:
        values = []
        try:
            with open(caminho, "r") as f:
                reader = csv.reader(f)
                next(reader)  # pular cabeçalho
                for _ in range(numAnalises):
                    row = next(reader)
                    values.append(float(row[1]))
        except FileNotFoundError:
            print(f"loadNovelties - Não foi possível abrir o arquivo: {caminho}")
            exit(1)
        return values

    @staticmethod
    def loadMetrics(caminho: str, numAnalises: int,
                    acuracias: List[float], precisoes: List[float],
                    recalls: List[float], f1Scores: List[float],
                    unkR: List[float], unknownRates: List[float]):
        try:
            with open(caminho, "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                if header != ["Tempo", "Acuracia", "Precision", "Recall", "F1-Score", "unkMen", "unknownRate"]:
                    print(f"loadMetrics - Formato de cabeçalho inválido no arquivo: {caminho}")
                    exit(1)

                for _ in range(numAnalises):
                    row = next(reader)
                    # colunas: Tempo, Acurácia, Precision, Recall, F1, unkMen, unknownRate
                    acuracias.append(float(row[1]) * 100)
                    precisoes.append(float(row[2]) * 100)
                    recalls.append(float(row[3]) * 100)
                    f1Scores.append(float(row[4]) * 100)
                    unkR.append(float(row[5]))
                    unknownRates.append(float(row[6]) * 100)
        except FileNotFoundError:
            print(f"loadMetrics - Não foi possível abrir o arquivo: {caminho}")
            exit(1)
