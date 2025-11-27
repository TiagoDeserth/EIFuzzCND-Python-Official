import os
import csv
from typing import Dict, List
from ConfusionMatrix.Metrics import Metrics

class ConfusionMatrix:
    def __init__(self):
        self.matrix: Dict[float, Dict[float, int]] = {}
        self.lastMerge: Dict[float, float] = {}

    def addInstance(self, trueClass: float, predictedClass: float):
        if trueClass not in self.matrix:
            self._addClass(trueClass)
        if predictedClass not in self.matrix:
            self._addClass(predictedClass)

        count = self.matrix[trueClass][predictedClass]
        self.matrix[trueClass][predictedClass] = count + 1

    def _addClass(self, classLabel: float):
        self.matrix[classLabel] = {}
        for otherClass in self.matrix.keys():
            self.matrix[classLabel][otherClass] = self.matrix[classLabel].get(otherClass, 0)
            self.matrix[otherClass][classLabel] = self.matrix[otherClass].get(classLabel, 0)

    def printMatrix(self):
        print("\nConfusion Matrix:")
        print("\t" + "\t".join(str(c) for c in self.matrix.keys()))
        for trueClass in self.matrix.keys():
            row = [str(self.matrix[trueClass].get(pred, 0)) for pred in self.matrix.keys()]
            print(f"{trueClass}\t" + "\t".join(row))

    def saveMatrix(self, dataset: str, latencia: int, percentLabeled: float):
        current = os.path.abspath(".")
        filePath = os.path.join(current, "datasets", dataset, "graphics_data",
                                f"{dataset}{latencia}-{percentLabeled}-matrix.csv")

        os.makedirs(os.path.dirname(filePath), exist_ok=True)

        file_exists = os.path.isfile(filePath)

        mode = "a" if file_exists else "w"
        with open(filePath, mode, newline="") as csvfile:
            writer = csv.writer(csvfile)

            if file_exists:
                writer.writerow([])  # buf_writer.newLine()
                writer.writerow(["Classes"] + list(self.matrix.keys()))
                writer.writerow([])  # escreve uma linha em branco
            else:
                writer.writerow(["Classes"] + list(self.matrix.keys()))

            for trueClass in self.matrix.keys():
                row = [trueClass] + [self.matrix[trueClass][pred] for pred in self.matrix.keys()]
                writer.writerow(row)

    def getClassesWithNonZeroCount(self) -> Dict[float, List[float]]:
        result: Dict[float, List[float]] = {}
        for trueClass in self.matrix.keys():
            if 0 <= trueClass < 100:
                predictedNonZero = []
                for predClass in self.matrix.keys():
                    if predClass > 100:
                        count = self.matrix[trueClass].get(predClass, 0)
                        if count > 0:
                            predictedNonZero.append(predClass)
                if predictedNonZero:
                    result[trueClass] = predictedNonZero
        return result

    def mergeClasses(self, labels: Dict[float, List[float]]):
        for srcLabel, destLabels in labels.items():
            if srcLabel not in self.matrix:
                continue
            row1 = self.matrix[srcLabel]

            for destLabel in destLabels:
                if destLabel in self.matrix and srcLabel != destLabel:
                    row2 = self.matrix[destLabel]

                    for column, value2 in row2.items():
                        row1[column] = row1.get(column, 0) + value2

                    self.matrix.pop(destLabel)

                    for rowLabel, row in self.matrix.items():
                        if destLabel in row:
                            value2 = row.get(destLabel)
                            row[srcLabel] = row.get(srcLabel, 0) + value2
                            row.pop(destLabel)

                    self.lastMerge[srcLabel] = destLabel

        for srcLabel, destLabel in self.lastMerge.items():
            if destLabel in self.matrix:
                self.mergeClasses({srcLabel: [destLabel]})

    def updateConfusionMatrix(self, trueLabel: float):
        if -1.0 in self.matrix.get(trueLabel, {}):
            self.matrix[trueLabel][-1.0] -= 1

    '''
    def calculateMetrics(self, tempo: int, unkMem: float, exc: float) -> Metrics:
        truePositive: float = 0
        falsePositive: float = 0
        falseNegative: float = 0
        totalSamples: float = 0

        for trueLabel, row in self.matrix.items():
            for predictedLabel, count in row.items():
                totalSamples += count
                if trueLabel is predictedLabel:
                    truePositive += count
                else:
                    falsePositive += count
                    falseNegative += count

        trueNegative: float = totalSamples - truePositive - falsePositive - falseNegative

        accuracy = (truePositive + trueNegative) / totalSamples if totalSamples > 0 else 0.0

        precision = truePositive / (truePositive + falsePositive) if (truePositive + falsePositive) > 0 else 0.0
        recall = truePositive / (truePositive + falseNegative) if (truePositive + falseNegative) > 0 else 0.0
        f1Score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        unknownRate: float = (unkMem / exc)

        return Metrics(accuracy, precision, recall, f1Score, tempo, unkMem, unknownRate)
    '''

    '''
    def calculateMetrics(self, tempo: int, unkMem: float, exc: float) -> Metrics:
        total = 0
        tp_total = 0

        precisions = []
        recalls = []

        for trueLabel, row in self.matrix.items():
            tp_cls = row.get(trueLabel, 0)
            fp_cls = sum(self.matrix[t].get(trueLabel, 0) for t in self.matrix if t != trueLabel)
            fn_cls = sum(row[p] for p in row if p != trueLabel)

            tp_total += tp_cls
            total += sum(row.values())

            prec = tp_cls / (tp_cls + fp_cls) if (tp_cls + fp_cls) > 0 else 0.0
            rec = tp_cls / (tp_cls + fn_cls) if (tp_cls + fn_cls) > 0 else 0.0

            precisions.append(prec)
            recalls.append(rec)

        accuracy = tp_total / total if total > 0 else 0.0
        precision = sum(precisions) / len(precisions) if precisions else 0.0
        recall = sum(recalls) / len(recalls) if recalls else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        unknownRate = (unkMem / exc) if exc > 0 else 0.0

        return Metrics(accuracy, precision, recall, f1, tempo, unkMem, unknownRate)
    '''

    '''
    def calculateMetrics(self, tempo: int, unkMem: float, exc: float) -> Metrics:
        truePositive: float = 0
        falsePositive: float = 0
        falseNegative: float = 0
        totalSamples: float = 0

        for trueLabel, row in self.matrix.items():
            for predictedLabel, count in row.items():
                totalSamples += count
                if trueLabel == predictedLabel:  # igual ao equals() do Java
                    truePositive += count
                else:
                    falsePositive += count if predictedLabel in row else 0
                    falseNegative += count if trueLabel in self.matrix else 0

        trueNegative: float = totalSamples - truePositive - falsePositive - falseNegative

        accuracy = (truePositive + trueNegative) / totalSamples if totalSamples > 0 else 0.0
        precision = truePositive / (truePositive + falsePositive) if (truePositive + falsePositive) > 0 else 0.0
        recall = truePositive / (truePositive + falseNegative) if (truePositive + falseNegative) > 0 else 0.0
        f1Score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        unknownRate: float = (unkMem / exc) if exc > 0 else 0.0

        return Metrics(accuracy, precision, recall, f1Score, tempo, unkMem, unknownRate)
    '''

    def calculateMetrics(self, tempo: int, unkMem: float, exc: float) -> Metrics:
        truePositive: float = 0
        falsePositive: float = 0
        falseNegative: float = 0
        trueNegative: float = 0  # Inicializa TN
        totalSamples: float = 0

        labels = list(self.matrix.keys())  # Obtém todas as classes presentes na matriz

        # 1. Calcula o total de amostras e o total de verdadeiros positivos (diagonal principal)
        for label in labels:
            row = self.matrix.get(label, {})
            for predicted_label, count in row.items():
                totalSamples += count
                if label == predicted_label:
                    truePositive += count

        # 2. Calcula Falsos Negativos (FN) e Falsos Positivos (FP) para cada classe e soma
        fp_total: float = 0
        fn_total: float = 0
        for i, current_label in enumerate(labels):
            # FN para current_label: soma da linha 'current_label', exceto a diagonal (TP)
            # (Quantas vezes era 'current_label' mas foi previsto como outra coisa)
            row_sum = sum(self.matrix.get(current_label, {}).values())
            tp_current = self.matrix.get(current_label, {}).get(current_label, 0)
            fn_total += (row_sum - tp_current)

            # FP para current_label: soma da coluna 'current_label', exceto a diagonal (TP)
            # (Quantas vezes foi previsto como 'current_label' mas era outra coisa)
            col_sum = 0
            for other_label in labels:
                col_sum += self.matrix.get(other_label, {}).get(current_label, 0)
            fp_total += (col_sum - tp_current)

        # Atribui os totais calculados
        falsePositive = fp_total
        falseNegative = fn_total

        # 3. Calcula Verdadeiros Negativos (TN)
        # TN = Total de amostras - (TP + FP + FN)
        # Cuidado: Esta fórmula simples para TN só funciona bem em classificação binária.
        # Em multiclasse, calcular TN por classe e fazer média é mais comum, mas complexo.
        # A abordagem Java parecia simplificar isso. Vamos seguir a fórmula mais simples
        # baseada no total, que pode ser uma aproximação razoável para a acurácia geral.
        trueNegative = totalSamples - truePositive - falsePositive - falseNegative
        # Garante que TN não seja negativo devido a possíveis inconsistências no cálculo de FP/FN multiclasse
        trueNegative = max(0.0, trueNegative)

        # 4. Calcula as métricas finais
        accuracy = (truePositive + trueNegative) / totalSamples if totalSamples > 0 else 0.0

        # Para Precisão, Recall, F1-Score em multiclasse, geralmente se usa média (macro, micro, weighted).
        # A implementação Java parecia calcular uma versão "geral" ou "micro" implícita.
        # Vamos usar os totais de TP, FP, FN para calcular essas métricas gerais:
        precision = truePositive / (truePositive + falsePositive) if (truePositive + falsePositive) > 0 else 0.0
        recall = truePositive / (truePositive + falseNegative) if (
                                                                              truePositive + falseNegative) > 0 else 0.0  # Também chamado de Sensibilidade
        f1Score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        unknownRate: float = (unkMem / exc) if exc > 0 else 0.0

        return Metrics(accuracy, precision, recall, f1Score, tempo, unkMem, unknownRate)

    def countUnknow(self) -> int:
        count = 0
        for row in self.matrix.values():
            count += row.get(-1.0, 0)
        return count

    def getNumberOfClasses(self) -> int:
        return len(self.matrix)
