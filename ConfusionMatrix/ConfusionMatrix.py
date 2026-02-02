import os
import csv
from typing import Dict, List
from ConfusionMatrix.Metrics import Metrics
from DebugLogger import DebugLogger

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

    # Trativa para tentar implementar a Janela Deslizante (Quando comento essa função)
    def updateConfusionMatrix(self, trueLabel: float):
        if -1.0 in self.matrix.get(trueLabel, {}):
            # New
            count = self.matrix[trueLabel][-1.0]
            self.matrix[trueLabel][-1.0] = count - 1

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


    # * calculateMetrics mais válida até o momento (Acurácia muito próxima à Precision)
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
        # A abordagem Java parecia simplificar isso
        # baseada no total, que pode ser uma aproximação razoável para a acurácia geral.
        trueNegative = totalSamples - truePositive - falsePositive - falseNegative
        # Garante que TN não seja negativo devido a possíveis inconsistências no cálculo de FP/FN multiclasse
        trueNegative = max(0.0, trueNegative)

        # 4. Calcula as métricas finais
        accuracy = (truePositive + trueNegative) / totalSamples if totalSamples > 0 else 0.0

        # Para Precisão, Recall, F1-Score em multiclasse, geralmente se usa média (macro, micro, weighted).
        # A implementação Java parecia calcular uma versão "geral" ou "micro" implícita.
        precision = truePositive / (truePositive + falsePositive) if (truePositive + falsePositive) > 0 else 0.0
        recall = truePositive / (truePositive + falseNegative) if (
                                                                              truePositive + falseNegative) > 0 else 0.0  # Também chamado de Sensibilidade
        f1Score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        unknownRate: float = (unkMem / exc) if exc > 0 else 0.0

        return Metrics(accuracy, precision, recall, f1Score, tempo, unkMem, unknownRate)

    # * calculateMetrics que tentei fazer para analisar mais afundo a Precision
    '''
    def calculateMetrics(self, tempo: int, unkMem: float, exc: float) -> Metrics:
        labels = list(self.matrix.keys())

        total = 0.0
        tp_total = 0.0
        precisions = []
        recalls = []

        for c in labels:
            row = self.matrix.get(c, {})
            tp = row.get(c, 0)

            fp = sum(self.matrix[true].get(c, 0) for true in labels if true != c)
            fn = sum(count for pred, count in row.items() if pred != c)

            tp_total += tp
            total += sum(row.values())

            prec_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            precisions.append(prec_c)
            recalls.append(rec_c)

        accuracy = tp_total / total if total > 0 else 0.0

        precision = sum(precisions) / len(precisions) if precisions else 0.0
        recall = sum(recalls) / len(recalls) if recalls else 0.0

        f1Score = (2 *  precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        unknownRate = (unkMem / exc) if exc > 0 else 0.0

        return Metrics(accuracy, precision, recall, f1Score, tempo, unkMem, unknownRate)
    '''

    '''
    def calculateMetrics(self, tempo: int, unkMem: float, exc: float) -> Metrics:
        """
        Calcula métricas para classificação multiclasse.
        Acurácia: (TP corretos) / (Total de exemplos)
        Precision/Recall: Média macro (média das métricas por classe)
        """
        labels = list(self.matrix.keys())

        DebugLogger.log(f"\n{'='*80}")
        DebugLogger.log(f"DEBUG calculateMetrics - Tempo: {tempo}")
        DebugLogger.log(f"{'='*80}")
        
        # Total de exemplos
        totalSamples: float = 0
        for row in self.matrix.values():
            totalSamples += sum(row.values())

        DebugLogger.log(f"Total de amostras: {totalSamples}")
        
        # TP total (soma da diagonal)
        truePositive: float = 0
        for label in labels:
            tp_label = self.matrix.get(label, {}).get(label, 0)
            truePositive += tp_label
            DebugLogger.log(f"  Classe {label}: TP = {tp_label}")
        
        DebugLogger.log(f"\nTP Total (diagonal): {truePositive}")
        
        # **ACURÁCIA SIMPLES: TP / Total**
        accuracy = truePositive / totalSamples if totalSamples > 0 else 0.0

        DebugLogger.log(f"Acurácia calculada: {truePositive}/{totalSamples} = {accuracy}")

        # Verificação de sanidade
        if accuracy > 1.0:
            DebugLogger.log(f"⚠️  ERRO! Acurácia > 1.0 detectada!")
            DebugLogger.log(f"   TP = {truePositive}, Total = {totalSamples}")
            DebugLogger.log(f"\nMatriz de Confusão:")
            for true_label in labels:
                row = self.matrix.get(true_label, {})
                DebugLogger.log(f"  Verdadeiro {true_label}: {dict(row)}")
        
        # **PRECISION E RECALL: Média Macro (por classe)**
        precisions = []
        recalls = []
        
        for label in labels:
            # TP para esta classe (diagonal)
            tp_class = self.matrix.get(label, {}).get(label, 0)
            
            # FP para esta classe (coluna - diagonal)
            # Quantas vezes previmos 'label' mas era outra classe
            fp_class = 0
            for true_label in labels:
                if true_label != label:
                    fp_class += self.matrix.get(true_label, {}).get(label, 0)
            
            # FN para esta classe (linha - diagonal)
            # Quantas vezes era 'label' mas previmos outra classe
            fn_class = 0
            row = self.matrix.get(label, {})
            for pred_label in labels:
                if pred_label != label:
                    fn_class += row.get(pred_label, 0)
            
            # Precision e Recall para esta classe
            precision_class = tp_class / (tp_class + fp_class) if (tp_class + fp_class) > 0 else 0.0
            recall_class = tp_class / (tp_class + fn_class) if (tp_class + fn_class) > 0 else 0.0

            DebugLogger.log(f"  Classe {label}: TP={tp_class}, FP={fp_class}, FN={fn_class}, Prec={precision_class:.4f}, Rec={recall_class:.4f}")
            
            precisions.append(precision_class)
            recalls.append(recall_class)
        
        # Média macro
        precision = sum(precisions) / len(precisions) if precisions else 0.0
        recall = sum(recalls) / len(recalls) if recalls else 0.0
        
        # F1-Score
        f1Score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        # Unknown Rate
        unknownRate: float = (unkMem / exc) if exc > 0 else 0.0

        DebugLogger.log(f"\nMétricas Finais:")
        DebugLogger.log(f"  Accuracy: {accuracy}")
        DebugLogger.log(f"  Precision (macro): {precision}")
        DebugLogger.log(f"  Recall (macro): {recall}")
        DebugLogger.log(f"  F1-Score: {f1Score}")
        DebugLogger.log(f"{'='*80}\n")
        
        return Metrics(accuracy, precision, recall, f1Score, tempo, unkMem, unknownRate)
    '''

    def countUnknow(self) -> int:
        count = 0
        for row in self.matrix.values():
            count += row.get(-1.0, 0)
        return count

    def getNumberOfClasses(self) -> int:
        return len(self.matrix)
