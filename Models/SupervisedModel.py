import numpy as np
from typing import List, Dict
from Structs.Example import Example
from Structs.SPFMiC import SPFMiC
from FuzzyFunctions.FuzzyFunctions import FuzzyFunctions
from FuzzyFunctions.DistanceMeasures import calculaDistanciaEuclidiana
from DebugLogger import DebugLogger
import math

class SupervisedModel:
    classifier: Dict[float, List[SPFMiC]] = {}

    def __init__(self, dataset: str, caminho: str, fuzzification: float, alpha: float, theta: float, K: int, minWeight: int):
        self.dataset = dataset
        self.caminho = caminho
        self.fuzzification = fuzzification
        self.alpha = alpha
        self.theta = theta
        self.K = K
        self.minWeight = minWeight

        self.knowLabels: List[float] = []
        #self.classifier: Dict[float, List[SPFMiC]] = {}

    def trainInitialModel(self, trainSet) -> None:
        chunk: List[Example] = []
        arr = np.asarray(trainSet)

        #DebugLogger.log(f"[DEBUG] trainInitialModel: trainSet.shape={arr.shape}, dtype={arr.dtype}") #<---

        for i in range(arr.shape[0]):
            # *Prints 11/11/2025
            #DebugLogger.log(f"[DEBUG Example] Linha de dados shape: {arr[i].shape} (index {i})")

            ex = Example(arr[i], True, i)
            chunk.append(ex)

        examplesByClass: Dict[float, List[Example]] = FuzzyFunctions.separateByClasses(chunk)
        classes: List[float] = list(examplesByClass.keys())
        DebugLogger.log(f"[DEBUG] trainInitialModel: classes detectadas={classes}")
        #DebugLogger.log(f"[DEBUG] trainInitialModel: classes detectadas={classes}, total_classes={len(classes)}")

        for j in range(len(examplesByClass)):
            cls: float = float(classes[j])
            lst: List[Example] = examplesByClass[cls]
            #DebugLogger.log(f"[DEBUG] trainInitialModel: classe={cls}, exemplos={len(lst)}")

            if len(lst) > self.K:
                if cls not in self.knowLabels:
                    self.knowLabels.append(cls)

                #DebugLogger.log(
                    #f"[DEBUG] trainInitialModel: executando fuzzyCMeans(K={self.K}, m={self.fuzzification})") #<---
                clusters = FuzzyFunctions.fuzzyCMeans(lst, self.K, self.fuzzification)

                '''
                try:
                    DebugLogger.log(
                        f"[DEBUG] fuzzyCMeans: centroids.shape={np.array(clusters.centroids).shape}, membership.shape={np.array(clusters.membership).shape}")
                except Exception:
                    DebugLogger.log(f"[DEBUG] fuzzyCMeans: centroids/membership shapes indisponíveis")
                '''

                spfmics: List[SPFMiC] = FuzzyFunctions.separateExamplesByClusterClassifiedByFuzzyCMeans(
                    lst, clusters.centroids, clusters.membership, cls, self.alpha, self.theta, self.minWeight, 0
                )

                #DebugLogger.log(f"[DEBUG] trainInitialModel: classe={cls}, SPFMiCs criados={len(spfmics)}")
                #DebugLogger.log(f"[DEBUG] trainInitialModel: classe={cls}, spfmics={len(spfmics)}")
                #self.classifier[cls] = spfmics #<---
                SupervisedModel.classifier[cls] = spfmics

        #DebugLogger.log(
            #f"[DEBUG] trainInitialModel: total_labels={len(self.knowLabels)}, total_classes={len(self.classifier)}")
        #DebugLogger.log("[DEBUG] Finalizando trainInitialModel()")

    def classifyNew(self, ins, updateTime: int):
        allSPFMiCSOfClassifier: List[SPFMiC] = []
        allSPFMiCSOfClassifier.extend(self.getAllSPFMiCsFromClassifier(SupervisedModel.classifier))

        #DebugLogger.log(
            #f"[DEBUG] classifyNew: exemplo.shape={np.asarray(ins).shape}, total_SPFMiCs={len(allSPFMiCSOfClassifier)}")

        return self.classify(allSPFMiCSOfClassifier, Example(np.asarray(ins), True), updateTime)

    #ORIGINAL
    @staticmethod
    def getAllSPFMiCsFromClassifier(classifier: Dict[float, List[SPFMiC]]) -> List[SPFMiC]:
        spfMiCS: List[SPFMiC] = []
        keys: List[float] = list(classifier.keys())
        for i in range(len(classifier)):
            spfMiCS.extend(classifier[keys[i]])

        return spfMiCS

    '''
    @staticmethod
    def getAllSPFMiCsFromClassifier(classifier: Dict[float, List[SPFMiC]]) -> List[SPFMiC]:
        spfMiCS: List[SPFMiC] = []
        for key in classifier.keys():
            spfMiCS.extend(classifier[key])
        return spfMiCS
    '''

    def classify(self, spfMiCS: List[SPFMiC], example: Example, updateTime: int) -> float:
        #DebugLogger.log(f"[DEBUG] classify: nSPFMiCs={len(spfMiCS)}, exemplo_dim={example.getPonto().shape}")

        tipicidades: List[float] = []
        pertinencia: List[float] = []
        auxSPFMiCs: List[SPFMiC] = []
        isOutlier: bool = True

        for i in range(len(spfMiCS)):
            distancia: float = calculaDistanciaEuclidiana(example, spfMiCS[i].getCentroide())
            raio = spfMiCS[i].getRadiusWithWeight()
            #print(f"Raio: {raio}")
            #if distancia <= spfMiCS[i].getRadiusWithWeight():
            if distancia <= raio:
                isOutlier = False
                tipicidades.append(spfMiCS[i].calculaTipicidade(example.getPonto(), spfMiCS[i].getN(), self.K))
                pertinencia.append(SupervisedModel.calculaPertinencia(example.getPonto(), spfMiCS[i].getCentroide(), self.fuzzification))
                auxSPFMiCs.append(spfMiCS[i])

        if isOutlier:
            #DebugLogger.log("[DEBUG] classify: exemplo classificado como desconhecido (outlier)")
            return -1

        maxValTip: float = max(tipicidades)
        indexMaxTip: int = tipicidades.index(maxValTip)

        maxValPer: float = max(pertinencia)
        #DebugLogger.log(
            #f"[DEBUG] classify: maxTip={maxValTip:.6f}, maxPer={maxValPer:.6f}, total_candidatos={len(auxSPFMiCs)}")

        spfmic: SPFMiC = auxSPFMiCs[indexMaxTip]
        index: int = spfMiCS.index(spfmic)

        spfMiCS[index].setUpdated(updateTime)
        spfMiCS[index].atribuiExemplo(example, maxValPer, 1)

        #DebugLogger.log(
            #f"[DEBUG] classify: exemplo atribuído ao cluster rotulo={spfMiCS[index].getRotulo()}, atualizado={updateTime}")
        return spfMiCS[index].getRotulo()

    @staticmethod
    def calculaPertinencia(dataPoints: np.ndarray, clusterCentroids: np.ndarray, m: float) -> float:
        dataPoints = np.asarray(dataPoints, dtype=float)
        clusterCentroids = np.asarray(clusterCentroids, dtype=float)
        distance: float = float(np.sum((dataPoints - clusterCentroids) ** 2))

        #DebugLogger.log(f"[DEBUG] calculaPertinencia: dist^2={distance:.6f}, m={m}")

        return math.exp(-distance / m)

    def getAllSPFMiCs(self) -> List[SPFMiC]:
        spfMiCS: List[SPFMiC] = []
        #spfMiCS.extend(self.getAllSPFMiCsFromClassifier(self.classifier))
        spfMiCS.extend(self.getAllSPFMiCsFromClassifier(SupervisedModel.classifier))

        #DebugLogger.log(f"[DEBUG] getAllSPFMiCs: total={len(spfMiCS)}")

        return spfMiCS

    def trainNewClassifier(self, chunk: List[Example], t: int) -> List[Example]:
        #DebugLogger.log(f"[DEBUG] trainNewClassifier: chunk={len(chunk)}, t={t}")

        newChunk: List[Example] = []
        examplesByClass = FuzzyFunctions.separateByClasses(chunk)
        classes: List[float] = list(examplesByClass.keys())

        #DebugLogger.log(f"[DEBUG] trainNewClassifier: classes={classes}")

        classifier_local: Dict[float, List[SPFMiC]] = {}

        for j in range(len(examplesByClass)):
            cls = float(classes[j])
            lst = examplesByClass[cls]

            #DebugLogger.log(f"[DEBUG] trainNewClassifier: classe={cls}, exemplos={len(lst)}")

            if len(lst) >= self.K * 2:
                if cls not in self.knowLabels:
                    self.knowLabels.append(cls)

                clusters = FuzzyFunctions.fuzzyCMeans(lst, self.K, self.fuzzification)

                #DebugLogger.log(
                    #f"[DEBUG] fuzzyCMeans (trainNew): centroids.shape={np.array(clusters.centroids).shape}, membership.shape={np.array(clusters.membership).shape}")

                spfmics = FuzzyFunctions.separateExamplesByClusterClassifiedByFuzzyCMeans(
                    lst, clusters.centroids, clusters.membership, cls, self.alpha, self.theta, self.minWeight, t
                )

                #DebugLogger.log(f"[DEBUG] trainNewClassifier(): gerou {len(spfmics)} SPFMiCs para classe {cls}")

                #classifier_local[cls] = spfmics
                SupervisedModel.classifier[cls] = spfmics

                #DebugLogger.log(f"[DEBUG] trainNewClassifier: classe={cls}, SPFMiCs={len(spfmics)}")

            else:
                newChunk.extend(lst)

        #DebugLogger.log(
            #f"[DEBUG] trainNewClassifier: total_new_labels={len(classifier_local)}, newChunk={len(newChunk)}")

        return newChunk

    @staticmethod
    def removeOldSPFMiCs(ts: int, currentTime: int) -> None:
        #DebugLogger.log(
            #f"[DEBUG] removeOldSPFMiCs: ts={ts}, currentTime={currentTime}, total_classes={len(self.classifier)}")

        #for cls, spfMiCSatuais in list(self.classifier.items()):
        for cls, spfMiCSatuais in list(SupervisedModel.classifier.items()):
            spfMiCSAux = list(spfMiCSatuais)
            for spf in spfMiCSatuais:
                if (currentTime - spf.getT() > ts) and (currentTime - spf.getUpdated() > ts):
                    spfMiCSAux.remove(spf)
            #self.classifier[cls] = spfMiCSAux
            SupervisedModel.classifier[cls] = spfMiCSAux
        #DebugLogger.log("[DEBUG] removeOldSPFMiCs: finalizado")