import numpy as np
from typing import List, Dict, Tuple, Optional
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
        #DebugLogger.log(f"[DEBUG] trainInitialModel: classes detectadas={classes}")
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

    # Código antigo antes da implementação da Janela 
    def classifyNew(self, ins, updateTime: int):
        allSPFMiCSOfClassifier: List[SPFMiC] = []
        allSPFMiCSOfClassifier.extend(self.getAllSPFMiCsFromClassifier(SupervisedModel.classifier))

        #DebugLogger.log(
            #f"[DEBUG] classifyNew: exemplo.shape={np.asarray(ins).shape}, total_SPFMiCs={len(allSPFMiCSOfClassifier)}")

        return self.classify(allSPFMiCSOfClassifier, Example(np.asarray(ins), True), updateTime)

    def classifyNewWithoutUpdate(self, ins, updateTime: int):
        '''
        NOVO: Classifica SEM atualizar o modelo (apenas predição)
        Retorna o rótulo classificado ou -1 se desconhecido
        '''

        allSPFMiCSOfClassifier: List[SPFMiC] = []
        allSPFMiCSOfClassifier.extend(self.getAllSPFMiCsFromClassifier(SupervisedModel.classifier))

        example = Example(np.asarray(ins), True, updateTime)

        tipicidades: List[float] = []
        pertinencia: List[float] = []
        auxSPFMiCs: List[SPFMiC] = []
        isOutlier: bool = True

        for i in range(len(allSPFMiCSOfClassifier)):
            distancia: float = calculaDistanciaEuclidiana(example, allSPFMiCSOfClassifier[i].getCentroide())
            raio = allSPFMiCSOfClassifier[i].getRadiusWithWeight()

            if distancia <= raio:
                isOutlier = False
                tipicidades.append(allSPFMiCSOfClassifier[i].calculaTipicidade(example.getPonto(), allSPFMiCSOfClassifier[i].getN(), self.K))
                pertinencia.append(SupervisedModel.calculaPertinencia(example.getPonto(), allSPFMiCSOfClassifier[i].getCentroide(), self.fuzzification))
                auxSPFMiCs.append(allSPFMiCSOfClassifier[i])


        if isOutlier:
            return -1
        
        maxValTip: float = max(tipicidades)
        indexMaxTip: int = tipicidades.index(maxValTip)
        spfmic: SPFMiC = auxSPFMiCs[indexMaxTip]

        # ? NÃO ATUALIZA, apenas retorna o rótulo
        return spfmic.getRotulo()

    # ! DEPOIS VOLTAR [TESTES - 28/01/2026]
    # ? Nova função
    def updateWithExample(self, ins, rotulo: float, updateTime: int):
        
        # **NOVO: Atualiza o modelo com um exemplo já classificado**
        #Usado pela janela deslizante para atualização incremental em lote.
        
        allSPFMiCSOfClassifier: List[SPFMiC] = []
        #self.getAllSPFMiCsFromClassifier(SupervisedModel.classifier)
        allSPFMiCSOfClassifier.extend(self.getAllSPFMiCsFromClassifier(SupervisedModel.classifier))

        # ? [begin] Logs para verificar [...]
        DebugLogger.log(f"[UPDATE] updateTime = {updateTime} | rotulo = {rotulo:.1f} | Total SPFMiCs = {len(allSPFMiCSOfClassifier)}")
        # ? [end]

        example = Example(np.asarray(ins), True, updateTime)

        # Encontra o SPFMiC correspondente ao rótulo com maior tipicidade
        tipicidades: List[float] = []
        pertinencias: List[float] = []
        candidatos: List[SPFMiC] = []

        for spfmic in allSPFMiCSOfClassifier:
            if spfmic.getRotulo() == rotulo:
                distancia = calculaDistanciaEuclidiana(example, spfmic.getCentroide())
                raio = spfmic.getRadiusWithWeight()

                if distancia <= raio:
                    tipicidade = spfmic.calculaTipicidade(example.getPonto(), spfmic.getN(), self.K)
                    pertinencia = SupervisedModel.calculaPertinencia(example.getPonto(), spfmic.getCentroide(), self.fuzzification)
                    tipicidades.append(tipicidade)
                    pertinencias.append(pertinencia)
                    candidatos.append(spfmic)
        
        if len(candidatos) > 0:
            maxValTip = max(tipicidades)
            indexMaxTip = tipicidades.index(maxValTip)

            spfmic_escolhido = candidatos[indexMaxTip]
            pertinencia_escolhida = pertinencias[indexMaxTip]
            tipicidade_escolhida = tipicidades[indexMaxTip]

            # ? [begin] Logs para verificar [...]
            DebugLogger.log(f"[UPDATE SUCCESS] updateTime = {updateTime} | rotulo = {rotulo:.1f} | SPFMiC atualizado | tip = {tipicidade_escolhida:.4f} | pert = {pertinencia_escolhida:.4f}")
            # ? [end]

            # Atualiza Incrementalmente
            spfmic_escolhido.setUpdated(updateTime)
            spfmic_escolhido.atribuiExemplo(example, pertinencia_escolhida, tipicidade_escolhida)

            return

        else:
            # ? [begin] Logs para verificar [...]
            DebugLogger.log(f"[UPDATE FAIL] updateTime = {updateTime} | rotulo = {rotulo:.1f} | NENHUM SPFMiC candidato encontrado!")
            # ? [end]
            return

    '''
    # ? Nova função - Colocada aqui em 28/01/2026 - TEHO QUE RODAR PARA TESTAR, AINDA NÃO RODEI COM ESSA MODIFICAÇÃO
    def updateWithExample(self, ins, rotulo: float, updateTime: int):
        """
        Atualiza o modelo com um exemplo já classificado, mas escolhe o SPFMiC
        exatamente como o classify() antigo: dentre TODOS os SPFMiCs que cobrem o ponto,
        escolhe o de maior tipicidade e atualiza esse SPFMiC.
        """
        allSPFMiCS: List[SPFMiC] = []
        allSPFMiCS.extend(self.getAllSPFMiCsFromClassifier(SupervisedModel.classifier))

        example = Example(np.asarray(ins), True, updateTime)

        tipicidades: List[float] = []
        pertinencias: List[float] = []
        candidatos: List[SPFMiC] = []

        # 1) Mesma regra do classify(): candidatos são TODOS que cobrem o ponto
        for spfmic in allSPFMiCS:
            distancia = calculaDistanciaEuclidiana(example, spfmic.getCentroide())
            raio = spfmic.getRadiusWithWeight()
            if distancia <= raio:
                tip = spfmic.calculaTipicidade(example.getPonto(), spfmic.getN(), self.K)
                per = SupervisedModel.calculaPertinencia(example.getPonto(), spfmic.getCentroide(), self.fuzzification)
                tipicidades.append(tip)
                pertinencias.append(per)
                candidatos.append(spfmic)

        if len(candidatos) == 0:
            DebugLogger.log(
                f"[UPDATE FAIL] updateTime={updateTime} | rotulo={rotulo:.1f} | nenhum SPFMiC cobriu o ponto"
            )
            return

        # 2) Escolhe o vencedor por max tipicidade (igual ao classify)
        idx = tipicidades.index(max(tipicidades))
        spf_vencedor = candidatos[idx]
        per_vencedor = max(pertinencias)  # igual ao classify: usa maxValPer

        # 3) Atualiza exatamente como no classify(): tipicidade fixa = 1
        spf_vencedor.setUpdated(updateTime)
        spf_vencedor.atribuiExemplo(example, per_vencedor, 1)

        DebugLogger.log(
            f"[UPDATE SUCCESS] updateTime={updateTime} | "
            f"rotulo_janela={rotulo:.1f} | rotulo_spf={spf_vencedor.getRotulo():.1f} | "
            f"tip_max={tipicidades[idx]:.4f} | per_max={per_vencedor:.4f}"
    )
    '''
  
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

    '''
    # ? NEW: classify_predict
    def classify_predict(self, ins, updateTime: int) -> Tuple[float, Optional[SPFMiC], Optional[float], Optional[float]]:
        # *Prediz um rótulo para 'ins' sem causar efeitos colaterais (sem atribuiExemplo e sem setUpdate)
        # *Retorna : (pred_label, spf_reference_or_None, maxPertinencia_or_None)

        example = Example(np.asarray(ins), True, updateTime)
        allSPFMiCSOfClassifier: List[SPFMiC] = []
        allSPFMiCSOfClassifier.extend(self.getAllSPFMiCsFromClassifier(SupervisedModel.classifier))

        tipicidades: List[float] = []
        pertinencia: List[float] = []
        auxSPFMiCs: List[SPFMiC] = []
        isOutlier: bool = True

        for i in range(len(allSPFMiCSOfClassifier)):
            spf = allSPFMiCSOfClassifier[i]
            distancia: float = calculaDistanciaEuclidiana(example, spf.getCentroide())
            raio = spf.getRadiusWithWeight()
            if distancia <= raio:
                isOutlier = False
                tipicidades.append(spf.calculaTipicidade(example.getPonto(), spf.getN(), self.K))
                pertinencia.append(SupervisedModel.calculaPertinencia(example.getPonto(), spf.getCentroide(), self.fuzzification))
                auxSPFMiCs.append(spf)

        if isOutlier:
            return -1, None, None, None
        
        maxValTip: float = max(tipicidades)
        indexMaxTip: int = tipicidades.index(maxValTip)
        maxValPer: float = max(pertinencia)
        chosen_spf: SPFMiC = auxSPFMiCs[indexMaxTip]

        return chosen_spf.getRotulo(), chosen_spf, float(maxValPer), float(maxValTip)
    '''
