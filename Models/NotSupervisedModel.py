from typing import List
from Structs.Example import Example
from Structs.SPFMiC import SPFMiC
from FuzzyFunctions.DistanceMeasures import calculaDistanciaEuclidiana
from DebugLogger import DebugLogger
from Models.SupervisedModel import SupervisedModel
import numpy as np

class NotSupervisedModel:
    def __init__(self):
        self.spfMiCS: List[SPFMiC] = []

    def classify(self, example: Example, K: float, updated: int) -> float:
        tipicidades: List[float] = []
        auxSPFMiCs: List[SPFMiC] = []
        isOutlier = True

        for spf in self.spfMiCS:
            distancia: float = calculaDistanciaEuclidiana(example, spf.getCentroide())
            if distancia <= spf.getRadiusUnsupervised():
                isOutlier = False
                tipicidades.append(spf.calculaTipicidade(example.getPonto(), spf.getN(), K))
                auxSPFMiCs.append(spf)

        if isOutlier:
            return -1.0

        maxVal = max(tipicidades)
        indexMax = tipicidades.index(maxVal)

        spfmic: SPFMiC = auxSPFMiCs[indexMax]
        idx = self.spfMiCS.index(spfmic)

        self.spfMiCS[idx].setUpdated(updated)
        #self.spfMiCS[idx].setUpdated(float(updated))

        return self.spfMiCS[idx].getRotulo()


    def addNewSPFMiC(self, spfmic: SPFMiC, supervised_model=None):
        """
        Adiciona um novo SPFMiC ao modelo não supervisionado
        e opcionalmente sincroniza com o modelo supervisionado.
        """
        if spfmic is None:
            return

        # 1️⃣ Adiciona localmente
        self.spfMiCS.append(spfmic)

        # 2️⃣ Se o modelo supervisionado for fornecido, sincroniza
        if supervised_model is not None:
            rotulo = spfmic.getRotulo()
            # Se a classe ainda não existe, cria lista
            if rotulo not in supervised_model.classifier:
                supervised_model.classifier[rotulo] = []
                if rotulo not in supervised_model.knowLabels:
                    supervised_model.knowLabels.append(rotulo)

            # Evita duplicatas do mesmo centroide
            ja_existe = any(
                np.allclose(spfmic.getCentroide(), existente.getCentroide())
                for existente in supervised_model.classifier[rotulo]
            )
            if not ja_existe:
                supervised_model.classifier[rotulo].append(spfmic)
    

    def removeOldSPFMiCs(self, ts: int, currentTime: int) -> None:
        spfMiCSAux = list(self.spfMiCS)
        print(f"Aqui está: {spfMiCSAux}")
        for spf in self.spfMiCS:
            if (currentTime - spf.getT() > ts) and (currentTime - spf.getUpdated() > ts):
                spfMiCSAux.remove(spf)
        self.spfMiCS = spfMiCSAux

