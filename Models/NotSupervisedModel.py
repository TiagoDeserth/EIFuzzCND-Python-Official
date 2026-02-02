from typing import List, Tuple, Optional, Dict
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

    '''
    # ? NEW: classify_predict
    def classify_predict(self, example: Example, K: float) -> Tuple[float, Optional[SPFMiC], Optional[float]]:
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
            return -1.0, None, None
        
        maxVal = max(tipicidades)
        indexMax = tipicidades.index(maxVal)
        spfmic: SPFMiC = auxSPFMiCs[indexMax]
        
        return spfmic.getRotulo(), spfmic, float(maxVal)
    '''
        

    # ! 26/01/2026
    '''
    def updateWithExample(self, ins, rotulo: float, updateTime: int):
  
        #    Atualiza incrementalmente um SPFMiC do MCD com novo exemplo.
        #    Usado pela janela deslizante.
        
        example = Example(np.asarray(ins), True, updateTime)
        
        # Procura SPFMiCs com o rótulo especificado
        candidatos: List[Tuple[SPFMiC, float, int]] = []
        
        for idx, spfmic in enumerate(self.spfMiCS):
            if spfmic.getRotulo() == rotulo:
                distancia = calculaDistanciaEuclidiana(example, spfmic.getCentroide())
                raio = spfmic.getRadiusUnsupervised()
                
                if distancia <= raio:
                    # Calcula tipicidade (K fixo como no classify)
                    K = 5  # Ou passe como parâmetro se necessário
                    tipicidade = spfmic.calculaTipicidade(example.getPonto(), spfmic.getN(), K)
                    candidatos.append((spfmic, tipicidade, idx))
        
        if len(candidatos) > 0:
            # Escolhe o de maior tipicidade
            melhor = max(candidatos, key=lambda x: x[1])
            spfmic_escolhido = melhor[0]
            tipicidade_escolhida = melhor[1]
            idx_escolhido = melhor[2]
            
            # ✅ ATUALIZA COMPLETO (igual ao SupervisedModel):
            self.spfMiCS[idx_escolhido].setUpdated(updateTime)
            
            # ✅ Calcula pertinência (exponencial da distância)
            import math
            dataPoints = example.getPonto()
            centroide = spfmic_escolhido.getCentroide()
            distance_sq = float(np.sum((dataPoints - centroide) ** 2))
            m = 2.0  # fuzzification padrão (ou passe como parâmetro)
            pertinencia = math.exp(-distance_sq / m)
            
            # ✅ CHAMA atribuiExemplo() (atualiza centroide/raio/peso)
            self.spfMiCS[idx_escolhido].atribuiExemplo(example, pertinencia, tipicidade_escolhida)
            
            DebugLogger.log(f"[UPDATE MCD] tempo={updateTime} | rotulo={rotulo:.1f} | tip={tipicidade_escolhida:.4f} | pert={pertinencia:.4f}")
        else:
            DebugLogger.log(f"[UPDATE MCD FAIL] tempo={updateTime} | rotulo={rotulo:.1f} | Nenhum SPFMiC candidato")
'''

