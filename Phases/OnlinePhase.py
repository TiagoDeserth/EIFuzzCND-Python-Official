import random
from typing import List, Set, Dict
import numpy as np
from Models.SupervisedModel import SupervisedModel
from Models.NotSupervisedModel import NotSupervisedModel
from Structs.Example import Example
from Structs.SPFMiC import SPFMiC
from FuzzyFunctions.FuzzyFunctions import FuzzyFunctions
from FuzzyFunctions.DistanceMeasures import calculaDistanciaEuclidiana
from ConfusionMatrix.ConfusionMatrix import ConfusionMatrix
from ConfusionMatrix.Metrics import Metrics
from Output.HandlesFiles import HandlesFiles
from DebugLogger import DebugLogger
from collections import deque, defaultdict
from ClassMapper import ClassMapper

try:
    from scipy.io import arff
    import pandas as pd
except ImportError:
    arff = None
    pd = None

class OnlinePhase:
    def __init__(self, caminho: str, supervisedModel: SupervisedModel, latencia: int, tChunk: int, T: int,
                 kShort: int, phi: float, ts: int, minWeight: int, percentLabeled: float, 
                 windowSize: int
                 ):
        self.kShort: int = kShort
        self.ts: int = ts
        self.minWeight: int = minWeight
        self.T: int = T
        self.caminho: str = caminho
        self.latencia: int = latencia
        self.tChunk: int = tChunk
        self.supervisedModel = supervisedModel
        self.notSupervisedModel = NotSupervisedModel()
        self.phi: float = phi
        self.existNovelty: bool = False
        self.nPCount: float = 100
        self.novelties: List[float] = []
        self.percentLabeled: float = percentLabeled
        self.results: List[Example] = []
        self.divisor: int = 1000
        self.tamConfusion: int = 0

        # ? Janel para updates
        self.windowSize: int = windowSize
        self.classifiedWindow: List[Dict] = []

        DebugLogger.log(f"[INIT] OnlinePhase com windowSize = {self.windowSize}")

    def initialize(self, dataset: str):
        #Log 30/10
        #DebugLogger.log("[DEBUG] Iniciando OnlinePhase.initialize()")

        np.random.seed(42)
        random.seed(42)

        esperandoTempo = None
        nExeTemp = 0

        confusionMatrix = ConfusionMatrix()
        confusionMatrixOriginal = ConfusionMatrix()
        append = False
        metrics: Metrics
        listaMetricas: List[Metrics] = []

        path = f"{self.caminho}{dataset}-instances.arff"
        if arff is None or pd is None:
            raise RuntimeError("scipy.io.arff e pandas são necessários para ler ARFF.")

        # !data_np, meta = arff.loadarff(path)
        # !df = pd.DataFrame(data_np)
        #values = df.values
        #data = values
        # !df[df.columns[-1]] = pd.to_numeric(df[df.columns[-1]].astype(str), errors="coerce")
        # !data = df.values

        # *Modificações para transformação de labels nominais para numéricos (float)
        data_np, meta = arff.loadarff(path)
        df = pd.DataFrame(data_np)
        # !class_col = df.columns[-1]
        # !df[class_col] = df[class_col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
        # !classes_uniques = df[class_col].unique()
        # !class_for_index = {classe: idx for idx, classe in enumerate(classes_uniques)}
        # !df[class_col] = df[class_col].map(class_for_index).astype(float)
        # !data = df.values

        class_col = df.columns[-1]

        mapper = ClassMapper()
        class_mapping = mapper.get_mapping()

        ''' Testes 10/12
        nominal_values = meta[class_col][1]
        class_for_index = {val.decode() if isinstance(val, bytes) else val: idx
                           for idx, val in enumerate(nominal_values)}
        index_for_class = {idx: val for val, idx in class_for_index.items()}

        df[class_col] = df[class_col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
        df[class_col] = df[class_col].map(class_for_index).astype(float)
        '''

        df[class_col] = df[class_col].apply(
            lambda x: class_mapping.get(x.decode() if isinstance(x, bytes) else x, -1.0)
        )

        data = df.values
        
        #DebugLogger.log("Primeiros 10 rótulos (classe) [OnlinePhase]: ", [row[-1] for row in data[:10]])
        #DebugLogger.log("Últimos 10 rótulos (classe) [OnlinePhase]: ", [row[-1] for row in data[-10:]])

        #Log 30/10
        #DebugLogger.log(f"[DEBUG] Dataset carregado: {data.shape[0]} instâncias, {data.shape[1]} atributos")

        unkMem: List[Example] = []
        esperandoTempo = data
        labeledMem: List[Example] = []
        trueLabels: Set[float] = set()

        tempoLatencia = 0

        # ? Total
        #total_len = data.shape[0]

        for tempo in range(data.shape[0]):
        #for tempo in range(total_len):
            ins_array = np.asarray(data[tempo], dtype=float)
            exemplo = Example(ins_array, True, tempo)

            # !rotulo: float = self.supervisedModel.classifyNew(ins_array, tempo)

            rotulo: float = self.supervisedModel.classifyNewWithoutUpdate(ins_array, tempo)

            #DebugLogger.log(f"[STEP {tempo}] Classe real: {exemplo.getRotuloVerdadeiro()}  |  Classe prevista: {rotulo}")

            exemplo.setRotuloClassificado(rotulo)

            # ? [begin] Logs para verificar se a implementação da Janela Deslizante está funcionando
            if tempo % 100 == 0:
                DebugLogger.log(f"[STEP {tempo}] Classe real = {exemplo.getRotuloVerdadeiro():.1f} | MCC classificou = {rotulo:.1f}")
            # ? [end]
            
            # ---> CÓDIGO ANTIGO <---
            if (exemplo.getRotuloVerdadeiro() not in trueLabels) or \
               (confusionMatrixOriginal.getNumberOfClasses() != self.tamConfusion):
                trueLabels.add(exemplo.getRotuloVerdadeiro())
                self.tamConfusion = confusionMatrixOriginal.getNumberOfClasses()

                # ? [begin] Logs para verificar [...]
                DebugLogger.log(f"[NEW TRUE CLASS] Tempo = {tempo} | Classe = {exemplo.getRotuloVerdadeiro():.1f} | Total classes = {self.tamConfusion}")
                # ? [end]

            '''
            # ---> CÓDIGO ANTIGO: Antes da tentativa de implementação da Janela Deslizante <---
            if rotulo == -1 or rotulo == -1.0:
                rotulo = self.notSupervisedModel.classify(exemplo, self.supervisedModel.K, tempo)
                exemplo.setRotuloClassificado(rotulo)
                if rotulo == -1 or rotulo == -1.0:
                    unkMem.append(exemplo)
                    if len(unkMem) >= self.T:
                        #Log 30/10
                        #DebugLogger.log(
                            #f"[DEBUG] initialize(): chamando multiClassNoveltyDetection com unkMem={len(unkMem)} no tempo={tempo}")

                        unkMem = self.multiClassNoveltyDetection(unkMem, tempo, confusionMatrix, confusionMatrixOriginal)

                        #Log 30/10
                        #DebugLogger.log(
                            #f"[DEBUG] initialize(): retorno multiClassNoveltyDetection, unkMem={len(unkMem)} no tempo={tempo}")
            '''
            # ---> CÓDIGO ANTIGO: Antes da tentativa de implementação da Janela Deslizante <---

            # ---> CÓDIGO ANTIGO (JANELA): Depois da segunda tentiva de implementação da Janela Deslizante <---
            '''          
            if rotulo != -1 and rotulo != -1.0:
                # Armazena na Janela (SEM ATUALIZAR O MODELO)
                self.classifiedWindow.append({
                    'exemplo': exemplo,
                    'rotulo': rotulo,
                    'tempo': tempo,
                    'ins_array': ins_array
                })

                if len(self.classifiedWindow) >= self.windowSize:
                    #DebugLogger.log(f"[WINDOW] Janela cheia ({len(self.classifiedWindow)} exemplos) no tempo = {tempo}. Processando...")
                    self.processClassifiedWindow()
            
            else:
                rotulo = self.notSupervisedModel.classify(exemplo, self.supervisedModel.K, tempo)
                exemplo.setRotuloClassificado(rotulo)

                if rotulo != -1 and rotulo != -1.0:
                    # Armazena na Janela (SEM ATUALIZAR O MODELO)
                    self.classifiedWindow.append({
                        'exemplo': exemplo,
                        'rotulo': rotulo,
                        'tempo': tempo,
                        'ins_array': ins_array
                    })

                if len(self.classifiedWindow) >= self.windowSize:
                    #DebugLogger.log(f"[WINDOW] Janela cheia ({len(self.classifiedWindow)} exemplos) no tempo = {tempo}. Processando...")
                    self.processClassifiedWindow()

                if rotulo == -1 or rotulo == -1.0:
                    unkMem.append(exemplo)
                    if len (unkMem) >= self.T:
                        unkMem = self.multiClassNoveltyDetection(unkMem, tempo, confusionMatrix, confusionMatrixOriginal)
            '''
            # ---> CÓDIGO ANTIGO (JANELA): Depois da segunda tentiva de implementação da Janela Deslizante <---

            if rotulo == -1 or rotulo == -1.0:
                # ? [begin] Logs para verificar [...]
                if tempo % 100 == 0:
                    DebugLogger.log(f"[STEP {tempo}] MCC não classificou (-1.0), tentando MCD...")
                # ? [begin]

                rotulo = self.notSupervisedModel.classify(exemplo, self.supervisedModel.K, tempo)
                exemplo.setRotuloClassificado(rotulo)

                # ? [begin] Logs para verificar [...]
                if tempo % 100 == 0:
                    DebugLogger.log(f"[STEP {tempo}] MCD classificou = {rotulo:.1f}")
                # ? [end]

            if rotulo != -1 and rotulo != -1.0:
                self.classifiedWindow.append({
                    'exemplo': exemplo,
                    'rotulo': rotulo,
                    'tempo': tempo,
                    'ins_array': ins_array
                })

                # ? [begin] Logs para verificar [...]
                if tempo % 100 == 0 or len(self.classifiedWindow) == self.windowSize:
                    DebugLogger.log(f"[WINDOW] Tempo = {tempo} | Janela: {len(self.classifiedWindow)} / {self.windowSize} | Adicionado: rotulo = {rotulo:.1f}")
                # ? [end]

                if len (self.classifiedWindow) >= self.windowSize:

                    # ? [begin] Logs para verificar [...]
                    DebugLogger.log(f"[WINDOW FULL] Tempo = {tempo} | Processando {len(self.classifiedWindow)} exemplos...")
                    # ? [end]

                    self.processClassifiedWindow()

                    # ? [begin] Logs para verificar [...]
                    DebugLogger.log(f"[WINDOW DONE] Tempo = {tempo} | Janela limpa. Modelo atualizado.")
                    # ? [end]

            else: # rotulo == -1
                unkMem.append(exemplo)

                # ? [begin] Logs para verificar [...]
                if len(unkMem) % 10 == 0:
                    DebugLogger.log(f"[UNKMEM] Tempo = {tempo} | unkMem size = {len(unkMem)} / {self.T} | Classe real = {exemplo.getRotuloVerdadeiro():.1f}")
                # ? [end]

                if len(unkMem) >= self.T:

                    # ? [begin] Logs para verificar [...]
                    DebugLogger.log(f"[NOVELTY DETECTION] Tempo = {tempo} | unkMem = {len(unkMem)} >= T = {self.T} | Chamando multiClassNoveltyDetection...")
                    # ? [end]

                    # ? [begin] Logs para verificar [...]
                    unkMem_antes = len(unkMem)
                    # ? [end]

                    unkMem = self.multiClassNoveltyDetection(unkMem, tempo, confusionMatrix, confusionMatrixOriginal)

                    # ? [begin] Logs para verificar [...]
                    unkMem_depois = len(unkMem)
                    DebugLogger.log(f"[NOVELTY RESULT] Tempo = {tempo} | unkMem: {unkMem_antes} -> {unkMem_depois} | Removidos: {unkMem_antes - unkMem_depois}")
                    # ? [end]

            self.results.append(exemplo)
            confusionMatrix.addInstance(exemplo.getRotuloVerdadeiro(), exemplo.getRotuloClassificado())

            tempoLatencia += 1
            if tempoLatencia >= self.latencia:
                if (random.random() < self.percentLabeled) or (len(labeledMem) == 0):
                    labeledExample = Example(esperandoTempo[nExeTemp], True, tempo)
                    labeledMem.append(labeledExample)

                if len(labeledMem) >= self.tChunk:

                    # ? [begin] Logs para verificar [...]
                    DebugLogger.log(f"[RETRAIN] Tempo = {tempo} | labeledMem = {len(labeledMem)} | Re-treinamento classificador...")
                    # ? [end]

                    #Log 30/10
                    #DebugLogger.log(f"[DEBUG] Re-treinando classificador em tempo={tempo}, labeledMem={len(labeledMem)}")

                    labeledMem = self.supervisedModel.trainNewClassifier(labeledMem, tempo)
                    labeledMem.clear()
                nExeTemp += 1

            self.supervisedModel.removeOldSPFMiCs(self.latencia + self.ts, tempo)
            #self.notSupervisedModel.removeOldSPFMiCs(self.latencia + self.ts, tempo)
            self.removeOldUnknown(unkMem, self.ts, tempo)

            if (tempo > 0) and (tempo % self.divisor == 0):
                confusionMatrix.mergeClasses(confusionMatrix.getClassesWithNonZeroCount())
                metrics: Metrics = confusionMatrix.calculateMetrics(tempo, confusionMatrix.countUnknow(), self.divisor)
                DebugLogger.log(f"[METRICS] Tempo: {tempo} | Acurácia: {metrics.getAccuracy():.4f} | Precision: {metrics.getPrecision():.4f} | Unknown Rate = {metrics.getUnknownRate():.4f}")

                listaMetricas.append(metrics)
                if self.existNovelty:

                    # ? [begin] Logs para verificar [...]
                    DebugLogger.log(f"[NOVELTY FLAG] Tempo = {tempo} | Novidade foi detectada neste intervalo!")
                    # ? [[end]
                        
                    self.novelties.append(1.0)
                    self.existNovelty = False
                else:
                    self.novelties.append(0.0)

        if len(self.classifiedWindow) > 0:
            #DebugLogger.log(f"[WINDOW] Processando {len(self.classifiedWindow)} exemplos restantes na janela ao final")
            self.processClassifiedWindow()

        for metrica in listaMetricas:
            tempo_idx = int(metrica.getTempo() / self.divisor)
            HandlesFiles.salvaMetrics(
                tempo_idx,
                metrica.getAccuracy(),
                metrica.getPrecision(),
                metrica.getRecall(),
                metrica.getF1Score(),
                dataset,
                self.latencia,
                self.percentLabeled,
                metrica.getUnkMem(),
                metrica.getUnknownRate(),
                append
            )
            append = True

        HandlesFiles.salvaNovidades(self.novelties, dataset, self.latencia, self.percentLabeled)
        HandlesFiles.salvaResultados(self.results, dataset, self.latencia, self.percentLabeled)

        #Log 30/10
        #DebugLogger.log("[DEBUG] Finalizando OnlinePhase.initialize()")

    '''
    ORIGINAL
    def multiClassNoveltyDetection(self, listaDesconhecidos: List[Example], tempo: int,
                                   confusionMatrix: ConfusionMatrix,
                                   confusionMatrixOriginal: ConfusionMatrix) -> List[Example]:

        if len(listaDesconhecidos) > self.kShort:
            clusters = FuzzyFunctions.fuzzyCMeans(listaDesconhecidos, self.kShort, self.supervisedModel.fuzzification)
            centroides_list = clusters.getClusters()
            DebugLogger.log(f"[DEBUG] multiClassNoveltyDetection: len(centroides_list)={len(centroides_list)}")

            silhuetas = FuzzyFunctions.fuzzySilhouette(clusters, listaDesconhecidos, self.supervisedModel.alpha)
            DebugLogger.log(f"[DEBUG] silhuetas={silhuetas}")

            silhuetasValidas: List[int] = []
            for i in range(len(silhuetas)):
                if (silhuetas[i] > 0) and (len(centroides_list[i]['points']) >= self.minWeight):
                    silhuetasValidas.append(i)

            sfMiCS: List[SPFMiC] = FuzzyFunctions.newSeparateExamplesByClusterClassifiedByFuzzyCMeans(
                listaDesconhecidos, clusters, -1, self.supervisedModel.alpha, self.supervisedModel.theta,
                self.minWeight, tempo)
            sfmicsConhecidos: List[SPFMiC] = self.supervisedModel.getAllSPFMiCs()
            frs: List[float] = []

            DebugLogger.log(
                f"[DEBUG] multiClassNoveltyDetection: silhuetasValidas={silhuetasValidas}, sfMiCS={len(sfMiCS)}, sfmicsConhecidos={len(sfmicsConhecidos)}")

            for i in range(len(centroides_list)):
                if (i in silhuetasValidas) and (not sfMiCS[i].isNullFunc()):
                    frs.clear()
                    for j in range(len(sfmicsConhecidos)):
                        di: float = sfmicsConhecidos[j].getRadiusND()
                        dj: float = sfMiCS[i].getRadiusND()
                        dist: float = (di + dj) / calculaDistanciaEuclidiana(sfmicsConhecidos[j].getCentroide(), sfMiCS[i].getCentroide())
                        frs.append((di + dj) / dist)
                    DebugLogger.log(f"[DEBUG] FRs(i={i}): {frs}, minFr={min(frs) if frs else None}, phi={self.phi}")

                    if len(frs) > 0:
                        minFr: float = min(frs)
                        indexMinFr: int = frs.index(minFr)
                        if minFr <= self.phi:
                            sfMiCS[i].setRotulo(sfmicsConhecidos[indexMinFr].getRotulo())
                            examples: List[Example] = centroides_list[i]['points']
                            rotulos: Dict[float, int] = {}
                            for j in range(len(examples)):
                                listaDesconhecidos.remove(examples[j])
                                trueLabel: float = examples[j].getRotuloVerdadeiro()
                                predictedLabel: float = sfMiCS[i].getRotulo()
                                self.updateConfusionMatrix(trueLabel, predictedLabel, confusionMatrix)
                                rotulos[trueLabel] = rotulos.get(trueLabel, 0) + 1

                            maiorValor: float = -float('inf')
                            maiorRotulo: float = -1.0
                            for key, val in rotulos.items():
                                if maiorValor < val:
                                    maiorValor = val
                                    maiorRotulo = key
                            if maiorRotulo == sfMiCS[i].getRotulo():
                                sfMiCS[i].setRotuloReal(maiorRotulo)
                                self.notSupervisedModel.spfMiCS.append(sfMiCS[i])

                        else:
                            self.existNovelty = True
                            novoRotulo: float = self.generateNPLabel()
                            sfMiCS[i].setRotulo(novoRotulo)
                            examples: List[Example] = centroides_list[i]['points']
                            rotulos: Dict[float, int] = {}
                            for j in range(len(examples)):
                                listaDesconhecidos.remove(examples[j])
                                trueLabel = examples[j].getRotuloVerdadeiro()
                                predictedLabel = sfMiCS[i].getRotulo()
                                self.updateConfusionMatrix(trueLabel, predictedLabel, confusionMatrix)
                                rotulos[trueLabel] = rotulos.get(trueLabel, 0) + 1

                            maiorValor = -float('inf')
                            maiorRotulo = -1.0
                            for key, val in rotulos.items():
                                if maiorValor < val:
                                    maiorValor = val
                                    maiorRotulo = key

                            sfMiCS[i].setRotuloReal(maiorRotulo)
                            self.notSupervisedModel.spfMiCS.append(sfMiCS[i])

        return listaDesconhecidos
    '''

    # ? Nova função para Janela
    def processClassifiedWindow(self):
        '''
        Processa todos os exemplos na Janela classificada
        Atualiza o modelo incrementalmente para cada exemplo
        '''
        #DebugLogger.log(f"[WINDOW] Iniciando processamento de {len(self.classifiedWindow)} exemplos")

        for item in self.classifiedWindow:
            exemplo = item['exemplo']
            rotulo = item['rotulo']
            tempo = item['tempo']
            ins_array = item['ins_array']

            self.supervisedModel.updateWithExample(ins_array, rotulo, tempo)

            # ! 26/01/2026
            # self.notSupervisedModel.updateWithExample(ins_array, rotulo, tempo)
        
        processed_count = len(self.classifiedWindow)
        self.classifiedWindow.clear()
        
        #DebugLogger.log(f"[WINDOW] Processamento concluído. {processed_count} exemplos atualizados. Janela limpa.")
        
    #AUXILIAR
    def multiClassNoveltyDetection(self, listaDesconhecidos: List[Example], tempo: int,
                                   confusionMatrix: ConfusionMatrix,
                                   confusionMatrixOriginal: ConfusionMatrix) -> List[Example]:

        #Log 30/10
        #DebugLogger.log(f"[DEBUG] multiClassNoveltyDetection: entrada len={len(listaDesconhecidos)}, tempo={tempo}")

        # ? [begin] Logs para verificar [...]
        DebugLogger.log(f"[NOVELTY] Tempo={tempo} | Entrada: {len(listaDesconhecidos)} desconhecidos")
        # ? [end]

        if len(listaDesconhecidos) > self.kShort:
            clusters = FuzzyFunctions.fuzzyCMeans(listaDesconhecidos, self.kShort, self.supervisedModel.fuzzification)

            # shapes de membership e centroides
            try:
                u_shape = clusters.membership.shape  # (n_exemplos, n_clusters) — conforme correção no FuzzyFunctions
            except Exception:
                u_shape = None

            centroides_list = clusters.getClusters()

            # ! VOLTAR LOG DEPOIS (12/12/2025)
            #DebugLogger.log(f"[DEBUG] kShort={self.kShort} | Clusters formados={len(centroides_list)}")

            #Log 30/10
            #DebugLogger.log(f"[DEBUG] Clusters formados={len(centroides_list)}")

            silhuetas: List[float] = FuzzyFunctions.fuzzySilhouette(clusters, listaDesconhecidos, self.supervisedModel.alpha)
            #DebugLogger.log(f"[DEBUG] silhuetas={silhuetas}")

            # ! VOLTAR LOG DEPOIS (12/12/2025)
            #DebugLogger.log(f"[DEBUG] Silhuetas calculadas={len(silhuetas)}")

            # resumo de quantidades de pontos por cluster (apenas tamanhos)
            pontos_por_cluster = [len(c['points']) for c in centroides_list]
            #DebugLogger.log(f"[DEBUG] multiClassNoveltyDetection: pontos_por_cluster={pontos_por_cluster}")

            silhuetasValidas: List[int] = []
            for i in range(len(silhuetas)):
                if (silhuetas[i] > 0) and (len(centroides_list[i]['points']) >= self.minWeight):
                    silhuetasValidas.append(i)

            # ! VOLTAR LOG DEPOIS (12/12/2025)
            #DebugLogger.log(f"[DEBUG] Silhuetas válidas={len(silhuetasValidas)}")

            #Log 30/10
            #DebugLogger.log(f"[DEBUG] Silhuetas válidas={len(silhuetasValidas)}")

            sfMiCS: List[SPFMiC] = FuzzyFunctions.newSeparateExamplesByClusterClassifiedByFuzzyCMeans(
                listaDesconhecidos, clusters, -1, self.supervisedModel.alpha, self.supervisedModel.theta,
                self.minWeight, tempo)
            sfmicsConhecidos: List[SPFMiC] = self.supervisedModel.getAllSPFMiCs()

            # ! VOLTAR LOG DEPOIS (12/12/2025)
            #DebugLogger.log(f"[DEBUG] sfMiCS gerados={len(sfMiCS)} | SPFMiCs conhecidos={len(sfmicsConhecidos)}")

            #print("sfmicsConhecidos: ", sfmicsConhecidos)

            #Log 30/10
            #DebugLogger.log(f"[DEBUG] sfMiCS={len(sfMiCS)}, sfmicsConhecidos={len(sfmicsConhecidos)}")

            frs: List[float] = []

            #DebugLogger.log(
                #f"[DEBUG] multiClassNoveltyDetection: silhuetasValidas={silhuetasValidas}, sfMiCS={len(sfMiCS)}, sfmicsConhecidos={len(sfmicsConhecidos)}")

            for i in range(len(centroides_list)):

                if (i in silhuetasValidas) and (not sfMiCS[i].isNullFunc()):
                    frs.clear()
                    for j in range(len(sfmicsConhecidos)):
                        di: float = sfmicsConhecidos[j].getRadiusND()
                        dj: float = sfMiCS[i].getRadiusND()
                        dist: float = (di + dj) / calculaDistanciaEuclidiana(sfmicsConhecidos[j].getCentroide(), sfMiCS[i].getCentroide())
                        # ! VOLTAR LOG DEPOIS (12/12/2025)
                        #DebugLogger.log(f"[DEBUG] Cluster {i} | di={di:.6f} | dj={dj:.6f} | dist={dist:.6f}")
                        #dist = calculaDistanciaEuclidiana(sfmicsConhecidos[j].getCentroide(), sfMiCS[i].getCentroide())
                        frs.append((di + dj) / dist)
                    #DebugLogger.log(f"[DEBUG] FRs(i={i}): {frs}, minFr={min(frs) if frs else None}, phi={self.phi}")

                    if len(frs) > 0:
                        minFr: float = min(frs)
                        indexMinFr: int = frs.index(minFr)

                        # ! VOLTAR LOG DEPOIS (12/12/2025)
                        #DebugLogger.log(f"[DEBUG] minFr={minFr:.6f} | phi={self.phi} | known={minFr <= self.phi}")

                        if minFr <= self.phi:

                            # ? [begin] Logs para verificar [...]
                            DebugLogger.log(f"[KNOWN CLASS] Tempo={tempo} | Cluster {i} é classe conhecida {sfmicsConhecidos[indexMinFr].getRotulo():.1f} | minFr={minFr:.4f}")
                            # ? [end]
                            #Log 30/10
                            #DebugLogger.log(f"[DEBUG] Cluster {i} conhecido (minFr={minFr})")

                            sfMiCS[i].setRotulo(sfmicsConhecidos[indexMinFr].getRotulo())
                            examples: List[Example] = centroides_list[i]['points']
                            rotulos: Dict[float, int] = {}
                            for j in range(len(examples)):
                                listaDesconhecidos.remove(examples[j])
                                trueLabel: float = examples[j].getRotuloVerdadeiro()
                                predictedLabel: float = sfMiCS[i].getRotulo()
                                self.updateConfusionMatrix(trueLabel, predictedLabel, confusionMatrix)
                                rotulos[trueLabel] = rotulos.get(trueLabel, 0) + 1

                            maiorValor: float = -float('inf')
                            maiorRotulo: float = -1.0
                            for key, val in rotulos.items():
                                if maiorValor < val:
                                    maiorValor = val
                                    maiorRotulo = key
                            if maiorRotulo == sfMiCS[i].getRotulo():
                                sfMiCS[i].setRotuloReal(maiorRotulo)
                                self.notSupervisedModel.spfMiCS.append(sfMiCS[i])
                                #self.notSupervisedModel.addNewSPFMiC(sfMiCS[i], self.supervisedModel)

                        else:
                            #Log 30/10
                            #DebugLogger.log(f"[DEBUG] NOVIDADE detectada no cluster {i} (minFr={minFr})")

                            # ! VOLTAR LOG DEPOIS (12/12/2025)
                            #DebugLogger.log(f"[DEBUG] Novo SPFMiC label={sfMiCS[i].getRotulo()} | Real={sfMiCS[i].getRotuloReal()}")

                            self.existNovelty = True
                            novoRotulo: float = self.generateNPLabel()

                            # ? [begin] Logs para verificar [...]
                            DebugLogger.log(f"[NEW CLASS!!!] Tempo={tempo} | Cluster {i} é NOVA CLASSE | minFr={minFr:.4f} > phi={self.phi} | Novo rótulo={novoRotulo:.1f}")
                            # ? [end]
                            
                            sfMiCS[i].setRotulo(novoRotulo)
                            examples: List[Example] = centroides_list[i]['points']
                            rotulos: Dict[float, int] = {}
                            for j in range(len(examples)):
                                listaDesconhecidos.remove(examples[j])
                                trueLabel = examples[j].getRotuloVerdadeiro()
                                predictedLabel = sfMiCS[i].getRotulo()
                                self.updateConfusionMatrix(trueLabel, predictedLabel, confusionMatrix)
                                rotulos[trueLabel] = rotulos.get(trueLabel, 0) + 1

                            maiorValor = -float('inf')
                            maiorRotulo = -1.0
                            for key, val in rotulos.items():
                                if maiorValor < val:
                                    maiorValor = val
                                    maiorRotulo = key

                            sfMiCS[i].setRotuloReal(maiorRotulo)
                            self.notSupervisedModel.spfMiCS.append(sfMiCS[i])
                            #self.notSupervisedModel.addNewSPFMiC(sfMiCS[i], self.supervisedModel)

        # ? [begin] Logs para verificar [...]
        DebugLogger.log(f"[NOVELTY] Tempo={tempo} | Saída: {len(listaDesconhecidos)} desconhecidos restantes")
        # ? [end]

        return listaDesconhecidos

    def generateNPLabel(self) -> float:
        self.nPCount += 1

        # ! VOLTAR LOG DEPOIS (12/12/2025)
        #DebugLogger.log(f"[DEBUG] Novo rótulo gerado: {self.nPCount}")

        return self.nPCount

    def removeOldUnknown(self, unkMem: List[Example], ts: int, ct: int) -> List[Example]:
        newUnkMem: List[Example] = []
        for i in range(len(unkMem)):
            if ct - unkMem[i].getTime() >= ts:
                newUnkMem.append(unkMem[i])
        return newUnkMem

    @staticmethod
    def updateConfusionMatrix(trueLabel: float, predictedLabel: float, confusionMatrix: ConfusionMatrix):
        confusionMatrix.addInstance(trueLabel, predictedLabel)
        # ! Trativa para tentar implementar a Janela Deslizante
        confusionMatrix.updateConfusionMatrix(trueLabel)

    def getTamConfusion(self) -> int:
        return self.tamConfusion

    def setTamConfusion(self, tamConfusion: int):
        self.tamConfusion = tamConfusion