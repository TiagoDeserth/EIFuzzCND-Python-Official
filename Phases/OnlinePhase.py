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
                 kShort: int, phi: float, ts: int, minWeight: int, percentLabeled: float):
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
        #self.windowSize: int = max(1, int(windowSize))
        #self._bufferUpdates = deque()

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
        
        print("Primeiros 10 rótulos (classe) [OnlinePhase]: ", [row[-1] for row in data[:10]])
        print("Últimos 10 rótulos (classe) [OnlinePhase]: ", [row[-1] for row in data[-10:]])

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
            rotulo: float = self.supervisedModel.classifyNew(ins_array, tempo)

            #print(f"[STEP {tempo}] Classe real: {exemplo.getRotuloVerdadeiro()}  |  Classe prevista: {rotulo}")

            exemplo.setRotuloClassificado(rotulo)

            # ---> CÓDIGO ANTIGO <---
            if (exemplo.getRotuloVerdadeiro() not in trueLabels) or \
               (confusionMatrixOriginal.getNumberOfClasses() != self.tamConfusion):
                trueLabels.add(exemplo.getRotuloVerdadeiro())
                self.tamConfusion = confusionMatrixOriginal.getNumberOfClasses()

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
            # * ---> Novo código <---
            pred_label_sup, spf_ref_sup, maxPert_sup, tip_sup = self.supervisedModel.classify_predict(ins_array, tempo)

            if pred_label_sup != -1.0:
                exemplo.setRotuloClassificado(pred_label_sup)

                # ? Se calcula o dist^2 agora e guardamos ponto e per para agregação
                dist_sq = 0.0
                if spf_ref_sup is not None:
                    dist = calculaDistanciaEuclidiana(exemplo.getPonto(), spf_ref_sup.getCentroide())
                    dist_sq = dist * dist

                # Se tip_sup estiver None, calcule pela função do SPF se possível
                tip_val = tip_sup if tip_sup is not None else (spf_ref_sup.calculaTipicidade(exemplo.getPonto(), spf_ref_sup.getN(), self.supervisedModel.K) if spf_ref_sup is not None else 1.0)
                pert_val = maxPert_sup if maxPert_sup is not None else 1.0

                self._bufferUpdates.append({
                    'ex': exemplo,
                    't': tempo,
                    'pred_label': pred_label_sup,
                    'spf_ref': spf_ref_sup,
                    #'pert': maxPert_sup if maxPert_sup is not None else 1.0,
                    'pert': float(pert_val),
                    'tip': float(tip_val),
                    'dist_sq': float(dist_sq)
                })
            else:
                pred_label_ns, spf_ref_ns, tip_ns = self.notSupervisedModel.classify_predict(exemplo, self.supervisedModel.K)
                exemplo.setRotuloClassificado(pred_label_ns)
                if pred_label_ns == -1.0:
                    unkMem.append(exemplo)

                    # ? Não se adiciona atualização para desconhecidos
                    self._bufferUpdates.append({
                        'ex': exemplo,
                        't': tempo,
                        'pred_label': -1.0,
                        'spf_ref': None,
                        'pert': 0.0,
                        'tip': 0.0,
                        'dist_sq': 0.0
                    })
                else:
                    dist = 0.0
                    if spf_ref_ns is not None:
                        dist = calculaDistanciaEuclidiana(exemplo.getPonto(), spf_ref_ns.getCentroide())
                        dist_sq = dist * dist

                    pert_val_ns = 1.0  # para não-supervisionado, pertinência tradicional não é definida — use 1.0 ou calcule se desejar
                    tip_val_ns = tip_ns if tip_ns is not None else (spf_ref_ns.calculaTipicidade(exemplo.getPonto(), spf_ref_ns.getN(), self.supervisedModel.K) if spf_ref_ns is not None else 1.0)

                    self._bufferUpdates.append({
                        'ex': exemplo,
                        't': tempo,
                        'pred_label': pred_label_ns,
                        'spf_ref': spf_ref_ns,
                        'pert': float(pert_val_ns),
                        'tip': float(tip_val_ns),
                        'dist_sq': float(dist_sq)
                    })
            # * ---> Fim do Novo código <---
            '''

            self.results.append(exemplo)
            confusionMatrix.addInstance(exemplo.getRotuloVerdadeiro(), exemplo.getRotuloClassificado())

            tempoLatencia += 1
            if tempoLatencia >= self.latencia:
                if (random.random() < self.percentLabeled) or (len(labeledMem) == 0):
                    labeledExample = Example(esperandoTempo[nExeTemp], True, tempo)
                    labeledMem.append(labeledExample)

                if len(labeledMem) >= self.tChunk:
                    #Log 30/10
                    #DebugLogger.log(f"[DEBUG] Re-treinando classificador em tempo={tempo}, labeledMem={len(labeledMem)}")

                    labeledMem = self.supervisedModel.trainNewClassifier(labeledMem, tempo)
                    labeledMem.clear()
                nExeTemp += 1

            self.supervisedModel.removeOldSPFMiCs(self.latencia + self.ts, tempo)
            #self.notSupervisedModel.removeOldSPFMiCs(self.latencia + self.ts, tempo)
            self.removeOldUnknown(unkMem, self.ts, tempo)

            '''
            # * ---> Novo código <---
            # ? Aplica Updates Agregados quando o buffer (janela) enche
            if len(self._bufferUpdates) >= self.windowSize:
                # ? Agregadores por SPFMiC (chave: id(spf) quando spf_ref disponível, senão chave por label + tipo)
                agg_by_spf = {}

                # ? Estrutura de fallback para SPFs não-supervisionados identificados por rótulo
                agg_by_label_ns = {}

                for item in list(self._bufferUpdates):
                    pred = item['pred_label']
                    if pred == -1.0:
                        continue
                    spf_ref = item['spf_ref']
                    pert = float(item['pert'])
                    tip = float(item['tip'])
                    ex = item['ex']
                    dist_sq = float(item['dist_sq'])
                    ponto = np.array(ex.getPonto(), dtype = float)
                    
                    if spf_ref is not None:
                        key = id(spf_ref)
                        if key not in agg_by_spf:
                            agg_by_spf[key] = {
                                'spf': spf_ref,
                                'N_add': 0.0,
                                'sum_pert_alpha': 0.0,
                                'sum_tip_theta': 0.0,
                                'sum_SSDe': 0.0,
                                'sum_CF1pert': np.zeros_like(spf_ref.getCF1pertinencias()),
                                'sum_CF1tip': np.zeros_like(spf_ref.getCF1tipicidades())
                            }
                        a = agg_by_spf[key]
                        a['N_add'] += 1.0
                        a['sum_pert_alpha'] += (pert ** spf_ref.alpha) if hasattr(spf_ref, 'alpha') else (pert ** self.supervisedModel.alpha)
                        a['sum_tip_theta'] += (tip ** spf_ref.theta) if hasattr(spf_ref, 'theta') else (tip ** self.supervisedModel.theta)
                        a['sum_SSDe'] += pert * dist_sq
                        a['sum_CF1pert'] += ponto * pert
                        a['sum_CF1tip'] += ponto * tip
                    else:
                        # ? Fallback para não-supervisionado: agrupar por rótulo
                        label = pred
                        if label not in agg_by_label_ns:
                            # ? Tenta achar spf existente no notSupervisedModel com esse rótulo
                            spf_match = None
                            for spf in self.notSupervisedModel.spfMiCS:
                                if spf.getRotulo() == label:
                                    spf_match = spf
                                    break
                            agg_by_label_ns[label] = {
                                'spf': spf_match,
                                'N_add': 0.0,
                                'sum_pert_alpha': 0.0,
                                'sum_tip_theta': 0.0,
                                'sum_SSDe': 0.0,
                                'sum_CF1pert': None if spf_match is None else np.zeros_like(spf_match.getCF1pertinencias()),
                                'sum_CF1tip': None if spf_match is None else np.zeros_like(spf_match.getCF1tipicidades())
                            }
                        b = agg_by_label_ns[label]
                        b['N_add'] += 1.0
                        # ? alpha/theta from found spf or supervisedModel defaults
                        alpha_local = b['spf'].alpha if (b['spf'] is not None and hasattr(b['spf'], 'alpha')) else self.supervisedModel.alpha
                        theta_local = b['spf'].theta if (b['spf'] is not None and hasattr(b['spf'], 'theta')) else self.supervisedModel.theta
                        b['sum_pert_alpha'] += (pert ** alpha_local)
                        b['sum_tip_theta'] += (tip ** theta_local)
                        b['sum_SSDe'] += pert * dist_sq
                        if b['sum_CF1pert'] is None:
                            b['sum_CF1pert'] = ponto * pert
                            b['sum_CF1tip'] = ponto * tip
                        else:
                            b['sum_CF1pert'] += ponto * pert
                            b['sum_CF1tip'] += ponto * tip
                
                # ? Agora aplicar atualizações agregadas
                # ? 1) supervisionados (usando referência direta ao objeto SPFMiC)
                for key, a in agg_by_spf.items():
                    spf_obj: SPFMiC = a['spf']
                    # ? Soma os componentes diretamente
                    spf_obj.N += a['N_add']
                    # ? Atualiza Me e Te
                    spf_obj.Me += a['sum_pert_alpha']
                    spf_obj.Te += a['sum_tip_theta']
                    spf_obj.SSDe += a['sum_SSDe']
                    # ? Atualiza CF1 vetoriais
                    cf1p = spf_obj.getCF1pertinencias()
                    cf1t = spf_obj.getCF1tipicidades()
                    spf_obj.setCF1pertinencias(cf1p + a['sum_CF1pert'])
                    spf_obj.setCF1tipicidades(cf1t + a['sum_CF1tip'])
                    # ? Atualiza tempo
                    spf_obj.setUpdated(tempo)
                    # ? Recalcula centroide a partir dos CF1s / Me/Te
                    spf_obj.atualizaCentroide()
                
                # ? 2) não-supervisionados (agrupados por label)
                for label, b in agg_by_label_ns.items():
                    if b['spf'] is not None:
                        spf_obj: SPFMiC = b['spf']
                        spf_obj.N += b['N_add']
                        spf_obj.Me += b['sum_pert_alpha']
                        spf_obj.Te += b['sum_tip_theta']
                        spf_obj.SSDe += b['sum_SSDe']
                        spf_obj.setCF1pertinencias(spf_obj.getCF1pertinencias() + b['sum_CF1pert'])
                        spf_obj.setCF1tipicidades(spf_obj.getCF1tipicidades() + b['sum_CF1tip'])
                        spf_obj.setUpdated(tempo)
                        spf_obj.atualizaCentroide()
                    else:
                        # ? Se não existe spf com esse rótulo nos não-supervisionados, opcionalmente criar
                        # ? Criando novo SPFMiC com centroid inicial = média dos pontos agregados
                        cf_sum = b['sum_CF1pert']
                        if cf_sum is not None:
                            centroide = (cf_sum / max(b['N_add'], 1.0))
                            novo_spf = SPFMiC(centroide, int(b['N_add']), self.supervisedModel.alpha, self.supervisedModel.theta, tempo)
                            novo_spf.setCF1pertinencias(b['sum_CF1pert'])
                            novo_spf.setCF1tipicidades(b['sum_CF1tip'])
                            novo_spf.setSSDe(b['sum_SSDe'])
                            novo_spf.setMe(b['sum_pert_alpha'])
                            novo_spf.setTe(b['sum_tip_theta'])
                            novo_spf.setRotulo(label)
                            novo_spf.setUpdated(tempo)
                            novo_spf.atualizaCentroide()
                            self.notSupervisedModel.spfMiCS.append(novo_spf)

                # ? Limpa o buffer
                self._bufferUpdates.clear()
                # * ---> Fim do Novo código <---
                '''

            if (tempo > 0) and (tempo % self.divisor == 0):
                confusionMatrix.mergeClasses(confusionMatrix.getClassesWithNonZeroCount())
                metrics: Metrics = confusionMatrix.calculateMetrics(tempo, confusionMatrix.countUnknow(), self.divisor)
                DebugLogger.log(f"Tempo: {tempo} | Acurácia: {metrics.getAccuracy()} | Precision: {metrics.getPrecision()}")

                listaMetricas.append(metrics)
                if self.existNovelty:
                    self.novelties.append(1.0)
                    self.existNovelty = False
                else:
                    self.novelties.append(0.0)

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

    #AUXILIAR
    def multiClassNoveltyDetection(self, listaDesconhecidos: List[Example], tempo: int,
                                   confusionMatrix: ConfusionMatrix,
                                   confusionMatrixOriginal: ConfusionMatrix) -> List[Example]:

        #Log 30/10
        #DebugLogger.log(f"[DEBUG] multiClassNoveltyDetection: entrada len={len(listaDesconhecidos)}, tempo={tempo}")

        if len(listaDesconhecidos) > self.kShort:
            clusters = FuzzyFunctions.fuzzyCMeans(listaDesconhecidos, self.kShort, self.supervisedModel.fuzzification)

            # shapes de membership e centroides
            try:
                u_shape = clusters.membership.shape  # (n_exemplos, n_clusters) — conforme correção no FuzzyFunctions
            except Exception:
                u_shape = None

            centroides_list = clusters.getClusters()

            DebugLogger.log(f"[DEBUG] kShort={self.kShort} | Clusters formados={len(centroides_list)}")

            #Log 30/10
            #DebugLogger.log(f"[DEBUG] Clusters formados={len(centroides_list)}")

            silhuetas: List[float] = FuzzyFunctions.fuzzySilhouette(clusters, listaDesconhecidos, self.supervisedModel.alpha)
            #DebugLogger.log(f"[DEBUG] silhuetas={silhuetas}")

            DebugLogger.log(f"[DEBUG] Silhuetas calculadas={len(silhuetas)}")

            # resumo de quantidades de pontos por cluster (apenas tamanhos)
            pontos_por_cluster = [len(c['points']) for c in centroides_list]
            #DebugLogger.log(f"[DEBUG] multiClassNoveltyDetection: pontos_por_cluster={pontos_por_cluster}")

            silhuetasValidas: List[int] = []
            for i in range(len(silhuetas)):
                if (silhuetas[i] > 0) and (len(centroides_list[i]['points']) >= self.minWeight):
                    silhuetasValidas.append(i)
            DebugLogger.log(f"[DEBUG] Silhuetas válidas={len(silhuetasValidas)}")

            #Log 30/10
            #DebugLogger.log(f"[DEBUG] Silhuetas válidas={len(silhuetasValidas)}")

            sfMiCS: List[SPFMiC] = FuzzyFunctions.newSeparateExamplesByClusterClassifiedByFuzzyCMeans(
                listaDesconhecidos, clusters, -1, self.supervisedModel.alpha, self.supervisedModel.theta,
                self.minWeight, tempo)
            sfmicsConhecidos: List[SPFMiC] = self.supervisedModel.getAllSPFMiCs()
            DebugLogger.log(f"[DEBUG] sfMiCS gerados={len(sfMiCS)} | SPFMiCs conhecidos={len(sfmicsConhecidos)}")

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
                        DebugLogger.log(f"[DEBUG] Cluster {i} | di={di:.6f} | dj={dj:.6f} | dist={dist:.6f}")
                        #dist = calculaDistanciaEuclidiana(sfmicsConhecidos[j].getCentroide(), sfMiCS[i].getCentroide())
                        frs.append((di + dj) / dist)
                    #DebugLogger.log(f"[DEBUG] FRs(i={i}): {frs}, minFr={min(frs) if frs else None}, phi={self.phi}")

                    if len(frs) > 0:
                        minFr: float = min(frs)
                        indexMinFr: int = frs.index(minFr)

                        DebugLogger.log(f"[DEBUG] minFr={minFr:.6f} | phi={self.phi} | known={minFr <= self.phi}")

                        if minFr <= self.phi:
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
                            DebugLogger.log(f"[DEBUG] Novo SPFMiC label={sfMiCS[i].getRotulo()} | Real={sfMiCS[i].getRotuloReal()}")

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
                            #self.notSupervisedModel.addNewSPFMiC(sfMiCS[i], self.supervisedModel)

        return listaDesconhecidos

    def generateNPLabel(self) -> float:
        self.nPCount += 1
        DebugLogger.log(f"[DEBUG] Novo rótulo gerado: {self.nPCount}")
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
        confusionMatrix.updateConfusionMatrix(trueLabel)

    def getTamConfusion(self) -> int:
        return self.tamConfusion

    def setTamConfusion(self, tamConfusion: int):
        self.tamConfusion = tamConfusion