import numpy as np
import skfuzzy as fuzz
from typing import List, Dict
from FuzzyFunctions.DistanceMeasures import calculaDistanciaEuclidiana
from Structs.Example import Example
from Structs.SPFMiC import SPFMiC
import math
import random
from DebugLogger import DebugLogger

class FuzzyClusterResult:
    """Wrapper para compatibilidade com Java FuzzyKMeansClusterer"""
    def __init__(self, centroids: np.ndarray, membership: np.ndarray, examples: List[Example]):
        self.centroids = centroids
        self.membership = membership
        self._examples = examples
        self._clusters_cache = None

    def getClusters(self):
        """Retorna lista de clusters com centroides e pontos"""
        if self._clusters_cache is not None:
            return self._clusters_cache
        
        clusters = []
        for i in range(self.centroids.shape[0]):
            indices = [j for j in range(self.membership.shape[0]) #<--- mudei o [1] para [0]
                      if np.argmax(self.membership[j, :]) == i] #<--- mudei o [:, j] para [j, :]
            points = [self._examples[j] for j in indices]
            
            cluster = {
                'centroid': self.centroids[i],
                'points': points
            }
            clusters.append(cluster)
        
        self._clusters_cache = clusters

        #print(f"[DEBUG] getClusters: {len(clusters)} clusters -> {[len(c['points']) for c in clusters]}")

        return clusters

class FuzzyFunctions:
    @staticmethod
    def fuzzyCMeans(examples: List[Example], K: int, fuzzification: float):

        #data = np.array([ex.getPonto() for ex in examples], dtype=np.float64).T
        data = np.array([ex.getPonto() for ex in examples], dtype=np.float64)
        #print(f"[DEBUG] Entrada fuzzyCMeans: data.shape={data.shape}")

        # *Prints 11/11/2025
        # *DebugLogger.log(f"[DEBUG FuzzyCMeans] data.shape={data.shape}")

        # Corrige se estiver transposta
        # !if data.shape[0] < data.shape[1]:
            # !data = data.T
            # !#DebugLogger.log(f"[DEBUG] Corrigido transpose: data agora {data.shape}")

        cntr, u, *_ = fuzz.cluster.cmeans(
            data.T, c=K, m=fuzzification, error=1e-6, maxiter=1000, init=None
        )
        # *Prints 11/11/2025
        # *DebugLogger.log(f"[DEBUG CMEANS] cntr.shape={cntr.shape}, u.shape={u.shape}")

        #print(f"[DEBUG] cmeans: cntr={cntr.shape}, u={u.shape}, u.min={u.min()}, u.max={u.max()}")

        #ALTERAÇÕES (POR FAVOR, DEUS)
        u_corrigido = u.T.copy()

        #print(f"[DEBUG] u_corrigido={u_corrigido.shape}, sample_row={u_corrigido[0][:5]}")

        #DebugLogger.log(f"[DEBUG] fuzzyCMeans: data={data.shape}, u_original={u.shape}, u_corrigido={u_corrigido.shape}, cntr={cntr.shape}")

        return FuzzyClusterResult(cntr, u_corrigido, examples)

    @staticmethod
    def getFirstAndSecondBiggerPertinence(valores: np.ndarray, j: int = 0):
        lista = sorted(valores, reverse=True)
        if len(lista) < 2:
            return (lista[0], 0.0) if lista else (0.0, 0.0)
        return lista[0], lista[1]

    @staticmethod
    def fuzzySilhouette(clusterer: FuzzyClusterResult, exemplos: List[Example], alpha: float):
        u = clusterer.membership
        #u_T = u.T
        nExemplos, K = u.shape
        silhuetas: List[float] = []
        numerador = 0.0
        denominador = 0.0
        #K = u_T.shape[1]
        apj: float = 0.0
        dqj: List[float] = []

        for i in range(K):
        #for i in range(u.shape[0]):
            for j in range(nExemplos):
                #indexClasse = np.argmax(u[:, j])
                #indexClasse = int(np.argmax(u[j])) #<---
                indexClasse: int = FuzzyFunctions.getIndiceDoMaiorValor(u[j])
                if indexClasse == i:
                    for k in range(nExemplos):
                        #if np.argmax(u[:, k]) == indexClasse:
                        #if int(np.argmax(u[k])) == indexClasse: #<---
                        if FuzzyFunctions.getIndiceDoMaiorValor(u[k]) == indexClasse:
                            apj += calculaDistanciaEuclidiana(exemplos[j].getPonto(), exemplos[k].getPonto())
                        else:
                            dqj.append(calculaDistanciaEuclidiana(exemplos[j].getPonto(), exemplos[k].getPonto()))
                    apj = apj / nExemplos
                    #apj /= max(nExemplos, 1)
                    if dqj:
                    #if len(dqj) > 0:
                        bpj: float = min(dqj)
                        #bpj: float = float(min(dqj))
                        sj: float = (bpj - apj) / max(apj, bpj)
                        #upj, uqj = FuzzyFunctions.getFirstAndSecondBiggerPertinence(u[:, j], j)
                        #upj, uqj = FuzzyFunctions.getFirstAndSecondBiggerPertinence(u[j], j)
                        maiorESegundaMaiorPertinencia = FuzzyFunctions.getFirstAndSecondBiggerPertinence(u[j], j)
                        upj = maiorESegundaMaiorPertinencia[0]
                        uqj = maiorESegundaMaiorPertinencia[1]
                        #numerador += ((upj - uqj) ** alpha) * sj
                        #denominador += (upj - uqj) ** alpha
                        numerador += math.pow((upj - uqj), alpha) * sj
                        denominador += math.pow((upj -uqj), alpha)
            fs: float = numerador / denominador if denominador != 0 else 0.0

            #DebugLogger.log(f"[DEBUG] fuzzySilhouette: u={u.shape}, examples={len(exemplos)}, silhuetas={len(silhuetas)}")
            silhuetas.append(fs)

        return silhuetas

    @staticmethod
    def calculaTipicidade(membership_matrix: np.ndarray):
        n, k = membership_matrix.shape
        typicality = np.zeros((n, k))
        for i in range(n):
            max_u_i = np.max(membership_matrix[i])
            for j in range(k):
                typicality[i][j] = membership_matrix[i][j] / max_u_i if max_u_i > 0 else 0

        #DebugLogger.log(f"[DEBUG] calculaTipicidade: input={membership_matrix.shape}, output={typicality.shape}")
        return typicality

    '''
    ORIGINAL
    @staticmethod
    def separateExamplesByClusterClassifiedByFuzzyCMeans(
            exemplos: List[Example],
            cntr: np.ndarray,
            u: np.ndarray,
            rotulo: float,
            alpha: float,
            theta: float,
            minWeight: int,
            t: int
    ) -> List[SPFMiC]:

        DebugLogger.log(
            f"[DEBUG] separateExamplesByClusterClassifiedByFuzzyCMeans: cntr={cntr.shape}, u={u.shape}, exemplos={len(exemplos)}")

        matrizTipicidade = FuzzyFunctions.calculaTipicidade(u)
        DebugLogger.log(f"[DEBUG] matrizTipicidade shape={matrizTipicidade.shape}")

        nExemplos, K = u.shape

        #K = cntr.shape[0]
        sfMiCS: List[SPFMiC] = []

        for j, centroide in enumerate(cntr):
            indices_do_cluster = [k for k in range(len(exemplos)) if np.argmax(u[k]) == j]
            nClusterPoints = len(indices_do_cluster)
            #DebugLogger.log(f"[DEBUG] cluster {j}: nClusterPoints={nClusterPoints}")
            
            if nClusterPoints == 0:
                continue

            spfmic = SPFMiC(centroide, nClusterPoints, alpha, theta, t) #<---
            spfmic.setRotulo(rotulo) #<---
            
            spfmic = None
            SSDe, Me, Te = 0.0, 0.0, 0.0
            CF1pertinencias = np.zeros_like(centroide, dtype=float)
            CF1tipicidades = np.zeros_like(centroide, dtype=float)
            #primeiro_ponto = True

            for k in indices_do_cluster:
                ex = exemplos[k]
                valorPert = float(u[k, j])
                valorTip = float(matrizTipicidade[k][j])
                ponto = np.array(ex.getPonto(), dtype=float)

                if spfmic is None:
                    spfmic = SPFMiC(centroide, nClusterPoints, alpha, theta, t)
                    spfmic.setRotulo(rotulo)

                dist = calculaDistanciaEuclidiana(centroide, ponto)
                Me += (valorPert ** alpha)
                Te += (valorTip ** theta)
                SSDe += valorPert * (dist ** 2)

                #22/10
                #if not primeiro_ponto:
                    #CF1pertinencias += ponto * valorPert
                    #CF1tipicidades += ponto * valorTip
                #else:
                    #primeiro_ponto = False

                CF1pertinencias += ponto * valorPert
                CF1tipicidades += ponto * valorTip

            if spfmic is not None and nClusterPoints >= minWeight:
                CF1pertinencias += spfmic.getCF1pertinencias()
                CF1tipicidades += spfmic.getCF1tipicidades()
                
                spfmic.setSSDe(SSDe)
                spfmic.setMe(Me)
                spfmic.setTe(Te)
                spfmic.setCF1pertinencias(CF1pertinencias)
                spfmic.setCF1tipicidades(CF1tipicidades)
                sfMiCS.append(spfmic)

        return sfMiCS
    '''

    @staticmethod
    def separateExamplesByClusterClassifiedByFuzzyCMeans(
            exemplos: List[Example],
            cntr: np.ndarray,
            u: np.ndarray,
            rotulo: float,
            alpha: float,
            theta: float,
            minWeight: int,
            t: int
    ) -> List[SPFMiC]:

        # Garante forma correta da matriz de pertinência
        #if u.shape[0] != len(exemplos):
            #u = u.T
            #DebugLogger.log(f"[DEBUG] Transposta aplicada em u: novo shape={u.shape}")

        matrizTipicidade = FuzzyFunctions.calculaTipicidade(u)
        K = cntr.shape[0]
        sfMiCS: List[SPFMiC] = []

        #DebugLogger.log(f"[DEBUG] separateExamplesByClusterClassifiedByFuzzyCMeans: cntr={cntr.shape}, u={u.shape}")

        for j, centroide in enumerate(cntr):
            # *Prints 11/11/2025
            # *DebugLogger.log(f"[DEBUG] centroide shape={centroide.shape}")

            indices_do_cluster = [k for k in range(len(exemplos)) if np.argmax(u[k]) == j]
            nClusterPoints = len(indices_do_cluster)

            if nClusterPoints == 0:
                continue

            spfmic = SPFMiC(centroide, nClusterPoints, alpha, theta, t)
            spfmic.setRotulo(rotulo)

            #spfmic = None
            SSDe, Me, Te = 0.0, 0.0, 0.0
            CF1pertinencias = np.zeros_like(centroide, dtype=float)
            CF1tipicidades = np.zeros_like(centroide, dtype=float)

            for k in indices_do_cluster:
                ex = exemplos[k]
                valorPert = float(u[k, j])
                valorTip = float(matrizTipicidade[k, j])
                ponto = np.array(ex.getPonto(), dtype=float)

                # *Prints 11/11/2025
                # *DebugLogger.log(f"[DEBUG Cluster] Centroide: {centroide.shape}, Ponto: {ponto.shape}")

                dist = calculaDistanciaEuclidiana(centroide, ponto)
                Me += (valorPert ** alpha)
                Te += (valorTip ** theta)
                SSDe += valorPert * (dist ** 2)
                CF1pertinencias += ponto * valorPert
                CF1tipicidades += ponto * valorTip

            if nClusterPoints >= minWeight:
                spfmic.setSSDe(SSDe)
                spfmic.setMe(Me)
                spfmic.setTe(Te)
                spfmic.setCF1pertinencias(CF1pertinencias)
                spfmic.setCF1tipicidades(CF1tipicidades)
                sfMiCS.append(spfmic)

            #DebugLogger.log(
                #f"[DEBUG] Cluster {j}: nPts={nClusterPoints}, SSDe={SSDe:.4f}, Me={Me:.4f}, Te={Te:.4f}"
            #)

        return sfMiCS

    '''
    ORIGINAL
    @staticmethod
    def newSeparateExamplesByClusterClassifiedByFuzzyCMeans(
            exemplos: List[Example],
            clusterer: FuzzyClusterResult,
            rotulo: float,
            alpha: float,
            theta: float,
            minWeight: int,
            t: int
    ) -> List[SPFMiC]:

        centroides_list = clusterer.getClusters()
        u = clusterer.membership
        sfMiCS: List[SPFMiC] = []

        DebugLogger.log(
            f"[DEBUG] newSeparateExamplesByClusterClassifiedByFuzzyCMeans: centroides={len(centroides_list)}, u={u.shape}")

        for j, cluster_info in enumerate(centroides_list):
            spfmic = None
            SSD = 0.0
            examples = cluster_info['points']
            nClusterPoints = len(examples)

            if nClusterPoints > 0:
                spfmic = SPFMiC(cluster_info['centroid'], nClusterPoints, alpha, theta, t)
                spfmic.setRotulo(rotulo)

                for ex in examples:
                    indexExample = exemplos.index(ex)
                    valorPert = float(u[indexExample, j]) #<---
                    dist = calculaDistanciaEuclidiana(cluster_info['centroid'], ex.getPonto())
                    SSD += valorPert * (dist ** 2)

                if nClusterPoints >= minWeight:
                    spfmic.setSSDe(SSD)

                sfMiCS.append(spfmic)

        return sfMiCS
    '''

    '''
    @staticmethod
    def newSeparateExamplesByClusterClassifiedByFuzzyCMeans(
            exemplos: List[Example],
            clusterer: FuzzyClusterResult,
            rotulo: float,
            alpha: float,
            theta: float,
            minWeight: int,
            t: int
    ) -> List[SPFMiC]:

        centroides_list = clusterer.getClusters()
        u = clusterer.membership
        sfMiCS: List[SPFMiC] = []

        #DebugLogger.log(
            #f"[DEBUG] newSeparateExamplesByClusterClassifiedByFuzzyCMeans: u={u.shape}, centroides={len(centroides_list)}")

        for j, cluster_info in enumerate(centroides_list):
            #spfmic = None
            #SSD = 0.0
            examples = cluster_info['points']
            nClusterPoints = len(examples)

            # MUDANÇA CRÍTICA: sempre cria um SPFMiC, mesmo vazio
            if nClusterPoints == 0:
                # Cria SPFMiC nulo/vazio para manter índice correto
                spfmic = SPFMiC(cluster_info['centroid'], 0, alpha, theta, t)
                spfmic.setRotulo(rotulo)
                spfmic.setNull(True)  # Marca como nulo
                sfMiCS.append(spfmic)  # ✅ SEMPRE adiciona
                continue

            spfmic = SPFMiC(cluster_info['centroid'], nClusterPoints, alpha, theta, t)
            spfmic.setRotulo(rotulo)
            SSD = 0.0

            
            #for ex in examples:
                #indexExample = exemplos.index(ex)
                #valorPert = float(u[indexExample, j])
                #dist = calculaDistanciaEuclidiana(cluster_info['centroid'], ex.getPonto())
                #SSD += valorPert * (dist ** 2)
            

            for k, ex in enumerate(examples):
            # Corrige o comportamento do Java:
            # Primeiro exemplo usa o índice global na lista 'exemplos',
            # demais usam o índice local k.
                if k == 0:
                    indexExample = exemplos.index(ex)
                else:
                    indexExample = k

                valorPert = float(u[indexExample, j])
                dist = calculaDistanciaEuclidiana(cluster_info['centroid'], ex.getPonto())
                SSD += valorPert * (dist ** 2)

            if nClusterPoints >= minWeight:
                spfmic.setSSDe(SSD)

            sfMiCS.append(spfmic)
            #DebugLogger.log(f"[DEBUG] Cluster {j}: nPts={nClusterPoints}, SSD={SSD:.4f}")

        return sfMiCS
    '''

    @staticmethod
    def newSeparateExamplesByClusterClassifiedByFuzzyCMeans(
            exemplos: List[Example],
            clusterer: FuzzyClusterResult,
            rotulo: float,
            alpha: float,
            theta: float,
            minWeight: int,
            t: int
    ) -> List[SPFMiC]:
        """
        Replica EXATAMENTE o comportamento do Java:
        - Sempre retorna uma lista com tamanho = número de centroides
        - Adiciona SPFMiC para CADA cluster, mesmo que seja None/null
        """
        
        sfMiCS: List[SPFMiC] = []
        matrizMembership = clusterer.membership
        centroides_list = clusterer.getClusters()
        
        # Itera sobre TODOS os centroides (como no Java)
        for j in range(len(centroides_list)):
            cluster_info = centroides_list[j]
            sfMiC = None
            SSD = 0.0
            examples = cluster_info['points']
            
            # Processa cada exemplo do cluster
            for k in range(len(examples)):
                indexExample = exemplos.index(examples[k])
                
                if sfMiC is None:
                    # Primeiro exemplo: cria o SPFMiC
                    sfMiC = SPFMiC(
                        cluster_info['centroid'],
                        len(examples),
                        alpha,
                        theta,
                        t
                    )
                    sfMiC.setRotulo(rotulo)
                    valorPertinencia = float(matrizMembership[indexExample][j])
                    ex = exemplos[k].getPonto()
                    distancia = calculaDistanciaEuclidiana(sfMiC.getCentroide(), ex)
                    SSD += valorPertinencia * (distancia ** 2)
                else:
                    # Exemplos subsequentes
                    valorPertinencia = float(matrizMembership[k][j])
                    ex = exemplos[k].getPonto()
                    distancia = calculaDistanciaEuclidiana(sfMiC.getCentroide(), ex)
                    SSD += valorPertinencia * (distancia ** 2)
            
            # Se o SPFMiC foi criado e atende ao peso mínimo
            if sfMiC is not None:
                if sfMiC.getN() >= minWeight:
                    sfMiC.setSSDe(SSD)
            
            # CRÍTICO: SEMPRE adiciona, mesmo que seja None (como no Java)
            sfMiCS.append(sfMiC)
        
        return sfMiCS


    @staticmethod
    def getIndiceDoMaiorValor(array: np.ndarray) -> int:
        index = 0
        maior = -1000000
        for i, val in enumerate(array):
            if val > maior:
                index = i
                maior = float(val)
        return index

    @staticmethod
    def separateByClasses(chunk: List[Example]) -> Dict[float, List[Example]]:
        examplesByClass: Dict[float, List[Example]] = {}
        for ex in chunk:
            r = ex.getRotuloVerdadeiro()
            if r not in examplesByClass:
                examplesByClass[r] = []
            examplesByClass[r].append(ex)
            #DebugLogger.log(f"[DEBUG] separateByClasses: { {k: len(v) for k, v in examplesByClass.items()} }")
        return examplesByClass

