import numpy as np
from FuzzyFunctions.DistanceMeasures import calculaDistanciaEuclidiana
from Structs.Example import Example
from DebugLogger import DebugLogger
import math

class SPFMiC:
    def __init__(self, centroide: np.ndarray, N: int, alpha: float, theta: float, t: int):
        # CORREÇÃO CRÍTICA: CF1s devem começar com CÓPIA do centroide (como no Java)
        self.CF1pertinencias = np.array(centroide, dtype=float).copy()
        self.CF1tipicidades = np.array(centroide, dtype=float).copy()
        self.centroide = centroide

        self.Me = 1.0
        self.Te = 1.0
        self.SSDe = 0.0
        self.N = float(N)

        self.t = float(t)
        self.updated = float(t)
        self.created = float(t)

        self.rotulo = -1.0
        self.rotuloReal = -1.0
        self.alpha = alpha
        self.theta = theta
        self.isObsolete = False
        self.isNull = False

    # ---------- Getters / Setters ----------
    def getCF1pertinencias(self) -> np.ndarray:
        return self.CF1pertinencias

    def setCF1pertinencias(self, arr):
        self.CF1pertinencias = np.array(arr, dtype=float)

    def getCF1tipicidades(self) -> np.ndarray:
        return self.CF1tipicidades

    def setCF1tipicidades(self, arr):
        self.CF1tipicidades = np.array(arr, dtype=float)

    def setSSDe(self, SSDe: float):
        self.SSDe = SSDe

    def getN(self) -> float:
        return self.N

    def getTheta(self) -> float:
        return self.theta

    def getT(self) -> float:
        return self.t

    def getRotulo(self) -> float:
        return self.rotulo

    def setRotulo(self, rotulo: float):
        self.rotulo = rotulo

    def getCentroide(self) -> np.ndarray:
        return self.centroide

    def setCentroide(self, c):
        self.centroide = np.array(c, dtype=float)

    def setMe(self, me: float):
        self.Me = me

    def isNullFunc(self) -> bool:
        return self.isNull

    def setNull(self, value: bool):  # ← ADICIONE ESTE MÉTODO
        self.isNull = value

    def getRotuloReal(self) -> float:
        return self.rotuloReal

    def setRotuloReal(self, rotuloReal: float):
        self.rotuloReal = rotuloReal

    def getUpdated(self) -> float:
        return self.updated

    def setUpdated(self, u: float):
        self.updated = u

    def getCreated(self) -> float:
        return self.created

    def setTe(self, te: float):
        self.Te = te

    def setObsolete(self, b: bool):
        self.isObsolete = bool(b)

    def isObsoleteFunc(self) -> bool:
        return self.isObsolete

    # ---------- Operações principais ----------
    def atualizaCentroide(self):
        nAtributos: int = len(self.CF1pertinencias)
        self.centroide = np.zeros(nAtributos)
        for i in range(nAtributos):
            self.centroide[i] = (
                (self.alpha * self.CF1pertinencias[i] + self.theta * self.CF1tipicidades[i]) /
                    (self.alpha * self.Te + self.theta * self.Me if (self.alpha * self.Te + self.theta * self.Me) != 0 else 1e-12)
            )

    def atribuiExemplo(self, exemplo: Example, pertinencia: float, tipicidade: float):
        dist: float = calculaDistanciaEuclidiana(exemplo.getPonto(), self.centroide)
        self.N += 1
        self.Me += math.pow(pertinencia, self.alpha)
        self.Te += math.pow(tipicidade, self.theta)
        self.SSDe += math.pow(dist, 2) * pertinencia
        for i in range(len(self.centroide)):
            self.CF1pertinencias[i] += exemplo.getPontoPorPosicao(i) * pertinencia
            self.CF1tipicidades[i] += exemplo.getPontoPorPosicao(i) * tipicidade
        self.atualizaCentroide()

    '''
    def calculaTipicidade(self, exemplo: np.ndarray, n: float, K: float) -> float:
        gamma_i = self.getGamma(K)
        dist = calculaDistanciaEuclidiana(exemplo, self.centroide)
        return 1/ (1 + pow(((self.theta / gamma_i) * dist), (1 / (n - 1))))
    '''

    def calculaTipicidade(self, exemplo: np.ndarray, n: float, K: float) -> float:
        gamma_i: float = self.getGamma(K)
        dist: float = calculaDistanciaEuclidiana(exemplo, self.centroide)

        if n <= 1 or gamma_i == 0:
            return 0.0

        expoente = 1.0 / (n - 1.0)
        return 1.0 / (1.0 + math.pow(((self.theta / gamma_i) * dist), expoente))

    def getGamma(self, K: float) -> float:
        return K * (self.SSDe / self.Me if self.Me != 0 else 1e-12)

    # ---------- Raio ----------
    def getRadiusWithWeight(self):
        if self.N == 0:
            return 0.0
        return np.sqrt((self.SSDe / self.N)) * 2.0
        #return math.sqrt((self.SSDe / self.N)) * 2

    def getRadiusND(self):
        if self.N == 0:
            return 0.0
        #return np.sqrt(self.SSDe / self.N)
        return math.sqrt((self.SSDe / self.N))

    def getRadiusUnsupervised(self):
        if self.N == 0:
            return 0.0
        #return np.sqrt(self.SSDe / self.N)
        return math.sqrt((self.SSDe / self.N))

    # ---------- Utilidades ----------
    def toDoubleArray(self):
        return self.centroide

    def __repr__(self):
        return f"SPFMiC(rotulo={self.rotulo}, N={self.N}, centroide={self.centroide.tolist()})"