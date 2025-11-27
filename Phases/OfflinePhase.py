from Models.SupervisedModel import SupervisedModel
import numpy as np
from DebugLogger import DebugLogger
import pandas as pd

class OfflinePhase:
    def __init__(self, dataset: str, caminho: str, fuzzification: float, alpha: float, theta: float, K: int, minWeight: int):
        self.dataset = dataset
        self.caminho = caminho
        self.fuzzification = fuzzification
        self.alpha = alpha
        self.theta = theta
        self.K = K
        self.minWeight = minWeight
        self.supervisedModel = None

    def inicializar(self, trainSet):
        """Recebe os dados já carregados (como no Java)."""
        if isinstance(trainSet, pd.DataFrame):
            trainSet = trainSet.to_numpy()

        if self.supervisedModel is None:
            self.supervisedModel = SupervisedModel(
                self.dataset,
                self.caminho,
                self.fuzzification,
                self.alpha,
                self.theta,
                self.K,
                self.minWeight)

        self.supervisedModel.trainInitialModel(trainSet)
        return self.supervisedModel
