import numpy as np
from typing import List

class Example:
    def __init__(self, ponto, comRotulo: bool, time: int = 0):
        """
        Construtor equivalente ao Java:
        - Se comRotulo=True: última posição do vetor é o rótulo verdadeiro.
        - Caso contrário: ponto inteiro são atributos e rótulo = -1.
        """
        arr = np.array(ponto, dtype=float).ravel()

        if comRotulo:
            if arr.size < 2:
                raise ValueError("Exemplo rotulado precisa de ao menos 1 atributo + 1 rótulo.")
            self.rotuloVerdadeiro = float(arr[-1])
            self.ponto = arr[:-1]
            self.time = time
        else:
            self.rotuloVerdadeiro = -1.0
            self.ponto = arr
            self.time = time

        self.rotuloClassificado = -1.0
        self.desconhecido = False
        self.time = int(time)

    # -------- Getters e Setters --------
    def getRotuloVerdadeiro(self) -> float:
        return self.rotuloVerdadeiro

    def getRotuloClassificado(self) -> float:
        return self.rotuloClassificado

    def setRotuloClassificado(self, rotulo: float):
        self.rotuloClassificado = float(rotulo)

    def getPonto(self) -> np.ndarray:
        return self.ponto

    def getPontoPorPosicao(self, i: int) -> float:
        return float(self.ponto[i])

    def getTime(self) -> int:
        return self.time

    # -------- Utilidades --------
    def toDoubleArray(self) -> np.ndarray:
        return np.copy(self.ponto)

    def getPoint(self) -> np.ndarray:
        return np.copy(self.ponto)

    def arrayToString(self) -> str:
        exemplo = str(self.ponto[0]) if len(self.ponto) > 0 else ""
        for i in range(1, len(self.ponto)):
            exemplo += "\t" + str(self.ponto[i])
        exemplo += "\t" + str(self.rotuloVerdadeiro)
        return exemplo

    def __repr__(self):
        return f"Example(ponto={self.ponto.tolist()}, rotuloVerdadeiro={self.rotuloVerdadeiro}, rotuloClassificado={self.rotuloClassificado}, time={self.time})"
