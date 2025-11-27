import numpy as np
from typing import List, Union
from Structs.Example import Example
import math
from DebugLogger import DebugLogger

'''
def calculaDistanciaEuclidiana(ponto1: Union[Example, List[float], np.ndarray],
                               ponto2: Union[List[float], np.ndarray]) -> float:
    """
    Versão fiel ao Java:
    - Se ponto1 é Example: usa .getPonto()
    - Se ponto1 é array-like: usa diretamente
    - Sempre calcula distância Euclidiana por soma de quadrados + sqrt (sem np.linalg.norm)
    """
    if isinstance(ponto1, Example):
        a = np.array(ponto1.getPonto(), dtype=np.float64)
    else:
        a = np.array(ponto1, dtype=np.float64)

    b = np.array(ponto2, dtype=np.float64)

    somatorio = 0.0
    for i in range(len(a)):
        somatorio += (a[i] - b[i]) ** 2

    return float(np.sqrt(somatorio))
'''

'''
def calculaDistanciaEuclidiana(ponto1, ponto2):
    a = np.asarray(ponto1.getPonto() if isinstance(ponto1, Example) else ponto1, dtype=np.float64)
    b = np.asarray(ponto2, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Dimensões incompatíveis para distância euclidiana.")
    return float(np.linalg.norm(a - b))
'''

'''
def calculaDistanciaEuclidiana(ponto1, ponto2):
    a = np.asarray(ponto1.getPonto() if isinstance(ponto1, Example) else ponto1, dtype=np.float64)
    b = np.asarray(ponto2, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError("Dimensões incompatíveis para distância euclidiana.")

    somatorio = np.sum((a - b) ** 2)
    return float(math.sqrt(somatorio))
'''

def calculaDistanciaEuclidiana(ponto1, ponto2):
    if isinstance(ponto1, Example):
        ponto1 = ponto1.getPonto()
    if isinstance(ponto2, Example):
        ponto2 = ponto2.getPonto()

    a = np.asarray(ponto1, dtype=np.float64)
    b = np.asarray(ponto2, dtype=np.float64)

    # *Prints 11/11/2025
    #DebugLogger.log(f"[DEBUG Distância] a.shape={a.shape}, b.shape={b.shape}")
    #DebugLogger.log(f"[DEBUG Distância] type(a)={type(a)}, type(b)={type(b)}")

    if a.shape != b.shape:
        raise ValueError(f"Dimensões incompatíveis: {a.shape} vs {b.shape}")

    return float(math.sqrt(np.sum((a - b) ** 2)))

