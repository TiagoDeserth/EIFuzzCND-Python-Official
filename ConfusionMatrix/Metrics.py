# Metrics.py

class Metrics:
    def __init__(self, accuracy: float, precision: float, recall: float, f1Score: float,
                 tempo: int, unkMem: float, unknownRate: float):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1Score = f1Score
        self.tempo = tempo
        self.unkMem = unkMem
        self.unknownRate = unknownRate

    def getAccuracy(self) -> float:
        return self.accuracy

    def getPrecision(self) -> float:
        return self.precision

    def getRecall(self) -> float:
        return self.recall

    def getF1Score(self) -> float:
        return self.f1Score

    def getTempo(self) -> int:
        return self.tempo

    def getUnkMem(self) -> float:
        return self.unkMem

    def getUnknownRate(self) -> float:
        return self.unknownRate

    def __repr__(self):
        return (f"Metrics(tempo={self.tempo}, acc={self.accuracy:.4f}, "
                f"prec={self.precision:.4f}, rec={self.recall:.4f}, "
                f"f1={self.f1Score:.4f}, unkMem={self.unkMem}, "
                f"unkRate={self.unknownRate:.4f})")



