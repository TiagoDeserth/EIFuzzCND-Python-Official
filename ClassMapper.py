# ClassMapper.py - Gerenciador centralizado de mapeamento de classes
import os
from scipy.io import arff
from typing import Dict

class ClassMapper:
    _instance = None
    _mapping = None
    _reverse_mapping = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ClassMapper, cls).__new__(cls)
        return cls._instance
    
    def initialize(self, caminho_dataset: str, dataset_name: str):
        """
        Inicializa o mapeamento baseado no arquivo de instâncias completo.
        Garante que todas as classes (incluindo novas) sejam mapeadas consistentemente.
        
        Args:
            caminho_dataset: Caminho para a pasta do dataset
            dataset_name: Nome do dataset (ex: 'kdd', 'rbf')
        """
        instances_path = os.path.join(caminho_dataset, f"{dataset_name}-instances.arff")
        
        data_np, meta = arff.loadarff(instances_path)
        class_col = list(meta.names())[-1]
        nominal_values = meta[class_col][1]
        
        # Cria mapeamento: classe_nominal -> índice_numérico
        self._mapping = {
            val.decode() if isinstance(val, bytes) else val: float(idx)
            for idx, val in enumerate(nominal_values)
        }
        
        # Cria mapeamento reverso: índice_numérico -> classe_nominal
        self._reverse_mapping = {v: k for k, v in self._mapping.items()}
        
        print("="*70)
        print("MAPEAMENTO DE CLASSES INICIALIZADO")
        print("="*70)
        print(f"Total de classes: {len(self._mapping)}")
        print(f"Mapeamento (primeiras 10):")
        for i, (classe, idx) in enumerate(list(self._mapping.items())[:10]):
            print(f"  {classe} -> {idx}")
        if len(self._mapping) > 10:
            print(f"  ... e mais {len(self._mapping) - 10} classes")
        print("="*70)
    
    def get_mapping(self) -> Dict[str, float]:
        """Retorna o dicionário de mapeamento classe -> índice"""
        if self._mapping is None:
            raise RuntimeError("ClassMapper não foi inicializado! Chame initialize() primeiro.")
        return self._mapping
    
    def get_reverse_mapping(self) -> Dict[float, str]:
        """Retorna o dicionário de mapeamento índice -> classe"""
        if self._reverse_mapping is None:
            raise RuntimeError("ClassMapper não foi inicializado! Chame initialize() primeiro.")
        return self._reverse_mapping
    
    def map_class(self, class_name: str) -> float:
        """Converte nome da classe para índice numérico"""
        if isinstance(class_name, bytes):
            class_name = class_name.decode()
        return self._mapping.get(class_name, -1.0)
    
    def reverse_map(self, index: float) -> str:
        """Converte índice numérico para nome da classe"""
        return self._reverse_mapping.get(index, "unknown")
    
    def get_num_classes(self) -> int:
        """Retorna o número total de classes"""
        return len(self._mapping) if self._mapping else 0
