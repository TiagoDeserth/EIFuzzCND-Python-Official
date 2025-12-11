# verify_mapping.py - Script para verificar mapeamento de classes

import pandas as pd
import os

def verificar_mapeamento():
    current = os.getcwd()
    dataset = "rbf"
    
    # Caminhos
    caminho_train = os.path.join(current, "datasets", dataset, f"{dataset}-train.csv")
    caminho_resultados = os.path.join(current, "datasets", dataset, "graphics_data", 
                                      f"{dataset}10000000-1.0-EIFuzzCND-Python-results.csv")
    
    print("="*70)
    print("VERIFICAÇÃO DE MAPEAMENTO: TREINO vs RESULTADOS ONLINE")
    print("="*70)
    
    # ========== ANÁLISE DO TREINO ==========
    print("\n📚 FASE 1: Analisando arquivo de TREINO")
    print("-"*70)
    
    df_train = pd.read_csv(caminho_train, encoding="latin1")
    print(f"Total de linhas no treino: {len(df_train)}")
    print(f"Total de colunas: {len(df_train.columns)}")
    print(f"\nPrimeiras 3 linhas da ÚLTIMA coluna (classe original):") 
    print(df_train.iloc[:3, -1])
    
    # Classes únicas no treino (valores originais)
    classes_originais_train = df_train.iloc[:, -1].dropna().unique()
    print(f"\n🏷️  Classes ORIGINAIS únicas no treino: {len(classes_originais_train)}")
    print(f"Primeiras 10: {sorted(classes_originais_train)[:10]}")
    print(f"Todas: {sorted(classes_originais_train)}")
    
    # ========== ANÁLISE DOS RESULTADOS ==========
    print("\n\n🔄 FASE 2: Analisando arquivo de RESULTADOS (online)")
    print("-"*70)
    
    df_results = pd.read_csv(caminho_resultados, encoding="latin1")
    print(f"Total de linhas nos resultados: {len(df_results)}")
    print(f"Colunas: {list(df_results.columns)}")
    print(f"\nPrimeiras 5 linhas do arquivo:")
    print(df_results.head())
    
    # Coluna 1 = Rótulo Verdadeiro
    rotulos_verdadeiros = df_results.iloc[:, 1]
    print(f"\n🏷️  Valores na coluna 1 (Rótulo Verdadeiro):")
    print(f"Tipo de dado: {rotulos_verdadeiros.dtype}")
    print(f"Primeiras 10 linhas: {rotulos_verdadeiros.head(10).tolist()}")
    print(f"Classes únicas na coluna 1: {sorted(rotulos_verdadeiros.unique())}")
    
    # ========== BUSCA POR SPY E IMAP ==========
    print("\n\n🔍 FASE 3: Buscando 'spy' e 'imap'")
    print("-"*70)
    
    print("\nNo arquivo de TREINO:")
    print(f"  'spy' aparece? {'spy' in classes_originais_train}")
    print(f"  'imap' aparece? {'imap' in classes_originais_train}")
    
    print("\nNo arquivo de RESULTADOS:")
    linha_spy = 225211
    linha_imap = 34268
    
    if len(df_results) >= linha_spy:
        valor_spy = df_results.iloc[linha_spy - 1, 1]
        print(f"  Linha {linha_spy}, coluna 1: {valor_spy} (tipo: {type(valor_spy).__name__})")
    else:
        print(f"  ⚠️  Arquivo tem apenas {len(df_results)} linhas")
    
    if len(df_results) >= linha_imap:
        valor_imap = df_results.iloc[linha_imap - 1, 1]
        print(f"  Linha {linha_imap}, coluna 1: {valor_imap} (tipo: {type(valor_imap).__name__})")
    
    # ========== DETECTAR CLASSES NOVAS ==========
    print("\n\n🆕 FASE 4: Detectando classes novas")
    print("-"*70)
    
    classes_resultados = set(rotulos_verdadeiros.unique())
    classes_treino_indices = set(range(len(classes_originais_train)))
    
    print(f"Classes no treino (como índices): {sorted(classes_treino_indices)}")
    print(f"Classes nos resultados: {sorted(classes_resultados)}")
    
    classes_novas = classes_resultados - classes_treino_indices
    print(f"\n✨ Classes que SÓ aparecem nos resultados: {sorted(classes_novas)}")
    
    print(f"\n📍 Primeira aparição de cada classe nova:")
    for classe_nova in sorted(classes_novas):
        primeira_linha = df_results[df_results.iloc[:, 1] == classe_nova].index[0] + 1
        momento = (primeira_linha - 1) // 1000
        print(f"   Classe {classe_nova}: linha {primeira_linha}, momento {momento}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    verificar_mapeamento()
