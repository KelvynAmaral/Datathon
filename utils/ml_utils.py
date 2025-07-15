import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

def calcular_status(score: float) -> tuple[str, str]:
    """Calcula o status e a cor correspondente com base no score."""
    if score >= 0.6:
        return "✅ Recomendado", "green"
    elif score >= 0.4:
        return "🟨 Potencial", "orange"
    else:
        return "❌ Baixa Aderência", "red"

def calcular_score_combinado(probabilidade: float, match_percent: float, similaridade: float, aderencia_academica: float) -> float:
    """Calcula o score combinado com pesos pré-definidos."""
    return (probabilidade * 0.3) + (match_percent * 0.4) + (similaridade * 0.2) + (aderencia_academica * 0.1)