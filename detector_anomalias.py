"""
Detector de Compras Anômalas
============================
Algoritmo que detecta transações fora do padrão usando:
  - Z-score         → identifica valores muito acima/abaixo da média histórica
  - IQR (Intervalo  → detecção robusta baseada em quartis, menos sensível a outliers
    Interquartil)
  - Regra de        → quantidade de itens muito acima da média histórica da categoria
    quantidade
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------

@dataclass
class Transacao:
    categoria: str
    valor_unitario: float
    quantidade: int = 1

    @property
    def total(self) -> float:
        return self.valor_unitario * self.quantidade


@dataclass
class ResultadoAnalise:
    transacao: Transacao
    eh_anomalia: bool
    z_score: Optional[float]
    motivos: list[str] = field(default_factory=list)
    media_historica: float = 0.0
    desvio_padrao: float = 0.0
    limiar_zscore: float = 0.0
    limiar_iqr: float = 0.0


# ---------------------------------------------------------------------------
# Perfis de categoria (média e desvio padrão esperados em R$)
# ---------------------------------------------------------------------------

PERFIS_CATEGORIA = {
    "passagem_aerea":  {"media": 900,  "std": 200,  "qty_media": 1.5, "icone": "✈️"},
    "mercado":         {"media": 250,  "std": 80,   "qty_media": 1,   "icone": "🛒"},
    "combustivel":     {"media": 180,  "std": 50,   "qty_media": 1,   "icone": "⛽"},
    "restaurante":     {"media": 120,  "std": 40,   "qty_media": 1,   "icone": "🍽️"},
    "eletronicos":     {"media": 2200, "std": 800,  "qty_media": 1,   "icone": "💻"},
    "hotel":           {"media": 400,  "std": 120,  "qty_media": 1,   "icone": "🏨"},
}


# ---------------------------------------------------------------------------
# Funções estatísticas auxiliares
# ---------------------------------------------------------------------------

def calcular_media(valores: list[float]) -> float:
    return sum(valores) / len(valores)


def calcular_desvio_padrao(valores: list[float]) -> float:
    if len(valores) < 2:
        return 0.0
    media = calcular_media(valores)
    variancia = sum((x - media) ** 2 for x in valores) / len(valores)
    return math.sqrt(variancia)


def calcular_iqr(valores: list[float]) -> tuple[float, float]:
    """Retorna (Q1, Q3) a partir dos dados ordenados."""
    ordenado = sorted(valores)
    n = len(ordenado)
    q1 = ordenado[n // 4]
    q3 = ordenado[(3 * n) // 4]
    return q1, q3


def calcular_zscore(valor: float, media: float, std: float) -> Optional[float]:
    if std == 0:
        return None
    return (valor - media) / std


# ---------------------------------------------------------------------------
# Detector principal
# ---------------------------------------------------------------------------

class DetectorAnomalias:
    """
    Mantém histórico de transações por categoria e analisa
    novas compras em busca de anomalias.

    Parâmetros
    ----------
    limiar_zscore : float
        Quantos desvios padrão acima da média disparam alerta (padrão 2.5).
    fator_iqr : float
        Multiplicador do IQR para limite superior (padrão 1.5 = regra de Tukey).
    fator_qty : float
        Quantas vezes acima da média de quantidade é suspeito (padrão 10×).
    min_historico : int
        Mínimo de transações históricas para usar estatísticas reais (senão usa perfil padrão).
    """

    def __init__(
        self,
        limiar_zscore: float = 2.5,
        fator_iqr: float = 1.5,
        fator_qty: float = 10.0,
        min_historico: int = 5,
    ):
        self.limiar_zscore = limiar_zscore
        self.fator_iqr = fator_iqr
        self.fator_qty = fator_qty
        self.min_historico = min_historico
        self.historico: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def adicionar_historico(self, transacao: Transacao) -> None:
        """Registra uma transação no histórico sem analisá-la."""
        cat = transacao.categoria
        self.historico.setdefault(cat, []).append(transacao.total)

    def analisar(self, transacao: Transacao) -> ResultadoAnalise:
        """
        Analisa uma nova transação e retorna o resultado com diagnóstico completo.
        A transação NÃO é adicionada ao histórico automaticamente — só faça isso
        após decidir que ela é legítima.
        """
        cat = transacao.categoria
        historico = self.historico.get(cat, [])
        perfil = PERFIS_CATEGORIA.get(cat, {"media": 500, "std": 200, "qty_media": 1})

        # Estatísticas: usa histórico real se suficiente, senão usa perfil padrão
        if len(historico) >= self.min_historico:
            media = calcular_media(historico)
            std = calcular_desvio_padrao(historico)
            q1, q3 = calcular_iqr(historico)
        else:
            media = perfil["media"]
            std = perfil["std"]
            q1 = media - std
            q3 = media + std

        iqr = q3 - q1
        limiar_zscore = media + self.limiar_zscore * std
        limiar_iqr = q3 + self.fator_iqr * iqr

        motivos = []
        eh_anomalia = False

        # --- Regra 1: Z-score ---
        z = calcular_zscore(transacao.total, media, std)
        if z is not None and abs(z) > self.limiar_zscore:
            eh_anomalia = True
            direcao = "acima" if z > 0 else "abaixo"
            motivos.append(
                f"Z-score {z:.2f} ({direcao} de {self.limiar_zscore}σ) — "
                f"total R$ {transacao.total:,.2f} vs média R$ {media:,.2f}"
            )

        # --- Regra 2: IQR (Tukey) ---
        if transacao.total > limiar_iqr:
            eh_anomalia = True
            motivos.append(
                f"Acima do limiar IQR de R$ {limiar_iqr:,.2f} "
                f"(Q3={q3:,.2f} + {self.fator_iqr}×IQR={iqr:,.2f})"
            )

        # --- Regra 3: Quantidade suspeita ---
        qty_media = perfil.get("qty_media", 1)
        if transacao.quantidade > max(qty_media * self.fator_qty, 5):
            eh_anomalia = True
            motivos.append(
                f"Quantidade {transacao.quantidade} unidades é "
                f"{transacao.quantidade / max(qty_media, 1):.0f}× a média histórica ({qty_media:.1f})"
            )

        return ResultadoAnalise(
            transacao=transacao,
            eh_anomalia=eh_anomalia,
            z_score=z,
            motivos=motivos,
            media_historica=media,
            desvio_padrao=std,
            limiar_zscore=limiar_zscore,
            limiar_iqr=limiar_iqr,
        )

    def registrar_se_normal(self, resultado: ResultadoAnalise) -> None:
        """Adiciona ao histórico somente transações normais."""
        if not resultado.eh_anomalia:
            self.adicionar_historico(resultado.transacao)


# ---------------------------------------------------------------------------
# Exibição formatada
# ---------------------------------------------------------------------------

def exibir_resultado(resultado: ResultadoAnalise) -> None:
    t = resultado.transacao
    perfil = PERFIS_CATEGORIA.get(t.categoria, {"icone": "🔹"})
    icone = perfil["icone"]
    linha = "─" * 60

    print(f"\n{linha}")
    print(f"  {icone}  {t.categoria.upper()}")
    print(f"  Valor unitário : R$ {t.valor_unitario:>10,.2f}")
    print(f"  Quantidade     : {t.quantidade:>10,}")
    print(f"  TOTAL          : R$ {t.total:>10,.2f}")
    print(f"  Média histórica: R$ {resultado.media_historica:>10,.2f}")
    print(f"  Limiar z-score : R$ {resultado.limiar_zscore:>10,.2f}")
    print(f"  Limiar IQR     : R$ {resultado.limiar_iqr:>10,.2f}")

    if resultado.z_score is not None:
        print(f"  Z-score        : {resultado.z_score:>10.2f}")

    if resultado.eh_anomalia:
        print(f"\n  🚨 ALERTA — COMPRA ANÔMALA DETECTADA")
        for i, motivo in enumerate(resultado.motivos, 1):
            print(f"     {i}. {motivo}")
    else:
        print(f"\n  ✅ Normal — dentro do padrão esperado")

    print(linha)


# ---------------------------------------------------------------------------
# Gerador de dados históricos sintéticos
# ---------------------------------------------------------------------------

def gerar_historico_sintetico(detector: DetectorAnomalias, n: int = 30) -> None:
    """Popula o detector com transações normais simuladas para cada categoria."""
    def normal(media, std):
        # Box-Muller (sem dependência externa)
        u1, u2 = random.random(), random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        return max(10.0, media + std * z)

    for cat, perfil in PERFIS_CATEGORIA.items():
        for _ in range(n):
            valor = normal(perfil["media"], perfil["std"])
            detector.adicionar_historico(Transacao(cat, round(valor, 2)))

    print(f"✔ {n} transações históricas por categoria carregadas.")


# ---------------------------------------------------------------------------
# Demonstração
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("   DETECTOR DE COMPRAS ANÔMALAS")
    print("   Z-score + IQR + Regra de Quantidade")
    print("=" * 60)

    detector = DetectorAnomalias(
        limiar_zscore=2.5,
        fator_iqr=1.5,
        fator_qty=10.0,
        min_historico=5,
    )

    # 1. Gera histórico base
    gerar_historico_sintetico(detector, n=30)

    # 2. Casos de teste
    casos = [
        # Normais
        Transacao("passagem_aerea",  850.00, 1),
        Transacao("mercado",         230.00, 1),
        Transacao("restaurante",     115.00, 1),
        Transacao("eletronicos",    2100.00, 1),

        # Anomalias
        Transacao("passagem_aerea",  900.00, 1000),  # 1000 passagens
        Transacao("mercado",        8500.00, 1),     # valor absurdo no mercado
        Transacao("restaurante",       5.00, 1),     # valor muito baixo
        Transacao("eletronicos",   45000.00, 1),     # eletrônico caríssimo
        Transacao("combustivel",      15.00, 500),   # 500 abastecimentos
        Transacao("hotel",          9800.00, 1),     # hotel suspeito
    ]

    print("\n📋 ANALISANDO TRANSAÇÕES:\n")

    normais, anomalias = 0, 0
    for transacao in casos:
        resultado = detector.analisar(transacao)
        exibir_resultado(resultado)
        if resultado.eh_anomalia:
            anomalias += 1
        else:
            normais += 1
            detector.registrar_se_normal(resultado)  # só adiciona ao histórico se normal

    print(f"\n{'=' * 60}")
    print(f"  RESUMO FINAL")
    print(f"  Total analisado : {len(casos)}")
    print(f"  Normais         : {normais}")
    print(f"  Anomalias       : {anomalias}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
