**Detector de Compras Anômalas**

Este projeto surgiu da ideia de identificar transações fora do padrão de forma inteligente — sem depender de regras fixas, mas sim do comportamento histórico de cada categoria de compra.

O detector combina três abordagens estatísticas: Z-score, IQR (Intervalo Interquartil) e uma regra de quantidade, aplicadas sobre um histórico de transações por categoria. Se uma compra se afasta muito do padrão esperado em valor ou quantidade, o sistema emite um alerta com os motivos detalhados.

**Como rodar**

Nenhuma instalação extra necessária — só Python 3.10+.

```bash
python detector_anomalias.py
```

**O que o projeto detecta**

- Valores muito acima ou abaixo da média histórica da categoria
- Compras com quantidade de itens desproporcional ao padrão
- Anomalias mesmo quando o histórico ainda é pequeno, usando perfis de referência por categoria

**Dupla:** Giovana Marsigli & Mariana Akemi
