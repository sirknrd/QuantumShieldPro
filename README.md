# 🛡️ Quantum Shield Pro | Financial Trading Terminal

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Quantum Shield Pro** es una terminal de análisis técnico construida con Streamlit para evaluar activos financieros con un enfoque de confluencia de señales, gestión de riesgo y visualización profesional.

---

## 🚀 Características

- **Motor de recomendación por confluencia:** combina tendencia, momentum, volatilidad y volumen para construir un score total.
- **Indicadores técnicos integrados:** EMA/SMA, RSI, MACD, ADX, ATR, Bollinger, StochRSI, Supertrend e Ichimoku.
- **Gestión de riesgo adaptativa:** métricas basadas en ATR y régimen de mercado.
- **Visualización avanzada:** velas, volumen, overlays y paneles técnicos con Plotly.
- **Radar multi-activo:** compara varios tickers en una sola vista.
- **Ranking de más activas S&P 500:** ordenado por dólar transado.

---

## 📁 Estructura del proyecto

```text
QuantumShieldPro/
├── .streamlit/config.toml   # Configuración visual de Streamlit
├── app_trading.py           # Aplicación principal (lógica + UI)
├── requirements.txt         # Dependencias Python
├── .gitignore               # Exclusiones de control de versiones
└── README.md                # Documentación del proyecto
```

---

## ⚙️ Requisitos

- Python 3.9 o superior
- pip

Dependencias definidas en `requirements.txt`.

---

## ▶️ Ejecución local

```bash
pip install -r requirements.txt
streamlit run app_trading.py
```

La app quedará disponible en la URL local que Streamlit muestra en consola.

---

## 🧠 Lógica funcional (resumen)

1. **Carga de mercado:** descarga OHLCV desde Yahoo Finance.
2. **Cálculo técnico:** genera indicadores con `pandas_ta`.
3. **Modelado de señal:** calcula señales parciales por grupo:
   - Tendencia
   - Momentum
   - Volatilidad
   - Volumen
4. **Score final:** aplica pesos dinámicos según régimen de mercado (ADX).
5. **Recomendación final:** clasifica en:
   - COMPRA FUERTE
   - COMPRA
   - NEUTRAL
   - VENTA
   - VENTA FUERTE

---

## ✅ Mejoras de robustez implementadas

- Validación de entradas para período e intervalo.
- Manejo defensivo de errores en descargas externas.
- Validación de columnas obligatorias antes de calcular indicadores.
- Corrección de contrato de tipos en la función de recomendación.
- Limpieza de lógica redundante para mayor mantenibilidad.

---

## ⚠️ Descargo de responsabilidad

Este proyecto es educativo y de apoyo analítico. **No constituye asesoría financiera.**  
Toda decisión de inversión debe evaluarse con gestión de riesgo y criterio propio.
