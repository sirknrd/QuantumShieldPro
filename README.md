# 🛡️ Quantum Shield Pro | Financial Trading Terminal

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Quantum Shield Pro** es una terminal de análisis técnico avanzado diseñada para proporcionar señales de trading de alta precisión mediante la confluencia de múltiples indicadores y una gestión de riesgo basada en volatilidad real.

---

## 🚀 Características Principales

* **Motor de Confluencia:** Algoritmo que valida señales cruzando **EMA 50**, **RSI** y **MACD** para reducir falsos positivos.
* **Gestión de Riesgo (ATR):** Cálculo dinámico de *Stop Loss* y *Take Profit* basado en el **Average True Range**, adaptándose a la volatilidad del mercado.
* **Visualización Profesional:** Gráficos interactivos de velas japonesas con **Bandas de Bollinger** y sombreado de áreas de volatilidad mediante Plotly.
* **Multi-Activo:** Soporta Criptomonedas (Yahoo Finance), Acciones Internacionales y el mercado chileno (IPSA).

---

## 🛠️ Arquitectura del Proyecto

```text
QuantumShieldPro/
├── .streamlit/          # Configuración de interfaz y tema oscuro
├── app_trading.py       # Lógica principal y UI de la aplicación
├── requirements.txt     # Dependencias del sistema
├── .gitignore           # Filtro de archivos para el repositorio
└── README.md            # Documentación técnica
