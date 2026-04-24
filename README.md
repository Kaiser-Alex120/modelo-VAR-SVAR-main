# VarPy — Modelos VAR/SVAR en Python

> Material para elaborar modelos VAR y SVAR con datos macroeconómicos del Perú.

## 📋 Descripción

Este repositorio contiene un pipeline completo para el análisis de series temporales multivariadas mediante modelos **VAR (Vector Autoregression)** y **SVAR (Structural VAR)**.

### Variables del modelo:
| Código | Descripción |
|--------|-------------|
| `Tasa` | Tasa de Referencia BCRP (%) |
| `Expec` | Expectativas Empresariales (3 meses) |
| `PBI` | Crecimiento PBI (% interanual) |
| `TInt` | Términos de Intercambio (var%) |

## 📁 Estructura del proyecto

```
modelo-VAR-SVAR/
---tutorial
  -tutorial_api
├── 02_preparacion_data.ipynb    # Limpieza y preparación de datos
├── 03_estacionariedad.ipynb     # Pruebas de estacionariedad (ADF, KPSS)
├── 04_var_svar.ipynb            # Estimación VAR/SVAR y diagnósticos
├── data/                        # Datos fuente
│   ├── DATA_DESTACIONALIZADA.xlsx
│   └── README.md
├── funciones/                   # Módulos auxiliares
│   ├── funciones_var_svars.py
│   ├── funciones_estacionariedadp.py
│   └── ...
├── resultados/                 # Gráficos y tablas generadas
├── tutorials/                  # Tutoriales adicionales
├── requirements.txt            # Dependencias Python
├── .gitignore                  # Archivos ignorados por git
└── README.md                   # Este archivo
```

## 🚀 Uso rápido

```bash
# 1. Crear entorno virtual
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar notebooks en orden:
#    01-tutorial_api
#    02_preparacion_data.ipynb
#    03_estacionariedad.ipynb
#    04_var_svar.ipynb
```

##  Notebooks

| Notebook | Contenido |
|----------|-----------|
| **02_preparacion_data.ipynb** | Carga, limpieza y fusión de datos |
| **03_estacionariedad.ipynb** | Pruebas ADF, KPSS, Zivot-Andrews |
| **04_var_svar.ipynb** | Estimación VAR, diagnósticos, IRF, FEVD |

##  Requisitos

- Python ≥ 3.10
- pandas, numpy, statsmodels, matplotlib, seaborn

---

**Autor:** Alex Simeon Bustillos | Centro de Análisis Económico (Perú, 2026)
