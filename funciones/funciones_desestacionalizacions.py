import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose, STL


def _descomponer_serie(serie: pd.Series, metodo: str, periodo: int):
    s = serie.dropna()
    if metodo == "stl":
        dec = STL(s, period=periodo, robust=True).fit()
        nombre = "STL (Loess)"
    elif metodo == "x13":
        try:
            from statsmodels.tsa.x13 import x13_arima_analysis
            res = x13_arima_analysis(s, x12path=None, trading=False, log=None)
            class _X13Dec:
                trend    = res.trend
                seasonal = res.seasonal
                resid    = res.irregular
            dec = _X13Dec()
            nombre = "X-13ARIMA-SEATS (Census)"
        except Exception as e:
            raise RuntimeError(f"X-13 no disponible: {e}")
    elif metodo in ("additive", "multiplicative"):
        dec = seasonal_decompose(s, model=metodo, period=periodo, extrapolate_trend="freq")
        nombre = f"Clásica {metodo}"
    else:
        raise ValueError(f"Método '{metodo}' no reconocido.")
    return dec, nombre


def desestacionalizar_serie(serie: pd.Series, metodo: str, periodo: int) -> pd.Series:
    dec, _ = _descomponer_serie(serie, metodo, periodo)
    s = serie.dropna()
    sa = s / dec.seasonal if metodo == "multiplicative" else s - dec.seasonal
    sa.name = serie.name
    return sa


def desestacionalizar_df(df: pd.DataFrame, metodo: str, periodo: int) -> pd.DataFrame:
    cols = {}
    for col in df.columns:
        try:
            cols[col] = desestacionalizar_serie(df[col], metodo, periodo)
        except Exception as e:
            print(f"  ⚠️  {col}: error — {e}")
            cols[col] = df[col]
    return pd.DataFrame(cols)


_COLORES = {
    "original": "#1A73E8",
    "trend":    "#D62728",
    "seasonal": "#2CA02C",
    "resid":    "#FF7F0E",
    "adjusted": "#9467BD",
}


def _panel(ax, x, y, color, title, ref_zero=False):
    ax.plot(x, y, color=color, linewidth=1.2)
    ax.set_title(title, fontsize=8.5, fontweight="bold")
    if ref_zero:
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.tick_params(axis="x", rotation=25, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)


def plot_descomposicion(df: pd.DataFrame, periodo: int = 12, metodo: str = "stl", guardar: str = None):
    vars_ = df.columns.tolist()
    n = len(vars_)
    fig, axes = plt.subplots(n, 5, figsize=(18, 3.5 * n), squeeze=False)
    try:
        _, nombre_metodo = _descomponer_serie(df[vars_[0]].dropna(), metodo, periodo)
    except Exception:
        nombre_metodo = metodo
    fig.suptitle(f"Descomposición Estacional — {nombre_metodo}  |  periodo = {periodo}",
                 fontsize=13, fontweight="bold", y=1.01)
    for i, var in enumerate(vars_):
        serie = df[var].dropna()
        if len(serie) < 2 * periodo:
            axes[i][0].text(0.5, 0.5, f"{var}\nDatos insuficientes",
                            ha="center", va="center", transform=axes[i][0].transAxes)
            for j in range(1, 5):
                axes[i][j].set_visible(False)
            continue
        try:
            dec, _ = _descomponer_serie(serie, metodo, periodo)
            sa = serie / dec.seasonal if metodo == "multiplicative" else serie - dec.seasonal
            idx = serie.index
            _panel(axes[i][0], idx, serie.values, _COLORES["original"],  f"Original — {var}")
            _panel(axes[i][1], idx, dec.trend,    _COLORES["trend"],     "Tendencia")
            _panel(axes[i][2], idx, dec.seasonal, _COLORES["seasonal"],  "Componente Estacional", ref_zero=True)
            _panel(axes[i][3], idx, dec.resid,    _COLORES["resid"],     "Residuo", ref_zero=True)
            _panel(axes[i][4], idx, sa.values,    _COLORES["adjusted"],  "Ajustada (SA)")
        except Exception as e:
            axes[i][0].text(0.5, 0.5, f"Error: {str(e)[:50]}",
                            ha="center", va="center", transform=axes[i][0].transAxes)
            for j in range(1, 5):
                axes[i][j].set_visible(False)
    plt.tight_layout()
    if guardar:
        Path(guardar).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(guardar, bbox_inches="tight", dpi=150)
        print(f"💾 Gráfico guardado: {guardar}")
    plt.show()


def _info_metodo(metodo: str):
    info = {
        "additive":       ("Clásica ADITIVA",        "SA = original − estacional"),
        "multiplicative": ("Clásica MULTIPLICATIVA",  "SA = original / estacional"),
        "stl":            ("STL (Loess) ",           "SA = original − estacional"),
        "x13":            ("X-13ARIMA-SEATS",         "SA = componente ajustado"),
    }
    nombre, formula = info.get(metodo, (metodo, ""))
    print("─" * 60)
    print(f"  Método : {nombre}")
    print(f"  Fórmula: {formula}")
    print("─" * 60)


def analizar_estacionalidad(df: pd.DataFrame,
                             periodo: int = 12,
                             metodo: str = "stl",
                             ruta_output: str = "data/DATA_DESTACIONALIZADA.xlsx",
                             guardar_grafico: str = None,
                             alpha: float = 0.05) -> pd.DataFrame:
    _info_metodo(metodo)

    print("\nGenerando gráfico de descomposición...")
    plot_descomposicion(df, periodo=periodo, metodo=metodo, guardar=guardar_grafico)

    print("\nDesestacionalizando series...")
    df_sa = desestacionalizar_df(df, metodo=metodo, periodo=periodo)

    ruta = Path(ruta_output)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df_sa.to_excel(ruta)
    print(f"\nDATA_DESTACIONALIZADA guardada: {ruta}")
    print(f"   Shape : {df_sa.shape}")
    print(f"   Rango : {df_sa.index.min().date()} → {df_sa.index.max().date()}")

    return df_sa