"""
varpy.filtro_datos
==================
Funciones para lectura, visualización y análisis estadístico
de series de tiempo macroeconómicas desde Excel.

Autor: VarPy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import matplotlib.gridspec as gridspec
from statsmodels.tsa.seasonal import seasonal_decompose, STL
 

#  Estilo global
plt.rcParams.update({
    "figure.dpi":        120,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8F9FA",
    "axes.edgecolor":    "#CCCCCC",
    "axes.grid":         True,
    "grid.color":        "white",
    "grid.linewidth":    1.0,
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})
#  Eventos históricos para marcar en gráficos
# Puedes agregar más eventos relevantes a esta configuración 
EVENTOS = {
    "COVID": ("2020-03-01", "2021-12-01", "#FF000020", "COVID-19"),
    "GFC":   ("2008-09-01", "2009-12-01", "#FF880020", "Crisis Fin. 2008"),
}
# Leer data desde Excel y preparar DataFrame
# Leer data siempre con esta función para asegurar formato correcto y estadísticas descriptivas iniciales
def leer_data(
    ruta: str = "data/DATA.xlsx",
    hoja: str = 0,
    labels: dict = None,
) -> pd.DataFrame:
    """
    Lee el Excel DATA.xlsx y devuelve un DataFrame con índice datetime.

    Parámetros
    ----------
    ruta   : Ruta al archivo Excel.
    hoja   : Nombre o índice de la hoja a leer.
    labels : Diccionario {codigo: label_largo}. Si se pasa, renombra columnas.

    Retorna
    -------
    pd.DataFrame con índice DatetimeIndex.
    """
    df = pd.read_excel(ruta, sheet_name=hoja, index_col=0, parse_dates=True)
    df.index.name = "Fecha"
    df = df.apply(pd.to_numeric, errors="coerce")
    if labels:
        df = df.rename(columns=labels)
    print(f" Data cargada: {df.shape[0]} obs × {df.shape[1]} variables")
    print(f"   Rango: {df.index.min().date()} → {df.index.max().date()}")
    print(f"   Variables: {list(df.columns)}")
    return df
# ESTADÍSTICOS DESCRIPTIVOS

def estadisticos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabla de estadísticos descriptivos ampliada.
    Incluye: obs, media, std, min, p25, mediana, p75, max, skew, kurt, NaN.
    """
    stats = df.describe(percentiles=[0.25, 0.5, 0.75]).T
    stats["skew"] = df.skew()
    stats["kurt"] = df.kurt()
    stats["NaN"]  = df.isna().sum()
    stats = stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "skew", "kurt", "NaN"]]
    stats.columns = ["Obs", "Media", "Std", "Mín", "P25", "Mediana", "P75", "Máx", "Asim.", "Curt.", "NaN"]
    stats["Obs"] = stats["Obs"].astype(int)
    stats["NaN"] = stats["NaN"].astype(int)
    return stats.round(3)
#  Helper: sombrear eventos en un Axes
def _marcar_eventos(ax, eventos_activos=("COVID", "GFC")):
    handles = []
    for key in eventos_activos:
        ini, fin, color, label = EVENTOS[key]
        ax.axvspan(pd.Timestamp(ini), pd.Timestamp(fin), color=color, zorder=0)
        handles.append(mpatches.Patch(color=color.replace("20", "60"), label=label))
    return handles

#  serie de tiempo con eventos marcados
def plot_series(
    df: pd.DataFrame,
    eventos_activos: tuple = ("COVID", "GFC"),
    ncols: int = 2,
    figsize_per: tuple = (7, 3),
    guardar: str = None,
):
    """
    Grafica cada variable como serie de tiempo en subplots.
    Marca automáticamente los eventos históricos configurados.
    """
    vars_ = df.columns.tolist()
    n     = len(vars_)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
        squeeze=False
    )
    fig.suptitle("Series de Tiempo", fontsize=14, fontweight="bold", y=1.01)

    evento_handles = []
    for i, var in enumerate(vars_):
        ax = axes[i // ncols][i % ncols]
        ax.plot(df.index, df[var], color="#1A73E8", linewidth=1.4)
        ax.set_title(var, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        handles = _marcar_eventos(ax, eventos_activos)
        if handles and not evento_handles:
            evento_handles = handles

    # Ocultar subplots vacíos
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    if evento_handles:
        fig.legend(handles=evento_handles, loc="lower center",
                   ncol=len(evento_handles), bbox_to_anchor=(0.5, -0.03),
                   frameon=False, fontsize=9)

    plt.tight_layout()
    if guardar:
        Path(guardar).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(guardar, bbox_inches="tight")
        print(f"💾 Guardado: {guardar}")
    plt.show()


#  HISTOGRAMAS + KDE

def plot_histogramas(
    df: pd.DataFrame,
    ncols: int = 2,
    bins: int = 30,
    figsize_per: tuple = (6, 3.5),
    guardar: str = None,
):
    """
    Histograma con curva KDE para cada variable.
    """
    vars_ = df.columns.tolist()
    n     = len(vars_)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
        squeeze=False
    )
    fig.suptitle("Distribución de Variables", fontsize=14, fontweight="bold", y=1.01)

    for i, var in enumerate(vars_):
        ax   = axes[i // ncols][i % ncols]
        data = df[var].dropna()
        ax.hist(data, bins=bins, color="#1A73E8", alpha=0.6, density=True, edgecolor="white")
        data.plot.kde(ax=ax, color="#D62728", linewidth=2)
        ax.set_title(var, fontsize=10, fontweight="bold")
        ax.set_ylabel("Densidad")
        ax.axvline(data.mean(),   color="#2CA02C", linestyle="--", linewidth=1.2,
                   label=f"Media={data.mean():.2f}")
        ax.axvline(data.median(), color="#FF7F0E", linestyle=":",  linewidth=1.2,
                   label=f"Mediana={data.median():.2f}")
        ax.legend(fontsize=7, frameon=False)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    plt.tight_layout()
    if guardar:
        Path(guardar).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(guardar, bbox_inches="tight")
        print(f" Guardado: {guardar}")
    plt.show()
#  BOXPLOTS
def plot_boxplots(
    df: pd.DataFrame,
    figsize: tuple = None,
    guardar: str = None,
):
    """
    Boxplot de todas las variables en un solo gráfico (normalizadas).
    """
    df_norm = (df - df.mean()) / df.std()

    figsize = figsize or (max(8, len(df.columns) * 1.5), 5)
    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(
        [df_norm[c].dropna() for c in df_norm.columns],
        labels=df_norm.columns,
        patch_artist=True,
        medianprops=dict(color="#D62728", linewidth=2),
        whiskerprops=dict(color="#555555"),
        capprops=dict(color="#555555"),
        flierprops=dict(marker="o", color="#1A73E8", alpha=0.4, markersize=4),
    )
    colors = plt.cm.tab10(np.linspace(0, 1, len(df.columns)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor((*color[:3], 0.5))

    ax.set_title("Boxplots (estandarizados)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Valor estandarizado (z-score)")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    if guardar:
        Path(guardar).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(guardar, bbox_inches="tight")
        print(f" Guardado: {guardar}")
    plt.show()

# MATRIZ DE CORRELACIÓN
def plot_correlacion(
    df: pd.DataFrame,
    metodo: str = "pearson",
    figsize: tuple = None,
    guardar: str = None,
):
    """
    Heatmap de correlación con anotaciones y máscara triangular superior.
    """
    corr    = df.corr(method=metodo)
    n       = len(corr)
    figsize = figsize or (max(6, n * 1.2), max(5, n * 1.1))

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, vmin=-1, vmax=1,
        linewidths=0.5, linecolor="white",
        ax=ax, annot_kws={"size": 9},
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title(f"Matriz de Correlación ({metodo.capitalize()})",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    if guardar:
        Path(guardar).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(guardar, bbox_inches="tight")
        print(f" Guardado: {guardar}")
    plt.show()
    return corr
#  ACF / PACF
def plot_acf_pacf(
    df: pd.DataFrame,
    lags: int = 24,
    guardar: str = None,
):
    """
    Gráficos ACF y PACF para cada variable.
    """
    vars_ = df.columns.tolist()
    n     = len(vars_)

    fig, axes = plt.subplots(n, 2, figsize=(12, 3.5 * n), squeeze=False)
    fig.suptitle("ACF y PACF por Variable", fontsize=14, fontweight="bold", y=1.01)

    for i, var in enumerate(vars_):
        serie = df[var].dropna()
        plot_acf( serie, lags=lags, ax=axes[i][0], alpha=0.05,
                  color="#1A73E8", title=f"ACF — {var}")
        plot_pacf(serie, lags=lags, ax=axes[i][1], alpha=0.05,
                  color="#D62728", title=f"PACF — {var}", method="ywm")

    plt.tight_layout()
    if guardar:
        Path(guardar).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(guardar, bbox_inches="tight")
        print(f" Guardado: {guardar}")
    plt.show()
