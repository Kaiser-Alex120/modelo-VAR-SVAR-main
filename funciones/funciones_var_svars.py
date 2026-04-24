"""
funciones_var_svar.py
=====================
Funciones reutilizables para estimación VAR/SVAR, diagnósticos,
causalidad de Granger, IRF, FEVD y pronósticos.
Uso en el notebook:
    from funciones.var_svar import *
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats

from statsmodels.tsa.api import VAR
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.stats.stattools import omni_normtest
from statsmodels.stats.diagnostic import normal_ad
from statsmodels.tsa.stattools import grangercausalitytests


# ESTIMACIÓN VAR

def estimar_var(data: pd.DataFrame, vars_: list, p: int = None,
                ic: str = "bic", max_lags: int = 8) -> tuple:
    """
    Estima un modelo VAR(p).

    Parámetros
    ----------
    data     : DataFrame con las series (ya transformadas / estacionarias)
    vars_    : lista de columnas a incluir
    p        : número de rezagos. Si es None, se selecciona automáticamente con `ic`
    ic       : criterio de información para selección automática ('aic','bic','hqic')
    max_lags : máximo de rezagos a evaluar en la selección automática

    Retorna
    -------
    (model, results) — VARProcess y VARResultsWrapper de statsmodels
    """
    df_var = data[vars_].dropna()
    model  = VAR(df_var)

    if p is None:
        results = model.fit(maxlags=max_lags, ic=ic)
        print(f" Lag óptimo seleccionado por {ic.upper()}: p = {results.k_ar}")
    else:
        results = model.fit(p)
        print(f" VAR({p}) estimado.")

    return model, results


def resumen_var(results, vars_: list):
    """
    Imprime el resumen general del VAR: muestra, log-verosimilitud y criterios
    de información en formato limpio.
    """
    nobs = results.nobs
    p    = results.k_ar
    llf  = results.llf

    print("=" * 50)
    print(f"  Modelo VAR({p}) — Resultados Generales")
    print("=" * 50)
    print(f"  Variables  : {vars_}")
    print(f"  Rezagos    : {p}")
    print(f"  Obs efectivas : {nobs}")
    print(f"  Log-likelihood: {llf:.4f}")
    print(f"  AIC  = {results.aic:.5f}")
    print(f"  HQIC = {results.hqic:.5f}")
    print(f"  SBIC = {results.bic:.5f}")
    print("=" * 50)


def coeficientes_var(results):
    """
    Muestra los coeficientes por ecuación con errores estándar,
    estadístico t y p-valor.
    """
    params   = results.params
    stderr   = results.bse
    tvalues  = params / stderr

    print("\n Coeficientes por ecuación")
    print("─" * 65)

    for col in params.columns:
        pvals = 2 * (1 - stats.norm.cdf(np.abs(tvalues[col])))
        df_eq = pd.DataFrame({
            "Variable": params.index,
            "Coef.":    params[col].round(5),
            "Std.Err.": stderr[col].round(5),
            "t":        tvalues[col].round(3),
            "P>|t|":    pvals.round(4),
        })
        sig = df_eq["P>|t|"].apply(
            lambda p: "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
        )
        df_eq["Sig."] = sig
        print(f"\n  Ecuación dependiente: {col}")
        print(df_eq.to_string(index=False))

    print("\n  Significancia: *** p<0.01  ** p<0.05  * p<0.10")


def r2_rmse_var(results, data: pd.DataFrame, vars_: list) -> pd.DataFrame:
    """
    Calcula R² y RMSE por ecuación.

    Retorna DataFrame con columnas: Variable, Parms, RMSE, R²
    """
    resid = results.resid
    p     = results.k_ar
    filas = []

    for var in vars_:
        y_true = data[var].iloc[p:]
        ss_res = np.sum(resid[var] ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2     = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        rmse   = np.sqrt(np.mean(resid[var] ** 2))
        filas.append({
            "Variable": var,
            "Parms":    len(results.params),
            "RMSE":     round(rmse, 4),
            "R²":       round(r2,   4),
        })

    tabla = pd.DataFrame(filas)
    print("\n Ajuste por ecuación")
    print(tabla.to_string(index=False))
    return tabla


#  DIAGNÓSTICOS

def test_normalidad(results, alpha: float = 0.05) -> pd.DataFrame:
    """
    Test Jarque-Bera por ecuación + Anderson-Darling global.

    Retorna DataFrame con resultados por variable.
    """
    resid    = results.resid
    nombres  = list(resid.columns)
    filas    = []

    print("\n🔍 Normalidad — Jarque–Bera por ecuación")
    print(f"  {'Variable':>10} {'JB stat':>10} {'p-value':>10} {'Skew':>8} {'Kurt':>8}  {'Normal?':>8}")
    print("  " + "─" * 60)

    for col in nombres:
        serie = resid[col].dropna()
        if len(serie) < 3:
            continue
        jb, p, skew, kurt = jarque_bera(serie)
        ok = " normal" if p > alpha else " no normal"
        print(f"  {col:>10} {jb:>10.3f} {p:>10.5f} {skew:>8.3f} {kurt:>8.3f}  {ok:>8}")
        filas.append({"Variable": col, "JB": jb, "p-JB": p,
                      "Skew": skew, "Kurt": kurt, "Normal": p > alpha})

    # Anderson-Darling global
    flat = resid.values.flatten()
    flat = flat[~np.isnan(flat)]
    if len(flat) > 10:
        ad_stat, ad_p = normal_ad(flat)
        print(f"\n  Anderson–Darling global: stat={ad_stat:.4f}, p={ad_p:.5f}  "
              f"{'✅' if ad_p > alpha else '❌'}")

    return pd.DataFrame(filas)


def test_autocorrelacion(results, lags: int = 10, alpha: float = 0.05) -> pd.DataFrame:
    """
    Test Ljung-Box por ecuación.

    Retorna DataFrame con resultados.
    """
    resid = results.resid
    filas = []

    print(f"\n🔍 Autocorrelación — Ljung–Box (lag={lags}) por ecuación")
    print(f"  {'Variable':>10} {'LB stat':>10} {'p-value':>10}  {'Sin autocorr?':>14}")
    print("  " + "─" * 50)

    for col in resid.columns:
        serie = resid[col].dropna()
        if len(serie) < lags + 5:
            print(f"  {col:>10}   Obs insuficientes")
            continue
        lb  = acorr_ljungbox(serie, lags=[lags], return_df=True)
        lb_stat = lb["lb_stat"].iloc[0]
        lb_p    = lb["lb_pvalue"].iloc[0]
        ok  = "✅" if lb_p > alpha else "❌"
        print(f"  {col:>10} {lb_stat:>10.3f} {lb_p:>10.5f}  {ok:>14}")
        filas.append({"Variable": col, "LB stat": lb_stat,
                      "p-LB": lb_p, "Sin autocorr": lb_p > alpha})

    return pd.DataFrame(filas)


def test_arch(results, lags: int = 5, alpha: float = 0.05) -> pd.DataFrame:
    """
    Test de heterocedasticidad condicional ARCH por ecuación.

    Retorna DataFrame con resultados.
    """
    resid = results.resid
    filas = []

    print(f"\n🔍 Heterocedasticidad — ARCH (lag={lags}) por ecuación")
    print(f"  {'Variable':>10} {'F stat':>8} {'p(F)':>9} {'Chi²':>9} {'p(Chi²)':>11}  {'Homoc.?':>8}")
    print("  " + "─" * 64)

    for col in resid.columns:
        serie = resid[col].dropna()
        if len(serie) < 10:
            print(f"  {col:>10}   Obs insuficientes")
            continue
        f_stat, f_p, chi2, chi2_p = het_arch(serie)[:4]
        ok = "✅" if chi2_p > alpha else "❌"
        print(f"  {col:>10} {f_stat:>8.3f} {f_p:>9.5f} {chi2:>9.3f} {chi2_p:>11.5f}  {ok:>8}")
        filas.append({"Variable": col, "F": f_stat, "p-F": f_p,
                      "Chi2": chi2, "p-Chi2": chi2_p, "Homoc": chi2_p > alpha})

    return pd.DataFrame(filas)


def diagnosticos_completos(results, alpha: float = 0.05,
                            lb_lags: int = 10, arch_lags: int = 5):
    """
    Corre los tres diagnósticos (normalidad, autocorrelación, ARCH)
    y muestra un resumen final semáforo.
    """
    df_norm = test_normalidad(results, alpha)
    df_lb   = test_autocorrelacion(results, lags=lb_lags, alpha=alpha)
    df_arch = test_arch(results, lags=arch_lags, alpha=alpha)

    print("\n" + "=" * 50)
    print("  RESUMEN DE DIAGNÓSTICOS")
    print("=" * 50)
    print("  ✅ = cumple supuesto  |  ❌ = viola supuesto")
    print(f"  Normalidad (JB)   : {'✅' if df_norm['Normal'].all() else '❌  algunas variables no normales'}")
    print(f"  Sin autocorr (LB) : {'✅' if df_lb['Sin autocorr'].all() else '❌  autocorrelación detectada'}")
    print(f"  Homoc. (ARCH)     : {'✅' if df_arch['Homoc'].all() else '❌  heterocedasticidad detectada'}")
    print("=" * 50)

    return df_norm, df_lb, df_arch



# CAUSALIDAD DE GRANGER

def granger_tabla(data: pd.DataFrame, vars_: list,
                  max_lag: int = 2, alpha: float = 0.05) -> pd.DataFrame:
    """
    Corre el test de causalidad de Granger para todos los pares de variables.

    Parámetros
    ----------
    data    : DataFrame (series estacionarias)
    vars_   : lista de variables
    max_lag : rezago máximo (debe coincidir con el orden del VAR)
    alpha   : nivel de significancia

    Retorna DataFrame ordenado con resultados y columna 'Causalidad Granger'.
    """
    filas = []
    df_g  = data[vars_].dropna()

    for y in vars_:
        for x in vars_:
            if x == y:
                continue
            test = grangercausalitytests(df_g[[y, x]], maxlag=max_lag, verbose=False)
            chi2, p_val, df_chi2 = test[max_lag][0]["ssr_chi2test"]
            filas.append({
                "Dependiente": y,
                "Excluida":    x,
                "Chi²":        round(chi2,  4),
                "gl":          int(df_chi2),
                "p-value":     round(p_val, 5),
                "Granger →":   "✅ Sí" if p_val < alpha else "❌ No",
            })

    tabla = (
        pd.DataFrame(filas)
        .sort_values(["Dependiente", "Excluida"])
        .reset_index(drop=True)
    )
    print("\n📋 Test de Causalidad de Granger")
    print(f"   Lag = {max_lag}  |  α = {alpha}")
    print(tabla.to_string(index=False))
    return tabla


def plot_granger_heatmap(granger_df: pd.DataFrame,
                         alpha: float = 0.05, title: str = "Causalidad de Granger"):
    """
    Visualiza la causalidad de Granger como un heatmap de p-valores.
    Fila = variable dependiente (efecto), columna = excluida (causa).
    """
    vars_dep = granger_df["Dependiente"].unique()
    vars_exc = granger_df["Excluida"].unique()
    all_vars = sorted(set(vars_dep) | set(vars_exc))

    pivot = granger_df.pivot(index="Dependiente", columns="Excluida", values="p-value")
    pivot = pivot.reindex(index=all_vars, columns=all_vars)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values.astype(float), cmap="RdYlGn_r",
                   vmin=0, vmax=0.15, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Causa (excluida)", fontsize=10)
    ax.set_ylabel("Efecto (dependiente)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                txt = f"{val:.3f}"
                color = "white" if val < alpha / 2 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="p-valor")
    plt.tight_layout()
    plt.show()
# ── Agregar a funciones_var_svar.py ──────────────────────────────────────────

def identificacion_cholesky(results, vars_: list) -> np.ndarray:
    """
    Muestra la matriz de Cholesky (P) usada en la identificación estructural
    y genera automáticamente una plantilla de signos para plot_irf_signos().

    Parámetros
    ----------
    results : VARResultsWrapper
    vars_   : lista de variables en el orden del sistema

    Retorna
    -------
    P : np.ndarray  — matriz triangular inferior de Cholesky (n x n)
    """
    sigma = results.sigma_u          # matriz de covarianza de residuos
    P     = np.linalg.cholesky(sigma)
    n     = len(vars_)

    print("=" * 55)
    print("  IDENTIFICACIÓN DE CHOLESKY")
    print("=" * 55)
    print("  Orden del sistema (exógeno → endógeno):")
    for i, v in enumerate(vars_):
        print(f"    {i+1}. {v}")

    print("\n  Matriz P (Cholesky de Σ_u):")
    df_P = pd.DataFrame(P.round(6), index=vars_, columns=vars_)
    print(df_P.to_string())

    print("\n  Estructura de restricciones (triangular inferior):")
    restricc = pd.DataFrame(
        [["libre" if j <= i else "0 (restric.)" for j in range(n)] for i in range(n)],
        index=vars_, columns=vars_
    )
    print(restricc.to_string())

    # ── Plantilla de signos lista para copiar ────────────────────────────────
    print("\n" + "=" * 55)
    print("  PLANTILLA — copia y edita SIGNOS_IRF:")
    print("  Valores: 1 = positivo | -1 = invertir | 0 = ocultar")
    print("=" * 55)
    header = "          # " + "   ".join(f"{v:<6}" for v in vars_)
    print(f"\n{header}   ← choque")
    print("SIGNOS_IRF = {")
    for i, resp in enumerate(vars_):
        fila = ", ".join([" 1"] * n)
        coma = "," if i < n - 1 else ""
        print(f"    '{resp}': [{fila}]{coma}   # respuesta de {resp}")
    print("}")
    print("# ↑ respuesta")

    return P

def plot_irf_signos(results, vars_: list, signos: dict,
                    steps: int = 20, orth: bool = True,
                    ci: bool = True, alpha_ci: float = 0.16,
                    shock_labels: dict = None,
                    shock_desc: dict = None,
                    normalizar_1std: bool = True,
                    figsize: tuple = (14, 11)):
    """
    IRF personalizada con control de signos, etiquetas de choque y escala.

    Parámetros
    ----------
    results         : VARResultsWrapper
    vars_           : lista de variables en orden del sistema
    signos          : dict {resp: [s1,s2,...]}  1=normal | -1=invertir | 0=ocultar
    steps           : horizonte de impulso-respuesta
    orth            : True = Cholesky
    ci              : mostrar bandas de confianza
    alpha_ci        : nivel bandas (0.16 ≈ ±1σ | 0.05 ≈ 95%)
    shock_labels    : dict {var: "etiqueta corta del choque"}
                      ej. {'Tasa': 'Choque monetario'}
    shock_desc      : dict {var: "descripción larga"}
                      ej. {'Tasa': '+1 d.e. ≈ +0.25 pp en tasa BCRP'}
    normalizar_1std : True = choque de 1 desviación estándar (recomendado)
    figsize         : tamaño figura
    """
    import matplotlib.patches as mpatches

    irf  = results.irf(steps)
    irfs = irf.orth_irfs if orth else irf.irfs      # (steps+1, n, n)
    n    = len(vars_)
    h    = np.arange(steps + 1)

    # ── Desviaciones estándar de cada choque ─────────────────────────────────
    sigma   = results.sigma_u
    P_chol  = np.linalg.cholesky(sigma)
    std_shocks = np.diag(P_chol)                    # 1 d.e. por choque

    # ── Etiquetas por defecto ─────────────────────────────────────────────────
    if shock_labels is None:
        shock_labels = {v: v for v in vars_}
    if shock_desc is None:
        shock_desc = {
            v: f"1 d.e. = {std_shocks[j]:.4f}" 
            for j, v in enumerate(vars_)
        }

    # ── Bandas de confianza (bootstrap) ──────────────────────────────────────
    tiene_ci = False
    lower = upper = None
    if ci:
        try:
            bands       = irf.errband_mc(orth=orth, svar=False, repl=500, signif=alpha_ci)
            lower, upper = bands
            tiene_ci    = True
        except Exception:
            tiene_ci = False

    colors = ["#1A73E8", "#D62728", "#2CA02C", "#FF7F0E"]
    tipo   = "Cholesky" if orth else "Reducida"
    escala = "1 d.e." if normalizar_1std else "unitario"

    fig, axes = plt.subplots(n, n, figsize=figsize, sharex=True)
    fig.suptitle(
        f"Funciones Impulso-Respuesta — Identificación {tipo}  |  Choque: {escala}",
        fontsize=13, fontweight="bold", y=1.01
    )

    for j, shock in enumerate(vars_):          # columnas = choques
        # ── Subtítulo de columna: descripción del choque ──────────────────
        axes[0][j].set_title(
            f"Choque: {shock_labels.get(shock, shock)}\n"
            f"({shock_desc.get(shock, '')})",
            fontsize=8.5, fontweight="bold",
            color=colors[j % len(colors)],
            pad=6,
        )

        for i, resp in enumerate(vars_):       # filas = respuestas
            ax = axes[i][j]
            s  = signos.get(resp, [1]*n)[j]

            # ── Etiqueta del eje Y (solo primera columna) ─────────────────
            if j == 0:
                ax.set_ylabel(resp, fontsize=9, fontweight="bold", labelpad=4)

            # ── Eje X (solo última fila) ──────────────────────────────────
            if i == n - 1:
                ax.set_xlabel("Horizonte (meses)", fontsize=8)

            # ── Celda oculta ──────────────────────────────────────────────
            if s == 0:
                ax.set_facecolor("#EEEEEE")
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                        ha="center", va="center", fontsize=11, color="#BBBBBB")
                ax.set_xticks([]); ax.set_yticks([])
                continue

            # ── IRF con escala 1 d.e. ─────────────────────────────────────
            factor = std_shocks[j] if normalizar_1std else 1.0
            y      = irfs[:, i, j] * s * (factor if not orth else 1.0)

            # Bandas
            if tiene_ci:
                y_lo = lower[:, i, j] * s * (factor if not orth else 1.0)
                y_hi = upper[:, i, j] * s * (factor if not orth else 1.0)
                if s == -1:
                    y_lo, y_hi = y_hi, y_lo
                ax.fill_between(h, y_lo, y_hi,
                                color=colors[j % len(colors)], alpha=0.15)
                ax.plot(h, y_lo, color=colors[j % len(colors)],
                        linewidth=0.6, linestyle=":", alpha=0.6)
                ax.plot(h, y_hi, color=colors[j % len(colors)],
                        linewidth=0.6, linestyle=":", alpha=0.6)

            ax.plot(h, y, color=colors[j % len(colors)], linewidth=1.8)
            ax.axhline(0, color="gray", linewidth=0.9, linestyle="--")

            # Marca si el signo está invertido
            if s == -1:
                ax.text(0.97, 0.97, "signo (−)", transform=ax.transAxes,
                        ha="right", va="top", fontsize=6.5,
                        color=colors[j % len(colors)], style="italic")

            ax.tick_params(labelsize=7)

    # ── Leyenda global inferior ───────────────────────────────────────────────
    parches = [
        mpatches.Patch(color=colors[j % len(colors)],
                       label=f"{shock_labels.get(v,v)}")
        for j, v in enumerate(vars_)
    ]
    if tiene_ci:
        parches.append(mpatches.Patch(color="gray", alpha=0.3,
                                      label=f"IC {int((1-alpha_ci)*100)}%"))

    fig.legend(handles=parches, loc="lower center", ncol=n + 1,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.show()
    return irf

# FEVD — DESCOMPOSICIÓN DE VARIANZA

def calcular_fevd(results, steps: int = 20):
    """
    Calcula la descomposición de varianza del error de pronóstico (FEVD).

    Retorna el objeto FEVD de statsmodels.
    """
    return results.fevd(steps)


def tabla_fevd(results, vars_: list, steps: int = 20,
               horizontes: list = None) -> dict:
    """
    Muestra tablas FEVD por variable dependiente.

    Parámetros
    ----------
    horizontes : lista de horizontes a mostrar (ej. [1, 4, 8, 12]).
                 Si es None, muestra todos.

    Retorna dict {var: DataFrame} con la descomposición por horizonte.
    """
    fevd    = calcular_fevd(results, steps)
    tablas  = {}

    print("\n DESCOMPOSICIÓN DE VARIANZA (FEVD)")
    print(f"   Horizonte máximo = {steps}  |  Identificación: Cholesky")

    for i, var in enumerate(vars_):
        df_fevd = pd.DataFrame(
            fevd.decomp[i, :, :],
            columns=vars_,
            index=[f"h={h}" for h in range(fevd.decomp.shape[1])],
        ).round(4)

        if horizontes:
            idx_sel = [f"h={h}" for h in horizontes if f"h={h}" in df_fevd.index]
            df_fevd = df_fevd.loc[idx_sel]

        print(f"\n  Variable dependiente: {var}")
        print(df_fevd.to_string())
        tablas[var] = df_fevd

    return tablas


def plot_fevd(results, vars_: list, steps: int = 20,
              figsize: tuple = (12, 8)):
    """
    Gráfico de barras apiladas para la FEVD.
    """
    fevd    = calcular_fevd(results, steps)
    n       = len(vars_)
    h_axis  = np.arange(steps + 1)

    colors = ["#1A73E8", "#D62728", "#2CA02C", "#FF7F0E",
              "#9467BD", "#8C564B", "#E377C2", "#7F7F7F"]

    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
    fig.suptitle("Descomposición de Varianza del Error de Pronóstico (FEVD — Cholesky)",
                 fontsize=12, fontweight="bold")

    for i, var in enumerate(vars_):
        ax = axes[i] if n > 1 else axes
        bottom = np.zeros(steps + 1)
        for j, shock in enumerate(vars_):
            vals = fevd.decomp[i, :, j]
            ax.bar(h_axis, vals, bottom=bottom,
                   color=colors[j % len(colors)], label=shock, alpha=0.85)
            bottom += vals
        ax.set_title(var, fontsize=10, fontweight="bold")
        ax.set_xlabel("Horizonte", fontsize=9)
        ax.set_ylim(0, 1)
        if i == 0:
            ax.set_ylabel("Proporción de varianza", fontsize=9)
        if i == n - 1:
            ax.legend(title="Choque", fontsize=7, loc="upper right")

    plt.tight_layout()
    plt.show()

# PRONÓSTICOS

def pronosticar(results, data: pd.DataFrame, vars_: list,
                steps: int = 12, alpha: float = 0.05) -> pd.DataFrame:
    """
    Genera pronósticos con intervalos de confianza.

    Retorna DataFrame con columnas: {var}_F, {var}_Lo, {var}_Hi
    """
    forecast, lower, upper = results.forecast_interval(
        data[vars_].values[-results.k_ar:],
        steps=steps,
        alpha=alpha,
    )

    cols_f = {f"{v}_F":  forecast[:, i] for i, v in enumerate(vars_)}
    cols_l = {f"{v}_Lo": lower[:, i]    for i, v in enumerate(vars_)}
    cols_h = {f"{v}_Hi": upper[:, i]    for i, v in enumerate(vars_)}

    df_f = pd.DataFrame({**cols_f, **cols_l, **cols_h}).round(4)
    df_f.index = [f"h+{h+1}" for h in range(steps)]

    print(f"\n📈 Pronósticos ({steps} pasos adelante, IC {int((1-alpha)*100)}%)")
    cols_show = [f"{v}_F" for v in vars_]
    print(df_f[cols_show].to_string())

    return df_f


def plot_pronosticos(results, data: pd.DataFrame, vars_: list,
                     steps: int = 12, alpha: float = 0.05,
                     n_hist: int = 40, figsize: tuple = (13, 7)):
    """
    Gráfico de pronósticos con historia reciente e intervalos de confianza.
    """
    df_f = pronosticar(results, data, vars_, steps=steps, alpha=alpha)
    n    = len(vars_)

    fig, axes = plt.subplots(
        (n + 1) // 2, 2, figsize=figsize, squeeze=False
    )
    fig.suptitle(f"Pronósticos VAR — {steps} pasos  (IC {int((1-alpha)*100)}%)",
                 fontsize=12, fontweight="bold")

    for idx, var in enumerate(vars_):
        ax     = axes[idx // 2][idx % 2]
        hist   = data[var].iloc[-n_hist:]
        x_hist = np.arange(len(hist))
        x_fore = np.arange(len(hist), len(hist) + steps)

        ax.plot(x_hist, hist.values, color="#1A73E8", linewidth=1.6, label="Histórico")
        ax.plot(x_fore, df_f[f"{var}_F"].values,
                color="#D62728", linewidth=1.6, linestyle="--", label="Pronóstico")
        ax.fill_between(x_fore,
                        df_f[f"{var}_Lo"].values,
                        df_f[f"{var}_Hi"].values,
                        color="#D62728", alpha=0.15, label="IC")

        ax.axvline(len(hist) - 1, color="gray", linewidth=0.9, linestyle=":")
        ax.set_title(var, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, frameon=False)
        ax.tick_params(labelsize=8)

    # Ocultar subplots vacíos
    for k in range(n, axes.shape[0] * 2):
        axes[k // 2][k % 2].set_visible(False)

    plt.tight_layout()
    plt.show()
    return df_f

# ESTABILIDAD DEL VAR

def estabilidad_var(results, vars_: list, figsize: tuple = (8, 8)) -> dict:
    """
    Analiza la estabilidad del VAR mediante las raíces del polinomio característico.

    Construye la matriz compañera, calcula sus eigenvalores y grafica
    el círculo unitario. Equivalente al comando 'varstable' de Stata.

    Parámetros
    ----------
    results : VARResultsWrapper
    vars_   : lista de nombres de variables (para el reporte)
    figsize : tamaño del gráfico

    Retorna
    -------
    dict con claves:
        'estable'     : bool
        'eigenvalores': np.ndarray de valores complejos
        'modulos'     : np.ndarray de módulos
        'tabla'       : pd.DataFrame con el reporte tabular
    """
    n          = results.neqs
    p          = results.k_ar
    coefs_var  = results.coefs          # (p, n, n)

    # ── Matriz compañera ──────────────────────────────────────────────────────
    coef_matrix = np.zeros((n, n * p))
    for i in range(p):
        coef_matrix[:, i * n:(i + 1) * n] = coefs_var[i]

    companion = np.zeros((n * p, n * p))
    companion[:n, :] = coef_matrix
    for i in range(1, p):
        companion[i * n:(i + 1) * n, (i - 1) * n:i * n] = np.eye(n)

    # ── Eigenvalores ──────────────────────────────────────────────────────────
    eig_vals   = np.linalg.eigvals(companion)
    modulos    = np.abs(eig_vals)
    stable_all = bool(np.all(modulos < 1))

    # ── Gráfico círculo unitario ──────────────────────────────────────────────
    theta = np.linspace(0, 2 * np.pi, 400)

    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]),
                             gridspec_kw={"width_ratios": [1, 1.2]})
    fig.suptitle(
        f"Estabilidad del VAR({p})  —  "
        f"{'✔ ESTABLE' if stable_all else '❌ INESTABLE'}",
        fontsize=13, fontweight="bold",
        color="#2c7a2c" if stable_all else "#c0392b",
    )

    # Panel izquierdo: círculo unitario
    ax = axes[0]
    ax.plot(np.cos(theta), np.sin(theta),
            linestyle="--", linewidth=1.5, color="black", label="Círculo unitario")
    ax.axhline(0, linewidth=0.7, color="gray")
    ax.axvline(0, linewidth=0.7, color="gray")

    for i, ev in enumerate(eig_vals):
        dentro = np.abs(ev) < 1
        color  = "#1A73E8" if dentro else "#D62728"
        ax.scatter(ev.real, ev.imag, color=color, s=70, zorder=5)
        ax.text(ev.real + 0.02, ev.imag + 0.02, str(i + 1), fontsize=8)

    # Leyenda manual
    from matplotlib.lines import Line2D
    leyenda = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1A73E8",
               markersize=9, label="Dentro CU (estable)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#D62728",
               markersize=9, label="Fuera CU (inestable)"),
    ]
    ax.legend(handles=leyenda, fontsize=8, frameon=True)
    ax.set_xlabel("Parte real",      fontsize=10)
    ax.set_ylabel("Parte imaginaria", fontsize=10)
    ax.set_title("Raíces del polinomio característico", fontsize=10, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    # Panel derecho: tabla de módulos como barras horizontales
    ax2    = axes[1]
    idx    = np.argsort(modulos)[::-1]
    etiq   = [f"λ{i+1}" for i in range(len(eig_vals))]
    colores = ["#D62728" if m >= 1 else "#1A73E8" for m in modulos[idx]]

    ax2.barh([etiq[i] for i in range(len(idx))],
             modulos[idx], color=colores, alpha=0.8, height=0.6)
    ax2.axvline(1.0, color="black", linewidth=1.5,
                linestyle="--", label="Límite (|λ|=1)")
    ax2.set_xlim(0, max(modulos.max() * 1.1, 1.15))
    ax2.set_xlabel("|λ|  (módulo del eigenvalor)", fontsize=10)
    ax2.set_title("Módulos ordenados", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ── Reporte tabular ───────────────────────────────────────────────────────
    filas = []
    print("\n" + "=" * 55)
    print("  ANÁLISIS DE ESTABILIDAD DEL VAR")
    print("=" * 55)
    print(f"  Modelo  : VAR({p})  |  Variables: {vars_}")
    print(f"  Eigenvalores evaluados: {len(eig_vals)}  (n×p = {n}×{p})")
    print(f"  {'λ':>3}  {'Parte Real':>12}  {'Parte Imag.':>12}  {'|λ|':>8}  {'Estado':>10}")
    print("  " + "─" * 52)

    for i, ev in enumerate(eig_vals):
        mod    = np.abs(ev)
        estado = "Estable" if mod < 1 else "❌ Inestable"
        print(f"  {i+1:>3}  {ev.real:12.4f}  {ev.imag:12.4f}  {mod:8.4f}  {estado:>10}")
        filas.append({"λ": i + 1, "Re": round(ev.real, 4),
                      "Im": round(ev.imag, 4), "|λ|": round(mod, 4),
                      "Estable": mod < 1})

    tabla = pd.DataFrame(filas)
    print("  " + "─" * 52)
    print(f"\n  Módulo máximo : {modulos.max():.6f}  (debe ser < 1)")
    print(f"\n  Conclusión    : El VAR({p}) es "
          f"{'ESTABLE ✔' if stable_all else 'INESTABLE ❌'} — "
          f"{'todas' if stable_all else 'no todas'} las raíces se encuentran "
          f"dentro del círculo unitario.")
    print("=" * 55)

    return {"estable": stable_all, "eigenvalores": eig_vals,
            "modulos": modulos, "tabla": tabla}
