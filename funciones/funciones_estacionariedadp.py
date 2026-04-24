import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from arch.unitroot import ADF, PhillipsPerron, KPSS, ZivotAndrews

_ALPHA = 0.05
_FREQ  = 12

OPCIONES_TRAFO = [
    ("Sin transformación — mantener en niveles",       "none"),
    ("Δy  —  Primera diferencia  (más común)",         "d1"),
    ("Δlog(y)×100  —  Dif. log × 100  (recomendada)", "d1_log"),
    ("Δ²y  —  Segunda diferencia",                     "d2"),
    ("YoY %  —  Variación año contra año",             "yoy"),
    ("Δlog anual × 100  —  Dif. log anual",            "yoy_log"),
]


def aplicar_trafo(serie: pd.Series, tipo: str, freq: int = _FREQ) -> pd.Series:
    if tipo == "none":
        return serie
    elif tipo == "d1":
        return serie.diff()
    elif tipo == "d1_log":
        return np.log(serie.clip(lower=1e-9)).diff() * 100
    elif tipo == "d2":
        return serie.diff().diff()
    elif tipo == "yoy":
        return (serie - serie.shift(freq)) / serie.shift(freq).abs() * 100
    elif tipo == "yoy_log":
        return (np.log(serie.clip(lower=1e-9)) - np.log(serie.shift(freq).clip(lower=1e-9))) * 100
    return serie


def correr_tests(serie: pd.Series, alpha: float = _ALPHA) -> dict:
    s = serie.dropna()

    def _safe(fn):
        try:
            return fn()
        except Exception:
            return None

    adf_r = _safe(lambda: ADF(s, trend="c"))
    adf = {
        "stat":  round(adf_r.stat,   4) if adf_r else np.nan,
        "pval":  round(adf_r.pvalue, 4) if adf_r else np.nan,
        "lags":  adf_r.lags               if adf_r else None,
        "I0":    (adf_r.pvalue < alpha)   if adf_r else None,
        "crit5": round(adf_r.critical_values["5%"], 3) if adf_r else np.nan,
    }

    pp_r = _safe(lambda: PhillipsPerron(s, trend="c"))
    pp = {
        "stat":  round(pp_r.stat,   4) if pp_r else np.nan,
        "pval":  round(pp_r.pvalue, 4) if pp_r else np.nan,
        "I0":    (pp_r.pvalue < alpha)  if pp_r else None,
        "crit5": round(pp_r.critical_values["5%"], 3) if pp_r else np.nan,
    }

    kp_r = _safe(lambda: KPSS(s, trend="c"))
    kpss = {
        "stat":  round(kp_r.stat,   4) if kp_r else np.nan,
        "pval":  round(kp_r.pvalue, 4) if kp_r else np.nan,
        "I0":    (kp_r.pvalue >= alpha) if kp_r else None,
        "crit5": round(kp_r.critical_values["5%"], 3) if kp_r else np.nan,
    }

    za_r = _safe(lambda: ZivotAndrews(s, trend="c"))
    za = {
        "stat":       round(za_r.stat,   4) if za_r else np.nan,
        "pval":       round(za_r.pvalue, 4) if za_r else np.nan,
        "I0":         (za_r.pvalue < alpha)  if za_r else None,
        "crit5":      round(za_r.critical_values["5%"], 3) if za_r else np.nan,
        "breakpoint": None,
    }

    votos_I0 = sum(1 for t in [adf, pp, kpss, za] if t["I0"] is True)
    return {"ADF": adf, "PP": pp, "KPSS": kpss, "ZA": za, "votos_I0": votos_I0}


def correr_todos(df: pd.DataFrame, alpha: float = _ALPHA) -> pd.DataFrame:
    filas = []
    for col in df.columns:
        r = correr_tests(df[col], alpha=alpha)
        filas.append({
            "Variable":   col,
            "ADF stat":   r["ADF"]["stat"],  "ADF p-val":  r["ADF"]["pval"],
            "PP stat":    r["PP"]["stat"],   "PP p-val":   r["PP"]["pval"],
            "KPSS stat":  r["KPSS"]["stat"], "KPSS p-val": r["KPSS"]["pval"],
            "ZA stat":    r["ZA"]["stat"],   "ZA p-val":   r["ZA"]["pval"],
            "Votos I(0)": r["votos_I0"],
        })
    tabla = pd.DataFrame(filas).set_index("Variable")
    print(tabla.to_string())
    return tabla


def plot_variable_con_tests(col: str, df: pd.DataFrame,
                             tabla: pd.DataFrame, alpha: float = _ALPHA):
    r     = tabla.loc[col]
    serie = df[col].dropna()
    votos = r["Votos I(0)"]
    ok    = votos >= 3
    color = "#1A73E8" if ok else "#D62728"

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    fig.suptitle(f"{col}  —  Votos I(0): {votos}/4", fontsize=13, fontweight="bold",
                 color="#2c7a2c" if ok else "#c0392b")

    ax = axes[0]
    ax.plot(df.index, df[col], color=color, linewidth=1.5)
    ax.axhline(serie.mean(), color="gray", linestyle="--", linewidth=0.9,
               label=f"Media = {serie.mean():.2f}")
    ax.set_title("Serie de tiempo", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(axis="x", rotation=25)

    ax2 = axes[1]
    ax2.hist(serie, bins=20, density=True, color=color, alpha=0.55, edgecolor="white")
    try:
        serie.plot.kde(ax=ax2, color="#333333", linewidth=1.8)
    except Exception:
        pass
    ax2.set_title("Distribución", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Densidad")

    ax3 = axes[2]
    tests   = ["ADF", "PP", "KPSS", "ZA"]
    pvals   = [r["ADF p-val"], r["PP p-val"], r["KPSS p-val"], r["ZA p-val"]]
    labels3 = [
        f"ADF  (p={r['ADF p-val']:.3f})",
        f"PP   (p={r['PP p-val']:.3f})",
        f"KPSS (p={r['KPSS p-val']:.3f})*",
        f"ZA   (p={r['ZA p-val']:.3f})",
    ]
    colors_bar = []
    for t, p in zip(tests, pvals):
        try:
            p = float(p)
            verde = (p >= alpha) if t == "KPSS" else (p < alpha)
            colors_bar.append("#2ca02c" if verde else "#d62728")
        except Exception:
            colors_bar.append("#aaaaaa")

    pvals_num = [float(p) if not pd.isna(p) else 0 for p in pvals]
    ax3.barh(labels3, pvals_num, color=colors_bar, alpha=0.8, height=0.5)
    ax3.axvline(alpha, color="black", linewidth=1.4, linestyle="--", label=f"α = {alpha}")
    ax3.set_xlim(0, 1.05)
    ax3.set_title("P-valores por test", fontsize=10, fontweight="bold")
    ax3.set_xlabel("p-valor")
    ax3.legend(fontsize=8, frameon=False)
    ax3.text(0.97, -0.15, "* KPSS: verde = p ≥ α",
             transform=ax3.transAxes, fontsize=7, ha="right", color="gray")

    plt.tight_layout()
    plt.show()


def construir_ui(df: pd.DataFrame, tabla_tests: pd.DataFrame,
                 alpha: float = _ALPHA, freq: int = _FREQ,
                 ruta_output: str = "data/DATA_VAR.xlsx"):
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    state = {"df_work": df.copy(), "trafos": {}}

    check_vars = {}
    for col in df.columns:
        votos = tabla_tests.loc[col, "Votos I(0)"]
        no_estac = votos < 3
        check_vars[col] = widgets.Checkbox(
            value=no_estac,
            description=f"{col}   [Votos I(0): {votos}/4]",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="400px"),
        )

    tipo_dd = widgets.Dropdown(
        options=OPCIONES_TRAFO, value="d1", description="Transformación:",
        style={"description_width": "initial"}, layout=widgets.Layout(width="540px"),
    )

    freq_w = widgets.BoundedIntText(
        value=freq, min=1, max=365, description="Periodos/año (para YoY):",
        style={"description_width": "initial"}, layout=widgets.Layout(width="290px"),
    )

    def _toggle_freq(change):
        freq_w.layout.visibility = "visible" if tipo_dd.value in ("yoy", "yoy_log") else "hidden"
    tipo_dd.observe(_toggle_freq, names="value")
    _toggle_freq(None)

    btn_prev  = widgets.Button(description="👁  Vista previa",          button_style="info",    layout=widgets.Layout(width="170px"))
    btn_apply = widgets.Button(description="✅  Aplicar",               button_style="primary", layout=widgets.Layout(width="170px"))
    btn_reset = widgets.Button(description="↩  Resetear todo",         button_style="warning", layout=widgets.Layout(width="170px"))
    btn_save  = widgets.Button(description="💾  Guardar DATA_VAR.xlsx", button_style="success", layout=widgets.Layout(width="240px"))

    status = widgets.HTML("")
    out    = widgets.Output()

    ui = widgets.VBox([
        widgets.HTML("<hr><h4>① Selecciona variables a transformar:</h4>"),
        widgets.VBox(list(check_vars.values())),
        widgets.HTML("<br><h4>② Tipo de transformación:</h4>"),
        tipo_dd, freq_w,
        widgets.HTML("<br>"),
        widgets.HBox([btn_prev, btn_apply, btn_reset]),
        status,
        widgets.HTML("<hr><h4>③ Guardar resultado:</h4>"),
        btn_save,
        widgets.HTML("<hr>"),
        out,
    ])

    def on_preview(b):
        vars_sel = [c for c, cb in check_vars.items() if cb.value]
        tipo, freq_v = tipo_dd.value, freq_w.value
        with out:
            clear_output()
            if not vars_sel:
                print("⚠️  Ninguna variable seleccionada.")
                return
            if tipo == "none":
                print("ℹ️  Sin transformación — niveles originales.")
                return
            fig, axes = plt.subplots(len(vars_sel), 2, figsize=(13, 3.5 * len(vars_sel)), squeeze=False)
            label_corto = dict(OPCIONES_TRAFO)[tipo].split("—")[0].strip()
            fig.suptitle(f"Vista previa — {label_corto}", fontsize=13, fontweight="bold")
            for i, col in enumerate(vars_sel):
                s_trans = aplicar_trafo(df[col], tipo, freq_v).dropna()
                axes[i][0].plot(df.index, df[col], color="#1A73E8", linewidth=1.4)
                axes[i][0].set_title(f"{col} — Original", fontsize=10, fontweight="bold")
                axes[i][1].plot(s_trans.index, s_trans, color="#D62728", linewidth=1.4)
                axes[i][1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
                axes[i][1].set_title(f"{col} — {label_corto}", fontsize=10, fontweight="bold")
            plt.tight_layout()
            plt.show()

    def on_apply(b):
        vars_sel = [c for c, cb in check_vars.items() if cb.value]
        tipo, freq_v = tipo_dd.value, freq_w.value
        n_obs = 0
        with out:
            clear_output()
            if not vars_sel and tipo != "none":
                print("⚠️  Ninguna variable seleccionada.")
                return
            state["df_work"] = df.copy()
            state["trafos"]  = {}
            label = dict(OPCIONES_TRAFO)[tipo]
            for col in vars_sel:
                state["df_work"][col] = aplicar_trafo(df[col], tipo, freq_v)
                state["trafos"][col]  = label
            state["df_work"] = state["df_work"].dropna()
            n_obs = len(state["df_work"])
            if tipo == "none":
                print("ℹ️  Sin transformación — niveles originales.")
            else:
                print(f"✅ Transformación: {label}")
                print(f"   Transformadas : {vars_sel}")
                print(f"   En niveles    : {[c for c in df.columns if c not in vars_sel]}")
            print(f"   Observaciones : {n_obs}")
            if tipo != "none" and vars_sel:
                print("\n📊 Verificación ADF post-transformación:")
                for col in vars_sel:
                    s = state["df_work"][col].dropna()
                    if len(s) > 10:
                        try:
                            adf_r = ADF(s, trend="c")
                            ok = adf_r.pvalue < alpha
                            print(f"  {col:<30} p={adf_r.pvalue:.4f}   {'✅ Estacionaria' if ok else '❌ Aún no estacionaria'}")
                        except Exception:
                            print(f"  {col:<30} n/a")
        status.value = f"<span style='color:green;font-weight:bold'>✅ Aplicado — {n_obs} obs.</span>"

    def on_reset(b):
        state["df_work"] = df.copy()
        state["trafos"]  = {}
        for cb in check_vars.values():
            cb.value = False
        tipo_dd.value = "d1"
        status.value = "<span style='color:darkorange;font-weight:bold'>↩ Reseteado.</span>"
        with out:
            clear_output()
            print("↩ df_work reseteado a niveles originales.")

    def on_save(b):
        with out:
            clear_output()
            ruta = Path(ruta_output)
            ruta.parent.mkdir(parents=True, exist_ok=True)
            state["df_work"].to_excel(ruta)
            dw = state["df_work"]
            print(f"💾 DATA_VAR guardada: {ruta}")
            print(f"   Shape : {dw.shape}")
            print(f"   Rango : {dw.index.min().date()} → {dw.index.max().date()}")
            if state["trafos"]:
                for col, lbl in state["trafos"].items():
                    print(f"   · {col}: {lbl}")
            else:
                print("   ℹ️  Guardado en niveles originales.")
        status.value = f"<span style='color:green;font-weight:bold'>💾 Guardado en {ruta_output}</span>"

    btn_prev.on_click(on_preview)
    btn_apply.on_click(on_apply)
    btn_reset.on_click(on_reset)
    btn_save.on_click(on_save)

    display(ui)
    return state