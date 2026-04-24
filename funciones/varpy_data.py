"""
varpy.data
==========
Funciones para la extracción automática de datos macroeconómicos
desde el API del BCRP (Banco Central de Reserva del Perú) y el
Banco Mundial (World Bank).
Autor: alexsibu09@gmail.com
"""
import re
import requests
import pandas as pd
from io import StringIO
from typing import List, Optional
import html
# ─────────────────────────────────────────────
#  Parser universal de fechas BCRP (reconoce anual, trimestral, mensual y diario)
_MESES_ES = {
    "Ene": "Jan", "Feb": "Feb", "Mar": "Mar", "Abr": "Apr",
    "May": "May", "Jun": "Jun", "Jul": "Jul", "Ago": "Aug",
    "Set": "Sep", "Sep": "Sep", "Oct": "Oct", "Nov": "Nov", "Dic": "Dec"
}
def _parse_bcrp_date(s: str) -> pd.Timestamp:
    s = s.strip()

    # 1) Anual: solo 4 dígitos a tomar como año
    if re.fullmatch(r"\d{4}", s):
        return pd.Timestamp(f"{s}-01-01")
    # 2) Trimestral: se toma en cuenta 'IT.2010', 'IIT.2010', 'IIIT.2010', 'IVT.2010'
    m = re.fullmatch(r"(IV|III|II|I)T\.(\d{2,4})", s)
    if m:
        trimestre_map = {"I": 1, "II": 4, "III": 7, "IV": 10}
        mes  = trimestre_map[m.group(1)]
        anio = int(m.group(2))
        if anio < 100:
            anio += 2000 if anio < 50 else 1900
        return pd.Timestamp(f"{anio}-{mes:02d}-01")
    # Traducir meses españoles → inglés
    for es, en in _MESES_ES.items():
        s = s.replace(es, en)
    # 3) Diario:  va aparecer asi '01.Jan.10' (año 2 dígitos)
    if re.match(r"\d{2}\.\w{3}\.\d{2}$", s):
        try:
            return pd.to_datetime(s.replace(".", " "), format="%d %b %y")
        except ValueError:
            return pd.to_datetime(s.replace(".", " "), format="mixed", dayfirst=True)
    # 4) Mensual año 4 dígitos: 'Jan.2000'
    if re.match(r"\w{3}\.\d{4}$", s):
        try:
            return pd.to_datetime(s.replace(".", " "), format="%b %Y")
        except ValueError:
            return pd.to_datetime(s.replace(".", " "), format="mixed")
    # 5) Mensual año 2 dígitos: 'Jan.00'
    if re.match(r"\w{3}\.\d{2}$", s):
        try:
            return pd.to_datetime(s.replace(".", " "), format="%b %y")
        except ValueError:
            return pd.to_datetime(s.replace(".", " "), format="mixed")
    # 6) Intento genérico como último recurso
    return pd.to_datetime(s, dayfirst=True)
#  Función de descarga de datos del BCRP
def bcrp_fetch(
    series: List[str],
    fecha_inicio: str,
    fecha_fin: str,
    formato: str = "json",
    idioma: str = "ing",
    nombres: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Descarga series económicas del API del BCRP.

    Parámetros
    ----------
    series : list[str]
        Códigos de las series a descargar.
        Encuéntralos en: https://estadisticas.bcrp.gob.pe/estadisticas/series/ayuda/metadatos
    fecha_inicio : str
        - Diaria     : 'YYYY-MM-DD'  Ej: '2010-01-01'
        - Mensual    : 'YYYY-MM'     Ej: '2010-01'
        - Trimestral : 'YYYY-MM'     Ej: '2010-01'
        - Anual      : 'YYYY'        Ej: '2010'
    fecha_fin : str
        Mismo formato que fecha_inicio.
    nombres : list[str], optional
        Nombres personalizados para las columnas (sin incluir 'Fecha').
        Si se omite, se usan los códigos originales de las series.

    ADVERTENCIA DE UN SAN MARQUINO CARAJO
    -----
    No mezcles series de distinta frecuencia en una misma llamada.
    El API del BCRP no soporta combinaciones diario+mensual, etc.
    """
    url_base   = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api/"
    series_str = "-".join(series)
    url        = f"{url_base}{series_str}/csv/{fecha_inicio}/{fecha_fin}/{idioma}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Error al conectar con el API del BCRP: {e}")
    texto = response.text.replace("<br>", "\n")
    texto = html.unescape(texto)  # Decodificar entidades HTML
    df    = pd.read_csv(StringIO(texto))
    # Renombrar columnas
    col_names = ["Fecha"] + (nombres if nombres else series)
    if len(df.columns) != len(col_names):
        raise ValueError(
            f"Se esperaban {len(col_names)} columnas pero el API devolvió {len(df.columns)}.\n"
            f"Tip: no mezcles series de distinta frecuencia en una misma llamada."
        )
    df.columns = col_names

    # Parsear fechas con parser universal
    df["Fecha"] = df["Fecha"].astype(str).apply(_parse_bcrp_date)
    df.set_index("Fecha", inplace=True)

    # Convertir a numérico
    df = df.apply(pd.to_numeric, errors="coerce")

    print(f"✅ BCRP: {len(df)} observaciones | {df.index.min().date()} → {df.index.max().date()}")
    return df
def bcrp_metadata_url() -> str:
    """Retorna la URL del catálogo de metadatos del BCRP."""
    url = "https://estadisticas.bcrp.gob.pe/estadisticas/series/ayuda/metadatos"
    print(f"📖 Catálogo de series BCRP: {url}")
    return url




#  Funciones para extraer datos del WORLD BANK
def wb_fetch(
    paises: List[str],
    indicadores: List[str],
    año_inicio: int,
    año_fin: int,
    nombres: Optional[dict] = None,
    batch_size: int = 10000,
) -> pd.DataFrame:
    """
    Descarga indicadores del API del Banco Mundial (World Bank).

    Parámetros
    ----------
    paises : list[str]
        Códigos ISO3. Ej: ['PER', 'CHL', 'COL']
    indicadores : list[str]
        Códigos de indicadores. Ej: ['NY.GDP.MKTP.CD']
    año_inicio : int
    año_fin : int
    nombres : dict, optional
        Renombrar indicadores. Ej: {'NY.GDP.MKTP.CD': 'PIB'}
    """
    base_url  = "https://api.worldbank.org/v2/country"
    registros = []
    for indicador in indicadores:
        for pais in paises:
            params = {
                "date": f"{año_inicio}:{año_fin}",
                "format": "json",
                "per_page": batch_size,
            }
            url = f"{base_url}/{pais}/indicator/{indicador}"
            try:
                response = requests.get(url, params=params, timeout=15)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"    Error {pais} - {indicador}: {e}")
                continue

            json_data = response.json()
            if not isinstance(json_data, list) or len(json_data) < 2 or not json_data[1]:
                print(f"    Sin datos: {pais} - {indicador}")
                continue
            for obs in json_data[1]:
                registros.append({
                    "Pais":      obs["country"]["value"],
                    "ISO3":      pais.upper(),
                    "Año":       int(obs["date"]),
                    "Indicador": indicador,
                    "Valor":     obs["value"],
                })

    if not registros:
        raise ValueError("No se descargaron datos. Verifica los parámetros.")
    df_long = pd.DataFrame(registros)
    df_wide = df_long.pivot_table(
        index=["Pais", "ISO3", "Año"],
        columns="Indicador",
        values="Valor",
        aggfunc="first",
    ).reset_index()
    df_wide.columns.name = None
    if nombres:
        df_wide.rename(columns=nombres, inplace=True)
    df_wide.sort_values(["Pais", "Año"], inplace=True)
    df_wide.reset_index(drop=True, inplace=True)
    print(f"✅ World Bank: {len(df_wide)} filas | {df_wide['Pais'].nunique()} países | "
          f"{df_wide['Año'].min()} → {df_wide['Año'].max()}")
    return df_wide
def wb_metadata_url() -> str:
    """Retorna la URL del catálogo de indicadores del Banco Mundial."""
    url = "https://databank.worldbank.org/metadataglossary/all/series"
    print(f"📖 Catálogo World Bank: {url}")
    return url
LATAM = [
    "ARG", "BOL", "BRA", "CHL", "COL", "ECU",
    "PRY", "PER", "URY", "VEN", "MEX", "GTM",
    "BLZ", "SLV", "HND", "NIC", "CRI", "PAN",
    "CUB", "DOM", "HTI",
]
"""Lista de códigos ISO3 para países de América Latina y el Caribe."""