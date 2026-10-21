from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from red_neuronal import CENTRO_NORMALIZACION, ESCALA_NORMALIZACION, RedNeuronal

CARPETA = Path(__file__).resolve().parent
CIUDADES = {
    "Guadalajara": {
        "nombre_datos": "Guadalajara, Mexico",
        "centro": (20.6767, -103.3475),
        "color": [185, 89, 55, 215],
    },
    "Ámsterdam": {
        "nombre_datos": "Ámsterdam, Países Bajos",
        "centro": (52.3676, 4.9041),
        "color": [38, 119, 128, 215],
    },
}
CLASES = {0: "Guadalajara", 1: "Ámsterdam"}
COLORES = {ciudad["nombre_datos"]: ciudad["color"] for ciudad in CIUDADES.values()}

st.set_page_config(page_title="Clasificación GPS", layout="wide")
st.markdown(
    """
    <style>
      .block-container {max-width: 1200px; padding-top: 1.5rem;}
      h1 {font-size: 2rem !important;}
      .leyenda {display: inline-flex; align-items: center; margin-right: 1.2rem; gap: .4rem;}
      .punto {display: inline-block; width: .6rem; height: .6rem; border-radius: 50%;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cargar_archivos() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    gps = pd.read_csv(CARPETA / "datos_gps.csv", encoding="utf-8")
    predicciones = pd.read_csv(CARPETA / "predicciones.csv", encoding="utf-8")
    historial = pd.read_csv(CARPETA / "historial_entrenamiento.csv", encoding="utf-8")
    with (CARPETA / "pesos_y_resultados.json").open(encoding="utf-8") as archivo:
        resultados = json.load(archivo)

    gps = gps.merge(
        predicciones[["id", "probabilidad_clase_1", "prediccion", "correcta"]],
        on="id",
        how="inner",
        validate="one_to_one",
    )
    if len(gps) != 300:
        raise ValueError("Los archivos deben contener 300 puntos con sus predicciones.")
    return gps, historial, resultados


def token_mapbox() -> str:
    token = os.environ.get("MAPBOX_TOKEN") or os.environ.get("MAPBOX_API_KEY") or ""
    try:
        return token or st.secrets.get("MAPBOX_TOKEN", "")
    except (FileNotFoundError, KeyError, AttributeError):
        return token


def preparar_mapa(datos: pd.DataFrame, ciudades: list[str], color_por: str, token: str) -> pdk.Deck:
    puntos = datos.copy()
    if color_por == "Predicción":
        puntos["color"] = puntos["prediccion"].map(
            lambda clase: CIUDADES[CLASES[clase]]["color"]
        )
    elif color_por == "Resultado":
        puntos["color"] = puntos["correcta"].map(
            lambda correcto: [100, 116, 117, 205] if correcto else [205, 54, 46, 255]
        )
    else:
        puntos["color"] = puntos["ciudad"].map(COLORES)

    puntos["clase_predicha"] = puntos["prediccion"].map(CLASES)
    puntos["probabilidad"] = (100 * puntos["probabilidad_clase_1"]).round(2)

    capa = pdk.Layer(
        "ScatterplotLayer",
        data=puntos,
        get_position="[longitud, latitud]",
        get_fill_color="color",
        get_radius=140,
        radius_min_pixels=4,
        radius_max_pixels=13,
        stroked=True,
        get_line_color=[255, 255, 255, 165],
        line_width_min_pixels=1,
        pickable=True,
        auto_highlight=True,
    )

    if len(ciudades) == 2:
        vista = pdk.ViewState(latitude=37, longitude=-48, zoom=1.35, pitch=0)
    else:
        lat, lon = CIUDADES[ciudades[0]]["centro"]
        vista = pdk.ViewState(latitude=lat, longitude=lon, zoom=10.6, pitch=0)

    return pdk.Deck(
        layers=[capa],
        initial_view_state=vista,
        map_style=(
            "mapbox://styles/mapbox/light-v11"
            if token else "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
        ),
        map_provider="mapbox" if token else "carto",
        api_keys={"mapbox": token} if token else None,
        tooltip={
            "html": (
                "<b>{ciudad}</b><br/>"
                "{conjunto} · ID {id}<br/>"
                "Latitud: {latitud}<br/>"
                "Longitud: {longitud}<br/>"
                "Predicción: {clase_predicha}<br/>"
                "P(Ámsterdam): {probabilidad}%"
            ),
            "style": {"backgroundColor": "#20272A", "color": "#FAFAF7", "fontSize": "12px"},
        },
    )


def cargar_modelo_final(pesos: dict) -> RedNeuronal:
    modelo = RedNeuronal()
    estado = pesos["estados"]["final"]
    modelo.W = [np.asarray(w, dtype=np.float64) for w in estado["W"]]
    modelo.b = [np.asarray(b, dtype=np.float64) for b in estado["b"]]
    return modelo


datos, historial, resultados = cargar_archivos()

with st.sidebar:
    st.header("Filtros")
    ciudades = st.multiselect(
        "Ciudades",
        options=list(CIUDADES),
        default=list(CIUDADES),
    )
    conjunto = st.selectbox("Conjunto", ["Todos", "Entrenamiento", "Prueba"])
    color_por = st.selectbox("Color", ["Ciudad", "Predicción", "Resultado"])

st.title("Clasificación de coordenadas GPS")
met1, met2 = st.columns(2)
met1.metric("Puntos", len(datos))
met2.metric("Exactitud en prueba", f"{resultados['metricas_finales']['prueba']['exactitud']:.1%}")

mapa, modelo = st.tabs(["Mapa", "Modelo"])

with mapa:
    seleccion = datos[datos["ciudad"].isin([CIUDADES[c]["nombre_datos"] for c in ciudades])]
    if conjunto != "Todos":
        seleccion = seleccion[seleccion["conjunto"] == conjunto.lower()]

    if ciudades:
        if color_por == "Resultado":
            leyenda = (
                "<span class='leyenda'><span class='punto' style='background:#647475'></span>Acierto</span>"
                "<span class='leyenda'><span class='punto' style='background:#CD362E'></span>Error</span>"
            )
        else:
            leyenda = (
                "<span class='leyenda'><span class='punto' style='background:#B95937'></span>Guadalajara</span>"
                "<span class='leyenda'><span class='punto' style='background:#267780'></span>Ámsterdam</span>"
            )
        st.markdown(leyenda, unsafe_allow_html=True)
        st.pydeck_chart(
            preparar_mapa(seleccion, ciudades, color_por, token_mapbox()),
            use_container_width=True,
        )

        with st.expander("Ver datos"):
            tabla = seleccion[[
                "id", "ciudad", "latitud", "longitud", "conjunto", "etiqueta",
                "probabilidad_clase_1", "prediccion", "correcta",
            ]].rename(columns={
                "id": "ID", "ciudad": "Ciudad", "latitud": "Latitud", "longitud": "Longitud",
                "conjunto": "Conjunto", "etiqueta": "Clase real",
                "probabilidad_clase_1": "P(Ámsterdam)", "prediccion": "Clase predicha",
                "correcta": "Correcto",
            })
            st.dataframe(tabla, hide_index=True, use_container_width=True)
            st.download_button(
                "Descargar CSV",
                data=tabla.to_csv(index=False).encode("utf-8-sig"),
                file_name="puntos_filtrados.csv",
                mime="text/csv",
            )
    else:
        st.info("Selecciona al menos una ciudad.")

with modelo:
    grafica, parametros = st.columns([1.5, 1], gap="large")
    with grafica:
        st.subheader("Error de entrenamiento")
        hist = historial[historial["conjunto"] == "240 puntos"].rename(
            columns={"epoch": "Epoch", "mse_antes_paso": "MSE"}
        )
        st.line_chart(hist, x="Epoch", y="MSE", height=315)

    with parametros:
        st.subheader("Parámetros")
        st.markdown(
            "**Capas:** 2 → 4 → 8 → 1  \n"
            "**Activación:** ReLU / sigmoide  \n"
            f"**Tasa de aprendizaje:** {resultados['lr']}  \n"
            f"**Epochs:** {resultados['epochs']}"
        )

    st.divider()
    st.subheader("Clasificar una coordenada")
    referencia = st.selectbox("Ciudad de referencia", list(CIUDADES))
    lat_base, lon_base = CIUDADES[referencia]["centro"]
    with st.form("coordenada"):
        col_lat, col_lon = st.columns(2)
        with col_lat:
            lat = st.number_input("Latitud", value=float(lat_base), format="%.6f")
        with col_lon:
            lon = st.number_input("Longitud", value=float(lon_base), format="%.6f")
        enviar = st.form_submit_button("Clasificar")

    if enviar:
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            st.error("Coordenadas fuera del rango permitido.")
        else:
            entrada = (np.asarray([[lat, lon]]) - CENTRO_NORMALIZACION) / ESCALA_NORMALIZACION
            prob = float(cargar_modelo_final(resultados).predict_proba(entrada)[0, 0])
            clase = int(prob >= 0.5)
            st.write(f"**Predicción:** {CLASES[clase]}")
            st.write(f"P(Ámsterdam): {prob:.4f} · P(Guadalajara): {1 - prob:.4f}")
