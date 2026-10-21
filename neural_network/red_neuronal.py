"""
Red neuronal 2-4-8-1

150 puntos GPS en Guadalajara y 150 en Ámsterdam.
Activaciones ReLU / ReLU / sigmoide.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

SEED = 20261020
ARQUITECTURA = (2, 4, 8, 1)
LEARNING_RATE = 0.05
EPOCHS = 1200
CENTRO_NORMALIZACION = np.array([30.0, -53.0])
ESCALA_NORMALIZACION = np.array([20.0, 50.0])
CIUDADES = (
    ("Guadalajara, Mexico", 20.6767, -103.3475, 0),
    ("Ámsterdam, Países Bajos", 52.3676, 4.9041, 1),
)


def circulo_gps(lat_centro: float, lon_centro: float, n: int,
                radio_km: float, rng: np.random.Generator) -> np.ndarray:
    """Muestrea puntos aproximadamente uniformes en un disco alrededor del centro.

    Aproximacion local en km a grados.
    """
    radio = radio_km * np.sqrt(rng.random(n))
    angulo = rng.uniform(0, 2 * np.pi, n)
    delta_lat = radio * np.sin(angulo) / 110.57
    delta_lon = radio * np.cos(angulo) / (111.32 * np.cos(np.deg2rad(lat_centro)))
    return np.round(np.column_stack([lat_centro + delta_lat, lon_centro + delta_lon]), 6)


def preparar_datos(semilla: int = SEED):
    rng = np.random.default_rng(semilla)
    datos = []
    for ciudad, lat, lon, etiqueta in CIUDADES:
        puntos = circulo_gps(lat, lon, 150, 6.0, rng)
        for punto in puntos:
            datos.append({"ciudad": ciudad, "latitud": float(punto[0]),
                          "longitud": float(punto[1]), "etiqueta": etiqueta})

    # Particion estratificada: 120 entrenamiento + 30 prueba por ciudad.
    indices_entreno, indices_prueba = [], []
    for clase in (0, 1):
        indices = np.arange(150 * clase, 150 * (clase + 1))
        indices = rng.permutation(indices)
        indices_entreno.extend(indices[:120].tolist())
        indices_prueba.extend(indices[120:].tolist())
    indices_entreno = rng.permutation(indices_entreno)
    indices_prueba = rng.permutation(indices_prueba)

    gps = np.array([[d["latitud"], d["longitud"]] for d in datos], dtype=float)
    x = (gps - CENTRO_NORMALIZACION) / ESCALA_NORMALIZACION
    y = np.array([d["etiqueta"] for d in datos], dtype=float).reshape(-1, 1)
    conjunto = np.array(["prueba"] * len(datos), dtype=object)
    conjunto[indices_entreno] = "entrenamiento"
    for i, d in enumerate(datos):
        d.update({"id": i + 1, "x1": float(x[i, 0]), "x2": float(x[i, 1]),
                  "conjunto": str(conjunto[i])})
    return datos, x, y, indices_entreno, indices_prueba


def sigmoide(z: np.ndarray) -> np.ndarray:
    # Evita overflow numerico para argumentos extremos.
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class RedNeuronal:
    """Clasificador feedforward, backprop y descenso de gradiente en NumPy."""

    def __init__(self):
        # Inicialización He para romper la simetría entre neuronas.
        rng = np.random.default_rng(SEED)
        self.W = [rng.standard_normal((a, b)) * np.sqrt(2.0 / a)
                  for a, b in zip(ARQUITECTURA[:-1], ARQUITECTURA[1:])]
        self.b = [np.zeros((1, b), dtype=float)
                  for b in ARQUITECTURA[1:]]

    def estado(self) -> dict:
        return {"W": [m.copy() for m in self.W],
                "b": [v.copy() for v in self.b]}

    def propagacion(self, x: np.ndarray) -> dict:
        a0 = x
        z1 = a0 @ self.W[0] + self.b[0]
        a1 = np.maximum(0.0, z1)
        z2 = a1 @ self.W[1] + self.b[1]
        a2 = np.maximum(0.0, z2)
        z3 = a2 @ self.W[2] + self.b[2]
        a3 = sigmoide(z3)
        return {"a0": a0, "z1": z1, "a1": a1,
                "z2": z2, "a2": a2, "z3": z3, "a3": a3}

    def gradientes(self, x: np.ndarray, y: np.ndarray, cache: dict) -> dict:
        n = x.shape[0]
        a3, a2, a1 = cache["a3"], cache["a2"], cache["a1"]
        # Derivada de L = 1/(2N)*sum((pred-y)^2).
        d3 = (a3 - y) * a3 * (1 - a3)
        d2 = (d3 @ self.W[2].T) * (cache["z2"] > 0)
        d1 = (d2 @ self.W[1].T) * (cache["z1"] > 0)
        dW = [x.T @ d1 / n, a1.T @ d2 / n, a2.T @ d3 / n]
        db = [np.mean(d, axis=0, keepdims=True) for d in [d1, d2, d3]]
        return {"delta1": d1, "delta2": d2, "delta3": d3,
                "dW": dW, "db": db}

    def paso(self, x: np.ndarray, y: np.ndarray, tasa: float):
        cache = self.propagacion(x)
        grads = self.gradientes(x, y, cache)
        # W y b se actualizan despues de calcular todos los deltas con pesos previos.
        self.W = [w - tasa * dw for w, dw in zip(self.W, grads["dW"])]
        self.b = [b - tasa * db for b, db in zip(self.b, grads["db"])]
        mse = float(np.mean((cache["a3"] - y) ** 2))
        return cache, grads, mse

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.propagacion(x)["a3"]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(int)


def como_lista(estado):
    return {"W": [v.tolist() for v in estado["W"]],
            "b": [v.tolist() for v in estado["b"]]}


def metricas(modelo, x, y):
    proba = modelo.predict_proba(x)
    return {"mse": float(np.mean((proba - y) ** 2)),
            "exactitud": float(np.mean((proba >= 0.5) == y))}


def csv_guardar(ruta, nombres, filas):
    with ruta.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(nombres)
        writer.writerows(filas)


def ejecutar(destino: Path, epochs: int = EPOCHS):
    destino.mkdir(parents=True, exist_ok=True)
    registros, x, y, i_train, i_test = preparar_datos()
    modelo = RedNeuronal()
    historial = []

    # Entrenamiento con 240 puntos (120 de cada ciudad).
    for epoch in range(epochs):
        _, _, mse = modelo.paso(x[i_train], y[i_train], LEARNING_RATE)
        if epoch == 0 or (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            met_train = metricas(modelo, x[i_train], y[i_train])
            met_test = metricas(modelo, x[i_test], y[i_test])
            historial.append([epoch + 1, "240 puntos", mse,
                              met_train["exactitud"], met_test["exactitud"]])

    csv_guardar(destino / "datos_gps.csv",
                ["id", "ciudad", "latitud", "longitud", "etiqueta", "x1", "x2",
                 "conjunto"],
                [[d[k] for k in ["id", "ciudad", "latitud", "longitud", "etiqueta",
                                   "x1", "x2", "conjunto"]] for d in registros])

    csv_guardar(destino / "historial_entrenamiento.csv",
                ["epoch", "conjunto", "mse_antes_paso", "exactitud_entreno", "exactitud_prueba"],
                historial)

    pred = modelo.predict_proba(x).reshape(-1)
    csv_guardar(destino / "predicciones.csv",
                ["id", "ciudad", "clase_real", "probabilidad_clase_1",
                 "prediccion", "correcta", "conjunto"],
                [[d["id"], d["ciudad"], d["etiqueta"], float(pred[i]),
                  int(pred[i] >= 0.5), int((pred[i] >= .5) == d["etiqueta"]),
                  d["conjunto"]] for i, d in enumerate(registros)])

    resultado = {
        "descripcion": "Red neuronal NumPy 2-4-8-1 para clasificar GPS",
        "arquitectura": ARQUITECTURA,
        "activaciones": ["relu", "relu", "sigmoide"],
        "semilla": SEED,
        "lr": LEARNING_RATE,
        "epochs": epochs,
        "normalizacion": {"centro": CENTRO_NORMALIZACION.tolist(),
                           "escala": ESCALA_NORMALIZACION.tolist()},
        "ciudades": [{"nombre": v[0], "lat_centro": v[1], "lon_centro": v[2],
                     "etiqueta": v[3]} for v in CIUDADES],
        "estados": {"final": como_lista(modelo.estado())},
        "metricas_finales": {"entrenamiento": metricas(modelo, x[i_train], y[i_train]),
                             "prueba": metricas(modelo, x[i_test], y[i_test])},
    }
    with (destino / "pesos_y_resultados.json").open("w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    print("Arquitectura:", " - ".join(map(str, ARQUITECTURA)))
    print("Datos: 300 GPS (150 Guadalajara + 150 Ámsterdam)")
    print("Epochs:", epochs, "| Learning rate:", LEARNING_RATE)
    print("Exactitud train:", f"{resultado['metricas_finales']['entrenamiento']['exactitud']:.1%}")
    print("Exactitud test:", f"{resultado['metricas_finales']['prueba']['exactitud']:.1%}")
    print("Archivos guardados en:", destino.resolve())
    return {"resultado": resultado, "registros": registros, "x": x, "y": y,
            "i_train": i_train, "i_test": i_test,
            "modelo": modelo, "historial": historial}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", type=Path, default=Path(__file__).resolve().parent,
                        help="Carpeta para los archivos CSV y JSON")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Epochs de entrenamiento con los 240 puntos")
    args = parser.parse_args()
    ejecutar(args.salida, args.epochs)
