#!/usr/bin/env python3
import argparse
import io
import math
import random
import sys
import wave
from pathlib import Path

NOTAS = ("DO", "RE", "MI", "FA", "SOL", "LA", "SI")
FRECUENCIAS = (261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88)
MATRIZ = (
    (0.10, 0.35, 0.25, 0.05, 0.20, 0.05, 0.00),  # DO
    (0.20, 0.10, 0.30, 0.15, 0.15, 0.05, 0.05),  # RE
    (0.20, 0.30, 0.10, 0.20, 0.15, 0.05, 0.00),  # MI
    (0.10, 0.15, 0.35, 0.10, 0.25, 0.05, 0.00),  # FA
    (0.35, 0.05, 0.15, 0.15, 0.10, 0.15, 0.05),  # SOL
    (0.10, 0.15, 0.20, 0.10, 0.25, 0.10, 0.10),  # LA
    (0.50, 0.05, 0.10, 0.05, 0.20, 0.05, 0.05),  # SI
)
# Interstellar: LA-MI, SI-MI, DO-MI, RE-MI
NOTAS_INTERSTELLAR = ("LA", "SI", "DO", "RE", "MI")
FRECUENCIAS_INTERSTELLAR = (440.00, 493.88, 523.25, 587.33, 659.25)
MATRIZ_INTERSTELLAR = (
    (0.05, 0.05, 0.05, 0.05, 0.80),  # LA -> MI
    (0.05, 0.05, 0.05, 0.05, 0.80),  # SI -> MI
    (0.05, 0.05, 0.05, 0.05, 0.80),  # DO -> MI
    (0.05, 0.05, 0.05, 0.05, 0.80),  # RE -> MI
    (0.32, 0.26, 0.22, 0.18, 0.02),  # MI -> LA, SI, DO, RE o MI
)
TASA_MUESTREO = 44100


def sintetizar_tono(frecuencia, duracion):
    """Devuelve audio PCM mono a 16 bits de una nota y una pausa corta"""
    from array import array

    n = round(TASA_MUESTREO * duracion)
    rampa = max(1, round(TASA_MUESTREO * 0.015))
    muestras = array("h")
    for i in range(n):
        t = i / TASA_MUESTREO
        envolvente = min(1.0, i / rampa, (n - i - 1) / rampa)
        onda = (0.78 * math.sin(2 * math.pi * frecuencia * t)
                + 0.17 * math.sin(4 * math.pi * frecuencia * t)
                + 0.05 * math.sin(6 * math.pi * frecuencia * t))
        muestras.append(round(0.35 * 32767 * max(0.0, envolvente) * onda))
    muestras.extend([0] * round(TASA_MUESTREO * 0.035))
    if sys.byteorder != "little":
        muestras.byteswap()
    return muestras.tobytes()


def como_wav(audio_pcm):
    salida = io.BytesIO()
    with wave.open(salida, "wb") as archivo:
        archivo.setnchannels(1)
        archivo.setsampwidth(2)
        archivo.setframerate(TASA_MUESTREO)
        archivo.writeframes(audio_pcm)
    return salida.getvalue()


def reproducir(audio_pcm):
    """Intenta reproducir una nota"""
    datos = como_wav(audio_pcm)
    if sys.platform == "win32":
        try:
            import winsound
            winsound.PlaySound(datos, winsound.SND_MEMORY)
            return True
        except (ImportError, RuntimeError, OSError):
            return False


def generar(transiciones, duracion, semilla, archivo, reproducir_ahora,
            modelo="mayor"):
    if transiciones < 1 or duracion <= 0:
        raise ValueError("Se necesitan transiciones >= 1 y duración > 0.")
    if modelo == "mayor":
        notas, frecuencias, matriz = NOTAS, FRECUENCIAS, MATRIZ
        nota_inicial = "DO4"
    elif modelo == "interestelar":
        notas, frecuencias, matriz = NOTAS_INTERSTELLAR, FRECUENCIAS_INTERSTELLAR, MATRIZ_INTERSTELLAR
        nota_inicial = "LA4"
    else:
        raise ValueError("Modelo desconocido: " + modelo)
    if any(len(fila) != len(notas) or
           any(p < 0 for p in fila) or
           not math.isclose(sum(fila), 1.0, abs_tol=1e-9)
           for fila in matriz):
        raise ValueError("La matriz de transición no es válida.")

    azar = random.Random(semilla)
    estado = 0  # DO4 en mayor; LA4 en interestelar
    fragmentos = []
    hubo_audio = False
    print(f"Modelo: {modelo}. Estado inicial: {nota_inicial}. "
          f"Transiciones: {transiciones}")
    for paso in range(1, transiciones + 1):
        siguiente = azar.choices(range(len(notas)), weights=matriz[estado], k=1)[0]
        frecuencia = frecuencias[siguiente]
        print(f"{paso:02d}: {notas[estado]:3s} -> {notas[siguiente]:3s} "
              f"| P={matriz[estado][siguiente]:.2f} | {frecuencia:.2f} Hz")
        tono = sintetizar_tono(frecuencia, duracion)
        fragmentos.append(tono)
        if reproducir_ahora:
            funciono = reproducir(tono)
            hubo_audio = hubo_audio or funciono
            if not funciono:
                reproducir_ahora = False 
        estado = siguiente

    ruta = Path(archivo)
    ruta.write_bytes(como_wav(b"".join(fragmentos)))
    print(f"Melodía guardada en: {ruta.resolve()}")
    if not hubo_audio:
        print("Para escucharla, abre el WAV en un reproductor de audio.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Melodías mediante cadenas de Markov")
    parser.add_argument("--transiciones", type=int, default=16)
    parser.add_argument("--duracion", type=float, default=0.32,
                        help="Segundos por nota")
    parser.add_argument("--semilla", type=int, default=None,
                        help="Permite repetir la misma melodía")
    parser.add_argument("--archivo", default="melodia_markov.wav")
    parser.add_argument("--sin-reproducir", action="store_true",
                        help="Solo guardar el WAV; útil en equipos sin salida de audio")
    parser.add_argument("--modelo", choices=("mayor", "interestelar"),
                        default="mayor", help="Conjunto de notas y matriz")
    args = parser.parse_args()
    generar(args.transiciones, args.duracion, args.semilla,
            args.archivo, not args.sin_reproducir, args.modelo)
