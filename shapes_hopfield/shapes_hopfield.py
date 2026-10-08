"""Red de Hopfield: reconocimiento de caras feliz, triste y enojada.

Entrenamiento por regla de proyeccion (pseudo-inversa), util para imagenes
altamente correlacionadas. Recuperacion asincrona para lograr estabilidad.
"""

import sys


NOMBRES = ["feliz", "triste", "enojado"]
ARCHIVOS = ["dataset/feliz.txt", "dataset/triste.txt", "dataset/enojado.txt"]


def leer_figura(ruta):
    matriz = []
    with open(ruta, "r", encoding="utf-8") as archivo:
        for linea in archivo:
            if linea.strip():
                fila = []
                for pixel in linea.split():
                    if pixel == "0":
                        fila.append(0)
                    elif pixel == "1":
                        fila.append(1)
                    else:
                        raise ValueError("Se permiten solo 0 y 1: " + ruta)
                matriz.append(fila)

    if not matriz or not matriz[0]:
        raise ValueError("Figura vacia: " + ruta)

    columnas = len(matriz[0])
    for fila in matriz:
        if len(fila) != columnas:
            raise ValueError("La matriz no es rectangular: " + ruta)
    return matriz


def vectorizar(matriz):
    vector = []
    for fila in matriz:
        for pixel in fila:
            if pixel == 1:
                vector.append(1)
            else:
                vector.append(-1)
    return vector


def invertir_matriz(matriz):
    """Inversa de una matriz pequena usando Gauss-Jordan con listas."""
    n = len(matriz)
    aumentada = []
    for i in range(n):
        fila = matriz[i][:]
        for j in range(n):
            if i == j:
                fila.append(1.0)
            else:
                fila.append(0.0)
        aumentada.append(fila)

    for i in range(n):
        pivote = i
        for j in range(i + 1, n):
            if abs(aumentada[j][i]) > abs(aumentada[pivote][i]):
                pivote = j
        if abs(aumentada[pivote][i]) < 1e-10:
            raise ValueError("Figuras dependientes: no se puede invertir la matriz")
        aumentada[i], aumentada[pivote] = aumentada[pivote], aumentada[i]
        divisor = aumentada[i][i]
        for j in range(2 * n):
            aumentada[i][j] /= divisor

        for k in range(n):
            if k != i:
                factor = aumentada[k][i]
                for j in range(2 * n):
                    aumentada[k][j] -= factor * aumentada[i][j]

    inversa = []
    for fila in aumentada:
        inversa.append(fila[n:])
    return inversa


def entrenar(patrones):
    """Regla de proyeccion: W = X (X^T X)^(-1) X^T, W_ii = 0.

    Las caras comparten muchos pixeles. La regla de Hebb simple puede
    mezclarlas; la proyeccion permite conservar estos tres patrones.
    """
    if not patrones:
        raise ValueError("Se necesita al menos un patron")
    cantidad = len(patrones)
    neuronas = len(patrones[0])
    for patron in patrones:
        if len(patron) != neuronas:
            raise ValueError("Las figuras tienen dimensiones distintas")

    gram = []
    for p in range(cantidad):
        fila = []
        for q in range(cantidad):
            suma = 0
            for i in range(neuronas):
                suma += patrones[p][i] * patrones[q][i]
            fila.append(float(suma))
        gram.append(fila)
    gram_inversa = invertir_matriz(gram)

    pesos = []
    for i in range(neuronas):
        pesos.append([0.0] * neuronas)

    for i in range(neuronas):
        for j in range(i + 1, neuronas):
            peso = 0.0
            for p in range(cantidad):
                for q in range(cantidad):
                    peso += patrones[p][i] * gram_inversa[p][q] * patrones[q][j]
            pesos[i][j] = peso
            pesos[j][i] = peso

    return pesos


def recuperar(entrada, pesos, max_barridos=100):
    estado = entrada[:]
    barridos = 0

    while barridos < max_barridos:
        cambios = 0
        for i in range(len(estado)):
            campo = 0.0
            for j in range(len(estado)):
                campo += pesos[i][j] * estado[j]

            anterior = estado[i]
            if campo > 1e-9:
                estado[i] = 1
            elif campo < -1e-9:
                estado[i] = -1
            else:
                estado[i] = anterior

            if estado[i] != anterior:
                cambios += 1

        barridos += 1
        if cambios == 0:
            return estado, barridos, True

    return estado, barridos, False


def identificar(estado, patrones, nombres):
    for i in range(len(patrones)):
        if estado == patrones[i]:
            return nombres[i]
    return "No reconocida (posible estado espurio)"


def mostrar_figura(vector, filas, columnas):
    for i in range(filas):
        linea = ""
        for j in range(columnas):
            if vector[i * columnas + j] == 1:
                linea += "##"
            else:
                linea += "  "
        print(linea)


def cargar_patrones():
    figuras = []
    for ruta in ARCHIVOS:
        figuras.append(leer_figura(ruta))

    filas = len(figuras[0])
    columnas = len(figuras[0][0])
    patrones = []
    for figura in figuras:
        if len(figura) != filas or len(figura[0]) != columnas:
            raise ValueError("Todas las figuras deben medir M x N")
        patrones.append(vectorizar(figura))
    return patrones, filas, columnas


def main():
    patrones, filas, columnas = cargar_patrones()
    pesos = entrenar(patrones)
    print("Red de Hopfield: {} x {} = {} neuronas".format(
        filas, columnas, filas * columnas))
    print("Patrones almacenados:", ", ".join(NOMBRES))

    for i in range(len(patrones)):
        salida, pasos, estable = recuperar(patrones[i], pesos)
        print("Patron {} => {} ({} barridos, estable={})".format(
            NOMBRES[i], identificar(salida, patrones, NOMBRES), pasos, estable))

    ruta = sys.argv[1] if len(sys.argv) > 1 else "dataset/prueba_feliz.txt"
    figura = leer_figura(ruta)
    if len(figura) != filas or len(figura[0]) != columnas:
        raise ValueError("La prueba debe medir {} x {}".format(filas, columnas))

    entrada = vectorizar(figura)
    print("\nFigura de entrada ({}):".format(ruta))
    mostrar_figura(entrada, filas, columnas)
    salida, pasos, estable = recuperar(entrada, pesos)
    print("\nFigura recuperada:")
    mostrar_figura(salida, filas, columnas)
    print("\nReconocimiento:", identificar(salida, patrones, NOMBRES))
    print("Barridos:", pasos)
    print("Convergencia:", "si" if estable else "no")


if __name__ == "__main__":
    main()
