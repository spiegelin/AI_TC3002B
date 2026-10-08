"""Red Hopfield de figuras M x N usando solo listas y control de flujo de Python.

Entrenamiento: dataset/uno.txt, dataset/dos.txt
Prueba: dataset/x.txt
El archivo dataset/tres.txt es vacio en el template
"""


def leer_figura(ruta):
    matriz = []
    with open(ruta, "r", encoding="utf-8") as archivo:
        for linea in archivo:
            if linea.strip() != "":
                fila = []
                for valor in linea.split():
                    if valor == "0":
                        fila.append(0)
                    elif valor == "1":
                        fila.append(1)
                    else:
                        raise ValueError("Valor no permitido en " + ruta)
                matriz.append(fila)

    if not matriz or not matriz[0]:
        raise ValueError("Figura vacia: " + ruta)

    columnas = len(matriz[0])
    for fila in matriz:
        if len(fila) != columnas:
            raise ValueError("Matriz no rectangular: " + ruta)
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


def entrenar(patrones):
    n = len(patrones[0])
    pesos = [[0 for _ in range(n)] for _ in range(n)]

    for patron in patrones:
        if len(patron) != n:
            raise ValueError("Todos los patrones deben medir M x N")
        for i in range(n):
            for j in range(n):
                if i != j:
                    pesos[i][j] += patron[i] * patron[j]
    return pesos


def recuperar(entrada, pesos, max_iteraciones=100):
    estado = entrada[:]
    visitados = [estado[:]]
    iteraciones = 0

    while iteraciones < max_iteraciones:
        siguiente = [0] * len(estado)
        for i in range(len(estado)):
            campo = 0
            for j in range(len(estado)):
                campo += pesos[i][j] * estado[j]

            if campo > 0:
                siguiente[i] = 1
            elif campo < 0:
                siguiente[i] = -1
            else:
                siguiente[i] = estado[i]

        iteraciones += 1
        if siguiente == estado:
            return siguiente, iteraciones, True
        if siguiente in visitados:
            return siguiente, iteraciones, False
        visitados.append(siguiente[:])
        estado = siguiente
    return estado, iteraciones, False


def identificar(estado, patrones, nombres):
    for i in range(len(patrones)):
        if estado == patrones[i]:
            return nombres[i]
    return "No reconocido"


def mostrar_figura(vector, filas, columnas):
    for i in range(filas):
        linea = ""
        for j in range(columnas):
            if vector[i * columnas + j] == 1:
                linea += "##"
            else:
                linea += "  "
        print(linea)


def main():
    # Solo figuras existentes y no vacias del repositorio del profesor.
    nombres = ["1", "2"]
    rutas = ["dataset/uno.txt", "dataset/dos.txt"]

    figuras = []
    for ruta in rutas:
        figuras.append(leer_figura(ruta))

    filas = len(figuras[0])
    columnas = len(figuras[0][0])
    for figura in figuras:
        if len(figura) != filas or len(figura[0]) != columnas:
            raise ValueError("Dimensiones incompatibles")

    patrones = []
    for figura in figuras:
        patrones.append(vectorizar(figura))

    pesos = entrenar(patrones)
    print("Red de Hopfield: {} x {} = {} neuronas".format(
        filas, columnas, filas * columnas))
    print("Patrones almacenados:", ", ".join(nombres))

    for i in range(len(patrones)):
        resultado, barridos, estable = recuperar(patrones[i], pesos)
        print("Patron {} => {} ({} barridos, estable={})".format(
            nombres[i], identificar(resultado, patrones, nombres), barridos, estable))

    figura_entrada = leer_figura("dataset/x.txt")
    if len(figura_entrada) != filas or len(figura_entrada[0]) != columnas:
        raise ValueError("x.txt debe tener dimensiones M x N")

    entrada = vectorizar(figura_entrada)
    print("\nFigura de entrada (x.txt):")
    mostrar_figura(entrada, filas, columnas)

    recuperada, barridos, estable = recuperar(entrada, pesos)
    print("\nFigura recuperada:")
    mostrar_figura(recuperada, filas, columnas)
    print("\nReconocimiento:", identificar(recuperada, patrones, nombres))
    print("Barridos:", barridos)
    print("Convergencia:", "si" if estable else "no")


if __name__ == "__main__":
    main()
