# Instrucciones de ejecución
1. Instalar las dependencias:
```bash
python -m pip install -r requirements.txt
```
2. Obtener un token público de Mapbox en [https://account.mapbox.com/access-tokens/](https://account.mapbox.com/access-tokens/).

En la carpeta `.streamlit`, crear el archivo `secrets.toml` (copiar `secrets.toml.example`) y pegar el token:
```toml
MAPBOX_TOKEN = "pk.TOKEN_DE_MAPBOX"
```
3. Ejecutar la aplicación:
```bash
python -m streamlit run app.py
```
4. La aplicación debería estar corriendo en [http://localhost:8501](http://localhost:8501).

# Excel de comprobación
El Excel para comprobar los resultados es el siguiente [validacion_red_neuronal.xlsx](./validacion_red_neuronal.xlsx)