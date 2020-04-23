# Dash Application for Dynamic Routing

## Quick start

1. Primero debes clonar el repositorio y entrar en la carpeta del proyecto:

```sh
$ git clone https://github.com/IngenieriaUP/dash-rutas-ch.git
$ cd dash-rutas-ch/
```

2. Opcional: Crea un ambiente virtual para instalar las dependencias y actívalo:

```sh
$ python3 -m virtualenv .env
$ source .env/bin/activate
```

3. Instala las dependencias necesarias para la aplicacion:

```sh
(.env) $ pip install -r requirements.txt
```

4. Configura tus [API key de Mapbox](https://account.mapbox.com/) en un archivo llamado mykeys.py que debe contener:

```python
MAPBOX_API_KEY = [AQUI COLOCA TU API KEY DE MAPBOX]
```

5. Corre la aplicación para probarla mediante un navegador

```python
(.env) $ python app.py
```

6. Entra a la direccion http://127.0.0.1:8050/ y podrás ver la aplicación.

#### Aviso

Este proyecto fue una prueba de concepto, tiene 3 opciones para obtener las rutas de punto a punto:

- Mapbox Directions API (necesita API KEY) - ESTABLE
- Google Maps Directions API (necesita API KEY) - EN PRUEBA 
- Libreria NetworkX utilizando el grafo de las vias de la ciudad - EN PRUEBA
