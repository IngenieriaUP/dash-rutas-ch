# Import dependencies
from warnings import filterwarnings
filterwarnings('ignore')

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import flask
#import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import plotly
import os
#import osmnx as ox
import requests
import time
from dash.dependencies import Input, Output, State
from shapely.geometry import LineString, Point
import googlemaps

# Initialize dash application
app = dash.Dash()
server = app.server

if 'DYNO' in os.environ:
    app.scripts.append_script({
        'external_url': 'https://cdn.rawgit.com/chriddyp/ca0d8f02a1659981a0ea7f013a378bbd/raw/e79f3f789517deec58f41251f7dbb6bee72c44ab/plotly_ga.js'
    })

# API keys
mapbox_access_token = "pk.eyJ1IjoiY2xhdWRpbzk3IiwiYSI6ImNqbzM2NmFtMjB0YnUzd3BvenZzN3QzN3YifQ.heZHwQTY8TWhuO0u2-BxxA"
gmaps = googlemaps.Client(key='AIzaSyDo3XDr4zJsyynCuH9KMQc4IbPrI6YaNGY')

# Coordinate system
proj_utm = {'datum': 'WGS84', 'ellps': 'WGS84', 'proj': 'utm', 'zone': 18, 'units': 'm'}

# Datasets
map_data = pd.read_csv('~/data/input/school-hospitalMatrix.csv', index_col=0)
map_data["color"] = map_data.apply(lambda x: '#ff0000' if x['TYPE'] == "School" else '#0000ff', axis=1)
#map_geometry= map_data.apply(lambda x: Point(x['C_LONG'], x['C_LAT']), axis=1)
#map_data = gpd.GeoDataFrame(data=map_data, geometry=map_geometry,
#                            crs={'init': 'epsg:4326'})
#map_data = map_data.to_crs(proj_utm)
#map_data.reset_index(inplace=True)

print("Loading graph")
init = time.time()
#graph = pickle.load(open("data/input/lima_graph_proj.pk","rb"))
wait = time.time() - init
print("graph loaded in", wait)

print("Loading nodes")
init = time.time()
#nodes = ox.graph_to_gdfs(graph, nodes=True, edges=False)
wait = time.time() - init
print("nodes loaded in", wait)
print('#'*40)
#print('n_nodes:',len(nodes))
print('#'*40)

#print("Loading edges")
#init = time.time()
#edges = gpd.read_file('data/input/edges_proj.geojson')
#wait = time.time() - init
#print("edges loaded in", wait)

layout = dict(
    #autosize=True,
    height = '1080px',
    font=dict(color="#edefef"),
    titlefont=dict(
        color="#edefef",
        size='14'
    ),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    hovermode="closest",
    plot_bgcolor='#4b4b4b',
    paper_bgcolor='#4b4b4b',
    showlegend=False,
    title='Rutas de Evacuación de Colegio a Hospitales',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        style="light",
        center=dict(
            lon=-76.925824,
            lat=-12.081203,
        ),
        pitch=0,
        zoom=10
    )
)

initial_map = {
    "data":
    [
        {"type": "scattermapbox",
        "lat": list(map_data['C_LAT']),
        "lon": list(map_data['C_LONG']),
        "text": list(map_data['TYPE']),
        "customdata": list(map_data['LINK']),
        "mode": "markers",
        "marker":
            {
            "size": 5,
            "opacity": 0.7,
            "color": list(map_data['color'])
            }
        }
    ], "layout": layout
}

def gen_map(map_data, route_line):
    points = {
            "type": "scattermapbox",
            "lat": list(map_data['C_LAT']),
            "lon": list(map_data['C_LONG']),
            "text": list(map_data['TYPE']),
            "customdata": list(map_data['LINK']),
            "mode": "markers",
            "marker": {
                "size": 5,
                "opacity": 0.7,
                "color": list(map_data['color'])
                }
            }

    points_inner = {
            "type": "scattermapbox",
            "lat": list(map_data['C_LAT']),
            "lon": list(map_data['C_LONG']),
            "text": list(map_data['TYPE']),
            "customdata": list(map_data['LINK']),
            "mode": "markers",
            "marker": {
                "size": 3,
                "opacity": 0.7,
                "color": list(map_data['color'])
                }
            }

    route = {
            "type": "scattermapbox",
            "lat": [tuple_xy[0] for tuple_xy in route_line[:-1]],
            "lon": [tuple_xy[1] for tuple_xy in route_line[1:]],
            "mode": "lines+markers",
            "line": {
                "width": 7,
                "color": 'green',
                "opacity": 0.5
                },
            "marker": {
                "size": 7,
                "opacity": 0.7,
                "color": 'green'
                }
            }

    return {
        "data": [points, route],
        "layout": layout,
        }

app.layout = html.Div(children = [
    # Map
    html.Div(children = [
        dcc.Graph(id='map-graph')
        ])
    ],
    style={"padding-top": "20px"},
    className = '80 rows'
)

def get_boundingbox(x, y, margin):
    x = Point(x)
    y = Point(y)
    xy = gpd.GeoDataFrame(geometry=[x, y], crs=proj_utm)
    xmin, ymin, xmax, ymax = xy.unary_union.bounds
    xmin -= margin
    ymin -= margin
    xmax += margin
    ymax += margin
    print('bbox:',xmin, ymin, xmax, ymax)
    return xmin, ymin, xmax, ymax

def get_subgraph(graph, nodes, source, target):
    xmin, ymin, xmax, ymax = get_boundingbox(source, target, margin=50)
    subgraph_nodes_ix = nodes.cx[xmin:xmax, ymin:ymax].index
    print('n_subgraph_nodes:',len(subgraph_nodes_ix))
    print("Getting subgraph")
    init = time.time()
    subgraph = graph.subgraph(subgraph_nodes_ix)
    wait = time.time() - init
    print("subgraph loaded in", wait)
    return subgraph, subgraph_nodes_ix

def get_nearest_nodes(graph, source, target):
    print("Getting nearest nodes")
    init = time.time()
    origin_node = ox.get_nearest_node(G=graph, point=(source[1],source[0]), method='euclidean')
    target_node = ox.get_nearest_node(G=graph, point=(target[1], target[0]), method='euclidean')
    wait = time.time() - init
    print("nereast nodes loaded in", wait)
    return origin_node, target_node

def get_route_data(route, nodes):
    route_nodes = nodes.loc[route]
    print('n_route_nodes:',len(route_nodes))
    route_line = list(zip(route_nodes.lat, route_nodes.lon))
    route_linestr = LineString(route_nodes.geometry.values.tolist())
    route_geom = gpd.GeoDataFrame(crs=nodes.crs)
    route_geom['geometry'] = None
    route_geom['osmids'] = None
    route_geom.loc[0,'geometry'] = route_linestr
    route_geom.loc[0,'osmids'] = str(list(route_nodes['osmid'].values))
    route_geom['length_m'] = route_geom.length
    return route_geom, route_line

def ldict2ltup(d):
    #maps a list of dictionaries to a list of tuples
    return (d['lat'],d['lng'])

def llist2ltup(d):
    #maps a list of coordinate lists to a list of tuples
    return (d[1],d[0])

def get_directions_mapbox(source, target):
    # Mapbox driving direction API call
    source_str = "{},{}".format(source[1],source[0])
    target_str = "{},{}".format(target[1],target[0])
    coords = ";".join([source_str,target_str])
    ROUTE_URL = "https://api.mapbox.com/directions/v5/mapbox/driving/{0}.json?access_token={1}&overview=full&geometries=geojson".format(coords, mapbox_access_token)
    result = requests.get(ROUTE_URL)
    data = result.json()
    route_data = data["routes"][0]["geometry"]["coordinates"]
    return list(map(llist2ltup,route_data))

def get_directions_google(gmaps, origin, destination):
    #reverse_geocode_result = gmaps.reverse_geocode(source)
    dirs = gmaps.directions(origin=origin, destination=destination)
    overview_polyline = dirs[0].get('overview_polyline')
    if overview_polyline is not None:
        #print(overview_polyline)
        route_decoded = googlemaps.convert.decode_polyline(overview_polyline['points'])
    else:
        pass
    #lats = []
	#lons = []

	#for i in directions['routes']:
	#	for j, k in i.items():
	#		if j == 'legs':
	#			for l in k:
	#				for m, n in l.items():
	#					if m == 'steps':
	#						for o in n:
	#							for p, q in o.items():
	#								if p == 'end_location':
	#									for l1, l2 in q.items():
	#										if l1 == 'lat':
	#											lats.append(l2)
	#										if l1 == 'lng':
	#											lons.append(l2)

    #route = []

    #steps = dirs[0]['legs'][0]['steps']
    #for step in steps:
    #    route.append((step['start_location']['lat'], step['start_location']['lng']))
    #    route.append((step['end_location']['lat'], step['end_location']['lng']))
        #if step == dirs[0]['legs'][0]['steps'][-1]:
        #    route.append((step['end_location']['lat'], step['end_location']['lng']))
    return list(map(ldict2ltup,route_decoded))

def get_scattermap_lines(source, target):
    # Filter graph to reduce time
    subgraph, subgraph_nodes_ix = get_subgraph(graph, nodes, source, target)
    # Get nearest nodes in the subgraph
    source_node_id, target_node_id = get_nearest_nodes(graph, source, target)

    print('#'*20,'source and target nodes')
    print(source_node_id)
    print(target_node_id)
    print('#'*30)
    # Get shortest_path (list of nodes)
    print("Getting shortest path")
    init = time.time()
    opt_route = nx.shortest_path(G=graph, source=source_node_id,
                                 target=target_node_id, weight='length')
    wait = time.time() - init
    print("shortest path in", wait)
    print('#'*40)
    print(opt_route)
    print('#'*40)

    # Get route data
    route_df, route_line = get_route_data(opt_route, nodes)

    return route_line

# Functions to update mapscatter component
@app.callback(
              Output('map-graph', 'figure'),
              [Input('map-graph', 'clickData')],
              [State('map-graph', 'relayoutData')])
def _update_routes(clickData, relayoutData):
    if clickData == None:
        print('Initial Map')
        return initial_map
    else:
        print('#'*30)
        print(clickData)
        print('#'*30)
        print(relayoutData)
        print('Click Event')
        source_data = map_data.loc[clickData['points'][0]['pointIndex'],:]
        target_data = map_data.loc[source_data['LINK'],:]
        source = tuple([source_data['C_LAT'], source_data['C_LONG']])
        target = tuple([target_data['C_LAT'], target_data['C_LONG']])
        #source = tuple([source_data.geometry.x, source_data.geometry.y])
        #target = tuple([target_data.geometry.x, target_data.geometry.y])
        #route_line = get_scattermap_lines(source, target)
        #route_line = get_directions(gmaps, source, target)
        route_line = get_directions_mapbox(source, target)
        route_line.insert(0, tuple([source_data['C_LAT'], source_data['C_LONG']]))
        route_line.append(tuple([target_data['C_LAT'], target_data['C_LONG']]))
        new_map = gen_map(map_data, route_line)
        if relayoutData:
            if 'mapbox.center' in relayoutData:
                new_map['layout']['mapbox']['center'] = relayoutData['mapbox.center']
            if 'mapbox.zoom' in relayoutData:
                new_map['layout']['mapbox']['zoom'] = relayoutData['mapbox.zoom']

        return new_map


# Boostrap CSS.
external_css = ["https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
                "//fonts.googleapis.com/css?family=Raleway:400,300,600",
                "//fonts.googleapis.com/css?family=Dosis:Medium",
                "https://cdn.rawgit.com/plotly/dash-app-stylesheets/62f0eb4f1fadbefea64b2404493079bf848974e8/dash-uber-ride-demo.css",
                "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"]

for css in external_css:
    app.css.append_css({"external_url": css})

if __name__ == '__main__':
    app.run_server(debug=True)
