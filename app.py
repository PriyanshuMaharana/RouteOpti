from flask import Flask, render_template, send_file, jsonify
import osmnx as ox
import networkx as nx
import folium
import random
from itertools import permutations
from shapely.geometry import Polygon, Point
import geopandas as gpd

# Flask app initialization
app = Flask(__name__)

# VRP Solver class (as defined in the previous code)
class VRPSolver:
    def __init__(self, graph, n_ants=20, evaporation_rate=0.1, alpha=1, beta=2):
        self.graph = graph
        self.n_ants = n_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.pheromone = {}
        self.shortest_paths = {}
        self.shortest_distances = {}

    def precompute_distances(self, stops: List[int]):
        """Precompute shortest paths between all stops"""
        for start, end in permutations(stops, 2):
            try:
                path = nx.shortest_path(self.graph, start, end, weight='length')
                distance = nx.shortest_path_length(self.graph, start, end, weight='length')
                self.shortest_paths[(start, end)] = path
                self.shortest_distances[(start, end)] = distance
                self.pheromone[(start, end)] = 1.0
            except nx.NetworkXNoPath:
                continue

    def solve_vrp(self, depot: int, stops: List[int], max_iterations=100) -> Tuple[List[int], float]:
        all_points = [depot] + stops
        self.precompute_distances(all_points)

        best_route = None
        best_distance = float('inf')

        for iteration in range(max_iterations):
            # Generate routes for each ant
            for _ in range(self.n_ants):
                route = self.construct_route(depot, stops)
                if route:
                    distance = self.calculate_total_distance(route)
                    if distance < best_distance:
                        best_distance = distance
                        best_route = route

            # Update pheromones
            self.update_pheromones(best_route, best_distance)

        return best_route, best_distance

    def construct_route(self, depot: int, stops: List[int]) -> List[int]:
        route = [depot]
        remaining_stops = stops.copy()

        while remaining_stops:
            current = route[-1]
            next_stop = self.select_next_stop(current, remaining_stops)
            if next_stop is None:
                break
            route.append(next_stop)
            remaining_stops.remove(next_stop)

        route.append(depot)  # Return to depot
        return route

    def select_next_stop(self, current: int, remaining_stops: List[int]) -> int:
        if not remaining_stops:
            return None

        probabilities = []
        for stop in remaining_stops:
            if (current, stop) not in self.shortest_distances:
                continue

            distance = self.shortest_distances[(current, stop)]
            pheromone = self.pheromone.get((current, stop), 0.1)

            probability = (pheromone ** self.alpha) * ((1.0 / distance) ** self.beta)
            probabilities.append((stop, probability))

        if not probabilities:
            return None

        total = sum(p[1] for p in probabilities)
        if total == 0:
            return random.choice(remaining_stops)

        r = random.random() * total
        current_sum = 0
        for stop, prob in probabilities:
            current_sum += prob
            if current_sum >= r:
                return stop

        return probabilities[-1][0]

    def calculate_total_distance(self, route: List[int]) -> float:
        total_distance = 0
        for i in range(len(route) - 1):
            if (route[i], route[i+1]) not in self.shortest_distances:
                return float('inf')
            total_distance += self.shortest_distances[(route[i], route[i+1])]
        return total_distance

    def update_pheromones(self, best_route: List[int], best_distance: float):
        # Evaporation
        for start, end in self.pheromone:
            self.pheromone[(start, end)] *= (1 - self.evaporation_rate)

        # Add new pheromones
        if best_route:
            deposit = 1.0 / best_distance
            for i in range(len(best_route) - 1):
                self.pheromone[(best_route[i], best_route[i+1])] = deposit

    def get_detailed_route(self, route: List[int]) -> List[int]:
        """Convert stop-to-stop route into detailed node path"""
        detailed_route = []
        for i in range(len(route) - 1):
            path = self.shortest_paths.get((route[i], route[i+1]), [])
            detailed_route.extend(path[:-1])  # Avoid duplicating connecting nodes
        detailed_route.append(route[-1])
        return detailed_route

# Isochrone calculation and visualization
def generate_isochrone_map(graph, center_point, max_distance_km=100):
    """Generate and return a folium map with the isochrone area highlighted"""
    # Convert max_distance_km to meters
    max_distance_m = max_distance_km * 1000
    # Use OSMnx to generate an isochrone from the center point (depot)
    isochrone = ox.distance.great_circle_vec(center_point[0], center_point[1], max_distance_m)
    
    # Create a polygon around the isochrone to highlight non-accessible areas
    polygon = Polygon(isochrone)
    
    # Create a folium map centered on the depot
    m = folium.Map(location=center_point, zoom_start=5)  # Zoom level adjusted for India

    # Highlight accessible area
    folium.GeoJson(polygon).add_to(m)
    
    return m

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/optimize-route", methods=["GET"])
def optimize_route():
    try:
        location = "India"  # Fetch OSM data for India
        graph = ox.graph_from_place(location, network_type='drive')

        largest_component = max(nx.strongly_connected_components(graph), key=len)
        graph = graph.subgraph(largest_component).copy()

        nodes = list(graph.nodes())
        depot = random.choice(nodes)
        stops = random.sample([n for n in nodes if n != depot], 10)

        solver = VRPSolver(graph)
        route, total_distance = solver.solve_vrp(depot, stops)

        if route:
            detailed_route = solver.get_detailed_route(route)
            map_viz = visualize_route(graph, detailed_route, stops, depot)

            # Generate the isochrone map (non-accessible areas)
            depot_coords = (graph.nodes[depot]['y'], graph.nodes[depot]['x'])
            isochrone_map = generate_isochrone_map(graph, depot_coords)

            output_file = "optimized_delivery_route.html"
            isochrone_map.save(output_file)

            return jsonify({
                "message": "Route optimized successfully!",
                "route_order": route,
                "total_distance_km": total_distance / 1000,
                "html_file": output_file
            })
        else:
            return jsonify({"message": "No valid route found", "status": "error"}), 500
    except Exception as e:
        return jsonify({"message": str(e), "status": "error"}), 500

@app.route("/show-map")
def show_map():
    file_path = "optimized_delivery_route.html"
    if os.path.exists(file_path):
        return send_file(file_path)
    else:
        return "Map not generated yet. Optimize the route first by visiting /optimize-route", 404

if __name__ == "__main__":
    app.run(debug=True)

