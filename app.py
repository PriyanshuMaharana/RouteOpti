from firebase_admin import credentials, firestore, initialize_app
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
import json
import os
from tqdm import tqdm
import folium
import osmnx as ox
import networkx as nx
from folium import plugins
import polyline
import math
from haversine import haversine
import firebase_admin
import traceback

class FirestoreRouteOptimizer:
    def __init__(self, credentials_path: str):
        """Initialize the FirestoreRouteOptimizer with Firebase credentials."""
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(credentials_path)
                initialize_app(cred)
            self.db = firestore.client()
            self.graph = None
        except Exception as e:
            print(f"Error initializing Firebase: {e}")
            raise

    def parse_coordinates(self, coord: firestore.GeoPoint) -> Tuple[float, float]:
        """Parse a GeoPoint object into (latitude, longitude)."""
        try:
            lat_value = coord.latitude
            lng_value = coord.longitude
            return lat_value, lng_value
        except Exception as e:
            print(f"Error parsing coordinates: {e}")
            raise

    def fetch_deliveries(self, driver_id: str) -> List[Dict]:
        """Fetch pending deliveries for a specific driver."""
        try:
            deliveries_ref = self.db.collection('drivers').document(driver_id).collection('deliveries')
            query = deliveries_ref.where('status', '==', 'pending')\
                                .order_by('priority', direction=firestore.Query.DESCENDING)
            
            deliveries = []
            for doc in query.stream():
                delivery_data = doc.to_dict()
                delivery_data['id'] = doc.id
                delivery_data['location'] = self.parse_coordinates(delivery_data['location'])
                deliveries.append(delivery_data)
            
            return deliveries
        except Exception as e:
            print(f"Error fetching deliveries: {e}")
            raise

    def ant_colony_optimization(self, distances: np.ndarray, n_ants: int = 10, n_iterations: int = 50,
                              decay: float = 0.1, alpha: float = 1, beta: float = 2) -> List[int]:
        """Perform Ant Colony Optimization to find the optimal route."""
        try:
            print("Starting Ant Colony Optimization...")
            n_points = len(distances)
            pheromone = np.ones((n_points, n_points)) / n_points
            best_path = None
            best_path_length = float('inf')
            
            for iteration in tqdm(range(n_iterations), desc="ACO Progress"):
                paths = np.zeros((n_ants, n_points), dtype=int)
                path_lengths = np.zeros(n_ants)
                
                for ant in range(n_ants):
                    current = 0
                    unvisited = list(range(1, n_points))
                    path = [current]
                    path_length = 0
                    
                    while unvisited:
                        if np.random.random() < 0.1:
                            next_point = np.random.choice(unvisited)
                        else:
                            pheromone_values = pheromone[current, unvisited]
                            distance_values = distances[current, unvisited]
                            
                            with np.errstate(divide='ignore'):
                                probabilities = (pheromone_values ** alpha) * ((1.0 / np.maximum(distance_values, 1e-10)) ** beta)
                            
                            probabilities = np.nan_to_num(probabilities)
                            sum_prob = probabilities.sum()
                            if sum_prob == 0:
                                next_point = np.random.choice(unvisited)
                            else:
                                probabilities = probabilities / sum_prob
                                next_point = unvisited[np.random.choice(len(unvisited), p=probabilities)]
                        
                        path_length += distances[current, next_point]
                        current = next_point
                        path.append(current)
                        unvisited.remove(current)
                    
                    path_length += distances[current, 0]
                    paths[ant] = path
                    path_lengths[ant] = path_length
                    
                    if path_length < best_path_length:
                        best_path_length = path_length
                        best_path = path.copy()
            
                pheromone *= (1 - decay)
                for ant in range(n_ants):
                    for i in range(n_points - 1):
                        current = paths[ant, i]
                        next_point = paths[ant, i + 1]
                        pheromone[current, next_point] += 1.0 / path_lengths[ant]
                        pheromone[next_point, current] += 1.0 / path_lengths[ant]
            
            if best_path is None:
                raise ValueError("No valid path found")
                
            print(f"Best path length: {best_path_length:.2f} km")
            return best_path
            
        except Exception as e:
            print(f"Error in ACO algorithm: {e}")
            raise

    def get_shortest_path(self, origin: Tuple[float, float], destination: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Get the shortest path between two points using OSMnx."""
        try:
            if self.graph is None:
                print("Downloading street network data...")
                self.graph = ox.graph_from_point(
                    origin, 
                    dist=10000,  # 10km radius
                    network_type='drive'
                )
                
            orig_node = ox.nearest_nodes(self.graph, origin[1], origin[0])
            dest_node = ox.nearest_nodes(self.graph, destination[1], destination[0])
            
            route = nx.shortest_path(self.graph, orig_node, dest_node, weight='length')
            
            path_coords = []
            for node in route:
                path_coords.append((
                    self.graph.nodes[node]['y'],
                    self.graph.nodes[node]['x']
                ))
            
            return path_coords
        except Exception as e:
            print(f"Error calculating shortest path: {e}")
            return [origin, destination]

    def calculate_route_metrics(self, path_coords: List[Tuple[float, float]]) -> Dict:
        """Calculate metrics for a route segment."""
        try:
            total_distance = 0
            for i in range(len(path_coords) - 1):
                total_distance += haversine(path_coords[i], path_coords[i + 1])
            
            # Estimate time assuming average speed of 30 km/h in urban areas
            estimated_time = (total_distance / 30) * 60  # Convert to minutes
            
            return {
                'distance_km': round(total_distance, 2),
                'estimated_time_mins': round(estimated_time, 2)
            }
        except Exception as e:
            print(f"Error calculating route metrics: {e}")
            raise

    def create_route_map(self, optimal_route: Dict) -> str:
        """Create an interactive map with the optimized delivery route."""
        try:
            start_location = optimal_route['start_location']
            m = folium.Map(
                location=start_location,
                zoom_start=13,
                tiles='OpenStreetMap'
            )

            # Add marker for starting point
            folium.Marker(
                start_location,
                popup='Start Location',
                icon=folium.Icon(color='green', icon='info-sign')
            ).add_to(m)

            route_group = folium.FeatureGroup(name='Delivery Route')
            metrics_group = folium.FeatureGroup(name='Route Metrics')

            # Plot each delivery point and the route between points
            previous_point = start_location
            total_distance = 0
            total_time = 0

            for idx, delivery in enumerate(optimal_route['route_sequence'], 1):
                current_point = delivery['location']
                
                # Get the actual street route between points
                path_coords = self.get_shortest_path(previous_point, current_point)
                
                # Calculate metrics for this segment
                metrics = self.calculate_route_metrics(path_coords)
                total_distance += metrics['distance_km']
                total_time += metrics['estimated_time_mins']

                # Draw the route line
                route_line = folium.PolyLine(
                    locations=path_coords,
                    weight=3,
                    color='blue',
                    opacity=0.8,
                    popup=f"Segment {idx}: {metrics['distance_km']}km, {metrics['estimated_time_mins']}min"
                )
                route_line.add_to(route_group)

                # Add marker for delivery point
                folium.Marker(
                    current_point,
                    popup=f"""
                    <b>Delivery {idx}</b><br>
                    Customer: {delivery['customer_name']}<br>
                    Address: {delivery['address']}<br>
                    Instructions: {delivery['instructions']}<br>
                    Contact: {delivery['contact']}<br>
                    Priority: {delivery.get('priority', 'Normal')}<br>
                    Distance from previous: {metrics['distance_km']}km<br>
                    Estimated time: {metrics['estimated_time_mins']}min
                    """,
                    icon=folium.Icon(
                        color='red' if delivery.get('priority', 0) > 1 else 'orange',
                        icon='info-sign'
                    )
                ).add_to(route_group)

                previous_point = current_point

            # Add total metrics to the map
            folium.Rectangle(
                bounds=[[start_location[0] - 0.02, start_location[1] - 0.02],
                       [start_location[0] - 0.01, start_location[1] + 0.02]],
                color="white",
                fill=True,
                popup=f"""
                <b>Total Route Metrics</b><br>
                Total Distance: {round(total_distance, 2)}km<br>
                Total Time: {round(total_time, 2)}min
                """
            ).add_to(metrics_group)

            route_group.add_to(m)
            metrics_group.add_to(m)
            folium.LayerControl().add_to(m)

            # Save the map
            map_path = 'delivery_route.html'
            m.save(map_path)
            print(f"Map saved to {map_path}")
            return map_path

        except Exception as e:
            print(f"Error creating route map: {e}")
            traceback.print_exc()
            raise

    def calculate_optimal_route(self, driver_id: str) -> Dict:
        """Calculate the optimal delivery route for a driver."""
        try:
            deliveries = self.fetch_deliveries(driver_id)
            if not deliveries:
                print("No pending deliveries found.")
                return {"error": "No pending deliveries found"}
            
            driver_doc = self.db.collection('drivers').document(driver_id).get()
            if not driver_doc.exists:
                raise ValueError(f"Driver {driver_id} not found")
            
            driver_data = driver_doc.to_dict()
            if 'current_location' not in driver_data:
                raise ValueError("Driver location not found")
            
            start_location = self.parse_coordinates(driver_data['current_location'])
            points = [start_location]
            
            delivery_map = {}
            for delivery in deliveries:
                points.append(delivery['location'])
                delivery_map[delivery['location']] = delivery
            
            n_points = len(points)
            distances = np.zeros((n_points, n_points))
            for i in range(n_points):
                for j in range(n_points):
                    distances[i, j] = haversine(points[i], points[j])
            
            route_indices = self.ant_colony_optimization(distances)
            route_sequence = [points[idx] for idx in route_indices]
            
            # Build the result dictionary
            result = {
                "start_location": start_location,
                "route_sequence": [
                    {
                        "delivery_id": delivery_map[point]['id'],
                        "address": delivery_map[point]['address'],
                        "location": point,
                        "customer_name": delivery_map[point]['customer_name'],
                        "instructions": delivery_map[point]['instructions'],
                        "contact": delivery_map[point]['contact'],
                        "priority": delivery_map[point].get('priority', 0)
                    }
                    for point in route_sequence[1:]
                ]
            }
            
            # Create and save the route map
            map_path = self.create_route_map(result)
            result['map_path'] = map_path
            
            return result
        except Exception as e:
            print(f"Error calculating optimal route: {e}")
            traceback.print_exc()
            raise

if __name__ == "__main__":
    # Path to your Firebase credentials JSON file
    FIREBASE_CREDENTIALS = "<Firebase Creds Here>"
    DRIVER_ID = "driver_123"  # Replace with actual driver ID

    try:
        optimizer = FirestoreRouteOptimizer(credentials_path=FIREBASE_CREDENTIALS)

        print("\nCalculating optimal route...")
        optimal_route = optimizer.calculate_optimal_route(driver_id=DRIVER_ID)

        if "error" in optimal_route:
            print(f"Error: {optimal_route['error']}")
        else:
            print("\nOptimal Route Details:")
            print(json.dumps(optimal_route, indent=4))
            print(f"\nRoute map saved to: {optimal_route['map_path']}")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
