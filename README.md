# RouteOpti

The RouteOpti is a Python-based application that helps delivery drivers calculate the optimal route for completing deliveries. The solution leverages Firebase for data storage and retrieval, Ant Colony Optimization (ACO) for optimizing the delivery route, and OSMnx for fetching the shortest paths using OpenStreetMap (OSM) data.
The tool calculates an optimized route that minimizes travel distance and time, factoring in priority levels, traffic data (if available), and other factors. It generates an interactive map for visualization and provides useful metrics such as total distance and estimated time.

```python
pip install firebase-admin numpy osmnx networkx folium haversine tqdm
```
### Setup

    - Firebase Credentials: Obtain your Firebase credentials in JSON format. You can generate these credentials from the Firebase console and store the JSON file securely.

    - Configure Firebase Credentials: Replace the placeholder <Firebase Creds Here> in the script with the path to your Firebase credentials JSON file.

 ### How It Works

    - Initialization : The FirestoreRouteOptimizer class initializes Firebase with the provided credentials and connects to the Firestore database.

    - Fetching Deliveries: The fetch_deliveries() method fetches all pending deliveries for a specified driver, sorted by priority.

    - Ant Colony Optimization (ACO): The ant_colony_optimization() method performs optimization on the delivery points using Ant Colony Optimization to find the most efficient route.

    - Shortest Path Calculation: The get_shortest_path() method calculates the shortest path between two geographical points using OSMnx and OpenStreetMap data.

    - Route Metrics: The calculate_route_metrics() method calculates the total distance and estimated time for each segment of the route.

    - Route Map Generation: The create_route_map() method generates an interactive map that shows the optimized route and displays markers for the start point and each delivery point.

    - Optimal Route Calculation: The calculate_optimal_route() method integrates the aforementioned methods to calculate the optimal route for a driver. The result includes the route sequence, start location, and the saved map path

## Usage

    - Set the FIREBASE_CREDENTIALS and DRIVER_ID in the main script to your respective Firebase credentials and driver ID.

- Run the script:
    
    ```python
    python firestore_route_optimizer.py
    ```


    This will calculate the optimal route for the given driver and print the details of the optimized route, including the map path.

    The optimized route map will be saved as delivery_route.html. You can open this file in a web browser to view the interactive map.

## Features

    - Firebase Integration: Data is fetched directly from Firebase Firestore.
    - Ant Colony Optimization: Optimizes the delivery route for the driver     using Ant Colony Optimization (ACO).
    - Shortest Path Calculation: Uses OSMnx and OpenStreetMap for calculating the shortest paths between locations.
    - Interactive Map: Generates a detailed map that displays the optimized route.
    - Route Metrics: Provides total distance and estimated travel time for the entire route and each segment.

Error Handling

The application includes basic error handling with informative messages for common issues such as:

    - Firebase initialization errors
    - Missing or incorrect delivery data
    - Calculation issues during Ant Colony Optimization or shortest path retrieval
    - Invalid driver ID or location in Firestore

## License

This project is licensed under the MIT License