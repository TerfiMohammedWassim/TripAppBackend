from flask import Flask, request, jsonify
import json
import random
import math
import numpy as np

app = Flask(__name__)

def distance(city1, city2):
    dx = city1["x"] - city2["x"]
    dy = city1["y"] - city2["y"]
    return math.sqrt(dx ** 2 + dy ** 2)

def total_distance(route, cities):
    total = sum(distance(cities[route[i]], cities[route[i + 1]]) for i in range(len(route) - 1))
    total += distance(cities[route[-1]], cities[route[0]])
    return total

def selection(population, cities, k=5):
    selected = random.sample(population, k)
    return min(selected, key=lambda route: total_distance(route, cities))

def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(1, size), 2))
    child = [-1] * size
    
    child[0] = parent1[0]
    
    child[start:end] = parent1[start:end]
    
    fill_values = [city for city in parent2 if city not in child]
    index = 0
    for i in range(1, size):
        if child[i] == -1:
            child[i] = fill_values[index]
            index += 1
    
    return child

def mutate(route, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(1, len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

def initial_population(size, num_cities, start_city_index):
    population = []
    
    for _ in range(size):
        other_cities = list(range(num_cities))
        other_cities.remove(start_city_index)
        random.shuffle(other_cities)
        route = [start_city_index] + other_cities
        population.append(route)
        
    return population

def genetic_algorithm(cities, population_size=100, generations=100, mutation_rate=0.01, start_city_index=0):
    print("Starting genetic algorithm...")
    
    population_size = int(population_size)
    generations = int(generations)
    mutation_rate = float(mutation_rate)
    num_cities = len(cities)
    
    population = initial_population(population_size, num_cities, start_city_index)
    
    best_route_overall = None
    best_distance_overall = float('inf')
    
    for generation in range(generations):
        new_population = []
        
        for _ in range(population_size):
            parent1 = selection(population, cities)
            parent2 = selection(population, cities)
            
            child = crossover(parent1, parent2)
            
            child = mutate(child, mutation_rate)
            
            new_population.append(child)
        
        population = new_population
        
        current_best_route = min(population, key=lambda route: total_distance(route, cities))
        current_best_distance = total_distance(current_best_route, cities)
        
        if current_best_distance < best_distance_overall:
            best_distance_overall = current_best_distance
            best_route_overall = current_best_route
        
        if generation % 10 == 0:
            print(f"Generation {generation}: Best distance = {current_best_distance:.2f}")
    
    best_route_names = [cities[i]["name"] for i in best_route_overall]
    
    print("✅ Algorithm completed!")
    print(f"🔝 Best route found: {best_route_names}")
    print(f"📏 Total distance: {best_distance_overall:.2f}")
    
    return best_route_names, best_distance_overall

@app.route('/solve', methods=['POST'])
def solve():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "No JSON data received"}), 400

        starting_city = data.get('startingCity')
        mutation_rate = float(data.get('mutationRate', 0.01))
        generations = int(data.get('generations', 100))
        population_size = int(data.get('populationSize', 100))
        all_cities = data.get('allCities', [])
        
        # Validate input data
        if not starting_city:
            return jsonify({"error": "Starting city is required"}), 400
        if not all_cities:
            return jsonify({"error": "City list is required"}), 400
            
        cities = {}
        for i, city in enumerate(all_cities):
            cities[i] = {
                "name": city["name"],
                "x": city["x"],
                "y": city["y"]
            }
        
       
        start_city_index = None
        for i, city in cities.items():
            if city["name"].strip().lower() == starting_city["name"].strip().lower():
                start_city_index = i
                break
        
        if start_city_index is None:
            return jsonify({"error": f"Starting city '{starting_city['name']}' not found in city list"}), 400
        
        
        best_route_names, best_distance = genetic_algorithm(
            cities=cities, 
            population_size=population_size, 
            generations=generations, 
            mutation_rate=mutation_rate, 
            start_city_index=start_city_index
        )
        
       
        route_coordinates = []
        for city_name in best_route_names:
            city_index = next(i for i, city in cities.items() if city["name"] == city_name)
            route_coordinates.append({
                "name": city_name,
                "x": cities[city_index]["x"],
                "y": cities[city_index]["y"],
                "count" : cities[city_index]["count"]
            })
        
        return jsonify({
            "bestRoute": best_route_names,
            "totalDistance": round(best_distance, 2),
            "routeCoordinates": route_coordinates
        })

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)