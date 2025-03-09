from flask import Flask, request, jsonify
import json
<<<<<<< HEAD
import random
import math
import matplotlib.pyplot as plt
import numpy as np

=======
>>>>>>> 5613296fbe95b9f2bd9a8f223c824987ce93a065

app = Flask(__name__)

# Fonction pour calculer la distance entre deux villes 
def distance(city1, city2):
    dx = city1["x"] - city2["x"]
    dy = city1["y"] - city2["y"]
    return math.sqrt(dx ** 2 + dy ** 2)

# Fonction pour calculer la distance totale d’un chemin
def total_distance(route, cities):
    return sum(distance(cities[route[i]], cities[route[i + 1]]) for i in range(len(route) - 1))

# Sélectionner une route optimale par tournoi
def selection(population, cities, k=5):
    selected = random.sample(population, k)
    return min(selected, key=lambda route: total_distance(route, cities))

# Crossover (croisement entre deux parents)
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

# Mutation : échanger deux villes au hasard
def mutate(route, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

# Générer une population initiale
def initial_population(size, cities, start_city_index):
    num_cities = len(cities)
    population = []
    
    for _ in range(size):
            other_cities = list(range(num_cities))  
            other_cities.remove(start_city_index)  
            random.shuffle(other_cities)  
            route = [start_city_index] + other_cities  
            population.append(route)  
    return population




# Algorithme génétique principal
def genetic_algorithm(cities, population_size, generations, mutation_rate, start_city):
    print("l'algorithme génétique...")
    # Vérification et conversion des types
    population_size = int(population_size)
    generations = int(generations)
    mutation_rate = float(mutation_rate)

    population = initial_population(population_size, cities, start_city)
    for _ in range(generations):
        new_population = [] 
        for _ in range(population_size):
            parent1 = selection(population, cities)
            parent2 = selection(population, cities)
            
            child = crossover(parent1, parent2)
            
            child = mutate(child, mutation_rate)
            
            new_population.append(child)
            
        population = sorted(new_population, key=lambda route: total_distance(route, cities))[:population_size]
        
    
    best_route = min(population, key=lambda route: total_distance(route, cities))
    best_route_names = [cities[i]["name"] for i in best_route] # Convertir indices → noms
    

    best_distance = total_distance(best_route, cities)

    print("✅ Algorithme terminé !")
    print(f"🔝 Meilleur chemin trouvé : {best_route_names}")
    print(f"📏 Distance totale : {best_distance}")
    return best_route_names, total_distance

# Route Flask pour résoudre le problème
@app.route('/solve', methods=['POST'])
def solve():
<<<<<<< HEAD
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "No JSON data received"}), 400

        starting_city = data.get('startingCity')
        mutation_rate = data.get('mutationRate')
        generations = data.get('generations')
        all_cities = data.get('allCities', [])

        cities = {
            i: {
            "name": city["name"],  
            "x": city["x"],        
            "y": city["y"]         
            }
            for i, city in enumerate(all_cities)
        }
        start_city_index = None  

        for i, city in cities.items(): 
            if city["name"].strip().lower() == starting_city["name"].strip().lower():
                    start_city_index = i  
       
        
        # Exécuter l’algorithme génétique
        best_route, best_distance = genetic_algorithm(
            cities, population_size=100, generations=generations, 
            mutation_rate=mutation_rate, start_city=start_city_index
        )
        # Convertir les indices en noms de villes
        best_route_names = [cities[i]["name"] for i in best_route]

        return jsonify({
            "bestRoute": best_route_names,
            "totalDistance": best_distance
        })



    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
=======
    data = request.get_json()
    print(data)
    starting_city = data.get('startingCity')
    mutation_rate = data.get('mutationRate')
    generation = data.get('generations')
    target_city = data.get('targetCity')
    allCities = data.get('allCities')
    
    print(target_city)
    print(mutation_rate)
    print(generation)
    print(starting_city)
    print(allCities)
    if data is None:
        return jsonify({"error": "No JSON data received"}), 400
>>>>>>> 5613296fbe95b9f2bd9a8f223c824987ce93a065

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
