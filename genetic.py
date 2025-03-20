import numpy as np
from collections import defaultdict
import random

def calculate_distance(city1, city2):
    return np.sqrt((city1['x'] - city2['x'])**2 + (city1['y'] - city2['y'])**2)

def genetic_algorithm(all_cities, starting_city, mutation_rate=0.01, generations=100, population_size=50):
    city_dict = {city['name']: city for city in all_cities}
    
    city_names = [city['name'] for city in all_cities if city['name'] != starting_city['name']]
    
    population = []
    for _ in range(population_size):
        route = [starting_city['name']] + random.sample(city_names, len(city_names))
        population.append(route)
    
    def calculate_fitness(route):
        total_distance = 0
        for i in range(len(route) - 1):
            city1 = city_dict[route[i]]
            city2 = city_dict[route[i + 1]]
            total_distance += calculate_distance(city1, city2)
        total_distance += calculate_distance(city_dict[route[-1]], city_dict[route[0]])
        return 1 / total_distance if total_distance > 0 else float('inf')
    
    def selection(population, fitness_scores):
        selected = []
        for _ in range(len(population)):
            tournament = random.sample(range(len(population)), 3)
            best_idx = max(tournament, key=lambda idx: fitness_scores[idx])
            selected.append(population[best_idx])
        return selected
    
    def crossover(parent1, parent2):
        child = [parent1[0]]
        
        start, end = sorted(random.sample(range(1, len(parent1)), 2))
        child_subset = parent1[start:end+1]
        
        remaining_cities = [city for city in parent2[1:] if city not in child_subset]
        child.extend(child_subset + remaining_cities)
        
        return child
    
    def mutate(route, mutation_rate):
        if random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(1, len(route)), 2)
            route[idx1], route[idx2] = route[idx2], route[idx1]
        return route
    
    best_route = None
    best_fitness = 0
    
    for generation in range(generations):
        fitness_scores = [calculate_fitness(route) for route in population]
        
        best_idx = fitness_scores.index(max(fitness_scores))
        if fitness_scores[best_idx] > best_fitness:
            best_fitness = fitness_scores[best_idx]
            best_route = population[best_idx]
        
        if generation % 10 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness}")
        
        selected = selection(population, fitness_scores)
        
        next_generation = []
        for i in range(0, len(selected), 2):
            if i + 1 < len(selected):  
                parent1, parent2 = selected[i], selected[i+1]
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
                next_generation.append(mutate(child1, mutation_rate))
                next_generation.append(mutate(child2, mutation_rate))
        
        population = next_generation[:population_size]
    
    best_distance = 0
    for i in range(len(best_route) - 1):
        city1 = city_dict[best_route[i]]
        city2 = city_dict[best_route[i + 1]]
        best_distance += calculate_distance(city1, city2)
    best_distance += calculate_distance(city_dict[best_route[-1]], city_dict[best_route[0]])
    
    return best_route, best_distance


