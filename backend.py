from flask import Flask, request, jsonify
import json
import random
import math
import numpy as np
from collections import defaultdict

app = Flask(__name__)

class GeneticAlgorithm:
    def __init__(self, cities, starting_city, mutation_rate=0.01, population_size=100, generations=1000):
        self.cities = cities
        self.starting_city = starting_city
        self.mutation_rate = float(mutation_rate)
        self.population_size = int(population_size)
        self.generations = int(generations)
        
        
        self.city_names = set([city['name'] for city in cities])
        self.city_names.add(starting_city['name'])
        
        self.city_dict = {city['name']: city for city in cities}
        if starting_city['name'] not in self.city_dict:
            self.city_dict[starting_city['name']] = starting_city
    
    def get_city_by_name(self, name):
        if name == self.starting_city['name']:
            return self.starting_city
        return self.city_dict.get(name)
    
    def create_individual(self):
        individual = {}
        for target_city in self.city_names:
            if target_city == self.starting_city['name']:
                continue
                
            path = [self.starting_city['name']]
            current_city = self.starting_city
            
            visited = {self.starting_city['name']}
            
            reached_target = False
            
            for _ in range(len(self.city_names) * 2): 
                if not current_city or 'goingTo' not in current_city or not current_city['goingTo']:
                    break
                    
                options = current_city['goingTo']
                if not options:
                    break
                    
                
                valid_options = [opt for opt in options if opt['name'] not in visited]
                if not valid_options:
                    break
                
                next_city_data = random.choice(valid_options)
                next_city_name = next_city_data['name']
                path.append(next_city_name)
                visited.add(next_city_name)

                if next_city_name == target_city:
                    reached_target = True
                    break
                    
                current_city = self.get_city_by_name(next_city_name)
            
            if reached_target:
                individual[target_city] = path
        
        return individual
    
    def create_initial_population(self):
        population = []
        for _ in range(self.population_size):
            individual = self.create_individual()
            population.append(individual)
        return population
    
    def calculate_path_cost(self, path):
        if not path or len(path) <= 1:
            return float('inf')
            
        total_cost = 0
        for i in range(len(path) - 1):
            current_city_name = path[i]
            next_city_name = path[i + 1]
            
            current_city = self.get_city_by_name(current_city_name)
            
            if not current_city or 'goingTo' not in current_city:
                return float('inf')
                
            connection = next((conn for conn in current_city['goingTo'] if conn['name'] == next_city_name), None)
            
            if not connection:
                return float('inf')
                
            total_cost += connection['cost']
        
        return total_cost
    
    def calculate_fitness(self, individual):
        if not individual:
            return 0
            
        fitness = 0
        
        num_cities_reached = len(individual)
        fitness += num_cities_reached * 10  
        
        total_inverse_cost = 0
        for target, path in individual.items():
            path_cost = self.calculate_path_cost(path)
            if path_cost < float('inf'):
                total_inverse_cost += 100.0 / (path_cost + 1) 
        
        fitness += total_inverse_cost
        
        return fitness
    
    def select_parent(self, population, fitnesses):
        tournament_size = 3
        tournament = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament]
        winner_idx = tournament[tournament_fitnesses.index(max(tournament_fitnesses))]
        return population[winner_idx]
    
    def crossover(self, parent1, parent2):
        child = {}
        
        all_targets = set(parent1.keys()) | set(parent2.keys())
        
        for target in all_targets:
            if target in parent1 and target in parent2:
                cost1 = self.calculate_path_cost(parent1[target])
                cost2 = self.calculate_path_cost(parent2[target])
                
                if cost1 <= cost2:
                    child[target] = parent1[target].copy()
                else:
                    child[target] = parent2[target].copy()
            elif target in parent1:
                child[target] = parent1[target].copy()
            elif target in parent2:
                child[target] = parent2[target].copy()
        
        return child
    
    def mutate(self, individual):
        if random.random() > self.mutation_rate or not individual:
            return individual
            
        mutated = {}
        for target, path in individual.items():
            if random.random() < self.mutation_rate and len(path) > 2:
                mutation_point = random.randint(0, len(path) - 2)
                
                new_path = path[:mutation_point + 1]
                current_city_name = new_path[-1]
                current_city = self.get_city_by_name(current_city_name)
                
                visited = set(new_path)
                reached_target = False
                
                for _ in range(len(self.city_names) * 2):  
                    if not current_city or 'goingTo' not in current_city or not current_city['goingTo']:
                        break
                        
                    options = current_city['goingTo']
                    if not options:
                        break
                        
                    valid_options = [opt for opt in options if opt['name'] not in visited]
                    if not valid_options:
                        break
                    
                    next_city_data = random.choice(valid_options)
                    next_city_name = next_city_data['name']
                    new_path.append(next_city_name)
                    visited.add(next_city_name)
                    
                    if next_city_name == target:
                        reached_target = True
                        break
                        
                    current_city = self.get_city_by_name(next_city_name)
                
              
                if reached_target:
                    mutated[target] = new_path
                else:
                    mutated[target] = path.copy()
            else:
                mutated[target] = path.copy()
        
        return mutated
    
    def evolve(self):
        
        population = self.create_initial_population()
        best_individual = None
        best_fitness = -1
        
        for generation in range(self.generations):
            # Calculate fitness for each individual
            fitnesses = [self.calculate_fitness(individual) for individual in population]
            
            # Find the best individual
            max_fitness_idx = fitnesses.index(max(fitnesses))
            current_best = population[max_fitness_idx]
            current_best_fitness = fitnesses[max_fitness_idx]
            
            # Update overall best if needed
            if current_best_fitness > best_fitness:
                best_individual = current_best
                best_fitness = current_best_fitness
            
            # Create the next generation
            new_population = []
            
            # Keep the best individual (elitism)
            new_population.append(current_best)
            
            # Fill the rest of the population through selection, crossover, and mutation
            while len(new_population) < self.population_size:
                parent1 = self.select_parent(population, fitnesses)
                parent2 = self.select_parent(population, fitnesses)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
        
        # Format results with costs for each path
        result = {}
        if best_individual:
            for target, path in best_individual.items():
                cost = self.calculate_path_cost(path)
                if cost < float('inf'):
                    result[target] = {
                        "path": path,
                        "cost": cost,
                        "pathLength": len(path)
                    }
        
        return result

@app.route('/solve', methods=['POST'])
def solve():
    try:
        data = request.get_json()
        
        # Extract parameters
        mutation_rate = data.get("mutationRate", "0.01")
        generations = data.get("generations", "1000")
        population_size = data.get("population", "100")
        starting_city = data.get("startingCity")
        all_cities = data.get('allCities', [])
        
        # Validate input
        if not starting_city:
            return jsonify({"error": "Starting city is required"}), 400
        if not all_cities:
            return jsonify({"error": "City list is required"}), 400
        
        # Initialize and run genetic algorithm
        ga = GeneticAlgorithm(
            cities=all_cities,
            starting_city=starting_city,
            mutation_rate=mutation_rate,
            population_size=population_size,
            generations=generations
        )
        
        best_paths = ga.evolve()
        
        # Prepare the result
        result = {
            "bestPaths": best_paths,
            "totalCitiesReached": len(best_paths),
            "totalCities": len(ga.city_names) - 1  # Subtract 1 to exclude starting city
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        print(f'An exception occurred: {str(e)}')
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)