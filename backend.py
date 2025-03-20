from flask import Flask, request, jsonify
import random
import numpy as np
from genetic import genetic_algorithm  

app = Flask(__name__)

@app.route('/solve', methods=['POST'])
def solve():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body is required"}), 400
        
        mutation_rate = float(data.get("mutationRate", 0.01))
        generations = int(data.get("generations", 1000))
        population_size = int(data.get("population", 100))
        all_cities = data.get('allCities', [])

        if not all_cities:
            return jsonify({"error": "City list is required"}), 400

        starting_city = random.choice(all_cities)  
        
        best_route, best_distance = genetic_algorithm(all_cities, starting_city, mutation_rate, generations, population_size)
        
        result = {
            "bestPaths": best_route,
            "totalCitiesReached": np.round(1 / best_distance, 2),
            "totalCities": len(all_cities), 
        }
        
        
        return jsonify(result), 200

    except Exception as e:
        print(f'An exception occurred: {str(e)}')
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
