from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route('/')
def home():
    return "Backend is running!"

@app.route('/solve', methods=['POST'])
def solve():
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

    return jsonify({"message": "Solution calculated!", "received_data": data})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
