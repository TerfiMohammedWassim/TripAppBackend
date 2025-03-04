from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Backend is running!"

@app.route('/solve', methods=['POST'])
def solve():
    data = request.get_json()
    starting_city = data.get('StartingCity')
    print(data)
    print(starting_city)
    
    if data is None:
        return jsonify({"error": "No JSON data received"}), 400

    return jsonify({"message": "Solution calculated!", "received_data": data})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
