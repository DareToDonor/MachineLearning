from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from chatbot import clean_up_sentence, predict_class, get_response, intents

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({"response": "Welcome ML API"})

@app.route('/chat', methods=['POST'])
def chat():
    try :
        data = request.get_json()
        message = data['message']

        if message.lower() == 'exit':
            return jsonify({"response": "Goodbye!"})

        ints = predict_class(message)
        res = get_response(ints, intents)

        return jsonify({
        "status": "success",
        "message": "response succcess",
        "data": res,
        }), 200
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return jsonify({
            "message": "An error occurred",
            "status": "error",
            "error": str(e),
        }), 500  

# Run the Flask app
if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
