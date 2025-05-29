from flask import Flask, request, jsonify, render_template
from utils import get_response, predict_class

app = Flask(__name__, template_folder='template')
@app.route('/')
def home():
    return render_template('index.html')

@app.route(rule='/handle_message', methods=['POST'])
def handle_message():
    user_message = request.json['message']
    intents_list = predict_class(user_message)
    response = get_response(intents_list)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')