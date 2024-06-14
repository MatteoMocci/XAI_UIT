from flask import Flask, request, jsonify
from flask_cors import CORS
from new_text_explain import text_explain_script
from timeseries_explain import timeseries_explain_script
from image_explain import image_explain_script
from flask_socketio import SocketIO

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")



@app.route('/explain_text', methods=['POST'])
def explain_text():
    text_explain_script(request.form['text'], socketio, model_name=request.form['selected_model'])
    return '0'

@app.route('/explain_image', methods=['POST'])
def explain_picture():
    user_data = request.files['file']  # Assuming the user data is sent as JSON

    image_explain_script(user_data, socketio, model_name=request.form['selected_model'])
    return '0'

@app.route('/explain_time-series', methods=['POST'])
def explain_timeseries():
    user_data = request.files['file']  # Assuming the user data is sent as JSON

    timeseries_explain_script(user_data, socketio, model_name=request.form['selected_model'])
    return '0'


if __name__ == '__main__':
    socketio.run(app)
