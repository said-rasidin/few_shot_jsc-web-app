from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
from app.torch_utils import *

app = Flask(__name__)

MODEL = load_model("app/proto_mobilenetv3_13class.pth")
UPLOAD_FOLDER = "app/static"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET','POST'])
def upload_predict():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_location = os.path.join(app.config['UPLOAD_FOLDER'], "image.PNG")
            file.save(image_location)
            sim_all, sim_max, predict_class = predict_one_img(MODEL, file)
            return render_template("home.html", cl_1 = sim_all[0], cl_2 = sim_all[1], cl_3 = sim_all[2], 
                                    cl_4 = sim_all[3], cl_5 = sim_all[4], cl_6 = sim_all[5], cl_7 = sim_all[6], 
                                    cl_8 = sim_all[7], cl_9 = sim_all[8], cl_10 = sim_all[9], cl_11 = sim_all[10],
                                    cl_12 = sim_all[11], cl_13 = sim_all[12],
                                    prediction=predict_class, 
                                    similar=sim_max, 
                                    image_loc=image_location)
    return render_template("home.html",cl_1 = 0, cl_2 = 0, cl_3 = 0, 
                                    cl_4 = 0, cl_5 = 0, cl_6 = 0, cl_7 = 0, 
                                    cl_8 = 0, cl_9 = 0, cl_10 = 0, cl_11 = 0,
                                    cl_12 = 0, cl_13 = 0,
                                    prediction=None, 
                                    similar=0, 
                                    image_loc=None)

@app.route('/visual-service', methods=['POST'])
def predict_API():
    if request.method == 'POST':
        image_file = request.files.get('file')
        if image_file:
            image_byte = image_file.read() 
            sim_all, sim_max, predict_class = predict_one_img(MODEL, image_file)
            return jsonify({'Top_Predict':
                            {'Prediction': predict_class, 'Similarity_score':sim_max},
                            'All_Predict': 
                            {'Clean Water Network':sim_all[0],
                            'Communication Network': sim_all[1],
                            'Flood':sim_all[2],
                            'Garbage':sim_all[3],
                            'Gutter Cover':sim_all[4],
                            'Illegal Parking':sim_all[5],
                            'Layout and Building':sim_all[6],
                            'Park':sim_all[7],
                            'Road':sim_all[8],
                            'Sidewalk':sim_all[9],
                            'Tree':sim_all[10],
                            'Vandalism':sim_all[11],
                            'Waterway':sim_all[12]}}
                            )
    return jsonify({'Top_Predict':
                    {'Prediction': 'None', 'Similarity_score':0},
                    'All_Predict': 
                    {'Clean Water Network':0,
                    'Communication Network':0,
                    'Flood':0,
                    'Garbage':0,
                    'Gutter Cover':0,
                    'Illegal Parking':0,
                    'Layout and Building':0,
                    'Park':0,
                    'Road':0,
                    'Sidewalk':0,
                    'Tree':0,
                    'Vandalism':0,
                    'Waterway':0}}
                    )