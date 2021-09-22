from flask import Flask, request, render_template, jsonify, flash, redirect
from werkzeug.utils import secure_filename
import os
from app.torch_utils import *

app = Flask(__name__)

MODEL = load_model("app/proto_mobilenetv3_13class.pth")
UPLOAD_FOLDER = "app/static"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
INDEX_LABEL = {0: 'Clean Water Network',
                1: 'Communication Network',
                2: 'Flood',
                3: 'Garbage',
                4: 'Gutter Cover',
                5: 'Illegal Parking',
                6: 'Layout and Building',
                7: 'Park',
                8: 'Road',
                9: 'Sidewalk',
                10: 'Tree',
                11: 'Vandalism',
                12: 'Waterway'}

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
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_location = os.path.join(app.config['UPLOAD_FOLDER'], "image.PNG")
            file.save(image_location)
            sim_all, sim_max, predict_class = predict_one_img(MODEL, file)
            return render_template("index.html", cl_1 = sim_all[0], cl_2 = sim_all[1], cl_3 = sim_all[2], 
                                    cl_4 = sim_all[3], cl_5 = sim_all[4], cl_6 = sim_all[5], cl_7 = sim_all[6], 
                                    cl_8 = sim_all[7], cl_9 = sim_all[8], cl_10 = sim_all[9], cl_11 = sim_all[10],
                                    cl_12 = sim_all[11], cl_13 = sim_all[12],
                                    prediction=predict_class, 
                                    similar=sim_max, 
                                    image_loc=image_location)
    return render_template("index.html",cl_1 = 0, cl_2 = 0, cl_3 = 0, 
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
        if image_file and allowed_file(image_file.filename):
            image_byte = image_file.read() 
            sim_all, sim_max, predict_class = predict_one_img(MODEL, image_file)
            return jsonify({'class': list(INDEX_LABEL.values()),
                            'similarity_score' : sim_all,
                            'top_predict': predict_class,
                            'top_similarity':sim_max}
                            )
    return jsonify({'class': list(INDEX_LABEL.values()),
                            'similarity_score' : 0,
                            'top_predict': 0,
                            'top_similarity':0}
                            )