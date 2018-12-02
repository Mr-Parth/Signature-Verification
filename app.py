
'''
|========================================================================================================|
|````````````````````````````````````````````````````````````````````````````````````````````````````````|
|`````````````````````````````````██████╗  █████╗ ████████╗ █████╗ ``````````````````````````````````````|
|`````````````````````````````````██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗``````````````````````````````````````|
|`````````````````````````````````██║``██║███████║```██║```███████║``````````````````````````````````````|
|`````````````````````````````````██║``██║██╔══██║```██║```██╔══██║``````````````````````````````````````|
|`````````````````````````````````██████╔╝██║``██║```██║```██║``██║``````````````````````````````````````|
|`````````````````````````````````╚═════╝ ╚═╝``╚═╝```╚═╝```╚═╝``╚═╝``````````````````````````````````````|
|``````````````````````██████╗ ███████╗███╗```██╗██████╗ ███████╗██████╗ ███████╗````````````````````````|
|``````````````````````██╔══██╗██╔════╝████╗``██║██╔══██╗██╔════╝██╔══██╗██╔════╝````````````````````````|
|``````````````````````██████╔╝█████╗``██╔██╗`██║██║``██║█████╗``██████╔╝███████╗````````````````````````|
|``````````````````````██╔══██╗██╔══╝``██║╚██╗██║██║``██║██╔══╝``██╔══██╗╚════██║````````````````````````|
|``````````````````````██████╔╝███████╗██║ ╚████║██████╔╝███████╗██║``██║███████║````````````````````````|
|``````````````````````╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝``╚═╝╚══════╝````````````````````````|
|========================================================================================================|
|                        Project for Axis Bank ( SIGNATURE RECOGNITION SYSTEM )                          |
|========================================================================================================|

'''
from flask import Flask,render_template,request, redirect, url_for
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import time,csv,os
import pandas as pd
from prediction.preprocess.normalize import preprocess_signature
from prediction.predictor import *
import prediction.signet as signet
from prediction.cnn_model import TF_CNNModel
from prediction.lasagne_to_tf import copy_initializer, transpose_copy_initializer

app = Flask(__name__)

UPLOAD_FOLDER = '/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DEBUG'] = True
socketio = SocketIO(app,binary = True)
model = mod_load(TF_CNNModel,signet)
sess = ses_init()
## creating the PNG file of blob data--------------------------
def createPng(data,name="tmpImage.png"):
    fh = open(name, "wb")
   
    fh.write(data)
    fh.close()

## storing in Database(CSV) -----------------------------------
def writeCSV(data):
    base = [[data['accountNo'],data['customerName'],os.path.join("account_images",str(data['accountNo'])+".png") ]]
    Database = open('Database.csv','a')
    with Database:
        writer = csv.writer(Database)
        writer.writerows(base)
        return True
        
def matchSign(data):
    ## create PNG Image of receiving BLOB Data-------------------
    createPng(data['finalImage'])
    real=pd.read_csv("Database.csv")
    r = list(real[real.account_no == int(data["accountNo"])].signature_image)
    if(len(r) == 0):
        scanResult = {
            'action': 'scan',
            'accountNo' : data['accountNo'],
            'customerName' : data['customerName'],
            'verdict' : "LOST"        
        }
        return scanResult	
    else:
        #Prediction
        dist = euc_distance(sess,list(real[real.account_no == int(data["accountNo"])].signature_image)[0],"tmpImage.png",model,preprocess_signature)
    	## Scan result will generate after compare the image -------------
        scanResult = {
        	'action': 'scan',
        	'accountNo' : data['accountNo'],
        	'customerName' : data['customerName'],
        	'verdict' : "PASS" if dist < 9 else "FAIL",
        	'confidence' : str(dist),
    	}
        return scanResult

def addData(data):
    
    ## response JSON package for showing results in frontend -----
    response = {
            'action': 'add',
            'accountNo' : data['accountNo'],
            'verdict' : 'added',
            'customerName' : data['customerName'] 
        }
    if(writeCSV(data)):
        createPng(data["finalImage"],name=os.path.join("account_images",str(data['accountNo'])+".png"))
        response['performed'] = True
    else:
        response['performed'] = False
    return response

@app.route('/index')
def index():
    return render_template('index.html' )



@app.route('/databenders/asrs/api/start',methods=['POST'])
def uploadImage():
    if request.method =='POST':
        file = request.files['file[]']
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            scanResult = {
                'action': 'scan',
                'accountNo' : '123456789',
                'customerName' : 'test',
                'verdict' : 'PASS',
                'confidence' : 0.755,
            }
    return scanResult


@socketio.on('connect')
def connect():
    print(' CONNECTED ! ')
    socketio.emit('test','server is ready...')

@socketio.on('scan-blob')
def scanBlob(blob):
    if(blob != '' ):
        print('Server is receving Blob...')
        Result = matchSign(blob)
        time.sleep(2)
        socketio.emit('result',Result)
    else:
        print('Blob is empty')

@socketio.on('add-blob')
def addBlob(blob):
    if(blob != '' ):
        print('Server is receving addBlob...')
        time.sleep(2)
        Result = addData(blob)
        socketio.emit('result',Result)  
    else:
        print('addBlob is empty')
    
if __name__ == '__main__':
    socketio.run(app)
