import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory , send_file
from werkzeug import secure_filename
from flask import jsonify
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from PIL import Image
import PIL.ImageOps
from tensorflow.keras.applications.vgg16 import VGG16 , preprocess_input
from tensorflow.keras import Model
from tensorflow.keras.layers import Input,Lambda,Flatten,Dot
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.python.keras.backend import set_session


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['DATABASE'] = 'database/'
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])



def load_data_directory():
    img_list=pd.read_csv('name_list.csv') 
    return img_list , np.load('img_feature_f.npy'),np.load('img_feature_b.npy')
    
def load_model():
    vgg16=VGG16(weights='imagenet',include_top=False) #no fc layer , output shape = (7,7,512)
    flat=Flatten()(vgg16.output)
    l2_norm=Lambda(lambda x: K.l2_normalize(x,axis=1))(flat)
    model=Model(vgg16.input , l2_norm)
    return model

def resize_aspect_ratio(img,size): 
    w,h=img.size #col , row
    target_w , target_h = size
    ratio=min(target_w/w , target_h/h)
    new_img=img.resize( (int(w*ratio),int(h*ratio)) )
    return new_img

def padding(img,size,color=(255,255,255)):
    w,h=img.size #col , row
    target_w , target_h = size
    new_img=PIL.ImageOps.expand(img,((target_w-w)//2,(target_h-h)//2),color).resize(size)
    return new_img

#load img , resize as ratio , pad
def imgs_preprocess_sect(img_names,resize_size=(200,200),target_size=(224,224),directory='database/'):
    #param img_names: list of img name
    #output: np array for img (batch,row,col,channel)
    output=[]
    for name in img_names:
        img=Image.open(directory+name)
        img=resize_aspect_ratio(img,resize_size)
        img=padding(img,target_size)
        output.append(np.array(img))
    return np.array(output)

def searching_display(top_20,imgs,img_name):
    row=5+1
    col=8
    fig=plt.figure(figsize=(4*col,4*row))
    img_f,img_b=imgs
    fig.add_subplot(row,col,1)
    plt.gca().set_title('{}'.format(img_name[0]))
    plt.imshow(img_f)
    fig.add_subplot(row,col,2)
    plt.gca().set_title('{}'.format(img_name[1]))
    plt.imshow(img_b)
    
    for i,index in enumerate(top_20):
        tmp_f=db_list['front'].iloc[index]
        tmp_b=db_list['back'].iloc[index]
        img_f=np.squeeze(imgs_preprocess_sect([tmp_f]))
        img_b=np.squeeze(imgs_preprocess_sect([tmp_b]))
        fig.add_subplot(row,col,i*2+1+8)
        plt.gca().set_title('{}:{}'.format(i,tmp_f))
        plt.imshow(img_f)
        fig.add_subplot(row,col,i*2+2+8)
        plt.gca().set_title('{}:{}'.format(i,tmp_b))
        plt.imshow(img_b)
    return fig

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    file2 = request.files['file2']
    
    if file and allowed_file(file.filename) and file2 and allowed_file(file2.filename):
        filename = secure_filename(file.filename)
        filename2 = secure_filename(file2.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
        return redirect(url_for('uploaded_file', filename=filename ,filename2=filename2))

@app.route('/uploads/<filename>/<filename2>')
def uploaded_file(filename,filename2):

    model=load_model()
    img_f=Image.open("uploads/"+filename)
    img_f=resize_aspect_ratio(img_f,(200,200))
    img_f=padding(img_f,(224,224))
    img_f_in=preprocess_input(np.expand_dims(np.array(img_f),axis=0)) 
    query_feature_front = model.predict(img_f_in)[0]      
    sim_front=query_feature_front.dot(feature_f.T)
    
    img_b=Image.open("uploads/"+filename2)
    img_b=resize_aspect_ratio(img_b,(200,200))
    img_b=padding(img_b,(224,224))
    img_b_in=preprocess_input(np.expand_dims(np.array(img_b),axis=0))
    query_feature_back=model.predict(img_b_in)[0]
    sim_back=query_feature_back.dot(feature_b.T)
    top20=np.argsort(sim_front+sim_back)[-20:][::-1]
    
    fig=searching_display(top20,[img_f,img_b],[filename,filename2])
    fig.savefig('output/{}.jpg'.format(filename))    
    return send_file("output/{}.jpg".format(filename), mimetype='image/jpg')

if __name__ == '__main__':
    db_list , feature_f , feature_b =load_data_directory()
    #model=load_model()
    app.debug=True
    app.run(host="0.0.0.0",port=int("8080"))

