import os
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
#from flask import jsonify
#from sklearn.externals import joblib
import pandas as pd
import numpy as np
from PIL import Image
import PIL.ImageOps
from tensorflow.keras.applications.vgg16 import VGG16 , preprocess_input
from tensorflow.keras import Model
from tensorflow.keras.layers import Lambda,Flatten
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.python.keras.backend import set_session
import redis,base64,sys,uuid,json,time
from threading import Thread


##redis param:
IMAGE_WIDTH=224
IMAGE_HEIGHT=224
IMAGE_CHANS=3
IMAGE_DTYPE="float32"
IMAGE_QUEUE="image_queue"
BATCH_SIZE=1
SERVER_SLEEP=0.25
CLIENT_SLEEP=0.25



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['DATABASE'] = 'database/'
app.config['ALLOWED_EXTENSIONS'] = set(['JPG','JPEG', 'png', 'jpg', 'jpeg', 'gif'])

db=redis.StrictRedis(host='localhost' , port=6379 ,db=0)
#model=None #global initial Var. model

###redis encode , decode function
def base64_encode_image(a):
	# base64 encode the input NumPy array
	return base64.b64encode(a).decode("utf-8")
 
def base64_decode_image(a, dtype, shape):
	# if this is Python 3, we need the extra step of encoding the
	# serialized NumPy string as a byte object
	if sys.version_info.major == 3:
		a = bytes(a, encoding="utf-8")
	# convert the string to a NumPy array using the supplied data
	# type and target shape
	a = np.frombuffer(base64.decodebytes(a), dtype=dtype)
	a = a.reshape(shape)
	# return the decoded image
	return a

###deep learning function
def load_data_directory():
    img_list=pd.read_csv('name_list.csv') 
    return img_list , np.load('img_feature_f.npy'),np.load('img_feature_b.npy')

def fig2array (fig):
    # draw the renderer
    fig.canvas.draw ()
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_rgb(), dtype=np.uint8 ).reshape(h,w,3)
    return buf 
   
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

def model_process():
    global model
    global graph , sess
    graph=tf.get_default_graph()
    sess=tf.Session(graph=graph)
    K.set_session(sess)
    
    model=load_model()
    while True:
        queue=db.blpop(IMAGE_QUEUE,timeout=1) #tuple ('name','content')
        if queue != None:
            
            queue=json.loads(queue[1].decode("utf-8"))
            q_id=queue["id"]
            img_f=base64_decode_image(queue["image_f"],IMAGE_DTYPE,(224,224,3))
            img_b=base64_decode_image(queue["image_b"],IMAGE_DTYPE,(224,224,3))
            img_f_in=preprocess_input(np.expand_dims(np.array(img_f),axis=0))
            img_b_in=preprocess_input(np.expand_dims(np.array(img_b),axis=0))
            f_name=queue["filename"]
            f_name2=queue["filename2"]
            
            with graph.as_default():
                K.set_session(sess)
                query_feature_front = model.predict(img_f_in)[0]   
                query_feature_back=model.predict(img_b_in)[0]
                
            sim_front=query_feature_front.dot(feature_f.T)
            sim_back=query_feature_back.dot(feature_b.T)
            top20=np.argsort(sim_front+sim_back)[-20:][::-1]
            fig=searching_display(top20,[img_f.astype(np.uint8),img_b.astype(np.uint8)],[f_name,f_name2])
            result=fig2array(fig).astype('float32') #type =uint8
            result=result.copy(order="C")
            r={"shape":result.shape,"result":base64_encode_image(result)}
            db.set(q_id,json.dumps(r))
#            del model
#            gc.collect()
        time.sleep(SERVER_SLEEP)
    
###flask function
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

    #send image to redis server 
    img_f=Image.open("uploads/"+filename)
    img_f=resize_aspect_ratio(img_f,(200,200))
    img_f=padding(img_f,(224,224))
    img_f=np.array(img_f).astype('float32')
    #img_f_in=preprocess_input(np.expand_dims(np.array(img_f),axis=0))# shape 1,224,224,3
    img_f=img_f.copy(order="C") #this is important!!! to ensure Np array is C-continus
    
    img_b=Image.open("uploads/"+filename2)
    img_b=resize_aspect_ratio(img_b,(200,200))
    img_b=padding(img_b,(224,224))
    img_b=np.array(img_b).astype('float32')
    #img_b_in=preprocess_input(np.expand_dims(np.array(img_b),axis=0))
    img_b=img_b.copy(order="C")#this is important!!! to ensure Np array is C-continus
    
    #產生ID，避免redis傳輸與取值的時候發生 hash/key 衝突的情況
    k=str(uuid.uuid4())
    d={"id":k,"image_f":base64_encode_image(img_f),
       "image_b":base64_encode_image(img_b),
       "filename":filename,"filename2":filename2}
    db.rpush(IMAGE_QUEUE,json.dumps(d))
    
    #receive serching result from redis stack
    while True:
        output=db.get(k)
        if output is not None:
            output=json.loads(output.decode("utf-8"))
            r_shape=tuple(output["shape"])
            result=base64_decode_image(output['result'],"float32",r_shape).astype(np.uint8) #np array
            #result=base64_decode_image(output,"uint8",(1728,2304,3)) #np array
            db.delete(k)
            break
        time.sleep(CLIENT_SLEEP)
    
    im=Image.fromarray(result)
    im.save("output/{}.jpg".format(filename))
    #fig=searching_display(top20,[img_f,img_b],[filename,filename2])
    #fig.savefig('output/{}.jpg'.format(filename))    
    return send_file("output/{}.jpg".format(filename), mimetype='image/jpg')

if __name__ == '__main__':
    
    print("* Starting model service...")
    db_list , feature_f , feature_b =load_data_directory()
    t=Thread(target=model_process)
    t.daemon=True
    t.start()
    
    print("* Starting web service...")
    app.debug=True
    app.run(host="127.0.0.1",port=int("5000"),use_reloader=False)


####test code
#    img_f=Image.open('test.jpg')
#    img_b=Image.open('test.jpg')
#    filename='test.jpg'
#    filename2='test.jpg'