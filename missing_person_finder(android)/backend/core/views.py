from django.shortcuts import render
from rest_framework.views import    APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from .serializers import RegisterSerializer
from .models import Register


import os
from PIL import Image
from mtcnn.mtcnn import MTCNN
from numpy import asarray
from os import listdir
from tensorflow.keras.models import load_model
from numpy import expand_dims
import base64
import numpy as np
from numpy import savez_compressed
import firebase_admin
from firebase_admin import credentials
#from firebase_admin import firestore
from firebase_admin import db
import requests
from numpy import load
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
import urllib.request
from sklearn.svm import SVC

cred = credentials.Certificate(r"C:\Users\harid\Desktop\serviceaccountkey.json")

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://slacks-1008c.firebaseio.com/'
})

#firebase_admin.initialize_app(cred)

#db = firestore.client()

ref = db.reference('server/')
users_ref = ref.child('missing data')
#doc_ref = db.collection(u'missing_persons')

FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
	base_path=os.path.abspath(os.path.dirname(__file__)))

def extract_face(filename, required_size=(160, 160)):
	# load image from file
	# convert to RGB, if needed
    image = Image.open(filename)
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
	# create the detector, using default weights
    detector = MTCNN()
	# detect faces in the image
    results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
	# bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
	# extract the face
    face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array
'''
def load_faces(directory):
	faces1 = list()
	# enumerate files
	for filename in listdir(directory):
		# path
		path = directory + filename
		# get face
		face2 = extract_face(path)
		# store
		faces1.append(face2)
	return faces1
'''

def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]


def front(request):
    context={ }
    return render(request,"index.html",context)
def fronts(request, id):
    context={ }
    return render(request,"index.html",context)

class PostView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        posts = Register.objects.all()
        serializer = RegisterSerializer(posts, many=True)
        return Response(serializer.data)

    def post(self, request, *args, **kwargs):
        datas=request.data
        image=datas['image']
        #encoded_string = base64.b64encode(image.read())
        #print(image)

        #cred = credentials.Certificate(r"C:\Users\harid\Desktop\serviceaccountkey.json")
        #firebase_admin.initialize_app(cred)
        

        firstname=datas['firstname']
        lastname=datas['lastname']
        print(firstname)
        print(lastname)
        #print(datas)
        #image = "media/media/"
        #print(image)
        face = extract_face(image)
        print(face)
        print("testing..........")
        print(len(face))
        X = list()
        #labels = "Hari"
        #labels = [lbls for _ in range(len(face))]
        X.extend(face)
        #y.extend(labels)
        trainX = asarray(X)
        #print(trainy)
        trainX = expand_dims(trainX, axis=0)
        print(trainX.shape)
        #print(trainy.shape)
        savez_compressed('pikachu-faces-dataset.npz', trainX)
        model = load_model('facenet_keras.h5')
        print('Loaded Model')
        newTrainX = list()
        print("made list")
        for face_pixels in trainX:
            #print(face_pixels)
            embedding = get_embedding(model, face_pixels)
            newTrainX.append(embedding)
        newTrainX = asarray(newTrainX)
        print(newTrainX.shape)
        
        #embedding = asarray(embedding)
        #print(newTrainX)
        #print(newTrainX.shape)
        #print(newTrainX)
        #s = base64.b64encode(newTrainX)
        #print(s)
        #print(trainy)
        #savez_compressed('5-celebrity-faces-embeddings.npz', newTrainX, trainy)
        #savez_compressed('5-celebrity-faces-embeddings.npz', embedding, trainy)

        #r = base64.decodebytes(s)
        #q = np.frombuffer(r, dtype=np.float64)
        #print(print(np.allclose(q, newTrainX)))
        embedded_data = newTrainX.tolist()
        #embedded_data = embedding.tolist()

        print('done.............')
        # Adding encodings to firebase...............................................................
        '''
        users_ref.update({
            firstname+lastname :{
                'firstname': firstname,
                'lastname': lastname,
                'encoding': embedded_data
            }
        }) 
        '''
        #............................................................................................
        
        '''
        ref = db.reference('server/missing data')
        stored_data = ref.get()
        missing_persons = list(i for i in stored_data.keys())
        print(missing_persons)
        
        #print(type(asarray(missing_persons)))
        #print(stored_data[[i for i in stored_data.keys()]]['encoding'])
        encodings = list()
        for j in missing_persons:
            encodings.append(stored_data[j]['encoding'][0])
        encodings = asarray(encodings)
        print(encodings.shape)
        trainy = asarray(missing_persons)
        print(trainy.shape)
        savez_compressed('pikachu-faces-embeddings.npz', encodings, trainy)

        #perfected.................................................................................................................................
        '''
        '''
        data_pik = load('pikachu-faces-embeddings.npz')
        trainX, trainy = data_pik['arr_0'], data_pik['arr_1']
        in_encoder = Normalizer(norm='l2')
        trainX = in_encoder.transform(trainX)
        out_encoder = LabelEncoder()
        out_encoder.fit(trainy)
        trainy = out_encoder.transform(trainy)

        model = SVC(kernel='linear', probability=True)
        model.fit(trainX, trainy)

        #################################################################################################################################################
        
        new_ref = db.reference('Image').get()
        #new_stored_data = new_ref.get()
        Image_data = list(i for i in new_ref.keys())
        imagearray = list()

        for k in Image_data:
            imagearray.append(new_ref[k]['imageUrl'])
        

        #savez_compressed('face-embeddings.npz', encodings, trainy)

        print(imagearray[0])
        #im = Image.open(requests.get(imagearray[0], stream=True).raw) 
        urllib.request.urlretrieve(imagearray[0], "gfg.jpg")
        #face1 = extract_face("gfg.jpg")
        #img = Image.open("gfg.jpg")
        #img.show()

        #............................................................................................................................................
        face1 = extract_face("gfg.jpg")
        X1 = list()
        X1.extend(face1)
        trainX1 = asarray(X1)
        trainX1 = expand_dims(trainX1, axis=0)
        #print(trainy.shape)
        savez_compressed('pikachu-faces-dataset1.npz', trainX1)
        model1 = load_model('facenet_keras.h5')
        newTrainX1 = list()
        for face_pixels1 in trainX1:
            #print(face_pixels1)
            embedding1 = get_embedding(model1, face_pixels1)
            newTrainX1.append(embedding1)
        newTrainX1 = asarray(newTrainX1)
        print(newTrainX1.shape)
        yhat_class = model.predict(newTrainX1)
        predict_names = out_encoder.inverse_transform(yhat_class)
        print(predict_names[0])
        db.reference('Image').delete()
        #.............................................................................................................................................


        
        new_X = list()
        new_X.extend(new_image)
        testXX = asarray(new_X)
        savez_compressed('test-faces-dataset.npz', testXX)

        new_data = load('test-faces-dataset.npz')
        testXX = new_data['arr_0']
        #print(testX)
        newTestX = list()
        new_embedding = get_embedding(model, testXX)
        newTestX.append(new_embedding)
        newTestX = asarray(newTestX)
        savez_compressed('new-faces-embeddings.npz', newTestX)
        new_data = load('new-faces-embeddings.npz')
        testXX = new_data['arr_0']
        
        testX = in_encoder.transform(testXX)
        samples = expand_dims(testX, axis=0)
        print(samples.shape)
        yhat_class = model.predict(samples)
        #print(yhat_class)
        predict_names = out_encoder.inverse_transform(yhat_class)
        print(predict_names)

    
        
        
        print("Done............................................................")
        
    
        #posts_serializer = RegisterSerializer(data=request.data)
        #print(posts_serializer) 
        '''
        return Response(embedded_data)
       