from django.shortcuts import render
from rest_framework.views import    APIView
from rest_framework.parsers import MultiPartParser, FormParser,FileUploadParser
from rest_framework.response import Response
from rest_framework import status
from .serializers import RegisterSerializer
from .models import Register
from .helper import modify_input_for_multiple_files
import cv2
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
from google.cloud import storage
import firebase_admin
from firebase_admin import credentials
#from firebase_admin import firestore
from firebase_admin import db

cred = credentials.Certificate(r"C:\Users\harid\Desktop\serviceaccountkey.json")

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://missing-person-finder-8e324-default-rtdb.firebaseio.com/'
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

        cred = credentials.Certificate(r"C:\Users\harid\Desktop\serviceaccountkey.json")
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
        X, y = list(), list()
        labels = "Hari"
        #labels = [lbls for _ in range(len(face))]
        X.extend(face)
        y.extend(labels)
        trainX, trainy = asarray(X), asarray(y)
        #print(trainX.shape, trainy.shape)
        model = load_model('facenet_keras.h5')
        print('Loaded Model')
        newTrainX = list()
        print("made list")
        embedding = get_embedding(model, trainX)
        newTrainX.append(embedding)
        newTrainX = asarray(newTrainX)
        print(newTrainX)
        #print(newTrainX.shape)
        #print(newTrainX)
        #s = base64.b64encode(newTrainX)
        #print(s)
        #print(trainy)
        savez_compressed('5-celebrity-faces-embeddings.npz', newTrainX, trainy)

        #r = base64.decodebytes(s)
        #q = np.frombuffer(r, dtype=np.float64)
        #print(print(np.allclose(q, newTrainX)))
        embedded_data = newTrainX.tolist()
        users_ref.update({
            firstname+lastname :{
                'firstname': firstname,
                'lastname': lastname,
                'encoding': embedded_data
            }
        }) 
        ref = db.reference('server/missing data')
        stored_data = ref.get()
        missing_persons = list(i for i in stored_data.keys())
        #print(type(asarray(missing_persons)))
        #print(stored_data[[i for i in stored_data.keys()]]['encoding'])
        encodings = list()
        for j in missing_persons:
            encodings.append(stored_data[j]['encoding'])
        encodings = asarray(encodings)
        trainy = asarray(missing_persons)
        savez_compressed('face-embeddings.npz', encodings, trainy)
        print("Done............................................................")
        
        posts_serializer = RegisterSerializer(data=datas)
        #posts_serializer = RegisterSerializer(data=request.data)
        print(posts_serializer)
        if posts_serializer.is_valid():
            posts_serializer.save()
            return Response(posts_serializer.data, status=status.HTTP_201_CREATED)
        else:
            print('error', posts_serializer.errors)
            return Response(posts_serializer.errors, status=status.HTTP_400_BAD_REQUEST)