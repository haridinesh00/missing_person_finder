import cv2 
import os
import glob
from PIL import Image
import csv
face_class = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def face_extractor (photo):
        gphoto = cv2.cvtColor(photo, cv2.COLOR_BGRZGRAY) 
        detected = face_class.detectMultiscale(gphoto)

        if detected == (): 
            return None

        else :
            (x,y,w,h) = detected[0] 
            cphoto = photo [y:y+h, x:x+w ] 
            return cphoto




parent_dir = "C://Users//harid//Desktop//Finding-missing-person-using-AI-master//app//train//hariddinesh1"

parent_dir= "C://Users//harid//Desktop//Finding-missing-person-using-AI-master//app//test//hariddinesh1" 

images_train = [cv2.imread(file) for file in glob.glob("C://Users//harid//Desktop//Finding-missing-person-using-AI-master//app//train//hariddinesh1//*.jpg")]
images_test = [cv2.imread(file) for file in glob.glob("C://Users//harid//Desktop//Finding-missing-person-using-AI-master//app//test//hariddinesh1//*.jpg")]


for file in glob.glob("C:\\Users\\harid\\Desktop\\Finding-missing-person-using-AI-master\\app\\train\\hariddinesh1\\*.jpg"):
    print(file)
    images_train2 = cv2.imread(file)
    face = cv2.resize(images_train2, (200,200))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    cv2.imshow(face)
    cv2.imwrite("C:\\Users\\harid\\Desktop\\Finding-missing-person-using-AI-master\\app\\train\\hariddinesh\\*.jpg",face)

for file in glob.glob("C:\\Users\\harid\\Desktop\\Finding-missing-person-using-AI-master\\app\\test\\hariddinesh1\\*.jpg"):
    print(file)
    images_train2 = cv2.imread(file)
    face = cv2.resize(images_train2, (200,200))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    cv2.imshow(face)
    cv2.imwrite("C:\\Users\\harid\\Desktop\\Finding-missing-person-using-AI-master\\app\\test\\hariddinesh\\*.jpg",face)


'''
for file in images_test:
    images_test = cv2.imread(file)
    face = cv2.resize(images_test, (200,200))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("C://Users//harid//Desktop//Finding-missing-person-using-AI-master//app//test//hariddinesh//*.jpg", face)
'''
'''
path = "C:/Users/harid/Desktop/Finding-missing-person-using-AI-master/app/train/hariddinesh1"
dirs = os.listdir( path )


def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((200,200), Image.ANTIALIAS)
            #imResize = cv2.cvtColor(imResize, cv2.COLOR_BGR2GRAY)
            imResize.save(f + ' resized.jpg', 'JPEG', quality=90)
'''

'''
for image_file_name in os.listdir('C:\\Users\\harid\\Desktop\\Finding-missing-person-using-AI-master\\app\\train\\hariddinesh1\\'):
    if image_file_name.endswith(".jpg"):
        print(image_file_name)
        img_train1 = Image.open('C:\\Users\\harid\\Desktop\\Finding-missing-person-using-AI-master\\app\\train\\hariddinesh1\\'+image_file_name)
        face = cv2.resize((200,200), img_train1)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        img_train1.save('C:\\Users\\harid\\Desktop\\Finding-missing-person-using-AI-master\\app\\train\\hariddinesh' + '.jpg')

'''






from keras.applications import vgg16

img_rows, img_cols = 224, 224

model= vgg16.VGG16 (weights = 'imagenet',
include_top = False,
input_shape= (img_rows, img_cols, 3))

for layer in model.layers:
    layer.trainable = False

model.summary()

def layer_adder (bottom_model, num_classes):

    top_model= bottom_model.output

    top_model= GlobalAveragePooling2D() (top_model) 
    top_model= Dense(1024, activation='relu') (top_model)

    top_model= Dense(512, activation='relu') (top_model)

    top_model= Dense(num_classes, activation='softmax') (top_model)

    return top_model

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D 
from tensorflow.keras.layers import BatchNormalization

from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

from keras.models import Model

# Set our class number to 3 (Young, Middle, Old)

num_classes = 4

FC_Head = layer_adder (model, num_classes)

model = Model (inputs = model.input, outputs = FC_Head)

print(model.summary())


from keras.preprocessing.image import ImageDataGenerator

train_data_dir= "C://Users//harid//Desktop//Finding-missing-person-using-AI-master//app//train//hariddinesh//"
validation_data_dir = "C://Users//harid//Desktop//Finding-missing-person-using-AI-master//app//test//hariddinesh//"
#Let's use some data augmentaiton

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45, 
    horizontal_flip=True,
    width_shift_range=0.3,
    height_shift_range=0.3,
    fill_mode= 'nearest')

validation_datagen = ImageDataGenerator (rescale=1./255)

#set our batch size (typically on most mid tier systems we'LL use 16-32)

batch_size=1

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_rows, img_cols), batch_size=batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(validation_data_dir, target_size=(img_rows, img_cols), batch_size = batch_size, class_mode='categorical')


from tensorflow.keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("face_detector.h15",
                              monitor="val_loss",
                              mode="min",
                              save_best_only = True,
                              verbose=1 )
earlystop =  EarlyStopping(monitor = 'val_loss',
                           min_delta = 0,
                           patience = 3,
                           verbose = 1,
                           restore_best_weights = True)
callbacks = [earlystop, checkpoint]

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'Adam',
              metrics = ['accuracy'])

nb_train_samples = 280
nb_validation_samples = 120

epochs =1
batch_size = 1

history = model.fit_generator(
    train_generator,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
)
from keras.models import load_model

classifier = load_model( 'face_detector.h15')

import os

import cv2

import numpy as np

from os import listdir
from os.path import isfile, join


stored_dict = { "[e]": "shikhar ", "[1]": "aman" }

stored_dict_n = {"[0]": "shikhar ", "[1]": "aman" }

def draw_test (name, pred, im):

    face = stored_dict[str(pred)]
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(im, 80, 0, 0, 100,cv2.BORDER_CONSTANT, value=BLACK)
    cv2.putText(expanded_image, face, (20, 60), cv2. FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    cv2.imshow (name, expanded_image)

def getRandomImage(path):
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path,x)),os.listdir(path)))
    random_directory = np.random.randint(0,len(folders))
    path_class = folders[random_directory]
    print("Class - " + str(pathg_class))
    file_path = path + path_class
    file_names = [f for f in listdir(file_path) if isfile(join(ile_path, f))]
    random_file_index = np.random.randint(0,len(file_names))
    image_name = file_names[random_file_index]
    return cv2.imread(file_path+"/"+image_name)

for i in range(0,10):
    input_im = getRandomImage("")
    input_original = input_im_copy()
    input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)

    input_im = cv2.resize(input_im, (224,224), interpolation = cv2.INTER_LINEAR)
    input_im = input_im / 255.
    input_im = input_im.reshape(1,224,224,3)

    res =np.argmax(classifier.predict(input_im, 1, verbose=0), axis=1)

    draw_test("Predictiom", res, input_original)
    cv2.waitKey(0)

cv2.destroyAllWindows()

