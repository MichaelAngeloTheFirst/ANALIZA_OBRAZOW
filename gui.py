import PySimpleGUI as sg
import cv2 as cv
import io
from PIL import Image, ImageTk
import numpy as np
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


# model
model = tf.keras.models.load_model("our_model.model")

def predict_model(path):
    temp = np.empty((1,256,256,3))
    temp[0]=cv.resize(plt.imread(path),(256,256),interpolation= cv.INTER_NEAREST)
    print(np.argmax(model.predict([temp])))
    return np.argmax(model.predict([temp]))

def model_labels(val):
    if val == 0:
        print("Deinopis")
        return "Deinopis"
    if val == 1:
        print("Red Knee")
        return "Red Knee"
    if val == 2:
        print("Peackock")
        return "Peackock"

predict_model("train/pea.jpg")

choices = glob.glob("*.jpg")
choices.remove("temp.jpg")


w, h = 640, 480

def resize_image(image_path, resize=None): 
    # if isinstance(image_path, str):
    # img = PIL.Image.open(image_path)
    img = image_path


    # else:
    #     try:
    #         img = PIL.Image.open(io.BytesIO(base64.b64decode(image_path)))
    #     except Exception as e:
    #         data_bytes_io = io.BytesIO(image_path)
    #         img = PIL.Image.open(data_bytes_io)

    # cur_width, cur_height = img.size
    cur_width, cur_height, dim = img.shape
    if resize:
        new_width, new_height = resize
        scale = min(new_height/cur_height, new_width/cur_width)
        # img = img.resize((int(cur_width*scale), int(cur_height*scale)), PIL.Image.Resampling.LANCZOS)
        img = cv.resize(img, ((int(cur_width*scale), int(cur_height*scale))) )
    # bio = io.BytesIO()
    # img.save(bio, format="PNG")
    # del img
    return img



def kmeans(image, K_input):
    img = image
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = K_input
    ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    cv.imwrite('temp.jpg', res2)
   


sg.theme('DarkGreen3')   
# All the stuff inside your window.
layout = [  [sg.Text("Nasza skuteczność (accuracy): ", )],
            [sg.T("")], 
            [sg.Text("Choose a file: "), sg.Input(), sg.FileBrowse(key="-IN-")],
            [sg.Button("Submit")],
            [sg.Text("Nasza predykcja: ")],
            [sg.Text(key='-OUTPUT-')],
            [sg.Text("Nasz obrazek: "),
            sg.Image(key='-IMAGE-',  size=(8, 4), pad=(0,0), expand_x=False, expand_y=False),
            sg.Text("Pająk: "),
            sg.Image(key='-IMAGE2-', size=(8, 4), pad=(0,0), expand_x=True, expand_y=True)],
            [sg.Button('Exit')]
           

            
            ]

# sg.theme_background_color('#FF0000')


# Create the Window
window = sg.Window('Rozpoznawanie pająków', layout, size=(1000, 700), finalize=True)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED or event == 'Exit':
        break

    # if event == 'Ok':
    if event == "Submit":
        val = predict_model(values["-IN-"])
        print(values["-IN-"])
        labels = model_labels(val)
        window['-OUTPUT-'].update(value=labels)
        img = cv.imdecode(np.fromfile(values['-IN-'], dtype=np.uint8), cv.IMREAD_UNCHANGED)
        img = cv.resize(img, (256,256))
        imgbytes = cv.imencode('.ppm', img)[1].tobytes() 
        window['-IMAGE-'].update(data = imgbytes)
        window['-IMAGE2-'].update(data = imgbytes)


window.close()
