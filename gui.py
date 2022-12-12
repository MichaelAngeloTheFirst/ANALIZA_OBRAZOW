import PySimpleGUI as sg
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# model
model = tf.keras.models.load_model("our_model.model")


def predict_model(path):
    temp = np.empty((1,256,256,3))
    temp[0]=cv.resize(plt.imread(path),(256,256),interpolation= cv.INTER_NEAREST)
    print(np.argmax(model.predict([temp])))
    return np.argmax(model.predict([temp]))


def model_labels(val):
    if val == 0:
        return "train/expected_files/dein.001.jpg", "Deinopis"
    if val == 1:
        return "train/expected_files/red.002.jpg", "Red Knee"
    if val == 2:
        return "train/expected_files/pea.001.jpg", "Peacock"


w, h = 640, 480

def get_image(path): 
    img = cv.imdecode(np.fromfile(path, dtype=np.uint8), cv.IMREAD_UNCHANGED)
    img = cv.resize(img, (256,256))
    imgbytes = cv.imencode('.ppm', img)[1].tobytes() 
    return imgbytes


sg.theme('DarkGreen3')   


layout = [  [sg.Column([[sg.Button('Exit' )]], element_justification='right', expand_x=True)],
            [sg.Text("Nasza skuteczność (accuracy) wynosi około 84,9% ", pad=(0,10))],
            [sg.Text("Choose a file: ", pad=(0,10)), sg.Input(), sg.FileBrowse(key="-IN-")],
            [sg.Button("Submit", pad=(0,10))],
            [sg.Text("", key='-OUTPUT-', pad=(0,10))],
            [sg.Text(" Nasz obrazek:", key='-TXT-', size=(25, 2), pad=(0, 0), expand_x=True, expand_y=False, visible=False),
            sg.Text(" Obraz przewidzianego pająka:", key='-TXT2-', size=(25, 2), pad=(0, 0), expand_x=True, expand_y=False, visible=False)],
            [sg.Image(key='-IMAGE-',  size=(256, 256), pad=(0,0), expand_x=False, expand_y=False),
            sg.Image(key='-IMAGE2-', size=(256, 256), pad=(0,0), expand_x=True, expand_y=False)]
            ]

# sg.theme_background_color('#FF0000')

window = sg.Window('Rozpoznawanie pająków', layout, size=(800, 600), finalize=True)
while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED or event == 'Exit':
        break

    if event == "Submit":
        val = predict_model(values["-IN-"])

        path, label = model_labels(val)
        window['-OUTPUT-'].update(value = " Nasza predykcja: " + label)

        imgbytes = get_image(values['-IN-'])
        window['-TXT-'].update(visible=True)
        window['-TXT2-'].update(visible=True)
        window['-IMAGE-'].update(data = imgbytes)
        imgbytes = get_image(path)
        window['-IMAGE2-'].update(data = imgbytes)


window.close()