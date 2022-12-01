import PySimpleGUI as sg
import cv2 as cv
import io
import PIL.Image
# import base64
import numpy as np
import glob


choices = glob.glob("*.jpg")
choices.remove("temp.jpg")


w, h = 640, 480

def resize_image(image_path, resize=None): 
    # if isinstance(image_path, str):
    img = PIL.Image.open(image_path)
    # else:
    #     try:
    #         img = PIL.Image.open(io.BytesIO(base64.b64decode(image_path)))
    #     except Exception as e:
    #         data_bytes_io = io.BytesIO(image_path)
    #         img = PIL.Image.open(data_bytes_io)

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height/cur_height, new_width/cur_width)
        img = img.resize((int(cur_width*scale), int(cur_height*scale)), PIL.Image.Resampling.LANCZOS)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    del img
    return bio.getvalue()



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
layout = [  [sg.Text('Select your image')],
            [sg.Listbox(choices, size=(20, len(choices)), key='-IMG-')],
            [sg.Text('Select your K:')],
            [sg.Spin([i for i in range(1, 25)],     initial_value=1, size=(4, 2), enable_events=True, key='-K-')],
            [sg.Button('Ok'), sg.Button('Cancel')],
            [sg.Graph(
            canvas_size=(w, h),
            graph_bottom_left=(0, 0),
            graph_top_right=(w, h),
            key="-GRAPH-",
            change_submits=True,  # mouse click events
            # background_color='lightblue',
            drag_submits=True), ]
            
            ]

# sg.theme_background_color('#FF0000')


# Create the Window
window = sg.Window('Color Segmentation', layout, size=(1000, 700), finalize=True)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED or event == 'Cancel':
        break

    if event == 'Ok':
        if values['-IMG-'] and values['-K-']:    
            image = cv.imread(values['-IMG-'][0])
            K = int(values['-K-'])
            res2 = kmeans(image, K)

            window["-GRAPH-"].erase()
            window["-GRAPH-"].draw_image(data= resize_image('temp.jpg', resize=(w,h)), location=(0, h))

window.close()

