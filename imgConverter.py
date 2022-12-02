import cv2


def imgChange(img):
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    th, im_gray_th_otsu = cv2.threshold(im_gray, 128, 192, cv2.THRESH_OTSU)

    return im_gray_th_otsu