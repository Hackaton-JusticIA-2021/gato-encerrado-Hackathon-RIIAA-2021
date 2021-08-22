# funciones de preproceso
import cv2
import numpy as np
from PIL import Image, ImageFilter
from PIL import ImageEnhance
from kraken import binarization
import pytesseract


def get_grayscale(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def thresholding(image_path):
    from keras.preprocessing import image
    img = image.load_img(image_path, grayscale=True)
    img = image.img_to_array(img, dtype='uint8')
    th = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return th


def canny(image_path):
    image = cv2.imread(image_path)
    return cv2.Canny(image, 100, 200)


def invert_image(image_path):
    image = cv2.imread(image_path)
    return cv2.bitwise_not(image)


def word_boxes(image_path):
    image = cv2.imread(image_path)
    from pytesseract import Output
    d = pytesseract.image_to_data(image, output_type=Output.DICT)
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:
            (x, y, w, h) = (d['left'][i], d['top']
                            [i], d['width'][i], d['height'][i])
            img = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return img


def noise_removal(image_path):
    image = cv2.imread(image_path)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=2)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image


def thin_font(image_path):
    """ input: imagen sin ruido """
    image = cv2.imread(image_path)
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image


def thick_font(image_path):
    """ input: imagen sin ruido """
    image = cv2.imread(image_path)
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image


def remove_borders(image_path):  # Rolando ya hizo esto
    """ input: imagen sin ruido """
    image = cv2.imread(image_path)
    contours, heiarchy = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return crop

# preprocesamiento con kraken y pillow


def pillow_image_detail(image_path):
    im = Image.open(image_path)
    im2 = im.filter(ImageFilter.DETAIL)
    return im2


def pillow_sharpness(image_path):
    im = Image.open(image_path)
    enhancer = ImageEnhance.Sharpness(im)
    return enhancer.enhance(10.0)


def pillow_edges(image_path):
    im = Image.open(image_path)
    return im.filter(ImageFilter.FIND_EDGES)


def aplicar_preproceso(img_path, filtro):
    imagen_preprocesada, filtro_name = filtro(img_path)
    return imagen_preprocesada, filtro_name


def guardar_imagen_cv2(img_path, filtro, filtro_name="", file_path=""):
    """
    img_path: directorio de la imagen
    filtro: funcion para aplicar filtro
    file_path: directorio donde guardaremos las imagen
    """
    try:
        imagen_preprocesada = aplicar_preproceso(img_path, filtro)
    except:
        print("Error, image path incorrecto")
    filename = file_path + img_path.replace('.JPG', '').replace(
        "/", f"-{filtro_name}-")+".JPG"
    cv2.imwrite(filename, imagen_preprocesada)
    print("Error, file_path incorrecto")
    return filename


# se guardan diferente las imagenes de pillow - cv2
def guardar_imagen_pillow(img_path, filtro, filtro_name="", file_path=""):
    """
    img_path: directorio de la imagen
    filtro: funcion para aplicar filtro
    file_path: directorio donde guardaremos las imagen
    """
    try:
        im = filtro(img_path)
    except:
        print("Error, image path incorrecto")
    filename = file_path+img_path.replace("/", f"-{filtro_name}-")
    im.save(filename)


def guardar_imagen(img_path, filtro, filtro_name="", file_path="", pillow=False):
    if(pillow):
        guardar_imagen_pillow(img_path, filtro, filtro_name, file_path)
    else:
        guardar_imagen_cv2(img_path, filtro, file_path)
