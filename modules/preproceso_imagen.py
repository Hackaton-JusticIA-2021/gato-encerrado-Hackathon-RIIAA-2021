# funciones de preproceso
import cv2
import numpy as np
from PIL import Image, ImageFilter
from PIL import ImageEnhance
from kraken import binarization
from kraken import blla
import pytesseract
from pathlib import Path
import os 


def extraer_bloques_texto(img_path, sufijo = '_bloques', output_dir = ''):
  '''
  Extrae los bloques de una imagen y guarda la imagen obtenida.
  Primero obtiene poligonos que representan regiones con texto.
  Luego genera rectangulos que contienen a dichas regiones.
  Finalmente, recorta la imagen usando al menor rectangulo 
  que contiene a todas los rectangulos anteriores.

  Parameters
    -------------
    img_path: str
      Ruta del archivo
    output_dir: str
      Ruta del directorio para guardar las imagenes
  '''  

  # imagen de PIL para kraken
  im = Image.open(img_path)
  # imagen de openvc para manipulacion de geometria
  im_cv = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

  # segmentacion de la imagen
  # se podría convertir a imagen binaria antes de la segmentación
  # se prefiere la imagen original, para no perder escritura con lapiz
  baseline_seg = blla.segment(im)

  # coordenadas de los poligonos que contienen regiones con texto
  text_regions = baseline_seg['regions']['text']
 
  # regiones de texto con rectangulos
  # imagen de salida (rectangulos), lienzo en blanco del mismo tamaño que la original
  # para que el lienzo sea negro, fijar out = 0
  out_rect = np.zeros_like(im_cv)
  out_rect = 255 - out_rect

  # se crean mascaras para los rectangulos que contienen regiones con texto
  for region in text_regions:
    pts = np.array(region, dtype = np.int32)
    # se encuentra el rectangulo que contiene al poligono
    x_r, y_r, w_r, h_r = cv2.boundingRect(pts)
    mask = np.zeros((im_cv.shape[0], im_cv.shape[1]))
    mask = mask.astype(np.bool)
    # se copia solamente la region que estan en la mascara
    out_rect[y_r:y_r+h_r, x_r:x_r+w_r] = im_cv[y_r:y_r+h_r, x_r:x_r+w_r]

  region_total = [punto for region in text_regions for punto in region]

  # se eliminan regiones de la imagen sin texto
  # agregar padding si hace falta
  x, y, w, h = cv2.boundingRect(np.array(region_total, dtype = np.int32))
  out_rect_cut = out_rect[y:y+h, x:x+w]

  # obtiene el nombre de la imagen
  img_name = Path(img_path).name
  # cambia el sufijo
  output_filename = img_name.replace('.JPG', '') + sufijo + '.JPG'
  # crea el path de salida
  output_dir = Path(output_dir)
  output_path = output_dir / output_filename
  output_path_str = str(output_path)

  cv2.imwrite(output_path_str, out_rect_cut)


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


def guardar_imagen_cv2(img_path, filtro, filtro_name="", output_dir=""):
    """
    img_path: directorio de la imagen
    filtro: funcion para aplicar filtro
    file_path: directorio donde guardaremos las imagen
    """
    try:
        imagen_preprocesada = aplicar_preproceso(img_path, filtro)
    except:
        print("Error, image path incorrecto")
    img_name = Path(img_path).name #cambia el sufijo
    output_filename = img_name.replace('.JPG','')+f"-{filtro_name}"+".JPG"
    output_dir = Path(output_dir)
    output_path = output_dir/output_filename
    cv2.imwrite(str(output_path), imagen_preprocesada)
    


# se guardan diferente las imagenes de pillow - cv2
def guardar_imagen_pillow(img_path, filtro, filtro_name="", output_dir=""):
    """
    img_path: directorio de la imagen
    filtro: funcion para aplicar filtro
    file_path: directorio donde guardaremos las imagen
    """
    try:
        im = filtro(img_path)
        img_name = Path(img_path).name #cambia el sufijo
        output_filename = img_name.replace('.JPG','')+f"-{filtro_name}"
        output_dir = Path(output_dir)
        output_path = output_dir/output_filename
        im.save(output_path)
    except:
        print("Error, image path incorrecto")
    


def guardar_imagen(img_path, filtro, filtro_name="", output_dir="", pillow=False):
    if(pillow):
        guardar_imagen_pillow(img_path, filtro, filtro_name, file_path)
    else:
        guardar_imagen_cv2(img_path, filtro, file_path)
        
        

def get_line(image_path,save_path):
#Segmentamos imagen
  im=Image.open(image_path)
  name=Path(image_path).name[:-4]
  baseline_seg = blla.segment(im)
  results=[]
  if not os.path.exists(save_path):
      os.mkdir(save_path)
  folder_path=str(Path(save_path)/name)
  os.mkdir(folder_path)
  
  #Nos moveremos en cada fila identificada
  for k,segment in enumerate(baseline_seg['lines']):
    sub_im=np.array(im)
    blank=np.zeros(sub_im.shape)+255
    #Lista de fronteras y lineas bases
    boundaries=np.array(segment['boundary'])
    lines=np.array(segment['baseline'])
    #extremos y
    y_max=max(boundaries[:,1])
    y_min=min(boundaries[:,1])
    #coordenadas base en x
    x_left=min(lines[:,0])
    x_right=max(lines[:,0])
    #coordenadas base en y
    y_left=lines[0,1]
    y_right=lines[-1,1]
    #Coordenadas y superiores 
    ind=boundaries[:,1]<min(y_left,y_right)
    sub_bound=boundaries[ind]
    #x para cada limite superior de y
    x_cut=sorted(list(set(list(sub_bound[:,0]))))
    # pendiente linea de base
    m=(y_right-y_left)/(x_right-x_left)
    x_1=x_left
    
    for i in range(len(x_cut)):
      x_2=x_cut[i]
      #Selelccionamos coordenada y
      index=sub_bound[:,0]==x_2
      y_top=min(sub_bound[index,1])
      #Calculamos aumento en base
      delta=int((x_2-x_left)*m)
      #Submatriz con la línea 
      blank[y_top-10:(y_left+delta+10),x_1:x_2]=sub_im[y_top-10:(y_left+delta+10),x_1:x_2]
      x_1=x_2
      
  #última iteracion
    delta=int((x_right-x_left)*m)
    blank[y_top-10:(y_left+delta+10),x_1:x_right]=sub_im[y_top-10:(y_left+delta+10),x_1:x_right]
    result=blank[y_min:y_max,:]
    results.append(result)
    img_res=Image.fromarray(result).convert('RGB')
    img_res.save(str(Path(folder_path)/str(i)+'.jpg'))
    
  return results

