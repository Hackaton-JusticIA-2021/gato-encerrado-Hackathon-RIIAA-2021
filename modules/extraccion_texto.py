import pytesseract
from PIL import Image


def extraer_texto_imagen(image_path):
    im = Image.open(image_path)
    custom_config = r'--oem 3 --psm 6 --lang spa'
    texto = pytesseract.image_to_string(im, config=custom_config)
    return texto


def guardar_texto(image_path, files_path):
    """
    image_path: directorio donde guarda las imagenes
    files_path: directorio donde guarda los archivos de texto
    """
    texto = extraer_texto_imagen(image_path)
    filename = image_path.replace('.JPG', '.txt').replace("/", "")
    os.chdir(dir+"files_path")
    f = open(filename, "x")
    try:
        f.write(texto)
    except:
        print("error en image_path ")
    finally:
        f.close()
        os.chdir(dir)
      # Muestra
