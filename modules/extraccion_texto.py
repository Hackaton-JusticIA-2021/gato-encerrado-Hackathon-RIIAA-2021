import pytesseract
from PIL import Image


def extraer_texto_imagen(image_path, psm=6):
    im = Image.open(image_path)
    custom_config = r'--oem 3 --psm '+str(psm)+' --lang spa'
    texto = pytesseract.image_to_string(im, config=custom_config)
    return texto


def guardar_texto(image_path, files_path):
    """
    image_path: directorio donde guarda las imagenes
    files_path: directorio donde guarda los archivos de texto
    """
    texto = extraer_texto_imagen(image_path)
    img_name = Path(image_path).name
    output_filename = img_name.replace('.JPG', '.txt')
    output_path=str(Path(files_path)/output_filename)
    try:
        f = open(output_path, "x")
        f.write(texto)
    except:
        print("error en image_path ")
    finally:
        f.close()
    return texto
