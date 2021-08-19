import numpy as np
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import cv2

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import fitz #pip install PyMuPDF
from PIL import Image

# PDF to image function
def pdf_to_img(pdf_file):
    doc = fitz.open(pdf_file)
    images = []
    for page in doc:
        zoom_x = 4.0  # horizontal zoom
        zomm_y = 4.0  # vertical zoom
        mat = fitz.Matrix(zoom_x, zomm_y)  # zoom factor 2 in each dimension
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images
    

# Import image and return duple (pg, img)
def import_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (1,img)
    
# Image to text function    
def ocr_core(file):
    text = pytesseract.image_to_string(file, config='--psm 3')
    return text

# Image preprocesing
def transform_image(image):
    if type(image) != np.ndarray: 
        image = np.array(image)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = cv2.GaussianBlur(image, (5,5), 0)
    #image = cv2.medianBlur(image, 3)
    #image = cv2.Canny(image,200,300)
    

    return image

# Boxes detection letter
def detect_paragraph(image):
    #h, w, c = image.shape
    boxes = pytesseract.image_to_boxes(image)
    return boxes


def print_pages(pdf_file, pdf=True):

    text = ''
    
    #if pdf: #If the file is a pdf we transform it to image 
    #images = pdf_to_img(pdf_file)
      
    #for pg, img in enumerate(pdf_file):
    for img in pdf_file:

        img = transform_image(img)    #image preprocesing
        ''' #This lines write  a file wich is saved in the folder now we want return it
        file = open(f'{pdf_file}-page-{pg+1}.txt'.replace('.pdf', ''), 'w')
        file.write(ocr_core(img))
        print(file)
        file.close()
        '''
        text +=  ocr_core(img)
        #print(img)
        #print(pg)

    return text


'''
text = print_pages('alice copy.pdf')
print(text)
'''

# file = pdf_to_img('//Users/a.uceda/Documents/AI/ai-master/last_version/tests/documents/galton.pdf')                 #Import pdf
# output = print_pages(file, pdf=True) 
# #print(file)
# print(output)
