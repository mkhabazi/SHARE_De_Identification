### THIS FILE HAS THE FUNCTIONS TO IMPORT, EXTRACT TEXT, AND REWRITE TEXT FROM IMAGES.

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math 

from paddleocr import PaddleOCR, draw_ocr
from PIL import Image


# Function to calculate the distance between 2 points and the angle with respect to the x-axis
def lines_carac(p1, p2):
    # Distance between the points
    p1_p2 = math.sqrt( (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 )
    # Angle
    if p2[0] - p1[0] == 0: # To solve math error ZeroDivisionError: float division by zero
        p2[0] += 0.0001
    p1_p2_angle = math.atan( (p2[1] - p1[1])/(p2[0] - p1[0]) )
    p1_p2_angle = math.degrees(p1_p2_angle)
    return p1_p2, p1_p2_angle

# The function detects the scale of the text to adjust it to the spacing
def text_size(text, l, fontFace=cv2.FONT_HERSHEY_SIMPLEX, thickness=3):
    scale = 0.25
    # getTextSize return the text size with this characteristics
    while cv2.getTextSize(text, fontFace=fontFace, fontScale=scale+0.25, thickness=thickness)[0][0] < 0.9*l:
        scale += 0.25
    return scale       

def import_transform(img_path):

    image = cv2.imread(img_path)     # import original image
    #image = cv2.flip(image, 1)    # USE FOR: xr4.jpeg => obtain the simetrical image 
    #image_or = cv2.rotate(image_or, cv2.ROTATE_90_COUNTERCLOCKWISE) 
    #image = cv2.rotate(image, cv2.ROTATE_180)

    x_size = image.shape[1]
    y_size = image.shape[0]
    #########################################

    # Create 2 copies of the image
    img = np.copy(image)           #image to extract and predict text

    ###### Preprocesing 2 image  ############
    # image color to gray 
    #img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #image bgr to rgb
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # resize image
    scale = 5                           # percent of original size
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    # delete noise
    img= cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.medianBlur(img, 5)
    #_, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    #img = cv2.Canny(img, 1,150)
    ###########################################

    return image, img, (x_size, y_size)

def image2txt(img_path):

    img = import_transform(img_path)[1]

    #### Select the best option depend of datases##

    ocr = PaddleOCR(lang="en")
    #ocr = PaddleOCR(use_angle_cls=True, lang="en") # The model file will be downloaded automatically when executed for the first time
    #ocr = PaddleOCR(use_angle_cls=True)
    #ocr = PaddleOCR()
    ###############################################

    # Detect the texts and thier positions in the image
    result = ocr.ocr(img, cls=True)

    return result

def rewrite_image(result, img_path):

    image, _, size = import_transform(img_path)

    for i in result:

        position = i[0]  #obtain 4 corners of the text
        
        # Calculate the distance and angle between the corners
        x1_x2, x1_x2_angle = lines_carac(position[0], position[1])
        x4_x3, x4_x3_angle = lines_carac(position[3], position[2])

        h_long = (x1_x2 + x4_x3)/2              # "horizontal" length
        h_ang = (x1_x2_angle + x4_x3_angle)/2   # "horizontal" angle

        x2_x3, x2_x3_angle = lines_carac(position[1], position[2])
        x1_x4, x1_x4_angle = lines_carac(position[0], position[3])

        v_long = (x2_x3 + x1_x4)/2              # "vertical" length
        v_ang = (x2_x3_angle + x1_x4_angle)/2   # "vertical" angle
        
        # Detect the direction of the text and the angle by comparing the lengths.
        if h_long >= v_long:
            text_ang = -h_ang

            xi = position[3][0]
            yi = position[3][1]
            xf = position[1][0]
            yf = position[1][1]

            l = int(max(x1_x2, x4_x3)) # box length
            h = int(max(x2_x3, x1_x4)) # box hight

        else:
            text_ang = -v_ang

            xi = position[0][0]
            yi = position[0][1]
            xf = position[2][0]
            yf = position[2][1]

            h = int(max(x1_x2, x4_x3)) # box length
            l = int(max(x2_x3, x1_x4)) # box hight

        ###### If the text is not horizontal
        if abs(text_ang) > 5:
            
            
            ########### create a black image with the box and text
            
            image_text = np.zeros((image.shape[0], image.shape[1]+l, 3))                                                 # Array with the same dimenstion than the image
            image_text = cv2.rectangle(image_text, ( int(xi), int(yi) ), ( int(xi+l), int(yi-h)), (255,255,255), -1)  # Create the black rectangule
            
            ### To understand the process with images uncomment these lines
            # plt.imshow(image_text)
            # plt.show()
            # plt.close()

            scale = text_size(i[-1][0], l, fontFace=cv2.FONT_HERSHEY_SIMPLEX, thickness=6)           # Detect the text size
            image_text = cv2.putText(image_text, i[-1][0], (int(xi), int(yi-(h/4))) , cv2.FONT_HERSHEY_SIMPLEX, scale, (1,1,1), int(scale*2) )  # Write the text
            #image_text = cv2.putText(image_text, i[-1][0], (0, int(l-(h/4)) ), cv2.FONT_HERSHEY_SIMPLEX, scale, (10.,10.,10.), int(scale*2) )  # Write the text

            ### To understand the process with images uncomment these lines
            # plt.imshow(image_text)
            # plt.show()
            # plt.close()

            # Rotate the box and text
            M = cv2.getRotationMatrix2D( (int(xi), int(yi)), text_ang, 1 )
            image_text = cv2.warpAffine( image_text, M, (image.shape[1]+l ,image.shape[0]) )
            image_text = image_text[:, :image.shape[1]]

            # Paste the text and the box in the image

            image = np.where(image_text == (255,255,255), (255,255,255), image) 
            image = np.where(image_text == (1,1,1), (0,0,0), image) 
            

        else:
            # Detect the corners' position of the text 
            position = i[0]        
            xi = min([position[0][0], position[3][0]])
            xf = max([position[1][0], position[2][0]])
            yi = min([position[0][1], position[1][1]])
            yf = max([position[2][1], position[3][1]])
            pointi = (int(xi), int(yi) )
            pointf = (int(xf), int(yf) )

            # Create a box on the text
            color = (255., 255., 255.)

            image = cv2.rectangle(image, pointi, pointf, color, -1)      #print the box

            scale = text_size(i[-1][0], max([h_long,v_long]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, thickness=6) # Select the text size
            image = cv2.putText(image, i[-1][0], (int(xi) , int(yf) - int((yf-yi)//4) ), cv2.FONT_HERSHEY_SIMPLEX, scale, (100,100,100), int(scale*2) ) #print the text
        ################################
    
    image = cv2.resize(image, (size[0], size[1]), interpolation = cv2.INTER_AREA)
    return image

# output = image2txt('/Users/m.khabazi/Desktop/BSI/last_version/xr5.jpeg')
# txt = ''
# for text in output:
#             text = text[-1][0]
#             txt += text + "\n"

# print(txt)

# image = rewrite_image(output, '/Users/m.khabazi/Desktop/BSI/last_version/xr5.jpeg')
# plt.imshow(image)
# plt.show()
# plt.close()


