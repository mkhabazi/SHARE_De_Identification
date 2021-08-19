
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math 

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

##### Images used like example ##########
img_path ='xr5.jpeg'
################################

###### Preprocesing 1 image  ############
image_or = cv2.imread(img_path)     # import original image
image_or = cv2.flip(image_or, 1)    # USE FOR: xr4.jpeg => obtain the simetrical image 
#image_or = cv2.rotate(image_or, cv2.ROTATE_90_COUNTERCLOCKWISE) 
image_or = cv2.rotate(image_or, cv2.ROTATE_180)
#########################################

# Create 2 images copy
image = np.copy(image_or)         #image to show 
img = np.copy(image_or)           #image to extract and predict text

###### Preprocesing 2 image  ############
# image color to gray 
#img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#image bgr to rgb
image_or = cv2.cvtColor(image_or, cv2.COLOR_BGR2RGB)
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
print('1--------------image imported')

#This function use PaddleOCR to detect the text in the image and rewrite it 
def text_detection_paddleocr(img, image):
    from paddleocr import PaddleOCR, draw_ocr
    from PIL import Image

    #ocr = PaddleOCR(lang="en")
    #ocr = PaddleOCR(use_angle_cls=True, lang="en") # The model file will be downloaded automatically when executed for the first time
    ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)
    #ocr = PaddleOCR()
    
    # Detect the texts and position in the image
    result = ocr.ocr(img, cls=True)
    print('2--------------- text extract')
    #print(result)
    #result = ocr.ocr(img)

    #####THIS LIST WILL BE NECESARY IN THE INTEGRATION WITH THE PROJECT
    #output = []
    #text_areas = []
    #text = []
    #####################

    for i in result:

        #print(i)

        position = i[0]  #obtain the 4 corners to the text
        #I could not confirm but I think that the output structure is [[up-left corner], [up-right], [down-right], [down-left]]
        
        # Calculate the distance and angle between the corners
        x1_x2, x1_x2_angle = lines_carac(position[0], position[1])
        x4_x3, x4_x3_angle = lines_carac(position[3], position[2])

        h_long = (x1_x2 + x4_x3)/2              # "horizontal" longitude
        h_ang = (x1_x2_angle + x4_x3_angle)/2   # "horizontal" angle

        x2_x3, x2_x3_angle = lines_carac(position[1], position[2])
        x1_x4, x1_x4_angle = lines_carac(position[0], position[3])

        v_long = (x2_x3 + x1_x4)/2              # "vertical" longitude
        v_ang = (x2_x3_angle + x1_x4_angle)/2   # "vertical" angle
        
        # Detect the direction of the text and the angle by comparing the lengths.
        if h_long >= v_long:
            text_ang = -h_ang

            xi = position[3][0]
            yi = position[3][1]
            xf = position[1][0]
            yf = position[1][1]

            l = int(max(x1_x2, x4_x3)) # box longitude
            h = int(max(x2_x3, x1_x4)) # box hight

        else:
            text_ang = -v_ang

            xi = position[0][0]
            yi = position[0][1]
            xf = position[2][0]
            yf = position[2][1]

            h = int(max(x1_x2, x4_x3)) # box longitude
            l = int(max(x2_x3, x1_x4)) # box hight

        print('3----------characteristics ')
        ###### If the text his not horizontal
        if abs(text_ang) > 5:
            
            
            ########### create a black image with the box and text
            
            #image_text = np.zeros_like(image)
            image_text = np.zeros((image.shape[0], image.shape[1]+l, 3))                                                 # Array with the same dimenstion than the image
            image_text = cv2.rectangle(image_text, ( int(xi), int(yi) ), ( int(xi+l), int(yi-h)), (255,255,255), -1)  # Create the black rectangule
            #image_text = cv2.rectangle(image_text, ( 0, int(l) ), ( int(l), int(l-h)), (255.,255.,255.), -1)

            ### To understan the process with images uncoment this lines
            # plt.imshow(image_text)
            # plt.show()
            # plt.close()

            scale = text_size(i[-1][0], l, fontFace=cv2.FONT_HERSHEY_SIMPLEX, thickness=6)           # Detect the text size
            image_text = cv2.putText(image_text, i[-1][0], (int(xi), int(yi-(h/4))) , cv2.FONT_HERSHEY_SIMPLEX, scale, (1,1,1), int(scale*2) )  # Write the text
            #image_text = cv2.putText(image_text, i[-1][0], (0, int(l-(h/4)) ), cv2.FONT_HERSHEY_SIMPLEX, scale, (10.,10.,10.), int(scale*2) )  # Write the text

            ### To understan the process with images uncoment this lines
            # plt.imshow(image_text)
            # plt.show()
            # plt.close()

            # Rotate the box and text
            M = cv2.getRotationMatrix2D( (int(xi), int(yi)), text_ang, 1 ) 
            #M = cv2.getRotationMatrix2D( (0, l), text_ang, 1 ) 
            image_text = cv2.warpAffine( image_text, M, (image.shape[1]+l ,image.shape[0]) )
            image_text = image_text[:, :image.shape[1]]
            
            ### To understan the process with images uncoment this lines
            # plt.imshow(image_text)
            # plt.show()
            # plt.close()

            # f, arr = plt.subplots(1,2)
            # arr[0].imshow(image_text)
            # arr[1].imshow(image)
            # plt.show()
            
            ################ IN PROCESS ############
            # Paste the text and the box in the image
            # image = np.where(image_text == (255.,255.,255.), image_text, image) 
            # image = np.where(image_text == (100.,100.,100.), image_text, image) 

            image = np.where(image_text == (255,255,255), (255,255,255), image) 
            image = np.where(image_text == (1,1,1), (0,0,0), image) 

            # print(image.shape, image[int(yi-l):int(yi+l), int(xi):].shape, image_text.shape)
            # image = np.where(image_text == (255.,255.,255.), image_text, image[int(yi-l):int(yi+l), int(xi):int(xi+l)]) 
            # image[int(yi-l):int(yi+l), int(xi):int(xi+l)] = np.where(image_text == (100.,100.,100.), image_text, image[int(yi-l):int(yi+l), int(xi):int(xi+l)]) 

            # plt.imshow(image)
            # plt.show()
            # plt.close() 
            #################################            
            

        else:
            # Detect the corners position to the text 
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
    
    image = cv2.resize(image, (image_or.shape[1], image_or.shape[0]), interpolation = cv2.INTER_AREA)
    return image


def text_detection_paddleocr_2(image):
    from paddleocr import PaddleOCR, draw_ocr
    from PIL import Image

    ocr = PaddleOCR(lang='en') # need to run only once to download and load model into memory
    #result = ocr.ocr(img_path)
    result = ocr.ocr(image)
    # for line in result:
    #     print(line)

    # Visualization
    from PIL import Image
    #image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    print(boxes, txts, scores)
    im_show = draw_ocr(image, boxes, txts, scores)
    #im_show = Image.fromarray(im_show)
    #im_show.save('result.jpg')
    # for line in result:
    #     print(line)

    # draw result
    # image = Image.open(img_path).convert('RGB')
    # boxes = [line[0] for line in result]
    # txts = [line[1][0] for line in result]
    # scores = [line[1][1] for line in result]
    # im_show = draw_ocr(image, boxes, txts, scores, font_path='/path/to/PaddleOCR/doc/fonts/simfang.ttf')
    # im_show = Image.fromarray(im_show)
    # im_show.save('result.jpg')


def text_detection_tesseract(img, image):
    import pytesseract
    #print( pytesseract.image_to_string(img, config='--psm 3') )
    texts = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT )
    for x, y, w, h in zip(texts['left'], texts['top'], texts['width'], texts['height']):
        color = (255., 0, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    print(texts)
    plt.imshow(image)
    plt.show()



#text_detection_tesseract(img, image)
image = text_detection_paddleocr(img, image)
#image = text_detection_paddleocr_2(img_path)
#image = text_detection_paddleocr_2(img)
# f, arr = plt.subplots(1,2)
# arr[0].imshow(image)
# arr[1].imshow(image_or)
# plt.show()
print(image)
