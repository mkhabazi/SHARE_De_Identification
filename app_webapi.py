
import datetime
import codecs
from neuroner.conll_to_brat import output_entities
import json
from neuroner import utils
import pdf_ocr  # this function take a np.array and return a the text
# import pydicom as dicom
import tensorflow
# import dicom as dc
import image_ocr as iocr
import os
from neuroner import neuromodel
from flask import Flask,render_template,url_for,request,jsonify,send_file
#import jwt
from functools import wraps
import cv2
##########################

#print('okay 1')
def read_txt(file):
    file = open(file, 'r').read()                   #Import the .txt file 
    return file

def pdf2txt(file):
    file = pdf_ocr.pdf_to_img(file)                 #Import pdf
    output = pdf_ocr.print_pages(file, pdf=True)    #Extract text from image

    return output

def import_file():
     # Are we reciving something?
    if 'file' not in request.files:
            return jsonify({'message' : 'Any file'})

    file = request.files['file']
    print(file)
    filename = file.filename
    file_name = str(filename)
    file.save(file_name)
    return filename, file_name


app = Flask(__name__)
#app.config['SECRET_KEY']='Th1s1ss3cr3t'
nn = neuromodel.NeuroNER()
'''
def token_required(f):  
    @wraps(f)  
    def decorator(*args, **kwargs):

        token = None 

        if 'x-access-tokens' in request.headers:  
            token = request.headers['x-access-tokens'] 


        if not token:  
            return jsonify({'message': 'a valid token is missing'})   

        data = jwt.decode(token, app.config['SECRET_KEY']) 
        if data['public_id'] !=  '0123456789':
            return jsonify({'message': 'token is invalid'})  

        return f(*args,  **kwargs)  
    return decorator 
'''
@app.route('/')
def home():
    #token = jwt.encode({'public_id': '0123456789', 'exp' : datetime.datetime.utcnow() + datetime.timedelta(minutes=999999)}, app.config['SECRET_KEY'])
    #return jsonify({'token' : token.decode('UTF-8')}) 
    return jsonify({'message' : 'hello'})



 
@app.route('/predict', methods=['POST'])
#@token_required
def predict():

    file, name = import_file()
    print(file, name)
    format = name.split('.')[-1]
    name = name.split('.')[0]
    print(format)

    if format == 'txt':
        
        text = read_txt(file)
        os.remove(file)
        tags = nn.predict(text)
        tags = str(tags)


    elif format == 'pdf':

        text = pdf2txt(file)
        os.remove(file)
        tags = nn.predict(text)
        tags = str(tags)

    
    elif format in ['jpeg', 'png', 'jpg', 'jpe', 'tiff', 'tif']:

        txt = ''
        output = iocr.image2txt(file)
        os.remove(file)
        #print('The text was detected')
        for text in output:
            text = text[-1][0]
            txt += text + " \n "

        tags = nn.predict(txt)
        tags = str(tags)
        
    
    with open('output/' + name + '_predict.txt', "w") as f:
        f.write(tags)

    return send_file('output/' + name + "_predict.txt", as_attachment=True)


@app.route('/deidentify', methods=['POST'])
#@token_required
def deidentify():

    #credential_level = request.args.get('credential_level', type=int)
    credential_level = 1
    
    file, name = import_file()
    format = name.split('.')[-1]
    name = name.split('.')[0]
    
    level_1_keep = []
    level_2_keep = ['DATE']
    level_3_keep = ['DATE', 'AGE']

    if credential_level == 1:
        keep_list = level_1_keep
    elif credential_level == 2:
        keep_list = level_2_keep
    elif credential_level == 3:
        keep_list = level_3_keep

    
    if format == 'txt':
        
        text = read_txt(file)
        os.remove(file) # Remove the original file 
        tags = nn.predict(text)
        output_sentence = text 
        dif = 0 # when we change a word by type, the positions of the other words after this one change 
        if credential_level != 4:
            for tag in tags:
                if (tag['type'] not in keep_list):
                    
                    #The text before the word + the type + the text after the word
                    output_sentence = output_sentence[:tag['start']+dif] + '[['+tag['type']+']]' + output_sentence[tag['end']+dif:]
                    dif += len(str('[['+tag['type']+']]')) - len(tag['text'])

        with open('output/' + name + '_deidentify.txt', "w") as f:
            f.write(output_sentence)
        return send_file('output/' + name + "_deidentify.txt", as_attachment=True)
        
    elif format == 'pdf':

        text = pdf2txt(file)
        os.remove(file)            # Remove the original file 
        tags = nn.predict(text)
        output_sentence = text
        dif = 0                    # when we change a word by type, the positions of the other words after this one change 
        if credential_level != 4:
            for tag in tags:
                if (tag['type'] not in keep_list):

                    #The text before the word + the type + the text after the word
                    output_sentence = output_sentence[:tag['start']+dif] + '[['+tag['type']+']]' + output_sentence[tag['end']+dif:]
                    dif += len(str('[['+tag['type']+']]')) - len(tag['text'])

        with open('output/' + name + '_deidentify.txt', "w") as f:
            f.write(output_sentence)
        return send_file('output/' + name + "_deidentify.txt", as_attachment=True)

    
    elif format in ['jpeg', 'png', 'jpg', 'jpe', 'tiff', 'tif']:

        txt = ''
        output = iocr.image2txt(file)
        

        for text in output:
            text = text[-1][0]
            txt += text + " \n "
        output_sentence = txt

        tags = nn.predict(txt)
        if credential_level != 4:
            for tag in tags:

                if (tag['type'] not in keep_list):
                    continue

                    #The text before the word + the type + the text after the word
                    # output_sentence = output_sentence[:tag['start']+dif] + '[['+tag['type']+']]' + output_sentence[tag['end']+dif:]
                    # dif += len(str('[['+tag['type']+']]')) - len(tag['text'])
                
                for i in range(len(output)):
                    text = output[i][-1][0]
                    if (tag['type'] not in keep_list):
                        output_sentence = text.replace(tag['text'], '[['+tag['type']+']]') #Is it possible that this line delete more info thats necessary?               
                        output[i][-1][0] = output_sentence
                    
        output = iocr.rewrite_image(output, file)
        os.remove(file)
        cv2.imwrite('output/' + name + '_deidentify.' + format, output)

        # with open('output/' + name + '_image_deidentify.txt', "w") as f:
        #     f.write(output_sentence)

        return jsonify({'message' : 'The image file and txt were saved in the output folder in EC2'})
        #return send_file('output/' + name + '_deidentify.' + format, as_attachment=True)
        

    # elif format in ['DCM', 'dcm']:
    #     file = dicom.read_file(file)
    #     os.remove(file)
    #     if credential_level != 4:
    #         output_sentence = dc.dicom_class(file, keep_list)

    #     #print(output_sentence)

    #     #output_sentence.save_as('output/' + name + "_deidentify.DCM")
    #     dicom.filewriter.write_file('output/' + name + "_deidentify.DCM", output_sentence)
    #     return jsonify({'message' : 'The DCM file was saved in the output folder in EC2'})


@app.route('/createBratFiles', methods=['POST'])
#@token_required
def createBratFiles():

    file, name = import_file()
    format = name.split('.')[-1]
    name = name.split('.')[0]

    
    if format == 'txt':
        
        text = read_txt(file)
        tags = nn.predict(text)
        output_text_filepath = name + '_brat.txt'
        with codecs.open(output_text_filepath, 'w', 'UTF-8') as f:
            f.write(text)

        entities = []
        for tag in tags:
            entity = {}
            entity['label'] = tag['type']
            entity['text'] = tag['text']
            entity['start'] = int(tag['start'])
            entity['end'] = int(tag['end'])
            entities.append(entity)

    
    elif format == 'pdf':

        text = pdf2txt(file)
        tags = nn.predict(text)
        output_text_filepath = name + '_brat.txt'
        with codecs.open(output_text_filepath, 'w', 'UTF-8') as f:
            f.write(text)

        entities = []
        for tag in tags:
            entity = {}
            entity['label'] = tag['type']
            entity['text'] = tag['text']
            entity['start'] = int(tag['start'])
            entity['end'] = int(tag['end'])
            entities.append(entity)



    # elif format == 'dicom':
    #     text = dicom.read_file(file)
    #     output_text_filepath = name + '_brat.txt'
    #     with codecs.open(output_text_filepath, 'w', 'UTF-8') as f:
    #         f.write(text)
        
    #     entities = []
    #     for element in file:
    #         entity = {}
    #         for i in dc.dictionaries:
    #             if str(element.tag) == i['VR']:
    #                 entity['label'] = i['value']
    #                 entity['text'] = element['text']
    #                 #entity['start'] = int(tag['start'])   # ?
    #                 #entity['end'] = int(tag['end'])       # ?
    #                 entities.append(entity)

    return send_file(output_entities('Brat', name, entities, output_text_filepath, text, overwrite=True))

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=4000)
