# THIS FUNCTION READS A FOLDER, SORTS THE FILES AND IMPORTS THEM
import os

#This function creates an output folder if it does not exist
def output_folder(folder_path):
    output_folder = os.path.join(folder_path, 'output')
    if os.path.isdir(output_folder):
        print('The folder exist')
    else:
        os.mkdir(output_folder)
        print('The following folder has been created')

# This function creates a list per type of file
def files_list(folder_path):
    files = os.listdir(folder_path)
    txt_files = []
    images_files = []
    pdf_files = []
    dicom_files = []
    for file in os.listdir(folder_path):
        extension = file.split('.')[-1]

        if extension == 'txt':
            txt_files.append(os.path.join(folder_path, file))
        elif extension == 'pdf':
            pdf_files.append(os.path.join(folder_path, file))
        elif extension in ['jpeg', 'png']:
            images_files.append(os.path.join(folder_path, file))
        elif extension == 'dicom':
            dicom_files.append(os.path.join(folder_path, file))
        
    return txt_files, images_files, pdf_files, dicom_files


        




path = '/Users/a.uceda/Documents/AI/ai-master/last version/documents'
txt_files, images_files, pdf_files, dicom_files = files_list(path)
print('txt:', txt_files)
print('pdf:', pdf_files)
print('image:', images_files)
print('dicom:', dicom_files)
#files_detections( path )
