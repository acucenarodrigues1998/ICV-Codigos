import pydicom as dicom
from glob import glob
import os, os.path
import numpy as np

#Funcao que carrega as imagens e armazena-as em uma lista
def load_scan(path):
    slices = []
    for image in path:
        # Read in each image and construct 3d arrays
        ds = dicom.dcmread(image)
        slices.append(ds)
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
    print(len(slices))
        
    return slices

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


image_directory = "/home/pavic/Imagens/LIDC-IDRI-Incomplete/*"
output_path = "/home/pavic/Imagens/NPY/"
path_to_images = glob(image_directory)
path_to_images = sorted(path_to_images)
largest_stack = 0
patient_path = []

for patient in path_to_images:
    subdirs = glob(patient + "/*")
    # 1st Branch
    first_subdir = glob(subdirs[0]+"/*")[0]
    first_len = len([name for name in os.listdir(first_subdir + "/")])
    # 2nd Branch
    second_subdir = glob(subdirs[-1] + "/*")[0]
    second_len = len([name for name in os.listdir(second_subdir + "/")])

    # If 1st longer then build structure from 1st data
    # If 2nd longer then build structure from 2nd data
    if first_len > second_len:
        patient_image_folder = first_subdir + "/*.dcm"
    else:
        patient_image_folder = second_subdir + "/*.dcm"
        
    patient_path.append(glob(patient_image_folder))


for id_exam, exam in enumerate(patient_path):
    patient = load_scan(exam)
    id=patient[0].PatientID
    print(str(id_exam)+" - Patient "+id)
    imgs = get_pixels_hu(patient)
    data = []
    for im in patient:
        data.append(str(float(im.SliceLocation))+"\n")
    imgs = get_pixels_hu(patient)
    np.save(output_path + "full_images/fullimages_%s.npy" % (id), imgs)
    arquivo = open('/home/pavic/Imagens/NPY/SliceLocation/SL_'+id+'.txt', 'w')
    arquivo.writelines(data)
    arquivo.close()
