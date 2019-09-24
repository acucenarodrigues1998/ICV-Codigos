#from builtins import sorted, int, str, enumerate, open, float

import numpy as np
from glob import glob
from skimage import morphology, measure
from sklearn.cluster import KMeans
from skimage.segmentation import slic
import os


# Padronização dos valores dos pixels
def make_lungmask(img, display=False):
    row_size = img.shape[0]
    col_size = img.shape[1]

    mean = np.mean(img)
    std = np.std(img)
    img = img - mean
    img = img / std
    # Encontre o valor médio de pixel perto
    # dos pulmões para renormalizar imagens desbotadas
    middle = img[int(col_size / 5):int(col_size / 5 * 4), int(row_size / 5):int(row_size / 5 * 4)]
    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)
    # Para melhorar a descoberta do limite, estou movendo o underflow e overflow no espectro de pixels
    img[img == max] = mean
    img[img == min] = mean
    #
    # Usando KMeans para separar o primeiro plano (tecido mole / osso) e fundo (pulmão / ar)
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image

    # Primeiro corroer os elementos mais finos, depois dilatar para incluir alguns dos pixels em torno do pulmão.
    # Nós não queremos acidentalmente cortar o pulmão.

    eroded = morphology.erosion(thresh_img, np.ones([3, 3]))
    dilation = morphology.dilation(eroded, np.ones([8, 8]))

    labels = measure.label(dilation)  # Etiquetas diferentes são exibidas em cores diferentes
    #label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2] - B[0] < row_size / 10 * 9 and B[3] - B[1] < col_size / 10 * 9 and B[0] > row_size / 5 and B[
            2] < col_size / 5 * 4:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size, col_size], dtype=np.int8)
    mask[:] = 0

    #
    #  Depois que apenas os pulmões são deixados, fazemos outra grande dilatação para preencher e retirar a máscara do pulmão
    #
    for N in good_labels:
        mask = mask + np.where(labels == N, 1, 0)
    # mask = morphology.dilation(mask,np.ones([10,10])) # uma última dilatação

    par = mask * img

    _im = par * 20
    segments = slic(_im, n_segments=100, compactness=10, sigma=1)
    label_image = segments * mask

    return label_image


def create_regions_lists(img, img_org, loc):
    regions = []
    labels = np.unique(img)
    
    for i in labels:
        if i != 0:
            temp = np.zeros([512, 512])
            temp = temp + np.where(img == i, 1.0, 0.0)
            reg = img_org * temp
            regions.append(reg)
    regions.append(loc)

    return regions


data_path = '/home/pavic/Imagens/NPY/'
output_path = working_path = "/media/pavic/DISPOSITIVO/RegioesSlic/"
imagens = glob(data_path + 'full_images/*.npy')
slice_locations = glob(data_path+"SliceLocation/*.txt")
imagens = sorted(imagens)
slice_locations = sorted(slice_locations)

for id_img, exam in enumerate(imagens):
    imgs_to_process = np.load(exam)
    print(len(imgs_to_process))
    arquivo = open(slice_locations[id_img], 'r')
    listaLocations = arquivo.readlines()
    arquivo.close()
    image_labels = []
    id = exam.split('/')[6].split('.')[0]
    for id_img1, img in enumerate(imgs_to_process):
        print("Exame: " + id +" Slice: " + str(id_img1+1))
        image_labels.append(make_lungmask(img))

    caminho = output_path + id
    os.mkdir(caminho)
    for id_img2, img in enumerate(image_labels):
        print(listaLocations[id_img2])
        print("Salvando Patient " + id +": " + str(listaLocations[id_img2]))
        np.save(caminho + "/slice_%s.npy" % (str(id_img2)), create_regions_lists(img, imgs_to_process[id_img2], listaLocations[id_img2]))



