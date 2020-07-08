import os
import itertools as it
import cv2
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops, multiblock_lbp, hog
import numpy as np
import pickle as pkl
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
from skimage import filters
from tqdm import tqdm
import itertools
from skimage import feature
import mahotas
import random

_log2 = np.log(2)
_log2_sq = _log2 * _log2
sqrt_log2 = np.sqrt(_log2)
_2pi = 2 * np.pi


def gabor_kernel(p, q, L, sx, sy, u0, alpha, M):
    sx2 = sx * sx
    sy2 = sy * sy
    c = (M + 1) / 2
    ap = alpha ** (-p)
    tq = np.pi * q / L
    cos_tq = np.cos(tq)
    sin_tq = np.sin(tq)
    f_exp = 2 * np.pi * 1j * u0

    X = ap * np.repeat(np.arange(M) - c, M).reshape(M, M)
    Y = X.T

    _X = X * cos_tq + Y * sin_tq
    _Y = Y * cos_tq - X * sin_tq

    f = np.exp(-.5 * (_X * _X / sx2 + _Y * _Y / sy2)) * np.exp(f_exp * _X)
    return f * ap / (2 * np.pi * sx * sy)


def gabor_features(image, region=None, *, rotations=8, dilations=8, freq_h=2, freq_l=.1, mask=21, show=False, labels=False):
    '''\
    gabor_features(image, region=None, *, rotations=8, dilations=8, freq_h=2, freq_l=.1, mask=21, show=False, labels=False)
    (TODO)
    Parameters
    ----------
    Returns
    -------
    See Also
    --------
    Examples
    --------
    '''

    if mask % 2 == 0:
        raise ValueError(
            "`mask` value must be an odd positive integer, not '{mask}'")

    if region is None:
        region = np.ones_like(image)

    if show:
        print('--- extracting Gabor features...')

    alpha = (freq_h / freq_l) ** (1 / (dilations - 1))
    sx = sqrt_log2 * (alpha + 1) / (2 * np.pi * freq_h * (alpha-1))
    sy = sqrt_log2 - (2*_log2 / (_2pi * sx * freq_h))**2 /\
                     (_2pi * np.tan(np.pi / (2 * rotations)) *
                      (freq_h - 2 * np.log(1/4/np.pi**2/sx**2/freq_h)))
    u0 = freq_h

    k = np.where(region.astype(bool))
    N, M = image.shape

    g = np.zeros((dilations, rotations))
    size_out = image.shape + np.repeat(mask, 2) - 1
    Iw = np.fft.fft2(image, size_out)
    n1 = (mask + 1) // 2

    for p, q in it.product(range(dilations), range(rotations)):
        f = gabor_kernel(p, q, rotations, sx, sy, u0, alpha, mask)
        Ir = np.real(np.fft.ifft2(Iw * np.fft.fft2(np.real(f), size_out)))
        Ii = np.real(np.fft.ifft2(Iw * np.fft.fft2(np.imag(f), size_out)))
        Ir = Ir[n1:n1+N, n1:n1+M]
        Ii = Ii[n1:n1+N, n1:n1+M]
        Iout = np.sqrt(Ir*Ir + Ii*Ii)
        g[p, q] = Iout[k].mean()

    gmax = g.max()
    gmin = g.min()
    J = (gmax - gmin) / gmin
    features = np.hstack([g.ravel(), gmax, gmin, J])

    if labels:
        gabor_labels = np.hstack([
            [f'Gabor({p},{q})' for p, q in it.product(
                range(dilations), range(rotations))],
            ['Gabor-max', 'Gabor-min', 'Gabor-J']
        ])

        return gabor_labels, features

    return features

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def noisy_crop(img, y1, y2, x1, x2, noise):
    h, w = img.shape[0], img.shape[1]
    dy = y2 - y1
    dx = x2 - x1
    assert dy > 0
    assert dx > 0
    y1 = np.random.randint(max(y1 - noise, 0), y1 + min(h - y2, noise) + 1)
    y2 = y1 + dy
    x1 = np.random.randint(max(x1 - noise, 0), x1 + min(w - x2, noise) + 1)
    x2 = x1 + dx
    assert 0 <= y1 == y2 - dy < h
    assert 0 <= x1 == x2 - dx < w
    crop = img[y1: y2, x1: x2]
    return crop

def random_crop(img, n_crops=5, noise=10):
    old_shape = img.shape[:2]
    list_crops = []
    list_crops.append(cv2.resize(noisy_crop(img, 0,110,45,200, 0), old_shape))
    for _ in range(1, n_crops):        
        img2 = noisy_crop(img, 0,110,45,200, noise)
        if random.randint(0, 1):
            img2 = rotate_image(img2, 4)
            img2 = img2[3:old_shape[0]-3, 3:old_shape[1]-3]
        if random.randint(0, 1):
            img2 = cv2.flip(img2, 1)
        img2 = cv2.resize(img2, old_shape)
        list_crops.append(img2)
    return list_crops

def get_features_images(root, imgs, path_load, methods = ['lbp', 'glcm', 'EV', 'GABOR', 'SEGL', 'HOG'], full_face = True, crop_enable = False, n_crops = 5, noise = 20):
    assert path_load is not None
    labels = []
    features = []
    images_t = []
    # Se leen las imágenes de cada una de las clases y se obtienen las features para cada una de ella.
    if path_load is not None and os.path.exists(path_load):
        with open(path_load, "rb") as input_file:
            features, labels, images_t = pkl.load(input_file)
        print("feature file loaded")
    else: 
        for img in tqdm(imgs):
            class_id = img.split("_")[0]
            class_id = class_id.replace("FM","")
            class_id = int(class_id) - 1
            list_img = []
            img_info = cv2.imread(os.path.join(root, img), cv2.IMREAD_COLOR)
            if full_face:
                list_img.append(img_info)
            if crop_enable:
                crops = random_crop(img_info, n_crops, noise)
                list_img.extend(crops)
            for e, img_s in enumerate(list_img):
                feature_img = compute_features(img_s, methods)
                features.append(feature_img)
                labels.append(class_id)
                images_t.append(cv2.cvtColor(img_s, cv2.COLOR_RGB2GRAY))

        features, labels, images_t = np.array(features), np.array(labels), np.array(images_t)
        with open(path_load, "wb") as output_file:
            info = features, labels, images_t
            pkl.dump(info, output_file)
    
    return features, labels, images_t


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
 
    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = local_binary_pattern(image, self.numPoints,
            self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, self.numPoints + 3),
            range=(0, self.numPoints + 2))
 
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
 
        # return the histogram of Local Binary Patterns
        return hist


def compute_features(img, methods = ['lbp', 'glcm', 'EV', 'GABOR', 'SEGL', 'HOG']):
    # Entrega las features por imagen
    # img = cv2.imread(path_img, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    edge = filters.sobel(img_gray)
    
    list_features = ()
    if 'lbp' in methods:
        lbp_method = LocalBinaryPatterns(100, 1)
        # LBP
        lbp = lbp_method.describe(img_gray)
        lbp_b = lbp_method.describe(img[:,:,0])
        lbp_g = lbp_method.describe(img[:,:,1])
        lbp_r = lbp_method.describe(img[:,:,2])

        # n_bins = int(lbp.max() + 1)
        # hist_lbp, _ = np.histogram(lbp, density=False, bins=n_bins, range=(0, n_bins))

        # n_bins_b = int(lbp_b.max() + 1)
        # hist_lbp_p, _ = np.histogram(lbp_b, density=False, bins=n_bins_b, range=(0, n_bins_b))

        # n_bins_g = int(lbp_g.max() + 1)
        # hist_lbp_g, _ = np.histogram(lbp_g, density=False, bins=n_bins_g, range=(0, n_bins_g))

        # n_bins_r = int(lbp_r.max() + 1)
        # hist_lbp_r, _ = np.histogram(lbp_r, density=False, bins=n_bins_r, range=(0, n_bins_r))

        list_features += (lbp,lbp_b,lbp_g,lbp_r)
    
    if 'lbp_p' in methods:
        windowsize_r = 16
        windowsize_c = 16
        for r in range(0,img.shape[0] - windowsize_r, windowsize_r):
            for c in range(0,img.shape[1] - windowsize_c, windowsize_c):
                window = img[r:r+windowsize_r,c:c+windowsize_c]
                window_gray = cv2.cvtColor(window, cv2.COLOR_RGB2GRAY)

                lbp_method = LocalBinaryPatterns(100, 1)
                # LBP
                lbp = lbp_method.describe(window_gray)
                lbp_b = lbp_method.describe(window[:,:,0])
                lbp_g = lbp_method.describe(window[:,:,1])
                lbp_r = lbp_method.describe(window[:,:,2])
                
                list_features += (lbp,lbp_b,lbp_g,lbp_r)
    
    if 'glcm' in methods:

        #GLCM (Haralick)
        GLCM = greycomatrix(img_gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
        GLCM_inv = np.expand_dims(np.mean(GLCM, axis = 3), axis=3)
        contrast = greycoprops(GLCM_inv, 'contrast')[0]
        dissimilarity = greycoprops(GLCM_inv, 'dissimilarity')[0]
        homogeneity = greycoprops(GLCM_inv, 'homogeneity')[0]
        energy = greycoprops(GLCM_inv, 'energy')[0]
        correlation = greycoprops(GLCM_inv, 'correlation')[0]
        ASM = greycoprops(GLCM_inv, 'ASM')[0]

        list_features += (contrast,dissimilarity,homogeneity,energy,correlation,ASM)

    if 'lbp' in methods and 'glcm' in methods and 'EV' in methods:

        lbp_energy = np.sum(lbp**2)
        image_energy = np.sum(img_gray**2)
        edge_energy = np.sum(edge**2)
        
        #ENERGY VARIATION
        energy_variation = [abs(lbp_energy - image_energy), abs(ASM - image_energy), abs(edge_energy - image_energy)]

        list_features += (energy_variation,)

    if 'GABOR' in methods:
        #GABOR
        features = gabor_features(img_gray)
        list_features += (features,)

    if 'SEGL' in methods and 'lbp' in methods:
        #SEGL
        GLCM_SEGL = greycomatrix(lbp.astype(np.uint8), [1], [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
        GLCM_SEGL = np.concatenate((GLCM_SEGL, np.expand_dims(np.mean(GLCM_SEGL, axis = 3), axis=3)),axis=3)
        features_SEGL = np.zeros((GLCM_SEGL.shape[3], 6))
        for i in range(GLCM_SEGL.shape[3]):
            SEGL_i = GLCM_SEGL[:,:,0,i]
            edge_SEGL = filters.sobel(SEGL_i)
            edge_SEGL = np.expand_dims(np.expand_dims(edge_SEGL, axis=2), axis = 3)
            features_SEGL[i,0] = greycoprops(edge_SEGL, 'contrast')[0]
            features_SEGL[i,1] = greycoprops(edge_SEGL, 'dissimilarity')[0]
            features_SEGL[i,2] = greycoprops(edge_SEGL, 'homogeneity')[0]
            features_SEGL[i,3] = greycoprops(edge_SEGL, 'energy')[0]
            features_SEGL[i,4] = greycoprops(edge_SEGL, 'correlation')[0]
            features_SEGL[i,5] = greycoprops(edge_SEGL, 'ASM')[0]
        features_SEGL = features_SEGL.ravel()

        list_features += (features_SEGL,)

    if 'HOG' in methods:
        #HOG
        fd = hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(3, 3), block_norm = 'L2-Hys')
        list_features += (fd,)
    
    if 'HOG_p' in methods:
        windowsize_r = 16
        windowsize_c = 16
        for r in range(0,img.shape[0] - windowsize_r, windowsize_r):
            for c in range(0,img.shape[1] - windowsize_c, windowsize_c):
                window = img[r:r+windowsize_r,c:c+windowsize_c]
                window_gray = cv2.cvtColor(window, cv2.COLOR_RGB2GRAY)

                fd_p = hog(window_gray, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(3, 3), block_norm = 'L2-Hys')
                list_features += (fd_p,)
    
    if 'zernike' in methods:
        degrees = [0,1,2,3,4,5,6,7,8,9,10]
        for d in degrees:
            moments = mahotas.features.zernike_moments(img_gray, radius = 100, degree = d)
            list_features += (moments,)

    if 'zernike_p' in methods:
        windowsize_r = 16
        windowsize_c = 16
        for r in range(0,img.shape[0] - windowsize_r, windowsize_r):
            for c in range(0,img.shape[1] - windowsize_c, windowsize_c):
                window = img[r:r+windowsize_r,c:c+windowsize_c]
                window_gray = cv2.cvtColor(window, cv2.COLOR_RGB2GRAY)

                degrees = [0,1,2,3,4,5,6,7,8,9,10]
                for d in degrees:
                    moments_p = mahotas.features.zernike_moments(window_gray, radius = 100, degree = d)
                    list_features += (moments_p,)
    
    #Concat Features
    if 'fourier' in methods:
        f = np.fft.fft2(img_gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        magnitude_spectrum = magnitude_spectrum.ravel()
        list_features += (magnitude_spectrum,)

    X_feature = np.concatenate(list_features)
    # if X_feature.shape[0] > 200:
    #     X_feature = X_feature[:200]
    
    return X_feature


def create_dataset(root="FaceMask166", methods=['lbp'], split='A',
                   path_template_train = 'features_split_{}_ff_{}_{}_cropenable_{}_ncrops_{}_noise_{}.pkl',
                   path_template_val_test = 'features_split_{}_ff_{}_{}.pkl',
                   full_face = True, crop_enable = False, n_crops = 10, noise = 20):

    archivos = [x for x in os.listdir(root) if ".DS" not in x]
    desglosados = [x.replace("FM000", "").replace(".jpg", "") for x in archivos]
    desglosados = [x.split("_") for x in desglosados]
    desglosados = [(int(x), int(y)) for (x, y) in desglosados]

    ds_size = {'A': 16, 'B': 40, 'C': 100, 'D': 166}
    train_ids = [i for i, (x, y) in enumerate(desglosados) if x <= ds_size[split] and y <= 3]
    val_ids = [i for i, (x, y) in enumerate(desglosados) if x <= ds_size[split] and y == 4]
    test_ids = [i for i, (x, y) in enumerate(desglosados) if x <= ds_size[split] and 4 < y]

    train_imgs = [archivos[x] for x in train_ids]
    val_imgs = [archivos[x] for x in val_ids]
    test_imgs = [archivos[x] for x in test_ids]
    
    print("Features de Train")
    path_train_load = path_template_train.format(split, full_face, 'train', crop_enable, n_crops, noise)
    x_train, gt_train, img_train = get_features_images(root, train_imgs, path_train_load, methods, full_face, crop_enable, n_crops, noise)
    
    print("Features de Val")
    path_val_load = path_template_val_test.format(split, full_face, 'val')
    x_val, gt_val, img_val = get_features_images(root, val_imgs, path_val_load, methods, full_face)
    
    print("Features de Test")
    path_test_load = path_template_val_test.format(split, full_face, 'test')
    x_test, gt_test, img_test = get_features_images(root, test_imgs, path_test_load, methods, full_face)

    return x_train, gt_train, x_val, gt_val, x_test, gt_test, train_imgs, val_imgs, test_imgs, img_train, img_val, img_test