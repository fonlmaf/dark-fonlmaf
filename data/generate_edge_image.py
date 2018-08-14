import cv2
import os
import numpy as np
from PIL import Image
import scipy.signal as signal

class Hog_descriptor():
    def __init__(self,img,cell_size=4,bin_size=16):
        self.img = img
        self.img = np.sqrt(img / np.max(img))
        self.img = img * 255
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 / self.bin_size
        assert type(self.bin_size) == int, "bin_size should be integer,"
        assert type(self.cell_size) == int, "cell_size should be integer,"
        assert type(self.angle_unit) == int, "bin_size should be divisible by 360"

    def extract(self):
        height, width = self.img.shape
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)
        cell_gradient_vector = np.zeros((height / self.cell_size, width / self.cell_size, self.bin_size))
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)
        return cell_gradient_vector 

    def global_gradient(self):
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        return gradient_magnitude, gradient_angle
    


    def global_Laplace_gradient(self):
        def func(x,y,sigma=1):
            return 100*(1/(2*np.pi*sigma))*np.exp(-((x-2)**2+(y-2)**2)/(2.0*sigma**2))
        gaussian_op= np.fromfunction(func,(1,1),sigma=1.5)
        img_blur = signal.convolve2d(self.img,gaussian_op,mode="same")
        gradient_values = cv2.Laplacian(img_blur, cv2.CV_64F) #, ksize=1)
        #gradient_values = np.absolute(gradient_values)
        #gradient_values = np.uint8(gradient_values)
        return gradient_values

    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers

    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit) % self.bin_size
        mod = gradient_angle % self.angle_unit
        return idx, (idx + 1) % self.bin_size, mod


# import the directory file
pwd_path = os.getcwd()
dir_train= pwd_path + '/coco/trainvalno5k.txt'
dir_valid= pwd_path + '/coco/5k.txt'

fopen1=open(dir_train)
dirs_train=fopen1.readlines()
fopen2=open(dir_valid)
dirs_valid=fopen2.readlines()
fopen2.close()
dirs=dirs_train+dirs_valid

# generate and store the edge images
for dir in dirs:
    dir=dir.strip('\n')
    img = cv2.imread(dir,cv2.IMREAD_GRAYSCALE) 
    hog = Hog_descriptor(img)
    gradient_values = hog.global_Laplace_gradient()
    dir_w = dir.replace('2014/','2014_lap/')
    cv2.imwrite(dir_w,gradient_values) 

