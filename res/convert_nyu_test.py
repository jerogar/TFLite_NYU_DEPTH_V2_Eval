import numpy as np
import cv2
from zipfile import ZipFile
from io import BytesIO

def extract_zip(input_zip):
    input_zip=ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}

data = extract_zip('nyu_test.zip')

rgb = np.load(BytesIO(data['eigen_test_rgb.npy']))
depth = np.load(BytesIO(data['eigen_test_depth.npy']))
crop = np.load(BytesIO(data['eigen_test_crop.npy']))

for iter in range(rgb.shape[0]):
    cv2.imwrite("img/"+"source" +str(iter).zfill(5) +".jpg", cv2.cvtColor(rgb[iter,:,:,:], cv2.COLOR_BGR2RGB))
    np.savetxt("gt/gt"+str(iter).zfill(5) + ".csv", depth[iter,:,:], fmt='%f', delimiter=',')

print('convert comp.\n')
