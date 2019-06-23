# Grayscale/color Canny edge detection
# author: Muhammet Bastan, mubastan@gmail.com
# date: March 2012

import numpy, scipy, scipy.ndimage
from scipy.ndimage import filters
from scipy.ndimage import measurements
from numpy import *
from gradient import *
import sys
from pylab import *

def getFractile(nms, th=1.0, fraction=0.30, bins=255):
    nzrind = nms >= th
    nzvals = nms[nzrind]
    minVal = nzvals.min()
    maxVal = nzvals.max()
    
    #figure(); hist(nzvals, bins=bins); draw()    
    H, e = numpy.histogram(nzvals, bins)
    
    nzr_frac = fraction*len(nzvals)    
    sum = 0.0
    i = 0
    while i < bins and sum < nzr_frac:
        sum += H[i]
        i += 1
    return (maxVal-minVal)*i/float(bins) + minVal


## nonmaximum suppression
# Gm: gradient magnitudes
# Gd: gradient directions, -pi/2 to +pi/2
# return: nms, gradient magnitude if local max, 0 otherwise
def nonmaxsupress(Gm, Gd, th=1.0):
    nms = zeros(Gm.shape, Gm.dtype)   
    h,w = Gm.shape    
    for x in range(1, w-1):
        for y in range(1, h-1):            
            mag = Gm[y,x]
            if mag < th: continue        
            teta = Gd[y,x]            
            dx, dy = 0, -1      # abs(orient) >= 1.1781, teta < -67.5 degrees and teta > 67.5 degrees
            if abs(teta) <= 0.3927: dx, dy = 1, 0       # -22.5 <= teta <= 22.5
            elif teta < 1.1781 and teta > 0.3927: dx, dy = 1, 1     # 22.5 < teta < 67.5 degrees
            elif teta > -1.1781 and teta < -0.3927: dx, dy = 1, -1  # -67.5 < teta < -22.5 degrees            
            if mag > Gm[y+dy,x+dx] and mag > Gm[y-dy,x-dx]: nms[y,x] = mag    
    return nms

### hysteresis thresholding
def hysteresisThreshold(nms, thLow, thHigh, binaryEdge = True):
    labels, n = measurements.label(nms > thLow, structure=ones((3,3)))
    for i in range(1, n):
        upper = amax(nms[labels==i])
        if upper < thHigh: labels[labels==i] = 0
    if binaryEdge: return 255*(labels>0)        
    else: return nms*(labels>0)

def detect_gray(image, thLow, thHigh, binaryEdge=True):
    Gm, Gd = gray_gradient(image)
    #print 'Gm max:', Gm.max(), mean(Gm)
    nms = nonmaxsupress(Gm, Gd, th=1.0)
    #fr = getFractile(nms, th=1.0, fraction=0.30, bins=255)
    #print 'Fractile:', fr, thLow, thHigh
    edge = hysteresisThreshold(nms, thLow, thHigh, binaryEdge)    
    return edge, nms

def detect_rgb(image, thLow, thHigh, gtype=0, binaryEdge=True):
    Gm, Gd = rgb_gradient(image, gtype) 
    #print 'Gm max:', Gm.max(), mean(Gm)
    nms = nonmaxsupress(Gm, Gd, th=1.0)
    #fr = getFractile(nms, th=1.0, fraction=0.50, bins=255)
    #print 'Fractile:', fr, thLow, thHigh
    edge = hysteresisThreshold(nms, thLow, thHigh, binaryEdge)
    return edge, nms

def gradient_vector(image):
    return rgb_gradient_vector(image)

def detect_multichannel(images, thLow, thHigh, gtype=0, binaryEdge=True):
    Gm, Gd = multi_gradient(images, gtype) 
    #print 'Gm max:', Gm.max(), mean(Gm)
    nms = nonmaxsupress(Gm, Gd, th=1.0)
    #fr = getFractile(nms, th=1.0, fraction=0.50, bins=255)
    #print 'Fractile:', fr, thLow, thHigh
    edge = hysteresisThreshold(nms, thLow, thHigh, binaryEdge)
    return edge, nms
  

def edge_detection_rgb(orig_image):
    
    # image = scipy.misc.imread(orig_image, False)
    # plt.figure()
    # plt.imshow(orig_image[:,:])
    # plt.show()
    image = orig_image
    sigma = 0.5
    image = scipy.ndimage.filters.gaussian_filter(image, sigma)
    
    imgMax = image.max()
    if imgMax > 255:
        image *= (255.0/65535)
        print 'Image max value:', imgMax
    
    tlow = 200
    thigh = 300
    # if argc == 4:
    #     tlow = float(sys.argv[2])
    #     thigh = float(sys.argv[3])
    
    edge = detect_rgb(image, tlow, thigh, 1)
    edge = edge[0]
    # plt.figure()
    # plt.imshow(edge[:,:])
    # plt.show()
    return edge

if __name__ == '__main__':
    edge_detection_rgb('/home/silvia/Desktop/edge_det_test.png')
    