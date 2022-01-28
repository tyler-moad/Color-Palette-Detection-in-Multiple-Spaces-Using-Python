from skimage.color import *
from skimage.io import *
from sklearn.metrics import pairwise_distances_argmin
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from PIL import ImageColor
from sklearn.isotonic import IsotonicRegression
import fnmatch

class Palette:
    n_colors = 0
    rgb = None
    #img = None
    weights = None
    hsv = None
    hues = None
    saturations = None
    values = None
    angles = None # on circle of Adobe
    xCoordinates = None # on circle of Adobe
    yCoordinates = None # on circle of Adobe
    adobeRepresentation = True
    cvtHue2Adobe = None
    #cvtAdobe2Hue = None
    
    def setAngles(self,angles):
        self.angles = angles
    
    def computeWeights(self, img):
        ## Computes ratio of pixels/total pixels, where pixels[i] are colors closest to colors i
        im_hsv = rgb2hsv(img)
        height, width = im_hsv.shape[:2]
        points = np.reshape(im_hsv,(height*width,3))
        labels = pairwise_distances_argmin(points,self.hsv)
        weights = np.zeros((self.n_colors,))
        for i  in range(self.n_colors):
            count_ = (labels == i).sum()
            weight_ = count_/len(points)
            weights[i] = weight_
        return weights
    
    def computeAngles(self):
        hues_ = self.hues * 360
        angles = self.cvtHue2Adobe.predict(hues_)
        self.angles = angles
    
    def computeCoordinates(self):
        x_ = self.saturations * np.cos(self.angles * 2 * np.pi / 360)
        y_ = self.saturations * np.sin(self.angles * 2 * np.pi / 360)
        self.xCoordinates = x_
        self.yCoordinates = y_
    
    def initAdobeConverter(self):
        circle_angles = [0,17.142,34.28,51.42,72.4,97.2,122,129.16,136.34,143.33,150,157.84,165,173.83,182.6,191.5,200.34,209.16,218,227.5,237,246.5,256,265.5,275,284,293.3,302.5,311.67,320.84,330,335,340,345,350,355,359.5]
        hues_angles = np.array([i*10  for i in range(36)]+[359])
        hue2circle = IsotonicRegression(increasing = True).fit(hues_angles,circle_angles)
        self.cvtHue2Adobe = hue2circle
        #circle2hue = IsotonicRegression(increasing = True).fit(circle_angles,hues_angles)
    """
    A refaire
    """
    def adobeColorWheel(self):
        hues = np.array([i*10  for i in range(36)]+[359])
        angles = [0,17.142,34.28,51.42,72.4,97.2,122,129.16,136.34,143.33,150,157.84,165,173.83,182.6,191.5,200.34,209.16,218,227.5,237,246.5,256,265.5,275,284,293.3,302.5,311.67,320.84,330,335,340,345,350,355,359.5]
        hue2angle = IsotonicRegression(increasing = True).fit(hues,angles)
        angle2hue = IsotonicRegression(increasing = True).fit(angles,hues)
        degree_samples = np.array([2*i for i in range(180)])
        hues_sample = angle2hue.predict(degree_samples)
        hues_sample , degree_samples = hues_sample/360 , degree_samples/360
        points_x, points_y = np.cos(2*np.pi*degree_samples), np.sin(2*np.pi*degree_samples)
        hsv_colors = np.hstack([hues_sample.reshape(-1,1),np.ones((len(hues_sample),2))])
        s_samples = [0.02 * i for i in range(1,50)]
        for s in s_samples:
            points_x_, points_y_ =  s * np.cos(2*np.pi*degree_samples), s * np.sin(2*np.pi*degree_samples)
            hsv_colors_ = np.hstack([hues_sample.reshape(-1,1),s * np.ones((len(hues_sample),1)),0.9 * np.ones((len(hues_sample),1))])
            points_x = np.vstack([points_x,points_x_])
            points_y = np.vstack([points_y,points_y_])
            hsv_colors = np.vstack([hsv_colors,hsv_colors_])
        points_c = hsv2rgb(hsv_colors)
        return points_x, points_y, points_c
    """
    """
    def __init__(self, rgb = None, img = None, hsv = None):
        
        if rgb is None and hsv is None:
            print("must input colors palette in rgb or HSV")
            
        else:
            if rgb is not None :
                self.rgb = rgb
            else: 
                self.rgb = (hsv2rgb(hsv)*256).astype(int)
            
            if hsv is not None:
                self.hsv = hsv
            else:
                self.hsv = rgb2hsv(rgb/256)

        self.n_colors = len(self.rgb)
        
        """if hsv is not None:
            self.hsv = hsv
        else:
            self.hsv = rgb2hsv(rgb/256)
        """
        self.hues = self.hsv[:,0]
        self.saturations = self.hsv[:,1]
        self.values = self.hsv[:,2]
        
        if img is not None:
            weights = self.computeWeights(img)
        else:
            weights = np.zeros((self.n_colors,)) + 1/self.n_colors
        self.weights = weights
        
        if self.adobeRepresentation:
            self.initAdobeConverter()
            self.computeAngles()
            self.computeCoordinates()
            
    def plot_palette(self):
        indices = np.array([[ i for i in range(self.n_colors)]])
        fig = plt.figure()
        plt.imshow(self.rgb[indices].astype('uint8'))
        plt.axis("off")
        plt.show()
        return fig
        
    def plot_circle(self):
        points_x,points_y,points_c = self.adobeColorWheel()
        plt.scatter(points_x,points_y, c = points_c , s =50)
        plt.scatter(self.xCoordinates, self.yCoordinates , c = self.rgb/256, s = 3000 * self.weights, edgecolors = "white", linewidths = 2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')
        plt.show()
        
    def summary(self):
        angles_ = self.angles
        weights_ = self.weights
        index = np.argmax(weights_)
        #sorted_weights = weights_[index] # everything sorted by weights
        #sorted_angles = angles_[index]
        #sorted_colors = self.hsv[index]
        ecart = angles_ - angles_[index]
        for i in range(len(ecart)):
            if ecart[i] >= 180:
                ecart[i] = 360 - ecart[i]
            elif ecart[i] <= -180:
                ecart[i] = 360 + ecart[i]
        
        data = dict()
        for i in range(self.n_colors):
            data["col_" + str(i+1) + "_h"] = self.hsv[i][0]
            data["col_" + str(i+1) + "_s"] = self.hsv[i][1]
            data["col_" + str(i+1) + "_v"] = self.hsv[i][2]
            data["col_" + str(i+1) + "_w"] = self.weights[i]
            data["col_" + str(i+2) + "_ecart_angulaire"] = ecart[i]
        return data
