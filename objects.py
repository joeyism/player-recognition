from sklearn.cluster import KMeans
import numpy as np


class Rect(object):
    x = None
    y = None
    w = None
    h = None

    def __init__(self, l):
        self.x = l[0]
        self.y = l[1]
        self.w = l[2]
        self.h = l[3]

class Ellipse(object):
    x = None
    y = None
    w = None
    h = None
    
    def __init__(self, rect):
        x = rect.x
        y = rect.y
        w = rect.w
        h = rect.h
        
        self.x = int(x + w/2)
        self.y = int(y + h*7/8)
        self.w = int(w*1/2)
        self.h = int(h/10)
    
    def center(self):
        return (self.x, self.y)
    
    def axes(self):
        return (self.w, self.h)

class PlayerImage(object):
    colors = None
    image = None

    def __init__(self, image, coord):
        self.image = image[coord.y: coord.y + coord.h, coord.x: coord.x + coord.w]
        height, width, dim = self.image.shape

        img_vec = np.reshape(self.image, [height * width, dim] )
        kmeans = KMeans(n_clusters=3)
        kmeans.fit( img_vec )

        colors = []
        for center in kmeans.cluster_centers_:
            colors.append((int(center[0]), int(center[1]), int(center[2])))
        self.colors = colors

