from sklearn.cluster import KMeans
import numpy as np
import cv2


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
    main_color = None
    image = None
    histogram = None

    def __init__(self, image, coord):
 
        def centroid_histogram(clt):
            numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
            (hist, _) = np.histogram(clt.labels_, bins = numLabels)
            hist = hist.astype("float")
            hist /= hist.sum()
            return hist

        #center
        self.image = image[coord.y: coord.y + coord.h, coord.x: coord.x + coord.w]
        height, width, dim = self.image.shape

        #crop
        img = self.image[int(height/4):int(2*height/4), int(width/4):int(3*width/4), :]
        self.image = img

        height, width, dim = img.shape

        # normalize
        normalizedImg = np.zeros((38, 38))

        self.image = cv2.normalize(self.image,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
        self.histogram = cv2.calcHist([self.image],[0],None,[256],[0,256])

        #kmeans
        img_vec = np.reshape(img, [height * width, dim] )
        kmeans = KMeans(n_clusters=5)
        kmeans.fit( img_vec )

        colors = []
        for center in kmeans.cluster_centers_:
            colors.append((int(center[0]), int(center[1]), int(center[2])))
        self.colors = colors
        self.main_color = self.colors[np.argsort(centroid_histogram(kmeans))[-2]]

class ImageAddition(object):
    ellipse = None
    playerImage = None

    def __init__(self, ellipse, playerImage):
        self.ellipse = ellipse
        self.playerImage = playerImage

class CorrelationLine(object):
    pt1 = None
    pt2 = None
    score = None
    color = None

    def __init__(self, pt1, pt2, score, color):
        self.pt1 = pt1
        self.pt2 = pt2
        self.score = score
        self.color = color
     
class CorrelationLines(list):
    pass

class ImageAdditions(list):

    def get_histogram_correlation(self):
        corrs = CorrelationLines()
        for i, addition1 in enumerate(self):
            for j, addition2 in enumerate(self):
                if i >= j:
                    continue
                corrs.append(
                        CorrelationLine(
                            addition1.ellipse.center(), 
                            addition2.ellipse.center(), 
                            cv2.compareHist(
                                addition1.playerImage.histogram, 
                                addition2.playerImage.histogram, 
                                cv2.HISTCMP_CORREL
                                ),
                            addition1.playerImage.main_color
                            )
                        )
        return corrs
