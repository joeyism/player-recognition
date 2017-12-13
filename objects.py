from sklearn.cluster import KMeans
from multiprocessing import Pool
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
    coord = None
    main_color = None
    image = None
    histogram = None
    weight = None
    debug = 0

    def __init__(self, image, coord, weight, debug =0):
        self.weight = weight
        self.debug = debug
        self.coord = coord
        self.image = image[coord.y: coord.y + coord.h, coord.x: coord.x + coord.w]

    def process(self):

        boundaries = [
            ([17, 15, 100], [50, 56, 200]),
            ([86, 31, 4], [220, 88, 50]),
            ([25, 146, 190], [62, 174, 250]),
            ([103, 86, 65], [145, 133, 128]),
            ([80, 0, 0], [255, 133, 133]),
            ([200,200,200], [255, 255, 255])
        ]
        filtered_images = []
        colors = []
        scores = []
        color_score_pair = []

        def centroid_histogram(clt):
            numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
            (hist, _) = np.histogram(clt.labels_, bins = numLabels)
            hist = hist.astype("float")
            hist /= hist.sum()
            return hist

        def filter_by_boundary(img, boundary):
            lower, upper = boundary
            lower = np.array(lower, dtype = "uint8")
            upper = np.array(upper, dtype = "uint8")
     
            mask = cv2.inRange(img, lower, upper)
            output = cv2.bitwise_and(img, img, mask = mask)
            return output
     

        #center
        height, width, dim = self.image.shape

        #crop
        img = self.image[int(height/4):int(2*height/4), int(width/4):int(3*width/4), :]
        self.image = img
        height, width, dim = img.shape

        # normalize
        normalizedImg = np.zeros((30, 30))
        self.image = cv2.normalize(self.image,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
        self.histogram = cv2.calcHist([self.image],[0],None,[256],[0,256])

        kmeans = KMeans(n_clusters=2)
        for j, boundary in enumerate(boundaries):
            img = filter_by_boundary(self.image, boundary)
            img_vec = np.reshape(img, [height * width, dim] )
            filtered_images.append(img)

            #kmeans
            kmeans.fit( img_vec )
            score = centroid_histogram(kmeans)

            for i, center in enumerate(kmeans.cluster_centers_):
                color = (int(center[0]), int(center[1]), int(center[2]))
                if color == (0,0,0):
                    continue
                colors.append(color)
                scores.append(score[i])



        self.colors = colors
        try:
            self.main_color = self.colors[np.argsort(scores)[-1]]
        except:
            self.main_color = (255,255,255)

        #print(self.colors)
        #print(scores)
        #print(self.main_color)
        #print("\n")

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
