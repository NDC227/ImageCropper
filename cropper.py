import cv2 as cv
import numpy as np
from shapely.geometry import Polygon
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from pathlib import Path

PROCESS_RES = (480, 480)
GAUSSIAN_K_SIZE = (3, 3)
CANNY_LOWER = 100
CANNY_UPPER = 200
STRUCTURE_ELEMENT_SIZE = (3, 3) #(7, 7)
MIN_CONTOUR_AREA = 0
DISTANCE_THRESHOLD_SCALE = 0.16
AREA_PERCENTILE_THRESH = 40

def centerAndCropImage(image):
    # Read in image and get size info
    # Also make copy for drawing on
    img = cv.imread(image)
    height, width = img.shape[:-1]
    target_size = [min(height, width), min(height, width)]
    resize_scale = [width / PROCESS_RES[1], height / PROCESS_RES[0]]
    img_copy = img.copy()

    # Image pre-processing to prepare for edge detection
    def preprocess(img):
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_gray = cv.resize(img_gray, PROCESS_RES)
        img_gray = cv.GaussianBlur(img_gray, GAUSSIAN_K_SIZE, 0)
        cv.imshow("gray", img_gray)
        return img_gray

    img_gray = preprocess(img)

    # Canny edge detection
    def getCannyEdges(img):
        edges = cv.Canny(img, CANNY_LOWER, CANNY_UPPER)
        print("edges", edges)
        structuring_element = cv.getStructuringElement(cv.MORPH_RECT, STRUCTURE_ELEMENT_SIZE)
        edges = cv.dilate(edges, structuring_element)
        cv.imshow("edges", edges)
        k = cv.waitKey(0)
        return edges
    
    edges = getCannyEdges(img_gray)
    # quit(0)

    # Make histogram of gray-scale pixel values and find value at given percentile
    def pixelAtPercentile(img):
        hist = cv.calcHist([img], [0], None, [256], [0, 256])
        percentile = 0.25
        value_at_percentile = 0
        count = 0
        while value_at_percentile < 256 and count < percentile * PROCESS_RES[0] * PROCESS_RES[1]:
            count += hist[value_at_percentile]
            value_at_percentile += 1
        print("value_at_percentile", value_at_percentile)
        return hist, value_at_percentile

    # hist, value_at_percentile = pixelAtPercentile(img_gray)
    # quit(0)

    # Threshold pixel values at a given threshold_value and dilate to denoise
    def thresholdAndDilate(img, threshold_value):
        ret, thresh = cv.threshold(img, threshold_value, 255, cv.THRESH_BINARY_INV)
        structuring_element = cv.getStructuringElement(cv.MORPH_RECT, STRUCTURE_ELEMENT_SIZE)
        # cv.imshow("before_dilate", thresh)
        thresh = cv.dilate(thresh, structuring_element)
        # cv.imshow("after_dilate", thresh)
        # thresh = cv.copyMakeBorder(thresh, 5, 5, 5, 5, cv.BORDER_CONSTANT, value=255)
        return thresh
    
    # thresh = thresholdAndDilate(img_gray, value_at_percentile)    
    # cv.imshow("thresh", thresh)
    # k = cv.waitKey(0)
    # quit(0)

    # Get the contours given a processed image (eg. Canny edges) (and NOT filter out small contours)
    def getContours(img, edges):
        contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        kept_contours = []
        kept_contours_areas = []
        for i in range(len(contours)):
            contour = [list(point[0]) for point in contours[i]]
            # print(contour)
            if len(contour) > 3:
                contour_polygon = Polygon(contour)
                contour_area = contour_polygon.area
                if contour_area > MIN_CONTOUR_AREA:
                    kept_contours.append(np.asarray([[point] for point in contour]))
                    kept_contours_areas.append(contour_area)

        print("len(contours)", len(contours))
        print("len(kept_contours)", len(kept_contours))
        # print(contours[0])
        # print(np.asarray(kept_contours[0]))

        contours = kept_contours
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        # cv.imshow("gray", img)
        # cv.imshow("contours", cv.drawContours(img, contours, -1, (0, 255, 0), 3))
        # k = cv.waitKey(0)
        return contours, kept_contours_areas

    contours, kept_contours_areas = getContours(img_gray, edges)
    print("len(contours) before filter", len(contours))
    # quit(0)

    # Filter out the bottom x percentile of contours based on area
    def filterContours(contours, contours_areas, percentile):
        area_threshold = np.percentile(contours_areas, percentile)
        kept_contours = []
        kept_contours_areas = []
        for i in range(len(contours)):
            if contours_areas[i] > area_threshold:
                kept_contours.append(contours[i])
                kept_contours_areas.append(contours_areas[i])
        return kept_contours, kept_contours_areas
    
    contours, contours_areas = filterContours(contours, kept_contours_areas, AREA_PERCENTILE_THRESH)
    print("len(contours) after filter", len(contours))
    cv.imshow("gray", img_gray)
    cv.imshow("contours", cv.drawContours(img_gray, contours, -1, (0, 255, 0), 3))
    k = cv.waitKey(0)

    # Helper function to get the centroid of a series of points
    def getCentroid(points):
        x, y = zip(*points)
        n = len(x)
        centroid = [int(sum(x) / n), int(sum(y) / n)]
        return centroid

    # Get a list of centroids for each contour
    def extractCentroids(contours):
        centroids = []
        for contour in contours:
            contour = [list(point[0]) for point in contour]
            centroid = getCentroid(contour)
            centroid = [int(centroid[i] * resize_scale[i]) for i in range(len(centroid))]
            centroids.append(centroid)
        return centroids
    
    centroids = extractCentroids(contours)
    # print("centroids", centroids)

    # Cluster the contours' centroids
    def centroidClustering(centroids):
        distance_threshold = target_size[0] * DISTANCE_THRESHOLD_SCALE
        print("distance_threshold", distance_threshold)
        clustering = AgglomerativeClustering(n_clusters=None, linkage="single", distance_threshold=distance_threshold)
        labels = clustering.fit_predict(centroids)
        print("clustering.distances_", clustering.distances_)
        num_clusters = max(labels) + 1
        # print("labels", labels)

        clusters = {k: [centroids[i] for i in range(len(centroids)) if labels[i] == k] for k in range(num_clusters)}
        return labels, clusters
    
    labels, clusters = centroidClustering(centroids)
    num_clusters = len(clusters)
    print("clusters", clusters)
    # print("num_clusters", num_clusters)
    cluster_centroids = {k: getCentroid(clusters[k]) for k in range(num_clusters)}
    # print("cluster_centroids", cluster_centroids)

    # Method 1: find the best cluster by getting the cluster with the most items in it
    def findBestCluster(clusters):
        cluster_sizes = [len(clusters[cluster_id]) for cluster_id in clusters]
        print("cluster_sizes", cluster_sizes)
        max_size = 0
        max_size_idx = 0
        for i in range(len(cluster_sizes)):
            cluster_size = cluster_sizes[i]
            if cluster_size > max_size:
                max_size = cluster_size
                max_size_idx = i

        # max_size_idx = 6
        return max_size_idx
    
    # Method 2: find the best cluster by considering total area in the cluster's contours
    def findBestClusterByArea(contours):
        cluster_total_areas = defaultdict(int)
        max_area = 0
        max_area_idx = 0
        print(len(contours))
        print(len(contours_areas))
        print(len(labels))
        for i in range(len(contours)):
            contour = contours[i]
            contour_area = contours_areas[i]
            contour_label = labels[i]
            cluster_total_areas[contour_label] += contour_area
            if cluster_total_areas[contour_label] > max_area:
                max_area = cluster_total_areas[contour_label]
                max_area_idx = contour_label

        # for cluster_id in clusters:
        #     clusters_at_id = clusters[cluster_id]
        #     cluster_total_area = 0
        #     for i in range(len(clusters_at_id)):
        #         contour = [list(point[0]) for point in contours[i]]
        #         contour_polygon = Polygon(contour)
        #         cluster_total_area += contour_polygon.area
        #     cluster_total_areas.append(cluster_total_area)

        # max_area_idx = list(cluster_total_areas.values()).index(max(cluster_total_areas.values()))
        print("cluster_total_areas", cluster_total_areas)
        return max_area_idx
    
    max_size_idx = findBestCluster(clusters)
    max_area_idx = findBestClusterByArea(contours)
    # best_cluster_centroid = cluster_centroids[max_size_idx]
    best_cluster_centroid = cluster_centroids[max_area_idx]
    print("max_size_idx", max_size_idx)
    print("max_area_idx", max_area_idx)
    print("best_cluster_centroid", best_cluster_centroid)

    # # Second pass of clustering (not useful)
    # clustering_2 = AgglomerativeClustering(n_clusters=None, distance_threshold=400)
    # labels_2 = clustering_2.fit_predict(list(cluster_centroids.values()))
    # print("labels_2", labels_2)

    # cv.drawContours(img_copy, contours, -1, (0,255,0), 3)
    # for i in range (len(cluster_centroids)):
    #     cluster_centroid = cluster_centroids[i]
    #     cv.circle(img_copy, cluster_centroid, 5, (0, 255, 0), -1)
    #     # cv.putText(img_copy, str(i), cluster_centroid, fontFace=0, fontScale=0.3, color=(0, 0, 0))
    #     cv.putText(img_copy, str(labels_2[i]), cluster_centroid, fontFace=0, fontScale=0.3, color=(0, 0, 0))

    # cv.drawContours(thresh, contours, -1, 127, 3)
    # cv.imshow("contours", thresh)

    # Given a point of interest, crop the image to the maximum-size square around the point
    def getNewBounds(best_cluster_centroid, target_size, height, width):
        print("target_size", target_size)
        x1, y1 = [int(best_cluster_centroid[i] - target_size[i] / 2) for i in range(len(target_size))]
        x2, y2 = [int(best_cluster_centroid[i] + target_size[i] / 2) for i in range(len(target_size))]
        print("x1, y1, x2, y2", x1, y1, x2, y2)
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > width:
            x1 -= (x2 - width)
            x2 = width
        if y2 > height:
            y1 -= (y2 - height)
            y2 = height
        print("x1, y1, x2, y2", x1, y1, x2, y2)
        return x1, y1, x2, y2

    x1, y1, x2, y2 = getNewBounds(best_cluster_centroid, target_size, height, width)

    # Draw contour centroids and write their labels. Also draw the point of interest
    def debugDraw(img, centroids, labels):
        for i in range (len(centroids)):
            centroid = centroids[i]
            cv.circle(img, centroid, 5, (0, 255, 0), -1)
            cv.putText(img, str(labels[i]), centroid, fontFace=0, fontScale=0.3, color=(0, 0, 0))
        cv.circle(img, best_cluster_centroid, 5, (0, 0, 255), -1)
        cv.imshow("centroids", img)
    
    debugDraw(img_copy, centroids, labels)

    # Crop the image given bounds found by getNewBounds()
    def cropImage(img, x1, x2, y1, y2):
        cropped = img[y1:y2, x1:x2]
        cv.imshow("cropped", cropped)
        cv.imwrite("cropped/" + image.split("/")[-1], cropped)
    
    cropImage(img, x1, x2, y1, y2)
    # cv.imshow("image", img)
    k = cv.waitKey(0)

image = "taiwan_chicken.jpg"
# image = "oyster_omelet.jpg"
# image = "sausage.jfif"
# image = "ice_cream_roll.JPG"
# image = "taro_balls.jpg"
# image = "taiwan_map.png"
# centerAndCropImage("inputs/" + image)

images = list(map(str, Path("inputs/").glob("*")))
for image in images:
    centerAndCropImage(image)