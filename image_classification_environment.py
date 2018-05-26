'''
Developer: Abhishek Manoj Sharma
Course: CS 256 Section 2
Date: October 15, 2017
Class: Environment
'''

from image_classification_agent import Agent
from PIL import Image, ImageStat, ImageFilter
from random import randint
import os
import csv
import math
import random
import webbrowser
import sys
import matplotlib.pyplot as plot

class Environment:

    LANDSCAPES_DIRECTORY = "images/landscapes/"
    HEADSHOTS_DIRECTORY = "images/headshots/"
    ALL_IMAGE_VALUES = []
    NUMBER_OF_FEATURES = 6
    CLUSTER_ITEMS_ALL = []
    AGGLOMERATIVE_CLUSTERS = []
    AGGLOMERATIVE_COUNT = 0
    HIERARCHICAL_MATRIX = []

    #getImageData() method traverses through the images/landscapes and images/portraits directories and calculates image features
    def getImageData(self):
        image_directories = [self.LANDSCAPES_DIRECTORY, self.HEADSHOTS_DIRECTORY]
        print "Retrieving image data from:", image_directories
        image_details = []
        for directory in image_directories:
            for filename in os.listdir(directory):
                if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
                    image_path = directory + filename
                    current_image = self.getRGB(Image.open(image_path))
                    current_image.insert(0, filename)
                    current_image.append(filename.split("_")[0])
                    image_details.append(current_image)

        return image_details

    #getRGB() method applies the FIND_EDGES filter from Pillow library and then calculates each pixel's RGB value's mean and standard deviation
    def getRGB(self, image_file):
        image_file = image_file.convert('RGB')
        image_file = image_file.filter(ImageFilter.FIND_EDGES)
        image_stats = ImageStat.Stat(image_file)
        return image_stats.mean  + image_stats.stddev
        #return image_stats.mean

    #predictImage() prepares the trainin set and sends it along with value of k to agent for image prediction
    def predictImage(self,image_path,k):
        if os.path.isfile(image_path):
            current_image = self.getRGB(Image.open(image_path))
            if len(self.ALL_IMAGE_VALUES) == 0:
                self.ALL_IMAGE_VALUES = self.getImageData()
            a = Agent()
            prediction = a.predictSingleImage(current_image,self.ALL_IMAGE_VALUES,k)
            print "----------------------\nPrediction (k = " + str(k) + "): " + prediction+ "\n----------------------"


            length_of_entries = len(self.ALL_IMAGE_VALUES[0])
            for i in range (0,len(self.ALL_IMAGE_VALUES)):
                del self.ALL_IMAGE_VALUES[i][length_of_entries-1]

            new_switch = raw_input("Press N to search new image, or any key to go back to main menu: ")
            if new_switch=='N' or new_switch=='n':
                new_image_path = raw_input("Enter image path with extension: ")
                new_k = raw_input("Enter the value of k: ")

                # checking if k is an integer
                if new_k.isdigit():
                    self.predictImage(new_image_path,int(new_k))
                else:
                    return

        #shows this message if the image path entered is incorrect
        else:
            print "Image file does not exist"

    #createFolds() method is used for 3 fold validation where it creates 3 training sets and 3 validation sets to measure the agent accuracy
    def createFolds(self):

        print "Building training and validation sets with image attributes.\nPlease wait, this may take about a minute."
        landscape_list = []
        headshot_list = []
        for file in os.listdir(self.LANDSCAPES_DIRECTORY):
            if file.lower().endswith(".jpeg") or file.lower().endswith(".jpg"):
                landscape_list.append(self.LANDSCAPES_DIRECTORY + file)

        for file in os.listdir(self.HEADSHOTS_DIRECTORY):
            if file.lower().endswith(".jpeg") or file.lower().endswith(".jpg"):
                headshot_list.append(self.HEADSHOTS_DIRECTORY + file)
        if len(landscape_list)<10 or len(headshot_list)<10:
            print "\nRequired at least 10 images for landscape and headshot each"
            print "Found",len(landscape_list),"in landscape and",len(headshot_list),"in headshot"
            return
        if (len(landscape_list)<>len(headshot_list)) or (len(landscape_list)+len(headshot_list)%6<>0):
            if len(landscape_list) == len(headshot_list):
                count = len(landscape_list) + len(headshot_list)
                while count%6<>0:
                    print len(landscape_list)
                    del landscape_list[len(landscape_list)-1]
                    del headshot_list[len(headshot_list)-1]
                    count = len(landscape_list) + len(headshot_list)
            elif len(landscape_list) > len(headshot_list):
                length_difference = len(landscape_list) - len(headshot_list)
                del landscape_list[-length_difference:]
                if len(landscape_list) == len(headshot_list):
                    count = len(landscape_list) + len(headshot_list)
                    while count % 6 <> 0:
                        print len(landscape_list)
                        del landscape_list[len(landscape_list) - 1]
                        del headshot_list[len(headshot_list) - 1]
                        count = len(landscape_list) + len(headshot_list)
            else:
                length_difference = len(headshot_list) - len(landscape_list)
                del headshot_list[-length_difference:]
                if len(landscape_list) == len(headshot_list):
                    count = len(landscape_list) + len(headshot_list)
                    while count % 6 <> 0:
                        print len(landscape_list)
                        del landscape_list[len(landscape_list) - 1]
                        del headshot_list[len(headshot_list) - 1]
                        count = len(landscape_list) + len(headshot_list)
        TOTAL_IMAGES = len(landscape_list) + len(headshot_list)
        n1 = TOTAL_IMAGES / 3
        n2 = TOTAL_IMAGES / 6
        n3 = TOTAL_IMAGES / 2
        random.shuffle(landscape_list)
        random.shuffle(headshot_list)

        # building 3 training sets
        training_1 = landscape_list[:n1]
        training_1.extend(headshot_list[:n1])
        training_2 = landscape_list[n1:]
        training_2.extend(landscape_list[:n2])
        training_2.extend(headshot_list[n1:])
        training_2.extend(headshot_list[:n2])
        training_3 = landscape_list[n2:n3]
        training_3.extend(headshot_list[n2:n3])

        # building 3 validation sets
        validation_1 = landscape_list[n1:]
        validation_1.extend(headshot_list[n1:])
        validation_2 = landscape_list[n2:n1]
        validation_2.extend(headshot_list[n2:n1])
        validation_3 = landscape_list[:n2]
        validation_3.extend(headshot_list[:n2])
        training_length = len(training_1)
        validation_length = len(validation_1)

        training_1_data = []
        training_2_data = []
        training_3_data = []
        validation_1_data = []
        validation_2_data = []
        validation_3_data = []
        for i in range(0,training_length):
            temp_list = self.getRGB(Image.open(training_1[i]))
            temp_list.insert(0, training_1[i])
            training_1_data.append(temp_list)

            temp_list = self.getRGB(Image.open(training_2[i]))
            temp_list.insert(0, training_2[i])
            training_2_data.append(temp_list)

            temp_list = self.getRGB(Image.open(training_3[i]))
            temp_list.insert(0, training_3[i])
            training_3_data.append(temp_list)
        print "Training sets built of sizes: " + str(len(training_1_data)) + ", " + str(len(training_2_data)) + ", " + str(len(training_3_data))
        for i in range(0,validation_length):
            temp_list = self.getRGB(Image.open(validation_1[i]))
            temp_list.insert(0, validation_1[i])
            validation_1_data.append(temp_list)

            temp_list = self.getRGB(Image.open(validation_2[i]))
            temp_list.insert(0, validation_2[i])
            validation_2_data.append(temp_list)

            temp_list = self.getRGB(Image.open(validation_3[i]))
            temp_list.insert(0, validation_3[i])
            validation_3_data.append(temp_list)
        print "Validation sets built of sizes: " + str(len(validation_1_data)) +", " + str(len(validation_2_data))+ ", " + str(len(validation_3_data))

        correct_1 = self.computeFoldAccuracy(len(validation_1_data),validation_1_data,training_1_data)
        correct_2 = self.computeFoldAccuracy(len(validation_2_data),validation_2_data,training_2_data)
        correct_3 = self.computeFoldAccuracy(len(validation_3_data), validation_3_data, training_3_data)

        print "\n----------------------\nAccuracies for fold 1\n---------------------"
        self.printFoldValue(correct_1,len(validation_1_data))
        print "\n----------------------\nAccuracies for fold 2\n---------------------"
        self.printFoldValue(correct_2, len(validation_2_data))
        print "\n----------------------\nAccuracies for fold 3\n---------------------"
        self.printFoldValue(correct_3, len(validation_3_data))

        new_switch = raw_input("Press G to view graph, or any key to go back to main menu: ")
        if new_switch == 'G' or new_switch == 'g':
            print "Graph opened. Please close graph to proceed."
            self.plotGraph(correct_1,correct_2,correct_3,len(validation_1_data))
        else:
            return

    #plotGraph() method is used for plotting the graph to show the accuracy differences in 3-fold validations
    def plotGraph(self,count1,count2,count3,size):
        accuracy_1 = []
        accuracy_2 = []
        accuracy_3 = []
        for i in range(0,10):
            accuracy_1.append(float(str("{:.2f}".format((float(count1[i]) / size) * 100))))
            accuracy_2.append(float(str("{:.2f}".format((float(count2[i]) / size) * 100))))
            accuracy_3.append(float(str("{:.2f}".format((float(count3[i]) / size) * 100))))

        k_list = list(range(1,11))

        plot.xlabel("Value of k")
        plot.ylabel("Accuracy (in %)")
        plot.xticks(k_list)
        plot.plot(k_list,accuracy_1, '--o')
        plot.plot(k_list, accuracy_2, '--o')
        plot.plot(k_list, accuracy_3, '--o')
        plot.legend(['Fold 1', 'Fold 2', 'Fold 3'], loc='upper left')
        plot.title("K vs Accuracy for all folds")
        plot.gcf().canvas.set_window_title("Line Graph - Accuracy vs Values of K")  # Credits: Solution posted by user'itoed' on https://github.com/jupyter/notebook/issues/919
        plot.show()

    #computeFoldAccuracy() method is used for computing the accuracy for each fold based on the predictions received from the agent
    def computeFoldAccuracy(self,validation_length,v_data,t_data):
        validation_data = list(v_data)
        training_data = list(t_data)
        correct_count = [0] * 10
        for k in range(1,11):
            for i in range(0, validation_length):
                if "landscape" in validation_data[i][0]:
                    class_label="landscape"
                else:
                    class_label = "headshot"
                a = Agent()
                prediction =  a.predictSingleImage(validation_data[i][1:7],training_data,k)
                length_of_entries = len(training_data[0])
                for n in range(0, len(training_data)):
                    del training_data[n][length_of_entries - 1]
                if class_label == prediction:
                    correct_count[k-1]+=1
        return correct_count

    #printFoldValue() is used for printing the accuracies of different folds for different values of k in an organized way
    def printFoldValue(self,correct_list,total_count):
        for i in range(0,10):
            print "K = " + str(i+1) + ":",
            print str("{:.2f}".format((float(correct_list[i])/total_count)*100)) + "%",
            print "(" + str(correct_list[i]) + " out of " + str(total_count) +")"

    #KMeansClusterImages() method is used to perform K-Means (K=2) on the image dataset
    def KMeansClusterImages(self):
        landscape_list = os.listdir(self.LANDSCAPES_DIRECTORY)
        headshot_list = os.listdir(self.HEADSHOTS_DIRECTORY)
        for i in range(0,len(landscape_list)):
            if landscape_list[i].lower().endswith(".jpg") or landscape_list[i].lower().endswith(".jpeg"):
                landscape_list[i] = self.LANDSCAPES_DIRECTORY + landscape_list[i]
            else:
                del landscape_list[i]
        for i in range(0,len(headshot_list)):
            if headshot_list[i].lower().endswith(".jpg") or headshot_list[i].lower().endswith(".jpeg"):
                headshot_list[i] = self.HEADSHOTS_DIRECTORY + headshot_list[i]
            else:
                del headshot_list[i]

        centroid_1 = []
        r = randint(0, len(landscape_list) - 1)
        centroid_1.append(landscape_list[r])
        landscape_list.pop(r)
        a = self.getRGB(Image.open(centroid_1[0]))
        a.insert(0, centroid_1[0])
        centroid_1[0] = a

        centroid_2 = []
        r = randint(0, len(headshot_list) - 1)
        centroid_2.append(headshot_list[r])
        headshot_list.pop(r)
        a = self.getRGB(Image.open(centroid_2[0]))
        a.insert(0, centroid_2[0])
        centroid_2[0] = a

        for i in range(len(landscape_list)):
            a = self.getRGB(Image.open(landscape_list[i]))
            a.insert(0, landscape_list[i])
            landscape_list[i] = a
            centroid_h_distance = self.calcEuclidean(centroid_2[0][1:7],landscape_list[i][1:7])
            centroid_l_distance = self.calcEuclidean(centroid_1[0][1:7],landscape_list[i][1:7])
            landscape_list[i].append(centroid_h_distance)
            landscape_list[i].append(centroid_l_distance)
            if landscape_list[i][7] > landscape_list[i][8]:
                centroid_1.append(landscape_list[i])
            else:
                centroid_2.append(landscape_list[i])

        for i in range(len(headshot_list)):
            a = self.getRGB(Image.open(headshot_list[i]))
            a.insert(0, headshot_list[i])
            headshot_list[i] = a
            centroid_h_distance = self.calcEuclidean(centroid_2[0][1:7], headshot_list[i][1:7])
            centroid_l_distance = self.calcEuclidean(centroid_1[0][1:7], headshot_list[i][1:7])
            headshot_list[i].append(centroid_h_distance)
            headshot_list[i].append(centroid_l_distance)
            if headshot_list[i][7] > headshot_list[i][8]:
                centroid_1.append(headshot_list[i])
            else:
                centroid_2.append(headshot_list[i])

        #printing HTML code to open in browser to view the images in each cluster
        html_head = "<html><head><title>K-Means Clustering</title></head><body><h3>K-Means Clustering, K=2</h3>"

        f = open("kmeans_clusters.html", "w")
        f.write(html_head + "<h3>Cluster 1</h3>")
        random.shuffle(centroid_2)
        cluster_1_counts = [0] * 2
        for i in range(0,len(centroid_2)):
            html_body = '<img src="' + centroid_2[i][0] + '" width="100" height="100"> '
            f.write(html_body)
            if "headshot" in centroid_2[i][0]:
                cluster_1_counts[0]+=1
            else:
                cluster_1_counts[1]+=1

        f.write("<h3>Cluster 2</h3>")
        random.shuffle(centroid_1)
        cluster_2_counts = [0] * 2
        for i in range(0,len(centroid_1)):
            html_body = '<img src="' + centroid_1[i][0] + '" width="100" height="100"> '
            f.write(html_body)
            if "headshot" in centroid_1[i][0]:
                cluster_2_counts[0]+=1
            else:
                cluster_2_counts[1]+=1
        f.write("</body></html>")
        f.close()

        # opening the browser to load cluster.html file
        webbrowser.open("kmeans_clusters.html")
        print "K-means clustering results opened as webpage in browser"

        print "\nCluster 1:"
        print str(cluster_1_counts[0]) + " headshots"
        print str(cluster_1_counts[1]) + " landscapes"

        print "\nCluster 2:"
        print str(cluster_2_counts[0]) + " headshots"
        print str(cluster_2_counts[1]) + " landscapes"

        new_switch = raw_input("\nPress C to perform K-means cluster again, or any key to go back to main menu: ")
        if new_switch == 'C' or new_switch == 'c':
            self.KMeansClusterImages()
        else:
            return

    #calcEuclidean() returns the Euclidean distance between the two values (percept, lookup) passed to it
    def calcEuclidean(self, percept, lookup_value):
        euclidean_distance = 0.00;
        count = len(percept) - 1
        while count >= 0:
            f1 = float(percept[count])
            f2 = float(lookup_value[count])
            euclidean_distance = euclidean_distance + (f1 - f2) ** 2
            count -= 1
        return euclidean_distance