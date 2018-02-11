'''
Developer: Abhishek Manoj Sharma
Course: CS 256 Section 2
Homework: 3
Date: October 15, 2017
Class: Agent
'''

from operator import itemgetter
from collections import Counter

class Agent:

    #predictSingleImage() method is used for predicting whether an input image is headshot or landscape based on the value of k
    def predictSingleImage(self, current_image_percept, training_image_set,k):
        image_set_with_distances=[]

        for value in training_image_set:
            distance = self.calcEuclidean(current_image_percept,value[1:10])
            value.append(distance)
            image_set_with_distances.append(value)

        image_set_with_distances = sorted(image_set_with_distances, key=itemgetter(len(image_set_with_distances[0])-1))
        prediction = self.actuator(image_set_with_distances,k)
        return prediction


    #actuator() method is used for sorting the k neareast neighbors and sending the predictions to the environment
    def actuator(self,image_set_with_distances,k):

        k = int(k)
        predicted_values = []
        prediction = []
        class_label = len(image_set_with_distances[0])-2
        #print "Class label", class_label
        #for i in range(1,k+1):
        for j in range(0,k):
            if "landscape" in image_set_with_distances[j][0]:
                predicted_values.append("landscape")
            else:
                predicted_values.append("headshot")
        prediction = Counter(predicted_values).most_common()[0][0]
        return prediction

    '''
    The calcEuclidean() method accepts the percept and lookup value to calculate the distance between them.
    It returns the Euclidean distance between them as a float data type.
    '''
    def calcEuclidean(self, percept, lookup_value):
        euclidean_distance = 0.00;
        count = len(percept) - 1

        while count >= 0:
            f1 = float(percept[count])
            f2 = float(lookup_value[count])
            euclidean_distance = euclidean_distance + (f1 - f2) ** 2
            count -= 1
        return euclidean_distance
