'''
Developer: Abhishek Manoj Sharma
Course: CS 256 Section 2
Date: October 15, 2017
Input: Two arguments
1. Image path
2. Value of k
'''

from argparse import ArgumentParser
from image_classification_environment import Environment

#Implementing argparse for command line interface
help_description='The program will display a command line interface to ' \
                 '1) Predict whether an image is landscape or headshot, ' \
                 '2) Measure agent accuracy using three folds, ' \
                 '3) K-Means clustering\n (k=2).' \
                 ' Please note that all images shall be in directories "images/landscapes" or "images/headshots". ' \
                 'Addtionally, please make sure that there are at least 10 landscape and headshot images each ' \
                 'for 3 fold validation."'

parser = ArgumentParser(description=help_description)
args = parser.parse_args()

#printMenu() method contains the options for the menu-driven logic allowing user to make choices
def printMenu():
    print "\n--------------------\nType the corresponding number and press enter"
    print "1. Predict image (Landscape or Headshot)"
    print "2. Test agent accuracy using 3-folds"
    print "3. K-means clustering clustering (K=2)"
    print "4. Quit"
    option = raw_input("Enter the number: ")
    try:
        option = int(option)
        if option == 1:
            image_path = raw_input("\n---------------\nEnter image path with extension: ")
            k = raw_input("Enter the value of k: ")

            # checking if k is an integer
            if k.isdigit():
                e = Environment()
                e.predictImage(image_path,int(k))
                printMenu()
            else:
                print "Invalid value of k, try again"
                printMenu()
        elif option == 2:
            e = Environment()
            e.createFolds()
            printMenu()
        elif option == 3:
            print "K-Means Clustering in progress. Please wait."
            e = Environment()
            e.KMeansClusterImages()
            printMenu()
        elif option == 4:
            print "Quitting program"
            exit(0)
        else:
            print "Invalid option selected.\nTry again"
            printMenu()
    # throws an error if the input is not an integer
    except ValueError:
        print "Invalid option selected.\nTry againn"
        printMenu()

printMenu()