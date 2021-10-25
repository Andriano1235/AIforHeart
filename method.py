# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 23:35:17 2021

@author: ASUS A412DA
"""

# Import the necessary libraries
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
# Importing Image class from PIL module
from PIL import Image
import glob
import xlsxwriter as xls
from skimage.feature import greycomatrix, greycoprops
import math
from scipy import stats
from keras.models import Sequential
from keras.layers import Dense

# ConvertToRGB Function


def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def methode(location=''):
    # Import Library Haar Cascade
    haar_cascade_face = cv2.CascadeClassifier(
        'haarcascades/haarcascade_frontalface_alt2.xml')
    eyes_cascade = cv2.CascadeClassifier(
        'haarcascades/haarcascade_eye_tree_eyeglasses.xml')
    
    # Import image
    img = cv2.imread(location)
    # Resize image for standard image
    basewidth = 2000
    wpercent = float(basewidth/img.shape[1])
    hsize = int(img.shape[0] * wpercent)
    dsize = (basewidth, hsize)
    img = cv2.resize(img, dsize)

#     # Save resized image
#     cv2.imwrite('Resize Image/result1.jpg', img)

#     # Loading the image to be tested
#     test_image = cv2.imread('Resize Image/result1.jpg')

    # Converting to grayscale as opencv expects detector takes in input gray scale images
    gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#make picture gray
    faces = face_cascade.detectMultiScale(gray_picture, 1.1, 5)

    # Face Detect
#     faces_rects = haar_cascade_face.detectMultiScale(
#         test_image_gray, scaleFactor=1.1, minNeighbors=2)
#     eyes_rects = eyes_cascade.detectMultiScale(
#         test_image, scaleFactor=1.1, minNeighbors=3)

#     # Let us print the no. of faces found
#     print('Faces found: ', len(faces_rects))
#     print('Eyes found: ', len(eyes_rects))
#     # print('Eyes found: ', len(eyes_rects_1))

    # Coding coordinate face detector
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        gray_face = gray_picture[y:y+h, x:x+w] # cut the gray face frame out
        face = img[y:y+h, x:x+w] # cut the face frame out
        
#         Save file face
        face_file_name = "Result by File/face.jpg"
        plt.imsave(face_file_name, convertToRGB(face))
        
        eyes = eye_cascade.detectMultiScale(gray_face, 1.25, 5, 5)
        height = np.size(face, 0) # get face frame height
        for (x2, y2, w2, h2) in eyes:
            if y+h > height/2: # pass if the eye is at the bottom
                pass
            cv2.rectangle(face,(x2,y2),(x2+w2,y2+h2),(0,0,255),10)
            
            #Forehead
            roi_color_forehead = None
            roi_color_forehead = img[y:y+400, x+500:x+900]

            #Left Crowsfeet
            roi_color_left_crowsfeet = None
            roi_color_left_crowsfeet = face[y2+50:y2+350, x2+250:x2+550]

            #Left Canthus
            roi_color_left_canthus = None
            roi_color_left_canthus = face[y2+50:y2+350, x2-100:x2+200]

            #Bridge Nose
            roi_color_bridge_Nose = None
            roi_color_bridge_Nose = face[y2+50:y2+350, x2-300:x2]

            #Right Canthus
            roi_color_right_canthus = None
            roi_color_right_canthus = face[y2+50:y2+350, x2-500:x2-200]

            #Right Crowsfeet
            roi_color_right_croswfeet = None
            roi_color_right_croswfeet = face[y2+50:y2+350, x2-850:x2-550]
#     for (x, y, w, h) in faces_rects:
#         cv2.rectangle(test_image, (x, y), (x+w, y+h),
#                       (255, 0, 0), 0)  # tebal garis 0,1,2,3 dst
#         print("here1")
#         for (x, y, w, h) in faces_rects:
#             # tebal garis 0,1,2,3 dst
#             test_image, (x, y), (x+w, y+h), (255, 0, 0), 0

#             # Save image face
#             roi_gray = test_image_gray[y:y+h, x:x+w]
#             roi_color = test_image[y:y+h, x:x+w]
#             print("here2a")
#             face_file_name = "Result by File/face.jpg"
#             plt.imsave(face_file_name, convertToRGB(roi_color))
#             print("here2b")
#             eyes = eyes_cascade.detectMultiScale(roi_gray)
#             for (x2, y2, w2, h2) in eyes:
#                 # Comment this line if no need draw rectangle
#                 cv2.rectangle(roi_color, (x2, y2),
#                               (x2+w2, y2+h2), (0, 255, 0), 0)
#                 # Forehead
#                 roi_color_forehead = test_image[y:y+400, x+500:x+900]
#                 print(roi_color_forehead)

#                 # Left Crowsfeet
#                 roi_color_left_crowsfeet = roi_color[y2 +
#                                                      50:y2+350, x2+250:x2+550]

#                 # Left Canthus
#                 roi_color_left_canthus = roi_color[y2+50:y2+350, x2-100:x2+200]

#                 # Bridge Nose
#                 roi_color_bridge_Nose = roi_color[y2+50:y2+350, x2-300:x2]
#                 print(roi_color_bridge_Nose)

#                 # Right Canthus
#                 roi_color_right_canthus = roi_color[y2 +
#                                                     50:y2+350, x2-500:x2-200]

#                 # Right Crowsfeet
#                 roi_color_right_croswfeet = roi_color[y2 +
#                                                       50:y2+350, x2-850:x2-550]

    # Save Result
    # Save result face
    cv2.imwrite('facedetect.jpg', test_image)
    # saving ROI image
    print("here2c")
    forehead = convertToRGB(roi_color_forehead)
    print("here2d")
    # forehead_file_name = "Result by File/crop_forehead" + str(y) + ".jpg"
    forehead_file_name = "Result by File/Forehead.jpg"
    plt.imsave(forehead_file_name, forehead)
    print("here2e")
    # left_canthus_file_name = "Result by File/crop_left_canthus" + str(y) + ".jpg"
    left_canthus_file_name = "Result by File/Left Canthus.jpg"
    plt.imsave(left_canthus_file_name, convertToRGB(roi_color_left_canthus))

    # left_crowsfeet_file_name = "Result by File/crop_left_crowsfeet" + str(y) + ".jpg"
    left_crowsfeet_file_name = "Result by File/Left Crowsfeet.jpg"
    plt.imsave(left_crowsfeet_file_name,
               convertToRGB(roi_color_left_crowsfeet))

    # bridge_Nose_file_name = "Result by File/crop_bridge_Nose" + str(y) + ".jpg"
    bridge_Nose_file_name = "Result by File/Bridge Nose.jpg"
    plt.imsave(bridge_Nose_file_name, convertToRGB(roi_color_bridge_Nose))

    # right_canthus_file_name = "Result by File/crop_right_canthus" + str(y) + ".jpg"
    right_canthus_file_name = "Result by File/Right Canthus.jpg"
    plt.imsave(right_canthus_file_name, convertToRGB(roi_color_right_canthus))

    # right_croswfeet_file_name = "Result by File/crop_right_croswfeet" + str(y) + ".jpg"
    right_croswfeet_file_name = "Result by File/Right Crowsfeet.jpg"
    plt.imsave(right_croswfeet_file_name,
               convertToRGB(roi_color_right_croswfeet))

    # Resize images before using GLCM extraction
    source_folder = 'Result by File/'

    destination_folder = 'Dataset/'
    directory = os.listdir(source_folder)

    for item in directory:
        img = Image.open(source_folder+item)
        imgResize = img.resize((400, 400), Image.ANTIALIAS)
        imgResize.save(destination_folder + item[:-4] + '.png', quality=90)

    print("Berhasil Resize")

    # Extract Feature

    book = xls.Workbook('testing.xlsx')
    sheet = book.add_worksheet()
    sheet.write(0, 0, 'file')

    column = 1

    # kolom fitur glcm
    glcm_feature = ['correlation', 'homogeneity',
                    'dissimilarity', 'contrast', 'energy', 'ASM']
    angle = ['0', '45', '90', '135']
    for i in glcm_feature:
        for j in angle:
            sheet.write(0, column, i+" "+j)
            column += 1

    # Citra Forehead
    roi_wajah = ['Forehead']
    # sum_each_type = 7
    row = 1

    # for i in data positif_type:
    for i in roi_wajah:
        #     for j in range(1,sum_each_type+1):
        for j in range(1):
            column = 0
            file_name = 'Dataset/Forehead.png'
            print(file_name)
            sheet.write(row, column, 'Forehead')
            column += 1

            # preprocessing
            img = cv2.imread(file_name)
            grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, img1 = cv2.threshold(
                grayscale, 165, 255, cv2.THRESH_BINARY_INV)
            # cv2.imshow('img',img1)
            cv2.imwrite('Threshold/Result/Forehead/Result ' +
                        i+str(j)+'.png', img1)  # save image

            img1 = cv2.dilate(img1.copy(), None, iterations=5)
            img1 = cv2.erode(img1.copy(), None, iterations=5)
            b, g, r = cv2.split(img)
            rgba = [b, g, r, img1]
            dst = cv2.merge(rgba, 4)

            contours, hierarchy = cv2.findContours(
                img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            select = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(select)
            png = dst[y:y+h, x:x+w]
            gray = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)

        # GLCM
            distances = [5]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            levels = 256
            symetric = True
            normed = True

            glcm = greycomatrix(gray, distances, angles,
                                levels, symetric, normed)

            glcm_props = [propery for name in glcm_feature for propery in greycoprops(glcm, name)[
                0]]
            for iten in glcm_props:
                sheet.write(row, column, iten)
                column += 1

            row += 1

    # Citra Bridge Nose
    roi_wajah = ['Bridge Nose']
    # sum_each_type = 7

    # for i in data positif_type:
    for i in roi_wajah:
        for j in range(1):
            column = 0
            file_name = 'Dataset/Bridge Nose.png'
            print(file_name)
            sheet.write(row, column, 'Bridge Nose')
            column += 1

            # preprocessing
            img = cv2.imread(file_name)
            grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, img1 = cv2.threshold(
                grayscale, 140, 255, cv2.THRESH_BINARY_INV)
            # cv2.imshow('img',img1)
            cv2.imwrite('Threshold/Result/Bridge Nose/Result ' +
                        i+str(j)+'.png', img1)  # save image

            img1 = cv2.dilate(img1.copy(), None, iterations=5)
            img1 = cv2.erode(img1.copy(), None, iterations=5)
            b, g, r = cv2.split(img)
            rgba = [b, g, r, img1]
            dst = cv2.merge(rgba, 4)

            contours, hierarchy = cv2.findContours(
                img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            select = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(select)
            png = dst[y:y+h, x:x+w]
            gray = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)

        # GLCM
            distances = [5]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            levels = 256
            symetric = True
            normed = True

            glcm = greycomatrix(gray, distances, angles,
                                levels, symetric, normed)

            glcm_props = [propery for name in glcm_feature for propery in greycoprops(glcm, name)[
                0]]
            for iten in glcm_props:
                sheet.write(row, column, iten)
                column += 1

            row += 1

    # Citra Left Canthus
    roi_wajah = ['Left Canthus']
    # sum_each_type = 7

    for i in roi_wajah:
        for j in range(1):
            column = 0
            file_name = 'Dataset/Left Canthus.png'
            print(file_name)
            sheet.write(row, column, 'Left Canthus')
            column += 1

            # preprocessing
            img = cv2.imread(file_name)
            grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, img1 = cv2.threshold(
                grayscale, 90, 255, cv2.THRESH_BINARY_INV)
            # cv2.imshow('img',img1)
            cv2.imwrite('Threshold/Result/Left Canthus/Result ' +
                        i+str(j)+'.png', img1)  # save image

            img1 = cv2.dilate(img1.copy(), None, iterations=5)
            img1 = cv2.erode(img1.copy(), None, iterations=5)
            b, g, r = cv2.split(img)
            rgba = [b, g, r, img1]
            dst = cv2.merge(rgba, 4)

            contours, hierarchy = cv2.findContours(
                img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            select = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(select)
            png = dst[y:y+h, x:x+w]
            gray = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)

        # GLCM
            distances = [5]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            levels = 256
            symetric = True
            normed = True

            glcm = greycomatrix(gray, distances, angles,
                                levels, symetric, normed)

            glcm_props = [propery for name in glcm_feature for propery in greycoprops(glcm, name)[
                0]]
            for iten in glcm_props:
                sheet.write(row, column, iten)
                column += 1

            row += 1

    # Citra Left Crowsfeet
    roi_wajah = ['Left Crowsfeet']
    # sum_each_type = 7

    for i in roi_wajah:
        for j in range(1):
            column = 0
            file_name = 'Dataset/Left Crowsfeet.png'
            print(file_name)
            sheet.write(row, column, 'Left Crowsfeet')
            column += 1

            # preprocessing
            img = cv2.imread(file_name)
            grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, img1 = cv2.threshold(
                grayscale, 90, 255, cv2.THRESH_BINARY_INV)
            # cv2.imshow('img',img1)
            cv2.imwrite('Threshold/Result/Left Crowsfeet/Result ' +
                        i+str(j)+'.png', img1)  # save image

            img1 = cv2.dilate(img1.copy(), None, iterations=5)
            img1 = cv2.erode(img1.copy(), None, iterations=5)
            b, g, r = cv2.split(img)
            rgba = [b, g, r, img1]
            dst = cv2.merge(rgba, 4)

            contours, hierarchy = cv2.findContours(
                img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            select = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(select)
            png = dst[y:y+h, x:x+w]
            gray = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)

        # GLCM
            distances = [5]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            levels = 256
            symetric = True
            normed = True

            glcm = greycomatrix(gray, distances, angles,
                                levels, symetric, normed)

            glcm_props = [propery for name in glcm_feature for propery in greycoprops(glcm, name)[
                0]]
            for iten in glcm_props:
                sheet.write(row, column, iten)
                column += 1

            row += 1

    # Citra Right Canthus
    roi_wajah = ['Right Canthus']
    # sum_each_type = 7

    # for i in data positif_type:
    for i in roi_wajah:
        for j in range(1):
            column = 0
            file_name = 'Dataset/Right Canthus.png'
            print(file_name)
            sheet.write(row, column, 'Right Canthus')
            column += 1

            # preprocessing
            img = cv2.imread(file_name)
            grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, img1 = cv2.threshold(
                grayscale, 90, 255, cv2.THRESH_BINARY_INV)
            # cv2.imshow('img',img1)
            cv2.imwrite('Threshold/Result/Right Canthus/Result ' +
                        i+str(j)+'.png', img1)  # save image

            img1 = cv2.dilate(img1.copy(), None, iterations=5)
            img1 = cv2.erode(img1.copy(), None, iterations=5)
            b, g, r = cv2.split(img)
            rgba = [b, g, r, img1]
            dst = cv2.merge(rgba, 4)

            contours, hierarchy = cv2.findContours(
                img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            select = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(select)
            png = dst[y:y+h, x:x+w]
            gray = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)

        # GLCM
            distances = [5]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            levels = 256
            symetric = True
            normed = True

            glcm = greycomatrix(gray, distances, angles,
                                levels, symetric, normed)

            glcm_props = [propery for name in glcm_feature for propery in greycoprops(glcm, name)[
                0]]
            for iten in glcm_props:
                sheet.write(row, column, iten)
                column += 1

            row += 1

    # Citra Right Crowsfeet
    roi_wajah = ['Right Crowsfeet']
    # sum_each_type = 7

    # for i in data positif_type:
    for i in roi_wajah:
        for j in range(1):
            column = 0
            file_name = 'Dataset/Right Crowsfeet.png'
            print(file_name)
            sheet.write(row, column, 'Right Crowsfeet')
            column += 1

            # preprocessing
            img = cv2.imread(file_name)
            grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, img1 = cv2.threshold(
                grayscale, 90, 255, cv2.THRESH_BINARY_INV)
            # cv2.imshow('img',img1)
            cv2.imwrite('Threshold/Result/Right Crowsfeet/Result ' +
                        i+str(j)+'.png', img1)  # save image

            img1 = cv2.dilate(img1.copy(), None, iterations=5)
            img1 = cv2.erode(img1.copy(), None, iterations=5)
            b, g, r = cv2.split(img)
            rgba = [b, g, r, img1]
            dst = cv2.merge(rgba, 4)

            contours, hierarchy = cv2.findContours(
                img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            select = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(select)
            png = dst[y:y+h, x:x+w]
            gray = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)

        # GLCM
            distances = [5]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            levels = 256
            symetric = True
            normed = True

            glcm = greycomatrix(gray, distances, angles,
                                levels, symetric, normed)

            glcm_props = [propery for name in glcm_feature for propery in greycoprops(glcm, name)[
                0]]
            for iten in glcm_props:
                sheet.write(row, column, iten)
                column += 1

            row += 1

    book.close()

    # Show Result
    df = pd.read_excel('testing.xlsx')

    print(df.shape)

    # Prediction With ANN
    # Training
    train = pd.read_excel("training.xlsx")
    train = train.drop("tipe", axis=1)
    x = train.drop("Decision", axis=1)
    y = train["Decision"]

    # Build Model

    model = Sequential()
    model.add(Dense(100, input_dim=24, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy",
                  optimizer="adam", metrics=["accuracy"])

    model.fit(x, y, epochs=200, batch_size=1)

    _, accuracy = model.evaluate(x, y)
    print("Model accuracy: %.2f" % (accuracy*100))

    # Testing
    test = pd.read_excel("testing.xlsx")
    test = test.drop('file', axis=1)

    # Predict Result
    predictions = model.predict(test)  # make predictions
    # #round the prediction
    rounded = [round(test[0]) for test in predictions]
    rounded

    # Create Conclusion
    ans = sum(rounded)
    # display sum
    print('Sum of the array is ', ans)
    hasil = {}
    if ans > 3:
        hasil["risk"] = "Anda memiliki risiko tinggi jantung koroner"
        hasil["recomend"] = "Segera konsultasikan ke dokter"
        print("Anda memiliki risiko tinggi jantung koroner")
    else:
        hasil["risk"] = "Anda memiliki risiko rendah jantung koroner"
        hasil["recomend"] = "Jaga kesehatan selalu"
        print("Anda memiliki risiko rendah jantung koroner")

    return hasil
