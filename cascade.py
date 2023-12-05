
import cv2
import pickle
import os
import numpy as np
from config import data_directory, training_directory
from skin_detection import detect_skin
from boosting import boosted_predict



# def cascade(gray_faces, color_faces, weak_classifiers, cascade_classifiers_list, cascade_threshold_list)
# it returns the indicies of the windows that are faces and the corresponding adaboost response
# values for those windows


def cascade(gray_faces, color_faces, model, weak_classifiers, cascade_classifiers_list, cascade_threshold_list, positive_histogram, negative_histogram):
    
    skin_prediction_threshold = .2
    cur_threshold_index = 0
    filtered_gray_faces = gray_faces
    filtered_colored_faces = color_faces
    filtered_indicies = range(len(gray_faces))
    filtered_scores = []

    for num_classifiers in cascade_classifiers_list:
        face_response_threshold = cascade_threshold_list[cur_threshold_index]
        # used as temp list to keep track of non-eliminated faces
        gray_passed_round = []
        # used as temp list to keep track of non-eliminated colored images
        colored_passed_round = []

        indicies_passed_round = []
        scores_passed_round = []

        print("i made it to classifier stage ", cur_threshold_index)
        for i in range(len(filtered_gray_faces)):
            #call boosted_predct with current amount of classifiers
            gray_image = filtered_gray_faces[i][:, :]
            color_image = filtered_colored_faces[i][:,:]

            face_prediction = boosted_predict(gray_image, model, weak_classifiers, num_classifiers)
            skin_prediction = ((detect_skin(color_image, positive_histogram, negative_histogram) >= 0.5).sum() / 10000)
            #if % or num pixels is colored??
            if face_prediction > face_response_threshold and skin_prediction >= skin_prediction_threshold:
                gray_passed_round.append(gray_image)
                colored_passed_round.append(color_image)
                indicies_passed_round.append(filtered_indicies[i])
                scores_passed_round.append(face_prediction)

        # iterate threshold (e.g raising the threshold each wave)
        print("i solved the 91000 windows")
        print(len(gray_passed_round))
        cur_threshold_index += 1
        filtered_gray_faces = gray_passed_round
        filtered_colored_faces = colored_passed_round
        filtered_indicies = indicies_passed_round
        filtered_scores = scores_passed_round
        print("i made it through ", i, " classifier stages")
    

    return filtered_indicies, filtered_scores