import cv2
import os
import pickle
import numpy as np

from boosting import integral_image
from boosting import generate_classifier
from boosting import eval_weak_classifier
from boosting import adaboost
from boosting import boosted_predict
from config import training_directory



def train_scaled_model(passed_faces, passed_nonfaces, scale):
    """
    Trains an AdaBoost model based on desired image dimensions

    Args:
        passed_faces: grayscale images of faces training set at scale x scale pixel dimensions.
        passed_nonfaces: grayscale images of nonfaces training set at scale x scale pixel dimensions.
        scale: the desired dimensions of image size that the model is trained on

    Returns:
        Nothing:
        It saves the weak classifiers and model to .pkl's:
            "classifiers_scale_{scale}.pkl" 
            "model_scale_{scale}.pkl"
    """
    
    # Load face and non-face data
    faces = np.copy(passed_faces)
    nonfaces = np.copy(passed_nonfaces)
    face_vertical = scale 
    face_horizontal = scale

    # Define the number of weak classifiers to generate
    number = 7777

    # Generate the list of weak classifiers
    weak_classifiers = [generate_classifier(face_vertical, face_horizontal) for _ in range(number)]

    #name the classifier based on its name
    classifier_file_name = f"classifiers_scale_{scale}.pkl"
    with open(training_directory + "/" + classifier_file_name, "wb") as f:
        pickle.dump(weak_classifiers,f)


    # faces and nonfaces are given as (2400, 100, 100) matrices (probably)
    # Initialize empty lists to store the integral images
    face_integrals = []
    nonface_integrals = []

    # Compute the integral images for all faces
    for face in faces:
        face_integrals.append(integral_image(face))

    # Compute the integral images for all non-faces
    for nonface in nonfaces:
        nonface_integrals.append(integral_image(nonface))

    # Convert the lists of integral images into NumPy arrays
    face_integrals = np.array(face_integrals)
    nonface_integrals = np.array(nonface_integrals)

    # Concatenate the face_integrals and nonface_integrals along the first axis
    # resulting in a new array of shape (4800, 100, 100) assuming the integrals
    # retain the original image shape
    examples = np.concatenate((face_integrals, nonface_integrals), axis=0)

    # Set the labels for the face (1) and non-face (-1) examples
    labels = np.array([1] * len(faces) + [-1] * len(nonfaces))
    print(f"labels made for scale {scale}")

    example_number = examples.shape[0]
    classifier_number = len(weak_classifiers)

    # Initialize an array to hold the responses
    responses = np.zeros((example_number, classifier_number))

    # Loop through each example and classifier
    for example in range(example_number):
        integral = examples[example, :, :]
        for feature in range(classifier_number):
            classifier = weak_classifiers[feature]
            responses[example, feature] = eval_weak_classifier(classifier, integral)

    num_rounds = 25  # Define the number of rounds

    boosted_classifier = adaboost(responses, labels, num_rounds)  # Run the AdaBoost algorithm

    #name the model based on its scale
    model_file_name = f"model_scale_{scale}.pkl"

    with open(training_directory + "/" + model_file_name, "wb") as f:
        pickle.dump(boosted_classifier,f)

    # Predict the label of the 200th face example
    prediction = boosted_predict(faces[20, :, :], boosted_classifier, weak_classifiers, num_rounds)
    print(f"Prediction of face with scale {scale}: {prediction}")

    # Predict the label of the 500th non-face example
    prediction = boosted_predict(nonfaces[50, :, :], boosted_classifier, weak_classifiers, num_rounds)
    print(f"Prediction of nonface with scale {scale}:  {prediction}")

    #**********************************************

    # Evaluate the boosted classifier on the faces and non-faces
    face_predictions = boosted_predict(faces, boosted_classifier, weak_classifiers, num_rounds)
    nonface_predictions = boosted_predict(nonfaces, boosted_classifier, weak_classifiers, num_rounds)

    face_response_threshold = 0  # A face is detected if the response is greater than or equal to this threshold

    # Calculate the accuracy of the predictions
    face_accuracy = np.sum(face_predictions >= face_response_threshold) / len(face_predictions)
    nonface_accuracy = np.sum(nonface_predictions < face_response_threshold) / len(nonface_predictions)

    # Print the results
    print(f"Face accuracy for scale {scale}: {face_accuracy}")
    print(f"Non-face accuracy for scale {scale}: {nonface_accuracy}")

    return