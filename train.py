import cv2
import os
import pickle
import numpy as np
from config import data_directory, training_directory

from boosting import integral_image
from boosting import generate_classifier
from boosting import eval_weak_classifier
from boosting import adaboost
from boosting import boosted_predict

# Split the training_nonfaces into 100x100 windows and put them in training/training_nonfaces folder
nonfaces_directory = data_directory + "/training_nonfaces"
faces_directory = data_directory + "/training_faces"
face_files = os.listdir(faces_directory)
nonface_files = os.listdir(nonfaces_directory)
training_nonfaces = []
training_faces = []
training_faces_grayscale = []
training_nonfaces_grayscale = []
nonface_num = 0

#for each nonface get all 100x100 windows
for nonface_file in nonface_files:
    nonface = cv2.imread(nonfaces_directory + "/" + nonface_file)
    nonface_gray = cv2.imread(nonfaces_directory + "/" + nonface_file, cv2.IMREAD_GRAYSCALE)
    nonface_shape = nonface.shape

    # iterate over image and take only complete 100x100 windows
    for i in range(0, nonface_shape[0] - 100, 100):
        for j in range(0, nonface_shape[1] - 100, 100):
            training_nonfaces.append(np.copy(nonface[i:i+100, j:j+100]))
            training_nonfaces_grayscale.append(np.copy(nonface_gray[i:i+100, j:j+100]))

for face_file in face_files:
    training_faces.append(cv2.imread(faces_directory + "/" + face_file))
    training_faces_grayscale.append(cv2.imread(faces_directory + "/" + face_file, cv2.IMREAD_GRAYSCALE))

# Adaboost/rectangle filters - figure it out from his boosting one - #2
# Load face and non-face data
faces = np.copy(training_faces_grayscale)
nonfaces = np.copy(training_nonfaces_grayscale)
face_vertical = 100
face_horizontal = 100

# Define the number of weak classifiers to generate
number = 1000

# Generate the list of weak classifiers
weak_classifiers = [generate_classifier(face_vertical, face_horizontal) for _ in range(number)]

with open(training_directory + "/classifiers.pkl", "wb") as f:
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
print(f"labels made")

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

num_classifiers = 15  # Define the number of weak classifiers to to select

boosted_classifier = adaboost(responses, labels, num_classifiers)  # Run the AdaBoost algorithm
with open(training_directory + "/model.pkl", "wb") as f:
    pickle.dump(boosted_classifier,f)

# Predict the label of the 200th face example
prediction = boosted_predict(faces[20, :, :], boosted_classifier, weak_classifiers, num_classifiers)
print(f"Prediction: {prediction}")

# Predict the label of the 500th non-face example
prediction = boosted_predict(nonfaces[50, :, :], boosted_classifier, weak_classifiers, num_classifiers)
print(f"Prediction: {prediction}")

#**********************************************

# Evaluate the boosted classifier on the faces and non-faces
face_predictions = boosted_predict(faces, boosted_classifier, weak_classifiers, num_classifiers)
nonface_predictions = boosted_predict(nonfaces, boosted_classifier, weak_classifiers, num_classifiers)

face_response_threshold = 0  # A face is detected if the response is greater than or equal to this threshold

# Calculate the accuracy of the predictions
face_accuracy = np.sum(face_predictions >= face_response_threshold) / len(face_predictions)
nonface_accuracy = np.sum(nonface_predictions < face_response_threshold) / len(nonface_predictions)

# Print the results
print(f"Face accuracy: {face_accuracy}")
print(f"Non-face accuracy: {nonface_accuracy}")


