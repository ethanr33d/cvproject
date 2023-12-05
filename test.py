import cv2
import pickle
import os
import numpy as np
from config import data_directory, training_directory
from skin_detection import detect_skin
from boosting import boosted_predict
from cascade import cascade
from draw_rectangle import draw_rectangle
from filter_detected_windows import filter_detected_windows

# Get test data
nonfaces_directory = data_directory + "/test_nonfaces"
cropped_faces_directory = data_directory + "/test_cropped_faces"
photo_faces_directory = data_directory + "/test_face_photos"
cropped_face_files = os.listdir(cropped_faces_directory)
nonface_files = os.listdir(nonfaces_directory)
photo_face_files = os.listdir(photo_faces_directory)
test_nonfaces_grayscale = []
test_cropped_faces_grayscale = []
test_photos_faces = []
test_photos_faces_grayscale = []
nonface_num = 0
################################################################################################
#IMAGE EXTRACTION
# for each nonface get all 100x100 windows
for nonface_file in nonface_files:
    nonface_gray = cv2.imread(nonfaces_directory + "/" + nonface_file, cv2.IMREAD_GRAYSCALE)
    nonface_shape = nonface_gray.shape

    # iterate over image and take only complete 100x100 windows
    for i in range(0, nonface_shape[0] - 100, 100):
        for j in range(0, nonface_shape[1] - 100, 100):
            test_nonfaces_grayscale.append(np.copy(nonface_gray[i:i+100, j:j+100]))

for face_file in cropped_face_files:
    test_cropped_faces_grayscale.append(cv2.imread(cropped_faces_directory + "/" + face_file, cv2.IMREAD_GRAYSCALE))

for face_file in photo_face_files:
    if face_file != "face_annotations.py":
        color_img = cv2.imread(photo_faces_directory + "/" + face_file)
        test_photos_faces.append(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
        test_photos_faces_grayscale.append(cv2.imread(photo_faces_directory + "/" + face_file, cv2.IMREAD_GRAYSCALE))
    
################################################################################################
#MODEL EXTRACTION FROM STORED PICKLES
#grab the saved weakclassifiers
with open(training_directory + "/classifiers.pkl", "rb") as f:
    weak_classifiers= pickle.load(f)

boosted_classifier = []
with open(training_directory + "/model.pkl", "rb") as f:
    boosted_classifier = pickle.load(f)

################################################################################################
#RUNNING MODEL ON GREYSCALE CROPPED IMAGES

# Load face and non-face data
gray_faces = np.copy(test_cropped_faces_grayscale)
gray_nonfaces = np.copy(test_nonfaces_grayscale)
# print(f"Filtered Faces size: {len(gray_faces)}")
# print(f"Filtered Non-Faces size: {len(gray_nonfaces)}")

# Adjust these lists for thresholds and number of used classfiers
num_classifiers_list = [3, 5, 15]
threshold_list = [-3, -1.25, 0]
cur_threshold_index = 0

filtered_faces = np.copy(gray_faces)
filtered_nonfaces = np.copy(gray_nonfaces)

for num_classifiers in num_classifiers_list:

    #call boosted_predct with current amount of classifiers
    face_predictions = boosted_predict(filtered_faces, boosted_classifier, weak_classifiers, num_classifiers)
    nonface_predictions = boosted_predict(filtered_nonfaces, boosted_classifier, weak_classifiers, num_classifiers)

    face_response_threshold = threshold_list[cur_threshold_index]
    # iterate threshold (e.g raising the threshold each wave)
    cur_threshold_index += 1

    filtered_faces = filtered_faces[face_predictions >= face_response_threshold]
    filtered_nonfaces = filtered_nonfaces[nonface_predictions >= face_response_threshold]

    # Print the sizes of filtered_faces and filtered_nonfaces
    # print(f"Iteration with {num_classifiers} classifiers:")
    # print(f"Filtered Faces size: {len(filtered_faces)}")
    # print(f"Filtered Non-Faces size: {len(filtered_nonfaces)}")
    # print(f"-------------------------------------------------")

# Calculate the accuracy of the predictions
# Takes the finished length of the faces / original length. For 100% accuracy there would no faces removed from the filtered faces list
# and zero images remaining in the filtered nonface list
face_accuracy = len(filtered_faces) / len(gray_faces)
nonface_accuracy = 1 - (len(filtered_nonfaces) / len(gray_nonfaces))

# Print the results
print(f"Greyscale Face accuracy: {face_accuracy}")
print(f"Greyscale Non-face accuracy: {nonface_accuracy}")
print("-------------------------")

################################################################################################
#RUNNING MODEL 
# Load images for face detection
gray_faces = test_photos_faces_grayscale #greyscale for model usage
colored_faces = test_photos_faces #still have color for skin detection

# Load skin detection histograms
positive_histogram = np.load("positive_histogram.npy")
negative_histogram = np.load("negative_histogram.npy")

# Adjust these lists for thresholds and number of used classfiers
num_classifiers_list = [3, 5, 15]
threshold_list = [-3, -1.25, 0]

# def cascade(gray_faces, color_faces, weak_classifiers, cascade_classifiers_list, cascade_threshold_list)
# it returns the indicies of the windows that are faces


# for every image 
#   make windows
#   cascade
#   draw bounding box around windows that are faces

# for each image make windows, run face detection, and interpret results
for imgindex in range(len(test_photos_faces)):
    gray_image = test_photos_faces_grayscale[imgindex]
    color_image = test_photos_faces[imgindex]
    result_image = np.copy(color_image)
    gray_windows = []
    color_windows = []
    windowinfo = [] # parallel array that stores the position in the real image of top left corner of each window 

    # make windows of size 100x100
    for i in range(gray_image.shape[0] - 100):
        for j in range(gray_image.shape[1] - 100):
            gray_windows.append(gray_image[i:i+100, j:j+100])
            color_windows.append(color_image[i:i+100, j:j+100])
            windowinfo.append((i,j)) # store the position of the top left corner of window in parallel array
    
    print("i made ", len(gray_windows), " windows!!!")
    # run face detection
    face_windows, face_scores = cascade(gray_windows, 
                           color_windows, 
                           boosted_classifier,
                           weak_classifiers, 
                           num_classifiers_list, 
                           threshold_list, 
                           positive_histogram, 
                           negative_histogram
                           )

    
    # gather positions of detected windows
    detection_positions = []
    for face_window_index in face_windows:
        detection_positions.append(windowinfo[face_window_index])
    

    # filter out overlapping boxes
    face_positions = filter_detected_windows(detection_positions, face_scores)

    # write resulting face windows to file
    for face_pos in face_positions:
        top = face_pos[0]
        left = face_pos[1]
        draw_rectangle(result_image, top, top + 100, left, left + 100) # draw bounding box

    cv2.imwrite("/workspaces/cvproject/photo_output/" + photo_face_files[imgindex], cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    
# filter_detected_windows(positions, scores)
# returns the positions of the best array 

# # Print the results
# print(f"Greyscale Face accuracy: {face_accuracy}")
# print(f"Greyscale Non-face accuracy: {nonface_accuracy}")
# print("-------------------------")