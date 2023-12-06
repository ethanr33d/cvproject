import cv2
import pickle
import os
import numpy as np
from config import data_directory, training_directory
from boosting import boosted_predict
from draw_rectangle import draw_rectangle
from filter_detected_windows import filter_detected_windows
from multiscale_face_positions import multiscale_face_positions
from compute_metrics import compute_metrics

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
# for each nonface get all 60x60 windows
for nonface_file in nonface_files:
    nonface_gray = cv2.imread(nonfaces_directory + "/" + nonface_file, cv2.IMREAD_GRAYSCALE)
    nonface_shape = nonface_gray.shape

    # iterate over image and take only complete 60x60 windows
    for i in range(0, nonface_shape[0] - 60, 60):
        for j in range(0, nonface_shape[1] - 60, 60):
            test_nonfaces_grayscale.append(np.copy(nonface_gray[i:i+60, j:j+60]))

for face_file in cropped_face_files:
    face = cv2.imread(cropped_faces_directory + "/" + face_file, cv2.IMREAD_GRAYSCALE)
    face = face[30:90, 20:80]
    test_cropped_faces_grayscale.append(face)

pure_photos = []
# get rid of pesky face_annotations file
for face_file in photo_face_files:
    if face_file != "face_annotations.py":
        pure_photos.append(face_file)

photo_face_files = pure_photos

for face_file in photo_face_files:
    color_img = cv2.imread(photo_faces_directory + "/" + face_file)
    test_photos_faces.append(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    test_photos_faces_grayscale.append(cv2.imread(photo_faces_directory + "/" + face_file, cv2.IMREAD_GRAYSCALE))
    
################################################################################################
#MODEL EXTRACTION FROM STORED PICKLES
#grab the saved weakclassifiers
with open(training_directory + "/classifiers_scale_60.pkl", "rb") as f:
    weak_classifiers_60= pickle.load(f)

with open(training_directory + "/classifiers_scale_40.pkl", "rb") as f:
    weak_classifiers_40= pickle.load(f)

boosted_classifier_60 = []
with open(training_directory + "/model_scale_60.pkl", "rb") as f:
    boosted_classifier_60 = pickle.load(f)

boosted_classifier_40 = []
with open(training_directory + "/model_scale_40.pkl", "rb") as f: 
    boosted_classifier_40 = pickle.load(f)


################################################################################################
#RUNNING MODEL ON GREYSCALE CROPPED IMAGES

# Load face and non-face data
gray_faces = np.copy(test_cropped_faces_grayscale)
gray_nonfaces = np.copy(test_nonfaces_grayscale)
# print(f"Filtered Faces size: {len(gray_faces)}")
# print(f"Filtered Non-Faces size: {len(gray_nonfaces)}")

# Adjust these lists for thresholds and number of used classfiers
num_classifiers_list = [3, 20, 50]
threshold_list = [-3, -1.25, 0]
cur_threshold_index = 0

filtered_faces = np.copy(gray_faces)
filtered_nonfaces = np.copy(gray_nonfaces)

for num_classifiers in num_classifiers_list:

    #call boosted_predct with current amount of classifiers
    face_predictions = boosted_predict(filtered_faces, boosted_classifier_60, weak_classifiers_60, num_classifiers)
    nonface_predictions = boosted_predict(filtered_nonfaces, boosted_classifier_60, weak_classifiers_60, num_classifiers)

    face_response_threshold = threshold_list[cur_threshold_index]
    # iterate threshold (e.g raising the threshold each wave)
    cur_threshold_index += 1

    filtered_faces = filtered_faces[face_predictions >= face_response_threshold]
    filtered_nonfaces = filtered_nonfaces[nonface_predictions >= face_response_threshold]

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
#RUNNING MODEL TO DETECT FACES IN IMAGES
# Load images for face detection
gray_faces = test_photos_faces_grayscale #greyscale for model usage
colored_faces = test_photos_faces #still have color for skin detection

# Load skin detection histograms
positive_histogram = np.load("positive_histogram.npy")
negative_histogram = np.load("negative_histogram.npy")

# Adjust these lists for thresholds and number of used classfiers
num_classifiers_list = [3, 20, 50]
threshold_list = [-3, 0, 3]

result_annotations = []
# for each image make windows, run face detection, and interpret results
for imgindex in range(len(test_photos_faces)):
    print("running ", photo_face_files[imgindex])
    gray_image = test_photos_faces_grayscale[imgindex]
    color_image = test_photos_faces[imgindex]
    result_image = np.copy(color_image)
    file_name = photo_face_files[imgindex]
    # gather face positions for 3 different scales
    # EACH FUNCTION RECIEVES ITS OWN MODEL TRAINED ON THE SPECIFIED SCALE
    face_positions_40px, face_scores_40px = multiscale_face_positions(gray_image, 
                           color_image,
                           boosted_classifier_40,#specific model to 40x40px models
                           weak_classifiers_40, 
                           num_classifiers_list, 
                           threshold_list, 
                           0.67,
                           file_name
                           )
    face_positions_60px, face_scores_60px = multiscale_face_positions(gray_image, 
                           color_image, 
                           boosted_classifier_60,#specific model to 60x60px models
                           weak_classifiers_60, 
                           num_classifiers_list, 
                           threshold_list, 
                           1,
                           file_name
                           )
    
    # concatenate the best face positions for all 3 scales
    all_face_positions = face_positions_40px + face_positions_60px
    # print("concatenated face positions " , all_face_positions)
    all_face_scores = face_scores_40px + face_scores_60px
    # print("concatenated face scores " , all_face_scores)
    # run one last time to find the best of the 3 scales
    face_positions, _ = filter_detected_windows(all_face_positions, all_face_scores)
    print("An image finshed!! ==============================================")

    annotation = {
        "photo_file_name": file_name,
        "faces": []
    }

    # draw rectangle on output immage and add bounding box to annotations
    for face_pos in face_positions:
        scale = int(face_pos[2] * 60)
        top = int(face_pos[0])
        left = int(face_pos[1])
        right = left + scale
        bottom = top + scale
        draw_rectangle(result_image, top, bottom, left, right) # draw bounding box
        annotation["faces"].append([top, bottom, left, right])

    # write output image and add final annotation
    cv2.imwrite("/workspaces/cvproject/photo_output/" + photo_face_files[imgindex], cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    result_annotations.append(annotation)
    

# compute and display final metrics for face_photos
compute_metrics(result_annotations)