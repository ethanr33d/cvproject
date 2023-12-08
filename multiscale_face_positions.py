import cv2
import numpy as np

from draw_rectangle import draw_rectangle
from cascade import cascade
from skin_detection import detect_skin
from filter_detected_windows import filter_detected_windows
from config import training_directory

positive_histogram = np.load(training_directory + "/positive_histogram.npy")
negative_histogram = np.load(training_directory + "/negative_histogram.npy")
skin_threshold = 0.3
pixel_granularity = 3

def multiscale_face_positions(gray_image, color_image, model, weak_classifiers, cascade_classifiers_list, cascade_threshold_list, scale, filename):
    """
    Adjusts images based on scale, filters out windows that have no skin, 
        cascades remaining windows of that image, then returns best face positions(coordinates on original image) and the scores(boost) of those windows

    Args:
        gray_image: grayscale version of color_image
        color_image: colored image
        model: model(based on scale)
        weak_classifiers: weak classifiers in accordance to the model(scale)
        cascade_classifiers_list: list of how many classifiers will be used for boost_predict cascade
        cascade_threshold_list: list of thresholds used in each cascade stage
        scale: scaled of image (40x40px or 60x60px)
        filename: used for naming outputted images(mostly for debug)

    Returns:
         face_positions: list of best windows: tuple(y,x,scale) - coordinates of window and its scale
         scores: a parralel list synced with face_positions - giving the boost_predict score of that window

    """
    skin_scores = detect_skin(color_image, positive_histogram, negative_histogram)
    gray_windows = []
    windowinfo = [] # parallel array that stores the position in the real image of top left corner of each window 
    window_size = int(60 * scale)
    real_skin_threshold = int(window_size * window_size * skin_threshold) # pixels needed per window
    num_windows_processed = 0

    # make windows of size 60x60 * scale
    for i in range(0, gray_image.shape[0] - window_size, pixel_granularity):
        for j in range(0, gray_image.shape[1] - window_size, pixel_granularity):
            num_windows_processed += 1
            skin_score = (skin_scores[i:i+window_size, j:j+window_size] >= 0.5).sum()

            # if skin is above threshold keep window
            if (skin_score >= real_skin_threshold):
                window = gray_image[i:i+window_size, j:j+window_size]
                gray_windows.append(window)
                windowinfo.append((i,j, scale)) # store the position of the top left corner of window in parallel array
            
    
    print(num_windows_processed, " windows created")
    print(len(gray_windows), " windows passed skin detection")

    # VISUALIZATION CODE - contains a hardcoded path to visualization folder, overlays skin detected windows onto image
    # result_image2 = np.copy(color_image)

    # for i in range(len(gray_windows)):
    #     face_pos = windowinfo[i]
    #     top = int(face_pos[0])
    #     left = int(face_pos[1])
    #     scaled = int(face_pos[2] * 60)
    #     draw_rectangle(result_image2, top, top + scaled, left, left + scaled) # draw bounding box
    # isZero = "ZERO" if len(gray_windows) == 0 else ""
    # cv2.imwrite("/workspaces/cvproject/skin_photos/" + filename + str(scale) + isZero + ".jpg", cv2.cvtColor(result_image2, cv2.COLOR_BGR2RGB))

    # run skin detection 
    face_windows, face_scores = cascade(gray_windows, 
                           model,
                           weak_classifiers, 
                           cascade_classifiers_list, 
                           cascade_threshold_list, 
                           )

    # gather positions of detected windows
    detection_positions = []
    for face_window_index in face_windows:
        detection_positions.append(windowinfo[face_window_index])

    # find strongest boxes and delete overlapping ones
    face_positions, scores = filter_detected_windows(detection_positions, face_scores)

    # VISUALIZATION CODE - contains a hardcoded path to visualization folder, visualizes detected boxes with strongest boxes
    # result_image = np.copy(color_image)
    
    # for face_pos in detection_positions:
    #     top = int(face_pos[0])
    #     left = int(face_pos[1])
    #     scaled = int(face_pos[2] * 60)
    #     draw_rectangle(result_image, top, top + scaled, left, left + scaled) # draw bounding box


    # for face_pos in face_positions:
    #     top = int(face_pos[0])
    #     left = int(face_pos[1])
    #     scaled = int(face_pos[2] * 60)
    #     draw_rectangle(result_image, top, top + scaled, left, left + scaled, (255,0,0)) # draw bounding box
    # cv2.imwrite("/workspaces/cvproject/bounding_photos/" + filename + str(scale) + ".jpg", cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))

    return face_positions, scores
