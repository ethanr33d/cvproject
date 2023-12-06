import cv2
import numpy as np
from cascade import cascade
from skin_detection import detect_skin
from filter_detected_windows import filter_detected_windows

positive_histogram = np.load("positive_histogram.npy")
negative_histogram = np.load("negative_histogram.npy")
skin_threshold = 0.3
pixel_granularity = 1

def multiscale_face_positions(gray_image, color_image, boosted_classifier, weak_classifiers, cascade_classifiers_list, cascade_threshold_list, scale):
    skin_scores = detect_skin(color_image, positive_histogram, negative_histogram)
    gray_windows = []
    windowinfo = [] # parallel array that stores the position in the real image of top left corner of each window 
    window_size = int(100 * scale)
    real_skin_threshold = int(window_size * window_size * skin_threshold)

    # make windows of size 100x100
    for i in range(0, gray_image.shape[0] - window_size, pixel_granularity):
        for j in range(0, gray_image.shape[1] - window_size, pixel_granularity):
            skin_score = (skin_scores[i:i+window_size, j:j+window_size] >= 0.5).sum()

            if (skin_score >= real_skin_threshold):
                window = gray_image[i:i+window_size, j:j+window_size]
                gray_windows.append(window)
                windowinfo.append((i,j, scale)) # store the position of the top left corner of window in parallel array
            
    
    print("i made ", len(gray_windows), " windows!!!")

    # run face detection
    face_windows, face_scores = cascade(gray_windows, 
                           boosted_classifier,
                           weak_classifiers, 
                           cascade_classifiers_list, 
                           cascade_threshold_list, 
                           )

    
    # gather positions of detected windows
    detection_positions = []
    for face_window_index in face_windows:
        detection_positions.append(windowinfo[face_window_index])

    from draw_rectangle import draw_rectangle
    import random

    result_image = np.copy(color_image)
    face_positions, scores = filter_detected_windows(detection_positions, face_scores)
    for face_pos in face_positions:
        top = int(face_pos[0])
        left = int(face_pos[1])
        scale = int(face_pos[2] * 100)
        draw_rectangle(result_image, top, top + scale, left, left + scale) # draw bounding box

    cv2.imwrite("/workspaces/cvproject/photo" + str(random.randint(0,1000)) + ".jpg", cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    #return filter_detected_windows(detection_positions, face_scores)
    return face_positions, scores
