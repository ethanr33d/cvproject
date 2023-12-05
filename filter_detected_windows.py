import cv2
import numpy as np
from config import data_directory, training_directory

# filter_detected_windows(positions, scores)
# returns the positions of the best array 


#positions is a list of coordinates(y,x)
#scores is a list of adaboost scores in parallel index with positions list
def filter_detected_windows(positions, scores):
    #loop until theres no more overlap between any positions
    #while(box overlapping is still present)
    best_array  = []
    working_positions_array = np.copy(positions)
    working_scores_array = np.copy(scores)

    while(len(working_positions_array) != 0):
        argmax_index = np.argmax(working_scores_array)
        best_position = working_positions_array[argmax_index]
        best_array.append(best_position)

        filtered_positions = []
        filtered_scores = []

        for position in working_positions_array:
            current_pos_cent_y = position[1]
            current_pos_cent_x = position[0]
            dist_x = abs(best_position[0] - current_pos_cent_x)
            dist_y = abs(best_position[1] - current_pos_cent_y)

            if (dist_x > 50 and dist_y > 50):
                filtered_positions.append(position)
                filtered_scores.append(np.max(working_scores_array))
        
        working_positions_array = filtered_positions
        working_scores_array = filtered_scores

    return best_array