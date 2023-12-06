import numpy as np
from config import data_directory, training_directory

# filter_detected_windows(positions, scores)
# returns the positions of the best array 


#positions is a list of coordinates(y,x)
#scores is a list of adaboost scores in parallel index with positions list
def filter_detected_windows(positions, scores):
    best_position_array  = []
    best_scores_array = []
    working_positions_array = np.copy(positions)
    working_scores_array = np.copy(scores)

    #loop until theres no more overlap between any positions
    #while(box overlapping is still present)
    while(len(working_positions_array) != 0):
        # get index of best score
        argmax_index = np.argmax(working_scores_array)
        best_position = working_positions_array[argmax_index]

        # add best position and corresponding score to best array
        best_position_array.append(best_position)
        best_scores_array.append(working_scores_array[argmax_index])
        filtered_positions = []
        filtered_scores = []
        
        best_pos_left = best_position[1]
        best_pos_top = best_position[0]
        best_scale = best_position[2]
        best_pos_right = best_pos_left + int(100 * best_scale)
        best_pos_bottom = best_pos_top + int(100 * best_scale)
        # go through remaining boxes and compare them to the box with the best score
        # then check if the working box is overlapping at any point with the best boz and if it IS overlapping, no longer consider it (deletion)
        for posIndex in range(len(working_positions_array)):
            if (posIndex == argmax_index): continue
            position = working_positions_array[posIndex]
            current_pos_left = position[1]
            current_pos_top = position[0]
            current_scale = position[2]
            current_pos_right = current_pos_left + int(100 * current_scale)
            current_pos_bottom = current_pos_top + int(100 * current_scale)
            
            # check if working box is overlapping with best box, if they are NOT overlapping, then let worker box pass into next iteration
            if not (current_pos_left < best_pos_right and current_pos_right > best_pos_left
               and current_pos_top < best_pos_bottom and current_pos_bottom > current_pos_top):
                filtered_positions.append(position)
                filtered_scores.append(working_scores_array[posIndex])
        
        
        working_positions_array = filtered_positions
        working_scores_array = filtered_scores

    return best_position_array, best_scores_array