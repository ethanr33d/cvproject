
from boosting import boosted_predict



# def cascade(gray_faces, color_faces, weak_classifiers, cascade_classifiers_list, cascade_threshold_list)
# it returns the indicies of the windows that are faces and the corresponding adaboost response
# values for those windows


def cascade(gray_faces, model, weak_classifiers, cascade_classifiers_list, cascade_threshold_list):
    """
    recieves a list of windows from an image, runs the list through the adaboost cascade, returns the indicies and scores(adaboost)
        of the windows that made it through the entire cascade

    Args:
        gray_faces: list of gray images.
        model: model(based on scale)
        weak_classifiers: weak classifiers in accordance to the model(scale)
        cascade_classifiers_list: list of how many classifiers will be used for boost_predict
        cascade_threshold_list: list of thresholds used in each cascade stage

    Returns:
         filtered_indicies: returns the indexes of the windows it was given that pass the cascade
         filtered_scores: in parallel with indicies, returns the scores off those windows that passed the cascade

    """
    cur_threshold_index = 0
    filtered_gray_faces = gray_faces
    filtered_indicies = range(len(gray_faces))
    filtered_scores = []

    for num_classifiers in cascade_classifiers_list:
        face_response_threshold = cascade_threshold_list[cur_threshold_index]
        # used as temp list to keep track of non-eliminated faces
        gray_passed_round = []
        indicies_passed_round = []
        scores_passed_round = []

        for i in range(len(filtered_gray_faces)):
            #call boosted_predct with current amount of classifiers
            gray_image = filtered_gray_faces[i][:, :]
            
            face_prediction = boosted_predict(gray_image, model, weak_classifiers, num_classifiers)
            #if % or num pixels is colored??
            if face_prediction > face_response_threshold:
                gray_passed_round.append(gray_image)
                indicies_passed_round.append(filtered_indicies[i])
                scores_passed_round.append(face_prediction)

        # iterate threshold (e.g raising the threshold each wave)
        print(" ", len(gray_passed_round), " windows remaining after cascade stage:" , cur_threshold_index + 1,)
        cur_threshold_index += 1
        filtered_gray_faces = gray_passed_round
        filtered_indicies = indicies_passed_round
        filtered_scores = scores_passed_round
        
    

    return filtered_indicies, filtered_scores