import cv2
from draw_rectangle import draw_rectangle
from config import data_directory

# from face_annotations.py
annotations = [
    {'photo_file_name': 'clintonAD2505_468x448.jpg', 'faces': [[146, 226, 96, 176], [56, 138, 237, 312]]},
    {'photo_file_name': 'DSC01181.JPG', 'faces': [[141, 181, 157, 196], [144, 184, 231, 269]]},
    {'photo_file_name': 'DSC01418.JPG', 'faces': [[122, 147, 263, 285], [129, 151, 305, 328]]},
    {'photo_file_name': 'DSC02950.JPG', 'faces': [[126, 239, 398, 501]]},
    {'photo_file_name': 'DSC03292.JPG', 'faces': [[92, 177, 169, 259], [122, 200, 321, 402]]},
    {'photo_file_name': 'DSC03318.JPG', 'faces': [[188, 246, 178, 238], [157, 237, 333, 414]]},
    {'photo_file_name': 'DSC03457.JPG', 'faces': [[143, 174, 127, 157], [91, 120, 177, 206], [94, 129, 223, 257]]},
    {'photo_file_name': 'DSC04545.JPG', 'faces': [[56, 86, 119, 151]]},
    {'photo_file_name': 'DSC04546.JPG', 'faces': [[105, 137, 193, 226]]},
    {'photo_file_name': 'DSC06590.JPG', 'faces': [[167, 212, 118, 158], [191, 228, 371, 407]]},
    {'photo_file_name': 'DSC06591.JPG', 'faces': [[180, 222, 290, 330], [260, 313, 345, 395]]},
    {'photo_file_name': 'IMG_3793.JPG', 'faces': [[172, 244, 135, 202], [198, 253, 275, 331], [207, 264, 349, 410], [160, 233, 452, 517]]},
    {'photo_file_name': 'IMG_3794.JPG', 'faces': [[169, 211, 109, 148], [154, 189, 201, 235], [176, 204, 314, 342], [170, 206, 445, 483], [144, 191, 550, 592]]},
    {'photo_file_name': 'IMG_3840.JPG', 'faces': [[200, 268, 150, 212], [202, 262, 261, 323], [222, 286, 371, 430], [154, 237, 477, 549]]},
    {'photo_file_name': 'jackie-yao-ming.jpg', 'faces': [[45, 77, 93, 124], [61, 91, 173, 200]]},
    {'photo_file_name': 'katie-holmes-tom-cruise.jpg', 'faces': [[55, 102, 93, 141], [72, 116, 197, 241]]},
    {'photo_file_name': 'mccain-palin-hairspray-horror.jpg', 'faces': [[58, 139, 100, 179], [102, 177, 254, 331]]},
    {'photo_file_name': 'obama8.jpg', 'faces': [[85, 157, 109, 180]]},
    {'photo_file_name': 'phil-jackson-and-michael-jordan.jpg', 'faces': [[34, 75, 58, 92], [32, 75, 152, 193]]},
    {'photo_file_name': 'the-lord-of-the-rings_poster.jpg', 'faces': [[222, 267, 0, 35], [129, 170, 6, 40], [13, 81, 26, 84], [22, 92, 120, 188], [35, 94, 225, 276], [190, 255, 235, 289], [301, 345, 257, 298]]}
]
def intersection_area(r1_dims, r2_dims):
    r1_top, r1_bottom, r1_left, r1_right = r1_dims
    r2_top, r2_bottom, r2_left, r2_right = r2_dims

    #compute x overlap
    length_x = max(r1_right, r2_right) - min(r1_left,r2_left)
    intersection_x = length_x - abs(r1_left - r2_left) - abs(r1_right - r2_right)

    #computer y overlap
    length_y = max(r1_bottom, r2_bottom) - min(r1_top, r2_top)
    intersection_y = length_y - abs(r1_bottom - r2_bottom) - abs(r1_top - r2_top)
    
    #area of intersection
    return intersection_x * intersection_y

def union_area(r1_dims, r2_dims):
    intersection = intersection_area(r1_dims, r2_dims)
    r1_top, r1_bottom, r1_left, r1_right = r1_dims
    r2_top, r2_bottom, r2_left, r2_right = r2_dims
    
    r1_area = (r1_bottom - r1_top) * (r1_right - r1_left)
    r2_area = (r2_bottom - r2_top) * (r2_right -r2_left)
    
    # Union(r1,r2) = area_r1 + area_r2 - intersection(r1,r2)
    return r1_area + r2_area - intersection 

def compute_metrics(query_annotations):
    """
    Computes the accuracy and number of false positives against the ground truth face_photo annotations

    Args:
        query_annotations: list of annotations to compute metrics for in the same format as original annotations
        
    Returns:
         None

    """
    global_matches = 0
    total_faces = 0
    falsies = 0
    
    # Iterate over the annotations
    for annotationIndex in range(len(annotations)):
        correct_annotation = annotations[annotationIndex]
        file_name = correct_annotation["photo_file_name"]
        query_annotation = query_annotations[annotationIndex]
        visualization = cv2.imread(file_name)

        correct_faces = correct_annotation['faces']
        query_faces = query_annotation['faces']*
        total_faces += len(correct_faces)
        num_matches = 0
    
        for face in correct_faces:
            top, bottom, left, right = face
            draw_rectangle(visualization, top, bottom, left, right, (0, 255, 0))

        for working_query in query_faces:
            for faceIndex in range(len(correct_faces)):
                face = correct_faces[faceIndex]
                union = union_area(working_query, face)
                intersection = intersection_area(working_query, face)
                # Intersection over Union (IoU) ratio. Mathematically IoU = (Area of Overlap) / (Area of Union) > 0.5.
                IoU = (intersection)/(union)
                if IoU > 0.5:
                    del(correct_faces[faceIndex])
                    num_matches+=1
                    break
                else:
                    
        global_matches += num_matches
        falsies += (len(query_faces) - num_matches)
    
    accuracy = global_matches / total_faces

    print(total_faces)
    print("Total accuracy: ", accuracy * 100, "%")
    print("Number of false positives: ", falsies)