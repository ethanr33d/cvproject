import cv2

def draw_rectangle(img, top, bottom, left, right, color=(255,255,255)):
    # Determine if the image is grayscale or color
    if len(img.shape) == 2 or img.shape[2] == 1:  # Grayscale
        color = 255  # White color for grayscale
    
    thickness = 2
    cv2.rectangle(img, (left, top), (right, bottom), color, thickness)
    return img