import cv2
import dlib
import numpy

predictor_path = "data/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

class TooManyFaces(Exception):
    # print('Too many faces in the given image')
    pass

class NoFaces(Exception):
    # print("no face in the given image")
    pass

def get_landmarks(img):
    rects = detector(img, 1)

    if (len(rects) > 1):
        raise TooManyFaces

    if (len(rects) == 0):
        raise NoFaces


    return numpy.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])


def annotate_landmarks(img, landmarks):
    img = img.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(img, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.3,
                    color=(176, 254, 55))
        cv2.circle(img, pos, 3, color=(255, 0, 255))
        
    return img

image = cv2.imread("images/IMG_20170811_212205_390.jpg")
landmarks = get_landmarks(image)

# print(landmarks)
image_with_landmarks = annotate_landmarks(image, landmarks)

cv2.imshow("Result Picture", image_with_landmarks)
cv2.imwrite('images/image_with_landmarks.jpg', image_with_landmarks)
cv2.waitKey(0)
cv2.destroyAllWindows()
