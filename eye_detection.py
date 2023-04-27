#!/usr/bin/env python3

import cv2
from datetime import date
from optparse import OptionParser
from time import strftime, localtime
from colorama import Fore, Back, Style

GREEN = (0, 255, 0)
CYAN = (255, 255, 0)

scale_factor = 1.3
min_neighbors = 5
cascade_file_face = "haarcascade_frontalface_default.xml"
cascade_file_eye = "haarcascade_eye.xml"

status_color = {
	'+': Fore.GREEN,
	'-': Fore.RED,
	'*': Fore.YELLOW,
	':': Fore.CYAN,
	' ': Fore.WHITE,
}

def get_time():
	return strftime("%H:%M:%S", localtime())
def display(status, data):
	print(f"{status_color[status]}[{status}] {Fore.BLUE}[{date.today()} {get_time()}] {status_color[status]}{Style.BRIGHT}{data}{Fore.RESET}{Style.RESET_ALL}")

def get_arguments(*args):
	parser = OptionParser()
	for arg in args:
		parser.add_option(arg[0], arg[1], dest=arg[2], help=arg[3])
	return parser.parse_args()[0]

def draw_rects(image, rects, color):
    for x, y, w, h in rects:
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
def detect_faces(image, face_classifier, scale_factor, min_neighbors, localize=True):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    if localize:
        draw_rects(image, faces, GREEN)
    return faces
def detect_eye(image, eye_classifier, localize=True):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_classifier.detectMultiScale(gray_image)
    if localize:
        draw_rects(image, eyes, CYAN)
    return eyes

if __name__ == "__main__":
    data = get_arguments(('-i', "--image", "image", "Path to the Image file (If not specified, will take frame from Camera Stream)"),
                         ('-f', "--cascade-file-face", "cascade_file_face", f"Path to cascade File for Face Detection (Default={cascade_file_face})"),
                         ('-e', "--cascade-file-eye", "cascade_file_eye", f"Path to cascade File for Eye Detection (Default={cascade_file_eye})"),
                         ('-s', "--scale-factor", "scale_factor", f"Scale Factor (Default={scale_factor})"),
                         ('-m', "--min-neighbors", "min_neighbors", f"Minimum Neighbors (Default={min_neighbors})"))
    if not data.cascade_file_face:
        data.cascade_file_face = cascade_file_face
    if not data.cascade_file_eye:
        data.cascade_file_eye = cascade_file_eye
    if not data.scale_factor:
        data.scale_factor = scale_factor
    else:
        data.scale_factor = int(data.scale_factor)
    if not data.min_neighbors:
        data.min_neighbors = min_neighbors
    else:
        data.min_neighbors = int(data.min_neighbors)
    face_classifier = cv2.CascadeClassifier(data.cascade_file_face)
    eye_classifier = cv2.CascadeClassifier(data.cascade_file_eye)
    if data.image:
        try:
            image = cv2.imread(data.image)
        except:
            display('-', f"Failed to read the Image : {Back.MAGENTA}{data.image}{Back.RESET}")
        faces = detect_faces(image, face_classifier, data.scale_factor, data.min_neighbors, localize=True)
        print(f"\r{Fore.CYAN}{Back.MAGENTA}{len(faces)}{Back.RESET} {Fore.GREEN}Faces Detected{Fore.RESET} ,", end='')
        for face in faces:
            x, y, w, h = face
            face_image = image[x:x+w, y:y+h]              # Cropping out the face to make the program fast
            eyes = detect_eye(face_image, eye_classifier, localize=True)
            print(f"{Fore.CYAN}{Back.MAGENTA}{len(eyes)}{Back.RESET} {Fore.GREEN}Eyes Detected{Fore.RESET}", end='')
        cv2.imshow("Image", image)
        cv2.waitKey()
    else:
        video_capture = cv2.VideoCapture(0)
        while True:
            ret, frame = video_capture.read()
            if not ret:
                display('-', "Failed to get Frame from the Camera")
                break
            faces = detect_faces(frame, face_classifier, data.scale_factor, data.min_neighbors, localize=True)
            print(f"\r{Fore.CYAN}{Back.MAGENTA}{len(faces)}{Back.RESET} {Fore.GREEN}Faces Detected{Fore.RESET}, ", end='')
            for face in faces:
                x, y, w, h = face
                face_image = frame[x:x+w, y:y+h]          # Cropping out the face to make the program fast
                eyes = detect_eye(face_image, eye_classifier, localize=True)
                print(f"{Fore.CYAN}{Back.MAGENTA}{len(eyes)}{Back.RESET} {Fore.GREEN}Eyes Detected{Fore.RESET}", end='')
            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) >= 0:
                break
        video_capture.release()
        cv2.destroyAllWindows()
    print()