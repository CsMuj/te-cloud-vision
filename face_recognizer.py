import os, io
from google.cloud import vision
import cv2
from time import sleep
import pprint 

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'GCVision.json' #the credetials to talk to the API.
client = vision.ImageAnnotatorClient()
print("starting object recognizer...")

object_to_find = 0

def get_face_from_cloud(image_path):

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    likelihood_name = ('Unknown', 'Very unlikely', 'Unlikely', 'Possible',
                       'Likely', 'Very likey')

    image = vision.Image(content=content)
    response = client.face_detection(image=image) #full response from Google cloud vision
    faces = response.face_annotations #array of objects labeled
    #print(face_annotations)
    print('Faces:')

    for face in faces:
        print('anger: {}'.format(likelihood_name[face.anger_likelihood]))
        print('joy: {}'.format(likelihood_name[face.joy_likelihood]))
        print('surprise: {}'.format(likelihood_name[face.surprise_likelihood]))

        # vertices = (['({},{})'.format(vertex.x, vertex.y)
        #             for vertex in face.bounding_poly.vertices])

        # print('face bounds: {}'.format(','.join(vertices)))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

def get_image_from_frame(cap):
    ret, frame = cap.read()
    file = 'frame.png'
    cv2.imwrite(file,frame)
    cv2.imshow('frame',frame) #show camera output
    return file

def start_camera():
    global object_to_find
    # os.system('sudo modprobe bcm2835-v4l2') #Force the Raspberry Pi to use the the Picamera, which CV2 will need to capture each frame.

    cap = cv2.VideoCapture(0)
    print("Starting camera")

    while True:
        
        img = get_image_from_frame(cap)
        key = cv2.waitKey(0) #press 0 to move through frames
        object_to_find = get_face_from_cloud(img)

        if key == ord('q'): #press q to quit
            break
    
    cap.release() #release the object when the app quits.
    cv2.destroyAllWindows()

start_camera()