import numpy as np
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
from PIL import Image

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./face_landmark.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def process_input(img_path,new_w,new_h):
    img = Image.open(img_path)
    img = img.resize((new_w, new_h))
    return img

def inference_facemesh(img_path):
    interpreter = tf.lite.Interpreter(model_path='face_landmark.tflite')
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    floating_model = input_details[0]['dtype'] == np.float32
    
    req_height = input_details[0]['shape'][1]
    req_width = input_details[0]['shape'][2]
    img = process_input(img_path,req_width,req_height)
    input_data = np.expand_dims(img, axis=0)
    
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) /127.5
        
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    results.shape = (468,3)
    
    return results

face_mesh_results = inference_facemesh('./test.jpg')
landmark_points = face_mesh_results[:,0:2]
landmark_points = [tuple(map(int,i)) for i in landmark_points]

points = np.array(landmark_points, np.int32)
convex_hul = cv2.convexHull(points)

img = cv2.imread("test.jpg")
img = cv2.resize(img, (192, 192))

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(img_gray)

# cv2.polylines(img,[convex_hul], True, (255,0,0), 3)
cv2.fillConvexPoly(mask,convex_hul,255)

face_img_1 = cv2.bitwise_or(img,img,mask=mask )

# Delaunav Triangulation
rect = cv2.boundingRect(convex_hul)
subdiv = cv2.Subdiv2D(rect)
subdiv.insert(landmark_points)
triangles = subdiv.getTriangleList()
triangles = np.array(triangles,dtype = np.int32)
print(len(triangles),len(landmark_points))

for t in triangles:
    pt1 = (t[0], t[1])
    pt2 = (t[2], t[3])
    pt3 = (t[4], t[5])
    cv2.line(img, pt1, pt2, (0, 0, 255), 2)
    cv2.line(img, pt2, pt3, (0, 0, 255), 2)
    cv2.line(img, pt1, pt3, (0, 0, 255), 2)

cv2.imshow("Image_1",img)
cv2.imshow("Image_2",mask)
cv2.imshow("face_img_1",face_img_1)
cv2.waitKey(0)
cv2.destroyAllWindows()
