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

input_details, output_details

# Load and process image


def process_input(img_path, new_w=192, new_h=192, img_format='pil_image'):
    if img_format == 'pil_image':
        img = Image.open(img_path)
        img = img.resize((new_w, new_h))

    if img_format == 'cv2':
        img = cv2.imread(img_path)
        img = cv2.resize(img, (new_w, new_h))

    return img


# Inference on image
def inference_facemesh(img_path):
    interpreter = tf.lite.Interpreter(model_path='face_landmark.tflite')
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    floating_model = input_details[0]['dtype'] == np.float32

    req_height = input_details[0]['shape'][1]
    req_width = input_details[0]['shape'][2]
    img = process_input(img_path, req_width, req_height)
    input_data = np.expand_dims(img, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    results.shape = (468, 3)

    return results


def display_image(face_mesh_result, img_path):
    plt.imshow(process_input(img_path, 192, 192), zorder=1)
    x, y = face_mesh_result[:, 0], face_mesh_result[:, 1]
    plt.scatter(x, y, zorder=2, s=1.0)
    plt.show()


source_path = './bradley_cooper.jpg'
target_path = './jim_carrey.jpg'
source_mesh_results = inference_facemesh(source_path)
target_mesh_results = inference_facemesh(target_path)

display_image(source_mesh_results, source_path)
display_image(target_mesh_results, target_path)


def process_landmark_points(face_mesh_result):
    landmark_points_org = face_mesh_result[:, 0:2]
    landmark_points_org = [
        list(map(int, i)) for i in landmark_points_org
    ]
    landmark_points = [tuple(i) for i in landmark_points_org]
    points = np.array(landmark_points, np.int32)
    convex_hul = cv2.convexHull(points)

    return points, convex_hul


source_landmark_points, source_convex_hul = process_landmark_points(
    source_mesh_results)
target_landmark_points, target_convex_hul = process_landmark_points(
    target_mesh_results)

img_1 = process_input(source_path,img_format='cv2')
img_gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
mask_1 = np.zeros_like(img_gray_1)
face_img_1 = cv2.bitwise_or(img_1, img_1, mask=mask_1)

# Delaunav Triangulation for face 1
rect = cv2.boundingRect(source_convex_hul)
subdiv = cv2.Subdiv2D(rect)
subdiv.insert(source_landmark_points)
triangles_face1 = subdiv.getTriangleList()
triangles_face1 = np.array(triangles_face1, dtype=np.int32)

triangles_1_index = []
for t in triangles_face1:
    pt1 = (t[0], t[1])
    pt1_index = np.where((source_landmark_points == pt1).all(axis=1))[0][0]
    pt2 = (t[2], t[3])
    pt2_index = np.where((source_landmark_points == pt2).all(axis=1))[0][0]
    pt3 = (t[4], t[5])
    pt3_index = np.where((source_landmark_points == pt3).all(axis=1))[0][0]
    triangles_1_index.append([pt1_index, pt2_index, pt3_index])

