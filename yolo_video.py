#---- import library ----
import cv2
import numpy as np

#---- read video ----
capture = cv2.VideoCapture('./computer vision/yolo_test.mp4')

# Image label
"""
0 = person
2 = car
"""

#--- define classes what we want to predict -----
classes = ['car','person']

#--- define yolo as our model ----
neural_network = cv2.dnn.readNetFromDarknet('yolov3.cfg','yolov3.weights')
neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
neural_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#---- create function to find object ---
#setting threshold
threshold = 0.3

#define size
yolo_image_size = 320

#supression threshold
suppression_threshold = 0.3
def find_objects(model_output):
    bbox_locations = []
    class_ids = []
    confindence_values = []

    for output in model_output:
        for prediction in output:
            class_probabilities = prediction[5:]
            class_id = np.argmax(class_probabilities)
            confidence = class_probabilities[class_id]

            if confidence > threshold:
                #--- get widht and height -----
                w,h = int(prediction[2]*yolo_image_size),int(prediction[3]*yolo_image_size)

                #---- center of coordinate ----
                x,y = int(prediction[0]*yolo_image_size-w/2),int(prediction[1]*yolo_image_size-h/2)

                #--- store the values ---
                bbox_locations.append([x,y,w,h])
                class_ids.append(class_id)
                confindence_values.append(float(confidence))

    box_indexes_to_keep = cv2.dnn.NMSBoxes(bbox_locations,confindence_values,threshold,suppression_threshold)

    return box_indexes_to_keep,bbox_locations,class_ids,confindence_values

def show_detected_object(img,bbox_ids,all_bounding_boxes,class_ids,confidence_values,width_ratio,height_ratio):
    for index in bbox_ids:
        bounding_box = all_bounding_boxes[index]
        x,y,w,h = int(bounding_box[0]),int(bounding_box[1]),int(bounding_box[2]),int(bounding_box[3])

        #transform location
        x = int(x*width_ratio)
        y = int(y*height_ratio)
        w = int(w*width_ratio)
        h = int(h*height_ratio)

        if class_ids[index]==2:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            class_with_confidence = 'CAR ' + str(int(confidence_values[index]*100)) + "%"
            cv2.putText(img,class_with_confidence,(x,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(255,0,0),1)

        if class_ids[index]==0:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            class_with_confidence = 'PERSON ' + str(int(confidence_values[index]*100)) + "%"
            cv2.putText(img,class_with_confidence,(x,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(255,0,0),1)

while True:


    frame_grab,frame = capture.read()

    if not frame_grab:
        break

    # --- get width and height from the frame ----
    original_width, original_height = frame.shape[1], frame.shape[0]

    #--- the image into a BLOB ----
    blob = cv2.dnn.blobFromImage(frame,1/255,(320,320),True,crop=False)
    neural_network.setInput(blob)

    #---- get the layer and output result from model -----
    layer_names = neural_network.getLayerNames()

    output_names = [layer_names[index - 1] for index in neural_network.getUnconnectedOutLayers()]
    outputs = neural_network.forward(output_names)
    predicted_objects, bbox_locations, class_label_ids, conf_values = find_objects(outputs)
    show_detected_object(frame, predicted_objects, bbox_locations, class_label_ids, conf_values,
                         original_width / yolo_image_size, original_height / yolo_image_size)

    cv2.imshow('YOLOV3 Algorithm', frame)
    cv2.waitKey(1)

capture.release()
cv2.destroyAllWindows()
