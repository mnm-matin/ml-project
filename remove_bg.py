
# Import necessary libraries
import cv2
import numpy as np
import urllib.request

# Load the pre-trained Deep Learning model for image segmentation
url = 'https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt'
filename = 'faster_rcnn_inception_v2_coco_2018_01_28.pbtxt'
urllib.request.urlretrieve(url, filename)

url = 'http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz'
filename = 'faster_rcnn_inception_v2_coco_2018_01_28.tar.gz'
urllib.request.urlretrieve(url, filename)

model = cv2.dnn.readNetFromTensorflow('faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb',
                                      'faster_rcnn_inception_v2_coco_2018_01_28.pbtxt')

# Load the input image
img = cv2.imread('input_image.jpg')

# Convert the image to a blob for feeding it into the network
blob = cv2.dnn.blobFromImage(img, swapRB=True, crop=False)

# Set the blob as input to the network
model.setInput(blob)

# Forward pass through the model
output = model.forward()

# Get the segmented mask with confidence score greater than 0.5
for i in range(output.shape[2]):
    confidence = output[0, 0, i, 2]
    if confidence > 0.5:
        class_id = int(output[0, 0, i, 1])
        if class_id == 15: # 15 is the class ID of "person" in COCO dataset
            mask = output[0, 0, i, 3:]
            mask = np.reshape(mask, (img.shape[0], img.shape[1], 1))
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            mask = (mask > 0.5).astype(np.uint8) * 255

# Apply the mask to the input image to remove the background
result = cv2.bitwise_and(img, img, mask=mask)

# Save the output image
cv2.imwrite('output_image.jpg', result)
