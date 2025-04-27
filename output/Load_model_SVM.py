# import the necessary packages
import numpy as np
import cv2
import pickle
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input

# 1. Khởi tạo danh sách 10 loài vật
classLabels = ["bo", "buom", "ech", "ga", "meo", "ngua", "ran", "rua", "tho", "voi"]

# 2. Nạp hình ảnh cần phân lớp
print("[INFO] Đang nạp ảnh mẫu để phân lớp...")
imagePath = "image//4.jpg"  # đường dẫn ảnh
image = load_img(imagePath, target_size=(224, 224))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)

# 3. Trích xuất đặc trưng bằng VGG16
batchImages = []
batchImages.append(image)
batchImages = np.vstack(batchImages)

model1 = VGG16(weights="imagenet", include_top=False)
features = model1.predict(batchImages)
features = features.reshape((features.shape[0], 7 * 7 * 512))

# 4. Nạp model đã train
print("[INFO] Nạp model mạng pre-trained ...")
model = pickle.load(open('output//model1.cpickle', 'rb'))

# 5. Dự đoán nhãn
print("[INFO] Đang dự đoán để phân lớp...")
preds = model.predict(features)
label = classLabels[int(preds[0])]  # Lấy nhãn theo vị trí số

# 6. Hiển thị kết quả lên ảnh
image = cv2.imread(imagePath)
cv2.putText(image, "Label: {}".format(label), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow("Image", image)
cv2.waitKey(0)
