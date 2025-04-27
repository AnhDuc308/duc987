import cv2
import numpy as np
import pickle
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input

# 1. Khởi tạo danh sách nhãn của các loài vật
classLabels = ["bo", "buom", "ech", "ga", "meo", "ngua", "ran", "rua", "tho", "voi"]

# 2. Nạp model đã huấn luyện
print("[INFO] Nạp model mạng pre-trained ...")
model = pickle.load(open('output//model1.cpickle', 'rb'))

# 3. Mở video
videoPath = "video/0.mp4"  # Đường dẫn đến video
cap = cv2.VideoCapture(videoPath)

# Kiểm tra nếu video mở thành công
if not cap.isOpened():
    print("Error: Không thể mở video.")
    exit()

while True:
    # Đọc từng khung hình của video
    ret, frame = cap.read()
    if not ret:
        break  # Nếu không có khung hình thì thoát vòng lặp

    # Thay đổi kích thước khung hình cho phù hợp với mô hình
    image = cv2.resize(frame, (224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # Trích xuất đặc trưng với VGG16
    model1 = VGG16(weights="imagenet", include_top=False)
    features = model1.predict(image)
    features = features.reshape((features.shape[0], 7 * 7 * 512))

    # Dự đoán nhãn
    preds = model.predict(features)
    label = classLabels[int(preds[0])]

    # Hiển thị nhãn lên khung hình
    cv2.putText(frame, "Label: {}".format(label), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Hiển thị khung hình
    cv2.imshow("Frame", frame)

    # Nếu nhấn phím 'q' sẽ thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng và đóng video
cap.release()
cv2.destroyAllWindows()
