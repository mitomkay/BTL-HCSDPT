import os
import cv2
from numpy import  save

# Trích xuất đặc trưng màu sắc
def extract_color_features(image):
    # Chuyển đổi ảnh từ RGB sang HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Tạo histogram màu sắc theo các kênh Hue, Saturation và Value
    hist = cv2.calcHist(
        [hsv_image], [0, 1, 2], None, [12, 12, 3], [0, 180, 0, 256, 0, 256]
    )
    # Chuẩn hóa histogram (tính xác suất xuất hiện của mỗi giá trị màu)
    hist = cv2.normalize(hist, hist).flatten()
    return hist

source = "data_new"
x = os.listdir(source)

image_paths = []  # Danh sách chứa đường dẫn tới 100 ảnh chim
labels = []  # Danh sách chứa nhãn của từng ảnh
features = []  # Danh sách chứa các vector đặc trưng ảnh

# Thêm đường dẫn và nhãn tương ứng(tên) của ảnh chim vào các danh sách trên
for i in range(0, 10):
    y = os.listdir(source + "/" + x[i])
    for j in range(0, 10):
        image_paths.append(source + "/" + x[i] + "/" + y[j])
        labels.append(x[i])

for image_path in image_paths:
    image = cv2.imread(
        image_path
    )  # Trả về một đối tượng ma trận numpy biểu diễn hình ảnh
    image = cv2.resize(
        image, (350, 700)
    )  # Thay đổi kích thước của hình ảnh về tỉ lệ 350x700
    # Trích xuất đặc trưng màu sắc
    color_feature = extract_color_features(image)
    # Thêm đặc trưng mới vào mảng
    features.append(color_feature)

# Lưu các đặc trưng vào file images_feature.npy
save("images_features.npy", features)
