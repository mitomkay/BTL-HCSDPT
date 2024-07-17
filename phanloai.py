import os
import cv2

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image, ImageDraw, ImageFont
from numpy import load
from xulydactrung import extract_color_features

source = "data_new"
x = os.listdir(source)

image_paths = []  # Danh sách chứa đường dẫn tới 100 ảnh chim
labels = []  # Danh sách chứa nhãn của từng ảnh
features = []  # Danh sách chứa các vector đặc trưng ảnh

# Thêm đường dẫn và nhãn tương ứng(tên) của ảnh chim vào các danh sách trên
for i in range(0, 10):
    # print(os.listdir(source + "/" + x[i]))
    y = os.listdir(source + "/" + x[i])
    for j in range(0, 10):
        image_paths.append(source + "/" + x[i] + "/" + y[j])
        labels.append(x[i])
# print(image_paths)

features = load("images_features.npy")

# Sử dụng mô hình học máy theo thuật K láng giềng gần nhất (KNN)
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2
)  # x: tập vector đặc trưng, y: tập các nhãn, Lấy 20% tập dữ liệu để test, 80% để train

k = 3  # Số lượng láng giềng gần nhất để tham khảo khi dự đoán

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)

# Tính độ chính xác của mô hình
y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# print(y_test)
# print(y_pred)

# Nhân dạng ảnh chim với đầu vào là ảnh không có trong tập dữ liệu
nhap = input()
input_path = "test/" + nhap + ".jpg"  # Đương dẫn ảnh đầu vào

image_input = cv2.imread(
    input_path
)  # Trả về một đối tượng ma trận numpy biểu diễn hình ảnh
image_input = cv2.resize(
    image_input, (350, 700)
)  # Thay đổi kích thước của hình ảnh về tỉ lệ 350x700
feature_output = extract_color_features(image_input).reshape(
    1, -1
)  # Trích xuất đặc trưng màu sắc của ảnh đầu vào
# print(feature_output)
print(knn.predict(feature_output))

# In ra ảnh cần nhận dạng kèm nhãn phân loại
# Tạo đối tượng Image từ tệp ảnh
image = Image.open(input_path)
# Tạo đối tượng ImageDraw để vẽ lên ảnh
draw = ImageDraw.Draw(image)

# Tạo đối tượng ImageFont cho phông chữ
font_path = "Arial.ttf"
font = ImageFont.truetype(font_path, size=24)  # Thay đổi font và kích thước tùy ý

# Vị trí và nội dung của văn bản
text_position = (150, 20)  # Tọa độ x, y để vẽ văn bản
text_content = knn.predict(feature_output)  # Nội dung văn bản

# Màu sắc của văn bản
text_color = (255, 0, 0)  # Màu RGB, ở đây là đỏ

# Vẽ văn bản lên ảnh
draw.text(text_position, text_content[0], font=font, fill=text_color)

# Hiển thị ảnh
image.show()
