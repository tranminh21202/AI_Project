import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tkinter as tk

# Hàm xử lý sự kiện khi bấm nút "Predict"
def predict():
    input_mail = [entry.get()]
    input_data_features = feature_extraction.transform(input_mail)
    prediction = model.predict(input_data_features)
    if prediction[0] == 1:
        result_label.config(text="Đây là email thường (ham)")
    else:
        result_label.config(text="Đây là email rác (spam)")

# Đọc dữ liệu từ file csv
raw_mail_data = pd.read_csv("mail_data1.csv")

# Thay thế các giá trị null bằng chuỗi rỗng
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), "")

# Đổi nhãn "spam" thành 0, "ham" thành 1
mail_data.loc[mail_data["Category"] == "spam", "Category",] = 0
mail_data.loc[mail_data["Category"] == "ham", "Category",] = 1

X = mail_data["Message"]
Y = mail_data["Category"]

# Chia dữ liệu thành tập train và tập test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Tạo đối tượng TfidfVectorizer để trích xuất đặc trưng
feature_extraction = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype("int")
Y_test = Y_test.astype("int")

# Huấn luyện mô hình SVM
model = svm.SVC(kernel = 'linear', C=1.0)
#model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Tạo GUI
root = tk.Tk()
root.title("Phân loại email")
root.geometry("600x350")

# Tạo widget Label và Entry để nhập nội dung email
label = tk.Label(root, text="Nhập nội dung email:")
label.pack(pady=10)

entry = tk.Entry(root, width=50)
entry.pack()

# Tạo widget Button để dự đoán
button = tk.Button(root, text="Predict", command=predict)
button.pack(pady=10)

# Tạo widget Label để hiển thị kết quả
result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()