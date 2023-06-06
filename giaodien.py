
import streamlit as st
from pathlib import Path
from keras.preprocessing import image
import pandas as pd
import numpy as np
from PIL import Image

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score
import os
import matplotlib.pyplot as plt
# from win32com.client import Dispatch

import pickle
import warnings
warnings.filterwarnings(action='ignore')


import seaborn as sns
sns.set_style('whitegrid')

#đọc mô hình, hình ảnh, data
class_names = ['IncorrectlyWornMask', 'WithMask','WithoutMask']
svm_sklearn1 = pickle.load(open("models/svm_sklearn.pkl","rb"))
df_compare = pickle.load(open("models/dataframe_compare.pkl","rb"))
svm_classifiers = pickle.load(open("models/svm_build.pkl","rb"))

st.sidebar.title("Hệ thống nhận diện đeo khẩu trang với thuật toán SVM")
st.sidebar.markdown("Hình ảnh của bạn là: ")
st.sidebar.markdown("🚫IncorrectlyWornMask ✅With mash 🍄Without mask")


@st.cache
def loadimg():
    #đọc dữ liệu hình ảnh, chuyển đổi
    p = Path("data/")
    dirs = p.glob("*")
    labels_dict = {'IncorrectlyWornMask':0,'WithMask':1,'WithoutMask':2 }

    image_data = []
    labels = []
    for folder_dir in dirs:
        print(str(folder_dir))
        label = str(folder_dir).split("/")[-1]
        print("doc anh thanh cong thu muc:",label)
        for img_path in folder_dir.glob("*"):
            img = image.load_img(img_path, target_size=(32,32))
            img_array = image.img_to_array(img)
            image_data.append(img_array)
            labels.append(labels_dict[label])
        


    ## Chuyển đổi dữ liệu thành mảng numpy 
    image_data = np.array(image_data, dtype='float32')/255.0
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=45)

    ## Chuyển đổi dữ liệu cho phân loại Một vs Một
    #train
    M = X_train.shape[0]
    X_train = X_train.reshape(M,-1)
    return X_train,y_train

image_data = []
labels = []
image_data,labels = loadimg()

#dữ liệu biểu đồ tròn
df_labels = pd.DataFrame(
    labels,
    columns=['label']
)

#dự đoán
switcher = { 0 : 'IncorrectlyWornMask',1 :'WithMask', 2 :'WithoutMask'}


number_of_classes = len(np.unique(labels))
def binaryPredict(x,w,b):
    z = np.dot(x,w.T) + b
    if z >= 0:
        return 1
    else:
        return -1

def predict(x):
    count = np.zeros((number_of_classes,))
    for i in range(number_of_classes):
        for j in range(i+1, number_of_classes):
            w,b = svm_classifiers[i][j]
            #
            z = binaryPredict(x,w,b)
            #(lớp có tổng điểm lớn nhất) được dự đoán là nhãn lớp.
            if z==1:
                count[j] += 1
            else:
                count[i] += 1

    final_prediction = np.argmax(count)
    return final_prediction

def coverimagetoarray(pathimg):
    imgpre = image.load_img("imgthucte/"+str(pathimg), target_size=(32, 32))
    imgpre_array = image.img_to_array(imgpre)
    imgpre_array = np.array(imgpre_array, dtype='float32') / 255.0
    imgpre_array = imgpre_array.reshape(-1, )
    return imgpre_array

col1, col2 = st.columns(2)

with col1:
    st.header("Phần trăm dữ liệu")


    labels_circe =  'IncorrectlyWornMask','WithMask','WithoutMask'
    sizes = [2000,2000,2000]
    explode = (0, 0.05,0.05)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots(1)
    ax1.pie(sizes, explode=explode, labels=labels_circe, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig1)
with col2:
    st.header("Số lượng dữ liệu")
    fig2 = plt.figure(figsize=(8, 6))
    sns.countplot(data=df_labels, x='label')
    st.pyplot(fig2)
def plot_metrics(metrics_list):
    st.set_option('deprecation.showPyplotGlobalUse', False)



    if 'Confusion Matrix' in metrics_list:

        st.subheader("Confusion Matrix")
        # st.image(image1, caption='Ma trận nhầm lẫn với tập test')

    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        # st.image(image2, caption='Đường cong ROC với tập test')



class_names = ['Not Spam', 'Spam']


st.sidebar.subheader("Choose Classifier")


classifier = st.sidebar.selectbox("Classification Algorithms",
                                     ("Support Vector Machine (thư viện)",
                                         "Support Vector Machine (Tự xây dựng)"
                                      ))

if classifier == 'Support Vector Machine (thư viện)':

    metrics = st.sidebar.multiselect("Chọn chỉ số lập biểu đồ?",
                                     ('Confusion Matrix','ROC Curve'))

    st.subheader("SVM thư viện")
    # image1 = Image.open('imagemodels/cfm_sklearn.png')
    # image2 = Image.open('imagemodels/Multiclass ROC sklearn.png')
    accuracy = df_compare['Accuracy'][1]
    precision = df_compare['Precision score'][1]
    recall = df_compare['Recall score'][1]
    f1score = df_compare['F1 score'][1]
    # y_pred = model.predict(image_data_test)
    st.write("Accuracy ", accuracy.round(4)*100)
    st.write("Precision score ", precision.round(4)*100)
    st.write("Recall score ", recall.round(4)*100)
    st.write("F1 score ", f1score.round(4)*100)
    plot_metrics(metrics)
    #dự đoán


if classifier == 'Support Vector Machine (Tự xây dựng)':

    metrics = st.sidebar.multiselect("Chọn chỉ số lập biểu đồ?",
                                     ('Confusion Matrix','ROC Curve'))

    st.subheader("SVM tự xây dựng")
    # image1 = Image.open('imagemodels/cfm_build.png')
    # image2 = Image.open('imagemodels/Multiclass ROC sklearn.png')
    accuracy = df_compare['Accuracy'][0]
    precision = df_compare['Precision score'][0]
    recall = df_compare['Recall score'][0]
    f1score = df_compare['F1 score'][0]
    #y_pred = model.predict(image_data_test)
    st.write("Accuracy ", accuracy.round(4)*100)
    st.write("Precision score ", precision.round(4)*100)
    st.write("Recall score ", recall.round(4)*100)
    st.write("F1 score ", f1score.round(4)*100)
    plot_metrics(metrics)


if st.sidebar.checkbox("Hiển thị bảng đánh giá", False):
    st.subheader("Đánh giá chất lượng 2 mô hình ")
    st.write(df_compare)
st.balloons()

col3, col4, col5 = st.columns(3)

with col3:
    st.header("IncorrectWornMask - WithMask")
    image3 = Image.open('imagemodels/plot_loss_0-1.png')
    st.image(image3,"Biểu đồ sự mất mát qua các lần đào tạo")

with col4:
    st.header("IncorrectWornMask - WithoutMask")
    image3 = Image.open('imagemodels/plot_loss_0-2.png')
    st.image(image3, "Biểu đồ sự mất mát qua các lần đào tạo")

with col5:
    st.header("WithMask - WithoutMask")
    image3 = Image.open('imagemodels/plot_loss_1-2.png')
    st.image(image3, "Biểu đồ sự mất mát qua các lần đào tạo")

#giao dien side bar dự đoán

switcher = { 0 : 'IncorrectlyWornMask',1 :'WithMask', 2 :'WithoutMask'}

uploaded_file = st.sidebar.file_uploader("Upload Files", type=['png', 'jpeg', 'jpg'])
if uploaded_file is not None:
    st.header("Hình ảnh được dự đoán")
    image3 = Image.open(uploaded_file)
    st.image(image3, "Hình ảnh đã tải lên để dự đoán")

if st.sidebar.button("Predict"):
    if classifier == 'Support Vector Machine (thư viện)':
        imgpre_array = coverimagetoarray(uploaded_file.name)
        pre = svm_sklearn1.predict([imgpre_array])
        print(pre)
        st.sidebar.success(" <" + switcher.get(pre[0], "nothing") + ">  :))")
    if classifier == 'Support Vector Machine (Tự xây dựng)':
        imgpre_array = coverimagetoarray(uploaded_file.name)
        pre = predict(imgpre_array)
        st.sidebar.success(" <" + switcher.get(pre, "nothing") + ">  :))")
st.write()



