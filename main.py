
import streamlit as st
from keras.preprocessing import image
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import warnings


warnings.filterwarnings(action='ignore')

import seaborn as sns
sns.set_style('whitegrid')

#đọc mô hình, hình ảnh, data
class_names = ['IncorrectlyWornMask', 'WithMask','WithoutMask']
svm_sklearn1 = pickle.load(open("models/svm_sklearn.pkl","rb"))
knn_model = pickle.load(open("models/knn.pkl","rb"))
df_compare_svm = pickle.load(open("models/dataframe_compare.pkl","rb"))
svm_classifiers = pickle.load(open("models/svm_build.pkl","rb"))
voting_classification_model = pickle.load(open("models/voting.pkl","rb"))
rf_model = pickle.load(open("models/rf.pkl","rb"))



# quanlity load file
svm_quanlity = pickle.load(open("models/svm_quality.pkl","rb"))
seft_svm_quanlity = pickle.load(open("models/seft_svm_quality.pkl","rb"))
knn_quanlity = pickle.load(open("models/knn_quality.pkl","rb"))
voting_quanlity = pickle.load(open("models/voting_quality.pkl","rb"))
rf_quanlity = pickle.load(open("models/rf_quality.pkl","rb"))



st.sidebar.title("Hệ thống nhận diện đeo khẩu trang")
st.sidebar.markdown("Hình ảnh của bạn là: ")
st.sidebar.markdown("🚫IncorrectlyWornMask ✅With mash 🍄Without mask")


# @st.cache
def loadTrainData():
    pickle_in = open("pickle/X.pickle", "rb")
    X = pickle.load(pickle_in)
    pickle_in = open("pickle/y.pickle", "rb")
    y = pickle.load(pickle_in)
    # Split our data into testing and training.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
    return X_train,y_train


image_data = []
labels = []
image_data,labels = loadTrainData()

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
    pickle_in = open("pickle/y.pickle", "rb")
    y = pickle.load(pickle_in)
    sizes = [(y == 0).sum(),(y == 1).sum(),(y == 2).sum()]
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






class_names = ['Not Spam', 'Spam']


st.sidebar.subheader("Choose Classifier")


classifier = st.sidebar.selectbox("Classification Algorithms",
                                     ("Support Vector Machine (thư viện)",
                                         "Support Vector Machine (Tự xây dựng)",
                                         "KNN",
                                         "Random forest",
                                         "SVM + KNN + Random forest"
                                      ))

if classifier == 'Support Vector Machine (thư viện)':
    st.subheader("SVM thư viện")
    image1 = Image.open('imagemodels/SVM_sklearn.png')
    st.image(image1, caption='SVM confusion matrix', use_column_width=True)
    accuracy = df_compare_svm['Accuracy'][1]
    precision = df_compare_svm['Precision score'][1]
    recall = df_compare_svm['Recall score'][1]
    f1score = df_compare_svm['F1 score'][1]
    st.write("Accuracy ", accuracy.round(4)*100)
    st.write("Precision score ", precision.round(4)*100)
    st.write("Recall score ", recall.round(4)*100)
    st.write("F1 score ", f1score.round(4)*100)


if classifier == "SVM + KNN + Random forest":
    st.subheader("SVM + KNN + Random forest")
    image1 = Image.open('imagemodels/voting_confusion.png')
    st.image(image1, caption="SVM + KNN + Random forest", use_column_width=True)
    accuracy = voting_quanlity[0]
    precision = voting_quanlity[1]
    recall = voting_quanlity[2]
    f1score = voting_quanlity[3]
    st.write("Accuracy ", accuracy.round(4)*100)
    st.write("Precision score ", precision.round(4)*100)
    st.write("Recall score ", recall.round(4)*100)
    st.write("F1 score ", f1score.round(4)*100)


if classifier == 'KNN':
    st.subheader("KNN")
    image1 = Image.open('imagemodels/knn_confusion.png')
    st.image(image1, caption='KNN confusion matrix', use_column_width=True)
    accuracy = knn_quanlity[0]
    precision = knn_quanlity[1]
    recall = knn_quanlity[2]
    f1score = knn_quanlity[3]
    st.write("Accuracy ", accuracy.round(4)*100)
    st.write("Precision score ", precision.round(4)*100)
    st.write("Recall score ", recall.round(4)*100)
    st.write("F1 score ", f1score.round(4)*100)


if classifier == 'Random forest':
    st.subheader("Random forest")
    image1 = Image.open('imagemodels/rf_confusion.png')
    st.image(image1, caption='Random forest confusion matrix', use_column_width=True)
    accuracy = rf_quanlity[0]
    precision = rf_quanlity[1]
    recall = rf_quanlity[2]
    f1score = rf_quanlity[3]
    st.write("Accuracy ", accuracy.round(4)*100)
    st.write("Precision score ", precision.round(4)*100)
    st.write("Recall score ", recall.round(4)*100)
    st.write("F1 score ", f1score.round(4)*100)



if classifier == 'Support Vector Machine (Tự xây dựng)':
    st.subheader("SVM tự xây dựng")
    image1 = Image.open('imagemodels/SVM_seft_confusion.png')
    st.image(image1, caption='SVM self confusion matrix', use_column_width=True)
    accuracy = df_compare_svm['Accuracy'][0]
    precision = df_compare_svm['Precision score'][0]
    recall = df_compare_svm['Recall score'][0]
    f1score = df_compare_svm['F1 score'][0]
    st.write("Accuracy ", accuracy.round(4)*100)
    st.write("Precision score ", precision.round(4)*100)
    st.write("Recall score ", recall.round(4)*100)
    st.write("F1 score ", f1score.round(4)*100)
 

if st.sidebar.checkbox("Hiển thị bảng đánh giá", False):
    st.subheader("Đánh giá chất lượng các mô hình ")
    compare = pd.DataFrame({
    'Accuracy':[ seft_svm_quanlity[0] ,svm_quanlity[0],knn_quanlity[0],rf_quanlity[0], voting_quanlity[0]],
    'Precision score': [seft_svm_quanlity[1],svm_quanlity[1],knn_quanlity[1],rf_quanlity[1], voting_quanlity[1]],
    'Recall score': [seft_svm_quanlity[2],svm_quanlity[2],knn_quanlity[2],rf_quanlity[2], voting_quanlity[2]],
    'F1 score': [seft_svm_quanlity[3],svm_quanlity[3],knn_quanlity[3],rf_quanlity[3], voting_quanlity[3]]

    })
    compare.index = ['Self-built SVM model','Model SVM library', "KNN","Randrom forest", "SVM + KNN + Random forest" ]
    st.write(compare)
st.balloons()



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
    if classifier == 'KNN':
        imgpre_array = coverimagetoarray(uploaded_file.name)
        pre = knn_model.predict([imgpre_array])
        st.sidebar.success(" <" + switcher.get(pre[0], "nothing") + ">  :))")
    if classifier == 'Random forest':
        imgpre_array = coverimagetoarray(uploaded_file.name)
        pre = rf_model.predict([imgpre_array])
        st.sidebar.success(" <" + switcher.get(pre[0], "nothing") + ">  :))")
    if classifier == "SVM + KNN + Random forest":
        imgpre_array = coverimagetoarray(uploaded_file.name)
        pre = voting_classification_model.predict([imgpre_array])
        st.sidebar.success(" <" + switcher.get(pre[0], "nothing") + ">  :))")
st.write()



