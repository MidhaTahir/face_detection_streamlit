import streamlit as st
import cv2
import numpy as np
from PIL import Image
from time import sleep

print(cv2.__version__)

@st.cache
def load_image(img):
	im = Image.open(img)
	return im


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def detect_faces(image):
    new_img = np.array(image)
    img = cv2.cvtColor(new_img,1)
    gray = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
    return img,faces

def main():
    st.header('Face Detection Application')
    img = st.file_uploader("Uploads an image",type=['jpg','png','jpeg'])

    if img is not None:
        my_image = load_image(img)
        st.subheader('You uploaded below image: ')
        st.image(my_image,width=100)


    if st.button("DETECT"):
        if img is not None:
            final_img,final_faces = detect_faces(my_image)
            st.image(final_img,width=400)
            st.success("The total faces on image are: {}".format(len(final_faces)))
            sleep(2)
            st.balloons()
        if img is None:
            st.warning('Please upload image')
    
    
    

    if st.button('About me'):
        st.markdown('Follow me on Github: [Midha Tahir](https://github.com/MidhaTahir/)')


if __name__ == '__main__':
    main()