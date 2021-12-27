from functools import wraps
import streamlit as st
from helper import load_model, convert, load_img, foods, top_5_predictions

model = load_model("model.h5")

st.markdown("# Food classifier")
st.markdown("<h6>Check the sidebar for the types of food that can be detected!</h6>", unsafe_allow_html=True)

st.sidebar.selectbox(
    "List of food classes that are able to predict",
    foods
)

image_file = st.file_uploader("Choose an image file", type=['jpeg', 'jpg', 'png'])

if image_file:
  st.image(load_img(image_file))

  img = convert(image_file)

  custom_preds = model.predict(img)

  pred_class = foods[custom_preds.argmax()]

  st.write(f"Predicted: {pred_class}, with an accuracy of {custom_preds[0].max()*100:.0f}%!")

  st.markdown("## Top 5 predictions")

  top_5_predictions(custom_preds[0])