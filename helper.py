import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
sns.set()

foods = np.array(['apple_pie',
 'baby_back_ribs',
 'baklava',
 'beef_carpaccio',
 'beef_tartare',
 'beet_salad',
 'beignets',
 'bibimbap',
 'bread_pudding',
 'breakfast_burrito',
 'bruschetta',
 'caesar_salad',
 'cannoli',
 'caprese_salad',
 'carrot_cake',
 'ceviche',
 'cheese_plate',
 'cheesecake',
 'chicken_curry',
 'chicken_quesadilla',
 'chicken_wings',
 'chocolate_cake',
 'chocolate_mousse',
 'churros',
 'clam_chowder',
 'club_sandwich',
 'crab_cakes',
 'creme_brulee',
 'croque_madame',
 'cup_cakes',
 'deviled_eggs',
 'donuts',
 'dumplings',
 'edamame',
 'eggs_benedict',
 'escargots',
 'falafel',
 'filet_mignon',
 'fish_and_chips',
 'foie_gras',
 'french_fries',
 'french_onion_soup',
 'french_toast',
 'fried_calamari',
 'fried_rice',
 'frozen_yogurt',
 'garlic_bread',
 'gnocchi',
 'greek_salad',
 'grilled_cheese_sandwich',
 'grilled_salmon',
 'guacamole',
 'gyoza',
 'hamburger',
 'hot_and_sour_soup',
 'hot_dog',
 'huevos_rancheros',
 'hummus',
 'ice_cream',
 'lasagna',
 'lobster_bisque',
 'lobster_roll_sandwich',
 'macaroni_and_cheese',
 'macarons',
 'miso_soup',
 'mussels',
 'nachos',
 'omelette',
 'onion_rings',
 'oysters',
 'pad_thai',
 'paella',
 'pancakes',
 'panna_cotta',
 'peking_duck',
 'pho',
 'pizza',
 'pork_chop',
 'poutine',
 'prime_rib',
 'pulled_pork_sandwich',
 'ramen',
 'ravioli',
 'red_velvet_cake',
 'risotto',
 'samosa',
 'sashimi',
 'scallops',
 'seaweed_salad',
 'shrimp_and_grits',
 'spaghetti_bolognese',
 'spaghetti_carbonara',
 'spring_rolls',
 'steak',
 'strawberry_shortcake',
 'sushi',
 'tacos',
 'takoyaki',
 'tiramisu',
 'tuna_tartare',
 'waffles'])

foods_classes = {}
for i, food in enumerate(foods):
  foods_classes[i] = food

@st.cache(allow_output_mutation=True)
def load_model(m):
    model = tf.keras.models.load_model(m, custom_objects={"KerasLayer": hub.KerasLayer})
    return model

@st.cache(allow_output_mutation=True)
def load_img(img_path):
  img = Image.open(img_path)
  return img

def convert(image_file):
  img_array = np.array(load_img(image_file))
  # img_array = img_array / 255.0 # using efficient net architecture means we do not need to normalize images!
  img = tf.image.resize(img_array, size=(224,224))
  img = tf.expand_dims(img, axis=0)
  return img

def top_5_predictions(predictions_probabilities):
  top_5_pred_indexes = predictions_probabilities.argsort()[-5:][::-1]
  top_5_pred_labels = foods[top_5_pred_indexes]
  top_5_pred_values = predictions_probabilities[top_5_pred_indexes]
  fig, ax = plt.subplots()
  set_color = ['gray' if (x < tf.reduce_max(top_5_pred_values)) else 'green' for x in top_5_pred_values]
  bar_plot = ax.bar(np.arange(len(top_5_pred_labels)), top_5_pred_values, color=set_color)
  for idx,rect in enumerate(bar_plot):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                str(f"{top_5_pred_values[idx]*100:.1f}%"),
                ha='center', va='bottom', rotation=0, color=set_color[idx])
  ax.set_xticks(np.arange(len(top_5_pred_labels)), labels=top_5_pred_labels, rotation="vertical")
  st.pyplot(fig)