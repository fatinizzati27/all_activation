import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Activation Functions Comparison")

st.title("Comparison of Activation Functions")
st.write("This app visualises ReLU, Sigmoid and Tanh activation functions in a single frame.")

x = np.linspace(-10, 10, 400)

relu = np.maximum(0, x)
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)

fig, ax = plt.subplots()
ax.plot(x, relu, label="ReLU")
ax.plot(x, sigmoid, label="Sigmoid")
ax.plot(x, tanh, label="Tanh")

ax.set_xlabel("Weighted Sum (z)")
ax.set_ylabel("Activation Output")
ax.set_title("Activation Functions Comparison")
ax.legend()

st.pyplot(fig)
