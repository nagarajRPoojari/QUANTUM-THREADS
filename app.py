import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Streamlit Visualization Example")

st.sidebar.header("User Input Parameters")

num_points = st.sidebar.slider("Select the number of data points", 10, 100, 50)

data = np.random.randn(num_points, 2)

# Scatter plot
st.header("Scatter Plot")
st.write("Here is a simple scatter plot:")
st.pyplot(plt.scatter(data[:, 0], data[:, 1]))
st.write("You can customize this section based on your data and visualization needs.")

# Line chart
st.header("Line Chart")
st.write("And here is a basic line chart:")
st.line_chart(data)

# Bar chart
st.header("Bar Chart")
st.write("Let's also include a bar chart:")
st.bar_chart(data)


# Data table
st.header("Data Table")
st.write("You can also display the raw data as a table:")
st.write(data)

