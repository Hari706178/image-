import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Image Pooling Visualizer", layout="wide")
st.title("🖼️ Image Pooling Visualizer")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    rgb_array = np.array(img)

    st.sidebar.header("⚙️ Controls")
    size = st.sidebar.selectbox("Compressed Size", [16, 32])
    pool_size = st.sidebar.slider("Pool Size", 2, 4, 2)
    stride = st.sidebar.slider("Stride", 1, 4, 2)

    # Grayscale
    grayscale = np.mean(rgb_array, axis=2)

    # Resize
    resized = Image.fromarray(grayscale.astype(np.uint8)).resize((size, size))
    matrix = np.array(resized)

    def sliding_pooling(matrix, pool_size, stride, mode):
        h, w = matrix.shape
        out_h = (h - pool_size) // stride + 1
        out_w = (w - pool_size) // stride + 1
        pooled = np.zeros((out_h, out_w))

        for i in range(0, h - pool_size + 1, stride):
            for j in range(0, w - pool_size + 1, stride):
                window = matrix[i:i+pool_size, j:j+pool_size]
                if mode == "min":
                    pooled[i//stride, j//stride] = np.min(window)
                elif mode == "max":
                    pooled[i//stride, j//stride] = np.max(window)
                else:
                    pooled[i//stride, j//stride] = np.mean(window)
        return pooled

    min_pool = sliding_pooling(matrix, pool_size, stride, "min")
    max_pool = sliding_pooling(matrix, pool_size, stride, "max")
    avg_pool = sliding_pooling(matrix, pool_size, stride, "avg")

    st.subheader("Results")
    c1, c2, c3, c4 = st.columns(4)

    c1.image(matrix, caption="Original", clamp=True)
    c2.image(min_pool, caption="Min Pool", clamp=True)
    c3.image(max_pool, caption="Max Pool", clamp=True)
    c4.image(avg_pool, caption="Avg Pool", clamp=True)

    with st.expander("View Matrices"):
        st.write("Original:", matrix)
        st.write("Min Pool:", min_pool)
        st.write("Max Pool:", max_pool)
        st.write("Avg Pool:", avg_pool)

else:
    st.info("Upload an image to start")
