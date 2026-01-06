import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------
# App Configuration
# ------------------------------------
st.set_page_config(page_title="Image Pooling Visualizer", layout="wide")
st.title("🖼️ Image Pooling Visualizer (Min / Max / Avg)")
st.markdown("Upload an image and explore **sliding pooling operations** interactively.")

# ------------------------------------
# Image Upload
# ------------------------------------
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # ------------------------------------
    # Read & Convert Image
    # ------------------------------------
    img = Image.open(uploaded_file).convert("RGB")
    rgb_array = np.array(img)

    # ------------------------------------
    # Sidebar Controls (Impressive Part)
    # ------------------------------------
    st.sidebar.header("⚙️ Controls")

    resize_option = st.sidebar.selectbox(
        "Compressed Image Size",
        ["16 x 16", "32 x 32"]
    )

    pool_size = st.sidebar.slider("Pooling Window Size", 2, 4, 2)
    stride = st.sidebar.slider("Stride", 1, 4, 2)

    pooling_mode = st.sidebar.selectbox(
        "Pooling Type",
        ["All", "Min", "Max", "Average"]
    )

    # ------------------------------------
    # Grayscale Conversion
    # ------------------------------------
    grayscale = np.mean(rgb_array, axis=2)

    # ------------------------------------
    # Resize
    # ------------------------------------
    size = 16 if resize_option == "16 x 16" else 32
    img_resized = Image.fromarray(grayscale.astype(np.uint8)).resize((size, size))
    matrix = np.array(img_resized)

    # ------------------------------------
    # Pooling Function
    # ------------------------------------
    def sliding_pooling(matrix, pool_size=2, stride=2, mode="avg"):
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
                elif mode == "avg":
                    pooled[i//stride, j//stride] = np.mean(window)

        return pooled

    # ------------------------------------
    # Apply Pooling
    # ------------------------------------
    min_pool = sliding_pooling(matrix, pool_size, stride, "min")
    max_pool = sliding_pooling(matrix, pool_size, stride, "max")
    avg_pool = sliding_pooling(matrix, pool_size, stride, "avg")

    # ------------------------------------
    # Display Images
    # ------------------------------------
    st.subheader("🔍 Visual Results")

    cols = st.columns(4)
    cols[0].image(matrix, caption=f"Original {size}x{size}", clamp=True)

    if pooling_mode in ["Min", "All"]:
        cols[1].image(min_pool, caption="Min Pooling", clamp=True)

    if pooling_mode in ["Max", "All"]:
        cols[2].image(max_pool, caption="Max Pooling", clamp=True)

    if pooling_mode in ["Average", "All"]:
        cols[3].image(avg_pool, caption="Average Pooling", clamp=True)

    # ------------------------------------
    # Histogram Comparison (Extra Feature)
    # ------------------------------------
    st.subheader("📊 Intensity Distribution (Histogram)")

    fig, ax = plt.subplots()
    ax.hist(matrix.flatten(), bins=30, alpha=0.5, label="Original")

    if pooling_mode in ["Min", "All"]:
        ax.hist(min_pool.flatten(), bins=30, alpha=0.5, label="Min Pool")

    if pooling_mode in ["Max", "All"]:
        ax.hist(max_pool.flatten(), bins=30, alpha=0.5, label="Max Pool")

    if pooling_mode in ["Average", "All"]:
        ax.hist(avg_pool.flatten(), bins=30, alpha=0.5, label="Avg Pool")

    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    ax.legend()

    st.pyplot(fig)

    # ------------------------------------
    # Matrix View (Exam Friendly)
    # ------------------------------------
    with st.expander("📋 View Matrices (Numerical Values)"):
        st.write("Original Matrix:", matrix)
        st.write("Min Pool Matrix:", min_pool)
        st.write("Max Pool Matrix:", max_pool)
        st.write("Average Pool Matrix:", avg_pool)

else:
    st.info("👆 Please upload an image to begin.")
