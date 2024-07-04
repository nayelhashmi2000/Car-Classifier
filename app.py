import streamlit as st
from ultralytics import YOLO
import os
import zipfile
import tempfile
from PIL import Image
import io

def classifier(input_folder, model, names, progress_bar, progress_text):
    total_files = len(os.listdir(input_folder))
    for count, image in enumerate(os.listdir(input_folder)):
        image_path = os.path.join(input_folder, image)
        results = model([image_path])  # return a list of Results objects
        # Process results list
        for result in results:
            probs = result.probs  # Probs object for classification outputs
            identified_class = probs.top1  # highest confidence class
            name_identified = names[identified_class]  # get class name
            name_of_class = name_identified.split("_")[0]
            # Rename file to the name of the identified class number
            os.rename(image_path, os.path.join(input_folder, f"{name_of_class}_{count}.jpg"))
        progress_bar.progress((count + 1) / total_files)
        progress_text.text(f"Processed {count + 1}/{total_files}")
    return total_files

def create_zip(input_folder):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, files in os.walk(input_folder):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, os.path.relpath(file_path, input_folder))
    zip_buffer.seek(0)
    return zip_buffer

def main():
    st.set_page_config(page_title="Image Classifier and Renamer", layout="wide")
    
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Image Classifier and Renamer</h1>", unsafe_allow_html=True)

    # Load model once and store it in session state
    if 'model' not in st.session_state:
        st.session_state.model = YOLO('best.pt')  # Load your pretrained YOLO model
        st.session_state.names = st.session_state.model.names  # Get class names

    # Initialize session state for uploaded files and file uploader key
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0

    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []

    with st.form(key='upload_form', clear_on_submit=True):
        files = st.file_uploader(
            "Choose images", 
            accept_multiple_files=True, 
            type=["jpg", "jpeg", "png"], 
            key=st.session_state["file_uploader_key"]
        )
        submit_button = st.form_submit_button(label='Process')

    if files:
        st.session_state["uploaded_files"] = files

    if submit_button and st.session_state["uploaded_files"]:
        with tempfile.TemporaryDirectory() as input_folder:
            for uploaded_file in st.session_state["uploaded_files"]:
                image = Image.open(uploaded_file)
                image.save(os.path.join(input_folder, uploaded_file.name))

            model = st.session_state.model
            names = st.session_state.names

            st.markdown("<h3 style='text-align: center; color: #4CAF50;'>Processing Images...</h3>", unsafe_allow_html=True)
            progress_text = st.empty()
            progress_bar = st.progress(0)
            total_files = classifier(input_folder, model, names, progress_bar, progress_text)

            zip_buffer = create_zip(input_folder)
            st.session_state.zip_buffer = zip_buffer  # Store zip buffer in session state
            st.session_state.processed = True  # Indicate that processing is complete

            # Display summary message
            # st.markdown(f"<h3 style='text-align: center; color: #4CAF50;'>Processed {total_files} files successfully!</h3>", unsafe_allow_html=True)

    # Display download button only after processing is complete
    if 'processed' in st.session_state and st.session_state.processed:
        st.markdown("<h3 style='text-align: center; color: #4CAF50;'>Download Processed Images</h3>", unsafe_allow_html=True)
        if st.download_button(
            label="Download ZIP",
            data=st.session_state.zip_buffer,
            file_name="classified_images.zip",
            mime="application/zip"
        ):
            # Remove uploaded images and reset session state
            st.session_state["uploaded_files"] = []
            st.session_state.pop('zip_buffer', None)
            st.session_state.pop('processed', None)
            st.session_state["file_uploader_key"] += 1
            st.experimental_rerun()  # Rerun the app to refresh the state

if __name__ == '__main__':
    main()