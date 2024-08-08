import os
import torch
import base64
import requests
import numpy as np
import pandas as pd
from PIL import Image
from gtts import gTTS
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForSeq2SeqLM


def speak_text(text):
    tts = gTTS(text)
    audio_path = "caption.mp3"
    tts.save(audio_path)
    return audio_path

@st.cache_resource
def load_model_and_processor():
    processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")
    return processor, model


@st.cache_resource
def load_xray_model_and_processor():
    xray_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    xray_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return xray_processor, xray_model


# Define the model metrics
model_metrics = [
    {"metric": "Accuracy", "value": "85%"},
    {"metric": "Precision", "value": "82%"},
    {"metric": "Recall", "value": "88%"},
    {"metric": "F1 Score", "value": "85%"},
]

# Check if GPU is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to predict caption
def predict_caption(image, processor, model):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption

# Function to predict caption for X-ray images
def predict_xray_caption(image, processor, model):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption


# Function to convert image to base64
def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

# def display_speaker_icon(audio_path):
#     st.markdown(f"""
#     <div style="position: relative; display: inline-block;">
#         <button onclick="document.getElementById('audio').play()">
#             <img src="https://example.com/speaker_icon.png" style="width:50px; height:50px; cursor: pointer;"/>
#         </button>
#         <audio id="audio" src="{audio_path}"></audio>
#     </div>
#     """, unsafe_allow_html=True)

# Base64 encoded images (Replace these with your actual image paths)
use_case_images_base64 = {
    "E-commerce": convert_image_to_base64("images/ecom.jpeg"),
    "Digital Marketing": convert_image_to_base64("images/Digital Marketing.jpeg"),
    "Media and Entertainment": convert_image_to_base64("images/Media and Entertainment.jpeg"),
    "Healthcare": convert_image_to_base64("images/health.jpeg"),
    "Accessibility Solutions": convert_image_to_base64("images/accessibility.jpeg"),
    "Visually Impaired": convert_image_to_base64("images/visually impaired.jpeg")
}
xrayimage = convert_image_to_base64("static/xray.jpg")
# Convert images to base64
video_icon_base64 = convert_image_to_base64("images/video icon.png")
multilingual_icon_base64 = convert_image_to_base64("images/translate.png")
creative_icon_base64 = convert_image_to_base64("images/creative icon.png")

# Set the page configuration
st.set_page_config(
    page_title="Image Captioning for Visually Impaired",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
body, html {
    height: 100%;
    margin: 0;
    overflow: hidden; /* Prevent scrolling */
}

.main-container {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100vh; /* Full viewport height */
    overflow: hidden; /* Prevent scrolling */
}

.section {
    flex: 1;
    padding: 10px;
    overflow: hidden; /* Prevent scrolling */
}

.header, .footer {
    flex: 0 0 auto;
    padding: 10px;
    background-color: #04142b;
    color: white;
    text-align: center;
}

.content {
    flex: 1;
    overflow: hidden; /* Prevent scrolling */
}





.conclusion-section {
    height: 100vh; /* Full viewport height */
    overflow: hidden; /* Prevent scrolling */
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

.conclusion-section h1 {
    margin-bottom: 20px;
}

.conclusion-section p {
    margin-bottom: 20px;
}

.conclusion-section div {
    display: flex;
    align-items: center;
    margin-bottom: 0px;
}

.conclusion-section img {
    width: 50px;
    height: 50px;
    margin-right: 20px;
}

.stButton button {
    background-color: #00BFFF !important; /* Deep Sky Blue */
    color: white;
    border: none;
    padding: 12px 28px;
    text-align: center;
    font-size: 16px;
    border-radius: 8px;
    margin: 6px 3px;
    cursor: pointer;
    transition: background-color 0.3s, transform 0.3s;
}

.stButton button:hover {
    background-color: #1E90FF; /* Dodger Blue */
    transform: scale(1.05);
}

.stFileUploader label {
    font-size: 16px;
    font-weight: bold;
    color: #00BFFF; /* Deep Sky Blue */
}

.header {
    display: flex;
    justify-content: center;
    align-items: center;
}

.header img {
    width: 150px;
    height: auto;
}

.title {
    font-size: 36px;
    font-weight: bold;
    text-align: center;
    color: #00BFFF; /* Deep Sky Blue */
}

.section {
    background-color: #87CEFA; /* Light Sky Blue */
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0px 6px 12px rgba(0,0,0,0.1);
}

.section h1, .section h2, .section p, .section li {
    color: #000000; /* White text color */
}




.mentor p {
    color: #FFFFFF;
    font-size: 18px;
    text-align: center;
}

.section-content {
    background-color: #E0FFFF; /* Light Cyan */
    padding: 15px;
    border-radius: 8px;
    margin-top: 10px;
    box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
}

.section-content h2 {
    font-size: 24px;
    color: #00BFFF; /* Deep Sky Blue */
    margin-bottom: 10px;
}

.st-expander {
    background-color: #00BFFF; /* Deep Sky Blue */
    padding: 12px;
    border-radius: 15px;
    margin-bottom: 12px;
}

.expander-container {
    background-color: #FFFFFF; /* White background for boxes */
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 6px 12px rgba(0,0,0,0.1);
    color: #333;
    margin-bottom: 12px; /* Space between expanders */
}

.expander-header {
    font-size: 24px;
    font-weight: bold;
    color: #00BFFF; /* Deep Sky Blue */
    margin-bottom: 12px;
    text-align: center;
}

.expander-header img {
    height: 200px; /* Adjusted height for images */
    border-radius: 10px;
    object-fit: cover;
    display: block;
    margin: 0 auto 12px auto; /* Centered with margin */
}

.expander-content {
    font-size: 16px;
}

.code-explanation-container {
    display: flex;
    flex-direction: column;
    gap: 24px;
}

.code-step {
    background-color: #E0FFFF; /* Light Cyan */
    padding: 10px;
    padding-top:0px;
    padding-bottom:0px;
    border-radius: 12px;
    box-shadow: 0px 6px 12px rgba(0,0,0,0.1);
}

.code-step h2 {
    color: #00BFFF; /* Deep Sky Blue */
    margin-bottom: 4px;
}

/* Image Captioning Section */
.caption-section {
    background-color: #F0FFFF; /* Azure */
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
    box-shadow: 0px 6px 12px rgba(0,0,0,0.2);
}

.caption-section h1 {
    color: #00CED1; /* Dark Turquoise */
    font-size: 32px;
    text-align: center;
    margin-bottom: 15px;
}

.caption-section .upload-section {
    background-color: #FFFFFF; /* White background */
    padding: 20px;
    border-radius: 12px;
    border: 2px solid #00CED1; /* Dark Turquoise border */
}

.caption-section .upload-section img {
    border-radius: 8px;
    box-shadow: 0px 6px 12px rgba(0,0,0,0.3);
    display: block;
    margin: 0 auto;
}

.caption-section .stFileUploader label {
    color: #00CED1; /* Dark Turquoise */
}

.caption-section .stButton {
    display: flex;
    justify-content: center;
    margin-top: 20px;
}

.caption-section .stButton button {
    background-color: #00CED1; /* Dark Turquoise */
    color: white;
    border: none;
    padding: 14px 28px;
    font-size: 16px;
    border-radius: 12px;
    transition: background-color 0.3s, transform 0.3s;
}

.caption-section .stButton button:hover {
    background-color: #20B2AA; /* Light Sea Green */
    transform: scale(1.05);
}

.caption-section .generated-caption {
    text-align: left;
    align:center;
    color: #333;
    margin-top: 20px;
    overflow: hidden; /* Hide overflow text */
    text-overflow: ellipsis; /* Add ellipsis for overflow text */
}

.generated-caption {
    text-align: left;
    # font-size: 20px;
    margin-top: 20px;
    overflow: hidden; /* Hide overflow text */
    text-overflow: clip; /* Clip the text without ellipsis */
}

.streamlit-expanderHeader {
    color: white !important;
}

.generated-caption {
    color: #FFFFFF;
}

.blue-button {
    background-color: #20B2AA;
    color: white;
    border: none;
    padding: 12px 28px;
    text-align: center;
    font-size: 16px;
    border-radius: 8px;
    margin: 6px 3px;
    cursor: pointer;
    transition: background-color 0.3s, transform 0.3s;
}

.blue-button:hover {
    background-color: #00BFFF; /* Deep Sky Blue */
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to",
                           ["Home", "About Project", "Business Use Cases", "Image Captioning","Conclusion"])

if section == "Home":
    st.markdown(f"""
    <div class="full-screen-container home-container">
        <div class="header">
            <div class="title">Image Captioning Using Transformers</div>
        </div>
    </div>
    """, unsafe_allow_html=True)





elif section == "About Project":
    st.markdown("""
    <div class="section">
        <h1>About the Project</h1>
        <p>This project aims to create an image captioning application that generates descriptive captions for images using advanced deep learning models. The application is designed to be particularly beneficial for visually impaired individuals, providing them with an audio description of visual content.</p>
        
        <p>In this project, we've implemented the following features:</p>
        <ul>
            <li><strong>Model Integration:</strong> We integrated pre-trained models from Hugging Face's Transformers library, including models specialized for general image captioning and X-ray image captioning.</li>
            <li><strong>GPU Utilization:</strong> The application checks for available GPU resources and uses them for faster processing, enhancing the performance of the caption generation.</li>
            <li><strong>Text-to-Speech Conversion:</strong> After generating the captions, the application uses Google Text-to-Speech (gTTS) to convert the text into audio, making the captions accessible in audio format.</li>
            <li><strong>Streamlit Interface:</strong> The application is built with Streamlit, providing a user-friendly interface for uploading images, generating captions, and playing the audio output.</li>
            <li><strong>Custom Styling:</strong> The app features custom CSS styling to enhance the user experience, ensuring that the interface is visually appealing and easy to navigate.</li>
            <li><strong>Business Use Cases:</strong> We also showcase different business use cases where image captioning can be beneficial, such as in healthcare, digital marketing, and accessibility solutions.</li>
        </ul>
        <p>Overall, this project demonstrates the potential of AI in creating inclusive technologies that can significantly improve the lives of visually impaired individuals.</p>
    </div>
    """, unsafe_allow_html=True)

elif section == "Business Use Cases":
    st.markdown("""
    <div class="section">
        <h1>Business Use Cases</h1>
    </div>
    """, unsafe_allow_html=True)

    use_cases = {
        "Visually Impaired": """
            <ul>
                <li><strong>Descriptive Captions:</strong> Generate detailed and accurate captions for images to help visually impaired individuals understand visual content.</li>
                <li><strong>Accessibility:</strong> Enhance accessibility by providing text descriptions for images, making digital content more inclusive.</li>
                <li><strong>Assistive Technology:</strong> Integrate with assistive technologies like screen readers to provide real-time image descriptions.</li>
            </ul>
        """,
        "Healthcare": """
            <ul>
                <li><strong>X-ray Image Captioning:</strong> Provide detailed descriptions of X-ray images to support radiologists and other healthcare professionals in their analyses.</li>
                <li><strong>Telemedicine:</strong> Enhance telemedicine platforms by providing detailed descriptions of visual data shared between patients and doctors.</li>
                <li><strong>Patient Education:</strong> Help patients understand medical images and reports through clear and concise captions.</li>
            </ul>
        """,
        "Digital Marketing": """
            <ul>
                <li><strong>Engaging Content:</strong> Generate engaging content with accurate and descriptive captions for images used in marketing campaigns.</li>
                <li><strong>SEO Improvement:</strong> Well-described images improve search engine optimization (SEO), leading to better search engine rankings.</li>
                <li><strong>Social Media Marketing:</strong> Create compelling captions for images posted on social media to attract more engagement and shares.</li>
            </ul>
        """,
        "Media and Entertainment": """
            <ul>
                <li><strong>Content Creation:</strong> Assist content creators by automatically generating descriptions and tags for images and videos.</li>
                <li><strong>Video Summarization:</strong> Use image captioning to summarize video content by creating descriptions for key frames.</li>
                <li><strong>Enhanced User Experience:</strong> Provide better user experiences by offering detailed descriptions of visual content.</li>
            </ul>
        """,
        "Accessibility Solutions": """
            <ul>
                <li><strong>Assistive Technology:</strong> Use image captioning to create assistive technologies for visually impaired individuals, enhancing their ability to understand visual content.</li>
                <li><strong>Inclusive Content:</strong> Ensure that digital content is inclusive and accessible to everyone by providing descriptive captions for images and videos.</li>
                <li><strong>Compliance:</strong> Meet legal and regulatory requirements for accessibility by providing descriptive text for all visual content.</li>
            </ul>
        """
    }

    # Display use cases in expanders
    for use_case, details in use_cases.items():
        with st.expander(f":blue[{use_case}]"):
            st.markdown(f"""
            <div class="expander-container">
                <div class="expander-header">
                    {use_case}
                    <img src="data:image/jpeg;base64,{use_case_images_base64[use_case]}" alt="{use_case}">
                </div>
                <div class="expander-content">
                    {details}
                
            </div>
            """, unsafe_allow_html=True)

# Image Captioning Section
if section == "Image Captioning":
    st.markdown("""
    <div class="section">
        <h1>Image Captioning</h1>
        <p>Upload an image to get a caption generated by the model.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-content">
        <h2>Use Case 1: Image Captioning for Visually Impaired</h2>
        <p>Upload an image for image captioning for visually impaired to generate a caption.</p>
    </div>
    """, unsafe_allow_html=True)

    # File uploader for Use Case 1
    uploaded_file1 = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="uploader1")
    if uploaded_file1 is not None:
        image1 = Image.open(uploaded_file1)
        st.markdown(f"""
         <div class="expander-header">
             <img src="data:image/jpeg;base64,{base64.b64encode(uploaded_file1.getvalue()).decode()}" alt="Uploaded Image">
         </div>
         """, unsafe_allow_html=True)

        # Display the button only if an image is uploaded
        if st.button("Generate Caption", key="generate_caption1"):
            with st.spinner('Generating caption...'):
                processor, model = load_model_and_processor()
                caption1 = predict_caption(image1, processor, model)
                st.markdown(f"<div class='generated-caption'><strong>Generated Caption:</strong> {caption1}</div>",
                            unsafe_allow_html=True)

                # Generate the caption audio
                audio_path = speak_text(caption1)


                # Play the caption immediately
                st.audio(audio_path, format="audio/mp3")
    else:
        # Optional: Display a message when no image is uploaded
        st.markdown("<p style='color: gray;'>Please upload an image to generate a caption.</p>", unsafe_allow_html=True)


# Conclusion section with icons and descriptions
if section == "Conclusion":
    st.markdown("""
    <div class="section">
        <h1>Conclusion and Future Scope</h1>
        <p>Image captioning has made significant progress in identifying the context and describing the input image, but there are ongoing efforts to
improve its accuracy, fluency, and ability to capture nuanced details and context.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="section">
        <div style="display: flex; align-items: center; margin-bottom: 0px;">
            <div style="flex: 0 0 50px;">
                <img src="data:image/png;base64,{video_icon_base64}" alt="Video Icon" style="width: 50px; height: 50px;">
            </div>
            <div style="flex: 1; padding-left: 20px;">
                <h2>Video Captioning</h2>
                <p>Extending image captioning to generate descriptions for video sequences.</p>
            </div>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 0px;">
            <div style="flex: 0 0 50px;">
                <img src="data:image/png;base64,{multilingual_icon_base64}" alt="Multilingual Icon" style="width: 50px; height: 50px;">
            </div>
            <div style="flex: 1; padding-left: 20px;">
                <h2>Multilingual Captioning</h2>
                <p>Developing models that can generate captions in multiple languages.</p>
            </div>
        </div>
        <div style="display: flex; align-items: center;">
            <div style="flex: 0 0 50px;">
                <img src="data:image/png;base64,{creative_icon_base64}" alt="Creative Icon" style="width: 50px; height: 50px;">
            </div>
            <div style="flex: 1; padding-left: 20px;">
                <h2>Creative Captioning</h2>
                <p>Generating more descriptive and engaging captions that go beyond simple descriptions.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
