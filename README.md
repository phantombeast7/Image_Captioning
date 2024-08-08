# Image Captioning Project

You can access the application directly here: [Image Captioning Transformer](https://imagecaptiontransformer.streamlit.app/)

This project provides an accessible tool for generating image captions and converting them to audio, particularly useful for visually impaired individuals. It utilizes advanced deep learning models for captioning and integrates text-to-speech conversion for enhanced accessibility.

## Overview

This document details an image captioning project with functionalities including:

* **Image Captioning:** Generates descriptive captions for general images using pre-trained models from Hugging Face's Transformers library.
* **Text-to-Speech Conversion:** Converts generated captions into audio using Google Text-to-Speech (gTTS).
* **Streamlit Interface:** Creates an interactive web interface with Streamlit for uploading images, generating captions, and playing the audio output.
* **Custom Styling:** Applies custom CSS to enhance the user interface and user experience.

## Running the Code

**Note:** This section is only relevant if you want to set up the application locally. If you have already deployed the application, skip to the "Using the Application" section.

### Set Up Your Development Environment

* Open PyCharm or your preferred IDE.
* **Create a Virtual Environment:**
    * Open the terminal in PyCharm or your command line interface.
    * Run the following command to create a virtual environment:
        ```bash
        python -m venv venv
        ```
    * Activate the virtual environment:
        * **On Windows:**
            ```bash
            venv\Scripts\activate
            ```
        * **On macOS/Linux:**
            ```bash
            source venv/bin/activate
            ```
* **Install Required Libraries:**
    * Install the necessary Python libraries by running:
        ```bash
        pip install torch transformers Pillow gtts streamlit numpy pandas
        ```
* **Add Your Code:**
    * Save the provided code in a file named `main.py`.

### Run the Application

* To start the application, use the following command in your terminal:
    ```bash
    streamlit run main.py
    ```

## Access the Application

Once the application is running locally, open your web browser and navigate to:

* `http://localhost:8501`

## Using the Application

* Go to the "Image Captioning" section on the application.
* Upload an image.
* The application will generate a caption and audio for you.

## Conclusion

Your image captioning tool is ready for use! Enjoy generating captions and listening to the audio descriptions.
