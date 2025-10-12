### **README.md Content**
# Brain Tumor Detection using AI 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://btd-aiml-hbsclhvfay6ua5b26fnutd.streamlit.app/)

An end-to-end deep learning application that classifies brain tumors from MRI scans into four categories: Glioma, Meningioma, Pituitary, and No Tumor. The project uses a Convolutional Neural Network (CNN) and is deployed as an interactive web application using Streamlit.

**You can access the live application here:**
**[https://btd-aiml-hbsclhvfay6ua5b26fnutd.streamlit.app/](https://btd-aiml-hbsclhvfay6ua5b26fnutd.streamlit.app/)**

---

##  Features

-   **User-Friendly Interface:** A simple and intuitive web UI built with Streamlit.
-   **Image Upload:** Allows users to upload MRI scans in JPG, JPEG, or PNG format.
-   **Real-Time Prediction:** Utilizes a trained TensorFlow/Keras model to provide instant classification.
-   **Result Display:** Shows the prediction result (Tumor Type or No Tumor) along with the model's confidence score.
-   **Online Accessibility:** Deployed on Streamlit Community Cloud, making it accessible to anyone with a web browser.

---

##  Tech Stack

-   **Backend & Model:** Python, TensorFlow, Keras
-   **Frontend:** Streamlit
-   **Image Processing:** OpenCV, Pillow
-   **Data Handling:** NumPy, Scikit-learn
-   **Deployment:** Streamlit Community Cloud, GitHub, Git LFS

---

##  Project Structure

```

├── app.py                  \# The main Streamlit application file
├── brain\_tumor\_model.h5    \# The pre-trained Keras CNN model file
├── requirements.txt        \# Python dependencies for the app
├── packages.txt            \# System-level dependencies for deployment
├── train\_model.py          \# Script used to train the CNN model
└── .gitignore              \# Files to be ignored by Git

````

---

##  How to Run Locally

To run this project on your local machine, follow these steps:

1.  **Clone the Repository:**
    ```bash
    # Neeche diye gaye URL ko apni repository ke URL se replace karein
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install Git LFS and Pull the Model File:**
    * Install Git LFS from [here](https://git-lfs.github.com/).
    * Pull the large model file:
        ```bash
        git lfs pull
        ```

3.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    ```
    * **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **MacOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your web browser.

---

##  Model Training

The CNN model was trained on the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle, which contains over 7,000 images across four classes. The model architecture consists of three convolutional blocks followed by dense layers, compiled with the Adam optimizer and trained for 10 epochs. The entire training process is detailed in the `train_model.py` script.

---

##  Disclaimer

This project is developed for educational and demonstration purposes only. The predictions made by the AI model are not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any medical concerns.
