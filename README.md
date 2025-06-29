# Resume to JD Matcher

A Streamlit application that matches resumes to job descriptions using natural language processing and machine learning.

## Features

- Upload job descriptions and resumes in PDF format.
- Extract and parse text from PDFs.
- Calculate matching scores between resumes and job descriptions.
- Display and download top matching resumes.

## Installation

To run this application locally, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/gratefulvortex/Recruitment.git
    cd <your-repository-directory>
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv myenv
    ```

3. **Activate the virtual environment**:
    - On macOS and Linux:
        ```sh
        source myenv/bin/activate
        ```
    - On Windows:
        ```sh
        myenv\Scripts\activate
        ```

4. **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

5. **Download the spaCy model**:
    ```sh
    python -m spacy download en_core_web_sm
    ```

## Usage

1. **Run the Streamlit application**:
    ```sh
    streamlit run app.py
    ```

2. **Open your web browser** and go to `http://localhost:8501` to view the application.

## Deployment

To deploy this application on Streamlit Cloud, follow these steps:

1. **Push your code to a GitHub repository**:
    ```sh
    git add .
    git commit -m "Initial commit"
    git push origin main
    ```

2. **Go to [Streamlit Cloud](https://share.streamlit.io/)** and sign in with your GitHub account.

3. **Click on "New app"** and select the repository where your app is located.

4. **Choose the branch** (e.g., `main`) and the main file path (e.g., `app.py`).

5. **Click "Deploy"**.

Streamlit Cloud will automatically handle the installation of dependencies and the setup of your environment based on the `requirements.txt` file.

## License

This project is licensed under the MIT License.
