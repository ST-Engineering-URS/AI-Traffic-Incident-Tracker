# AI-Traffic-Incident-Tracker

# Traffic Image Analysis & Accident Detection Toolkit

This repository contains a set of Python scripts designed for processing and analyzing traffic-related video and images. The toolkit includes functionalities for extracting frames from videos, displaying image collections in a graphical interface, and performing automated accident detection in images using a Vision-Language Model (Janus-Pro-7B) with Retrieval Augmented Generation (RAG).

**Note:** The provided code snippets represent distinct functionalities. They can be used individually or potentially combined into a larger workflow.

## Features

* **Video Frame Extraction:** Extract images from a video file at a configurable frame interval.
* **Image Display GUI:** A simple Tkinter-based application to display sequences of images from multiple folders simultaneously in a grid layout, with a fullscreen viewing option on click.
* **AI Accident Detection (RAG-based):**
    * Utilizes the Janus-Pro-7B Vision-Language Model from DeepSeek-AI.
    * Builds a local knowledge base using ChromaDB for text-based traffic scene descriptions (accidents vs. non-accidents).
    * Generates descriptions for input images.
    * Performs RAG by retrieving relevant context from the knowledge base based on the image description.
    * Uses the VL model to classify images as containing an accident ("yes") or not ("no") based on the image and the retrieved context.
    * Processes images in specified directories and outputs a list of detected accident image filenames.

## Prerequisites

* Python 3.8+
* A CUDA-enabled GPU is **highly recommended** for the AI Accident Detection script due to the use of large language models.
* Hugging Face Account and Access Token (needed for downloading model weights). You should set this as an environment variable `HF_TOKEN`.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install required Python packages:**
    Create a `requirements.txt` file with the following content:
    ```
    opencv-python
    Pillow
    sentence-transformers
    chromadb
    transformers
    huggingface_hub
    torch # Ensure you install the correct version for your CUDA setup
    # Note: The 'janus' package used in the third script
    # appears to be a custom implementation or from a specific source.
    # You may need to include the 'janus' source code or follow
    # specific installation instructions not provided in the snippets.
    # For DeepSeek-AI Janus, it might involve specific transformer
    # versions or source code from their repositories.
    ```
    Then install:
    ```bash
    pip install -r requirements.txt
    ```
    * **Regarding the `janus` package:** The provided code imports from `janus.models` and `janus.utils.io`. The `transformers` library is used to load `AutoModelForCausalLM` from `deepseek-ai/Janus-Pro-7B`. It's possible `janus` is intended to be a local module included in the project, or it might be a dependency requiring installation from a specific source. Please ensure you have the necessary `janus` code or installation method configured.
    * **PyTorch (`torch`):** Make sure you install the version compatible with your CUDA setup for GPU acceleration. Refer to the official PyTorch installation guide.

4.  **Set your Hugging Face Token:**
    It's best practice to set your Hugging Face token as an environment variable:
    * On Linux/macOS:
        ```bash
        export HF_TOKEN='YOUR_HF_TOKEN_HERE'
        ```
        (Add this line to your `~/.bashrc`, `~/.zshrc`, etc., to make it permanent)
    * On Windows:
        ```cmd
        set HF_TOKEN=YOUR_HF_TOKEN_HERE
        ```
        (For permanent setting, use System Properties -> Environment Variables)

## Usage

The repository contains three main scripts (assuming these filenames based on the code):

* `video_to_frames.py`
* `image_display_gui.py`
* `accident_detector_rag.py`

You will need to **edit the Python files directly** to configure paths and parameters as they are hardcoded in the provided snippets. Look for variables like `video_path`, `output_dir`, `folders`, `base_dir`, `chroma_db_path`, `output_file_path`, `frame_interval`, `fps`, `thumb_size`, etc.

### 1. Video Frame Extraction (`video_to_frames.py`)

This script extracts frames from a specified video.

1.  Edit `video_to_frames.py` and set the `video_path`, `output_dir`, and `frame_interval`.
2.  Run the script:
    ```bash
    python video_to_frames.py
    ```
    The extracted images will be saved in the `output_dir`.

### 2. Image Display GUI (`image_display_gui.py`)

This script launches a GUI to display image sequences from multiple folders.

1.  Edit `image_display_gui.py` and set the `folders` list to the directories you want to display. Configure `rows`, `cols`, `fps`, and `thumb_size` as needed.
2.  Run the script:
    ```bash
    python image_display_gui.py
    ```
    A window will open displaying images. Click an image to view it fullscreen (fullscreen view currently has hardcoded status text). Press `Esc` to exit fullscreen.

### 3. AI Accident Detection (`accident_detector_rag.py`)

This script performs RAG-based accident detection on images. This script requires GPU acceleration for the Vision-Language Model.

1.  Edit `accident_detector_rag.py` and set the `base_dir`, `input_folders` (list of directories containing images to analyze), `chroma_db_path`, and `output_file_path`.
2.  **Knowledge Base Setup:** The script needs to build or load the ChromaDB knowledge base. The first time you run it, or if you need to rebuild the knowledge base (e.g., after changing the description lists), set the `build_kb_flag = True` in the script. For subsequent runs, you can set `build_kb_flag = False` to load the existing database.
3.  Run the script:
    ```bash
    python accident_detector_rag.py
    ```
    The script will process images in the specified `input_folders`, classify them, and write the filenames of images detected as accidents to the `output_file_path`.

## Project Structure

Assuming you save the snippets as separate files:
```
