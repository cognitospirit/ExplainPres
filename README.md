<img width="892" height="1166" alt="image" src="https://github.com/user-attachments/assets/f879bff7-bcfa-4353-ba55-26b55de69419" /># ExplainPres
Prescription Translation App (Local) Prototype
# Prescription Translator AI

This is a desktop application built to demystify medical prescriptions. It uses a powerful local AI vision model to read a picture of a prescription and explain what the medicines are for in simple, everyday language, either in English or Hindi.



---

## Introduction (The Need)

Have you ever looked at a doctor's prescription and felt a bit lost? The handwriting can be hard to read, and the names of the medicines are often complex. This can be a real barrier, especially for elderly individuals or anyone who isn't familiar with medical terms.

I built this tool to bridge that gap. The goal is to empower patients by giving them a clear, simple understanding of their own medication, right on their own computer, without needing to upload sensitive information to the internet. Gemini Pro 2.5 was used for assistance in coding and writing.

---

## Methods (How It Works)

This application is built entirely in **Python** and uses a modern GUI created with the **CustomTkinter** library. The "brain" of the app is Microsoft's `Phi-3.5-vision-instruct`, a multimodal AI model that can understand both images and text.

The process happens in two main steps, all locally on your machine:

1.  **Optical Character Recognition (OCR):** First, you load an image of the prescription. The Phi-3.5 Vision model analyzes the image and extracts all the written text it can find—the names of the medicines, dosages, and frequencies.

2.  **Explanation & Translation:** The text extracted in the first step is then fed back into the *same* Phi-3.5 model with a new set of instructions. This time, it's asked to act as a helpful medical assistant, explain each medicine in layman's terms, and translate the entire explanation into your chosen language (English or Hindi).

Because everything runs locally, your medical information stays completely private.

---

## How to Run

To get this application running on your Windows machine, follow these steps.

1.  **Set up a Python Environment:** It's highly recommended to use a virtual environment to avoid conflicts with other projects. If you use Conda, you can set one up like this:
    ```bash
    conda create -n vision_app python=3.11
    conda activate vision_app
    ```

2.  **Install Dependencies:** You'll need to install all the required libraries. The most important part is getting the correct versions for PyTorch (depending on your hardware) and `transformers`.
    * **Install PyTorch:** Go to the [PyTorch website](https://pytorch.org/get-started/locally/) and get the command for your specific system (CPU or GPU/CUDA).
    * **Install other packages:**
        ```bash
        pip install customtkinter Pillow transformers==4.41.2
        ```

3.  **Run the Script:** Once everything is installed, simply run the main Python file from your terminal:
    ```bash
    python ExplainPres.py
    ```
    The first time you run it, the Phi-3.5 model (which is several gigabytes) will be downloaded and cached on your computer. Please be patient, as this can take some time depending on your internet connection. Subsequent launches will be much faster.

---

## Dependencies

Here is the complete list of libraries you need to install:

* `torch` & `torchvision` (from the official PyTorch website)
* `transformers==4.41.2` (This specific version is crucial for compatibility)
* `customtkinter`
* `Pillow`

---

## Discussion

This project is a practical demonstration of how powerful local, multimodal AI models have become. We can now build tools that understand the world through images and provide helpful information, all without relying on cloud APIs. This has huge implications for privacy and accessibility.

While this tool is designed to be helpful, it is important to remember that it is **not a substitute for professional medical advice**. The explanations are for informational purposes only. Always consult with your doctor or pharmacist if you have any questions about your medication.

The quality of the output depends heavily on the clarity of the prescription image. A clear, well-lit photo will always yield better results.
