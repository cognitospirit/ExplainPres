import customtkinter as ctk
from PIL import Image
import threading
import webbrowser
import os
import tempfile
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# --- Main Application Class ---
class PrescriptionTranslatorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Configuration ---
        self.title("Prescription Translator (Local Vision Model)")
        self.geometry("600x750")
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=2) # Give more weight to the result textbox

        # --- State Variables ---
        self.original_image = None
        self.language = ctk.StringVar(value="English")
        self.phi_model = None
        self.phi_processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- UI WIDGETS ---

        # 1. Header
        self.header_frame = ctk.CTkFrame(self, corner_radius=10)
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self.header_frame.grid_columnconfigure(0, weight=1)
        self.header_label = ctk.CTkLabel(self.header_frame, text="Prescription Translator", font=ctk.CTkFont(size=20, weight="bold"))
        self.header_label.grid(row=0, column=0, padx=10, pady=10)

        # 2. Image Display & Upload
        self.image_frame = ctk.CTkFrame(self, height=300)
        self.image_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.image_frame.grid_propagate(False)
        self.image_frame.grid_columnconfigure(0, weight=1)
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_label = ctk.CTkLabel(self.image_frame, text="Load a prescription image to begin.", text_color="gray")
        self.image_label.grid(row=0, column=0)

        # 3. Controls
        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        self.controls_frame.grid_columnconfigure([0, 1, 2], weight=1)

        self.load_button = ctk.CTkButton(self.controls_frame, text="Load Image", command=self.load_image_dialog)
        self.load_button.grid(row=0, column=0, padx=5, pady=10)

        self.language_segmented_button = ctk.CTkSegmentedButton(self.controls_frame, values=["English", "Hindi"], variable=self.language)
        self.language_segmented_button.grid(row=0, column=1, padx=5, pady=10)

        self.translate_button = ctk.CTkButton(self.controls_frame, text="Translate & Explain", command=self.start_translation, state="disabled")
        self.translate_button.grid(row=0, column=2, padx=5, pady=10)

        # 4. Results Textbox
        self.result_textbox = ctk.CTkTextbox(self, wrap="word", state="disabled", font=("Arial", 14))
        self.result_textbox.grid(row=3, column=0, sticky="nsew", padx=10, pady=10)

        # 5. Print Button
        self.print_button = ctk.CTkButton(self, text="Print Translation", command=self.print_translation, state="disabled")
        self.print_button.grid(row=4, column=0, padx=10, pady=10)
        
        # 6. Status Bar
        self.status_label = ctk.CTkLabel(self, text="Loading Vision Model... Please wait.", text_color="gray")
        self.status_label.grid(row=5, column=0, sticky="w", padx=10, pady=5)
        
        # --- Load Phi-3.5 Vision Model in Background ---
        threading.Thread(target=self.load_phi_model, daemon=True).start()

    def load_phi_model(self):
        """Loads the Phi-3.5 Vision model and processor in a background thread."""
        try:
            model_id = "microsoft/Phi-3.5-vision-instruct"
            self.phi_model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                device_map=self.device, 
                trust_remote_code=True, 
                torch_dtype="auto",
                _attn_implementation='eager'
            )
            # --- UPDATED: Use num_crops for better OCR ---
            self.phi_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=4)
            self.status_label.configure(text=f"Vision Model Ready ({self.device.upper()}).", text_color="green")
            self.translate_button.configure(state="normal")
        except Exception as e:
            self.status_label.configure(text=f"Error loading model: {e}", text_color="red")
            self.update_textbox(f"Failed to load the vision model. Please check your internet connection and try restarting the app.\n\nError: {e}")

    def load_image_dialog(self):
        """Opens a file dialog to select an image."""
        filepath = ctk.filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if filepath:
            self.original_image = Image.open(filepath).convert("RGB")
            display_image = ctk.CTkImage(light_image=self.original_image, size=(400, 280))
            self.image_label.configure(image=display_image, text="")
            self.print_button.configure(state="disabled")

    def start_translation(self):
        """Starts the AI processing in a new thread to keep the GUI responsive."""
        if self.original_image is None:
            self.update_textbox("Error: Please load an image first.")
            return
        
        self.translate_button.configure(state="disabled", text="Processing...")
        self.print_button.configure(state="disabled")
        threading.Thread(target=self.get_ai_explanation, daemon=True).start()

    def _generate_response(self, prompt_text):
        """Helper function to generate a response from the model using the chat template."""
        messages = [{"role": "user", "content": f"<|image_1|>\n{prompt_text}"}]
        prompt = self.phi_processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.phi_processor(prompt, images=[self.original_image], return_tensors="pt").to(self.device)
        
        generation_args = {"max_new_tokens": 1000, "temperature": 0.0, "do_sample": False}
        generate_ids = self.phi_model.generate(**inputs, eos_token_id=self.phi_processor.tokenizer.eos_token_id, **generation_args)
        
        # Remove the input tokens from the generated output
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.phi_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response

    def get_ai_explanation(self):
        """
        Two-step AI process using only the local Phi-3.5 Vision model with improved prompting.
        """
        try:
            # --- Step 1: OCR with Phi-3.5 Vision ---
            self.update_textbox("Step 1/2: Reading text from the prescription image...")
            ocr_prompt_text = "Extract all text from this prescription image. List each medicine, its dosage, and frequency exactly as written. If the image does not contain a prescription or is unreadable, respond with only the word 'Error'."
            extracted_text = self._generate_response(ocr_prompt_text)

            if not extracted_text or "error" in extracted_text.lower():
                raise ValueError("No prescription found, or the image is unreadable.")

            # --- Step 2: Explanation and Translation with Phi-3.5 Vision ---
            self.update_textbox(f"Step 2/2: Explaining and translating to {self.language.get()}...")
            
            explanation_prompt_text = f"""
            Based on the following text which was extracted from the image:
            ---
            {extracted_text}
            ---
            Your task is to act as a helpful medical assistant.
            1. Identify each medication from the text.
            2. For each one, provide a simple, one-sentence explanation in layman's terms of its general use.
            3. Translate the medication name, its dosage, and the simple explanation into {self.language.get()}.
            4. Format the final output clearly with headings for each medicine. Do not include any warnings or disclaimers.
            """
            final_explanation = self._generate_response(explanation_prompt_text)
            
            self.update_textbox(final_explanation)
            self.print_button.configure(state="normal")

        except Exception as e:
            self.update_textbox(f"An error occurred during processing:\n\n{e}")
        finally:
            self.translate_button.configure(state="normal", text="Translate & Explain")

    def update_textbox(self, text):
        """Helper function to safely update the GUI from a background thread."""
        self.result_textbox.configure(state="normal")
        self.result_textbox.delete("1.0", "end")
        self.result_textbox.insert("1.0", text)
        self.result_textbox.configure(state="disabled")

    def print_translation(self):
        """Saves the result to a temporary HTML file and opens it in a web browser for printing."""
        content = self.result_textbox.get("1.0", "end").strip()
        if not content:
            return

        # Basic conversion from markdown-like text to HTML
        html_content = content.replace("\n", "<br>")
        html_content = f"<html><head><title>Prescription Translation</title></head><body><pre style='font-family: Arial, sans-serif; font-size: 14px; white-space: pre-wrap;'>{html_content}</pre></body></html>"
        
        try:
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html", encoding='utf-8') as f:
                f.write(html_content)
                webbrowser.open('file://' + os.path.realpath(f.name))
        except Exception as e:
            self.update_textbox(f"Could not open print dialog: {e}")


if __name__ == "__main__":
    app = PrescriptionTranslatorApp()
    app.mainloop()
