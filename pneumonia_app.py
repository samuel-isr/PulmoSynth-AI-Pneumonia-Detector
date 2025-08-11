import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
import os
import threading
class PneumoniaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PulmoSynth AI - Standalone Analyzer")
        self.root.geometry("850x650")
        self.root.configure(bg="#0F1014")

        self.model1 = None
        self.model2 = None
        
        self.create_widgets()
        threading.Thread(target=self.load_models, daemon=True).start()

    def load_models(self):
        """Loads the two trained .h5 models."""
        try:
            self.status_label.config(text="Status: Loading AI models")
            model1_path = 'pneumonia_model_EfficientNetB0.h5'
            model2_path = 'pneumonia_model_DenseNet121.h5'
            
            if not os.path.exists(model1_path) or not os.path.exists(model2_path):
                messagebox.showerror("Fatal Error", "Model files not found!\nPlease ensure 'pneumonia_model_EfficientNetB0.h5' and 'pneumonia_model_DenseNet121.h5' are in the same folder as this application.")
                self.root.quit()
                return
            self.model1 = load_model(model1_path, compile=False)
            self.model2 = load_model(model2_path, compile=False)
            self.status_label.config(text="Status: AI Ready", fg="#2ED573")
            self.upload_button.config(state=tk.NORMAL)
            print("Models loaded successfully.")
        except Exception as e:
            self.status_label.config(text="Status: Model Loading Failed!", fg="#FF4757")
            messagebox.showerror("Model Loading Error")
            self.root.quit()

    def create_widgets(self):
        main_frame = tk.Frame(self.root, bg="#0F1014")
        main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        title_label = tk.Label(main_frame, text="PulmoSynth AI", bg="#0F1014", fg="white", font=("Inter", 28, "bold"))
        title_label.pack(pady=(0, 20))
        self.image_label = tk.Label(main_frame, bg="#1c1c1e", text="Upload an X-Ray Image to Begin Analysis", fg="white", font=("Inter", 12))
        self.image_label.pack(pady=10, fill=tk.BOTH, expand=True)
        self.upload_button = tk.Button(main_frame, text="Select Image", command=self.upload_image, bg="#007BFF", fg="white", font=("Inter", 14, "bold"), relief=tk.FLAT, padx=20, pady=10, state=tk.DISABLED)
        self.upload_button.pack(pady=20)
        self.result_label = tk.Label(main_frame, text="", bg="#0F1014", fg="white", font=("Inter", 24, "bold"))
        self.result_label.pack(pady=10)
        self.confidence_label = tk.Label(main_frame, text="", bg="#0F1014", fg="white", font=("Inter", 16))
        self.confidence_label.pack()
        self.status_label = tk.Label(self.root, text="Status: Initializing...", bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#1c1c1e", fg="white")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def upload_image(self):
        file_path = filedialog.askopenfilename(title="Select an X-Ray Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        try:
            img = Image.open(file_path)
            img.thumbnail((450, 450))
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
            self.result_label.config(text="Analyzing...", fg="white")
            self.confidence_label.config(text="")
            self.status_label.config(text="Status: Performing analysis...")
            self.root.update_idletasks()
            threading.Thread(target=self.predict, args=(file_path,), daemon=True).start()
        except Exception as e:
            messagebox.showerror("Image Error", f"Failed to open or display the image: {e}")
            self.status_label.config(text="Status: Error", fg="red")

    def predict(self, file_path):
        try:
            processed_img1 = self.preprocess_image(file_path, efficientnet_preprocess)
            processed_img2 = self.preprocess_image(file_path, densenet_preprocess)
            pred1 = self.model1.predict(processed_img1, verbose=0)[0][0]
            pred2 = self.model2.predict(processed_img2, verbose=0)[0][0]
            final_pred_score = (float(pred1) + float(pred2)) / 2
            if final_pred_score > 0.5:
                label = "Pneumonia"
                confidence = final_pred_score * 100
                self.result_label.config(fg="#FF4757")
            else:
                label = "Normal"
                confidence = (1 - final_pred_score) * 100
                self.result_label.config(fg="#2ED573")
            self.result_label.config(text=label)
            self.confidence_label.config(text=f"{confidence:.2f}% Confidence")
            self.status_label.config(text="Status: Analysis Complete", fg="#2ED573")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during AI analysis: {e}")
            self.result_label.config(text="Analysis Failed", fg="red")
            self.confidence_label.config(text="")
            self.status_label.config(text="Status: Error", fg="red")

    def preprocess_image(self, img_path, preprocess_func):
        """Helper function to load and prepare an image for a model."""
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        return preprocess_func(img_array_expanded)
    
if __name__ == "__main__":
    root = tk.Tk()
    app = PneumoniaApp(root)
    root.mainloop()
