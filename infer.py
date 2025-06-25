import easyocr
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from googletrans import Translator
import google.generativeai as genai
from difflib import SequenceMatcher
import numpy as np
import warnings
import time

# Suppress warnings
warnings.filterwarnings("ignore")

class ArabicTranslationEvaluator:
    def __init__(self, gemini_api_key):
        # Initialize OCR reader first
        print("Initializing EasyOCR...")
        self.reader = easyocr.Reader(['ar'], gpu=True)  # Set gpu=True if you have CUDA
        
        # Then initialize translation models
        self.initialize_huggingface_models()
        self.initialize_google_translator()
        self.initialize_gemini(gemini_api_key)
        
    def initialize_huggingface_models(self):
        # Helsinki-NLP model for Arabic to English translation
        print("Loading Helsinki-NLP model...")
        self.helsinki_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-ar-en")
        self.helsinki_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-tc-big-ar-en")
        
        # ALLaM model
        print("Loading ALLaM model...")
        self.llama_tokenizer = AutoTokenizer.from_pretrained("ALLaM-AI/ALLaM-7B-Instruct-preview")
        self.llama_model = AutoModelForCausalLM.from_pretrained("ALLaM-AI/ALLaM-7B-Instruct-preview")
        print("All models loaded successfully!")
    
    def initialize_google_translator(self):
        self.google_translator = Translator()
    
    def initialize_gemini(self, api_key):
        try:
            genai.configure(api_key=api_key, transport='rest')
            
            # Verify model availability
            available_models = [m.name for m in genai.list_models()]
            if 'models/gemini-pro' in available_models:
                model_name = 'gemini-pro'
            elif 'models/gemini-1.0-pro' in available_models:
                model_name = 'gemini-1.0-pro'
            else:
                raise ValueError(f"No Gemini model found. Available: {available_models}")
                
            self.gemini_model = genai.GenerativeModel(model_name)
            print(f"Initialized Gemini with model: {model_name}")
        except Exception as e:
            print(f"Failed to initialize Gemini: {str(e)}")
            raise
    
    def extract_text_from_image(self, image_path):
        """Extract Arabic text from image using EasyOCR"""
        try:
            start_time = time.time()
            # Read the image
            result = self.reader.readtext(image_path, detail=0)
            
            # Combine all detected text
            arabic_text = " ".join(result).strip()
            
            if not arabic_text:
                return {"error": "No text detected in image", "status": "failed"}
            
            print(f"OCR completed in {time.time() - start_time:.2f} seconds")
            return {"text": arabic_text, "status": "success"}
        except Exception as e:
            return {"error": f"OCR Error: {str(e)}", "status": "failed"}
    
    def translate_with_helsinki(self, arabic_text):
        """Translate using Helsinki-NLP model"""
        try:
            inputs = self.helsinki_tokenizer(arabic_text, return_tensors="pt")
            outputs = self.helsinki_model.generate(**inputs)
            return self.helsinki_tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            return f"Translation error: {str(e)}"
    
    def translate_with_llama(self, arabic_text):
        """Translate using ALLaM model with instruction prompting"""
        try:
            prompt = f"Translate the following Arabic text to English accurately:\n\n{arabic_text}\n\nTranslation:"
            inputs = self.llama_tokenizer(prompt, return_tensors="pt")
            outputs = self.llama_model.generate(**inputs, max_length=500)
            translation = self.llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Clean up the output to remove the prompt
            return translation.replace(prompt, "").strip()
        except Exception as e:
            return f"Translation error: {str(e)}"
    
    def translate_with_google(self, arabic_text):
        """Translate using Google Translate API"""
        try:
            translation = self.google_translator.translate(arabic_text, src='ar', dest='en')
            return translation.text
        except Exception as e:
            return f"Google Translate error: {str(e)}"
    
    def translate_with_gemini(self, arabic_text):
        """Translate using Gemini"""
        try:
            prompt = f"Translate the following Arabic text to English accurately and preserve the meaning:\n\n{arabic_text}"
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Gemini error: {str(e)}"
    
    def calculate_similarity(self, text1, text2):
        """Calculate similarity score between two texts"""
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def evaluate_translations(self, arabic_text):
        """Get translations from all methods and compare with Gemini"""
        if not arabic_text:
            return {"error": "No text provided for translation", "status": "failed"}
        
        print("\nGenerating translations...")
        start_time = time.time()
        
        # Get all translations
        translations = {
            "Helsinki-NLP": self.translate_with_helsinki(arabic_text),
            "ALLaM": self.translate_with_llama(arabic_text),
            "Google Translate": self.translate_with_google(arabic_text),
            "Gemini": self.translate_with_gemini(arabic_text)
        }
        
        print(f"Translations completed in {time.time() - start_time:.2f} seconds")
        
        # Calculate accuracy scores compared to Gemini
        gemini_translation = translations["Gemini"]
        accuracy_scores = {}
        
        for method, translation in translations.items():
            if method != "Gemini":
                if "error" not in str(gemini_translation).lower() and "error" not in str(translation).lower():
                    accuracy_scores[method] = self.calculate_similarity(translation, gemini_translation)
                else:
                    accuracy_scores[method] = 0.0
            else:
                accuracy_scores[method] = 1.0  # Gemini compared to itself
        
        return {
            "translations": translations,
            "accuracy_scores": accuracy_scores,
            "status": "success"
        }
    
    def process_image(self, image_path):
        """Complete pipeline from image to evaluation"""
        print(f"\nProcessing image: {image_path}")
        
        # Step 1: Extract Arabic text from image
        ocr_result = self.extract_text_from_image(image_path)
        
        if ocr_result["status"] == "failed":
            return {"error": ocr_result["error"], "status": "failed"}
        
        arabic_text = ocr_result["text"]
        print(f"\nExtracted Arabic Text:\n{arabic_text}")
        
        # Step 2: Get translations and evaluations
        evaluation = self.evaluate_translations(arabic_text)
        
        if evaluation["status"] == "failed":
            return {"error": evaluation["error"], "status": "failed"}
        
        # Return complete results
        return {
            "original_text": arabic_text,
            **evaluation,
            "status": "success"
        }


if __name__ == "__main__":
    try:
        # Initialize with your Gemini API key
        print("Initializing translation system...")
        start_time = time.time()
        
        # Replace with your actual Gemini API key
        evaluator = ArabicTranslationEvaluator("AIzaSyCpDBj-V3T3DUEey03SmGmAgONnTGN_tEs")
        
        print(f"System initialized in {time.time() - start_time:.2f} seconds")
        
        # Process an Arabic image
        image_path = "data\pic_8.jpeg"  # Change to your image path
        results = evaluator.process_image(image_path)
        
        if results["status"] == "failed":
            print(f"\nError: {results['error']}")
        else:
            # Print results
            print("\nTranslation Results:")
            for method, translation in results["translations"].items():
                print(f"\n{method} Translation:")
                print(translation)
            
            print("\nAccuracy Scores (compared to Gemini):")
            for method, score in results["accuracy_scores"].items():
                print(f"{method}: {score:.2f}")
                
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
