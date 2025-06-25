import os
import json
import easyocr
from groq import Groq
import google.generativeai as genai
from PIL import Image
import dotenv
from datetime import datetime

# Load environment variables
dotenv.load_dotenv()

# Initialize APIs
# =================================================================
# Groq setup
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Gemini setup (updated to use latest models)
genai.configure(api_key=os.getenv("GEMINI_KEY"))
try:
    # Try the newest flash model first, fall back to pro if needed
    gemini_flash = genai.GenerativeModel('gemini-1.5-flash')
    gemini_pro = genai.GenerativeModel('gemini-1.5-pro')
    print("✓ Gemini models initialized successfully (1.5-flash and 1.5-pro available)")
except Exception as e:
    print(f"Gemini initialization error: {e}")
    gemini_flash = None
    gemini_pro = None

# EasyOCR setup
reader = easyocr.Reader(['ar'])

# Core Functions
# =================================================================
def extract_text_from_image(image_path):
    """Extract Arabic text from image using EasyOCR"""
    result = reader.readtext(image_path)
    return " ".join([text[1] for text in result])

def translate_with_groq(text):
    """Translate Arabic to English using Groq/Llama"""
    response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
            "role": "user",
            "content": f"Translate this Arabic text to English exactly without additions:\n{text}"
        }],
        temperature=0.3,
        max_tokens=1024
    )
    return response.choices[0].message.content

def translate_with_gemini(text, model_type='flash'):
    """Translate Arabic to English using Gemini's latest models"""
    if model_type == 'flash' and gemini_flash:
        model = gemini_flash
    elif model_type == 'pro' and gemini_pro:
        model = gemini_pro
    else:
        return "Gemini translation unavailable"
    
    try:
        response = model.generate_content(
            f"Provide exact English translation of this Arabic text without any additional commentary or explanations:\n{text}",
            generation_config={
                "temperature": 0.2,  # Lower temperature for more precise translations
                "max_output_tokens": 1000
            }
        )
        return response.text
    except Exception as e:
        print(f"Gemini translation error ({model_type}): {e}")
        return f"Translation error: {str(e)}"

# Processing Pipeline
# =================================================================
def process_images_folder(folder_path):
    results = []
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            
            print(f"\nProcessing {filename}...")
            
            try:
                # Extract Arabic text
                arabic_text = extract_text_from_image(image_path)
                
                # Get translations
                groq_translation = translate_with_groq(arabic_text)
                gemini_flash_translation = translate_with_gemini(arabic_text, 'flash')
                gemini_pro_translation = translate_with_gemini(arabic_text, 'pro')
                
                # Store results
                results.append({
                    "image_file": filename,
                    "arabic_text": arabic_text,
                    "translations": {
                        "groq_llama": groq_translation,
                        "google_gemini_flash": gemini_flash_translation,
                        "google_gemini_pro": gemini_pro_translation
                    },
                    "processing_time": datetime.now().isoformat()
                })
                
                print(f"✓ Completed {filename}")
                
            except Exception as e:
                print(f"✗ Error processing {filename}: {str(e)}")
                results.append({
                    "image_file": filename,
                    "error": str(e)
                })
    
    return results

# Main Execution
# =================================================================
if __name__ == "__main__":
    # Configuration
    IMAGE_FOLDER = "data"  # Folder containing Arabic images
    OUTPUT_JSON = "translations_results.json"
    
    # Process all images
    translation_results = process_images_folder(IMAGE_FOLDER)
    
    # Save results
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(translation_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Results saved to {OUTPUT_JSON}")
    print(f"Processed {len(translation_results)} images")