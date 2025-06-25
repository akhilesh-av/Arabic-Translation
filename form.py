from groq import Groq
import easyocr
import dotenv
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Load environment variables
dotenv.load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize EasyOCR reader for Arabic
reader = easyocr.Reader(['ar'])

def extract_text_with_positions(image_path):
    """Extract Arabic text from image with bounding box positions"""
    result = reader.readtext(image_path)
    
    text_data = []
    for detection in result:
        bbox = detection[0]  # Bounding box coordinates
        text = detection[1]  # Extracted text
        confidence = detection[2]  # Confidence score
        
        # Calculate center position and dimensions
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        text_data.append({
            'text': text,
            'bbox': bbox,
            'position': {
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
                'center_x': (x_min + x_max) // 2,
                'center_y': (y_min + y_max) // 2,
                'width': x_max - x_min,
                'height': y_max - y_min
            },
            'confidence': confidence
        })
    
    return text_data

def get_translation(text):
    """Get simple translation using Groq's model"""
    client = Groq(api_key=GROQ_API_KEY)
    
    prompt = f"""
    Translate this Arabic text to English:
    {text}
    
    Rules:
    1. Keep it concise for form display
    2. Output ONLY the translation
    """
    
    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Translation error: {e}")
        return "Translation failed"

def get_background_color(image, position):
    """
    Analyze the background around text area to get dominant background color
    """
    # Convert coordinates to integers
    x_min = int(position['x_min'])
    y_min = int(position['y_min'])
    x_max = int(position['x_max'])
    y_max = int(position['y_max'])
    
    # Expand the area to sample background
    padding = 10
    sample_x_min = max(0, x_min - padding)
    sample_y_min = max(0, y_min - padding)
    sample_x_max = min(image.shape[1], x_max + padding)
    sample_y_max = min(image.shape[0], y_max + padding)
    
    # Extract border pixels around the text area
    border_pixels = []
    
    # Top border
    if sample_y_min < y_min:
        border_pixels.extend(image[sample_y_min:y_min, sample_x_min:sample_x_max].reshape(-1, 3))
    
    # Bottom border
    if sample_y_max > y_max:
        border_pixels.extend(image[y_max:sample_y_max, sample_x_min:sample_x_max].reshape(-1, 3))
    
    # Left border
    if sample_x_min < x_min:
        border_pixels.extend(image[sample_y_min:sample_y_max, sample_x_min:x_min].reshape(-1, 3))
    
    # Right border
    if sample_x_max > x_max:
        border_pixels.extend(image[sample_y_min:sample_y_max, x_max:sample_x_max].reshape(-1, 3))
    
    if border_pixels:
        border_pixels = np.array(border_pixels)
        # Get the most common color (mode)
        from scipy import stats
        try:
            mode_color = stats.mode(border_pixels, axis=0, keepdims=False)[0]
            return tuple(map(int, mode_color))
        except:
            # Fallback to mean color
            mean_color = np.mean(border_pixels, axis=0)
            return tuple(map(int, mean_color))
    
    # Fallback to white if no border pixels found
    return (255, 255, 255)

def intelligent_text_removal(image, position):
    """
    Remove text using content-aware filling techniques
    """
    # Convert coordinates to integers
    x_min = int(position['x_min'])
    y_min = int(position['y_min'])
    x_max = int(position['x_max'])
    y_max = int(position['y_max'])
    
    # Create a mask for the text area
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    mask[y_min:y_max, x_min:x_max] = 255
    
    # Use OpenCV's inpainting to fill the text area
    try:
        result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return result
    except:
        # Fallback: use background color analysis
        bg_color = get_background_color(image, position)
        result = image.copy()
        result[y_min:y_max, x_min:x_max] = bg_color
        return result

def get_text_color(image, position):
    """
    Analyze surrounding text to determine appropriate text color
    """
    # Convert coordinates to integers
    x_min = int(position['x_min'])
    y_min = int(position['y_min'])
    x_max = int(position['x_max'])
    y_max = int(position['y_max'])
    
    # Expand area to find nearby text
    padding = 20
    sample_x_min = max(0, x_min - padding)
    sample_y_min = max(0, y_min - padding) 
    sample_x_max = min(image.shape[1], x_max + padding)
    sample_y_max = min(image.shape[0], y_max + padding)
    
    # Get the area around the text
    surrounding_area = image[sample_y_min:sample_y_max, sample_x_min:sample_x_max]
    
    # Convert to grayscale to find dark areas (likely text)
    gray = cv2.cvtColor(surrounding_area, cv2.COLOR_RGB2GRAY)
    
    # Find pixels that are likely text (darker pixels)
    dark_threshold = np.percentile(gray, 25)  # Bottom 25% of brightness
    dark_pixels = surrounding_area[gray < dark_threshold]
    
    if len(dark_pixels) > 0:
        # Return the average color of dark pixels
        avg_color = np.mean(dark_pixels, axis=0)
        return tuple(map(int, avg_color))
    
    # Fallback to black
    return (0, 0, 0)


def calculate_font_size(text, max_width, max_height):
    """Calculate appropriate font size based on text length and available space"""
    # Simple font size calculation
    if len(text) == 0:
        return 12
    
    # Base calculation
    width_based = max(8, min(max_width // len(text) * 2, 24))
    height_based = max(8, min(max_height * 0.6, 24))
    
    return min(width_based, height_based)

def create_bilingual_form(image_path, output_path="translated_form.png"):
    """
    Create a form with Arabic text intelligently replaced by English translations
    """
    
    # Extract text with positions
    print("Extracting text from image...")
    text_data = extract_text_with_positions(image_path)
    
    if not text_data:
        print("No Arabic text found in the image!")
        return None
    
    # Load the original image
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Create a copy for processing
    processed_image = original_image.copy()
    
    # First pass: Remove all Arabic text using intelligent filling
    print("Removing original Arabic text...")
    for i, item in enumerate(text_data):
        print(f"Removing text {i+1}/{len(text_data)}")
        processed_image = intelligent_text_removal(processed_image, item['position'])
    
    # Convert to PIL for text rendering
    pil_image = Image.fromarray(processed_image)
    draw = ImageDraw.Draw(pil_image)
    
    # Second pass: Add English translations
    print("Adding English translations...")
    for i, item in enumerate(text_data):
        print(f"Translating text {i+1}/{len(text_data)}: {item['text'][:30]}...")
        
        # Get translation
        translation = get_translation(item['text'])
        
        # Get original text position
        pos = item['position']
        
        # Calculate font size to fit in the original text area
        font_size = calculate_font_size(translation, pos['width'], pos['height'])
        
        # Try to load font with better quality
        try:
            font_paths = [
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/calibri.ttf",
                "arial.ttf",
                "/System/Library/Fonts/Arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            ]
            
            font = None
            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except:
                    continue
            
            if font is None:
                font = ImageFont.load_default()
                
        except:
            font = ImageFont.load_default()
        
        # Get appropriate text color based on surrounding context
        text_color = get_text_color(original_image, pos)
        
        # Position the translated text at the center of the original text area
        text_x = pos['center_x']
        text_y = pos['center_y']
        
        # Draw the translated text with context-appropriate color
        try:
            draw.text(
                (text_x, text_y), 
                translation, 
                fill=text_color,
                font=font,
                anchor="mm"  # middle-middle
            )
        except:
            # Fallback without anchor for older PIL versions
            bbox = draw.textbbox((0, 0), translation, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            adjusted_x = text_x - text_width // 2
            adjusted_y = text_y - text_height // 2
            draw.text((adjusted_x, adjusted_y), translation, fill=text_color, font=font)
    
    # Save the translated form
    pil_image.save(output_path)
    print(f"Translated form saved to: {output_path}")
    
    return len(text_data)

def main():
    # Path to your form image
    image_path = "00100.jpeg"
    
    # Create translated form with Arabic text replaced by English
    print("Creating translated form with Arabic text replaced by English...")
    
    result = create_bilingual_form(
        image_path, 
        "translated_form_6.png"
    )
    
    if result:
        print(f"Successfully processed {result} text blocks")
        print("Translated form created successfully!")
        print("Arabic text has been replaced with English translations in black color")
    else:
        print("Failed to create translated form")

if __name__ == "__main__":
    main()

# from groq import Groq
# import easyocr
# import dotenv
# import os
# import cv2
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont

# # Load environment variables
# dotenv.load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# # Initialize EasyOCR reader for Arabic
# reader = easyocr.Reader(['ar'])

# def extract_text_with_positions(image_path):
#     """Extract Arabic text from image with bounding box positions"""
#     result = reader.readtext(image_path)
    
#     text_data = []
#     for detection in result:
#         bbox = detection[0]  # Bounding box coordinates
#         text = detection[1]  # Extracted text
#         confidence = detection[2]  # Confidence score
        
#         # Calculate center position and dimensions
#         x_coords = [point[0] for point in bbox]
#         y_coords = [point[1] for point in bbox]
        
#         x_min, x_max = min(x_coords), max(x_coords)
#         y_min, y_max = min(y_coords), max(y_coords)
        
#         text_data.append({
#             'text': text,
#             'bbox': bbox,
#             'position': {
#                 'x_min': x_min,
#                 'y_min': y_min,
#                 'x_max': x_max,
#                 'y_max': y_max,
#                 'center_x': (x_min + x_max) // 2,
#                 'center_y': (y_min + y_max) // 2,
#                 'width': x_max - x_min,
#                 'height': y_max - y_min
#             },
#             'confidence': confidence
#         })
    
#     return text_data

# def get_translation(text):
#     """Get simple translation using Groq's model"""
#     client = Groq(api_key=GROQ_API_KEY)
    
#     prompt = f"""
#     Translate this Arabic text to English:
#     {text}
    
#     Rules:
#     1. Keep it concise for form display
#     2. Output ONLY the translation
#     """
    
#     try:
#         completion = client.chat.completions.create(
#             model="meta-llama/llama-4-scout-17b-16e-instruct",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.3,
#             max_tokens=1024
#         )
#         return completion.choices[0].message.content.strip()
#     except Exception as e:
#         print(f"Translation error: {e}")
#         return "Translation failed"

# def calculate_font_size(text, max_width, max_height):
#     """Calculate appropriate font size based on text length and available space"""
#     # Simple font size calculation
#     if len(text) == 0:
#         return 12
    
#     # Base calculation
#     width_based = max(8, min(max_width // len(text) * 2, 24))
#     height_based = max(8, min(max_height * 0.6, 24))
    
#     return min(width_based, height_based)

# def create_bilingual_form(image_path, output_path="bilingual_form.png", translation_position="below"):
#     """
#     Create a bilingual form with translations embedded
#     """
    
#     # Extract text with positions
#     print("Extracting text from image...")
#     text_data = extract_text_with_positions(image_path)
    
#     if not text_data:
#         print("No Arabic text found in the image!")
#         return None
    
#     # Load the original image
#     original_image = cv2.imread(image_path)
#     original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
#     # Convert to PIL for text rendering
#     pil_image = Image.fromarray(original_image)
#     draw = ImageDraw.Draw(pil_image)
    
#     # Process each text element
#     for i, item in enumerate(text_data):
#         print(f"Processing text {i+1}/{len(text_data)}: {item['text'][:30]}...")
        
#         # Get translation
#         translation = get_translation(item['text'])
        
#         # Calculate font size
#         pos = item['position']
#         font_size = calculate_font_size(translation, pos['width'], pos['height'])
        
#         # Try to load font, fallback to default
#         try:
#             # Try common font paths
#             font_paths = [
#                 "arial.ttf",
#                 "C:/Windows/Fonts/arial.ttf",
#                 "/System/Library/Fonts/Arial.ttf",
#                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
#             ]
            
#             font = None
#             for font_path in font_paths:
#                 try:
#                     font = ImageFont.truetype(font_path, font_size)
#                     break
#                 except:
#                     continue
            
#             if font is None:
#                 font = ImageFont.load_default()
                
#         except:
#             font = ImageFont.load_default()
        
#         # Determine translation position
#         if translation_position == "below":
#             text_x = pos['center_x']
#             text_y = pos['y_max'] + 10
#             anchor = "mt"  # middle-top
#         elif translation_position == "above":
#             text_x = pos['center_x']
#             text_y = pos['y_min'] - 10
#             anchor = "mb"  # middle-bottom
#         elif translation_position == "overlay":
#             text_x = pos['center_x']
#             text_y = pos['center_y']
#             anchor = "mm"  # middle-middle
            
#             # Add background rectangle for overlay
#             bbox = draw.textbbox((text_x, text_y), translation, font=font, anchor=anchor)
#             padding = 5
#             draw.rectangle([
#                 bbox[0] - padding, bbox[1] - padding,
#                 bbox[2] + padding, bbox[3] + padding
#             ], fill=(255, 255, 255, 200), outline=(0, 0, 0))
            
#         else:  # side
#             text_x = pos['x_max'] + 10
#             text_y = pos['center_y']
#             anchor = "lm"  # left-middle
        
#         # Draw translation text
#         try:
#             draw.text(
#                 (text_x, text_y), 
#                 translation, 
#                 fill=(0, 0, 255),  # Blue color for translations
#                 font=font,
#                 anchor=anchor
#             )
#         except:
#             # Fallback without anchor for older PIL versions
#             draw.text((text_x, text_y), translation, fill=(0, 0, 255), font=font)
        
#         # Draw bounding box around original text
#         draw.rectangle([
#             pos['x_min'], pos['y_min'],
#             pos['x_max'], pos['y_max']
#         ], outline=(255, 0, 0), width=2)
    
#     # Save the bilingual form
#     pil_image.save(output_path)
#     print(f"Bilingual form saved to: {output_path}")
    
#     return len(text_data)

# def main():
#     # Path to your form image
#     image_path = "001.jpeg"
    
#     # Create bilingual form with translations below original text
#     print("Creating bilingual form with translations below original text...")
    
#     result = create_bilingual_form(
#         image_path, 
#         "bilingual_form_below2.png", 
#         translation_position="below"
#     )
    
#     if result:
#         print(f"Successfully processed {result} text blocks")
#         print("Bilingual form created successfully!")
#     else:
#         print("Failed to create bilingual form")

# if __name__ == "__main__":
#     main()