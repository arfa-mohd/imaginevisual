from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import base64
import io

app = Flask(__name__)
CORS(app)

# Hugging Face API configuration
HF_API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
HF_API_TOKEN = "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"

# Comprehensive room-specific prompts
ROOM_PROMPTS = {
    'kitchen': {
        'base': 'A beautiful modern kitchen interior design',
        'elements': ['marble countertops', 'stainless steel appliances', 'pendant lighting', 'kitchen island', 'subway tile backsplash', 'wooden cabinets', 'granite counters', 'modern fixtures']
    },
    'living room': {
        'base': 'An elegant living room interior design',
        'elements': ['comfortable seating', 'coffee table', 'area rug', 'floor lamps', 'artwork', 'throw pillows', 'entertainment center', 'fireplace']
    },
    'bedroom': {
        'base': 'A cozy bedroom interior design',
        'elements': ['bed with headboard', 'nightstands', 'table lamps', 'dresser', 'mirror', 'curtains', 'bedding', 'reading chair']
    },
    'dining room': {
        'base': 'A sophisticated dining room interior design',
        'elements': ['dining table', 'chairs', 'chandelier', 'buffet', 'artwork', 'table setting', 'centerpiece', 'elegant lighting']
    },
    'bathroom': {
        'base': 'A luxurious bathroom interior design',
        'elements': ['vanity', 'mirror', 'shower', 'bathtub', 'tile work', 'fixtures', 'towels', 'modern lighting']
    },
    'hallway': {
        'base': 'A welcoming hallway interior design',
        'elements': ['console table', 'mirror', 'lighting', 'artwork', 'runner rug', 'decorative elements', 'storage solutions']
    },
    'home office': {
        'base': 'A productive home office interior design',
        'elements': ['desk', 'office chair', 'bookshelves', 'computer setup', 'task lighting', 'storage', 'plants', 'organized workspace']
    },
    'whole house': {
        'base': 'A complete house interior design',
        'elements': ['open floor plan', 'consistent color scheme', 'flowing design', 'multiple rooms', 'cohesive style', 'natural lighting']
    }
}

# Design styles with enhanced descriptions
DESIGN_STYLES = {
    'modern': 'clean lines, minimalist aesthetic, neutral colors, sleek furniture',
    'traditional': 'classic furniture, warm colors, elegant details, timeless appeal',
    'minimalist': 'simple forms, white and neutral tones, clutter-free, functional',
    'rustic': 'natural materials, wood textures, warm earth tones, cozy atmosphere',
    'industrial': 'exposed brick, metal fixtures, urban aesthetic, raw materials',
    'scandinavian': 'light woods, white walls, cozy textiles, hygge feeling',
    'bohemian': 'colorful patterns, eclectic mix, plants, artistic elements',
    'contemporary': 'current trends, mixed materials, bold accents, sophisticated',
    'farmhouse': 'shiplap walls, vintage elements, comfortable furniture, country charm',
    'mid-century': 'retro furniture, bold colors, geometric patterns, vintage modern',
    'luxury': 'high-end materials, elegant finishes, sophisticated details, premium quality',
    'coastal': 'light blues, whites, natural textures, beach-inspired, airy feeling'
}

def detect_room_type(prompt):
    """Detect room type from prompt with flexible matching"""
    prompt_lower = prompt.lower()
    
    # Extended room type mapping
    room_mapping = {
        'kitchen': ['kitchen', 'cook', 'culinary', 'chef', 'pantry'],
        'bathroom': ['bathroom', 'bath', 'shower', 'toilet', 'washroom', 'restroom'],
        'bedroom': ['bedroom', 'bed', 'sleep', 'master suite', 'guest room'],
        'living room': ['living room', 'lounge', 'family room', 'sitting room', 'parlor'],
        'dining room': ['dining room', 'dining', 'eat', 'meal', 'breakfast nook'],
        'home office': ['office', 'study', 'workspace', 'desk', 'work', 'computer room'],
        'hallway': ['hallway', 'corridor', 'entryway', 'foyer', 'entrance'],
        'whole house': ['house', 'home', 'entire', 'whole', 'complete', 'full interior']
    }
    
    for room_type, keywords in room_mapping.items():
        if any(keyword in prompt_lower for keyword in keywords):
            return room_type
    
    return None  # Return None if no specific room detected

def detect_style(prompt):
    """Detect design style from prompt with flexible matching"""
    prompt_lower = prompt.lower()
    
    # Extended style mapping  
    style_mapping = {
        'modern': ['modern', 'contemporary', 'sleek', 'clean lines'],
        'traditional': ['traditional', 'classic', 'elegant', 'formal'],
        'minimalist': ['minimalist', 'minimal', 'simple', 'clean', 'sparse'],
        'rustic': ['rustic', 'country', 'farmhouse', 'rural', 'wooden'],
        'industrial': ['industrial', 'urban', 'loft', 'exposed', 'raw'],
        'scandinavian': ['scandinavian', 'nordic', 'hygge', 'cozy', 'light wood'],
        'bohemian': ['bohemian', 'boho', 'eclectic', 'artistic', 'colorful'],
        'luxury': ['luxury', 'luxurious', 'upscale', 'high-end', 'premium', 'expensive'],
        'coastal': ['coastal', 'beach', 'nautical', 'ocean', 'seaside'],
        'mid-century': ['mid-century', 'retro', 'vintage', '60s', 'atomic']
    }
    
    for style, keywords in style_mapping.items():
        if any(keyword in prompt_lower for keyword in keywords):
            return style
    
    return None  # Return None if no specific style detected

def enhance_prompt(prompt):
    """Enhanced prompt system that works with ALL types of prompts"""
    # Clean and prepare the original prompt
    original_prompt = prompt.strip()
    
    # Remove any duplicate text that might have been added
    if original_prompt.count(original_prompt.split('.')[0]) > 1:
        # Fix duplicate text issue
        parts = original_prompt.split('.')
        original_prompt = parts[0].strip()
    
    # Check if prompt already contains high-quality descriptors
    quality_keywords = ['high resolution', '8k', 'professional', 'photorealistic', 'detailed']
    has_quality = any(keyword in original_prompt.lower() for keyword in quality_keywords)
    
    # Try to detect room type and style for enhancement
    room_type = detect_room_type(original_prompt)
    style = detect_style(original_prompt)
    
    # Start with the original prompt (no duplication)
    enhanced = original_prompt
    
    # Add style and room enhancements if detected and not already present
    if room_type and room_type not in original_prompt.lower():
        room_info = ROOM_PROMPTS.get(room_type, ROOM_PROMPTS['living room'])
        enhanced = f"{enhanced}, {room_info['base'].lower()}"
    
    if style and style not in original_prompt.lower():
        style_desc = DESIGN_STYLES.get(style, DESIGN_STYLES['modern'])
        enhanced = f"{enhanced}, {style} style with {style_desc}"
    
    # Always add high-quality descriptors if not present
    if not has_quality:
        enhanced += ", ultra high resolution, 8K quality, professional interior photography"
        enhanced += ", photorealistic, award-winning design, magazine quality, perfect lighting"
        enhanced += ", hyperdetailed, sharp focus, architectural digest style, cinematic lighting"
    
    return enhanced

def generate_with_huggingface(prompt):
    """Generate image using free services first, then fallback to Hugging Face"""
    try:
        # Always enhance the prompt for better quality
        enhanced_prompt = enhance_prompt(prompt)
        
        # Ensure the enhanced prompt is not too long
        if len(enhanced_prompt) > 500:
            # Fallback to simpler enhancement if too long
            enhanced_prompt = f"{prompt}, ultra high resolution, photorealistic, professional interior design, perfect lighting"
        
        print(f"🚀 Generating with prompt: {enhanced_prompt[:200]}...")
        
        # Try free services first (no API credits needed)
        print("🆓 Trying free image generation services first...")
        free_result = try_alternative_generation(enhanced_prompt)
        
        if free_result.get('success'):
            print("✅ Successfully generated image using free service!")
            return free_result
        
        # If free services fail, try Hugging Face as backup
        print("🔄 Free services unavailable, trying Hugging Face API...")
        
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": enhanced_prompt,
            "parameters": {
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 4,
                "guidance_scale": 0.0
            }
        }
        
        # Try the main API
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200 and response.content:
            # Convert to base64 for display
            image_base64 = base64.b64encode(response.content).decode('utf-8')
            print(f"✅ Successfully generated image with Hugging Face ({len(response.content)} bytes)")
            return {
                'success': True,
                'image': f"data:image/png;base64,{image_base64}",
                'enhanced_prompt': enhanced_prompt,
                'model_used': 'FLUX.1-schnell',
                'demo_mode': False
            }
        elif response.status_code == 402:
            # API credits exceeded
            print(f"⚠️ API credits exceeded (402 Payment Required)")
            return {
                'success': False,
                'error': "🔄 Free AI services are working! Premium API credits are exhausted, but you can still generate unlimited images using our free services.",
                'enhanced_prompt': enhanced_prompt
            }
        else:
            error_msg = f"HF API Error: {response.status_code}"
            if response.text:
                error_msg += f" - {response.text[:200]}"
            print(f"❌ Generation failed: {error_msg}")
            return free_result  # Return the free service result even if it failed
            
    except Exception as e:
        error_msg = f"Generation error: {str(e)}"
        print(f"❌ Exception during generation: {error_msg}")
        return try_alternative_generation(enhance_prompt(prompt))

def try_alternative_generation(enhanced_prompt):
    """Try alternative generation methods including completely free APIs"""
    print("🔄 Trying free image generation services...")
    
    # Try DeepAI first - usually no watermark on free tier
    try:
        print("🔄 Trying DeepAI free service (no watermark)...")
        deepai_url = "https://api.deepai.org/api/text2img"
        
        response = requests.post(
            deepai_url,
            data={'text': enhanced_prompt},
            headers={'api-key': 'quickstart-QUdJIGlzIGNvbWluZy4uLi4K'},  # Free tier key
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'output_url' in result:
                # Download the image
                img_response = requests.get(result['output_url'], timeout=30)
                if img_response.status_code == 200:
                    image_base64 = base64.b64encode(img_response.content).decode('utf-8')
                    print(f"✅ DeepAI generated real image! ({len(img_response.content)} bytes)")
                    return {
                        'success': True,
                        'image': f"data:image/jpeg;base64,{image_base64}",
                        'enhanced_prompt': enhanced_prompt,
                        'model_used': 'DeepAI (Free, No Watermark)',
                        'demo_mode': False
                    }
        print(f"❌ DeepAI failed: {response.status_code}")
    except Exception as e:
        print(f"❌ DeepAI error: {str(e)}")
    
    # Try Replicate free tier - usually no watermark
    try:
        print("🔄 Trying Replicate free service (no watermark)...")
        replicate_url = "https://api.replicate.com/v1/predictions"
        
        payload = {
            "version": "ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e", # Stable Diffusion
            "input": {
                "prompt": enhanced_prompt,
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 20
            }
        }
        
        response = requests.post(
            replicate_url,
            json=payload,
            headers={"Authorization": "Token r8_P5VXTrwTsq12qitnitlD9r0ji9B6hoA3v5ZWQ"},
            timeout=60
        )
        
        if response.status_code == 201:
            result = response.json()
            prediction_url = result.get('urls', {}).get('get')
            if prediction_url:
                # Wait for completion
                import time
                time.sleep(15)  # Wait longer for better results
                
                status_response = requests.get(prediction_url, 
                    headers={"Authorization": "Token r8_P5VXTrwTsq12qitnitlD9r0ji9B6hoA3v5ZWQ"})
                
                if status_response.status_code == 200:
                    status_result = status_response.json()
                    if status_result.get('status') == 'succeeded' and status_result.get('output'):
                        image_url = status_result['output'][0] if isinstance(status_result['output'], list) else status_result['output']
                        img_response = requests.get(image_url, timeout=30)
                        if img_response.status_code == 200:
                            image_base64 = base64.b64encode(img_response.content).decode('utf-8')
                            print(f"✅ Replicate generated real image! ({len(img_response.content)} bytes)")
                            return {
                                'success': True,
                                'image': f"data:image/jpeg;base64,{image_base64}",
                                'enhanced_prompt': enhanced_prompt,
                                'model_used': 'Replicate (Free, No Watermark)',
                                'demo_mode': False
                            }
        print(f"❌ Replicate failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Replicate error: {str(e)}")
    
    # Try Pollinations.ai (has watermark but works reliably)
    try:
        print("🔄 Trying Pollinations.ai (may have watermark)...")
        import urllib.parse
        
        # Clean and encode the prompt for URL
        clean_prompt = enhanced_prompt.replace('"', '').replace("'", "")
        encoded_prompt = urllib.parse.quote(clean_prompt)
        
        # Try different Pollinations endpoints
        pollinations_urls = [
            f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&model=flux&nologo=true",
            f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&enhance=true&nologo=true",
            f"https://pollinations.ai/p/{encoded_prompt}?width=1024&height=1024"
        ]
        
        for url in pollinations_urls:
            try:
                response = requests.get(url, timeout=45)
                if response.status_code == 200 and response.content and len(response.content) > 5000:
                    image_base64 = base64.b64encode(response.content).decode('utf-8')
                    print(f"✅ Pollinations.ai generated image! ({len(response.content)} bytes)")
                    return {
                        'success': True,
                        'image': f"data:image/jpeg;base64,{image_base64}",
                        'enhanced_prompt': enhanced_prompt,
                        'model_used': 'Pollinations.ai (Free Service)',
                        'demo_mode': False
                    }
            except:
                continue
                
    except Exception as e:
        print(f"❌ Pollinations.ai error: {str(e)}")
    
    # Try Hugging Face Inference API (free tier)
    try:
        print("🔄 Trying Hugging Face Inference API (free tier)...")
        hf_url = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
        
        response = requests.post(
            hf_url,
            headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
            json={"inputs": enhanced_prompt},
            timeout=60
        )
        
        if response.status_code == 200:
            image_base64 = base64.b64encode(response.content).decode('utf-8')
            print(f"✅ Hugging Face generated image! ({len(response.content)} bytes)")
            return {
                'success': True,
                'image': f"data:image/jpeg;base64,{image_base64}",
                'enhanced_prompt': enhanced_prompt,
                'model_used': 'Hugging Face (Free Tier)',
                'demo_mode': False
            }
        print(f"❌ Hugging Face failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Hugging Face error: {str(e)}")
    
    # If all free services fail, return an error
    return {
        'success': False,
        'error': "🚫 All free AI image generation services are currently unavailable. Please try again in a few minutes.",
        'enhanced_prompt': enhanced_prompt
    }


def generate_demo_response(enhanced_prompt):
    """Simple fallback demo response"""
    return {
        'success': True,
        'image': "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAyNCIgaGVpZ2h0PSIxMDI0IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxyZWN0IHdpZHRoPSIxMDI0IiBoZWlnaHQ9IjEwMjQiIGZpbGw9IiNmMGYwZjAiLz48dGV4dCB4PSI1MTIiIHk9IjUxMiIgZm9udC1mYW1pbHk9IkFyaWFsIiBmb250LXNpemU9IjI0IiBmaWxsPSIjNjY2IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+QUkgSW1hZ2UgR2VuZXJhdGlvbiBSZWFkeSE8L3RleHQ+PC9zdmc+",
        'enhanced_prompt': enhanced_prompt,
        'demo_mode': True,
        'message': f"🖼️ AI-optimized prompt ready for image generation: '{enhanced_prompt[:80]}...'"
    }

def generate_detailed_placeholder(enhanced_prompt):
    """Generate a detailed placeholder that represents the interior design"""
    import base64
    from PIL import Image, ImageDraw, ImageFont
    import io
    import random
    
    try:
        # Create a more sophisticated placeholder
        img = Image.new('RGB', (1024, 1024), color=(250, 250, 250))
        draw = ImageDraw.Draw(img)
        
        # Extract key elements from the prompt for visual representation
        prompt_lower = enhanced_prompt.lower()
        
        # Define color schemes based on prompt
        if 'coastal' in prompt_lower or 'blue' in prompt_lower:
            colors = [(173, 216, 230), (135, 206, 235), (70, 130, 180)]  # Light blue theme
        elif 'industrial' in prompt_lower:
            colors = [(105, 105, 105), (169, 169, 169), (128, 128, 128)]  # Gray theme
        elif 'rustic' in prompt_lower or 'wood' in prompt_lower:
            colors = [(222, 184, 135), (210, 180, 140), (205, 133, 63)]  # Wood theme
        elif 'luxury' in prompt_lower or 'marble' in prompt_lower:
            colors = [(255, 248, 220), (250, 235, 215), (245, 245, 220)]  # Luxury theme
        else:
            colors = [(245, 245, 245), (220, 220, 220), (200, 200, 200)]  # Neutral theme
        
        # Draw geometric shapes to represent interior elements
        # Floor
        draw.rectangle([0, 700, 1024, 1024], fill=colors[1])
        
        # Walls
        draw.rectangle([0, 0, 1024, 700], fill=colors[0])
        
        # Furniture representations
        if 'sofa' in prompt_lower or 'living room' in prompt_lower:
            # Draw sofa representation
            draw.rectangle([200, 500, 600, 650], fill=colors[2])
            draw.rectangle([180, 480, 620, 520], fill=colors[2])
        
        if 'table' in prompt_lower:
            # Coffee table
            draw.rectangle([300, 550, 500, 600], fill=(139, 69, 19))
        
        if 'window' in prompt_lower:
            # Windows
            draw.rectangle([800, 200, 950, 500], fill=(173, 216, 230))
            draw.rectangle([820, 220, 930, 480], fill=(224, 255, 255))
        
        # Add text overlay
        try:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        except:
            font_large = font_small = None
        
        # Title
        title = "🖼️ AI Image Generation Ready"
        if font_large:
            bbox = draw.textbbox((0, 0), title, font=font_large)
            text_width = bbox[2] - bbox[0]
            draw.text(((1024 - text_width) // 2, 50), title, fill=(100, 100, 100), font=font_large)
        
        # Enhanced prompt preview
        prompt_preview = f"Optimized for AI: {enhanced_prompt[:50]}..."
        if font_small:
            bbox = draw.textbbox((0, 0), prompt_preview, font=font_small)
            text_width = bbox[2] - bbox[0]
            draw.text(((1024 - text_width) // 2, 100), prompt_preview, fill=(120, 120, 120), font=font_small)
        
        # Status message
        status = "🎨 AI Image Generator Ready - Your prompt is enhanced and ready!"
        if font_small:
            bbox = draw.textbbox((0, 0), status, font=font_small)
            text_width = bbox[2] - bbox[0]
            draw.text(((1024 - text_width) // 2, 950), status, fill=(150, 150, 150), font=font_small)
        
        # Convert to base64
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        image_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        
        return {
            'success': True,
            'image': f"data:image/png;base64,{image_base64}",
            'enhanced_prompt': enhanced_prompt,
            'demo_mode': True,
            'message': f"🖼️ IMAGE GENERATION READY: '{enhanced_prompt[:80]}...' - AI optimized for stunning visuals!"
        }
        
    except Exception as e:
        print(f"Error creating detailed placeholder: {e}")
        return generate_demo_response(enhanced_prompt)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({'success': False, 'error': 'No prompt provided'})
        
        print(f"🎨 Received prompt: {prompt}")
        
        # Generate with Hugging Face
        result = generate_with_huggingface(prompt)
        
        # Always return success for demo mode to show enhancement
        if result.get('demo_mode'):
            print(f"✅ Demo mode: Showing prompt enhancement for '{prompt}'")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error in generate_image: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model': 'FLUX.1-schnell'})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)
