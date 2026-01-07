import torch
import os
from PIL import Image
from diffusers import LongCatImageEditPipeline
from pathlib import Path

if __name__ == '__main__':
    device = torch.device('cuda')

    # Setup input and output folders
    input_folder = '../data/images/'
    output_folder = '../data/edit_images/'
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Initialize the pipeline
    pipe = LongCatImageEditPipeline.from_pretrained("meituan-longcat/LongCat-Image-Edit", torch_dtype=torch.bfloat16)
    pipe.to(device, torch.bfloat16)  # Uncomment for high VRAM devices (Faster inference)
    #pipe.enable_model_cpu_offload()  # Offload to CPU to save VRAM (Required ~18 GB); slower but prevents OOM

    # Define prompts
    prompts = {
        #'face_enhancement': 'Keep the background exactly the same. Rotate the face toward the camera so the eyes and facial expression are clearly visible. Add a soft, natural smile that conveys warmth and comfort. Maintain natural lighting and candid lifestyle photography style.',
        'text_translation': 'Keep the background completely unchanged. Locate all text content in the image, translate it to Simplified Chinese, and overlay it back in the exact same position using the same font style and formatting. Produce an image identical to the original except with translated text.'
    }

    # Get all image files from input folder
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    image_files = [f for f in os.listdir(input_folder) 
                   if os.path.isfile(os.path.join(input_folder, f)) 
                   and Path(f).suffix.lower() in image_extensions]

    print(f"Found {len(image_files)} images in {input_folder}")

    # Process each image with both prompts
    for image_file in image_files:
        print(f"\nProcessing: {image_file}")
        
        # Load image
        img_path = os.path.join(input_folder, image_file)
        img = Image.open(img_path).convert('RGB')
        
        # Get filename without extension
        filename_base = Path(image_file).stem
        filename_ext = Path(image_file).suffix
        
        # Process with each prompt
        for prompt_name, prompt_text in prompts.items():
            print(f"  Applying {prompt_name}...")
            
            # Generate edited image
            image = pipe(
                img,
                prompt_text,
                negative_prompt='',
                guidance_scale=4.5,
                num_inference_steps=50,
                num_images_per_prompt=1,
                generator=torch.Generator("cpu").manual_seed(43)
            ).images[0]
            
            # Save output with descriptive name
            output_filename = f"{filename_base}_{prompt_name}{filename_ext}"
            output_path = os.path.join(output_folder, output_filename)
            image.save(output_path)
            print(f"  Saved: {output_filename}")

    print(f"\n✓ All images processed! Results saved to {output_folder}")
