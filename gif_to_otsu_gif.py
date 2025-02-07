from PIL import Image, ImageSequence
import numpy as np
import cv2
import os

def process_gif(input_gif_path, output_gif_path):
    # Open the input GIF
    gif = Image.open(input_gif_path)
    
    frames = []
    durations = []

    # Iterate over each frame in the GIF
    for frame in ImageSequence.Iterator(gif):
        # Get the frame duration
        duration = frame.info.get('duration', 100)  # Default to 100ms if duration not found
        durations.append(duration)
        
        # Convert frame to RGB if it's in palette mode
        if frame.mode == 'P':
            frame = frame.convert('RGB')
        
        # Convert frame to numpy array
        frame_np = np.array(frame)
        
        # Convert to grayscale
        frame_gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
        
        # Apply Otsu's thresholding
        _, thresh = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert numpy array back to PIL Image
        frame_pil = Image.fromarray(thresh)
        
        # Append processed frame to list
        frames.append(frame_pil)
    
    # Save all frames as a new GIF
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0
    )


folder = "/home/ilmari/python/limited-angle-tomography/BenchmarkFullHTCModel_AutoEnc=Patch40LV10Stride10_Coeff=0.1_Filt=5.0_Model=True_P=2_TV=1.0_Time=200"

for file in os.listdir(folder):
    if file.endswith(".gif"):
        input_gif_path = os.path.join(folder, file)
        output_gif_path = os.path.join(folder, file.replace(".gif", "_otsu.gif"))
        process_gif(input_gif_path, output_gif_path)
        print(f"Processed {input_gif_path} and saved to {output_gif_path}")

