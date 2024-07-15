#  Image Generation with Pre-trained Models 

Welcome to the Image Generation with Pre-trained Models project! This repository brings the magic of AI-driven creativity to your fingertips. Here, you’ll learn how to transform textual descriptions into vivid images using state-of-the-art generative models like DALL-E-MINI and Stable Diffusion, with a special focus on the latter, powered by the `diffusers` library.

##  Introduction

Imagine conjuring up images from mere words—capturing the essence of your imagination visually! This project enables you to do just that by leveraging the pre-trained Stable Diffusion model. Whether you're an artist, a developer, or simply an AI enthusiast, this project provides a gateway to explore and create stunning visuals from textual prompts.

Stable Diffusion is a powerful text-to-image model that uses advanced deep learning techniques to generate high-quality images from textual descriptions. This approach not only showcases the potential of AI in creative fields but also opens new avenues for practical applications.

##  Features

- **Text-to-Image Generation**: Transform your text prompts into visually appealing images using the cutting-edge Stable Diffusion model.
- **Customizable Inference**: Adjust the number of inference steps and guidance scale to fine-tune the image generation process for optimal results.
- **CUDA Support**: Leverage GPU acceleration (if available) for faster and more efficient image creation.

##  Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU (optional but recommended)

### Libraries

Install the required Python libraries using pip:

```bash
pip install torch diffusers
``` 
### Usage  
1. **Clone the Repository**
```
git clone https://github.com/SheldonChetty/PRODIGY_GA_02.git
cd PRODIGY_GA_02
```
2. **Run the Script**
  ```
python generate_image.py
```
3. **Provide Input**  
The script will prompt you to enter a description for the image and optional settings for the number of inference steps and guidance scale.
 
4. **View the Generated Image**  
The generated image will be saved as `generated_image.png` in the current directory and optionally displayed.  

## Code Explanation
### Main Script: generate_image.py

```python
import torch
from diffusers import StableDiffusionPipeline

def main():
    # Load the pre-trained model
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to(device)

    # Prompt the user for input
    prompt = input("Enter the description for the image you want to generate: ")

    # Additional settings for improving quality
    num_inference_steps = int(input("Enter the number of inference steps (default 50): ") or "50")
    guidance_scale = float(input("Enter the guidance scale (default 7.5): ") or "7.5")

    # Generate the image
    print("Generating image... Please wait.")
    with torch.autocast("cuda"):
        image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]

    # Save the image
    image_filename = "generated_image.png"
    image.save(image_filename)
    print(f"Image saved as {image_filename}")

    # Display the image (optional)
    image.show()

if __name__ == "__main__":
    main()
```

### Functions and Implementation Details
- Model Loading: Loads the pre-trained Stable Diffusion model using `StableDiffusionPipeline.from_pretrained(model_id)`.  
- Device Setup: Checks for CUDA availability and sets the device accordingly.  
- User Input: Prompts the user to enter a description and optional settings for image generation.  
- Image Generation: Utilizes the Stable Diffusion pipeline to generate the image based on the user's input.  
- Saving and Displaying: Saves the generated image to a file and optionally displays it.  

## Customization
- Inference Steps: Adjust the number of inference steps to control the quality and computation time of the image generation process. Higher values typically yield better quality but require more computation.  
- Guidance Scale: Adjust the guidance scale to control the adherence of the generated image to the text prompt. Higher values usually make the image more closely match the description.

## Ouput Images 
![Screenshot 2024-07-14 234707](https://github.com/user-attachments/assets/428458af-5366-49d1-a559-c95017445ad0)

![Screenshot 2024-07-14 232829](https://github.com/user-attachments/assets/f973a64f-85ef-4857-932d-819547801a81)



  ## Contributing
Contributions to improve this project are welcome! If you'd like to contribute:

- Fork the repository.
- Create a new branch with a descriptive name (`git checkout -b my-branch-name`).
- Make your changes and commit them (`git commit -am 'Add some feature'`).
- Push to the branch (`git push origin my-branch-name`).
- Create a new Pull Request.
