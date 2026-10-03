import torch
import traceback
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler


def main():
    small_model = "runwayml/stable-diffusion-v1-5"
    print("Welcome to AI Image Generator")
    print("This is free and unlimited")
    print("Please wait as we are loading the application")

    image_number = 0
    run = True

    while run:
        print("\n--- MENU ---")
        print("Press 1 to create an image")
        print("Press 2 to edit an image created")
        print("Press 3 to quit")

        try:
            Option = int(input("Which option: "))
        except ValueError:
            print("Please enter a valid number (1, 2, or 3).")
            continue

        if Option == 1:
            print("Loading Text-to-Image model...")

            pipe = StableDiffusionPipeline.from_pretrained(small_model, torch_dtype=torch.float32,
                                                           low_cpu_mem_usage=True, safety_checker=None,
                                                           requires_safety_checker=False,
                                                           local_files_only=False)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

            pipe.enable_attention_slicing()
            pipe = pipe.to("cpu")

            initial_prompt = input("Enter the INITIAL prompt to create the image: ")
            print("Generating base image (this may take a few minutes on CPU)...")

            initial_result = pipe(
                prompt=initial_prompt, num_inference_steps=20,
                guidance_scale=7.5, height=512, width=512)
            base_image = initial_result.images[0]

            filename = f"base_image_{image_number}.png"
            base_image.save(filename)
            print(f"Base image saved as '{filename}'.")
            image_number += 1

            del pipe

        elif Option == 2:
            try:
                target_index = image_number - 1 if image_number > 0 else 0
                base_image = Image.open(f"base_image_{target_index}.png").convert("RGB")
            except FileNotFoundError:
                print("No base image found! Please create an image using Option 1 first.")
                continue

            print("Loading Image-to-Image editing model...")

            pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                small_model, torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                safety_checker=None, requires_safety_checker=False,
                local_files_only=False)
            pipe_img2img.scheduler = DPMSolverMultistepScheduler.from_config(pipe_img2img.scheduler.config)

            pipe_img2img.enable_attention_slicing()
            pipe_img2img = pipe_img2img.to("cpu")

            edit_prompt = input("Enter the NEW prompt to edit/modify the image: ")
            print("Applying edits to the image...")

            edited_result = pipe_img2img(
                prompt=edit_prompt, image=base_image, strength=0.75,
                num_inference_steps=25, guidance_scale=7.5)
            final_image = edited_result.images[0]

            final_filename = f"edited_image_{target_index}.png"
            final_image.save(final_filename)
            print(f"Image edited and saved successfully as '{final_filename}'!")

            del pipe_img2img

        elif Option == 3:
            print("Exiting application. Goodbye!")
            run = False
        else:
            print("Invalid option. Please choose 1, 2, or 3.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[CRITICAL ERROR DETECTED]")
        traceback.print_exc()
    finally:
        input("\nPress Enter to exit the application...")
