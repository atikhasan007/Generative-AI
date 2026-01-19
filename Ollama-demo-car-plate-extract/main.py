import os
import base64 
from io import BytesIO
import argparse
from PIL import Image
from langchain_ollama import OllamaLLM


llm = OllamaLLM(model="qwen2.5vl:3b")


def convert_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG") 
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str



prompt = (
            "You are a visual analysis assistant. From the given car image, extract the following "
            "information and return it in strict JSON format:\n"
            "{\n"
            "  \"license_plate\": \"\",\n"
            "  \"brand\": \"\",\n"
            "  \"model\": \"\",\n"
            "  \"variant\": \"\"\n"
            "  \"state_name\": \"\"\n"
            "  \"car_color\": \"\"\n"
            "}\n"
            "If any field is missing or not visible, leave it as an empty string. "
            "Only return the JSON. No explanation."
        )


def main():
    parser = argparse.ArgumentParser(description="Extract car info from an image using Qwen2.5-VL")
    parser.add_argument("image_path", type=str, help="Path to the car image file")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: File not found -> {args.image_path}")
        return
    
    try:
        file_path = args.image_path
        pil_image = Image.open(file_path)
        image_b64 = convert_to_base64(pil_image)

        llm_with_image_context = llm.bind(images=[image_b64])
        output = llm_with_image_context.invoke(prompt)
        print("Extracted Vehicle Info:\n", output)


    except Exception as e:
        print(f"Error processing image: {e}")
        

if __name__ == "__main__":
    main()

