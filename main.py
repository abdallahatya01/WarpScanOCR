import cv2
import numpy as np
from imutils.perspective import four_point_transform
import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration

os.makedirs("output", exist_ok=True)

from google.colab import files
uploaded = files.upload()

image = cv2.imread("/content/drive/MyDrive/images/003.jpg")
orig = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

document_contour = None
for contour in contours:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4:
        document_contour = approx
        break

if document_contour is None:
    print("لم يتم العثور على مستند")
else:
    warped = four_point_transform(orig, document_contour.reshape(4, 2))
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_warped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_not(binary)
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    white_bg = np.ones_like(warped, dtype=np.uint8) * 255
    final_colored = np.where(mask_colored == 255, warped, white_bg)

    out_path = "output/scanned_colored.jpg"
    cv2.imwrite(out_path, final_colored)

    model_id = "/content/drive/MyDrive/models/qwen2vl-2b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    pil_image = Image.open(out_path)
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": "extract the text in the image"}
        ]}
    ]
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text_prompt, images=pil_image, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=512)
    extracted_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    with open("output/recognized.txt", "w", encoding="utf-8") as f:
        f.write(extracted_text)

    print(extracted_text)

    plt.imshow(cv2.cvtColor(final_colored, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()