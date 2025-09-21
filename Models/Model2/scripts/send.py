from google import genai
from google.genai import types
import mimetypes

client = genai.Client(api_key="AIzaSyD5vWRUZG-ksss782D_AP85YsNUHUprrPg")
filepath = "test_images/potato_leaf.jpg"
mime_type, _ = mimetypes.guess_type(filepath)
# Agar tumhare paas local image hai:
with open(filepath, "rb") as f:
    image_bytes = f.read()

contents = [
    types.Content(
        role="user",
        parts=[
            types.Part( text=(
        "You are a plant pathology expert. "
        "The image shows a leaf. Identify the crop and the exact disease name. "
        "Respond strictly in the format: <Plant Name> - <Disease Name>. "
        "Do not add any explanation or extra text."
        "Examples:\n"
        "Tomato leaf with brown circular spots -> Tomato - Early Blight\n"
        "Potato leaf with irregular dark spots -> Potato - Late Blight\n"
        "Pepper leaf with small black lesions -> Pepper - Bacterial Spot\n\n"
        "Now classify the following image in the same format:"
    )),
            types.Part(
                inline_data=types.Blob(
                    mime_type="image/jpeg",
                    data=image_bytes
                )
            )
        ],
    )
]

response = client.models.generate_content(
    model="gemini-2.5-flash",   # ya jo model tum use karna chahte ho
    contents=contents,
)

print(response.text)

