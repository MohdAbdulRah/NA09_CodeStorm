from google import genai
from google.genai import types
import mimetypes

client = genai.Client(api_key="AIzaSyD5vWRUZG-ksss782D_AP85YsNUHUprrPg")

def detect_disease(filepath):
    """Detect plant + disease from image"""
    mime_type, _ = mimetypes.guess_type(filepath)

    with open(filepath, "rb") as f:
        image_bytes = f.read()

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(text=(
                    "You are a plant pathology expert. "
                    "The image shows a leaf. Identify the crop and the exact disease name. "
                    "Respond strictly in the format: <Plant Name> - <Disease Name>. "
                    "Do not add any explanation or extra text.\n"
                    "Examples:\n"
                    "Tomato leaf with brown circular spots -> Tomato - Early Blight\n"
                    "Potato leaf with irregular dark spots -> Potato - Late Blight\n"
                    "Pepper leaf with small black lesions -> Pepper - Bacterial Spot\n\n"
                    "Now classify the following image in the same format:"
                )),
                types.Part(
                    inline_data=types.Blob(
                        mime_type=mime_type,
                        data=image_bytes
                    )
                )
            ],
        )
    ]

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
    )
    return response.text.strip()

def chatbot(disease_name):
    """Simple chatbot for Q&A about the disease"""
    print(f"\n✅ Detected: {disease_name}")
    print("💬 Ask me about precautions or medicines (type 'exit' to quit)\n")

    history = [
        types.Content(
            role="system",
            parts=[types.Part(text=(
                "You are an agricultural assistant. "
                "Always answer briefly and practically. "
                "If user asks about precautions or treatment, give step-by-step actions and suitable medicines. "
                "Always keep answers under 5 sentences."
            ))]
        ),
        types.Content(
            role="assistant",
            parts=[types.Part(text=f"Disease detected: {disease_name}. I can suggest precautions and treatments.")]
        )
    ]

    while True:
        user_input = input("👨‍🌾 You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting chatbot...")
            break

        history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=history
        )

        reply = response.text.strip()
        print(f"🤖 Bot: {reply}")

        history.append(types.Content(role="assistant", parts=[types.Part(text=reply)]))


if __name__ == "__main__":
    # Step 1: Detect disease
    filepath = "test_images/potato_leaf.jpg"
    disease = detect_disease(filepath)

    # Step 2: Start chatbot
    chatbot(disease)
