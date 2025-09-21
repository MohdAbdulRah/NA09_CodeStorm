# from google import genai
# from google.genai import types
# import mimetypes

# client = genai.Client(api_key="AIzaSyD5vWRUZG-ksss782D_AP85YsNUHUprrPg")

# def chatbot(disease_name):
#     """Simple chatbot for Q&A about the disease"""
#     print(f"\n✅ Detected: {disease_name}")
#     print("💬 Ask me about precautions or medicines (type 'exit' to quit)\n")

#     # Initialize conversation history
#     history = [
#         types.Content(
#             role="model",
#             parts=[types.Part(text=(
#                 "You are an agricultural assistant. "
#                 "Always answer briefly and practically. "
#                 "If user asks about precautions or treatment, give step-by-step actions and suitable medicines. "
#                 "Always keep answers under 5 sentences."
#             ))]
#         ),
#         types.Content(
#             role="model",
#             parts=[types.Part(text=f"Disease detected: {disease_name}. I can suggest precautions and treatments.")]
#         )
#     ]

#     while True:
#         user_input = input("👨‍🌾 You: ")
#         if user_input.lower() in ["exit", "quit"]:
#             print("👋 Exiting chatbot...")
#             break

#         # Append user message
#         history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))

#         # Generate model response
#         response = client.models.generate_content(
#             model="gemini-2.5-flash",
#             contents=history
#         )

#         reply = response.text.strip()
#         print(f"🤖 Bot:\n {reply}")

#         # Append model response to history
#         history.append(types.Content(role="model", parts=[types.Part(text=reply)]))


# if __name__ == "__main__":
   
#     disease = input("🌱🦠 Disease Name : ")
#     # Step 2: Start chatbot
#     chatbot(disease)

from google import genai
from google.genai import types
import mimetypes
from colorama import Fore, Style, init

# Initialize colorama for colored console text
init(autoreset=True)

client = genai.Client(api_key="AIzaSyD5vWRUZG-ksss782D_AP85YsNUHUprrPg")

# Function to convert markdown-style **bold** to terminal bold
def format_bold(text):
    import re
    # Replace **word** with ANSI bold
    return re.sub(r"\*\*(.*?)\*\*", "\033[1m\\1\033[0m", text)

def chatbot(disease_name):
    """Simple chatbot for Q&A about the disease"""
    print(f"\n✅ Detected: {Fore.GREEN}{Style.BRIGHT}{disease_name}{Style.RESET_ALL}")
    print("💬 Ask me about precautions or medicines (type 'exit' to quit)\n")

    # Initialize conversation history
    history = [
        types.Content(
            role="model",
            parts=[types.Part(text=(
                "You are an agricultural assistant. "
                "Always answer briefly and practically. "
                "If user asks about precautions or treatment, give step-by-step actions and suitable medicines. "
                "Always keep answers under 5 sentences. Use bold for key terms."
            ))]
        ),
        types.Content(
            role="model",
            parts=[types.Part(text=f"Disease detected: {disease_name}. I can suggest precautions and treatments.")]
        )
    ]

    while True:
        user_input = input(f"{Fore.YELLOW}👨‍🌾 You:{Style.RESET_ALL} ")
        if user_input.lower() in ["exit", "quit"]:
            print(f"{Fore.CYAN}👋 Exiting chatbot...{Style.RESET_ALL}")
            break

        # Append user message
        history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))

        # Generate model response
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=history
        )

        reply = response.text.strip()
        reply = format_bold(reply)  # Convert markdown bold to terminal bold
        print(f"{Fore.BLUE}🤖 Bot:{Style.RESET_ALL}\n {Style.BRIGHT}{reply}{Style.RESET_ALL}\n")

        # Append model response to history
        history.append(types.Content(role="model", parts=[types.Part(text=reply)]))


if __name__ == "__main__":
    disease = input("🌱🦠 Disease Name : ")
    chatbot(disease)


