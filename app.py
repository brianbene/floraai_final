import streamlit as st
from openai import OpenAI
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="FloraAI",
    page_icon="🌿",
    layout="wide"
)

# --- Custom CSS for Gardening Theme ---
st.markdown("""
    <style>
    .stApp {
        background-color: #fdfcf0;
    }
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #2e8b57; 
        text-align: center;
        font-weight: bold;
    }
    .sub-header {
        color: #556b2f;
        text-align: center;
        font-style: italic;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .chat-message.user {
        background-color: #e0f2f1;
        border-left: 5px solid #009688;
    }
    .chat-message.bot {
        background-color: #f1f8e9;
        border-left: 5px solid #8bc34a;
    }
    .prediction-box {
        background-color: #dcedc8;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        border: 2px solid #8bc34a;
    }
    </style>
""", unsafe_allow_html=True)

# --- Global Constants ---
CLASS_NAMES = [
    "Beets", "Broccoli", "Cabbage", "Carrots", "Cauliflower", "Celery", "Corn",
    "Cucumbers", "Garlic", "Green_Beans", "Kale", "Lettuce", "Onions", "Peas",
    "Peppers", "Potatoes", "Radishes", "Spinach", "Summer_Squash", "Tomatoes"
]
MODEL_PATH = "models/resnet50_vegetables.pth"

# --- PERPLEXITY CONFIGURATION ---
PPLX_API_KEY = ""
BASE_URL = "https://api.perplexity.ai"

# Initialize Client
client = OpenAI(api_key=PPLX_API_KEY, base_url=BASE_URL)

# --- COMPREHENSIVE SYSTEM PROMPT ---
FLORA_SYSTEM_PROMPT = """
You are Flora, the ancient Goddess of Fertility and the world's most knowledgeable, scientific gardening assistant. Your mission is to guide users to cultivate thriving gardens tailored specifically to their local environment.

**YOUR PRIME DIRECTIVES:**

1.  **CONTEXT IS KING:** You have access to online data. You MUST use the user's provided **geographical location** to look up their current weather and climate zone if needed. 
    * *CRITICAL:* If the user has NOT stated their location yet, you must politely but firmly ask for it before providing specific planting advice.

2.  **PLANT ANALYSIS MODE:** When the user presents an identified plant (e.g., "I have identified a Tomato plant"), you must generate a comprehensive, location-specific cultivation guide.

3.  **RESPONSE STRUCTURE:** When discussing a specific plant, you must use the following format:
    * **🌿 The Goddess's Verdict:** A brief, mystical acknowledgment of the plant.
    * **🌍 Local Viability (Pros & Cons):** specifically for the user's stated location.
    * **🌱 Planting Strategy:** precise timing (months), spacing, seed depth, and sun exposure requirements.
    * **💧 Hydro-Logic:** Detailed watering needs.
    * **🛡️ Guardian's Care (Treatment):** Specific pests/diseases common to this plant *in their region* and organic methods to treat them.

4.  **TONE:** Your voice is wise, nurturing, and earthy, yet precise and grounded in agricultural science.
"""


# --- Enforce valid user-assistant alternation in message sequence ---
def clean_message_sequence(messages):
    """
    Keeps system messages at front, then enforces user/assistant alternating pattern.
    Removes any consecutive user or consecutive assistant messages beyond the first.
    Also ensures first conversational message after system is user role.
    """
    system_msgs = [m for m in messages if m["role"] == "system"]
    conv_msgs = [m for m in messages if m["role"] != "system"]

    cleaned = []
    last_role = None
    for msg in conv_msgs:
        # Skip if same role as previous (enforces alternation)
        if last_role == msg["role"]:
            continue
        cleaned.append(msg)
        last_role = msg["role"]

    # CRITICAL: After system messages, first message MUST be user role
    # If first conversational message is assistant, skip it or convert
    if cleaned and cleaned[0]["role"] == "assistant":
        # Option 1: Skip the initial assistant greeting for API
        cleaned = cleaned[1:]
        # Option 2: Prepend a dummy user message (less ideal)

    return system_msgs + cleaned


# --- Helper to call Perplexity ---
def get_perplexity_response(messages):
    """
    Calls the Perplexity API with the current chat history.
    Enforces valid message role pattern before sending.
    """
    full_history = clean_message_sequence([{"role": "system", "content": FLORA_SYSTEM_PROMPT}] + messages)

    print("DEBUG: Sending messages sequence:")
    for i, m in enumerate(full_history):
        print(f"{i}: Role={m['role']}, Content={m['content'][:50]!r}")

    models_to_try = [
        "llama-3.1-sonar-small-128k-online",
        "llama-3.1-sonar-large-128k-online",
        "sonar-pro",
        "sonar"
    ]

    last_error = None
    for model_name in models_to_try:
        try:
            stream = client.chat.completions.create(
                model=model_name,
                messages=full_history,
                stream=True
            )
            return stream
        except Exception as e:
            last_error = e
            print(f"Model {model_name} failed: {e}")
            continue

    raise last_error


# --- Model Loading Logic ---
@st.cache_resource
def load_trained_model():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))

        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model = model.to(device)
            model.eval()
            return model, device
        else:
            return None, None

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def predict_image(image, model, device):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
    return CLASS_NAMES[preds.item()]


# --- Sidebar: Configuration & Tools ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/potted-plant.png", width=100)
    st.title("FloraAI Controls")

    st.markdown("---")
    st.markdown("### 📷 Garden Eye")

    input_method = st.radio("Choose input method:", ["Upload Photo", "Live Camera"])

    img_file_buffer = None

    if input_method == "Upload Photo":
        img_file_buffer = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    else:
        img_file_buffer = st.camera_input("Take a picture")

    # Load Model
    model, device = load_trained_model()
    if not model:
        st.warning(f"⚠️ Model not found at `{MODEL_PATH}`.")

# --- Main Content ---
st.markdown("<h1 class='main-header'>🌿 FloraAI</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='sub-header'>Goddess of Fertility & Intelligent Gardening</h3>", unsafe_allow_html=True)

# --- Chat State Management ---
if "messages" not in st.session_state:
    # Start with assistant greeting for DISPLAY ONLY
    # It will be filtered out when sending to API
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Greetings, mortal. I am Flora. Before we begin our cultivation, tell me: **Where is your garden located, and what are your goals for this harvest?**"
        }
    ]

# --- Image Analysis Logic ---
if "last_processed_file" not in st.session_state:
    st.session_state.last_processed_file = None

if img_file_buffer:
    image = Image.open(img_file_buffer).convert('RGB')

    with st.sidebar:
        st.image(image, caption='Analyzed Plant', use_column_width=True)

    if model and (st.session_state.last_processed_file != img_file_buffer):
        st.session_state.last_processed_file = img_file_buffer

        # 1. Run Inference
        predicted_class = predict_image(image, model, device)

        # 2. Visual Feedback
        st.markdown(f"""
        <div class="prediction-box">
            <h3>👁️ Flora's Vision</h3>
            <p>I have identified this plant as: <strong>{predicted_class}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # 3. Construct Prompt
        analysis_prompt = (
            f"Flora, I have just identified a **{predicted_class}** plant in my garden using your vision. "
            f"Please provide the 'The Goddess's Verdict', 'Local Viability', 'Planting Strategy', "
            f"'Hydro-Logic', and 'Guardian's Care' for this {predicted_class} based on the location I previously mentioned."
        )

        # 4. Add visual marker for display
        st.session_state.messages.append({"role": "user", "content": f"*[Analyzed image: {predicted_class}]*"})

        # 5. Build API messages - replace visual marker with actual prompt
        api_messages = st.session_state.messages[:-1] + [{"role": "user", "content": analysis_prompt}]

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                stream = get_perplexity_response(api_messages)

                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.rerun()

            except Exception as e:
                st.error(f"Flora is unreachable: {e}")

# --- Display Chat Interface ---
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Chat Input ---
if prompt := st.chat_input("Consult the Goddess..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            stream = get_perplexity_response(st.session_state.messages)

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"The spirits are silent: {e}")
