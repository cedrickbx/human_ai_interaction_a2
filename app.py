import streamlit as st
import os
from groq import Groq, BadRequestError, RateLimitError
from dotenv import load_dotenv
import datetime

# Load environment variables
load_dotenv()

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
client = None
if api_key:
    client = Groq(api_key=api_key)

# Page configuration
st.set_page_config(
    page_title="Trip Planning Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics (Safe for Light/Dark mode)
st.markdown("""
<style>
    /* Chat Messages */
    .stChatMessage {
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    /* Input Fields */
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
        border-radius: 8px;
    }
    
    /* Info Box */
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- Tools ---
# Web search disabled by user request

# Sidebar for User Context and Settings
with st.sidebar:
    st.title("🌍 Trip Settings")
    st.markdown("Customize your journey details below.")
    
    st.markdown("### 👤 Context & Preferences")
    
    # User Inputs
    interests = st.text_area("What are your interests?", placeholder="e.g., History, Food, Nature, Adventure", height=100, key="interests_input")
    
    # Budget Range
    budget_range = st.slider(
        "Budget Range ($)",
        min_value=0,
        max_value=10000,
        value=(1000, 3000),
        step=100,
        help="Select your estimated budget range for the trip.",
        key="budget_slider"
    )
    
    # Date Range
    today = datetime.date.today()
    tomorrow = today + datetime.timedelta(days=1)
    date_range = st.date_input(
        "Trip Dates",
        value=(today, tomorrow),
        min_value=today,
        help="Select the start and end date of your trip.",
        key="date_picker"
    )
    
    # Calculate duration if range is selected
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date = date_range[0]
        end_date = date_range[1]
        duration = (end_date - start_date).days + 1
        st.caption(f"Duration: {duration} days")
    else:
        start_date = today
        end_date = None
        duration = "Unknown"
        st.warning("Please select an end date.")
    
    st.markdown("### 🕌 Social Norms & Preferences")
    halal_option = st.checkbox("Strictly Halal (Muslim Friendly)", key="halal_check")
    veg_option = st.checkbox("Vegetarian / Vegan Friendly", key="veg_check")
    alcohol_free_option = st.checkbox("Alcohol-Free Environment", key="alcohol_free_check")
    family_option = st.checkbox("Family / Kid Friendly", key="family_check")
    accessibility_option = st.checkbox("Wheelchair Accessible", key="access_check")
    
    st.markdown("---")
    
    # Effective Dismissal
    st.markdown("### 🛑 Controls")
    if st.button("Reset Conversation", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Chat Interface
st.title("✈️ Trip Planning AI Assistant")
st.markdown("#### Your personal travel companion powered by Groq")

with st.expander("ℹ️ System Capabilities", expanded=True):
    st.markdown(
        """
        **I can:**
        - Plan trips based on interests & budget.
        - Suggest activities & restaurants.
        - Respect social norms (Halal, Veg, Family-friendly, etc.).
        - **Search the web**.
        
        **I cannot:**
        - Book flights or hotels.
        """
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    if message["role"] in ["user", "assistant"]:
        role_icon = "👤" if message["role"] == "user" else "🤖"
        content = message.get("content")
        if content:
            with st.chat_message(message["role"], avatar=role_icon):
                st.markdown(content)

# Chat Input
if prompt := st.chat_input("Tell me about your trip ideas..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Check if API key is present
    if not client:
        st.error("⚠️ Groq API Key not found. Please set it in the `.env` file.")
        st.stop()

    # Construct System Prompt with Context
    system_prompt = f"""
    You are a helpful and knowledgeable Trip Planning Assistant that can help with tourism-related queries. 
    
    **User Context:**
    - Interests: {interests}
    - Budget Range: ${budget_range[0]} - ${budget_range[1]}
    - Trip Dates: {start_date} to {end_date if end_date else 'Unspecified'}
    - Duration: {duration} days
    
    **Social Norms & Preferences:**
    {'- User requires strictly Halal food and places. Ensure all recommendations are Halal-certified or Muslim-friendly.' if halal_option else ''}
    {'- User prefers Vegetarian or Vegan friendly options.' if veg_option else ''}
    {'- User prefers an Alcohol-Free environment.' if alcohol_free_option else ''}
    {'- User is travelling with family/kids. Suggest family-friendly activities and amenities.' if family_option else ''}
    {'- User requires Wheelchair Accessible locations.' if accessibility_option else ''}
    {'No specific social norm restrictions provided.' if not any([halal_option, veg_option, alcohol_free_option, family_option, accessibility_option]) else ''}
    
    **Your Goal:**
    - Help the user plan a trip that matches their preferences.
    - Provide contextually relevant information.
    - If the user's request is vague, ask clarifying questions to scope the services (e.g., "Where would you like to go?", "What kind of food do you like?").
    - Remember recent interactions in the conversation.
    - Be polite, professional, and enthusiastic.
    - Try to keep response short and concise.
    
    **Limitations:**
    - You cannot book anything.
    - **IMPORTANT:** If you are not sure or do not know the facts, then just state so. Do not hallucinate or make up information.
    """

    # Truncate chat history to ensure we don't hit rate limits (keep last 10 messages)
    # This is a simple heuristic; for production, counting tokens is better.
    max_history = 10
    recent_messages = st.session_state.messages[-max_history:] if len(st.session_state.messages) > max_history else st.session_state.messages
    
    # Prepare messages for API call
    messages = [{"role": "system", "content": system_prompt}] + recent_messages

    # Generate response
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # First API Call (Decision to use tool or not)
            model_name = os.getenv("GROQ_MODEL")
            if not model_name:
                st.error("⚠️ GROQ_MODEL not set in .env file.")
                st.stop()

            response = client.chat.completions.create(
                messages=messages,
                model=model_name,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
            # Add final response to session state
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except BadRequestError as e:
            if "tool_use_failed" in str(e):
                st.error("⚠️ The AI model failed to use the search tool correctly. This often happens with smaller models.")
                st.info("💡 **Tip:** Try switching to a larger model like `llama-3.1-70b-versatile` in your `.env` file.")
                st.code(f"Error details: {e}")
            else:
                st.error(f"API Error: {e}")
        except RateLimitError as e:
            st.error("⚠️ Rate Limit Exceeded.")
            st.info("You have hit the usage limit for this model (Tokens Per Minute).")
            st.markdown("""
            **Suggestions:**
            1. **Wait a moment** and try again.
            2. **Reset the conversation** using the button in the sidebar to clear history.
            3. **Switch models** in your `.env` file (e.g., try `llama-3.1-70b-versatile` or `llama3-70b-8192`).
            """)
            st.code(f"Error details: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
