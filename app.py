import os
from dotenv import load_dotenv
import gradio as gr
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage # Import these
from langchain.agents import tool

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")

# HIPAA Note: No real PHI is stored. This simulates safe interactions.

# Memory
# For conversational agents, the memory object is often directly passed to the agent,
# and it handles updating its own state.
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) # Add return_messages=True

# Tool 1: Book appointment
@tool
def book_appointment(name: str, symptoms: str, time: str) -> str:
    """
    Book an appointment given patient name, symptoms, and appointment time.
    """
    return f"✅ Appointment booked for {name} at {time} for symptoms: {symptoms}."

# Tool 2: Check availability
@tool
def check_availability(symptoms: str) -> str:
    """
    Check available appointment slots for given symptoms.
    """
    return "🕒 Available slots: 10:00 AM, 2:00 PM, 4:30 PM"

# Tools list
tools = [
    Tool(name="BookAppointment", func=book_appointment, description="Book appointment using name, symptoms, and time."),
    Tool(name="CheckAvailability", func=check_availability, description="Check available times for a given symptom.")
]

# LLaMA 3.1 via OpenRouter
llm = ChatOpenAI(
    model_name="meta-llama/llama-3.1-8b-instruct", # Changed to the Llama 3.1 ID
    temperature=0.3,
    streaming=False
)

# Create agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

def chat_with_agent(message, history):
    # Gradio's history is a list of lists: [[user_msg, bot_msg], ...]
    # Langchain's memory expects a list of BaseMessage objects (HumanMessage, AIMessage)
    # We need to convert Gradio's history format to Langchain's message format
    
    # Initialize Langchain's chat history based on Gradio's history
    # The agent's memory will manage the full history internally,
    # but we need to ensure it's properly initialized with the current conversation.
    
    # Clear agent's memory and re-populate it with the current Gradio history
    memory.clear() 
    for user_msg, ai_msg in history:
        memory.save_context({"input": user_msg}, {"output": ai_msg})

    try:
        # Pass the new user message to the agent
        response = agent.run(input=message) # Use input= for explicit naming
        # Update Gradio's history with the new user message and agent's response
        history.append((message, response))
        return "", history
    except Exception as e:
        # If an error occurs, append the user message and an error message to history
        history.append((message, f"❌ Error: {str(e)}"))
        return "", history


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 LLaMA 3 - HIPAA-Friendly Appointment Booking Agent")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your question...")
    clear = gr.Button("Clear Chat")

    msg.submit(chat_with_agent, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: [], None, chatbot)

demo.launch()
