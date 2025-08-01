import os
from dotenv import load_dotenv
import gradio as gr
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import tool

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")

# HIPAA Note: No real PHI is stored. This simulates safe interactions.

# Memory
memory = ConversationBufferMemory(memory_key="chat_history")

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

# LLaMA 3 via OpenRouter
llm = ChatOpenAI(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
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

# # Agent conversation handler
# def chat_with_agent(message, history):
#     try:
#         response = agent.run(message)
#         history.append((message, response))
#         return "", history
#     except Exception as e:
#         return "", history + [(message, f"❌ Error: {str(e)}")]
    
def chat_with_agent(message, history):
    if history is None:
        history = []
    response = agent.run(message)
    history.append((message, response))
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
