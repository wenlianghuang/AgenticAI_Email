from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain.schema import SystemMessage
from typing import TypedDict, Annotated, List
from difflib import SequenceMatcher
from langchain.llms.base import LLM
from dotenv import load_dotenv
import openvino_genai as ov_genai
from langchain.memory import ConversationBufferMemory
from typing import List, Optional
from langchain.tools import Tool, StructuredTool
import os
from agent_tool.weather import get_weather_by_city
from agent_tool.sendEmail import send_email, SendEmailInput
from agent_tool.searchWikipedia import search_wikipedia
from agent_tool.stockReport import get_financial_tool, StockReport
from agent_tool.meeting_mode import meeting_mode
from agent_tool.recording_vedio import recording_vedio
load_dotenv()

model_path = 'Phi-35_mini_instruct_refined'
pipe = ov_genai.LLMPipeline(model_path, device='NPU')
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

class OpenVINO_LLM(LLM):
    def __init__(self, pipeline: ov_genai.LLMPipeline, **kwargs):
        super().__init__(**kwargs)
        self._pipeline = pipeline

    @property
    def pipeline(self) -> ov_genai.LLMPipeline:
        return self._pipeline

    @property
    def _llm_type(self) -> str:
        return "openvino_genai"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = []
        def capture_output(subword):
            response.append(subword)
            return False
        self.pipeline.generate(prompt, streamer=capture_output)
        return "".join(response)

llm = OpenVINO_LLM(pipeline=pipe)

# Define tools
def open_calculator() -> str:
    try:
        os.startfile("calc.exe")
        return "✅ I have opened the Windows Calculator successfully."
    except Exception as e:
        return f"❌ Failed to open calculator: {str(e)}"

def open_paint() -> str:
    try:
        os.startfile("mspaint.exe")
        return "✅ I have opened Microsoft Paint."
    except Exception as e:
        return f"❌ Failed to open paint: {str(e)}"

calculator_tool = Tool(
    name="Open Calculator",
    func=lambda _: open_calculator(),
    description="Opens the Windows Calculator."
)

paint_tool = Tool(
    name="Open Paint",
    func=lambda _: open_paint(),
    description="Opens Microsoft Paint."
)

weather_tool = Tool(
    name="Get Weather",
    func=lambda city: get_weather_by_city(city),
    description="Returns weather info given a city name."
)

send_email_tool = StructuredTool(
    name="SendEmail",
    func=send_email,
    description="Sends an email. Input must include recipient, subject, and body.",
    args_schema=SendEmailInput
)

wikipedia_tool = Tool(
    name="Search Wikipedia",
    func=lambda query: search_wikipedia(query),
    description="Returns a Wikipedia summary given a topic.",
    args_schema=StockReport
)

stock_report_tool = Tool(
    name="Get Stock Report",
    func=get_financial_tool,
    description="Fetches and plots the quarterly revenue data for a given stock symbol."
)

meeting_mode_tool = Tool(
    name="Activate Meeting Mode", # 強化工具名稱的唯一性
    func=lambda _: meeting_mode(),
    description="Activates meeting mode in a video conference."
)

recording_vedio_tool = Tool(
    name="Start Video Recording",
    func=lambda _: recording_vedio(),
    #description="Starts video recording using the camera and microphone."
    description="Starts to record video and save it in the local disk."
)
def simple_chatbot_response(user_input: str) -> str:
    prompt = (
        f"You are a helpful chatbot. Answer the user's question concisely and informatively. "
        f"Provide your response step by step if necessary.\n\n"
        f"User: {user_input}\n"
        f"Chatbot:"
    )
    response = []

    def capture_output(subword):
        response.append(subword)
        print(subword, end="", flush=True)
        return False

    llm.pipeline.generate(prompt, streamer=capture_output)
    return "".join(response).strip()

chatbot_tool = Tool(
    name="Chatbot",
    func=lambda question: simple_chatbot_response(question),
    description="Answers general questions or engages in casual conversation."
)

tools = [calculator_tool, paint_tool, weather_tool, send_email_tool, wikipedia_tool, stock_report_tool, meeting_mode_tool,recording_vedio_tool ,chatbot_tool]

class AgentState(TypedDict):
    messages: Annotated[List, add_messages]

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def call_agent(data: AgentState):
    prompt = (
        "You are an AI assistant with reasoning capabilities. "
        "When the user asks for an action, respond with the exact tool name to execute the action. "
        "If the user is asking a general question or engaging in casual conversation, respond with 'Chatbot'. "
        "Do not provide explanations or step-by-step reasoning.\n\n"
        "Available tools:\n"
        + "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
        + "\n\n"
        + "User input:\n"
        + "".join([m.content for m in data["messages"] if hasattr(m, 'content')])
    )
    response = llm._call(prompt).strip()
    last_message = data["messages"][-1].content.lower()

    # Exact match for tool names
    for tool in tools:
        if tool.name.lower() in response.lower():
            return execute_tool(tool, last_message,data)

    # Fuzzy matching for tool names or descriptions
    best_match = None
    best_score = 0.0
    for tool in tools:
        for keyword in [tool.name.lower(), tool.description.lower()]:
            # Give higher weight to tool names
            name_score = similar(last_message, tool.name.lower()) * 1.5
            desc_score = similar(last_message, tool.description.lower())
            score = max(name_score, desc_score)
            #score = similar(last_message, keyword)
            if score > best_score:
                best_score = score
                best_match = tool

    if best_match and best_score > 0.8:
        return execute_tool(best_match, last_message,data)

    # Default to chatbot response
    chatbot_response = simple_chatbot_response(last_message)
    return {"messages": data["messages"] + [SystemMessage(content=chatbot_response), SystemMessage(content="[FINISH RESPONSE]")]}

def execute_tool(tool, last_message,data):
    if tool.name == "Search Wikipedia":
        print("🤖 Please provide a topic to search on Wikipedia.")
        topic = input("Topic: ")
        tool_result = tool.func(topic)
    elif tool.name == "Get Weather":
        print("🤖 Please provide the city name for weather information.")
        city = input("City: ")
        tool_result = tool.func(city)
    elif tool.name == "SendEmail":
        print("🤖 Please provide the recipient, subject, and the concept of day.")
        recipient = input("Recipient: ")
        subject = input("Subject: ")
        prompt = (
            f"You are an AI assistant. The user wants to send an email with the subject '{subject}'. "
            f"Generate a detailed and professional email body based on the user's input: '{last_message}'."
        )
        body = llm._call(prompt).strip()
        tool_result = tool.func(recipient, subject, body)
    elif tool.name == "Get Stock Report":
        print("🤖 Please provide the stock symbol for the report.")
        symbol = input("Stock Symbol: ")
        print("Choose an option:")
        print("1. Quarterly Revenue and YOY Comparison")
        print("2. Daily Candle Chart")
        choice = input("Enter your choice (1 or 2): ").strip()
        tool_result = tool.func(symbol, choice)
    elif tool.name == "Meeting Mode":
        tool_result = tool.func("")
    elif tool.name == "Start Video Recording":
        tool_result = tool.func("")
    else:
        tool_result = tool.func("")
    return {"messages": data["messages"] + [SystemMessage(content=tool_result), SystemMessage(content="[TOOL_EXECUTED]")]}

def call_tools_with_feedback(data: AgentState):
    if any("[TOOL_EXECUTED]" in msg.content for msg in data["messages"]) or any("[FINISH RESPONSE]" in msg.content for msg in data["messages"]):
        return data
    last_message = data["messages"][-1].content.lower()
    best_match = None
    best_score = 0.0

    for tool in tools:
        for keyword in [tool.name.lower(), tool.description.lower()]:
            score = similar(last_message, keyword)
            if score > best_score:
                best_score = score
                best_match = tool

    if best_match and best_score > 0.8:
        return execute_tool(best_match, last_message)

    return {"messages": data["messages"] + [SystemMessage(content="No tool matched in call_tools_with_feedback.")]}

def should_end(data: AgentState) -> bool:
    last_msg = data["messages"][-1].content
    return any(stop_word in last_msg for stop_word in ["✅", "❌"])

def build_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_agent)
    workflow.add_node("tool_call", call_tools_with_feedback)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", "tool_call")
    workflow.add_conditional_edges(
        "tool_call", 
        should_end,
        "end",
        "agent",
    )
    return workflow.compile()