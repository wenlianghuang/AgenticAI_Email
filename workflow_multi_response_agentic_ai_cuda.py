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
from agent_tool.meeting_mode import meeting_mode,MeetingModeType
from agent_tool.recording_video import recording_video
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
load_dotenv()

model_name = "microsoft/Phi-4"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device="cuda:0", torch_dtype=torch.float16, low_cpu_mem_usage=True)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

class TransformersLLM(LLM):
    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.tokenizer = tokenizer

    @property
    def _llm_type(self) -> str:
        return "transformers"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=512,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

llm = TransformersLLM(model=model, tokenizer=tokenizer)

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
    #func=lambda _: meeting_mode(),
    func=meeting_mode,
    description="Activates meeting mode in a video conference.",
    args_schema=MeetingModeType
)

recording_vedio_tool = Tool(
    name="Start Video Recording",
    func=lambda filename: recording_video(filename),
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
            name_score = similar(last_message, tool.name.lower()) * 1.5 # Higher weight for name
            # Give lower weight to tool descriptions
            desc_score = similar(last_message, tool.description.lower())
            # Combine scores
            score = max(name_score, desc_score)
            #score = similar(last_message, keyword)
            if score > best_score:
                best_score = score
                best_match = tool

    if best_match and best_score > 0.5:
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
    elif tool.name == "Activate Meeting Mode":
        prompt = (
            f"You are an AI assistant. The user wants to set the meeting mode. "
            f"Generate appropriate values for the following parameters based on the user's input: "
            f"'brightness', 'master_volume', and 'app_volume'. "
            f"Ensure the values are within a valid range and suitable for a meeting environment. "
            f"Only provide the values in the format: brightness=<value>, master_volume=<value>, app_volume=<value>."
        )
        response = llm._call(prompt).strip()
        #print(f"🤖 Meeting mode parameters: {response}")
        # 清理並解析 response
        try:
            # 提取最後一行包含 brightness, master_volume, app_volume 的部分
            last_line = next(
                line for line in reversed(response.splitlines()) 
                if "brightness=" in line and "master_volume=" in line and "app_volume=" in line
            )
            # 移除多餘的換行符號並分割
            response_lines = last_line.replace("\n", "").split(", ")
            # 提取數值
            brightness, master_volume, app_volume = [
                int(value.split('=')[1]) for value in response_lines if '=' in value
            ]
            print(f"brightness: {brightness}, master_volume: {master_volume}, app_volume: {app_volume}")
            tool_result = tool.func(brightness=brightness, master_volume=master_volume, app_volume=app_volume)
        except (ValueError, IndexError) as e:
            # 捕捉錯誤並提供回饋
            tool_result = f"❌ Failed to parse meeting mode parameters: {str(e)}. Please ensure the input format is correct."
            print(tool_result)
        tool_result = tool.func(brightness=brightness,master_volume=master_volume,app_volume=app_volume)
        # 使用語言模型生成回應
        if "✅" in tool_result:  # 成功的情況
            response_prompt = (
                f"You are an AI assistant. The user has successfully started a video recording with the filename '{filename}'. "
                f"Generate a friendly and professional response to confirm this action."
                f"Only provide the response without any additional text."
            )
        else:
            response_prompt = (
                f"You are an AI assistant. The user attempted to start a video recording with the filename '{filename}', but it failed. "
                f"Generate a polite and helpful response explaining the failure and suggesting possible solutions."
                f"Only provide the reponse without any additional text."
            )
        # stream generation response 
        response = []
        def capture_output(subword):
            response.append(subword)
            print(subword, end="", flush=True)  # 即時輸出
            return False
        llm.pipeline.generate(response_prompt, streamer=capture_output)
        #tool_result = llm._call(response_prompt).strip()
        tool_result = "".join(response).strip()
    elif tool.name == "Get Stock Report":
        print("🤖 Please provide the stock symbol for the report.")
        symbol = input("Stock Symbol: ")
        print("Choose an option:")
        print("1. Quarterly Revenue and YOY Comparison")
        print("2. Daily Candle Chart")
        choice = input("Enter your choice (1 or 2): ").strip()
        tool_result = tool.func(symbol, choice)
    elif tool.name == "Start Video Recording":
        print("🤖 Please provide the file name to save the video recording.")
        filename = input("File Name: ")
        prompt = (
            f"You are an AI assistant. The user wants to save a video recording. "
            f"Generate a valid filename based on the user's input: '{filename}'. "
            f"Ensure the filename does not contain spaces by replacing them with underscores."
            f"Only provide the filename without any additional text."
        )
        filename = llm._call(prompt).strip().replace(" ", "_")
        tool_result = tool.func(filename)
        # 使用語言模型生成回應
        if "✅" in tool_result:  # 成功的情況
            response_prompt = (
                f"You are an AI assistant. The user has successfully started a video recording with the filename '{filename}'. "
                f"Generate a friendly and professional response to confirm this action."
                f"Only provide the response without any additional text."
            )
        else:
            response_prompt = (
                f"You are an AI assistant. The user attempted to start a video recording with the filename '{filename}', but it failed. "
                f"Generate a polite and helpful response explaining the failure and suggesting possible solutions."
                f"Only provide the reponse without any additional text."
            )
        # stream generation response 
        response = []
        def capture_output(subword):
            response.append(subword)
            print(subword, end="", flush=True)  # 即時輸出
            return False
        llm.pipeline.generate(response_prompt, streamer=capture_output)
        #tool_result = llm._call(response_prompt).strip()
        tool_result = "".join(response).strip()
    #else:
    #    tool_result = tool.func("")
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