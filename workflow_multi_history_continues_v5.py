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
from agent_tool.recording_video import recording_video
from agent_tool.set_brightness import set_brightness
from agent_tool.set_volume import set_master_volume
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
    #args_schema=StockReport
)

stock_report_tool = Tool(
    name="Get Stock Report",
    func=get_financial_tool,
    description="Fetches and plots the quarterly revenue data for a given stock symbol.",
    args_schema=StockReport
)
set_brightness_tool = Tool(
    name="Set Brightness",
    func=lambda brightness: set_brightness(brightness),
    description="Set the brightness of the screen to a specified value (0-100)."
)

set_volume_tool = Tool(
    name="Set Volume",
    func=lambda master_volume: set_master_volume(master_volume),
    description="Sets the master volume to specified number."
)


recording_vedio_tool = Tool(
    name="Start Video Recording",
    func=lambda filename: recording_video(filename),
    #description="Starts video recording using the camera and microphone."
    description="Starts to record video and save it in the local disk."
)
def simple_chatbot_response(user_input: str) -> str:
    '''
    prompt = (
        f"You are a helpful chatbot. Answer the user's question concisely and directly. "
        f"Provide only the necessary information in one or two sentences. "
        f"Do not include any additional explanations, commentary, self-reflection, or repeated statements. "
        f"Do not generate hypothetical scenarios or additional questions.\n\n"
        f"User: {user_input}\n"
        f"Chatbot:"
    )
    '''
    response = []

    def capture_output(subword):
        response.append(subword)
        print(subword, end="", flush=True)
        return False
    llm.pipeline.generate(user_input, streamer=capture_output)
    full_response = "".join(response).strip()

    # Save the user input and AI response to memory
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(full_response)
    
    #llm.pipeline.generate(prompt, streamer=capture_output)
    #full_response = "".join(response).strip()
    # Post-process to remove any repeated "Chatbot:" or unnecessary content
    #if "Chatbot:" in "".join(response):
    #    full_response = full_response.split("Chatbot:")[0].strip()

    return "".join(response).strip()
    
    #return full_response

chatbot_tool = Tool(
    name="Chatbot",
    func=lambda question: simple_chatbot_response(question),
    description="Answers general questions or engages in casual conversation."
)

#tools = [calculator_tool, paint_tool, weather_tool, send_email_tool, wikipedia_tool, stock_report_tool, meeting_mode_tool,recording_vedio_tool ,chatbot_tool]
#tools = [calculator_tool, paint_tool, weather_tool, send_email_tool, wikipedia_tool, stock_report_tool, meeting_mode_tool,recording_vedio_tool]
tools = [weather_tool, send_email_tool, wikipedia_tool, stock_report_tool,set_brightness_tool,set_volume_tool ,recording_vedio_tool]
class AgentState(TypedDict):
    #messages: Annotated[List, add_messages]
    messages: Optional[Annotated[List, add_messages]]  # 用于存储消息列表
    selected_tool: Optional[str]  # 用于存储选择的工具名称
    user_input: Optional[str]     # 用于存储用户的最后输入

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def call_agent(data: AgentState):
    # Combine chat history into the prompt
    chat_history = "\n".join(
        [f"User: {msg.content}" if i % 2 == 0 else f"Assistant: {msg.content}" 
         for i, msg in enumerate(memory.chat_memory.messages)]
    )
    #if len(chat_history) >= 1024:
    #    chat_history = chat_history[1024:] 
    
    # 20250515 Ensure chat history does not exceed the length limit
    max_length = 1024  # Define a configurable length limit
    if len(chat_history) > max_length:
        # Split messages and truncate from the start while preserving message boundaries
        messages = chat_history.split("\n")
        truncated_history = []
        current_length = 0
        
        # Add messages from the end until the limit is reached
        for msg in reversed(messages):
            if current_length + len(msg) + 1 > max_length:  # +1 for the newline character
                break
            truncated_history.insert(0, msg)
            current_length += len(msg) + 1
        
        chat_history = "\n".join(truncated_history)
    
    prompt = (
        "You are an AI assistant with reasoning capabilities. "
        "When the user asks for an action, respond with the exact tool name to execute the action. "
        #"If the user is asking a general question or engaging in casual conversation, respond with 'Chatbot'. "
        "Do not provide explanations or step-by-step reasoning.\n\n"
        "Available tools:\n"
        + "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
        + "\n\n"
        + "Chat History:\n"
        + chat_history
        + "\n\n"
        + "User input:\n"
        + "".join([m.content for m in data["messages"] if hasattr(m, 'content')])
    )
    response = llm._call(prompt).strip()
    #print("response:", response)
    last_message = data["messages"][-1].content.lower()
    #print("last_message:", last_message)
    # Save the user input and assistant response to memory
    memory.chat_memory.add_user_message(last_message)
    memory.chat_memory.add_ai_message(response)

    # Exact match for tool names
    for tool in tools:
        if tool.name.lower() in response.lower():
            print(f"Exact match found: {tool.name}")
            return {"messages": [SystemMessage(content="")],"selected_tool": tool, "user_input": last_message}
            #return execute_tool(tool, last_message,data)

    # Fuzzy matching for tool names or descriptions
    best_match = None
    best_score = 0.0

    combined_input = chat_history +"\n" + last_message
    for tool in tools:
        for keyword in [tool.name.lower(), tool.description.lower()]:
            # Give higher weight to tool names
            name_score = similar(last_message, tool.name.lower()) * 1.5 # Higher weight for name
            #name_score = similar(combined_input, tool.name.lower()) * 1.5 # Higher weight for name
            # Give lower weight to tool descriptions
            desc_score = similar(last_message, tool.description.lower())
            #desc_score = similar(combined_input, tool.description.lower())
            # Combine scores
            score = max(name_score, desc_score)
            print(f"Keyword: {keyword}, Score: {score}")
            #score = similar(last_message, keyword)
            if score > best_score:
                best_score = score
                best_match = tool
                

    if best_match and best_score > 0.3:
        return {"messages": [SystemMessage(content="")],"selected_tool": best_match, "user_input": last_message}
    else:
        # Default to chatbot response
        chatbot_response = simple_chatbot_response(last_message)
        memory.chat_memory.add_ai_message(chatbot_response)  # Save chatbot response to memory
    #return {"messages": data["messages"] + [SystemMessage(content=chatbot_response), SystemMessage(content="[FINISH RESPONSE]")]}
    return {"messages": [SystemMessage(content="[FINISH RESPONSE]")],"selected_tool": None, "user_input": None}
def execute_meeting_mode(data: AgentState):
    tool = data["selected_tool"]
    last_message = data["user_input"]

    if tool.name == "Set Brightness":
        # 直接用 AI 決定亮度
        prompt = (
            f"You are an AI assistant. The user wants to set the screen brightness. "
            f"Generate a valid brightness level (0-100) based on the user's input: '{last_message}'. "
            f"Only provide the brightness level as a number, no extra text."
        )
        brightness = llm._call(prompt).strip()
        tool_result = tool.func(brightness)
        print(f"🤖: {tool_result}" )
        return {
            "messages": data["messages"] + [SystemMessage(content=tool_result), SystemMessage(content="[TOOL_EXECUTED]")],
            "selected_tool": None,
            "user_input": tool_result  # 把結果傳給下一步
        }
    elif tool.name == "Set Volume":
        # 直接用 AI 決定音量
        prompt = (
            f"You are an AI assistant. The user wants to set the master volume. "
            f"Based on the previous tool result: '{last_message}', generate a valid volume level (0-100). "
            f"Only provide the volume level as a number, no extra text."
        )
        master_volume = llm._call(prompt).strip()
        tool_result = tool.func(master_volume)
        print(f"🤖: {tool_result}" )
        return {
            "messages": data["messages"] + [SystemMessage(content=tool_result), SystemMessage(content="[TOOL_EXECUTED]")],
            "selected_tool": None,
            "user_input": tool_result
        }   

def should_end(data: AgentState) -> bool:
    if not data["messages"]:
        print("Debug: No messages in data.")
        return False
    last_msg = data["messages"][-1].content
    print(f"Debug: Last message content in should_end: {last_msg}")
    result = any(stop_word in last_msg for stop_word in ["✅", "❌", "[NEXT]"])
    print(f"Debug: should_end result: {result}")
    return result
def build_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_agent)
    workflow.add_node("bridge_tool", lambda data: {"messages": data["messages"], "selected_tool": None, "user_input": data["user_input"]})
    workflow.add_node("set_brightness_tool", lambda data: execute_meeting_mode({**data, "selected_tool": set_brightness_tool}))
    # 新增 agent 回應節點
    def agent_after_brightness(data):
        # 用 AI 理解 set_brightness_tool 的回應
        last_tool_result = data["user_input"]
        prompt = (
            f"You are an AI assistant. The previous tool result was: '{last_tool_result}'. "
            f"Summarize or interpret this result for the next step."
        )
        ai_response = llm._call(prompt).strip()
        return {"messages": data["messages"] + [SystemMessage(content=ai_response)], "selected_tool": None, "user_input": ai_response}
    workflow.add_node("agent_after_brightness", agent_after_brightness)
    workflow.add_node("set_volume_tool", lambda data: execute_meeting_mode({**data, "selected_tool": set_volume_tool}))
    def end_for(data):
        #print("Workflow has reached the end.")
        #print("Final messages:", [msg.content for msg in data["messages"]])
        return {
            "messages": data["messages"] + [SystemMessage(content="Workflow ended.")],
            "selected_tool": None,
            "user_input": None
        }
    workflow.add_node("end_for", end_for)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", "set_brightness_tool")
    workflow.add_edge("set_brightness_tool", "agent_after_brightness")
    workflow.add_edge("agent_after_brightness", "set_volume_tool")
    workflow.add_edge("set_volume_tool", "end_for")
    workflow.set_finish_point("end_for")
    return workflow.compile()
    