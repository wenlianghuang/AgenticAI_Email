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
    #args_schema=StockReport
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
    description="The meeting is ready to start, activates meeting mode and checks the brightness and volume settings.",
    args_schema=MeetingModeType
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
tools = [weather_tool, send_email_tool, wikipedia_tool, stock_report_tool, meeting_mode_tool,recording_vedio_tool]
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
    print("response:", response)
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
# 保存上下文資訊的全域變數
meeting_mode_context = {"brightness": None, "master_volume": None, "app_volume": None}
def execute_tool(data: AgentState):
    tool = data["selected_tool"]
    last_message = data["user_input"]
    global meeting_mode_context  # 使用全域變數保存上下文資訊
    if tool is None:
        return {"messages": data["messages"] + [SystemMessage(content="No tool matched in execute_tool.")], "selected_tool": None, "user_input": None}
    if tool.name == "Search Wikipedia":
        print("🤖 Please provide a topic to search on Wikipedia.")
        topic = input("Topic: ")
        tool_result = tool.func(topic)
        return {"messages": data["messages"] + [SystemMessage(content=tool_result), SystemMessage(content="[TOOL_EXECUTED]"),SystemMessage(content="[NEXT]")],"selected_tool": None, "user_input": None}

    elif tool.name == "Get Weather":
        print("🤖 Please provide the city name for weather information.")
        city = input("City: ")
        tool_result = tool.func(city)
        print("tool_result:", tool_result)
        return {"messages": data["messages"] + [SystemMessage(content=tool_result), SystemMessage(content="[TOOL_EXECUTED]"),SystemMessage(content="[NEXT]")],"selected_tool": None, "user_input": None}
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
            f"The parameters 'brightness', 'master_volume', and 'app_volume' must be within the range of 0 to 100. "
            f"The user has provided the following preferences: {last_message}. "
            f"Use the following context for missing parameters: {meeting_mode_context}. "
            f"Here is the chat history for reference: {memory.chat_memory.messages}. "  # 包含 chat_history
            f"Do not change the value of parameters explicitly mentioned to be kept unchanged by the user. "
            f"Generate appropriate values for the missing parameters based on the user's intent. "
            f"But the missing parameters must not over 50. "
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
            # 更新全局上下文
            meeting_mode_context["brightness"] = brightness
            meeting_mode_context["master_volume"] = master_volume
            meeting_mode_context["app_volume"] = app_volume
            tool_result = tool.func(brightness=brightness, master_volume=master_volume, app_volume=app_volume)
        except (ValueError, IndexError) as e:
            # 捕捉錯誤並提供回饋
            tool_result = f"❌ Failed to parse meeting mode parameters: {str(e)}. Please ensure the input format is correct."
            print(tool_result)
        #tool_result = tool.func(brightness=brightness,master_volume=master_volume,app_volume=app_volume)
        # 使用語言模型生成回應
        if "✅" in tool_result:  # 成功的情況
            response_prompt = (
                f"You are an AI assistant. The user activate meeting mode has successfully."
                f"Generate a friendly and professional response to confirm this action."
                f"Only provide the response without any additional text."
            )
        else:
            response_prompt = (
                f"You are an AI assistant. The user attempted to activate meegint mode, but it failed. "
                f"Generate a polite and helpful response explaining the failure and suggesting possible solutions."
                f"Only provide the reponse without any additional text."
            )
        # stream generation response 
        response = []
        def capture_output(subword):
            response.append(subword)
            print(subword, end="", flush=True)  # 即時輸出
            return False
        print("🤖 ")
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
    #elif tool is None and last_message is None:
    #    return {"messages": data["messages"] + [SystemMessage(content="No tool matched in execute_tool.")], "selected_tool": None, "user_input": None}
    #elif tool.name == "Chatbot":
        # 直接使用工具的 func 方法
    #    tool_result = tool.func(last_message)
    #    return {"messages": [SystemMessage(content="[FINISH RESPONSE]")]}
    print("Return messages:", [msg.content for msg in data["messages"]])
    #return {"messages": data["messages"] + [SystemMessage(content=tool_result), SystemMessage(content="[TOOL_EXECUTED]"),SystemMessage(content="[NEXT]")],"selected_tool": None, "user_input": None}

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

    return {"messages": data["messages"] + [SystemMessage(content="No tool matched in call_tools_with_feedback.")], "selected_tool": None, "user_input": None}
'''
def should_end(data: AgentState) -> bool:
    last_msg = data["messages"][-1].content
    print(f"Debug: Last message content in should_end: {last_msg}")
    return any(stop_word in last_msg for stop_word in ["✅", "❌","[NEXT]"])
    #return any(stop_word in last_msg for stop_word in ["✅", "❌"])
'''
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
    
    # 添加 agent 節點
    workflow.add_node("agent", call_agent)
    
    # 添加 execute_tool 節點
    workflow.add_node("execute_tool", execute_tool)
    
    # 添加 weather_tool 節點
    workflow.add_node("weather_tool", lambda data: execute_tool({**data, "selected_tool": weather_tool}))
    
    # 添加 stock_report_tool 節點
    #workflow.add_node("stock_report_tool", lambda data: execute_tool({**data, "selected_tool": stock_report_tool}))
    workflow.add_node("wikipedia_tool", lambda data: execute_tool({**data, "selected_tool": wikipedia_tool}))
    # 添加 end 節點
    #workflow.add_node("end_for", lambda data: {"messages": data["messages"] + [SystemMessage(content="Workflow ended.")], "selected_tool": None, "user_input": None})
    def start_for(data):
        # 打印一些資訊
        print("Workflow has reached the start.")
        print("Final messages:", [msg.content for msg in data["messages"]])
        return {
            "messages": data["messages"] + [SystemMessage(content="Workflow start.")],
            "selected_tool": None,
            "user_input": None
        }
    workflow.add_node("start_for", start_for)
    def end_for(data):
        # 打印一些資訊
        print("Workflow has reached the end.")
        print("Final messages:", [msg.content for msg in data["messages"]])
        return {
            "messages": data["messages"] + [SystemMessage(content="Workflow ended.")],
            "selected_tool": None,
            "user_input": None
        }
    workflow.add_node("end_for", end_for)
    def test_for(data):
        # 打印一些資訊
        print("Workflow has reached the test.")
        print("Final messages:", [msg.content for msg in data["messages"]])
        return {
            "messages": data["messages"] + [SystemMessage(content="Workflow test.")],
            "selected_tool": None,
            "user_input": None
        }
    workflow.add_node("test_for", test_for)
    # 設置入口點
    workflow.set_entry_point("agent")
    
    # 添加邊：從 agent 到 weather_tool
    #workflow.add_edge("agent", "weather_tool")
    workflow.add_edge("agent", "start_for")
    # 添加邊：從 weather_tool 到 stock_report_tool
    #workflow.add_edge("weather_tool", "stock_report_tool")
    #workflow.add_edge("weather_tool", "wikipedia_tool")
    # 添加條件邊：從 weather_tool 到 stock_report_tool 或 end
    workflow.add_conditional_edges(
        "start_for",
        #lambda data: any("[NEXT]" in msg.content for msg in data["messages"]),  # 用戶輸入 "next" 繼續
        lambda data: should_end(data),  # 用戶輸入 "next" 繼續
        #lambda data: print(f"Debug: should_end result: {should_end(data)}") or should_end(data),
        #should_end,
        #"wikipedia_tool",  # 如果應該結束，跳到 end
        #"test_for",  # 如果應該結束，跳到 end
        #"end_for",
        {
            True: "test_for",   # 如果 decision 為 True，跳轉到 B
            False: "end_for"   # 如果 decision 為 False，跳轉到 C
        }
    )
    '''
    workflow.add_conditional_edges(
        "weather_tool",
        #lambda data: any("[NEXT]" in msg.content for msg in data["messages"]),  # 用戶輸入 "next" 繼續
        lambda data: should_end(data),  # 用戶輸入 "next" 繼續
        #lambda data: print(f"Debug: should_end result: {should_end(data)}") or should_end(data),
        #should_end,
        #"wikipedia_tool",  # 如果應該結束，跳到 end
        "test_for",  # 如果應該結束，跳到 end
        "end_for",
    )
    '''
    
    '''
    # 添加條件邊：從 stock_report_tool 到 end 或 agent
    workflow.add_conditional_edges(
        #"stock_report_tool", 
        "wikipedia_tool",
        #"weather_tool",
        lambda data: should_end(data),
        #lambda data: any("[NEXT]" in msg.content for msg in data["messages"]),  # 用戶輸入 "next" 繼續
        "end",  # 如果應該結束，跳到 end
        "agent",  # 否則返回 agent
    )
    '''
    
    return workflow.compile()