from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain.schema import SystemMessage
from typing import TypedDict, Annotated, List
from difflib import SequenceMatcher
#from console_chatbot_langgraph import tools, call_agent, call_tools_with_feedback, should_end
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
#from agent_tool.stockReport import get_quarterly_revenue_and_plot
from agent_tool.stockReport import get_financial_tool, StockReport
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
    #func=lambda symbol: get_quarterly_revenue_and_plot(symbol),
    func=get_financial_tool,
    description="Fetches and plots the quarterly revenue data for a given stock symbol."
)
def simple_chatbot_response(user_input: str) -> str:
    '''
    prompt = (
        f"You are a helpful chatbot. Answer the user's question concisely and informatively."
        f"User: {user_input}\n"
        f"Chatbot:"
    )
    return llm._call(prompt).strip()
    '''
    prompt = (
        f"You are a helpful chatbot. Answer the user's question concisely and informatively. "
        f"Provide your response step by step if necessary.\n\n"
        f"User: {user_input}\n"
        f"Chatbot:"
    )
    response = []

    # 使用 streamer 逐步生成回應
    def capture_output(subword):
        response.append(subword)
        print(subword, end="", flush=True)  # 即時輸出每個生成的部分
        return False  # 繼續生成

    llm.pipeline.generate(prompt, streamer=capture_output)
    return "".join(response).strip()
chatbot_tool = Tool(
    name="Chatbot",
    func=lambda question: simple_chatbot_response(question),
    description="Answers general questions or engages in casual conversation."
)
tools = [calculator_tool, paint_tool, weather_tool, send_email_tool, wikipedia_tool,stock_report_tool,chatbot_tool]
#tools = [calculator_tool, paint_tool, weather_tool, send_email_tool, wikipedia_tool,stock_report_tool]
# Define agent input/output schema
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]

# Define the core logic node
def call_agent(data: AgentState):
    #print("run in call_agent")
    prompt = (
        "You are an AI assistant with reasoning capabilities. "
        "When the user asks for an action, respond with the exact tool name to execute the action. "
        #"If the user is asking a general question or engaging in casual conversation, respond with 'Chatbot'. "
        #"If the user is asking a general question or engaging in casual conversation, respond appropriately. "
        "Do not provide explanations or step-by-step reasoning.\n\n"
        "Available tools:\n"
        + "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
        + "\n\n"
        + "User input:\n"
        + "".join([m.content for m in data["messages"] if hasattr(m, 'content')])
    )
    response = llm._call(prompt).strip()
    last_message = data["messages"][-1].content.lower()
    # Match the response to a tool
    for tool in tools:
        if tool.name.lower() in response.lower():
            #if tool.name == "Chatbot":
            #    # 問答模式
            #    tool_result = tool.func(last_message)
            if tool.name == "Search Wikipedia":
                # Prompt the user for a topic to search
                print("🤖 Please provide a topic to search on Wikipedia.")
                topic = input("Topic: ")
                tool_result = tool.func(topic)
            elif tool.name == "Get Weather":
                print("🤖 Please provide the city name for weather information. again")
                city = input("City: ")
                tool_result = tool.func(city)
            elif tool.name == "SendEmail":
                print("🤖 Please provide the recipient, subject and the concept of day.")
                recipient = input("Recipient: ")
                subject = input("Subject: ")
                # Use LLM to generate a detailed email body
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
                prompt = (
                    f"You are an AI assistant. The user wants to get a stock report for the company '{symbol}'. "
                    f"Find the stock ticker symbol for this company and provide it to the user."
                    f"Only return the stock ticker symbol, nothing else."
                )
                stock_ticker = llm._call(prompt).strip()
                
                tool_result = tool.func(stock_ticker,choice)
                #tool_result = tool.func(symbol)    
            else:
                tool_result = tool.func("")
            #else:
            #    print("Run normal QA")
            #    tool_result = tool.func(last_message)
            
            #print(f"Tool result: {tool_result}")
            # Add a marker to indicate the tool has been executed
            return {"messages": data["messages"] + [SystemMessage(content=tool_result), SystemMessage(content="[TOOL_EXECUTED]")]} # 多給一個標記，讓後面不會重複執行
        else:
            chatbot_response = simple_chatbot_response(last_message)
            return {"messages": data["messages"] + [SystemMessage(content=chatbot_response),SystemMessage(content="[FINISH RESPONSE]")]} # 多給一個標記，讓後面不會重複執行

    return {"messages": data["messages"] + [SystemMessage(content="No matching tool found, try to find a more deep tool.")]} # 這是在沒有找到對應的tool,才會繼續往"call_tools_with_feedback"走
    #return {"messages": data["messages"] + [SystemMessage(content="[NO_TOOL_MATCHED]")]} # 這是在沒有找到對應的tool,才會繼續往"call_tools_with_feedback"走

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def call_tools_with_feedback(data: AgentState):
    # Check if the tool has already been executed
    if any("[TOOL_EXECUTED]" in msg.content for msg in data["messages"]) or any("[FINISH RESPONSE]" in msg.content for msg in data["messages"]):
        return data  # Return the data as-is without executing any tool
    print("run in call_tools_with_feedback")
    last_message = data["messages"][-1].content.lower()
    best_match = None
    best_score = 0.0

    # Clean the LLM response, extract possible tool name or description
    cleaned_message = last_message.split("\n")[0]

    for tool in tools:
        #print(f"Tool name: {tool.name}, Tool description: {tool.description}")
        # Match against both tool name and description
        for keyword in [tool.name.lower(), tool.description.lower()]:
            score = similar(cleaned_message, keyword)
            #print(f"Best match: {best_match.name if best_match else 'None'}, Score: {best_score}, Current score: {score}")
            if score > best_score:
                best_score = score
                best_match = tool
                #print(f"Best match: {best_match.name if best_match else 'None'} with score: {best_score}")
    #print(f"Best match: {best_match.name if best_match else 'None'} with score: {best_score}")
    # Only execute the tool if the match score is above a stricter threshold
    if best_match and best_score > 0.8:
        if best_match.name == "SendEmail":
             # Prompt the user for additional details
            print("🤖 Please provide the email details.")
            recipient = input("Recipient: ")
            subject = input("Subject: ")
            # Use LLM to generate a detailed email body
            prompt = (
                f"You are an AI assistant. The user wants to send an email with the subject '{subject}'. "
                f"Generate a detailed and professional email body based on the user's input: '{last_message}'."
            )
            body = llm._call(prompt).strip()
            #body = input("Body: ")
            tool_result = best_match.func(recipient, subject, body)
            #tool_result = best_match.func({"recipient": recipient, "subject": subject, "body": body})
        elif best_match.name == "Search Wikipedia":
            # Prompt the user for a topic to search
            print("🤖 Please provide a topic to search on Wikipedia.")
            topic = input("Topic: ")
            tool_result = best_match.func(topic)
        else:
            tool_result = best_match.func("")
        if "✅" in tool_result:
            return {"messages": data["messages"] + [SystemMessage(content=tool_result)]}
        else:
            return {"messages": data["messages"] + [SystemMessage(content=f"Tool failed: {tool_result}. Trying alternatives...")]}
    
    return {"messages": data["messages"] + [SystemMessage(content="No tool matched in call_tools_with_feedback.")]}

def should_end(data: AgentState) -> bool:
    last_msg = data["messages"][-1].content
    return any(stop_word in last_msg for stop_word in ["✅", "❌"])

# Build the graph
def build_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_agent)
    workflow.add_node("tool_call", call_tools_with_feedback)
    workflow.set_entry_point("agent")
    #workflow.set_entry_point("tool_call")
    workflow.add_edge("agent", "tool_call")
    #workflow.add_edge("tool_call", "agent") # Add an edge from tool_call to agent for feedback loop
    workflow.add_conditional_edges(
        "tool_call", 
        should_end,
        "end",
        "agent",
    )
    return workflow.compile()