from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain.schema import SystemMessage, HumanMessage
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool, StructuredTool
import openvino_genai as ov_genai
from typing import TypedDict, Annotated, List, Optional
import os
from dotenv import load_dotenv
from difflib import SequenceMatcher

# Custom imports
from agent_tool.weather import get_weather_by_city
from agent_tool.sendEmail import send_email, SendEmailInput
from agent_tool.searchWikipedia import search_wikipedia

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
    description="Returns a Wikipedia summary given a topic."
)

tools = [calculator_tool, paint_tool, weather_tool, send_email_tool, wikipedia_tool]

# Define agent input/output schema
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]

# Define the core logic node
def call_agent(data: AgentState):
    prompt = (
        "You are an AI assistant with reasoning capabilities. "
        "When the user asks for an action, respond with the exact tool name to execute the action. "
        "Do not provide explanations or step-by-step reasoning.\n\n"
        "Available tools:\n"
        + "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
        + "\n\n"
        + "User input:\n"
        + "".join([m.content for m in data["messages"] if hasattr(m, 'content')])
    )
    response = llm._call(prompt).strip()

    # Match the response to a tool
    for tool in tools:
        if tool.name.lower() in response.lower():
            tool_result = tool.func("")
            # Add a marker to indicate the tool has been executed
            return {"messages": data["messages"] + [SystemMessage(content=tool_result), SystemMessage(content="[TOOL_EXECUTED]")]} # 多給一個標記，讓後面不會重複執行

    return {"messages": data["messages"] + [SystemMessage(content="No matching tool found.")]}

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# 20250502 run in call_tools_with_feedback when tool is not executed => 找不到我定義的工具才往下找
def call_tools_with_feedback(data: AgentState):
    # Check if the tool has already been executed
    if any("[TOOL_EXECUTED]" in msg.content for msg in data["messages"]):
        return data  # Return the data as-is without executing any tool

    last_message = data["messages"][-1].content.lower()
    best_match = None
    best_score = 0.0

    # Clean the LLM response to extract possible tool name or description
    cleaned_message = last_message.split("\n")[0]   # Use the first line of the response 

    for tool in tools:
        # Match against both tool name and description
        for keyword in [tool.name.lower(), tool.description.lower()]:
            score = similar(cleaned_message, keyword)
            if score > best_score:
                best_score = score
                best_match = tool
    # Only execute the tool if the match score is above a stricter threshold
    if best_match and best_score > 0.5:
        tool_result = best_match.func("")
        if "✅" in tool_result:
            return {"messages": data["messages"] + [SystemMessage(content=tool_result)]}
        else:
            return {"messages": data["messages"] + [SystemMessage(content=f"Tool failed: {tool_result}. Trying alternatives...")]}
    
    return {"messages": data["messages"] + [SystemMessage(content="No tool matched.")]}

def should_end(data: AgentState) -> bool:
    last_msg = data["messages"][-1].content
    return any(stop_word in last_msg for stop_word in ["✅", "❌"])

# Build the graph
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

app = workflow.compile()

# Console chatbot
def main():
    print("Welcome to the Console Chatbot! Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        inputs = {"messages": [HumanMessage(content=user_input)]}
        result = app.invoke(inputs)
        for msg in result["messages"]:
            if "[TOOL_EXECUTED]" not in msg.content:
                print(f"🤖 {msg.content}")

if __name__ == "__main__":
    main()