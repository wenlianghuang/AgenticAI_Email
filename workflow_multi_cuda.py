from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain.schema import SystemMessage
from typing import TypedDict, Annotated, List
from difflib import SequenceMatcher
#from console_chatbot_langgraph import tools, call_agent, call_tools_with_feedback, should_end
from langchain.llms.base import LLM
from dotenv import load_dotenv
#import openvino_genai as ov_genai
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain.memory import ConversationBufferMemory
from typing import List, Optional
from langchain.tools import Tool, StructuredTool
import os
from agent_tool.weather import get_weather_by_city
from agent_tool.sendEmail import send_email, SendEmailInput
from agent_tool.searchWikipedia import search_wikipedia

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
            print(f"Tool result: {tool_result}")
            # Add a marker to indicate the tool has been executed
            return {"messages": data["messages"] + [SystemMessage(content=tool_result), SystemMessage(content="[TOOL_EXECUTED]")]} # 多給一個標記，讓後面不會重複執行

    return {"messages": data["messages"] + [SystemMessage(content="No matching tool found.")]}

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def call_tools_with_feedback(data: AgentState):
    # Check if the tool has already been executed
    if any("[TOOL_EXECUTED]" in msg.content for msg in data["messages"]):
        return data  # Return the data as-is without executing any tool

    last_message = data["messages"][-1].content.lower()
    best_match = None
    best_score = 0.0

    # Clean the LLM response, extract possible tool name or description
    cleaned_message = last_message.split("\n")[0]

    for tool in tools:
        # Match against both tool name and description
        for keyword in [tool.name.lower(), tool.description.lower()]:
            score = similar(cleaned_message, keyword)
            if score > best_score:
                best_score = score
                best_match = tool
    print(f"Best match: {best_match.name if best_match else 'None'} with score: {best_score}")
    # Only execute the tool if the match score is above a stricter threshold
    if best_match and best_score > 0.5:
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
        else:
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