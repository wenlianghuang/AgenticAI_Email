# Import necessary modules and libraries
# langchain: Used for creating and managing AI agents and tools
# openvino_genai: Used for initializing and using OpenVINO's LLM pipeline
# smtplib, email.mime: Used for sending emails via SMTP
# dotenv: Used for loading environment variables from a .env file
# os: Provides functions to interact with the operating system
from langchain.agents import initialize_agent, Tool, AgentExecutor, AgentType
from langchain.tools import StructuredTool
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferMemory
import openvino_genai as ov_genai
from typing import List, Optional
from pydantic import BaseModel
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os

import json
import urllib.request
import urllib.parse
import ssl
# Load environment variables from a .env file
# This is used to securely store sensitive information like passwords
load_dotenv()
API_KEY = os.getenv('AccuWeatherAPIKey')
# Initialize OpenVINO LLM pipeline
# The model_path specifies the pre-trained model to use
# The device parameter specifies the hardware (e.g., NPU) for running the model
model_path = 'Phi-35_mini_instruct_refined'
pipe = ov_genai.LLMPipeline(model_path, device='NPU')
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# Define a custom LLM class for OpenVINO
# This class integrates OpenVINO's LLM pipeline with LangChain's LLM interface
class OpenVINO_LLM(LLM):
    def __init__(self, pipeline: ov_genai.LLMPipeline, **kwargs):
        super().__init__(**kwargs)
        self._pipeline = pipeline

    # Property to access the pipeline
    @property
    def pipeline(self) -> ov_genai.LLMPipeline:
        return self._pipeline

    # Property to define the LLM type
    @property
    def _llm_type(self) -> str:
        return "openvino_genai"

    # Method to process a prompt and generate a response
    

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = []
        # Capture the output generated by the pipeline
        def capture_output(subword):
            response.append(subword)
            return False
        self.pipeline.generate(prompt, streamer=capture_output)
        return "".join(response)
    
# Instantiate the OpenVINO LLM
openvino_llm = OpenVINO_LLM(pipeline=pipe)

# Function to send an email
# Parameters:
# - recipient: The email address of the recipient
# - subject: The subject of the email
# - body: The body content of the email
def send_email(recipient: str, subject: str, body: str):
    # Sender email and password (retrieved from environment variables)
    sender_email = "wenliangmatt@gmail.com"
    sender_password = os.getenv('wenliangmattapppwd')  # App-specific password for Gmail
    #print(f"DEBUG: sender_password = {sender_password}")  # Debug message
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    try:
        # Create the email message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Connect to the SMTP server and send the email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Start TLS encryption
            server.login(sender_email, sender_password)  # Login to the SMTP server
            server.send_message(msg)  # Send the email

        #print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Function to open the calculator application
# Returns a success or error message
def open_calculator() -> str:
    try:
        #print("DEBUG: open_calculator() 被觸發！")  # Debug message
        os.startfile("calc.exe")  # Open the calculator application
        return "Calculator opened successfully in Windows system! No further actions are needed."
    except FileNotFoundError:
        return "Calculator application not found."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def open_paint() -> str:
    try:
        os.startfile("mspaint.exe")  # Open the Paint application
        return "Paint opened successfully!"
    except FileNotFoundError:
        return "Paint application not found."
# 判斷城市名稱的語言
def detect_language(city_name):
    if any(u'\u4e00' <= char <= u'\u9fff' for char in city_name):  # 檢查是否包含中文字符
        return "zh-tw"
    return "en-us"
# 查詢城市的 Location Key
def get_location_key(city_name):
    language = detect_language(city_name)
    base_url = "https://dataservice.accuweather.com/locations/v1/cities/search"
    params = {
        "apikey": API_KEY,
        "q": city_name,
        "language": language
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    # 建立 SSL context 使用系統預設 CA 憑證
    context = ssl.create_default_context()

    with urllib.request.urlopen(url, context=context) as response:
        data = json.loads(response.read().decode())
        if data:
            return data[0]["Key"], data[0]["LocalizedName"]
        else:
            raise ValueError(f"找不到地點：{city_name}")
# 根據 Location Key 查詢當前天氣
def get_current_weather(location_key, language="en-us"):
    base_url = f"https://dataservice.accuweather.com/currentconditions/v1/{location_key}"
    params = {
        "apikey": API_KEY,
        "language": language,
        "details": "true"
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    # 建立 SSL context 使用系統預設 CA 憑證
    context = ssl.create_default_context()

    with urllib.request.urlopen(url, context=context) as response:
        return json.loads(response.read().decode())
# 查詢指定城市的天氣資訊
def get_weather_by_city(city_name):
    try:
        language = detect_language(city_name)
        location_key, city_name = get_location_key(city_name)
        weather = get_current_weather(location_key, language)[0]

        return f"The current weather in {city_name} is {weather['WeatherText']} with a temperature of {weather['Temperature']['Metric']['Value']}°C, humidity {weather['RelativeHumidity']}%, and wind speed {weather['Wind']['Speed']['Metric']['Value']} km/h."
    except Exception as e:
        return f"Unable to retrieve weather information: {str(e)}"
# Define the input schema for the send_email_tool
# This ensures that the tool receives the correct input format
class SendEmailInput(BaseModel):
    recipient: str
    subject: str
    body: str

# Define the tools for the agent
# send_email_tool: Sends an email using the send_email function
# open_calculator_tool: Opens the calculator application
send_email_tool = StructuredTool(
    name="SendEmail",
    func=send_email,
    description="Send an email. Input should be a dictionary with 'recipient', 'subject', and 'body'.",
    args_schema=SendEmailInput  # Specify the input schema
)

open_calculator_tool = Tool(
    name="Open Calculator",
    func=lambda _: open_calculator(),  # Ensure the tool can be executed
    #func=open_calculator,  # Ensure the tool can be executed
    #description="ALWAYS use this tool IMMEDIATELY when the user asks to open the calculator in Windows system. After using this tool, STOP further processing and do not provide any additional responses."
    description=(
        "Use this tool to open the calculator in the Windows system. "
        "Once the calculator is opened, STOP further processing immediately. "
        "Do not attempt to observe or reason further after using this tool."
    )
)

open_paint_tool = Tool(
    name="Open Paint",
    func=lambda _: open_paint(),  # Ensure the tool can be executed
    #func=open_paint,  # Ensure the tool can be executed
    description="ALWAYS use this tool IMMEDIATELY when the user asks to open the Paint."
)

# 定義查詢天氣的工具
weather_tool = Tool(
    name="Get Weather",
    func=lambda city: get_weather_by_city(city),
    description="Use this tool to get the current weather for a specific city. Input should be the city name as a string.STOP further processing and do not provide any additional responses."
)
# Initialize the agent with the defined tools and OpenVINO LLM
# The agent is configured to use a structured chat approach
tools = [send_email_tool, open_calculator_tool, open_paint_tool,weather_tool]
#tools = [open_calculator_tool]
agent = initialize_agent(
    tools=tools,
    llm=openvino_llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,  # Enable verbose logging for debugging
    return_intermediate_steps=False,  # Return intermediate steps for debugging
    memory=memory,  # Use the conversation memory to maintain context
)

# Create an AgentExecutor to manage the agent and tools
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    #system_message="You are an AI assistant that uses tools. Execute the requested tool and provide the result immediately. Do not re-execute the same tool unnecessarily.",
    #system_message = (
    #    "You are an AI assistant that uses tools. "
    #    "Execute the requested tool and provide the result immediately. "
    #    "Once a tool has returned a valid result, STOP further processing."
    #    "Do not attempt to reason or take additional actions after the tool has been executed. "
    #),
    system_message=(
        "You are an AI assistant that uses tools. "
        "When a tool like 'Open Calculator' or 'Open Paint' is executed, "
        "STOP further processing immediately and do not provide any additional responses. "
        "For other tools, execute them and provide the result."
    ),
    handle_parse_errors=False,  # If the response do not have a perfect result, it will show the error or
)
#result = agent_executor.invoke({"input": "Open the calculator"})
#print(result['output'])  # Print the result of the agent's execution
# Test the agent by sending a test email
'''
response = agent.run({
    "input": {
        "recipient": "wenlianghuang08@gmail.com",
        "subject": "Test Email",
        "body": "This is a test email sent using LangChain."
    }
})
'''