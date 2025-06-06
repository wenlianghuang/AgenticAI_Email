#from Agentic_AI_SendEmail import agent,openvino_llm,agent_executor
from Agentic_AI_Tool import agent
def main():
    print("Welcome to the Console Chatbot! Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit' or user_input.lower() == 'cls':
            print("Goodbye!")
            break
        
        # 使用 Agent 處理輸入
        try:
            if "send email" in user_input.lower():
                print("Bot: I can help you send an email. Please provide the details.")
                recipient = input("Recipient: ")
                subject = input("Subject: ")
                body = input("Body: ")
                response = agent.run({
                    "input": {
                        "recipient": recipient,
                        "subject": subject,
                        "body": body
                    }
                })
                print(f"Bot: {response}")
            else:
                response = agent.run({"input": user_input})
                print(f"Bot: {response}")
        except Exception as e:
            print(f"Bot: Sorry, I couldn't process your request. Error: {e}")
        '''
        if "send email" in user_input.lower():
            print("Bot: I can help you send an email. Please provide the details.")
            recipient = input("Recipient: ")
            subject = input("Subject: ")
            body = input("Body: ")
            try:
                response = agent.run({
                    "input": {
                        "recipient": recipient,
                        "subject": subject,
                        "body": body
                    }
                })
                print(f"Bot: {response}")
            except Exception as e:
                print(f"Bot: Failed to send email. Error: {e}")
        '''
if __name__ == "__main__":
    main()