from Agentic_AI_SendEmail import agent,openvino_llm,agent_executor

def main():
    print("Welcome to the Console Chatbot! Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        # 使用 Agent 處理輸入
        try:
            #response = agent.run(user_input)

            response = agent.run({"input": user_input})
            #response = agent_executor.invoke({"input": user_input})
            #response = agent_executor.run({"input": user_input})
            print(f"Bot: {response}")

            #if "Stopping further processing" in response:
            #    continue
        except Exception as e:
            print(f"Bot: Sorry, I couldn't process your request. Error: {e}")
        
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

if __name__ == "__main__":
    main()