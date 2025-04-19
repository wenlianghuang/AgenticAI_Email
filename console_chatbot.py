from Agentic_AI_SendEmail import agent

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
            print(f"Bot: {response}")
        except Exception as e:
            print(f"Bot: Sorry, I couldn't process your request. Error: {e}")

if __name__ == "__main__":
    main()