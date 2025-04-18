from Agentic_AI_SendEmail import openvino_llm

def main():
    print("Welcome to the Console Chatbot! Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        print("Bot: ", end="", flush=True)
        def capture_output(subword):
            print(subword, end="", flush=True)  # 实时输出每个 token
            return False
        
        openvino_llm.pipeline.generate(user_input, streamer=capture_output)
        print()
if __name__ == "__main__":
    main()