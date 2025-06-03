from langchain.schema import HumanMessage
# workflow_multi import build_workflow
#from workflow_multi_history import build_workflow
#from workflow_multi_fuzzy_tool import build_workflow
#from workflow_multi_repond_agentic_ai import build_workflow
#from workflow_multi_history import build_workflow
#from workflow_multi_history_continues import build_workflow
from workflow_multi_history_continues_v3 import build_workflow
app = build_workflow()

# Console chatbot
def main():
    print("Welcome to the Console Chatbot! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        inputs = {"messages": [HumanMessage(content=user_input)],"selected_tool": None,"user_input": user_input}
        result = app.invoke(inputs)
        '''
        print('Outside all result:', result)
        

        for msg in result["messages"]:
            if "[TOOL_EXECUTED]" not in msg.content and user_input not in msg.content:
                print(f"🤖 {msg.content}")
            elif msg.content == None:
                break
        '''
if __name__ == "__main__":
    main()