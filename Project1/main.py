from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os


load_dotenv()
@tool 
def greet(name:str)->str:
    """Useful for greeting people."""
    print("The External Tool has been Called!")
    return f"Hello,{name} ! Nice to meet you."

def main():

    # Create Gemini model instance
    llm= ChatGoogleGenerativeAI(api_key=os.getenv("GEMINI_API_KEY"),model="gemini-2.5-flash",temperature=0)

    tools=[]
    agent_executor=create_react_agent(llm,tools)

    print("Welcome to the Gemini Agent. Type 'exit' to quit.")
    print("You can ask me to perform calucations or chat with me.")

    while True:
        user_input=input("\nYou: ").strip()

        if user_input == "quit":
            break;

        print("\n Assistant: ",end="")

        for chunk in agent_executor.stream({"messages":[HumanMessage(content=user_input)]}):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    print(message.content,end="")
        print()
    
if __name__ == "__main__":
    main()









