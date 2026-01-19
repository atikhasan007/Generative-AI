import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage



# Access the variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY


chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")
search = TavilySearchResults(max_results=3)

# res = search.invoke("Tell me the recent movies list in 2025")
# print(res)

tool = [search]

agent_executor = create_react_agent(chatModel, tool)


response = agent_executor.invoke({"messages": [HumanMessage(content="Tell me the recent movies list in 2025")]})

print(response['messages'])