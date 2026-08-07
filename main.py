from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langfuse import get_client


load_dotenv()

langfuse_handler = CallbackHandler()

search = DuckDuckGoSearchRun(name="web_search")

class Source(BaseModel):
    """
    Scheme for a source used by the agent
    """
    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """"
    Scheme for the agent response with answer and sources
    """
    answer: str = Field(description="The answer from the agent to the query")
    sources: List[Source] = Field(default_factory=list, description="list of sources used to generate the answer")

llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0, max_retries=3)
tools = [search]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {"messages": HumanMessage(content="Search for the recent jobs for AI engineers in Karachi on Linkedin and provide the answer with the sources.")},
        config={"callbacks": [langfuse_handler]}
    )
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
    get_client().flush()

    