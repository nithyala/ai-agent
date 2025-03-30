from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.embedder.openai import OpenAIEmbedder
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]="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"
openai_api_key = os.getenv("OPENAI_API_KEY")
agent = Agent(
    model=OpenAIChat(id="gpt-4o", api_key=openai_api_key),
    description="You are a Thai cuisine expert!",
    instructions=[
        "Search your knowledge base for Thai recipes.",
        "If the question is better suited for the web, search the web to fill in gaps.",
        "Prefer the information in your knowledge base over the web results."
    ],
    knowledge=PDFUrlKnowledgeBase(
        urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
        vector_db=LanceDb(
            uri="tmp/lancedb",
            table_name="recipes",
            search_type=SearchType.hybrid,
            embedder=OpenAIEmbedder(id="text-embedding-3-small", api_key=openai_api_key),
        ),
    ),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True
)

# Comment out after the knowledge base is loaded
if agent.knowledge is not None:
    agent.knowledge.load()

agent.print_response("How do I make chicken and galangal in coconut milk soup", stream=True)
agent.print_response("What is the history of Thai curry?", stream=True)
