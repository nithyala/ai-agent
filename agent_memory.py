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
# App title
# ✅ App title
import streamlit as st

st.title("🍜 Thai Cuisine Expert")


# ✅ Show initial diagnostic message
st.markdown("### 🛠️ App is running...")

# ✅ Ensure LanceDB directory exists
os.makedirs("tmp/lancedb", exist_ok=True)

# ✅ Load Agent inside try block
try:
    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
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
                embedder=OpenAIEmbedder(id="text-embedding-3-small"),
            ),
        ),
        tools=[DuckDuckGoTools()],
        show_tool_calls=True,
        markdown=True
    )

    if agent.knowledge is not None:
        agent.knowledge.load()

    st.success("✅ Agent loaded successfully!")

except Exception as e:
    st.error(f"⚠️ Error loading agent: {e}")
    st.stop()  # Stop execution if agent fails to load

# ✅ User input
query = st.text_input("Ask a Thai food question:")

# ✅ Handle response
if query:
    try:
        response = agent.chat(query)
        st.markdown(response)
    except Exception as e:
        st.error(f"⚠️ Error during response: {e}")
