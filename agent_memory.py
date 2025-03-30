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
if not os.environ.get("OPENAI_API_KEY"):
    st.error("❌ OpenAI API key not found. Please check your environment settings.")
    st.stop()
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
if query:
    try:
        # Check what .run() returns
        response = agent.run(query)
        st.write("🧪 Raw response type:", type(response))
        st.write("🧪 Raw response content:", response)

        # Try rendering nicely if it's a dict with 'output'
        if isinstance(response, dict) and "output" in response:
            st.markdown(response["output"])
        else:
            st.markdown(str(response))

    except Exception as e:
        st.error(f"⚠️ Error during response: {e}")
