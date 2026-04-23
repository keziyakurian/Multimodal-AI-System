import os
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain.schema import SystemMessage
from dotenv import load_dotenv

load_dotenv()

def _get_secret(key: str) -> str:
    """Reads a secret from st.secrets (Streamlit Cloud) or os.environ (local)."""
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, ""))
    except Exception:
        return os.getenv(key, "")

class AgenticEngine:
    def __init__(self, vector_db_query_fn):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=0,
            api_key=_get_secret("GROQ_API_KEY")
        )
        self.vector_db_query_fn = vector_db_query_fn
        
        # Define Tools
        @tool
        def search_documents(query: str) -> str:
            """Searches the document vault for information related to the query. 
            Use this for any questions about invoices, medical records, or banking documents."""
            return self.vector_db_query_fn(query)

        @tool
        def draft_email(recipient: str, subject: str, body: str) -> str:
            """Drafts a professional email based on the provided details. 
            Useful for the 'Secretary Agent' tasks like asking for discounts or follow-ups."""
            draft = f"To: {recipient}\nSubject: {subject}\n\n{body}"
            return f"DRAFT CREATED:\n---\n{draft}\n---\n(In a real scenario, this would be sent to an email API)."

        self.tools = [search_documents, draft_email]
        
        # Setup Agent
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an Agentic Voice Document Assistant. You help users reason across their documents and execute tasks like drafting emails. Always be professional and concise."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def run(self, user_input: str, chat_history: list = []):
        """Runs the agent on the user input."""
        try:
            response = self.agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            return response["output"]
        except Exception as e:
            return f"Agent Error: {str(e)}"
