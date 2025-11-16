"""
Financial CRAG System
"""

import os
import logging
from datetime import datetime, timedelta
from typing import TypedDict, List, Literal, Dict, Any

import yfinance as yf
from newsapi import NewsApiClient
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from tavily import TavilyClient
from langgraph.graph import StateGraph, END

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# STATE DEFINITION
# =============================================================================

class CRAGState(TypedDict):
    """CRAG workflow state"""
    question: str
    ticker: str
    documents: List[Document]
    web_results: str
    generation: str
    quality: Literal["correct", "ambiguous", "incorrect"]


# =============================================================================
# DATA EXTRACTION
# =============================================================================

class DataExtractor:
    """Yfinance + NewsAPI"""

    def __init__(self, newsapi_key: str):
        self.newsapi = NewsApiClient(api_key=newsapi_key)

    def get_stock_data(self, ticker: str) -> str:
        """Temel veriler"""
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1mo")

        data = f"""
            STOCK: {ticker} - {info.get('longName', 'N/A')}
            Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}
            
            CURRENT PRICE: ${info.get('currentPrice', 'N/A')}
            Market Cap: ${info.get('marketCap', 'N/A'):,} 
            P/E Ratio: {info.get('trailingPE', 'N/A')}
            Forward P/E: {info.get('forwardPE', 'N/A')}
            PEG Ratio: {info.get('pegRatio', 'N/A')}
            Beta: {info.get('beta', 'N/A')}
            
            52-WEEK RANGE: ${info.get('fiftyTwoWeekLow', 'N/A')} - ${info.get('fiftyTwoWeekHigh', 'N/A')}
            
            1-MONTH PERFORMANCE:
            Start: ${hist['Close'].iloc[0]:.2f}
            Current: ${hist['Close'].iloc[-1]:.2f}
            Return: {((hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100):.2f}%
            Avg Volume: {hist['Volume'].mean():,.0f}
            
            DESCRIPTION: {info.get('longBusinessSummary', 'N/A')[:500]}...
            """
        return data

    def get_news(self, ticker: str, days: int = 7) -> str:
        """Get News"""
        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            stock = yf.Ticker(ticker)
            company_name = stock.info.get('longName', ticker)

            articles = self.newsapi.get_everything(
                q=company_name,
                from_param=from_date,
                language='en',
                sort_by='relevancy',
                page_size=10
            )

            news_text = []
            for article in articles.get('articles', [])[:10]:
                news_text.append(
                    f"[{article['publishedAt'][:10]}] {article['title']}\n"
                    f"{article['description'] or article['content'][:200]}"
                )

            return "\n\n".join(news_text) if news_text else "No recent news."
        except:
            return "News unavailable."


# =============================================================================
# CRAG ASSESSMENT
# =============================================================================

class CRAGAssessor:
    """Quality Assessor"""

    ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are a document quality assessor for financial Q&A.

Evaluate if retrieved documents contain sufficient information to answer the question.

Return ONLY one word:
- "correct" = Documents fully answer the question
- "ambiguous" = Partial information, web search would help
- "incorrect" = Documents don't answer, web search required

Question: {question}
Documents: {documents}"""),
        ("human", "Assessment:")
    ])

    def __init__(self, llm: ChatOpenAI):
        self.chain = self.ASSESSMENT_PROMPT | llm | StrOutputParser()

    def assess(self, question: str, documents: List[Document]) -> str:
        docs_text = "\n\n".join([d.page_content[:500] for d in documents[:3]])
        result = self.chain.invoke({"question": question, "documents": docs_text})

        # Parse result
        result_lower = result.lower().strip()
        if "correct" in result_lower:
            return "correct"
        elif "incorrect" in result_lower:
            return "incorrect"
        else:
            return "ambiguous"


# =============================================================================
# CRAG WORKFLOW NODES
# =============================================================================

class CRAGWorkflow:
    """CRAG workflow nodes"""

    GENERATION_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "You are a financial analyst. Answer based on the context."),
        ("human", "Question: {question}\n\nContext: {context}")
    ])

    def __init__(self, vectorstore, llm: ChatOpenAI, assessor: CRAGAssessor, tavily_client=None):
        self.vectorstore = vectorstore
        self.llm = llm
        self.assessor = assessor
        self.tavily_client = tavily_client

    # NODE 1: Retrieve
    def retrieve(self, state: CRAGState) -> CRAGState:
        logger.info("🔍 Retrieving documents...")
        docs = self.vectorstore.similarity_search(state['question'], k=5)
        state["documents"] = docs
        return state

    # NODE 2: Assess Quality (CRAG CORE!)
    def assess(self, state: CRAGState) -> CRAGState:
        logger.info("⚖️ Assessing quality...")
        quality = self.assessor.assess(state['question'], state['documents'])
        state["quality"] = quality
        logger.info(f"  Quality: {quality.upper()}")
        return state

    # NODE 3: Web Search (conditionally executed)
    def web_search(self, state: CRAGState) -> CRAGState:
        logger.info("🌐 Searching web...")
        if not self.tavily_client:
            state["web_results"] = "[Web search disabled]"
            return state

        try:
            query = f"{state['ticker']} stock {state['question']}"
            response = self.tavily_client.search(query=query, max_results=3)

            state["web_results"] = "\n\n".join([
                f"{r.get('content', '')}" for r in response.get('results', [])
            ])
        except Exception as e:
            state["web_results"] = f"[Error: {e}]"

        return state

    # NODE 4: Generate Answer
    def generate(self, state: CRAGState) -> CRAGState:
        logger.info("🤖 Generating answer...")

        # Context based on quality
        if state["quality"] == "correct":
            # Use only local documents
            context = "\n\n".join([doc.page_content for doc in state["documents"]])
        else:
            # Combine local + web
            local = "\n\n".join([doc.page_content[:800] for doc in state["documents"][:2]])
            context = f"Local:\n{local}\n\nWeb:\n{state.get('web_results', '')}"

        # Generate
        chain = self.GENERATION_PROMPT | self.llm | StrOutputParser()
        state["generation"] = chain.invoke({
            "question": state["question"],
            "context": context
        })

        return state

    # ROUTING: Decide path after assessment
    def route(self, state: CRAGState) -> str:
        """CRAG routing logic"""
        return "generate" if state["quality"] == "correct" else "web_search"


# =============================================================================
# MAIN SYSTEM
# =============================================================================

class FinancialCRAG:
    """Main CRAG System"""

    def __init__(self):
        # API keys
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.newsapi_key = os.getenv("NEWSAPI_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")

        self.extractor = DataExtractor(self.newsapi_key)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.assessor = CRAGAssessor(self.llm)
        self.tavily = TavilyClient(api_key=tavily_key) if tavily_key else None

        self.vectorstore = None
        self.app = None

        logger.info("✅ FinancialCRAG initialized")

    def setup(self, ticker: str) -> None:
        """Setup for ticker"""
        logger.info(f"\n{'=' * 60}\nSETUP: {ticker}\n{'=' * 60}")

        # Collect data
        stock_data = self.extractor.get_stock_data(ticker)
        news_data = self.extractor.get_news(ticker)

        # Create documents
        documents = [
            Document(page_content=stock_data, metadata={"source": "yfinance", "ticker": ticker}),
            Document(page_content=news_data, metadata={"source": "news", "ticker": ticker})
        ]

        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
            collection_name="crag-compact"
        )

        # Build CRAG graph
        workflow = CRAGWorkflow(self.vectorstore, self.llm, self.assessor, self.tavily)
        graph = StateGraph(CRAGState)

        # Add nodes
        graph.add_node("retrieve", workflow.retrieve)
        graph.add_node("assess", workflow.assess)
        graph.add_node("web_search", workflow.web_search)
        graph.add_node("generate", workflow.generate)

        # Define flow
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "assess")
        graph.add_conditional_edges(
            "assess",
            workflow.route,
            {"web_search": "web_search", "generate": "generate"}
        )
        graph.add_edge("web_search", "generate")
        graph.add_edge("generate", END)

        self.app = graph.compile()
        logger.info("✅ Setup complete\n")

    def query(self, question: str, ticker: str = "") -> dict:
        """Execute query"""
        if not self.app:
            raise RuntimeError("Call setup() first")

        logger.info(f"\n{'=' * 60}\nQUERY: {question}\n{'=' * 60}")

        # Run CRAG workflow
        result = self.app.invoke({
            "question": question,
            "ticker": ticker,
            "documents": [],
            "web_results": "",
            "generation": "",
            "quality": "unknown"
        })

        return {
            "question": question,
            "answer": result["generation"],
            "quality": result["quality"],
            "used_web": bool(result.get("web_results"))
        }


# =============================================================================
# Main
# =============================================================================

def main():
    system = FinancialCRAG()
    system.setup("AAPL")

    questions = [
        "What is Apple's P/E ratio?",
        "Why did Apple stock move today?",
        "Should I buy Apple stock?"
    ]

    for q in questions:
        result = system.query(q, "AAPL")
        print(f"\nQ: {q}")
        print(f"Quality: {result['quality'].upper()} | Web: {'✅' if result['used_web'] else '❌'}")
        print(f"A: {result['answer']}\n" + "=" * 80)


if __name__ == "__main__":
    main()