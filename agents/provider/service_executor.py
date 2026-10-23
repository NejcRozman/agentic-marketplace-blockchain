"""Literature Analysis Agent for the agentic marketplace."""

import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Annotated, Sequence, Literal, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

try:
    from ..config import config, get_architecture
except ImportError:
    from config import Config, get_architecture
    config = Config()

logger = logging.getLogger(__name__)


class ResilientOpenAIEmbeddings(OpenAIEmbeddings):
    """OpenAI embeddings wrapper with retries for transient empty responses."""

    def __init__(
        self,
        *args,
        max_retries: int = 3,
        retry_delay_seconds: float = 0.5,
        cost_tracker: Optional[Any] = None,
        input_price_per_1k: float = 0.0,
        scope: str = "service_embedding",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._max_retries = max_retries
        self._retry_delay_seconds = retry_delay_seconds
        self._cost_tracker = cost_tracker
        self._input_price_per_1k = max(0.0, float(input_price_per_1k or 0.0))
        self._scope = str(scope or "service_embedding")

    @staticmethod
    def _is_retryable_embedding_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "no embedding data received" in message or "embedding" in message

    def _estimate_input_tokens(self, texts: List[str], chunk_size: Optional[int]) -> int:
        """Estimate billable embedding input tokens using the embedding tokenizer."""
        if not texts:
            return 0

        try:
            _chunk_size = int(chunk_size or self.chunk_size)
            _, _, _, token_counts = self._tokenize(texts, _chunk_size)
            return int(sum(max(0, int(count or 0)) for count in token_counts))
        except Exception as exc:
            logger.debug("Failed to estimate embedding tokens: %s", exc)
            return 0

    def _record_embedding_cost(self, input_tokens: int) -> None:
        """Record embedding token costs in the shared cost tracker when available."""
        if not self._cost_tracker or input_tokens <= 0 or self._input_price_per_1k <= 0.0:
            return

        total_cost = (input_tokens / 1000) * self._input_price_per_1k
        if total_cost <= 0.0:
            return

        self._cost_tracker.add_llm_cost(
            cost_usd=total_cost,
            context=f"{self._scope}:{self.model} ({input_tokens} tokens)",
            input_tokens=input_tokens,
            output_tokens=0,
        )

    def embed_documents(
        self,
        texts: List[str],
        chunk_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                embeddings = super().embed_documents(texts, chunk_size=chunk_size, **kwargs)
                input_tokens = self._estimate_input_tokens(texts, chunk_size=chunk_size)
                self._record_embedding_cost(input_tokens)
                return embeddings
            except Exception as exc:
                last_exc = exc
                if attempt >= self._max_retries or not self._is_retryable_embedding_error(exc):
                    raise
                logger.warning(
                    "Embedding request failed (attempt %s/%s): %s; retrying...",
                    attempt,
                    self._max_retries,
                    exc,
                )
                time.sleep(self._retry_delay_seconds * attempt)
        if last_exc:
            raise last_exc
        return []

    def embed_query(self, text: str) -> List[float]:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return super().embed_query(text)
            except Exception as exc:
                last_exc = exc
                if attempt >= self._max_retries or not self._is_retryable_embedding_error(exc):
                    raise
                logger.warning(
                    "Embedding query failed (attempt %s/%s): %s; retrying...",
                    attempt,
                    self._max_retries,
                    exc,
                )
                time.sleep(self._retry_delay_seconds * attempt)
        if last_exc:
            raise last_exc
        return []


class AgentState(TypedDict, total=False):
    """State for the LangGraph agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_calls_used: int
    retrieval_rounds_used: int
    budget_exhausted: bool


class ServiceExecutor:
    """
    Service Executor that performs RAG-based service execution.
    
    This agent:
    - Loads PDFs provided by clients
    - Creates vector database for semantic search
    - Performs literature analyses based on prompts
    - Provides properly cited responses
    """
    
    SYSTEM_PROMPT = """You are a literature analysis assistant specialized in analyzing research papers.
Your responsibilities:
1. Answer questions about research papers accurately and comprehensively
2. ALWAYS cite your sources using the format [SourceID] after each statement
3. When comparing papers, clearly indicate which paper each finding comes from
4. Summarize papers with proper citations
5. Generate literature analyses with consistent citation format


Citation Rules:
- Every factual statement must be followed by a citation
- Use the exact citation format provided by the search_literature tool
- Never rename, expand, or reinterpret a source ID
- Do not convert source IDs into author/title citations
- When multiple sources support a statement, list all citations
- Be precise about which paper each piece of information comes from

Primary Objective:
Maximize answer quality on these dimensions: completeness, depth, citations, clarity, and synthesis.

Grounding Rule (overrides everything else):
You are analyzing documents the client supplied. Your own recollection of these papers is
not evidence and must not be used. State a fact only if it appears in a passage returned by
search_literature in this conversation.
If you have not retrieved the passage that supports a claim, do not make the claim. Write
instead: NOT FOUND IN RETRIEVED CONTEXT: <what is missing>.
An answer that is partly "not found" is correct and expected when the search budget did not
surface the evidence. An answer that fills the gap from memory is wrong even if it is true.

Quality Rules:
Answer every part of the user prompt directly and explicitly.
Base every important claim on retrieved evidence from the papers.
Do not make claims that are not supported by retrieved evidence.
If evidence is missing or ambiguous, say so clearly instead of guessing.
Do not reward yourself for verbosity; prioritize substance, support, and structure.
When multiple papers are relevant, explicitly compare them: agreements, disagreements, differences in assumptions, methods, findings, and limitations.
Use the exact source IDs from the tool output and preserve them exactly in citations.

Search Policy:
Use search_literature to gather enough evidence to fully answer the prompt.
Never declare evidence missing while you still have search budget. If a required detail is not in
what you have retrieved, search again with different wording - the paper's own terms, the name of
the component, the metric, the section topic - and keep going until either you find it or the budget
is exhausted. Giving up early with budget left is a failure, not caution.
Continue searching when you still need missing evidence, cross-paper comparison, disagreement analysis, limitation analysis, or citation verification.
Do not spend search budget on near-duplicate searches once additional searches are no longer improving completeness, support, or synthesis.
Use remaining search budget to strengthen weakly supported parts of the answer, not to repeat already supported claims.

Required Answer Structure:
Direct answer to the prompt
Evidence and findings from the relevant sources
Cross-paper synthesis or comparison when applicable
Limitations, uncertainties, or open questions
A structured table comparing papers (authors, findings, methods, relevance)

Final Check Before Responding:
Every important factual claim has an exact citation in the form [SourceID].
Every part of the prompt has been answered.
Cross-paper agreements or disagreements are stated when relevant.
No unsupported claim remains.
The response is clear, concise, and well-structured.

- Do not finalize until you have searched for:
  1. direct evidence
  2. alternative supporting evidence
  3. contradictions or limitations
  4. cross-source comparisons
  5. citation verification
- If earlier searches are insufficient, reformulate the query rather than concluding early. If you don't have enough information always try to search for more information instead of guessing or concluding!
"""
    
    def __init__(
        self,
        agent_id: str,
        workspace_dir: Optional[str] = None,
        cost_tracker: Optional[Any] = None,
        architecture: Optional[str] = None
    ):
        """
        Initialize the Literature Analysis Agent.
        
        Args:
            agent_id: Unique identifier for this agent instance
            workspace_dir: Directory for storing RAG databases (optional)
            cost_tracker: Optional CostTracker instance for tracking LLM costs
            architecture: Architecture profile (uses config.architecture when omitted)
        """
        self.agent_id = agent_id
        self.workspace_dir = Path(workspace_dir or config.workspace_dir) / agent_id
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.cost_tracker = cost_tracker
        self.architecture = architecture or config.architecture
        self.arch_config = get_architecture(self.architecture)
        self.llm_model = getattr(config, "service_llm_override", None) or self.arch_config.service_llm
        
        # Initialize LLM and embeddings
        if not config.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        # Use config-based parameters (supports quality profiles)
        self.llm = ChatOpenAI(
            model=self.llm_model,
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
            temperature=config.rag_temperature
        )
        self.active_llm_model = self.llm_model
        self.active_input_price_per_1k = None
        self.active_output_price_per_1k = None
        self.embeddings = ResilientOpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
            max_retries=4,
            retry_delay_seconds=0.75,
            cost_tracker=self.cost_tracker,
            input_price_per_1k=float(getattr(config, "embedding_input_price_per_1k", 0.0)),
            scope="service_embedding",
        )
        
        # RAG parameters from config
        self.chunk_size = config.rag_chunk_size
        self.chunk_overlap = config.rag_chunk_overlap
        self.retrieval_k = config.rag_retrieval_k
        
        # RAG components (initialized when PDFs are loaded)
        self.vectorstore = None
        self.retriever = None
        self.current_pdf_directory = None
        
        # LangGraph workflow
        self.graph = None
        self.tools = []
        
        logger.info(f"Initialized ServiceExecutor: {agent_id} "
               f"(model={self.llm_model}, temp={config.rag_temperature}, k={config.rag_retrieval_k})")
    
    def _extract_paper_metadata(self, document: Any) -> Dict[str, str]:
        """Extract the filename-based source identity for a document."""
        metadata = document.metadata
        source_path = str(metadata.get("source", ""))
        source_file = os.path.basename(source_path)
        return {
            "source_id": Path(source_file).stem or "unknown_source",
            "source_file": source_file,
        }

    def load_pdfs(self, pdf_directory: str, force_rebuild: bool = False) -> bool:
        """
        Load PDFs and create/update vector database.
        
        Args:
            pdf_directory: Path to directory containing PDF files
            force_rebuild: Force rebuild of vector database even if it exists
            
        Returns:
            True if successful, False otherwise
        """
        try:
            pdf_dir = Path(pdf_directory)
            if not pdf_dir.exists():
                logger.error(f"PDF directory does not exist: {pdf_directory}")
                return False
            
            pdf_files = list(pdf_dir.glob("*.pdf"))
            if not pdf_files:
                logger.error(f"No PDF files found in {pdf_directory}")
                return False
            
            logger.info(f"Found {len(pdf_files)} PDF files")
            
            # Set up persist directory for this PDF collection
            collection_name = f"literature_analysis_{self.agent_id}"
            persist_directory = self.workspace_dir / "RAGfiles" / pdf_dir.name
            persist_directory.mkdir(parents=True, exist_ok=True)
            
            # Check if we need to rebuild
            needs_rebuild = force_rebuild or not (persist_directory / "chroma.sqlite3").exists()
            
            if needs_rebuild:
                logger.info("Building vector database from PDFs...")
                
                # Load PDFs
                loader = PyPDFDirectoryLoader(str(pdf_dir))
                documents = loader.load()
                logger.info(f"Loaded {len(documents)} pages from PDFs")
                
                # Extract and add metadata
                for doc in documents:
                    paper_metadata = self._extract_paper_metadata(doc)
                    doc.metadata.update(paper_metadata)
                
                # Split documents using config-based parameters
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                splits = text_splitter.split_documents(documents)
                logger.info(f"Created {len(splits)} text chunks")
                
                # Create vector database
                self.vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=self.embeddings,
                    collection_name=collection_name,
                    persist_directory=str(persist_directory)
                )
                logger.info("Vector database created successfully")
            else:
                logger.info("Loading existing vector database...")
                self.vectorstore = Chroma(
                    collection_name=collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=str(persist_directory)
                )
                logger.info("Loaded existing vector database")
            
            # Create retriever using config-based parameters
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.retrieval_k}
            )
            
            self.current_pdf_directory = pdf_directory
            
            # Rebuild LangGraph with the search tool
            self._build_graph()
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading PDFs: {e}", exc_info=True)
            return False
    
    def _apply_effort_model(self, effort_budget: Dict[str, Any]) -> None:
        """Swap the execution model for this run when the tier names one.

        Kept off the constructor deliberately: the tier is known per request, not per
        provider, and a provider that sells three tiers runs all three.
        """
        self.active_input_price_per_1k = effort_budget.get("input_price_per_1k")
        self.active_output_price_per_1k = effort_budget.get("output_price_per_1k")

        model = effort_budget.get("model")
        if not model or model == self.active_llm_model:
            return

        self.active_llm_model = str(model)
        self.llm = ChatOpenAI(
            model=self.active_llm_model,
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
            temperature=config.rag_temperature,
        )
        logger.info("Effort tier model: %s", self.active_llm_model)

    def _apply_retrieval_k(self, retrieval_k: int) -> None:
        """Set how many passages each search returns for this execution.

        Effort is an evidence budget, not just a search count: with a fixed number of
        passages per search, two searches saturate quality at every task depth.
        Rebuilding the retriever per run lets the tier move both factors.
        """
        if not self.vectorstore:
            return

        k = max(1, int(retrieval_k or self.retrieval_k))
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def _create_search_tool(self):
        """Create the search_literature tool with access to the retriever."""
        retriever = self.retriever
        
        @tool
        def search_literature(query: str) -> str:
            """
            Search through research papers to find relevant information.
            Returns content with source IDs indicating which paper each piece of information comes from.
            Use this tool to answer questions about research papers, find specific information,
            compare findings, or gather information for literature analyses.
            """
            docs = retriever.invoke(query)
            
            if not docs:
                return "No relevant information found in the research papers."
            
            # Format results with citations
            results = []
            for i, doc in enumerate(docs, 1):
                source_id = doc.metadata.get('source_id')
                if not source_id:
                    source_id = Path(str(doc.metadata.get('source', ''))).stem or 'unknown_source'
                
                citation = f"[{source_id}]"
                content = doc.page_content.strip()
                
                results.append(f"Source {i} {citation}:\n{content}\n")
            
            return "\n---\n".join(results)
        
        return search_literature

    @staticmethod
    def _extract_response_text(content: Any) -> str:
        """Extract text payloads from heterogeneous message content shapes.

        Some model/providers return a plain string while others return a list of
        structured blocks. This method collects all text-like fields and joins
        them. If no text is present, it returns an empty string.
        """
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                extracted = ServiceExecutor._extract_response_text(item)
                if isinstance(extracted, str) and extracted.strip():
                    parts.append(extracted)
            return "\n".join(parts).strip() if parts else ""

        if isinstance(content, dict):
            parts: List[str] = []

            for key in ("text", "content", "output_text"):
                value = content.get(key)
                extracted = ServiceExecutor._extract_response_text(value)
                if isinstance(extracted, str) and extracted.strip():
                    parts.append(extracted)

            message = content.get("message")
            if not parts and message is not None:
                extracted = ServiceExecutor._extract_response_text(message)
                if isinstance(extracted, str) and extracted.strip():
                    parts.append(extracted)

            return "\n".join(parts).strip() if parts else ""

        text_attr = getattr(content, "text", None)
        if isinstance(text_attr, str):
            return text_attr

        content_attr = getattr(content, "content", None)
        if content_attr is not None:
            extracted = ServiceExecutor._extract_response_text(content_attr)
            if isinstance(extracted, str):
                return extracted

        return ""

    def _build_graph(
        self,
        max_tool_calls: Optional[int] = None,
        max_retrieval_rounds: Optional[int] = None,
    ):
        """
        Build the LangGraph workflow.
        """
        if not self.retriever:
            logger.warning("Cannot build graph without retriever. Load PDFs first.")
            return
        
        active_llm = self.llm
        
        # Create tools
        search_tool = self._create_search_tool()
        self.tools = [search_tool]
        
        # Bind tools to active LLM
        llm_with_tools = active_llm.bind_tools(self.tools)
        
        # Create callback if cost tracking enabled
        callbacks = None
        if self.cost_tracker:
            try:
                from ..common.cost_tracker import LLMCostCallback
            except ImportError:
                from common.cost_tracker import LLMCostCallback

            callback = LLMCostCallback(
                cost_tracker=self.cost_tracker,
                model=active_llm.model_name,  # Use active LLM's model
                input_price_per_1k=float(
                    self.active_input_price_per_1k
                    if self.active_input_price_per_1k is not None
                    else config.service_llm_input_price_per_1k
                ),
                output_price_per_1k=float(
                    self.active_output_price_per_1k
                    if self.active_output_price_per_1k is not None
                    else config.service_llm_output_price_per_1k
                ),
                scope="service_execution",
            )
            callbacks = [callback]
        
        # Define graph nodes
        def call_llm(state: AgentState) -> AgentState:
            """Call the LLM with tools and system prompt."""
            messages = state['messages']
            
            # Add system prompt if not already present
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=self.SYSTEM_PROMPT)] + messages
            
            # Invoke with callbacks if provided
            if callbacks:
                response = llm_with_tools.invoke(messages, config={"callbacks": callbacks})
            else:
                response = llm_with_tools.invoke(messages)
            return {'messages': [response]}
        
        def should_continue(state: AgentState) -> Literal["tools", "end"]:
            """Determine if we should use tools or end."""
            last_message = state['messages'][-1]

            if not (hasattr(last_message, 'tool_calls') and last_message.tool_calls):
                return "end"

            if bool(state.get("budget_exhausted", False)):
                return "end"

            return "tools"
        
        def call_tools(state: AgentState) -> AgentState:
            """Execute tool calls and return results."""
            last_message = state['messages'][-1]
            tool_calls = last_message.tool_calls
            
            tools_dict = {tool.name: tool for tool in self.tools}
            
            results = []
            tool_calls_used = int(state.get("tool_calls_used", 0) or 0)
            retrieval_rounds_used = int(state.get("retrieval_rounds_used", 0) or 0)
            remaining_tool_calls = None
            remaining_retrieval_rounds = None

            if max_tool_calls is not None:
                remaining_tool_calls = max(0, max_tool_calls - tool_calls_used)

            if max_retrieval_rounds is not None:
                remaining_retrieval_rounds = max(0, max_retrieval_rounds - retrieval_rounds_used)

            tool_call_count = 0
            retrieval_round_count = 0
            budget_exhausted = bool(state.get("budget_exhausted", False))
            for tool_call in tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                is_retrieval_tool = tool_name == "search_literature"

                allow_tool_call = True
                if remaining_tool_calls is not None and tool_call_count >= remaining_tool_calls:
                    allow_tool_call = False
                if (
                    allow_tool_call
                    and is_retrieval_tool
                    and remaining_retrieval_rounds is not None
                    and retrieval_round_count >= remaining_retrieval_rounds
                ):
                    allow_tool_call = False
                
                if not allow_tool_call:
                    budget_exhausted = True
                    result = (
                        "Skipped tool call because the per-prompt tool/retrieval budget "
                        "was exhausted. Produce the best final answer from the evidence already retrieved."
                    )
                elif tool_name not in tools_dict:
                    result = f"Error: Tool {tool_name} not found"
                else:
                    tool = tools_dict[tool_name]
                    result = tool.invoke(tool_args)
                
                tool_message = ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call['id'],
                    name=tool_name
                )
                results.append(tool_message)

                if not allow_tool_call:
                    logger.info(
                        "Skipping tool call '%s' due to prompt budget: remaining_tool_calls=%s, remaining_retrieval_rounds=%s",
                        tool_name,
                        max(0, (remaining_tool_calls or 0) - tool_call_count) if remaining_tool_calls is not None else "unbounded",
                        max(0, (remaining_retrieval_rounds or 0) - retrieval_round_count) if remaining_retrieval_rounds is not None else "unbounded",
                    )
                    continue

                tool_call_count += 1
                if is_retrieval_tool:
                    retrieval_round_count += 1

            return {
                'messages': results,
                'tool_calls_used': int(state.get("tool_calls_used", 0) or 0) + tool_call_count,
                'retrieval_rounds_used': int(state.get("retrieval_rounds_used", 0) or 0) + retrieval_round_count,
                'budget_exhausted': budget_exhausted,
            }
        
        # Build graph
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", call_llm)
        workflow.add_node("tools", call_tools)
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "end": END
            }
        )
        workflow.add_edge("tools", "agent")
        workflow.set_entry_point("agent")
        
        self.graph = workflow.compile()
        logger.info("LangGraph workflow built successfully")

    @staticmethod
    def _normalize_effort_tier(effort_tier: Optional[str]) -> Optional[str]:
        """Normalize effort names to strict low/medium/high tiers."""
        if effort_tier is None:
            return None

        normalized = str(effort_tier).strip().lower()
        if normalized in {"low", "medium", "high"}:
            return normalized
        return None

    def perform_analysis(
        self,
        pdf_directory: str,
        prompts: List[str],
        force_rebuild: bool = False,
        effort_tier: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform a literature analysis based on provided PDFs and prompts.
        
        Args:
            pdf_directory: Path to directory containing PDF files
            prompts: List of questions/prompts for the literature analysis
            force_rebuild: Force rebuild of vector database
            effort_tier: Optional effort tier for execution budget selection ("low", "medium", "high")
            
        Returns:
            Dictionary containing analysis results:
            {
                "success": bool,
                "agent_id": str,
                "pdf_directory": str,
                "prompts": List[str],
                "responses": List[Dict[str, Any]],
                # [{"prompt", "response", "retrieval_rounds_used", "tool_calls_used",
                #   "budget_exhausted", "usage_measured", "prompt_duration_seconds"}]
                "retrieval_rounds_total": Optional[int],
                "error": Optional[str]
            }

        ``retrieval_rounds_used`` is how many times the agent went back to the
        documents for *this one prompt* (each prompt gets a fresh state with the
        counter at 0). It is ``None`` when the value could not be observed rather than 0, because a
        silent 0 is indistinguishable from "no digging" and would poison any
        regression fitted on it; ``usage_measured`` says which case applies.
        """
        result = {
            "success": False,
            "agent_id": self.agent_id,
            "pdf_directory": pdf_directory,
            "prompts": prompts,
            "responses": [],
            "retrieval_rounds_total": None,
            "error": None,
            "partial_fallback_used": False
        }
        
        try:
            normalized_effort_tier = self._normalize_effort_tier(effort_tier)
            if effort_tier and normalized_effort_tier is None:
                logger.warning(f"Unknown effort tier '{effort_tier}', using default effort tier")

            resolved_effort_tier = normalized_effort_tier or config.default_effort_tier
            effort_budget = config.get_service_execution_budget_for_effort(resolved_effort_tier)
            logger.info(
                "Using effort tier '%s' (input='%s'): max_tool_calls=%s, max_retrieval_rounds=%s, retrieval_k=%s, model=%s",
                effort_budget["effort_tier"],
                effort_tier,
                effort_budget["max_tool_calls"],
                effort_budget["max_retrieval_rounds"],
                effort_budget.get("retrieval_k"),
                self.llm_model,
            )

            # Load PDFs if not already loaded or if directory changed
            if self.current_pdf_directory != pdf_directory or force_rebuild:
                if not self.load_pdfs(pdf_directory, force_rebuild):
                    result["error"] = "Failed to load PDFs"
                    return result
            
            # Model and retriever first: the graph closes over both when it is built.
            self._apply_effort_model(effort_budget)
            self._apply_retrieval_k(effort_budget.get("retrieval_k", self.retrieval_k))

            # IMPORTANT: Rebuild graph for each service to prevent context accumulation.
            self._build_graph(
                max_tool_calls=effort_budget["max_tool_calls"],
                max_retrieval_rounds=effort_budget["max_retrieval_rounds"],
            )
            
            if not self.graph:
                result["error"] = "Graph not initialized. Load PDFs first."
                return result
            
            # Process each prompt
            recursion_limit = int(getattr(config, "service_recursion_limit", 25))
            for prompt in prompts:
                logger.info(f"Processing prompt: {prompt[:100]}...")

                # Create initial state with the prompt
                initial_state = {
                    "messages": [HumanMessage(content=prompt)],
                    "tool_calls_used": 0,
                    "retrieval_rounds_used": 0,
                    "budget_exhausted": False,
                }

                # Run the graph (bounded to avoid infinite tool loops)
                prompt_started = time.perf_counter()
                try:
                    graph_result = self.graph.invoke(
                        initial_state,
                        config={"recursion_limit": recursion_limit}
                    )
                except Exception as e:
                    err_text = str(e)
                    if "GRAPH_RECURSION_LIMIT" in err_text or "Recursion limit" in err_text:
                        logger.warning(
                            "Service reasoning recursion limit reached for one prompt; "
                            "falling back to retrieval-only answer"
                        )
                        response_text = self._retrieval_only_fallback(prompt)
                        # The graph raised, so its state never came back: usage is
                        # unknown, not zero. Recording 0 here would look like a
                        # trivially easy prompt when it was in fact the hardest.
                        result["responses"].append({
                            "prompt": prompt,
                            "response": response_text,
                            "retrieval_rounds_used": None,
                            "tool_calls_used": None,
                            "budget_exhausted": None,
                            "usage_measured": False,
                            "prompt_duration_seconds": round(time.perf_counter() - prompt_started, 3),
                        })
                        result["partial_fallback_used"] = True
                        continue
                    else:
                        raise

                # Extract the final response
                final_message = graph_result['messages'][-1]

                logger.info(
                    "Prompt usage: tool_calls_used=%s, retrieval_rounds_used=%s",
                    graph_result.get("tool_calls_used", 0),
                    graph_result.get("retrieval_rounds_used", 0),
                )

                # Extract text robustly across provider/model content shapes.
                response_text = self._extract_response_text(final_message.content)

                if not response_text.strip():
                    logger.warning(
                        "Empty response text extracted for prompt: %s",
                        prompt[:100],
                    )
                
                result["responses"].append({
                    "prompt": prompt,
                    "response": response_text,
                    "retrieval_rounds_used": int(graph_result.get("retrieval_rounds_used", 0) or 0),
                    "tool_calls_used": int(graph_result.get("tool_calls_used", 0) or 0),
                    "budget_exhausted": bool(graph_result.get("budget_exhausted", False)),
                    "usage_measured": True,
                    "prompt_duration_seconds": round(time.perf_counter() - prompt_started, 3),
                })
            
            measured = [
                r["retrieval_rounds_used"] for r in result["responses"]
                if r.get("usage_measured") and r.get("retrieval_rounds_used") is not None
            ]
            # Total over measured prompts only. Left None when nothing was
            # measured so a consumer cannot mistake "unobserved" for "zero".
            result["retrieval_rounds_total"] = sum(measured) if measured else None
            result["success"] = True
            logger.info(f"Literature analysis completed successfully with {len(prompts)} prompts")
            
        except Exception as e:
            logger.error(f"Error performing analysis: {e}", exc_info=True)
            result["error"] = str(e)
        
        return result

    def _retrieval_only_fallback(self, prompt: str) -> str:
        """Build a deterministic answer from retriever output when agent reasoning loops."""
        try:
            if not self.retriever:
                return (
                    "Unable to complete full reasoning for this prompt due to recursion limits, "
                    "and no retriever is available for fallback."
                )

            docs = self.retriever.invoke(prompt)
            if not docs:
                return (
                    "Unable to complete full reasoning for this prompt due to recursion limits, "
                    "and no relevant sources were retrieved."
                )

            lines = []
            for i, doc in enumerate(docs[:1], 1):
                snippet = " ".join(str(doc.page_content).split())
                if len(snippet) > 1500:
                    snippet = snippet[:1500] + "..."
                source_id = doc.metadata.get('source_id')
                if not source_id:
                    source_id = Path(str(doc.metadata.get('source', ''))).stem or 'unknown_source'
                lines.append(f"{i}. [{source_id}] {snippet}")

            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"Retrieval-only fallback failed: {e}")
            return (
                "Unable to complete full reasoning for this prompt due to recursion limits. "
                f"Fallback also failed: {e}"
            )
