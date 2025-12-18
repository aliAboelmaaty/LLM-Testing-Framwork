"""
LLM Service Module

Handles communication with multiple LLM providers:
- Gemini (via Files API with resumable upload)
- Replicate (Gemma3, DeepSeek, GPT-5, etc.)

Features:
- PDF processing with multiple fallback strategies
- OCR support for image-based PDFs
- Citation enforcement and grounding verification
- Resilient error handling with retries
"""

from typing import Optional, Dict, Any, List, Tuple, Iterable
import mimetypes
import time
import os
import io
import re
import requests
import json

# Import from core (avoid circular imports by importing only what's needed)
from .core import LLMProvider, ContextMode


# ================= PDF Extraction Cache =================

# Global cache for PDF extraction and chunking
# Key: (manual_path, chunk_size) -> (full_text, chunks)
# This prevents repeated extraction of the same PDF across models/repetitions
_PDF_CACHE: Dict[Tuple[str, int], Tuple[str, List[str]]] = {}


# ================= Utilities =================

def _safe_imports():
    """Safely import optional dependencies"""
    mods = {}
    try:
        import pypdf as _pypdf
        mods["pypdf"] = _pypdf
    except Exception:
        mods["pypdf"] = None
    try:
        from PyPDF2 import PdfReader as _PdfReader
        mods["PyPDF2"] = _PdfReader
    except Exception:
        mods["PyPDF2"] = None
    try:
        import pytesseract as _pytesseract
        mods["pytesseract"] = _pytesseract
    except Exception:
        mods["pytesseract"] = None
    try:
        from pdf2image import convert_from_path as _convert_from_path
        mods["convert_from_path"] = _convert_from_path
    except Exception:
        mods["convert_from_path"] = None
    try:
        from PIL import Image as _PILImage
        mods["PIL"] = True
    except Exception:
        mods["PIL"] = None
    return mods


def _chunk_text(s: str, max_chars: int) -> Iterable[str]:
    """Yield chunks <= max_chars, cut on paragraph boundaries when possible."""
    if len(s) <= max_chars:
        yield s
        return
    paras = s.split("\n\n")
    buf = []
    cur = 0
    for p in paras:
        p2 = (p + "\n\n")
        if cur + len(p2) <= max_chars:
            buf.append(p2)
            cur += len(p2)
        else:
            if buf:
                yield "".join(buf)
            if len(p2) > max_chars:
                start = 0
                while start < len(p2):
                    yield p2[start:start+max_chars]
                    start += max_chars
                buf, cur = [], 0
            else:
                buf, cur = [p2], len(p2)
    if buf:
        yield "".join(buf)


def get_contexts(
    manual_path: str,
    context_mode: ContextMode,
    retrieval_query: str,
    top_k: int = 6,
    chunk_size: int = 2000,
    max_context_chars: int = 40000,
    manual_full_max_chunks: int = 40
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Get contexts for a given manual and context mode with caching.

    Implements PDF extraction caching and handles all three context modes:
    - BASELINE: Returns empty list
    - MANUAL_FULL: Returns all chunks (up to manual_full_max_chunks or max_context_chars)
    - RAG_RETRIEVAL: Returns top-k retrieved chunks (up to max_context_chars)

    Args:
        manual_path: Path to PDF manual
        context_mode: Context mode (BASELINE, MANUAL_FULL, or RAG_RETRIEVAL)
        retrieval_query: Query for retrieval (NOT full prompt, short query only)
        top_k: Number of chunks to retrieve for RAG
        chunk_size: Size of chunks in characters
        max_context_chars: Maximum total context characters (default: 40000)
        manual_full_max_chunks: Maximum chunks for MANUAL_FULL mode (default: 40)

    Returns:
        Tuple of (contexts_list, metadata_dict)
        metadata includes: cache_hit, total_chunks, chunks_used, truncated, context_chars
    """
    metadata = {
        "cache_hit": False,
        "total_chunks": 0,
        "chunks_used": 0,
        "truncated": False,
        "context_chars": 0
    }

    if context_mode == ContextMode.BASELINE:
        return [], metadata

    # Check cache first
    cache_key = (manual_path, chunk_size)
    if cache_key in _PDF_CACHE:
        full_text, chunks = _PDF_CACHE[cache_key]
        metadata["cache_hit"] = True
        metadata["total_chunks"] = len(chunks)
        print(f"[CACHE HIT] PDF: {os.path.basename(manual_path)} ({len(chunks)} chunks)")
    else:
        # Extract and chunk PDF
        print(f"[CACHE MISS] Extracting PDF: {os.path.basename(manual_path)}")
        full_text = _extract_pdf_text_locally(manual_path, max_chars=700_000, do_ocr=True)

        if full_text.startswith("[[PDF TEXT EXTRACTION"):
            # Extraction failed - cache state becomes irrelevant/untrustworthy
            print(f"[ERROR] PDF extraction failed for {os.path.basename(manual_path)}")
            # TRI-STATE: Set cache_hit to None (cache was checked, but operation failed)
            metadata["cache_hit"] = None
            return [], metadata

        chunks = list(_chunk_text(full_text, max_chars=chunk_size))
        metadata["total_chunks"] = len(chunks)

        # Cache the result
        _PDF_CACHE[cache_key] = (full_text, chunks)
        print(f"[CACHE STORED] Cached {len(chunks)} chunks for {os.path.basename(manual_path)}")

    if context_mode == ContextMode.MANUAL_FULL:
        # MANUAL_FULL: Return chunks in order, stopping when we hit either limit:
        # 1. manual_full_max_chunks (chunk count limit)
        # 2. max_context_chars (character limit)
        selected_chunks = []
        total_chars = 0

        for i, chunk in enumerate(chunks):
            # Check if adding this chunk would exceed limits
            if i >= manual_full_max_chunks:
                metadata["truncated"] = True
                print(f"[WARNING] MANUAL_FULL truncated by chunk limit: {len(chunks)} -> {manual_full_max_chunks} chunks")
                break

            if total_chars + len(chunk) > max_context_chars:
                metadata["truncated"] = True
                print(f"[WARNING] MANUAL_FULL truncated by char limit: {total_chars} -> {max_context_chars} chars")
                break

            selected_chunks.append(chunk)
            total_chars += len(chunk)

        metadata["chunks_used"] = len(selected_chunks)
        metadata["context_chars"] = total_chars
        return selected_chunks, metadata

    elif context_mode == ContextMode.RAG_RETRIEVAL:
        # Retrieve top-k chunks using retrieval_query (NOT full prompt)
        from .retrieval import retrieve_chunks
        retrieved = retrieve_chunks(retrieval_query, chunks, top_k=top_k, method="bm25")

        # Truncate retrieved chunks to max_context_chars if needed
        selected_chunks = []
        total_chars = 0

        for chunk in retrieved:
            if total_chars + len(chunk) > max_context_chars:
                metadata["truncated"] = True
                print(f"[WARNING] RAG_RETRIEVAL truncated by char limit: {len(retrieved)} chunks -> {len(selected_chunks)} chunks ({total_chars} chars)")
                break
            selected_chunks.append(chunk)
            total_chars += len(chunk)

        metadata["chunks_used"] = len(selected_chunks)
        metadata["context_chars"] = total_chars
        print(f"[RETRIEVAL] Retrieved {len(selected_chunks)} chunks ({total_chars} chars) using query: '{retrieval_query[:50]}...'")
        return selected_chunks, metadata

    else:
        raise ValueError(f"Unknown context_mode: {context_mode}")


def _extract_pdf_text_locally(pdf_path: str, max_chars: int = 120_000, do_ocr: bool = True) -> str:
    """
    Extract text from PDF using pypdf/PyPDF2 with page markers.
    Falls back to OCR if text extraction returns empty.

    Args:
        pdf_path: Path to PDF file
        max_chars: Maximum characters to extract
        do_ocr: Whether to use OCR fallback for image-based PDFs

    Returns:
        Extracted text with page markers "[p.1] ... [p.2] ..." (with [[TRUNCATED]] marker if needed)
    """
    mods = _safe_imports()

    # 1) Text-native extraction with page markers
    try:
        text_pages: List[str] = []
        if mods["pypdf"] is not None:
            reader = mods["pypdf"].PdfReader(pdf_path)
            for page_num, pg in enumerate(reader.pages, start=1):
                page_text = pg.extract_text() or ""
                if page_text.strip():
                    text_pages.append(f"[p.{page_num}] {page_text}")
        elif mods["PyPDF2"] is not None:
            reader = mods["PyPDF2"](pdf_path)
            for page_num, pg in enumerate(reader.pages, start=1):
                page_text = pg.extract_text() or ""
                if page_text.strip():
                    text_pages.append(f"[p.{page_num}] {page_text}")
        else:
            text_pages = []
        joined = "\n".join(text_pages).strip()
        if joined:
            return joined[:max_chars] + ("\n[[TRUNCATED]]" if len(joined) > max_chars else "")
    except Exception:
        pass

    # 2) OCR fallback with page markers
    if do_ocr:
        try:
            if mods["convert_from_path"] is None or mods["pytesseract"] is None or not mods["PIL"]:
                return "[[PDF TEXT EXTRACTION RETURNED EMPTY (no OCR available)]]"
            images = mods["convert_from_path"](pdf_path, dpi=200, fmt="png")
            ocr_texts: List[str] = []
            for page_num, img in enumerate(images, start=1):
                txt = mods["pytesseract"].image_to_string(img) or ""
                if txt.strip():
                    ocr_texts.append(f"[p.{page_num}] {txt}")
                if sum(len(x) for x in ocr_texts) >= max_chars:
                    break
            ocr_joined = "\n".join(ocr_texts).strip() if ocr_texts else ""
            if not ocr_joined:
                return "[[PDF TEXT EXTRACTION RETURNED EMPTY (OCR produced no text)]]"
            return ocr_joined[:max_chars] + ("\n[[TRUNCATED]]" if len(ocr_joined) > max_chars else "")
        except Exception as e:
            return f"[[PDF TEXT EXTRACTION FAILED (OCR path): {e}]]"

    return "[[PDF TEXT EXTRACTION RETURNED EMPTY]]"


# ================= Grounding & Citation Checking =================

_GROUNDING_RED_FLAGS = [
    r"\baccording to (typical|general|common)\b",
    r"\b(in|under) (most|many|typical|general) (cases|conditions)\b",
    r"\bgenerally\b",
    r"\btypically\b",
    r"\bas a rule\b",
    r"\bindustry standard(s)?\b",
]

_CITATION_HINTS = [
    r"\(p\.\s*\d+\)",  # (p. 12)
    r"\bpage[s]?\s*\d+\b",  # page 12
    r"\bsection\s*\d+(\.\d+)*\b"  # Section 1.2
]


def _looks_ungrounded(answer: str) -> bool:
    """Check if answer appears to lack grounding in provided context"""
    if not answer or not answer.strip():
        return True
    a = answer.strip().lower()
    for pat in _GROUNDING_RED_FLAGS:
        if re.search(pat, a):
            return True
    if len(a) > 300 and not any(re.search(p, answer, flags=re.I) for p in _CITATION_HINTS):
        return True
    return False


# ================= Main LLM Service Class =================

class LLMService:
    """
    Service for interacting with multiple LLM providers.

    Supports:
    - Gemini (via Files API)
    - Replicate models (Gemma3, DeepSeek, GPT-5)
    - PDF processing with OCR fallback
    - Citation enforcement
    """

    def __init__(
        self,
        replicate_api_token: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-2.5-flash"
    ):
        """
        Initialize LLM Service.

        Args:
            replicate_api_token: Replicate API token
            gemini_api_key: Google Gemini API key
            gemini_model: Gemini model to use (default: gemini-2.5-flash)
        """
        # Load from environment if not provided
        if replicate_api_token is None:
            replicate_api_token = os.getenv("REPLICATE_API_TOKEN")
        if gemini_api_key is None:
            gemini_api_key = os.getenv("GEMINI_API_KEY")

        self.replicate_api_token = replicate_api_token
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model

        # Initialize Replicate client
        self.replicate_client = None
        if self.replicate_api_token:
            try:
                import replicate
                self.replicate_client = replicate.Client(api_token=self.replicate_api_token)
            except ImportError:
                print("Warning: replicate package not installed. Replicate providers will not work.")

        # Gemini configuration
        self.gemini_configured = bool(self.gemini_api_key)
        if self.gemini_configured:
            self.gemini_api_url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent"
                f"?key={self.gemini_api_key}"
            )

        # Model slugs for Replicate
        # Gemma3 family (Google DeepMind)
        self.gemma3_4b_it = "google-deepmind/gemma-3-4b-it:00139d2960396352b671f7b5c2ece5313bf6d45fe0a052efe14f023d2a81e196"
        self.gemma3_12b_it = "google-deepmind/gemma-3-12b-it:5a0df3fa58c87fbd925469a673fdb16f3dd08e6f4e2f1a010970f07b7067a81c"
        self.gemma3_27b_it = "google-deepmind/gemma-3-27b-it:c0f0aebe8e578c15a7531e08a62cf01206f5870e9d0a67804b8152822db58c54"
        # Other models
        self.deepseek_model = "deepseek-ai/deepseek-r1"
        self.gpt5_model = "openai/gpt-5"

    def _post_with_backoff(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        tries: int = 3,
        base: float = 0.8
    ) -> requests.Response:
        """Make HTTP POST with exponential backoff retry, including rate limit handling"""
        for attempt in range(tries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=90)

                # Handle rate limiting specifically (429 Too Many Requests)
                if resp.status_code == 429:
                    if attempt < tries - 1:
                        # Wait longer for rate limits (60 seconds base, exponential)
                        wait_time = 60 * (2 ** attempt)
                        print(f"Rate limited (429). Waiting {wait_time}s before retry {attempt + 1}/{tries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise RuntimeError(f"Rate limited after {tries} attempts. Try again later or get a new API key.")

                resp.raise_for_status()
                return resp
            except requests.exceptions.HTTPError as e:
                if attempt == tries - 1:
                    raise
                time.sleep(base * (2 ** attempt))
            except Exception as e:
                if attempt == tries - 1:
                    raise
                time.sleep(base * (2 ** attempt))
        raise RuntimeError("Unreachable")

    def _normalize_replicate_output(self, out: Any) -> str:
        """Normalize Replicate output to string"""
        if isinstance(out, list):
            return "".join(map(str, out))
        return str(out)

    def ask(
        self,
        question: str,
        provider: LLMProvider = LLMProvider.GEMINI,
        system_prompt: Optional[str] = None,
        temperature: float = 0.4,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Ask a question without RAG (baseline mode).

        Args:
            question: Question to ask
            provider: LLM provider to use
            system_prompt: Optional system prompt
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter (1.0 = disabled, lower = more focused)
            seed: Optional random seed for reproducibility (provider support varies)
            **kwargs: Additional provider-specific arguments

        Returns:
            Dictionary with 'answer' key and optional metadata
        """
        if provider == LLMProvider.GEMINI:
            # Gemini: pass top_p via kwargs to generation_config
            return self._gemini_call(question, system_prompt, temperature, max_tokens, top_p=top_p)
        elif provider in (LLMProvider.GEMMA3_4B, LLMProvider.GEMMA3_12B, LLMProvider.GEMMA3_27B,
                          LLMProvider.GEMMA3, LLMProvider.DEEPSEEK, LLMProvider.GPT5):
            # Inject generation params into kwargs for Replicate
            kwargs.setdefault("temperature", temperature)
            kwargs.setdefault("max_tokens", max_tokens)
            kwargs.setdefault("top_p", top_p)
            if seed is not None:
                kwargs.setdefault("seed", seed)
            return self._replicate_call(question, provider, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def ask_with_pdf(
        self,
        question: str,
        pdf_path: str,
        provider: LLMProvider = LLMProvider.GEMINI,
        stream: bool = False,
        system_prompt: Optional[str] = None,
        temperature: float = 0.4,
        max_tokens: int = 8192,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Ask a question about a PDF document.

        Args:
            question: Question to ask
            pdf_path: Path to PDF file
            provider: LLM provider to use
            stream: Whether to use streaming (Replicate only)
            system_prompt: Optional system prompt
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific arguments

        Returns:
            Dictionary with 'answer' key and optional metadata
        """
        if provider == LLMProvider.GEMINI:
            return self._gemini_pdf_call(question, pdf_path, system_prompt, max_output_tokens=max_tokens, temperature=temperature)
        elif provider in (LLMProvider.GEMMA3_4B, LLMProvider.GEMMA3_12B, LLMProvider.GEMMA3_27B,
                          LLMProvider.GEMMA3, LLMProvider.DEEPSEEK, LLMProvider.GPT5):
            # Inject generation params into kwargs for Replicate
            kwargs.setdefault("temperature", temperature)
            kwargs.setdefault("max_tokens", max_tokens)
            return self._replicate_pdf_call(question, pdf_path, provider, stream, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def ask_with_manual(
        self,
        question: str,
        manual_path: str,
        provider: LLMProvider,
        context_mode: ContextMode,
        retrieval_query: Optional[str] = None,
        top_k: int = 6,
        **gen_kwargs
    ) -> Dict[str, Any]:
        """
        DEPRECATED: This method is no longer used and will raise an error.

        Use get_contexts() + template rendering in runner.py instead.
        See runner._run_with_repetitions() for the correct implementation.
        """
        raise RuntimeError(
            "ask_with_manual() is DEPRECATED and removed. "
            "Use get_contexts() + PromptTemplate rendering instead. "
            "See framework/runner.py:_run_with_repetitions() for correct usage."
        )

    # ================= Gemini Implementation =================

    def _gemini_call(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.4,
        max_output_tokens: int = 8192,
        top_p: float = 1.0
    ) -> Dict[str, Any]:
        """Call Gemini without PDF (baseline mode)"""
        if not self.gemini_configured:
            raise RuntimeError("Gemini API not configured")

        # Prepare request
        system_instr = {"parts": [{"text": system_prompt}]} if system_prompt else None
        contents = [{
            "role": "user",
            "parts": [{"text": question}]
        }]

        generation_config = {
            "maxOutputTokens": max_output_tokens,
            "temperature": temperature,
            "topP": top_p  # Gemini uses camelCase
        }

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config
        }
        if system_instr:
            payload["systemInstruction"] = system_instr

        headers = {"Content-Type": "application/json"}
        resp = self._post_with_backoff(self.gemini_api_url, headers, payload)

        if resp.status_code != 200:
            raise RuntimeError(f"Gemini HTTP {resp.status_code}: {resp.text}")

        data = resp.json()

        # Extract text from candidates
        def _extract_text(js: Dict[str, Any]) -> str:
            cands = js.get("candidates") or []
            texts: List[str] = []
            for c in cands:
                parts = ((c.get("content") or {}).get("parts") or [])
                t = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p)
                if t:
                    texts.append(t)
            return "\n".join(texts).strip()

        full_text = _extract_text(data)

        if not full_text:
            return {"answer": "No response generated.", "raw_response": data, "contexts": []}


        return {"answer": full_text.strip(), "raw_response": data, "contexts": []}

    def _gemini_upload_file(self, pdf_path: str, display_name: Optional[str] = None) -> Dict[str, Any]:
        """Upload PDF to Gemini Files API using resumable upload"""
        if not self.gemini_configured:
            raise RuntimeError("Gemini API not configured")
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        upload_init_url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={self.gemini_api_key}"
        size = os.path.getsize(pdf_path)
        mime = mimetypes.guess_type(pdf_path)[0] or "application/pdf"
        display_name = display_name or os.path.basename(pdf_path)

        # Initialize upload
        init_headers = {
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(size),
            "X-Goog-Upload-Header-Content-Type": mime,
            "Content-Type": "application/json",
        }
        init_body = {"file": {"display_name": display_name}}
        init_resp = requests.post(upload_init_url, headers=init_headers, json=init_body, timeout=90)
        init_resp.raise_for_status()

        upload_url = init_resp.headers.get("X-Goog-Upload-URL") or init_resp.headers.get("x-goog-upload-url")
        if not upload_url:
            raise RuntimeError("Missing X-Goog-Upload-URL in response headers")

        # Upload file
        with open(pdf_path, "rb") as f:
            data = f.read()

        upload_headers = {
            "Content-Length": str(len(data)),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        }

        up = requests.post(upload_url, headers=upload_headers, data=data, timeout=300)
        up.raise_for_status()
        file_obj = up.json()

        # Wait for file to be processed
        name = (file_obj.get("name") or (file_obj.get("file") or {}).get("name"))
        if name:
            meta_url = f"https://generativelanguage.googleapis.com/v1beta/{name}?key={self.gemini_api_key}"
            deadline = time.time() + 30.0
            while time.time() < deadline:
                meta = requests.get(meta_url, timeout=30)
                if meta.status_code == 200:
                    js = meta.json()
                    state = js.get("state")
                    if state == "ACTIVE":
                        break
                    if state in {"FAILED", "DELETED"}:
                        raise RuntimeError(f"File entered terminal state: {state}")
                time.sleep(0.5)

        return file_obj

    def _gemini_pdf_call(
        self,
        question: str,
        pdf_path: str,
        system_prompt: Optional[str] = None,
        max_output_tokens: int = 8192,
        temperature: float = 0.4,
        max_continuations: int = 2
    ) -> Dict[str, Any]:
        """Call Gemini with PDF using Files API, and expose contexts for RAG metrics"""
        if not self.gemini_configured:
            raise RuntimeError("Gemini API not configured")

        # 1) Extract manual text LOCALLY for metrics (contexts)
        manual_text = _extract_pdf_text_locally(pdf_path, max_chars=700_000, do_ocr=True)
        if manual_text.startswith("[[PDF TEXT EXTRACTION"):
            # local extraction failed → no contexts available
            contexts: List[str] = []
        else:
            # split manual text into chunks so metrics can treat them as context units
            # Using 4000 chars for better semantic coherence (increased from 2000)
            contexts = list(_chunk_text(manual_text, max_chars=4000))

        # 2) Upload PDF for Gemini's Files API (for the actual reasoning)
        file_obj = self._gemini_upload_file(pdf_path)
        file_uri = ((file_obj.get("file") or {}).get("uri") or file_obj.get("uri"))
        file_name = ((file_obj.get("file") or {}).get("name") or file_obj.get("name"))
        if not file_uri and file_name:
            file_uri = f"https://generativelanguage.googleapis.com/v1beta/{file_name}"
        if not file_uri:
            raise RuntimeError(f"Could not resolve file URI from response: {file_obj}")

        # 3) Prepare request to Gemini
        system_instr = {"parts": [{"text": system_prompt}]} if system_prompt else None
        contents = [{
            "role": "user",
            "parts": [
                {"text": question},
                {"fileData": {"mimeType": "application/pdf", "fileUri": file_uri}}
            ]
        }]

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_output_tokens,
                "temperature": temperature
            }
        }
        if system_instr:
            payload["systemInstruction"] = system_instr

        headers = {"Content-Type": "application/json"}
        resp = self._post_with_backoff(self.gemini_api_url, headers, payload)

        if resp.status_code != 200:
            raise RuntimeError(f"Gemini HTTP {resp.status_code}: {resp.text}")

        data = resp.json()  

            # 4) Extract text from all candidates
        def _extract_text(js: Dict[str, Any]) -> str:
            cands = js.get("candidates") or []
            texts: List[str] = []
            for c in cands:
                parts = ((c.get("content") or {}).get("parts") or [])
                t = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p)
                if t:
                    texts.append(t)
            return "\n".join(texts).strip()

        full_text = _extract_text(data)

        if not full_text:
            return {
                "answer": "Not found in the manual.",
                "raw_response": data,
                "contexts": contexts,   # may be empty if extraction failed
            }

        return {
            "answer": full_text.strip(),
            "raw_response": data,
            "contexts": contexts,       # manual chunks used by RAG metrics
        }

    # ================= Replicate Implementation =================

    def _replicate_pdf_call(
        self,
        question: str,
        pdf_path: str,
        provider: LLMProvider,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Call Replicate models with PDF (falls back to text extraction)"""
        if not self.replicate_client:
            raise RuntimeError("Replicate not configured")

        # Select model
        if provider == LLMProvider.GEMMA3_4B:
            model_slug = self.gemma3_4b_it
        elif provider == LLMProvider.GEMMA3_12B:
            model_slug = self.gemma3_12b_it
        elif provider in (LLMProvider.GEMMA3, LLMProvider.GEMMA3_27B):
            model_slug = self.gemma3_27b_it
        elif provider == LLMProvider.DEEPSEEK:
            model_slug = self.deepseek_model
        elif provider == LLMProvider.GPT5:
            model_slug = self.gpt5_model
        else:
            raise ValueError(f"Unsupported Replicate provider: {provider}")

        # Extract text from PDF (with OCR fallback)
        manual_text = _extract_pdf_text_locally(pdf_path, max_chars=700_000, do_ocr=True)

        if manual_text.startswith("[[PDF TEXT EXTRACTION"):
            # Failed to extract text
            if self.gemini_configured:
                # Fallback to Gemini
                try:
                    # Gemini call may or may not include contexts; pass through as-is
                    return self._gemini_pdf_call(question, pdf_path)
                except Exception:
                    pass
            return {
                "answer": "Not found in the manual.",
                "raw_response": {"reason": "extraction_failed"},
                "contexts": [],      # no usable manual text
            }
        # Using 4000 chars for better semantic coherence (increased from 2000)
        contexts = list(_chunk_text(manual_text, max_chars=4000))
        
        # Create grounded prompt
        grounded_prompt = (
            "Answer STRICTLY and ONLY using the following manual content. "
            "If the answer is not present, reply 'Not found in the manual.' "
            "For every factual statement include an inline citation with a page or section, e.g., (p. 12) or (Section 1.2). "
            "Do NOT rely on outside knowledge.\n\n"
            "----- BEGIN MANUAL TEXT -----\n"
            f"{manual_text}\n"
            "----- END MANUAL TEXT -----\n\n"
            f"Question: {question}\n"
            "IMPORTANT: Include (p. X) or (Section Y) after each claim."
        )

        # Run model
        inputs = {"prompt": grounded_prompt}
        if kwargs:
            inputs.update(kwargs)

        out = self.replicate_client.run(model_slug, input=inputs)
        text = self._normalize_replicate_output(out)

        return {
            "answer": text.strip() if text else "No response generated.",
            "raw_response": out,
            "contexts": contexts,   # baseline → no retrieved documents
        }


    def _replicate_call(
        self,
        question: str,
        provider: LLMProvider,
        **kwargs
    ) -> Dict[str, Any]:
        """Call Replicate models without PDF (baseline mode)"""
        if not self.replicate_client:
            raise RuntimeError("Replicate not configured")

        # Select model
        if provider == LLMProvider.GEMMA3_4B:
            model_slug = self.gemma3_4b_it
        elif provider == LLMProvider.GEMMA3_12B:
            model_slug = self.gemma3_12b_it
        elif provider in (LLMProvider.GEMMA3, LLMProvider.GEMMA3_27B):
            model_slug = self.gemma3_27b_it
        elif provider == LLMProvider.DEEPSEEK:
            model_slug = self.deepseek_model
        elif provider == LLMProvider.GPT5:
            model_slug = self.gpt5_model
        else:
            raise ValueError(f"Unsupported Replicate provider: {provider}")

        # Run model with plain question (no RAG)
        inputs = {"prompt": question}
        if kwargs:
            inputs.update(kwargs)

        out = self.replicate_client.run(model_slug, input=inputs)
        text = self._normalize_replicate_output(out)

        return {
            "answer": text.strip() if text else "No response generated.",
            "raw_response": out,
            "contexts": [],       # baseline => no RAG contexts
        }