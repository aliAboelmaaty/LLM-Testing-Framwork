"""
Metrics Module

Comprehensive set of metrics for evaluating LLM outputs across different domains.

Metrics Categories:
1. RAG-specific: citation_coverage, citation_correctness, context_precision/recall, faithfulness
2. Quality: answer_relevancy, answer_correctness (with ground truth)
3. Readability: fkgl (Flesch-Kincaid Grade Level)
4. Structure: structural_completeness
5. Hallucination: hallucination_rate
6. Domain-specific: scenario_identification_rate, property_identification_rate
7. Consistency: output_consistency
8. OCR quality: cer, wer
"""

from typing import List, Dict, Any, Optional, Set
import re
import json


# ================= Text Processing Utilities =================

def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences, filtering out structural elements.

    Excludes:
    - Markdown headers (lines starting with #, **, etc.)
    - Very short segments (< 10 chars)
    - Lines that are just labels (ending with :)
    """
    text = (text or "").strip()
    if not text:
        return []

    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\u0027\u002D\u0028])', text.strip())

    # Filter out structural elements
    sentences = []
    for p in parts:
        p = p.strip()
        if not p:
            continue

        # Skip markdown headers and formatting
        if p.startswith('#') or p.startswith('**') or p.startswith('> '):
            continue

        # Skip very short segments (likely labels or formatting)
        if len(p) < 10:
            continue

        # Skip lines that end with : (section headers)
        if p.rstrip(':').strip() != p.strip() and len(p.split()) <= 4:
            continue

        sentences.append(p)

    return sentences


def _tokens(text: str) -> List[str]:
    """Extract tokens from text"""
    return re.findall(r"[A-Za-z0-9\u0027]+", (text or "").lower())


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Calculate Jaccard similarity between two sets"""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _cosine_tfidf(a: str, b: str) -> Optional[float]:
    """Calculate TF-IDF cosine similarity between two texts"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return None

    if not a.strip() or not b.strip():
        return 0.0

    vect = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    X = vect.fit_transform([a, b])
    return float(cosine_similarity(X[0], X[1])[0, 0])


# ================= Citation & Grounding Metrics =================

# Citation patterns (more permissive)
_CITATION_PAT = re.compile(
    r"\((?:(?:[^)]*?(?:p\.|pp\.|pg\.|page|pages|section)\s*\d[\d\u002E\u002D]*[^)]*))\)",
    re.I,
)


def _best_overlap_score(span: str, contexts: List[str]) -> float:
    """Calculate best overlap score between span and any context"""
    span_tokens = _tokens(span)
    if not span_tokens:
        return 0.0

    best = 0.0
    for c in contexts:
        ctx_tokens = _tokens(c)
        if not ctx_tokens:
            continue

        span_set, ctx_set = set(span_tokens), set(ctx_tokens)
        inter = len(span_set & ctx_set)
        prec = inter / len(span_set) if span_set else 0.0
        rec = inter / len(ctx_set) if ctx_set else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

        # Character-level overlap
        char_overlap = _jaccard(
            set(re.findall(r"..", " ".join(span_tokens))),
            set(re.findall(r"..", " ".join(ctx_tokens))),
        )

        score = 0.7 * f1 + 0.3 * char_overlap
        best = max(best, score)

    return best


def citation_coverage(
    answer: str,
    contexts: Optional[List[str]] = None,
    overlap_threshold: float = 0.18,
) -> float:
    """
    Fraction of sentences with EXPLICIT citations only.

    CRITICAL: This metric now returns 0.0 if no explicit citations are found,
    even if the answer overlaps with contexts. This enforces citation discipline.

    Args:
        answer: LLM answer text
        contexts: List of context documents (no longer used for implicit scoring)
        overlap_threshold: Deprecated parameter (kept for backward compatibility)

    Returns:
        Coverage score between 0.0 and 1.0
    """
    sents = _split_sentences(answer)
    if not sents:
        return 0.0

    # Count only explicit citations (p. X), (Section Y), etc.
    explicit = [s for s in sents if _CITATION_PAT.search(s)]
    return len(explicit) / len(sents)


def _citation_correctness_with_seed(
    answer: str,
    contexts: List[str],
    threshold: float = 0.18,
    sample_k: int = 10,
    seed: int = 42,
) -> float:
    """
    Check if EXPLICITLY cited sentences are actually grounded in contexts.

    CRITICAL: This metric now returns 0.0 if no explicit citations exist.
    It does NOT fall back to implicit overlap scoring.

    Args:
        answer: LLM answer text
        contexts: List of context documents
        threshold: Minimum overlap to consider grounded
        sample_k: Maximum sentences to sample for efficiency
        seed: Random seed for sampling (for reproducibility)

    Returns:
        Correctness score between 0.0 and 1.0
    """
    # Extract only sentences with explicit citations
    sents = [s for s in _split_sentences(answer) if _CITATION_PAT.search(s)]

    if not contexts:
        return 0.0

    # If no citations found, return 0.0 (no implicit fallback)
    if not sents:
        return 0.0

    # Sample sentences if too many (use provided seed for reproducibility)
    import random
    rnd = random.Random(seed)
    sample = sents if len(sents) <= sample_k else rnd.sample(sents, sample_k)

    # Check how many cited sentences are grounded in contexts
    ok = 0
    for s in sample:
        if _best_overlap_score(s, contexts) >= threshold:
            ok += 1

    return ok / len(sample) if sample else 0.0


# Standalone function for backward compatibility
def citation_correctness(
    answer: str,
    contexts: List[str],
    threshold: float = 0.18,
    sample_k: int = 10,
    seed: int = 42,
) -> float:
    """
    Backward compatibility wrapper for citation_correctness.

    DEPRECATED: This function uses a hard-coded seed (42) by default.
    For reproducibility, use MetricCalculator instance which respects experiment seed.

    Args:
        answer: LLM answer
        contexts: Context documents
        threshold: Overlap threshold
        sample_k: Max sentences to sample
        seed: Random seed (default: 42 - NOT RECOMMENDED for thesis)

    Returns:
        Citation correctness score (0.0-1.0)

    Warning:
        The default seed=42 is provided only for backward compatibility.
        For thesis work, ALWAYS use MetricCalculator(random_seed=your_seed).citation_correctness()
    """
    return _citation_correctness_with_seed(answer, contexts, threshold, sample_k, seed=seed)


def faithfulness(answer: str, contexts: List[str]) -> float:
    """
    Overlap-based grounding score.

    Measures how well each sentence in the answer is grounded in the contexts.

    Args:
        answer: LLM answer text
        contexts: List of context documents

    Returns:
        Faithfulness score between 0.0 and 1.0
    """
    sents = _split_sentences(answer)
    if not sents or not contexts:
        return 0.0

    scores = [_best_overlap_score(s, contexts) for s in sents]
    return float(sum(scores) / len(scores))


# ================= Context Quality Metrics =================

def _is_context_used(unit_text: str, answer: str, q: str) -> float:
    """Check if a context unit is used in the answer"""
    return _best_overlap_score(unit_text, [answer + "\n" + q])


def context_precision(
    contexts: List[str],
    question: str,
    answer: str,
    use_threshold: float = 0.18,
) -> float:
    """
    Precision of retrieved context.

    What fraction of retrieved contexts are actually used?

    Args:
        contexts: Retrieved context documents
        question: Original question
        answer: LLM answer
        use_threshold: Minimum overlap to consider "used"

    Returns:
        Precision score between 0.0 and 1.0
    """
    if not contexts:
        return 0.0

    used_flags = [
        1 if _is_context_used(c, answer, question) >= use_threshold else 0
        for c in contexts
    ]
    precision = sum(used_flags) / len(contexts) if contexts else 0.0
    return precision


def context_recall(
    contexts: List[str],
    question: str,
    answer: str,
    use_threshold: float = 0.18,
) -> float:
    """
    Recall of retrieved context.

    How much of the answer is covered by the retrieved contexts?

    Args:
        contexts: Retrieved context documents
        question: Original question
        answer: LLM answer
        use_threshold: Minimum overlap to consider "used"

    Returns:
        Recall score between 0.0 and 1.0
    """
    if not contexts:
        return 0.0

    used_flags = [
        1 if _is_context_used(c, answer, question) >= use_threshold else 0
        for c in contexts
    ]

    used_text = " ".join([c for c, f in zip(contexts, used_flags) if f])
    all_text = " ".join(contexts)

    cover_used = _best_overlap_score(answer, [used_text]) if used_text else 0.0
    cover_all = _best_overlap_score(answer, [all_text]) or 1e-9

    recall = min(1.0, cover_used / cover_all) if cover_all > 0 else 0.0
    return recall


# ================= Answer Quality Metrics =================

def answer_relevancy(question: str, answer: str) -> float:
    """
    TF-IDF cosine similarity between question and answer.

    Measures how relevant the answer is to the question.

    Args:
        question: Original question
        answer: LLM answer

    Returns:
        Relevancy score between 0.0 and 1.0
    """
    cos = _cosine_tfidf(question, answer)
    if cos is not None:
        return cos
    # Fallback to Jaccard
    return _jaccard(set(_tokens(question)), set(_tokens(answer)))


def _extract_diagnosis_section(answer: str) -> str:
    """
    Extract the diagnosis section from a structured LLM answer.

    Looks for patterns like:
    - **Diagnosis:** <text>
    - Diagnosis: <text>
    - ## Diagnosis

    Args:
        answer: LLM answer text

    Returns:
        Extracted diagnosis text (or empty string if not found)
    """
    lines = answer.split('\n')
    diagnosis_text = []
    in_diagnosis = False

    for line in lines:
        line_lower = line.lower().strip()

        # Check if this line starts a diagnosis section
        if 'diagnosis' in line_lower and ':' in line:
            # Extract text after "Diagnosis:"
            parts = line.split(':', 1)
            if len(parts) > 1:
                diagnosis_text.append(parts[1].strip())
            in_diagnosis = True
            continue

        # Check for markdown heading
        if line_lower.startswith('#') and 'diagnosis' in line_lower:
            in_diagnosis = True
            continue

        # Stop at next section heading
        if in_diagnosis and (line.startswith('#') or (line.endswith(':') and len(line.split()) <= 4)):
            break

        # Collect diagnosis text
        if in_diagnosis and line.strip():
            diagnosis_text.append(line.strip())

    return ' '.join(diagnosis_text)


def _normalize_diagnosis(text: str) -> str:
    """
    Normalize diagnosis text for comparison.

    - Lowercase
    - Remove punctuation
    - Collapse whitespace

    Args:
        text: Diagnosis text

    Returns:
        Normalized text
    """
    import string
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text


def _answer_correctness_with_embedder(
    llm_answer: str,
    ground_truth: Dict[str, Any],
    embedder: Optional[Any] = None
) -> float:
    """
    DIAGNOSIS-GROUNDED METRIC - Compare extracted diagnosis with ground truth.

    This metric specifically evaluates diagnostic capability by:
    1. Extracting the "Diagnosis" section from the LLM answer
    2. Comparing only that section to ground_truth["diagnosis"]
    3. Using exact/normalized match as primary score
    4. Falling back to semantic similarity for partial credit

    For diagnosis tasks:
        ground_truth = {"diagnosis": "Pump blocked", "root_cause": "Foreign object"}
        Extracts "Diagnosis:" section from llm_answer, compares with "Pump blocked"

    For other tasks (repurposing, ML recommendation):
        Falls back to keyword matching across all ground truth fields

    Args:
        llm_answer: LLM's answer
        ground_truth: Dictionary with expected outputs (must include "diagnosis" key for diagnosis tasks)
        embedder: Pre-loaded SentenceTransformer model (optional, improves performance)

    Returns:
        Correctness score between 0.0 and 1.0
    """
    if not llm_answer or not ground_truth:
        return 0.0

    # Diagnosis task: extract and compare diagnosis section
    if "diagnosis" in ground_truth:
        gt_diagnosis = str(ground_truth["diagnosis"]).strip()
        if not gt_diagnosis:
            return 0.0

        # Extract diagnosis from answer
        extracted_diagnosis = _extract_diagnosis_section(llm_answer)

        # If extraction failed, fall back to searching whole answer
        if not extracted_diagnosis:
            extracted_diagnosis = llm_answer

        # Normalize both for comparison
        norm_extracted = _normalize_diagnosis(extracted_diagnosis)
        norm_gt = _normalize_diagnosis(gt_diagnosis)

        # 1. Exact match after normalization
        if norm_extracted == norm_gt:
            return 1.0

        # 2. Check if ground truth is substring of extracted (partial match)
        if norm_gt in norm_extracted:
            return 0.9

        # 3. Token overlap (precision-focused)
        extracted_tokens = set(_tokens(extracted_diagnosis))
        gt_tokens = set(_tokens(gt_diagnosis))

        if not gt_tokens:
            return 0.0

        # Precision: what fraction of extracted tokens are in ground truth?
        # Recall: what fraction of ground truth tokens are in extracted?
        precision = len(extracted_tokens & gt_tokens) / len(extracted_tokens) if extracted_tokens else 0.0
        recall = len(extracted_tokens & gt_tokens) / len(gt_tokens) if gt_tokens else 0.0

        # F1 score (balanced precision and recall)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # 4. Semantic similarity as fallback (use shared embedder if available)
        if embedder is not None:
            try:
                embeddings = embedder.encode([extracted_diagnosis, gt_diagnosis])
                from sklearn.metrics.pairwise import cosine_similarity
                semantic_sim = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])

                # Weighted combination: 60% token overlap, 40% semantic similarity
                return 0.6 * f1 + 0.4 * semantic_sim
            except Exception:
                # Fall back to token overlap if embedding fails
                return f1
        else:
            # No embedder available, use token overlap only
            return f1

    # Non-diagnosis tasks: keyword matching across all ground truth fields
    gt_keywords = set()
    for v in ground_truth.values():
        if isinstance(v, str):
            gt_keywords.update(_tokens(v))
        elif isinstance(v, list):
            for item in v:
                gt_keywords.update(_tokens(str(item)))

    answer_tokens = set(_tokens(llm_answer))

    if not gt_keywords:
        return 0.0

    # Jaccard similarity
    return _jaccard(answer_tokens, gt_keywords)


# Standalone function for backward compatibility
def answer_correctness(llm_answer: str, ground_truth: Dict[str, Any]) -> float:
    """
    Backward compatibility wrapper for answer_correctness.

    For best performance, use MetricCalculator instance instead.
    """
    return _answer_correctness_with_embedder(llm_answer, ground_truth, embedder=None)


# ================= Domain-Specific Metrics =================

def scenario_identification_rate(
    llm_scenarios: List[str],
    ground_truth_scenarios: List[str]
) -> float:
    """
    NEW - Dörnbach's SI metric.

    What fraction of documented scenarios did the LLM identify?

    SI = len(identified_scenarios) / len(ground_truth_scenarios)

    Args:
        llm_scenarios: Scenarios identified by LLM
        ground_truth_scenarios: Expected scenarios from documentation

    Returns:
        SI score between 0.0 and 1.0
    """
    if not ground_truth_scenarios:
        return 1.0 if not llm_scenarios else 0.0

    # Normalize both lists to lowercase tokens
    gt_set = {tuple(_tokens(s)) for s in ground_truth_scenarios}
    llm_set = {tuple(_tokens(s)) for s in llm_scenarios}

    # Count matches
    matches = 0
    for gt_scenario in gt_set:
        for llm_scenario in llm_set:
            # Check if there's significant overlap
            if _jaccard(set(gt_scenario), set(llm_scenario)) >= 0.5:
                matches += 1
                break

    return matches / len(ground_truth_scenarios)


def property_identification_rate(
    llm_properties: List[str],
    ground_truth_properties: List[str]
) -> float:
    """
    NEW - Dörnbach's PI metric.

    What fraction of required properties did the LLM identify?

    PI = len(identified_properties) / len(ground_truth_properties)

    Args:
        llm_properties: Properties identified by LLM
        ground_truth_properties: Expected properties from documentation

    Returns:
        PI score between 0.0 and 1.0
    """
    if not ground_truth_properties:
        return 1.0 if not llm_properties else 0.0

    # Similar to SI, but for properties
    gt_set = {tuple(_tokens(p)) for p in ground_truth_properties}
    llm_set = {tuple(_tokens(p)) for p in llm_properties}

    matches = 0
    for gt_prop in gt_set:
        for llm_prop in llm_set:
            if _jaccard(set(gt_prop), set(llm_prop)) >= 0.5:
                matches += 1
                break

    return matches / len(ground_truth_properties)


def output_consistency(answers: List[str]) -> float:
    """
    NEW - Sonntag's OCR metric (Output Consistency Rate).

    Given multiple runs with same input, how consistent are outputs?

    OCR = 1.0 if all answers identical
    OCR = 0.0 if all answers completely different

    Uses pairwise semantic similarity across all answers.

    Args:
        answers: List of answers from multiple runs

    Returns:
        Consistency score between 0.0 and 1.0
    """
    if not answers or len(answers) < 2:
        return 1.0  # Single answer is perfectly consistent

    # Calculate pairwise similarities
    similarities = []
    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            sim = _cosine_tfidf(answers[i], answers[j])
            if sim is not None:
                similarities.append(sim)
            else:
                # Fallback to Jaccard
                sim = _jaccard(set(_tokens(answers[i])), set(_tokens(answers[j])))
                similarities.append(sim)

    if not similarities:
        return 1.0

    return sum(similarities) / len(similarities)


# ================= Hallucination Detection =================

def hallucination_rate(
    faithfulness_score: float,
    citation_correctness: float,
    f_thresh: float = 0.25,
    c_thresh: float = 0.30,
) -> int:
    """
    Binary hallucination detection.

    Returns 1 if likely hallucinated, 0 if grounded.

    The answer is considered hallucinated if either:
    - Faithfulness score is below the faithfulness threshold, OR
    - Citation correctness is below the citation threshold

    Args:
        faithfulness_score: Faithfulness score (0.0-1.0)
        citation_correctness: Citation correctness score (0.0-1.0)
        f_thresh: Faithfulness threshold (default: 0.25)
        c_thresh: Citation correctness threshold (default: 0.30)

    Returns:
        1 if hallucinated, 0 if grounded

    Note:
        This metric should only be computed when contexts are available.
        For BASELINE mode (no contexts), return NaN instead.
    """
    return 1 if (faithfulness_score < f_thresh or citation_correctness < c_thresh) else 0


# ================= Readability Metrics =================

_VOWELS = "aeiouy"


def _count_syllables(word: str) -> int:
    """Count syllables in a word"""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0

    count = 0
    prev_vowel = False
    for ch in w:
        is_v = ch in _VOWELS
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v

    if w.endswith("e") and count > 1:
        count -= 1

    return max(1, count)


def fkgl(text: str) -> float:
    """
    Flesch-Kincaid Grade Level (readability).

    Higher scores = more difficult to read.

    Args:
        text: Text to analyze

    Returns:
        Grade level (e.g., 8.0 = 8th grade reading level)
    """
    sents = _split_sentences(text)
    words = _tokens(text)

    if not sents or not words:
        return 0.0

    syllables = sum(_count_syllables(w) for w in words)
    ASL = len(words) / len(sents)  # Average Sentence Length
    ASW = syllables / len(words)   # Average Syllables per Word

    return 0.39 * ASL + 11.8 * ASW - 15.59


# ================= Structural Completeness =================

def structural_completeness(answer: str, contexts: Optional[List[str]] = None) -> float:
    """
    Check for expected structural elements in the answer.

    For diagnosis tasks, checks for:
    - Citations (>= 50% coverage)
    - Safety warnings
    - Tools & Parts section

    Args:
        answer: LLM answer
        contexts: Optional contexts (for citation checking)

    Returns:
        Completeness score between 0.0 and 1.0
    """
    cov = citation_coverage(answer, contexts or [])

    checks = {
        "citations": cov >= 0.5,
        "warnings": (">" in answer)
                    or ("warning" in answer.lower())
                    or ("caution" in answer.lower()),
        "tools_parts": ("tools & parts" in answer.lower())
                       or ("tools and parts" in answer.lower())
                       or ("### tools" in answer.lower()),
    }

    return sum(1 for _, v in checks.items() if v) / len(checks)


# ================= OCR Quality Metrics =================

def _levenshtein(a: List, b: List) -> int:
    """Calculate Levenshtein distance between two sequences"""
    n, m = len(a), len(b)
    dp = list(range(m + 1))

    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur

    return dp[m]


def cer(ocr_text: str, gt_text: str) -> float:
    """
    Character Error Rate.

    CER = edit_distance(chars) / len(ground_truth_chars)

    Args:
        ocr_text: OCR-extracted text
        gt_text: Ground truth text

    Returns:
        CER score (lower is better)
    """
    a_c = list(ocr_text or "")
    b_c = list(gt_text or "")

    if not b_c:
        return 0.0 if not a_c else 1.0

    return _levenshtein(a_c, b_c) / len(b_c)


def wer(ocr_text: str, gt_text: str) -> float:
    """
    Word Error Rate.

    WER = edit_distance(words) / len(ground_truth_words)

    Args:
        ocr_text: OCR-extracted text
        gt_text: Ground truth text

    Returns:
        WER score (lower is better)
    """
    a_w = (ocr_text or "").split()
    b_w = (gt_text or "").split()

    if not b_w:
        return 0.0 if not a_w else 1.0

    return _levenshtein(a_w, b_w) / len(b_w)


# ================= Metrics Calculator =================

class MetricCalculator:
    """
    Centralized metric calculator with shared embedding model.

    Instantiates embedding model once for efficiency and determinism.
    """

    def __init__(self, random_seed: int = 42):
        """
        Initialize metric calculator with embedding model.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed

        # Instantiate embedding model once (expensive operation)
        # This ensures:
        # 1. Model is only downloaded/loaded once per experiment
        # 2. Consistent environment across metric calculations
        # 3. Better performance (no repeated instantiation)
        self.embedder = None
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            # Embedding model not available - metrics will fall back to token-based scoring
            pass

    # Citation & Grounding
    citation_coverage = staticmethod(citation_coverage)
    faithfulness = staticmethod(faithfulness)

    def citation_correctness(self, answer: str, contexts: List[str], threshold: float = 0.18, sample_k: int = 10) -> float:
        """
        Instance method wrapper for citation_correctness that uses configured random_seed.
        """
        return _citation_correctness_with_seed(answer, contexts, threshold, sample_k, seed=self.random_seed)

    # Context Quality
    context_precision = staticmethod(context_precision)
    context_recall = staticmethod(context_recall)

    # Answer Quality
    answer_relevancy = staticmethod(answer_relevancy)

    # Domain-Specific
    scenario_identification_rate = staticmethod(scenario_identification_rate)
    property_identification_rate = staticmethod(property_identification_rate)
    output_consistency = staticmethod(output_consistency)

    # Hallucination
    hallucination_rate = staticmethod(hallucination_rate)

    # Readability
    fkgl = staticmethod(fkgl)

    # Structural
    structural_completeness = staticmethod(structural_completeness)

    # OCR Quality
    cer = staticmethod(cer)
    wer = staticmethod(wer)

    def answer_correctness(self, llm_answer: str, ground_truth: Dict[str, Any]) -> float:
        """
        Instance method wrapper for answer_correctness that uses shared embedder.
        """
        return _answer_correctness_with_embedder(llm_answer, ground_truth, self.embedder)


# ================= Metrics Configuration =================

class MetricsConfig:
    """Predefined metric sets for different task types"""

    DIAGNOSIS_METRICS = [
        "answer_correctness",
        "citation_coverage",
        "citation_correctness",
        "faithfulness",
        "context_precision",
        "context_recall",
        "hallucination_rate",
        "structural_completeness",
        "answer_relevancy",
        "fkgl",
    ]

    REPURPOSING_METRICS = [
        "scenario_identification_rate",
        "property_identification_rate",
        "answer_correctness",
        "output_consistency",
    ]

    ML_RECOMMENDATION_METRICS = [
        "answer_correctness",
        "output_consistency",
        "answer_relevancy",
    ]

    @staticmethod
    def get_metrics_for_task(task_type: str) -> List[str]:
        """Get recommended metrics for a task type"""
        if task_type == "diagnosis":
            return MetricsConfig.DIAGNOSIS_METRICS
        elif task_type == "repurposing":
            return MetricsConfig.REPURPOSING_METRICS
        elif task_type == "ml_recommendation":
            return MetricsConfig.ML_RECOMMENDATION_METRICS
        else:
            return MetricsConfig.DIAGNOSIS_METRICS  # Default
