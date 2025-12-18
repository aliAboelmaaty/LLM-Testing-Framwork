"""
Experiment Runner Module

Orchestrates LLM evaluation experiments following research methodology.

Features:
- Multi-model testing
- Baseline vs RAG comparison
- Multiple repetitions for consistency analysis
- Comprehensive metrics calculation
- Progress tracking
- Resilient error handling

TRI-STATE OPERATIONAL FIELDS:
===============================
Operational/debug fields use tri-state logic (True/False/None) to distinguish:
- True: Operation executed and succeeded
- False: Operation executed but failed/not used
- None: Operation not applicable or result not trustworthy

Fields using tri-state:
- pdf_extraction_cache_hit: None when baseline, no manual, or runtime error
- cache_hit: (backward compat alias for pdf_extraction_cache_hit)
- retrieval_executed: None when baseline, no manual, or runtime error
- manual_full_loaded: None when baseline, no manual, or runtime error

When fields are None:
1. context_mode == BASELINE: Cache/retrieval operations don't apply
2. manual_available == False: No manual document to process
3. had_error == True: Runtime error makes operational data untrustworthy
4. pdf_extraction_failed == True: PDF extraction failed, cache state unknown

This prevents misleading False values when operations were never attempted.
"""

from typing import List, Dict, Any, Optional
import json
import re
from tqdm import tqdm

from .core import ExperimentConfig, TestCase, ExperimentResults, LLMProvider, ContextMode
from .dataset import Dataset
from .prompts import PromptLibrary
from .metrics import MetricCalculator, MetricsConfig
from .llm_service import LLMService


# ================= Provenance Helper =================

def compute_provenance_fields(
    first_output: Dict[str, Any],
    contexts: List[str],
    run_id: str,
    model: LLMProvider,
    llm_service: 'LLMService',
    temperature_used: float,
    max_tokens_used: int,
    top_p_used: float,
    prompt_variant: Optional[str]
) -> Dict[str, Any]:
    """
    CENTRALIZED provenance computation to guarantee consistency.

    Computes all reproducibility and audit trail fields in ONE place.
    This ensures no competing logic can create inconsistent hashes.

    Args:
        first_output: First repetition output dict
        contexts: List of context strings
        run_id: Unique run identifier
        model: LLM provider
        llm_service: LLM service instance
        temperature_used: Temperature parameter used
        max_tokens_used: Max tokens parameter used
        prompt_variant: Prompt variant name

    Returns:
        Dict with all provenance fields (hashes, identifiers, audit trail)
    """
    import hashlib

    # Extract strings (use empty string as default, never None)
    prompt_str = first_output.get("prompt", "")
    retrieval_query_str = first_output.get("retrieval_query", "")
    context_str = "\n\n".join(contexts) if contexts else ""

    # CRITICAL: Always compute hashes, even for empty strings
    # Empty string -> deterministic hash (not None)
    # This ensures hash fields are ALWAYS present and NEVER None
    prompt_hash = hashlib.sha256(prompt_str.encode('utf-8')).hexdigest()
    context_hash = hashlib.sha256(context_str.encode('utf-8')).hexdigest()

    # Get model identifier (provider-specific)
    if model == LLMProvider.GEMINI:
        model_identifier = llm_service.gemini_model
        provider_backend = "gemini"
    elif model in (LLMProvider.GEMMA3_4B, LLMProvider.GEMMA3_12B, LLMProvider.GEMMA3_27B, LLMProvider.GEMMA3):
        if model == LLMProvider.GEMMA3_4B:
            model_identifier = llm_service.gemma3_4b_it
        elif model == LLMProvider.GEMMA3_12B:
            model_identifier = llm_service.gemma3_12b_it
        else:
            model_identifier = llm_service.gemma3_27b_it
        provider_backend = "replicate"
    elif model == LLMProvider.DEEPSEEK:
        model_identifier = llm_service.deepseek_model
        provider_backend = "replicate"
    elif model == LLMProvider.GPT5:
        model_identifier = llm_service.gpt5_model
        provider_backend = "replicate"
    else:
        model_identifier = model.value
        provider_backend = "unknown"

    # Return all provenance fields in one dict
    return {
        # Hash-based reproducibility fields
        "prompt_hash": prompt_hash,
        "context_hash": context_hash,
        "retrieval_query": retrieval_query_str,

        # Model identification fields
        "model_identifier": model_identifier,
        "provider_backend": provider_backend,

        # Protocol parameters (FAIRNESS)
        "temperature_used": temperature_used,
        "max_tokens_used": max_tokens_used,
        "top_p_used": top_p_used,
        "prompt_variant": prompt_variant if prompt_variant else "default",
        "seed_used": first_output.get("seed_used", None),

        # Audit trail (REPRODUCIBILITY)
        "run_id": run_id,

        # Full text for exact replay (REPRODUCIBILITY)
        "prompt_text": prompt_str,
        "contexts_text": context_str
    }


# ================= Experiment Runner =================

class ExperimentRunner:
    """
    Runs LLM evaluation experiments following methodology from research papers.

    Implements:
    - Multiple repetitions (for consistency analysis - Sonntag's OCR)
    - Baseline vs RAG comparison
    - Complexity-based analysis (Dörnbach's clusters)
    - Comprehensive metrics calculation
    """

    def __init__(
        self,
        config: ExperimentConfig,
        llm_service: LLMService
    ):
        """
        Initialize experiment runner.

        Args:
            config: Experiment configuration
            llm_service: LLM service instance
        """
        self.config = config
        self.llm_service = llm_service
        self.results: List[Dict[str, Any]] = []
        # Pass random_seed to MetricCalculator for reproducible metric calculations
        self.metrics_calculator = MetricCalculator(random_seed=config.random_seed)

    def run(self, dataset: Dataset, verbose: bool = True) -> ExperimentResults:
        """
        Main experiment execution.

        Process:
        1. For each model in config.models
        2. For each context_mode in config.context_modes
        3. For each test case in dataset
        4. Run config.n_repetitions times
        5. Calculate all configured metrics
        6. Aggregate results

        Args:
            dataset: Dataset of test cases
            verbose: Whether to show progress bars

        Returns:
            ExperimentResults with all data and analysis tables
        """
        # REPRODUCIBILITY: Set global random seed for consistent behavior
        import random
        random.seed(self.config.random_seed)

        # Also seed numpy if available for full reproducibility
        try:
            import numpy as np
            np.random.seed(self.config.random_seed)
        except ImportError:
            pass

        # REPRODUCIBILITY: Generate unique run ID and capture git commit
        import uuid
        import subprocess
        from datetime import datetime

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Starting Experiment: {self.config.task_type}")
            print(f"Run ID: {run_id}")
            print(f"Models: {[m.value for m in self.config.models]}")
            print(f"Context Modes: {[cm.value for cm in self.config.context_modes]}")
            print(f"Test Cases: {len(dataset)}")
            print(f"Repetitions: {self.config.n_repetitions}")
            print(f"Random Seed: {self.config.random_seed}")
            print(f"{'='*60}\n")

        # Progress bar setup
        total_runs = (
            len(self.config.models) *
            len(self.config.context_modes) *
            len(dataset) *
            self.config.n_repetitions
        )

        # REPRODUCIBILITY: Sort test cases by case_id for deterministic ordering
        sorted_test_cases = sorted(dataset, key=lambda tc: tc.case_id)

        with tqdm(total=total_runs, desc="Running experiments", disable=not verbose) as pbar:
            for model in self.config.models:
                for context_mode in self.config.context_modes:
                    if verbose:
                        print(f"\nTesting {model.value} ({context_mode.value})...")

                    for test_case in sorted_test_cases:
                        # Mixed dataset behavior: skip MANUAL_FULL/RAG_RETRIEVAL when no manual exists
                        if context_mode != ContextMode.BASELINE and not test_case.context_document:
                            # Skip this test case for non-baseline modes
                            if pbar:
                                pbar.update(self.config.n_repetitions)
                            continue

                        # Run with repetitions
                        outputs = self._run_with_repetitions(
                            model=model,
                            test_case=test_case,
                            context_mode=context_mode,
                            pbar=pbar
                        )

                        # CHANGE #1: Determine extraction failure BEFORE sanity check
                        first_output = outputs[0] if outputs else {}
                        contexts = first_output.get("contexts", [])
                        if isinstance(contexts, str):
                            try:
                                contexts = json.loads(contexts) if contexts else []
                            except json.JSONDecodeError:
                                contexts = []

                        # Check if there was a runtime error first
                        had_runtime_error = (
                            first_output.get("error") or
                            str(first_output.get("answer", "")).startswith("[ERROR:")
                        )

                        # Check if extraction failed (manual exists but contexts empty for non-baseline)
                        # FIX: Only flag extraction failure if there was NO runtime error
                        pdf_extraction_failed = (
                            context_mode != ContextMode.BASELINE and
                            test_case.context_document and
                            len(contexts) == 0 and
                            not had_runtime_error
                        )

                        # CHANGE #5: Lightweight guard - check context count consistency across reps
                        context_count_first = len(contexts)
                        context_inconsistent = False
                        for i, out in enumerate(outputs[1:], start=1):
                            out_contexts = out.get("contexts", [])
                            if isinstance(out_contexts, str):
                                try:
                                    out_contexts = json.loads(out_contexts) if out_contexts else []
                                except json.JSONDecodeError:
                                    out_contexts = []
                            if len(out_contexts) != context_count_first:
                                print(f"WARNING: Inconsistent context count across repetitions for {test_case.case_id}: "
                                      f"rep 0 had {context_count_first}, rep {i} had {len(out_contexts)}")
                                context_inconsistent = True
                                break

                        # Sanity check: validate contexts based on context_mode
                        sanity_check_passed, sanity_error_msg = self._sanity_check_contexts(
                            outputs, context_mode, self.config.top_k_retrieval
                        )

                        # CHANGE #5: Mark as sanity failure if contexts inconsistent
                        if context_inconsistent:
                            sanity_check_passed = False
                            sanity_error_msg = f"Inconsistent context counts across repetitions (first={context_count_first})"

                        # FAIRNESS: Schema validation (ONLY if explicitly enabled)
                        # WARNING: Disabled by default because default prompts don't match the hardcoded schema
                        # Only enable if using custom prompts that request: Diagnosis, Root Cause, Steps, Safety, Citations
                        if sanity_check_passed and self.config.enforce_output_schema and self.config.task_type == "diagnosis":
                            first_answer = first_output.get("answer", "")
                            schema_passed, schema_error = self._check_output_schema(first_answer)
                            if not schema_passed:
                                sanity_check_passed = False
                                sanity_error_msg = f"Schema violation: {schema_error}"

                        # CHANGE #2, #3, #7: Handle sanity failure properly
                        if not sanity_check_passed:
                            # Sanity check failed - set metrics to NaN (not scored)
                            print(f"SANITY CHECK FAILED for {test_case.case_id}: {sanity_error_msg}")
                            metrics = {metric_name: float('nan') for metric_name in self.config.metrics}
                            metrics["sanity_check_failed"] = 1.0
                            metrics["had_error"] = 0.0  # FIX: Always set had_error (sanity ≠ runtime error)
                            metrics["pdf_extraction_failed"] = 1.0 if pdf_extraction_failed else 0.0
                            # CHANGE #7: Store explicit sanity error message
                            sanity_error_detail = f"SANITY: {sanity_error_msg}"
                        else:
                            # Calculate metrics normally
                            metrics = self._calculate_metrics(outputs, test_case, context_mode)
                            metrics["sanity_check_failed"] = 0.0
                            # Add pdf_extraction_failed if not already set
                            if "pdf_extraction_failed" not in metrics:
                                metrics["pdf_extraction_failed"] = 1.0 if pdf_extraction_failed else 0.0
                            sanity_error_detail = ""

                        # TRI-STATE: Extract debug info with proper None handling
                        # Set operational fields to None when not applicable or not trustworthy

                        # Determine if operations are applicable/trustworthy
                        manual_available = bool(test_case.context_document)
                        operations_not_applicable = (
                            context_mode == ContextMode.BASELINE or  # Baseline doesn't use manual
                            not manual_available or                   # No manual to process
                            had_runtime_error or                      # Runtime error = untrustworthy
                            pdf_extraction_failed                     # PDF extraction failed = cache state unknown
                        )

                        # Cache fields: None if operations not applicable, else get from output
                        if operations_not_applicable:
                            pdf_cache_hit = None
                            retrieval_exec = None
                            manual_loaded = None
                        else:
                            # Trust the values from output (could be True/False)
                            pdf_cache_hit = first_output.get("pdf_extraction_cache_hit", first_output.get("cache_hit", None))
                            retrieval_exec = first_output.get("retrieval_executed", None)
                            manual_loaded = first_output.get("manual_full_loaded", None)

                        # PROVENANCE: Determine protocol parameters
                        temperature_used = self.config.temperature
                        if context_mode == ContextMode.BASELINE:
                            max_tokens_used = self.config.baseline_max_tokens if self.config.baseline_max_tokens is not None else self.config.max_tokens
                        else:
                            max_tokens_used = self.config.max_tokens

                        # PROVENANCE: Centralized computation (guarantees consistency)
                        provenance = compute_provenance_fields(
                            first_output=first_output,
                            contexts=contexts,
                            run_id=run_id,
                            model=model,
                            llm_service=self.llm_service,
                            temperature_used=temperature_used,
                            max_tokens_used=max_tokens_used,
                            top_p_used=self.config.top_p,
                            prompt_variant=self.config.prompt_variant
                        )

                        debug_info = {
                            "contexts_count": len(contexts),
                            "context_chars": first_output.get("context_chars", 0),
                            "prompt_chars": first_output.get("prompt_chars", 0),
                            "total_chunks": first_output.get("total_chunks", 0),
                            "chunks_used": first_output.get("chunks_used", 0),
                            "truncated": first_output.get("truncated", False),
                            # TRI-STATE: Operational fields (True/False/None)
                            # None = not applicable or not trustworthy
                            "pdf_extraction_cache_hit": pdf_cache_hit,
                            "cache_hit": pdf_cache_hit,  # Backward compatibility
                            "retrieval_executed": retrieval_exec,
                            "manual_full_loaded": manual_loaded,
                            # CHANGE #7: Combine runtime error and sanity error
                            "error": sanity_error_detail or first_output.get("error", ""),
                            # PROVENANCE: All fields from centralized computation
                            **provenance
                        }

                        # Store result
                        self.results.append({
                            "model": model.value,
                            "context_mode": context_mode.value,
                            "uses_context": context_mode != ContextMode.BASELINE,  # CHANGE #6: Renamed from use_rag_compat
                            "use_rag_compat": context_mode != ContextMode.BASELINE,  # Backwards compatibility - keep for now
                            "case_id": test_case.case_id,
                            "complexity": test_case.metadata.get("complexity", 1),
                            "manual_available": bool(test_case.context_document),
                            **metrics,
                            **debug_info
                        })

        return self._aggregate_results()

    def _run_with_repetitions(
        self,
        model: LLMProvider,
        test_case: TestCase,
        context_mode: ContextMode,
        pbar: Optional[tqdm] = None
    ) -> List[Dict[str, Any]]:
        """
        Run same test case n times to measure consistency.

        Args:
            model: LLM provider
            test_case: Test case to run
            context_mode: Context mode (BASELINE, MANUAL_FULL, or RAG_RETRIEVAL)
            pbar: Progress bar to update

        Returns:
            List of LLM outputs (one per repetition)
        """
        outputs = []

        # Select template based on context_mode
        # CRITICAL FIX: BASELINE uses template WITHOUT {context}
        # MANUAL_FULL and RAG_RETRIEVAL use template WITH {context}
        if self.config.custom_prompt:
            template = self.config.custom_prompt
        else:
            use_rag = (context_mode != ContextMode.BASELINE)
            template = PromptLibrary.get_template(
                self.config.task_type,
                use_rag=use_rag,
                variant=self.config.prompt_variant
            )

        # Extract retrieval query from input_data (separate from full prompt)
        # This is used for BM25 retrieval, NOT the full rendered prompt
        retrieval_query = (
            test_case.input_data.get("fault_description") or
            test_case.input_data.get("question") or
            test_case.input_data.get("component") or
            test_case.input_data.get("problem_description") or
            str(list(test_case.input_data.values())[0] if test_case.input_data else "")[:500]
        )

        for i in range(self.config.n_repetitions):
            # REPRODUCIBILITY: Compute deterministic seed per repetition
            seed_used = self.config.random_seed + i

            try:
                if context_mode == ContextMode.BASELINE:
                    # BASELINE: No context - render prompt without context parameter
                    prompt = template.render(**test_case.input_data)
                    baseline_max = self.config.baseline_max_tokens if self.config.baseline_max_tokens is not None else self.config.max_tokens# new
                    result = self.llm_service.ask(
                        question=prompt,
                        provider=model,
                        temperature=self.config.temperature,
                        max_tokens=baseline_max,
                        top_p=self.config.top_p,
                        seed=seed_used  # Pass seed to provider (if supported)
                    )
                    result["contexts"] = []
                    # TRI-STATE: BASELINE doesn't use retrieval/caching (not applicable = None)
                    result["retrieval_executed"] = None
                    result["manual_full_loaded"] = None
                    result["pdf_extraction_cache_hit"] = None
                    result["cache_hit"] = None
                    # PROVENANCE: Store prompt for hash computation
                    result["prompt"] = prompt
                    result["retrieval_query"] = ""  # No retrieval in BASELINE
                    result["seed_used"] = seed_used  # Track seed for reproducibility

                elif test_case.context_document:
                    # MANUAL_FULL or RAG_RETRIEVAL: Get contexts, then render prompt
                    from .llm_service import get_contexts

                    # Get provider-aware budgets
                    budgets = self.config.get_context_budgets(model)

                    contexts, metadata = get_contexts(
                        manual_path=test_case.context_document,
                        context_mode=context_mode,
                        retrieval_query=retrieval_query,  # Use short query, NOT full prompt
                        top_k=self.config.top_k_retrieval,
                        chunk_size=budgets["chunk_size"],
                        max_context_chars=budgets["max_context_chars"],
                        manual_full_max_chunks=budgets["manual_full_max_chunks"]
                    )

                    # Render prompt WITH contexts
                    context_text = "\n\n".join(contexts)
                    prompt = template.render(context=context_text, **test_case.input_data)

                    # Pass fully-rendered prompt to LLM (NO double wrapping)
                    result = self.llm_service.ask(
                        question=prompt,
                        provider=model,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        top_p=self.config.top_p,
                        seed=seed_used  # Pass seed to provider (if supported)
                    )
                    result["contexts"] = contexts
                    # FIX #1: Rename for clarity - this is PDF extraction caching, not retrieval caching
                    result["pdf_extraction_cache_hit"] = metadata.get("cache_hit", False)
                    result["cache_hit"] = metadata.get("cache_hit", False)  # Backward compatibility
                    result["total_chunks"] = metadata.get("total_chunks", 0)
                    result["chunks_used"] = metadata.get("chunks_used", 0)
                    result["truncated"] = metadata.get("truncated", False)
                    result["context_chars"] = metadata.get("context_chars", 0)
                    result["prompt_chars"] = len(prompt)
                    # TRI-STATE: Only set to True when applicable, None when not applicable
                    # RAG_RETRIEVAL: retrieval_executed=True, manual_full_loaded=None
                    # MANUAL_FULL: retrieval_executed=None, manual_full_loaded=True
                    if context_mode == ContextMode.RAG_RETRIEVAL:
                        result["retrieval_executed"] = True
                        result["manual_full_loaded"] = None
                    else:  # MANUAL_FULL
                        result["retrieval_executed"] = None
                        result["manual_full_loaded"] = True
                    # PROVENANCE: Store prompt and retrieval_query for hash computation
                    result["prompt"] = prompt
                    result["retrieval_query"] = retrieval_query
                    result["seed_used"] = seed_used  # Track seed for reproducibility

                else:
                    # No manual available but not baseline mode
                    result = {
                        "answer": "[ERROR: No manual document provided]",
                        "error": "No context_document in test case",
                        "contexts": [],
                        # TRI-STATE: Manual missing = operations not executed (None)
                        "retrieval_executed": None,
                        "manual_full_loaded": None,
                        "pdf_extraction_cache_hit": None,
                        "cache_hit": None,
                        "prompt": "",
                        "retrieval_query": "",
                        "seed_used": seed_used
                    }

                outputs.append(result)

            except Exception as e:
                # Error handling: store error message
                # TRI-STATE: Runtime error = operations not trustworthy (None)
                outputs.append({
                    "answer": f"[ERROR: {str(e)}]",
                    "error": str(e),
                    "contexts": [],
                    "retrieval_executed": None,
                    "manual_full_loaded": None,
                    "pdf_extraction_cache_hit": None,
                    "cache_hit": None,
                    "prompt": "",
                    "retrieval_query": "",
                    "seed_used": seed_used
                })

            if pbar:
                pbar.update(1)

        return outputs

    def _calculate_metrics(
        self,
        outputs: List[Dict[str, Any]],
        test_case: TestCase,
        context_mode: ContextMode
    ) -> Dict[str, float]:
        """
        Calculate all configured metrics for this test case.

        Args:
            outputs: List of LLM outputs from repetitions
            test_case: Original test case
            context_mode: Context mode used

        Returns:
            Dictionary of metric scores
        """
        metrics: Dict[str, float] = {}

        # 1) No outputs at all → everything is 0
        if not outputs:
            for metric_name in self.config.metrics:
                metrics[metric_name] = 0.0
            metrics["had_error"] = 1.0
            return metrics

        primary_output = outputs[0]
        primary_answer = primary_output.get("answer", "")

        # 2) If first output is an error, treat whole case as failed
        if primary_output.get("error") or str(primary_answer).startswith("[ERROR:"):
            for metric_name in self.config.metrics:
                metrics[metric_name] = 0.0
            metrics["had_error"] = 1.0
            return metrics

        # 3) Extract contexts if available
        contexts = primary_output.get("contexts", [])
        if isinstance(contexts, str):
            try:
                contexts = json.loads(contexts) if contexts else []
            except json.JSONDecodeError:
                contexts = []

        # 4) Get question from input_data
        question = test_case.input_data.get("question") or \
                   test_case.input_data.get("fault_description", "")

        # Note: pdf_extraction_failed is now determined before sanity check in main loop
        # (removed duplicate logic here)

        # 5) Calculate each configured metric
        # RAG-specific metrics require contexts (applicable to MANUAL_FULL and RAG_RETRIEVAL)
        RAG_METRICS = {
            "faithfulness", "citation_coverage", "citation_correctness",
            "context_precision", "context_recall", "structural_completeness"
        }

        for metric_name in self.config.metrics:
            # Skip RAG metrics for BASELINE (set to NaN = not applicable)
            if metric_name in RAG_METRICS and context_mode == ContextMode.BASELINE:
                metrics[metric_name] = float('nan')
                continue

            try:
                score = self._calculate_single_metric(
                    metric_name,
                    primary_answer,
                    contexts,
                    question,
                    test_case,
                    outputs
                )
                metrics[metric_name] = round(score, 3)
            except Exception as e:
                metrics[metric_name] = 0.0
                print(f"Warning: Failed to calculate {metric_name}: {e}")

        # CRITICAL: Ensure had_error is ALWAYS present (robustness fix)
        if "had_error" not in metrics:
            metrics["had_error"] = 0.0

        return metrics


    def _calculate_single_metric(
        self,
        metric_name: str,
        answer: str,
        contexts: List[str],
        question: str,
        test_case: TestCase,
        all_outputs: List[Dict[str, Any]]
    ) -> float:
        """Calculate a single metric"""

        if metric_name == "answer_correctness":
            return self.metrics_calculator.answer_correctness(
                answer,
                test_case.ground_truth
            )

        elif metric_name == "citation_coverage":
            return self.metrics_calculator.citation_coverage(
                answer,
                contexts
            )

        elif metric_name == "citation_correctness":
            return self.metrics_calculator.citation_correctness(
                answer,
                contexts
            )

        elif metric_name == "faithfulness":
            return self.metrics_calculator.faithfulness(
                answer,
                contexts
            )

        elif metric_name == "context_precision":
            return self.metrics_calculator.context_precision(
                contexts,
                question,
                answer
            )

        elif metric_name == "context_recall":
            return self.metrics_calculator.context_recall(
                contexts,
                question,
                answer
            )

        elif metric_name == "answer_relevancy":
            return self.metrics_calculator.answer_relevancy(
                question,
                answer
            )

        elif metric_name == "hallucination_rate":
            # Hallucination can only be computed when contexts exist
            if not contexts:
                return float("nan")

            # Calculate dependencies
            faith = self.metrics_calculator.faithfulness(answer, contexts)
            cit_corr = self.metrics_calculator.citation_correctness(answer, contexts)

            # Use configurable thresholds from ExperimentConfig
            return float(self.metrics_calculator.hallucination_rate(
                faith,
                cit_corr,
                f_thresh=self.config.hallucination_faithfulness_thresh,
                c_thresh=self.config.hallucination_citation_thresh
            ))

        elif metric_name == "fkgl":
            return self.metrics_calculator.fkgl(answer)

        elif metric_name == "structural_completeness":
            return self.metrics_calculator.structural_completeness(
                answer,
                contexts
            )

        elif metric_name == "output_consistency":
            # Calculate consistency across all repetitions
            # Filter out error outputs before calculating consistency
            valid_answers = [
                out.get("answer", "")
                for out in all_outputs
                if not out.get("error") and not str(out.get("answer", "")).startswith("[ERROR:")
            ]
            if len(valid_answers) < 2:
                return 1.0  # Single valid answer is perfectly consistent
            return self.metrics_calculator.output_consistency(valid_answers)

        elif metric_name == "scenario_identification_rate":
            # Extract scenarios from answer and ground truth
            llm_scenarios = self._extract_scenarios_from_answer(answer)
            gt_scenarios = test_case.ground_truth.get("scenarios", [])
            return self.metrics_calculator.scenario_identification_rate(
                llm_scenarios,
                gt_scenarios
            )

        elif metric_name == "property_identification_rate":
            # Extract properties from answer and ground truth
            llm_properties = self._extract_properties_from_answer(answer)
            gt_properties = test_case.ground_truth.get("properties", [])
            return self.metrics_calculator.property_identification_rate(
                llm_properties,
                gt_properties
            )

        elif metric_name == "cer":
            ocr_text = test_case.ground_truth.get("ocr_text", "")
            gt_text = test_case.ground_truth.get("gt_text", "")
            if ocr_text and gt_text:
                return self.metrics_calculator.cer(ocr_text, gt_text)
            return 0.0

        elif metric_name == "wer":
            ocr_text = test_case.ground_truth.get("ocr_text", "")
            gt_text = test_case.ground_truth.get("gt_text", "")
            if ocr_text and gt_text:
                return self.metrics_calculator.wer(ocr_text, gt_text)
            return 0.0

        else:
            # Unknown metric
            return 0.0

    def _extract_scenarios_from_answer(self, answer: str) -> List[str]:
        """Extract repurposing scenarios from LLM answer"""
        # Look for "Component | Target System" format
        lines = answer.split('\n')
        scenarios = []
        for line in lines:
            if '|' in line:
                parts = line.split('|')
                if len(parts) >= 2:
                    target = parts[1].strip()
                    if target and not target.lower().startswith('target'):
                        scenarios.append(target)
        return scenarios

    def _extract_properties_from_answer(self, answer: str) -> List[str]:
        """Extract technical properties from LLM answer"""
        # Look for "Property: Value" format
        lines = answer.split('\n')
        properties = []
        for line in lines:
            if ':' in line:
                prop = line.split(':')[0].strip()
                if prop and len(prop.split()) <= 5:  # Reasonable property name
                    properties.append(prop)
        return properties

    def _check_output_schema(
        self,
        answer: str,
        required_sections: Optional[List[str]] = None
    ) -> tuple[bool, str]:
        """
        FAIRNESS: Validate that output follows required schema across all conditions.

        WARNING: This is DISABLED BY DEFAULT (config.enforce_output_schema=False) because
        the hardcoded schema does NOT match the default diagnosis prompts!

        Default required sections (if enabled):
        - Diagnosis
        - Root Cause
        - Steps
        - Safety
        - Citations

        Compatible prompts: NONE of the built-in prompts! You must use custom prompts.
        - DIAGNOSIS_BASELINE asks for: Diagnosis, Recommended Action, Tools & Parts, Safety Warnings
        - DIAGNOSIS_BASELINE_V1 asks for: Diagnosis, Root Cause, Repair Procedure, Required Materials, Safety
        - etc.

        To use schema validation:
        1. Create custom prompts that request the exact sections above
        2. Set config.enforce_output_schema = True
        3. OR modify required_sections to match your prompts

        Args:
            answer: LLM answer text
            required_sections: List of required section headers (uses default if None)

        Returns:
            Tuple of (passed: bool, error_message: str)
        """
        if required_sections is None:
            # Default schema for diagnosis task
            required_sections = ["Diagnosis", "Root Cause", "Steps", "Safety", "Citations"]

        # Skip schema check for error outputs
        if answer.startswith("[ERROR:"):
            return True, ""

        missing_sections = []
        for section in required_sections:
            # Case-insensitive search for section headers
            # Matches "## Diagnosis", "**Diagnosis:**", "Diagnosis:", etc.
            pattern = rf'(?:^|\n)(?:#+\s*)?(?:\*\*)?{re.escape(section)}(?:\*\*)?(?::|$)'
            if not re.search(pattern, answer, re.IGNORECASE):
                missing_sections.append(section)

        if missing_sections:
            return False, f"Missing required sections: {', '.join(missing_sections)}"

        return True, ""

    def _sanity_check_contexts(
        self,
        outputs: List[Dict[str, Any]],
        context_mode: ContextMode,
        top_k: int
    ) -> tuple[bool, str]:
        """
        Sanity check: validate that contexts match expected counts for context_mode.

        Rules:
        - BASELINE: len(contexts) == 0
        - MANUAL_FULL: len(contexts) > 0
        - RAG_RETRIEVAL: 0 < len(contexts) <= top_k

        Args:
            outputs: List of LLM outputs from repetitions
            context_mode: Context mode used
            top_k: Top-k parameter for RAG retrieval

        Returns:
            Tuple of (passed: bool, error_message: str)
        """
        if not outputs:
            return False, "No outputs to check"

        # Check first output (representative)
        first_output = outputs[0]

        # Skip check if there was an error in the output
        if first_output.get("error") or str(first_output.get("answer", "")).startswith("[ERROR:"):
            return True, ""  # Don't fail sanity check for error cases

        contexts = first_output.get("contexts", [])

        # Handle string contexts (legacy)
        if isinstance(contexts, str):
            try:
                import json
                contexts = json.loads(contexts) if contexts else []
            except json.JSONDecodeError:
                contexts = []

        num_contexts = len(contexts)

        if context_mode == ContextMode.BASELINE:
            if num_contexts != 0:
                return False, f"BASELINE should have 0 contexts, got {num_contexts}"

        elif context_mode == ContextMode.MANUAL_FULL:
            if num_contexts == 0:
                return False, f"MANUAL_FULL should have >0 contexts, got 0"

        elif context_mode == ContextMode.RAG_RETRIEVAL:
            if num_contexts == 0:
                return False, f"RAG_RETRIEVAL should have >0 contexts, got 0"
            if num_contexts > top_k:
                return False, f"RAG_RETRIEVAL should have <={top_k} contexts, got {num_contexts}"

        return True, ""

    def _aggregate_results(self) -> ExperimentResults:
        """
        Aggregate results and create analysis tables.

        Returns:
            ExperimentResults with analysis tables
        """
        # CRITICAL: Pass metrics list as ALLOWLIST for aggregation
        # Only columns in config.metrics will be averaged in performance tables
        return ExperimentResults(self.results, metrics=self.config.metrics)


# ================= Helper Functions =================

def run_quick_experiment(
    task_type: str,
    test_cases: List[TestCase],
    model: LLMProvider = LLMProvider.GEMINI,
    llm_service: Optional[LLMService] = None,
    metrics: Optional[List[str]] = None
) -> ExperimentResults:
    """
    Quick experiment runner for testing.

    Args:
        task_type: Type of task
        test_cases: List of test cases
        model: LLM provider to use
        llm_service: LLM service instance (creates default if None)
        metrics: List of metrics (uses defaults if None)

    Returns:
        ExperimentResults
    """
    # Create default LLM service if not provided
    if llm_service is None:
        llm_service = LLMService()

    # Use default metrics for task type if not provided
    if metrics is None:
        metrics = MetricsConfig.get_metrics_for_task(task_type)

    # Create config
    config = ExperimentConfig(
        task_type=task_type,
        models=[model],
        metrics=metrics,
        n_repetitions=1,
        include_baseline=False
    )

    # Create dataset
    from .dataset import Dataset
    dataset = Dataset(test_cases)

    # Run experiment
    runner = ExperimentRunner(config, llm_service)
    return runner.run(dataset, verbose=True)
