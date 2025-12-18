"""
Core data structures and base classes for the LLM Evaluation Framework.

This module defines the fundamental abstractions that make the framework
domain-agnostic and reusable across different evaluation tasks.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
import pandas as pd


# ================= LLM Provider Enum =================

class LLMProvider(Enum):
    """Supported LLM providers"""
    GEMINI = "gemini"
    GEMMA3_4B = "gemma3-4b"
    GEMMA3_12B = "gemma3-12b"
    GEMMA3_27B = "gemma3-27b"
    GEMMA3 = "gemma3-27b"  # Alias for backward compatibility
    DEEPSEEK = "deepseek"
    GPT5 = "gpt5"


# ================= Context Mode Enum =================

class ContextMode(Enum):
    """Context modes for experiments"""
    BASELINE = "baseline"  # No manual
    MANUAL_FULL = "manual_full"  # Full manual text (no retrieval)
    RAG_RETRIEVAL = "rag_retrieval"  # Retrieve top-k chunks from manual


# ================= Test Case Structure =================

@dataclass
class TestCase:
    """
    Generic test case structure that works for ANY domain.

    Examples:
    - Diagnosis: fault_description → expected_diagnosis
    - Repurposing: component_name → expected_scenarios
    - ML Recommendation: problem_description → expected_algorithms

    Attributes:
        case_id: Unique identifier for this test case
        input_data: Flexible dict containing any input fields
        ground_truth: Flexible dict containing expected outputs
        metadata: Additional information (complexity, category, etc.)
        context_document: Optional path to PDF/document for RAG
    """
    case_id: str
    input_data: Dict[str, Any]
    ground_truth: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    context_document: Optional[str] = None

    def __post_init__(self):
        """Validate test case after initialization"""
        if not self.case_id:
            raise ValueError("case_id cannot be empty")
        if not isinstance(self.input_data, dict):
            raise TypeError("input_data must be a dictionary")
        if not isinstance(self.ground_truth, dict):
            raise TypeError("ground_truth must be a dictionary")


# ================= Experiment Configuration =================

@dataclass
class ExperimentConfig:
    """
    Configuration for any LLM evaluation experiment.

    This class defines all parameters needed to run a reproducible experiment
    following the methodology from published research papers.

    Attributes:
        task_type: Type of task ("diagnosis", "repurposing", "ml_recommendation", etc.)
        models: List of LLM providers to test
        metrics: List of metric names to calculate
        n_repetitions: Number of times to run each test (for consistency analysis)
        context_modes: List of context modes to test (default: [BASELINE, RAG_RETRIEVAL])
        temperature: LLM temperature parameter (default: 0.2)
        max_tokens: Maximum output tokens (default: 2048)
        top_p: Nucleus sampling parameter (default: 1.0, typical: 0.9)
        random_seed: Random seed for reproducibility
        custom_prompt: Optional custom PromptTemplate to use instead of default
        prompt_variant: Optional variant name (e.g., "v1", "v2", "concise") to use specific prompt version
        top_k_retrieval: Number of chunks to retrieve in RAG_RETRIEVAL mode
        hallucination_faithfulness_thresh: Faithfulness threshold for hallucination detection (default: 0.25)
        hallucination_citation_thresh: Citation correctness threshold for hallucination detection (default: 0.30)
        chunk_size: Size of text chunks in characters (default: 1000)
        max_context_chars: Maximum context characters to use (default: 40000)
        manual_full_max_chunks: Maximum chunks for MANUAL_FULL mode (default: 40)
        context_budget_overrides: Per-provider budget overrides (dict[provider_value, dict[budget_params]])
        include_baseline: DEPRECATED - use context_modes instead
        enforce_output_schema: Validate output structure (default: False). WARNING: Only enable with compatible prompts!
    """
    task_type: str
    models: List[LLMProvider]
    metrics: List[str]
    n_repetitions: int = 5
    context_modes: Optional[List['ContextMode']] = None
    temperature: float = 0.2
    max_tokens: int = 2048
    baseline_max_tokens: Optional[int] = None
    top_p: float = 1.0  # Nucleus sampling parameter (1.0 = disabled, 0.9 = typical)
    random_seed: int = 42
    custom_prompt: Optional[Any] = None  # PromptTemplate (avoiding circular import)
    prompt_variant: Optional[str] = None
    top_k_retrieval: int = 6
    hallucination_faithfulness_thresh: float = 0.25
    hallucination_citation_thresh: float = 0.30
    chunk_size: int = 1000
    max_context_chars: int = 40000
    manual_full_max_chunks: int = 40
    context_budget_overrides: Optional[Dict[str, Dict[str, int]]] = None
    include_baseline: Optional[bool] = None  # Backward compatibility
    enforce_output_schema: bool = False  # FAIRNESS: Validate output structure (disabled by default)

    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.task_type:
            raise ValueError("task_type cannot be empty")
        if not self.models:
            raise ValueError("models list cannot be empty")
        if not self.metrics:
            raise ValueError("metrics list cannot be empty")
        if self.n_repetitions < 1:
            raise ValueError("n_repetitions must be >= 1")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

        # Backward compatibility: map include_baseline to context_modes
        if self.context_modes is None:
            if self.include_baseline is not None:
                import warnings
                warnings.warn(
                    "DEPRECATED: 'include_baseline' parameter is deprecated. "
                    "Use 'context_modes' instead with explicit [ContextMode.BASELINE, ContextMode.RAG_RETRIEVAL].",
                    DeprecationWarning,
                    stacklevel=2
                )
                # Legacy behavior
                if self.include_baseline:
                    self.context_modes = [ContextMode.BASELINE, ContextMode.RAG_RETRIEVAL]
                else:
                    self.context_modes = [ContextMode.RAG_RETRIEVAL]
            else:
                # Default: compare baseline vs RAG
                self.context_modes = [ContextMode.BASELINE, ContextMode.RAG_RETRIEVAL]

    def get_context_budgets(self, provider: LLMProvider) -> Dict[str, int]:
        """
        Resolve context budgets for a specific provider.

        Returns global defaults unless overridden in context_budget_overrides.

        Args:
            provider: LLM provider to get budgets for

        Returns:
            Dictionary with keys: chunk_size, max_context_chars, manual_full_max_chunks
        """
        # Start with global defaults
        budgets = {
            "chunk_size": self.chunk_size,
            "max_context_chars": self.max_context_chars,
            "manual_full_max_chunks": self.manual_full_max_chunks,
        }

        # Apply provider-specific overrides if they exist
        if self.context_budget_overrides and provider.value in self.context_budget_overrides:
            overrides = self.context_budget_overrides[provider.value]
            budgets.update(overrides)

        return budgets


# ================= Experiment Results =================

def _sanitize_csv_value(value: Any) -> Any:
    """
    Sanitize CSV value to prevent Excel formula injection.

    Handles both single-line and multiline strings. For multiline content,
    checks each line and prefixes dangerous lines with a tab character.

    Dangerous patterns:
    - Lines starting with =, +, -, @ (formula injection)
    - This includes the first line and any subsequent lines in multiline text

    Args:
        value: Cell value to sanitize

    Returns:
        Sanitized value (dangerous lines prefixed with tab)
    """
    if not isinstance(value, str) or len(value) == 0:
        return value

    # Check if multiline
    if '\n' in value:
        # Sanitize each line independently
        lines = value.split('\n')
        sanitized_lines = []
        for line in lines:
            if len(line) > 0 and line[0] in ('=', '+', '-', '@'):
                sanitized_lines.append('\t' + line)
            else:
                sanitized_lines.append(line)
        return '\n'.join(sanitized_lines)
    else:
        # Single line: check first character
        if value[0] in ('=', '+', '-', '@'):
            return '\t' + value
        return value


class ExperimentResults:
    """
    Container for experiment results with analysis methods.

    This class stores raw results and provides methods to generate
    publication-ready tables following the format from research papers.

    Methods generate tables like:
    - Table 1: Overall performance (per model, averaged across all test cases)
    - Table 2: Performance by complexity cluster
    - Table 3: Output consistency analysis
    """

    def __init__(self, raw_results: List[Dict[str, Any]], metrics: Optional[List[str]] = None):
        """
        Initialize results container.

        Args:
            raw_results: List of dictionaries, each containing metrics for one test run
            metrics: REQUIRED list of metric names from ExperimentConfig.metrics (allowlist for aggregation)
        """
        self.raw_results = raw_results
        self.df = pd.DataFrame(raw_results) if raw_results else pd.DataFrame()

        # CRITICAL: Store metrics as ALLOWLIST for aggregation
        # Only columns in this list will be averaged in performance tables
        # This prevents accidental averaging of debug/operational fields
        self.metrics = metrics if metrics is not None else []

        # Analysis tables (generated lazily)
        self._overall_table: Optional[pd.DataFrame] = None
        self._complexity_table: Optional[pd.DataFrame] = None
        self._consistency_table: Optional[pd.DataFrame] = None

    @property
    def overall_table(self) -> pd.DataFrame:
        """
        Overall performance table (like Table 1 in both papers).

        Returns DataFrame with columns:
        - model: Model name
        - use_rag: Whether RAG was used
        - [metric columns]: Average scores for each metric
        """
        if self._overall_table is None:
            self._overall_table = self._create_overall_table()
        return self._overall_table

    @overall_table.setter
    def overall_table(self, value: pd.DataFrame):
        self._overall_table = value

    @property
    def complexity_table(self) -> pd.DataFrame:
        """
        Performance by complexity table (like Dörnbach's Table 2).

        Returns DataFrame with performance broken down by complexity cluster.
        """
        if self._complexity_table is None:
            self._complexity_table = self._create_complexity_table()
        return self._complexity_table

    @complexity_table.setter
    def complexity_table(self, value: pd.DataFrame):
        self._complexity_table = value

    @property
    def consistency_table(self) -> pd.DataFrame:
        """
        Output consistency analysis (like Sonntag's OCR metric).

        Returns DataFrame with consistency scores per model.
        """
        if self._consistency_table is None:
            self._consistency_table = self._create_consistency_table()
        return self._consistency_table

    @consistency_table.setter
    def consistency_table(self, value: pd.DataFrame):
        self._consistency_table = value

    def _create_overall_table(self) -> pd.DataFrame:
        """Create overall performance table"""
        if self.df.empty:
            return pd.DataFrame()

        # CHANGE #4: Filter out both runtime errors AND sanity check failures
        had_error = self.df.get("had_error", False).fillna(False).astype(bool)
        sanity_failed = self.df.get("sanity_check_failed", False).fillna(False).astype(bool)
        valid_df = self.df[~(had_error | sanity_failed)]

        if valid_df.empty:
            return pd.DataFrame()

        # Group by model and context mode
        group_cols = ["model"]
        if "context_mode" in valid_df.columns:
            group_cols.append("context_mode")
        elif "use_rag" in valid_df.columns:
            # Backward compatibility
            group_cols.append("use_rag")

        # STRICT ALLOWLIST: Only average columns explicitly requested in config.metrics
        # This prevents accidental averaging of debug/operational fields
        # If someone adds "llm_call_duration" or "prompt_tokens_count" without adding to metrics,
        # it will NOT be averaged (safe by default)

        # Audit trail fields (never metrics, always excluded)
        AUDIT_TRAIL_FIELDS = {"run_id"}

        metric_cols = [
            m for m in self.metrics
            if m in valid_df.columns and m not in AUDIT_TRAIL_FIELDS
        ]

        # Validate: warn if requested metrics are missing
        missing_metrics = [m for m in self.metrics if m not in valid_df.columns]
        if missing_metrics:
            import warnings
            warnings.warn(
                f"Requested metrics not found in results: {missing_metrics}. "
                f"These will be skipped in aggregation.",
                UserWarning
            )

        # THESIS REQUIREMENT: Add denominators (n_total, n_valid, n_excluded, excluded_rate)
        # This shows reviewers how many runs contributed to each mean
        overall_means = valid_df.groupby(group_cols)[metric_cols].mean()

        # Compute denominators per group
        n_valid = valid_df.groupby(group_cols).size()
        n_total = self.df.groupby(group_cols).size()
        n_excluded = n_total - n_valid
        excluded_rate = (n_excluded / n_total).fillna(0.0)

        # Add denominator columns to the means table
        overall_means['n_total'] = n_total
        overall_means['n_valid'] = n_valid
        overall_means['n_excluded'] = n_excluded
        overall_means['excluded_rate'] = excluded_rate

        return overall_means

    def _create_complexity_table(self) -> pd.DataFrame:
        """Create performance by complexity table"""
        if self.df.empty or "complexity" not in self.df.columns:
            return pd.DataFrame()

        # CHANGE #4: Filter out both runtime errors AND sanity check failures
        had_error = self.df.get("had_error", False).fillna(False).astype(bool)
        sanity_failed = self.df.get("sanity_check_failed", False).fillna(False).astype(bool)
        valid_df = self.df[~(had_error | sanity_failed)]

        if valid_df.empty:
            return pd.DataFrame()

        # Group by model, context mode, and complexity
        group_cols = ["model"]
        if "context_mode" in valid_df.columns:
            group_cols.append("context_mode")
        elif "use_rag" in valid_df.columns:
            # Backward compatibility
            group_cols.append("use_rag")
        group_cols.append("complexity")

        # STRICT ALLOWLIST: Only average metrics from config.metrics
        metric_cols = [m for m in self.metrics if m in valid_df.columns]

        # THESIS REQUIREMENT: Add denominators (n_total, n_valid, n_excluded, excluded_rate)
        complexity_means = valid_df.groupby(group_cols)[metric_cols].mean()

        # Compute denominators per group
        n_valid = valid_df.groupby(group_cols).size()
        n_total = self.df.groupby(group_cols).size()
        n_excluded = n_total - n_valid
        excluded_rate = (n_excluded / n_total).fillna(0.0)

        # Add denominator columns
        complexity_means['n_total'] = n_total
        complexity_means['n_valid'] = n_valid
        complexity_means['n_excluded'] = n_excluded
        complexity_means['excluded_rate'] = excluded_rate

        return complexity_means

    def _create_consistency_table(self) -> pd.DataFrame:
        """Create consistency analysis table"""
        if self.df.empty or "output_consistency" not in self.df.columns:
            return pd.DataFrame()

        # CHANGE #4: Filter out both runtime errors AND sanity check failures
        had_error = self.df.get("had_error", False).fillna(False).astype(bool)
        sanity_failed = self.df.get("sanity_check_failed", False).fillna(False).astype(bool)
        valid_df = self.df[~(had_error | sanity_failed)]

        if valid_df.empty:
            return pd.DataFrame()

        group_cols = ["model"]
        if "context_mode" in valid_df.columns:
            group_cols.append("context_mode")
        elif "use_rag" in valid_df.columns:
            # Backward compatibility
            group_cols.append("use_rag")

        consistency = valid_df.groupby(group_cols)["output_consistency"].mean()
        return consistency.to_frame()

    def export_manifest(
        self,
        output_dir: str,
        config: 'ExperimentConfig',
        dataset_path: Optional[str] = None,
        run_id: Optional[str] = None
    ):
        """
        Export run manifest for full reproducibility.

        Creates run_manifest.json with:
        - Run ID (for joining to raw_results.csv)
        - Full experiment configuration
        - Environment information (Python, packages, git)
        - Model identifiers
        - Dataset information

        Args:
            output_dir: Directory to save manifest
            config: Experiment configuration
            dataset_path: Optional path to dataset file
            run_id: Optional run ID (extracted from results if not provided)
        """
        import os
        import sys
        import subprocess
        import hashlib
        import json
        from datetime import datetime

        os.makedirs(output_dir, exist_ok=True)

        # Extract run_id from dataframe if not provided
        if run_id is None and not self.df.empty and "run_id" in self.df.columns:
            # All rows should have same run_id, take first
            run_id = self.df["run_id"].iloc[0]

        manifest = {
            "run_id": run_id,  # CRITICAL: Allows joining manifest to raw_results
            "timestamp": datetime.now().isoformat(),
            "experiment_config": {
                "task_type": config.task_type,
                "models": [m.value for m in config.models],
                "context_modes": [cm.value for cm in config.context_modes],
                "metrics": config.metrics,
                "n_repetitions": config.n_repetitions,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "baseline_max_tokens": config.baseline_max_tokens,
                "random_seed": config.random_seed,
                "top_k_retrieval": config.top_k_retrieval,
                "chunk_size": config.chunk_size,
                "max_context_chars": config.max_context_chars,
                "manual_full_max_chunks": config.manual_full_max_chunks,
                "hallucination_thresholds": {
                    "faithfulness": config.hallucination_faithfulness_thresh,
                    "citation": config.hallucination_citation_thresh
                }
            },
            "environment": {
                "python_version": sys.version,
                "platform": sys.platform,
            },
            "dataset": {
                "n_test_cases": len(self.df) if not self.df.empty else 0,
            }
        }

        # Add dataset hash if path provided
        if dataset_path and os.path.exists(dataset_path):
            with open(dataset_path, 'rb') as f:
                dataset_bytes = f.read()
                manifest["dataset"]["path"] = dataset_path
                manifest["dataset"]["hash"] = hashlib.sha256(dataset_bytes).hexdigest()

        # Try to get key package versions
        try:
            import pkg_resources
            key_packages = [
                'sentence-transformers',
                'sklearn',
                'scikit-learn',
                'replicate',
                'google-generativeai',
                'pypdf',
                'PyPDF2',
                'pandas',
                'tqdm'
            ]
            versions = {}
            for pkg in key_packages:
                try:
                    ver = pkg_resources.get_distribution(pkg).version
                    versions[pkg] = ver
                except Exception:
                    pass
            if versions:
                manifest["environment"]["package_versions"] = versions
        except Exception:
            pass

        # Write manifest
        manifest_path = os.path.join(output_dir, "run_manifest.json")
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"Manifest exported to: {manifest_path}")

    def export_all(self, output_dir: str, config: Optional['ExperimentConfig'] = None, dataset_path: Optional[str] = None):
        """
        Export all tables and manifest to CSV/JSON files.

        CSV files use German/European format for direct opening in Excel:
        - Semicolon (;) as delimiter
        - Comma (,) as decimal separator
        - UTF-8 with BOM encoding

        Args:
            output_dir: Directory to save CSV files
            config: Optional experiment configuration (for manifest)
            dataset_path: Optional dataset path (for manifest)
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Overall performance
        if not self.overall_table.empty:
            df_overall = self.overall_table.reset_index()
            # Sanitize string columns for CSV injection prevention
            for col in df_overall.columns:
                if df_overall[col].dtype == 'object':
                    df_overall[col] = df_overall[col].apply(_sanitize_csv_value)
            df_overall.to_csv(
                os.path.join(output_dir, "table1_overall_performance.csv"),
                index=False,
                sep=";",
                decimal=",",
                encoding="utf-8-sig"
            )

        # By complexity
        if not self.complexity_table.empty:
            df_complexity = self.complexity_table.reset_index()
            # Sanitize string columns for CSV injection prevention
            for col in df_complexity.columns:
                if df_complexity[col].dtype == 'object':
                    df_complexity[col] = df_complexity[col].apply(_sanitize_csv_value)
            df_complexity.to_csv(
                os.path.join(output_dir, "table2_complexity_analysis.csv"),
                index=False,
                sep=";",
                decimal=",",
                encoding="utf-8-sig"
            )

        # Consistency
        if not self.consistency_table.empty:
            df_consistency = self.consistency_table.reset_index()
            # Sanitize string columns for CSV injection prevention
            for col in df_consistency.columns:
                if df_consistency[col].dtype == 'object':
                    df_consistency[col] = df_consistency[col].apply(_sanitize_csv_value)
            df_consistency.to_csv(
                os.path.join(output_dir, "table3_output_consistency.csv"),
                index=False,
                sep=";",
                decimal=",",
                encoding="utf-8-sig"
            )

        # Raw results - split into two files for usability and reproducibility
        if not self.df.empty:
            # 1. raw_results_view.csv: Human-friendly view (drop long text fields)
            view_columns = [col for col in self.df.columns if col not in ('prompt_text', 'contexts_text')]
            df_view = self.df[view_columns].copy()

            # Sanitize all string columns to prevent CSV injection
            for col in df_view.columns:
                if df_view[col].dtype == 'object':  # String columns
                    df_view[col] = df_view[col].apply(_sanitize_csv_value)

            df_view.to_csv(
                os.path.join(output_dir, "raw_results_view.csv"),
                index=False,
                sep=";",
                decimal=",",
                encoding="utf-8-sig"
            )

            # 2. raw_results_audit.jsonl: Complete audit trail (full prompt/context text)
            # JSONL format: one JSON object per line, preserves all data for reproducibility
            import json
            audit_path = os.path.join(output_dir, "raw_results_audit.jsonl")
            with open(audit_path, 'w', encoding='utf-8') as f:
                for _, row in self.df.iterrows():
                    json.dump(row.to_dict(), f, ensure_ascii=False)
                    f.write('\n')

            print(f"Raw results exported:")
            print(f"  - View: raw_results_view.csv ({len(df_view)} rows, {len(view_columns)} columns)")
            print(f"  - Audit: raw_results_audit.jsonl (full reproducibility data)")

        # Export manifest if config provided
        if config:
            self.export_manifest(output_dir, config, dataset_path)

    def summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for the experiment.

        Returns:
            Dictionary with summary statistics
        """
        if self.df.empty:
            return {}

        # STRICT ALLOWLIST: Only use metrics from config.metrics
        metric_cols = [m for m in self.metrics if m in self.df.columns]

        # CHANGE #4: Filter out both runtime errors AND sanity check failures for averaging
        had_error = self.df.get("had_error", False).fillna(False).astype(bool)
        sanity_failed = self.df.get("sanity_check_failed", False).fillna(False).astype(bool)
        valid_df = self.df[~(had_error | sanity_failed)]

        return {
            "n_test_cases": len(self.df),
            "n_valid_cases": len(valid_df),
            "n_models": self.df["model"].nunique() if "model" in self.df.columns else 0,
            "metrics": metric_cols,
            "avg_scores": valid_df[metric_cols].mean().to_dict() if not valid_df.empty else {},
        }

    def __repr__(self) -> str:
        """String representation of results"""
        n_results = len(self.raw_results)
        n_models = self.df["model"].nunique() if not self.df.empty and "model" in self.df.columns else 0
        return f"ExperimentResults(n_results={n_results}, n_models={n_models})"


# ================= Helper Functions =================

def validate_test_case(test_case: TestCase, task_type: Optional[str] = None) -> bool:
    """
    Validate that a test case has required fields for a specific task type.

    Args:
        test_case: TestCase to validate
        task_type: Optional task type for task-specific validation
                   ("diagnosis", "repurposing", "ml_recommendation", etc.)

    Returns:
        True if valid, raises ValueError otherwise
    """
    # Basic validation
    if not test_case.case_id:
        raise ValueError("Test case must have a case_id")

    if not test_case.input_data:
        raise ValueError(f"Test case {test_case.case_id} has empty input_data")

    if not test_case.ground_truth:
        raise ValueError(f"Test case {test_case.case_id} has empty ground_truth")

    # CRITICAL: Check for ground-truth leakage
    # Prevent ground-truth field names from appearing in input_data
    # This ensures the model cannot simply copy answers from the input
    for gt_key in test_case.ground_truth.keys():
        if gt_key in test_case.input_data:
            raise ValueError(
                f"GROUND-TRUTH LEAKAGE in test case {test_case.case_id}: "
                f"Field '{gt_key}' appears in BOTH input_data and ground_truth. "
                f"This allows the model to see the answer in the prompt. "
                f"Ground-truth fields must ONLY appear in ground_truth, never in input_data."
            )

    # Task-specific validation
    if task_type == "diagnosis":
        # Diagnosis tasks require specific input fields
        required_inputs = ["fault_description", "appliance"]
        for field in required_inputs:
            if field not in test_case.input_data or not test_case.input_data[field]:
                raise ValueError(
                    f"Diagnosis test case {test_case.case_id} missing required input field: '{field}'. "
                    f"Available fields: {list(test_case.input_data.keys())}"
                )

        # Diagnosis tasks require diagnosis in ground truth
        if "diagnosis" not in test_case.ground_truth or not test_case.ground_truth["diagnosis"]:
            raise ValueError(
                f"Diagnosis test case {test_case.case_id} missing required ground_truth field: 'diagnosis'. "
                f"Available fields: {list(test_case.ground_truth.keys())}"
            )

    elif task_type == "repurposing":
        # Repurposing tasks require component and scenarios
        if "component" not in test_case.input_data or not test_case.input_data["component"]:
            raise ValueError(
                f"Repurposing test case {test_case.case_id} missing required input field: 'component'"
            )

        if "scenarios" not in test_case.ground_truth or not test_case.ground_truth["scenarios"]:
            raise ValueError(
                f"Repurposing test case {test_case.case_id} missing required ground_truth field: 'scenarios'"
            )

    elif task_type == "ml_recommendation":
        # ML recommendation tasks require problem description
        if "problem_description" not in test_case.input_data or not test_case.input_data["problem_description"]:
            raise ValueError(
                f"ML recommendation test case {test_case.case_id} missing required input field: 'problem_description'"
            )

        if "ml_type" not in test_case.ground_truth or not test_case.ground_truth["ml_type"]:
            raise ValueError(
                f"ML recommendation test case {test_case.case_id} missing required ground_truth field: 'ml_type'"
            )

    return True
