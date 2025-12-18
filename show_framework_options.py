"""
Framework Options Guide
=======================
This script displays ALL available options in the framework:
- LLM Models
- Metrics
- Prompt Variants
- How to build complete experiments

Run: python show_framework_options.py
"""

import sys
import io

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from framework import LLMProvider
from framework.prompts import PromptLibrary
from framework.metrics import MetricsConfig


def print_section(title):
    """Print a formatted section header"""
    print()
    print("="*80)
    print(f"  {title}")
    print("="*80)
    print()


def show_llm_models():
    """Display all available LLM models"""
    print_section("AVAILABLE LLM MODELS")

    models = [
        ("GEMINI", "Google Gemini (Pro)", "General purpose, supports PDF upload via Files API"),
        ("GEMMA3_4B", "Google Gemma 3 (4B)", "Lightweight model, faster responses"),
        ("GEMMA3_12B", "Google Gemma 3 (12B)", "Balanced performance and speed"),
        ("GEMMA3_27B", "Google Gemma 3 (27B)", "Large model, best quality"),
        ("DEEPSEEK", "DeepSeek", "Coding and reasoning focused"),
        ("GPT5", "GPT-5", "OpenAI's latest (via Replicate)"),
    ]

    print(f"{'Model Name':<20} {'Full Name':<30} {'Description'}")
    print("-" * 80)
    for name, full, desc in models:
        print(f"{name:<20} {full:<30} {desc}")

    print()
    print("Usage in ExperimentConfig:")
    print("  models=[LLMProvider.GEMINI, LLMProvider.GEMMA3_12B]")
    print()
    print("API Keys Required (in .env file):")
    print("  - GEMINI_API_KEY         (for Gemini)")
    print("  - REPLICATE_API_KEY      (for Gemma, DeepSeek, GPT-5)")


def show_metrics():
    """Display all available metrics"""
    print_section("AVAILABLE METRICS")

    metrics = {
        "RAG-Specific Metrics": [
            ("citation_coverage", "Fraction of sentences with citations (0-1)"),
            ("citation_correctness", "Are citations grounded in context? (0-1)"),
            ("faithfulness", "Overlap-based grounding score (0-1)"),
            ("context_precision", "Precision of retrieved context (0-1)"),
            ("context_recall", "Recall of retrieved context (0-1)"),
        ],
        "Quality Metrics": [
            ("answer_correctness", "Semantic similarity with ground truth (0-1)"),
            ("answer_relevancy", "TF-IDF cosine similarity with question (0-1)"),
            ("hallucination_rate", "Binary: 0=grounded, 1=hallucinated"),
        ],
        "Domain-Specific Metrics": [
            ("scenario_identification_rate", "For repurposing: fraction of scenarios found"),
            ("property_identification_rate", "For repurposing: fraction of properties found"),
            ("output_consistency", "Consistency across repetitions (0-1)"),
        ],
        "Readability & Structure": [
            ("fkgl", "Flesch-Kincaid Grade Level (readability score)"),
            ("structural_completeness", "Has required sections? (0-1)"),
        ],
        "OCR Metrics": [
            ("cer", "Character Error Rate (for OCR evaluation)"),
            ("wer", "Word Error Rate (for OCR evaluation)"),
        ],
    }

    for category, metric_list in metrics.items():
        print(f"{category}:")
        print("-" * 80)
        for name, desc in metric_list:
            print(f"  {name:<30} {desc}")
        print()

    print("Pre-configured Metric Sets:")
    print("-" * 80)
    print(f"  {'DIAGNOSIS_METRICS':<30} {MetricsConfig.DIAGNOSIS_METRICS}")
    print(f"  {'REPURPOSING_METRICS':<30} {MetricsConfig.REPURPOSING_METRICS}")
    print(f"  {'ML_RECOMMENDATION_METRICS':<30} {MetricsConfig.ML_RECOMMENDATION_METRICS}")
    print()
    print("Usage in ExperimentConfig:")
    print("  # Use pre-configured set:")
    print("  metrics=MetricsConfig.DIAGNOSIS_METRICS")
    print()
    print("  # Or choose your own:")
    print("  metrics=['answer_correctness', 'faithfulness', 'citation_coverage']")


def show_prompts():
    """Display all available prompt variants"""
    print_section("AVAILABLE PROMPT VARIANTS")

    templates = PromptLibrary.list_available_templates()

    # Group by task type
    task_groups = {
        "Diagnosis": [],
        "Repurposing": [],
        "ML Recommendation": [],
    }

    for name, desc in templates.items():
        if "diagnosis" in name.lower():
            task_groups["Diagnosis"].append((name, desc))
        elif "repurposing" in name.lower():
            task_groups["Repurposing"].append((name, desc))
        elif "ml_recommendation" in name.lower():
            task_groups["ML Recommendation"].append((name, desc))

    for task, prompts in task_groups.items():
        if prompts:
            print(f"{task} Prompts:")
            print("-" * 80)
            for name, desc in prompts:
                print(f"  {name}")
                print(f"    -> {desc}")
                print()

    print()
    print("How to Use Prompt Variants:")
    print("-" * 80)
    print()
    print("Option 1: Use default prompt (no variant specified)")
    print("  config = ExperimentConfig(task_type='diagnosis')")
    print()
    print("Option 2: Choose a built-in variant")
    print("  config = ExperimentConfig(")
    print("      task_type='diagnosis',")
    print("      prompt_variant='v1'  # detailed, comprehensive")
    print("  )")
    print()
    print("Option 3: Create custom prompt")
    print("  from framework.prompts import CustomPromptBuilder")
    print("  builder = CustomPromptBuilder()")
    print("  builder.set_background('You are...')")
    print("  builder.set_task('...')")
    print("  custom = builder.build()")
    print("  config = ExperimentConfig(")
    print("      task_type='diagnosis',")
    print("      custom_prompt=custom")
    print("  )")


def show_complete_example():
    """Show how to build a complete experiment"""
    print_section("COMPLETE EXPERIMENT EXAMPLE")

    example = '''
# =====================================================================
# Step 1: Import Framework
# =====================================================================
from framework import (
    TestCase,
    Dataset,
    ExperimentConfig,
    ExperimentRunner,
    LLMProvider,
    LLMService,
    ResultsAnalyzer,
    MetricsConfig
)

# =====================================================================
# Step 2: Create Test Cases
# =====================================================================
test_cases = [
    TestCase(
        case_id="001",
        input_data={
            "fault_description": "Error E18",
            "appliance": "Bosch dishwasher"
        },
        ground_truth={
            "diagnosis": "Drain pump blockage"
        },
        metadata={"complexity": 1},
        context_document="manuals/bosch.pdf"  # Optional: for RAG
    ),
    # ... add more test cases
]

dataset = Dataset(test_cases)

# =====================================================================
# Step 3: Configure Experiment
# =====================================================================
config = ExperimentConfig(
    # Task type
    task_type="diagnosis",

    # Models to test
    models=[
        LLMProvider.GEMINI,
        LLMProvider.GEMMA3_12B
    ],

    # Metrics to calculate
    metrics=MetricsConfig.DIAGNOSIS_METRICS,
    # Or custom: metrics=["answer_correctness", "faithfulness"]

    # Experiment parameters
    n_repetitions=5,          # Run each test 5 times
    include_baseline=True,     # Compare with/without RAG
    temperature=0.2,           # Lower = more consistent
    max_tokens=2048,

    # Prompt customization (optional)
    prompt_variant="v1",       # or None for default
    # OR custom_prompt=my_prompt
)

# =====================================================================
# Step 4: Run Experiment
# =====================================================================
llm_service = LLMService()  # Reads API keys from .env
runner = ExperimentRunner(config, llm_service)
results = runner.run(dataset, verbose=True)

# =====================================================================
# Step 5: Analyze Results
# =====================================================================
analyzer = ResultsAnalyzer(results)

# Print to console
analyzer.print_summary()

# Export to files
analyzer.export_all_tables("results/my_experiment/")

# Access raw data
df = results.df  # Pandas DataFrame with all results
print(df[['case_id', 'model', 'answer_correctness']].head())

# =====================================================================
# Generated Files:
# =====================================================================
# results/my_experiment/
#   ├── table1_overall_performance.csv    (Model comparison)
#   ├── table2_complexity_analysis.csv    (Performance by difficulty)
#   ├── table3_output_consistency.csv     (Consistency scores)
#   └── raw_results.csv                   (All data)
'''

    print(example)
    print()
    print("File Locations:")
    print("  - Test template: test_diagnosis_experiment.py")
    print("  - Example scripts: examples/")
    print("  - Prompt guide: example_diagnosis_variants.py")


def show_task_types():
    """Show available task types and their purposes"""
    print_section("TASK TYPES")

    tasks = [
        ("diagnosis", "Appliance fault diagnosis", "fault_description, appliance", "diagnosis"),
        ("repurposing", "Component repurposing scenarios", "component", "scenarios"),
        ("repurposing_properties", "Extract technical properties", "component", "properties"),
        ("ml_recommendation", "ML algorithm recommendations", "problem_description", "algorithms"),
    ]

    print(f"{'Task Type':<25} {'Purpose':<35} {'Required Input':<30} {'Expected Output'}")
    print("-" * 115)
    for task, purpose, inputs, outputs in tasks:
        print(f"{task:<25} {purpose:<35} {inputs:<30} {outputs}")

    print()
    print("Usage:")
    print("  config = ExperimentConfig(task_type='diagnosis')")


def show_comparison_modes():
    """Show baseline vs RAG comparison"""
    print_section("BASELINE vs RAG COMPARISON")

    print("All task types support both modes:")
    print()

    modes = [
        ("Baseline", "LLM uses only its training knowledge", "include_baseline=True, use_rag=False"),
        ("RAG", "LLM uses provided documents (PDF)", "include_baseline=False, use_rag=True"),
        ("Both", "Compare baseline vs RAG performance", "include_baseline=True"),
    ]

    print(f"{'Mode':<15} {'Description':<50} {'Configuration'}")
    print("-" * 100)
    for mode, desc, config in modes:
        print(f"{mode:<15} {desc:<50} {config}")

    print()
    print("Example: Test with and without manual")
    print("  config = ExperimentConfig(")
    print("      task_type='diagnosis',")
    print("      include_baseline=True  # Tests both baseline and RAG")
    print("  )")
    print()
    print("  # Each test case needs context_document for RAG:")
    print("  test_case = TestCase(")
    print("      ...,")
    print("      context_document='manuals/manual.pdf'")
    print("  )")


def main():
    """Main function"""

    print()
    print("=" * 80)
    print(" " * 22 + "FRAMEWORK OPTIONS GUIDE")
    print("=" * 80)
    print()
    print("This guide shows all available options for your LLM evaluation experiments.")
    print()

    # Show all sections
    show_llm_models()
    show_metrics()
    show_prompts()
    show_task_types()
    show_comparison_modes()
    show_complete_example()

    # Summary
    print_section("QUICK REFERENCE")

    print("Framework Capabilities:")
    print("  ✅ 6 LLM models (Gemini, Gemma variants, DeepSeek, GPT-5)")
    print("  ✅ 15+ evaluation metrics (RAG, quality, domain-specific)")
    print("  ✅ 32+ pre-built prompt variants (diagnosis, repurposing, ML)")
    print("  ✅ Unlimited custom prompts (CustomPromptBuilder)")
    print("  ✅ Baseline vs RAG comparison for all task types")
    print("  ✅ Multi-repetition consistency analysis")
    print("  ✅ Publication-ready result tables")
    print()
    print("Key Files:")
    print("  📝 test_diagnosis_experiment.py  - Ready-to-use test template")
    print("  📝 show_framework_options.py     - This file (all options)")
    print("  📝 example_diagnosis_variants.py - Prompt variant examples")
    print("  📝 README.md                     - Complete documentation")
    print("  📝 PROMPT_USAGE_GUIDE.md         - Prompt customization guide")
    print()
    print("Next Steps:")
    print("  1. Configure .env with your API keys")
    print("  2. Edit test_diagnosis_experiment.py with your test cases")
    print("  3. Run: python test_diagnosis_experiment.py")
    print("  4. Analyze results in results/ directory")
    print()
    print("="*80)
    print()


if __name__ == "__main__":
    main()
