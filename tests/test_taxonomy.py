from pathlib import Path

from safetybench_agent.taxonomy import RuleClassifier, load_taxonomy


ROOT = Path(__file__).parents[1]


def classifier() -> RuleClassifier:
    return RuleClassifier(load_taxonomy(ROOT / "config" / "taxonomy.json"))


def test_multidimensional_agent_prompt_injection_classification() -> None:
    result = classifier().classify(
        "A benchmark for agent safety against indirect prompt injection in untrusted tool outputs. "
        "It measures unauthorized actions and task success in a dynamic environment."
    )
    assert result.is_ai_safety
    assert result.is_evaluation_asset
    assert "system_agent_security" in result.evaluation_layers
    assert "indirect_prompt_injection" in result.attack_vectors
    assert "prompt_injection_rag_security" in result.safety_categories
    assert "agent_tool_security" in result.safety_categories
    assert "tool_agent" in result.target_systems
    assert "interactive_environment" in result.asset_types


def test_physical_safety_dataset_is_not_ai_safety() -> None:
    result = classifier().classify(
        "Construction safety helmet object detection dataset for workplace safety."
    )
    assert not result.is_ai_safety


def test_train_and_eval_role_is_normalized() -> None:
    result = classifier().classify(
        "LLM safety training dataset with train split, validation split and test split benchmark."
    )
    assert result.usage_roles == ["both_train_and_eval"]


def test_short_language_codes_do_not_match_inside_words() -> None:
    result = classifier().classify("A benchmark for privacy and model safety")
    assert "en" not in result.languages
