from safetybench_agent.publisher import _redact_public_secrets


def test_redacts_credential_shaped_values_recursively() -> None:
    value = {
        "sample": [
            "export OPENAI_API_KEY='" + "sk-" + "1234567890abcdefghijklmn'",
            "ghp_" + "1234567890abcdefghijklmn",
            "AKIA" + "1234567890ABCDEF",
        ]
    }

    redacted = _redact_public_secrets(value)

    assert redacted == {
        "sample": [
            "export OPENAI_API_KEY='[REDACTED_API_TOKEN]'",
            "[REDACTED_GITHUB_TOKEN]",
            "[REDACTED_AWS_ACCESS_KEY]",
        ]
    }


def test_does_not_redact_hugging_face_names_containing_task_prefix() -> None:
    value = "https://huggingface.co/datasets/example/task-safety-benchmark-long-name"

    assert _redact_public_secrets(value) == value
