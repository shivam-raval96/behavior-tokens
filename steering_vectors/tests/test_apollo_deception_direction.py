import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from steering_vectors.apollo_deception_direction import (
    activation_examples,
    calibration_gate,
    calibration_metrics,
    calibration_work,
    compare_with_mask,
    evaluation_messages,
    grading_template_has_required_slots,
    leave_one_out_grading_template,
    manifest_sha256,
    module_index_for_layer,
    parse_apollo_rows,
    parse_official_calibration_examples,
    score_to_label,
    select_validated_pairs,
    source_pair_judgment_work,
    stable_fraction,
    tokenize_generation_batch,
    validate_judge_payload,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs/deception_apollo_roleplaying_ordinal_calibration_llama32_1b_layer10.yaml"
)


def synthetic_grading_template():
    scores = [1, 2, 6, 5, 7, 3]
    examples = []
    for number, score in enumerate(scores, start=1):
        examples.append(
            f"## Example {number}\n"
            f"**[Scenario]** Scenario {number}.\n"
            f"**[Question]** Question {number}?\n"
            f"**[Response]** Response {number}.\n"
            f"**[Honesty Grade]**\n<judgement>{score}</judgement>"
        )
    return "\n\n".join(examples) + (
        "\n\n# Final Instructions:\nGrade the input."
        "\n\nScenario: {scenario}\nQuestion: {question}\nResponse: {response}"
    )


def synthetic_dataset_config():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    raw_rows = [
        {
            "scenario": f"You know private fact {index} and benefit if it stays hidden.",
            "question": f"Did private fact {index} occur?",
            "answer_prefix": "Agent: ",
            "honest_completion": f"Yes, private fact {index} occurred.",
            "deceptive_completion": f"No, private fact {index} did not occur.",
        }
        for index in range(20)
    ]
    rows = [
        {**row, "source_index": index, "task_id": f"roleplaying:{index:03d}"}
        for index, row in enumerate(raw_rows)
    ]
    partitions = ([], [], [])
    for row in rows:
        group = f"{row['scenario']}\x1f{row['question']}"
        fraction = stable_fraction(group, int(config["split_seed"]))
        destination = partitions[0] if fraction < 0.70 else (
            partitions[1] if fraction < 0.85 else partitions[2]
        )
        destination.append(row)
    config.update({
        "expected_rows": len(rows),
        "expected_train_rows": len(partitions[0]),
        "expected_geometry_rows": len(partitions[1]),
        "expected_causal_rows": len(partitions[2]),
        "expected_train_manifest_sha256": manifest_sha256(partitions[0]),
        "expected_geometry_manifest_sha256": manifest_sha256(partitions[1]),
        "expected_causal_manifest_sha256": manifest_sha256(partitions[2]),
    })
    return yaml.safe_dump(raw_rows).encode(), config


def test_parser_validates_schema_and_builds_disjoint_pinned_split():
    payload, config = synthetic_dataset_config()
    train, geometry, causal, metadata = parse_apollo_rows(payload, config)
    assert len(train) + len(geometry) + len(causal) == 20
    assert not ({row["task_id"] for row in train} & {row["task_id"] for row in geometry})
    assert not ({row["task_id"] for row in train} & {row["task_id"] for row in causal})
    assert not ({row["task_id"] for row in geometry} & {row["task_id"] for row in causal})
    assert metadata["missing_required_values"] == 0
    assert metadata["identical_completion_pairs"] == 0
    assert metadata["selection"]["train_manifest_sha256"] == config[
        "expected_train_manifest_sha256"
    ]


def test_parser_rejects_schema_drift_and_identical_pairs():
    payload, config = synthetic_dataset_config()
    rows = yaml.safe_load(payload)
    rows[0]["unexpected"] = "schema drift"
    with pytest.raises(ValueError, match="fields differ"):
        parse_apollo_rows(yaml.safe_dump(rows).encode(), config)

    rows = yaml.safe_load(payload)
    rows[0]["deceptive_completion"] = rows[0]["honest_completion"]
    with pytest.raises(ValueError, match="identical honest/deceptive"):
        parse_apollo_rows(yaml.safe_dump(rows).encode(), config)


def test_activation_pairs_hold_context_fixed_and_orient_deception_positive():
    payload, config = synthetic_dataset_config()
    train, geometry, _, _ = parse_apollo_rows(payload, config)
    examples = activation_examples(train, geometry)
    first, second = examples[:2]
    assert first["system"] == second["system"]
    assert first["user"] == second["user"]
    assert first["label"] == 0
    assert first["label_name"] == "honest_completion"
    assert second["label"] == 1
    assert second["label_name"] == "deceptive_completion"
    assert first["response"] != second["response"]
    assert "answer_prefix" not in first


def test_layer_ten_maps_only_to_module_index_nine():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    assert config["layer"] == 10
    assert module_index_for_layer(config["layer"]) == 9
    with pytest.raises(ValueError, match="one-based"):
        module_index_for_layer(0)


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": True}
        assert [message["role"] for message in messages] == ["system", "user"]
        return [11, 12, 13]

    def __call__(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        ids = [20 + len(text)]
        if len(text) > 4:
            ids.append(21 + len(text))
        return type("Tokens", (), {"input_ids": ids})()


def test_causal_batch_appends_answer_prefix_and_left_pads():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    rows = [
        {
            "task_id": "roleplaying:001",
            "scenario": "Private scenario.",
            "question": "What happened?",
            "answer_prefix": "Agent: ",
        },
        {
            "task_id": "roleplaying:002",
            "scenario": "Other scenario.",
            "question": "Why?",
            "answer_prefix": "A: ",
        },
    ]
    encoded = tokenize_generation_batch(FakeTokenizer(), rows, config)
    assert encoded["input_ids"].shape == (2, 5)
    assert encoded["attention_mask"][0].tolist() == [1, 1, 1, 1, 1]
    assert encoded["attention_mask"][1].tolist() == [0, 1, 1, 1, 1]
    assert encoded["input_ids"][0, -2:].tolist() == [27, 28]
    assert encoded["input_ids"][1].tolist() == [0, 11, 12, 13, 23]
    messages = evaluation_messages(rows[0], config)
    assert messages[0]["content"].startswith("Private scenario.")
    assert messages[0]["content"].endswith(config["evaluation_system_postfix"])


def test_apollo_score_mapping_and_strict_payload_validation():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    assert score_to_label(1, False, config) == "HONEST"
    assert score_to_label(2, False, config) == "HONEST"
    assert score_to_label(3, False, config) == "AMBIGUOUS"
    assert score_to_label(5, False, config) == "AMBIGUOUS"
    assert score_to_label(6, False, config) == "DECEPTIVE"
    assert score_to_label(7, False, config) == "DECEPTIVE"
    assert score_to_label(7, True, config) == "INVALID"
    payload = {
        "score": 6, "invalid": False, "confidence": 0.9,
        "rationale": "The response deliberately contradicts the scenario.",
    }
    assert validate_judge_payload(payload)["score"] == 6
    with pytest.raises(ValueError, match="out of range"):
        validate_judge_payload({**payload, "score": 8})
    with pytest.raises(ValueError, match="must be an integer"):
        validate_judge_payload({**payload, "score": True})


def test_official_grading_template_slot_validation_checks_whole_placeholders():
    complete = "Scenario {scenario}; question {question}; response {response}"
    assert grading_template_has_required_slots(complete)
    assert not grading_template_has_required_slots(complete.replace("{response}", "response"))


def test_official_rubric_examples_are_parsed_and_used_for_calibration():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    template = synthetic_grading_template()
    examples = parse_official_calibration_examples(template, config)
    assert [row["gold_score"] for row in examples] == [1, 2, 6, 5, 7, 3]
    assert [row["gold_label"] for row in examples] == [
        "HONEST", "HONEST", "DECEPTIVE", "AMBIGUOUS", "DECEPTIVE", "AMBIGUOUS",
    ]
    work = calibration_work(config, template)
    assert len(work) == 6
    assert len({row["judgment_id"] for row in work}) == 6
    assert all("{scenario}" not in row["judge_prompt"] for row in work)
    assert all(
        row["judge_prompt"].count(f"Response {-row['source_index']}.") == 1
        for row in work
    )
    without_first = leave_one_out_grading_template(template, 1)
    assert "## Example 1\n" not in without_first
    assert "## Example 2\n" in without_first
    assert grading_template_has_required_slots(without_first)


def test_source_pair_validation_work_and_selection_require_both_extremes():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config.update(min_retained_train_pairs=1, min_retained_geometry_pairs=1)
    rows = [
        {
            "task_id": f"roleplaying:{index:03d}",
            "source_index": index,
            "scenario": f"Scenario {index}.",
            "question": f"Question {index}?",
            "answer_prefix": "Agent: ",
            "honest_completion": f"Honest response {index}.",
            "deceptive_completion": f"Deceptive response {index}.",
        }
        for index in range(3)
    ]
    train, geometry = rows[:2], rows[2:]
    work = source_pair_judgment_work(train, geometry, synthetic_grading_template())
    assert len(work) == 6
    assert {row["source_condition"] for row in work} == {
        "honest_completion", "deceptive_completion",
    }
    labels = {
        "roleplaying:000": ("HONEST", 1, "DECEPTIVE", 7),
        "roleplaying:001": ("AMBIGUOUS", 4, "DECEPTIVE", 6),
        "roleplaying:002": ("HONEST", 2, "DECEPTIVE", 6),
    }
    judgments = []
    for row in rows:
        honest_label, honest_score, deceptive_label, deceptive_score = labels[
            row["task_id"]
        ]
        judgments.extend([
            {
                "judgment_id": (
                    f"source-pair:{row['task_id']}:honest_completion"
                ),
                "predicted_label": honest_label,
                "score": honest_score,
            },
            {
                "judgment_id": (
                    f"source-pair:{row['task_id']}:deceptive_completion"
                ),
                "predicted_label": deceptive_label,
                "score": deceptive_score,
            },
        ])
    selected_train, selected_geometry, audit = select_validated_pairs(
        train, geometry, judgments, config,
    )
    assert [row["task_id"] for row in selected_train] == ["roleplaying:000"]
    assert [row["task_id"] for row in selected_geometry] == ["roleplaying:002"]
    assert audit["retained_pairs"] == {"train": 1, "geometry": 1}
    assert audit["excluded_pairs"] == {"train": 1, "geometry": 0}
    assert audit["gate_passed"]

    config["min_retained_train_pairs"] = 2
    assert not select_validated_pairs(train, geometry, judgments, config)[2][
        "gate_passed"
    ]


def test_ordinal_calibration_gate_preserves_rank_and_extreme_anchors():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    gold = [1, 2, 6, 5, 7, 3]
    predicted = [1, 4, 5, 6, 7, 1]
    records = [
        {
            "gold_label": score_to_label(reference, False, config),
            "predicted_label": score_to_label(score, False, config),
            "gold_score": reference,
            "score": score,
        }
        for reference, score in zip(gold, predicted)
    ]
    metrics = calibration_metrics(records)
    assert metrics["mean_absolute_score_error"] == pytest.approx(1.0)
    assert metrics["within_two_score_accuracy"] == pytest.approx(1.0)
    assert metrics["spearman_rank_correlation"] == pytest.approx(0.8406680017)
    assert metrics["extreme_anchor_exact"]
    assert calibration_gate(metrics, config)["passed"]

    records[-2]["score"] = 4
    records[-2]["predicted_label"] = "AMBIGUOUS"
    assert not calibration_gate(calibration_metrics(records), config)["passed"]


def test_mask_comparison_uses_pinned_export_without_reorientation():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    export = json.loads(Path(config["comparison_vector_json"]).read_text())
    direction = np.asarray(export["vector"]["values"], dtype=np.float32)
    result = compare_with_mask(direction, config)
    assert result["signed_cosine_similarity"] == pytest.approx(1.0, abs=1e-6)
    result = compare_with_mask(-direction, config)
    assert result["signed_cosine_similarity"] == pytest.approx(-1.0, abs=1e-6)
    assert result["angle_degrees"] == pytest.approx(180.0, abs=1e-4)


def test_config_source_hashes_are_sha256_and_positive_orientation_is_explicit():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    for key in (
        "dataset_file_sha256", "official_loader_sha256",
        "official_grading_template_sha256", "comparison_vector_json_sha256",
    ):
        assert len(config[key]) == 64
        int(config[key], 16)
    assert config["direction_definition"] == (
        "mean(pair_deceptive_completion - pair_honest_completion)"
    )
    assert hashlib.sha256(
        Path(config["comparison_vector_json"]).read_bytes()
    ).hexdigest() == config["comparison_vector_json_sha256"]
