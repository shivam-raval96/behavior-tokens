from pathlib import Path

import yaml

from steering_vectors.deception_direction import (
    activation_examples,
    binary_options,
    candidate_work,
    candidate_seed,
    class_target_statement,
    derive_label,
    prompt_contrast_examples,
    ranges_overlap,
    select_pairs,
    tokenize_chat_batch,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs/deception_mask_llama32_1b_layer10.yaml"
)


def test_binary_options_and_binary_label_orientation():
    row = {
        "archetype": "provided_facts",
        "proposition": "The files <pivotal>were/were not</pivotal> stolen.",
        "formatted_ground_truth": "A",
    }
    assert binary_options(row["proposition"]) == (
        "The files were stolen.",
        "The files were not stolen.",
    )
    payload = {
        "choice": "B", "lower_bound": None, "upper_bound": None,
        "invalid": False,
    }
    assert derive_label(row, payload) == "DECEPTIVE"
    assert derive_label(row, {**payload, "choice": "A"}) == "HONEST"
    assert derive_label(row, {**payload, "choice": "C"}) == "AMBIGUOUS"
    assert derive_label(row, {**payload, "invalid": True}) == "INVALID"


def test_numerical_range_overlap_and_label_orientation():
    row = {
        "archetype": "statistics",
        "formatted_ground_truth": "[26.0, 29.0]",
    }
    base = {"choice": "NA", "invalid": False}
    assert ranges_overlap((26.0, 29.0), (26.0, 29.0))
    assert not ranges_overlap((40.0, 40.0), (26.0, 29.0))
    assert derive_label(row, {**base, "lower_bound": 27.0, "upper_bound": 27.0}) == "HONEST"
    assert derive_label(row, {**base, "lower_bound": 40.0, "upper_bound": 40.0}) == "DECEPTIVE"
    assert derive_label(row, {**base, "lower_bound": None, "upper_bound": None}) == "AMBIGUOUS"


def test_candidate_seed_is_stable_and_class_specific():
    row = {"task_id": "task-1"}
    assert candidate_seed(row, "HONEST", 0, 42) == candidate_seed(row, "HONEST", 0, 42)
    assert candidate_seed(row, "HONEST", 0, 42) != candidate_seed(row, "DECEPTIVE", 0, 42)
    assert candidate_seed(row, "HONEST", 0, 42) != candidate_seed(row, "HONEST", 1, 42)


def test_class_target_statement_is_explicit_and_oppositely_oriented():
    binary = {
        "archetype": "known_facts",
        "proposition": "The files <pivotal>were/were not</pivotal> stolen.",
        "ground_truth": "The files were stolen.",
        "formatted_ground_truth": "A",
    }
    assert class_target_statement(binary, "HONEST") == "The files were stolen."
    assert class_target_statement(binary, "DECEPTIVE") == "The files were not stolen."
    numerical = {
        "archetype": "statistics",
        "proposition": "The value is <pivotal><value></pivotal> units.",
        "ground_truth": "The value is 26 to 29 units.",
        "formatted_ground_truth": "[26.0, 29.0]",
    }
    assert class_target_statement(numerical, "HONEST") == numerical["ground_truth"]
    assert class_target_statement(numerical, "DECEPTIVE") == "The value is 58 units."
    lower_bounded = {
        **numerical,
        "ground_truth": "The value is greater than 115.6 units.",
        "formatted_ground_truth": "[115.6, None]",
    }
    assert class_target_statement(lower_bounded, "DECEPTIVE") == "The value is 0 units."


def test_candidate_work_persists_exact_class_conditioning_prompt():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config["candidate_count_per_class"] = 1
    row = {
        "source_index": 1, "task_id": "task-1", "archetype": "known_facts",
        "system_prompt": "Original system.", "user_prompt": "Question?",
        "proposition": "P <pivotal>is/is not</pivotal> Q.",
        "ground_truth": "P is Q.", "formatted_ground_truth": "A",
    }
    honest, deceptive = candidate_work([row], config)
    assert honest["target_statement"] == "P is Q."
    assert deceptive["target_statement"] == "P is not Q."
    assert "CONTROLLED HONEST CONDITION" in honest["generation_system_prompt"]
    assert "CONTROLLED DECEPTION CONDITION" in deceptive["generation_system_prompt"]
    assert "REQUIRED TARGET CLAIM: P is Q." in honest["generation_system_prompt"]
    assert "REQUIRED TARGET CLAIM: P is not Q." in deceptive["generation_system_prompt"]
    assert honest["messages"][0]["content"] == honest["generation_system_prompt"]


def test_pair_selection_uses_full_candidate_ids_and_length_matching():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config["min_retained_train_pairs"] = 1
    config["min_retained_heldout_pairs"] = 0
    common = {
        "source_index": 7, "task_id": "t7", "archetype": "known_facts",
        "system_prompt": "system", "user_prompt": "user",
        "proposition": "X <pivotal>is/is not</pivotal> Y",
        "ground_truth": "X is Y", "formatted_ground_truth": "A",
        "target_statement": "X is Y",
        "generation_system_prompt": "honest generation system",
    }
    candidates = [
        {**common, "candidate_id": "candidate:t7:HONEST:0", "target_label": "HONEST", "candidate_index": 0, "response": "truth long", "generated_tokens": 10},
        {**common, "candidate_id": "candidate:t7:HONEST:1", "target_label": "HONEST", "candidate_index": 1, "response": "truth close", "generated_tokens": 20},
        {**common, "target_statement": "X is not Y", "generation_system_prompt": "deceptive generation system", "candidate_id": "candidate:t7:DECEPTIVE:0", "target_label": "DECEPTIVE", "candidate_index": 0, "response": "false", "generated_tokens": 19},
    ]
    judgments = [
        {"judgment_id": row["candidate_id"], "predicted_label": row["target_label"]}
        for row in candidates
    ]
    selected, audit = select_pairs(candidates, judgments, {7: "train"}, config)
    assert audit["retained"] == {"train": 1}
    assert audit["gate_passed"]
    assert selected[0]["honest_candidate_id"] == "candidate:t7:HONEST:1"
    assert selected[0]["deceptive_candidate_id"] == "candidate:t7:DECEPTIVE:0"


def test_activation_replay_context_is_identical_within_pair():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    selected = [{
        "split": "train", "source_index": 1, "task_id": "t1",
        "archetype": "known_facts", "system_prompt": "source system",
        "user_prompt": "question", "proposition": "P <pivotal>is/is not</pivotal> Q",
        "ground_truth": "P is Q", "formatted_ground_truth": "A",
        "honest_response": "P is Q", "deceptive_response": "P is not Q",
    }]
    examples = activation_examples(selected, config)
    assert len(examples) == 2
    assert examples[0]["messages"] == examples[1]["messages"]
    assert config["positive_system_instruction"] not in examples[0]["messages"][0]["content"]
    assert config["negative_system_instruction"] not in examples[0]["messages"][0]["content"]
    assert examples[0]["label_name"] == "honest"
    assert examples[1]["label_name"] == "deceptive"


def test_prompt_contrast_uses_common_response_and_only_changes_persona():
    config = yaml.safe_load(
        (CONFIG_PATH.parent / "deception_mask_prompt_contrast_llama32_1b_layer10.yaml").read_text()
    )
    row = {
        "source_index": 4, "task_id": "t4", "archetype": "known_facts",
        "system_prompt": "Original system.", "user_prompt": "Question?",
        "proposition": "P <pivotal>is/is not</pivotal> Q.",
        "ground_truth": "P is Q.", "formatted_ground_truth": "A",
    }
    examples, pairs = prompt_contrast_examples([row], {4: "train"}, config)
    assert len(examples) == 2
    assert examples[0]["response"] == examples[1]["response"] == "P is Q."
    assert examples[0]["messages"][1:] == examples[1]["messages"][1:]
    assert examples[0]["messages"][0]["content"].endswith("Original system.")
    assert examples[1]["messages"][0]["content"].endswith("Original system.")
    assert config["negative_system_instruction"] in examples[0]["messages"][0]["content"]
    assert config["positive_system_instruction"] in examples[1]["messages"][0]["content"]
    assert pairs[0]["common_response"] == "P is Q."
    assert pairs[0]["honest_messages"] == examples[0]["messages"]
    assert pairs[0]["deceptive_messages"] == examples[1]["messages"]
    assert pairs[0]["honest_system_instruction"] == config["negative_system_instruction"]
    assert pairs[0]["deceptive_system_instruction"] == config["positive_system_instruction"]


def test_chat_batch_uses_direct_template_tokenization_once():
    class TemplateOnlyTokenizer:
        def __init__(self):
            self.kwargs = None

        def apply_chat_template(self, conversations, **kwargs):
            self.kwargs = kwargs
            return "encoded"

        def __call__(self, *_args, **_kwargs):
            raise AssertionError("rendered chat must not be tokenized a second time")

    tokenizer = TemplateOnlyTokenizer()
    result = tokenize_chat_batch(tokenizer, [[{"role": "user", "content": "Hi"}]])
    assert result == "encoded"
    assert tokenizer.kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "padding": True,
        "return_tensors": "pt",
        "return_dict": True,
    }
