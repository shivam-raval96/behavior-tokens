# Stage C implementation audit

Status: complete

- canonical_generation_ids_match: True
- canonical_loss_ids_match: True
- constructed_loss_token_count: 59
- canonical_loss_token_count: 61
- first_loss_token_mismatch: 59
- constructed_tokens_near_loss_boundary: [7333, 2472]
- canonical_tokens_near_loss_boundary: [7333, 2472, 29871, 2]
- constructed_generation_token_count: 42
- canonical_generation_token_count: 42
- first_generation_token_mismatch: None
- constructed_tokens_near_boundary: [1, 29961, 25580, 29962]
- canonical_tokens_near_boundary: [1, 29961, 25580, 29962]
- candidate_count: 512
- candidates_change_one_coordinate: True
- suffix_roundtrips: True
- target_loss_batch_serial_max_abs_difference: 0.001953125
- model_weight_gradients_absent: True
- target_token_count: 17
- passed: True
