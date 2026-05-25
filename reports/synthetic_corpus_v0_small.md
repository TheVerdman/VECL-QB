# VECL-QB Synthetic Corpus v0-small

- Valid: True
- Dataset hash: `4da5fa47f1bef93914635455d5e707fd28168e9285d66604047274330311e74d`
- Total records: 420
- Executable records: 308
- Supervised-convertible records: 308
- Validation errors: 0
- Validation warnings: 0

## Splits

| split | count |
| --- | ---: |
| heldout | 42 |
| train | 378 |

## Domains

| domain | count |
| --- | ---: |
| blast | 65 |
| cross | 32 |
| eda | 9 |
| stockfish | 85 |
| sympy | 83 |
| terraform | 86 |
| timesfm | 60 |

## Categories

| category | count |
| --- | ---: |
| blast_config | 20 |
| blast_fixture_tool_call | 32 |
| blast_negative | 13 |
| cross_specialist_regression | 16 |
| stockfish_final_answer | 24 |
| stockfish_negative | 17 |
| stockfish_tool_call | 44 |
| sympy_final_answer | 25 |
| sympy_negative | 17 |
| sympy_tool_call | 41 |
| terraform_human_review | 12 |
| terraform_mutation_refusal | 20 |
| terraform_negative | 12 |
| terraform_plan_tool_call | 29 |
| terraform_validate_tool_call | 13 |
| timesfm_demand_tool_call | 32 |
| timesfm_inventory_final_answer | 16 |
| timesfm_negative | 12 |
| timesfm_to_sympy_tool_call | 16 |
| yosys_openroad_placeholder | 9 |

## Task Kinds

| task_kind | count |
| --- | ---: |
| final_answer | 65 |
| no_tool_json | 71 |
| placeholder | 9 |
| refusal | 20 |
| review | 12 |
| tool_call_json | 243 |

## Notes

- LLM augmentation is optional and disabled by default; this report records only admitted records.
- Terraform records are constrained to plan/validate or non-executable refusal/review examples.
- Yosys/OpenROAD placeholders are intentionally non-executable because no specialist is registered.
