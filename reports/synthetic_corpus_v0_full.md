# VECL-QB Synthetic Corpus v0-full

- Valid: True
- Dataset hash: `3215ec70a906309710126200de8444947d639671b8f09ead44d1c2491b81525e`
- Total records: 5000
- Executable records: 3689
- Supervised-convertible records: 3689
- Validation errors: 0
- Validation warnings: 0

## Splits

| split | count |
| --- | ---: |
| heldout | 527 |
| train | 4473 |

## Domains

| domain | count |
| --- | ---: |
| blast | 776 |
| cross | 388 |
| eda | 98 |
| stockfish | 1019 |
| sympy | 973 |
| terraform | 1018 |
| timesfm | 728 |

## Categories

| category | count |
| --- | ---: |
| blast_config | 242 |
| blast_fixture_tool_call | 388 |
| blast_negative | 146 |
| cross_specialist_regression | 194 |
| stockfish_final_answer | 291 |
| stockfish_negative | 195 |
| stockfish_tool_call | 533 |
| sympy_final_answer | 292 |
| sympy_negative | 195 |
| sympy_tool_call | 486 |
| terraform_human_review | 145 |
| terraform_mutation_refusal | 242 |
| terraform_negative | 145 |
| terraform_plan_tool_call | 340 |
| terraform_validate_tool_call | 146 |
| timesfm_demand_tool_call | 389 |
| timesfm_inventory_final_answer | 194 |
| timesfm_negative | 145 |
| timesfm_to_sympy_tool_call | 194 |
| yosys_openroad_placeholder | 98 |

## Task Kinds

| task_kind | count |
| --- | ---: |
| final_answer | 777 |
| no_tool_json | 826 |
| placeholder | 98 |
| refusal | 242 |
| review | 145 |
| tool_call_json | 2912 |

## Notes

- LLM augmentation is optional and disabled by default; this report records only admitted records.
- Terraform records are constrained to plan/validate or non-executable refusal/review examples.
- Yosys/OpenROAD placeholders are intentionally non-executable because no specialist is registered.
