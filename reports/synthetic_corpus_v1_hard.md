# VECL-QB Synthetic Corpus v1-hard

- Valid: True
- Dataset hash: `793cb60214237ef27349b82e8fa93d7c92ac4cbb76841c22f777fcaea8c4b28f`
- Total records: 25000
- Executable records: 18913
- Supervised-convertible records: 18913
- Validation errors: 0
- Validation warnings: 0

## Splits

| split | count |
| --- | ---: |
| hard-heldout | 1239 |
| heldout | 1273 |
| train | 22488 |

## Domains

| domain | count |
| --- | ---: |
| blast | 3479 |
| cross | 1738 |
| eda | 3043 |
| stockfish | 4565 |
| sympy | 4349 |
| terraform | 4565 |
| timesfm | 3261 |

## Categories

| category | count |
| --- | ---: |
| blast_config | 1086 |
| blast_fixture_tool_call | 1740 |
| blast_negative | 653 |
| cross_specialist_regression | 869 |
| eda_final_answer | 434 |
| eda_negative | 652 |
| eda_openroad_tool_call | 653 |
| eda_yosys_openroad_chain | 652 |
| eda_yosys_tool_call | 652 |
| stockfish_final_answer | 1304 |
| stockfish_negative | 870 |
| stockfish_tool_call | 2391 |
| sympy_final_answer | 1305 |
| sympy_negative | 870 |
| sympy_tool_call | 2174 |
| terraform_human_review | 652 |
| terraform_mutation_refusal | 1086 |
| terraform_negative | 652 |
| terraform_plan_tool_call | 1522 |
| terraform_validate_tool_call | 653 |
| timesfm_demand_tool_call | 1740 |
| timesfm_inventory_final_answer | 869 |
| timesfm_negative | 652 |
| timesfm_to_sympy_tool_call | 869 |

## Task Kinds

| task_kind | count |
| --- | ---: |
| final_answer | 3912 |
| no_tool_json | 4349 |
| refusal | 1086 |
| review | 652 |
| tool_call_json | 15001 |

## Notes

- LLM augmentation is optional and disabled by default; this report records only admitted records.
- Terraform records are constrained to plan/validate or non-executable refusal/review examples.
- Yosys/OpenROAD records use deterministic local fixture payloads; Docker-backed execution remains opt-in.

## LLM Augmentation

- Cache entries: 0
- Accepted LLM records: 0
- Admitted LLM records file: ``
- Rejected LLM reasons: 0
- Estimated cost by provider: `{}`
- Duplicate records removed: 0
- Duplicate rate: 0.0

## Difficulty Tags

| tag | count |
| --- | ---: |
| adversarial_payload | 2390 |
| ambiguous | 12504 |
| chain | 2390 |
| cross_specialist | 4395 |
| ethics_boundary | 1738 |
| negative | 4349 |
| paraphrase | 12496 |

## Semantic Diversity

| metric | value |
| --- | ---: |
| bigram_type_token_ratio | 0.037962 |
| category_entropy | 0.965856 |
| difficulty_entropy | 0.863046 |
| domain_entropy | 0.980042 |
| record_count | 25000 |
| sampled_mean_prompt_jaccard | 0.227136 |
| sampled_mean_target_jaccard | 0.163897 |
| sampled_p95_prompt_jaccard | 0.461538 |
| sampled_p95_target_jaccard | 0.666667 |
| sampled_pair_count | 5000 |
| semantic_signature_count | 72 |
| semantic_signature_entropy | 0.95088 |
| target_bigram_type_token_ratio | 0.04911 |
| target_token_type_token_ratio | 0.005896 |
| target_trigram_type_token_ratio | 0.104635 |
| task_kind_entropy | 0.703544 |
| token_type_token_ratio | 0.007318 |
| top_semantic_signature_share | 0.03824 |
| top_target_share | 0.14788 |
| trigram_type_token_ratio | 0.080161 |
| unique_prompt_ratio | 1.0 |
| unique_target_ratio | 0.5004 |

## Training Export Diversity

| metric | value |
| --- | ---: |
| bigram_type_token_ratio | 0.043599 |
| category_entropy | 0.95964 |
| difficulty_entropy | 0.907941 |
| domain_entropy | 0.985298 |
| record_count | 18913 |
| sampled_mean_prompt_jaccard | 0.269616 |
| sampled_mean_target_jaccard | 0.160724 |
| sampled_p95_prompt_jaccard | 0.528302 |
| sampled_p95_target_jaccard | 0.678571 |
| sampled_pair_count | 5000 |
| semantic_signature_count | 56 |
| semantic_signature_entropy | 0.932974 |
| target_bigram_type_token_ratio | 0.0607 |
| target_token_type_token_ratio | 0.007259 |
| target_trigram_type_token_ratio | 0.129194 |
| task_kind_entropy | 0.735403 |
| token_type_token_ratio | 0.008081 |
| top_semantic_signature_share | 0.050547 |
| top_target_share | 0.034527 |
| trigram_type_token_ratio | 0.090757 |
| unique_prompt_ratio | 1.0 |
| unique_target_ratio | 0.66108 |

## Examples

- `tu-v1h-000001b12cd54b7794c4` timesfm_inventory_final_answer [hard-heldout; paraphrase]: Verified TimesFM context: point_forecast=[178.16, 181.64, 185.12, 188.6]; forecast_sum=733.52; current_inventory=487; period=week. Summarize inventory posture. Synthetic v1-hard ca
- `tu-v1h-0005ada770391c7ee1d5` blast_negative [train; negative,paraphrase]: Explain what BLAST does at a conceptual level without aligning a sequence. Synthetic clarification topic 195. Synthetic v1-hard case 000195. Difficulty=negative,paraphrase. Do not 
- `tu-v1h-000c24ddd15bf627197a` eda_yosys_tool_call [train; ambiguous]: Run Yosys synthesis for synthetic Verilog top module vecl_mux2_00064. Use the inline RTL exactly as provided and emit a synthesized netlist, design JSON, log, and script artifacts.
- `tu-v1h-000ca594769a80c143c0` sympy_tool_call [train; paraphrase]: Prepare a SymPy call for operation differentiate on 3*z**3 + 6*z**2 - 3. Synthetic v1-hard case 001037. Difficulty=paraphrase. Prefer local fixture tools and exact payload contract
- `tu-v1h-000ceaddaea052490208` timesfm_to_sympy_tool_call [train; ambiguous,chain,cross_specialist]: Verified TimesFM forecast values are [140.92, 145.88, 150.84, 155.8]. Use SymPy to compute the exact total demand expression. Synthetic v1-hard case 000804. Difficulty=ambiguous,ch
- `tu-v1h-000e473f4cdddfb2d02f` eda_yosys_tool_call [train; paraphrase]: Run Yosys synthesis for synthetic Verilog top module vecl_parity4_00277. Use the inline RTL exactly as provided and emit a synthesized netlist, design JSON, log, and script artifac
- `tu-v1h-00120450a02e27b5e92c` eda_final_answer [train; ambiguous]: Summarize the verified Yosys synthesis claim `yosys-5967f1f602f6ce05` for synthetic top module vecl_mux2_00324; mention only the bounded local artifact facts. Synthetic v1-hard cas
- `tu-v1h-0015ab391684ef3c3318` timesfm_to_sympy_tool_call [train; chain,cross_specialist,paraphrase]: Verified TimesFM forecast values are [207.6, 211.85, 216.1, 220.35]. Use SymPy to compute the exact total demand expression. Synthetic v1-hard case 000765. Difficulty=chain,cross_s
