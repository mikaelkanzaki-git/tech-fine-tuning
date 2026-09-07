# Auditoria do dataset SFT

Exemplos analisados: 16375.
Registros com ao menos um alerta: 7148.

## Comprimento e alertas por split

| Split | Exemplos | Com alerta | P50 caracteres | P95 | P99 | Máximo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 13234 | 5779 | 871 | 3210 | 7611 | 26980 |
| validation | 1561 | 644 | 892 | 3163 | 5769 | 21381 |
| test | 1580 | 725 | 896 | 3499 | 8192 | 19615 |

## Alertas

- `answer_repeats_question`: 1
- `duplicate_answer_within_split`: 577
- `duplicate_question_within_split`: 2767
- `estimated_context_overflow`: 152
- `hpo_boilerplate`: 2243
- `question_overlap_across_splits`: 958
- `repeated_sentence`: 90
- `resource_navigation_answer`: 1085

## Sobreposição de perguntas entre splits

- `train` x `validation`: 174
- `train` x `test`: 191
- `validation` x `test`: 23

A estimativa de tokens usa quatro caracteres por token e serve apenas como triagem. A validação exata deve usar o tokenizer fixado na configuração do modelo.
