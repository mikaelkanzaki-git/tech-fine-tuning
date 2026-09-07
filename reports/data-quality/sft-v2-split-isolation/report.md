# Auditoria do dataset SFT

Exemplos analisados: 16375.
Registros com ao menos um alerta: 7596.

## Comprimento e alertas por split

| Split | Exemplos | Com alerta | P50 caracteres | P95 | P99 | Máximo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 13332 | 6188 | 872 | 3208 | 7501 | 24668 |
| validation | 1560 | 702 | 880 | 3258 | 6436 | 21381 |
| test | 1483 | 706 | 903 | 3382 | 9069 | 26980 |

## Alertas

- `answer_repeats_question`: 494
- `duplicate_answer_within_split`: 576
- `duplicate_question_within_split`: 3410
- `estimated_context_overflow`: 152
- `hpo_boilerplate`: 2243
- `repeated_sentence`: 90
- `resource_navigation_answer`: 1085

## Sobreposição de perguntas entre splits

- `train` x `validation`: 0
- `train` x `test`: 0
- `validation` x `test`: 0

A estimativa de tokens usa quatro caracteres por token e serve apenas como triagem. A validação exata deve usar o tokenizer fixado na configuração do modelo.
