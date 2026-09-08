# Auditoria do dataset SFT

Exemplos analisados: 15290.
Registros com ao menos um alerta: 3665.

## Comprimento e alertas por split

| Split | Exemplos | Com alerta | P50 caracteres | P95 | P99 | Máximo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 12444 | 2993 | 824 | 3102 | 7763 | 24668 |
| validation | 1457 | 330 | 830 | 3211 | 6636 | 21381 |
| test | 1389 | 342 | 826 | 3361 | 9071 | 26980 |

## Alertas

- `answer_repeats_question`: 5
- `duplicate_answer_within_split`: 578
- `duplicate_question_within_split`: 3013
- `estimated_context_overflow`: 157
- `repeated_sentence`: 90

## Sobreposição de perguntas entre splits

- `train` x `validation`: 0
- `train` x `test`: 0
- `validation` x `test`: 0

A estimativa de tokens usa quatro caracteres por token e serve apenas como triagem. A validação exata deve usar o tokenizer fixado na configuração do modelo.
