# Auditoria do dataset SFT

`audit-sft` inspeciona o dataset conversacional antes do treinamento. A auditoria não reescreve
respostas, não move registros entre splits e não altera o manifesto de origem.

## Execução

```powershell
uv run tech-fine-tuning audit-sft `
  --source "artifacts/sft" `
  --output "reports/data-quality/sft-baseline" `
  --max-sequence-length 2048
```

O diretório de saída deve ser novo ou estar vazio. Cada execução produz:

```text
reports/data-quality/<auditoria>/
├── audit-summary.json   Contagens agregadas por split e tipo de alerta
├── issues.jsonl         Registros afetados, sem copiar o texto das respostas
├── report.md            Resumo legível
└── audit-manifest.json  Hashes das entradas e saídas
```

## Alertas

- `hpo_boilerplate`: explicação genérica da Human Phenotype Ontology repetida nas respostas;
- `resource_navigation_answer`: resposta que apenas encaminha para recursos de diagnóstico ou
  manejo;
- `answer_repeats_question`: resposta que começa repetindo integralmente a pergunta;
- `repeated_sentence`: a mesma frase longa aparece três ou mais vezes na resposta;
- `estimated_context_overflow`: estimativa simples acima do limite configurado;
- `duplicate_question_within_split`: pergunta normalizada repetida no mesmo split;
- `duplicate_answer_within_split`: resposta normalizada repetida no mesmo split;
- `question_overlap_across_splits`: a mesma pergunta normalizada aparece em splits diferentes.

A normalização de perguntas usa Unicode NFKC, ignora maiúsculas, minúsculas e diferenças de
espaços e remove espaços antes de pontuação. O controle é mais estrito que a separação por
documento porque documentos diferentes podem conter a mesma pergunta.

## Uso do resultado

Alertas não autorizam alterações médicas automáticas. O relatório serve para definir regras
determinísticas de curadoria. Sobreposição entre splits precisa ser resolvida antes de uma nova
comparação; caso contrário, a avaliação pode medir memorização da pergunta em vez de generalização.
