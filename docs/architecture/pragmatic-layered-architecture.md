# Arquitetura em Camadas Pragmática

## Status

Adotada pelo `tech-fine-tuning` desde a primeira fatia executável. A organização segue a referência do
`ms-ai-agent`, adaptada somente às responsabilidades existentes neste serviço.

## Estrutura atual

```text
src/tech_fine_tuning/
├── models/          Registros canônicos mínimos e exemplos SFT
├── services/        Preparação, treinamento e avaliação
├── integrations/    Filesystem, datasets e futuramente Unsloth
├── config/          Settings e composição
├── errors.py        Erros estáveis apresentados pela CLI
└── runner.py        Entrada local
```

`repositories/` não existe porque nenhum estado de domínio é persistido em banco. `api/` não
existe porque esta etapa é executada por CLI. Esses diretórios só serão criados quando houver
uma responsabilidade concreta.

## Direção das dependências

```text
runner ───────────────> services ───────────────> models
  │                         │
  └────> config             └────> integrations/dataset
```

- `models/` não importa SDKs ou camadas externas.
- `services/` decide as regras de aceitação e isolamento do treinamento.
- `integrations/` traduz JSON/JSONL e, futuramente, encapsulará Unsloth e Transformers.
- `config/` contém apenas settings e composição.
- o `tech-fine-tuning` depende do contrato publicado pelo `tech-ingestao`, nunca do pacote Python dele.

## Limite entre ingestão e treinamento

O `tech-ingestao` decide quais dados estão aptos: normaliza, sanitiza, deduplica e divide. O
`tech-fine-tuning` decide como os dados aprovados são apresentados ao modelo: instrução de sistema,
mensagens, chat template, tokenização, packing e hiperparâmetros.

A primeira projeção usa mensagens independentes de modelo. O chat template não é persistido no
JSONL porque depende do tokenizer do modelo base selecionado. Essa separação permite trocar
Llama, Mistral, Gemma ou Qwen sem reexecutar a ingestão.
