# Contrato do dataset SFT

O comando `prepare-sft` deriva exemplos conversacionais do schema canônico `1.1` produzido pelo
`tech-ingestao`. A derivação não altera os splits e não aplica tokenizer ou chat template.

## Exemplo

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a medical information assistant..."
    },
    {
      "role": "user",
      "content": "What are the symptoms of diabetes?"
    },
    {
      "role": "assistant",
      "content": "Common symptoms include..."
    }
  ],
  "metadata": {
    "canonical_schema_version": "1.1",
    "record_id": "...",
    "document_id": "...",
    "content_sha256": "...",
    "pii_status": "not_detected",
    "source": {
      "dataset": "MedQuAD",
      "relative_path": "5_NIDDK_QA/0000035.xml",
      "upstream_revision": "...",
      "url": "...",
      "license": "CC-BY-4.0"
    }
  }
}
```

Somente `messages` será entregue ao tokenizer durante o treinamento. `metadata` sustenta
rastreabilidade, depuração e avaliação, e o manifesto declara explicitamente
`metadata_is_training_text: false`.

## Splits

- `train.jsonl`: atualiza os pesos do modelo.
- `validation.jsonl`: acompanha perda e qualidade durante o desenvolvimento.
- `test.jsonl`: permanece reservado para avaliação final.

O serviço verifica de forma independente que nenhum `document_id`, `record_id` ou
`content_sha256` aparece em mais de um split. As contagens precisam coincidir com o manifesto
canônico.

## Reprodutibilidade

O `manifest.json` SFT registra:

- SHA-256 do manifesto canônico e de cada arquivo de entrada;
- revisão upstream do MedQuAD;
- instrução de sistema e seu SHA-256;
- contagens por split e registros com conteúdo redigido;
- SHA-256 dos arquivos SFT gerados;
- resultado das verificações de vazamento.

O arquivo de mensagens ainda é independente de modelo. Depois que o modelo base for escolhido,
o carregador de treinamento aplicará `tokenizer.apply_chat_template` em memória e registrará o
nome e a revisão do tokenizer no manifesto do treinamento.
