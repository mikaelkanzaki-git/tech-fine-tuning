# Curadoria conservadora do dataset SFT

`curate-sft` cria uma nova derivação do dataset já auditado. A etapa não inventa, resume ou
corrige fatos médicos. O perfil `medquad_conservative_v1` executa apenas operações determinísticas:

- remove a repetição literal da pergunta no início da resposta;
- exclui respostas que apenas encaminham o leitor para outros recursos;
- preserva listas de sinais e sintomas, removendo somente o texto explicativo repetido da HPO.

Duplicidades, respostas longas e repetições clínicas não são removidas automaticamente. Esses
alertas continuam visíveis na auditoria porque escolher uma resposta médica preferencial exige um
critério adicional.

## Execução

```powershell
uv run tech-fine-tuning curate-sft `
  --source "artifacts/sft-v2" `
  --output "artifacts/sft-candidate-v3" `
  --system-prompt "You are a medical information assistant. Answer only the question asked with concise educational information grounded in reliable medical sources. If reliable information is unavailable, say that it is unknown instead of inventing details. Do not prescribe medication, diagnose a patient, or replace a qualified healthcare professional."
```

`--system-prompt` é opcional. Quando informado, substitui somente a mensagem de sistema dos três
splits e registra o texto e seu SHA-256 em `manifest.json`; perguntas e respostas médicas continuam
inalteradas pelas regras abaixo.

O diretório de saída deve ser novo ou estar vazio. Ele contém os três splits, `manifest.json` e
`excluded.jsonl`. O arquivo de excluídos registra apenas `record_id`, split e motivo; as respostas
descartadas permanecem recuperáveis no dataset de origem. O manifesto guarda hashes de todas as
entradas e saídas, contagens por regra e a verificação de isolamento entre splits.

Depois da curadoria, execute `audit-sft` sobre a nova saída antes de treinar.
