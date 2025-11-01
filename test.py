from ollama import generate
# Non serve importare ChatResponse per 'generate'

# La nostra grammatica
gbnf_grammar = r'root ::= ("Giorgia" | "Sebastiano")'

# 'generate' usa 'prompt' (stringa) e 'options' come argomenti separati
response = generate(
    model='mistral:7b',
    prompt="Who's faster?",  # <-- Usa 'prompt'
    options={
        "grammar": gbnf_grammar
    }
)

# 'generate' salva la risposta nella chiave 'response'
print(response['response'])