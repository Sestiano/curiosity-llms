import requests
import json

# La nostra grammatica
gbnf_grammar = r'root ::= ("Giorgia" | "Sebastiano")'

# L'endpoint API di Ollama per il completamento (non-chat)
# NOTA: Usiamo /api/generate, non /api/chat, per lo stesso motivo
# di LM Studio: evitare conflitti con i template di chat.
url = "http://localhost:11434/api/generate"

# Un prompt semplice
prompt = "Domanda: Chi è più veloce? Risposta:"

payload = {
    "model": "llama3",  # Sostituisci con il tuo modello Ollama
    "prompt": prompt,
    "stream": False,
    "options": {
        "grammar": gbnf_grammar,  # <-- Passiamo la grammatica qui
        "num_predict": 2          # Limitiamo i token generati
    }
}

print("Chiamo Ollama con grammatica nativa...")

try:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    response_data = response.json()
    
    risposta = response_data['response'].strip()
    print(f"\nRisposta forzata da Ollama: {risposta}")

    if risposta in ["Giorgia", "Sebastiano"]:
        print("(Successo! La grammatica è stata rispettata)")
    else:
        print("(Fallimento! La grammatica è stata ignorata)")

except Exception as e:
    print(f"Errore: Assicurati che Ollama sia in esecuzione. {e}")