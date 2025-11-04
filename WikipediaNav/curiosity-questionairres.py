from llama_cpp import Llama, LlamaGrammar
import json

# Scala Likert per i questionari sulla curiosità
Scale = {
    1: "Does not describes me at all",
    2: "Barely describes me",
    3: "Somewhat describes me",
    4: "Neutral",
    5: "Generally describes me",
    6: "Mostly describes me",
    7: "Completely describes me"
}

def create_likert_grammar():
    """
    Crea una grammatica LLama.cpp che forza il modello a rispondere
    con un singolo rating della scala Likert (1-7).
    """
    return LlamaGrammar.from_string('''root ::= rating-value
rating-value ::= "1" | "2" | "3" | "4" | "5" | "6" | "7"''')


# Five-Dimensional Curiosity Scale (5DC)
# Kashdan, T. B., et al. (2018)

CURIOSITY_DIMENSIONS = {
    "joyous_exploration": {
        "name": "Joyous Exploration",
        "items": [
            "I view challenging situations as an opportunity to grow and learn.",
            "I am always looking for experiences that challenge how I think about myself and the world.",
            "I seek out situations where it is likely that I will have to think in depth about something.",
            "I enjoy learning about subjects that are unfamiliar to me.",
            "I find it fascinating to learn new information."
        ]
    },
    "deprivation_sensitivity": {
        "name": "Deprivation Sensitivity",
        "items": [
            "Thinking about solutions to difficult conceptual problems can keep me awake at night.",
            "I can spend hours on a single problem because I just can't rest without knowing the answer.",
            "I feel frustrated if I can't figure out the solution to a problem, so I work even harder to solve it.",
            "I work relentlessly at problems that I feel must be solved.",
            "It frustrates me not having all the information I need."
        ]
    },
    "stress_tolerance": {
        "name": "Stress Tolerance",
        "items": [
            "The smallest doubt can stop me from seeking out new experiences.",
            "I cannot handle the stress that comes from entering uncertain situations.",
            "I find it hard to explore new places when I lack confidence in my abilities.",
            "I cannot function well if I am unsure whether a new experience is safe.",
            "It is difficult to concentrate when there is a possibility that I will be taken by surprise."
        ],
        "reverse_scored": True
    },
    "social_curiosity": {
        "name": "Social Curiosity",
        "items": [
            "I like to learn about the habits of others.",
            "I like finding out why people behave the way they do.",
            "When other people are having a conversation, I like to find out what it's about.",
            "When around other people, I like listening to their conversations.",
            "When people quarrel, I like to know what's going on."
        ]
    },
    "thrill_seeking": {
        "name": "Thrill Seeking",
        "items": [
            "The anxiety of doing something new makes me feel excited and alive.",
            "Risk-taking is exciting to me.",
            "When I have free time, I want to do things that are a little scary.",
            "Creating an adventure as I go is much more appealing than a planned adventure.",
            "I prefer friends who are excitingly unpredictable."
        ]
    }
}


def create_single_item_prompt(item_number, item_text):
    """
    Crea un prompt per valutare un singolo item del questionario.
    """
    scale_description = "\n".join([f"{k}: {v}" for k, v in Scale.items()])
    
    return f"""You are a Large Language Model acting as a Wikipedia navigator.
        You are completing the Five-Dimensional Curiosity Scale questionnaire.
        
        Please rate how well the following statement describes you or your typical behavior.
        
        Use this scale:
        {scale_description}
        
        
        STATEMENT {item_number}
        {item_text}
        
        Respond with only a single number from 1 to 7:"""


def evaluate_all_items(llm_model, output_file="curiosity_results.json"):
    """
    Valuta tutti i 25 item del questionario uno alla volta e salva i risultati in un file JSON.
    
    Args:
        llm_model: Istanza del modello Llama
        output_file: Nome del file JSON dove salvare i risultati
    
    Returns:
        dict: Dizionario con tutti i risultati
    """
    grammar = create_likert_grammar()
    all_results = {}
    
    item_number = 1
    
    for dim_key, dimension in CURIOSITY_DIMENSIONS.items():
        print(f"\n=== Evaluating {dimension['name']} ===")
        dimension_results = []
        
        for item_text in dimension["items"]:
            print(f"\nItem {item_number}: {item_text}")
            
            prompt = create_single_item_prompt(item_number, item_text)
            
            response = llm_model(
                prompt,
                grammar=grammar,
                max_tokens=5,
                temperature=0.7,
                stop=["\n"]
            )
            
            rating = int(response["choices"][0]["text"].strip())
            print(f"Rating: {rating}")
            
            dimension_results.append({
                "item_number": item_number,
                "item_text": item_text,
                "rating": rating
            })
            
            item_number += 1
        
        all_results[dim_key] = {
            "name": dimension["name"],
            "reverse_scored": dimension.get("reverse_scored", False),
            "items": dimension_results
        }
    
    # Salva i risultati in un file JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to {output_file}")
    
    return all_results

# Esempio di utilizzo
if __name__ == "__main__":
    # Carica il modello (modifica il path secondo necessità)
    model_path = "../models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    
    print("Loading model...")
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=4,
        verbose=False
    )
    
    print("Model loaded successfully!")
    
    # Valuta tutti gli item uno alla volta
    results = evaluate_all_items(llm, output_file="curiosity_results.json")
    
    print("\n=== Evaluation Complete ===")
    print(f"Total items evaluated: 25")

