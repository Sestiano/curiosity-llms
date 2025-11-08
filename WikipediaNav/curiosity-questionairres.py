from llama_cpp import Llama, LlamaGrammar
import json
import numpy as np

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


def calculate_dimension_scores(results):
    """
    Calcola i punteggi medi per ogni dimensione secondo Kashdan et al. (2018).
    
    Returns:
        dict: Punteggi da 1-7 per ogni dimensione + percentile sample
    """
    scores = {}
    
    for dim_key, dimension in results.items():
        ratings = [item["rating"] for item in dimension["items"]]
        
        # Applica reverse scoring SOLO per stress_tolerance
        if dimension.get("reverse_scored", False):
            ratings = [8 - r for r in ratings]  # 8-x inverte la scala 1-7
        
        mean_score = sum(ratings) / len(ratings)
        
        scores[dim_key] = {
            "name": dimension["name"],
            "mean_score": mean_score,
            "raw_ratings": ratings,
            # Calcola anche deviazione standard per robustezza
            "std": np.std(ratings),
            # Normalizza in POMP (Percentage of Maximum Possible) come negli studi
            "pomp": ((mean_score - 1) / 6) * 100  # da scala 1-7 a 0-100%
        }
    
    return scores


def _add_secondary_modifiers(profile, scores):
    """Aggiungi modifiche basate su altre dimensioni (effetto più debole)."""
    
    # Stress Tolerance: alto → più disposto a esplorare incerto
    if scores["stress_tolerance"]["mean_score"] >= 5.0:
        profile["system_prompt_additions"].append(
            "You handle uncertainty well - don't hesitate to explore unfamiliar territory."
        )
    elif scores["stress_tolerance"]["mean_score"] <= 3.0:
        profile["system_prompt_additions"].append(
            "Build confidence gradually - start with familiar concepts before venturing far."
        )
    
    # Joyous Exploration: influenza l'entusiasmo
    if scores["joyous_exploration"]["mean_score"] >= 5.0:
        profile["system_prompt_additions"].append(
            "Express genuine enthusiasm for discovering new information."
        )
    
    # Social Curiosity: focus su people-related topics
    if scores["social_curiosity"]["mean_score"] >= 5.0:
        profile["system_prompt_additions"].append(
            "When relevant, prioritize information about people, social phenomena, and human behavior."
        )


def assign_navigation_style_scientific(scores):
    """
    Assegna stile di navigazione basato su evidenze da Lydon-Staley et al. (2021).
    
    Focus su Deprivation Sensitivity come predittore principale dello stile.
    """
    profile = {
        "deprivation_score": scores["deprivation_sensitivity"]["mean_score"],
        "deprivation_pomp": scores["deprivation_sensitivity"]["pomp"],
        "navigation_style": None,
        "reinforcement_tendency": None,
        "regularity_preference": None,
        "system_prompt_additions": []
    }
    
    ds_score = scores["deprivation_sensitivity"]["mean_score"]
    
    # Soglie basate su distribuzioni degli studi (media campione ≈ 4.0, SD ≈ 1.1)
    # Alta DS: > media + 0.5 SD (≈ 4.5+)
    # Bassa DS: < media - 0.5 SD (≈ 3.5-)
    
    if ds_score >= 4.5:
        profile["navigation_style"] = "hunter"
        profile["reinforcement_tendency"] = "high"  # Torna spesso su pagine visitate
        profile["regularity_preference"] = "short_steps"  # Preferisce concetti vicini
        profile["system_prompt_additions"].extend([
            "When you encounter a knowledge gap, pursue closely related concepts systematically.",
            "Revisit previous topics to build deeper understanding before moving to new areas.",
            "Focus on eliminating uncertainty about specific topics through targeted exploration.",
            "Prefer depth over breadth - exhaust related concepts before large conceptual jumps."
        ])
        
    elif ds_score <= 3.5:
        profile["navigation_style"] = "busybody"
        profile["reinforcement_tendency"] = "low"
        profile["regularity_preference"] = "long_jumps"  # Salta tra concetti distanti
        profile["system_prompt_additions"].extend([
            "Sample diverse concepts without dwelling on any single topic for too long.",
            "Make surprising conceptual leaps between distant topics.",
            "Prioritize breadth of exploration over systematic depth.",
            "Follow whatever sparks immediate interest, even if unrelated to previous topics."
        ])
    
    else:
        profile["navigation_style"] = "mixed"
        profile["reinforcement_tendency"] = "moderate"
        profile["regularity_preference"] = "balanced"
        profile["system_prompt_additions"].extend([
            "Balance systematic exploration with occasional diverse jumps.",
            "Return to previous concepts when uncertainty arises, but also explore new areas.",
            "Adapt your strategy based on the information encountered."
        ])
    
    # Aggiungi modificatori dalle altre dimensioni (meno peso)
    _add_secondary_modifiers(profile, scores)
    
    return profile


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
    
    # Calcola i punteggi per dimensione
    print("\n=== Calculating Dimension Scores ===")
    scores = calculate_dimension_scores(results)
    
    print("\nDimension Scores:")
    for dim_key, dim_data in scores.items():
        print(f"\n{dim_data['name']}:")
        print(f"  Mean Score: {dim_data['mean_score']:.2f}/7")
        print(f"  POMP: {dim_data['pomp']:.1f}%")
        print(f"  Std Dev: {dim_data['std']:.2f}")
    
    # Assegna lo stile di navigazione basato sui punteggi
    print("\n=== Assigning Navigation Style ===")
    navigation_profile = assign_navigation_style_scientific(scores)
    
    print(f"\nNavigation Style: {navigation_profile['navigation_style'].upper()}")
    print(f"Deprivation Sensitivity Score: {navigation_profile['deprivation_score']:.2f}/7")
    print(f"Reinforcement Tendency: {navigation_profile['reinforcement_tendency']}")
    print(f"Regularity Preference: {navigation_profile['regularity_preference']}")
    
    print("\nSystem Prompt Additions:")
    for i, addition in enumerate(navigation_profile['system_prompt_additions'], 1):
        print(f"  {i}. {addition}")
    
    # Salva il profilo completo
    complete_profile = {
        "dimension_scores": scores,
        "navigation_profile": navigation_profile
    }
    
    profile_file = "navigation_profile.json"
    with open(profile_file, 'w', encoding='utf-8') as f:
        json.dump(complete_profile, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Complete profile saved to {profile_file}")
