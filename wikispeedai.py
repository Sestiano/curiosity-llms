import wikipedia
import requests
import json
from datetime import datetime
import warnings
from tqdm import tqdm
import os
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
import random

# --- Setup ---
wikipedia.set_lang('en')
warnings.filterwarnings('ignore')

# ===== CONFIGURATION =====
def get_lm_studio_model_name(lm_studio_url):
    """Retrieve the current model name from LM Studio API."""
    try:
        models_url = lm_studio_url.replace('/v1/chat/completions', '/v1/models')
        response = requests.get(models_url, timeout=20)
        response.raise_for_status()
        result = response.json()
        if 'data' in result and len(result['data']) > 0:
            # Get the first model (usually the loaded one)
            model_name = result['data'][0].get('id', 'lm_studio')
            print(f"Auto-detected LM Studio model: {model_name}")
            return model_name
        else:
            print("No model found in LM Studio response, using default 'lm_studio'")
            return 'lm_studio'
    except Exception as e:
        print(f"Could not retrieve model name from LM Studio: {e}")
        print("Using default 'lm_studio' as model name")
        return 'lm_studio'

def load_config():
    """Load and validate configuration from config.json or use defaults"""
    # Default configuration values
    defaults = {
        'start_pages': ['Vaccine', 'Albert Einstein'],
        'target': 'Renaissance',
        'iterations_per_start_page': 100,
        'temperatures': [0.3, 1.5],
        'personalities': ['baseline', 'busybody', 'hunter', 'dancer'],
        'lm_studio_url': "http://localhost:1234/v1/chat/completions",
      # "model_name": "qwen/qwen3-4b-thinking-2507", add this line in config.json with the exact model name if you're going to use version 1 
        'fuzzy_match_threshold': 0.95,
        'max_loop_repetitions': 3,
        'max_correction_attempts': 2  # Max retries after an error like hallucination
    }
    
    # Load configuration from file if it exists, otherwise use defaults
    config_file = 'config.json'
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = defaults

    # Auto-detect model name from LM Studio if not specified
    if 'model_name' not in config or not config['model_name']:
        print("Model name not specified in config, auto-detecting from LM Studio...")
        config['model_name'] = get_lm_studio_model_name(config.get('lm_studio_url', defaults['lm_studio_url']))
    else:
        print(f"Using model name from config: {config['model_name']}")
    
    # Create output directory name from model name
    model_name = config['model_name']
    config['output_dir'] = model_name.replace('/', '_').replace(':', '_').replace('\\', '_')

    # Normalize start_pages (sp) and temperature (t) to always be a list
    sp = config.get('start_pages', [config.get('start_page', 'Vaccine')])
    config['start_pages'] = [sp] if isinstance(sp, str) else sp
    t = config.get('temperatures', [0.3])
    config['temperatures'] = [t] if isinstance(t, (int, float)) else t
    
    return config

CONFIG = load_config()

# Define the baseline prompt that is always used
BASELINE_PROMPT = """
You are a Large Language Model acting as a Wikipedia navigator. 
Your goal is to reach the TARGET article by selecting every time a link.

Rules:
- Read the CURRENT PAGE CONTENT to understand the topic and available connections
- Choose ONE link that brings you towards the target's webpage
- You MUST reply in JSON format with TWO fields:
  * "link": the EXACT LINK name from the available options
  * "reason": a BRIEF explanation of why you chose this link
"""

# Define personality traits that are ADDED to the baseline
PERSONALITY_TRAITS = {
    "baseline": "",  # No additional traits
    
    "busybody": """
You MUST have a BUSYBODY type personality.
You like to explore many different things without going to deep.
You move quickly from one idea to another, following what feels new or interesting.
Your knowledge is wide and varied, often about culture and geography.
Keep these elements in mind now that you act as a BUSYBODY type of personality.
""",
    
    "hunter": """
You MUST have a HUNTER type personality.
You look for clear answers and follow a straight path.
You stay focused on one topic and explore it deeply.
Your knowledge is strong but narrow, often in science or technical fields.
Keep these elements in mind now that you act as a HUNTER type of personality.
""",
    
    "dancer": """
You MUST have a DANCER type personality.
You jump between distant ideas and mixes them in new ways.
You like to break rules and connect things that others don't.
Your curiosity leads to creative thoughs and fresh ways of understanding.
Keep these elements in mind now that you act as a DANCER type of personality.
"""
}

def create_navigation_prompt(current_article, target_article, links, path, page_content=None):
    """Create a prompt for the LLM with current context and available links"""
    
    # Show recent navigation path for context
    recent_path = ' '.join(path[:-1]) if len(path) > 1 else {start_page}

    # Format links as a list
    links_list = '\n'.join(links)
    
    # Include page content to help LLM make informed decisions
    content_text = page_content if page_content else "[no content available]"

    # Build the complete prompt with all necessary information
    prompt = f""" Your TARGET page of Wikipedia is: "{target_article}" 
                  Your CURRENT page of Wikipedia is: "{current_article}"
                  Your PREVIOUS PAGE was: {recent_path}
                  
                  Select the link by reading the CURRENT PAGE: {content_text}
                  
                  AVAILABLE LINKS: {links_list}
                  
                  TASK: Choose ONE link that brings you towards the target's webpage: "{target_article}".
                  Remember to respond in JSON format with both "link" and "reason" fields.
                  """

    return prompt

def create_disambiguation_prompt(original_choice, options, target_article, path):
    """
    Create a prompt for the LLM to resolve a disambiguation page.
    
    When Wikipedia returns multiple options for an ambiguous article name,
    this function creates a prompt asking the LLM to select the most relevant option.
    The options are shuffled to prevent position bias in the model's selection.
    
    Args:
        original_choice: The ambiguous article name that led to disambiguation
        options: List of possible disambiguation options from Wikipedia
        target_article: The final target article we're trying to reach
        path: List of articles visited so far in the navigation
    
    Returns:
        A formatted prompt string for the LLM
    """
    # Show recent navigation path for context
    recent_path = ' '.join(path[:-1]) if len(path) > 1 else {start_page}

    # Shuffle the options to avoid position bias (e.g., model preferring first option)
    shuffled_options = options.copy()
    random.shuffle(shuffled_options)
    options_list = '\n'.join(shuffled_options)

    prompt = f"""Your previous choice "{original_choice}" was AMBIGUOS.
                 Your PREVIOUS PAGE was: {recent_path}

                 To continue, you must choose ONE link that brings you towards the target's webpage: "{target_article}".
                 AVAILABLE OPTIONS:
                 {options_list}
                 
                 TASK: Choose the BEST option from the list to reach the target. Respond in JSON format with both "link" and "reason" fields.
                """
    return prompt

def create_correction_prompt(invalid_choice, target_article, links, path):
    """
    Create a prompt for the LLM to correct an invalid link selection.
    
    When the LLM hallucinates or selects a link that doesn't exist in the available options,
    this function creates a correction prompt to guide the model to choose a valid link.
    The links are shuffled to prevent position bias in the corrected selection.
    
    Args:
        invalid_choice: The invalid link name that was hallucinated or incorrectly selected
        target_article: The final target article we're trying to reach
        links: List of actually available links on the current page
        path: List of articles visited so far in the navigation
    
    Returns:
        A formatted correction prompt string for the LLM
    """
    # Show recent navigation path for context
    recent_path = ' '.join(path[:-1]) if len(path) > 1 else {start_page}

    # Shuffle the links to avoid position bias (e.g., model always picking first link after error)
    shuffled_links = links.copy()
    random.shuffle(shuffled_links)
    links_list = '\n'.join(shuffled_links)

    prompt = f"""CORRECTION REQUIRED: Your previous choice "{invalid_choice}" was INVALID because it is NOT in the list of available links.

                 Your TARGET is : "{target_article}"
                 Your PREVIOUS PAGE was: {recent_path}
                 
                 Please review the AVAILABLE LINKS carefully. You must choose one of these available links: {links_list}
                 
                 TASK: Choose a VALID link from the list. Respond in JSON.
                """
    return prompt

# ===== API & WIKIPEDIA FUNCTIONS =====
def call_lm_studio(messages, temperature=0.3):
    """Call LM Studio API."""
    payload = {
        "model": CONFIG.get('model_name', 'local-model'),
        "messages": messages, 
        "temperature": temperature,
        "max_tokens": -1,  # -1 means use maximum available
        "stream": False
    }
    try:
        response = requests.post(CONFIG['lm_studio_url'], json=payload, timeout=1200)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except requests.exceptions.RequestException as e:
        print(f"Error calling LM Studio: {e}")
        # Print response content if available for debugging
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return None

def get_personality_prompt(personality_name="baseline"):
    """Get the combined system prompt."""
    traits = PERSONALITY_TRAITS.get(personality_name, "")
    return f"{BASELINE_PROMPT}\n{traits}" if traits else BASELINE_PROMPT

def get_wikipedia_page(title):
    """
    Loads a Wikipedia page with scientifically sound error handling.
    It reports ambiguities instead of resolving them.

    Returns: tuple (status, result)
    - ('success', page_object)
    - ('disambiguation', options_list)
    - ('not_found', None)
    - ('error', error_message)
    """
    try:
        page = wikipedia.page(title, auto_suggest=False, redirect=True)
        return 'success', page
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"INFO: '{title}' is a disambiguation page. Returning options.")
        return 'disambiguation', e.options
    except wikipedia.exceptions.PageError:
        print(f"INFO: Page '{title}' not found. Trying with auto-suggest...")
        try:
            page = wikipedia.page(title, auto_suggest=True, redirect=True)
            print(f"INFO: Found alternative page: '{page.title}'")
            return 'success', page
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"INFO: Suggestion for '{title}' also led to a disambiguation.")
            return 'disambiguation', e.options
        except wikipedia.exceptions.PageError:
            print(f"ERROR: No page found for '{title}', even with suggestions.")
            return 'not_found', None
        except Exception as e:
            print(f"ERROR: Unexpected error during auto-suggest for '{title}': {e}")
            return 'error', str(e)
    except Exception as e:
        print(f"ERROR: Unexpected error while loading '{title}': {e}")
        return 'error', str(e)

def get_page_links(page, exclude_keywords=None):
    """Extract links from a Wikipedia page in order of appearance."""
    if not page: return []
    exclude = exclude_keywords or ['Category:', 'Template:', 'File:', 'Portal:', 'Help:', 'Wikipedia:', 'Talk:', 'Special:']
    try:
        from urllib.parse import unquote
        soup = BeautifulSoup(page.html(), 'html.parser')
        content = soup.find('div', {'id': 'mw-content-text'})
        if not content:
            return [l for l in page.links if not any(kw in l for kw in exclude)]
        
        seen, links = set(), []
        for a in content.find_all('a', href=True):
            href = a.get('href', '')
            if href.startswith('/wiki/'):
                title = unquote(href.replace('/wiki/', '').replace('_', ' '))
                if not any(kw in title for kw in exclude) and title not in seen:
                    seen.add(title)
                    links.append(title)
        return links
    except Exception as e:
        print(f"Error parsing HTML: {e}, using fallback.")
        try:
            return [l for l in page.links if not any(kw in l for kw in exclude)]
        except:
            return []

def fuzzy_similarity(s1, s2):
    """Calculate fuzzy similarity between two strings."""
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

def parse_model_choice(response, available_links):
    """Parse the model's response and match it to an available link."""
    if not response: return None, {}
    
    metadata = {"fallback_used": False, "match_type": "unknown", "similarity_score": 0.0, "reason": "", "raw_choice": ""}
    
    try:
        response_data = json.loads(response.strip())
        chosen_text = response_data.get("link", "").strip()
        metadata["reason"] = response_data.get("reason", "").strip()
    except json.JSONDecodeError:
        print(f"JSON parsing failed, treating response as plain text link.")
        chosen_text = response.strip().strip('"\'.,:;!?()[]{}')
        metadata["reason"] = "No reason provided (non-JSON response)"
    
    metadata["raw_choice"] = chosen_text
    if not chosen_text: return None, metadata

    for link in available_links:
        if link.lower() == chosen_text.lower():
            metadata.update({"match_type": "exact_match", "similarity_score": 1.0})
            return link, metadata

    threshold = CONFIG.get('fuzzy_match_threshold', 0.95)
    best_link, best_score = max(((link, fuzzy_similarity(chosen_text, link)) for link in available_links), key=lambda item: item[1], default=(None, 0.0))
    
    if best_score >= threshold:
        metadata.update({"fallback_used": True, "match_type": "fuzzy_match", "similarity_score": best_score})
        print(f"FALLBACK: Fuzzy match ({best_score:.2f}) - '{chosen_text}' -> '{best_link}'")
        return best_link, metadata

    metadata.update({"fallback_used": True, "match_type": "no_match_hallucination", "similarity_score": best_score})
    print(f"HALLUCINATION: No match for '{chosen_text}' (best: {best_score:.2f})")
    return None, metadata

# ===== EXPERIMENT RUNNER =====
def run_navigation_experiment(start_article, target_article, temperature=0.3, personality="baseline"):
    current, path, visited = start_article, [start_article], {start_article.lower()}
    success, detailed_steps, loop_detected = False, [], False
    error_type = None  # Track why the experiment failed
    
    fallback_stats = {
        "total_fallbacks": 0, "fuzzy_match_fallbacks": 0, "total_hallucinations": 0,
        "fallback_rate": 0.0, "avg_similarity_score": 0.0,
        "steps_with_fallback": [], "successful_fallback_steps": [], "fallback_success_rate": 0.0
    }
    
    max_loop_repetitions = CONFIG.get('max_loop_repetitions', 3)
    max_correction_attempts = CONFIG.get('max_correction_attempts', 2)
    recent_transitions = []
    system_prompt = get_personality_prompt(personality)
    
    step = 0
    while step < 50:  # Max steps to prevent infinite runs
        if current.lower() == target_article.lower():
            success = True
            break
        
        # --- 1. LOAD PAGE (with disambiguation handling loop) ---
        page, status = None, ''
        temp_current = current
        
        for _ in range(max_correction_attempts):
            status, result = get_wikipedia_page(temp_current)
            
            if status == 'success':
                page = result
                current = page.title  # Update to actual title after redirects
                break
            
            elif status == 'disambiguation':
                print("Disambiguation found. Asking LLM to choose.")
                options = result
                prompt = create_disambiguation_prompt(temp_current, options, target_article, path)
                response = call_lm_studio([{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}], temperature)
                
                chosen_option, _ = parse_model_choice(response, options)
                
                if chosen_option:
                    temp_current = chosen_option
                else:
                    print("ERROR: LLM failed to resolve disambiguation. Halting.")
                    status = 'error'
                    error_type = "disambiguation_resolution_failed"
                    break
            else: # 'not_found' or 'error'
                error_type = f"page_load_failed_{status}"
                break
        
        if not page or status != 'success':
            if not error_type:
                error_type = "page_load_unknown_error"
            break

        # --- 2. EXTRACT LINKS ---
        available = [l for l in get_page_links(page) if l.lower() not in visited]
        if not available:
            print("Dead end reached.")
            error_type = "dead_end_no_links"
            break
        
        # --- 3. GET LLM CHOICE (with self-correction loop) ---
        chosen, metadata = None, {}
        for attempt in range(max_correction_attempts):
            prompt = create_navigation_prompt(current, target_article, available, path, page.content) if attempt == 0 \
                else create_correction_prompt(metadata.get("raw_choice"), target_article, available, path)
            
            response = call_lm_studio([{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}], temperature)
            if not response:
                error_type = "api_call_failed"
                break
            
            chosen, metadata = parse_model_choice(response, available)
            metadata["correction_attempts"] = attempt
            
            if chosen: break
        
        # --- 4. PROCESS RESULT ---
        if metadata.get("fallback_used"):
            fallback_stats["total_fallbacks"] += 1
            fallback_stats["steps_with_fallback"].append(step)
            if metadata["match_type"] == "fuzzy_match": fallback_stats["fuzzy_match_fallbacks"] += 1
            elif metadata["match_type"] == "no_match_hallucination": fallback_stats["total_hallucinations"] += 1
        
        if not chosen:
            print("LLM failed to provide a valid link after retries. Halting.")
            if not error_type:
                error_type = "hallucination_max_retries_exceeded"
            break
            
        # --- 5. LOOP DETECTION & STATE UPDATE ---
        transition = (current.lower(), chosen.lower())
        recent_transitions.append(transition)
        if len(recent_transitions) > max_loop_repetitions * 2: recent_transitions.pop(0)
        
        if recent_transitions.count(transition) >= max_loop_repetitions:
            loop_detected = True
            error_type = "loop_detected"
            print(f"LOOP DETECTED: Transition '{current}' -> '{chosen}' repeated.")
            break
            
        detailed_steps.append({
            "step_number": step, "origin_page": current, "destination_page": chosen,
            "reason": metadata.get("reason", ""), "available_links_count": len(available),
            "total_visited": len(visited), "fallback_used": metadata.get("fallback_used", False),
            "match_type": metadata.get("match_type", "unknown"), "similarity_score": metadata.get("similarity_score", 0.0),
            "correction_attempts": metadata.get("correction_attempts", 0)
        })
        
        current = chosen
        path.append(current)
        visited.add(current.lower())
        step += 1

    # --- 6. FINAL STATISTICS CALCULATION ---
    total_steps = len(detailed_steps)
    if total_steps > 0:
        fallback_stats["fallback_rate"] = (fallback_stats["total_fallbacks"] / total_steps) * 100
        
        sim_scores = [s["similarity_score"] for s in detailed_steps if s.get("similarity_score", 0) > 0]
        if sim_scores:
            fallback_stats["avg_similarity_score"] = sum(sim_scores) / len(sim_scores)
        
        if success:
            fallback_stats["successful_fallback_steps"] = [s["step_number"] for s in detailed_steps if s["fallback_used"]]
            if fallback_stats["total_fallbacks"] > 0:
                fallback_stats["fallback_success_rate"] = (len(fallback_stats["successful_fallback_steps"]) / fallback_stats["total_fallbacks"]) * 100

    return {
        "success": success, "steps": len(path) - 1, "path": path,
        "detailed_steps": detailed_steps, "fallback_statistics": fallback_stats,
        "start": start_article, "target": target_article, "model": CONFIG.get('model_name', 'unknown'),
        "personality": personality, "lm_studio_url": CONFIG['lm_studio_url'],
        "total_pages_visited": len(visited), "loop_detected": loop_detected,
        "error_type": error_type  # NEW: Track failure reason
    }

# ===== MAIN EXECUTION =====
if __name__ == '__main__':
    if 'model_name' not in CONFIG or CONFIG['model_name'] == 'lm_studio':
        detected_model = get_lm_studio_model_name()
        CONFIG['model_name'] = detected_model
        CONFIG['output_dir'] = detected_model.replace('/', '_').replace(':', '_').replace('\\', '_')
    
    print(f"Using model: {CONFIG['model_name']}")
    print(f"Output directory: {CONFIG['output_dir']}")

    all_results = []
    personalities = CONFIG.get('personalities', ['baseline'])
    start_pages = CONFIG['start_pages']
    temperatures = CONFIG['temperatures']
    iterations = CONFIG['iterations_per_start_page']
    
    for temp in temperatures:
        for start_page in start_pages:
            for personality in personalities:
                for i in tqdm(range(iterations), desc=f"{start_page} (T={temp}, P={personality})"):
                    # Set random seed for reproducibility
                    random.seed(f"{start_page}_{temp}_{personality}_{i}")
                    
                    result = run_navigation_experiment(start_page, CONFIG['target'], temp, personality)
                    
                    result.update({
                        'config_start_page': start_page, 
                        'temperature': temp, 
                        'iteration_in_group': i + 1
                    })
                    all_results.append(result)

                    # Save individual result IMMEDIATELY after each iteration
                    temp_str = str(temp).replace('.', '_')
                    output_dir = os.path.join(CONFIG['output_dir'], 
                                              f"temp_{temp_str}_personality_{personality}", 
                                              start_page.replace(' ', '_'))
                    os.makedirs(output_dir, exist_ok=True)
                    
                    filename = os.path.join(output_dir, f"result_{i+1:03d}.json")
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nCompleted {len(all_results)} iterations. Individual results saved in: {CONFIG['output_dir']}")