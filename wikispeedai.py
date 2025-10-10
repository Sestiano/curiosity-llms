import wikipedia
import requests
import json
from datetime import datetime
import warnings
from tqdm import tqdm
import os
from bs4 import BeautifulSoup
from difflib import SequenceMatcher

wikipedia.set_lang('en')
warnings.filterwarnings('ignore')

# ===== CONFIGURATION =====
def load_config():
    """Load and validate configuration from config.json or use defaults"""
    # Default configuration values
    defaults = {
        'start_pages': ['Vaccine', 'Philosophy'],
        'target': 'Renaissance',
        'iterations_per_start_page': 10,
        'temperatures': [0.3, 1.5],
        'personalities': ['baseline', 'busybody', 'hunter', 'dancer'],
        'lm_studio_url': "http://localhost:1234/v1/chat/completions",
        'fuzzy_match_threshold': 0.95,
        'max_loop_repetitions': 3,
    }
    
    # Load configuration from file if exists, otherwise use defaults
    config_file = 'config.json'
    config = json.load(open(config_file, 'r', encoding='utf-8')) if os.path.exists(config_file) else defaults

    # Create output directory name from model name
    model_name = config.get('model_name', 'lm_studio')
    config['output_dir'] = model_name.replace('/', '_').replace(':', '_').replace('\\', '_')

    # Normalize start_pages (sp) and temperature (t) to always be a list
    sp = config.get('start_pages', [config.get('start_page', 'Vaccine')])
    config['start_pages'] = [sp] if isinstance(sp, str) else sp
    t = config.get('temperatures', [0.3])
    config['temperatures'] = [t] if isinstance(t, (int, float)) else t
    
    return config

# Load global configuration
CONFIG = load_config()
start_pages = CONFIG['start_pages']
temperatures = CONFIG['temperatures']
iterations = CONFIG['iterations_per_start_page']

# Define the baseline prompt that is always used
BASELINE_PROMPT = """
You are a Large Language Model acting as a Wikipedia navigator. 
Your goal is to reach the TARGET article by selecting every time a link.

Rules:
- Read the CURRENT PAGE CONTENT to understand the topic and available connections
- Choose ONE link that brings you towards the target's webpage
- You MUST reply in JSON format with TWO fields:
  * "link": the exact link name from the available options
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

def get_lm_studio_model_name():
    """
    Retrieve the current model name from LM Studio API.
    Returns the model name or 'lm_studio' as fallback.
    """
    try:
        # LM Studio's /v1/models endpoint returns info about loaded models
        models_url = CONFIG['lm_studio_url'].replace('/v1/chat/completions', '/v1/models')
        response = requests.get(models_url, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        # Get the first available model
        if 'data' in result and len(result['data']) > 0:
            model_name = result['data'][0].get('id', 'lm_studio')
            print(f"Detected LM Studio model: {model_name}")
            return model_name
        else:
            print("No model found in LM Studio response, using default")
            return 'lm_studio'
    except Exception as e:
        print(f"Could not retrieve model name from LM Studio: {e}")
        return 'lm_studio'

def call_lm_studio(messages, temperature=0.3):
    """Call LM Studio API with the given messages and temperature setting"""
    
    # Prepare the API request payload
    payload = {
        "messages": messages,
        "temperature": temperature,
    }
    
    try:
        # Send request to LM Studio with generous timeout for slow models
        response = requests.post(CONFIG['lm_studio_url'], json=payload, timeout=1200)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except requests.exceptions.RequestException as e:
        print(f"Error calling LM Studio: {e}")
        return None

def get_personality_prompt(personality_name="baseline"):
    """
    Get the system prompt by combining the baseline with personality traits.
    The baseline instructions are always included, and personality traits are added on top.
    Valid personality names: 'baseline', 'busybody', 'hunter', 'dancer'
    """
    traits = PERSONALITY_TRAITS.get(personality_name, PERSONALITY_TRAITS["baseline"])
    
    # Combine baseline + personality traits
    if traits:
        return BASELINE_PROMPT + "\n" + traits
    else:
        return BASELINE_PROMPT

def get_wikipedia_page(title):
    """
    Load a Wikipedia page by title.
    Handles disambiguation errors, redirects, and missing pages.
    """
    try:
        # Try to get the page without suggestions
        page = wikipedia.page(title, auto_suggest=False)
        return page
    except wikipedia.exceptions.PageError:
        # If not found, try with auto-suggestions enabled
        try:
            print(f'Page "{title}" not found, trying with suggestions...')
            page = wikipedia.page(title, auto_suggest=True)
            print(f'Found alternative: {page.title}')
            return page
        except Exception as e:
            print(f'Still not found: {title}')
            return None
    except wikipedia.exceptions.DisambiguationError as e:
        # Handle disambiguation pages by selecting the first option
        print(f'Disambiguation for {title}, using {e.options[0]}')
        try:
            return wikipedia.page(e.options[0])
        except Exception:
            return None
    except Exception as e:
        print(f'Error loading page {title}: {e}')
        return None
    
def get_page_links(page, exclude_keywords=None):
    """Extract links from a Wikipedia page in order of appearance."""
    if not page:
        return []
    
    # Define keywords for non-content pages to exclude
    exclude = exclude_keywords or ['Category:', 'Template:', 'File:', 'Portal:', 
                                   'Help:', 'Wikipedia:', 'Talk:', 'Special:']
    
    try:
        from urllib.parse import unquote
        # Parse HTML to extract links in their actual order
        soup = BeautifulSoup(page.html(), 'html.parser')
        content = soup.find('div', {'id': 'mw-content-text'})
        
        # Fallback to default links if HTML parsing fails
        if not content:
            return [l for l in page.links if not any(kw in l for kw in exclude)]
        
        seen = set()
        links = []
        
        # Extract all wiki links from content
        for a in content.find_all('a', href=True):
            href = a.get('href', '')
            if not href.startswith('/wiki/'):
                continue
            
            # Decode URL and convert to readable title
            title = unquote(href.replace('/wiki/', '').replace('_', ' '))
            
            # Skip excluded pages and duplicates
            if any(kw in title for kw in exclude) or title in seen:
                continue
            
            seen.add(title)
            links.append(title)
        
        return links
    
    except Exception as e:
        print(f"Error parsing HTML: {e}, using fallback")
        return [l for l in page.links if not any(kw in l for kw in exclude)]

def fuzzy_similarity(s1, s2):
    """
    Calculate fuzzy similarity between two strings using SequenceMatcher.
    Returns a value between 0 and 1, where 1 is an exact match.
    """
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

def parse_model_choice(response, available_links, target_article=None):
    """
    Parse the model's response and match it to an available link.
    Expects JSON format with 'link' and 'reason' fields.
    Uses exact matching first, then fuzzy matching as fallback.
    Detects hallucinations when the model suggests non-existent links.
    Returns: (chosen_link, metadata) where metadata includes the reason.
    """
    if not response:
        return None, None
    
    # Initialize metadata for tracking match quality
    metadata = {"fallback_used": False, "match_type": None, "similarity_score": 0.0, "reason": ""}
    
    # Try to parse JSON response
    chosen = None
    reason = ""
    
    try:
        # Attempt to parse JSON response
        response_data = json.loads(response.strip())
        chosen = response_data.get("link", "").strip()
        reason = response_data.get("reason", "").strip()
    except json.JSONDecodeError:
        # Fallback: treat entire response as link name (backward compatibility)
        print(f"JSON parsing failed, treating response as plain text link")
        chosen = response.strip().strip('"\'.,:;!?()[]{}')
        reason = "No reason provided (non-JSON response)"
    
    # Store the reason in metadata
    metadata["reason"] = reason
    
    if not chosen:
        return None, None

    # Try exact match first (case-insensitive)
    for link in available_links:
        if link.lower() == chosen.lower():
            metadata.update({"match_type": "exact_match", "similarity_score": 1.0})
            return link, metadata

    # If no exact match, try fuzzy matching
    threshold = CONFIG.get('fuzzy_match_threshold', 0.75)
    best_link, best_score = None, 0.0
    
    for link in available_links:
        score = fuzzy_similarity(chosen, link)
        if score > best_score:
            best_score, best_link = score, link
    
    # Accept fuzzy match if above threshold
    if best_score >= threshold:
        metadata.update({"fallback_used": True, "match_type": "fuzzy_match", "similarity_score": best_score})
        print(f"FALLBACK: Fuzzy match ({best_score:.2f}) - '{chosen}' -> '{best_link}'")
        return best_link, metadata

    # No match found - model hallucinated a link
    metadata.update({"fallback_used": True, "match_type": "no_match_hallucination", "similarity_score": best_score})
    print(f"HALLUCINATION: No match for '{chosen}' (best: {best_score:.2f})")
    return None, metadata

def run_navigation_experiment(start_article, target_article, temperature=0.3, personality="baseline"):
    """
    Run a single Wikipedia navigation experiment from start to target article.
    The LLM navigates by selecting links at each step until reaching the target or getting stuck.
    Args:
        start_article: Starting Wikipedia page
        target_article: Target Wikipedia page to reach
        temperature: LLM temperature setting (0.0-2.0)
        personality: LLM personality type
    """
    
    # Initialize navigation state
    current, path, visited = start_article, [start_article], {start_article.lower()}
    success, detailed_steps = False, []
    
    # Initialize statistics for tracking fallback and hallucination behavior
    fallback_stats = {
        "total_fallbacks": 0, "fuzzy_match_fallbacks": 0, "total_hallucinations": 0,
        "fallback_rate": 0.0, "avg_similarity_score": 0.0, "fallback_success_rate": 0.0,
        "steps_with_fallback": [], "successful_fallback_steps": []
    }
    
    # Loop detection to prevent getting stuck in cycles
    max_loop_repetitions = CONFIG.get('max_loop_repetitions', 3)
    recent_transitions = []
    loop_detected = False
    
    # Get the system prompt for the selected personality
    system_prompt = get_personality_prompt(personality)
    
    step = 0
    # Main navigation loop
    while True:
        
        # Check if we've reached the target
        if current.lower() == target_article.lower():
            success = True
            break
        
        # Load the current Wikipedia page
        page = get_wikipedia_page(current)
        if not page:
            break
        
        # Update current title (may differ due to redirects)
        current = page.title
        # Get available links, excluding already visited pages
        available = [l for l in get_page_links(page) if l.lower() not in visited]
        
        # Dead end - no more links to explore
        if not available:
            break
        
        # Create prompt with current context and ask LLM to choose next link
        prompt = create_navigation_prompt(current, target_article, available, path, page.content)
        response = call_lm_studio([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ], temperature=temperature)
        
        # LLM call failed
        if not response:
            break
        
        # Parse the LLM's choice and match it to an actual link
        chosen, metadata = parse_model_choice(response, available, target_article)
        
        # Track fallback statistics for analysis
        if metadata and metadata.get("fallback_used"):
            fallback_stats["total_fallbacks"] += 1
            fallback_stats["steps_with_fallback"].append(step)
            
            if metadata.get("match_type") == "fuzzy_match":
                fallback_stats["fuzzy_match_fallbacks"] += 1
            elif metadata.get("match_type") == "no_match_hallucination":
                fallback_stats["total_hallucinations"] += 1
        
        # Model hallucinated or failed to choose - end navigation
        if not chosen:
            break
        
        # Loop detection: track recent page transitions
        transition = (current.lower(), chosen.lower())
        recent_transitions.append(transition)

        # Keep only recent transitions for loop detection
        if len(recent_transitions) > max_loop_repetitions * 2:
            recent_transitions.pop(0)

        # Count how many times this transition occurred
        transition_count = recent_transitions.count(transition)
        
        # If same transition repeated too many times, stop (stuck in a loop)
        if transition_count >= max_loop_repetitions:
            loop_detected = True
            print(f"LOOP DETECTED: Transition '{current}' -> '{chosen}' repeated {transition_count} times")
            break
        
        # Record detailed information about this step for analysis
        detailed_steps.append({
            "step_number": step,
            "origin_page": current,
            "destination_page": chosen,
            "reason": metadata.get("reason", ""),
            "available_links_count": len(available),
            "total_visited": len(visited),
            "fallback_used": metadata.get("fallback_used", False),
            "match_type": metadata.get("match_type", "unknown"),
            "similarity_score": metadata.get("similarity_score", 0.0)
        })
        
        # Move to the chosen page
        current = chosen
        path.append(current)
        visited.add(current.lower())
        step += 1
    
    total_steps = len(path) - 1
    if total_steps > 0:
        fallback_stats["fallback_rate"] = (fallback_stats["total_fallbacks"] / total_steps) * 100
        
        sim_scores = [s["similarity_score"] for s in detailed_steps if s.get("similarity_score", 0) > 0]
        if sim_scores:
            fallback_stats["avg_similarity_score"] = sum(sim_scores) / len(sim_scores)
        
        if success and fallback_stats["total_fallbacks"] > 0:
            fallback_stats["successful_fallback_steps"] = fallback_stats["steps_with_fallback"][:]
            fallback_stats["fallback_success_rate"] = (
                len(fallback_stats["successful_fallback_steps"]) / fallback_stats["total_fallbacks"] * 100
            )
    
    return {
        "success": success, "steps": len(path) - 1, "path": path,
        "detailed_steps": detailed_steps, "fallback_statistics": fallback_stats,
        "start": start_article, "target": target_article,
        "model": CONFIG.get('model_name', 'unknown'),
        "personality": personality,
        "lm_studio_url": CONFIG['lm_studio_url'],
        "total_pages_visited": len(visited),
        "loop_detected": loop_detected
    }

# ===== MAIN EXECUTION =====
# Automatically detect the model name from LM Studio if not specified in config
if 'model_name' not in CONFIG or CONFIG['model_name'] == 'lm_studio':
    detected_model = get_lm_studio_model_name()
    CONFIG['model_name'] = detected_model
    # Update output directory with detected model name
    CONFIG['output_dir'] = detected_model.replace('/', '_').replace(':', '_').replace('\\', '_')
    print(f"Using model: {CONFIG['model_name']}")
    print(f"Output directory: {CONFIG['output_dir']}")

all_results = []

# Get list of personalities to test
personalities = CONFIG.get('personalities', ['baseline', 'busybody', 'hunter', 'dancer'])
total_iterations = len(start_pages) * len(temperatures) * len(personalities) * iterations

iteration_counter = 0

# Run experiments for each temperature setting
for temperature in temperatures:
    
    # Run experiments for each starting page
    for start_page in start_pages:
        
        # Run experiments for each personality
        for personality in personalities:
            
            # Run multiple iterations to gather statistics
            for i in tqdm(range(iterations), desc=f"{start_page} (T={temperature}, P={personality})", unit="iter"):
                iteration_counter += 1
                
                # Execute single navigation experiment
                result = run_navigation_experiment(start_page, CONFIG['target'], 
                                                  temperature=temperature, 
                                                  personality=personality)
                result.update({
                    'config_start_page': start_page, 
                    'temperature': temperature, 
                    'personality': personality,
                    'iteration_in_group': i + 1
                })
                all_results.append(result)

                # Save individual result to file
                temp_str = str(temperature).replace('.', '_')
                output_dir = os.path.join(CONFIG['output_dir'], 
                                         f"temp_{temp_str}_personality_{personality}", 
                                         start_page.replace(' ', '_'))
                os.makedirs(output_dir, exist_ok=True)
                
                filename = os.path.join(output_dir, f"result_{i+1:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

# ===== SAVE ALL RESULTS =====
# Save comprehensive summary of all experiments
summary_file = os.path.join(CONFIG['output_dir'], f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
os.makedirs(CONFIG['output_dir'], exist_ok=True)
with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\nCompleted {len(all_results)} iterations. Results saved to: {summary_file}")