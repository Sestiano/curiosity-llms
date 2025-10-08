import wikipedia
import requests
import json
from datetime import datetime
import warnings
from tqdm import tqdm
import os
from bs4 import BeautifulSoup
from difflib import SequenceMatcher

# Set Wikipedia language to English and suppress warnings
wikipedia.set_lang('en')
warnings.filterwarnings('ignore')

# ===== CONFIGURATION =====
def load_config():
    """Load and validate configuration from config.json or use defaults"""
    # Default configuration values
    defaults = {
        'start_pages': ['Vaccine', 'Philosophy', 'Ancient Rome'],
        'target': 'Renaissance',
        'iterations_per_start_page': 100,
        'temperatures': [0.3, 1.5],
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

    # Normalize start_pages to always be a list
    sp = config.get('start_pages', [config.get('start_page', 'Vaccine')])
    config['start_pages'] = [sp] if isinstance(sp, str) else sp
    
    # Normalize temperatures to always be a list
    t = config.get('temperatures', [0.3])
    config['temperatures'] = [t] if isinstance(t, (int, float)) else t
    
    return config

# Load global configuration
CONFIG = load_config()
start_pages = CONFIG['start_pages']
temperatures = CONFIG['temperatures']
iterations = CONFIG['iterations_per_start_page']

# System prompt that guides the LLM navigation behavior
SYSTEM_PROMPT = """
You are a Large Language Model acting as a Wikipedia navigator. 
Your goal is to reach the TARGET article by selecting every time a link.

Rules:
- Read the CURRENT PAGE CONTENT to understand the topic and available connections
- Choose ONE link that brings you towards the target's webpage
- You MUST reply ONLY with the exact link name from the available options

"""

def create_navigation_prompt(current_article, target_article, links, path, page_content=None):
    """Create a prompt for the LLM with current context and available links"""
    
    # Show recent navigation path for context
    recent_path = ' '.join(path[:-1]) if len(path) > 1 else {start_page}

    # Format links as a list
    links_list = '\n'.join(links)
    
    # Include page content to help LLM make informed decisions
    content_text = page_content if page_content else "[No content available]"

    # Build the complete prompt with all necessary information
    prompt = f""" Your TARGET page of Wikipedia is: "{target_article}" 
                  Your CURRENT page of Wikipedia is: "{current_article}"
                  Your PREVIOUS PAGE was: {recent_path}
                  
                  Select the link by reading the CURRENT PAGE: {content_text}
                  
                  AVAILABLE LINKS: {links_list}
                  
                  TASK: Choose ONE link that brings you towards the target's webpage: "{target_article}".
                  """

    return prompt

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
                                   'Help:', 'Wikipedia:', 'Talk:', 'Special:', 
                                   'List of', 'Timeline of']
    
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
    Uses exact matching first, then fuzzy matching as fallback.
    Detects hallucinations when the model suggests non-existent links.
    """
    if not response:
        return None, None
    
    # Initialize metadata for tracking match quality
    metadata = {"fallback_used": False, "match_type": None, "similarity_score": 0.0}
    
    # Clean the response from common punctuation and formatting
    chosen = response.strip().strip('"\'.,:;!?()[]{}')
    
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

def run_navigation_experiment(start_article, target_article, temperature=0.3):
    """
    Run a single Wikipedia navigation experiment from start to target article.
    The LLM navigates by selecting links at each step until reaching the target or getting stuck.
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
            {"role": "system", "content": SYSTEM_PROMPT},
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
        "lm_studio_url": CONFIG['lm_studio_url'],
        "total_pages_visited": len(visited),
        "loop_detected": loop_detected
    }

# ===== MAIN EXECUTION =====
all_results = []
total_iterations = len(start_pages) * len(temperatures) * iterations

iteration_counter = 0

# Run experiments for each temperature setting
for temperature in temperatures:
    
    # Run experiments for each starting page
    for start_page in start_pages:
        
        # Run multiple iterations to gather statistics
        for i in tqdm(range(iterations), desc=f"{start_page} (T={temperature})", unit="iter"):
            iteration_counter += 1
            
            # Execute single navigation experiment
            result = run_navigation_experiment(start_page, CONFIG['target'], temperature=temperature)
            result.update({'config_start_page': start_page, 'temperature': temperature, 'iteration_in_group': i + 1})
            all_results.append(result)

            # Save individual result to file
            temp_str = str(temperature).replace('.', '_')
            output_dir = os.path.join(CONFIG['output_dir'], f"temp_{temp_str}", start_page.replace(' ', '_'))
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