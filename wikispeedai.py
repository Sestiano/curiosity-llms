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
from urllib.parse import unquote
import csv
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum


class Personality(Enum):
    """Enumeration of available personality types for navigation."""
    BASELINE = "baseline"
    BUSYBODY = "busybody"
    HUNTER = "hunter"
    DANCER = "dancer"
    
    @property
    def traits(self) -> str:
        """Return the trait description for this personality."""
        return PERSONALITY_TRAITS[self.value]


class Temperature(Enum):
    """Predefined temperature values for LLM experimentation."""
    LOW = 0.3
    MEDIUM = 0.7
    HIGH = 1.0
    VERY_HIGH = 1.5
    
    @classmethod
    def from_value(cls, value: float) -> 'Temperature':
        """Find the closest predefined temperature or create custom."""
        for temp in cls:
            if abs(temp.value - value) < 0.01:
                return temp
        # If custom value, return it as float (not enum)
        return value


# Personality traits dictionary (used by Enum)
PERSONALITY_TRAITS = {
    "baseline": "",
    
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


@dataclass
class NavigationStep:
    """Represents a single step in the navigation path."""
    step_number: int
    origin_page: str
    destination_page: str
    reason: str
    available_links_count: int
    total_visited: int
    fallback_used: bool = False
    match_type: str = "unknown"
    similarity_score: float = 0.0
    correction_attempts: int = 0
    available_links_map: Dict[str, str] = field(default_factory=dict)


@dataclass
class FallbackStatistics:
    """Statistics about fallback mechanisms used during navigation."""
    total_fallbacks: int = 0
    fuzzy_match_fallbacks: int = 0
    total_hallucinations: int = 0
    fallback_rate: float = 0.0
    avg_similarity_score: float = 0.0
    steps_with_fallback: List[int] = field(default_factory=list)
    successful_fallback_steps: List[int] = field(default_factory=list)
    fallback_success_rate: float = 0.0


@dataclass
class NavigationResult:
    """Complete result of a navigation experiment."""
    success: bool
    steps: int
    path: List[str]
    detailed_steps: List[NavigationStep]
    fallback_statistics: FallbackStatistics
    start: str
    target: str
    model: str
    personality: str
    lm_studio_url: str
    total_pages_visited: int
    loop_detected: bool
    error_type: Optional[str]
    config_start_page: str = ""
    temperature: float = 0.3
    iteration_in_group: int = 0


class WikiNavigator:
    """Wikipedia navigation experiment using LLM with different personalities and temperatures."""
    
    # Define the baseline prompt that is always used
    BASELINE_PROMPT = """
You are a Large Language Model acting as a Wikipedia navigator. 
Your goal is to reach the TARGET article by selecting every time a link.

Rules:

- Read the \"CURRENT PAGE CONTENT\" to understand the topic and available connections

- Choose ONE link that brings you towards the target's webpage

- You MUST reply in JSON format with TWO fields:
  * \"link\": the EXACT LINK name from the available options
  * \"reason\": a BRIEF explanation of why you chose this link
"""
    
    def __init__(self, config_file: str = 'config.json'):
        """Initialize the WikiNavigator with configuration."""
        wikipedia.set_lang('en')
        warnings.filterwarnings('ignore')
        
        self.config = self._load_config(config_file)
        self.all_available_links_set: Set[Tuple[str, str]] = set()
        self.all_chosen_links_set: Set[Tuple[str, str]] = set()
        
        self._setup_logging()
        
        logging.info(f"Using model: {self.config['model_name']}")
        logging.info(f"Output directory: {self.config['output_dir']}")
    
    def _setup_logging(self) -> None:
        """Configure logging with file and console handlers."""
        log_filename = f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
    
    def _get_lm_studio_model_name(self, lm_studio_url: str, timeout: int = 120) -> str:
        """Retrieve the current model name from LM Studio API."""
        try:
            models_url = lm_studio_url.replace('/v1/chat/completions', '/v1/models')
            response = requests.get(models_url, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            if 'data' in result and len(result['data']) > 0:
                model_name = result['data'][0].get('id', 'lm_studio')
                logging.info(f"Auto-detected LM Studio model: {model_name}")
                return model_name
            else:
                logging.warning("No model found in LM Studio response, using default 'lm_studio'")
                return 'lm_studio'
        except Exception as e:
            logging.error(f"Could not retrieve model name from LM Studio: {e}")
            logging.warning("Using default 'lm_studio' as model name")
            return 'lm_studio'
    
    def _load_config(self, config_file: str) -> Dict:
        """Load and validate configuration from config.json or use defaults"""
        defaults = {
            'start_pages': ['Vaccine', 'Albert Einstein'],
            'target': 'Renaissance',
            'iterations_per_start_page': 100,
            'temperatures': [Temperature.LOW.value, Temperature.VERY_HIGH.value],
            'personalities': [p.value for p in Personality],
            'lm_studio_url': "http://localhost:1234/v1/chat/completions",
            'fuzzy_match_threshold': 0.95,
            'max_loop_repetitions': 3,
            'max_correction_attempts': 1,
            'api_timeout': 1200,
            'model_detection_timeout': 120
        }
        
        config_path = Path(config_file)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            for key, value in defaults.items():
                config.setdefault(key, value)
        else:
            config = defaults

        if 'model_name' not in config or not config['model_name']:
            logging.info("Model name not specified in config, auto-detecting from LM Studio...")
            config['model_name'] = self._get_lm_studio_model_name(
                config.get('lm_studio_url', defaults['lm_studio_url']),
                timeout=config.get('model_detection_timeout', defaults['model_detection_timeout'])
            )
        else:
            logging.info(f"Using model name from config: {config['model_name']}")
        
        model_name = config['model_name']
        config['output_dir'] = model_name.replace('/', '_').replace(':', '_').replace('\\', '_')

        sp = config.get('start_pages', [config.get('start_page', 'Vaccine')])
        config['start_pages'] = [sp] if isinstance(sp, str) else sp
        t = config.get('temperatures', [Temperature.LOW.value])
        config['temperatures'] = [t] if isinstance(t, (int, float)) else t
        
        # Validate personalities
        valid_personalities = [p.value for p in Personality]
        config['personalities'] = [p for p in config.get('personalities', ['baseline']) 
                                   if p in valid_personalities]
        if not config['personalities']:
            logging.warning("No valid personalities found in config, using baseline")
            config['personalities'] = [Personality.BASELINE.value]
        
        return config
    
    def create_navigation_prompt(self, current_article: str, target_article: str, 
                                links: List[str], path: List[str], 
                                page_content: Optional[str] = None) -> str:
        """Create a prompt for the LLM with current context and available links"""
        
        # Show recent navigation path for context
        recent_path = ' '.join(path[:-1]) if len(path) > 1 else path[0]

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
                      Remember to respond in JSON format with both \"link\" and \"reason\" fields.
                      """

        return prompt

    def create_disambiguation_prompt(self, original_choice: str, options: List[str], 
                                   target_article: str, path: List[str]) -> str:
        """Create a prompt for the LLM to resolve a disambiguation page."""
        # Show recent navigation path for context
        recent_path = ' '.join(path[:-1]) if len(path) > 1 else path[0]

        # Shuffle the options to avoid position bias (e.g., model preferring first option)
        shuffled_options = options.copy()
        random.shuffle(shuffled_options)
        options_list = '\n'.join(shuffled_options)

        prompt = f"""Your previous choice "{original_choice}" was AMBIGUOS.
                     Your PREVIOUS PAGE was: {recent_path}

                     To continue, you must choose ONE link that brings you towards the target's webpage: "{target_article}".
                     AVAILABLE OPTIONS:
                     {options_list}
                     
                     TASK: Choose the BEST option from the list to reach the target. Respond in JSON format with both \"link\" and \"reason\" fields.
                    """
        return prompt

    def create_correction_prompt(self, invalid_choice: str, target_article: str, 
                               links: List[str], path: List[str]) -> str:
        """Create a prompt for the LLM to correct an invalid link selection."""
        # Show recent navigation path for context
        recent_path = ' '.join(path[:-1]) if len(path) > 1 else path[0]

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
    
    def call_lm_studio(self, messages: List[Dict], temperature: float = 0.3) -> Optional[str]:
        """Call LM Studio API."""
        payload = {
            "model": self.config.get('model_name', 'local-model'),
            "messages": messages, 
            "temperature": temperature,
            "max_tokens": -1,  # -1 means use maximum available
            "stream": False
        }
        try:
            response = requests.post(
                self.config['lm_studio_url'], 
                json=payload, 
                timeout=self.config.get('api_timeout', 1200)
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error calling LM Studio: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logging.error(f"Response content: {e.response.text}")
            return None
    
    def get_personality_prompt(self, personality_name: str = "baseline") -> str:
        """Get the combined system prompt."""
        traits = self.PERSONALITY_TRAITS.get(personality_name, "")
        return f"{self.BASELINE_PROMPT}\n{traits}" if traits else self.BASELINE_PROMPT
    
    def get_wikipedia_page(self, title: str) -> Tuple[str, any]:
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
            logging.info(f"'{title}' is a disambiguation page. Returning options.")
            return 'disambiguation', e.options
        except wikipedia.exceptions.PageError:
            logging.info(f"Page '{title}' not found. Trying with auto-suggest...")
            try:
                page = wikipedia.page(title, auto_suggest=True, redirect=True)
                logging.info(f"Found alternative page: '{page.title}'")
                return 'success', page
            except wikipedia.exceptions.DisambiguationError as e:
                logging.warning(f"Suggestion for '{title}' also led to a disambiguation.")
                return 'disambiguation', e.options
            except wikipedia.exceptions.PageError:
                logging.error(f"No page found for '{title}', even with suggestions.")
                return 'not_found', None
            except Exception as e:
                logging.error(f"Unexpected error during auto-suggest for '{title}': {e}")
                return 'error', str(e)
        except Exception as e:
            logging.error(f"Unexpected error while loading '{title}': {e}")
            return 'error', str(e)
    
    def get_page_links(self, page, exclude_keywords: Optional[List[str]] = None) -> List[Tuple[str, str]]:
        """
        Fetches a Wikipedia page HTML and extracts internal links, returning them
        as a list of (anchor_text, page_title) tuples.
        """
        if not page:
            return []

        exclude = exclude_keywords or [
            'Category:', 'Template:', 'File:', 'Portal:', 'Help:', 'Wikipedia:',
            'Talk:', 'Special:', 'Template talk:'
        ]
        
        try:
            soup = BeautifulSoup(page.html(), 'html.parser')
            content = soup.find('div', {'id': 'mw-content-text'}) or soup.find('div', {'class': 'mw-parser-output'})
            
            if not content:
                logging.warning(f"Could not find the main content area for page '{page.title}'.")
                return []

            seen_anchors, links = set(), []
            for a in content.find_all('a', href=True):
                href = a.get('href', '')
                
                if href.startswith('/wiki/'):
                    page_title = unquote(href.replace('/wiki/', '')).replace('_', ' ')

                    if '#' not in page_title and not any(kw in page_title for kw in exclude):
                        anchor_text = a.get_text(strip=True)
                        
                        if anchor_text and not (anchor_text.startswith('[') and anchor_text.endswith(']')) and anchor_text not in seen_anchors:
                            seen_anchors.add(anchor_text)
                            links.append((anchor_text, page_title))
            return links
            
        except Exception as e:
            logging.error(f"An error occurred while parsing HTML for links in page '{page.title}': {e}")
            return []
    
    @staticmethod
    def fuzzy_similarity(s1: str, s2: str) -> float:
        """Calculate fuzzy similarity between two strings."""
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
    
    def parse_model_choice(self, response: Optional[str], 
                          available_links: List[str]) -> Tuple[Optional[str], Dict]:
        """Parse the model's response and match it to an available link."""
        if not response: 
            return None, {}
        
        metadata = {"fallback_used": False, "match_type": "unknown", "similarity_score": 0.0, "reason": "", "raw_choice": ""}
        
        try:
            response_data = json.loads(response.strip())
            chosen_text = response_data.get("link", "").strip()
            metadata["reason"] = response_data.get("reason", "").strip()
        except json.JSONDecodeError:
            logging.warning(f"JSON parsing failed, treating response as plain text link.")
            chosen_text = response.strip().strip('"\',:;!?()[]{}')
            metadata["reason"] = "No reason provided (non-JSON response)"
        
        metadata["raw_choice"] = chosen_text
        if not chosen_text: 
            return None, metadata

        for link in available_links:
            if link.lower() == chosen_text.lower():
                metadata.update({"match_type": "exact_match", "similarity_score": 1.0})
                return link, metadata

        threshold = self.config.get('fuzzy_match_threshold', 0.95)
        best_link, best_score = max(
            ((link, self.fuzzy_similarity(chosen_text, link)) for link in available_links), 
            key=lambda item: item[1], 
            default=(None, 0.0)
        )
        
        if best_score >= threshold:
            metadata.update({"fallback_used": True, "match_type": "fuzzy_match", "similarity_score": best_score})
            logging.info(f"FALLBACK: Fuzzy match ({best_score:.2f}) - '{chosen_text}' -> '{best_link}'")
            return best_link, metadata

        metadata.update({"fallback_used": True, "match_type": "no_match_hallucination", "similarity_score": best_score})
        logging.warning(f"HALLUCINATION: No match for '{chosen_text}' (best: {best_score:.2f})")
        return None, metadata
    
    def _handle_page_loading(self, current_page_title: str, target_article: str, 
                           path: List[str], system_prompt: str, 
                           temperature: float, max_attempts: int) -> Tuple[any, Optional[str]]:
        """Handles page loading, including disambiguation resolution."""
        page, status = None, ''
        temp_current = current_page_title

        for _ in range(max_attempts):
            status, result = self.get_wikipedia_page(temp_current)
            
            if status == 'success':
                page = result
                break
            
            elif status == 'disambiguation':
                logging.info("Disambiguation found. Asking LLM to choose.")
                options = result
                prompt = self.create_disambiguation_prompt(temp_current, options, target_article, path)
                response = self.call_lm_studio([{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}], temperature)
                
                chosen_option, _ = self.parse_model_choice(response, options)
                
                if chosen_option:
                    temp_current = chosen_option
                else:
                    logging.error("LLM failed to resolve disambiguation. Halting.")
                    return None, "disambiguation_resolution_failed"
            else: # 'not_found' or 'error'
                return None, f"page_load_failed_{status}"
                
        if not page or status != 'success':
            return None, "page_load_unknown_error"
            
        return page, None
    
    def _get_llm_choice(self, page, target_article: str, path: List[str], 
                       available_anchors: List[str], system_prompt: str, 
                       temperature: float, max_attempts: int) -> Tuple[Optional[str], Dict, Optional[str]]:
        """Handles LLM choice, including self-correction loop."""
        chosen_anchor, metadata = None, {}
        
        for attempt in range(max_attempts):
            prompt = self.create_navigation_prompt(page.title, target_article, available_anchors, path, page.content) if attempt == 0 else self.create_correction_prompt(metadata.get("raw_choice"), target_article, available_anchors, path)
            
            response = self.call_lm_studio([{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}], temperature)
            if not response:
                return None, {}, "api_call_failed"
            
            chosen_anchor, metadata = self.parse_model_choice(response, available_anchors)
            metadata["correction_attempts"] = attempt
            
            if chosen_anchor:
                return chosen_anchor, metadata, None
                
        logging.warning("LLM failed to provide a valid link after retries. Halting.")
        return None, metadata, "hallucination_max_retries_exceeded"
    
    def run_navigation_experiment(self, start_article: str, target_article: str, 
                                 temperature: float = 0.3, 
                                 personality: str = "baseline") -> NavigationResult:
        """Run a single navigation experiment."""
        current, path, visited = start_article, [start_article], {start_article.lower()}
        success, detailed_steps, loop_detected = False, [], False
        error_type = None

        fallback_stats = FallbackStatistics()
        
        max_loop_repetitions = self.config.get('max_loop_repetitions', 3)
        max_correction_attempts = self.config.get('max_correction_attempts', 2)
        recent_transitions = []
        system_prompt = self.get_personality_prompt(personality)
        
        step = 0
        while step < 50:  # Max steps to prevent infinite runs
            if current.lower() == target_article.lower():
                success = True
                break
            
            # --- 1. LOAD PAGE ---
            page, error_type = self._handle_page_loading(current, target_article, path, system_prompt, temperature, max_correction_attempts)
            if error_type:
                break
            current = page.title  # Update to actual title after redirects

            # --- 2. EXTRACT LINKS ---
            all_links = self.get_page_links(page)
            available_links_map = {anchor: title for anchor, title in all_links if title.lower() not in visited}
            
            self.all_available_links_set.update([(title, anchor) for anchor, title in available_links_map.items()])

            if not available_links_map:
                logging.warning("Dead end reached.")
                error_type = "dead_end_no_links"
                break
            
            available_anchors = list(available_links_map.keys())
            
            # --- 3. GET LLM CHOICE ---
            chosen_anchor, metadata, error_type = self._get_llm_choice(page, target_article, path, available_anchors, system_prompt, temperature, max_correction_attempts)
            
            # --- 4. PROCESS RESULT ---
            if metadata.get("fallback_used"):
                fallback_stats.total_fallbacks += 1
                fallback_stats.steps_with_fallback.append(step)
                if metadata["match_type"] == "fuzzy_match": 
                    fallback_stats.fuzzy_match_fallbacks += 1
                elif metadata["match_type"] == "no_match_hallucination": 
                    fallback_stats.total_hallucinations += 1
            
            if error_type:
                break
                
            # --- 5. LOOP DETECTION & STATE UPDATE ---
            chosen_title = available_links_map[chosen_anchor]

            self.all_chosen_links_set.add((chosen_title, chosen_anchor))

            transition = (current.lower(), chosen_title.lower())
            recent_transitions.append(transition)
            if len(recent_transitions) > max_loop_repetitions * 2: 
                recent_transitions.pop(0)
            
            if recent_transitions.count(transition) >= max_loop_repetitions:
                loop_detected = True
                error_type = "loop_detected"
                logging.warning(f"LOOP DETECTED: Transition '{current}' -> '{chosen_title}' repeated.")
                break
                
            detailed_steps.append(NavigationStep(
                step_number=step,
                origin_page=current,
                destination_page=chosen_title,
                reason=metadata.get("reason", ""),
                available_links_count=len(available_anchors),
                total_visited=len(visited),
                fallback_used=metadata.get("fallback_used", False),
                match_type=metadata.get("match_type", "unknown"),
                similarity_score=metadata.get("similarity_score", 0.0),
                correction_attempts=metadata.get("correction_attempts", 0),
                available_links_map=available_links_map
            ))
            
            current = chosen_title
            path.append(current)
            visited.add(current.lower())
            step += 1

        # --- 6. FINAL STATISTICS CALCULATION ---
        total_steps = len(detailed_steps)
        if total_steps > 0:
            fallback_stats.fallback_rate = (fallback_stats.total_fallbacks / total_steps) * 100
            
            sim_scores = [s.similarity_score for s in detailed_steps if s.similarity_score > 0]
            if sim_scores:
                fallback_stats.avg_similarity_score = sum(sim_scores) / len(sim_scores)
            
            if success:
                fallback_stats.successful_fallback_steps = [s.step_number for s in detailed_steps if s.fallback_used]
                if fallback_stats.total_fallbacks > 0:
                    fallback_stats.fallback_success_rate = (len(fallback_stats.successful_fallback_steps) / fallback_stats.total_fallbacks) * 100

        return NavigationResult(
            success=success,
            steps=len(path) - 1,
            path=path,
            detailed_steps=detailed_steps,
            fallback_statistics=fallback_stats,
            start=start_article,
            target=target_article,
            model=self.config.get('model_name', 'unknown'),
            personality=personality,
            lm_studio_url=self.config['lm_studio_url'],
            total_pages_visited=len(visited),
            loop_detected=loop_detected,
            error_type=error_type
        )
    
    def save_individual_result(self, result: NavigationResult, start_page: str, 
                             temp: float, personality: str, iteration: int) -> None:
        """Saves a single experiment result to a structured directory."""
        temp_str = str(temp).replace('.', '_')
        output_dir = Path(self.config['output_dir']) / f"temp_{temp_str}_personality_{personality}" / start_page.replace(' ', '_')
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = output_dir / f"result_{iteration+1:03d}.json"
        
        # Convert dataclasses to dict for JSON serialization
        result_dict = asdict(result)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
    
    def save_csv_reports(self) -> None:
        """Write summary CSV files for available and chosen links."""
        logging.info("\nWriting summary CSV files...")
        
        self._write_csv_report(
            'report_all_available_links.csv',
            sorted(self.all_available_links_set),
            "available links"
        )
        
        self._write_csv_report(
            'report_all_chosen_links.csv',
            sorted(self.all_chosen_links_set),
            "chosen links"
        )
    
    def _write_csv_report(self, filename: str, data: List[Tuple[str, str]], 
                         description: str) -> None:
        """Helper method to write CSV reports with error handling."""
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['page_title', 'anchor_text'])
                writer.writerows(data)
            logging.info(f"Saved {len(data)} unique {description} to {filename}")
        except Exception as e:
            logging.error(f"Error writing {filename}: {e}")
    
    def run_experiments(self) -> List[NavigationResult]:
        """Run all experiments based on configuration."""
        all_results = []
        personalities = self.config.get('personalities', ['baseline'])
        start_pages = self.config['start_pages']
        temperatures = self.config['temperatures']
        iterations = self.config['iterations_per_start_page']

        for temp in temperatures:
            for start_page in start_pages:
                for personality in personalities:
                    for i in tqdm(range(iterations), desc=f"{start_page} (T={temp}, P={personality})"):
                        # Set random seed for reproducibility
                        random.seed(f"{start_page}_{temp}_{personality}_{i}")
                        
                        result = self.run_navigation_experiment(
                            start_page, 
                            self.config['target'], 
                            temp, 
                            personality
                        )
                        
                        # Update result with experiment metadata
                        result.config_start_page = start_page
                        result.temperature = temp
                        result.iteration_in_group = i + 1
                        
                        all_results.append(result)

                        # Save individual result IMMEDIATELY after each iteration
                        self.save_individual_result(result, start_page, temp, personality, i)

        # Write summary CSV files
        self.save_csv_reports()

        logging.info(f"\nCompleted {len(all_results)} iterations. Individual results saved in: {self.config['output_dir']}")
        return all_results


if __name__ == '__main__':
    navigator = WikiNavigator()
    navigator.run_experiments()