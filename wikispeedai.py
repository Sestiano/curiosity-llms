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


class LLMBackend(Enum):
    """Enumeration of supported LLM backends."""
    LM_STUDIO = "lm_studio"
    OLLAMA = "ollama"


class Personality(Enum):
    """Enumeration of available personality types for navigation."""
    BASELINE = "baseline"
    BUSYBODY = "busybody"
    HUNTER = "hunter"
    DANCER = "dancer"


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
        return value


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
    backend: str
    backend_url: str
    total_pages_visited: int
    loop_detected: bool
    error_type: Optional[str]
    config_start_page: str = ""
    temperature: float = 0.3
    iteration_in_group: int = 0


class WikiNavigator:
    """Wikipedia navigation experiment using LLM with different personalities and temperatures."""
    
    def __init__(self, config_file: str = 'config.json'):
        """Initialize the WikiNavigator with configuration."""
        wikipedia.set_lang('en')
        warnings.filterwarnings('ignore')
        
        self.config = self._load_config(config_file)
        self.all_available_links_set: Set[Tuple[str, str]] = set()
        self.all_chosen_links_set: Set[Tuple[str, str]] = set()
        
        self.cache_dir = Path(__file__).parent / "wiki_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        self._setup_logging()
        
        logging.info(f"Using backend: {self.config['backend']}")
        logging.info(f"Using model: {self.config['model_name']}")
        logging.info(f"Output directory: {self.config['output_dir']}")
    
    def _setup_logging(self) -> None:
        """Configure logging with file and console handlers."""
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        log_filename = log_dir / f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        detailed_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        simple_formatter = logging.Formatter('%(message)s')

        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(detailed_formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(simple_formatter)

        logger = logging.getLogger()
        if logger.hasHandlers():
            logger.handlers.clear()

        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    def _console_print(self, message: str) -> None:
        """Print directly to console bypassing logging filters."""
        print(message)
    
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
    
    def _get_ollama_model_name(self, ollama_url: str, timeout: int = 120) -> str:
        """Retrieve the current model name from Ollama API."""
        try:
            # Get the base URL (remove /api/chat or /api/generate)
            base_url = ollama_url.rsplit('/api/', 1)[0]
            tags_url = f"{base_url}/api/tags"
            
            response = requests.get(tags_url, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            
            if 'models' in result and len(result['models']) > 0:
                model_name = result['models'][0].get('name', 'ollama')
                logging.info(f"Auto-detected Ollama model: {model_name}")
                return model_name
            else:
                logging.warning("No model found in Ollama response, using default 'ollama'")
                return 'ollama'
        except Exception as e:
            logging.error(f"Could not retrieve model name from Ollama: {e}")
            logging.warning("Using default 'ollama' as model name")
            return 'ollama'
    
    def _load_config(self, config_file: str) -> Dict:
        """Load and validate configuration from config.json."""
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Determine backend
        backend = config.get('backend', 'lm_studio').lower()
        if backend not in ['lm_studio', 'ollama']:
            logging.warning(f"Invalid backend '{backend}', defaulting to 'lm_studio'")
            backend = 'lm_studio'
        config['backend'] = backend

        # Set appropriate URL based on backend
        if backend == 'lm_studio':
            config['backend_url'] = config.get('lm_studio_url', "http://localhost:1234/v1/chat/completions")
            if 'model_name' not in config or not config['model_name']:
                logging.info("Model name not specified, auto-detecting from LM Studio...")
                config['model_name'] = self._get_lm_studio_model_name(
                    config['backend_url'],
                    timeout=config.get('model_detection_timeout', 120)
                )
        else:  # ollama
            config['backend_url'] = config.get('ollama_url', "http://localhost:11434/api/chat")
            if 'model_name' not in config or not config['model_name']:
                logging.info("Model name not specified, auto-detecting from Ollama...")
                config['model_name'] = self._get_ollama_model_name(
                    config['backend_url'],
                    timeout=config.get('model_detection_timeout', 120)
                )
        
        model_name = config['model_name']
        config['output_dir'] = model_name.replace('/', '_').replace(':', '_').replace('\\', '_')

        sp = config.get('start_pages', ['Vaccine'])
        config['start_pages'] = [sp] if isinstance(sp, str) else sp
        t = config.get('temperatures', [0.3])
        config['temperatures'] = [t] if isinstance(t, (int, float)) else t
        
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
        recent_path = ' '.join(path[:-1]) if len(path) > 1 else path[0]
        links_list = '\n'.join(links)
        content_text = page_content if page_content else "[no content available]"
        
        prompt_template = self.config['prompts']['navigation']
        return prompt_template.format(
            target_article=target_article,
            current_article=current_article,
            recent_path=recent_path,
            content_text=content_text,
            links_list=links_list
        )

    def create_disambiguation_prompt(self, original_choice: str, options: List[str],
                                     target_article: str, path: List[str]) -> str:
        """Create a prompt for the LLM to resolve a disambiguation page."""
        recent_path = ' '.join(path[:-1]) if len(path) > 1 else path[0]
        shuffled_options = options.copy()
        random.shuffle(shuffled_options)
        options_list = '\n'.join(shuffled_options)

        prompt_template = self.config['prompts']['disambiguation']
        return prompt_template.format(
            original_choice=original_choice,
            recent_path=recent_path,
            target_article=target_article,
            options_list=options_list
        )

    def create_correction_prompt(self, invalid_choice: str, target_article: str,
                                 links: List[str], path: List[str]) -> str:
        """Create a prompt for the LLM to correct an invalid link selection."""
        recent_path = ' '.join(path[:-1]) if len(path) > 1 else path[0]
        shuffled_links = links.copy()
        random.shuffle(shuffled_links)
        links_list = '\n'.join(shuffled_links)

        prompt_template = self.config['prompts']['correction']
        return prompt_template.format(
            invalid_choice=invalid_choice,
            target_article=target_article,
            recent_path=recent_path,
            links_list=links_list
        )

    def call_llm(self, messages: List[Dict], temperature: float = 0.3) -> Optional[str]:
        """Call LLM API (LM Studio or Ollama based on configuration)."""
        if self.config['backend'] == 'lm_studio':
            return self._call_lm_studio(messages, temperature)
        else:
            return self._call_ollama(messages, temperature)

    def _call_lm_studio(self, messages: List[Dict], temperature: float = 0.3) -> Optional[str]:
        """Call LM Studio API."""
        payload = {
            "model": self.config.get('model_name', 'local-model'),
            "messages": messages, 
            "temperature": temperature,
            "max_tokens": -1,
            "stream": False
        }
        try:
            response = requests.post(
                self.config['backend_url'], 
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

    def _call_ollama(self, messages: List[Dict], temperature: float = 0.3) -> Optional[str]:
        """Call Ollama API."""
        payload = {
            "model": self.config.get('model_name', 'llama2'),
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        try:
            response = requests.post(
                self.config['backend_url'],
                json=payload,
                timeout=self.config.get('api_timeout', 1200)
            )
            response.raise_for_status()
            result = response.json()
            return result['message']['content'].strip()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error calling Ollama: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logging.error(f"Response content: {e.response.text}")
            return None
    
    def get_personality_prompt(self, personality: str = "baseline") -> str:
        """Get the combined system prompt for a given personality."""
        traits = self.config.get('personality_traits', {}).get(personality)

        if traits is None:
            logging.warning(f"Invalid or missing personality '{personality}', using baseline")
            traits = self.config.get('personality_traits', {}).get('baseline', '')

        baseline_prompt = self.config['prompts']['baseline']
        return f"{baseline_prompt}\n{traits}" if traits else baseline_prompt
    
    def get_wikipedia_page(self, title: str) -> Tuple[str, any]:
        """Loads a Wikipedia page with error handling."""
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

    def _get_cached_page_html(self, page) -> str:
        """Loads page HTML from cache or fetches and saves it."""
        safe_title = "".join(c for c in page.title if c.isalnum() or c in (' ', '_')).rstrip()
        cache_file = self.cache_dir / f"{safe_title}.html"

        if cache_file.exists():
            logging.info(f"CACHE HIT: Loading '{page.title}' from cache.")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            logging.info(f"CACHE MISS: Fetching '{page.title}' from Wikipedia and caching.")
            try:
                html_content = page.html()
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                return html_content
            except Exception as e:
                logging.error(f"Could not fetch or cache page '{page.title}': {e}")
                return ""
    
    def get_page_links(self, page, exclude_keywords: Optional[List[str]] = None) -> Tuple[List[Tuple[str, str]], str]:
        """Fetches a Wikipedia page HTML from cache or web and extracts internal links and text content."""
        if not page:
            return [], ""

        exclude = exclude_keywords or [
            'Category:', 'Template:', 'File:', 'Portal:', 'Help:', 'Wikipedia:',
            'Talk:', 'Special:', 'Template talk:'
        ]
        
        try:
            html_content = self._get_cached_page_html(page)
            if not html_content:
                return [], ""

            soup = BeautifulSoup(html_content, 'html.parser')
            content = soup.find('div', {'id': 'mw-content-text'}) or soup.find('div', {'class': 'mw-parser-output'})
            
            if not content:
                logging.warning(f"Could not find the main content area for page '{page.title}'.")
                return [], ""

            page_text = content.get_text(strip=True, separator=' ')

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
            return links, page_text
            
        except Exception as e:
            logging.error(f"An error occurred while parsing HTML for links in page '{page.title}': {e}")
            return [], "" 
    
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
                response = self.call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}], temperature)
                
                chosen_option, _ = self.parse_model_choice(response, options)
                
                if chosen_option:
                    temp_current = chosen_option
                else:
                    logging.error("LLM failed to resolve disambiguation. Halting.")
                    return None, "disambiguation_resolution_failed"
            else:
                return None, f"page_load_failed_{status}"
                
        if not page or status != 'success':
            return None, "page_load_unknown_error"
            
        return page, None
    
    def _get_llm_choice(self, page_title: str, page_content: str, target_article: str, path: List[str], 
                       available_anchors: List[str], system_prompt: str, 
                       temperature: float, max_attempts: int) -> Tuple[Optional[str], Dict, Optional[str]]:
        """Handles LLM choice, including self-correction loop."""
        chosen_anchor, metadata = None, {}
        
        for attempt in range(max_attempts):
            prompt = self.create_navigation_prompt(page_title, target_article, available_anchors, path, page_content) if attempt == 0 else self.create_correction_prompt(metadata.get("raw_choice"), target_article, available_anchors, path)
            
            response = self.call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}], temperature)
            if not response:
                return None, {}, "api_call_failed"
            
            chosen_anchor, metadata = self.parse_model_choice(response, available_anchors)
            metadata["correction_attempts"] = attempt
            
            if chosen_anchor:
                return chosen_anchor, metadata, None
                
        logging.warning("LLM failed to provide a valid link after retries. Halting.")
        return None, metadata, "hallucination_max_retries_exceeded"
    
    def _calculate_final_statistics(self, detailed_steps: List[NavigationStep], success: bool) -> FallbackStatistics:
        """Calculate fallback and other statistics from the detailed steps of a run."""
        stats = FallbackStatistics()
        total_steps = len(detailed_steps)
        if total_steps == 0:
            return stats

        stats.steps_with_fallback = [s.step_number for s in detailed_steps if s.fallback_used]
        stats.total_fallbacks = len(stats.steps_with_fallback)
        stats.fuzzy_match_fallbacks = len([s for s in detailed_steps if s.match_type == "fuzzy_match"])
        stats.total_hallucinations = len([s for s in detailed_steps if s.match_type == "no_match_hallucination"])
        
        stats.fallback_rate = (stats.total_fallbacks / total_steps) * 100

        sim_scores = [s.similarity_score for s in detailed_steps if s.similarity_score > 0 and s.match_type == "fuzzy_match"]
        if sim_scores:
            stats.avg_similarity_score = sum(sim_scores) / len(sim_scores)

        if success:
            stats.successful_fallback_steps = stats.steps_with_fallback
            if stats.total_fallbacks > 0:
                stats.fallback_success_rate = (len(stats.successful_fallback_steps) / stats.total_fallbacks) * 100
        
        return stats

    def run_navigation_experiment(self, start_article: str, target_article: str, 
                                 temperature: float = 0.3, 
                                 personality: str = "baseline") -> NavigationResult:
        """Run a single navigation experiment."""
        start_time = datetime.now()
        
        current, path, visited = start_article, [start_article], {start_article.lower()}
        success, detailed_steps, loop_detected = False, [], False
        error_type = None

        max_loop_repetitions = self.config.get('max_loop_repetitions', 3)
        max_correction_attempts = self.config.get('max_correction_attempts', 2)
        recent_transitions = []
        system_prompt = self.get_personality_prompt(personality)
        
        step = 0
        while step < 50:
            if current.lower() == target_article.lower():
                success = True
                elapsed = (datetime.now() - start_time).total_seconds()
                self._console_print(f"  🎯 TARGET REACHED in {step} steps! ({elapsed:.1f}s)")
                break
            
            page, error_type = self._handle_page_loading(current, target_article, path, system_prompt, temperature, max_correction_attempts)
            if error_type:
                self._console_print(f"  ❌ ERROR: {error_type}")
                break
            current = page.title

            all_links, page_content = self.get_page_links(page)
            available_links_map = {anchor: title for anchor, title in all_links if title.lower() not in visited}
            
            self.all_available_links_set.update([(current, title) for anchor, title in available_links_map.items()])

            if not available_links_map:
                logging.warning("Dead end reached.")
                error_type = "dead_end_no_links"
                self._console_print(f"  ❌ DEAD END at '{current}'")
                break
            
            available_anchors = list(available_links_map.keys())
            
            chosen_anchor, metadata, error_type = self._get_llm_choice(
                page.title, page_content, target_article, path, 
                available_anchors, system_prompt, temperature, max_correction_attempts
            )
            
            if metadata.get("fallback_used"):
                if metadata["match_type"] == "fuzzy_match": 
                    self._console_print(f"  ⚠️  Step {step}: {current} → {available_links_map.get(chosen_anchor, '?')} (fuzzy: {metadata['similarity_score']:.2f})")
                elif metadata["match_type"] == "no_match_hallucination": 
                    self._console_print(f"  ⚠️  Step {step}: Hallucination detected")
            
            if error_type:
                self._console_print(f"  ❌ ERROR: {error_type}")
                break
                
            chosen_title = available_links_map[chosen_anchor]

            self.all_chosen_links_set.add((current, chosen_title))

            transition = (current.lower(), chosen_title.lower())
            recent_transitions.append(transition)
            if len(recent_transitions) > max_loop_repetitions * 2: 
                recent_transitions.pop(0)
            
            if recent_transitions.count(transition) >= max_loop_repetitions:
                loop_detected = True
                error_type = "loop_detected"
                logging.warning(f"LOOP DETECTED: Transition '{current}' -> '{chosen_title}' repeated.")
                self._console_print(f"  🔄 LOOP DETECTED: {current} ↔ {chosen_title}")
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

        elapsed_time = (datetime.now() - start_time).total_seconds()
        fallback_stats = self._calculate_final_statistics(detailed_steps, success)

        if success:
            self._console_print(f"  ✅ SUCCESS in {step} steps | Fallbacks: {fallback_stats.total_fallbacks} ({fallback_stats.fallback_rate:.1f}%) | Time: {elapsed_time:.1f}s")
        else:
            self._console_print(f"  ❌ FAILED: {error_type} after {step} steps | Time: {elapsed_time:.1f}s")

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
            backend=self.config['backend'],
            backend_url=self.config['backend_url'],
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

        detailed_csv_filename = output_dir / f"result_{iteration+1:03d}_possible_links.csv"
        self.save_detailed_links_csv(result.detailed_steps, detailed_csv_filename)

        result_dict = asdict(result)
        for step in result_dict.get('detailed_steps', []):
            if 'available_links_map' in step:
                del step['available_links_map']

        filename = output_dir / f"result_{iteration+1:03d}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

    def save_detailed_links_csv(self, detailed_steps: List[NavigationStep], filename: Path) -> None:
        """Saves the available links from each step to a detailed CSV file."""
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['step_number', 'origin_page', 'anchor_text', 'destination_page_title'])
                for step in detailed_steps:
                    for anchor, title in step.available_links_map.items():
                        writer.writerow([step.step_number, step.origin_page, anchor, title])
        except Exception as e:
            logging.error(f"Error writing detailed links CSV {filename}: {e}")
    
    def save_csv_reports(self) -> None:
        """Write summary CSV files for available and chosen links."""
        logging.info("\nWriting summary CSV files...")
        
        self._write_csv_report(
            'network_possible_edges.csv',
            sorted(list(self.all_available_links_set)),
            "possible navigation edges",
            ['source_page', 'destination_page']
        )
        
        self._write_csv_report(
            'network_explored_edges.csv',
            sorted(list(self.all_chosen_links_set)),
            "explored navigation edges",
            ['source_page', 'destination_page']
        )

    def save_summary_csv(self, all_results: List[NavigationResult]) -> None:
        """Saves a summary of all experiment runs to a single CSV file."""
        if not all_results:
            logging.warning("No results to save to summary CSV.")
            return

        output_dir = Path(self.config['output_dir'])
        summary_filename = output_dir / "experiment_summary.csv"
        
        headers = [
            "iteration_in_group", "success", "steps", "start", "target", 
            "temperature", "personality", "backend", "total_pages_visited", 
            "loop_detected", "error_type", "total_fallbacks", 
            "fallback_rate", "total_hallucinations", "avg_similarity_score",
            "path"
        ]

        try:
            with open(summary_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for result in all_results:
                    row = {
                        "iteration_in_group": result.iteration_in_group,
                        "success": result.success,
                        "steps": result.steps,
                        "start": result.start,
                        "target": result.target,
                        "temperature": result.temperature,
                        "personality": result.personality,
                        "backend": result.backend,
                        "total_pages_visited": result.total_pages_visited,
                        "loop_detected": result.loop_detected,
                        "error_type": result.error_type,
                        "total_fallbacks": result.fallback_statistics.total_fallbacks,
                        "fallback_rate": f"{result.fallback_statistics.fallback_rate:.2f}%",
                        "total_hallucinations": result.fallback_statistics.total_hallucinations,
                        "avg_similarity_score": f"{result.fallback_statistics.avg_similarity_score:.2f}",
                        "path": " -> ".join(result.path)
                    }
                    writer.writerow(row)
            logging.info(f"Summary of all experiments saved to {summary_filename}")
            self._console_print(f"Summary CSV saved to: {summary_filename}")
        except Exception as e:
            logging.error(f"Error writing summary CSV: {e}")
    
    def _write_csv_report(self, filename: str, data: List[Tuple[str, str]], 
                         description: str, headers: List[str]) -> None:
        """Helper method to write CSV reports with error handling."""
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
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
        
        total_experiments = len(temperatures) * len(start_pages) * len(personalities) * iterations
        self._console_print("\n" + "="*70)
        self._console_print(f"🚀 STARTING EXPERIMENTS")
        self._console_print(f"   Backend: {self.config['backend'].upper()}")
        self._console_print(f"   Total runs: {total_experiments}")
        self._console_print(f"   Start pages: {', '.join(start_pages)}")
        self._console_print(f"   Target: {self.config['target']}")
        self._console_print(f"   Temperatures: {temperatures}")
        self._console_print(f"   Personalities: {personalities}")
        self._console_print("="*70 + "\n")

        experiment_num = 0
        success_count = 0
        total_steps = []
        
        for temp in temperatures:
            for start_page in start_pages:
                for personality in personalities:
                    for i in tqdm(range(iterations), 
                                desc=f"{start_page[:20]} (T={temp}, P={personality[:4]})",
                                leave=False):
                        experiment_num += 1
                        
                        self._console_print(f"\n[{experiment_num}/{total_experiments}] {start_page} → {self.config['target']} | T={temp} | P={personality}")
                        
                        random.seed(f"{start_page}_{temp}_{personality}_{i}")
                        
                        result = self.run_navigation_experiment(
                            start_page, 
                            self.config['target'], 
                            temp, 
                            personality
                        )
                        
                        result.config_start_page = start_page
                        result.temperature = temp
                        result.iteration_in_group = i + 1
                        
                        all_results.append(result)
                        
                        if result.success:
                            success_count += 1
                            total_steps.append(result.steps)

                        self.save_individual_result(result, start_page, temp, personality, i)

        self.save_csv_reports()
        self.save_summary_csv(all_results)

        avg_steps = sum(total_steps) / len(total_steps) if total_steps else 0
        self._console_print("\n" + "="*70)
        self._console_print("📈 EXPERIMENT SUMMARY")
        self._console_print("="*70)
        self._console_print(f"Backend: {self.config['backend'].upper()}")
        self._console_print(f"Total runs: {len(all_results)}")
        self._console_print(f"Success rate: {success_count/len(all_results)*100:.1f}% ({success_count}/{len(all_results)})")
        self._console_print(f"Avg steps (successful): {avg_steps:.1f}")
        self._console_print(f"Results saved in: {self.config['output_dir']}")
        self._console_print("="*70 + "\n")

        logging.info(f"Completed {len(all_results)} iterations. Individual results saved in: {self.config['output_dir']}")
        return all_results


if __name__ == '__main__':
    navigator = WikiNavigator()
    navigator.run_experiments()