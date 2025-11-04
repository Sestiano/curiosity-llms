from llama_cpp import Llama, LlamaGrammar
import os
import wikipedia
import json
import csv
import logging
from bs4 import BeautifulSoup
from urllib.parse import unquote
from pathlib import Path
from datetime import datetime

PROMPT = """You are a Large Language Model acting as a Wikipedia navigator.

We would like you to open a new tab on your browser and visit https://wikipedia.org/. 
We would like you to spend max {max_steps} steps while reading about whatever you want on Wikipedia.

For example, if you wanted to learn more about Philadelphia, you could go to the Philadelphia Wikipedia page.
You can read through the page. You can also click (choose) on links you find interesting or you can use the search bar to search for new topics.

There is no right or wrong way to do this. We are interested in what it is that people read about when they are not forced to read about anything in particular.

---

CURRENT PAGE: {current_page}

CONTENT PREVIEW:
{content_preview}

AVAILABLE LINKS:
{links}

---

Choose the link that interests you most and respond in JSON format: {{"link": "exact link name", "reason": "brief explanation of why you chose it"}}"""

class WikiNavigator:
    def __init__(self, config_file='config.json'):
        wikipedia.set_lang('en')
        self.config = self._load_config(config_file)
        self.cache_dir = Path("wiki_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.llm = None
        self.all_edges = []
        self.total_input_tokens = 0
        self._setup_logging()
        
    def _load_config(self, config_file):
        if not Path(config_file).exists():
            return {
                "model_path": "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                "start_page": "Wikipedia",
                "max_steps": 10,
                "runs": 5,
                "output_dir": "results"
            }
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        config.setdefault("model_path", "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
        config.setdefault("start_page", "Wikipedia")
        config.setdefault("max_steps", 10)
        config.setdefault("runs", 5)
        config.setdefault("output_dir", "results")
        return config
    
    def _setup_logging(self):
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logging.getLogger().handlers[1].setLevel(logging.WARNING)
    
    def _init_llm(self):
        if self.llm is None:
            model_path = os.path.join(os.path.dirname(__file__), self.config['model_path'])
            n_gpu = self.config.get('n_gpu_layers', -1)
            n_ctx = self.config.get('n_ctx', 8192)  # Default to 8192 if not specified
            logging.info(f"Loading model: {model_path}")
            logging.info(f"GPU layers: {n_gpu} (-1 = all layers on GPU, 0 = CPU only)")
            logging.info(f"Context window: {n_ctx} tokens")
            # Set verbose=False to reduce console output from llama.cpp
            self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu, verbose=False)
            logging.info("Model loaded successfully")
    
    def _reset_llm(self):
        """Resetta il contesto del modello per evitare di superare n_ctx."""
        if self.llm is not None:
            logging.info("Resetting model context to free memory")
            self.llm.reset()
            logging.info("Model context cleared")
    
    def _cache_path(self, page_title):
        safe_title = "".join(c for c in page_title if c.isalnum() or c in (' ', '_')).rstrip()
        return self.cache_dir / f"{safe_title}.txt"
    
    def _get_cached_content(self, page_title):
        cache_file = self._cache_path(page_title)
        if cache_file.exists():
            logging.info(f"Cache hit: {page_title}")
            return cache_file.read_text(encoding='utf-8')
        logging.info(f"Cache miss: {page_title}")
        return None
    
    def _save_cached_content(self, page_title, content):
        self._cache_path(page_title).write_text(content, encoding='utf-8')
        logging.info(f"Cached content for: {page_title}")
    
    def get_wikipedia_page(self, title):
        try:
            page = wikipedia.page(title, auto_suggest=False, redirect=True)
            logging.info(f"Loaded page: {page.title}")
            return 'success', page
        except wikipedia.exceptions.DisambiguationError as e:
            logging.info(f"Disambiguation page: {title}")
            return 'disambiguation', e.options
        except wikipedia.exceptions.PageError:
            logging.warning(f"Page not found: {title}, trying auto-suggest")
            try:
                page = wikipedia.page(title, auto_suggest=True, redirect=True)
                logging.info(f"Found alternative: {page.title}")
                return 'success', page
            except:
                logging.error(f"No page found for: {title}")
                return 'not_found', None
        except Exception as e:
            logging.error(f"Error loading page {title}: {e}")
            return 'error', str(e)
    
    def get_page_links(self, page_title):
        status, result = self.get_wikipedia_page(page_title)
        if status != 'success':
            return [], ""
        
        page = result
        content = self._get_cached_content(page.title) or page.content
        if content == page.content:
            self._save_cached_content(page.title, content)
        
        soup = BeautifulSoup(page.html(), 'html.parser')
        content_div = soup.find('div', {'id': 'mw-content-text'}) or soup.find('div', {'class': 'mw-parser-output'})
        if not content_div:
            return [], content
        
        exclude = ['Category:', 'Template:', 'File:', 'Portal:', 'Help:', 'Wikipedia:', 'Talk:', 'Special:', 'Template talk:']
        seen, links = set(), []
        
        for a in content_div.find_all('a', href=True):
            href = a.get('href', '')
            if not href.startswith('/wiki/'):
                continue
            
            title = unquote(href.replace('/wiki/', '')).replace('_', ' ')
            if '#' in title or any(kw in title for kw in exclude):
                continue
            
            text = a.get_text(strip=True)
            if text and not (text.startswith('[') and text.endswith(']')) and text not in seen:
                seen.add(text)
                links.append(text)
        
        return links, content
    
    def create_grammar(self, links):
        # Escape double quotes and backslashes in link titles for grammar
        escaped_links = [link.replace('\\', '\\\\').replace('"', '\\"') for link in links]
        alternatives = ' | '.join([f'"{link}"' for link in escaped_links])
        return LlamaGrammar.from_string(f'''root ::= object
object ::= "{{" ws "\\"link\\"" ws ":" ws link-value ws "," ws "\\"reason\\"" ws ":" ws reason-value ws "}}"
link-value ::= "\\"" ({alternatives}) "\\""
reason-value ::= "\\"" reason-text "\\""
reason-text ::= [^"]*
ws ::= [ \\t\\n]*''')
    
    def parse_response(self, response):
        try:
            data = json.loads(response.strip())
            return data.get('link', '').strip(), data.get('reason', '').strip()
        except:
            return response.strip().strip('"'), "No reason provided"
    
    def explore(self, start_page):
        """Esplora Wikipedia liberamente partendo da una pagina per max_steps passi."""
        self._init_llm()
        
        # Reset token counter for this exploration
        self.total_input_tokens = 0
        
        current = start_page
        path = [start_page]
        visited = {start_page.lower()}
        steps_data = []
        max_steps = self.config['max_steps']
        
        start_time = datetime.now()
        logging.info(f"Starting free exploration from: {start_page} (max {max_steps} steps)")
        
        for step in range(max_steps):
            logging.info(f"Step {step + 1}/{max_steps}: Current page = {current}")
            
            links, content = self.get_page_links(current)
            available = links  # Usa tutti i link disponibili come in wikispeedai.py
            
            if not available:
                logging.warning(f"No links available at: {current}")
                break
            
            logging.info(f"Available links: {len(available)}")
            
            # Crea il prompt
            content_preview = content[:500] if content else "[no content]"
            prompt = PROMPT.format(
                max_steps=max_steps,
                current_page=current,
                content_preview=content_preview,
                links='\n'.join(f"- {link}" for link in available)
            )
            
            # Genera grammar per forzare una scelta valida
            grammar = self.create_grammar(available)
            
            # Count input tokens
            input_tokens = len(self.llm.tokenize(prompt.encode('utf-8')))
            self.total_input_tokens += input_tokens
            logging.info(f"Input tokens: {input_tokens} (Total: {self.total_input_tokens})")
            
            # Chiedi all'LLM di scegliere
            output = self.llm(prompt, max_tokens=512, grammar=grammar)
            chosen, reason = self.parse_response(output['choices'][0]['text'].strip())
            
            if chosen not in available:
                logging.warning(f"Invalid choice: {chosen}, using first available link")
                chosen = available[0]
                reason = "Fallback choice"
            
            logging.info(f"Chosen: {chosen} - Reason: {reason}")
            
            # Crea la mappa dei link disponibili (come in wikispeedai.py)
            available_links_map = {link: link for link in available}
            
            steps_data.append({
                'step': step + 1,
                'from': current,
                'to': chosen,
                'reason': reason,
                'available_links': len(available),
                'input_tokens': input_tokens,  # Save tokens for this step
                'available_links_map': available_links_map
            })
            
            self.all_edges.append((current, chosen))
            
            # Naviga alla pagina scelta
            status, result = self.get_wikipedia_page(chosen)
            if status == 'disambiguation':
                logging.info(f"Disambiguation page, choosing first option")
                chosen = result[0] if result else chosen
                status, result = self.get_wikipedia_page(chosen)
            
            if status == 'success':
                current = result.title
            else:
                current = chosen
            
            path.append(current)
            visited.add(current.lower())
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logging.info(f"Exploration completed: {len(path)} pages visited in {elapsed:.1f}s")
        logging.info(f"Total input tokens used: {self.total_input_tokens}")
        
        return {
            'start': start_page,
            'steps': len(steps_data),
            'path': path,
            'steps_data': steps_data,
            'time': elapsed,
            'unique_pages': len(visited),
            'total_input_tokens': self.total_input_tokens
        }
    
    def save_result(self, result, run_number):
        """Salva i risultati di una singola esplorazione."""
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Salva available links CSV PRIMA di rimuovere available_links_map (come in wikispeedai.py)
        with open(output_dir / f"exploration_{run_number:03d}_available_links.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'origin_page', 'link_text', 'destination_page'])
            for step_data in result['steps_data']:
                for link, dest in step_data.get('available_links_map', {}).items():
                    writer.writerow([step_data['step'], step_data['from'], link, dest])
        
        # Crea una copia e rimuovi available_links_map per JSON (come in wikispeedai.py)
        result_copy = result.copy()
        result_copy['steps_data'] = [
            {k: v for k, v in step.items() if k != 'available_links_map'}
            for step in result['steps_data']
        ]
        
        # Salva JSON con tutti i dettagli (senza available_links_map)
        with open(output_dir / f"exploration_{run_number:03d}.json", 'w', encoding='utf-8') as f:
            json.dump(result_copy, f, indent=2, ensure_ascii=False)
        
        # Salva CSV con i passi
        with open(output_dir / f"exploration_{run_number:03d}_steps.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['step', 'from', 'to', 'reason', 'available_links', 'input_tokens'])
            writer.writeheader()
            for step in result_copy['steps_data']:
                writer.writerow(step)
        
        logging.info(f"Saved exploration {run_number}")
    
    def save_edges_csv(self):
        """Salva tutti gli edge del grafo di navigazione."""
        filename = Path(self.config['output_dir']) / "all_edges.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['source', 'target'])
            writer.writerows(sorted(set(self.all_edges)))
        logging.info(f"Saved {len(set(self.all_edges))} unique edges")
    
    def save_nodes_csv(self, results):
        """Salva tutti i nodi unici (pagine visitate) del grafo."""
        # Raccogli tutte le pagine uniche visitate
        all_nodes = set()
        for result in results:
            all_nodes.update(result['path'])
        
        filename = Path(self.config['output_dir']) / "all_nodes.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['page_title'])
            for node in sorted(all_nodes):
                writer.writerow([node])
        
        logging.info(f"Saved {len(all_nodes)} unique nodes (pages visited)")
    
    def save_summary(self, results):
        """Salva un riassunto di tutte le esplorazioni."""
        filename = Path(self.config['output_dir']) / "summary.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'run', 'start', 'steps', 'unique_pages', 'time', 'path'
            ])
            writer.writeheader()
            
            for i, r in enumerate(results, 1):
                writer.writerow({
                    'run': i,
                    'start': r['start'],
                    'steps': r['steps'],
                    'unique_pages': r['unique_pages'],
                    'time': f"{r['time']:.1f}",
                    'path': ' → '.join(r['path'])
                })
        
        logging.info(f"Summary saved to {filename}")
        
        # Statistiche
        avg_steps = sum(r['steps'] for r in results) / len(results) if results else 0
        avg_unique = sum(r['unique_pages'] for r in results) / len(results) if results else 0
        avg_time = sum(r['time'] for r in results) / len(results) if results else 0
        
        logging.info(f"Statistics: Avg steps: {avg_steps:.1f}, Avg unique pages: {avg_unique:.1f}, Avg time: {avg_time:.1f}s")
    
    def run_experiments(self):
        """Esegue multiple esplorazioni libere di Wikipedia."""
        start_page = self.config['start_page']
        runs = self.config['runs']
        max_steps = self.config['max_steps']
        
        logging.info(f"Starting {runs} exploration runs")
        logging.info(f"Start page: {start_page}, Max steps: {max_steps}")
        
        all_results = []
        
        for run in range(1, runs + 1):
            logging.info(f"Run {run}/{runs}")
            result = self.explore(start_page)
            all_results.append(result)
            self.save_result(result, run)
            
            # Pulisci la memoria del modello dopo ogni run per evitare di superare n_ctx
            self._reset_llm()
        
        self.save_edges_csv()
        self.save_nodes_csv(all_results)
        self.save_summary(all_results)
        
        logging.info(f"All experiments completed. Results in: {self.config['output_dir']}")

if __name__ == "__main__":
    navigator = WikiNavigator()
    navigator.run_experiments()