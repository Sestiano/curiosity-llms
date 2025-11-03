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

class WikiNavigator:
    def __init__(self, config_file='config.json'):
        wikipedia.set_lang('en')
        self.config = self._load_config(config_file)
        self.cache_dir = Path("wiki_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.llm = None
        self.all_edges = []
        self._setup_logging()
        
    def _load_config(self, config_file):
        if not Path(config_file).exists():
            return {
                "model_path": "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                "start_pages": ["Peanut"],
                "target": "Jupiter",
                "iterations_per_start_page": 5,
                "personalities": ["baseline"],
                "max_correction_attempts": 2,
                "max_loop_repetitions": 3,
                "output_dir": "results",
                "prompts": {},
                "personality_traits": {}
            }
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        config.setdefault("model_path", "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
        config.setdefault("max_correction_attempts", 2)
        config.setdefault("max_loop_repetitions", 3)
        config.setdefault("output_dir", "results")
        config.setdefault("personalities", ["baseline"])
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
            logging.info(f"Loading model: {model_path}")
            self.llm = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=0, verbose=False)
            logging.info("Model loaded successfully")
    
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
        alternatives = ' | '.join([f'"{link}"' for link in links])
        return LlamaGrammar.from_string(f'''root ::= object
object ::= "{{" ws members ws "}}"
members ::= pair (ws "," ws pair)*
pair ::= "\\"link\\"" ws ":" ws string ws "," ws "\\"reason\\"" ws ":" ws string
string ::= "\\"" ({alternatives}) "\\""
ws ::= [ \\t\\n]*''')
    
    def get_personality_prompt(self, personality):
        baseline = self.config.get('prompts', {}).get('baseline', '')
        traits = self.config.get('personality_traits', {}).get(personality, '')
        return f"{baseline}\n{traits}" if traits else baseline
    
    def _format_prompt(self, template, **kwargs):
        return template.format(**kwargs) if template else ""
    
    def create_navigation_prompt(self, current, target, path, content, links):
        return self._format_prompt(
            self.config.get('prompts', {}).get('navigation', ''),
            target_article=target,
            current_article=current,
            recent_path=' -> '.join(path[-3:]) if len(path) > 1 else current,
            content_text=content[:1000] if content else "[no content]",
            links_list='\n'.join(links)
        )
    
    def create_disambiguation_prompt(self, choice, target, path, options):
        return self._format_prompt(
            self.config.get('prompts', {}).get('disambiguation', ''),
            original_choice=choice,
            recent_path=' -> '.join(path[-3:]) if len(path) > 1 else path[0],
            target_article=target,
            options_list='\n'.join(options)
        )
    
    def create_correction_prompt(self, invalid_choice, target, path, links):
        return self._format_prompt(
            self.config.get('prompts', {}).get('correction', ''),
            invalid_choice=invalid_choice,
            target_article=target,
            recent_path=' -> '.join(path[-3:]) if len(path) > 1 else path[0],
            links_list='\n'.join(links)
        )
    
    def parse_response(self, response):
        try:
            data = json.loads(response.strip())
            return data.get('link', '').strip(), data.get('reason', '').strip()
        except:
            return response.strip().strip('"'), "No reason provided"
    
    def navigate(self, start, target, personality='baseline'):
        self._init_llm()
        
        current, path, visited = start, [start], {start.lower()}
        detailed_steps, corrections_used = [], 0
        recent_transitions = []
        max_loop = self.config['max_loop_repetitions']
        
        start_time = datetime.now()
        logging.info(f"Starting navigation: {start} -> {target} (personality: {personality})")
        
        system_prompt = self.get_personality_prompt(personality)
        
        step = 0
        while current.lower() != target.lower():
            logging.info(f"Step {step}: Current page = {current}")
            
            links, content = self.get_page_links(current)
            available = [link for link in links if link.lower() not in visited]
            
            if not available:
                logging.warning(f"Dead end reached at: {current}")
                return {
                    'success': False, 'steps': step, 'path': path, 'detailed_steps': detailed_steps,
                    'start': start, 'target': target, 'personality': personality,
                    'error_type': 'dead_end', 'time': (datetime.now() - start_time).total_seconds(),
                    'corrections_used': corrections_used, 'loop_detected': False
                }
            
            logging.info(f"Available links: {len(available)}")
            grammar = self.create_grammar(available)
            chosen, reason = None, ""
            
            for attempt in range(self.config['max_correction_attempts']):
                prompt = self.create_navigation_prompt(current, target, path, content, available) if attempt == 0 \
                         else self.create_correction_prompt(chosen, target, path, available)
                
                if attempt > 0:
                    corrections_used += 1
                    logging.info(f"Correction attempt {attempt}")
                
                output = self.llm(prompt, max_tokens=150, grammar=grammar)
                chosen, reason = self.parse_response(output['choices'][0]['text'].strip())
                
                if chosen in available:
                    logging.info(f"Valid link chosen: {chosen}")
                    break
                else:
                    logging.warning(f"Invalid link chosen: {chosen}")
            
            if chosen not in available:
                logging.error(f"Failed to find valid link after {self.config['max_correction_attempts']} attempts")
                return {
                    'success': False, 'steps': step, 'path': path, 'detailed_steps': detailed_steps,
                    'start': start, 'target': target, 'personality': personality,
                    'error_type': 'invalid_link', 'time': (datetime.now() - start_time).total_seconds(),
                    'corrections_used': corrections_used, 'loop_detected': False
                }
            
            available_links_map = {link: link for link in available}
            
            detailed_steps.append({
                'step': step, 'from': current, 'to': chosen, 'reason': reason,
                'available_links': len(available), 'corrections': attempt,
                'available_links_map': available_links_map
            })
            
            self.all_edges.append((current, chosen))
            
            transition = (current.lower(), chosen.lower())
            recent_transitions.append(transition)
            if len(recent_transitions) > max_loop * 2:
                recent_transitions.pop(0)
            
            if recent_transitions.count(transition) >= max_loop:
                logging.warning(f"Loop detected: {current} <-> {chosen} (repeated {max_loop} times)")
                return {
                    'success': False, 'steps': step, 'path': path, 'detailed_steps': detailed_steps,
                    'start': start, 'target': target, 'personality': personality,
                    'error_type': 'loop_detected', 'time': (datetime.now() - start_time).total_seconds(),
                    'corrections_used': corrections_used, 'loop_detected': True
                }
            
            status, result = self.get_wikipedia_page(chosen)
            if status == 'disambiguation':
                logging.info(f"Resolving disambiguation for: {chosen}")
                prompt = self.create_disambiguation_prompt(chosen, target, path, result)
                output = self.llm(prompt, max_tokens=150, grammar=self.create_grammar(result))
                chosen, _ = self.parse_response(output['choices'][0]['text'].strip())
                logging.info(f"Disambiguation resolved to: {chosen}")
                status, result = self.get_wikipedia_page(chosen)
            
            current = result.title if status == 'success' else chosen
            path.append(current)
            visited.add(current.lower())
            step += 1
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logging.info(f"SUCCESS! Reached target in {step} steps ({elapsed:.1f}s)")
        print(f"  SUCCESS in {step} steps ({elapsed:.1f}s)")
        
        return {
            'success': True, 'steps': step, 'path': path, 'detailed_steps': detailed_steps,
            'start': start, 'target': target, 'personality': personality,
            'error_type': None, 'time': elapsed, 'corrections_used': corrections_used,
            'loop_detected': False
        }
    
    def save_result(self, result, start_page, personality, iteration):
        output_dir = Path(self.config['output_dir']) / f"personality_{personality}" / start_page.replace(' ', '_')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result_copy = result.copy()
        for step in result_copy.get('detailed_steps', []):
            if 'available_links_map' in step:
                del step['available_links_map']
        
        with open(output_dir / f"result_{iteration:03d}.json", 'w', encoding='utf-8') as f:
            json.dump(result_copy, f, indent=2, ensure_ascii=False)
        
        with open(output_dir / f"result_{iteration:03d}_steps.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['step', 'from', 'to', 'reason', 'available_links', 'corrections'])
            writer.writeheader()
            for step in result['detailed_steps']:
                writer.writerow({k: v for k, v in step.items() if k != 'available_links_map'})
        
        with open(output_dir / f"result_{iteration:03d}_available_links.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'origin_page', 'link_text', 'destination_page'])
            for step_data in result['detailed_steps']:
                for link, dest in step_data.get('available_links_map', {}).items():
                    writer.writerow([step_data['step'], step_data['from'], link, dest])
        
        logging.info(f"Saved result {iteration} for {start_page}/{personality}")
    
    def save_edges_csv(self):
        filename = Path(self.config['output_dir']) / "network_edges.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['source', 'target'])
            writer.writerows(sorted(set(self.all_edges)))
        logging.info(f"Saved {len(set(self.all_edges))} unique edges to {filename}")
        print(f"Saved {len(set(self.all_edges))} unique edges to {filename}")
    
    def save_summary(self, results):
        filename = Path(self.config['output_dir']) / "summary.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'iteration', 'start', 'target', 'personality', 'success', 'steps', 
                'error_type', 'time', 'corrections_used', 'loop_detected', 'path'
            ])
            writer.writeheader()
            
            for i, r in enumerate(results, 1):
                writer.writerow({
                    'iteration': i, 'start': r['start'], 'target': r['target'],
                    'personality': r['personality'], 'success': r['success'], 'steps': r['steps'],
                    'error_type': r.get('error_type', ''), 'time': f"{r['time']:.1f}",
                    'corrections_used': r.get('corrections_used', 0), 
                    'loop_detected': r.get('loop_detected', False),
                    'path': ' -> '.join(r['path'])
                })
        
        logging.info(f"Summary saved to {filename}")
        print(f"Summary saved to {filename}")
        
        stats = {}
        for r in results:
            p = r['personality']
            if p not in stats:
                stats[p] = {'total': 0, 'success': 0, 'steps': [], 'loops': 0, 'corrections': 0}
            
            stats[p]['total'] += 1
            stats[p]['corrections'] += r.get('corrections_used', 0)
            if r['success']:
                stats[p]['success'] += 1
                stats[p]['steps'].append(r['steps'])
            if r.get('loop_detected'):
                stats[p]['loops'] += 1
        
        print("\nStatistics by Personality:")
        for p in sorted(stats.keys()):
            s = stats[p]
            rate = s['success'] / s['total'] * 100 if s['total'] > 0 else 0
            avg = sum(s['steps']) / len(s['steps']) if s['steps'] else 0
            avg_corr = s['corrections'] / s['total'] if s['total'] > 0 else 0
            print(f"  {p}: {s['success']}/{s['total']} ({rate:.1f}%) - Avg steps: {avg:.1f} - Loops: {s['loops']} - Avg corrections: {avg_corr:.1f}")
            logging.info(f"Stats {p}: {s['success']}/{s['total']} ({rate:.1f}%) - Avg: {avg:.1f} - Loops: {s['loops']}")
    
    def run_experiments(self):
        start_pages = self.config.get('start_pages', ['Vaccine'])
        target = self.config['target']
        iterations = self.config.get('iterations_per_start_page', 5)
        personalities = self.config.get('personalities', ['baseline'])
        
        total = len(start_pages) * len(personalities) * iterations
        logging.info(f"Starting {total} experiments")
        logging.info(f"Start pages: {start_pages}")
        logging.info(f"Target: {target}")
        logging.info(f"Personalities: {personalities}")
        
        print(f"\nStarting {total} experiments")
        print(f"Start pages: {', '.join(start_pages)}")
        print(f"Target: {target}")
        print(f"Personalities: {', '.join(personalities)}\n")
        
        all_results = []
        success_count = 0
        
        exp_num = 0
        for start in start_pages:
            for personality in personalities:
                for i in range(iterations):
                    exp_num += 1
                    logging.info(f"Experiment {exp_num}/{total}: {start} -> {target} | {personality} | Run {i+1}")
                    print(f"[{exp_num}/{total}] {start} -> {target} | {personality} (Run {i+1}/{iterations})")
                    
                    result = self.navigate(start, target, personality)
                    all_results.append(result)
                    
                    if result['success']:
                        success_count += 1
                    
                    self.save_result(result, start, personality, i + 1)
        
        self.save_edges_csv()
        self.save_summary(all_results)
        
        success_rate = success_count / len(all_results) * 100 if all_results else 0
        logging.info(f"Experiments completed: {success_count}/{len(all_results)} ({success_rate:.1f}%)")
        
        print(f"\nCompleted {len(all_results)} experiments")
        print(f"Success rate: {success_count}/{len(all_results)} ({success_rate:.1f}%)")
        print(f"Results saved in: {self.config['output_dir']}")

if __name__ == "__main__":
    navigator = WikiNavigator()
    navigator.run_experiments()