"""
Download Wikipedia hyperlink structure for analysis.

This script downloads all hyperlinks from Wikipedia pages found in the AI navigation results,
ensuring compatibility with the notebook navigation approach.
It now uses the same robust page fetching and link parsing logic as wikispeedai.py.
"""

import json
import pickle
import time
from pathlib import Path
import wikipedia
from bs4 import BeautifulSoup
from urllib.parse import unquote
import logging
from typing import List, Tuple, Optional

# --- Functions copied from wikispeedai.py for identical logic ---

def get_wikipedia_page(title: str) -> Tuple[str, any]:
    """Loads a Wikipedia page with error handling. Identical to wikispeedai.py."""
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

def get_page_links(page, exclude_keywords: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """Fetches a Wikipedia page HTML and extracts internal links. Identical to wikispeedai.py."""
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

# --- Adapted function to bridge new logic with script's purpose ---

def get_wikipedia_links(title: str) -> List[str]:
    """
    Get all hyperlink titles from a Wikipedia page.
    This function uses the robust page fetching and parsing logic and adapts the
    output to what this script requires (a list of page titles).
    """
    # In case of disambiguation, this script will just take the first option.
    status, result = get_wikipedia_page(title)
    
    page = None
    if status == 'success':
        page = result
    elif status == 'disambiguation':
        first_option = result[0] if result else None
        if first_option:
            print(f"  Disambiguation for '{title}', using first option: '{first_option}'")
            status, page = get_wikipedia_page(first_option)
            if status != 'success':
                return []
        else:
            return []

    if page:
        # get_page_links now returns List[Tuple[anchor, title]]
        # This script only needs the titles.
        links_with_anchors = get_page_links(page)
        return [link_title for anchor, link_title in links_with_anchors]
        
    return []

def main():
    # Use results directory and search recursively
    data_dir = Path("results")
    output_file = Path("wikipedia_links.pkl")
    backup_file = Path("wikipedia_links_backup.pkl")
    
    # Check if results directory exists
    if not data_dir.exists():
        print(f"Error: Directory '{data_dir}' not found!")
        return
    
    # Backup old file if exists
    if output_file.exists():
        import shutil
        shutil.copy(output_file, backup_file)
        print(f"Backed up old file to: {backup_file}\n")
    
    # Load all unique pages from results (search recursively)
    all_pages = set()
    json_files = sorted(data_dir.glob("**/result_*.json"))
    
    print(f"Scanning results directory: {data_dir.absolute()}")
    print(f"Loading pages from {len(json_files)} JSON files...\n")
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
            # Considering all paths, not just successful ones, might be better
            # to capture all visited nodes.
            path = result.get("path", [])
            all_pages.update(path)
    
    print(f"Found {len(all_pages)} unique pages\n")
    
    # Download Wikipedia links
    wikipedia_links = {}
    failed_pages = []
    
    # Setup basic logging for the functions that need it
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    for i, page in enumerate(sorted(all_pages), 1):
        print(f"[{i}/{len(all_pages)}] {page}")
        links = get_wikipedia_links(page)
        wikipedia_links[page] = links
        
        if len(links) == 0:
            failed_pages.append(page)
            print(f"  No links found")
        else:
            print(f"  {len(links)} links")
        
        time.sleep(0.1)  # Be nice to Wikipedia
    
    # Save to file
    with open(output_file, 'wb') as f:
        pickle.dump(wikipedia_links, f)
    
    print(f"\nTotal pages: {len(wikipedia_links)}")
    print(f"Total links: {sum(len(links) for links in wikipedia_links.values())}")
    if len(wikipedia_links) > 0:
        print(f"Avg links per page: {sum(len(links) for links in wikipedia_links.values()) / len(wikipedia_links):.1f}")
    if failed_pages:
        print(f"\nFailed pages ({len(failed_pages)}): {failed_pages}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()