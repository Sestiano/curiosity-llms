"""
Download Wikipedia hyperlink structure for analysis.

This script downloads all hyperlinks from Wikipedia pages found in the AI navigation results,
ensuring compatibility with the notebook navigation approach.
Uses wikipedia library page.links property - same logic as wikispeedia.ipynb.
"""

import json
import pickle
import time
from pathlib import Path
import wikipedia
from bs4 import BeautifulSoup
from urllib.parse import unquote


def get_wikipedia_page(title):
    """
    Carica una pagina Wikipedia dato il titolo.
    Gestisce errori di disambiguazione, redirect e pagine non trovate.
    Same logic as wikispeedia.ipynb.
    """
    try:
        # Prova a cercare la pagina con il titolo esatto
        page = wikipedia.page(title, auto_suggest=False)
        return page
    except wikipedia.exceptions.PageError:
        # Se non trova la pagina, prova con auto_suggest=True
        try:
            print(f'  Page "{title}" not found, trying with suggestions...')
            page = wikipedia.page(title, auto_suggest=True)
            print(f'  Found alternative: {page.title}')
            return page
        except Exception as e:
            print(f'  Still not found: {title}')
            return None
    except wikipedia.exceptions.DisambiguationError as e:
        # Se sono presenti più opzioni utilizziamo la prima presente
        print(f'  Disambiguation for {title}, using {e.options[0]}')
        try:
            return wikipedia.page(e.options[0])
        except Exception:
            return None
    except Exception as e:
        print(f'  Error loading page {title}: {e}')
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
            try:
                return [l for l in page.links if not any(kw in l for kw in exclude)]
            except (KeyError, AttributeError) as e:
                print(f"Error accessing page.links: {e}")
                return []
        
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
        try:
            return [l for l in page.links if not any(kw in l for kw in exclude)]
        except (KeyError, AttributeError) as fallback_error:
            print(f"Fallback also failed: {fallback_error}. Returning empty list.")
            return []

def get_wikipedia_links(title, max_retries=3):
    """Get all hyperlinks from a Wikipedia page using wikipedia library."""
    page = get_wikipedia_page(title)
    if page:
        return get_page_links(page)
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
            if result.get("success"):
                path = result.get("path", [])
                all_pages.update(path)
    
    print(f"Found {len(all_pages)} unique pages\n")
    
    # Download Wikipedia links
    wikipedia_links = {}
    failed_pages = []
    
    for i, page in enumerate(sorted(all_pages), 1):
        print(f"[{i}/{len(all_pages)}] {page}")
        links = get_wikipedia_links(page)
        wikipedia_links[page] = links
        
        if len(links) == 0:
            failed_pages.append(page)
            print(f"  No links found")
        else:
            print(f"  {len(links)} links")
        
        time.sleep(0.5)  # Be nice to Wikipedia
    
    # Save to file
    with open(output_file, 'wb') as f:
        pickle.dump(wikipedia_links, f)
    
    print(f"Total pages: {len(wikipedia_links)}")
    print(f"Total links: {sum(len(links) for links in wikipedia_links.values())}")
    print(f"Avg links per page: {sum(len(links) for links in wikipedia_links.values()) / len(wikipedia_links):.1f}")
    if failed_pages:
        print(f"\nFailed pages ({len(failed_pages)}): {failed_pages}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
