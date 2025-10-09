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
    """
    Estrae i links dalla pagina Wikipedia.
    Filtra i link non desiderati (categorie, template, ecc.).
    Same logic as wikispeedia.ipynb.
    """
    if not page:
        return []
    
    if exclude_keywords is None:
        exclude_keywords = ['Category:', 'Template:', 'File:', 'Portal:', 
                           'Help:', 'Wikipedia:', 'Talk:', 'Special:', 
                           'List of', 'Timeline of']
        
    links = page.links
    
    # Filtra i link indesiderati
    filtered_links = []
    for link in links:
        if not any(kw in link for kw in exclude_keywords):
            filtered_links.append(link)
    
    return filtered_links

def get_wikipedia_links(title, max_retries=3):
    """Get all hyperlinks from a Wikipedia page using wikipedia library."""
    page = get_wikipedia_page(title)
    if page:
        return get_page_links(page)
    return []

def main():
    data_dir = Path("C:\Users\sebas\Desktop\seb\Wikispeedai\Wikispeedai\lm_studio\temp_0_3\Vaccine")
    output_file = Path("C:\Users\sebas\Desktop\seb\Wikispeedai\Wikispeedai\lm_studio\temp_0_3")
    backup_file = Path("C:\Users\sebas\Desktop\seb\Wikispeedai\Wikispeedai\lm_studio\temp_0_3")
    
    # Backup old file if exists
    if output_file.exists():
        import shutil
        shutil.copy(output_file, backup_file)
        print(f"Backed up old file to: {backup_file}\n")
    
    # Load all unique pages from results
    all_pages = set()
    json_files = sorted(data_dir.glob("result_*.json"))
    
    print(f"Loading pages from {len(json_files)} JSON files...")
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
