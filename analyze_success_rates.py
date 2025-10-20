"""
Script per analizzare i tassi di successo in results-2 e identificare i motivi dei fallimenti.
"""

import json
import os
import glob
from pathlib import Path
from collections import defaultdict
import pandas as pd

def analyze_result_file(file_path):
    """
    Analizza un singolo file JSON per determinare se è un successo o un fallimento.
    
    Returns:
        dict: {
            'success': bool,
            'reason': str (se fallimento),
            'path_length': int,
            'target_reached': bool,
            'data': dict (dati originali)
        }
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Estrai informazioni chiave
        path = data.get('path', [])
        target = data.get('target', '')
        end_page = path[-1] if path else None
        
        # Determina successo
        success = (end_page == target) if end_page and target else False
        
        result = {
            'success': success,
            'path_length': len(path),
            'target': target,
            'end_page': end_page,
            'start_page': path[0] if path else None,
            'file': file_path
        }
        
        # Identifica motivo del fallimento
        if not success:
            if not path:
                result['failure_reason'] = 'Empty path'
            elif not target:
                result['failure_reason'] = 'No target specified'
            elif len(path) >= 50:  # Assumendo un limite di step
                result['failure_reason'] = f'Max steps reached (ended at: {end_page})'
            else:
                result['failure_reason'] = f'Target not reached (ended at: {end_page})'
        else:
            result['failure_reason'] = None
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'failure_reason': f'Error reading file: {str(e)}',
            'path_length': 0,
            'target': None,
            'end_page': None,
            'start_page': None,
            'file': file_path
        }

def extract_metadata_from_path(file_path):
    """
    Estrae temperatura, pagina iniziale e personalità dal percorso del file.
    """
    parts = Path(file_path).parts
    
    metadata = {
        'model': None,
        'temperature': None,
        'personality': None,
        'starting_page': None
    }
    
    # Trova il modello (es: openai_gpt-oss-20b)
    for i, part in enumerate(parts):
        if 'openai' in part or 'gpt' in part or 'llama' in part:
            metadata['model'] = part
            break
    
    # Estrai temperatura e personalità
    for part in parts:
        if part.startswith('temp_'):
            import re
            match = re.match(r'temp_(\d+)_(\d+)_personality_(.+)', part)
            if match:
                temp_int, temp_dec, personality = match.groups()
                metadata['temperature'] = f"{temp_int}.{temp_dec}"
                metadata['personality'] = personality
    
    # Trova la pagina iniziale
    for i, part in enumerate(parts):
        if metadata['temperature'] and not part.endswith('.json') and not part.startswith('temp_'):
            if part not in ['results-2', metadata['model']]:
                metadata['starting_page'] = part
                break
    
    return metadata

def analyze_directory(base_dir='results-2'):
    """
    Analizza tutti i file JSON in results-2 e calcola le statistiche.
    """
    print(f"Analyzing {base_dir}...")
    
    # Trova tutti i file JSON
    json_files = glob.glob(f"{base_dir}/**/*.json", recursive=True)
    print(f"Found {len(json_files)} files")
    
    # Analizza tutti i file
    results = []
    for file_path in json_files:
        analysis = analyze_result_file(file_path)
        metadata = extract_metadata_from_path(file_path)
        
        # Combina analisi e metadata
        combined = {**metadata, **analysis}
        results.append(combined)
    
    # Crea DataFrame per analisi
    df = pd.DataFrame(results)
    
    # Prepara il dizionario con tutte le analisi
    analysis_results = {
        'summary': {},
        'by_model': {},
        'by_temperature': {},
        'by_personality': {},
        'by_starting_page': {},
        'by_combination': {},
        'failure_reasons': {},
        'best_configurations': []
    }
    
    # === STATISTICHE GLOBALI ===
    total = len(df)
    successes = df['success'].sum()
    failures = total - successes
    success_rate = (successes / total * 100) if total > 0 else 0
    
    analysis_results['summary'] = {
        'total_experiments': total,
        'successes': int(successes),
        'failures': int(failures),
        'success_rate': round(success_rate, 2),
        'avg_path_length': round(df['path_length'].mean(), 2),
        'avg_path_length_success': round(df[df['success']]['path_length'].mean(), 2),
        'avg_path_length_failure': round(df[~df['success']]['path_length'].mean(), 2)
    }
    
    print(f"\nGlobal success rate: {success_rate:.2f}% ({successes}/{total})")
    
    # === ANALISI PER MODELLO ===
    if 'model' in df.columns and df['model'].notna().any():
        for model in df['model'].dropna().unique():
            model_df = df[df['model'] == model]
            model_successes = model_df['success'].sum()
            model_total = len(model_df)
            model_rate = (model_successes / model_total * 100) if model_total > 0 else 0
            
            analysis_results['by_model'][model] = {
                'successes': int(model_successes),
                'total': int(model_total),
                'success_rate': round(model_rate, 2),
                'avg_path_length': round(model_df['path_length'].mean(), 2)
            }
    
    # === ANALISI PER TEMPERATURA ===
    for temp in sorted(df['temperature'].dropna().unique()):
        temp_df = df[df['temperature'] == temp]
        temp_successes = temp_df['success'].sum()
        temp_total = len(temp_df)
        temp_rate = (temp_successes / temp_total * 100) if temp_total > 0 else 0
        
        analysis_results['by_temperature'][temp] = {
            'successes': int(temp_successes),
            'total': int(temp_total),
            'success_rate': round(temp_rate, 2),
            'avg_path_length': round(temp_df['path_length'].mean(), 2)
        }
    
    # === ANALISI PER PERSONALITÀ ===
    for personality in sorted(df['personality'].dropna().unique()):
        pers_df = df[df['personality'] == personality]
        pers_successes = pers_df['success'].sum()
        pers_total = len(pers_df)
        pers_rate = (pers_successes / pers_total * 100) if pers_total > 0 else 0
        
        analysis_results['by_personality'][personality] = {
            'successes': int(pers_successes),
            'total': int(pers_total),
            'success_rate': round(pers_rate, 2),
            'avg_path_length': round(pers_df['path_length'].mean(), 2)
        }
    
    # === ANALISI PER PAGINA INIZIALE ===
    for start_page in sorted(df['starting_page'].dropna().unique()):
        page_df = df[df['starting_page'] == start_page]
        page_successes = page_df['success'].sum()
        page_total = len(page_df)
        page_rate = (page_successes / page_total * 100) if page_total > 0 else 0
        
        analysis_results['by_starting_page'][start_page] = {
            'successes': int(page_successes),
            'total': int(page_total),
            'success_rate': round(page_rate, 2),
            'avg_path_length': round(page_df['path_length'].mean(), 2)
        }
    
    # === ANALISI COMBINATA (Temperatura × Personalità) ===
    for temp in sorted(df['temperature'].dropna().unique()):
        for personality in sorted(df['personality'].dropna().unique()):
            comb_df = df[(df['temperature'] == temp) & (df['personality'] == personality)]
            if len(comb_df) > 0:
                comb_successes = comb_df['success'].sum()
                comb_total = len(comb_df)
                comb_rate = (comb_successes / comb_total * 100)
                
                key = f"temp_{temp}_personality_{personality}"
                analysis_results['by_combination'][key] = {
                    'temperature': temp,
                    'personality': personality,
                    'successes': int(comb_successes),
                    'total': int(comb_total),
                    'success_rate': round(comb_rate, 2),
                    'avg_path_length': round(comb_df['path_length'].mean(), 2)
                }
    
    # === MOTIVI DEI FALLIMENTI ===
    failures_df = df[~df['success']]
    if len(failures_df) > 0:
        failure_reasons = failures_df['failure_reason'].value_counts()
        
        for reason, count in failure_reasons.items():
            percentage = (count / len(failures_df) * 100)
            analysis_results['failure_reasons'][str(reason)] = {
                'count': int(count),
                'percentage': round(percentage, 2)
            }
    
    # === MIGLIORI CONFIGURAZIONI ===
    if len(df) > 0:
        grouped = df.groupby(['temperature', 'personality']).agg({
            'success': ['sum', 'count']
        }).reset_index()
        grouped.columns = ['temperature', 'personality', 'successes', 'total']
        grouped['success_rate'] = (grouped['successes'] / grouped['total'] * 100)
        grouped = grouped.sort_values('success_rate', ascending=False)
        
        for idx, row in grouped.head(5).iterrows():
            analysis_results['best_configurations'].append({
                'rank': int(idx + 1),
                'temperature': row['temperature'],
                'personality': row['personality'],
                'successes': int(row['successes']),
                'total': int(row['total']),
                'success_rate': round(row['success_rate'], 2)
            })
    
    # === SALVA RISULTATI ===
    output_file = 'success_analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print(f"Analysis saved to: {output_file}")
    
    return analysis_results

if __name__ == "__main__":
    results = analyze_directory('results-2')
    print("\n✓ Analysis completed!")
