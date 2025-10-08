# WikiSpeedAI

An experimental framework for testing Large Language Models' ability to navigate Wikipedia by selecting links to reach a target article.

## Overview

WikiSpeedAI challenges LLMs to navigate from a starting Wikipedia page to a target page by strategically selecting links. The system tracks navigation paths, measures success rates, and analyzes model behavior including hallucinations and fallback mechanisms.

## Features

- **Multi-temperature testing**: Run experiments with different temperature settings to analyze model behavior
- **Fuzzy matching**: Handles cases where the model's response doesn't exactly match available links
- **Hallucination tracking**: Monitors when the model suggests non-existent links
- **Detailed analytics**: Tracks fallback statistics, similarity scores, and navigation paths
- **Batch experiments**: Run multiple iterations across different starting pages
- **Loop detection**: Automatically detects when the model gets stuck in navigation loops

## Requirements

```bash
pip install wikipedia requests beautifulsoup4 tqdm
```

## Configuration

Create a `config.json` file with your settings:

```json
{
  "start_pages": ["Vaccine", "Philosophy", "Ancient Rome"],
  "target": "Renaissance",
  "iterations_per_start_page": 100,
  "temperatures": [0.3, 1.5],
  "lm_studio_url": "http://localhost:1234/v1/chat/completions",
  "model_name": "lm_studio",
  "fuzzy_match_threshold": 0.95,
  "max_loop_repetitions": 3
}
```

### Configuration Parameters

- **start_pages**: List of Wikipedia articles to start navigation from
- **target**: Target Wikipedia article to reach
- **iterations_per_start_page**: Number of experiments to run for each starting page
- **temperatures**: List of temperature values to test (controls model randomness)
- **lm_studio_url**: LM Studio API endpoint URL
- **model_name**: Model identifier (used for output directory naming)
- **fuzzy_match_threshold**: Minimum similarity score (0-1) for fuzzy link matching
- **max_loop_repetitions**: Maximum allowed repeated transitions before detecting a loop

## Usage

1. Start LM Studio with your chosen model
2. Configure `config.json` with your experiment parameters
3. Run the experiment:

```bash
python wikispeedai.py
```

## Output

Results are saved in JSON format with the following structure:

```
{model_name}/
  temp_{temperature}/
    {start_page}/
      result_001_20241006_183935.json
      result_002_20241006_184016.json
      ...
  all_results_{timestamp}.json
```

Each result file contains:
- **success**: Whether the target was reached
- **steps**: Number of navigation steps taken
- **path**: Complete navigation path
- **detailed_steps**: Step-by-step information with link choices
- **fallback_statistics**: Metrics on fuzzy matching and hallucinations
- **loop_detected**: Whether a navigation loop was detected

## How It Works

1. **Page Loading**: Loads Wikipedia pages and extracts available links
2. **LLM Decision**: Presents current page content and available links to the LLM
3. **Link Matching**: Matches LLM response to actual links (exact or fuzzy)
4. **Navigation**: Follows the chosen link and repeats until target reached or stuck
5. **Loop Detection**: Monitors repeated transitions to prevent infinite loops
6. **Result Tracking**: Records all navigation details and statistics

## Key Functions

- `run_navigation_experiment()`: Executes a single navigation experiment
- `call_lm_studio()`: Communicates with LM Studio API
- `parse_model_choice()`: Matches model responses to available links
- `get_page_links()`: Extracts links from Wikipedia pages in order
- `fuzzy_similarity()`: Calculates string similarity for fallback matching

