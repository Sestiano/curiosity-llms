# WikiSpeedAI

An experimental framework for testing Large Language Models' ability to navigate Wikipedia by selecting links to reach a target article.

## Overview

WikiSpeedAI challenges LLMs to navigate from a starting Wikipedia page to a target page by strategically selecting links. The system tracks navigation paths, measures success rates, and analyzes model behavior including hallucinations and fallback mechanisms.

## Features

- Multi-temperature testing for analyzing model behavior variability
- Personality-based system prompts to test different navigation strategies
- Fuzzy matching for handling inexact model responses
- Hallucination tracking for non-existent link suggestions
- Detailed analytics including fallback statistics and similarity scores
- Batch experiments across multiple starting pages
- Loop detection to identify when models get stuck
- Comprehensive JSON output for each experiment

## Requirements

```bash
pip install wikipedia requests beautifulsoup4 tqdm
```

## Configuration

Create a `config.json` file with your settings:

```json
{
  "start_pages": ["Albert Einstein", "Computer Science"],
  "target": "Stockholm",
  "iterations_per_start_page": 20,
  "temperatures": [0.3, 1.5],
  "personalities": ["baseline", "busybody", "hunter", "dancer"],
  "lm_studio_url": "http://localhost:1234/v1/chat/completions",
  "model_name": "lm_studio",
  "output_dir": "results",
  "fuzzy_match_threshold": 0.95,
  "max_loop_repetitions": 3,
  "exclude_keywords": ["Category:", "Template:", "File:", "Portal:", "Help:", "Wikipedia:", "Talk:", "Special"]
}
```

### Configuration Parameters

- **start_pages**: List of Wikipedia articles to start navigation from
- **target**: Target Wikipedia article to reach
- **iterations_per_start_page**: Number of experiments to run for each starting page
- **temperatures**: List of temperature values to test (controls model randomness)
- **personalities**: List of personality types for system prompts (baseline, busybody, hunter, dancer)
- **lm_studio_url**: LM Studio API endpoint URL
- **model_name**: Model identifier (used for output directory naming)
- **output_dir**: Directory where results are saved
- **fuzzy_match_threshold**: Minimum similarity score (0-1) for fuzzy link matching
- **max_loop_repetitions**: Maximum allowed repeated transitions before detecting a loop
- **exclude_keywords**: List of keywords to filter out from available links

## Usage

1. Start LM Studio with your chosen model
2. Configure `config.json` with your experiment parameters
3. Run the experiment:

```bash
python wikispeedai.py
```

## Personality Types

The system supports different personality types that modify the navigation strategy:

- **baseline**: Standard navigation approach without additional traits
- **busybody**: Focuses on exploring diverse topics and connections
- **hunter**: Aggressive, goal-oriented navigation
- **dancer**: Creative, lateral thinking approach

## Output

Results are saved in JSON format with the following structure:

```
results/
  temp_{temperature}_personality_{personality}/
    {start_page}/
      result_001_20251010_124348.json
      result_002_20251010_124412.json
      ...
```

Each result file contains:

- **success**: Whether the target was reached
- **steps**: Number of navigation steps taken
- **path**: Complete navigation path
- **detailed_steps**: Step-by-step information with link choices and reasoning
- **fallback_statistics**: Metrics on fuzzy matching and hallucinations
- **loop_detected**: Whether a navigation loop was detected
- **temperature**: Temperature setting used
- **personality**: Personality type used
- **timestamp**: When the experiment was conducted

## How It Works

1. **Configuration Loading**: Loads experiment parameters from config.json
2. **Page Loading**: Retrieves Wikipedia pages and extracts available links in order
3. **LLM Decision**: Presents current page content and available links to the LLM
4. **Link Matching**: Matches LLM response to actual links using exact or fuzzy matching
5. **Navigation**: Follows the chosen link and repeats until target is reached or stuck
6. **Loop Detection**: Monitors repeated transitions to prevent infinite loops
7. **Result Tracking**: Records all navigation details and statistics in JSON format

## Key Functions

- `run_navigation_experiment()`: Executes a single navigation experiment
- `call_lm_studio()`: Communicates with LM Studio API
- `parse_model_choice()`: Matches model responses to available links
- `get_page_links()`: Extracts links from Wikipedia pages in order
- `fuzzy_similarity()`: Calculates string similarity for fallback matching
- `get_personality_prompt()`: Retrieves system prompt based on personality type
- `detect_loop()`: Identifies repeated navigation patterns

## Analysis

Use `analysis.ipynb` to analyze experimental results:

- Load and aggregate data from multiple experiments
- Calculate success rates by temperature and personality
- Analyze navigation path lengths
- Compare fallback and hallucination rates
- Generate visualizations of navigation patterns

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

