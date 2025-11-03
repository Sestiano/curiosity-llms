# WikiSpeedAI

An experimental framework for testing Large Language Models' ability to navigate Wikipedia by selecting links to reach a target article using local LLMs with grammar-constrained generation.

## Overview

WikiSpeedAI challenges LLMs to navigate from a starting Wikipedia page to a target page by strategically selecting links. The system uses **grammar-constrained generation** to force the model to choose only valid links, eliminating hallucinations while tracking navigation paths, measuring success rates, and analyzing model behavior across different personalities.

## Features

- **Grammar-constrained generation** using llama-cpp-python (no hallucinations)
- **Local model execution** (no API calls, faster and more reliable)
- **Multiple personality types** for testing different navigation strategies
- **Loop detection** to identify when models get stuck in cycles
- **Detailed tracking** of all available links at each step
- **Comprehensive logging** system with file and console output
- **Disambiguation handling** for ambiguous Wikipedia pages
- **Correction system** with retry mechanism for invalid choices
- **JSON + reason** output format for interpretable decisions
- **Batch experiments** across multiple starting pages and personalities
- **Statistics by personality** including success rates and average steps

## Requirements

```bash
pip install llama-cpp-python wikipedia beautifulsoup4
```

You'll also need a GGUF format model file (e.g., Mistral, Llama, etc.).

## Configuration

Create a `config.json` file with your settings:

```json
{
  "model_path": "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
  "start_pages": ["Vaccine", "Computer Science"],
  "target": "Adolf Hitler",
  "iterations_per_start_page": 10,
  "personalities": ["baseline", "busybody", "hunter", "dancer"],
  "max_loop_repetitions": 3,
  "max_correction_attempts": 2,
  "output_dir": "results",
  "personality_traits": {
    "baseline": "",
    "busybody": "You MUST have a BUSYBODY type personality...",
    "hunter": "You MUST have a HUNTER type personality...",
    "dancer": "You MUST have a DANCER type personality..."
  },
  "prompts": {
    "baseline": "You are a Large Language Model acting as a Wikipedia navigator...",
    "navigation": "Your TARGET page is: \"{target_article}\"...",
    "disambiguation": "Your previous choice was AMBIGUOUS...",
    "correction": "CORRECTION REQUIRED: Your previous choice was INVALID..."
  }
}
```

### Configuration Parameters

- **model_path**: Path to your GGUF model file (relative or absolute)
- **start_pages**: List of Wikipedia articles to start navigation from
- **target**: Target Wikipedia article to reach
- **iterations_per_start_page**: Number of experiments to run for each starting page
- **personalities**: List of personality types for system prompts (baseline, busybody, hunter, dancer)
- **max_loop_repetitions**: Maximum allowed repeated transitions before detecting a loop (default: 3)
- **max_correction_attempts**: Number of retry attempts for invalid link choices (default: 2)
- **output_dir**: Directory where results are saved (default: "results")
- **personality_traits**: Custom personality descriptions for each type
- **prompts**: Template prompts for navigation, disambiguation, and correction

## Usage

1. Place your GGUF model in the `models/` directory
2. Configure `config.json` with your experiment parameters
3. Run the experiment:

```bash
python wikispeedai.py
```

The system will:
- Load the local model (first run may take time)
- Execute experiments for all combinations of start_pages × personalities × iterations
- Save results in structured directories
- Display progress and statistics in real-time
- Generate comprehensive logs in the `logs/` directory

## Personality Types

The system supports different personality types that modify the navigation strategy:

- **baseline**: Standard navigation approach focused on reaching the target efficiently
- **busybody**: Explores diverse topics, wide but shallow knowledge, focuses on culture and geography
- **hunter**: Goal-oriented with deep focus, strong in science and technical fields
- **dancer**: Creative lateral thinking, connects distant ideas in unconventional ways

## Output

Results are saved in a structured directory format:

```
results/
  personality_baseline/
    Vaccine/
      result_001.json
      result_001_steps.csv
      result_001_available_links.csv
    Computer_Science/
      result_001.json
      ...
  personality_hunter/
    ...
  network_edges.csv
  summary.csv
logs/
  experiment_20251103_143022.log
wiki_cache/
  Vaccine.txt
  Computer_Science.txt
  ...
```

### Output Files

**Per Experiment:**
- `result_XXX.json`: Complete result with path, steps, timing, and metadata
- `result_XXX_steps.csv`: Step-by-step navigation with reasons and corrections
- `result_XXX_available_links.csv`: All available links at each step (for network analysis)

**Aggregate:**
- `network_edges.csv`: All unique navigation edges explored across experiments
- `summary.csv`: Summary table with success rates, steps, errors, and paths

**Logs:**
- Detailed timestamped logs with INFO level for file, WARNING level for console

Each result JSON contains:

- **success**: Whether the target was reached
- **steps**: Number of navigation steps taken
- **path**: Complete navigation path (list of page titles)
- **detailed_steps**: Step-by-step information with:
  - Link chosen and reason provided by the model
  - Number of available links at that step
  - Number of correction attempts needed
- **personality**: Personality type used
- **corrections_used**: Total corrections across all steps
- **loop_detected**: Whether a navigation loop was detected
- **error_type**: Type of failure (dead_end, invalid_link, loop_detected, or null)
- **time**: Execution time in seconds

## How It Works

1. **Configuration Loading**: Loads experiment parameters from config.json with sensible defaults
2. **Model Loading**: Loads GGUF model using llama-cpp-python (only once, reused for all experiments)
3. **Page Loading**: Retrieves Wikipedia pages with caching and extracts available links
4. **Grammar Creation**: Generates BNF grammar forcing JSON output with valid link choices only
5. **LLM Decision**: Presents page content and available links to the model with personality-specific prompt
6. **Grammar-Constrained Generation**: Model output is forced to be valid JSON with a link from available options
7. **Navigation**: Follows the chosen link, tracks visited pages to avoid loops
8. **Loop Detection**: Monitors repeated transitions (A→B→A→B...) to prevent infinite cycles
9. **Disambiguation Handling**: Resolves ambiguous Wikipedia pages by presenting options
10. **Result Tracking**: Saves detailed navigation data, statistics, and logs

## Key Advantages Over API-Based Approach

- **Zero Hallucinations**: Grammar ensures only valid links can be chosen
- **Faster**: Local execution, no network latency
- **More Reliable**: No API timeouts or rate limits
- **Simpler**: ~50% less code than original implementation
- **Interpretable**: Every choice includes a reason from the model
- **Comprehensive**: Tracks all available links for network analysis

## Key Functions

- `navigate()`: Executes a single navigation experiment with loop detection
- `get_page_links()`: Extracts links from Wikipedia pages and caches content
- `create_grammar()`: Generates BNF grammar for constrained generation
- `parse_response()`: Extracts link and reason from JSON response
- `create_navigation_prompt()`: Creates context-aware navigation prompt
- `create_disambiguation_prompt()`: Handles ambiguous page resolution
- `create_correction_prompt()`: Requests valid link after invalid choice
- `get_personality_prompt()`: Retrieves system prompt with personality traits
- `save_result()`: Saves JSON, CSV, and detailed link tracking
- `run_experiments()`: Orchestrates batch experiments across all configurations

## Analysis

Use `analysis.ipynb` to analyze experimental results:

- Load and aggregate data from multiple experiments
- Calculate success rates by personality type
- Analyze navigation path lengths and efficiency
- Compare correction usage across personalities
- Study loop patterns and dead ends
- Visualize navigation networks from edge data
- Analyze most commonly chosen links
- Compare reasoning strategies between personalities

## Example Statistics Output

```
Statistics by Personality:
  baseline: 8/10 (80.0%) - Avg steps: 4.2 - Loops: 0 - Avg corrections: 0.3
  busybody: 6/10 (60.0%) - Avg steps: 5.8 - Loops: 1 - Avg corrections: 0.8
  hunter: 9/10 (90.0%) - Avg steps: 3.7 - Loops: 0 - Avg corrections: 0.1
  dancer: 5/10 (50.0%) - Avg steps: 6.4 - Loops: 2 - Avg corrections: 1.2
```

## Architecture

**New Version (wikispeedai.py):**
- 429 lines
- Local llama-cpp-python execution
- Grammar-constrained generation
- No fuzzy matching needed
- Comprehensive logging system

**Backup Version (wikispeedai_backup.py):**
- 855 lines
- API-based (LM Studio/Ollama)
- Fuzzy matching for corrections
- Temperature testing
- Detailed fallback statistics

The new version is 50% smaller while adding features like detailed link tracking and better logging.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

