# Generative AI Dialogue Summarization with FLAN-T5

A practical implementation of dialogue summarization using Google's FLAN-T5 model, exploring various prompt engineering techniques from zero-shot to few-shot inference.

## Overview

This Jupyter notebook demonstrates how to use Large Language Models (LLMs) for dialogue summarization tasks. It provides hands-on experience with prompt engineering techniques and shows how different approaches affect the quality of generated summaries.

## Features

- **Multiple Inference Approaches**: Compare zero-shot, one-shot, and few-shot inference techniques
- **Prompt Engineering**: Explore different prompt templates and their impact on model performance
- **Configuration Tuning**: Experiment with generation parameters like temperature, top-k, and top-p
- **Real Dataset**: Uses the DialogSum dataset with 10,000+ dialogues and human-labeled summaries

## Dataset

The notebook uses the [DialogSum dataset](https://huggingface.co/datasets/knkarthick/dialogsum) from Hugging Face, which contains:
- 10,000+ dialogues between two people
- Manually labeled summaries for each dialogue
- Various conversation topics and scenarios

## Model

- **Base Model**: Google FLAN-T5 Base
- **Task**: Text-to-text generation (dialogue summarization)
- **Framework**: Hugging Face Transformers

## Requirements

### System Requirements
- **Instance Type**: ml.m5.2xlarge (8 vCPUs, 32 GiB RAM)
- **Python**: 3.12+

### Dependencies

```bash
pip install tensorflow==2.18.0 keras==3.9.0
pip install torch==2.5.1 torchdata==0.6.0
pip install datasets==2.17.0 transformers==4.38.2 evaluate==0.4.0 rouge_score==0.1.2 peft==0.3.0
```

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd dialogue-summarization
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter**
   ```bash
   jupyter notebook 1_summarize_dialogue.ipynb
   ```

## Notebook Structure

### 1. Setup & Dependencies
- Environment verification
- Package installation
- Model and tokenizer loading

### 2. Baseline Summarization
- Raw model performance without prompt engineering
- Understanding tokenization process

### 3. Zero-Shot Inference
- **Custom Prompts**: Create instruction-based prompts
- **FLAN-T5 Templates**: Use pre-built prompt templates
- Compare different prompt formulations

### 4. One-Shot & Few-Shot Inference
- **One-Shot**: Provide one example dialogue-summary pair
- **Few-Shot**: Use multiple examples for better context
- Analyze improvement in summary quality

### 5. Generation Parameters
- Experiment with `temperature`, `top_k`, `top_p`
- Control randomness and creativity in outputs
- Find optimal configuration for your use case

## Key Techniques Demonstrated

### Prompt Engineering
```python
prompt = f"""
Dialogue:

{dialogue}

Write a short summary
"""
```

### Few-Shot Learning
```python
# Provide examples before the target dialogue
prompt = f"""
Dialogue: {example_dialogue_1}
Summary: {example_summary_1}

Dialogue: {example_dialogue_2}
Summary: {example_summary_2}

Dialogue: {target_dialogue}
Summary:
"""
```

### Parameter Tuning
```python
generation_config = GenerationConfig(
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.9
)
```

## Example Results

**Input Dialogue:**
```
#Person1#: What time is it, Tom?
#Person2#: Just a minute. It's ten to nine by my watch.
#Person1#: Is it? I had no idea it was so late. I must be off now.
#Person2#: What's the hurry?
#Person1#: I must catch the nine-thirty train.
#Person2#: You've plenty of time yet. The railway station is very close.
```

**Generated Summary:**
```
#Person1# is in a hurry to catch a train. Tom tells #Person1# there is plenty of time.
```

## Usage Tips

1. **Start Simple**: Begin with zero-shot prompts before moving to few-shot
2. **Experiment with Prompts**: Small changes in wording can significantly affect results
3. **Monitor Context Length**: Stay within the 512 token limit for FLAN-T5
4. **Balance Examples**: For few-shot, 1-3 examples usually work best
5. **Tune Parameters**: Adjust temperature for creativity vs. consistency

## Exercises & Experiments

The notebook includes several exercises to help you explore:
- Different prompt formulations
- Varying numbers of few-shot examples
- Generation parameter effects
- Custom dialogue examples

## Limitations

- **Context Length**: Limited to 512 tokens for FLAN-T5 base
- **Domain Specificity**: Performance may vary across different dialogue types
- **Computational Resources**: Requires adequate memory for model loading

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Research for the FLAN-T5 model
- Hugging Face for the transformers library
- DialogSum dataset creators
- AWS for the educational framework

## Additional Resources

- [FLAN-T5 Paper](https://arxiv.org/abs/2210.11416)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [DialogSum Dataset](https://huggingface.co/datasets/knkarthick/dialogsum)

---

**Note**: This notebook is designed for educational purposes and demonstrates practical applications of prompt engineering with Large Language Models.