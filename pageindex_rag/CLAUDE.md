## Instructions for the construction

```models
- granite4.2:8b (for heavy lifitng)
- qwen2.5:3b (only generation)
```

#### Reconstruct the above given python notebook so as to :

- Migrate from nvidia's nim to ollama models completely
- I Have the PDFs and `policy_qa.json` inside a `data/` directory in the Project Root
- Reconstruct with minimal possible changes such that the entire pipeline works just fine
- Caching the results at all costs -> making it persistent across memory refreshes
- See to it that there are no bugs and minimal (no bullshit) comments
- there are a total of 83 questions therefore it'll take time to execute it therefore i want it u to have checkpoints at costs so that the progress is saved (cached) everytime
- cached results should be inside a `cache/` in json formats
- While evaluating : thinking = False (for whichever the model is the the JUDGE) + introduce parallelism so as to do the evaluation part as fast as possible
