the current setup uses Ollama setup for everything
but now my friend has given to me the inference endpoint of deepseek with an api key (its pay as u go one) -> baseUrl = https://api.deepseek.com
therefore now i wanna brainstorm with u, on how to do the migration to a openai compatible pipeline for the deepseek inference endpoint (api key)?
what i feel is : nuke the directories for the individual RAG strategies completely and then refer to ->

> ~/Projects/RAG_v1/run_pageindex_v4.ipynb
> ~/Projects/VectorRAG/noteboook/RAG_pipeline.ipynb

for vectorRAG (pageindex) and VectorRAG respectively, since they had been made on the openai compatible nvidia endpoints

<mark>Also we need to debug the current RAG stregies seperately to find out he bugs which are resulting in bad evaluations</mark>
The evaluations of the RAG strategies have to be published in the research_paper therefore they need to be concrete enough and legible for the purpose: research paper
that is why a solid rag pipieline with minimal bugs from retrieval to evaluation is very pivotal for the final results to be legible, if any one of the steps is bad, it makes the entire eval results bad

**Only use `deepseek-flash` and not the deepseek-v4-pro so as to not burn the usage, coze only a finite bucks are there in that account for me to exhaust**

Use /ste100 and /caveman to produce the final_plan.md explaining to me your plan as well as for your reference which executing the decided plan for the required execution
