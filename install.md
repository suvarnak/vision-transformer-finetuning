


```
docker build -t sue-finetuning-llm-img .

docker run -it --gpus all -v main.py:/app/main.py sue-finetuning-llm-img python main.py

docker run -it --gpus all -v  sue-finetuning-llm-img bash
```
