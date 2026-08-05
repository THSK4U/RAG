# vLLM Overview

vLLM is a fast and easy-to-use library for LLM inference and serving.

## Key Features

- **PagedAttention**: Efficiently manages attention key and value memory
- **Continuous batching**: Handles incoming requests without waiting for others to finish
- **Optimized CUDA kernels**: Custom GPU kernels for maximum throughput

## Architecture

vLLM uses a client-server architecture. The server hosts the model and handles
requests through an OpenAI-compatible REST API. Clients send prompts and receive
generated text in return.

### Scheduling

The scheduler decides which requests to process at each step. It uses a
first-come-first-served (FCFS) policy by default but supports other strategies.

### Memory Management

Memory is divided into fixed-size blocks. The KV cache is stored in these blocks,
and the paged attention mechanism allows non-contiguous memory allocation,
dramatically reducing waste compared to traditional methods.
