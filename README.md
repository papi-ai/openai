# PapiAI OpenAI Provider

[![Tests](https://github.com/papi-ai/openai/workflows/CI/badge.svg)](https://github.com/papi-ai/openai/actions?query=workflow%3ACI)

OpenAI provider for [PapiAI](https://github.com/papi-ai/papi-core) - A simple but powerful PHP library for building AI agents.

## Installation

```bash
composer require papi-ai/openai
```

## Usage

```php
use PapiAI\Core\Agent;
use PapiAI\OpenAI\OpenAIProvider;

$provider = new OpenAIProvider(
    apiKey: $_ENV['OPENAI_API_KEY'],
    defaultModel: OpenAIProvider::MODEL_GPT_4O,
);

$agent = new Agent(
    provider: $provider,
    model: 'gpt-4o',
    instructions: 'You are a helpful assistant.',
);

$response = $agent->run('Hello!');
echo $response->text;
```

## Available Models

```php
OpenAIProvider::MODEL_GPT_4O        // 'gpt-4o' (default, multimodal)
OpenAIProvider::MODEL_GPT_4O_MINI   // 'gpt-4o-mini' (fast, cost-effective)
OpenAIProvider::MODEL_GPT_4_TURBO   // 'gpt-4-turbo' (high quality)
OpenAIProvider::MODEL_O1_PREVIEW    // 'o1-preview' (reasoning)
OpenAIProvider::MODEL_O1_MINI       // 'o1-mini' (fast reasoning)
```

## Features

- Tool/function calling
- Vision/multimodal support
- Structured output (JSON mode)
- Streaming support

## License

MIT
