# PapiAI OpenAI Provider

[![CI](https://github.com/papi-ai/openai/workflows/CI/badge.svg)](https://github.com/papi-ai/openai/actions?query=workflow%3ACI) [![Latest Version](https://img.shields.io/packagist/v/papi-ai/openai.svg)](https://packagist.org/packages/papi-ai/openai) [![Total Downloads](https://img.shields.io/packagist/dt/papi-ai/openai.svg)](https://packagist.org/packages/papi-ai/openai) [![PHP Version](https://img.shields.io/packagist/php-v/papi-ai/openai.svg)](https://packagist.org/packages/papi-ai/openai) [![License](https://img.shields.io/packagist/l/papi-ai/openai.svg)](https://packagist.org/packages/papi-ai/openai)

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
    defaultModel: OpenAIProvider::MODEL_GPT_6_ASTRA,
);

$agent = new Agent(
    provider: $provider,
    model: OpenAIProvider::MODEL_GPT_6_ASTRA,
    instructions: 'You are a helpful assistant.',
);

$response = $agent->run('Hello!');
echo $response->text;
```

## Available Models

```php
OpenAIProvider::MODEL_GPT_6_ASTRA   // 'gpt-6-astra' (default, most capable)
OpenAIProvider::MODEL_GPT_5_6_SOL   // 'gpt-5.6-sol' (complex work)
OpenAIProvider::MODEL_GPT_5_6_TERRA // 'gpt-5.6-terra' (balanced)
OpenAIProvider::MODEL_GPT_5_6_LUNA  // 'gpt-5.6-luna' (cost-sensitive)
OpenAIProvider::MODEL_GPT_4O        // 'gpt-4o' (legacy, still served)
OpenAIProvider::MODEL_GPT_4O_MINI   // 'gpt-4o-mini' (legacy, still served)
```

## Features

- Tool/function calling
- Vision/multimodal support
- Structured output (JSON mode)
- Streaming support

## License

MIT
