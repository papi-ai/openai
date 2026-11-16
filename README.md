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
OpenAIProvider::MODEL_GPT_6_1_SOL   // 'gpt-6.1-sol'
OpenAIProvider::MODEL_GPT_6_1_TERRA // 'gpt-6.1-terra'
OpenAIProvider::MODEL_GPT_6_SOL     // 'gpt-6-sol'
OpenAIProvider::MODEL_GPT_6_LUNA    // 'gpt-6-luna'
OpenAIProvider::MODEL_GPT_5_6_SOL   // 'gpt-5.6-sol' (complex work)
OpenAIProvider::MODEL_GPT_5_6_TERRA // 'gpt-5.6-terra' (balanced)
OpenAIProvider::MODEL_GPT_5_6_LUNA  // 'gpt-5.6-luna' (cost-sensitive)
OpenAIProvider::MODEL_GPT_4O        // 'gpt-4o' (legacy, still served)
OpenAIProvider::MODEL_GPT_4O_MINI   // 'gpt-4o-mini' (legacy, still served)
```

## Endpoint

Chat, streaming and tool calling go through the Responses API (`POST /responses`). Tools are
declared flat, the system message travels as `instructions`, structured output under
`text.format`, and effort under `reasoning.effort`. Each request carries the full conversation
(`store: false`); nothing is kept on OpenAI's side between turns.

## Features

- Tool/function calling
- Vision/multimodal support
- Structured output (JSON mode)
- Streaming support

## Creating It by Name

An application that picks its provider from configuration can build this one without naming
its class:

```php
use PapiAI\Core\Factory\ComposerDiscovery;
use PapiAI\Core\Factory\ProviderRegistry;

$registry = new ProviderRegistry(new ComposerDiscovery());

$provider = $registry->create('openai', [
    'api_key' => $_ENV['OPENAI_API_KEY'],
]);
```

It takes `api_key` (required), `model`, `max_tokens`, `base_url`, `api_version` and `effort`.
Anything it is not given falls back to the constructor's defaults, and a missing or unknown setting
throws `ConfigurationException` before any request is made.

## License

MIT
