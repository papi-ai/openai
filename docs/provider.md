# OpenAI

OpenAI provider for PapiAI.

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

## Models

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

## Capabilities

| Capability | Supported |
|---|---|
| Chat | Yes |
| Streaming | Yes |
| Tool calling | Yes |
| Vision | Yes |
| Structured output | Yes |
| Embeddings | Yes |
| Text-to-speech | Yes |
| Transcription | Yes |

## Text-to-Speech

OpenAI supports TTS via the same provider package:

```php
$audio = $provider->synthesize('Hello world!', [
    'model' => 'tts-1',    // or 'tts-1-hd'
    'voice' => 'alloy',
]);
$audio->save('output.mp3');
```

## Transcription

```php
$transcription = $provider->transcribe('/path/to/audio.mp3', [
    'model' => 'whisper-1',
]);
echo $transcription->text;
```

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

## Requirements

- PHP 8.2+
- `ext-curl`
- `papi-ai/papi-core` ^0.14
