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
OpenAIProvider::MODEL_GPT_5_6_SOL   // 'gpt-5.6-sol' (complex work)
OpenAIProvider::MODEL_GPT_5_6_TERRA // 'gpt-5.6-terra' (balanced)
OpenAIProvider::MODEL_GPT_5_6_LUNA  // 'gpt-5.6-luna' (cost-sensitive)
OpenAIProvider::MODEL_GPT_4O        // 'gpt-4o' (legacy, still served)
OpenAIProvider::MODEL_GPT_4O_MINI   // 'gpt-4o-mini' (legacy, still served)
```

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

## Requirements

- PHP 8.2+
- `ext-curl`
- `papi-ai/papi-core` ^0.14
