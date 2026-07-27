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

## Models

```php
OpenAIProvider::MODEL_GPT_4O        // 'gpt-4o' (default, multimodal)
OpenAIProvider::MODEL_GPT_4O_MINI   // 'gpt-4o-mini' (fast, cost-effective)
OpenAIProvider::MODEL_GPT_4_TURBO   // 'gpt-4-turbo' (high quality)
OpenAIProvider::MODEL_O1_PREVIEW    // 'o1-preview' (reasoning)
OpenAIProvider::MODEL_O1_MINI       // 'o1-mini' (fast reasoning)
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
