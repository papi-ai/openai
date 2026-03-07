<?php

/*
 * This file is part of PapiAI,
 * A simple but powerful PHP library for building AI agents.
 *
 * (c) Marcello Duarte <marcello.duarte@gmail.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

declare(strict_types=1);

namespace PapiAI\OpenAI;

use Generator;
use PapiAI\Core\AudioResponse;
use PapiAI\Core\Contracts\EmbeddingProviderInterface;
use PapiAI\Core\Contracts\ProviderInterface;
use PapiAI\Core\Contracts\TextToSpeechProviderInterface;
use PapiAI\Core\Contracts\TranscriptionProviderInterface;
use PapiAI\Core\EmbeddingResponse;
use PapiAI\Core\Message;
use PapiAI\Core\Response;
use PapiAI\Core\Role;
use PapiAI\Core\StreamChunk;
use PapiAI\Core\ToolCall;
use PapiAI\Core\TranscriptionResponse;
use RuntimeException;

/**
 * OpenAI API Provider.
 *
 * Supports OpenAI models including:
 * - gpt-4o (latest, multimodal)
 * - gpt-4o-mini (fast, cost-effective)
 * - gpt-4-turbo (high quality)
 * - o1-preview (reasoning)
 * - o1-mini (fast reasoning)
 */
class OpenAIProvider implements ProviderInterface, EmbeddingProviderInterface, TextToSpeechProviderInterface, TranscriptionProviderInterface
{
    private const API_URL = 'https://api.openai.com/v1/chat/completions';
    private const EMBEDDINGS_API_URL = 'https://api.openai.com/v1/embeddings';
    private const AUDIO_SPEECH_API_URL = 'https://api.openai.com/v1/audio/speech';
    private const AUDIO_TRANSCRIPTIONS_API_URL = 'https://api.openai.com/v1/audio/transcriptions';

    public const MODEL_GPT_4_5 = 'gpt-4.5-preview';
    public const MODEL_GPT_4O = 'gpt-4o';
    public const MODEL_GPT_4O_MINI = 'gpt-4o-mini';
    public const MODEL_GPT_4_TURBO = 'gpt-4-turbo';
    public const MODEL_O1 = 'o1';
    public const MODEL_O1_PREVIEW = 'o1-preview';
    public const MODEL_O1_MINI = 'o1-mini';
    public const MODEL_O3_MINI = 'o3-mini';

    public function __construct(
        private readonly string $apiKey,
        private readonly string $defaultModel = self::MODEL_GPT_4O,
        private readonly int $defaultMaxTokens = 4096,
    ) {
    }

    #[\Override]
    public function chat(array $messages, array $options = []): Response
    {
        $payload = $this->buildPayload($messages, $options);
        $response = $this->request($payload);

        return Response::fromOpenAI($response, $messages);
    }

    #[\Override]
    public function stream(array $messages, array $options = []): iterable
    {
        $payload = $this->buildPayload($messages, $options);
        $payload['stream'] = true;

        foreach ($this->streamRequest($payload) as $event) {
            $delta = $event['choices'][0]['delta'] ?? [];
            if (isset($delta['content'])) {
                yield new StreamChunk($delta['content']);
            }
            if (($event['choices'][0]['finish_reason'] ?? null) !== null) {
                yield new StreamChunk('', isComplete: true);
            }
        }
    }

    #[\Override]
    public function supportsTool(): bool
    {
        return true;
    }

    #[\Override]
    public function supportsVision(): bool
    {
        return true;
    }

    #[\Override]
    public function supportsStructuredOutput(): bool
    {
        return true;
    }

    #[\Override]
    public function embed(string|array $input, array $options = []): EmbeddingResponse
    {
        $model = $options['model'] ?? 'text-embedding-3-small';
        $payload = [
            'model' => $model,
            'input' => is_array($input) ? $input : [$input],
        ];

        if (isset($options['dimensions'])) {
            $payload['dimensions'] = $options['dimensions'];
        }

        $response = $this->embeddingRequest($payload);

        $embeddings = array_map(
            fn (array $item) => $item['embedding'],
            $response['data']
        );

        return new EmbeddingResponse(
            embeddings: $embeddings,
            model: $response['model'] ?? $model,
            usage: $response['usage'] ?? [],
        );
    }

    #[\Override]
    public function synthesize(string $text, array $options = []): AudioResponse
    {
        $format = $options['format'] ?? 'mp3';
        $model = $options['model'] ?? 'tts-1';

        $payload = [
            'model' => $model,
            'input' => $text,
            'voice' => $options['voice'] ?? 'alloy',
            'response_format' => $format,
            'speed' => $options['speed'] ?? 1.0,
        ];

        if (isset($options['instructions'])) {
            $payload['instructions'] = $options['instructions'];
        }

        $data = $this->audioRequest($payload);

        return new AudioResponse(
            data: $data,
            format: $format,
            model: $model,
        );
    }

    #[\Override]
    public function transcribe(string $audioPath, array $options = []): TranscriptionResponse
    {
        $model = $options['model'] ?? 'whisper-1';

        $fields = [
            'model' => $model,
            'response_format' => 'verbose_json',
            'timestamp_granularities[]' => 'segment',
        ];

        if (isset($options['language'])) {
            $fields['language'] = $options['language'];
        }

        if (isset($options['prompt'])) {
            $fields['prompt'] = $options['prompt'];
        }

        $response = $this->transcriptionRequest($audioPath, $fields);

        $segments = array_map(
            fn (array $segment) => [
                'start' => (float) $segment['start'],
                'end' => (float) $segment['end'],
                'text' => $segment['text'],
            ],
            $response['segments'] ?? [],
        );

        return new TranscriptionResponse(
            text: $response['text'],
            model: $model,
            language: $response['language'] ?? null,
            duration: isset($response['duration']) ? (float) $response['duration'] : null,
            segments: $segments,
        );
    }

    #[\Override]
    public function getName(): string
    {
        return 'openai';
    }

    /**
     * Build the API request payload.
     */
    private function buildPayload(array $messages, array $options): array
    {
        $apiMessages = [];

        foreach ($messages as $message) {
            if ($message instanceof Message) {
                $apiMessages[] = $this->convertMessage($message);
            }
        }

        $payload = [
            'model' => $options['model'] ?? $this->defaultModel,
            'messages' => $apiMessages,
        ];

        if (isset($options['maxTokens'])) {
            $payload['max_tokens'] = $options['maxTokens'];
        }

        if (isset($options['temperature'])) {
            $payload['temperature'] = $options['temperature'];
        }

        if (isset($options['stopSequences'])) {
            $payload['stop'] = $options['stopSequences'];
        }

        // Handle structured output / JSON mode
        if (isset($options['outputSchema'])) {
            $payload['response_format'] = [
                'type' => 'json_schema',
                'json_schema' => [
                    'name' => 'response',
                    'schema' => $options['outputSchema'],
                ],
            ];
        }

        // Handle tools
        if (isset($options['tools']) && !empty($options['tools'])) {
            $payload['tools'] = $this->convertTools($options['tools']);
        }

        return $payload;
    }

    /**
     * Convert a Message to OpenAI API format.
     */
    private function convertMessage(Message $message): array
    {
        $apiMessage = [
            'role' => $this->convertRole($message->role),
        ];

        if ($message->isTool()) {
            $apiMessage['role'] = 'tool';
            $apiMessage['content'] = $message->content;
            $apiMessage['tool_call_id'] = $message->toolCallId;
        } elseif ($message->hasToolCalls()) {
            $apiMessage['content'] = $message->getText() ?: null;
            $apiMessage['tool_calls'] = array_map(function (ToolCall $tc) {
                return [
                    'id' => $tc->id,
                    'type' => 'function',
                    'function' => [
                        'name' => $tc->name,
                        'arguments' => json_encode($tc->arguments),
                    ],
                ];
            }, $message->toolCalls);
        } elseif (is_array($message->content)) {
            $apiMessage['content'] = $this->convertMultimodalContent($message->content);
        } else {
            $apiMessage['content'] = $message->content;
        }

        return $apiMessage;
    }

    /**
     * Convert multimodal content to OpenAI format.
     */
    private function convertMultimodalContent(array $content): array
    {
        $parts = [];

        foreach ($content as $part) {
            if ($part['type'] === 'text') {
                $parts[] = ['type' => 'text', 'text' => $part['text']];
            } elseif ($part['type'] === 'image') {
                $source = $part['source'];
                if ($source['type'] === 'url') {
                    $parts[] = [
                        'type' => 'image_url',
                        'image_url' => ['url' => $source['url']],
                    ];
                } else {
                    $parts[] = [
                        'type' => 'image_url',
                        'image_url' => [
                            'url' => "data:{$source['media_type']};base64,{$source['data']}",
                        ],
                    ];
                }
            }
        }

        return $parts;
    }

    /**
     * Convert tools from PapiAI format to OpenAI format.
     */
    private function convertTools(array $tools): array
    {
        $openaiTools = [];

        foreach ($tools as $tool) {
            if (is_array($tool)) {
                $openaiTools[] = [
                    'type' => 'function',
                    'function' => [
                        'name' => $tool['name'],
                        'description' => $tool['description'],
                        'parameters' => $tool['input_schema'] ?? $tool['parameters'] ?? ['type' => 'object', 'properties' => []],
                    ],
                ];
            }
        }

        return $openaiTools;
    }

    /**
     * Convert Role to OpenAI role string.
     */
    private function convertRole(Role $role): string
    {
        return match ($role) {
            Role::System => 'system',
            Role::User => 'user',
            Role::Assistant => 'assistant',
            Role::Tool => 'tool',
        };
    }

    /**
     * Make an API request.
     *
     * @codeCoverageIgnore
     */
    protected function request(array $payload): array
    {
        $ch = curl_init(self::API_URL);

        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($payload),
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => [
                'Content-Type: application/json',
                'Authorization: Bearer ' . $this->apiKey,
            ],
        ]);

        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);

        curl_close($ch);

        if ($error !== '') {
            throw new RuntimeException("OpenAI API request failed: {$error}");
        }

        $data = json_decode($response, true);

        if ($httpCode >= 400) {
            $errorMessage = $data['error']['message'] ?? 'Unknown error';
            throw new RuntimeException("OpenAI API error ({$httpCode}): {$errorMessage}");
        }

        return $data;
    }

    /**
     * Make an embedding API request.
     *
     * @codeCoverageIgnore
     */
    protected function embeddingRequest(array $payload): array
    {
        $ch = curl_init(self::EMBEDDINGS_API_URL);

        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($payload),
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => [
                'Content-Type: application/json',
                'Authorization: Bearer ' . $this->apiKey,
            ],
        ]);

        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);

        curl_close($ch);

        if ($error !== '') {
            throw new RuntimeException("OpenAI Embeddings API request failed: {$error}");
        }

        $data = json_decode($response, true);

        if ($httpCode >= 400) {
            $errorMessage = $data['error']['message'] ?? 'Unknown error';
            throw new RuntimeException("OpenAI Embeddings API error ({$httpCode}): {$errorMessage}");
        }

        return $data;
    }

    /**
     * Make a streaming API request.
     *
     * @codeCoverageIgnore
     *
     * @return Generator<array>
     */
    protected function streamRequest(array $payload): Generator
    {
        $ch = curl_init(self::API_URL);

        $buffer = '';
        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($payload),
            CURLOPT_HTTPHEADER => [
                'Content-Type: application/json',
                'Authorization: Bearer ' . $this->apiKey,
            ],
            CURLOPT_WRITEFUNCTION => function ($ch, $data) use (&$buffer) {
                $buffer .= $data;

                return strlen($data);
            },
        ]);

        curl_exec($ch);
        curl_close($ch);

        // Parse SSE events
        $lines = explode("\n", $buffer);
        foreach ($lines as $line) {
            $line = trim($line);
            if (str_starts_with($line, 'data: ')) {
                $json = substr($line, 6);
                if ($json === '[DONE]') {
                    break;
                }
                $event = json_decode($json, true);
                if ($event !== null) {
                    yield $event;
                }
            }
        }
    }

    /**
     * Make an audio speech API request.
     *
     * Returns raw audio bytes.
     *
     * @codeCoverageIgnore
     */
    protected function audioRequest(array $payload): string
    {
        $ch = curl_init(self::AUDIO_SPEECH_API_URL);

        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($payload),
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => [
                'Content-Type: application/json',
                'Authorization: Bearer ' . $this->apiKey,
            ],
        ]);

        /** @var string $response */
        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);

        curl_close($ch);

        if ($error !== '') {
            throw new RuntimeException("OpenAI Audio API request failed: {$error}");
        }

        if ($httpCode >= 400) {
            $data = json_decode($response, true);
            $errorMessage = $data['error']['message'] ?? 'Unknown error';
            throw new RuntimeException("OpenAI Audio API error ({$httpCode}): {$errorMessage}");
        }

        return $response;
    }

    /**
     * Make a transcription API request with multipart form data.
     *
     * @codeCoverageIgnore
     *
     * @return array<string, mixed>
     */
    protected function transcriptionRequest(string $audioPath, array $fields): array
    {
        $ch = curl_init(self::AUDIO_TRANSCRIPTIONS_API_URL);

        $fields['file'] = new \CURLFile($audioPath);

        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => $fields,
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => [
                'Authorization: Bearer ' . $this->apiKey,
            ],
        ]);

        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);

        curl_close($ch);

        if ($error !== '') {
            throw new RuntimeException("OpenAI Transcription API request failed: {$error}");
        }

        $data = json_decode($response, true);

        if ($httpCode >= 400) {
            $errorMessage = $data['error']['message'] ?? 'Unknown error';
            throw new RuntimeException("OpenAI Transcription API error ({$httpCode}): {$errorMessage}");
        }

        return $data;
    }
}
