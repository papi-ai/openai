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
use PapiAI\Core\Contracts\VideoProviderInterface;
use PapiAI\Core\Effort;
use PapiAI\Core\EmbeddingResponse;
use PapiAI\Core\Exception\AuthenticationException;
use PapiAI\Core\Exception\ProviderException;
use PapiAI\Core\Exception\RateLimitException;
use PapiAI\Core\Exception\UnknownEffortException;
use PapiAI\Core\JobStatus;
use PapiAI\Core\Message;
use PapiAI\Core\Response;
use PapiAI\Core\Role;
use PapiAI\Core\StreamChunk;
use PapiAI\Core\ToolCall;
use PapiAI\Core\ToolChoice;
use PapiAI\Core\TranscriptionResponse;
use PapiAI\Core\VideoResponse;
use RuntimeException;

/**
 * OpenAI API provider for PapiAI.
 *
 * Bridges PapiAI's core types (Message, Response, ToolCall) with OpenAI's
 * Chat Completions API, handling format conversion in both directions. Supports
 * chat completions, streaming, tool calling, vision (multimodal), structured
 * JSON output, text embeddings, text-to-speech, and audio transcription.
 *
 * Authentication is via Bearer token (or api-key header for Azure OpenAI).
 * All HTTP is done with ext-curl directly, with no HTTP abstraction layer.
 * The base URL is configurable for Azure OpenAI compatibility.
 *
 * @see https://platform.openai.com/docs/api-reference
 */
class OpenAIProvider implements ProviderInterface, EmbeddingProviderInterface, TextToSpeechProviderInterface, TranscriptionProviderInterface, VideoProviderInterface
{
    private const DEFAULT_BASE_URL = 'https://api.openai.com/v1';

    /**
     * OpenAI's own spellings for the two levels whose neutral names differ.
     */
    private const NATIVE_EFFORT = [
        'extra-high' => 'xhigh',
        'maximum' => 'max',
    ];

    public const MODEL_GPT_4_5 = 'gpt-4.5-preview';
    public const MODEL_GPT_4O = 'gpt-4o';
    public const MODEL_GPT_4O_MINI = 'gpt-4o-mini';
    public const MODEL_GPT_4_TURBO = 'gpt-4-turbo';
    public const MODEL_O1 = 'o1';
    public const MODEL_O1_PREVIEW = 'o1-preview';
    public const MODEL_O1_MINI = 'o1-mini';
    public const MODEL_O3_MINI = 'o3-mini';

    // Sora model aliases for video generation
    public const MODEL_SORA_2 = 'sora-2';
    public const MODEL_SORA_2_PRO = 'sora-2-pro';

    private readonly string $baseUrl;

    /**
     * Create a new OpenAI provider instance.
     *
     * When $baseUrl and $apiVersion are provided, the provider switches to
     * Azure OpenAI mode, using api-key header auth and appending api-version
     * as a query parameter.
     *
     * @param string      $apiKey          API key for authentication
     * @param string      $defaultModel    Model to use when none is specified in options
     * @param int         $defaultMaxTokens Default max tokens for completions
     * @param string|null $baseUrl         Custom base URL (for Azure OpenAI or proxies)
     * @param string|null $apiVersion      API version query parameter (enables Azure mode)
     * @param Effort|null $defaultEffort   Reasoning effort when none is given per call
     */
    public function __construct(
        private readonly string $apiKey,
        private readonly string $defaultModel = self::MODEL_GPT_4O,
        private readonly int $defaultMaxTokens = 4096,
        ?string $baseUrl = null,
        private readonly ?string $apiVersion = null,
        private readonly ?Effort $defaultEffort = null,
    ) {
        $this->baseUrl = rtrim($baseUrl ?? self::DEFAULT_BASE_URL, '/');
    }

    /**
     * Send a chat completion request to OpenAI.
     *
     * Converts PapiAI Messages to OpenAI's message format, sends the request,
     * and wraps the response in a core Response object.
     *
     * @param Message[] $messages Conversation history
     * @param array     $options  Provider options (model, maxTokens, temperature, tools, outputSchema, etc.)
     *
     * @return Response The parsed completion response
     *
     * @throws ProviderException       When the API returns an error
     * @throws AuthenticationException When the API key is invalid
     * @throws RateLimitException      When rate limits are exceeded
     */
    #[\Override]
    public function chat(array $messages, array $options = []): Response
    {
        $payload = $this->buildPayload($messages, $options);
        $response = $this->request($payload);

        return Response::fromOpenAI($response, $messages);
    }

    /**
     * Stream a chat completion response from OpenAI.
     *
     * Yields StreamChunk objects as server-sent events arrive, with the final
     * chunk marked as complete via its isComplete flag.
     *
     * @param Message[] $messages Conversation history
     * @param array     $options  Provider options (model, maxTokens, temperature, tools, etc.)
     *
     * @return iterable<StreamChunk> Stream of text chunks
     *
     * @throws ProviderException       When the API returns an error
     * @throws AuthenticationException When the API key is invalid
     * @throws RateLimitException      When rate limits are exceeded
     */
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

    /**
     * Indicate that this provider supports tool/function calling.
     *
     * @return bool Always true — OpenAI supports function calling natively
     */
    #[\Override]
    public function supportsTool(): bool
    {
        return true;
    }

    /**
     * Indicate that this provider supports vision (image) inputs.
     *
     * @return bool Always true — GPT-4o and GPT-4 Turbo accept image content
     */
    #[\Override]
    public function supportsVision(): bool
    {
        return true;
    }

    /**
     * Indicate that this provider supports structured JSON output via json_schema.
     *
     * @return bool Always true — OpenAI supports response_format with json_schema
     */
    #[\Override]
    public function supportsStructuredOutput(): bool
    {
        return true;
    }

    /**
     * Generate vector embeddings for the given input text(s).
     *
     * Uses OpenAI's Embeddings API (default model: text-embedding-3-small).
     * Accepts a single string or an array of strings for batch embedding.
     *
     * @param string|string[] $input Text(s) to embed
     * @param array           $options Options (model, dimensions)
     *
     * @return EmbeddingResponse The embedding vectors with model and usage metadata
     *
     * @throws ProviderException       When the API returns an error
     * @throws AuthenticationException When the API key is invalid
     * @throws RateLimitException      When rate limits are exceeded
     */
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

    /**
     * Convert text to speech audio using OpenAI's TTS API.
     *
     * Returns raw audio data in the requested format (default: mp3).
     *
     * @param string $text    The text to synthesize into speech
     * @param array  $options Options (model, voice, format, speed, instructions)
     *
     * @return AudioResponse The synthesized audio data with format and model metadata
     *
     * @throws ProviderException       When the API returns an error
     * @throws AuthenticationException When the API key is invalid
     * @throws RateLimitException      When rate limits are exceeded
     */
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

    /**
     * Transcribe audio to text using OpenAI's Whisper API.
     *
     * Sends the audio file as multipart form data and returns verbose JSON
     * with segment-level timestamps.
     *
     * @param string $audioPath Filesystem path to the audio file
     * @param array  $options   Options (model, language, prompt)
     *
     * @return TranscriptionResponse The transcribed text with segments and metadata
     *
     * @throws ProviderException       When the API returns an error
     * @throws AuthenticationException When the API key is invalid
     * @throws RateLimitException      When rate limits are exceeded
     */
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

    /**
     * Return the unique provider identifier.
     *
     * @return string Always 'openai'
     */
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

        // Forced tool choice. Validation lives in core and throws before any HTTP call.
        if (isset($options['toolChoice'])) {
            $choice = ToolChoice::fromOption($options['toolChoice'], $options['tools'] ?? []);

            if (!empty($options['tools'])) {
                $payload['tool_choice'] = $choice->toolName !== null
                    ? ['type' => 'function', 'function' => ['name' => $choice->toolName]]
                    : match ($choice->mode) {
                        ToolChoice::NONE => 'none',
                        ToolChoice::REQUIRED => 'required',
                        default => 'auto',
                    };
            }
        }

        $effort = $this->effortFor($options);

        if ($effort !== null) {
            $model = (string) ($options['model'] ?? $this->defaultModel);
            $narrowed = $effort->nearestOf($this->levelsFor($model));

            $payload['reasoning_effort'] = self::NATIVE_EFFORT[$narrowed->value] ?? $narrowed->value;
        }

        return $payload;
    }

    /**
     * The reasoning-effort levels a given model accepts.
     *
     * The set genuinely varies, and the API rejects a level the model does not know rather than
     * ignoring it, so a request that overshoots is a 400 and not a silent downgrade. `xhigh`
     * arrived with the 5.1 codex generation, `max` with 5.6, and `minimal` exists only on the
     * original GPT-5. Everything older, the o-series included, takes the three middle levels only.
     *
     * Decided from the model name because the API offers no way to ask.
     *
     * @return non-empty-list<Effort>
     */
    private function levelsFor(string $model): array
    {
        if (!preg_match('/gpt-5(?:\.(\d+))?/i', $model, $matches)) {
            // o-series and everything older.
            return [Effort::Low, Effort::Medium, Effort::High];
        }

        $minor = isset($matches[1]) ? (int) $matches[1] : 0;

        if ($minor >= 6) {
            return [Effort::None, Effort::Low, Effort::Medium, Effort::High, Effort::ExtraHigh, Effort::Maximum];
        }

        if ($minor >= 1) {
            return [Effort::None, Effort::Low, Effort::Medium, Effort::High, Effort::ExtraHigh];
        }

        // The original GPT-5 is the only model that took "minimal", and it has no xhigh.
        return [Effort::Minimal, Effort::Low, Effort::Medium, Effort::High];
    }

    /**
     * The effort this request asks for: the per-call option, else the provider default.
     *
     * @param array<string, mixed> $options The caller's request options
     *
     * @throws UnknownEffortException When the level is not one core defines
     */
    private function effortFor(array $options): ?Effort
    {
        if (!isset($options['effort'])) {
            return $this->defaultEffort;
        }

        $level = (string) $options['effort'];

        return Effort::tryFrom($level) ?? throw new UnknownEffortException($level);
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
     * Build the full URL for an endpoint, appending api-version for Azure.
     */
    private function buildUrl(string $path): string
    {
        $url = $this->baseUrl . $path;

        if ($this->apiVersion !== null) {
            $url .= '?api-version=' . $this->apiVersion;
        }

        return $url;
    }

    /**
     * Get the authorization headers based on the authentication mode.
     *
     * @return string[]
     */
    private function getAuthHeaders(): array
    {
        if ($this->apiVersion !== null) {
            return ['api-key: ' . $this->apiKey];
        }

        return ['Authorization: Bearer ' . $this->apiKey];
    }

    /**
     * Check the HTTP status code and throw appropriate exceptions.
     *
     * @param int $httpCode
     * @param array<string, mixed>|null $data
     * @param array<string, string> $responseHeaders
     *
     * @throws AuthenticationException
     * @throws RateLimitException
     * @throws ProviderException
     */
    private function handleErrorResponse(int $httpCode, ?array $data, array $responseHeaders = []): void
    {
        if ($httpCode < 400) {
            return;
        }

        if ($httpCode === 401) {
            throw new AuthenticationException(
                provider: 'openai',
                statusCode: $httpCode,
                responseBody: $data,
            );
        }

        if ($httpCode === 429) {
            $retryAfter = isset($responseHeaders['retry-after'])
                ? (int) $responseHeaders['retry-after']
                : null;

            throw new RateLimitException(
                provider: 'openai',
                retryAfterSeconds: $retryAfter,
                statusCode: $httpCode,
                responseBody: $data,
            );
        }

        $errorMessage = $data['error']['message'] ?? 'Unknown error';

        throw new ProviderException(
            message: "OpenAI API error ({$httpCode}): {$errorMessage}",
            provider: 'openai',
            statusCode: $httpCode,
            responseBody: $data,
        );
    }

    /**
     * Parse response headers from a cURL header callback.
     *
     * @param string $rawHeaders
     *
     * @return array<string, string>
     */
    private function parseResponseHeaders(string $rawHeaders): array
    {
        $headers = [];
        foreach (explode("\r\n", $rawHeaders) as $line) {
            if (str_contains($line, ':')) {
                [$key, $value] = explode(':', $line, 2);
                $headers[strtolower(trim($key))] = trim($value);
            }
        }

        return $headers;
    }

    /**
     * Send a JSON request to the Chat Completions endpoint and return the decoded response.
     *
     * Protected to allow test doubles to override HTTP transport.
     *
     * @param array<string, mixed> $payload The JSON-encodable request body
     *
     * @return array<string, mixed> The decoded JSON response
     *
     * @throws RuntimeException        When a cURL transport error occurs
     * @throws ProviderException       When the API returns an error status
     * @throws AuthenticationException When the API key is invalid (401)
     * @throws RateLimitException      When rate limits are exceeded (429)
     *
     * @codeCoverageIgnore
     */
    protected function request(array $payload): array
    {
        $url = $this->buildUrl('/chat/completions');
        $ch = curl_init($url);

        $rawHeaders = '';
        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($payload),
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => array_merge(
                ['Content-Type: application/json'],
                $this->getAuthHeaders(),
            ),
            CURLOPT_HEADERFUNCTION => function ($ch, $header) use (&$rawHeaders) {
                $rawHeaders .= $header;

                return strlen($header);
            },
        ]);

        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);

        curl_close($ch);

        if ($error !== '') {
            throw new RuntimeException("OpenAI API request failed: {$error}");
        }

        $data = json_decode($response, true);
        $this->handleErrorResponse($httpCode, $data, $this->parseResponseHeaders($rawHeaders));

        return $data;
    }

    /**
     * Send a JSON request to the Embeddings endpoint and return the decoded response.
     *
     * Protected to allow test doubles to override HTTP transport.
     *
     * @param array<string, mixed> $payload The JSON-encodable request body
     *
     * @return array<string, mixed> The decoded JSON response
     *
     * @throws RuntimeException        When a cURL transport error occurs
     * @throws ProviderException       When the API returns an error status
     * @throws AuthenticationException When the API key is invalid (401)
     * @throws RateLimitException      When rate limits are exceeded (429)
     *
     * @codeCoverageIgnore
     */
    protected function embeddingRequest(array $payload): array
    {
        $url = $this->buildUrl('/embeddings');
        $ch = curl_init($url);

        $rawHeaders = '';
        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($payload),
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => array_merge(
                ['Content-Type: application/json'],
                $this->getAuthHeaders(),
            ),
            CURLOPT_HEADERFUNCTION => function ($ch, $header) use (&$rawHeaders) {
                $rawHeaders .= $header;

                return strlen($header);
            },
        ]);

        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);

        curl_close($ch);

        if ($error !== '') {
            throw new RuntimeException("OpenAI Embeddings API request failed: {$error}");
        }

        $data = json_decode($response, true);
        $this->handleErrorResponse($httpCode, $data, $this->parseResponseHeaders($rawHeaders));

        return $data;
    }

    /**
     * Send a streaming request to the Chat Completions endpoint and yield parsed SSE events.
     *
     * Buffers the full response then parses server-sent events line by line,
     * yielding each decoded JSON event until the [DONE] sentinel.
     * Protected to allow test doubles to override HTTP transport.
     *
     * @param array<string, mixed> $payload The JSON-encodable request body (stream flag added by caller)
     *
     * @return Generator<int, array<string, mixed>> Decoded SSE event payloads
     *
     * @throws ProviderException       When the API returns an error status
     * @throws AuthenticationException When the API key is invalid (401)
     * @throws RateLimitException      When rate limits are exceeded (429)
     *
     * @codeCoverageIgnore
     */
    protected function streamRequest(array $payload): Generator
    {
        $url = $this->buildUrl('/chat/completions');
        $ch = curl_init($url);

        $buffer = '';
        $rawHeaders = '';
        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($payload),
            CURLOPT_HTTPHEADER => array_merge(
                ['Content-Type: application/json'],
                $this->getAuthHeaders(),
            ),
            CURLOPT_HEADERFUNCTION => function ($ch, $header) use (&$rawHeaders) {
                $rawHeaders .= $header;

                return strlen($header);
            },
            CURLOPT_WRITEFUNCTION => function ($ch, $data) use (&$buffer) {
                $buffer .= $data;

                return strlen($data);
            },
        ]);

        curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);

        if ($httpCode >= 400) {
            $data = json_decode($buffer, true);
            $this->handleErrorResponse($httpCode, $data, $this->parseResponseHeaders($rawHeaders));
        }

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
     * Send a JSON request to the Audio Speech endpoint and return raw audio bytes.
     *
     * Unlike other request methods, this returns the raw binary response
     * rather than decoded JSON, since the TTS API returns audio data directly.
     * Protected to allow test doubles to override HTTP transport.
     *
     * @param array<string, mixed> $payload The JSON-encodable request body
     *
     * @return string Raw audio bytes in the requested format
     *
     * @throws RuntimeException        When a cURL transport error occurs
     * @throws ProviderException       When the API returns an error status
     * @throws AuthenticationException When the API key is invalid (401)
     * @throws RateLimitException      When rate limits are exceeded (429)
     *
     * @codeCoverageIgnore
     */
    protected function audioRequest(array $payload): string
    {
        $url = $this->buildUrl('/audio/speech');
        $ch = curl_init($url);

        $rawHeaders = '';
        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($payload),
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => array_merge(
                ['Content-Type: application/json'],
                $this->getAuthHeaders(),
            ),
            CURLOPT_HEADERFUNCTION => function ($ch, $header) use (&$rawHeaders) {
                $rawHeaders .= $header;

                return strlen($header);
            },
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
            $this->handleErrorResponse($httpCode, $data, $this->parseResponseHeaders($rawHeaders));
        }

        return $response;
    }

    /**
     * Send a multipart form request to the Audio Transcriptions endpoint.
     *
     * Attaches the audio file via CURLFile and sends additional fields
     * as form data. Protected to allow test doubles to override HTTP transport.
     *
     * @param string               $audioPath Filesystem path to the audio file
     * @param array<string, mixed> $fields    Form fields (model, response_format, language, etc.)
     *
     * @return array<string, mixed> The decoded JSON response with transcription data
     *
     * @throws RuntimeException        When a cURL transport error occurs
     * @throws ProviderException       When the API returns an error status
     * @throws AuthenticationException When the API key is invalid (401)
     * @throws RateLimitException      When rate limits are exceeded (429)
     *
     * @codeCoverageIgnore
     */
    protected function transcriptionRequest(string $audioPath, array $fields): array
    {
        $url = $this->buildUrl('/audio/transcriptions');
        $ch = curl_init($url);

        $fields['file'] = new \CURLFile($audioPath);

        $rawHeaders = '';
        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => $fields,
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => $this->getAuthHeaders(),
            CURLOPT_HEADERFUNCTION => function ($ch, $header) use (&$rawHeaders) {
                $rawHeaders .= $header;

                return strlen($header);
            },
        ]);

        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);

        curl_close($ch);

        if ($error !== '') {
            throw new RuntimeException("OpenAI Transcription API request failed: {$error}");
        }

        $data = json_decode($response, true);
        $this->handleErrorResponse($httpCode, $data, $this->parseResponseHeaders($rawHeaders));

        return $data;
    }

    /**
     * Whether this provider supports video generation from text prompts.
     *
     * Supported via OpenAI's Sora models through the /videos endpoint.
     *
     * @return bool Always true for OpenAI
     */
    public function supportsVideoGeneration(): bool
    {
        return true;
    }

    /**
     * Generate a video from a text prompt using OpenAI's Sora API, blocking until ready.
     *
     * Submits the job, polls /videos/{id} with exponential backoff (1s doubling up to 10s)
     * until Sora finishes, then downloads and returns the finished clip.
     *
     * @param string $prompt Descriptive text prompt for video generation
     * @param array{
     *     model?: string,
     *     aspectRatio?: string,
     *     durationSeconds?: int,
     *     resolution?: string,
     *     fps?: int,
     *     image?: string,
     *     negativePrompt?: string,
     *     seconds?: int,
     *     size?: string,
     *     pollTimeoutSeconds?: int,
     * } $options Generation options (seconds/size are Sora-native aliases of durationSeconds/resolution)
     *
     * @return VideoResponse The generated video with the downloaded MP4 bytes
     *
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limits are exceeded (HTTP 429)
     * @throws ProviderException       When generation fails or times out
     * @throws RuntimeException        When a cURL request itself fails
     */
    public function generateVideo(string $prompt, array $options = []): VideoResponse
    {
        $jobId = $this->startVideo($prompt, $options);
        $timeoutSeconds = $options['pollTimeoutSeconds'] ?? 600;
        $delaySeconds = 1;
        $waited = 0;

        while (true) {
            $status = $this->videoStatus($jobId);

            if ($status->isCompleted()) {
                return $this->fetchVideo($jobId);
            }

            if ($status->isFailed()) {
                throw new ProviderException(
                    sprintf('Sora video generation failed: %s.', $status->error ?? 'unknown error'),
                    'openai',
                );
            }

            if ($waited >= $timeoutSeconds) {
                throw new ProviderException(
                    sprintf('Sora video generation timed out after %d seconds.', $timeoutSeconds),
                    'openai',
                );
            }

            $this->pause($delaySeconds);
            $waited += $delaySeconds;
            $delaySeconds = min($delaySeconds * 2, 10);
        }
    }

    /**
     * Submit a Sora video generation job and return the video id immediately.
     *
     * @param string $prompt  Descriptive text prompt for video generation
     * @param array  $options Generation options (see generateVideo())
     *
     * @return string The Sora video id, used as the job identifier
     *
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limits are exceeded (HTTP 429)
     * @throws ProviderException       When the API returns an error or no id
     * @throws RuntimeException        When the cURL request itself fails
     */
    public function startVideo(string $prompt, array $options = []): string
    {
        $payload = [
            'model' => $options['model'] ?? self::MODEL_SORA_2,
            'prompt' => $prompt,
        ];

        $seconds = $options['seconds'] ?? $options['durationSeconds'] ?? null;
        if ($seconds !== null) {
            $payload['seconds'] = (string) $seconds;
        }

        $size = $options['size'] ?? $options['resolution'] ?? null;
        if ($size !== null) {
            $payload['size'] = $size;
        }

        $response = $this->videoCreateRequest($payload);
        $id = $response['id'] ?? '';

        if ($id === '') {
            throw new ProviderException('Sora did not return a video id.', 'openai');
        }

        return $id;
    }

    /**
     * Poll the status of a submitted Sora video job.
     *
     * @param string $jobId The video id returned by startVideo()
     *
     * @return JobStatus Current status (queued/in_progress map to pending/running)
     *
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limits are exceeded (HTTP 429)
     * @throws ProviderException       When the API returns any other error
     * @throws RuntimeException        When the cURL request itself fails
     */
    public function videoStatus(string $jobId): JobStatus
    {
        $video = $this->videoRetrieveRequest($jobId);
        $status = $video['status'] ?? 'in_progress';

        return match ($status) {
            'completed' => new JobStatus($jobId, JobStatus::COMPLETED),
            'failed' => new JobStatus($jobId, JobStatus::FAILED, null, $video['error']['message'] ?? 'unknown error'),
            'queued' => new JobStatus($jobId, JobStatus::PENDING),
            default => new JobStatus($jobId, JobStatus::RUNNING),
        };
    }

    /**
     * Retrieve the finished video for a completed Sora job.
     *
     * @param string $jobId The video id returned by startVideo()
     *
     * @return VideoResponse The generated video with the downloaded MP4 bytes
     *
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limits are exceeded (HTTP 429)
     * @throws ProviderException       When the job is unfinished or failed
     * @throws RuntimeException        When the cURL request itself fails
     */
    public function fetchVideo(string $jobId): VideoResponse
    {
        $video = $this->videoRetrieveRequest($jobId);
        $status = $video['status'] ?? 'in_progress';

        if ($status === 'failed') {
            throw new ProviderException(
                sprintf('Sora video generation failed: %s.', $video['error']['message'] ?? 'unknown error'),
                'openai',
            );
        }

        if ($status !== 'completed') {
            throw new ProviderException(
                sprintf('Video job "%s" is not complete yet.', $jobId),
                'openai',
            );
        }

        $bytes = $this->videoContentRequest($jobId);
        $duration = isset($video['seconds']) ? (float) $video['seconds'] : null;

        return VideoResponse::fromBytes($bytes, (string) ($video['model'] ?? self::MODEL_SORA_2), 'video/mp4', $duration);
    }

    /**
     * Pause between poll attempts. Isolated so tests can override it to no-op.
     *
     * @param int $seconds Seconds to sleep
     */
    protected function pause(int $seconds): void
    {
        sleep($seconds);
    }

    /**
     * Create a Sora video job (POST /videos). Protected so test doubles can stub it.
     *
     * @param array<string, mixed> $payload The JSON-encodable request body
     *
     * @return array<string, mixed> The decoded video object
     *
     * @throws RuntimeException        When a cURL transport error occurs
     * @throws ProviderException       When the API returns an error status
     * @throws AuthenticationException When the API key is invalid (401)
     * @throws RateLimitException      When rate limits are exceeded (429)
     *
     * @codeCoverageIgnore
     */
    protected function videoCreateRequest(array $payload): array
    {
        $url = $this->buildUrl('/videos');
        $ch = curl_init($url);

        $rawHeaders = '';
        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($payload),
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => array_merge(
                ['Content-Type: application/json'],
                $this->getAuthHeaders(),
            ),
            CURLOPT_HEADERFUNCTION => function ($ch, $header) use (&$rawHeaders) {
                $rawHeaders .= $header;

                return strlen($header);
            },
        ]);

        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);

        curl_close($ch);

        if ($error !== '') {
            throw new RuntimeException("OpenAI Video API request failed: {$error}");
        }

        $data = json_decode($response, true);
        $this->handleErrorResponse($httpCode, $data, $this->parseResponseHeaders($rawHeaders));

        return $data;
    }

    /**
     * Retrieve a Sora video object (GET /videos/{id}). Protected so test doubles can stub it.
     *
     * @param string $videoId The Sora video id
     *
     * @return array<string, mixed> The decoded video object
     *
     * @throws RuntimeException        When a cURL transport error occurs
     * @throws ProviderException       When the API returns an error status
     * @throws AuthenticationException When the API key is invalid (401)
     * @throws RateLimitException      When rate limits are exceeded (429)
     *
     * @codeCoverageIgnore
     */
    protected function videoRetrieveRequest(string $videoId): array
    {
        $url = $this->buildUrl('/videos/' . $videoId);
        $ch = curl_init($url);

        $rawHeaders = '';
        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => $this->getAuthHeaders(),
            CURLOPT_HEADERFUNCTION => function ($ch, $header) use (&$rawHeaders) {
                $rawHeaders .= $header;

                return strlen($header);
            },
        ]);

        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);

        curl_close($ch);

        if ($error !== '') {
            throw new RuntimeException("OpenAI Video API request failed: {$error}");
        }

        $data = json_decode($response, true);
        $this->handleErrorResponse($httpCode, $data, $this->parseResponseHeaders($rawHeaders));

        return $data;
    }

    /**
     * Download the finished MP4 bytes (GET /videos/{id}/content). Protected so test doubles can stub it.
     *
     * @param string $videoId The Sora video id
     *
     * @return string Raw MP4 bytes
     *
     * @throws RuntimeException        When a cURL transport error occurs
     * @throws ProviderException       When the API returns an error status
     * @throws AuthenticationException When the API key is invalid (401)
     * @throws RateLimitException      When rate limits are exceeded (429)
     *
     * @codeCoverageIgnore
     */
    protected function videoContentRequest(string $videoId): string
    {
        $url = $this->buildUrl('/videos/' . $videoId . '/content');
        $ch = curl_init($url);

        $rawHeaders = '';
        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => $this->getAuthHeaders(),
            CURLOPT_HEADERFUNCTION => function ($ch, $header) use (&$rawHeaders) {
                $rawHeaders .= $header;

                return strlen($header);
            },
        ]);

        /** @var string $response */
        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);

        curl_close($ch);

        if ($error !== '') {
            throw new RuntimeException("OpenAI Video API request failed: {$error}");
        }

        if ($httpCode >= 400) {
            $data = json_decode($response, true);
            $this->handleErrorResponse($httpCode, $data, $this->parseResponseHeaders($rawHeaders));
        }

        return $response;
    }
}
