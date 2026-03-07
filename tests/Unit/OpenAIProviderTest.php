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

use PapiAI\Core\AudioResponse;
use PapiAI\Core\Contracts\EmbeddingProviderInterface;
use PapiAI\Core\Contracts\ProviderInterface;
use PapiAI\Core\Contracts\TextToSpeechProviderInterface;
use PapiAI\Core\Contracts\TranscriptionProviderInterface;
use PapiAI\Core\EmbeddingResponse;
use PapiAI\Core\Message;
use PapiAI\Core\Response;
use PapiAI\Core\StreamChunk;
use PapiAI\Core\ToolCall;
use PapiAI\Core\TranscriptionResponse;
use PapiAI\OpenAI\OpenAIProvider;

/**
 * Test subclass that stubs HTTP methods for unit testing.
 */
class TestableOpenAIProvider extends OpenAIProvider
{
    public array $lastPayload = [];
    public array $lastEmbeddingPayload = [];
    public array $lastAudioPayload = [];
    public string $lastAudioPath = '';
    public array $lastTranscriptionFields = [];
    public array $fakeResponse = [];
    public array $fakeEmbeddingResponse = [];
    public array $fakeStreamEvents = [];
    public string $fakeAudioResponse = '';
    public array $fakeTranscriptionResponse = [];

    protected function request(array $payload): array
    {
        $this->lastPayload = $payload;

        return $this->fakeResponse;
    }

    protected function embeddingRequest(array $payload): array
    {
        $this->lastEmbeddingPayload = $payload;

        return $this->fakeEmbeddingResponse;
    }

    protected function streamRequest(array $payload): Generator
    {
        $this->lastPayload = $payload;

        foreach ($this->fakeStreamEvents as $event) {
            yield $event;
        }
    }

    protected function audioRequest(array $payload): string
    {
        $this->lastAudioPayload = $payload;

        return $this->fakeAudioResponse;
    }

    protected function transcriptionRequest(string $audioPath, array $fields): array
    {
        $this->lastAudioPath = $audioPath;
        $this->lastTranscriptionFields = $fields;

        return $this->fakeTranscriptionResponse;
    }
}

describe('OpenAIProvider', function () {
    beforeEach(function () {
        $this->provider = new TestableOpenAIProvider('test-api-key');
    });

    describe('construction', function () {
        it('implements ProviderInterface', function () {
            expect($this->provider)->toBeInstanceOf(ProviderInterface::class);
        });

        it('returns openai as name', function () {
            expect($this->provider->getName())->toBe('openai');
        });
    });

    describe('capabilities', function () {
        it('supports tools', function () {
            expect($this->provider->supportsTool())->toBeTrue();
        });

        it('supports vision', function () {
            expect($this->provider->supportsVision())->toBeTrue();
        });

        it('supports structured output', function () {
            expect($this->provider->supportsStructuredOutput())->toBeTrue();
        });
    });

    describe('chat', function () {
        it('sends messages and returns a Response', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => ['role' => 'assistant', 'content' => 'Hello back!'],
                        'finish_reason' => 'stop',
                    ],
                ],
                'model' => 'gpt-4o',
                'usage' => ['prompt_tokens' => 10, 'completion_tokens' => 5],
            ];

            $response = $this->provider->chat([Message::user('Hello')]);

            expect($response)->toBeInstanceOf(Response::class);
            expect($response->text)->toBe('Hello back!');
        });

        it('includes system message in messages array', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => ['role' => 'assistant', 'content' => 'OK'],
                        'finish_reason' => 'stop',
                    ],
                ],
                'model' => 'gpt-4o',
                'usage' => ['prompt_tokens' => 10, 'completion_tokens' => 5],
            ];

            $this->provider->chat([
                Message::system('Be helpful'),
                Message::user('Hello'),
            ]);

            // OpenAI passes system as a regular message in the messages array
            expect($this->provider->lastPayload['messages'])->toHaveCount(2);
            expect($this->provider->lastPayload['messages'][0]['role'])->toBe('system');
            expect($this->provider->lastPayload['messages'][0]['content'])->toBe('Be helpful');
            expect($this->provider->lastPayload['messages'][1]['role'])->toBe('user');
        });

        it('uses default model', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => ['role' => 'assistant', 'content' => 'OK'],
                        'finish_reason' => 'stop',
                    ],
                ],
                'model' => 'gpt-4o',
                'usage' => ['prompt_tokens' => 10, 'completion_tokens' => 5],
            ];

            $this->provider->chat([Message::user('Hello')]);

            expect($this->provider->lastPayload['model'])->toBe('gpt-4o');
        });

        it('overrides model and options from parameters', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => ['role' => 'assistant', 'content' => 'OK'],
                        'finish_reason' => 'stop',
                    ],
                ],
                'model' => 'gpt-4-turbo',
                'usage' => ['prompt_tokens' => 10, 'completion_tokens' => 5],
            ];

            $this->provider->chat([Message::user('Hello')], [
                'model' => 'gpt-4-turbo',
                'maxTokens' => 8192,
                'temperature' => 0.5,
                'stopSequences' => ['END'],
            ]);

            expect($this->provider->lastPayload['model'])->toBe('gpt-4-turbo');
            expect($this->provider->lastPayload['max_tokens'])->toBe(8192);
            expect($this->provider->lastPayload['temperature'])->toBe(0.5);
            expect($this->provider->lastPayload['stop'])->toBe(['END']);
        });

        it('includes tools in payload converted to OpenAI format', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => ['role' => 'assistant', 'content' => 'OK'],
                        'finish_reason' => 'stop',
                    ],
                ],
                'model' => 'gpt-4o',
                'usage' => ['prompt_tokens' => 10, 'completion_tokens' => 5],
            ];

            $tools = [
                [
                    'name' => 'get_weather',
                    'description' => 'Get weather',
                    'input_schema' => ['type' => 'object', 'properties' => []],
                ],
            ];

            $this->provider->chat([Message::user('Hello')], ['tools' => $tools]);

            $expected = [
                [
                    'type' => 'function',
                    'function' => [
                        'name' => 'get_weather',
                        'description' => 'Get weather',
                        'parameters' => ['type' => 'object', 'properties' => []],
                    ],
                ],
            ];
            expect($this->provider->lastPayload['tools'])->toBe($expected);
        });

        it('converts tool result messages', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => ['role' => 'assistant', 'content' => 'The weather is sunny'],
                        'finish_reason' => 'stop',
                    ],
                ],
                'model' => 'gpt-4o',
                'usage' => ['prompt_tokens' => 10, 'completion_tokens' => 5],
            ];

            $this->provider->chat([
                Message::user('What is the weather?'),
                Message::assistant('Let me check', [
                    new ToolCall('tc_1', 'get_weather', ['city' => 'London']),
                ]),
                Message::toolResult('tc_1', ['temp' => 20]),
            ]);

            $messages = $this->provider->lastPayload['messages'];
            expect($messages)->toHaveCount(3);

            // Tool result message
            $toolMsg = $messages[2];
            expect($toolMsg['role'])->toBe('tool');
            expect($toolMsg['tool_call_id'])->toBe('tc_1');
        });

        it('converts assistant messages with tool calls', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => ['role' => 'assistant', 'content' => 'Done'],
                        'finish_reason' => 'stop',
                    ],
                ],
                'model' => 'gpt-4o',
                'usage' => ['prompt_tokens' => 10, 'completion_tokens' => 5],
            ];

            $this->provider->chat([
                Message::user('Hello'),
                Message::assistant('Let me help', [
                    new ToolCall('tc_1', 'search', ['q' => 'test']),
                ]),
                Message::toolResult('tc_1', 'result'),
            ]);

            $messages = $this->provider->lastPayload['messages'];
            $assistantMsg = $messages[1];
            expect($assistantMsg['role'])->toBe('assistant');
            expect($assistantMsg['content'])->toBe('Let me help');
            expect($assistantMsg['tool_calls'][0]['id'])->toBe('tc_1');
            expect($assistantMsg['tool_calls'][0]['type'])->toBe('function');
            expect($assistantMsg['tool_calls'][0]['function']['name'])->toBe('search');
            expect($assistantMsg['tool_calls'][0]['function']['arguments'])->toBe('{"q":"test"}');
        });

        it('converts multimodal messages with url images', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => ['role' => 'assistant', 'content' => 'I see a cat'],
                        'finish_reason' => 'stop',
                    ],
                ],
                'model' => 'gpt-4o',
                'usage' => ['prompt_tokens' => 100, 'completion_tokens' => 5],
            ];

            $this->provider->chat([
                Message::userWithImage('What is this?', 'https://example.com/cat.jpg'),
            ]);

            $messages = $this->provider->lastPayload['messages'];
            $content = $messages[0]['content'];
            expect($content)->toBeArray();
            expect($content[0]['type'])->toBe('text');
            expect($content[0]['text'])->toBe('What is this?');
            expect($content[1]['type'])->toBe('image_url');
            expect($content[1]['image_url']['url'])->toBe('https://example.com/cat.jpg');
        });

        it('converts multimodal messages with base64 images', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => ['role' => 'assistant', 'content' => 'I see a cat'],
                        'finish_reason' => 'stop',
                    ],
                ],
                'model' => 'gpt-4o',
                'usage' => ['prompt_tokens' => 100, 'completion_tokens' => 5],
            ];

            $this->provider->chat([
                Message::userWithImage('What is this?', 'base64data', 'image/png'),
            ]);

            $messages = $this->provider->lastPayload['messages'];
            $content = $messages[0]['content'];
            expect($content[1]['type'])->toBe('image_url');
            expect($content[1]['image_url']['url'])->toBe('data:image/png;base64,base64data');
        });

        it('handles response with tool calls', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => [
                            'role' => 'assistant',
                            'content' => 'Let me check',
                            'tool_calls' => [
                                [
                                    'id' => 'call_123',
                                    'type' => 'function',
                                    'function' => [
                                        'name' => 'get_weather',
                                        'arguments' => '{"city":"London"}',
                                    ],
                                ],
                            ],
                        ],
                        'finish_reason' => 'tool_calls',
                    ],
                ],
                'model' => 'gpt-4o',
                'usage' => ['prompt_tokens' => 10, 'completion_tokens' => 20],
            ];

            $response = $this->provider->chat([Message::user('Weather?')]);

            expect($response->hasToolCalls())->toBeTrue();
            expect($response->toolCalls)->toHaveCount(1);
            expect($response->toolCalls[0]->name)->toBe('get_weather');
            expect($response->toolCalls[0]->arguments)->toBe(['city' => 'London']);
        });

        it('includes output schema as response_format', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => ['role' => 'assistant', 'content' => '{"name":"test"}'],
                        'finish_reason' => 'stop',
                    ],
                ],
                'model' => 'gpt-4o',
                'usage' => ['prompt_tokens' => 10, 'completion_tokens' => 5],
            ];

            $schema = ['type' => 'object', 'properties' => ['name' => ['type' => 'string']]];
            $this->provider->chat([Message::user('Hello')], ['outputSchema' => $schema]);

            expect($this->provider->lastPayload['response_format'])->toBe([
                'type' => 'json_schema',
                'json_schema' => [
                    'name' => 'response',
                    'schema' => $schema,
                ],
            ]);
        });
    });

    describe('embed', function () {
        it('implements EmbeddingProviderInterface', function () {
            expect($this->provider)->toBeInstanceOf(EmbeddingProviderInterface::class);
        });

        it('embeds a single string input', function () {
            $this->provider->fakeEmbeddingResponse = [
                'data' => [
                    ['embedding' => [0.1, 0.2, 0.3], 'index' => 0],
                ],
                'model' => 'text-embedding-3-small',
                'usage' => ['prompt_tokens' => 5, 'total_tokens' => 5],
            ];

            $response = $this->provider->embed('Hello world');

            expect($response)->toBeInstanceOf(EmbeddingResponse::class);
            expect($response->embeddings)->toBe([[0.1, 0.2, 0.3]]);
            expect($response->model)->toBe('text-embedding-3-small');
            expect($this->provider->lastEmbeddingPayload['input'])->toBe(['Hello world']);
        });

        it('embeds an array of inputs', function () {
            $this->provider->fakeEmbeddingResponse = [
                'data' => [
                    ['embedding' => [0.1, 0.2], 'index' => 0],
                    ['embedding' => [0.3, 0.4], 'index' => 1],
                ],
                'model' => 'text-embedding-3-small',
                'usage' => ['prompt_tokens' => 10, 'total_tokens' => 10],
            ];

            $response = $this->provider->embed(['Hello', 'World']);

            expect($response->embeddings)->toBe([[0.1, 0.2], [0.3, 0.4]]);
            expect($response->count())->toBe(2);
            expect($this->provider->lastEmbeddingPayload['input'])->toBe(['Hello', 'World']);
        });

        it('uses custom model', function () {
            $this->provider->fakeEmbeddingResponse = [
                'data' => [
                    ['embedding' => [0.1], 'index' => 0],
                ],
                'model' => 'text-embedding-3-large',
                'usage' => ['prompt_tokens' => 5, 'total_tokens' => 5],
            ];

            $response = $this->provider->embed('Hello', ['model' => 'text-embedding-3-large']);

            expect($this->provider->lastEmbeddingPayload['model'])->toBe('text-embedding-3-large');
            expect($response->model)->toBe('text-embedding-3-large');
        });

        it('passes custom dimensions', function () {
            $this->provider->fakeEmbeddingResponse = [
                'data' => [
                    ['embedding' => [0.1, 0.2], 'index' => 0],
                ],
                'model' => 'text-embedding-3-small',
                'usage' => ['prompt_tokens' => 5, 'total_tokens' => 5],
            ];

            $this->provider->embed('Hello', ['dimensions' => 256]);

            expect($this->provider->lastEmbeddingPayload['dimensions'])->toBe(256);
        });

        it('does not include dimensions when not specified', function () {
            $this->provider->fakeEmbeddingResponse = [
                'data' => [
                    ['embedding' => [0.1], 'index' => 0],
                ],
                'model' => 'text-embedding-3-small',
                'usage' => ['prompt_tokens' => 5, 'total_tokens' => 5],
            ];

            $this->provider->embed('Hello');

            expect($this->provider->lastEmbeddingPayload)->not->toHaveKey('dimensions');
        });

        it('parses usage from response', function () {
            $this->provider->fakeEmbeddingResponse = [
                'data' => [
                    ['embedding' => [0.1, 0.2, 0.3], 'index' => 0],
                ],
                'model' => 'text-embedding-3-small',
                'usage' => ['prompt_tokens' => 8, 'total_tokens' => 8],
            ];

            $response = $this->provider->embed('Hello world');

            expect($response->usage)->toBe(['prompt_tokens' => 8, 'total_tokens' => 8]);
            expect($response->getPromptTokens())->toBe(8);
            expect($response->getTotalTokens())->toBe(8);
        });

        it('defaults to text-embedding-3-small model', function () {
            $this->provider->fakeEmbeddingResponse = [
                'data' => [
                    ['embedding' => [0.1], 'index' => 0],
                ],
                'model' => 'text-embedding-3-small',
                'usage' => ['prompt_tokens' => 5, 'total_tokens' => 5],
            ];

            $this->provider->embed('Hello');

            expect($this->provider->lastEmbeddingPayload['model'])->toBe('text-embedding-3-small');
        });
    });

    describe('stream', function () {
        it('yields StreamChunk for text deltas', function () {
            $this->provider->fakeStreamEvents = [
                ['choices' => [['delta' => ['content' => 'Hello'], 'finish_reason' => null]]],
                ['choices' => [['delta' => ['content' => ' world'], 'finish_reason' => null]]],
                ['choices' => [['delta' => [], 'finish_reason' => 'stop']]],
            ];

            $chunks = [];
            foreach ($this->provider->stream([Message::user('Hi')]) as $chunk) {
                $chunks[] = $chunk;
            }

            expect($chunks)->toHaveCount(3);
            expect($chunks[0])->toBeInstanceOf(StreamChunk::class);
            expect($chunks[0]->text)->toBe('Hello');
            expect($chunks[1]->text)->toBe(' world');
            expect($chunks[2]->isComplete)->toBeTrue();
        });

        it('sets stream flag in payload', function () {
            $this->provider->fakeStreamEvents = [
                ['choices' => [['delta' => [], 'finish_reason' => 'stop']]],
            ];

            iterator_to_array($this->provider->stream([Message::user('Hi')]));

            expect($this->provider->lastPayload['stream'])->toBeTrue();
        });

        it('ignores non-text delta events', function () {
            $this->provider->fakeStreamEvents = [
                ['choices' => [['delta' => ['role' => 'assistant'], 'finish_reason' => null]]],
                ['choices' => [['delta' => ['content' => 'Hi'], 'finish_reason' => null]]],
                ['choices' => [['delta' => [], 'finish_reason' => 'stop']]],
            ];

            $chunks = iterator_to_array($this->provider->stream([Message::user('Hi')]));

            expect($chunks)->toHaveCount(2); // text + complete
        });
    });

    describe('synthesize', function () {
        it('implements TextToSpeechProviderInterface', function () {
            expect($this->provider)->toBeInstanceOf(TextToSpeechProviderInterface::class);
        });

        it('sends default options', function () {
            $this->provider->fakeAudioResponse = 'fake-audio-bytes';

            $this->provider->synthesize('Hello world');

            expect($this->provider->lastAudioPayload['model'])->toBe('tts-1');
            expect($this->provider->lastAudioPayload['input'])->toBe('Hello world');
            expect($this->provider->lastAudioPayload['voice'])->toBe('alloy');
            expect($this->provider->lastAudioPayload['response_format'])->toBe('mp3');
            expect($this->provider->lastAudioPayload['speed'])->toBe(1.0);
        });

        it('accepts custom voice, model, format, and speed', function () {
            $this->provider->fakeAudioResponse = 'fake-audio-bytes';

            $this->provider->synthesize('Hello', [
                'model' => 'tts-1-hd',
                'voice' => 'nova',
                'format' => 'opus',
                'speed' => 1.5,
            ]);

            expect($this->provider->lastAudioPayload['model'])->toBe('tts-1-hd');
            expect($this->provider->lastAudioPayload['voice'])->toBe('nova');
            expect($this->provider->lastAudioPayload['response_format'])->toBe('opus');
            expect($this->provider->lastAudioPayload['speed'])->toBe(1.5);
        });

        it('includes instructions when provided', function () {
            $this->provider->fakeAudioResponse = 'fake-audio-bytes';

            $this->provider->synthesize('Hello', [
                'instructions' => 'Speak slowly and clearly',
            ]);

            expect($this->provider->lastAudioPayload['instructions'])->toBe('Speak slowly and clearly');
        });

        it('does not include instructions when not provided', function () {
            $this->provider->fakeAudioResponse = 'fake-audio-bytes';

            $this->provider->synthesize('Hello');

            expect($this->provider->lastAudioPayload)->not->toHaveKey('instructions');
        });

        it('returns an AudioResponse', function () {
            $this->provider->fakeAudioResponse = 'fake-audio-bytes';

            $response = $this->provider->synthesize('Hello world');

            expect($response)->toBeInstanceOf(AudioResponse::class);
            expect($response->data)->toBe('fake-audio-bytes');
            expect($response->format)->toBe('mp3');
            expect($response->model)->toBe('tts-1');
        });
    });

    describe('transcribe', function () {
        it('implements TranscriptionProviderInterface', function () {
            expect($this->provider)->toBeInstanceOf(TranscriptionProviderInterface::class);
        });

        it('sends basic transcription request', function () {
            $this->provider->fakeTranscriptionResponse = [
                'text' => 'Hello world',
                'language' => 'en',
                'duration' => 2.5,
                'segments' => [],
            ];

            $this->provider->transcribe('/path/to/audio.mp3');

            expect($this->provider->lastAudioPath)->toBe('/path/to/audio.mp3');
            expect($this->provider->lastTranscriptionFields['model'])->toBe('whisper-1');
            expect($this->provider->lastTranscriptionFields['response_format'])->toBe('verbose_json');
            expect($this->provider->lastTranscriptionFields['timestamp_granularities[]'])->toBe('segment');
        });

        it('includes language option when provided', function () {
            $this->provider->fakeTranscriptionResponse = [
                'text' => 'Bonjour le monde',
                'language' => 'fr',
                'duration' => 1.5,
                'segments' => [],
            ];

            $this->provider->transcribe('/path/to/audio.mp3', ['language' => 'fr']);

            expect($this->provider->lastTranscriptionFields['language'])->toBe('fr');
        });

        it('parses segments from response', function () {
            $this->provider->fakeTranscriptionResponse = [
                'text' => 'Hello world',
                'language' => 'en',
                'duration' => 2.5,
                'segments' => [
                    ['start' => 0.0, 'end' => 1.2, 'text' => 'Hello'],
                    ['start' => 1.2, 'end' => 2.5, 'text' => ' world'],
                ],
            ];

            $response = $this->provider->transcribe('/path/to/audio.mp3');

            expect($response->hasSegments())->toBeTrue();
            expect($response->segmentCount())->toBe(2);
            expect($response->segments[0]['start'])->toBe(0.0);
            expect($response->segments[0]['end'])->toBe(1.2);
            expect($response->segments[0]['text'])->toBe('Hello');
        });

        it('returns a TranscriptionResponse', function () {
            $this->provider->fakeTranscriptionResponse = [
                'text' => 'Hello world',
                'language' => 'en',
                'duration' => 2.5,
                'segments' => [],
            ];

            $response = $this->provider->transcribe('/path/to/audio.mp3');

            expect($response)->toBeInstanceOf(TranscriptionResponse::class);
            expect($response->text)->toBe('Hello world');
            expect($response->model)->toBe('whisper-1');
            expect($response->language)->toBe('en');
            expect($response->duration)->toBe(2.5);
        });

        it('includes prompt option when provided', function () {
            $this->provider->fakeTranscriptionResponse = [
                'text' => 'Hello world',
                'language' => 'en',
                'duration' => 2.5,
                'segments' => [],
            ];

            $this->provider->transcribe('/path/to/audio.mp3', ['prompt' => 'Context hint']);

            expect($this->provider->lastTranscriptionFields['prompt'])->toBe('Context hint');
        });
    });
});
