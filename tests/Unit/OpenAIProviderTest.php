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

            expect($response->usage->toArray())->toBe(['prompt_tokens' => 8, 'total_tokens' => 8]);
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
