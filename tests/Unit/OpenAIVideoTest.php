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

use PapiAI\Core\Contracts\VideoProviderInterface;
use PapiAI\Core\Exception\ProviderException;
use PapiAI\Core\VideoResponse;
use PapiAI\OpenAI\OpenAIProvider;

/**
 * Test subclass that stubs the Sora HTTP + sleep seams for unit testing.
 */
class TestableSoraProvider extends OpenAIProvider
{
    public array $lastCreatePayload = [];
    public array $createResponse = ['id' => 'video_abc'];

    /** @var array<int, array> Queue of videoRetrieveRequest() responses, consumed in order */
    public array $retrieveResponses = [];
    public string $fakeContent = 'fake-mp4-bytes';
    public int $pauseCalls = 0;

    protected function videoCreateRequest(array $payload): array
    {
        $this->lastCreatePayload = $payload;

        return $this->createResponse;
    }

    protected function videoRetrieveRequest(string $videoId): array
    {
        return array_shift($this->retrieveResponses) ?? [];
    }

    protected function videoContentRequest(string $videoId): string
    {
        return $this->fakeContent;
    }

    protected function pause(int $seconds): void
    {
        ++$this->pauseCalls;
    }
}

describe('OpenAIProvider video generation', function () {
    beforeEach(function () {
        $this->provider = new TestableSoraProvider('test-api-key');
    });

    describe('capabilities', function () {
        it('implements VideoProviderInterface', function () {
            expect($this->provider)->toBeInstanceOf(VideoProviderInterface::class);
        });

        it('supports video generation', function () {
            expect($this->provider->supportsVideoGeneration())->toBeTrue();
        });
    });

    describe('startVideo', function () {
        it('creates a job with the mapped payload and returns the id', function () {
            $jobId = $this->provider->startVideo('a cat surfing', [
                'model' => OpenAIProvider::MODEL_SORA_2_PRO,
                'durationSeconds' => 8,
                'resolution' => '1280x720',
            ]);

            expect($jobId)->toBe('video_abc');

            $payload = $this->provider->lastCreatePayload;
            expect($payload['model'])->toBe('sora-2-pro');
            expect($payload['prompt'])->toBe('a cat surfing');
            expect($payload['seconds'])->toBe('8');
            expect($payload['size'])->toBe('1280x720');
        });

        it('defaults to sora-2', function () {
            $this->provider->startVideo('a dog');

            expect($this->provider->lastCreatePayload['model'])->toBe('sora-2');
        });

        it('throws when no id is returned', function () {
            $this->provider->createResponse = [];

            expect(fn () => $this->provider->startVideo('x'))->toThrow(ProviderException::class);
        });
    });

    describe('videoStatus', function () {
        it('maps queued to pending', function () {
            $this->provider->retrieveResponses = [['status' => 'queued']];

            expect($this->provider->videoStatus('video_abc')->isPending())->toBeTrue();
        });

        it('maps in_progress to running', function () {
            $this->provider->retrieveResponses = [['status' => 'in_progress']];

            expect($this->provider->videoStatus('video_abc')->isRunning())->toBeTrue();
        });

        it('maps completed to completed', function () {
            $this->provider->retrieveResponses = [['status' => 'completed']];

            expect($this->provider->videoStatus('video_abc')->isCompleted())->toBeTrue();
        });

        it('maps failed and carries the error message', function () {
            $this->provider->retrieveResponses = [['status' => 'failed', 'error' => ['message' => 'safety block']]];

            $status = $this->provider->videoStatus('video_abc');

            expect($status->isFailed())->toBeTrue();
            expect($status->error)->toBe('safety block');
        });
    });

    describe('fetchVideo', function () {
        it('downloads the mp4 bytes for a completed job', function () {
            $this->provider->retrieveResponses = [
                ['status' => 'completed', 'model' => 'sora-2', 'seconds' => '8'],
            ];

            $video = $this->provider->fetchVideo('video_abc');

            expect($video)->toBeInstanceOf(VideoResponse::class);
            expect($video->data)->toBe('fake-mp4-bytes');
            expect($video->mimeType)->toBe('video/mp4');
            expect($video->model)->toBe('sora-2');
            expect($video->durationSeconds)->toBe(8.0);
        });

        it('throws when the job is not complete', function () {
            $this->provider->retrieveResponses = [['status' => 'in_progress']];

            expect(fn () => $this->provider->fetchVideo('video_abc'))->toThrow(ProviderException::class);
        });

        it('throws when the job failed', function () {
            $this->provider->retrieveResponses = [['status' => 'failed', 'error' => ['message' => 'nope']]];

            expect(fn () => $this->provider->fetchVideo('video_abc'))->toThrow(ProviderException::class);
        });
    });

    describe('generateVideo (blocking)', function () {
        it('polls until completed and returns the finished clip', function () {
            $this->provider->retrieveResponses = [
                ['status' => 'in_progress'],
                ['status' => 'completed'],
                ['status' => 'completed', 'model' => 'sora-2', 'seconds' => '8'],
            ];

            $video = $this->provider->generateVideo('a cat surfing');

            expect($video->data)->toBe('fake-mp4-bytes');
            expect($this->provider->pauseCalls)->toBe(1);
        });

        it('throws when generation fails', function () {
            $this->provider->retrieveResponses = [['status' => 'failed', 'error' => ['message' => 'blocked']]];

            expect(fn () => $this->provider->generateVideo('x'))->toThrow(ProviderException::class);
        });

        it('throws when polling exceeds the timeout', function () {
            $this->provider->retrieveResponses = [['status' => 'in_progress']];

            expect(fn () => $this->provider->generateVideo('x', ['pollTimeoutSeconds' => 0]))
                ->toThrow(ProviderException::class);
        });
    });
});
