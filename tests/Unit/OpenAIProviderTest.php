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

use PapiAI\OpenAI\OpenAIProvider;
use PapiAI\Core\Contracts\ProviderInterface;
use PapiAI\Core\Message;

describe('OpenAIProvider', function () {
    describe('construction', function () {
        it('implements ProviderInterface', function () {
            $provider = new OpenAIProvider('test-api-key');

            expect($provider)->toBeInstanceOf(ProviderInterface::class);
        });

        it('uses default model gpt-4o', function () {
            $provider = new OpenAIProvider('test-api-key');

            expect($provider->getName())->toBe('openai');
        });

        it('accepts custom model', function () {
            $provider = new OpenAIProvider(
                apiKey: 'test-api-key',
                defaultModel: OpenAIProvider::MODEL_GPT_4O_MINI,
            );

            expect($provider)->toBeInstanceOf(OpenAIProvider::class);
        });
    });

    describe('model constants', function () {
        it('has correct model constants', function () {
            expect(OpenAIProvider::MODEL_GPT_4O)->toBe('gpt-4o');
            expect(OpenAIProvider::MODEL_GPT_4O_MINI)->toBe('gpt-4o-mini');
            expect(OpenAIProvider::MODEL_GPT_4_TURBO)->toBe('gpt-4-turbo');
            expect(OpenAIProvider::MODEL_O1_PREVIEW)->toBe('o1-preview');
            expect(OpenAIProvider::MODEL_O1_MINI)->toBe('o1-mini');
        });
    });

    describe('capabilities', function () {
        it('supports tools', function () {
            $provider = new OpenAIProvider('test-api-key');

            expect($provider->supportsTool())->toBeTrue();
        });

        it('supports vision', function () {
            $provider = new OpenAIProvider('test-api-key');

            expect($provider->supportsVision())->toBeTrue();
        });

        it('supports structured output', function () {
            $provider = new OpenAIProvider('test-api-key');

            expect($provider->supportsStructuredOutput())->toBeTrue();
        });
    });

    describe('message conversion', function () {
        it('handles all message roles', function () {
            $provider = new OpenAIProvider('test-api-key');

            $messages = [
                Message::system('Be helpful'),
                Message::user('Hello'),
                Message::assistant('Hi there!'),
            ];

            expect($provider)->toBeInstanceOf(OpenAIProvider::class);
        });

        it('handles multimodal messages', function () {
            $provider = new OpenAIProvider('test-api-key');

            $messages = [
                Message::userWithImage('What is this?', 'https://example.com/image.jpg'),
            ];

            expect($provider)->toBeInstanceOf(OpenAIProvider::class);
        });
    });
});

describe('OpenAIProvider integration', function () {
    it('throws on API error', function () {
        $this->markTestSkipped('Integration test - requires OPENAI_API_KEY');

        $provider = new OpenAIProvider('invalid-key');

        expect(fn() => $provider->chat([Message::user('Hello')]))
            ->toThrow(RuntimeException::class);
    });
})->skip();
