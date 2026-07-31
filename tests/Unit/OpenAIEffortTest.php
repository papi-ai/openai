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

use PapiAI\Core\Effort;
use PapiAI\Core\Message;
use PapiAI\OpenAI\OpenAIProvider;

/**
 * Captures the request payload so effort mapping can be asserted without HTTP.
 */
class TestableOpenAIEffortProvider extends OpenAIProvider
{
    public array $lastPayload = [];

    protected function request(array $payload): array
    {
        $this->lastPayload = $payload;

        return ['choices' => [['message' => ['role' => 'assistant', 'content' => 'ok'], 'finish_reason' => 'stop']]];
    }
}

describe('OpenAIProvider reasoning effort', function () {
    beforeEach(function () {
        $this->provider = new TestableOpenAIEffortProvider('test-api-key');
        $this->chat = fn (array $options) => $this->provider->chat([Message::user('hi')], $options);
    });

    it('passes the level straight through, which is OpenAI\'s own vocabulary', function () {
        foreach (['low', 'medium', 'high'] as $level) {
            ($this->chat)(['effort' => $level]);

            expect($this->provider->lastPayload['reasoning_effort'])->toBe($level);
        }
    });

    it('sends nothing when the caller does not ask', function () {
        ($this->chat)([]);

        expect($this->provider->lastPayload)->not->toHaveKey('reasoning_effort');
    });

    it('rejects a level it does not recognise, before any HTTP call', function () {
        expect(fn () => ($this->chat)(['effort' => 'enormous']))
            ->toThrow(InvalidArgumentException::class, 'enormous');

        expect($this->provider->lastPayload)->toBe([]);
    });

    it('uses OpenAI\'s own spellings for the top two levels', function () {
        ($this->chat)(['effort' => 'extra-high', 'model' => 'gpt-5.6-terra']);
        expect($this->provider->lastPayload['reasoning_effort'])->toBe('xhigh');

        ($this->chat)(['effort' => 'maximum', 'model' => 'gpt-5.6-terra']);
        expect($this->provider->lastPayload['reasoning_effort'])->toBe('max');
    });

    it('narrows to what the target model actually accepts', function () {
        // The API rejects an unknown level outright, so overshooting is a 400, not a downgrade.
        $cases = [
            // model, asked for, expected
            ['gpt-5.6-sol', 'maximum', 'max'],
            ['gpt-5.5', 'maximum', 'xhigh'],       // no max before 5.6
            ['gpt-5', 'maximum', 'high'],          // no xhigh on the original GPT-5
            ['o3-mini', 'maximum', 'high'],        // o-series tops out at high
            ['gpt-5', 'minimal', 'minimal'],       // only the original GPT-5 has minimal
            ['gpt-5.6-sol', 'minimal', 'low'],     // 5.6 dropped minimal; ties round up, so low
            ['o3-mini', 'none', 'low'],            // o-series cannot switch reasoning off
        ];

        foreach ($cases as [$model, $asked, $expected]) {
            ($this->chat)(['effort' => $asked, 'model' => $model]);

            expect($this->provider->lastPayload['reasoning_effort'])
                ->toBe($expected, sprintf('%s asked for %s', $model, $asked));
        }
    });

    it('falls back to the provider default when the call does not say', function () {
        $provider = new TestableOpenAIEffortProvider('k', 'gpt-5', 4096, null, null, Effort::Minimal);
        $provider->chat([Message::user('hi')], []);

        expect($provider->lastPayload['reasoning_effort'])->toBe('minimal');
    });

    it('lets the call override the provider default', function () {
        $provider = new TestableOpenAIEffortProvider('k', 'gpt-4o', 4096, null, null, Effort::Minimal);
        $provider->chat([Message::user('hi')], ['effort' => 'high']);

        expect($provider->lastPayload['reasoning_effort'])->toBe('high');
    });
});
