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

use PapiAI\Core\Exception\AuthenticationException;
use PapiAI\Core\Exception\ProviderException;
use PapiAI\OpenAI\OpenAIProvider;

/**
 * Opens the reading of a streamed response to the tests, which stand in for the HTTP call.
 */
class TestableOpenAIStreamErrorsProvider extends OpenAIProvider
{
    public function read(int $httpCode, string $transportError, string $body): array
    {
        return iterator_to_array($this->eventsIn($httpCode, $transportError, $body), false);
    }
}

describe('OpenAIProvider streamed responses', function () {
    beforeEach(function () {
        $this->provider = new TestableOpenAIStreamErrorsProvider('test-api-key');
    });

    it('yields the events of a stream the API accepted, up to its end marker', function () {
        $body = "event: delta\ndata: {\"n\":1}\n\ndata: {\"n\":2}\n\ndata: not json\ndata: [DONE]\ndata: {\"n\":3}\n";

        expect($this->provider->read(200, '', $body))->toBe([['n' => 1], ['n' => 2]]);
    });

    it('throws when the API rejects the key, where it used to yield nothing', function () {
        expect(fn () => $this->provider->read(401, '', '{"error":{"message":"Invalid API key"}}'))
            ->toThrow(AuthenticationException::class);
    });

    it('throws for any other refusal, readable or not', function (int $httpCode, string $body) {
        expect(fn () => $this->provider->read($httpCode, '', $body))->toThrow(ProviderException::class);
    })->with([
        'a server error with a JSON body' => [500, '{"error":{"message":"Internal error"}}'],
        'a gateway error that is not JSON' => [502, '<html>Bad Gateway</html>'],
    ]);

    it('throws when the request never reached the API', function () {
        expect(fn () => $this->provider->read(0, 'Could not resolve host', ''))
            ->toThrow(RuntimeException::class, 'OpenAI API request failed: Could not resolve host');
    });
});
