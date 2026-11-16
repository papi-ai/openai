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

use PapiAI\Core\Message;
use PapiAI\Core\StreamChunk;
use PapiAI\Core\ToolCall;
use PapiAI\OpenAI\OpenAIProvider;

/**
 * Captures the request and replays a canned Responses API body, so the wire format can be
 * asserted without HTTP.
 */
class TestableResponsesProvider extends OpenAIProvider
{
    public array $lastPayload = [];
    public string $lastPath = '';
    public array $fakeResponse = ['output' => [], 'usage' => []];
    public array $fakeStreamEvents = [];

    protected function request(array $payload): array
    {
        $this->lastPayload = $payload;
        $this->lastPath = $this->chatPath();

        return $this->fakeResponse;
    }

    protected function streamRequest(array $payload): Generator
    {
        $this->lastPayload = $payload;
        $this->lastPath = $this->chatPath();

        yield from $this->fakeStreamEvents;
    }
}

describe('OpenAIProvider on the Responses API', function () {
    beforeEach(function () {
        $this->provider = new TestableResponsesProvider('test-api-key');
        $this->chat = fn (array $messages, array $options = []) => $this->provider->chat($messages, $options);
        $this->textResponse = fn (string $text) => [
            'id' => 'resp_1',
            'status' => 'completed',
            'output' => [
                ['type' => 'reasoning', 'id' => 'rs_1', 'summary' => []],
                ['type' => 'message', 'role' => 'assistant', 'status' => 'completed', 'content' => [
                    ['type' => 'output_text', 'text' => $text, 'annotations' => []],
                ]],
            ],
            'usage' => ['input_tokens' => 12, 'output_tokens' => 7, 'total_tokens' => 19],
        ];
    });

    describe('request shape', function () {
        it('posts to /responses, which is the only surface that takes tools and reasoning together', function () {
            ($this->chat)([Message::user('Hi')]);

            expect($this->provider->lastPath)->toBe('/responses');
        });

        it('sends the conversation as input items and the system message as instructions', function () {
            ($this->chat)([Message::system('Be terse.'), Message::user('Hi')]);

            expect($this->provider->lastPayload['instructions'])->toBe('Be terse.');
            expect($this->provider->lastPayload['input'])->toBe([
                ['role' => 'user', 'content' => 'Hi'],
            ]);
            expect($this->provider->lastPayload)->not->toHaveKey('messages');
        });

        it('keeps the request stateless, since papi resends the whole history itself', function () {
            ($this->chat)([Message::user('Hi')]);

            expect($this->provider->lastPayload['store'])->toBeFalse();
        });

        it('names the output ceiling max_output_tokens', function () {
            ($this->chat)([Message::user('Hi')], ['maxTokens' => 512]);

            expect($this->provider->lastPayload['max_output_tokens'])->toBe(512);
            expect($this->provider->lastPayload)->not->toHaveKey('max_completion_tokens');
        });

        it('declares tools flat, without the function wrapper Chat Completions used', function () {
            ($this->chat)([Message::user('Hi')], ['tools' => [
                ['name' => 'get_weather', 'description' => 'Weather', 'parameters' => ['type' => 'object', 'properties' => []]],
            ]]);

            expect($this->provider->lastPayload['tools'])->toBe([[
                'type' => 'function',
                'name' => 'get_weather',
                'description' => 'Weather',
                'parameters' => ['type' => 'object', 'properties' => []],
            ]]);
        });

        it('puts a JSON schema under text.format', function () {
            ($this->chat)([Message::user('Hi')], ['outputSchema' => ['type' => 'object', 'properties' => []]]);

            expect($this->provider->lastPayload['text']['format'])->toBe([
                'type' => 'json_schema',
                'name' => 'response',
                'schema' => ['type' => 'object', 'properties' => []],
            ]);
            expect($this->provider->lastPayload)->not->toHaveKey('response_format');
        });

        it('replays a prior assistant turn as a message item plus one function_call item per call', function () {
            ($this->chat)([
                Message::user('Weather?'),
                Message::assistant('Checking.', [new ToolCall('call_1', 'get_weather', ['city' => 'Lisbon'])]),
                Message::toolResult('call_1', '{"temp": 21}'),
            ]);

            expect($this->provider->lastPayload['input'])->toBe([
                ['role' => 'user', 'content' => 'Weather?'],
                ['role' => 'assistant', 'content' => [['type' => 'output_text', 'text' => 'Checking.']]],
                ['type' => 'function_call', 'call_id' => 'call_1', 'name' => 'get_weather', 'arguments' => '{"city":"Lisbon"}'],
                ['type' => 'function_call_output', 'call_id' => 'call_1', 'output' => '{"temp": 21}'],
            ]);
        });

        it('sends multimodal content as input_text and input_image parts', function () {
            ($this->chat)([Message::user([
                ['type' => 'text', 'text' => 'What is this?'],
                ['type' => 'image', 'source' => ['type' => 'url', 'url' => 'https://example.com/a.png']],
                ['type' => 'image', 'source' => ['type' => 'base64', 'media_type' => 'image/png', 'data' => 'AAAA']],
            ])]);

            expect($this->provider->lastPayload['input'][0]['content'])->toBe([
                ['type' => 'input_text', 'text' => 'What is this?'],
                ['type' => 'input_image', 'image_url' => 'https://example.com/a.png'],
                ['type' => 'input_image', 'image_url' => 'data:image/png;base64,AAAA'],
            ]);
        });
    });

    describe('response parsing', function () {
        it('reads the text from the message item and ignores reasoning items', function () {
            $this->provider->fakeResponse = ($this->textResponse)('Hello back');

            $response = ($this->chat)([Message::user('Hi')]);

            expect($response->text)->toBe('Hello back');
            expect($response->toolCalls)->toBe([]);
        });

        it('normalises usage, which is already in the input_tokens vocabulary', function () {
            $this->provider->fakeResponse = ($this->textResponse)('ok');

            $response = ($this->chat)([Message::user('Hi')]);

            expect($response->usage->inputTokens)->toBe(12);
            expect($response->usage->outputTokens)->toBe(7);
        });

        it('turns function_call items into ToolCalls keyed by call_id', function () {
            $this->provider->fakeResponse = [
                'status' => 'completed',
                'output' => [
                    ['type' => 'function_call', 'id' => 'fc_1', 'call_id' => 'call_9', 'name' => 'get_weather', 'arguments' => '{"city":"Lisbon"}', 'status' => 'completed'],
                ],
                'usage' => ['input_tokens' => 1, 'output_tokens' => 1],
            ];

            $response = ($this->chat)([Message::user('Weather?')]);

            expect($response->toolCalls)->toHaveCount(1);
            expect($response->toolCalls[0]->id)->toBe('call_9');
            expect($response->toolCalls[0]->name)->toBe('get_weather');
            expect($response->toolCalls[0]->arguments)->toBe(['city' => 'Lisbon']);
            expect($response->stopReason)->toBe('tool_calls');
        });

        it('reports a completed response as stop and a truncated one by its reason', function () {
            $this->provider->fakeResponse = ($this->textResponse)('done');
            expect(($this->chat)([Message::user('Hi')])->stopReason)->toBe('stop');

            $this->provider->fakeResponse = [
                'status' => 'incomplete',
                'incomplete_details' => ['reason' => 'max_output_tokens'],
                'output' => [['type' => 'message', 'role' => 'assistant', 'content' => [['type' => 'output_text', 'text' => 'par']]]],
                'usage' => [],
            ];
            expect(($this->chat)([Message::user('Hi')])->stopReason)->toBe('max_output_tokens');
        });
    });

    describe('streaming', function () {
        it('yields text deltas and completes on response.completed', function () {
            $this->provider->fakeStreamEvents = [
                ['type' => 'response.created', 'response' => ['id' => 'resp_1']],
                ['type' => 'response.output_text.delta', 'delta' => 'Hel'],
                ['type' => 'response.output_text.delta', 'delta' => 'lo'],
                ['type' => 'response.output_text.done', 'text' => 'Hello'],
                ['type' => 'response.completed', 'response' => ['status' => 'completed']],
            ];

            $chunks = iterator_to_array($this->provider->stream([Message::user('Hi')]));

            expect($this->provider->lastPayload['stream'])->toBeTrue();
            expect(array_map(fn (StreamChunk $c) => $c->text, $chunks))->toBe(['Hel', 'lo', '']);
            expect(end($chunks)->isComplete)->toBeTrue();
        });

        it('also completes on an incomplete or failed response, so the loop never hangs', function () {
            foreach (['response.incomplete', 'response.failed'] as $terminal) {
                $this->provider->fakeStreamEvents = [['type' => $terminal, 'response' => []]];

                $chunks = iterator_to_array($this->provider->stream([Message::user('Hi')]));

                expect(end($chunks)->isComplete)->toBeTrue();
            }
        });
    });
});
