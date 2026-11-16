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
use PapiAI\Core\Exception\ConfigurationException;
use PapiAI\Core\Factory\ComposerDiscovery;
use PapiAI\OpenAI\OpenAIProvider;
use PapiAI\OpenAI\OpenAIProviderFactory;

describe('OpenAIProviderFactory', function () {
    beforeEach(function () {
        $this->factory = new OpenAIProviderFactory();
        $this->property = fn (object $provider, string $name): mixed => (fn () => $this->{$name})->call($provider);
    });

    it('is named after its package', function () {
        expect($this->factory->name())->toBe('openai');
        expect($this->factory->package())->toBe('papi-ai/openai');
    });

    it('is declared in composer.json, where discovery looks for it', function () {
        $manifest = json_decode((string) file_get_contents(__DIR__ . '/../../composer.json'), true);

        expect($manifest['name'])->toBe($this->factory->package());
        expect($manifest['type'])->toBe(ComposerDiscovery::PACKAGE_TYPE);
        expect($manifest['extra']['papi']['factory'])->toBe(OpenAIProviderFactory::class);
    });

    it('takes what the constructor takes', function () {
        expect(array_map(fn ($setting) => $setting->name, $this->factory->settings()))->toBe(['api_key', 'model', 'max_tokens', 'base_url', 'api_version', 'effort']);
    });

    it('builds the provider from its settings', function () {
        $provider = $this->factory->create([
            'api_key' => 'k',
            'model' => 'custom-model',
            'max_tokens' => 1024,
            'base_url' => 'http://proxy.test/v1',
            'api_version' => '2025-01-01',
            'effort' => 'high',
        ]);

        expect($provider)->toBeInstanceOf(OpenAIProvider::class);
        expect(($this->property)($provider, 'apiKey'))->toBe('k');
        expect(($this->property)($provider, 'defaultModel'))->toBe('custom-model');
        expect(($this->property)($provider, 'defaultMaxTokens'))->toBe(1024);
        expect(($this->property)($provider, 'baseUrl'))->toBe('http://proxy.test/v1');
        expect(($this->property)($provider, 'apiVersion'))->toBe('2025-01-01');
        expect(($this->property)($provider, 'defaultEffort'))->toBe(Effort::High);
    });

    it('leaves what it is not given to the provider\'s own defaults', function () {
        $provider = $this->factory->create(['api_key' => 'k']);

        expect(($this->property)($provider, 'defaultModel'))->toBe(OpenAIProvider::MODEL_GPT_6_ASTRA);
    });

    it('refuses to build without its required settings', function () {
        expect(fn () => $this->factory->create([]))
            ->toThrow(ConfigurationException::class, 'Provider "openai" needs the "api_key" setting');
    });
});
