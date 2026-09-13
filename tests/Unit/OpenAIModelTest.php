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
use PapiAI\OpenAI\OpenAIModel;
use PapiAI\OpenAI\OpenAIProvider;

describe('OpenAIModel', function () {
    it('is the source of truth the old constants alias', function () {
        expect(OpenAIProvider::MODEL_GPT_4O)->toBe(OpenAIModel::Gpt4o->value);
        expect(OpenAIProvider::MODEL_O3_MINI)->toBe(OpenAIModel::O3Mini->value);
        expect(OpenAIProvider::MODEL_SORA_2)->toBe(OpenAIModel::Sora2->value);
        expect(OpenAIProvider::MODEL_GPT_6_ASTRA)->toBe('gpt-6-astra');
    });

    it('returns null for an ID it has not heard of, rather than throwing', function () {
        expect(OpenAIModel::tryFrom('gpt-7'))->toBeNull();
    });

    it('ships unique IDs', function () {
        $ids = array_map(fn (OpenAIModel $m) => $m->value, OpenAIModel::cases());

        expect($ids)->toBe(array_unique($ids));
    });

    describe('retirement', function () {
        it('knows the three that died in 2025', function () {
            expect(OpenAIModel::Gpt45Preview->retiredOn())->toBe('2025-07-14');
            expect(OpenAIModel::O1Preview->retiredOn())->toBe('2025-07-28');
            expect(OpenAIModel::O1Mini->retiredOn())->toBe('2025-10-27');
        });

        it('knows the October batch and where OpenAI sends them', function () {
            foreach ([OpenAIModel::Gpt4Turbo, OpenAIModel::O1, OpenAIModel::O3Mini] as $model) {
                expect($model->retiredOn())->toBe('2026-10-23');
                expect($model->replacement())->toBe(OpenAIModel::Gpt56Sol);
            }
        });

        it('knows Sora goes with no successor', function () {
            expect(OpenAIModel::Sora2->retiredOn())->toBe('2026-09-24');
            expect(OpenAIModel::Sora2->replacement())->toBeNull();
        });

        it('treats gpt-4o as legacy, not deprecated, which is what OpenAI says', function () {
            expect(OpenAIModel::Gpt4o->isDeprecated())->toBeFalse();
        });
    });

    describe('effort levels', function () {
        it('offers the full six on GPT-6 and the 5.6 trio', function () {
            $full = [Effort::None, Effort::Low, Effort::Medium, Effort::High, Effort::ExtraHigh, Effort::Maximum];

            expect(OpenAIModel::Gpt6Astra->effortLevels())->toBe($full);
            expect(OpenAIModel::Gpt56Luna->effortLevels())->toBe($full);
        });

        it('offers the three middle levels on the o-series and 4o', function () {
            expect(OpenAIModel::O3Mini->effortLevels())->toBe([Effort::Low, Effort::Medium, Effort::High]);
            expect(OpenAIModel::Gpt4o->effortLevels())->toBe([Effort::Low, Effort::Medium, Effort::High]);
        });

        it('offers none on the video models', function () {
            expect(OpenAIModel::Sora2->effortLevels())->toBe([]);
        });
    });
});
