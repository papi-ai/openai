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

use PapiAI\Core\Effort;

/**
 * Every OpenAI model this package knows, and what each one accepts.
 *
 * The model, not the provider, decides which reasoning-effort levels exist, because the API
 * rejects an unknown level with a 400 rather than ignoring it. Putting that here keeps the
 * provider free of string-sniffing and lets a watchdog enumerate what we ship.
 *
 * An ID we have not heard of is not an error: `tryFrom()` returns null and the provider infers
 * the generation from the name, assuming newer rather than older.
 *
 * Retirement dates are OpenAI's published shutdown dates, ISO formatted.
 *
 * @see https://developers.openai.com/api/docs/models
 * @see https://developers.openai.com/api/docs/deprecations
 */
enum OpenAIModel: string
{
    case Gpt6Astra = 'gpt-6-astra';
    case Gpt56Sol = 'gpt-5.6-sol';
    case Gpt56Terra = 'gpt-5.6-terra';
    case Gpt56Luna = 'gpt-5.6-luna';

    /** Legacy but not deprecated. */
    case Gpt4o = 'gpt-4o';
    /** Legacy but not deprecated. */
    case Gpt4oMini = 'gpt-4o-mini';

    /** @deprecated Shuts down 23 October 2026. Use Gpt56Sol. */
    case Gpt4Turbo = 'gpt-4-turbo';
    /** @deprecated Shuts down 23 October 2026. Use Gpt56Sol. */
    case O1 = 'o1';
    /** @deprecated Shuts down 23 October 2026. Use Gpt56Sol. */
    case O3Mini = 'o3-mini';
    /** @deprecated Shut down 14 July 2025; requests fail. */
    case Gpt45Preview = 'gpt-4.5-preview';
    /** @deprecated Shut down 28 July 2025; requests fail. */
    case O1Preview = 'o1-preview';
    /** @deprecated Shut down 27 October 2025; requests fail. */
    case O1Mini = 'o1-mini';

    /** @deprecated Sora and the Videos API shut down 24 September 2026, with no successor. */
    case Sora2 = 'sora-2';
    /** @deprecated Sora and the Videos API shut down 24 September 2026, with no successor. */
    case Sora2Pro = 'sora-2-pro';

    /**
     * The reasoning-effort levels this model accepts, in the neutral vocabulary.
     *
     * `xhigh` arrived with 5.1, `max` with 5.6, and `minimal` existed only on the original GPT-5.
     * Everything older, the o-series included, takes the three middle levels. Empty for the
     * video models, which have no such knob.
     *
     * @return list<Effort>
     */
    public function effortLevels(): array
    {
        return match ($this) {
            self::Gpt6Astra, self::Gpt56Sol, self::Gpt56Terra, self::Gpt56Luna
                => [Effort::None, Effort::Low, Effort::Medium, Effort::High, Effort::ExtraHigh, Effort::Maximum],
            self::Sora2, self::Sora2Pro => [],
            default => [Effort::Low, Effort::Medium, Effort::High],
        };
    }

    /**
     * Whether OpenAI has announced a shutdown for this model.
     */
    public function isDeprecated(): bool
    {
        return $this->retiredOn() !== null;
    }

    /**
     * The announced shutdown date, ISO formatted, or null while the model is not deprecated.
     */
    public function retiredOn(): ?string
    {
        return match ($this) {
            self::Gpt45Preview => '2025-07-14',
            self::O1Preview => '2025-07-28',
            self::O1Mini => '2025-10-27',
            self::Sora2, self::Sora2Pro => '2026-09-24',
            self::Gpt4Turbo, self::O1, self::O3Mini => '2026-10-23',
            default => null,
        };
    }

    /**
     * What OpenAI names as the replacement, when it names one we ship.
     */
    public function replacement(): ?self
    {
        return match ($this) {
            self::Gpt4Turbo, self::O1, self::O3Mini => self::Gpt56Sol,
            default => null,
        };
    }
}
