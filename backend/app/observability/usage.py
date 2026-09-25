from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class UsageRecord(BaseModel):
    provider: str
    model: str
    operation: str
    quantities: dict[str, float] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True, extra="forbid")


class PricingRate(BaseModel):
    provider: str
    model: str
    operation: str
    unit: str
    usd_per_unit: Decimal = Field(ge=0)
    effective_date: date
    source_note: str

    model_config = ConfigDict(frozen=True, extra="forbid")


class CostEstimate(BaseModel):
    estimated_usd: Decimal | None
    unavailable: tuple[str, ...] = ()

    model_config = ConfigDict(frozen=True)


class PricingCatalog:
    def __init__(
        self,
        rates: list[PricingRate] | None = None,
        *,
        version: str = "unversioned",
    ) -> None:
        self.version = version
        self._rates = {
            (rate.provider, rate.model, rate.operation, rate.unit): rate
            for rate in (rates or [])
        }

    def get(
        self,
        *,
        provider: str,
        model: str,
        operation: str,
        unit: str,
    ) -> PricingRate | None:
        return self._rates.get((provider, model, operation, unit))


class CostEstimator:
    def __init__(self, catalog: PricingCatalog) -> None:
        self._catalog = catalog

    def estimate(self, records: list[UsageRecord]) -> CostEstimate:
        total = Decimal(0)
        unavailable: list[str] = []
        for record in records:
            for unit, quantity in record.quantities.items():
                rate = self._catalog.get(
                    provider=record.provider,
                    model=record.model,
                    operation=record.operation,
                    unit=unit,
                )
                if rate is None:
                    unavailable.append(
                        f"{record.provider}/{record.model}/"
                        f"{record.operation}/{unit}"
                    )
                    continue
                total += Decimal(str(quantity)) * rate.usd_per_unit
        if unavailable:
            return CostEstimate(
                estimated_usd=None,
                unavailable=tuple(sorted(set(unavailable))),
            )
        return CostEstimate(estimated_usd=total)


OFFICIAL_GROQ_PRICING = PricingCatalog(
    [
        PricingRate(
            provider="groq",
            model="openai/gpt-oss-20b",
            operation="llm",
            unit="input_token",
            usd_per_unit=Decimal("0.075") / Decimal(1000000),
            effective_date=date(2026, 9, 24),
            source_note=(
                "Groq model documentation: "
                "https://console.groq.com/docs/model/openai/gpt-oss-20b"
            ),
        ),
        PricingRate(
            provider="groq",
            model="openai/gpt-oss-20b",
            operation="llm",
            unit="output_token",
            usd_per_unit=Decimal("0.30") / Decimal(1000000),
            effective_date=date(2026, 9, 24),
            source_note=(
                "Groq model documentation: "
                "https://console.groq.com/docs/model/openai/gpt-oss-20b"
            ),
        ),
        PricingRate(
            provider="groq",
            model="whisper-large-v3-turbo",
            operation="stt",
            unit="audio_second",
            usd_per_unit=Decimal("0.04") / Decimal(3600),
            effective_date=date(2026, 9, 24),
            source_note=(
                "Groq model documentation: "
                "https://console.groq.com/docs/model/whisper-large-v3-turbo"
            ),
        ),
        PricingRate(
            provider="groq",
            model="canopylabs/orpheus-v1-english",
            operation="tts",
            unit="character",
            usd_per_unit=Decimal(22) / Decimal(1000000),
            effective_date=date(2026, 9, 24),
            source_note=(
                "Groq TTS documentation: "
                "https://console.groq.com/docs/text-to-speech/orpheus"
            ),
        ),
    ],
    version="groq-2026-09-24",
)
