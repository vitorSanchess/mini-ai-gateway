"""PII guardrail for masking sensitive Brazilian & contact data before LLM calls (LGPD)."""

from dataclasses import dataclass, field
import re

# Formatted CPF (e.g., 123.456.789-00) and unformatted 11-digit candidates
_CPF_FORMATTED_RE = re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")
_CPF_RAW_RE = re.compile(r"\b\d{11}\b")

# Formatted CNPJ (e.g., 12.345.678/0001-99) and unformatted 14-digit candidates
_CNPJ_FORMATTED_RE = re.compile(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b")
_CNPJ_RAW_RE = re.compile(r"\b\d{14}\b")

# Standard email address pattern
_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")


def _is_valid_cpf(digits: str) -> bool:
    """Validate an 11-digit Brazilian CPF using its two mod-11 check digits."""
    if len(digits) != 11 or digits == digits[0] * 11:
        return False

    first_sum = sum(int(digits[i]) * (10 - i) for i in range(9))
    first_check = (first_sum * 10) % 11
    first_check = 0 if first_check == 10 else first_check
    if first_check != int(digits[9]):
        return False

    second_sum = sum(int(digits[i]) * (11 - i) for i in range(10))
    second_check = (second_sum * 10) % 11
    second_check = 0 if second_check == 10 else second_check
    return second_check == int(digits[10])


def _is_valid_cnpj(digits: str) -> bool:
    """Validate a 14-digit Brazilian CNPJ using its two mod-11 check digits."""
    if len(digits) != 14 or digits == digits[0] * 14:
        return False

    weights_1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    rem_1 = sum(int(digits[i]) * weights_1[i] for i in range(12)) % 11
    check_1 = 0 if rem_1 < 2 else 11 - rem_1
    if check_1 != int(digits[12]):
        return False

    weights_2 = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    rem_2 = sum(int(digits[i]) * weights_2[i] for i in range(13)) % 11
    check_2 = 0 if rem_2 < 2 else 11 - rem_2
    return check_2 == int(digits[13])


@dataclass(frozen=True)
class GuardrailResult:
    """Sanitized text plus aggregate redaction counts (never stores raw PII)."""

    sanitized_prompt: str
    sanitized_system_prompt: str | None
    pii_detected: bool
    redactions: dict[str, int] = field(default_factory=dict)


class PIIGuardrail:
    """Masks CPF, CNPJ, and email addresses before text leaves the gateway."""

    def _redact_text(self, text: str, counts: dict[str, int]) -> str:
        def _count_and_replace(
            pattern: re.Pattern[str],
            placeholder: str,
            pii_type: str,
            source: str,
            validator=None,
        ) -> str:
            def _repl(match: re.Match[str]) -> str:
                token = match.group(0)
                if validator is not None and not validator(token):
                    return token
                counts[pii_type] = counts.get(pii_type, 0) + 1
                return placeholder

            return pattern.sub(_repl, source)

        # Mask CNPJ before CPF so 14-digit numbers aren't partially inspected
        sanitized = _count_and_replace(
            _CNPJ_FORMATTED_RE, "[CNPJ_REDACTED]", "cnpj", text
        )
        sanitized = _count_and_replace(
            _CNPJ_RAW_RE, "[CNPJ_REDACTED]", "cnpj", sanitized, validator=_is_valid_cnpj
        )
        sanitized = _count_and_replace(
            _CPF_FORMATTED_RE, "[CPF_REDACTED]", "cpf", sanitized
        )
        sanitized = _count_and_replace(
            _CPF_RAW_RE, "[CPF_REDACTED]", "cpf", sanitized, validator=_is_valid_cpf
        )
        sanitized = _count_and_replace(
            _EMAIL_RE, "[EMAIL_REDACTED]", "email", sanitized
        )
        return sanitized

    def sanitize(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> GuardrailResult:
        """Return sanitized prompt/system_prompt and counts of masked PII types."""
        counts: dict[str, int] = {}
        clean_prompt = self._redact_text(prompt, counts)
        clean_system = (
            self._redact_text(system_prompt, counts)
            if system_prompt is not None
            else None
        )
        return GuardrailResult(
            sanitized_prompt=clean_prompt,
            sanitized_system_prompt=clean_system,
            pii_detected=bool(counts),
            redactions=counts,
        )
