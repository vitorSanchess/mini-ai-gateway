"""Unit tests for the LGPD PII guardrail."""

from app.services.guardrails import PIIGuardrail


def test_masks_formatted_cpf_cnpj_and_email() -> None:
    guardrail = PIIGuardrail()
    text = (
        "Cliente CPF 123.456.789-00, empresa CNPJ 12.345.678/0001-99, "
        "contato joao.silva@banco.com.br."
    )

    result = guardrail.sanitize(prompt=text)

    assert result.pii_detected is True
    assert "[CPF_REDACTED]" in result.sanitized_prompt
    assert "[CNPJ_REDACTED]" in result.sanitized_prompt
    assert "[EMAIL_REDACTED]" in result.sanitized_prompt
    assert "123.456.789-00" not in result.sanitized_prompt
    assert "12.345.678/0001-99" not in result.sanitized_prompt
    assert "joao.silva@banco.com.br" not in result.sanitized_prompt
    assert result.redactions == {"cnpj": 1, "cpf": 1, "email": 1}


def test_validates_unformatted_cpf_and_cnpj_check_digits() -> None:
    guardrail = PIIGuardrail()
    # 52998224725 is a mathematically valid CPF; 11222333000181 is a mathematically valid CNPJ
    # 99999999999 and 12345678901234 fail mod-11 check digits and must NOT be masked
    text = "CPF 52998224725, CNPJ 11222333000181, pedido 99999999999, protocolo 12345678901234"

    result = guardrail.sanitize(prompt=text)

    assert result.pii_detected is True
    assert "52998224725" not in result.sanitized_prompt
    assert "11222333000181" not in result.sanitized_prompt
    assert "99999999999" in result.sanitized_prompt
    assert "12345678901234" in result.sanitized_prompt
    assert result.redactions == {"cnpj": 1, "cpf": 1}


def test_sanitizes_system_prompt_and_aggregates_counts() -> None:
    guardrail = PIIGuardrail()
    result = guardrail.sanitize(
        prompt="Fale com maria@empresa.com",
        system_prompt="Responda como suporte@empresa.com para CPF 111.222.333-44",
    )

    assert result.pii_detected is True
    assert "[EMAIL_REDACTED]" in result.sanitized_prompt
    assert result.sanitized_system_prompt is not None
    assert "[EMAIL_REDACTED]" in result.sanitized_system_prompt
    assert "[CPF_REDACTED]" in result.sanitized_system_prompt
    assert result.redactions == {"email": 2, "cpf": 1}


def test_leaves_clean_prompt_unchanged() -> None:
    guardrail = PIIGuardrail()
    prompt = "Explique o que é um AI Gateway em uma frase."

    result = guardrail.sanitize(prompt=prompt)

    assert result.pii_detected is False
    assert result.sanitized_prompt == prompt
    assert result.redactions == {}
