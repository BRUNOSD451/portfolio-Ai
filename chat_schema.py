"""Contrato do chat compartilhado pela rota e pela validação de consumo."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HistoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)

    @field_validator("content")
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError("Mensagem vazia no histórico.")
        return value.strip()


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=2000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=12)

    @field_validator("message")
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError("Informe uma pergunta.")
        return value.strip()

    @field_validator("history")
    @classmethod
    def valid_history(cls, value):
        if len(value) % 2 or any(
            item.role != ("user" if index % 2 == 0 else "assistant")
            for index, item in enumerate(value)
        ):
            raise ValueError("O histórico deve conter pares de pergunta e resposta.")
        if sum(len(item.content) for item in value) > 12000:
            raise ValueError("Histórico muito longo.")
        return value
