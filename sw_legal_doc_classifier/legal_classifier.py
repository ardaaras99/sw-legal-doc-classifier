from agno.agent import Agent
from agno.models.openai import OpenAIChat
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    response_model: type[BaseModel] = Field(description="The response model of the agent")
    model_id: str = Field(description="The ID of the OpenAI model to use")
    max_chars_for_prompt: int = Field(description="Maximum number of characters to include in the prompt")
    debug_mode: bool = Field(description="Whether to enable debug mode")
    agent_description: str | None = Field(description="The description of the agent")
    agent_instructions_template: str | None = Field(description="The instructions template of the agent")
    agent_markdown: bool = Field(description="Whether to enable markdown in the agent's response")


class DocumentClassifier:
    """A class for classifying Turkish legal documents into predefined categories."""

    def __init__(self, agent_config: AgentConfig):
        self.config = agent_config
        self.agent = Agent(
            model=OpenAIChat(id=self.config.model_id),
            description=self.config.agent_description,
            instructions=self.config.agent_instructions_template,
            response_model=self.config.response_model,
            markdown=self.config.agent_markdown,
            debug_mode=self.config.debug_mode,
        )

    from typing import Any

    def get_response_and_scores(self, document_text: str) -> Any | None | BaseModel:
        response = self.agent.run(message=document_text)
        return response.content
