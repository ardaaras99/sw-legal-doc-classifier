# %%

# %%
import enum

from pydantic import BaseModel, Field
from rich import print as rprint

from sw_legal_doc_classifier.legal_classifier import AgentConfig, DocumentClassifier

DocumentType: type[enum.StrEnum] = enum.StrEnum(
    "DocumentType",
    [
        "Kira",
        "Hizmet",
        "İş",
        "Sigorta",
        "Lisans",
        "Franchise",
        "Distribütörlük",
        "Ortaklık",
        "Teminat",
        "Taşeronluk",
        "İpotek",
        "Tedarik",
        "Yatırım",
        "Danışmanlık",
        "Alım Satım",
    ],
)


class Trial(BaseModel):
    document_type: DocumentType
    score: int = Field(description="describes how confident the model is about the document type", ge=0, le=100)
    rationale: str = Field(description="sence bu döküman neden senin seçtiğin türe ait")


config = AgentConfig(
    response_model=Trial,
    # model_id="gpt-4.1-nano",
    model_id="gpt-4o-mini",
    max_chars_for_prompt=3000,
    debug_mode=True,
    agent_description="Sen bir classification uzmanısın, sana bir dökümanın alabileceği farklı türleri veriyorum, lütfen bunlar arasından en uygun olanı seç ve score ver, Cevaplarının türkçe olması gerekiyor.",
    agent_instructions_template="Sen bir classification uzmanısın, sana bir dökümanın alabileceği farklı türleri veriyorum, lütfen bunlar arasından en uygun olanı seç ve score ver, Cevaplarının türkçe olması gerekiyor.",
    agent_markdown=True,
)
classifier = DocumentClassifier(agent_config=config)

# Classify a PDF document
with open("data/trial.txt", encoding="utf-8") as file:
    document_text = file.read()
result = classifier.get_response_and_scores(document_text=document_text)
rprint(result)


# %%
