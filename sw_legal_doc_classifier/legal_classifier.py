import time
from typing import Any

from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field


class ScoreItemWithRationale(BaseModel):
    title: str = Field(..., description="Turkish title of the legal class.")
    score: float = Field(..., description="Relevance score, ideally between 0.0 and 1.0.")
    rationale: str = Field(
        ...,
        description="Brief explanation for why this specific score was assigned to this class based on the document content.",
    )


class LegalClassificationOutput(BaseModel):
    best_class_title: str = Field(..., description="Title of the best matching legal class.")
    rationale: str = Field(
        ...,
        description="Brief explanation for why this specific score was assigned to this class based on the document content. For low scores, What did you catch?",
    )
    key_terms: list[str] = Field(
        ..., description="Key legal terms found in the document that support the classification."
    )
    scores: list[ScoreItemWithRationale] = Field(
        ..., description="List of scores and rationales for all legal classes."
    )


class LegalClass(BaseModel):
    id: str
    title_tr: str


class LegalDocumentClassifier:
    """A class for classifying Turkish legal documents into predefined categories."""

    # Default legal classes in Turkish
    DEFAULT_LEGAL_CLASSES = [
        LegalClass(id="hizmet_sozlesmesi", title_tr="Hizmet Sözleşmesi"),
        LegalClass(id="is_sozlesmesi", title_tr="İş Sözleşmesi"),
        LegalClass(id="sigorta_sozlesmesi", title_tr="Sigorta Sözleşmesi"),
        LegalClass(id="kira_sozlesmesi", title_tr="Kira Sözleşmesi"),
        LegalClass(id="gizlilik_sozlesmesi", title_tr="Gizlilik Sözleşmesi (NDA)"),
        LegalClass(id="lisans_sozlesmesi", title_tr="Lisans Sözleşmesi"),
        LegalClass(id="franchise_sozlesmesi", title_tr="Franchise Sözleşmesi"),
        LegalClass(id="distrbutorluk_sozlesmesi", title_tr="Distribütörlük Sözleşmesi"),
        LegalClass(id="ortaklik_sozlesmesi", title_tr="Ortaklık Sözleşmesi"),
        LegalClass(id="teminat_sozlesmesi", title_tr="Teminat Sözleşmesi"),
        LegalClass(id="taseronluk_sozlesmesi", title_tr="Taşeronluk Sözleşmesi"),
        LegalClass(id="ipotek_sozlesmesi", title_tr="İpotek Sözleşmesi"),
        LegalClass(id="tedarik_sozlesmesi", title_tr="Tedarik Sözleşmesi"),
        LegalClass(id="yatirim_sozlesmesi", title_tr="Yatırım Sözleşmesi"),
        LegalClass(id="danismanlik_sozlesmesi", title_tr="Danışmanlık Sözleşmesi"),
        LegalClass(id="alim_satim_sozlesmesi", title_tr="Alım Satım Sözleşmesi"),
        LegalClass(id="acentelik_sozlesmesi", title_tr="Acentelik Sözleşmesi"),
        LegalClass(id="kefalet_sozlesmesi", title_tr="Kefalet Sözleşmesi"),
        LegalClass(id="gayrimenkul_satis_sozlesmesi", title_tr="Gayrimenkul Satış Sözleşmesi"),
        LegalClass(id="teknoloji_transfer_sozlesmesi", title_tr="Teknoloji Transfer Sözleşmesi"),
    ]

    def __init__(
        self,
        model_id: str = "gpt-4.1-nano",
        user_id: str = "default_user",
        legal_classes: list[LegalClass] | None = None,
        max_chars_for_prompt: int = 4000,
        debug_mode: bool = True,
        agent_description: str | None = None,
        agent_instructions_template: str | None = None,
        agent_markdown: bool = True,
        # Add more config options as needed
    ):
        """
        Initialize the legal document classifier.

        Args:
            model_id: The ID of the OpenAI model to use
            user_id: The ID of the user making the request
            legal_classes: A list of dictionaries containing legal classes
            max_chars_for_prompt: Maximum number of characters to include in the prompt
            debug_mode: Whether to enable debug mode
        """

        # Set instance variables
        self.model_id = model_id
        self.user_id = user_id
        self.legal_classes = legal_classes or self.DEFAULT_LEGAL_CLASSES
        self.max_chars_for_prompt = max_chars_for_prompt
        self.debug_mode = debug_mode
        self.agent_description = agent_description or (
            "You are an expert AI assistant specializing in Turkish legal document classification. "
            "Your task is to analyze a given legal document text and classify it according to a "
            "predefined list of Turkish legal contract types (Sözleşme Türleri)."
        )
        self.agent_instructions_template = (
            agent_instructions_template
            or """
            Please analyze the following legal document content.
            Based on the content, particularly the purpose of the document, obligations of the parties, specific clauses, subject matter, and legal terminology related to contracts:

            1. For EACH of the {num_classes} legal classes provided below, assign a relevance score from 0.0 (not relevant at all) to 1.0 (highly relevant).
            A document might be relevant to multiple classes to some degree, but focus on the primary nature of the document.

            2. Identify the single BEST and MOST SPECIFIC legal class that this document belongs to or is primarily concerned with.
            This should be one of the classes from the provided list.

            3. Extract key legal terms and phrases from the document that support your classification.

            Here are the Turkish Legal Classes:
            --- START LEGAL CLASSES ---
            {class_list_str}
            --- END LEGAL CLASSES ---

            Here is the document text to analyze:
            --- START DOCUMENT TEXT ---
            {prompt_document_text}
            --- END DOCUMENT TEXT ---

            Provide your output in the structured format requested.
        """
        )
        self.agent_markdown = agent_markdown

    def extract_text_from_pdf(self, pdf_path: str) -> str | None:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            The extracted text content or None if extraction failed
        """
        try:
            loader = PyPDFLoader(pdf_path, extract_images=False, mode="single")
            documents = loader.load()
            if not documents:
                print(f"Error: No documents found in {pdf_path}.")
                return None
            document = documents[0]
            if not document.page_content:
                print("Error: No text content found in the document.")
                return None
            return document.page_content
        except Exception as e:
            print(f"Error extracting text from PDF {pdf_path}: {e}")
            return None

    def _create_agent(self, document_text: str) -> Agent:
        prompt_document_text = document_text[: self.max_chars_for_prompt]
        if len(document_text) > self.max_chars_for_prompt:
            print(
                f"Warning: Document text truncated to {self.max_chars_for_prompt} characters for the prompt."
            )

        class_list_str = "\n".join(
            [f"- ID: {lc.id}, Title (TR): {lc.title_tr}" for lc in self.legal_classes]
        )
        agent_instructions = [
            self.agent_instructions_template.format(
                num_classes=len(self.legal_classes),
                class_list_str=class_list_str,
                prompt_document_text=prompt_document_text,
            )
        ]

        agent_config: dict[str, Any] = {
            "model": OpenAIChat(id=self.model_id),
            "description": self.agent_description,
            "instructions": agent_instructions,
            "response_model": LegalClassificationOutput,
            "markdown": self.agent_markdown,
            "debug_mode": self.debug_mode,
        }

        return Agent(**agent_config)

    def classify_pdf(self, pdf_path: str) -> dict[str, Any]:
        """
        Classify a legal document from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            A dictionary containing classification results
        """
        # Extract text from the PDF
        document_text = self.extract_text_from_pdf(pdf_path)
        if not document_text:
            return {"error": "Failed to extract text from PDF"}

        return self.classify_text(document_text, source_file=pdf_path)

    def classify_text(self, document_text: str, source_file: str | None = None) -> dict[str, Any]:
        """
        Classify a legal document from text content.

        Args:
            document_text: The text content of the document to classify
            source_file: Optional name of the source file for reference

        Returns:
            A dictionary containing classification results
        """
        # Create the agent
        agent = self._create_agent(document_text)

        # Run the agent
        file_info = f" for document: {source_file}" if source_file else ""
        if self.debug_mode:
            print(f"Running agent{file_info}...")

        start_time = time.time()

        try:
            response: RunResponse = agent.run(  # type: ignore
                "Classify the provided legal document based on your instructions.",
                user_id=self.user_id,
            )

            end_time = time.time()
            elapsed_time = end_time - start_time

            if self.debug_mode:
                print(f"Execution time: {elapsed_time:.2f} seconds")

            response_content = response.content
            if not response_content:
                raise ValueError("No content returned from the agent.")

            # Process and return the results
            results: dict[str, Any] = {
                "best_class": response_content.best_class_title,
                "rationale": response_content.rationale,
                "key_terms": response_content.key_terms,
                "normalized_scores": self._get_normalized_scores(response),
                "raw_scores": [
                    (score.title, score.score, score.rationale)
                    for score in response.content.scores  # type: ignore
                ],
                "execution_time": elapsed_time,
                "source_file": source_file,
            }

            return results
        except Exception as e:
            end_time = time.time()
            elapsed_time = end_time - start_time

            return {
                "error": str(e),
                "execution_time": elapsed_time,
                "source_file": source_file,
            }

    def _get_normalized_scores(self, response: RunResponse) -> dict[str, float]:
        """
        Normalize scores so that they sum to 1 and return all in descending order.

        Args:
            response: The agent's response

        Returns:
            A dictionary of normalized scores
        """
        response_content = response.content
        if not response_content or not hasattr(response_content, "scores"):
            raise ValueError("Invalid response content or no scores found.")
        scores = [(score.title, score.score) for score in response_content.scores]
        total = sum(score for _, score in scores)

        if total == 0:
            norm_scores = [(title, 0.0) for title, _ in scores]
        else:
            norm_scores = [(title, score / total) for title, score in scores]

        # Sort by numeric score, descending
        norm_scores_sorted = sorted(norm_scores, key=lambda x: x[1], reverse=True)

        # Format values for display
        return {title: float(f"{score:.2f}") for title, score in norm_scores_sorted}

    def get_response_and_scores(self, pdf_path: str) -> dict[str, float]:
        """
        Get both the raw agent response and normalized scores.

        Args:
        pdf_path: Path to the PDF file

        Returns:
        Tuple of (RunResponse, normalized_scores_dict)
        """
        document_text = self.extract_text_from_pdf(pdf_path)
        if not document_text:
            raise ValueError(f"Failed to extract text from PDF: {pdf_path}")

        agent = self._create_agent(document_text)
        # Run the agent
        file_info = f" for document: {pdf_path}" if pdf_path else ""
        if self.debug_mode:
            print(f"Running agent{file_info}...")

        start_time = time.time()

        response = agent.run(
            "Classify the provided legal document based on your instructions.",
            user_id=self.user_id,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time

        if self.debug_mode:
            print(f"Execution time: {elapsed_time:.2f} seconds")

        normalized_scores = self._get_normalized_scores(response)

        return normalized_scores
