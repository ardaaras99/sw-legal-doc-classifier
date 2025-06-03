from dotenv import load_dotenv

from sw_legal_doc_classifier.legal_classifier import LegalClass, LegalDocumentClassifier

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # You can customize legal classes if you want
    custom_legal_classes = [
        LegalClass(id="ozel_sozlesme", title_tr="Özel Sözleşme"),
        LegalClass(id="hizmet_sozlesmesi", title_tr="Hizmet Sözleşmesi"),
    ]

    # Create a classifier instance with custom configuration
    classifier = LegalDocumentClassifier(
        model_id="gpt-4.1-nano",  # or any other model you prefer
        user_id="user123",
        legal_classes=None,  # or your custom 20 legal class.You can add above.
        max_chars_for_prompt=3000,
        debug_mode=False,
        agent_description=None,  # or your custom description
        agent_instructions_template=None,  # or your custom instructions
        agent_markdown=True,
    )

    # Classify a PDF document
    pdf_file_path = "/home/faruk18/python-projects/sw-legal-doc-classifier/test.pdf"
    result = classifier.get_response_and_scores(pdf_file_path)

    print("Classification Result:", result)
