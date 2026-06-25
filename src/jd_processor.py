from docx import Document


def load_job_description(path):

    doc = Document(path)

    text = []

    for para in doc.paragraphs:
        text.append(
            para.text
        )

    return "\n".join(text)