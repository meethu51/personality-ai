import re


def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-z\s]", "", text)

    return text


def conversation_to_text(conv):

    if isinstance(conv, list):
        return " ".join([turn["text"] for turn in conv])

    return str(conv)