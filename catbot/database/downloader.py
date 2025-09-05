"""Downlaods documents from simple wikipedia"""

import wikipedia
import re
from langchain.schema import Document

wikipedia.set_lang("simple")


def download_and_chunk_wikipedia_articles(
    articles: list[str] = ["Cat"],
) -> list[Document]:
    """Downloads articles from simple wikipedia and splits them into sections based on headers"""

    sections = []
    cnt = 0
    for article_key in articles:

        article = wikipedia.page(article_key)
        text = article.content

        # Regular expression for MediaWiki section headers like "== Section =="
        section_pattern = re.compile(r"^(={2,})\s*(.*?)\s*\1$", re.MULTILINE)

        last_pos = 0
        last_header = "Introduction"

        for match in section_pattern.finditer(text):
            start, end = match.span()
            header = match.group(2).strip()
            # Add previous section
            if start > last_pos:
                content = text[last_pos:start].strip()
                if content:
                    sections.append(
                        Document(
                            page_content=article_key
                            + " - "
                            + last_header
                            + "\n\n"
                            + content,
                            metadata={
                                "title": article_key + " - " + last_header,
                                "source": content,
                            },
                            id=str(cnt),
                        )
                    )
                    cnt += 1
            last_header = header
            last_pos = end

        # Add the last section
        final_content = text[last_pos:].strip()
        if final_content:
            sections.append(
                Document(
                    page_content=article_key
                    + " - "
                    + last_header
                    + "\n\n"
                    + final_content,
                    metadata={
                        "title": article_key + " - " + last_header,
                        "source": final_content,
                    },
                    id=str(cnt),
                )
            )
            cnt += 1

    return sections
