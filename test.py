
import csv
import json
import os
import re



os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Silence Hugging Face warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM




PDF_FILE = "file.pdf"

OUTPUT_FILE = "flashcards_output.csv"

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


CARDS_PER_SECTION = 5

CHUNK_SIZE = 2500


def extract_pdf_text(pdf_path):

    if not os.path.exists(pdf_path):
        print(f"\nERROR: File not found:")
        print(pdf_path)
        return ""

    print("\nReading PDF...")

    reader = PdfReader(pdf_path)

    print(f"Total pages detected: {len(reader.pages)}")

    text = ""

    for page_number, page in enumerate(
        reader.pages,):

        extracted = page.extract_text()

        if extracted:

            text += (
                f"\n--- Page {page_number} ---\n"
            )

            text += extracted

            text += "\n"

        else:

            print(f"Warning: Page {page_number} "
                f"has no extractable text.")

    print(f"Characters extracted: {len(text):,}")
    return text
def clean_text(text):

    text = text.replace("\x00", " ")

    text = re.sub(r"[ \t]+"," ",text)
    text = re.sub(r"\n{3,}","\n\n",text)
    return text.strip()


# ============================================================
# SPLIT PDF
# ============================================================

def split_into_chunks(text):

    paragraphs = [
        p.strip()
        for p in text.split("\n")
        if p.strip()
    ]

    chunks = []

    current = ""

    for paragraph in paragraphs:

        if (
            len(current)
            + len(paragraph)
            + 1
            > CHUNK_SIZE
        ):

            if current:
                chunks.append(
                    current
                )

            current = paragraph

        else:

            if current:
                current += "\n"

            current += paragraph

    if current:
        chunks.append(
            current
        )

    return chunks


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\nLoading AI model...")
    print(MODEL_NAME)

    print(
        "\nRunning on CPU."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    )

    model.to("cpu")

    model.eval()

    print(
        "Model loaded successfully."
    )

    return tokenizer, model


# ============================================================
# GENERATE ONE CARD
# ============================================================

def generate_one_card(
    tokenizer,
    model,
    text,
    card_type
):

    # --------------------------------------------------------
    # MCQ
    # --------------------------------------------------------

    if card_type == "MCQ":

        instructions = """
Create ONE multiple-choice question.

Return ONLY this JSON:

{
  "type": "MCQ",
  "question": "question",
  "A": "option A",
  "B": "option B",
  "C": "option C",
  "D": "option D",
  "answer": "A"
}

Rules:
- Exactly four options.
- Exactly one correct answer.
- The answer must be A, B, C, or D.
- All options must be plausible.
- Base everything ONLY on the supplied text.
"""

    # --------------------------------------------------------
    # WRITTEN
    # --------------------------------------------------------

    elif card_type == "WRITTEN":

        instructions = """
Create ONE written-answer question.

Return ONLY this JSON:

{
  "type": "WRITTEN",
  "question": "question",
  "answer": "answer"
}

Rules:
- The question should require the student to write an answer.
- The answer should be concise but complete.
- Base everything ONLY on the supplied text.
"""

    # --------------------------------------------------------
    # TRUE / FALSE
    # --------------------------------------------------------

    else:

        instructions = """
Create ONE true/false question.

Return ONLY this JSON:

{
  "type": "TRUE_FALSE",
  "question": "statement",
  "answer": "True"
}

Rules:
- The answer must be exactly True or False.
- Make the statement unambiguous.
- Base everything ONLY on the supplied text.
"""

    prompt = f"""
You are an expert study-card generator.

{instructions}

Do NOT create multiple questions.

Do NOT add explanations.

Do NOT use markdown.

STUDY MATERIAL:

{text}
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You create accurate study flashcards. "
                "Follow the requested JSON format exactly."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    # CPU
    inputs = {
        key: value.to("cpu")
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,

            max_new_tokens=250,

            do_sample=False,

            pad_token_id=tokenizer.eos_token_id
        )

    # Only decode newly generated text
    generated_tokens = outputs[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]

    result = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return result


# ============================================================
# PARSE ONE CARD
# ============================================================

def parse_card(
    result,
    expected_type
):

    # Remove markdown fences
    result = result.replace(
        "```json",
        ""
    )

    result = result.replace(
        "```",
        ""
    )

    result = result.strip()

    # Find JSON object
    start = result.find("{")

    end = result.rfind("}")

    if start == -1 or end == -1:

        return None

    json_text = result[
        start:end + 1
    ]

    try:

        data = json.loads(
            json_text
        )

    except json.JSONDecodeError:

        return None

    if not isinstance(
        data,
        dict
    ):
        return None

    # ========================================================
    # MCQ
    # ========================================================

    if expected_type == "MCQ":

        question = str(
            data.get(
                "question",
                ""
            )
        ).strip()

        A = str(
            data.get(
                "A",
                ""
            )
        ).strip()

        B = str(
            data.get(
                "B",
                ""
            )
        ).strip()

        C = str(
            data.get(
                "C",
                ""
            )
        ).strip()

        D = str(
            data.get(
                "D",
                ""
            )
        ).strip()

        answer = str(
            data.get(
                "answer",
                ""
            )
        ).strip().upper()

        if not all([
            question,
            A,
            B,
            C,
            D
        ]):
            return None

        if answer not in [
            "A",
            "B",
            "C",
            "D"
        ]:
            return None

        front = (
            f"{question}\n\n"
            f"A) {A}\n"
            f"B) {B}\n"
            f"C) {C}\n"
            f"D) {D}"
        )

        back = (
            f"{answer}) "
            f"{data[answer]}"
        )

        return {
            "type": "MCQ",
            "front": front,
            "back": back
        }

    # ========================================================
    # WRITTEN
    # ========================================================

    if expected_type == "WRITTEN":

        question = str(
            data.get(
                "question",
                ""
            )
        ).strip()

        answer = str(
            data.get(
                "answer",
                ""
            )
        ).strip()

        if not question or not answer:
            return None

        return {
            "type": "Written",
            "front": question,
            "back": answer
        }

    # ========================================================
    # TRUE / FALSE
    # ========================================================

    if expected_type == "TRUE_FALSE":

        question = str(
            data.get(
                "question",
                ""
            )
        ).strip()

        answer = str(
            data.get(
                "answer",
                ""
            )
        ).strip().lower()

        if not question:
            return None

        if answer not in [
            "true",
            "false"
        ]:
            return None

        return {
            "type": "True/False",
            "front": (
                f"{question}\n\n"
                "True or False?"
            ),
            "back": answer.capitalize()
        }

    return None


# ============================================================
# GENERATE ALL CARDS
# ============================================================

def generate_all_flashcards(
    tokenizer,
    model,
    chunks
):

    all_cards = []

    # Every section gets this mix
    question_types = [
        "MCQ",
        "MCQ",
        "WRITTEN",
        "WRITTEN",
        "TRUE_FALSE"
    ]

    total_expected = (
        len(chunks)
        * len(question_types)
    )

    print("\n" + "=" * 60)

    print(
        f"Sections: {len(chunks)}"
    )

    print(
        f"Cards per section: "
        f"{len(question_types)}"
    )

    print(
        f"Maximum cards: "
        f"{total_expected}"
    )

    print("=" * 60)

    for section_number, chunk in enumerate(
        chunks,
        1
    ):

        print(
            f"\nSECTION "
            f"{section_number}/"
            f"{len(chunks)}"
        )

        print(
            "-" * 40
        )

        for card_number, card_type in enumerate(
            question_types,
            1
        ):

            print(
                f"Generating card "
                f"{card_number}/"
                f"{len(question_types)} "
                f"({card_type})..."
            )

            try:

                result = generate_one_card(
                    tokenizer,
                    model,
                    chunk,
                    card_type
                )

                card = parse_card(
                    result,
                    card_type
                )

                if card is None:

                    print(
                        "   FAILED to parse card."
                    )

                    print(
                        "   AI response:"
                    )

                    print(
                        "   "
                        + result[:500]
                    )

                    continue

                # Check duplicate
                duplicate = False

                for existing in all_cards:

                    if (
                        existing["front"]
                        .strip()
                        .lower()
                        ==
                        card["front"]
                        .strip()
                        .lower()
                    ):

                        duplicate = True
                        break

                if duplicate:

                    print(
                        "   Duplicate - skipped."
                    )

                else:

                    all_cards.append(
                        card
                    )

                    print(
                        "   SUCCESS!"
                    )

            except Exception as error:

                print(
                    f"   ERROR: {error}"
                )

    return all_cards


# ============================================================
# SAVE CSV
# ============================================================

def save_to_csv(
    cards,
    filename
):

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "Type",
            "Front",
            "Back"
        ])

        for card in cards:

            writer.writerow([
                card["type"],
                card["front"],
                card["back"]
            ])

    print(
        f"\nCSV saved successfully:"
    )

    print(
        os.path.abspath(filename)
    )


# ============================================================
# SUMMARY
# ============================================================

def show_summary(cards):

    mcq = sum(
        card["type"] == "MCQ"
        for card in cards
    )

    written = sum(
        card["type"] == "Written"
        for card in cards
    )

    true_false = sum(
        card["type"] == "True/False"
        for card in cards
    )

    print("\n")
    print("=" * 60)
    print("FLASHCARD SUMMARY")
    print("=" * 60)

    print(
        f"Total:       {len(cards)}"
    )

    print(
        f"MCQ:         {mcq}"
    )

    print(
        f"Written:     {written}"
    )

    print(
        f"True/False:  {true_false}"
    )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "PDF → AI FLASHCARD GENERATOR"
    )

    print(
        "CPU VERSION"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    text = extract_pdf_text(
        PDF_FILE
    )

    if not text:

        return

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    text = clean_text(
        text
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    chunks = split_into_chunks(
        text
    )

    print(
        f"\nDocument divided into "
        f"{len(chunks)} sections."
    )

    print(
        f"Target: approximately "
        f"{len(chunks) * CARDS_PER_SECTION} "
        f"flashcards."
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    tokenizer, model = load_model()
    cards = generate_all_flashcards(tokenizer,model,chunks)

    show_summary(
        cards
    )

    # --------------------------------------------------------
    # Display cards
    # --------------------------------------------------------

    for number, card in enumerate(
        cards,
        1
    ):

        print("\n" + "-" * 60)

        print(
            f"CARD {number}"
        )

        print(
            f"TYPE: {card['type']}"
        )

        print(
            f"\n{card['front']}"
        )

        print(
            f"\nANSWER:\n{card['back']}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if cards:

        save_to_csv(
            cards,
            OUTPUT_FILE
        )

    else:

        print(
            "\nNo cards were generated."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
