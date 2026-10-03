const askForm = document.getElementById("ask-form");
const question = document.getElementById("question");
const answerArea = document.getElementById("answer-area");
const answer = document.getElementById("answer");
const answerSources = document.getElementById("answer-sources");

askForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const value = question.value.trim();
    if (!value) return;

    answerArea.classList.remove("hidden");
    answer.textContent = "Epoch is searching your sources...";
    answerSources.innerHTML = "";

    const form = new FormData();
    form.append("question", value);

    try {
        const response = await fetch("/ask", {
            method: "POST",
            body: form
        });

        const data = await response.json();

        if (!response.ok) {
            answer.textContent = data.error || "Something went wrong.";
            return;
        }

        answer.textContent = data.answer || "No answer generated.";

        (data.sources || []).forEach(source => {
            const tag = document.createElement("span");
            tag.className = "source-tag";
            tag.textContent = `${source.filename} · ${source.score}`;
            answerSources.appendChild(tag);
        });

    } catch (error) {
        answer.textContent = "Could not connect to Epoch.";
    }
});


const resultCard = document.getElementById("result-card");
const resultTitle = document.getElementById("result-title");
const resultContent = document.getElementById("result-content");

document.querySelectorAll(".generate-btn").forEach(button => {
    button.addEventListener("click", async () => {
        const action = button.dataset.action;

        button.classList.add("loading");
        button.textContent = "Generating...";

        const form = new FormData();
        form.append("action", action);

        try {
            const response = await fetch("/generate", {
                method: "POST",
                body: form
            });

            const data = await response.json();

            if (!response.ok) {
                resultTitle.textContent = "Error";
                resultContent.textContent = data.error || "Generation failed.";
            } else {
                const titles = {
                    summary: "Study Summary",
                    notes: "Revision Notes",
                    flashcards: "Generated Flashcards",
                    questions: "Practice Questions"
                };

                resultTitle.textContent = titles[action];

                if (action === "flashcards" && Array.isArray(data.result)) {
                    renderFlashcards(data.result);
                } else {
                    resultContent.innerHTML = "";
                    const pre = document.createElement("pre");
                    pre.textContent = data.result;
                    resultContent.appendChild(pre);
                }
            }

            resultCard.classList.remove("hidden");
            resultCard.scrollIntoView({ behavior: "smooth", block: "start" });

        } catch (error) {
            resultTitle.textContent = "Error";
            resultContent.textContent = "Could not connect to Epoch.";
            resultCard.classList.remove("hidden");
        } finally {
            button.classList.remove("loading");

            const labels = {
                summary: "Generate summary",
                notes: "Generate notes",
                flashcards: "Generate cards",
                questions: "Generate questions"
            };

            button.textContent = labels[action];
        }
    });
});


function renderFlashcards(cards) {
    resultContent.innerHTML = "";

    if (!cards.length) {
        resultContent.textContent =
            "The model did not return cards in the expected format. Try generating again.";
        return;
    }

    cards.forEach((card, index) => {
        const wrapper = document.createElement("div");
        wrapper.className = "flashcard";

        const type = document.createElement("div");
        type.className = "flashcard-type";
        type.textContent = `${index + 1} · ${card.type}`;

        const q = document.createElement("div");
        q.className = "flashcard-question";
        q.textContent = card.question;

        wrapper.appendChild(type);
        wrapper.appendChild(q);

        if (card.type === "MCQ") {
            Object.entries(card.options).forEach(([letter, text]) => {
                const option = document.createElement("div");
                option.className = "option";
                option.textContent = `${letter}. ${text}`;
                wrapper.appendChild(option);
            });

            const reveal = document.createElement("div");
            reveal.className = "answer-reveal";
            reveal.textContent = `Correct answer: ${card.answer}`;
            wrapper.appendChild(reveal);
        } else {
            const reveal = document.createElement("div");
            reveal.className = "answer-reveal";
            reveal.textContent = `Answer: ${card.answer}`;
            wrapper.appendChild(reveal);
        }

        resultContent.appendChild(wrapper);
    });
}


document.getElementById("close-result").addEventListener("click", () => {
    resultCard.classList.add("hidden");
});
