async function debugCode() {
    const code = document.getElementById("code").value;
    const outputText = document.getElementById("output-text");

    // Clear previous results
    outputText.textContent = "Processing...";

    try {
        const response = await fetch("http://127.0.0.1:8000/debug", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ code }),
        });

        const result = await response.json();

        // Display results with OpenAI suggestions on the next line
        outputText.textContent = `
Syntax Error: ${result.syntax_error || "None"}
Predicted Error Type: ${result.predicted_error_type || "N/A"}
OpenAI Suggestions:
${result.openai_suggestions || "No suggestions available."}
        `;
    } catch (error) {
        outputText.textContent = `Error: ${error.message}`;
    }
}

function clearCode() {
    const codeInput = document.getElementById("code");
    const outputText = document.getElementById("output-text");

    // Clear the textarea and result output
    codeInput.value = "";
    outputText.textContent = "Your debug results will appear here.";
}