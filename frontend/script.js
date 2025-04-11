async function debugCode() {
    const code = document.getElementById('code').value;
    const response = await fetch('/debug', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code })
    });
    const result = await response.json();
    document.getElementById('result').innerText = result.result;
}