const canvas = document.getElementById("digit-canvas");
const ctx = canvas.getContext("2d");
// Initialize canvas with black background
ctx.fillStyle = "black"; // Black background
ctx.fillRect(0, 0, canvas.width, canvas.height);

let isDrawing = false;

// Setup canvas to draw
canvas.addEventListener("mousedown", () => (isDrawing = true));
canvas.addEventListener("mouseup", () => (isDrawing = false));
canvas.addEventListener("mousemove", draw);

function draw(event) {
    if (!isDrawing) return;
    ctx.lineWidth = 15; // Brush size
    ctx.lineCap = "round";
    ctx.strokeStyle = "white"; // White brush color


    // Get mouse position
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Draw on canvas
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

// helper function to read the canvas data and preprocess it
const readCanvasData = () => {
    // Select the canvas element
    const canvas = document.getElementById("digit-canvas");
    // Step 2: Create an offscreen canvas to resize the image
    const offscreenCanvas = document.createElement("canvas");
    offscreenCanvas.width = 28;
    offscreenCanvas.height = 28;
    const offscreenCtx = offscreenCanvas.getContext("2d");
    offscreenCtx.fillStyle = "black"; // Black background
    // Draw the current canvas data onto the offscreen canvas (resize to 28x28)
    offscreenCtx.drawImage(canvas, 0, 0, 28, 28);

    // Step 3: Extract resized image data
    const resizedData = offscreenCtx.getImageData(0, 0, 28, 28);

    // Step 4: Extract raw pixel values
    const pixelData = [];
    for (let i = 0; i < resizedData.data.length; i += 4) {
        const r = resizedData.data[i]; // Red channel
        pixelData.push(r); // Push only the red channel since it's grayscale
    }

    console.log(pixelData);
    // Step 5: Reshape pixel data to 28x28
    const reshapedData = [];
    for (let i = 0; i < pixelData.length; i += 28) {
        reshapedData.push(pixelData.slice(i, i + 28));
    }

    return reshapedData;
}

// Clear canvas
document.getElementById("clear-button").addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "black"; // Black background
    ctx.beginPath();
});

// predict digit
document.getElementById("predict-button").addEventListener("click", () => {
    // Read the canvas element
    const reshapedData = readCanvasData();

    // Step 5: Send raw grayscale data to the backend
    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pixels: reshapedData })
    })
        .then(response => response.json())
        .then(result => {
            console.log(`Prediction: ${result.prediction}`);
        })
        .catch(error => {
            console.error("Error:", error);
        });
});

// capture digits
document.getElementById("capture-button").addEventListener("click", () => {
    // read the canvas data
    const reshapedData = readCanvasData();

    // read the digit label
    const digitLabel = document.getElementById("digit_label").value;

    // send the data to the backend
    fetch("/capture", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pixels: reshapedData, label: digitLabel })
    })
        .then(response => response.json())
        .then(result => {
            console.log(`Capture: ${result.status}`);
        })
        .catch(error => {
            console.error("Error:", error);
        });


});