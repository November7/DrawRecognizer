const drawBoard = document.getElementById('drawBoard');
const ctx = drawBoard.getContext('2d', { willReadFrequently: true });
const predDiv = document.getElementById('predictions');
const modelSelect = document.getElementById('functionSelect');
const modelStatus = document.getElementById('modelStatus');
const probabilitiesContainer = document.getElementById('classProbabilities');

let drag = false;
let pos = { x: 0, y: 0 };
let lastTapTime = 0;
let model = null;
let classLabels = [];
const modelsBaseUrl = new URL('models/', window.location.href);
const fallbackModels = ['Cyfry-Mnist', 'Cyfry-TM', 'XO-TM'];
const DOUBLE_TAP_THRESHOLD_MS = 320;

function getModelAssetUrl(modelName, fileName = '') {
    return new URL(`${modelName}/${fileName}`, modelsBaseUrl).href;
}

function normalizeModelName(name) {
    return String(name || '')
        .replace(/\\/g, '/')
        .replace(/\/+$/, '')
        .split('/')
        .filter(Boolean)
        .pop() || '';
}

async function discoverFromManifest() {
    try {
        const response = await fetch(getModelAssetUrl('', 'index.json'));

        if (!response.ok) {
            return [];
        }

        const data = await response.json();

        if (!Array.isArray(data)) {
            return [];
        }

        return Array.from(
            new Set(
                data
                    .map(normalizeModelName)
                    .filter(Boolean)
            )
        ).sort((left, right) => left.localeCompare(right, 'pl'));
    }
    catch {
        return [];
    }
}

async function filterAvailableModels(models) {
    const checks = await Promise.all(
        models.map(async modelName => {
            try {
                const response = await fetch(getModelAssetUrl(modelName, 'model.json'), { cache: 'no-store' });
                return response.ok ? modelName : null;
            }
            catch {
                return null;
            }
        })
    );

    return checks.filter(Boolean);
}

modelSelect.addEventListener('change', loadSelectedModel);

async function discoverModels() {
    const fromManifest = await discoverFromManifest();

    if (fromManifest.length > 0) {
        return fromManifest;
    }

    try {
        const response = await fetch(modelsBaseUrl.href);

        if (response.ok) {
            const contentType = response.headers.get('content-type') || '';

            if (contentType.includes('application/json')) {
                const data = await response.json();

                if (Array.isArray(data)) {
                    return Array.from(
                        new Set(
                            data
                                .map(normalizeModelName)
                                .filter(Boolean)
                        )
                    ).sort((left, right) => left.localeCompare(right, 'pl'));
                }
            }

            const html = await response.text();
            const doc = new DOMParser().parseFromString(html, 'text/html');
            const links = Array.from(doc.querySelectorAll('a'));
            const discoveredModels = Array.from(
                new Set(
                    links
                        .map(link => link.getAttribute('href') || '')
                        .map(href => href.replace(/\\/g, '/'))
                        .map(href => href.split('?')[0].split('#')[0])
                        .filter(href => href.endsWith('/') && href !== '../' && href !== './')
                        .map(href => href.replace(/\/+$/, ''))
                        .map(href => href.split('/').filter(Boolean).pop() || '')
                        .map(normalizeModelName)
                        .filter(modelName => modelName && modelName.toLowerCase() !== 'models')
                )
            ).sort((left, right) => left.localeCompare(right, 'pl'));

            if (discoveredModels.length > 0) {
                return discoveredModels;
            }
        }
    }
    catch {
    }

    const availableFallbackModels = await filterAvailableModels(fallbackModels);

    if (availableFallbackModels.length > 0) {
        return availableFallbackModels;
    }

    throw new Error('Nie udało się wykryć modeli. Dodaj models/index.json albo upewnij się, że model.json jest dostępny.');
}

function renderModelOptions(models) {
    modelSelect.innerHTML = '';

    models.forEach(modelName => {
        const option = document.createElement('option');
        option.value = modelName;
        option.textContent = modelName;
        modelSelect.appendChild(option);
    });
}

async function loadModelMetadata(modelName, nClasses) {
    try {
        const response = await fetch(getModelAssetUrl(modelName, 'metadata.json'));

        if (!response.ok) {
            throw new Error('Brak metadanych modelu.');
        }

        const data = await response.json();

        if (Array.isArray(data.labels) && data.labels.length > 0) {
            return data.labels;
        }
    } 
    catch (error) {
            console.warn(`Nie można załadować metadanych dla modelu "${modelName}": ${error.message}`);
    }

    return Array.from({ length: nClasses }, (_, index) => `${index}`);
}

function renderProbabilityBars(labels, probabilities) {
    probabilitiesContainer.innerHTML = '';

    labels.forEach((label, index) => {
        const probability = probabilities[index] ?? 0;
        const percentage = (probability * 100).toFixed(2);

        const item = document.createElement('div');
        item.className = 'probability-item';

        const row = document.createElement('div');
        row.className = 'probability-row';

        const labelSpan = document.createElement('span');
        labelSpan.className = 'probability-label';
        labelSpan.textContent = label;

        const valueSpan = document.createElement('span');
        valueSpan.className = 'probability-value';
        valueSpan.textContent = `${percentage}%`;

        const track = document.createElement('div');
        track.className = 'probability-track';

        const fill = document.createElement('div');
        fill.className = 'probability-fill';
        fill.style.width = `${Math.max(0, Math.min(probability * 100, 100))}%`;

        row.appendChild(labelSpan);
        row.appendChild(valueSpan);
        track.appendChild(fill);
        item.appendChild(row);
        item.appendChild(track);
        probabilitiesContainer.appendChild(item);
    });
}

async function loadSelectedModel() {
    const modelName = modelSelect.value;

    if (!modelName) {
        return;
    }

    modelStatus.textContent = `Ładowanie modelu: ${modelName}`;
    modelSelect.disabled = true;

    try {
        model = await tf.loadLayersModel(getModelAssetUrl(modelName, 'model.json'));
        classLabels = await loadModelMetadata(modelName, model.output.shape[1]);
        modelStatus.textContent = `Załadowano model: ${modelName}`;
        clearCanvas();
    } 
    catch (error) {
        model = null;
        classLabels = [];
        predDiv.textContent = `Nie udało się załadować modelu: ${modelName}`;
        modelStatus.textContent = 'Błąd ładowania modelu.';
        probabilitiesContainer.innerHTML = '';
    } 
    finally {
        modelSelect.disabled = false;
    }
}

function setPos(e) {
    const point = getPointerPosition(e);
    pos.x = point.x;
    pos.y = point.y;
}

function getPointerPosition(e) {
    const rect = ctx.canvas.getBoundingClientRect();

    if (e.touches && e.touches.length > 0) {
        return {
            x: e.touches[0].clientX - rect.left,
            y: e.touches[0].clientY - rect.top
        };
    }

    if (e.changedTouches && e.changedTouches.length > 0) {
        return {
            x: e.changedTouches[0].clientX - rect.left,
            y: e.changedTouches[0].clientY - rect.top
        };
    }

    return {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
    };
}

function startDrawing(event) {
    event.preventDefault();
    drag = true;
    setPos(event);
}

function stopDrawing(event) {
    event.preventDefault();

    if (!drag) {
        return;
    }

    drag = false;
    predictModel();
}

function handlePointerTap(event) {
    // Gest czyszczenia podwójnym tapnięciem tylko dla wejścia dotykowego.
    const pointerType = event.pointerType || '';

    if (pointerType !== 'touch') {
        return;
    }

    const now = Date.now();

    if (now - lastTapTime <= DOUBLE_TAP_THRESHOLD_MS) {
        clearCanvas();
        lastTapTime = 0;
        return;
    }

    lastTapTime = now;
}

function handleLegacyTouchTap(event) {
    const now = Date.now();

    if (now - lastTapTime <= DOUBLE_TAP_THRESHOLD_MS) {
        event.preventDefault();
        clearCanvas();
        lastTapTime = 0;
        return;
    }

    lastTapTime = now;
}

function draw(event) {
    event.preventDefault();

    if (!drag) {
        return;
    }

    ctx.beginPath();
    ctx.lineWidth = 10;
    ctx.strokeStyle = 'white';
    ctx.lineCap = 'round';
    ctx.moveTo(pos.x, pos.y);
    setPos(event);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
}

function clearCanvas() {
    ctx.clearRect(0, 0, drawBoard.width, drawBoard.height);
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, drawBoard.width, drawBoard.height);
    predDiv.textContent = 'Oczekuje na rysunek...';
    renderProbabilityBars(classLabels, new Array(classLabels.length).fill(0));
}

function predictModel() {
    if (!model) {
        return;
    }

    const imageData = ctx.getImageData(0, 0, drawBoard.width, drawBoard.height);
    const width = model.input.shape[1];
    const height = model.input.shape[2];
    const depth = model.input.shape[3];
    const nClasses = model.output.shape[1];

    if (!classLabels || classLabels.length === 0) {
        classLabels = Array.from({ length: nClasses }, (_, index) => `${index}`);
    }

    const { predictionProbabilities, bestClassIndex } = tf.tidy(() => {
        const image = tf.browser.fromPixels(imageData, depth);
        const resizedImage = tf.image.resizeBilinear(image, [width, height]).expandDims(0);
        const normalizedImage = tf.cast(resizedImage, 'float32').div(255.0);
        const prediction = model.predict(normalizedImage);

        return {
            predictionProbabilities: Array.from(prediction.dataSync()),
            bestClassIndex: prediction.argMax(1).dataSync()[0]
        };
    });

    predDiv.textContent = `Klasyfikacja: ${classLabels[bestClassIndex]}`;
    renderProbabilityBars(classLabels, predictionProbabilities);
}

drawBoard.addEventListener('contextmenu', function (event) {
    event.preventDefault();
});

document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') {
        clearCanvas();
    }
});

if ('PointerEvent' in window) {
    drawBoard.addEventListener('pointerdown', startDrawing);
    drawBoard.addEventListener('pointermove', draw);
    drawBoard.addEventListener('pointerup', stopDrawing);
    drawBoard.addEventListener('pointerup', handlePointerTap);
    drawBoard.addEventListener('pointercancel', stopDrawing);
    drawBoard.addEventListener('pointerleave', stopDrawing);
}
else {
    drawBoard.addEventListener('mousedown', startDrawing);
    drawBoard.addEventListener('mousemove', draw);
    drawBoard.addEventListener('mouseup', stopDrawing);
    drawBoard.addEventListener('mouseleave', stopDrawing);

    drawBoard.addEventListener('touchstart', startDrawing, { passive: false });
    drawBoard.addEventListener('touchmove', draw, { passive: false });
    drawBoard.addEventListener('touchend', stopDrawing, { passive: false });
    drawBoard.addEventListener('touchend', handleLegacyTouchTap, { passive: false });
    drawBoard.addEventListener('touchcancel', stopDrawing, { passive: false });
}

async function initializeApp() {
    clearCanvas();

    try {
        const availableModels = await discoverModels();

        renderModelOptions(availableModels);
        modelStatus.textContent = `Wykryto ${availableModels.length} modele.`;
        await loadSelectedModel();
    } 
    catch (error) {
        predDiv.textContent = 'Nie udało się uruchomić aplikacji.';
        modelStatus.textContent = 'Błąd ładowania modeli.';
        modelSelect.disabled = true;
        probabilitiesContainer.innerHTML = '';
    }
}

initializeApp();