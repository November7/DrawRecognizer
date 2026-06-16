/**
 *  Draw Recognizer - A web application for drawing and recognizing images using ONNX Runtime Web.
 *  Author: Marcin Kowalski
 *  License: MIT
 */

// constants and state variables
const drawBoard                 =   document.getElementById('drawBoard');
const predDiv                   =   document.getElementById('predictions');
const modelSelect               =   document.getElementById('functionSelect');
const modelStatus               =   document.getElementById('modelStatus');
const probabilitiesContainer    =   document.getElementById('classProbabilities');
const mobilePredictionsSlot     =   document.getElementById('mobilePredictionsSlot');
const resultsPanel              =   document.querySelector('.results-panel');
const ctx                       =   drawBoard.getContext('2d', { willReadFrequently: true });
const preprocessCanvas          =   document.createElement('canvas');
const preprocessCtx             =   preprocessCanvas.getContext('2d', { willReadFrequently: true });
const plSort                    =   (left, right) => left.localeCompare(right, 'pl');
const modelsBaseUrl             =   new URL('models/', window.location.href);
const fallbackModels            =   ['test1.onnx'];
const DOUBLE_TAP_THRESHOLD_MS   =   320;
const singleColumnMediaQuery    =   window.matchMedia('(max-width: 960px)');

let drag                        =   false;
let pos                         =   { x: 0, y: 0 };
let lastTapTime                 =   0;
let session                     =   null;
let modelInputShape             =   [1, 28, 28, 1];
let modelInputRank              =   4;
let modelOutputName             =   '';
let classLabels                 =   [];

modelSelect.addEventListener('change', loadSelectedModel);

// utility functions

function getModelAssetUrl(pathName = '') {
    return new URL(pathName, modelsBaseUrl).href;
}

function normalizeModelName(name) {
    return String(name || '')
        .replace(/\\/g, '/')
        .replace(/^\.\//, '')
        .replace(/^\/+/, '')
        .replace(/\/+$/, '')
        .trim();
}

function placePredictionsByViewport() {
    if (!predDiv || !probabilitiesContainer) return;

    if (singleColumnMediaQuery.matches) {
        if (mobilePredictionsSlot && predDiv.parentElement !== mobilePredictionsSlot) {
            mobilePredictionsSlot.appendChild(predDiv);
        }
        return;
    }

    if (resultsPanel && predDiv.parentElement !== resultsPanel) {
        resultsPanel.insertBefore(predDiv, probabilitiesContainer);
    }
}

async function discoverFromManifest() {
    try {
        const response = await fetch(getModelAssetUrl('index.json'));

        if (!response.ok) return [];

        const data = await response.json();

        if (!Array.isArray(data)) return [];

        return Array.from(new Set(data
            .map(normalizeModelName)
            .filter(entry => entry.toLowerCase().endsWith('.onnx')))).sort(plSort);
    }
    catch {
        return [];
    }
}

async function filterAvailableModels(models) {
    const checks = await Promise.all(
        models.map(async modelPath => {
            try {
                const response = await fetch(getModelAssetUrl(modelPath), { cache: 'no-store' });
                return response.ok ? modelPath : null;
            }
            catch {
                return null;
            }
        })
    );

    return checks.filter(Boolean);
}

async function discoverModels() {
    const fromManifest = await discoverFromManifest();

    if (fromManifest.length > 0) return fromManifest;

    try {
        const response = await fetch(modelsBaseUrl.href);

        if (response.ok) {
            const contentType = response.headers.get('content-type') || '';

            if (contentType.includes('application/json')) {
                const data = await response.json();

                if (Array.isArray(data)) {
                    return Array.from(new Set(data
                        .map(normalizeModelName)
                        .filter(entry => entry.toLowerCase().endsWith('.onnx')))).sort(plSort);
                }
            }

            const html = await response.text();
            const doc = new DOMParser().parseFromString(html, 'text/html');
            const links = Array.from(doc.querySelectorAll('a'));
            const discoveredModels = Array.from(new Set(links.map(link => link.getAttribute('href') || '')
                .map(normalizeModelName)
                .filter(href => href.toLowerCase().endsWith('.onnx')))).sort(plSort);

            if (discoveredModels.length > 0) return discoveredModels;
        }
    }
    catch {
        // nop
    }

    const availableFallbackModels = await filterAvailableModels(fallbackModels);
    if (availableFallbackModels.length > 0) return availableFallbackModels;

    throw new Error('Nie udało się wykryć modeli. Dodaj models/index.json albo upewnij się, że pliki .onnx są dostępne.');
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

function softmax(values) {
    const maxVal = Math.max(...values);
    const exps = values.map(value => Math.exp(value - maxVal));
    const sum = exps.reduce((acc, val) => acc + val, 0) || 1;
    return exps.map(value => value / sum);
}

function normalizeScores(values) {
    const sum = values.reduce((acc, val) => acc + val, 0);
    const allBounded = values.every(value => Number.isFinite(value) && value >= 0 && value <= 1);

    if (allBounded && sum > 0.95 && sum < 1.05) return values;

    return softmax(values);
}

function readDimensionValue(dim, fallback) {
    return Number.isInteger(dim) && dim > 0 ? dim : fallback;
}

function resolveInputShape(inputMeta) {
    const dims = Array.isArray(inputMeta?.dimensions) ? inputMeta.dimensions : [];

    if (dims.length === 2) {
        const d0 = readDimensionValue(dims[0], 1);
        const d1 = readDimensionValue(dims[1], 784);
        return [d0, d1];
    }

    if (dims.length === 3) {
        const d0 = readDimensionValue(dims[0], 1);
        const d1 = readDimensionValue(dims[1], 28);
        const d2 = readDimensionValue(dims[2], 28);
        return [d0, d1, d2];
    }

    if (dims.length !== 4) return [1, 28, 28, 1];

    const d0 = readDimensionValue(dims[0], 1);
    const d1 = readDimensionValue(dims[1], 28);
    const d2 = readDimensionValue(dims[2], 28);
    const d3 = readDimensionValue(dims[3], 1);

    return [d0, d1, d2, d3];
}

function getModelMetadataCandidates(modelPath) {
    const cleanedPath = modelPath.replace(/\\/g, '/').replace(/^\/+/, '');
    const lastSlashIndex = cleanedPath.lastIndexOf('/');
    const directoryPath = lastSlashIndex >= 0 ? cleanedPath.slice(0, lastSlashIndex + 1) : '';
    const fileName = lastSlashIndex >= 0 ? cleanedPath.slice(lastSlashIndex + 1) : cleanedPath;
    const baseName = fileName.replace(/\.onnx$/i, '');

    return [
        `${directoryPath}metadata.json`,
        `${directoryPath}${baseName}.metadata.json`
    ];
}

async function loadModelMetadata(modelPath, nClasses) {
    const candidates = getModelMetadataCandidates(modelPath);

    for (const candidate of candidates) {
        try {
            const response = await fetch(getModelAssetUrl(candidate));

            if (!response.ok) continue;

            const data = await response.json();

            if (Array.isArray(data.labels) && data.labels.length > 0) return data.labels;
        }
        catch {
            // nop
        }
    }

    return Array.from({ length: nClasses }, (_, index) => `${index}`);
}

function renderProbabilityBars(labels, probabilities) {
    probabilitiesContainer.innerHTML = '';

    labels.forEach((label, index) => {
        const el = (tag, className, text) => {
            const e = document.createElement(tag);
            if (className) e.className = className;
            if (text !== undefined) e.textContent = text;
            return e;
        };

        const probability = probabilities[index] ?? 0;
        const percentage = (probability * 100).toFixed(2);

        const item = el('div', 'probability-item');
        const row = el('div', 'probability-row');
        const labelSpan = el('span', 'probability-label', label);
        const valueSpan = el('span', 'probability-value', `${percentage}%`);
        const track = el('div', 'probability-track');
        const fill = el('div', 'probability-fill');

        fill.style.width = `${Math.min(Math.max(probability * 100, 0), 100)}%`;

        row.append(labelSpan, valueSpan);
        track.append(fill);
        item.append(row, track);
        probabilitiesContainer.append(item);
    });
}

function getInputLayout(shape) {
    const channelsFirst = shape[1];
    const channelsLast = shape[3];

    if (channelsLast === 1 || channelsLast === 3) return 'NHWC';
    if (channelsFirst === 1 || channelsFirst === 3) return 'NCHW';

    return 'NCHW';
}

function getResizedPixelData(width, height) {
    preprocessCanvas.width = width;
    preprocessCanvas.height = height;
    preprocessCtx.drawImage(drawBoard, 0, 0, width, height);
    return preprocessCtx.getImageData(0, 0, width, height).data;
}

function fillFlatGrayData(target, pixelData, width, height) {
    const count = Math.min(target.length, width * height);

    for (let i = 0; i < count; i++) {
        const srcOffset = i * 4;
        target[i] = pixelData[srcOffset] / 255;
    }
}

function preprocessInputTensor(shape) {
    if (shape.length === 2) {
        const features = shape[1];
        const side = Math.round(Math.sqrt(features));
        const width = side * side === features ? side : features;
        const height = side * side === features ? side : 1;
        const pixelData = getResizedPixelData(width, height);
        const data = new Float32Array(shape[0] * shape[1]);

        for (let batch = 0; batch < shape[0]; batch++) {
            const offset = batch * shape[1];
            fillFlatGrayData(data.subarray(offset, offset + shape[1]), pixelData, width, height);
        }

        return new ort.Tensor('float32', data, shape);
    }

    if (shape.length === 3) {
        const height = shape[1];
        const width = shape[2];
        const pixelData = getResizedPixelData(width, height);
        const data = new Float32Array(shape[0] * shape[1] * shape[2]);

        for (let batch = 0; batch < shape[0]; batch++) {
            const offset = batch * width * height;
            fillFlatGrayData(data.subarray(offset, offset + (width * height)), pixelData, width, height);
        }

        return new ort.Tensor('float32', data, shape);
    }

    const layout = getInputLayout(shape);
    const channels = layout === 'NHWC' ? shape[3] : shape[1];
    const height = layout === 'NHWC' ? shape[1] : shape[2];
    const width = layout === 'NHWC' ? shape[2] : shape[3];

    const imageData = getResizedPixelData(width, height);
    const data = new Float32Array(shape[0] * shape[1] * shape[2] * shape[3]);

    for (let h = 0; h < height; h++) {
        for (let w = 0; w < width; w++) {
            const srcOffset = (h * width + w) * 4;
            const r = imageData[srcOffset] / 255;
            const g = imageData[srcOffset + 1] / 255;
            const b = imageData[srcOffset + 2] / 255;

            if (layout === 'NHWC') {
                const base = ((h * width) + w) * channels;

                if (channels === 1) {
                    data[base] = r;
                }
                else {
                    data[base] = r;
                    data[base + 1] = g;
                    data[base + 2] = b;
                }
            }
            else {
                const pixelIndex = h * width + w;

                if (channels === 1) {
                    data[pixelIndex] = r;
                }
                else {
                    const channelSize = width * height;
                    data[pixelIndex] = r;
                    data[channelSize + pixelIndex] = g;
                    data[(channelSize * 2) + pixelIndex] = b;
                }
            }
        }
    }

    return new ort.Tensor('float32', data, shape);
}

async function loadSelectedModel() {
    const modelPath = modelSelect.value;

    if (!modelPath) return;

    modelStatus.textContent = `Ładowanie modelu: ${modelPath}`;
    modelSelect.disabled = true;

    try {
        session = await ort.InferenceSession.create(getModelAssetUrl(modelPath), {
            executionProviders: ['wasm']
        });

        const inputName = session.inputNames[0];
        const outputName = session.outputNames[0];

        if (!inputName || !outputName) throw new Error('Model ONNX nie ma poprawnych wejść/wyjść.');

        modelInputShape = resolveInputShape(session.inputMetadata?.[inputName]);
        modelInputRank = modelInputShape.length;
        modelOutputName = outputName;

        const outputDims = session.outputMetadata?.[outputName]?.dimensions || [];
        const nClasses = Number.isInteger(outputDims[outputDims.length - 1])
            ? outputDims[outputDims.length - 1]
            : 10;

        classLabels = await loadModelMetadata(modelPath, nClasses);
        modelStatus.textContent = `Załadowano model: ${modelPath}`;
        clearCanvas();
    }
    catch (error) {
        session = null;
        classLabels = [];
        predDiv.textContent = `Nie udało się załadować modelu: ${modelPath}`;
        modelStatus.textContent = 'Błąd ładowania modelu.';
        probabilitiesContainer.innerHTML = '';
        console.error(error);
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

    if (!drag) return;

    drag = false;
    predictModel();
}

function handlePointerTap(event) {
    const pointerType = event.pointerType || '';

    if (pointerType !== 'touch') return;

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

    if (!drag) return;

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
    if (!session) return;
    void predictModelAsync();
}

async function predictModelAsync() {
    const inputName = session.inputNames[0];

    try {
        const inputTensor = preprocessInputTensor(modelInputShape);
        const outputs = await session.run({ [inputName]: inputTensor });
        const outputTensor = outputs[modelOutputName] || outputs[session.outputNames[0]];
        const rawScores = Array.from(outputTensor?.data || []);

        if (rawScores.length === 0) return;

        const predictionProbabilities = normalizeScores(rawScores);
        const bestClassIndex = predictionProbabilities.reduce((bestIndex, currentValue, currentIndex, arr) => {
            return currentValue > arr[bestIndex] ? currentIndex : bestIndex;
        }, 0);

        if (!classLabels || classLabels.length === 0) {
            classLabels = Array.from({ length: predictionProbabilities.length }, (_, index) => `${index}`);
        }

        predDiv.textContent = `Klasyfikacja: ${classLabels[bestClassIndex]}`;
        renderProbabilityBars(classLabels, predictionProbabilities);
    }
    catch (error) {
        const errorMessage = error?.message || 'nieznany błąd';
        predDiv.textContent = `Błąd inferencji modelu ONNX: ${errorMessage}`;
        modelStatus.textContent = `Błąd inferencji (${modelInputRank}D).`;
        console.error(error);
    }
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

placePredictionsByViewport();
