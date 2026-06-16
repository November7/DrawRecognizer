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
const {
    discoverModels,
    normalizeScores,
    resolveInputShape,
    loadModelMetadataFromOnnx,
    loadLabelsFromLabelEncoder,
    loadLabelsFromMetadataProps,
    resolveModelOutput,
    resolveOutputTensor
} = window.DrawRecognizerModelUtils;

let drag                        =   false;
let pos                         =   { x: 0, y: 0 };
let lastTapTime                 =   0;
let session                     =   null;
let modelInputShape             =   [1, 28, 28, 1];
let modelInputRank              =   4;
let modelOutputName             =   '';
let currentModelPath            =   '';
let classLabels                 =   [];

modelSelect.addEventListener('change', loadSelectedModel);

// utility functions

function getModelAssetUrl(pathName = '') {
    return new URL(pathName, modelsBaseUrl).href;
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


function renderModelOptions(models) {
    modelSelect.innerHTML = '';

    models.forEach(modelName => {
        const option = document.createElement('option');
        option.value = modelName;
        option.textContent = modelName;
        modelSelect.appendChild(option);
    });
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

function buildFlatGrayTensor(shape, width, height) {
    const pixelData = getResizedPixelData(width, height);
    const sampleSize = width * height;
    const vectorSize = shape.slice(1).reduce((acc, val) => acc * val, 1);
    const data = new Float32Array(shape[0] * vectorSize);

    for (let batch = 0; batch < shape[0]; batch++) {
        const offset = batch * vectorSize;
        fillFlatGrayData(data.subarray(offset, offset + Math.min(vectorSize, sampleSize)), pixelData, width, height);
    }

    return new ort.Tensor('float32', data, shape);
}

function fillFlatGrayData(target, pixelData, width, height) {
    const count = Math.min(target.length, width * height);

    for (let i = 0; i < count; i++) {
        const srcOffset = i * 4;
        target[i] = pixelData[srcOffset] / 255;
    }
}

function preprocessInputTensor(shape) {
    const rank = shape.length;

    // 2D: [N, features]
    if (rank === 2) {
        const features = shape[1];
        const side = Math.round(Math.sqrt(features));
        return buildFlatGrayTensor(shape, side, side);
    }

    // 3D: [N, H, W]
    if (rank === 3) {
        return buildFlatGrayTensor(shape, shape[1], shape[2]);
    }

    // 4D: obraz
    const [N, D1, D2, D3] = shape;

    const isNHWC = (D3 === 1 || D3 === 3);
    const channels = isNHWC ? D3 : D1;
    const height   = isNHWC ? D1 : D2;
    const width    = isNHWC ? D2 : D3;

    const imageData = getResizedPixelData(width, height);
    const data = new Float32Array(N * channels * width * height);

    for (let h = 0; h < height; h++) {
        for (let w = 0; w < width; w++) {
            const src = (h * width + w) * 4;
            const r = imageData[src] / 255;
            const g = imageData[src + 1] / 255;
            const b = imageData[src + 2] / 255;

            if (isNHWC) {
                const base = (h * width + w) * channels;
                if (channels === 1) data[base] = r;
                else {
                    data[base] = r;
                    data[base + 1] = g;
                    data[base + 2] = b;
                }
            } else {
                const pixelIndex = h * width + w;
                const channelSize = width * height;

                if (channels === 1) {
                    data[pixelIndex] = r;
                } else {
                    data[pixelIndex] = r;
                    data[channelSize + pixelIndex] = g;
                    data[channelSize * 2 + pixelIndex] = b;
                }
            }
        }
    }

    return new ort.Tensor('float32', data, shape);
}


function tryUpdateInputShapeFromOrtError(errorMessage) {
    const message = String(errorMessage || '');
    const matches = Array.from(message.matchAll(/index:\s*(\d+)\s*Got:\s*\d+\s*Expected:\s*(\d+)/g));

    if (matches.length === 0) return false;

    const updatedShape = [...modelInputShape];

    for (const match of matches) {
        const index = Number.parseInt(match[1], 10);
        const expected = Number.parseInt(match[2], 10);

        if (!Number.isInteger(index) || !Number.isInteger(expected) || expected <= 0) continue;
        if (index < 0 || index >= updatedShape.length) continue;

        updatedShape[index] = expected;
    }

    const changed = updatedShape.some((value, index) => value !== modelInputShape[index]);

    if (!changed) return false;

    modelInputShape = updatedShape;
    modelInputRank = updatedShape.length;
    modelStatus.textContent = `Dopasowano wejście modelu: [${updatedShape.join(', ')}]`;
    return true;
}

async function loadSelectedModel() {
    const modelPath = modelSelect.value;

    if (!modelPath) return;

    modelStatus.textContent = `Ładowanie modelu: ${modelPath}`;
    modelSelect.disabled = true;
    currentModelPath = modelPath;

    try {
        session = await ort.InferenceSession.create(getModelAssetUrl(modelPath), {
            executionProviders: ['wasm']
        });

        const inputName = session.inputNames[0];
        const outputName = session.outputNames[0];

        if (!inputName || !outputName) throw new Error('Model ONNX nie ma poprawnych wejść/wyjść.');

        modelInputShape = resolveInputShape(session.inputMetadata?.[inputName], currentModelPath);
        modelInputRank = modelInputShape.length;
        modelOutputName = outputName;

        const outputSelection = resolveModelOutput(session.outputNames, session.outputMetadata);

        modelOutputName = outputSelection.name || outputName;

        const nClasses = outputSelection.classCount;

        let labels = await loadModelMetadataFromOnnx(session, nClasses);

        if (!labels || labels.length === 0)
            labels = await loadLabelsFromMetadataProps(getModelAssetUrl, modelPath);

        if (!labels || labels.length === 0)
            labels = await loadLabelsFromLabelEncoder(getModelAssetUrl, modelPath);

        if (!labels || labels.length === 0)
            labels = Array.from({ length: nClasses }, (_, i) => String(i));


        classLabels = labels;



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

function updatePos(e) {
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
    updatePos(event);
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

    handleDoubleTap(event, false);
}

function handleDoubleTap(event, shouldPreventDefault) {
    const now = Date.now();

    if (now - lastTapTime <= DOUBLE_TAP_THRESHOLD_MS) {
        if (shouldPreventDefault && event?.preventDefault) {
            event.preventDefault();
        }
        clearCanvas();
        lastTapTime = 0;
        return;
    }

    lastTapTime = now;
}

function handleLegacyTouchTap(event) {
    handleDoubleTap(event, true);
}

function draw(event) {
    event.preventDefault();

    if (!drag) return;

    ctx.beginPath();
    ctx.lineWidth = 10;
    ctx.strokeStyle = 'white';
    ctx.lineCap = 'round';
    ctx.moveTo(pos.x, pos.y);
    updatePos(event);
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
    void predictModelAsync(0);
}

async function predictModelAsync(retryCount = 0) {
    const inputName = session.inputNames[0];

    try {
        const inputTensor = preprocessInputTensor(modelInputShape);
        const outputs = await session.run({ [inputName]: inputTensor });
        const selectedOutput = resolveOutputTensor(outputs, modelOutputName);

        if (!selectedOutput) return;

        modelOutputName = selectedOutput.name;

        const rawScores = Array.from(selectedOutput.tensor.data || []);

        if (rawScores.length === 0) return;

        const predictionProbabilities = normalizeScores(rawScores);
        const bestClassIndex = predictionProbabilities.reduce((bestIndex, currentValue, currentIndex, arr) => {
            return currentValue > arr[bestIndex] ? currentIndex : bestIndex;
        }, 0);

        if (!classLabels || classLabels.length !== predictionProbabilities.length) {
            classLabels = Array.from({ length: predictionProbabilities.length }, (_, index) => `${index}`);
        }

        predDiv.textContent = `Klasyfikacja: ${classLabels[bestClassIndex]}`;
        renderProbabilityBars(classLabels, predictionProbabilities);
    }
    catch (error) {
        const errorMessage = error?.message || 'nieznany błąd';

        if (retryCount < 1 && tryUpdateInputShapeFromOrtError(errorMessage)) {
            await predictModelAsync(retryCount + 1);
            return;
        }

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

function bindEvents(events, handler, options) {
    events.forEach(eventName => drawBoard.addEventListener(eventName, handler, options));
}

if ('PointerEvent' in window) {
    bindEvents(['pointerdown'], startDrawing);
    bindEvents(['pointermove'], draw);
    bindEvents(['pointerup', 'pointercancel', 'pointerleave'], stopDrawing);
    bindEvents(['pointerup'], handlePointerTap);
}
else {
    bindEvents(['mousedown'], startDrawing);
    bindEvents(['mousemove'], draw);
    bindEvents(['mouseup', 'mouseleave'], stopDrawing);
    bindEvents(['touchstart'], startDrawing, { passive: false });
    bindEvents(['touchmove'], draw, { passive: false });
    bindEvents(['touchend', 'touchcancel'], stopDrawing, { passive: false });
    bindEvents(['touchend'], handleLegacyTouchTap, { passive: false });
}

async function initializeApp() {
    clearCanvas();

    try {
        const availableModels = await discoverModels({ getModelAssetUrl, modelsBaseUrl, fallbackModels, plSort });

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
