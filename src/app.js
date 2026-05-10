/**
 *  Draw Recognizer - A web application for drawing and recognizing images using TensorFlow.js.
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
const plSort                    =   (left, right) => left.localeCompare(right, 'pl');
const modelsBaseUrl             =   new URL('models/', window.location.href);
const fallbackModels            =   ['Cyfry-Mnist', 'Cyfry-TM', 'XO-TM'];
const DOUBLE_TAP_THRESHOLD_MS   =   320;
const singleColumnMediaQuery    =   window.matchMedia('(max-width: 960px)');

let drag                        =   false;
let pos                         =   { x: 0, y: 0 };
let lastTapTime                 =   0;
let model                       =   null;
let classLabels                 =   [];

modelSelect.addEventListener('change', loadSelectedModel);

// utility functions

function getModelAssetUrl(modelName, fileName = '') {
    const relativePath = modelName ? `${modelName}/${fileName}` : fileName;
    return new URL(relativePath, modelsBaseUrl).href;
}

function normalizeModelName(name) {
    return String(name || '')
        .replace(/\\/g, '/')
        .replace(/\/+$/, '')
        .split('/')
        .filter(Boolean)
        .pop() || '';
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
        const response = await fetch(getModelAssetUrl('', 'index.json'));

        if (!response.ok) return [];

        const data = await response.json();

        if (!Array.isArray(data)) return [];

        return Array.from(new Set(data.map(normalizeModelName).filter(Boolean))).sort(plSort);
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



async function discoverModels() {
    const fromManifest = await discoverFromManifest();

    if (fromManifest.length > 0) return fromManifest;

    try {
        const response = await fetch(modelsBaseUrl.href);

        if (response.ok) {
            const contentType = response.headers.get('content-type') || '';

            if (contentType.includes('application/json')) {
                const data = await response.json();

                if (Array.isArray(data)) return Array.from(new Set(data.map(normalizeModelName).filter(Boolean))).sort(plSort); 
            }

            const html              = await response.text();
            const doc               = new DOMParser().parseFromString(html, 'text/html');
            const links             = Array.from(doc.querySelectorAll('a'));
            const discoveredModels  = Array.from(new Set(links.map(link => link.getAttribute('href') || '')
                                                              .map(href => href.replace(/\\/g, '/'))
                                                              .map(href => href.split('?')[0].split('#')[0])
                                                              .filter(href => href.endsWith('/') && href !== '../' && href !== './')
                                                              .map(href => href.replace(/\/+$/, ''))
                                                              .map(href => href.split('/').filter(Boolean).pop() || '')
                                                              .map(normalizeModelName)
                                                              .filter(modelName => modelName && modelName.toLowerCase() !== 'models'))).sort(plSort);

            if (discoveredModels.length > 0) return discoveredModels;
        }
    }
    catch {
        //nop
    }

    const availableFallbackModels = await filterAvailableModels(fallbackModels);
    if (availableFallbackModels.length > 0) return availableFallbackModels;

    throw new Error('Nie udało się wykryć modeli. Dodaj models/index.json albo upewnij się, że model.json jest dostępny.');
}

function renderModelOptions(models) {
    modelSelect.innerHTML = '';

    models.forEach(modelName => {
        const option        = document.createElement('option');
        option.value        = modelName;
        option.textContent  = modelName;
        modelSelect.appendChild(option);
    });
}

async function loadModelMetadata(modelName, nClasses) {
    try {
        const response = await fetch(getModelAssetUrl(modelName, 'metadata.json'));

        if (!response.ok) throw new Error('Brak metadanych modelu.');

        const data = await response.json();

        if (Array.isArray(data.labels) && data.labels.length > 0) return data.labels;
    } 
    catch (error) {
        console.warn(`Nie można załadować metadanych dla modelu "${modelName}": ${error.message}`);
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

        const probability       = probabilities[index] ?? 0;
        const percentage        = (probability * 100).toFixed(2); 

        const item              = el('div', 'probability-item');
        const row               = el('div', 'probability-row');
        const labelSpan         = el('span', 'probability-label', label);
        const valueSpan         = el('span', 'probability-value', `${percentage}%`);
        const track             = el('div', 'probability-track');
        const fill              = el('div', 'probability-fill');

        fill.style.width        = `${Math.min(Math.max(probability * 100, 0), 100)}%`;

        row.append(labelSpan, valueSpan);
        track.append(fill);
        item.append(row, track);
        probabilitiesContainer.append(item);

    });
}

async function loadSelectedModel() {
    const modelName = modelSelect.value;

    if (!modelName) return;

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
    pos.x       = point.x;
    pos.y       = point.y;
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
    if (!model) return;

    const imageData     = ctx.getImageData(0, 0, drawBoard.width, drawBoard.height);
    const width         = model.input.shape[1];
    const height        = model.input.shape[2];
    const depth         = model.input.shape[3];
    const nClasses      = model.output.shape[1];

    if (!classLabels || classLabels.length === 0) classLabels = Array.from({ length: nClasses }, (_, index) => `${index}`);

    const { predictionProbabilities, bestClassIndex } = tf.tidy(() => {
        const image             = tf.browser.fromPixels(imageData, depth);
        const resizedImage      = tf.image.resizeBilinear(image, [width, height]).expandDims(0);
        const normalizedImage   = tf.cast(resizedImage, 'float32').div(255.0);
        const prediction        = model.predict(normalizedImage);

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
    drawBoard.addEventListener('pointerdown',   startDrawing);
    drawBoard.addEventListener('pointermove',   draw);
    drawBoard.addEventListener('pointerup',     stopDrawing);
    drawBoard.addEventListener('pointerup',     handlePointerTap);
    drawBoard.addEventListener('pointercancel', stopDrawing);
    drawBoard.addEventListener('pointerleave',  stopDrawing);
}
else {
    drawBoard.addEventListener('mousedown',     startDrawing);
    drawBoard.addEventListener('mousemove',     draw);
    drawBoard.addEventListener('mouseup',       stopDrawing);
    drawBoard.addEventListener('mouseleave',    stopDrawing);
    drawBoard.addEventListener('touchstart',    startDrawing,           { passive: false });
    drawBoard.addEventListener('touchmove',     draw,                   { passive: false });
    drawBoard.addEventListener('touchend',      stopDrawing,            { passive: false });
    drawBoard.addEventListener('touchend',      handleLegacyTouchTap,   { passive: false });
    drawBoard.addEventListener('touchcancel',   stopDrawing,            { passive: false });
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

if (typeof singleColumnMediaQuery.addEventListener === 'function') {
    singleColumnMediaQuery.addEventListener('change', placePredictionsByViewport);
}
else if (typeof singleColumnMediaQuery.addListener === 'function') {
    singleColumnMediaQuery.addListener(placePredictionsByViewport);
}