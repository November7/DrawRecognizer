const drawBoard = document.getElementById('drawBoard');
const ctx = drawBoard.getContext('2d');
const predDiv = document.getElementById('predictions');
const modelSelect = document.getElementById('functionSelect');
const modelStatus = document.getElementById('modelStatus');
const probabilitiesContainer = document.getElementById('classProbabilities');

let drag = false;
let pos = { x: 0, y: 0 };
let model = null;
let classLabels = [];
const modelsBaseUrl = new URL('models/', window.location.href);

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

modelSelect.addEventListener('change', loadSelectedModel);

async function discoverModels() {
    const response = await fetch(modelsBaseUrl.href);
    
    if (!response.ok) {
        throw new Error('Nie udało się odczytać katalogu modeli.');
    }

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

    if (discoveredModels.length === 0) {
        throw new Error('Serwer nie udostępnia listingu katalogu models.');
    }

    return discoveredModels;
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
    } catch (error) {
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
        predictModel();
    } catch (error) {
        model = null;
        classLabels = [];
        predDiv.textContent = `Nie udało się załadować modelu: ${modelName}`;
        modelStatus.textContent = 'Błąd ładowania modelu.';
        probabilitiesContainer.innerHTML = '';
    } finally {
        modelSelect.disabled = false;
    }
}

function setPos(e) {
    pos.x = e.clientX - ctx.canvas.getBoundingClientRect().left;
    pos.y = e.clientY - ctx.canvas.getBoundingClientRect().top;
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

drawBoard.addEventListener('mousedown', function (event) {
    drag = true;
    setPos(event);
});

drawBoard.addEventListener('mouseup', function () {
    drag = false;
    predictModel();
});

drawBoard.addEventListener('mouseleave', function () {
    drag = false;
});

drawBoard.addEventListener('mousemove', function (event) {
    event.preventDefault();
    event.stopPropagation();

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
});

async function initializeApp() {
    clearCanvas();

    try {
        const availableModels = await discoverModels();

        renderModelOptions(availableModels);
        modelStatus.textContent = `Wykryto ${availableModels.length} modele.`;
        await loadSelectedModel();
    } catch (error) {
        predDiv.textContent = 'Nie udało się uruchomić aplikacji.';
        modelStatus.textContent = 'Serwer npx nie udostępnia katalogu models. Włącz directory listing albo dodaj endpoint JSON z listą modeli.';
        modelSelect.disabled = true;
        probabilitiesContainer.innerHTML = '';
    }
}

initializeApp();