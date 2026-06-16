(function () {
    function normalizeModelName(name) {
        return String(name || '')
            .replace(/\\/g, '/')
            .replace(/^\.\//, '')
            .replace(/^\/+/, '')
            .replace(/\/+$/, '')
            .trim();
    }

    function parseOnnxList(entries, plSort) {
        if (!Array.isArray(entries)) return [];

        return Array.from(new Set(entries
            .map(normalizeModelName)
            .filter(entry => entry.toLowerCase().endsWith('.onnx')))).sort(plSort);
    }

    async function fetchJson(url) {
        try {
            const response = await fetch(url);

            if (!response.ok) return null;

            return await response.json();
        }
        catch {
            return null;
        }
    }

    async function discoverFromManifest(getModelAssetUrl, plSort) {
        const data = await fetchJson(getModelAssetUrl('index.json'));
        return parseOnnxList(data, plSort);
    }

    async function filterAvailableModels(getModelAssetUrl, models) {
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

    async function discoverModels({ getModelAssetUrl, modelsBaseUrl, fallbackModels, plSort }) {
        const fromManifest = await discoverFromManifest(getModelAssetUrl, plSort);

        if (fromManifest.length > 0) return fromManifest;

        try {
            const response = await fetch(modelsBaseUrl.href);

            if (response.ok) {
                const contentType = response.headers.get('content-type') || '';

                if (contentType.includes('application/json')) {
                    const data = await response.json();
                    const models = parseOnnxList(data, plSort);

                    if (models.length > 0) return models;
                }

                const html = await response.text();
                const doc = new DOMParser().parseFromString(html, 'text/html');
                const links = Array.from(doc.querySelectorAll('a'));
                const discoveredModels = parseOnnxList(links.map(link => link.getAttribute('href') || ''), plSort);

                if (discoveredModels.length > 0) return discoveredModels;
            }
        }
        catch {
            // nop
        }

        const availableFallbackModels = await filterAvailableModels(getModelAssetUrl, fallbackModels);

        if (availableFallbackModels.length > 0) return availableFallbackModels;

        throw new Error('Nie udało się wykryć modeli. Dodaj models/index.json albo upewnij się, że pliki .onnx są dostępne.');
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
        const dims = inputMeta?.dimensions || [];
        const shape = dims.map(d => (Number.isInteger(d) && d > 0 ? d : 1));
        if (shape.length === 4) {
            return shape; // [N, C/H, H/W, C]
        }
        if (shape.length === 3) {
            return shape;
        }
        if (shape.length === 2) {
            return shape;
        }
        return [1, 28, 28, 1];
    }


    function getModelMetadataCandidates(modelPath) {
        const cleanedPath = modelPath.replace(/\\/g, '/').replace(/^\/+/, '');
        const lastSlashIndex = cleanedPath.lastIndexOf('/');
        const directoryPath = lastSlashIndex >= 0 ? cleanedPath.slice(0, lastSlashIndex + 1) : '';
        const fileName = lastSlashIndex >= 0 ? cleanedPath.slice(lastSlashIndex + 1) : cleanedPath;
        const baseName = fileName.replace(/\.onnx$/i, '');

        return [`${directoryPath}metadata.json`, `${directoryPath}${baseName}.metadata.json`];
    }

    async function loadModelMetadataFromOnnx(session, nClasses) {
        const meta = session.metadata?.customMetadataMap || {};

        const keys = ["labels", "label_names", "class_names", "classes"];

        for (const key of keys) {
            if (meta[key]) {
                const raw = meta[key].trim();
                const labels = raw.split(/[,;]+/).map(s => s.trim());
                if (labels.length > 0) return labels;
            }
        }

        // fallback
        return Array.from({ length: nClasses }, (_, i) => String(i));
    }

    async function loadLabelsFromMetadataProps(getModelAssetUrl, modelPath) {
        console.log(`Próbuję załadować etykiety z właściwości modelu dla ${modelPath}...`);
        const url = getModelAssetUrl(modelPath);
        const response = await fetch(url);
        const buffer = await response.arrayBuffer();
        const bytes = new Uint8Array(buffer);

        const labels = [];

        // Szukamy sekwencji "0a 06 6c 61 62 65 6c 73" → key="labels"
        const key = [0x0a, 0x06, 0x6c, 0x61, 0x62, 0x65, 0x6c, 0x73];

        for (let i = 0; i < bytes.length - key.length; i++) {

            // sprawdzamy, czy trafiliśmy na "labels"
            let match = true;
            for (let j = 0; j < key.length; j++) {
                if (bytes[i + j] !== key[j]) {
                    match = false;
                    break;
                }
            }
            if (!match) continue;

            // po "labels" jest pole value: 12 <len> <bytes>
            const tag = bytes[i + key.length];      // powinno być 0x12
            const len = bytes[i + key.length + 1];  // długość stringa

            if (tag !== 0x12) continue;
            if (len <= 0 || len > 32) continue;     // sanity check

            const start = i + key.length + 2;
            const end = start + len;

            const valueBytes = bytes.slice(start, end);
            const value = new TextDecoder().decode(valueBytes);

            return value.split(/[,;]+/).map(s => s.trim());
        }

        return null;
    }



    async function loadLabelsFromLabelEncoder(getModelAssetUrl, modelPath) {
        const url = getModelAssetUrl(modelPath);
        const response = await fetch(url);
        const buffer = await response.arrayBuffer();
        const bytes = new Uint8Array(buffer);

        let pos = 0;

        function readVarint() {
            let result = 0, shift = 0;
            while (true) {
                const byte = bytes[pos++];
                result |= (byte & 0x7F) << shift;
                if ((byte & 0x80) === 0) break;
                shift += 7;
            }
            return result;
        }

        function readBytes(length) {
            const out = bytes.slice(pos, pos + length);
            pos += length;
            return out;
        }

        const labels = [];

        while (pos < bytes.length) {
            const tag = readVarint();
            const wireType = tag & 7;

            if (wireType === 2) {
                const length = readVarint();
                const chunkStart = pos;
                const chunkEnd = pos + length;

                // Szukamy binarnego pola "classes_strings"
                const chunk = bytes.slice(chunkStart, chunkEnd);
                const text = new TextDecoder().decode(chunk);

                if (text.includes("classes_strings")) {
                    let innerPos = 0;

                    while (innerPos < chunk.length) {
                        const innerTag = chunk[innerPos++];
                        const innerWire = innerTag & 7;

                        if (innerWire === 2) {
                            const strLen = chunk[innerPos++];
                            const strBytes = chunk.slice(innerPos, innerPos + strLen);
                            innerPos += strLen;

                            const str = new TextDecoder().decode(strBytes).trim();
                            if (str.length > 0) labels.push(str);
                        } else {
                            innerPos++;
                        }
                    }
                }

                pos += length;
            } else {
                if (wireType === 0) readVarint();
                else if (wireType === 5) pos += 4;
                else if (wireType === 1) pos += 8;
                else break;
            }
        }

        return labels.length > 0 ? labels : null;
    }





    function getLastKnownDimension(dims) {
        if (!Array.isArray(dims) || dims.length === 0) return null;

        for (let index = dims.length - 1; index >= 0; index--) {
            const value = dims[index];

            if (Number.isInteger(value) && value > 0) return value;
        }

        return null;
    }

    function resolveModelOutput(outputNames, outputMetadata) {
        const metadata = outputMetadata || {};
        const entries = (outputNames || []).map(name => {
            const dims = metadata?.[name]?.dimensions || [];
            const classCount = getLastKnownDimension(dims);

            return {
                name,
                classCount: Number.isInteger(classCount) && classCount > 1 ? classCount : null,
                rank: Array.isArray(dims) ? dims.length : 0
            };
        });

        const classified = entries.filter(entry => Number.isInteger(entry.classCount));

        if (classified.length > 0) {
            classified.sort((left, right) => {
                if (left.classCount !== right.classCount) return left.classCount - right.classCount;
                return left.rank - right.rank;
            });

            return classified[0];
        }

        return {
            name: (outputNames || [])[0] || '',
            classCount: null,
            rank: 0
        };
    }

    function resolveOutputTensor(outputs, preferredOutputName) {
        const entries = Object.entries(outputs || {})
            .map(([name, tensor]) => ({
                name,
                tensor,
                length: Array.isArray(tensor?.data) || ArrayBuffer.isView(tensor?.data) ? tensor.data.length : 0
            }))
            .filter(entry => entry.length > 0);

        if (entries.length === 0) return null;

        const preferred = entries.find(entry => entry.name === preferredOutputName);

        if (preferred) return preferred;

        const withClasses = entries.filter(entry => entry.length > 1);

        if (withClasses.length > 0) {
            withClasses.sort((left, right) => left.length - right.length);
            return withClasses[0];
        }

        return entries[0];
    }

    window.DrawRecognizerModelUtils = {
        discoverModels,
        normalizeScores,
        resolveInputShape,
        loadModelMetadataFromOnnx,
        loadLabelsFromLabelEncoder,
        loadLabelsFromMetadataProps,
        resolveModelOutput,
        resolveOutputTensor
    };
})();
