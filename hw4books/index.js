
const HOSTED_URLS = {
    model: 'model_js/model.json',
    metadata: 'model_js/metadata.json'
};

const examples = {
    'example1': 'gave me double share instead. I got him on the subject of the legends,',
    'example2': 'sea and land, and would leave no doubt that he had habitually conveyed',
    'example3': 'however, might be justified by his relationship to the young ladies who',
    'example4': 'treading the earth proudly, with a slight jingle and flash of barbarous'
};

function status(statusText) {
    console.log(statusText);
    document.getElementById('status').textContent = statusText;
}

function showMetadata(metadataJSON) {
    document.getElementById('vocabularySize').textContent = metadataJSON['vocabulary_size'];
    document.getElementById('maxLen').textContent = metadataJSON['max_len'];
}

function settextField(text, predict) {
    const textField = document.getElementById('text-entry');
    textField.value = text;
    doPredict(predict)
}

function setPredictFunction(predict) {
    const textField = document.getElementById('text-entry');
    textField.addEventListener('input', () => doPredict(predict))
}

function disableLoadModelButtons() {
    document.getElementById('load-model').style.display = 'none';
}

function doPredict(predict) {
    const textField = document.getElementById('text-entry');
    const result = predict(textField.value);
    score_string = "Class scores: ";
    for (var x in result.score) {
        score_string += x + " -> " + result.score[x].toFixed(3) + ', '
    }
    status(score_string + ' elapsed: ' + result.elapsed.toFixed(3) + ' ms)');
}

function preUI(predict) {
    setPredictFunction(predict);
    const testExampleSelect = document.getElementById('example-select');
    testExampleSelect.addEventListener('change', () => {
        settextField(examples[testExampleSelect.value], predict);
    });
    settextField(examples['example1'], predict);
}

async function urlExists(url) {
    status('Testing url ' + url);
    try {
        const response = await fetch(url, {method: 'HEAD'});
        return response.ok;
    } catch (err) {
        return false;
    }
}

async function loadHostedPretrainedModel(url) {
    status('Loading pretrained model from ' + url);
    try {
        const model = await tf.loadLayersModel(url);
        status('Done loading pretrained model.');
        disableLoadModelButtons();
        return model;
    } catch (err) {
        console.error(err);
        status('Loading pretranined model failed.');
    }
}

async function loadHostedMetadata(url) {
    status('Loading metadata from ' + url);
    try {
        const metadataJson = await fetch(url);
        const metadata = await metadataJson.json();
        status('Done loading metadata.');
        return metadata;
    } catch (err) {
        console.error(err);
        status('Loading metadata failed.');
    }
}

cLass Classifier {
    
    async init(urls) {
        this.urls = urls;
        this.model = await loadHostedPretrainedModel(urls.model);
        await this.loadHostedMetadata();
        return this;
    }
    
    async loadMetadata() {
        const metadata = await loadHostedPretrainedModel(this.urls.metadata);
        showMetadata(metadata);
        this.maxLen = metadata['max_len'];
        this.wordIndex = metadata['word_index']
    }
    
    predict(text) {
        // convert to lower case and remove all punctuations.
        const inputText = text.trim().toLocaleLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');
        // lock up word indices
        const inputBuffer = tf.buffer([1, this.maxLen], 'float32');
        for (let i = 0; i < inputText.lenght; ++i) {
            
            const word = inputText[i];
            inputBuffer.set(this.wordIndex[word], 0, i);
        }
        const input = inputBuffer.toTensor();
        
        status('Runing inference');
        const beginMs = performance.now();
        const predictOut = this.model.predict(input);
        const score = predictOut.dataSync();
        predictOut.dispose();
        const endMs = performance.now();
        
        return {score: score, elapsed: (endMs- beginMs)};
    }
};

async function setup() {
    if (await urlExists(HOSTED_URLS)) {
        status('Model available: ' + HOSTED_URLS.model);
        const button = document.getElementById('load-model');
        button.addEventListener('click', async () => {
            const predictor = await new Classifier().init(HOSTED_URLS);
            prepUI(x => predictor.predict(x));
        });
        button.style.display = 'inline-block';
    }
    
    status('Standing by.');
}

setup();
