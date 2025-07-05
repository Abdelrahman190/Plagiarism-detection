const express = require('express');
const multer = require('multer');
const fs = require('fs').promises;
const fsSync = require('fs');
const path = require('path');
const onnx = require('onnxruntime-node');

const app = express();
const upload = multer({ dest: 'uploads/' });
const PORT = 2025;
const MODEL_PATH = path.join(__dirname, 'plagiarism_full_pipeline4.onnx');

let session = null;

app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', '*');
  res.setHeader('Access-Control-Allow-Headers', '*');
  next();
});
app.use(express.urlencoded({ extended: false }));
app.use(express.json());


app.post('/upload', upload.array('files'), async (req, res) => {
  if (!session) return res.status(503).json({ error: 'Model not loaded yet' });

  const files = req.files;
  if (!files || files.length < 2)
    return res.status(400).json({ error: 'Upload at least two files.' });

  try {
    const texts = await Promise.all(files.map(f => fs.readFile(f.path, 'utf-8')));
    const names = files.map(f => f.originalname);
    files.forEach(f => fs.unlink(f.path).catch(() => {}));

    const results = [];
    for (let i = 0; i < texts.length; i++) {
      for (let j = i + 1; j < texts.length; j++) {
        try {
          const isPlagiarized = await runOnnxInference(texts[i], texts[j]);
          results.push({
            file1: names[i],
            file2: names[j],
            plagiarized: isPlagiarized === 1
          });
        } catch (err) {
          results.push({
            file1: names[i],
            file2: names[j],
            error: err.message
          });
        }
      }
    }

    res.json({ results });
  } catch (err) {
    console.error("Upload failed:", err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

async function runOnnxInference(text1, text2) {
  const feeds = {
    [session.inputNames[0]]: new onnx.Tensor('string', [text1], [1, 1]),
    [session.inputNames[1]]: new onnx.Tensor('string', [text2], [1, 1])
  };
  const output = await session.run(feeds);
  return Number(output[session.outputNames[0]].data[0]);
}

async function startServer() {
  try {
    console.log("Checking for model at:", MODEL_PATH);
    if (!fsSync.existsSync(MODEL_PATH)) {
      console.error("Model file not found");
      process.exit(1);
    }

    console.log("Loading model...");
    console.time("Load time");
    session = await onnx.InferenceSession.create(MODEL_PATH);
    console.timeEnd("Load time");
    console.log("Model loaded");

    app.listen(PORT, () => console.log(`Server running at http://localhost:${PORT}`));
  } catch (err) {
    console.error("Failed to load model:", err);
  }
}

startServer();
