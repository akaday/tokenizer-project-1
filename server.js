const express = require('express');
const bodyParser = require('body-parser');

const app = express();
app.use(bodyParser.json());

function tokenizeText(text) {
    // Implement your tokenization logic here
    return text.split(' ');
}

app.post('/tokenize', (req, res) => {
    const text = req.body.text;
    const tokens = tokenizeText(text);
    res.json({ tokens });
});

app.listen(3000, () => {
    console.log('Server is running on port 3000');
});
