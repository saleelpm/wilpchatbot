import { useState, useCallback } from 'react';
import OpenAI from 'openai';
import * as pdfjs from "../node_modules/pdfjs-dist/build/pdf";
import * as pdfjsWorker from "../node_modules/pdfjs-dist/build/pdf.worker";

import './GptChatBot.css';

const apiKey = 'OpenAI Key Here';
const embeddingModel = 'text-embedding-ada-002';
const completionModel = 'text-davinci-003';
const aiTemperature = 1;
const maxTokens = 2048;
const domainDb = "domainData.csv"

pdfjs.GlobalWorkerOptions.workerSrc = pdfjsWorker;

const openai = new OpenAI({
    apiKey: apiKey,
    dangerouslyAllowBrowser: true
});

function GptChatBot() {
    const [statusMessage, setStatusMessage] = useState('');
    const [ask, setAsk] = useState([]);
    const [chats] = useState([]);
    const [pdfs, setPdfs] = useState([]);
    const [dragOver, setDragOver] = useState(false);
    const [Embeddings] = useState([]);

    const userAsked = async (event) => {
        event.preventDefault();

        setStatusMessage('Sending request...');
        chats.push({ id: chats.length + 1, sender: "user", message: ask, time: new Date().toLocaleTimeString() });

        let response = "";
        if (Embeddings.length > 0) {
            var askEmbedding = await mostRelevantEmbedding(ask);
            var askDomain = questionWithEmbedding(ask, askEmbedding);
            response = await askOpenAiGpt(askDomain);
        }
        else {
            response = await askOpenAiGpt(ask);
        }

        if (response) {
            setStatusMessage('Response received.');
            chats.push({ id: chats.length + 1, sender: "gpt", message: response, time: new Date().toLocaleTimeString() });
        }
        reset();
    };

    const reset = () => {
        setAsk('');
    };

    const askOpenAiGpt = async (input) => {
        try {
            const answer = await openai.completions.create({
                model: completionModel,
                prompt: input,
                max_tokens: maxTokens,
            });

            if (answer.choices && answer.choices.length > 0) {
                setStatusMessage('Response received.');
                return answer.choices[0].text;
            } else {
                console.error('No choices returned from API');
                setStatusMessage('Failed to get a response from the API.');
            }
        } catch (err) {
            console.error(err);
            setStatusMessage('An error occurred during the analysis.');
        }
        return null;
    };

    const ChatMessageItem = ({ item: { message, sender, time } }) => {
        return (
            <div className={`chat-bubble ${sender == "user" ? "right" : ""}`}>
                <div className="chat-bubble__right">
                    <p className="user-name">{sender} {time}</p>
                    <p className="user-message">{message}</p>
                </div>
            </div>
        );
    };

    const handleFileChange = useCallback((selectedFiles) => {
        setPdfs(Array.from(selectedFiles));

    }, []);

    const handleDragOver = useCallback((event) => {
        event.preventDefault();
        setDragOver(true);
    }, []);

    const handleDragLeave = useCallback(() => {
        setDragOver(false);
    }, []);

    const handleDrop = useCallback((event) => {
        event.preventDefault();
        setDragOver(false);
        const files = event.dataTransfer.files;
        if (files.length) {
            setPdfs(Array.from(files));
        }
    }, [handleFileChange]);

    const handleLoadPdfs = async (event) => {
        event.preventDefault();
        if (!pdfs) {
            setStatusMessage('No file selected!');
            return;
        }

        setStatusMessage('Processing request...');

        for (let i = 0; i < pdfs.length; i++) {
            var pdfData = await getTextFromPDF(pdfs[i]);
            var vector = await createEmbedding(pdfData);
            Embeddings.push({
                file: pdfs[i].name,
                vector: vector,
                text: pdfData,
            });
        }

        setStatusMessage('Processed Successfully!');
    }
    async function createEmbedding(text) {
        try {
            const response = await openai.embeddings.create({
                model: embeddingModel,
                input: text,
            });

            return response.data[0].embedding;
        } catch (error) {
            console.error(`Error in createEmbedding: ${error}`);
        }
    }

    function cosineSimilarity(vec1, vec2) {
        try {
            if (vec1.length !== vec2.length) {
                throw new Error("Vectors must have the same length");
            }

            let dotProduct = 0;
            let magnitude1 = 0;
            let magnitude2 = 0;

            for (let i = 0; i < vec1.length; i++) {
                dotProduct += vec1[i] * vec2[i];
                magnitude1 += Math.pow(vec1[i], 2);
                magnitude2 += Math.pow(vec2[i], 2);
            }

            magnitude1 = Math.sqrt(magnitude1);
            magnitude2 = Math.sqrt(magnitude2);

            return dotProduct / (magnitude1 * magnitude2);
        } catch (error) {
            console.error(`Error in cosineSimilarity: ${error}`);
        }
    }

    const mostRelevantEmbedding = async (question) => {
        const questionVector = await createEmbedding(question);

        const mostSimilarEmbedding = Embeddings.reduce(
            (best, embedding) => {
                const similarity = cosineSimilarity(questionVector, embedding.vector);
                if (similarity > best.similarity) {
                    return { embedding, similarity };
                } else {
                    return best;
                }
            },
            { embedding: null, similarity: -Infinity }
        );

        if (mostSimilarEmbedding.embedding) {
            console.log(
                `Most relevant embedding for '${question}' was found in ${mostSimilarEmbedding.embedding.file
                }`
            );
        }

        return mostSimilarEmbedding.embedding;
    };

    function questionWithEmbedding(question, embedding) {
        const prompt = `Answer the question as truthfully as possible using the provided text, and if the answer is not contained within the text below, say "I don't know"
            Context: ${embedding.text}
            Q: ${question}
            A:
        `;
        return prompt;
    }

    async function getTextFromPDF(pdfFile) {
        if (!pdfFile) return;
        var pdf = await pdfjs.getDocument(URL.createObjectURL(pdfFile));
        return pdf.promise.then(function (pdf) { 
            var maxPages = pdf.numPages;
            var countPromises = []; 
            for (var j = 1; j <= maxPages; j++) {
                var page = pdf.getPage(j);
                countPromises.push(page.then(function (page) { 
                    var textContent = page.getTextContent();
                    return textContent.then(function (text) { 
                        return text.items.map(function (s) { return s.str; }).join(''); 
                    });
                }));
            }            
            return Promise.all(countPromises).then(function (texts) {
                return texts.join('');
            });
        });
    }

    return (
        <div className="App">
            <div className='main-area'>
                <nav className="nav-bar">
                    <h2>BITS WILP Chat Bot</h2>
                </nav>
                <main className="chat-box">
                    <div className="messages-wrapper">
                        {chats?.map((message) => (
                            <ChatMessageItem item={message} />
                        ))}
                    </div>

                    <form onSubmit={(event) => userAsked(event)} className="send-message">
                        <label htmlFor="messageInput" hidden>
                            Enter Message
                        </label>
                        <input
                            type="text"
                            id="ask"
                            name="ask"
                            value={ask}
                            className="form-input__input"
                            placeholder="type message..."
                            onChange={event => setAsk(event.target.value)}
                        />
                        <button type="submit">Ask</button>
                    </form>

                </main>
            </div>
            <div className='side-bar'>
                <div className={`drop-area ${dragOver ? 'drag-over' : ''}`}
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onClick={() => document.getElementById('fileUpload').click()}
                >
                    <input
                        id="fileUpload"
                        type="file"
                        onChange={
                            (e) => handleFileChange(e.target.files)
                        }
                        accept="pdf/*"
                        style={{ display: 'none' }}
                    />
                    {pdfs.length ? (
                        <p>Below file(s) will be used for training the GPT.</p>
                    ) : (
                        <p>Drag and drop or browse the pdfs for training the GPT.</p>
                    )}
                    <div>
                        {pdfs &&
                            pdfs.map(
                                (item) => (
                                    <div>{item.name}</div>
                                )
                            )
                        }
                    </div>
                </div>
                <button onClick={handleLoadPdfs} className="app-button">
                    Upload Documents
                </button>
                {statusMessage && <p className="status-message">{statusMessage}</p>}
            </div>
        </div>
    );
}

export default GptChatBot;
