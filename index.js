import express from 'express';
import bodyParser from 'body-parser';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings } from '@langchain/openai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import OpenAI from 'openai';
import cheerio from 'cheerio';
import dotenv from 'dotenv';
import cors from 'cors';
import axios from 'axios';
import { clerkMiddleware, clerkClient, requireAuth, getAuth } from '@clerk/express'
// Load environment variables
dotenv.config();

const app = express();
const port = 3005;

app.use(cors({
  origin: "https://vidyagiri.vercel.app", // Set the allowed origin explicitly
  credentials: true, // Allow credentials (cookies, authorization headers, etc.)
  methods: ["GET", "POST", "PUT", "DELETE"],
  allowedHeaders: ["Content-Type", "Authorization"],
}));
// app.options("*", cors()); 
app.use(bodyParser.json());
app.use(clerkMiddleware());

// Configure OpenAI (Groq) and embeddings
const openai = new OpenAI({
  baseURL: 'https://api.groq.com/openai/v1',
  apiKey: process.env.GROQ_API_KEY,
});

const embeddings = new OpenAIEmbeddings({
  openAIApiKey: process.env.OPENAI_API_KEY, // Using OpenAI for embeddings
});

// Dummy detectVarkStyle function (replace with your own logic)
async function detectVarkStyle(message) {
  // For now, just return 'visual'. You can implement detection logic here.
  return 'visual';
}

// VARK style prompts
const varkPrompts = {
  visual: `Format the response with emphasis on visual organization:
- Use bullet points, numbered lists, and clear hierarchies.
- Include suggestions for diagrams, charts, or mind maps where applicable.
- Organize information in a structured, visual manner.
- Avoid lengthy paragraphs.
- Use spatial organization and layout to convey relationships.`,
  auditory: `Format the response for auditory learners:
- Emphasize verbal explanations and discussions.
- Suggest audio resources and verbal exercises.
- Include dialogue-style explanations.
- Recommend group discussions and verbal practice.
- Format as if explaining in a conversation.`,
  readWrite: `Format the response for read/write learners:
- Provide detailed written explanations.
- Include relevant terminology and definitions.
- Organize information in text-based formats.
- Suggest reading materials and writing exercises.
- Use clear, concise written language.`,
  kinesthetic: `Format the response for kinesthetic learners:
- Provide a brief definition of the topic.
- Include links to labs, interactive tools, or simulations where users can gain hands-on experience.
- Focus on practical applications and real-world scenarios.
- Avoid lengthy explanations.`
};

// Function to rephrase input using Groq (OpenAI)
async function rephraseInput(inputString) {
  console.log(`Rephrasing input...`);
  const groqResponse = await openai.chat.completions.create({
    model: "llama-3.3-70b-versatile",
    messages: [
      { role: "system", content: "You are a rephraser and always respond with a rephrased version of the input that is given to a search engine API. Always be succint and use the same words as the input. ONLY RETURN THE REPHRASED VERSION OF THE INPUT." },
      { role: "user", content: inputString },
    ],
  });
  console.log(`Input rephrased.`);
  return groqResponse.choices[0].message.content;
}

async function rephraseInputDiagram(inputString) {
  console.log(`Rephrasing input...`);
  const groqResponse = await openai.chat.completions.create({
    model: "llama-3.3-70b-versatile",
    messages: [
      { role: "system", content: "You are a rephraser and always respond with a rephrased version of the input that is given to a Rapid API that generates diagrams from query.So repharese the input so that it can be used to generate diagrams based on plantUML format. ONLY RETURN THE REPHRASED VERSION OF THE INPUT." },
      { role: "user", content: inputString },
    ],
  });
  console.log(`Input rephrased.`);
  return groqResponse.choices[0].message.content;
}

// Function to generate follow-up questions based on VARK style
async function generateVarkStyleFollowUpQuestions(responseText, varkStyle) {
  const stylePrompts = {
    visual: "Focus on visualization and diagram-related questions",
    auditory: "Focus on discussion and verbal explanation questions",
    readWrite: "Focus on reading and writing-based questions",
    kinesthetic: "Focus on practical application and hands-on activity questions"
  };

  const groqResponse = await openai.chat.completions.create({
    model: "llama-3.3-70b-versatile",
    messages: [
      {
        role: "system",
        content: `You are a question generator. Generate 3 follow-up questions based on the provided text. ${stylePrompts[varkStyle]}. Return the questions in an array format.`
      },
      {
        role: "user",
        content: `Generate 3 follow-up questions based on the following text:\n\n${responseText}\n\nReturn the questions in the following format: ["Question 1", "Question 2", "Question 3"]`
      }
    ],
  });
  return JSON.parse(groqResponse.choices[0].message.content);
}

// Function to perform a search using the Serper API
async function searchWithSerper(query, numberOfResults = 10) {
  try {
    const response = await fetch('https://google.serper.dev/search', {
      method: 'POST',
      headers: {
        'X-API-KEY': process.env.SERPER_API_KEY,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        q: query,
        num: numberOfResults
      })
    });

    if (!response.ok) {
      throw new Error(`Serper API error: ${response.status}`);
    }

    const data = await response.json();
    return data.organic || [];
  } catch (error) {
    console.error('Serper search error:', error);
    return [];
  }
}

// Function to use the search engine to get and process sources
async function searchEngineForSources(message, numberOfPagesToScan = 4) {
  console.log(`Initializing Search Engine Process with Serper`);

  // Rephrase the message for better search results
  const rephrasedMessage = await rephraseInput(message);
  console.log(`Rephrased search query: ${rephrasedMessage}`);

  // Get search results from Serper
  const searchResults = await searchWithSerper(rephrasedMessage, numberOfPagesToScan);
  console.log(`Received ${searchResults.length} results from Serper`);

  // Normalize the search results and filter out unwanted links
  const normalizedData = searchResults
    .filter(result => result.link && !result.link.includes('google.com'))
    .map(result => ({
      title: result.title,
      link: result.link,
      snippet: result.snippet
    }));

  // Process and vectorize the content from each result
  return await Promise.all(normalizedData.map(fetchAndProcess));
}

// Function to fetch page content with error handling
const fetchPageContent = async (link) => {
  console.log(`Fetching page content for ${link}`);
  try {
    const response = await fetch(link, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
      },
      timeout: 5000
    });

    if (!response.ok) {
      console.warn(`Failed to fetch ${link}: ${response.status}`);
      return "";
    }

    const text = await response.text();
    return extractMainContent(text);
  } catch (error) {
    console.error(`Error fetching ${link}:`, error);
    return '';
  }
};

// Function to extract main content from HTML using Cheerio
function extractMainContent(html) {
  try {
    const $ = cheerio.load(html);

    // Remove unnecessary elements
    $('script, style, nav, footer, iframe, img, header, aside, form, button').remove();

    // Try to extract from typical content containers
    let content = $('article, main, .content, .post-content').text();

    // If no specific container is found, fallback to body text
    if (!content.trim()) {
      content = $('body').text();
    }

    // Clean up and return the text
    return content
      .replace(/\s+/g, ' ')
      .replace(/\n+/g, ' ')
      .trim();
  } catch (error) {
    console.error('Error extracting content:', error);
    return '';
  }
}

// Function to process each search result and vectorize the content
const fetchAndProcess = async (item) => {
  const htmlContent = await fetchPageContent(item.link);

  // Skip if content is too short or empty
  if (!htmlContent || htmlContent.length < 250) {
    return null;
  }

  try {
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200
    });

    const splitText = await splitter.splitText(htmlContent);

    // Include the search snippet in metadata
    const metadata = {
      link: item.link,
      title: item.title,
      snippet: item.snippet
    };

    const vectorStore = await MemoryVectorStore.fromTexts(
      splitText,
      metadata,
      embeddings
    );

    return await vectorStore.similaritySearch(item.snippet, 2);
  } catch (error) {
    console.error(`Error processing ${item.link}:`, error);
    return null;
  }
};

// Main route handler for processing queries (POST /)
app.post('/', async (req, res) => {
  // const { userId } = getAuth(req); // Get user ID from Clerk
  const {
    message,
    returnSources = true,
    returnFollowUpQuestions = true,
    embedSourcesInLLMResponse = false,
    preferredStyle = null
  } = req.body;

  try {
    // Detect VARK style (or use the provided one)
    const varkStyle = preferredStyle || await detectVarkStyle(message);
    console.log(`Detected/Selected VARK style: ${varkStyle}`);

    // let existingMetadata = await clerkClient.users.getUser(userId);
    // let previousQueries = existingMetadata.publicMetadata?.queries || [];

    // // Append the new query
    // previousQueries.push({
    //   message,
    //   varkStyle,
    //   timestamp: new Date().toISOString()
    // });

    // // Update Clerk metadata
    // await clerkClient.users.updateUserMetadata(userId, {
    //   publicMetadata: { queries: previousQueries }
    // });

    // Get and process sources using the search engine function
    const sources = await searchEngineForSources(message);
    const sourcesParsed = sources
      .filter(Boolean)
      .map(group => group.map(doc => ({
        title: doc.metadata.title,
        link: doc.metadata.link,
        snippet: doc.metadata.snippet
      })))
      .flat()
      .filter((doc, index, self) =>
        index === self.findIndex(d => d.link === doc.link)
      );

    // Generate response with VARK-specific formatting
    const chatCompletion = await openai.chat.completions.create({
      messages: [
        {
          role: "system",
          content: `${varkPrompts[varkStyle]}
- Here is my query "${message}", respond back with an answer that is as long as possible.
- ${embedSourcesInLLMResponse ? "Return the sources used in the response with numbered annotations." : ""}`
        },
        {
          role: "user",
          content: `Here are the relevant search results: ${JSON.stringify(sources)}`
        }
      ],
      stream: true,
      model: "llama-3.3-70b-versatile"
    });

    // Stream the response back to the client
    let responseTotal = "";
    for await (const chunk of chatCompletion) {
      if (chunk.choices[0].delta && chunk.choices[0].finish_reason !== "stop") {
        responseTotal += chunk.choices[0].delta.content;
      } else {
        const responseObj = {
          varkStyle,
          answer: responseTotal,
          ...(returnSources && { sources: sourcesParsed }),
          ...(returnFollowUpQuestions && {
            followUpQuestions: await generateVarkStyleFollowUpQuestions(responseTotal, varkStyle)
          })
        };
        res.status(200).json(responseObj);
      }
    }
  } catch (error) {
    console.error('Error processing request:', error);
    res.status(500).json({
      error: 'An error occurred processing your request',
      details: error.message
    });
  }
});

app.post('/audio',async (req, res) => {
  // const { userId } = getAuth(req); // Get user ID from Clerk
  const {
    message,
    returnSources = true,
    returnFollowUpQuestions = true,
    embedSourcesInLLMResponse = false,
    preferredStyle = null
  } = req.body;

  try {
    // Detect VARK style (or use the provided one)
    const varkStyle = preferredStyle || await detectVarkStyle(message);
    console.log(`Detected/Selected VARK style: ${varkStyle}`);

    // let existingMetadata = await clerkClient.users.getUser(userId);
    // let previousQueries = existingMetadata.publicMetadata?.queries || [];

    // // Append the new query
    // previousQueries.push({
    //   message,
    //   varkStyle,
    //   timestamp: new Date().toISOString()
    // });

    // // Update Clerk metadata
    // await clerkClient.users.updateUserMetadata(userId, {
    //   publicMetadata: { queries: previousQueries }
    // });

    // Get and process sources using the search engine function
    const sources = await searchEngineForSources(message);
    const sourcesParsed = sources
      .filter(Boolean)
      .map(group => group.map(doc => ({
        title: doc.metadata.title,
        link: doc.metadata.link,
        snippet: doc.metadata.snippet
      })))
      .flat()
      .filter((doc, index, self) =>
        index === self.findIndex(d => d.link === doc.link)
      );

    // Generate response with VARK-specific formatting
    const chatCompletion = await openai.chat.completions.create({
      messages: [
        {
          role: "system",
          content: `${varkPrompts[varkStyle]}
- Here is my query "${message}", respond back with an answer that is please speak as if you're having a natural, friendly conversation with someone. Your tone should be warm and approachable, similar to a teacher explaining a topic in simple, clear language. Use everyday words and break down complex ideas into smaller, digestible parts to make the content easy to understand for audio learners. Aim to create an engaging, supportive learning experience that feels personal and encouraging.
- ${embedSourcesInLLMResponse ? "Return the sources used in the response with numbered annotations." : ""}`
        },
        {
          role: "user",
          content: `Here are the relevant search results: ${JSON.stringify(sources)}`
        }
      ],
      stream: true,
      model: "llama-3.3-70b-versatile"
    });

    // Stream the response back to the client
    let responseTotal = "";
    for await (const chunk of chatCompletion) {
      if (chunk.choices[0].delta && chunk.choices[0].finish_reason !== "stop") {
        responseTotal += chunk.choices[0].delta.content;
      } else {
        const responseObj = {
          varkStyle,
          answer: responseTotal,
          ...(returnSources && { sources: sourcesParsed }),
          ...(returnFollowUpQuestions && {
            followUpQuestions: await generateVarkStyleFollowUpQuestions(responseTotal, varkStyle)
          })
        };
        res.status(200).json(responseObj);
      }
    }
  } catch (error) {
    console.error('Error processing request:', error);
    res.status(500).json({
      error: 'An error occurred processing your request',
      details: error.message
    });
  }
});

// New endpoint to handle kinesthetic learning requests (POST /kinesthetic)
app.post('/kinesthetic',async (req, res) => {
  // const { userId } = getAuth(req); // Get user ID from Clerk
  const {
    message,
    returnSources = true,
    returnFollowUpQuestions = true,
    embedSourcesInLLMResponse = false
  } = req.body;

  try {
    // Set the VARK style to kinesthetic
    const varkStyle = 'kinesthetic';
    console.log(`Selected VARK style: ${varkStyle}`);

    // let previousQueries = existingMetadata.publicMetadata?.queries || [];

    // // Append the new query
    // previousQueries.push({
    //   message,
    //   varkStyle,
    //   timestamp: new Date().toISOString()
    // });

    // // Update Clerk metadata
    // await clerkClient.users.updateUserMetadata(userId, {
    //   publicMetadata: { queries: previousQueries }
    // });


    // Generate response with kinesthetic-specific formatting
    const chatCompletion = await openai.chat.completions.create({
      messages: [
        {
          role: "system",
          content: `${varkPrompts[varkStyle]}
- Here is my query "${message}", respond back with a brief definition of the topic and include links to popular learning platforms where users can gain hands-on experience. Ensure the URLs are plain and do not contain any additional formatting or special characters.`
        },
        {
          role: "user",
          content: `Here is the query: ${message}`
        }
      ],
      stream: true,
      model: "llama-3.3-70b-versatile"
    });

    // Stream the response back to the client
    let responseTotal = "";
    for await (const chunk of chatCompletion) {
      if (chunk.choices[0].delta && chunk.choices[0].finish_reason !== "stop") {
        responseTotal += chunk.choices[0].delta.content;
      } else {
        const responseObj = {
          varkStyle,
          answer: responseTotal,
          ...(returnFollowUpQuestions && {
            followUpQuestions: await generateVarkStyleFollowUpQuestions(responseTotal, varkStyle)
          })
        };
        res.status(200).json(responseObj);
      }
    }
  } catch (error) {
    console.error('Error processing request:', error);
    res.status(500).json({
      error: 'An error occurred processing your request',
      details: error.message
    });
  }
});


// New endpoint to generate a flowchart/diagram using the RapidAPI Diagram Generator (POST /diagram)
// app.post('/diagram', async (req, res) => {
//   let { query } = req.body; // Default value

//   try {
//     // Step 1: Rephrase user query
//     query = await rephraseInputDiagram(query);

//     // Step 2: Call the RapidAPI diagram generator
//     const options = {
//       method: 'POST',
//       url: 'https://ai-flowchart-diagram-generator.p.rapidapi.com/',
//       headers: {
//         'x-rapidapi-key': process.env.RAPIDAPI_KEY,
//         'x-rapidapi-host': 'ai-flowchart-diagram-generator.p.rapidapi.com',
//         'Content-Type': 'application/json'
//       },
//       data: {
//         jsonBody: {
//           function_name: 'diagram_generator',
//           query: query,
//           output_type: 'png'
//         }
//       }
//     };

//     const response = await axios.request(options);
//     console.log('Diagram API response:', response.data);

//     res.status(200).json(response.data);
//   } catch (error) {
//     console.error('Diagram API error:', error.response ? error.response.data : error.message);
//     res.status(500).json({
//       error: error.response ? error.response.data : 'Internal Server Error'
//     });
//   }
// });

app.post('/diagram', async (req, res) => {
  const {
    message,
    returnSources = true,
    returnFollowUpQuestions = true,
    embedSourcesInLLMResponse = false
  } = req.body;

  try {
    const varkStyle = 'visual';
    console.log(`Selected VARK style: ${varkStyle}`);

    // Call Groq (or OpenAI) API for Mermaid-based response
    const chatCompletion = await openai.chat.completions.create({
      messages: [
        {
          role: "system",
          content: `You are an expert assistant specialized for visual learners.
1. Start with a brief explanation of the topic.
2. Then, provide a structured Mermaid.js diagram.
3. Wrap the diagram in triple backticks with 'mermaid' keyword.
Only output the explanation and diagram.`
        },
        {
          role: "user",
          content: `Topic: ${message}`
        }
      ],
      stream: true,
      model: "llama-3.3-70b-versatile"
    });

    let responseTotal = "";
    for await (const chunk of chatCompletion) {
      if (chunk.choices[0].delta && chunk.choices[0].finish_reason !== "stop") {
        responseTotal += chunk.choices[0].delta.content;
      } else {
        const responseObj = {
          varkStyle,
          answer: responseTotal,
          ...(returnFollowUpQuestions && {
            followUpQuestions: await generateVarkStyleFollowUpQuestions(responseTotal, varkStyle)
          })
        };
        res.status(200).json(responseObj);
      }
    }
  } catch (error) {
    console.error('Error processing visual request:', error);
    res.status(500).json({
      error: 'An error occurred processing your request',
      details: error.message
    });
  }
});


app.post("/result", requireAuth(), async (req, res) => {
  const { userId } = getAuth(req); // Get user ID from Clerk
  let { preference } = req.body; // Extract preference from request body

  if (!preference) {
    return res.status(400).json({ message: "Missing required field: preference" });
  }

  // 🔹 Apply transformations to preference
  if (preference === "Multimodal") {
    preference = "Read";
  } else if (preference === "Read/Write") {
    preference = "Read";
  }

  try {
    await clerkClient.users.updateUserMetadata(userId, {
      publicMetadata: { preference, formSubmitted: true },
    });

    res.status(200).json({ message: "Results saved successfully", modifiedPreference: preference });
  } catch (error) {
    console.error("Error saving VARK results:", error);
    res.status(500).json({ message: "Failed to save results" });
  }
});

app.get("/",(req, res) => {
  res.send("Hello World");
})

// Start the server
app.listen(port, () => {
  console.log(`VARK-enhanced RAG server with Serper API is listening on port ${port}`);
});
